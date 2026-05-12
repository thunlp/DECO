# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-expert MoE health metrics.

These metrics catch the failure modes described in Step 3.5 Flash
(arXiv:2602.10604, sections 4.1.2 and 4.1.3): silent expert collapse and
localized activation blow-up. Both are invisible to router dispatch
statistics and to the training loss, but show up immediately in the
dispersion (max/median, min/median) of per-expert activation and weight
norms.

What we track per (layer, expert):
  * input_rms        : sqrt(mean over tokens of ||expert input||^2 / hidden_size)
                      — the routed hidden state before the expert fc1.
  * fc1_pre_act_rms  : sqrt(mean over tokens of ||fc1(x)||^2 / fc1_dim)
                      — the gate+up projection before SwiGLU.
  * gate_branch_rms  : RMS of the gate branch of the fc1 projection.
  * up_branch_rms    : RMS of the up branch of the fc1 projection.
  * intermediate_rms : sqrt(mean over tokens of ||SwiGLU intermediate||^2 / d_ff)
                      — the "RMS at the MoE FFN intermediate" recommended
                      by the paper.
  * output_rms       : sqrt(mean over tokens of ||expert output||^2 / hidden_size)
                      — the routed expert contribution before token
                      unpermutation / probability weighting.
  * swiglu_gain      : intermediate_rms / input_rms.
  * fc2_gain         : output_rms / intermediate_rms.
  * w_fc1_fronorm   : ||W_fc1||_F  (gate+up projection of the routed expert)
  * w_fc2_fronorm   : ||W_fc2||_F  (down projection of the routed expert)

What we log through TensorBoard / W&B:
  * <metric>/max_over_median  — early warning of localized blow-up
  * <metric>/min_over_median  — early warning of expert collapse
  * <metric>/max_over_mean
  * <metric>/min_over_mean

If --moe-expert-metrics-per-expert-logging is set, we additionally log
the full per-expert vector for each MoE layer.

Distributed handling:
  * intra-MoE TP : input hidden states are treated as replicated across
                   expert-TP ranks; for intermediate activations, sum-reduce per-token
                   sum-of-squares over ``expert_tensor_parallel`` (correct
                   because ||x||^2 is element-wise); for output activations,
                   sum-reduce the output vector first, then take its norm
                   (correct for row-parallel additive partial outputs).
  * DP / CP      : sum-reduce sum-of-squares AND token counts over
                   ``expert_data_parallel`` (which already absorbs CP).
  * EP           : sum-reduce a (num_layers, num_global_experts) tensor over
                   ``expert_model_parallel`` — non-local experts contribute 0.
  * PP           : sum-reduce the same tensor over the PP group — non-local
                   layers contribute 0.

Activation accumulation happens inline in the expert forward (under
``torch.no_grad``) so it composes correctly with selective activation
recompute / CheckpointWithoutOutput. Weight Frobenius norms are computed
on demand at log time from the registered expert modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from megatron.core import parallel_state


# ---------------------------------------------------------------------------
# Module registry: every TEGroupedMLP / GroupedMLP registers itself here so
# that we can compute weight norms on demand at log time without having to
# walk the whole model.
# ---------------------------------------------------------------------------

_REGISTERED_EXPERT_MODULES: List[Any] = []


def register_expert_module(module: Any) -> None:
    """Register a grouped-experts module so its weight norms can be polled."""
    _REGISTERED_EXPERT_MODULES.append(module)


def _iter_registered_modules():
    """Iterate over registered expert modules that have a layer_number assigned."""
    for module in _REGISTERED_EXPERT_MODULES:
        if getattr(module, "layer_number", None) is None:
            continue
        yield module


# ---------------------------------------------------------------------------
# Activation accumulator: per-step buffer of per-local-expert sum-of-squares
# and token counts, keyed by (metric_name, layer_number).
# ---------------------------------------------------------------------------


class _ActivationAccumulator:
    """Per-rank accumulator of per-expert activation statistics."""

    def __init__(self) -> None:
        # name -> { layer_number -> { 'sumsq': Tensor[num_local_experts],
        #                              'count': Tensor[num_local_experts],
        #                              'dim':   int (full feature dim) } }
        self._buf: Dict[str, Dict[int, Dict[str, Any]]] = {}

    def add(
        self,
        name: str,
        layer_number: int,
        sumsq_per_local_expert: torch.Tensor,
        count_per_local_expert: torch.Tensor,
        feature_dim: int,
    ) -> None:
        if layer_number is None:
            return
        per_metric = self._buf.setdefault(name, {})
        slot = per_metric.get(layer_number)
        if slot is None:
            per_metric[layer_number] = {
                "sumsq": sumsq_per_local_expert.detach().clone(),
                "count": count_per_local_expert.detach().clone(),
                "dim": int(feature_dim),
            }
        else:
            slot["sumsq"] = slot["sumsq"] + sumsq_per_local_expert.detach()
            slot["count"] = slot["count"] + count_per_local_expert.detach()
            # ``dim`` is a property of the layer, not of the micro-batch.
            assert slot["dim"] == int(feature_dim), (
                f"Inconsistent feature dim for {name} at layer {layer_number}: "
                f"{slot['dim']} vs {feature_dim}"
            )

    def items(self):
        return self._buf.items()

    def reset(self) -> None:
        self._buf.clear()


_ACCUMULATOR = _ActivationAccumulator()


def get_accumulator() -> _ActivationAccumulator:
    return _ACCUMULATOR


# ---------------------------------------------------------------------------
# Forward-time recording helpers (called from inside expert .forward).
# ---------------------------------------------------------------------------


def _expert_tp_all_reduce_sum_(tensor: torch.Tensor) -> None:
    """In-place sum-reduce ``tensor`` over the expert tensor-parallel group."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    try:
        group = parallel_state.get_expert_tensor_parallel_group(check_initialized=False)
    except AssertionError:
        group = None
    if group is None:
        return
    if torch.distributed.get_world_size(group=group) <= 1:
        return
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=group)


def record_intermediate_activation(
    layer_number: Optional[int],
    intermediate: torch.Tensor,
    tokens_per_expert: List[int],
    num_local_experts: int,
    name: str = "intermediate",
) -> None:
    """Accumulate per-local-expert sum-of-squares for a SwiGLU intermediate.

    ``intermediate`` has shape ``[sum(tokens_per_expert), ffn_per_partition]``
    (the feature dim is sharded by ``expert_tensor_parallel``). The per-token
    sum-of-squares is summed across the TP group to recover the full
    per-token squared norm before being grouped by expert.

    This function is a no-op if ``layer_number`` is None (e.g. layer-number
    propagation has not happened yet, which is the case for some test paths).
    """
    if layer_number is None:
        return
    if intermediate is None or intermediate.numel() == 0:
        # Even when this rank has no tokens routed to its experts, we still
        # need to register zero counts so the per-layer max/min over experts
        # is taken across all global experts.
        device = (
            intermediate.device
            if intermediate is not None
            else torch.cuda.current_device()
        )
        zeros = torch.zeros(num_local_experts, device=device, dtype=torch.float32)
        feature_dim_local = (
            intermediate.shape[-1] if intermediate is not None and intermediate.dim() >= 1 else 1
        )
        try:
            tp_size = parallel_state.get_expert_tensor_parallel_world_size()
        except Exception:
            tp_size = 1
        feature_dim = int(feature_dim_local) * max(int(tp_size), 1)
        get_accumulator().add(name, int(layer_number), zeros, zeros.clone(), feature_dim)
        return

    device = intermediate.device
    with torch.no_grad():
        # |x|^2 per token, summed over the local TP shard of the feature dim.
        per_token_sumsq = intermediate.detach().to(torch.float32).pow(2).sum(dim=-1)

        # Sum-reduce across expert TP so each rank holds the full per-token
        # squared norm. (||x||^2 is element-wise, so summing partial squared
        # norms is exact.)
        _expert_tp_all_reduce_sum_(per_token_sumsq)

        # Group by expert. Use repeat_interleave + scatter_add to avoid a
        # python-level loop over experts.
        counts = torch.tensor(tokens_per_expert, device=device, dtype=torch.long)
        assert counts.numel() == num_local_experts, (
            f"tokens_per_expert has {counts.numel()} entries but num_local_experts is "
            f"{num_local_experts}"
        )
        assert int(counts.sum().item()) == per_token_sumsq.numel(), (
            "sum(tokens_per_expert) must equal the number of permuted rows"
        )
        expert_idx = torch.repeat_interleave(
            torch.arange(num_local_experts, device=device), counts
        )
        sumsq = torch.zeros(num_local_experts, device=device, dtype=torch.float32)
        sumsq.scatter_add_(0, expert_idx, per_token_sumsq)

        count_t = counts.to(torch.float32)

        try:
            tp_size = parallel_state.get_expert_tensor_parallel_world_size()
        except Exception:
            tp_size = 1
        feature_dim = int(intermediate.shape[-1]) * max(int(tp_size), 1)

        get_accumulator().add(name, int(layer_number), sumsq, count_t, feature_dim)


def record_input_activation(
    layer_number: Optional[int],
    hidden_states: torch.Tensor,
    tokens_per_expert: List[int],
    num_local_experts: int,
    name: str = "input",
) -> None:
    """Accumulate per-local-expert sum-of-squares for routed expert inputs.

    Expert inputs are the hidden states after token dispatch and before the
    expert ``fc1``. In the DeepSeek-style grouped-MLP path used here, the
    expert input hidden dimension is not partitioned by expert tensor
    parallelism; each expert-TP rank sees the same routed hidden state and
    partitions only the ``fc1`` output / ``fc2`` input dimensions. Therefore
    this helper intentionally does *not* reduce over expert TP.
    """
    if layer_number is None:
        return
    if hidden_states is None or hidden_states.numel() == 0:
        device = hidden_states.device if hidden_states is not None else torch.cuda.current_device()
        zeros = torch.zeros(num_local_experts, device=device, dtype=torch.float32)
        feature_dim = (
            hidden_states.shape[-1] if hidden_states is not None and hidden_states.dim() >= 1 else 1
        )
        get_accumulator().add(name, int(layer_number), zeros, zeros.clone(), int(feature_dim))
        return

    device = hidden_states.device
    with torch.no_grad():
        per_token_sumsq = hidden_states.detach().to(torch.float32).pow(2).sum(dim=-1)
        counts = torch.tensor(tokens_per_expert, device=device, dtype=torch.long)
        assert counts.numel() == num_local_experts, (
            f"tokens_per_expert has {counts.numel()} entries but num_local_experts is "
            f"{num_local_experts}"
        )
        assert int(counts.sum().item()) == per_token_sumsq.numel(), (
            "sum(tokens_per_expert) must equal the number of permuted rows"
        )
        expert_idx = torch.repeat_interleave(
            torch.arange(num_local_experts, device=device), counts
        )
        sumsq = torch.zeros(num_local_experts, device=device, dtype=torch.float32)
        sumsq.scatter_add_(0, expert_idx, per_token_sumsq)
        get_accumulator().add(
            name,
            int(layer_number),
            sumsq,
            counts.to(torch.float32),
            int(hidden_states.shape[-1]),
        )


def record_fc1_pre_activation(
    layer_number: Optional[int],
    fc1_pre_act: torch.Tensor,
    tokens_per_expert: List[int],
    num_local_experts: int,
    gated_linear_unit: bool,
) -> None:
    """Accumulate RMS inputs to the expert nonlinearity.

    For SwiGLU experts this records three metrics in one pass:
    ``fc1_pre_act`` (gate+up together), ``gate_branch``, and ``up_branch``.
    The feature dimension is expert-TP sharded, so all three per-token
    sum-of-squares vectors are stacked and expert-TP SUM-reduced in a single
    collective.

    The values are recorded before activation. In configurations with FC1
    bias, this is the pre-bias projection output; DeepSeek-style MoE normally
    disables expert biases, which makes this exactly the SwiGLU pre-activation.
    """
    if layer_number is None:
        return

    metric_names = ("fc1_pre_act", "gate_branch", "up_branch")
    if fc1_pre_act is None or fc1_pre_act.numel() == 0:
        device = fc1_pre_act.device if fc1_pre_act is not None else torch.cuda.current_device()
        zeros = torch.zeros(num_local_experts, device=device, dtype=torch.float32)
        local_dim = fc1_pre_act.shape[-1] if fc1_pre_act is not None and fc1_pre_act.dim() else 1
        try:
            tp_size = parallel_state.get_expert_tensor_parallel_world_size()
        except Exception:
            tp_size = 1
        full_dim = int(local_dim) * max(int(tp_size), 1)
        branch_dim = full_dim // 2 if gated_linear_unit else full_dim
        for metric_name, dim in zip(metric_names, (full_dim, branch_dim, branch_dim)):
            get_accumulator().add(
                metric_name,
                int(layer_number),
                zeros,
                zeros.clone(),
                int(dim),
            )
        return

    device = fc1_pre_act.device
    with torch.no_grad():
        fc1 = fc1_pre_act.detach().to(torch.float32)
        full_sumsq = fc1.pow(2).sum(dim=-1)

        if gated_linear_unit:
            gate, up = torch.chunk(fc1, 2, dim=-1)
            gate_sumsq = gate.pow(2).sum(dim=-1)
            up_sumsq = up.pow(2).sum(dim=-1)
            per_token_sumsq = torch.stack((full_sumsq, gate_sumsq, up_sumsq), dim=0)
            local_branch_dim = gate.shape[-1]
        else:
            # Non-gated experts do not have gate/up branches; use zeros so the
            # fixed metric list still logs deterministically without extra work.
            zeros_like = torch.zeros_like(full_sumsq)
            per_token_sumsq = torch.stack((full_sumsq, zeros_like, zeros_like), dim=0)
            local_branch_dim = fc1.shape[-1]

        _expert_tp_all_reduce_sum_(per_token_sumsq)

        counts = torch.tensor(tokens_per_expert, device=device, dtype=torch.long)
        assert counts.numel() == num_local_experts, (
            f"tokens_per_expert has {counts.numel()} entries but num_local_experts is "
            f"{num_local_experts}"
        )
        assert int(counts.sum().item()) == per_token_sumsq.shape[-1], (
            "sum(tokens_per_expert) must equal the number of permuted rows"
        )
        expert_idx = torch.repeat_interleave(
            torch.arange(num_local_experts, device=device), counts
        )

        sumsq = torch.zeros(
            len(metric_names), num_local_experts, device=device, dtype=torch.float32
        )
        for metric_idx in range(len(metric_names)):
            sumsq[metric_idx].scatter_add_(0, expert_idx, per_token_sumsq[metric_idx])

        try:
            tp_size = parallel_state.get_expert_tensor_parallel_world_size()
        except Exception:
            tp_size = 1
        full_dim = int(fc1.shape[-1]) * max(int(tp_size), 1)
        branch_dim = int(local_branch_dim) * max(int(tp_size), 1)
        dims = (full_dim, branch_dim, branch_dim)
        count_t = counts.to(torch.float32)
        for metric_idx, metric_name in enumerate(metric_names):
            get_accumulator().add(
                metric_name,
                int(layer_number),
                sumsq[metric_idx],
                count_t,
                dims[metric_idx],
            )


def record_output_activation(
    layer_number: Optional[int],
    output: torch.Tensor,
    tokens_per_expert: List[int],
    num_local_experts: int,
    name: str = "output",
) -> None:
    """Accumulate per-local-expert sum-of-squares for routed expert outputs.

    Unlike the SwiGLU intermediate, the row-parallel down projection produces
    additive partial output contributions when expert tensor parallelism is
    enabled. Therefore we first SUM-reduce the output vector itself across the
    expert-TP group, and only then take the per-token norm. This is more
    expensive than the intermediate metric but gives the actual expert-output
    RMS instead of the norm of each rank's local contribution.
    """
    if layer_number is None:
        return
    if output is None or output.numel() == 0:
        device = output.device if output is not None else torch.cuda.current_device()
        zeros = torch.zeros(num_local_experts, device=device, dtype=torch.float32)
        feature_dim = output.shape[-1] if output is not None and output.dim() >= 1 else 1
        get_accumulator().add(name, int(layer_number), zeros, zeros.clone(), int(feature_dim))
        return

    device = output.device
    with torch.no_grad():
        full_output = output.detach().to(torch.float32).clone()
        _expert_tp_all_reduce_sum_(full_output)
        per_token_sumsq = full_output.pow(2).sum(dim=-1)

        counts = torch.tensor(tokens_per_expert, device=device, dtype=torch.long)
        assert counts.numel() == num_local_experts, (
            f"tokens_per_expert has {counts.numel()} entries but num_local_experts is "
            f"{num_local_experts}"
        )
        assert int(counts.sum().item()) == per_token_sumsq.numel(), (
            "sum(tokens_per_expert) must equal the number of permuted rows"
        )
        expert_idx = torch.repeat_interleave(
            torch.arange(num_local_experts, device=device), counts
        )
        sumsq = torch.zeros(num_local_experts, device=device, dtype=torch.float32)
        sumsq.scatter_add_(0, expert_idx, per_token_sumsq)

        get_accumulator().add(
            name,
            int(layer_number),
            sumsq,
            counts.to(torch.float32),
            int(output.shape[-1]),
        )


# ---------------------------------------------------------------------------
# Weight-norm helpers (called at log time, not on every forward).
# ---------------------------------------------------------------------------


def _flatten_weights(module: Any) -> List[Tuple[str, torch.Tensor]]:
    """Return [(role, weight)] pairs for every local expert in ``module``.

    Supports both TEGroupedMLP-style (``linear_fcN.weightI``) and the legacy
    GroupedMLP-style (``weight1`` / ``weight2`` viewed as an experts-major
    tensor of shape ``[hidden, num_local_experts * ffn_per_partition]``).
    """
    n = int(getattr(module, "num_local_experts", 0))
    if n <= 0:
        return []

    pairs: List[Tuple[str, torch.Tensor]] = []

    fc1 = getattr(module, "linear_fc1", None)
    fc2 = getattr(module, "linear_fc2", None)
    if fc1 is not None and hasattr(fc1, "weight0"):
        for i in range(n):
            w1 = getattr(fc1, f"weight{i}", None)
            if w1 is not None:
                pairs.append((f"w_fc1::{i}", w1))
        for i in range(n):
            w2 = getattr(fc2, f"weight{i}", None) if fc2 is not None else None
            if w2 is not None:
                pairs.append((f"w_fc2::{i}", w2))
        return pairs

    weight1 = getattr(module, "weight1", None)
    weight2 = getattr(module, "weight2", None)
    if weight1 is not None and weight2 is not None:
        # Legacy GroupedMLP: the weights are concatenated along the
        # experts axis.
        try:
            hidden = int(module.config.hidden_size)
            w1 = weight1.view(n, hidden, -1)
            w2 = weight2.view(n, -1, hidden)
            for i in range(n):
                pairs.append((f"w_fc1::{i}", w1[i]))
                pairs.append((f"w_fc2::{i}", w2[i]))
        except Exception:
            return []
    return pairs


def _gather_weight_normsq_per_local_expert(module: Any) -> Dict[str, torch.Tensor]:
    """Compute ||W||_F^2 per local expert for fc1 and fc2.

    Each weight is sharded across ``expert_tensor_parallel``; we sum
    element-wise squared values and then SUM-reduce across that group to
    recover the true Frobenius norm squared.
    """
    pairs = _flatten_weights(module)
    if not pairs:
        return {}

    n = int(module.num_local_experts)
    device = next(iter(p for _, p in pairs)).device
    fc1_sumsq = torch.zeros(n, device=device, dtype=torch.float32)
    fc2_sumsq = torch.zeros(n, device=device, dtype=torch.float32)

    for role, weight in pairs:
        kind, idx = role.split("::")
        i = int(idx)
        ssq = weight.detach().to(torch.float32).pow(2).sum()
        if kind == "w_fc1":
            fc1_sumsq[i] += ssq
        elif kind == "w_fc2":
            fc2_sumsq[i] += ssq

    _expert_tp_all_reduce_sum_(fc1_sumsq)
    _expert_tp_all_reduce_sum_(fc2_sumsq)

    return {"w_fc1": fc1_sumsq, "w_fc2": fc2_sumsq}


# ---------------------------------------------------------------------------
# Distributed reductions and per-layer dispersion summaries.
# ---------------------------------------------------------------------------


def _ep_pp_sum_reduce_(tensor: torch.Tensor) -> None:
    """In-place sum-reduce a ``(num_layers, num_experts)`` tensor over EP+PP.

    Each EP rank contributes only to the ``num_local_experts`` slice it owns;
    each PP rank contributes only to the layers it owns; everything else is
    zero.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return

    try:
        ep = parallel_state.get_expert_model_parallel_group(check_initialized=False)
    except AssertionError:
        ep = None
    if ep is not None and torch.distributed.get_world_size(group=ep) > 1:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=ep)

    try:
        pp = parallel_state.get_pipeline_model_parallel_group()
    except AssertionError:
        pp = None
    if pp is not None and torch.distributed.get_world_size(group=pp) > 1:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=pp)


def _dp_sum_reduce_(tensor: torch.Tensor) -> None:
    """In-place sum-reduce over the expert data-parallel group.

    For Megatron's MoE topology, the expert-DP group already absorbs the
    context-parallel dimension (see ``initialize_model_parallel`` —
    ``expert_data_parallel_size = world / (expert_tp * expert_ep * pp)``,
    which divides through the CP factor). Hence a single all-reduce here is
    enough to aggregate token statistics across all data/CP ranks.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    try:
        dp = parallel_state.get_expert_data_parallel_group()
    except AssertionError:
        dp = None
    if dp is None:
        return
    if torch.distributed.get_world_size(group=dp) <= 1:
        return
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=dp)


def _layer_dispersion(per_expert: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute max/min over median (and over mean) along the last dim.

    Experts with zero count (no contribution this step) are excluded from
    the statistics so they don't artificially drive ``min`` to zero. If an
    entire layer has zero contributions we report 0 for everything.
    """
    valid = per_expert > 0
    n_valid = valid.sum(dim=-1, dtype=torch.float32)
    safe = torch.where(valid, per_expert, torch.full_like(per_expert, float("nan")))

    median = torch.nanmedian(safe, dim=-1).values
    mean = torch.nansum(safe, dim=-1) / n_valid.clamp_min(1.0)

    inf = torch.full_like(per_expert, float("inf"))
    ninf = torch.full_like(per_expert, float("-inf"))
    max_v = torch.where(valid, per_expert, ninf).amax(dim=-1)
    min_v = torch.where(valid, per_expert, inf).amin(dim=-1)

    eps = torch.finfo(per_expert.dtype).tiny
    safe_med = median.clamp_min(eps)
    safe_mean = mean.clamp_min(eps)

    out = {
        "max_over_median": max_v / safe_med,
        "min_over_median": min_v / safe_med,
        "max_over_mean": max_v / safe_mean,
        "min_over_mean": min_v / safe_mean,
        "max": max_v,
        "min": min_v,
        "median": median,
        "mean": mean,
    }
    # Layers with no valid expert get zeroed out across the board.
    no_data = n_valid == 0
    if no_data.any():
        for k in out:
            out[k] = torch.where(no_data, torch.zeros_like(out[k]), out[k])
    return out


def _build_global_layer_expert_tensor(
    per_metric_buffer: Dict[int, Dict[str, Any]],
    num_layers: int,
    num_global_experts: int,
    num_local_experts: int,
    ep_rank: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
    """Build (num_layers, num_global_experts) sumsq and count tensors.

    Only this rank's local layers and local experts are populated. The
    remaining slots are 0, so a subsequent SUM-reduce across EP and PP fills
    in everything.

    Returns:
        sumsq, count, feature_dim, has_any_data
    """
    sumsq = torch.zeros(num_layers, num_global_experts, device=device, dtype=torch.float32)
    count = torch.zeros_like(sumsq)
    feature_dim = 0
    has_any = False

    expert_offset = ep_rank * num_local_experts
    for layer_number, slot in per_metric_buffer.items():
        if layer_number is None:
            continue
        idx = int(layer_number) - 1
        if idx < 0 or idx >= num_layers:
            # Layer numbers should be 1-indexed within [1, num_layers] but be
            # defensive — silently skip out-of-range entries rather than
            # crashing the training run for a logging side-effect.
            continue
        local_sumsq = slot["sumsq"].to(device=device, dtype=torch.float32)
        local_count = slot["count"].to(device=device, dtype=torch.float32)
        end = expert_offset + local_sumsq.numel()
        sumsq[idx, expert_offset:end] = local_sumsq
        count[idx, expert_offset:end] = local_count
        feature_dim = max(feature_dim, int(slot["dim"]))
        has_any = True

    return sumsq, count, feature_dim, has_any


def _log_per_expert(
    writer,
    wandb_writer,
    iteration: int,
    name: str,
    per_layer_per_expert: torch.Tensor,
) -> None:
    """Log the full ``(num_layers, num_experts)`` matrix as scalars.

    This is opt-in via ``--moe-expert-metrics-per-expert-logging`` because it
    can produce a large number of scalar series (num_moe_layers * num_experts).
    """
    n_layers, n_experts = per_layer_per_expert.shape
    for layer_idx in range(n_layers):
        row = per_layer_per_expert[layer_idx]
        if not bool(row.any()):
            continue
        for expert_idx in range(n_experts):
            v = row[expert_idx].item()
            tag = f"moe-experts/{name}_layer_{layer_idx}_expert_{expert_idx}"
            if writer is not None:
                writer.add_scalar(tag, v, iteration)
            if wandb_writer is not None:
                wandb_writer.log({tag: v}, iteration)


def _log_metric_summary(
    writer,
    wandb_writer,
    iteration: int,
    name: str,
    per_expert: torch.Tensor,
    per_layer_logging: bool,
    per_expert_logging: bool,
) -> None:
    """Log per-layer dispersion and optional per-expert scalar series."""
    layer_disp = _layer_dispersion(per_expert)

    if per_expert_logging:
        _log_per_expert(writer, wandb_writer, iteration, name, per_expert)

    for stat_name, stat_per_layer in layer_disp.items():
        tag = f"moe/expert-metrics/{name}/{stat_name}"
        # Average over MoE layers only (those with non-zero data) so the
        # cross-layer mean isn't dragged toward zero by dense layers and by
        # PP ranks that hold no MoE block.
        nz = stat_per_layer != 0
        n = nz.sum().clamp_min(1)
        mean_val = (stat_per_layer.to(torch.float32) * nz.to(torch.float32)).sum() / n
        mean_v = float(mean_val.item())
        if writer is not None:
            writer.add_scalar(tag, mean_v, iteration)
        if wandb_writer is not None:
            wandb_writer.log({tag: mean_v}, iteration)

        if per_layer_logging:
            stat_cpu = stat_per_layer.detach().to("cpu")
            for li in range(stat_cpu.numel()):
                v = float(stat_cpu[li].item())
                if v == 0.0:
                    continue
                layer_tag = f"{tag}_layer_{li}"
                if writer is not None:
                    writer.add_scalar(layer_tag, v, iteration)
                if wandb_writer is not None:
                    wandb_writer.log({layer_tag: v}, iteration)


# ---------------------------------------------------------------------------
# Public entry point: called once per logging step from training.py.
# ---------------------------------------------------------------------------


# Fixed list of metrics so every rank performs the same collective ops in the
# same order, regardless of whether this rank actually owns any MoE layer.
_ACTIVATION_METRIC_NAMES: Tuple[str, ...] = (
    "input",
    "fc1_pre_act",
    "gate_branch",
    "up_branch",
    "intermediate",
    "output",
)
_DERIVED_METRIC_NAMES: Tuple[str, ...] = ("swiglu_gain", "fc2_gain")
_WEIGHT_METRIC_NAMES: Tuple[str, ...] = ("w_fc1", "w_fc2")


def track_expert_health_metrics(
    iteration: int,
    writer,
    wandb_writer=None,
    *,
    num_layers: Optional[int] = None,
    num_global_experts: Optional[int] = None,
    feature_dim: Optional[int] = None,
    output_feature_dim: Optional[int] = None,
    per_layer_logging: bool = False,
    per_expert_logging: bool = False,
) -> None:
    """Aggregate, reduce, summarize, and log per-expert MoE metrics.

    Args:
        iteration: training step (used for the writer x-axis).
        writer: TensorBoard SummaryWriter or None.
        wandb_writer: W&B run handle or None.
        num_layers: total number of transformer layers (used to size the
            per-layer tracker). Must match across ranks.
        num_global_experts: total number of routed experts (EP-aggregated).
            Must match across ranks.
        feature_dim: ``moe_ffn_hidden_size`` — used to normalize activation
            sum-of-squares for the intermediate metric to RMS units. Must
            match across ranks.
        output_feature_dim: ``hidden_size`` — used to normalize expert-output
            sum-of-squares to RMS units. Defaults to ``feature_dim`` when not
            supplied for backwards compatibility.
        per_layer_logging: if True, log dispersion ratios per-layer in
            addition to the cross-layer mean.
        per_expert_logging: if True, also log the full per-expert vector
            for every layer (very verbose).

    Every rank that calls this function MUST call it at the same iteration —
    the function performs collective all-reduces over EP, DP, and PP. Calls
    are idempotent: per-step buffers are reset on exit.
    """
    if num_layers is None or num_global_experts is None:
        # Without these we cannot build a global layer-by-expert tensor.
        return

    if feature_dim is None or feature_dim <= 0:
        feature_dim = 1  # degrade gracefully; reported as "RMS / sqrt(dim)" anyway
    if output_feature_dim is None or output_feature_dim <= 0:
        output_feature_dim = feature_dim

    device = (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    ep_world_size = max(parallel_state.get_expert_model_parallel_world_size() or 1, 1)
    ep_rank = parallel_state.get_expert_model_parallel_rank() or 0
    if num_global_experts % ep_world_size != 0:
        # Pathological config — bail out without logging rather than crashing.
        get_accumulator().reset()
        return
    num_local_experts = num_global_experts // ep_world_size

    # ---------- 1. Activation metrics from the per-step accumulator ----------

    activation_buffers: Dict[str, Dict[int, Dict[str, Any]]] = dict(get_accumulator().items())

    # ---------- 2. Weight metrics — recomputed each call ---------------------

    weight_buffers: Dict[str, Dict[int, Dict[str, Any]]] = {n: {} for n in _WEIGHT_METRIC_NAMES}
    for module in _iter_registered_modules():
        per_expert = _gather_weight_normsq_per_local_expert(module)
        if not per_expert:
            continue
        layer_number = int(module.layer_number)
        for kind, sumsq in per_expert.items():
            weight_buffers[kind][layer_number] = {
                "sumsq": sumsq,
                "count": torch.ones_like(sumsq),
                "dim": 1,
            }

    # ---------- 3. Iterate over the FIXED metric list -----------------------
    # All ranks iterate the same names in the same order, so the all-reduces
    # match up even when some ranks have no contribution for a given metric.

    all_specs: List[Tuple[str, str, Dict[int, Dict[str, Any]]]] = []
    for name in _ACTIVATION_METRIC_NAMES:
        all_specs.append((name, "activation", activation_buffers.get(name, {})))
    for name in _WEIGHT_METRIC_NAMES:
        all_specs.append((name, "weight", weight_buffers.get(name, {})))

    activation_per_expert: Dict[str, torch.Tensor] = {}

    for name, kind, per_metric_buffer in all_specs:
        sumsq, count, metric_dim, _ = _build_global_layer_expert_tensor(
            per_metric_buffer=per_metric_buffer,
            num_layers=num_layers,
            num_global_experts=num_global_experts,
            num_local_experts=num_local_experts,
            ep_rank=ep_rank,
            device=device,
        )

        # DP sum-reduce only makes sense for token-aggregated activation
        # metrics. Weights are replicated across DP, so DP reduction would
        # over-count by the DP factor — skip it.
        if kind == "activation":
            _dp_sum_reduce_(sumsq)
            _dp_sum_reduce_(count)

        # EP + PP sum-reduce the (layers, experts) matrix on every rank.
        _ep_pp_sum_reduce_(sumsq)
        _ep_pp_sum_reduce_(count)

        # Compute per-(layer, expert) statistic.
        if kind == "activation":
            if metric_dim > 0:
                metric_feature_dim = metric_dim
            else:
                metric_feature_dim = output_feature_dim if name in ("input", "output") else feature_dim
            denom = count.clamp_min(1.0) * float(max(int(metric_feature_dim), 1))
            per_expert = torch.sqrt(sumsq / denom)
            per_expert = torch.where(count > 0, per_expert, torch.zeros_like(per_expert))
            activation_per_expert[name] = per_expert
        else:
            per_expert = torch.sqrt(sumsq)

        _log_metric_summary(
            writer,
            wandb_writer,
            iteration,
            name,
            per_expert,
            per_layer_logging,
            per_expert_logging,
        )

    eps = torch.finfo(torch.float32).tiny
    if "input" in activation_per_expert and "intermediate" in activation_per_expert:
        swiglu_gain = activation_per_expert["intermediate"] / activation_per_expert[
            "input"
        ].clamp_min(eps)
        swiglu_gain = torch.where(
            activation_per_expert["input"] > 0,
            swiglu_gain,
            torch.zeros_like(swiglu_gain),
        )
        _log_metric_summary(
            writer,
            wandb_writer,
            iteration,
            "swiglu_gain",
            swiglu_gain,
            per_layer_logging,
            per_expert_logging,
        )

    if "intermediate" in activation_per_expert and "output" in activation_per_expert:
        fc2_gain = activation_per_expert["output"] / activation_per_expert[
            "intermediate"
        ].clamp_min(eps)
        fc2_gain = torch.where(
            activation_per_expert["intermediate"] > 0,
            fc2_gain,
            torch.zeros_like(fc2_gain),
        )
        _log_metric_summary(
            writer,
            wandb_writer,
            iteration,
            "fc2_gain",
            fc2_gain,
            per_layer_logging,
            per_expert_logging,
        )

    # Drop the per-step buffer — the next step starts fresh.
    get_accumulator().reset()
