import torch
from typing import Union
from dataclasses import dataclass
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.blockffn.activations import get_activation_fn
from megatron.core.transformer.blockffn.statistics import ActivationStatistics
from megatron.core.transformer.moe.loss_free import LossFreeBalance
from megatron.core.transformer.blockffn.router_entropy import RegRouterEntropy
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.moe.moe_utils import switch_load_balancing_loss_func, save_to_aux_losses_tracker, MoEAuxLossAutoScaler


@dataclass
class BlockFFNSubmodules:
    # router module
    moe_router: Union[ModuleSpec, type] = None
    router_norm: Union[ModuleSpec, type] = None
    # expert module
    expert_gate_proj: Union[ModuleSpec, type] = None
    expert_up_proj: Union[ModuleSpec, type] = None
    expert_act_norm: Union[ModuleSpec, type] = None
    expert_down_proj: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class RouterFP32(MegatronModule):
    """Base Router class"""

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize the Router module.
        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
        """
        super().__init__(config)
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.layer_number = None

        self.weight = torch.nn.Parameter(
            torch.empty((self.config.num_moe_experts, self.config.hidden_size), dtype=torch.float32)
        )
        if config.perform_initialization:
            config.init_method(self.weight)
        self.weight.data = self.weight.data.to(dtype=config.params_dtype)
        setattr(self.weight, "sequence_parallel", config.sequence_parallel)

    def forward(self, input: torch.Tensor):
        """Forward pass of the router gate.
        Args:
            input (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Logits tensor.
        """
        if self.weight.device.type == "cpu":
            # move weights to GPU
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        # Convert to specified datatype for routing computation if enabled
        router_dtype = input.dtype
        if self.config.moe_router_dtype == "fp32":
            router_dtype = torch.float32
        elif self.config.moe_router_dtype == "fp64":
            router_dtype = torch.float64
        logits = torch.nn.functional.linear(input.to(router_dtype), self.weight.to(router_dtype))
        return logits

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the router."""
        self.layer_number = layer_number


class BlockFFNLayer(MegatronModule):
    def __init__(
        self, config: TransformerConfig, submodules: BlockFFNSubmodules = None, layer_number: int = None,
    ):
        super(BlockFFNLayer, self).__init__(config=config)
        self.submodules = submodules
        self.config = config
        self.num_experts, self.dim_expert, self.hidden_size = \
            config.num_moe_experts, config.moe_ffn_hidden_size, config.hidden_size
        self.dim_shared_expert = config.moe_shared_expert_intermediate_size
        self.layer_number = layer_number
        self.router_norm_type = config.router_norm_type
        self.router_norm_init_var = config.router_norm_init_var
        self.router_target_act_ratio = config.router_target_act_ratio
        self.expert_gated = self.config.gated_linear_unit and not config.expert_not_gated

        assert submodules.moe_router == RouterFP32
        self.moe_router = RouterFP32(self.config)
        self.router_act = get_activation_fn(config.router_act_func)
        self.router_norm = build_module(
            submodules.router_norm,
            config=self.config,
            hidden_size=(1 if self.config.router_norm_scalar else self.num_experts),
            eps=self.config.layernorm_epsilon,
            init_var=self.router_norm_init_var,
            norm_type=self.router_norm_type,
            fixed=self.config.router_norm_fixed,
        )

        self.expert_gate_proj = None
        if self.expert_gated:
            self.expert_gate_proj = build_module(
                submodules.expert_gate_proj,
                self.hidden_size,
                self.num_experts * self.dim_expert,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear,
                skip_bias_add=True,
                is_expert=False,
                tp_comm_buffer_name="expert_gate_proj",
            )

        self.expert_up_proj = build_module(
            submodules.expert_up_proj,
            self.hidden_size,
            self.num_experts * self.dim_expert,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="expert_up_proj",
        )
        self.expert_act = get_activation_fn(
            config.expert_act_func, config=config, norm_module=submodules.expert_act_norm,
            norm_dim=self.num_experts * self.dim_expert, norm_type=config.expert_act_norm_type,
            norm_init_var=config.expert_act_norm_init_var, block_size=self.dim_expert,
        )
        if torch.distributed.get_rank() == 0 and hasattr(self.expert_act, "named_parameters"):
            for n, p in self.expert_act.named_parameters():
                print("Expert Act Parameter:", n, p.shape)
        self.expert_down_proj = build_module(
            submodules.expert_down_proj,
            self.num_experts * self.dim_expert,
            self.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            input_is_parallel=True,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="expert_down_proj",
        )

        self.use_shared_expert = self.dim_shared_expert is not None and self.dim_shared_expert > 0
        if self.use_shared_expert:
            self.shared_experts = build_module(self.submodules.shared_experts, config=self.config)

        self.enable_expert_bias = self.config.moe_router_enable_expert_bias
        self.use_target_for_loss_free = self.config.moe_use_target_for_loss_free
        self.aux_loss_coeff = self.config.moe_aux_loss_coeff
        if self.enable_expert_bias:
            self.register_buffer(
                "expert_bias", torch.zeros(self.num_experts, dtype=torch.float32)
            )
        else:
            self.expert_bias = None
        if self.use_target_for_loss_free and self.enable_expert_bias:
            self.register_buffer(
                "local_targets_per_expert",
                torch.tensor(0., dtype=torch.float32),
                persistent=False,
            )
        else:
            self.local_targets_per_expert = None
        if self.enable_expert_bias or self.aux_loss_coeff > 0:
            self.register_buffer(
                "local_tokens_per_expert",
                torch.zeros(self.num_experts, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.local_tokens_per_expert = None

    def set_layer_number(self, layer_number: int = None):
        if layer_number is None or self.layer_number is not None:
            return
        self.layer_number = layer_number
        self.moe_router.set_layer_number(layer_number)

    def _maintain_float32_expert_bias(self):
        """
        Maintain the expert bias in float32.

        When using bf16/fp16, the expert bias gets converted to lower precision in Float16Module.
        We keep it in float32 to avoid routing errors when updating the expert_bias.
        """
        if hasattr(self, "expert_bias") and self.expert_bias is not None:
            if self.expert_bias.dtype != torch.float32:
                self.expert_bias.data = self.expert_bias.data.to(torch.float32)
            if self.use_target_for_loss_free and self.local_targets_per_expert.dtype != torch.float32:
                self.local_targets_per_expert.data = self.local_targets_per_expert.data.to(torch.float32)
        if hasattr(self, "local_tokens_per_expert") and self.local_tokens_per_expert is not None:
            if self.local_tokens_per_expert.dtype != torch.float32:
                self.local_tokens_per_expert.data = self.local_tokens_per_expert.data.to(torch.float32)

    @torch.no_grad()
    def record_pre_act_info(self, pre_act: torch.Tensor):
        if not ActivationStatistics.get_value_stat_enabled():
            return
        positive_mask = torch.gt(pre_act, 0)
        ActivationStatistics.record_layer_value("pos_ratio", positive_mask.sum().item() * 100 / positive_mask.numel())
        positive_values = torch.masked_select(pre_act, positive_mask)
        negative_values = torch.masked_select(pre_act, torch.logical_not(positive_mask)).abs()
        ActivationStatistics.record_layer_value("pos_mean", positive_values.mean().item() if positive_values.numel() > 0 else 0.0)
        ActivationStatistics.record_layer_value("neg_mean", negative_values.mean().item() if negative_values.numel() > 0 else 0.0)
        ActivationStatistics.record_layer_value("all_mean", pre_act.mean().item())
        ActivationStatistics.record_layer_value("abs_mean", pre_act.abs().mean().item())

    @torch.no_grad()
    def record_post_act_info(self, post_act: torch.Tensor):
        if not ActivationStatistics.get_value_stat_enabled():
            return
        positive_mask = torch.gt(post_act, 0)
        positive_values = torch.masked_select(post_act, positive_mask)
        negative_values = torch.masked_select(post_act, torch.logical_not(positive_mask)).abs()
        ActivationStatistics.record_layer_value("post_pos_mean", positive_values.mean().item() if positive_values.numel() > 0 else 0.0)
        ActivationStatistics.record_layer_value("post_neg_mean", negative_values.mean().item() if negative_values.numel() > 0 else 0.0)
        ActivationStatistics.record_layer_value("post_all_mean", post_act.mean().item())
        ActivationStatistics.record_layer_value("post_abs_mean", post_act.abs().mean().item())
        for val in [0.05, 0.1, 0.2]:
            ActivationStatistics.record_layer_value(f"post_gt_{val}", (post_act.abs() > val).sum().item() * 100 / post_act.numel())

    def apply_load_balancing_loss(self, router_score: torch.Tensor, routing_map: torch.Tensor):
        """Calculate auxiliary loss, attach gradient function to router_score and add to logging."""
        if self.aux_loss_coeff <= 0:
            return router_score

        assert torch.all(router_score >= 0)
        probs = router_score / (torch.sum(router_score, dim=-1, keepdim=True) + 1e-5)
        assert routing_map.ndim == 2 and routing_map.shape[-1] == self.num_experts
        tokens_per_expert = routing_map.sum(dim=0)

        aux_loss = switch_load_balancing_loss_func(
            probs=probs,
            tokens_per_expert=tokens_per_expert,
            topk=self.router_target_act_ratio * self.num_experts,
            moe_aux_loss_coeff=self.aux_loss_coeff,
        )
        save_to_aux_losses_tracker(
            "load_balancing_loss",
            aux_loss / self.aux_loss_coeff,
            self.layer_number,
            self.config.num_layers,
        )
        router_score = MoEAuxLossAutoScaler.apply(router_score, aux_loss)
        return router_score

    def forward(self, hidden_states: torch.Tensor):
        self._maintain_float32_expert_bias()

        ori_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        seq_len = hidden_states.shape[0]

        # router module forward
        raw_router_score = self.moe_router(hidden_states)  # [seq_len, num_experts]
        if self.enable_expert_bias:
            scores_for_routing = LossFreeBalance.apply_expert_bias(self.expert_bias, raw_router_score)
            routing_map = torch.gt(scores_for_routing, 0)
            router_score = self.router_act(raw_router_score) * routing_map.type_as(raw_router_score)
        else:
            router_score = self.router_act(raw_router_score)
            routing_map = torch.gt(router_score, 0)

        # statistics for the activation ratio
        ActivationStatistics.record_activation_num(router_score)
        # save expert activation for loss-free sparsification
        if torch.is_grad_enabled():
            with torch.no_grad():
                if self.local_tokens_per_expert is not None:
                    assert routing_map.ndim == 2 and routing_map.shape[-1] == self.num_experts
                    self.local_tokens_per_expert += routing_map.sum(dim=0)
                if self.use_target_for_loss_free:
                    self.local_targets_per_expert += routing_map.shape[0] * self.router_target_act_ratio

        if RegRouterEntropy.is_enabled():
            assert torch.all(router_score >= 0)
            router_score_scaled = router_score / (torch.sum(router_score, dim=-1, keepdim=True) + 1e-5)
            router_score = RegRouterEntropy.apply_router_entropy(router_score, router_score_scaled)
            # router_score = RegRouterEntropy.apply_chunk_regularization(router_score, router_score_scaled)
            # router_score = RegRouterEntropy.apply_transfer_regularization(router_score, raw_router_score)
            # router_score = RegRouterEntropy.apply_l1_load_balancing(router_score, router_score > 0, self.router_target_act_ratio)
        router_score = self.apply_load_balancing_loss(router_score, routing_map)

        router_score = self.router_norm(router_score)

        # expert module forward
        x_in, bias_parallel = self.expert_up_proj(hidden_states)  # [seq_len, num_experts * dim_expert]
        assert bias_parallel is None

        if self.expert_gated:
            x_gate, bias_parallel = self.expert_gate_proj(hidden_states)
            self.record_pre_act_info(x_gate)
            x_gate = self.expert_act(x_gate)
            self.record_post_act_info(x_gate)
            if x_gate.ndim == 3:
                x_in = x_in.view(seq_len, self.num_experts, self.dim_expert)
            x_in = x_in * x_gate
        else:
            self.record_pre_act_info(x_in)
            x_in = self.expert_act(x_in)
            self.record_post_act_info(x_in)
        if x_in.ndim == 3:
            scored_x_in = x_in * router_score.type_as(hidden_states).unsqueeze(-1)
        else:
            scored_x_in = x_in.view(seq_len, self.num_experts, self.dim_expert) * router_score.type_as(hidden_states).unsqueeze(-1)

        output, bias_parallel = self.expert_down_proj(scored_x_in.view(seq_len, self.num_experts * self.dim_expert))
        assert bias_parallel is None

        if self.use_shared_expert:
            output = output + self.shared_experts(hidden_states)
        output = output.view(*ori_shape)

        return output, None
