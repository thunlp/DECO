# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import torch

from megatron.core import parallel_state
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    save_to_aux_losses_tracker,
    sequence_load_balancing_loss_func,
    switch_load_balancing_loss_func,
    topk_softmax_with_capacity,
    z_loss_func,
    group_limited_topk,
)
from typing import Optional
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.blockffn.statistics import ActivationStatistics
from megatron.core.transformer.blockffn.router_entropy import RegRouterEntropy
from .loss_free import LossFreeBalance


class Router(ABC, MegatronModule):
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
        self.moe_aux_loss_func = None
        self.layer_number = None

        # Initialize the gate weights.
        # TODO: Add support for GPU initialization, which requires updating the golden values.
        self.weight = torch.nn.Parameter(
            torch.empty((self.config.num_moe_experts, self.config.hidden_size), dtype=torch.float32)
        )
        if config.perform_initialization:
            config.init_method(self.weight)
        self.weight.data = self.weight.data.to(dtype=config.params_dtype)
        setattr(self.weight, 'sequence_parallel', config.sequence_parallel)
        # If calculate per token loss, we need to scale up moe aux loss by the number of tokens.
        # So we need to know if the model is configured to calculate per token loss.
        self.calculate_per_token_loss = self.config.calculate_per_token_loss

    def gating(self, input: torch.Tensor, weight: Optional[torch.Tensor] = None):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        if weight is None:
            weight = self.weight
        if weight.device.type == 'cpu':
            # move weights to GPU
            weight.data = weight.data.to(device=torch.cuda.current_device())
        # Convert to specified datatype for routing computation if enabled
        router_dtype = input.dtype
        if self.config.moe_router_dtype == 'fp32':
            router_dtype = torch.float32
        elif self.config.moe_router_dtype == 'fp64':
            router_dtype = torch.float64
        logits = torch.nn.functional.linear(input.to(router_dtype), weight.to(router_dtype))
        return logits

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        """Routing function.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mapping.
        """
        raise NotImplementedError("Routing function not implemented.")

    @abstractmethod
    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        raise NotImplementedError("Forward function not implemented.")

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the router."""
        self.layer_number = layer_number


class TopKRouter(Router):
    """Route each token to the top-k experts."""

    def __init__(self, config: TransformerConfig, layer_number: Optional[int] = None) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config)
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.score_function = self.config.moe_router_score_function
        self.input_jitter = None
        self.aux_loss_coeff = config.moe_aux_loss_coeff

        self.enable_expert_bias = self.config.moe_router_enable_expert_bias
        if self.enable_expert_bias:
            self.register_buffer(
                'expert_bias', torch.zeros(self.config.num_moe_experts, dtype=torch.float32)
            )
        else:
            self.expert_bias = None
        if self.enable_expert_bias or self.aux_loss_coeff > 0:
            self.register_buffer(
                "local_tokens_per_expert",
                torch.zeros(self.num_experts, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.local_tokens_per_expert = None

    def _maintain_float32_expert_bias(self):
        """
        Maintain the expert bias in float32.

        When using bf16/fp16, the expert bias gets converted to lower precision in Float16Module.
        We keep it in float32 to avoid routing errors when updating the expert_bias.
        """
        if hasattr(self, 'expert_bias') and self.expert_bias is not None:
            if self.expert_bias.dtype != torch.float32:
                self.expert_bias.data = self.expert_bias.data.to(torch.float32)
        if hasattr(self, "local_tokens_per_expert") and self.local_tokens_per_expert is not None:
            if self.local_tokens_per_expert.dtype != torch.float32:
                self.local_tokens_per_expert.data = self.local_tokens_per_expert.data.to(torch.float32)

    def compute_routing_scores_for_aux_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute routing scores based on the score function.

        Args:
            logits (torch.Tensor): The logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            torch.Tensor: The normalized routing scores.
        """
        if self.score_function == "softmax":
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
        elif self.score_function == "sigmoid":
            scores = torch.sigmoid(logits)
            scores = (
                scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.topk > 1 else scores
            )
        else:
            raise ValueError(f"Invalid score_function: {self.score_function}")
        return scores

    def aux_loss_load_balancing(self, logits: torch.Tensor):
        """Apply auxiliary loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mask of token to experts assignment.
        """
        probs, routing_map, tokens_per_expert, dense_probs = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            deterministic_mode=self.config.deterministic_mode,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
        )

        if self.training and torch.is_grad_enabled():
            # Apply auxiliary load balancing loss
            # Skip auxiliary loss calculations when using torch.no_grad() or checkpointing.
            scores = self.compute_routing_scores_for_aux_loss(logits)
            aux_loss_func = partial(
                switch_load_balancing_loss_func,
                probs=scores,
                tokens_per_expert=tokens_per_expert,
                topk=self.topk,
            )
            probs = self.apply_load_balancing_loss(
                activation=probs, load_balancing_loss_func=aux_loss_func
            )
        return probs, routing_map, dense_probs

    def seq_aux_loss_load_balancing(self, logits: torch.Tensor, bsz: int, seq_length: int):
        """Apply sequence-auxiliary loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor after gating, shape: [num_tokens, num_experts].
            bsz (int): The batch size.
            seq_length (int): The sequence length.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mask of token to experts assignment.
        """

        probs, routing_map, tokens_per_expert, dense_probs = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            deterministic_mode=self.config.deterministic_mode,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
        )

        if self.training and torch.is_grad_enabled():
            # Apply sequence-auxiliary load balancing loss
            scores = self.compute_routing_scores_for_aux_loss(logits)
            aux_loss_func = partial(
                sequence_load_balancing_loss_func,
                probs=scores,
                routing_map=routing_map,
                batch_size=bsz,
                seq_length=seq_length,
                topk=self.topk,
            )
            probs = self.apply_load_balancing_loss(
                activation=probs, load_balancing_loss_func=aux_loss_func
            )

        return probs, routing_map, dense_probs

    def apply_load_balancing_loss(
        self, activation: torch.Tensor, load_balancing_loss_func: Callable
    ):
        """Calculate auxiliary loss, attach gradient function to activation and add to logging."""
        moe_aux_loss_coeff = self.aux_loss_coeff
        if moe_aux_loss_coeff == 0:
            return activation
        sequence_partition_group = None
        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            sequence_partition_group = parallel_state.get_context_parallel_group()
            moe_aux_loss_coeff /= parallel_state.get_tensor_model_parallel_world_size()
        elif parallel_state.get_tensor_and_context_parallel_world_size() > 1:
            sequence_partition_group = parallel_state.get_tensor_and_context_parallel_group()

        aux_loss = load_balancing_loss_func(
            moe_aux_loss_coeff=moe_aux_loss_coeff, sequence_partition_group=sequence_partition_group
        )
        save_to_aux_losses_tracker(
            "load_balancing_loss",
            aux_loss / moe_aux_loss_coeff,
            self.layer_number,
            self.config.num_layers,
            reduce_group=sequence_partition_group,
        )
        if self.calculate_per_token_loss:
            # Scale the aux_loss by the number of tokens.
            # The expected final scaling for aux_loss gradients is 1/(num_micro_batches * dp_size).
            # After commit 02648000, Megatron started using the number of total tokens to scale
            # gradients under the argument of calculate_per_token_loss,
            # which scales both the main_loss gradient and aux_loss gradient by
            # 1/(num_local_tokens * dp_size * num_micro_batches) in finalize_model_grads function.
            # To correct this scaling, we need to scale the aux_loss by num_local_tokens here.
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss * activation.shape[0])
        else:
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None and self.training and torch.is_grad_enabled():
            # Skip Z loss calculations when using torch.no_grad() or checkpointing.
            moe_z_loss_coeff = (
                self.config.moe_z_loss_coeff
                / parallel_state.get_tensor_and_context_parallel_world_size()
            )
            z_loss = z_loss_func(logits, moe_z_loss_coeff)
            scale_up = 1.0
            if self.calculate_per_token_loss:
                # The expected final scaling for z_loss gradients is
                # 1/(num_micro_batches * dp_size).
                # After commit 02648000, Megatron started using the number of total tokens
                # to scale gradients under the argument of calculate_per_token_loss,
                # which scales both the main_loss gradient and z_loss gradient by
                # 1/(num_local_tokens * dp_size * num_micro_batches) in finalize_model_grads().
                # To correct this scaling, we need to scale the z_loss by num_local_tokens here.
                logits = MoEAuxLossAutoScaler.apply(logits, z_loss * logits.shape[0])
            else:
                logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
            save_to_aux_losses_tracker(
                "z_loss", z_loss / moe_z_loss_coeff, self.layer_number, self.config.num_layers
            )
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            # Gather the logits from the TP region
            logits = gather_from_sequence_parallel_region(logits)

        if self.routing_type == "aux_loss":
            scores, routing_map, dense_probs = self.aux_loss_load_balancing(logits)
        elif self.routing_type == "seq_aux_loss":
            scores, routing_map, dense_probs = self.seq_aux_loss_load_balancing(logits, bsz, seq_length)
        elif self.routing_type == "none":
            # A naive top-k routing without load balancing
            scores, routing_map, _, dense_probs = topk_softmax_with_capacity(
                logits,
                self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
                drop_policy=self.config.moe_token_drop_policy,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                num_groups=self.config.moe_router_num_groups,
                group_topk=self.config.moe_router_group_topk,
                scaling_factor=self.config.moe_router_topk_scaling_factor,
                deterministic_mode=self.config.deterministic_mode,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
            )
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")
        # Prevent extra local tokens accumulation on evaluation or activation recomputation
        if hasattr(self, "local_tokens_per_expert") and self.local_tokens_per_expert is not None and torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert += routing_map.sum(dim=0)

        return scores, routing_map, dense_probs

    def forward(self, inputs: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self._maintain_float32_expert_bias()

        # Apply input jitter
        inputs = self.apply_input_jitter(inputs)
        logits = self.gating(inputs, weight=self.weight)
        scores, routing_map, dense_probs = self.routing(logits)

        """
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                sigmoid_score = torch.sigmoid(logits.float()).type_as(logits)
                overall_sum = sigmoid_score.sum(dim=-1).mean()
                save_to_aux_losses_tracker(
                    "sum_of_all_logits",
                    overall_sum,
                    self.layer_number,
                    self.config.num_layers,
                )
                topk_sum = torch.topk(sigmoid_score, k=self.topk, dim=-1)[0].sum(dim=-1).mean()
                save_to_aux_losses_tracker(
                    "sum_of_topk_logits",
                    topk_sum,
                    self.layer_number,
                    self.config.num_layers,
                )
        """

        return scores, routing_map, dense_probs


class ReMoERouter(Router):
    def __init__(self, config: TransformerConfig, layer_number: Optional[int] = None) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config)
        self.router_act = torch.nn.ReLU()
        self.target_act_ratio = config.router_target_act_ratio
        self.clamp_act_ratio = config.router_clamp_act_ratio

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        logits = logits.view(-1, self.config.num_moe_experts)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            # Gather the logits from the TP region
            logits = gather_from_sequence_parallel_region(logits)

        router_score = self.router_act(logits)
        routing_map = router_score > 0
        ActivationStatistics.record_activation_num(router_score)

        if RegRouterEntropy.is_enabled():
            router_score = RegRouterEntropy.apply_l1_load_balancing(router_score, routing_map, self.target_act_ratio)

        if 0 < self.clamp_act_ratio < 1.0:
            max_experts_per_token = int(self.clamp_act_ratio * self.config.num_moe_experts)
            active_experts_count = routing_map.sum(dim=-1)

            if (active_experts_count > max_experts_per_token).any():
                topk_vals, topk_indices = torch.topk(router_score, max_experts_per_token, dim=-1)
                routing_map = torch.zeros_like(routing_map).scatter_(-1, topk_indices, topk_vals > 0)

        # sorted_probs, sorted_indices = torch.sort(router_score, descending=True, dim=-1)
        # sorted_map = sorted_probs <= 0
        # sorted_indices = torch.where(sorted_map, -1, sorted_indices)
        # max_valid_num = max(sorted_probs.size(-1) - torch.min(torch.sum(sorted_map, dim=-1)).item(), 1)
        # assert torch.all(sorted_map[:, max_valid_num:])
        # sorted_probs = sorted_probs[:, :max_valid_num]
        # sorted_indices = sorted_indices[:, :max_valid_num]
        # assert torch.sum(routing_map) == torch.sum(sorted_indices != -1)
        # scores = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_probs)
        # routing_map = torch.zeros_like(logits).int().scatter(1, sorted_indices, 1).bool()
        return router_score, routing_map, logits

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)

        scores, routing_map, dense_probs = self.routing(logits)

        return scores, routing_map, dense_probs


class TopPRouter(Router):
    def __init__(self, config: TransformerConfig, layer_number: Optional[int] = None) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config)
        self.top_p = config.moe_router_topp

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        logits = logits.view(-1, self.config.num_moe_experts)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            # Gather the logits from the TP region
            logits = gather_from_sequence_parallel_region(logits)

        router_score = torch.abs(logits)
        router_score = router_score / (router_score.sum(dim=-1, keepdim=True) + 1e-20)

        sorted_probs, sorted_indices = torch.sort(router_score, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > self.top_p

        threshold_indices = mask.long().argmax(dim=-1)
        threshold_mask = torch.nn.functional.one_hot(threshold_indices, num_classes=sorted_indices.size(-1)).bool()

        mask = mask & ~threshold_mask
        sorted_probs = torch.where(mask, 0.0, sorted_probs)

        topk_masked_gates = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_probs)
        topk_map = torch.gt(topk_masked_gates, 0.0)
        ActivationStatistics.record_activation_num(topk_masked_gates)

        return topk_masked_gates, topk_map, router_score

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)

        scores, routing_map, dense_probs = self.routing(logits)

        return scores, routing_map, dense_probs
