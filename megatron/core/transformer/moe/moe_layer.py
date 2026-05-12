# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union
from copy import deepcopy

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import MoEAlltoAllSEQTokenDispatcher
from megatron.core.transformer.moe.router import TopKRouter, ReMoERouter, TopPRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
)
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler, save_to_aux_losses_tracker
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig, layer_number: Optional[int] = None):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router: TopKRouter = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)
        # Propagate the layer index to the experts so that the per-expert
        # health metrics tracker (see ``moe.expert_metrics``) can key its
        # recordings by layer. Implementations that don't track metrics
        # simply omit ``set_layer_number``, making this a no-op for them.
        if self.experts is not None and hasattr(self.experts, "set_layer_number"):
            self.experts.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
    ):
        self.submodules = submodules
        super(MoELayer, self).__init__(config=config, layer_number=layer_number)
        self.moe_layer_recompute = (
            config.recompute_granularity == "selective" and "moe" in config.recompute_modules
        )

        # Initialize router
        if config.router_type == "topk":
            self.router = TopKRouter(config=self.config, layer_number=layer_number)
        elif config.router_type == "remoe":
            self.router = ReMoERouter(config=self.config, layer_number=layer_number)
        elif config.router_type == "topp":
            self.router = TopPRouter(config=self.config, layer_number=layer_number)
        else:
            raise NotImplementedError(f"Router type {config.router_type} not implemented.")

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall_seq":
            self.token_dispatcher = MoEAlltoAllSEQTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        expert_config = deepcopy(self.config)
        expert_config.gated_linear_unit = self.config.gated_linear_unit and not self.config.expert_not_gated
        expert_config.bias_activation_fusion = self.config.bias_activation_fusion and not self.config.expert_not_gated
        self.experts = build_module(self.submodules.experts, self.num_local_experts, expert_config)

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(self.submodules.shared_experts, config=self.config)
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward(hidden_states):
            probs, routing_map, dense_probs = self.router(hidden_states)
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                hidden_states, probs, routing_map
            )
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)

            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)

            # if self.proxy_experts is not None and self.training and torch.is_grad_enabled():
            # DenseMixer Implementation
            if self.config.dense_mixer_enabled and self.training and torch.is_grad_enabled():
                assert isinstance(self.token_dispatcher, MoEAllGatherTokenDispatcher), "Proxy experts only work with MoEAllGatherTokenDispatcher"

                dim0, dim1, hidden_size = hidden_states.shape
                flat_hidden = hidden_states.view(-1, hidden_size)

                with torch.no_grad():
                    linear_fc1, linear_fc2 = [], []
                    for eid in range(self.num_local_experts):
                        linear_fc1.append(getattr(self.experts.linear_fc1, f"weight{eid}"))
                        linear_fc2.append(getattr(self.experts.linear_fc2, f"weight{eid}").T.unsqueeze(0))
                    linear_fc1, linear_fc2 = torch.cat(linear_fc1, dim=0), torch.cat(linear_fc2, dim=0)

                    fc1_out = torch.nn.functional.linear(flat_hidden, linear_fc1)
                    fc1_out = fc1_out.view(flat_hidden.shape[0], self.num_local_experts, self.config.moe_ffn_hidden_size * 2)
                    fc1_out_1, fc1_out_2 = torch.chunk(fc1_out, 2, dim=-1)
                    hidden = torch.nn.functional.silu(fc1_out_1) * fc1_out_2
                    dense_proxies = torch.einsum("bei,eih->beh", hidden, linear_fc2)

                    # self.proxy_experts.linear_fc1.copy_(linear_fc1)
                    # self.proxy_experts.linear_fc2.copy_(linear_fc2)
                    # dense_proxies = self.proxy_experts(flat_hidden)

                # Prevent Double-Gradients: Mask out the Top-K experts
                flat_routing_map = routing_map.view(-1, self.num_local_experts)
                unselected_mask = 1.0 - flat_routing_map.to(dense_probs.dtype)
                flat_dense_probs = dense_probs.view(-1, self.num_local_experts)
                unselected_probs = flat_dense_probs * unselected_mask

                weighted_for_ste = (dense_proxies.detach() * unselected_probs.unsqueeze(-1)).sum(dim=1)
                weighted_for_ste = weighted_for_ste.view(dim0, dim1, hidden_size)
                output = output + (weighted_for_ste - weighted_for_ste.detach())

            if self.use_shared_expert and not self.shared_expert_overlap:
                # if shared_expert_overlap is True, the expert calculation happens in
                # the token_dispatcher to overlap communications and computations
                output = output + self.shared_experts(hidden_states)
            return output, mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias
