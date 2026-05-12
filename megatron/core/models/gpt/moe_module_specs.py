# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import warnings
from typing import Optional

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import get_te_version, is_te_min_version
from megatron.core.transformer.blockffn.blockffn import BlockFFNLayer, BlockFFNSubmodules, RouterFP32
from megatron.core.transformer.blockffn.layernorm import CustomLayerNorm
from megatron.core.transformer.torch_norm import WrappedTorchNorm

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
        TENorm,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex  # pylint: disable=unused-import
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
    LNImpl = FusedLayerNorm
except ImportError:
    warnings.warn('Apex is not installed. Falling back to Torch Norm')
    LNImpl = WrappedTorchNorm


def get_moe_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    use_blockffn: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
    assert num_experts is not None
    if use_te:
        assert HAVE_TE
        layer_norm_impl = TENorm
    else:
        layer_norm_impl = LNImpl

    mlp = MLPSubmodules(
        linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
        linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
    )
    # shared experts spec
    shared_experts = ModuleSpec(module=SharedExpertMLP, params={"gate": False}, submodules=mlp)

    if use_blockffn:
        if use_te:
            # noinspection PyTypeChecker
            return ModuleSpec(
                module=BlockFFNLayer,
                submodules=BlockFFNSubmodules(
                    moe_router=RouterFP32,
                    router_norm=CustomLayerNorm,
                    expert_gate_proj=TEColumnParallelLinear,
                    expert_up_proj=TEColumnParallelLinear,
                    expert_act_norm=layer_norm_impl,
                    expert_down_proj=TERowParallelLinear,
                    shared_experts=shared_experts,
                )
            )
        else:
            # noinspection PyTypeChecker
            return ModuleSpec(
                module=BlockFFNLayer,
                submodules=BlockFFNSubmodules(
                    moe_router=RouterFP32,
                    router_norm=CustomLayerNorm,
                    expert_gate_proj=ColumnParallelLinear,
                    expert_up_proj=ColumnParallelLinear,
                    expert_act_norm=layer_norm_impl,
                    expert_down_proj=RowParallelLinear,
                    shared_experts=shared_experts,
                )
            )

    # experts spec
    if moe_grouped_gemm:
        ## use GroupedMLP
        if use_te and TEColumnParallelGroupedLinear is not None and not moe_use_legacy_grouped_gemm:
            ## use TEGroupedLinear
            expert_module = TEGroupedMLP
            expert_submodule = MLPSubmodules(
                linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
            )
        else:
            ## use legacy GroupedMLP
            expert_module = GroupedMLP
            expert_submodule = None
            warnings.warn(
                'The legacy GroupedMLP will be deprecated in Megatron-Core v0.12.0. '
                'Please update the TransformerEngine to version>=1.7.0 and use TEGroupedMLP.'
            )
    else:
        ## use SequentialMLP
        expert_module = SequentialMLP
        if use_te and not is_te_min_version("1.7.0.dev0"):
            warnings.warn(
                "Only transformer-engine>=1.7.0 supports MoE experts, "
                f"but your version is {get_te_version()}. Use local linear implementation instead."
            )
            expert_submodule = MLPSubmodules(
                linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear
            )
        else:
            expert_submodule = mlp

    experts = ModuleSpec(module=expert_module, submodules=expert_submodule)

    # MoE module spec
    moe_module_spec = ModuleSpec(
        module=MoELayer, submodules=MoESubmodules(experts=experts, shared_experts=shared_experts)
    )
    return moe_module_spec
