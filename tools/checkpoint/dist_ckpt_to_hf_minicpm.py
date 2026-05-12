"""Pretrain GPT."""
import sys
sys.path.append("./")
import os
import json
import torch

from typing import Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training.initialize import initialize_megatron
from megatron.training.training import setup_model_and_optimizer
from megatron.core import parallel_state
from megatron.core.enums import ModelType
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.spec_utils import import_module
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.transformer.moe.loss_free import LossFreeBalance
# from megatron.core.transformer.blockffn.loss_free import LossFreeTokenL1
from megatron.core.transformer.blockffn.router_entropy import RegRouterEntropy


def count_parameters(args, state_dict: dict):
    cnt, all_cnt = 0, 0
    for key, val in state_dict.items():
        tokens = key.split(".")
        if "mtp" in tokens or "_extra_state" in tokens or not isinstance(val, torch.Tensor):
            continue
        if "0" in tokens or "5" in tokens or all(not t.isdigit() for t in tokens):
            print_rank_0(f"{key} {val.shape}")
        multiplier = 1
        if "experts" in tokens and args.expert_model_parallel_size > 0:
            pass
        all_cnt += val.numel() * multiplier
        if "embedding" in tokens or "output_layer" in tokens or "embed_tokens" in tokens or "lm_head" in tokens:
            continue
        cnt += val.numel() * multiplier
    return cnt, all_cnt


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if args.use_blockffn or args.router_type in ["remoe"]:
        RegRouterEntropy.initialize(
            layer_number=args.num_layers,
            reg_coef_init=args.router_entropy_loss_coeff,
            reg_coef_multiplier=args.router_entropy_loss_coeff_multiplier,
            res_coef_resume=args.router_entropy_loss_coeff_resume,
            transfer_lambda=args.transfer_loss_coeff,
        )
    LossFreeBalance.set_update_method(args.moe_expert_bias_update_method)
    LossFreeBalance.set_apply_method(args.moe_expert_bias_apply_method)
    LossFreeBalance.set_print_expert_bias_step(args.print_expert_bias_step)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te, normalization=args.normalization, use_blockffn=args.use_blockffn)
            elif args.heterogeneous_layers_config_path is not None:
                transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm, use_blockffn=args.use_blockffn)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm,
                        normalization=args.normalization, use_blockffn=args.use_blockffn)
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, use_transformer_engine=use_te)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
        )
    non_embed_cnt, embed_cnt = count_parameters(args, model.state_dict())
    print_rank_0(">" * 6 + f"[non-embedding parameters: {non_embed_cnt}; embedding parameters: {embed_cnt}]" + "<" * 6)

    return model


if __name__ == "__main__":

    initialize_megatron(extra_args_provider=None,
                        args_defaults={'tokenizer_type': 'Llama2Tokenizer'})
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(model_provider, model_type=ModelType.encoder_or_decoder)
    args = get_args()
    model_real = model[0]

    # extract megatron state dict
    state_dict_megatron = model_real.state_dict()
    new_sd = dict()
    print(">" * 10, "before conversion", "<" * 10)
    for k in state_dict_megatron:
        if "_extra" in k:
            continue
        new_sd[k] = state_dict_megatron[k]
    non_embed_cnt_0, embed_cnt_0 = count_parameters(args, state_dict_megatron)
    print_rank_0(f"[non-embedding parameters: {non_embed_cnt_0}; embedding parameters: {embed_cnt_0}]")
    print(">" * 10, "before conversion", "<" * 10)

    # param name mapping
    state_dict_hf = dict()
    ep_idx = None

    state_dict_hf["model.embed_tokens.weight"] = state_dict_megatron["embedding.word_embeddings.weight"]
    state_dict_hf["model.norm.weight"] = state_dict_megatron["decoder.final_layernorm.weight"]
    if args.untie_embeddings_and_output_weights:
        state_dict_hf["lm_head.weight"] = state_dict_megatron["output_layer.weight"]

    assert args.num_attention_heads % args.num_query_groups == 0
    assert args.hidden_size % args.num_attention_heads == 0

    num_query_heads_per_group = args.num_attention_heads // args.num_query_groups
    kv_channels = args.kv_channels if args.kv_channels is not None else (args.hidden_size // args.num_attention_heads)
    moe_layer_freq = eval(args.moe_layer_freq) if isinstance(args.moe_layer_freq, str) else args.moe_layer_freq

    for layer_idx in range(args.num_layers):
        state_dict_hf[f"model.layers.{layer_idx}.input_layernorm.weight"] = state_dict_megatron[f"decoder.layers.{layer_idx}.self_attention.linear_qkv.layer_norm_weight"]
        qkv_proj = state_dict_megatron[f"decoder.layers.{layer_idx}.self_attention.linear_qkv.weight"]
        qkv_proj_split = torch.split(qkv_proj, split_size_or_sections=kv_channels, dim=0)

        q_proj_list, k_proj_list, v_proj_list = [], [], []
        for i in range(args.num_query_groups):
            q_proj_list.extend(qkv_proj_split[(num_query_heads_per_group + 2) * i: (num_query_heads_per_group + 2) * i + num_query_heads_per_group])
            k_proj_list.append(qkv_proj_split[(num_query_heads_per_group + 2) * i + num_query_heads_per_group])
            v_proj_list.append(qkv_proj_split[(num_query_heads_per_group + 2) * i + num_query_heads_per_group + 1])

        q_proj = torch.cat(q_proj_list, dim=0)
        k_proj = torch.cat(k_proj_list, dim=0)
        v_proj = torch.cat(v_proj_list, dim=0)

        state_dict_hf[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = q_proj
        state_dict_hf[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = k_proj
        state_dict_hf[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = v_proj
        state_dict_hf[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = state_dict_megatron[f"decoder.layers.{layer_idx}.self_attention.linear_proj.weight"]

        if not args.num_experts or args.moe_layer_freq[layer_idx] == 0:
            state_dict_hf[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = state_dict_megatron[
                f"decoder.layers.{layer_idx}.mlp.linear_fc1.layer_norm_weight"]
            linear1_fc_weight = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.linear_fc1.weight"]
            gate_proj, up_proj = torch.split(linear1_fc_weight,
                                             split_size_or_sections=(linear1_fc_weight.shape[0] // 2), dim=0)
            state_dict_hf[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = gate_proj
            state_dict_hf[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = up_proj
            state_dict_hf[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = state_dict_megatron[
                f"decoder.layers.{layer_idx}.mlp.linear_fc2.weight"]
        elif not args.use_blockffn:
            # router
            state_dict_hf[f"model.layers.{layer_idx}.mlp.router.weight"] = state_dict_megatron[
                f"decoder.layers.{layer_idx}.mlp.router.weight"]
            if args.moe_router_enable_expert_bias:
                state_dict_hf[f"model.layers.{layer_idx}.mlp.router.expert_bias"] = state_dict_megatron[
                    f"decoder.layers.{layer_idx}.mlp.router.expert_bias"]
            # shared experts
            state_dict_hf[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = state_dict_megatron[
                f"decoder.layers.{layer_idx}.pre_mlp_layernorm.weight"]
            if args.moe_shared_expert_intermediate_size is not None:
                linear1_fc_weight = state_dict_megatron[
                    f"decoder.layers.{layer_idx}.mlp.shared_experts.linear_fc1.weight"]
                gate_proj, up_proj = torch.split(linear1_fc_weight,
                                                 split_size_or_sections=(linear1_fc_weight.shape[0] // 2), dim=0)
                state_dict_hf[f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"] = gate_proj
                state_dict_hf[f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"] = up_proj
                state_dict_hf[f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"] = state_dict_megatron[
                    f"decoder.layers.{layer_idx}.mlp.shared_experts.linear_fc2.weight"]

            gate_proj, up_proj, down_proj = [], [], []
            for idx_expert in range(args.num_experts):
                if args.expert_not_gated:
                    up_proj.append(state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.experts.linear_fc1.weight{idx_expert}"])
                else:
                    linear1_fc_weight = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.experts.linear_fc1.weight{idx_expert}"]
                    gate_w, up_w = torch.split(linear1_fc_weight,
                                                     split_size_or_sections=(linear1_fc_weight.shape[0] // 2), dim=0)
                    gate_proj.append(gate_w)
                    up_proj.append(up_w)
                down_proj.append(state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.experts.linear_fc2.weight{idx_expert}"])
            if not args.expert_not_gated:
                state_dict_hf[f"model.layers.{layer_idx}.mlp.experts.gate_proj.weight"] = torch.stack(gate_proj)
            state_dict_hf[f"model.layers.{layer_idx}.mlp.experts.up_proj.weight"] = torch.stack(up_proj)
            state_dict_hf[f"model.layers.{layer_idx}.mlp.experts.down_proj.weight"] = torch.stack(down_proj)
        else:
            # BlockFFN conversion
            # router
            state_dict_hf[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = state_dict_megatron[f"decoder.layers.{layer_idx}.pre_mlp_layernorm.weight"]
            state_dict_hf[f"model.layers.{layer_idx}.mlp.moe_router.weight"] = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.moe_router.weight"]
            if not args.router_norm_fixed:
                state_dict_hf[f"model.layers.{layer_idx}.mlp.router_norm.weight"] = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.router_norm.weight"]

            # experts
            if not args.expert_not_gated:
                state_dict_hf[f"model.layers.{layer_idx}.mlp.expert_gate_proj.weight"] = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.expert_gate_proj.weight"]
            state_dict_hf[f"model.layers.{layer_idx}.mlp.expert_up_proj.weight"] = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.expert_up_proj.weight"]
            state_dict_hf[f"model.layers.{layer_idx}.mlp.expert_down_proj.weight"] = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.expert_down_proj.weight"]
            if args.expert_act_func in ["norm_silu", "norm_silu_nomean"]:
                state_dict_hf[f"model.layers.{layer_idx}.mlp.expert_act.rms_norm.weight"] = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.expert_act.rms_norm.weight"]

            # shared experts
            if args.moe_shared_expert_intermediate_size is not None:
                linear1_fc_weight = state_dict_megatron[
                    f"decoder.layers.{layer_idx}.mlp.shared_experts.linear_fc1.weight"]
                gate_proj, up_proj = torch.split(linear1_fc_weight,
                                                 split_size_or_sections=(linear1_fc_weight.shape[0] // 2), dim=0)
                state_dict_hf[f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"] = gate_proj
                state_dict_hf[f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"] = up_proj
                state_dict_hf[f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"] = state_dict_megatron[
                    f"decoder.layers.{layer_idx}.mlp.shared_experts.linear_fc2.weight"]

            if args.moe_router_enable_expert_bias:
                state_dict_hf[f"model.layers.{layer_idx}.mlp.expert_bias"] = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.expert_bias"]

    # save and generate hf repository
    os.makedirs(args.save, exist_ok=True)
    for k in state_dict_hf:
        state_dict_hf[k] = state_dict_hf[k].clone().contiguous()

    assert os.system(f"cp -r examples/minicpm4/base_files/* {args.save}/.") == 0
    # update configuration
    config_path = os.path.join(args.save, "config.json")
    with open(config_path, "r") as f:
        config_hf = json.load(f)
    update_keys = [
        "num_layers", "hidden_size", "ffn_hidden_size", "num_attention_heads", "num_query_groups",
        "norm_epsilon", "router_norm_fixed", "router_norm_scalar", "router_norm_init_var", "moe_expert_bias_apply_method",
        "use_blockffn", "num_experts", "moe_ffn_hidden_size", "moe_shared_expert_intermediate_size",
        "moe_layer_freq", "router_act_func", "router_norm_type", "expert_act_func", "router_type",
        "expert_act_norm_type", "moe_router_enable_expert_bias", "expert_not_gated", "moe_router_pre_softmax",
        "moe_router_topk", "moe_router_topp", "moe_router_score_function", "moe_router_topk_scaling_factor", "use_mup",
    ]
    for key in update_keys:
        assert hasattr(args, key)
        config_hf[key] = getattr(args, key)
    with open(config_path, "w") as f:
        json.dump(config_hf, f, indent=4)

    print("+" * 10, "after conversion", "+" * 10)
    non_embed_cnt, embed_cnt = count_parameters(args, state_dict_hf)
    assert non_embed_cnt_0 == non_embed_cnt and embed_cnt_0 == embed_cnt
    print_rank_0(f"[non-embedding parameters: {non_embed_cnt}; embedding parameters: {embed_cnt}]")
    print("+" * 10, "after conversion", "+" * 10)

    if args.expert_model_parallel_size == 1:
        torch.save(state_dict_hf, os.path.join(args.save, "pytorch_model.bin"))
    else:
        assert ep_idx is not None
        torch.save(state_dict_hf, os.path.join(args.save, f"pytorch_model.ep_{ep_idx}.bin"))
