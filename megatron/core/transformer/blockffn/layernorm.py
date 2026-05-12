import torch
from typing import Union
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


def rms_layernorm(hidden: torch.Tensor, weight: Union[torch.Tensor, float], eps: float) -> torch.Tensor:
    old_dtype = hidden.dtype
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


def normal_layernorm(hidden: torch.Tensor, weight: Union[torch.Tensor, float], eps: float) -> torch.Tensor:
    old_dtype = hidden.dtype
    hidden = hidden - torch.mean(hidden, dim=-1, keepdim=True)
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


def block_normal_layernorm(hidden: torch.Tensor, weight: Union[torch.Tensor, float], eps: float, num_blocks: int, block_size: int) -> torch.Tensor:
    old_dtype = hidden.dtype
    assert hidden.ndim == 2
    hidden = hidden - torch.mean(hidden, dim=-1, keepdim=True)
    hidden = hidden.view(hidden.shape[0], num_blocks, block_size)
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    if weight.ndim == 1:
        return hidden * weight.view(num_blocks, block_size)
    else:
        return hidden * weight


class CustomLayerNorm(MegatronModule):
    """Custom LayerNorm"""
    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        init_var: float = 1.0,
        norm_type: str = "rms",
        fixed: bool = False,
        block_size: int = -1,
    ):
        super().__init__(config)

        self.dim_norm = hidden_size
        self.eps = eps
        self.fixed = fixed
        self.init_var = init_var
        if self.fixed:
            self.weight = init_var
        else:
            self.weight = torch.nn.Parameter(torch.full((self.dim_norm,), self.init_var))
        assert norm_type in ["rms", "normal", "simple", "null", "block_normal"]
        self.norm_type = norm_type
        if self.norm_type == "block_normal":
            self.block_size = block_size
            assert self.dim_norm % self.block_size == 0
            self.num_blocks = self.dim_norm // self.block_size

    @torch.compile
    def forward(self, x: torch.Tensor):
        if self.norm_type == "rms":
            res = rms_layernorm(x, self.weight, self.eps)
        elif self.norm_type == "normal":
            res = normal_layernorm(x, self.weight, self.eps)
        elif self.norm_type == "block_normal":
            res = block_normal_layernorm(x, self.weight, self.eps, self.num_blocks, self.block_size)
        elif self.norm_type == "simple":
            res = x * self.weight
        else:
            res = x
        return res
