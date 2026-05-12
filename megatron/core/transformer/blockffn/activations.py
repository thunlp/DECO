import torch
import torch.nn.functional as F
from typing import Optional, Union
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.module import MegatronModule


class SquaredReLU(torch.nn.Module):
    def forward(self, x):
        return torch.square(F.relu(x))


class NullAct(torch.nn.Module):
    def forward(self, x):
        return x


class NormSiLU(MegatronModule):
    def __init__(self, config: TransformerConfig, norm_module: Union[ModuleSpec, type],
                 norm_dim: Optional[int] = None, norm_type: Optional[str] = None,
                 norm_init_var: Optional[float] = None, block_size: int = -1, activate_fn_type: str = "norm_silu"):
        super().__init__(config)
        assert config is not None and norm_module is not None and norm_dim is not None
        self.dim_norm = norm_dim
        self.init_var = norm_init_var
        self.eps = config.layernorm_epsilon
        self.block_size = block_size
        assert block_size > 0 and norm_dim % block_size == 0
        self.num_blocks = norm_dim // block_size
        assert norm_type == "normal"
        self.activate_fn_type = activate_fn_type
        assert self.activate_fn_type in ["norm_silu", "norm_silu_norms", "norm_silu_nomean", "silu"]

        assert self.init_var == 1.0
        self.rms_norm = None
        if self.activate_fn_type not in ["norm_silu_norms", "silu"]:
            self.rms_norm = build_module(
                norm_module, config=config, hidden_size=block_size, eps=self.eps,
            )
        self.silu = torch.nn.SiLU()

    @torch.compile
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        assert hidden.ndim == 2
        if self.activate_fn_type not in ["norm_silu_nomean", "silu"]:
            hidden = hidden - torch.mean(hidden, dim=-1, keepdim=True)
        if self.activate_fn_type not in ["norm_silu_norms", "silu"]:
            return self.silu(self.rms_norm(hidden.view(hidden.shape[0], self.num_blocks, self.block_size)))
        else:
            return self.silu(hidden)


def get_activation_fn(
    activate_fn: str, config: Optional[TransformerConfig] = None,
    norm_module: Union[ModuleSpec, type] = None, norm_dim: Optional[int] = None,
    norm_type: Optional[str] = None, norm_init_var: Optional[float] = None, block_size: int = -1,
):
    if activate_fn == "gelu":
        act = torch.nn.GELU()
    elif activate_fn == "silu":
        act = torch.nn.functional.silu
    elif activate_fn == "relu":
        act = torch.nn.ReLU()
    elif activate_fn == "sqrelu":
        act = SquaredReLU()
    elif activate_fn == "sigmoid":
        act = torch.nn.Sigmoid()
    elif activate_fn in ["norm_silu", "norm_silu_norms", "norm_silu_nomean"]:
        act = NormSiLU(config, norm_module, norm_dim, norm_type, norm_init_var, block_size, activate_fn)
    elif activate_fn == "null":
        act = NullAct()
    else:
        raise NotImplementedError(f"{activate_fn} is not supported")
    return act
