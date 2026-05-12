import torch
from typing import Tuple
from megatron.core import mpu
from collections import defaultdict


class ActivationStatistics:

    _activation_num: int = 0
    _total_num: int = 0
    _value_stat_enabled: bool = False
    _value_container: defaultdict = defaultdict(list)

    @classmethod
    def set_value_stat_enabled(cls, enabled: bool):
        cls._value_stat_enabled = enabled

    @classmethod
    def get_value_stat_enabled(cls) -> bool:
        return cls._value_stat_enabled

    @classmethod
    @torch.no_grad()
    def record_activation_num(cls, activation: torch.Tensor):
        loc_act, loc_tot = int(torch.sum(torch.ne(activation, 0)).item()), activation.numel()
        cls._activation_num += loc_act
        cls._total_num += loc_tot

    @classmethod
    def get_clear_activation_num(cls, accumulate: bool = False) -> Tuple[int, int, float]:
        activation_num, total_num = cls._activation_num, cls._total_num
        if not accumulate:
            cls._activation_num, cls._total_num = 0, 0
        activation_num = torch.tensor([activation_num], dtype=torch.long, device="cuda")
        total_num = torch.tensor([total_num], dtype=torch.long, device="cuda")
        torch.distributed.all_reduce(activation_num, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())
        torch.distributed.all_reduce(total_num, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())
        activation_num, total_num = activation_num.item(), total_num.item()
        activation_rate = round(activation_num * 100 / total_num, 2) if total_num > 0 else None
        return activation_num, total_num, activation_rate

    @classmethod
    def record_layer_value(cls, key: str, value: float):
        cls._value_container[key].append(value)

    @classmethod
    def get_clear_layer_values(cls) -> dict:
        value_container = cls._value_container
        cls._value_container = defaultdict(list)
        res_dict = {}
        for key, values in value_container.items():
            if len(values) == 0:
                continue
            for idx, val in enumerate(values):
                res_dict[f"stat/layer{idx:02}/{key}"] = val
            res_dict[f"stat/avg/{key}"] = sum(values) / len(values)
        return res_dict
