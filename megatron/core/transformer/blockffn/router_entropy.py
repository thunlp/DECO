import os
import torch
import torch.distributed
from typing import List
from megatron.core import mpu


class RouterEntropyAutoScaler(torch.autograd.Function):
    """An AutoScaler that compute and scales the grad for router entropy loss."""

    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, router_score: torch.Tensor, router_entropy: torch.Tensor):
        """Preserve the router_entropy by storing it in the context to avoid garbage collection.
        Args:
            router_score (torch.Tensor): The router_score tensor.
            router_entropy (torch.Tensor): The router_entropy loss tensor.
        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(router_entropy)
        return router_score

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for router entropy loss.
        Args:
            grad_output (torch.Tensor): The gradient of the output.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled router entropy loss gradient.
        """
        (router_entropy,) = ctx.saved_tensors
        router_entropy_backward_scale = RouterEntropyAutoScaler.main_loss_backward_scale
        scaled_router_entropy_grad = torch.ones_like(router_entropy) * router_entropy_backward_scale
        return grad_output, scaled_router_entropy_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the router entropy loss.
        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in matches the scale of the main_loss.
        """
        RouterEntropyAutoScaler.main_loss_backward_scale = scale


class RegRouterEntropy:

    _layer_number: int = -1
    _loss_container: List[float] = []

    _reg_coef: float = 0.
    _reg_coef_init: float = 0.
    _reg_coef_multiplier: float = None

    _chunk_regularization_length: int = 64
    _transfer_lambda: float = 0.
    _sigmoid_steep: float = 10.0

    @classmethod
    def is_enabled(cls):
        return cls._layer_number > 0 and cls._reg_coef > 0.

    @classmethod
    def set_layer_number(cls, layer_number: int):
        cls._layer_number = layer_number

    @classmethod
    def set_reg_coef(cls, reg_coef: float):
        cls._reg_coef = reg_coef

    @classmethod
    def set_transfer_lambda(cls, transfer_lambda: float):
        cls._transfer_lambda = transfer_lambda

    @classmethod
    def initialize(cls, layer_number: int, reg_coef_init: float, reg_coef_multiplier: float, res_coef_resume: float, transfer_lambda: float):
        cls._layer_number = layer_number
        cls._reg_coef_init = reg_coef_init
        cls._reg_coef = res_coef_resume if res_coef_resume is not None and res_coef_resume > 0 else reg_coef_init
        cls._reg_coef_multiplier = reg_coef_multiplier
        cls._transfer_lambda = transfer_lambda

    @classmethod
    def get_reg_coef(cls):
        return cls._reg_coef

    @classmethod
    def step_reg_coef(cls, cur_activation_rate: float, target_activation_rate: float) -> None:
        l1_coef, multiplier = cls._reg_coef, cls._reg_coef_multiplier
        if l1_coef is None or l1_coef <= 0 or multiplier is None or multiplier <= 0:
            return
        if cur_activation_rate > target_activation_rate:
            cls._reg_coef *= multiplier
        else:
            cls._reg_coef = max(l1_coef / multiplier, cls._reg_coef_init)

    @classmethod
    def apply_router_entropy(cls, router_score: torch.Tensor, router_score_scaled: torch.Tensor) -> torch.Tensor:
        if not cls.is_enabled():
            return router_score
        router_entropy = torch.sum(-router_score_scaled * torch.log(router_score_scaled + 1e-5), dim=-1)
        router_entropy = router_entropy.mean()
        # ALERT: save the loss for scheduling before multiplied by the factor!
        cls._loss_container.append(router_entropy.item())
        # loss multiplied by the factor
        router_entropy = router_entropy * cls._reg_coef / cls._layer_number
        if torch.isnan(router_entropy) or torch.isinf(router_entropy):
            rank = torch.distributed.get_rank()
            print(f"RANK {rank:02}: nan/inf router_entropy loss detected!")
            os.makedirs("logs/error", exist_ok=True)
            torch.save((router_score_scaled, router_entropy), f"logs/error/router_entropy_nan_{rank:02}.pkl")
            exit()
        router_score = RouterEntropyAutoScaler.apply(router_score, router_entropy)
        return router_score

    @classmethod
    def apply_chunk_regularization(cls, router_score: torch.Tensor, router_score_scaled: torch.Tensor) -> torch.Tensor:
        if not cls.is_enabled():
            return router_score
        seq_len, expert_num = router_score_scaled.shape
        chunk_size = cls._chunk_regularization_length
        assert seq_len % chunk_size == 0
        activation_group = router_score_scaled.view(seq_len // chunk_size, chunk_size, expert_num)

        # standard implementation, sum[ln(1-p)]
        activation_group = torch.sum(torch.log(1 - activation_group + 1e-5), dim=1)
        chunk_loss = 1 - torch.mean(torch.exp(activation_group))

        cls._loss_container.append(chunk_loss.item())
        # loss multiplied by the factor
        chunk_loss = chunk_loss * cls._reg_coef / cls._layer_number
        if torch.isnan(chunk_loss) or torch.isinf(chunk_loss):
            rank = torch.distributed.get_rank()
            print(f"RANK {rank:02}: nan/inf chunk_loss loss detected!")
            os.makedirs("logs/error", exist_ok=True)
            torch.save((router_score_scaled, chunk_loss), f"logs/error/chunk_loss_nan_{rank:02}.pkl")
            exit()
        router_score = RouterEntropyAutoScaler.apply(router_score, chunk_loss)
        return router_score

    @classmethod
    def apply_transfer_regularization(cls, router_score: torch.Tensor, raw_router_score: torch.Tensor):
        if not cls.is_enabled():
            return router_score
        assert raw_router_score.shape == router_score.shape, f"{raw_router_score.shape} != {router_score.shape}"
        left_score, right_score = raw_router_score[:-1], raw_router_score[1:]
        transfer_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            left_score * cls._sigmoid_steep, torch.sigmoid(right_score * cls._sigmoid_steep),
        )

        transfer_loss = transfer_loss * cls._transfer_lambda / cls._layer_number
        if torch.isnan(transfer_loss) or torch.isinf(transfer_loss):
            rank = torch.distributed.get_rank()
            print(f"RANK {rank:02}: nan/inf transfer_loss loss detected!")
            os.makedirs("logs/error", exist_ok=True)
            torch.save((raw_router_score, transfer_loss), f"logs/error/transfer_loss_nan_{rank:02}.pkl")
            exit()
        router_score = RouterEntropyAutoScaler.apply(router_score, transfer_loss)
        return router_score

    @classmethod
    def get_clear_router_entropy(cls) -> float:
        if not cls.is_enabled() or len(cls._loss_container) == 0:
            return 0.
        ave_router_entropy = sum(cls._loss_container) / len(cls._loss_container)
        cls._loss_container = []
        ave_router_entropy = torch.tensor([ave_router_entropy], dtype=torch.float, device="cuda")
        torch.distributed.all_reduce(ave_router_entropy, op=torch.distributed.ReduceOp.AVG, group=mpu.get_context_parallel_group())
        ave_router_entropy = ave_router_entropy.item()
        return ave_router_entropy

    @classmethod
    def apply_l1_load_balancing(cls, router_score: torch.Tensor, routing_map: torch.Tensor, target_act_ratio: int):
        if not cls.is_enabled():
            return router_score
        tokens_per_expert = routing_map.sum(dim=0)
        num_tokens = router_score.shape[0]
        aggregated_score_per_expert = router_score.sum(dim=0)
        l1_norm = torch.sum(aggregated_score_per_expert * tokens_per_expert) / (target_act_ratio * num_tokens * num_tokens)
        cls._loss_container.append(l1_norm.item())
        l1_norm = l1_norm * cls._reg_coef / cls._layer_number
        if torch.isnan(l1_norm) or torch.isinf(l1_norm):
            rank = torch.distributed.get_rank()
            print(f"RANK {rank:02}: nan/inf l1_norm loss detected!")
            os.makedirs("logs/error", exist_ok=True)
            torch.save((router_score, routing_map, l1_norm), f"logs/error/l1_norm_nan_{rank:02}.pkl")
            exit()
        router_score = RouterEntropyAutoScaler.apply(router_score, l1_norm)
        return router_score
