import torch
from megatron.core import parallel_state


class LossFreeBalance:

    _update_method: str = "sgd_sign"
    _apply_method: str = "base"
    _print_expert_bias_step: int = -1

    @classmethod
    def set_update_method(cls, update_method: str):
        cls._update_method = update_method

    @classmethod
    def set_apply_method(cls, apply_method: str):
        cls._apply_method = apply_method

    @classmethod
    @torch.no_grad()
    def update_expert_bias(cls, expert_bias: torch.Tensor, tokens_per_expert: torch.Tensor, expert_bias_update_rate: float, targets_per_expert: torch.Tensor = None) -> torch.Tensor:
        """Update expert bias for biased expert routing. See https://arxiv.org/abs/2408.15664v1#

        Args:
            expert_bias (torch.Tensor): The bias for each expert.
            tokens_per_expert (torch.Tensor): The number of tokens assigned to each expert.
            expert_bias_update_rate (float): The update rate for the expert bias.
            targets_per_expert (torch.Tensor): The target number of tokens assigned to each expert.
        """
        # All Reduce Across TPxCPxDP group
        torch.distributed.all_reduce(
            tokens_per_expert,
            group=parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True),
        )
        if targets_per_expert is not None:
            torch.distributed.all_reduce(
                targets_per_expert,
                group=parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True),
            )
            target_tokens = targets_per_expert
        else:
            target_tokens = tokens_per_expert.sum(dim=-1, keepdim=True) / tokens_per_expert.shape[-1]
        if cls._update_method == "sgd_sign":
            updated_expert_bias = expert_bias - expert_bias_update_rate * torch.sign(tokens_per_expert - target_tokens)
        else:
            raise NotImplementedError(f"invalid update method: {cls._update_method}")
        return updated_expert_bias

    @classmethod
    def apply_expert_bias(cls, expert_bias: torch.Tensor, router_scores: torch.Tensor) -> torch.Tensor:
        if cls._apply_method == "base":
            scores_for_routing = router_scores + expert_bias
        elif cls._apply_method == "rms":
            variance = router_scores.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
            scores_for_routing = router_scores + expert_bias.unsqueeze(0) * torch.sqrt(variance)
        else:
            raise NotImplementedError(f"invalid apply method: {cls._apply_method}")
        return scores_for_routing

    @classmethod
    def set_print_expert_bias_step(cls, print_expert_bias_step: int):
        cls._print_expert_bias_step = print_expert_bias_step

    @classmethod
    def print_expert_bias(cls, iteration: int, model: torch.nn.Module):
        if torch.distributed.get_rank() == 0 and cls._print_expert_bias_step > 0 and iteration % cls._print_expert_bias_step == 0:
            lid = 0
            for module in model.modules():
                if hasattr(module, "expert_bias"):
                    print(">" * 6, f"Expert Bias of MoELayer {lid:02}: {module.expert_bias}", "<" * 6)
                    lid += 1
