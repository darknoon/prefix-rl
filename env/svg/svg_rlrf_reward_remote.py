import modal
from typing import Any

SVGRewardService = modal.Cls.from_name(
    "prefix-rl-verl-reward-service",
    "SVGRewardService",  # class name
)


def compute_rewards_dict(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[Any],
    **kwargs,
) -> list[dict[str, float]]:
    """Offload SVG reward computation to Modal GPU service."""

    return SVGRewardService().compute_rewards_batch.remote(
        data_sources=data_sources,
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
    )
