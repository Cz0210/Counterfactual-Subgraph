"""Registry for controlled candidate-source ablation variants."""

from __future__ import annotations

from typing import Any

from src.candidates.random_brics_generator import RandomBRICSGenerator
from src.candidates.random_connected_subgraph_generator import (
    RandomConnectedSubgraphGenerator,
)


CANDIDATE_SOURCE_VARIANTS = (
    "chemllm_ppo",
    "chemllm_sft_only",
    "random_connected_size_matched",
    "random_brics_size_matched",
    "random_topk_from_chemllm_pool",
)


def build_random_generator(name: str) -> Any:
    if name == "random_connected_size_matched":
        return RandomConnectedSubgraphGenerator()
    if name == "random_brics_size_matched":
        return RandomBRICSGenerator()
    raise ValueError(
        f"{name!r} is not a random generator; expected one of "
        "random_connected_size_matched or random_brics_size_matched."
    )


__all__ = ["CANDIDATE_SOURCE_VARIANTS", "build_random_generator"]
