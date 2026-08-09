from __future__ import annotations

from src.chem.hard_deletion import CONNECTED_WNODE_CACHE_NAMESPACE
from src.eval.node_wasserstein_distance import (
    node_wasserstein_action_key,
    node_wasserstein_pair_key,
)


def test_connected_policy_uses_an_independent_distance_namespace() -> None:
    kwargs = {
        "canonical_smiles_a": "CCO",
        "canonical_smiles_b": "CC",
        "checkpoint_identity": "checkpoint",
        "feature_cost": "cosine",
        "node_mass": "uniform",
        "size_penalty_beta": 0.0,
    }
    legacy = node_wasserstein_pair_key(**kwargs)
    connected = node_wasserstein_pair_key(
        **kwargs, distance_namespace=CONNECTED_WNODE_CACHE_NAMESPACE
    )
    assert legacy != connected


def test_action_cache_key_changes_with_match_and_policy_context() -> None:
    common = {
        "canonical_parent_smiles": "CCOCC",
        "canonical_residual_smiles": "CC",
        "checkpoint_identity": "checkpoint",
        "feature_cost": "cosine",
        "node_mass": "uniform",
        "size_penalty_beta": 0.0,
        "distance_namespace": CONNECTED_WNODE_CACHE_NAMESPACE,
    }
    context = {
        "candidate_id": "candidate-1",
        "match_atom_indices": [2],
        "teacher_sha256": "a" * 64,
        "action_semantics_version": "connected_sanitized_residual_v1",
        "match_selection_policy": (
            "existential_min_wnode_among_valid_connected_strict_flips_v1"
        ),
        "distance_implementation_version": "wnode-v1",
    }
    first = node_wasserstein_action_key(**common, action_context=context)
    second = node_wasserstein_action_key(
        **common,
        action_context={**context, "match_atom_indices": [3]},
    )
    assert first != second
