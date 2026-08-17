from __future__ import annotations

import json
import inspect

import pytest

from src.eval.frozen_threshold_manifest import load_shared_frozen_thresholds
from src.eval.node_wasserstein_distance import (
    node_wasserstein_action_key,
    node_wasserstein_pair_key,
)


def _write(tmp_path, **updates):
    payload = {
        "thresholds": [0.01, 0.02, 0.03],
        "theta_star": 0.02,
        "cost_cap": 0.03,
        "threshold_fitted_on_test": False,
        "selection_used_test": False,
        "shared_across_methods": True,
        "cf_mode": "strict_flip",
    }
    payload.update(updates)
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_shared_frozen_threshold_is_rendered_without_fitting(tmp_path) -> None:
    contract = load_shared_frozen_thresholds(_write(tmp_path))
    assert contract["threshold_csv"] == "0.01,0.02,0.029999999999999999"
    assert contract["theta_star"] == 0.02
    assert contract["threshold_fitted_on_test"] is False


@pytest.mark.parametrize(
    "updates",
    [
        {"threshold_fitted_on_test": True},
        {"selection_used_test": True},
        {"shared_across_methods": False},
        {"cf_mode": "auto_quantile"},
    ],
)
def test_method_specific_or_test_fitted_threshold_fails_closed(tmp_path, updates) -> None:
    with pytest.raises(ValueError):
        load_shared_frozen_thresholds(_write(tmp_path, **updates))


def test_pair_cache_identity_is_threshold_independent() -> None:
    assert "threshold" not in inspect.signature(node_wasserstein_pair_key).parameters
    assert "threshold" not in inspect.signature(node_wasserstein_action_key).parameters
    key = node_wasserstein_pair_key(
        canonical_smiles_a="CCO",
        canonical_smiles_b="CCN",
        checkpoint_identity="frozen-molclr",
        feature_cost="cosine",
        node_mass="uniform",
        size_penalty_beta=0.0,
    )
    assert key == node_wasserstein_pair_key(
        canonical_smiles_a="CCN",
        canonical_smiles_b="CCO",
        checkpoint_identity="frozen-molclr",
        feature_cost="cosine",
        node_mass="uniform",
        size_penalty_beta=0.0,
    )
