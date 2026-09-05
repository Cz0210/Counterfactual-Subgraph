"""Synthetic no-fit regression checks for first-fit probability/OT correction."""
from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pytest

from src.ablations.gnn.temperature_repair import BoundRawDistance, FITTER_SHA, scaled_probabilities


def _old_match():
    from src.chem.hard_deletion import CONNECTED_ACTION_SEMANTICS
    return dict(parent_id="p", parent_smiles="CCO", candidate_id="r", match_index=0,
                match_atom_indices=[2], residual_smiles="CC", oracle_checkpoint_hash="frozen-weight",
                action_semantics_version=CONNECTED_ACTION_SEMANTICS, distance_ok=True,
                cf_flip=True, wnode_distance=0.125)


def _context():
    row = _old_match()
    return {key: row[key] for key in ("parent_id", "candidate_id", "match_index", "match_atom_indices",
                                      "action_semantics_version")} | {"oracle_checkpoint_id": "frozen-weight"}


def test_raw_distance_lookup_keeps_original_numeric_value_and_binding():
    old = _old_match()
    provider = BoundRawDistance([old], old_checkpoint="frozen-weight", new_checkpoint="frozen-weight",
                               parent_checkpoint_sha="parent-sha", distance_contract_sha="encoder-contract-sha")
    result = provider.distance_for_action("CCO", "CC", action_context=_context())
    assert result == {"ok": True, "distance": 0.125, "cache_hit": True}
    assert provider.used[0]["raw_wnode"] == old["wnode_distance"]
    assert provider.used[0]["parent_checkpoint_sha256"] == "parent-sha"
    assert provider.used[0]["distance_contract_sha256"] == "encoder-contract-sha"


@pytest.mark.parametrize("field,value", [
    ("parent_id", "wrong-parent"), ("candidate_id", "wrong-rule"),
    ("match_index", 1), ("match_atom_indices", [0]),
    ("oracle_checkpoint_id", "different-backbone"), ("action_semantics_version", "different-deletion"),
])
def test_raw_distance_never_loosens_exact_cache_provenance(field, value):
    provider = BoundRawDistance([_old_match()], old_checkpoint="frozen-weight", new_checkpoint="frozen-weight",
                               parent_checkpoint_sha="parent", distance_contract_sha="encoder")
    context = _context() | {field: value}
    with pytest.raises(ValueError, match="CACHE_PROVENANCE_GAP"):
        provider.distance_for_action("CCO", "CC", action_context=context)
    assert provider.used == []


def test_cross_backbone_cache_reuse_is_rejected_before_any_work():
    with pytest.raises(ValueError, match="UNCHANGED_MODEL_WEIGHTS"):
        BoundRawDistance([_old_match()], old_checkpoint="gin", new_checkpoint="gcn",
                         parent_checkpoint_sha="parent", distance_contract_sha="encoder")


def test_real_hard_deletion_uses_saved_ot_without_loading_solver():
    pytest.importorskip("rdkit")
    from src.eval.bace_frozen_gnn_contracts import BACEParent
    from src.eval.bace_frozen_gnn_verification import _evaluate_rows
    from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer, default_molecular_feature_schema
    provider = BoundRawDistance([_old_match()], old_checkpoint="frozen-weight", new_checkpoint="frozen-weight",
                               parent_checkpoint_sha="parent", distance_contract_sha="encoder")
    class ResidualOracle:
        def predict_records(self, graphs, **_kwargs):
            assert len(graphs) == 1
            return [{"predicted_label": 0, "probabilities": [0.8, 0.2]}]
    rows, matches = _evaluate_rows([BACEParent("p", "CCO", 1, 0)],
        [{"candidate_id": "r", "canonical_fragment": "O"}], oracle=ResidualOracle(),
        featurizer=MolecularGraphFeaturizer(default_molecular_feature_schema()), distance_provider=provider,
        oracle_batch_size=1, split="calibration", oracle_checkpoint_id="frozen-weight",
        parent_prediction_cache={"p": {"parent_smiles": "CCO", "pred_before": 1, "p_before": [0.3, 0.7]}})
    assert len(provider.used) == 1
    assert matches[0]["wnode_distance"] == rows[0]["wnode_distance"] == 0.125
    assert rows[0]["pair_strict_flip"] is True
    assert rows[0]["cf_drop"] == pytest.approx(0.5)
    assert rows[0]["best_match_atom_indices"] == [2]


def test_positive_scalar_scales_raw_logits_once_and_keeps_argmax():
    logits = np.asarray([[-2.0, 2.0], [4.0, -3.0]])
    result = scaled_probabilities(logits, 2.0)
    assert np.array_equal(result.argmax(1), logits.argmax(1))
    assert result[0, 1] == pytest.approx(1.0 / (1.0 + np.exp(-2.0)))
    assert result[0, 1] != pytest.approx(1.0 / (1.0 + np.exp(-1.0)))


def test_fitter_source_fingerprint_uses_declared_trailing_newline_convention():
    from src.oracles.gnn_oracle import fit_temperature_scaling
    assert hashlib.sha256(inspect.getsource(fit_temperature_scaling).rstrip("\n").encode()).hexdigest() == FITTER_SHA
