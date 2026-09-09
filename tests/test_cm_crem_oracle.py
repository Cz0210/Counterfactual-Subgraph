"""CPU engineering fixtures; these do NOT certify the actual frozen BACE run."""

import copy
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
Chem = pytest.importorskip("rdkit.Chem")

from src.baselines.cm_crem_oracle import (
    ENCODING_SCHEMA, FrozenCMOracle, FrozenCMWNode, encoding_digest,
    full_graph_pair, graph_identity, raw_distance_record, upstream_cam,
    upstream_top_atoms,
)
from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer
from src.models.molecular_gnn import MolecularGNN, MolecularGNNConfig
from src.oracles.gnn_oracle import GNNOracle


@pytest.fixture
def adapter():
    torch.manual_seed(7)
    schema = MolecularGraphFeaturizer().schema
    model = MolecularGNN(MolecularGNNConfig(backbone="gine", num_classes=2,
                          num_layers=3, hidden_dim=8, dropout=0.3,
                          normalization="batch_norm", residual=True),
                         node_cardinalities=schema.node_cardinalities,
                         edge_cardinalities=schema.edge_cardinalities)
    with torch.no_grad():
        model.classifier[-1].bias[0] = -1
        model.classifier[-1].bias[1] = 1
    oracle = GNNOracle(model, checkpoint_id="a" * 64, backbone="gine",
                       num_classes=2, source_label=1, temperature=1.5447202081060156,
                       edge_feature_dim=len(schema.edge_fields), device="cpu")
    return FrozenCMOracle(oracle, MolecularGraphFeaturizer(schema),
                          binding={"node_layer": "layers.2", "temperature_sha256": "b" * 64})


def test_real_gine_gradcam_same_forward_frozen_parameters_bn_rng_and_atom_map(adapter):
    from src.baselines.cm_crem_generation import load_parent_mol

    model = adapter.oracle.model
    snapshot = {k: v.clone() for k, v in model.state_dict().items()}
    flags = [p.requires_grad for p in model.parameters()]
    rng = torch.random.get_rng_state().clone()
    parent = {"parent_id": "fixture-0", "smiles": "N[C@@H](C)C(=O)O", "split": "train"}
    result = adapter.attribute_train_parent(parent)
    assert result["gradient_nonzero_count"] > 0
    assert result["forward_argmax_exact"] is True
    assert result["source_score"] == "calibrated_probability_source_class"
    assert result["node_layer"] == "layers.2"
    assert result["atom_mapping"] == list(range(6))
    assert all(torch.equal(snapshot[k], v) for k, v in model.state_dict().items())
    assert flags == [p.requires_grad for p in model.parameters()]
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(not module.training for module in model.modules())
    assert all(param.grad is None for param in model.parameters())
    assert len(model.layers[-1]._forward_hooks) == 0
    assert len(result["importances"]) == 6
    molecule = load_parent_mol(result["generation_request"])
    assert Chem.MolToSmiles(molecule) == Chem.MolToSmiles(Chem.MolFromSmiles(parent["smiles"]))


def test_attribution_repeat_is_exact_and_no_double_scaling(adapter):
    parent = {"parent_id": "p", "smiles": "CCOC(=O)N", "split": "train"}
    a = adapter.attribute_train_parent(parent)
    b = adapter.attribute_train_parent(parent)
    assert a["importances"] == b["importances"]
    assert a["gradient_sha256"] == b["gradient_sha256"]
    logits = np.array(a["prediction"]["logits"])
    expected = np.exp(logits / adapter.oracle.temperature - (logits / adapter.oracle.temperature).max())
    expected /= expected.sum()
    np.testing.assert_array_equal(expected, a["prediction"]["probabilities"])


def test_author_formula_is_node_channel_mean_not_standard_gradcam():
    activations = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    gradients = torch.tensor([[1., 3.], [-2., 0.], [1., 1.]])
    original = (activations * gradients.mean(dim=1, keepdim=True)).sum(dim=1)
    expected = (original - original.min()) / (original.max() - original.min() + 1e-9)
    assert torch.equal(upstream_cam(activations, gradients), expected)
    conventional = (activations * gradients.mean(dim=0, keepdim=True)).sum(dim=1)
    assert not torch.equal(expected, (conventional - conventional.min()) / (conventional.max() - conventional.min() + 1e-9))


@pytest.mark.parametrize("n,count", [(1, 1), (4, 1), (9, 1), (10, 2), (24, 4)])
def test_official_floor_and_stable_index_ties(n, count):
    assert upstream_top_atoms([0.] * n) == list(range(count))


@pytest.mark.parametrize("split", ["calibration", "test", None])
def test_nontrain_attribution_rejected(adapter, split):
    with pytest.raises(ValueError, match="train-only"):
        adapter.attribute_train_parent({"parent_id": "p", "smiles": "CCO", "split": split})


def test_non_source_does_not_make_mask(adapter):
    with torch.no_grad():
        adapter.oracle.model.classifier[-1].bias[:] = torch.tensor([100., -100.])
    result = adapter.attribute_train_parent({"parent_id": "p", "smiles": "CCO", "split": "train"})
    assert result["status"] == "BEFORE_NOT_SOURCE"
    assert "generation_request" not in result


def test_canonical_graph_identity_is_order_invariant_and_stereo_sensitive(adapter):
    assert graph_identity("OCC", adapter.featurizer)["candidate_id"] == graph_identity("CCO", adapter.featurizer)["candidate_id"]
    assert graph_identity("N[C@@H](C)C(=O)O", adapter.featurizer)["candidate_id"] != graph_identity("N[C@H](C)C(=O)O", adapter.featurizer)["candidate_id"]


@pytest.mark.parametrize("smiles,error", [("C.C", "DISCONNECTED"), ("[*]C", "DUMMY"), ("", "EMPTY")])
def test_invalid_graphs_reject_semantically(adapter, smiles, error):
    with pytest.raises(ValueError, match=error):
        graph_identity(smiles, adapter.featurizer)


def test_original_multicomponent_parent_kept_but_generated_prototype_rejected(adapter):
    parent = {"parent_id": "salt", "smiles": "CCO.CC", "split": "train"}
    # Production preserves the original frozen featurizer's own input gate.
    with pytest.raises(ValueError, match="exactly one connected component"):
        adapter.predict_rows([parent], split="train")
    # An explicitly different fixture featurizer proves the adapter adds no
    # generated-prototype gate to original parents; it is not a BACE override.
    adapter.featurizer = MolecularGraphFeaturizer(adapter.featurizer.schema,
                                                require_single_component=False)
    assert adapter.predict_rows([parent], split="train")[0]["parent_id"] == "salt"
    result = adapter.filter_generated(parent, {"parent_id": "salt", "status": "GENERATED",
        "retained_raw": [{"raw_id": "disconnected", "smiles": "CCC.CC"}]})
    assert result["rejected"][0]["reason"] == "DISCONNECTED_MOLECULE"


@pytest.mark.parametrize("status", ["NO_NATIVE_REPLACEMENT", "NO_REPLACEABLE_CONTEXT"])
def test_native_empty_terminals_are_explicit_and_cannot_hide_raw_outputs(adapter, status):
    parent = {"parent_id": "p", "smiles": "CCO", "split": "train"}
    generated = {"parent_id": "p", "status": status, "retained_raw": []}
    assert adapter.filter_generated(parent, generated)["unique_target_count"] == 0
    generated["retained_raw"] = [{"raw_id": "bad", "smiles": "CCC"}]
    with pytest.raises(ValueError, match="empty/context terminal"):
        adapter.filter_generated(parent, generated)


def test_filter_preserves_zero_and_infrastructure_error(adapter):
    parent = {"parent_id": "p", "smiles": "CCO", "split": "train"}
    result = adapter.filter_generated(parent, {"parent_id": "p", "status": "GENERATION_COMPLETE", "retained_raw": [
        {"raw_id": "unchanged", "smiles": "OCC"}, {"raw_id": "bad", "smiles": "C.C"},
        {"raw_id": "not_flip", "smiles": "CCCC"}]})
    assert result["unique_target_count"] == 0
    assert len(result["rejected"]) == 3
    with pytest.raises(ValueError, match="not scientific terminal"):
        adapter.filter_generated(parent, {"parent_id": "p", "status": "INFRASTRUCTURE_EIO", "retained_raw": []})


def test_filter_accepts_destination_and_retains_all_origins(adapter, monkeypatch):
    parent = {"parent_id": "p", "smiles": "CCO", "split": "train"}
    original_predict = adapter.predict_rows
    def fake_prediction(rows, *, split):
        records = original_predict(rows, split=split)
        for row in records:
            row["predicted_label"] = 1 if split == "train" else 0
        return records
    monkeypatch.setattr(adapter, "predict_rows", fake_prediction)
    result = adapter.filter_generated(parent, {"parent_id": "p", "status": "GENERATION_COMPLETE", "retained_raw": [
        {"raw_id": "a", "smiles": "CCCN"}, {"raw_id": "b", "smiles": "NCCC"}]})
    assert result["strict_flip_count"] == 2
    assert result["unique_target_count"] == 1
    assert len(result["accepted"][0]["origins"]) == 2


def test_raw_budget_precedes_prediction(adapter, monkeypatch):
    parent = {"parent_id": "p", "smiles": "CCO", "split": "train"}
    with pytest.raises(ValueError, match="truncation"):
        adapter.filter_generated(parent, {"parent_id": "p", "status": "GENERATION_COMPLETE",
            "retained_raw": [{"raw_id": str(i), "smiles": "CCC"} for i in range(129)]})


def _encoded(graph_id, H):
    record = {"schema_version": ENCODING_SCHEMA, "candidate_id": graph_id,
              "H": H, "atom_numbers": [6] * len(H), "embedding_dtype": "float32",
              "molclr_checkpoint_sha256": "c" * 64, "numerical_contract_sha256": "d" * 64,
              "node_extraction_version": "actual-fixture-version", "producer": {"device": "cpu"}}
    record["encoding_sha256"] = encoding_digest(record)
    return record


CONTRACT = {"numerical_contract_sha256": "d" * 64, "feature_cost": "cosine", "node_mass": "uniform", "size_penalty_beta": 0.0}


def test_raw_distance_uses_original_uniform_cosine_emd_function():
    calls = []
    def fixture_emd(a, b, M):
        calls.append((a, b, M))
        return min(M[0, 0] + M[1, 1], M[0, 1] + M[1, 0]) / 2
    left, right = _encoded("left", [[1., 0.], [0., 1.]]), _encoded("right", [[1., 0.], [1., 0.]])
    result = raw_distance_record(left, right, numerical_contract=CONTRACT, emd2_fn=fixture_emd)
    assert result["distance"] == .5
    np.testing.assert_array_equal(calls[0][0], [.5, .5])
    assert calls[0][2].dtype == np.float64
    assert result["distance_is_uncapped"] is True


@pytest.mark.parametrize("field", ["H", "producer", "molclr_checkpoint_sha256"])
def test_encoding_cache_conflict_fails_without_selecting_smaller_value(field):
    a, b = _encoded("a", [[1., 0.]]), _encoded("b", [[0., 1.]])
    b[field] = [[2., 0.]] if field == "H" else "conflicting"
    with pytest.raises(ValueError, match="identity mismatch"):
        raw_distance_record(a, b, numerical_contract=CONTRACT)


def test_emd_eio_escapes_not_infinity():
    a, b = _encoded("a", [[1., 0.]]), _encoded("b", [[0., 1.]])
    def bad_solver(*args):
        raise OSError(5, "Input/output error")
    with pytest.raises(OSError):
        raw_distance_record(a, b, numerical_contract=CONTRACT, emd2_fn=bad_solver)


def test_strict_flip_reduction_keeps_base_non_source_and_rejects_missing_distance():
    parent = {"parent_id": "p", "predicted_label": 0, "oracle_weight_sha256": "o", "temperature": 1.5, "full_graph_id": "pg"}
    proto = {"candidate_id": "c", "predicted_label": 0, "oracle_weight_sha256": "o", "temperature": 1.5, "full_graph_id": "cg"}
    result = full_graph_pair(parent, proto, raw_distance=None)
    assert result["kept_in_base_denominator"] is True
    assert result["failure_reason"] == "BEFORE_NOT_SOURCE" and result["distance"] is None
    parent["predicted_label"] = 1
    with pytest.raises(ValueError, match="missing a computed raw distance"):
        full_graph_pair(parent, proto, raw_distance=None)
    with pytest.raises(ValueError, match="complete parent/prototype"):
        full_graph_pair(parent, proto, raw_distance={"distance": .01, "parent_graph_id": "wrong"})


def test_wrong_oracle_or_node_layer_refused(adapter):
    oracle = adapter.oracle
    oracle.backbone = "gin"
    with pytest.raises(ValueError, match="not GIN/A"):
        FrozenCMOracle(oracle, adapter.featurizer, binding={})
    oracle.backbone = "gine"
    with pytest.raises(ValueError, match="final message layer"):
        FrozenCMOracle(oracle, adapter.featurizer, binding={"node_layer": "layers.0"})
