from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.ablations.llm import bace_common_downstream as route
from src.ablations.llm.contracts import canonical_json_sha256
from src.eval.bace_frozen_gnn_contracts import BACEParent, atomic_json, atomic_jsonl, read_json, sha256_file
from src.eval.mutagenicity_wnode_selector import derive_thresholds, VariantConfig


def calls():
    return [{"parent_id": "p", "parent_smiles": "CCC", "regime": regime, "shard_id": 0}
            for regime in ("base", "high_temperature")]


def attempts(variant="CHEMLLM_7B_OFF_THE_SHELF"):
    return [{**calls()[i // 4], "variant": variant, "attempt_index": i,
             "source_label": 1, "train_only": True, "raw_text": "C", "fragment_smiles": "C"} for i in range(8)]


def selector():
    return {"thresholds": derive_thresholds(np.array([0.1, 0.2, 0.3, 0.4, 0.5])),
            "variant": VariantConfig("fixture", False, 0.2, 0, 0, 0, False, False),
            "prefix_weights": tuple([1.] * 20), "local_swap_passes": 0,
            "input_sha256": "a" * 64}


def rules():
    return [{"candidate_id": f"c{i:02d}", "canonical_fragment": "C" * (i + 1)} for i in range(20)]


def test_attempt_matched_missing_duplicate_and_parent_drift():
    route.validate_attempts(attempts(), calls(), "CHEMLLM_7B_OFF_THE_SHELF")
    for rows in (attempts()[:-1], attempts() + attempts()[:1]):
        with pytest.raises(ValueError, match="PROPOSAL_ATTEMPT"):
            route.validate_attempts(rows, calls(), "CHEMLLM_7B_OFF_THE_SHELF")
    rows = attempts()
    rows[0]["parent_smiles"] = "CC"
    with pytest.raises(ValueError, match="parent_smiles"):
        route.validate_attempts(rows, calls(), "CHEMLLM_7B_OFF_THE_SHELF")


def test_native_candidate_receipt_binds_whole_spec(tmp_path):
    spec = {"variant": "CHEMLLM_7B_OFF_THE_SHELF", "calls": calls(), "task_spec_sha256": "a" * 64}
    atomic_jsonl(tmp_path / "candidate_pool.jsonl", attempts())
    receipt = {"status": "CANDIDATE_POOL_PASS", "spec_sha256": canonical_json_sha256(spec),
               "next_call": 2, "proposal_attempts": 8, "variant": spec["variant"],
               "test_loaded": False, "calibration_loaded": False, "training_performed": False,
               "candidate_pool_sha256": sha256_file(tmp_path / "candidate_pool.jsonl")}
    atomic_json(tmp_path / "candidate_generation_receipt.json", receipt)
    assert len(route.load_attempts(spec, tmp_path, "b" * 64)[0]) == 8
    receipt["test_loaded"] = True
    atomic_json(tmp_path / "candidate_generation_receipt.json", receipt)
    with pytest.raises(ValueError, match="test_loaded"):
        route.load_attempts(spec, tmp_path, "b" * 64)


def test_main_b10_merge_keeps_valid_train_nonflip_and_canonical_dedup():
    common = dict(parent_id="p", parse_ok=True, valid=True, connected=True,
                  direct_substructure=True, oracle_ok=True, cf_flip=False, cf_drop=.1)
    rows = [{**common, "candidate_id": "a", "final_fragment": "CC", "reward_total": .1},
            {**common, "candidate_id": "b", "final_fragment": "C(C)", "reward_total": .2},
            {**common, "candidate_id": "c", "final_fragment": "C", "reward_total": .3}]
    merged, universe = route.merge_scored_rows(rows)
    assert len(merged) == len(universe) == 2
    assert merged[-1]["candidate_id"] == "b"
    assert all(row["source_strict_flip_count"] == 0 for row in universe)
    assert len({row["candidate_id"] for row in universe}) == 2


def test_failed_attempts_and_brics_shortfall_remain_denominator():
    rows = [dict(parent_id="p", parse_ok=True, valid=True, connected=True,
                 direct_substructure=True, cf_flip=False, cf_drop=.2, final_fragment="C"),
            dict(parent_id="p", parse_ok=False, valid=False, final_fragment=None)]
    result = route.candidate_metrics(rows, [{}, {"proposal_shortfall": True}])
    assert result["parse_rate"] == .5
    assert result["proposal_attempts"] == 2
    assert result["proposal_shortfall"] == 1
    assert result["strict_flip_rate"] == 0
    assert result["projection_rate"] == 0


def test_real_main_hard_deletion_scorer_retains_invalid_attempt():
    from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer, default_molecular_feature_schema
    from src.eval.bace_frozen_gnn_pool import _score_generated_candidates
    class TinyOracle:
        def predict_records(self, graphs, batch_size):
            return [{"predicted_label": int(g.smiles == "CCC"),
                     "probabilities": [.2, .8] if g.smiles == "CCC" else [.8, .2]} for g in graphs]
    parent = BACEParent("p", "CCC", 1, 0)
    rows = _score_generated_candidates([(parent, 0, "C", "C"), (parent, 1, "bad", "bad")],
        oracle=TinyOracle(), featurizer=MolecularGraphFeaturizer(default_molecular_feature_schema()),
        stage="LLM_COMMON_TRAIN_ONLY", shard_index=0, oracle_batch_size=2, checkpoint_id="a" * 64)
    assert len(rows) == 2 and rows[0]["cf_flip"] is True
    assert rows[0]["residual_smiles"] == "CC"
    assert rows[1]["parse_ok"] is False and rows[1]["cf_flip"] is False
    assert all(row["projection_used"] is False for row in rows)


def test_brics_old_manifest_without_direct_reference_closes_both_links(tmp_path):
    from src.eval.bace_frozen_gnn_contracts import file_identity
    rows = [{"parent_id": "p", "attempt_index": i, "variant": "BRICS_FIXED", "source_label": 1,
             "fragment_smiles": "C" if i == 0 else None, "proposal_shortfall": i != 0,
             "oracle_used": False, "calibration_loaded": False, "test_loaded": False} for i in range(8)]
    atomic_jsonl(tmp_path / "brics_proposal_pool.jsonl", rows[:1])
    atomic_jsonl(tmp_path / "attempts.jsonl", rows)
    common = {"status": "PASS", "calibration_loaded": False, "test_loaded": False,
              "reference_contract": {"path": "/historical/ref.json", "sha256": "a" * 64}}
    atomic_json(tmp_path / "brics_vocab_manifest.json", {**common, "source_split": "train", "oracle_fields_read": []})
    atomic_json(tmp_path / "brics_proposal_shortfall_receipt.json", {**common,
        "candidate_duplication_used": False, "oracle_ranking_used": False, "shortfall_is_not_backfilled": True,
        "attempt_records": file_identity(tmp_path / "attempts.jsonl")})
    proposal = {"status": "PASS", "calibration_loaded": False, "test_loaded": False, "oracle_used": False,
        "candidate_pool": file_identity(tmp_path / "brics_proposal_pool.jsonl"),
        "attempt_records": file_identity(tmp_path / "attempts.jsonl"),
        "vocabulary_manifest": file_identity(tmp_path / "brics_vocab_manifest.json"),
        "shortfall_receipt": file_identity(tmp_path / "brics_proposal_shortfall_receipt.json")}
    atomic_json(tmp_path / "brics_proposal_manifest.json", proposal)
    spec = {"variant": "BRICS_FIXED", "calls": calls(), "adopted_brics": {
        name: file_identity(tmp_path / name) for name in ("brics_proposal_pool.jsonl", "brics_proposal_manifest.json",
                                                        "brics_vocab_manifest.json", "brics_proposal_shortfall_receipt.json")}}
    loaded, _ = route.load_attempts(spec, tmp_path, "a" * 64)
    assert len(loaded) == 8 and sum(row["proposal_shortfall"] for row in loaded) == 7
    with pytest.raises(ValueError, match="BRICS_reference"):
        route.load_attempts(spec, tmp_path, "b" * 64)
    # The same immutable BRICS JSONs remain readable on HPC: only physical
    # lookup changes, never the embedded AutoDL provenance or its SHA.
    from src.ablations.llm.portable_inputs import PortableInputs, SCHEMA
    portable_root = tmp_path / "portable"
    portable_root.mkdir()
    sources = {}
    for source in tmp_path.iterdir():
        if not source.is_file():
            continue
        destination = portable_root / source.name
        destination.write_bytes(source.read_bytes())
        sources[str(source)] = {"relative": source.name, "sha256": sha256_file(source), "size": source.stat().st_size}
    manifest = {"schema_version": SCHEMA, "variant": "BRICS_FIXED", "source_files": sources,
        "original_manifests_modified": False, "model_weights_copied": False}
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    atomic_json(portable_root / "portable_manifest.json", manifest)
    relocated, evidence = route.load_attempts(spec, portable_root, "a" * 64,
                                             file_resolver=PortableInputs(portable_root).resolve)
    assert relocated == loaded and evidence["pool_sha256"] == sha256_file(tmp_path / "brics_proposal_pool.jsonl")


def heldout_fixture(tmp_path, monkeypatch, *, empty=None):
    bundle, output = tmp_path / "bundle", tmp_path / "result"
    bundle.mkdir()
    output.mkdir()
    for split in ("calibration", "test"):
        (bundle / f"{split}.csv").write_text("fixture\n")
    manifest = {"splits": {s: f"{s}.csv" for s in ("calibration", "test")},
                "files": {p.name: {"size": p.stat().st_size, "sha256": sha256_file(p)} for p in bundle.iterdir()}}
    observed = []
    def load(path, source_label):
        observed.append(path.stem)
        assert source_label == 1
        if path.stem == "test":
            freeze = read_json(output / "selector_manifest.json")
            assert freeze["selection_frozen"] is True and freeze["test_loaded"] is False
            assert len(freeze["ordered_rule_ids"]) == 20
        return [] if path.stem == empty else [BACEParent(path.stem + "_p", "CCC", 1, 0)]
    monkeypatch.setattr(route, "load_bace_parents", load)
    # Deliberately wrong-predicted true-source parent is still in main cohort.
    monkeypatch.setattr(route.evaluation, "_predict", lambda parents, *args:
                        [{"predicted_label": 0, "probabilities": [.8, .2]} for _ in parents])
    def pairs(parents, candidates, **kwargs):
        assert kwargs["predictions"][parents[0].parent_id]["predicted_label"] == 0
        return [{"parent_id": p.parent_id, "candidate_id": c["candidate_id"],
                 "applicable": True, "pair_strict_flip": False, "wnode_distance": None, "cf_drop": None}
                for p in parents for c in candidates]
    monkeypatch.setattr(route.evaluation, "_pairs", pairs)
    kwargs = dict(bundle=bundle, manifest=manifest, output=output, universe=rules(), selector=selector(),
                  oracle=object(), featurizer=None, distance=None, binding="b" * 64, batch_size=2, pause=lambda: False)
    return kwargs, observed


def test_real_selector_then_test_and_main_true_source_cohort(tmp_path, monkeypatch):
    kwargs, observed = heldout_fixture(tmp_path, monkeypatch)
    result = route._heldout(**kwargs)
    assert observed == ["calibration", "test"]
    assert result["state"] == "PASS" and result["metrics"]["CCRCov@20"] == 0
    assert result["metrics"]["conditional_median_WNode"] is None
    assert len((kwargs["output"] / "table2_k10.csv").read_text().splitlines()) == 2
    assert route._heldout(**kwargs)["metrics"] == result["metrics"]


@pytest.mark.parametrize("empty", ["calibration", "test"])
def test_empty_cohorts_not_fabricated(tmp_path, monkeypatch, empty):
    kwargs, observed = heldout_fixture(tmp_path, monkeypatch, empty=empty)
    if empty == "calibration":
        with pytest.raises(ValueError, match="EMPTY_CALIBRATION"):
            route._heldout(**kwargs)
        assert observed == ["calibration"]
    else:
        result = route._heldout(**kwargs)
        assert result["metrics"]["CCRCov@20"] is None


def test_resume_rejects_frozen_selection_change(tmp_path, monkeypatch):
    kwargs, _ = heldout_fixture(tmp_path, monkeypatch)
    route._heldout(**kwargs)
    frozen = read_json(kwargs["output"] / "selector_manifest.json")
    frozen["ordered_rule_ids"].reverse()
    atomic_json(kwargs["output"] / "selector_manifest.json", frozen)
    with pytest.raises(ValueError, match="resume_sealed_artifact"):
        route._heldout(**kwargs)


def test_pause_before_test_does_not_parse_test(tmp_path, monkeypatch):
    kwargs, observed = heldout_fixture(tmp_path, monkeypatch)
    kwargs["pause"] = lambda: (kwargs["output"] / "selector_manifest.json").exists()
    assert route._heldout(**kwargs)["state"] == "PAUSED_AT_SAFE_PARENT_BOUNDARY"
    assert observed == ["calibration"]


def test_independent_registry_refuses_main_and_conflicting_variant(tmp_path):
    output = tmp_path / "result"
    output.mkdir()
    audit = {"binding_sha256": "a" * 64}
    atomic_json(output / "final_audit.json", audit)
    with pytest.raises(ValueError, match="MUST_NOT_BE_MAIN"):
        route._index_result(tmp_path / "control" / "fast16_matrix_authority", "BRICS_FIXED", output, audit)
    registry = tmp_path / "llm_registry"
    route._index_result(registry, "BRICS_FIXED", output, audit)
    route._index_result(registry, "BRICS_FIXED", output, audit)
    with pytest.raises(ValueError, match="VARIANT_CONFLICT"):
        route._index_result(registry, "BRICS_FIXED", output, {"binding_sha256": "b" * 64})


def test_independent_gnn_gate_precedes_every_model_or_output(tmp_path, monkeypatch):
    import src.ablations.llm.corrected_core_gate as verifier
    archive = tmp_path / "archive.tar.gz"
    archive.write_bytes(b"fixture-not-science")
    def blocked(*_args):
        raise ValueError("WAITING_GNN_CORE_SEED7_CORRECTED_PASS")
    monkeypatch.setattr(verifier, "require_corrected_gnn_core", blocked)
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="CORRECTED_PASS"):
        route.run_downstream(task_spec=tmp_path / "missing", candidate_root=tmp_path / "missing_pool",
            gnn_input_bundle=tmp_path / "missing_bundle", gnn_verified_archive=archive,
            gnn_verified_sha256=sha256_file(archive), registry_root=tmp_path / "llm_registry", output_root=output)
    assert not output.exists()
