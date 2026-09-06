from copy import deepcopy
from pathlib import Path

import pytest

from src.ablations.llm import bace_l0_successor as successor
from src.eval.bace_frozen_gnn_contracts import atomic_json, atomic_jsonl, stable_sha256, sha256_file


def seal(path, value):
    atomic_json(path, {**value, "self_sha256": stable_sha256(value)})


def fixture(tmp_path):
    source = tmp_path / "old_failed"
    source.mkdir()
    (source / "writer.lock").touch()
    proof = {"state": "GNN_CORE_SEED7_CORRECTED_PASS", "seed": 7,
        "validation_counts": {k: 187 for k in ("gin", "gcn", "gatv2", "gatedgcn_plus")},
        "counts": {"calibration": 288, "test": 614}, "raw_ot_recomputed_count": 0,
        "cache_provenance_gaps": [], "main_matrix_write": False, "repair_selected_using_test": False,
        **{k: True for k in ("all_weights_unchanged", "gine_unchanged", "candidate_pool_unchanged",
                              "selectors_frozen_before_test", "native_common_metrics_replayed")},
        **{k: "a" * 64 for k in ("independent_science_replay_sha256", "original_package_sha256",
                                 "repair_contract_sha256", "corrective_audit_sha256", "sha256")}, "bytes": 123}
    manifest = {"variant": "BRICS_FIXED", "main_matrix_write": False, "binding_sha256": "b" * 64,
                "gnn_independent_core": proof}
    seal(source / "run_manifest.json", manifest)
    atomic_json(source / "terminal.json", {"candidate_padding_used": False, "required_rules": 20,
        "state": "SCIENTIFIC_FAILED_INSUFFICIENT_VALID_UNIQUE_RULES", "test_loaded": False, "valid_unique_rules": 15})
    rows = []
    for p in range(386):
        current = [{"parent_id": f"p{p}", "candidate_index": i, "candidate_id": f"{p}-{i}",
            "final_fragment": "C" * (1 + (p + i) % 15), "reward_total": .1, "cf_drop": .1, "cf_flip": False,
            **{k: True for k in ("parse_ok", "valid", "connected", "direct_substructure", "oracle_ok")},
            "test_loaded": False, "calibration_loaded": False, "rf_oracle_used": False} for i in range(8)]
        rows.extend(current)
        seal(source / "parent_checkpoints" / "train" / f"{p}.json", {"binding_sha256": "b" * 64, "rows": current})
    from src.ablations.llm.bace_common_downstream import merge_scored_rows
    merged, universe = merge_scored_rows(rows)
    atomic_jsonl(source / "scored_attempts.jsonl", rows)
    atomic_jsonl(source / "candidate_pool.jsonl", merged)
    atomic_jsonl(source / "candidate_universe.jsonl", universe)
    atomic_json(source / "candidate_metrics.json", {"proposal_attempts": 3088})
    accepted = tmp_path / "accepted.json"
    atomic_json(accepted, proof)
    return source, accepted


def test_completed_3088_ledger_adopted_without_oracle_or_archive_replay(tmp_path, monkeypatch):
    source, accepted = fixture(tmp_path)
    before = {str(p): sha256_file(p) for p in source.rglob("*") if p.is_file()}
    import src.ablations.llm.corrected_core_gate as gate
    monkeypatch.setattr(gate, "require_corrected_gnn_core", lambda *_: pytest.fail("No GNN replay"))
    result = successor.prepare(source_train_root=source, corrected_package_receipt=accepted, output_root=tmp_path / "fresh")
    assert result["train_attempts"] == 3088 and result["valid_unique_rules"] == 15
    reopened = successor.load_train_adoption(tmp_path / "fresh" / "protocol_overlay.json")
    assert len(reopened["scored"]) == 3088
    assert before == {str(p): sha256_file(p) for p in source.rglob("*") if p.is_file()}
    assert result["gnn_package_replayed"] is result["train_oracle_repeated"] is False


def test_wrong_checkpoint_or_changed_ledger_fail_closed(tmp_path):
    source, accepted = fixture(tmp_path)
    checkpoint = source / "parent_checkpoints/train/0.json"
    payload = __import__("json").loads(checkpoint.read_text())
    payload["rows"][0]["cf_drop"] = .8
    payload.pop("self_sha256")
    seal(checkpoint, payload)
    with pytest.raises(ValueError, match="CHECKPOINT_MISMATCH"):
        successor.prepare(source_train_root=source, corrected_package_receipt=accepted, output_root=tmp_path / "fresh")


def test_existing_evaluation_not_silently_recomputed(tmp_path):
    source, accepted = fixture(tmp_path)
    atomic_jsonl(source / "test_pairs.jsonl", [])
    with pytest.raises(ValueError, match="DO_NOT_RECOMPUTE"):
        successor.prepare(source_train_root=source, corrected_package_receipt=accepted, output_root=tmp_path / "fresh")


def test_small_gnn_acceptance_wrong_hash_blocks(tmp_path):
    source, accepted = fixture(tmp_path)
    payload = __import__("json").loads(accepted.read_text())
    payload["sha256"] = "c" * 64
    atomic_json(accepted, payload)
    with pytest.raises(ValueError, match="CORRECTIVE_RECEIPT_MISMATCH"):
        successor.prepare(source_train_root=source, corrected_package_receipt=accepted, output_root=tmp_path / "fresh")


def test_overlay_change_after_adoption_blocks(tmp_path):
    source, accepted = fixture(tmp_path)
    successor.prepare(source_train_root=source, corrected_package_receipt=accepted, output_root=tmp_path / "fresh")
    atomic_json(source / "candidate_metrics.json", {"proposal_attempts": 9})
    with pytest.raises(ValueError, match="SOURCE_ARTIFACT_CHANGED"):
        successor.load_train_adoption(tmp_path / "fresh/protocol_overlay.json")


def test_slurm_is_cpu_only_thin_paired_and_no_generation():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/slurm/run_bace_l0_evaluation_successor.sh").read_text()
    assert "--partition=intel" in text and "#SBATCH --gres" not in text
    assert 'CUDA_VISIBLE_DEVICES=""' in text
    assert "scripts/hpc/llm/run_bace_l0_evaluation_successor.py" in text
    assert "--config configs/hpc.yaml" in text
    assert "source ~/.bashrc" in text and "export PYTHONPATH=$PWD" in text


def test_real_saved_matrix_package_replays_without_model_or_ot(tmp_path, monkeypatch):
    from test_bace_llm_common_downstream import heldout_fixture
    from src.ablations.llm import bace_common_downstream as common
    kwargs, _ = heldout_fixture(tmp_path, monkeypatch, rule_count=15)
    common._heldout(**kwargs)
    science, bundle = kwargs["output"], kwargs["bundle"]
    (science / "writer.lock").touch()
    atomic_json(bundle / "bundle_manifest.json", kwargs["manifest"])
    atomic_json(science / "run_manifest.json", {"bundle_sha256": sha256_file(bundle / "bundle_manifest.json")})
    atomic_jsonl(science / "candidate_universe.jsonl", kwargs["universe"])
    audit = {"state": "PASS", "main_matrix_write": False, "selection_policy": successor.POLICY,
             "binding_sha256": "b" * 64,
             "files": {p.name: sha256_file(p) for p in science.iterdir() if p.is_file() and p.name != "writer.lock"}}
    atomic_json(science / "final_audit.json", audit)
    monkeypatch.setattr(common.evaluation, "frozen_selector", lambda *_: kwargs["selector"])
    monkeypatch.setattr(common.evaluation, "_distance", lambda *_: pytest.fail("No OT during package"))
    result = successor.package(science_root=science, gnn_input_bundle=bundle, output_root=tmp_path / "package")
    assert result["state"] == "PASS" and result["K_EFFECTIVE"] == 15
    import tarfile
    with tarfile.open(result["path"]) as archive:
        assert "result/test_pairs.jsonl" in archive.getnames()
        assert "result/final_audit.json" in archive.getnames()
        assert all("cache/" not in n for n in archive.getnames())
    # Import checks small transport/inner hashes only; no numerical work repeats.
    import_root = tmp_path / "autodl_fresh"
    publication = successor.import_package(archive=result["path"], package_receipt=tmp_path / "package/result_package.json",
        output_root=import_root, registry_root=tmp_path / "llm_registry")
    assert publication["state"] == "PASS" and publication["scientific_recomputation"] is False
    assert (import_root / "result/table2_k10.csv").read_bytes() == (science / "table2_k10.csv").read_bytes()
    with pytest.raises(ValueError, match="FRESH_DISJOINT"):
        successor.import_package(archive=result["path"], package_receipt=tmp_path / "package/result_package.json",
            output_root=import_root, registry_root=tmp_path / "llm_registry")
