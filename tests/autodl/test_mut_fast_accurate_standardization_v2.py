from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autodl.run_comrecgc_standardized_continuation import ContinuationInputs
from scripts.autodl import run_mut_comrecgc_parity_standardization as standardizer
from src.baselines.comrecgc.contracts import (
    CF_MODE,
    DISTANCE_LINE,
    METHOD,
    sha256_file,
    stable_json_sha256,
)


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _historical_adoption(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "historical-generation"
    source.mkdir()
    (source / "counterfactuals.pt").write_bytes(b"historical trace-on fixture\n")
    lineage = source / "trace/candidate_action_lineage.json"
    _json(lineage, {"status": "PASS", "candidate_count": standardizer.SOURCE_CANDIDATE_COUNT})

    equivalence = tmp_path / "evidence/equivalence.json"
    equivalence_payload = {
        "schema_version": "mut_checkpoint_instrumentation_equivalence_v1",
        "status": "PASS",
        "paper_eligible": False,
        "dataset": "mutagenicity",
        "steps": 500,
        "seed": 0,
        "source_algorithm_commit": standardizer.SOURCE_PROJECT_COMMIT,
        "execution_instrumentation_commit": standardizer.INSTRUMENTATION_PROJECT_COMMIT,
        "step_action_trace_exact": True,
        "rng_state_exact": True,
        "checkpoint_mirror_verified": True,
        "checkpoint_resume_exercised": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "payload_equivalence": {
            "failures": [],
            "candidate_parity": {"trace_parity_passed": True},
        },
        "source_audit": {
            "legacy": {
                "project_commit": standardizer.SOURCE_PROJECT_COMMIT,
                "inventory_sha256": standardizer.LEGACY_SOURCE_INVENTORY_SHA256,
            },
            "instrumented": {
                "project_commit": standardizer.INSTRUMENTATION_PROJECT_COMMIT,
                "inventory_sha256": (
                    standardizer.INSTRUMENTATION_SOURCE_INVENTORY_SHA256
                ),
            },
            "delta_audit": {"status": "PASS", "failures": []},
        },
        "failures": [],
    }
    equivalence_payload["summary_sha256"] = stable_json_sha256(equivalence_payload)
    _json(equivalence, equivalence_payload)
    (equivalence.parent / "PASS").write_text("PASS\n", encoding="utf-8")

    common = tmp_path / "historical-common"
    external = common / "external"
    vectors = external / "vectors.npy"
    vectors.parent.mkdir(parents=True)
    vectors.write_bytes(b"vectors\n")
    vectors_sha = sha256_file(vectors)
    pair_manifest = external / "pair-store.json"
    _json(
        pair_manifest,
        {
            "run_complete": True,
            "vectors_path": str(vectors.resolve()),
            "vectors_sha256": vectors_sha,
            "scientific_identity": {
                "dataset": "mutagenicity",
                "counterfactuals_sha256": standardizer.SOURCE_PAYLOAD_SHA256,
                "candidate_graph_hashes_sha256": "b" * 64,
                "generation_indices_sha256": "c" * 64,
            },
        },
    )
    dbscan_manifest = external / "dbscan.json"
    _json(
        dbscan_manifest,
        {
            "run_complete": True,
            "scientific_identity": {
                "vectors_path": str(vectors.resolve()),
                "vectors_sha256": vectors_sha,
            },
        },
    )
    common_manifest = common / "run_manifest.json"
    _json(
        common_manifest,
        {
            "dataset": "mutagenicity",
            "method": METHOD,
            "run_complete": True,
            "source_counterfactuals_sha256": standardizer.SOURCE_PAYLOAD_SHA256,
            "common_recourse_count": 19,
            "external_memory_artifacts": {
                "engine": "external_memory_exact_v1",
                "pair_store_manifest": str(pair_manifest.resolve()),
                "pair_store_manifest_sha256": sha256_file(pair_manifest),
                "dbscan_manifest": str(dbscan_manifest.resolve()),
                "dbscan_manifest_sha256": sha256_file(dbscan_manifest),
            },
        },
    )

    universe = "b" * 64
    receipt = tmp_path / "historical-adoption.json"
    receipt_payload = {
            "schema_version": standardizer.HISTORICAL_ADOPTION_SCHEMA,
            "status": "PASS",
            "dataset": "mutagenicity",
            "method": METHOD,
            "historical_artifact_adopted": True,
            "historical_source_trace_enabled": True,
            "traceoff_reference_rerun": False,
            "trace_parity_passed": False,
            "500_step_semantic_equivalence_passed": True,
            "adoption_without_full_50k_parity_rerun_authorized": True,
            "generation_complete": True,
            "generation_steps": 50_000,
            "M_MAX": 50_000,
            "M_EFFECTIVE": 50_000,
            "candidate_capacity": 100_000,
            "candidate_count": standardizer.SOURCE_CANDIDATE_COUNT,
            "lineage_pass": True,
            "candidate_freeze_pass": True,
            "checkpoint_reload_pass": True,
            "no_test_leakage": True,
            "calibration_loaded": False,
            "test_loaded": False,
            "no_active_writer": True,
            "pair_store_reused": True,
            "dbscan_reused": True,
            "pair_store_recompute_performed": False,
            "dbscan_recompute_performed": False,
            "candidate_universe_sha": universe,
            "pair_store_source_candidate_universe_sha": universe,
            "dbscan_source_candidate_universe_sha": universe,
            "candidate_universe_binding_state": "PASS",
            "transitive_binding_kind": (
                "pair_candidate_universe_via_exact_generation_payload_and_dbscan_vectors"
            ),
            "pair_candidate_graph_hashes_sha256": universe,
            "dbscan_native_candidate_universe_field_present": False,
            "dbscan_universe_binding_via_pair_vectors": True,
            "source_generation_root": str(source.resolve()),
            "source_payload_path": str((source / "counterfactuals.pt").resolve()),
            "source_payload_sha256": standardizer.SOURCE_PAYLOAD_SHA256,
            "source_lineage_path": str(lineage.resolve()),
            "source_lineage_sha256": sha256_file(lineage),
            "500_step_semantic_equivalence_receipt_path": str(equivalence.resolve()),
            "500_step_semantic_equivalence_receipt_sha256": sha256_file(equivalence),
            "source_common_recourse_root": str(common.resolve()),
            "source_common_recourse_manifest_path": str(common_manifest.resolve()),
            "source_common_recourse_manifest_sha256": sha256_file(common_manifest),
            "source_pair_store_manifest_path": str(pair_manifest.resolve()),
            "source_pair_store_manifest_sha256": sha256_file(pair_manifest),
            "source_dbscan_manifest_path": str(dbscan_manifest.resolve()),
            "source_dbscan_manifest_sha256": sha256_file(dbscan_manifest),
            "common_recourse_count": 19,
    }
    receipt_payload["binding_sha256"] = stable_json_sha256(receipt_payload)
    _json(receipt, receipt_payload)
    return receipt, source


def _inputs(tmp_path: Path, *, source: Path, output: Path) -> ContinuationInputs:
    directories = {
        name: tmp_path / name for name in ("upstream", "dataset", "molclr")
    }
    for path in directories.values():
        path.mkdir(exist_ok=True)
    files = {
        name: tmp_path / name
        for name in (
            "distance.pt",
            "dataset.csv",
            "teacher.pkl",
            "molclr.pt",
            "thresholds.json",
        )
    }
    for path in files.values():
        path.write_bytes(b"fixture\n")
    return ContinuationInputs(
        dataset="mutagenicity",
        source_generation_root=source,
        upstream_root=directories["upstream"],
        dataset_dir=directories["dataset"],
        source_csv=None,
        distance_checkpoint=files["distance.pt"],
        dataset_csv=files["dataset.csv"],
        teacher_path=files["teacher.pkl"],
        molclr_root=directories["molclr"],
        molclr_checkpoint=files["molclr.pt"],
        thresholds_path=files["thresholds.json"],
        output_root=output,
        device="cpu",
        theta_star=None,
        cost_cap=None,
    )


def test_historical_adoption_reopens_trace_on_50k_transitive_binding(
    tmp_path: Path,
) -> None:
    receipt, source = _historical_adoption(tmp_path)

    reopened = standardizer._validate_historical_adoption(receipt, source_root=source)

    assert reopened["historical_source_trace_enabled"] is True
    assert reopened["traceoff_reference_rerun"] is False
    assert reopened["trace_parity_passed"] is False
    assert reopened["generation_steps"] == 50_000
    assert reopened["M_EFFECTIVE"] == 50_000
    assert reopened["transitive_binding_kind"] == (
        "pair_candidate_universe_via_exact_generation_payload_and_dbscan_vectors"
    )
    assert reopened["candidate_universe_sha"] == "b" * 64
    assert reopened["pair_candidate_graph_hashes_sha256"] == "b" * 64
    assert reopened["dbscan_native_candidate_universe_field_present"] is False


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("historical_source_trace_enabled", False),
        ("traceoff_reference_rerun", True),
        ("trace_parity_passed", True),
        ("500_step_semantic_equivalence_passed", False),
        ("adoption_without_full_50k_parity_rerun_authorized", False),
        ("pair_store_source_candidate_universe_sha", "e" * 64),
    ),
)
def test_historical_adoption_rejects_false_or_drifted_claims(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    receipt, source = _historical_adoption(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload[field] = bad_value
    _json(receipt, payload)

    with pytest.raises(ValueError, match="Historical Mut adoption is invalid"):
        standardizer._validate_historical_adoption(receipt, source_root=source)


def test_historical_adoption_rejects_receipt_universe_not_in_pair_identity(
    tmp_path: Path,
) -> None:
    receipt, source = _historical_adoption(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    for field in (
        "candidate_universe_sha",
        "pair_store_source_candidate_universe_sha",
        "dbscan_source_candidate_universe_sha",
        "pair_candidate_graph_hashes_sha256",
    ):
        payload[field] = "e" * 64
    _json(receipt, payload)

    with pytest.raises(ValueError, match="pair_store_candidate_universe_binding"):
        standardizer._validate_historical_adoption(receipt, source_root=source)


def test_historical_adoption_rejects_minimal_forged_equivalence_receipt(
    tmp_path: Path,
) -> None:
    receipt, source = _historical_adoption(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    equivalence = Path(payload["500_step_semantic_equivalence_receipt_path"])
    _json(
        equivalence,
        {
            "schema_version": "mut_checkpoint_instrumentation_equivalence_v1",
            "status": "PASS",
            "paper_eligible": False,
            "dataset": "mutagenicity",
            "steps": 500,
            "step_action_trace_exact": True,
            "rng_state_exact": True,
            "checkpoint_mirror_verified": True,
            "checkpoint_resume_exercised": True,
            "calibration_loaded": False,
            "test_loaded": False,
        },
    )
    payload["500_step_semantic_equivalence_receipt_sha256"] = sha256_file(
        equivalence
    )
    payload["binding_sha256"] = stable_json_sha256(
        {key: item for key, item in payload.items() if key != "binding_sha256"}
    )
    _json(receipt, payload)

    with pytest.raises(ValueError, match="historical_generation_evidence"):
        standardizer._validate_historical_adoption(receipt, source_root=source)


def test_historical_adoption_rejects_binding_digest_drift(tmp_path: Path) -> None:
    receipt, source = _historical_adoption(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["common_recourse_count"] += 1
    _json(receipt, payload)

    with pytest.raises(ValueError, match="binding_sha256"):
        standardizer._validate_historical_adoption(receipt, source_root=source)


def test_run_emits_truthful_fast_accurate_v2_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, source = _historical_adoption(tmp_path)
    inputs = _inputs(tmp_path, source=source, output=tmp_path / "fresh-output")
    monkeypatch.setattr(
        standardizer,
        "validate_adopted_generation",
        lambda _inputs: {
            "source_generation_root": str(source.resolve()),
            "counterfactual_candidate_count": standardizer.SOURCE_CANDIDATE_COUNT,
        },
    )
    monkeypatch.setattr(standardizer, "verify_checkout", lambda *_args, **_kwargs: {"passed": True})
    monkeypatch.setattr(standardizer, "_git_head", lambda: "a" * 40)
    monkeypatch.setattr(
        standardizer,
        "_verify_adopted_generation_integrity",
        lambda _adoption: {"status": "PASS"},
    )

    def _no_science_commands(
        current: ContinuationInputs, **_kwargs: object
    ) -> list[tuple[str, list[str], Path, str]]:
        teacher_sha = sha256_file(current.teacher_path)
        _json(
            current.output_root / "standardized/run_manifest.json",
            {
                "dataset_key": "mutagenicity",
                "cf_mode": CF_MODE,
                "distance_line": DISTANCE_LINE,
                "teacher_sha256": teacher_sha,
            },
        )
        _json(
            current.output_root / "standardized/freeze_manifest.json",
            {"dataset_key": "mutagenicity"},
        )
        return []

    monkeypatch.setattr(standardizer, "_commands", _no_science_commands)

    result = standardizer.run(
        inputs,
        common_adoption_path=None,
        historical_adoption_path=receipt,
    )

    assert result["schema_version"] == standardizer.FAST_ACCURATE_RUN_SCHEMA
    assert result["historical_source_trace_enabled"] is True
    assert result["traceoff_reference_rerun"] is False
    assert result["trace_parity_passed"] is False
    assert result["500_step_semantic_equivalence_passed"] is True
    assert result["adoption_without_full_50k_parity_rerun_authorized"] is True
    assert result["generation_steps"] == result["M_EFFECTIVE"] == 50_000
    assert result["trace_parity_path"] is None
    assert result["trace_parity_sha256"] is None
    assert result["candidate_universe_sha"] == "b" * 64
    assert result["pair_candidate_graph_hashes_sha256"] == "b" * 64
    assert result["dbscan_native_candidate_universe_field_present"] is False
    assert result["dbscan_universe_binding_via_pair_vectors"] is True
    assert (inputs.output_root / "PASS").read_bytes() == b"PASS\n"
