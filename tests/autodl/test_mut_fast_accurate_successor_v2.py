from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import scripts.autodl.run_mut_comrecgc_parity_standardization as standardizer
import scripts.autodl.run_mut_fast_accurate_v2 as successor
from scripts.autodl.run_four_gpu_recovery_controller import load_controller_manifest


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cgroup_v1_read_only_mount_detection_is_exact() -> None:
    row = (
        "31 24 0:28 / /sys/fs/cgroup/memory ro,nosuid,nodev,noexec,relatime "
        "- cgroup cgroup rw,memory\n"
    )
    assert successor._mount_is_read_only(row, Path("/sys/fs/cgroup/memory"))


def test_spec_accepts_absolute_python_symlink_but_normalizes_target(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    control = runtime / "control"
    repairs = (
        runtime
        / "outputs/autodl/paper_matrix/four_methods_four_datasets_v1/repairs"
    )
    project = tmp_path / "project"
    legacy = tmp_path / "legacy"
    instrumented = tmp_path / "instrumented"
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    for path in (control, repairs, project, legacy, instrumented, proc, cgroup):
        path.mkdir(parents=True, exist_ok=True)
    real_python = tmp_path / "python-real"
    real_python.write_text("#!/bin/sh\n", encoding="utf-8")
    real_python.chmod(0o755)
    python_link = tmp_path / "python"
    python_link.symlink_to(real_python)
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "configs/autodl/mut_fast_accurate_v2.template.json"
        ).read_text(encoding="utf-8")
        .replace("__PROJECT_ROOT__", str(project))
        .replace("__LEGACY_PROJECT_ROOT__", str(legacy))
        .replace("__INSTRUMENTATION_PROJECT_ROOT__", str(instrumented))
        .replace("__TIMESTAMP__", "20260901T000000Z")
    )
    fixture.update(
        {
            "runtime_root": str(runtime),
            "control_root": str(control),
            "fresh_output_root": str(repairs / "fresh"),
            "python": str(python_link),
            "proc_root": str(proc),
            "mountinfo_path": str(tmp_path / "mountinfo"),
            "cgroup_memory_root": str(cgroup),
            "historical_source_root": str(project),
            "completed_common_root": str(project),
        }
    )
    (tmp_path / "mountinfo").write_text("fixture\n", encoding="utf-8")
    spec = _json(tmp_path / "spec.json", fixture)
    loaded = successor.load_spec(spec)
    assert loaded["python"] == str(real_python.resolve())


def test_manifest_reuses_exclusive_controller_but_stops_before_unapproved_adoption(
    tmp_path: Path,
) -> None:
    runtime = Path("/fixture/counterfactual-subgraph-runtime")
    control = runtime / "control"
    threshold = Path("/fixture/thresholds.json")
    spec_path = Path("/fixture/spec.json")
    replay = {
        key: str(Path("/fixture") / key)
        for key in (
            "upstream_root", "dataset_dir", "gnn_checkpoint", "distance_checkpoint"
        )
    }
    spec = {
        "spec_path": str(spec_path),
        "controller_id": "mut-fast-fixture",
        "runtime_root": str(runtime),
        "control_root": str(control),
        "fresh_output_root": str(
            runtime
                / "outputs/autodl/paper_matrix/four_methods_four_datasets_v1/repairs/mut-fast-fixture"
        ),
        "gpu_min_free_memory_mb": 16000,
        "gpu_max_utilization_percent": 10,
        "legacy_project_root": "/fixture/legacy",
        "instrumentation_project_root": "/fixture/instrumented",
        "historical_source_root": "/fixture/historical",
        "allow_trace_on_historical_adoption": False,
        "replay": replay,
        "standardization": {
                "dataset_csv": "/fixture/heldout.csv",
                "teacher_path": "/fixture/teacher.pkl",
                "molclr_root": "/fixture/molclr",
                "molclr_checkpoint": "/fixture/molclr.pt",
            "thresholds_path": str(threshold),
        },
    }
    path = tmp_path / "controller.json"
    successor.build_controller_manifest(spec, path)
    manifest = load_controller_manifest(path)
    assert [task.task_id for task in manifest.tasks] == [
        "mut_fast_equivalence_500",
    ]
    equivalence = manifest.by_id["mut_fast_equivalence_500"]
    assert equivalence.resource == "gpu"
    assert equivalence.gpu_lock_mode == "exclusive"
    template = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "configs/autodl/mut_fast_accurate_v2.template.json"
        ).read_text(encoding="utf-8")
    )
    assert template["allow_trace_on_historical_adoption"] is False


def test_stage_wrapper_never_stops_old_waiter_and_uses_monitored_runner() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/autodl/run_mut_fast_accurate_stage_v2.sh").read_text()
    assert "run-equivalence" in text
    assert "memory_monitor" not in text  # monitoring belongs to the Python child
    for forbidden in ("pkill", "killall", "kill -", "SIGKILL", "drop_caches"):
        assert forbidden not in text
    python_text = (root / "scripts/autodl/run_mut_fast_accurate_v2.py").read_text()
    assert "start_new_session=True" in python_text
    assert "os.killpg(pgid, signal.SIGTERM)" in python_text
    assert "usage > 0.8 * limit" in python_text
    assert "checkpoint_window_peak" in python_text
    assert "except BaseException:" in python_text
    assert "leave its generation descendants orphaned" in python_text
    assert "derive_empirical_memory_admission" in python_text
    assert '"parent_cgroup_max_usage_attributable_to_mut": False' in python_text


def test_binding_receipt_reopens_in_truthful_standardizer(
    tmp_path: Path, monkeypatch
) -> None:
    payload = tmp_path / "historical/counterfactuals.pt"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"frozen-historical-payload")
    payload_sha = _sha(payload)
    monkeypatch.setattr(successor, "SOURCE_PAYLOAD_SHA256", payload_sha)
    monkeypatch.setattr(standardizer, "SOURCE_PAYLOAD_SHA256", payload_sha)

    lineage = _json(
        payload.parent / "trace/candidate_action_lineage.json",
        {"candidate_count": 100235},
    )
    source_manifest = _json(
        payload.parent / "run_manifest.json",
        {
            "counterfactuals_sha256": payload_sha,
            "parameters": {"steps": 50000, "candidate_capacity": 100000},
        },
    )
    generation_sha = _sha(source_manifest)
    vectors = tmp_path / "pair/vectors.npy"
    vectors.parent.mkdir(parents=True)
    vectors.write_bytes(b"vectors")
    vectors_sha = _sha(vectors)
    pair = _json(
        tmp_path / "pair/run_manifest.json",
        {
            "run_complete": True,
            "vectors_path": str(vectors),
            "vectors_sha256": vectors_sha,
            "scientific_identity": {
                "dataset": "mutagenicity",
                "counterfactuals_sha256": payload_sha,
                "generation_manifest_sha256": generation_sha,
                "dataset_fingerprint": successor.SOURCE_DATASET_SHA256,
                "parent_ids_sha256": successor.SOURCE_PARENT_ORDER_SHA256,
                "candidate_count": 50620,
                "candidate_graph_hashes_sha256": "c" * 64,
                "generation_indices_sha256": "d" * 64,
            },
        },
    )
    common = tmp_path / "common"
    pair_adoption = _json(
        common / "external_memory/pair_store_adoption/run_manifest.json",
        {"source_manifest_path": str(pair), "source_manifest_sha256": _sha(pair)},
    )
    dbscan = _json(
        common / "external_memory/dbscan/run_manifest.json",
        {
            "run_complete": True,
            "approximation_used": False,
            "scientific_identity": {
                "vectors_path": str(vectors), "vectors_sha256": vectors_sha
            },
        },
    )
    common_manifest = _json(
        common / "run_manifest.json",
        {
            "dataset": "mutagenicity",
            "method": "COMRECGC",
            "run_complete": True,
            "counterfactuals_sha256": payload_sha,
            "generation_manifest_sha256": generation_sha,
            "common_recourse_count": 100,
            "external_memory_artifacts": {
                "engine": "external_memory_exact_v1",
                "pair_store_manifest": str(pair),
                "pair_store_manifest_sha256": _sha(pair),
                "dbscan_manifest": str(dbscan),
                "dbscan_manifest_sha256": _sha(dbscan),
            },
        },
    )
    inventory = _json(
        tmp_path / "inventory/historical_inventory.json",
        {
            "schema_version": "mut_historical_50k_inventory_v2",
            "status": "PASS",
            "trace_parity_passed": False,
            "source": {
                "status": "PASS",
                "source_candidate_count": 100235,
                "source_payload_actual_sha256": payload_sha,
                "calibration_loaded": False,
                "test_loaded": False,
            },
            "lineage": {
                "path": str(lineage),
                "sha256": _sha(lineage),
                "candidate_count": 100235,
                "candidate_lineage_resolved_count": 100235,
                "recorded_action_replay_mismatch_count": 0,
            },
        },
    )
    equivalence_payload = {
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
    }
    equivalence = _json(tmp_path / "equivalence/equivalence.json", equivalence_payload)
    monkeypatch.setattr(
        successor,
        "validate_instrumentation_equivalence_gate",
        lambda **_: {**equivalence_payload, "path": str(equivalence), "sha256": _sha(equivalence)},
    )
    monkeypatch.setattr(
        standardizer,
        "validate_instrumentation_equivalence_gate",
        lambda **_: {**equivalence_payload, "path": str(equivalence), "sha256": _sha(equivalence)},
    )
    monkeypatch.setattr(
        successor,
        "scan_live_writers",
        lambda *_args, **_kwargs: {"writable_fd_count": 0, "writers": []},
    )
    spec = {
        "historical_source_root": str(payload.parent),
        "completed_common_root": str(common),
        "proc_root": str(tmp_path),
        "allow_historical_adoption_without_full_50k_parity": True,
    }
    output = tmp_path / "adoption"
    result = successor.publish_adoption(
        spec=spec,
        inventory_gate=inventory,
        equivalence_gate=equivalence,
        output_dir=output,
    )
    assert result["candidate_universe_binding_state"] == "PASS"
    reopened = standardizer._validate_historical_adoption(
        output / "historical_adoption.json", source_root=payload.parent
    )
    assert reopened["historical_artifact_adopted"] is True
    assert reopened["trace_parity_passed"] is False
