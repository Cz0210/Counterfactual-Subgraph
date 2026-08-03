from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.baselines.gcfexplainer import recover_mutagenicity_vrrw_run as recovery


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "scripts/slurm/gcfexplainer/recover_mutagenicity_vrrw.sh"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class RecoveryCase:
    def __init__(self, tmp_path: Path, payload: object | None = None) -> None:
        torch = pytest.importorskip("torch")
        self.failed_run = tmp_path / "failed_vrrw"
        self.failed_run.mkdir()
        self.output_dir = tmp_path / "recovered_vrrw"
        self.gnn = tmp_path / "gnn_model_best.pth"
        self.neurosed = tmp_path / "mutagenicity_neurosed.pt"
        self.gnn.write_bytes(b"frozen-mutagenicity-gnn")
        self.neurosed.write_bytes(b"frozen-mutagenicity-neurosed")
        self.parent_ids = [f"MUT_{index:04d}" for index in range(1448)]
        if payload is None:
            payload = {
                "graph_map": {
                    "hash-b": {"node": "b"},
                    "hash-a": {"node": "a"},
                },
                "graph_index_map": {"hash-b": 0, "hash-a": 1},
                "counterfactual_candidates": [
                    {
                        "graph_hash": "hash-b",
                        "frequency": 9,
                        "importance_parts": [0.9, 0.2],
                    },
                    {
                        "graph_hash": "hash-a",
                        "frequency": 4,
                        "importance_parts": [0.8, 0.1],
                    },
                ],
                "MAX_COUNTERFACTUAL_SIZE": 100000,
                "traversed_hashes": ["hash-b", "hash-a"],
                "input_graphs_covered": torch.zeros(1448),
            }
        self.counterfactuals = self.failed_run / "counterfactuals.pt"
        torch.save(payload, self.counterfactuals)
        self.config = {
            "dataset": "Mutagenicity",
            "dataset_name": "mutagenicity",
            "dataset_dir": str(tmp_path / "phase_a_dataset"),
            "official_root": str(tmp_path / "gcfexplainer_official"),
            "profile": "full",
            "gnn_checkpoint": str(self.gnn.resolve()),
            "gnn_checkpoint_sha256": _sha256(self.gnn),
            "neurosed_checkpoint": str(self.neurosed.resolve()),
            "neurosed_checkpoint_sha256": _sha256(self.neurosed),
            "parent_limit": 1448,
            "generation_source_parent_rows": 1448,
            "generation_parent_ids": list(self.parent_ids),
            "generation_source_cohort_hash": recovery.stable_json_sha256(
                self.parent_ids
            ),
            "M": 50000,
            "alpha": 1.0,
            "alpha_endpoint_branch": "individual_only",
            "theta": 0.05,
            "theta_source": "official_vrrw_mutagenicity_default",
            "distance_normalization": (
                "neurosed_divided_by_sum_graph_element_counts"
            ),
            "teleport": 0.1,
            "dynamic_teleportation": True,
            "candidate_capacity": 100000,
            "sample": False,
            "sample_size": 10000,
            "seed": 13,
            "node_feature_dim": 9,
            "calibration_loaded": False,
            "test_loaded": False,
            "resume_mode": "deterministic_restart_from_seed",
            "official_compatibility_patches": [
                recovery.VRRW_ALPHA_ENDPOINT_PATCH
            ],
        }
        self.write_config()
        predictions = self.failed_run / "internal_gnn_predictions.jsonl"
        with predictions.open("w", encoding="utf-8") as handle:
            for index, parent_id in enumerate(self.parent_ids):
                handle.write(
                    json.dumps(
                        {
                            "molecule_id": parent_id,
                            "source_graph_hash": f"source-{index}",
                            "project_label": 1,
                            "official_gnn_prediction": index % 2,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        failure = {
            "run_complete": False,
            "stage": "vrrw_runtime",
            "job_id": "2074516",
            "error_type": "OSError",
            "error": "[Errno 28] No space left on device while persisting artifact",
            "calibration_loaded": False,
            "test_loaded": False,
        }
        _write_json(self.failed_run / "_RUN_FAILED.json", failure)
        _write_json(self.failed_run / "failure_summary.json", failure)
        (self.failed_run / "visited_graph_universe.pt").write_bytes(b"x" * 128)

    def write_config(self) -> None:
        self.config.pop("config_fingerprint", None)
        self.config["config_fingerprint"] = recovery.stable_json_sha256(
            self.config
        )
        _write_json(self.failed_run / "resolved_config.json", self.config)

    def request(self, **changes: object) -> recovery.RecoveryRequest:
        request = recovery.RecoveryRequest(
            failed_run_dir=self.failed_run.resolve(),
            counterfactuals_path=self.counterfactuals.resolve(),
            output_dir=self.output_dir.resolve(),
            expected_profile="full",
            expected_parent_limit=1448,
            expected_m=50000,
            expected_alpha=1.0,
            expected_theta=0.05,
            expected_seed=13,
            expected_job_id="2074516",
            expected_bytes=self.counterfactuals.stat().st_size,
            expected_sha256=_sha256(self.counterfactuals),
        )
        return replace(request, **changes)

    def argv(self, **changes: object) -> list[str]:
        request = self.request(**changes)
        return [
            "--failed-run-dir",
            str(request.failed_run_dir),
            "--counterfactuals-path",
            str(request.counterfactuals_path),
            "--output-dir",
            str(request.output_dir),
            "--expected-profile",
            request.expected_profile,
            "--expected-parent-limit",
            str(request.expected_parent_limit),
            "--expected-m",
            str(request.expected_m),
            "--expected-alpha",
            str(request.expected_alpha),
            "--expected-theta",
            str(request.expected_theta),
            "--expected-seed",
            str(request.expected_seed),
            "--expected-job-id",
            request.expected_job_id,
            "--expected-bytes",
            str(request.expected_bytes),
            "--expected-sha256",
            request.expected_sha256,
        ]


@pytest.fixture
def case(tmp_path: Path) -> RecoveryCase:
    return RecoveryCase(tmp_path)


def test_single_authoritative_counterfactual_artifact_recovers(case: RecoveryCase) -> None:
    manifest = recovery.recover_vrrw_run(case.request())
    assert manifest["run_complete"] is True
    assert manifest["recovered_run"] is True
    assert manifest["algorithm_rerun"] is False
    assert manifest["counterfactual_candidate_count"] == 2
    assert (case.output_dir / "_RUN_COMPLETE.json").is_file()
    assert not (case.output_dir / recovery.VISITED_UNIVERSE_NAME).exists()


def test_missing_official_runtime_copy_is_not_required(case: RecoveryCase) -> None:
    assert not (case.failed_run / recovery.REDUNDANT_COPY_RELATIVE_PATH).exists()
    recovery.recover_vrrw_run(case.request())
    audit = json.loads(
        (case.output_dir / "recovery_manifest.json").read_text(encoding="utf-8")
    )
    assert audit["redundant_official_copy_status"] == "missing_not_required"


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"expected_bytes": 1}, "size mismatch"),
        ({"expected_sha256": "0" * 64}, "SHA256 mismatch"),
    ),
)
def test_artifact_size_or_sha_mismatch_is_rejected(
    case: RecoveryCase,
    change: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(recovery.GCFExplainerVRRWRecoveryError, match=message):
        recovery.recover_vrrw_run(case.request(**change))
    assert not (case.output_dir / "_RUN_COMPLETE.json").exists()


def test_torch_load_failure_is_rejected(tmp_path: Path) -> None:
    case = RecoveryCase(tmp_path)
    case.counterfactuals.write_bytes(b"not-a-torch-archive")
    with pytest.raises(recovery.GCFExplainerVRRWRecoveryError, match="torch.load"):
        recovery.recover_vrrw_run(case.request())


@pytest.mark.parametrize("payload", ({}, {"counterfactual_candidates": []}))
def test_empty_or_incomplete_payload_is_rejected(
    tmp_path: Path,
    payload: object,
) -> None:
    case = RecoveryCase(tmp_path, payload=payload)
    with pytest.raises(recovery.GCFExplainerVRRWRecoveryError, match="payload"):
        recovery.recover_vrrw_run(case.request())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("profile", "smoke", "profile"),
        ("parent_limit", 64, "parent_limit"),
        ("M", 500, "M"),
        ("alpha", 0.5, "alpha"),
        ("theta", 0.1, "theta"),
        ("seed", 12, "seed"),
    ),
)
def test_resolved_config_contract_mismatch_is_rejected(
    case: RecoveryCase,
    field: str,
    value: object,
    message: str,
) -> None:
    case.config[field] = value
    case.write_config()
    with pytest.raises(recovery.GCFExplainerVRRWRecoveryError, match=message):
        recovery.recover_vrrw_run(case.request())


def test_parent_id_count_and_hash_are_validated(case: RecoveryCase) -> None:
    case.config["generation_parent_ids"] = case.parent_ids[:-1]
    case.config["generation_source_cohort_hash"] = recovery.stable_json_sha256(
        case.parent_ids[:-1]
    )
    case.write_config()
    with pytest.raises(recovery.GCFExplainerVRRWRecoveryError, match="ID count"):
        recovery.recover_vrrw_run(case.request())

    case.config["generation_parent_ids"] = list(case.parent_ids)
    case.config["generation_source_cohort_hash"] = "0" * 64
    case.write_config()
    with pytest.raises(recovery.GCFExplainerVRRWRecoveryError, match="ID hash"):
        recovery.recover_vrrw_run(case.request())


@pytest.mark.parametrize("checkpoint_field", ("gnn_checkpoint", "neurosed_checkpoint"))
def test_checkpoint_hash_mismatch_is_rejected(
    case: RecoveryCase,
    checkpoint_field: str,
) -> None:
    Path(case.config[checkpoint_field]).write_bytes(b"changed-after-vrrw")
    with pytest.raises(recovery.GCFExplainerVRRWRecoveryError, match="Checkpoint SHA256"):
        recovery.recover_vrrw_run(case.request())


def test_internal_predictions_require_exact_1448_parent_order(case: RecoveryCase) -> None:
    predictions = case.failed_run / "internal_gnn_predictions.jsonl"
    lines = predictions.read_text(encoding="utf-8").splitlines()
    predictions.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(recovery.GCFExplainerVRRWRecoveryError, match="row count"):
        recovery.recover_vrrw_run(case.request())


def test_candidate_order_and_bytes_are_unchanged(case: RecoveryCase) -> None:
    source_payload = recovery._torch_load(case.counterfactuals)
    source_order = [
        row["graph_hash"] for row in source_payload["counterfactual_candidates"]
    ]
    source_sha = _sha256(case.counterfactuals)
    recovery.recover_vrrw_run(case.request())
    recovered_path = case.output_dir / "counterfactuals.pt"
    recovered_payload = recovery._torch_load(recovered_path)
    recovered_order = [
        row["graph_hash"]
        for row in recovered_payload["counterfactual_candidates"]
    ]
    assert recovered_order == source_order == ["hash-b", "hash-a"]
    assert _sha256(recovered_path) == source_sha


def test_same_filesystem_prefers_hardlink(case: RecoveryCase) -> None:
    manifest = recovery.recover_vrrw_run(case.request())
    target = case.output_dir / "counterfactuals.pt"
    assert manifest["artifact_materialization_mode"] == "hardlink"
    assert os.stat(case.counterfactuals).st_ino == os.stat(target).st_ino


def test_cross_filesystem_link_failure_uses_atomic_copy(
    case: RecoveryCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_link(_source: Path, _target: Path) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(recovery.os, "link", fail_link)
    manifest = recovery.recover_vrrw_run(case.request())
    assert manifest["artifact_materialization_mode"] == "atomic_copy"
    assert _sha256(case.output_dir / "counterfactuals.pt") == _sha256(
        case.counterfactuals
    )


def test_failed_run_files_are_not_modified(case: RecoveryCase) -> None:
    before = {
        path.relative_to(case.failed_run): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            _sha256(path),
        )
        for path in case.failed_run.rglob("*")
        if path.is_file()
    }
    recovery.recover_vrrw_run(case.request())
    after = {
        path.relative_to(case.failed_run): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            _sha256(path),
        )
        for path in case.failed_run.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_summary_marks_visited_universe_optional_not_required(case: RecoveryCase) -> None:
    assert recovery.summary_requires_visited_graph_universe() is False
    manifest = recovery.recover_vrrw_run(case.request())
    assert manifest["visited_graph_universe_status"] == "optional_not_required"


def test_recovery_fails_if_summary_requires_unrebuildable_visited_universe(
    case: RecoveryCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recovery,
        "summary_requires_visited_graph_universe",
        lambda: True,
    )
    assert recovery.main(case.argv()) == 2
    assert not (case.output_dir / "_RUN_COMPLETE.json").exists()
    assert (case.output_dir / "_RUN_FAILED.json").is_file()


def test_algorithm_failure_cannot_be_recovered_as_persistence_failure(
    case: RecoveryCase,
) -> None:
    marker = {
        "run_complete": False,
        "stage": "vrrw_runtime",
        "error_type": "ValueError",
        "error": "importance matrix shape mismatch",
    }
    _write_json(case.failed_run / "_RUN_FAILED.json", marker)
    _write_json(case.failed_run / "failure_summary.json", marker)
    (case.failed_run / "visited_graph_universe.pt").unlink()
    with pytest.raises(recovery.GCFExplainerVRRWRecoveryError, match="not proven"):
        recovery.recover_vrrw_run(case.request())


def test_complete_marker_is_written_only_after_full_validation(case: RecoveryCase) -> None:
    assert recovery.main(case.argv(expected_sha256="0" * 64)) == 2
    assert (case.output_dir / "failure_summary.json").is_file()
    assert (case.output_dir / "_RUN_FAILED.json").is_file()
    assert not (case.output_dir / "_RUN_COMPLETE.json").exists()


def test_calibration_or_test_lineage_is_rejected(case: RecoveryCase) -> None:
    case.config["test_loaded"] = True
    case.write_config()
    with pytest.raises(recovery.GCFExplainerVRRWRecoveryError, match="test_loaded"):
        recovery.recover_vrrw_run(case.request())


def test_nonempty_output_is_blocked_without_modification(case: RecoveryCase) -> None:
    case.output_dir.mkdir()
    sentinel = case.output_dir / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    assert recovery.main(case.argv()) == 2
    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    assert sorted(path.name for path in case.output_dir.iterdir()) == ["sentinel.txt"]


def test_recovery_does_not_modify_official_source(case: RecoveryCase) -> None:
    official_source = ROOT / "baselines/gcfexplainer_official/vrrw.py"
    before = _sha256(official_source)
    recovery.recover_vrrw_run(case.request())
    assert _sha256(official_source) == before


def test_recovery_wrapper_is_cpu_only_and_storage_gated() -> None:
    result = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    text = WRAPPER.read_text(encoding="utf-8")
    cpu = re.search(r"#SBATCH --cpus-per-task=(\d+)", text)
    memory = re.search(r"#SBATCH --mem=(\d+)G", text)
    assert cpu and int(cpu.group(1)) == 1
    assert memory and int(memory.group(1)) <= 16
    assert "#SBATCH --gres=" not in text
    assert "At least 5 GiB free space is required" in text
    assert "count=256" in text
    assert "conv=fsync" in text
    assert "--counterfactuals-path" in text
    assert "--expected-bytes" in text
    assert "--expected-sha256" in text
    assert '[[ "$EXPECTED_JOB_ID" == "2074516" ]]' in text
    assert '[[ "$EXPECTED_BYTES" == "210914250" ]]' in text
    assert recovery.REDUNDANT_COPY_RELATIVE_PATH.as_posix() not in text
    assert "unset http_proxy" not in text
    assert "unset https_proxy" not in text
    assert "unset all_proxy" not in text


def test_cli_requires_all_explicit_recovery_contract_arguments() -> None:
    required = {
        action.dest
        for action in recovery.build_parser()._actions
        if action.required
    }
    assert required == {
        "failed_run_dir",
        "counterfactuals_path",
        "output_dir",
        "expected_profile",
        "expected_parent_limit",
        "expected_m",
        "expected_alpha",
        "expected_theta",
        "expected_seed",
        "expected_job_id",
        "expected_bytes",
        "expected_sha256",
    }
