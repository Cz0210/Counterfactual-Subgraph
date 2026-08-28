from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts import run_tastemolnet_globalgce_smoke as cli
from src.baselines import tastemolnet_globalgce_smoke as t8
from src.utils import tastemolnet_t8_globalgce_release as release


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True).encode("utf-8")


def test_release_config_is_exact_and_default_disabled() -> None:
    payload = json.loads(release.RELEASE_CONFIG_PATH.read_text(encoding="utf-8"))
    assert set(payload) == release.RELEASE_KEYS
    assert payload["release_enabled"] is False
    assert payload["release_state"] == release.DISABLED_RELEASE_STATE
    assert payload["gpu_index"] == 2
    assert all(payload[field] is None for field in release.RELEASE_PIN_FIELDS)
    with pytest.raises(
        release.TasteT8ReleaseDisabled,
        match="TASTE_T8_GLOBALGCE_EXECUTION_NOT_RELEASED",
    ):
        release.assert_execution_released()


def test_cli_refuses_disabled_release_before_science_paths(capsys) -> None:
    assert cli.main(
        [
            "--config",
            str(PROJECT_ROOT / "configs/hpc.yaml"),
            "--output-dir",
            "/caller/selected/output",
        ]
    ) == 78
    captured = capsys.readouterr()
    assert "TASTE_T8_GLOBALGCE_EXECUTION_NOT_RELEASED" in captured.err
    assert captured.out == ""


def test_validate_only_refuses_without_independent_managed_authority() -> None:
    with pytest.raises(
        t8.TasteGlobalGCESmokeError,
        match="independent managed-v2 verifier authority adapter",
    ):
        cli.main(
            [
                "--config",
                str(PROJECT_ROOT / "configs/hpc.yaml"),
                "--output-dir",
                "/not-opened/self-signed-root",
                "--validate-only",
            ]
        )


def _external_managed_v2_evidence(closure: dict) -> dict:
    return {
        "schema_version": t8.MANAGED_V2_EXTERNAL_AUTHORITY_SCHEMA,
        "status": "HELD_ACTIVE_VALID",
        "protocol": t8.MANAGED_V2_PROTOCOL,
        "protocol_source_commit": t8.MANAGED_V2_SOURCE_COMMIT,
        "task_id": t8.MANAGED_TASK_ID,
        "run_id": "controller-run-v2",
        "stage": t8.STAGE,
        "authority_record_sha256": "1" * 64,
        "active_generation_sha256": "2" * 64,
        "child_identity_sha256": "3" * 64,
        "process_lineage_sha256": "4" * 64,
        "expected_closure_sha256": t8._canonical_sha256(closure),
        "gpu_index": 2,
        "gpu_uuid": "GPU-managed-v2-fixture",
        "gpu_lock_mode": "exclusive",
        "auto_terminate_uncontrolled_children": False,
    }


class _HeldExternalManagedV2Fixture:
    def __init__(self, evidence: dict) -> None:
        self.evidence = json.loads(json.dumps(evidence))

    def revalidate_t8_managed_v2_authority(self) -> dict:
        return json.loads(json.dumps(self.evidence))

    def revalidate_t8_official_startup_authority(self) -> dict:
        return {"fixture": "captured independently before worker startup"}


def test_external_managed_v2_authority_is_held_and_cross_bound() -> None:
    closure = {"expected": {"task": t8.MANAGED_TASK_ID, "gpu": 2}}
    held = _HeldExternalManagedV2Fixture(
        _external_managed_v2_evidence(closure)
    )
    binding = release._managed_binding(held, expected_closure=closure)
    assert binding["protocol"] == "managed_execution_v2"
    assert binding["protocol_source_commit"] == t8.MANAGED_V2_SOURCE_COMMIT
    assert binding["expected_closure_sha256"] == t8._canonical_sha256(closure)
    assert binding["same_child_revalidated_at_terminal"] is True

    with pytest.raises(
        t8.TasteGlobalGCESmokeError,
        match="held external object",
    ):
        release._managed_binding(
            _external_managed_v2_evidence(closure),
            expected_closure=closure,
        )
    with pytest.raises(
        t8.TasteGlobalGCESmokeError,
        match="authority changed",
    ):
        release._managed_binding(held, expected_closure={"forged": True})


def test_external_managed_v2_authority_rejects_legacy_holder_shape() -> None:
    class LegacyHolder:
        def revalidate(self) -> dict:
            return {"receipt_kind": "taste_t8_gpu2_v1"}

    with pytest.raises(
        t8.TasteGlobalGCESmokeError,
        match="adapter is absent",
    ):
        release._managed_binding(LegacyHolder(), expected_closure={})


def test_checkpoint_contract_never_opens_nontrain_split_paths() -> None:
    model = b"one frozen model"
    checkpoint_id = hashlib.sha256(model).hexdigest()
    feature = {"schema_sha256": "f" * 64}
    files = {
        role: {"path": f"/deliberately-absent/{role}.csv", "sha256": key * 64}
        for role, key in zip(
            ("train", "validation", "calibration", "test"),
            ("1", "2", "3", "4"),
            strict=True,
        )
    }
    split = {
        "schema_version": "molecular_gnn_split_manifest_v1",
        "dataset": "tastemolnet",
        "roles": {
            "train": "model_fitting",
            "validation": "checkpoint_selection_and_temperature_calibration",
            "calibration": "reserved_for_threshold_and_selector_only",
            "test": "frozen_model_final_quality_evaluation",
        },
        "files": files,
        "train_manifest": {
            "schema_version": "molecular_graph_dataset_v1",
            "num_records": 6,
            "num_classes": 3,
            "label_counts": {"0": 2, "1": 2, "2": 2},
            "split_counts": {"train": 6},
            "source_path": files["train"]["path"],
            "source_sha256": files["train"]["sha256"],
            "dataset_fingerprint": "5" * 64,
            "feature_schema_sha256": feature["schema_sha256"],
        },
        "validation_manifest": {},
        "calibration_loaded_for_training": False,
        "test_loaded_for_training": False,
        "test_evaluated_during_training": False,
        "test_used_for_checkpoint_selection": False,
    }
    payloads = {
        "model.pt": model,
        "model_card.json": _json_bytes(
            {
                "dataset": "tastemolnet",
                "oracle_backend": "gnn",
                "rf_oracle_used": False,
                "backbone": "gine",
                "num_classes": 3,
                "source_label": 1,
                "profile": "full",
                "checkpoint_id": checkpoint_id,
            }
        ),
        "feature_schema.json": _json_bytes(feature),
        "label_map.json": _json_bytes(
            {"0": "Bitter", "1": "Sweet", "2": "Tasteless"}
        ),
        "split_manifest.json": _json_bytes(split),
        "test_evaluation_status.json": _json_bytes(
            {
                "schema_version": "molecular_gnn_test_evaluation_status_v1",
                "status": "NOT_EVALUATED",
                "test_loaded": False,
                "reason": "held_out_until_frozen_final_evaluation",
                "path": files["test"]["path"],
                "sha256": files["test"]["sha256"],
            }
        ),
        "temperature_scaling.json": _json_bytes(
            {
                "temperature": 1.25,
                "selection_split": "validation",
                "test_used_for_fit": False,
            }
        ),
    }
    contract, train = release._checkpoint_train_contract(
        payloads=payloads,
        checkpoint_evidence={"checkpoint_id": checkpoint_id},
    )
    assert train == Path("/deliberately-absent/train.csv")
    assert contract["row_count"] == 6
    assert contract["label_counts"] == {"0": 2, "1": 2, "2": 2}


def test_autodl_wrapper_is_disabled_managed_gpu2_and_exact_terminal() -> None:
    path = PROJECT_ROOT / "scripts/autodl/run_tastemolnet_globalgce_smoke.sh"
    text = path.read_text(encoding="utf-8")
    assert text.count("TASTE_T8_GLOBALGCE_WRAPPER_RELEASED=") == 1
    assert "TASTE_T8_GLOBALGCE_WRAPPER_RELEASED=0" in text
    refusal = text.index("TASTE_T8_GLOBALGCE_WRAPPER_NOT_RELEASED")
    assert refusal < text.index("source \"$SCRIPT_DIR/common.sh\"")
    assert "T8_MANAGED_V2_GPU_ACTIVE_AUTHORITY_ADAPTER_NOT_FROZEN" in text
    assert text.index("T8_MANAGED_V2_GPU_ACTIVE_AUTHORITY_ADAPTER_NOT_FROZEN") < (
        text.index('exec "$AUTODL_PYTHON" -I -B "$RUNNER"')
    )
    assert "exp_run.py" not in text
    assert "--execution-receipt-kind" not in text
    assert "--strict-result-validator" not in text
    assert "--required-log-marker" not in text
    assert "--required-output-file" not in text
    assert "--set inference.fallback_to_heuristic=false" in text
    assert "export PYTHONNOUSERSITE=1" in text
    assert '"$AUTODL_PYTHON" -I -B "$RUNNER"' in text
    assert "rf_model.pkl" not in text.lower()
    assert "bace" not in text.lower()


def test_slurm_is_static_refusal_with_full_cli_parity() -> None:
    path = PROJECT_ROOT / "scripts/slurm/run_tastemolnet_globalgce_smoke.sh"
    text = path.read_text(encoding="utf-8")
    for literal in (
        "#SBATCH --partition=A800",
        "#SBATCH --gres=gpu:a800:1",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
        "source ~/.bashrc",
        "conda activate smiles_pip118",
        "cd /share/home/u20526/czx/counterfactual-subgraph",
        "export PYTHONPATH=$PWD",
    ):
        assert literal in text
    assert text.index("exit 64") < text.index(
        "python -I -B scripts/run_tastemolnet_globalgce_smoke.py"
    )
    for option in (
        "--config configs/hpc.yaml",
        "--stage T8_GLOBALGCE_SMOKE",
        "--t2-adoption",
        "--t3-output",
        "--t4-output",
        "--gnn-checkpoint",
        "--train-csv",
        "--official-root",
        "--downstream-policy",
        "--base-policy",
        "--state-dir",
        "--output-dir",
        "--set inference.fallback_to_heuristic=false",
    ):
        assert option in text


def test_production_runner_uses_only_native_three_class_globalgce() -> None:
    runner = (
        PROJECT_ROOT / "scripts/run_tastemolnet_globalgce_smoke.py"
    ).read_text(encoding="utf-8")
    assert "OfficialGlobalGCEMutagenicityGenerator" in runner
    assert "FrozenTasteGINEScorer" in runner
    assert "source_label=SOURCE_LABEL" in runner
    assert "num_classes=NUM_CLASSES" in runner
    assert "frozen_gine_payloads=inputs.checkpoint_payloads" in runner
    assert "native_train_payload=inputs.train_bytes" in runner
    assert "require_isolated_imports=True" in runner
    assert "RandomForestClassifier" not in runner
    assert "rf_model.pkl" not in runner
    assert "bace" not in runner.lower()
    release_source = (
        PROJECT_ROOT / "src/utils/tastemolnet_t8_globalgce_release.py"
    ).read_text(encoding="utf-8")
    assert "hold_injected_active_execution" not in release_source
    assert "managed_holder" not in release_source
    assert "taste_t8_gpu2_v1" not in release_source
    assert "taste_t8_v1" not in release_source
    assert "revalidate_t8_managed_v2_authority" in release_source
    assert "managed-v2 external GPU/ACTIVE" in release_source
    assert "create_managed_attempt" in runner
    assert "create_worker_staging" in runner
    assert "seal_t8_worker_evidence" in runner
    assert "publish_terminal_output" not in runner


def _make_clean_official_git_checkout(root: Path) -> None:
    files = {
        ".gitignore": "__pycache__/\n*.pyc\n",
        "src/main.py": "# main\n",
        "src/models/__init__.py": "# models\n",
        "src/models/GlobalGCE.py": "class GlobalGCE: pass\n",
        "src/data/data_preprocess.py": "# data\n",
    }
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=T8 Test",
            "-c",
            "user.email=t8@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )


def test_official_git_authority_rejects_ignored_pyc_runtime_code(
    tmp_path: Path,
) -> None:
    root = tmp_path / "official"
    _make_clean_official_git_checkout(root)
    release._official_git_snapshot(root)
    cache = root / "src/models/__pycache__"
    cache.mkdir()
    (cache / "GlobalGCE.cpython-311.pyc").write_bytes(b"attacker")
    with pytest.raises(
        t8.TasteGlobalGCESmokeError,
        match="ignored runtime files",
    ):
        release._official_git_snapshot(root)


def test_official_git_authority_rejects_ignored_root_utils_pyc(
    tmp_path: Path,
) -> None:
    root = tmp_path / "official"
    _make_clean_official_git_checkout(root)
    cache = root / "src/__pycache__"
    cache.mkdir()
    (cache / "utils.cpython-311.pyc").write_bytes(b"attacker")
    with pytest.raises(
        t8.TasteGlobalGCESmokeError,
        match="ignored runtime files",
    ):
        release._official_git_snapshot(root)


def test_official_git_authority_rejects_untracked_runtime_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "official"
    _make_clean_official_git_checkout(root)
    (root / "src/models/attacker.py").write_text(
        "raise RuntimeError('attacker')\n",
        encoding="utf-8",
    )
    with pytest.raises(t8.TasteGlobalGCESmokeError, match="not clean"):
        release._official_git_snapshot(root)
