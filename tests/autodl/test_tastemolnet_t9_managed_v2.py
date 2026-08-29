from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.baselines.comrecgc.held_upstream import OFFICIAL_SOURCE_SHA256
from src.baselines.tastemolnet_comrecgc_smoke import PASS_MARKER
from src.utils.managed_execution_v2 import (
    create_managed_attempt,
    create_worker_staging,
    load_verified_gate,
)
from src.utils.tastemolnet_t9_managed_v2 import (
    INPUT_AUTHORITY_FILE,
    SCIENCE_FILE,
    T9_INPUT_AUTHORITY_SCHEMA,
    TRUST_MODEL,
    open_t9_sealed,
    seal_t9_worker_evidence,
    t9_managed_input_hashes,
    verify_and_publish_t9_sealed,
)
from tests.baselines.test_tastemolnet_comrecgc_smoke import _valid_result


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _authority(tmp_path: Path) -> dict[str, object]:
    t2_files = {
        name: f"{index + 1:x}" * 64
        for index, name in enumerate(
            (
                "PASS",
                "artifact_hashes.json",
                "gate.json",
                "input_hashes.json",
                "sha256s.txt",
                "source_evidence.json",
                "verification.json",
            )
        )
    }
    t2 = {
        "schema_version": "tastemolnet_t9_t2_receipt_binding_v1",
        "status": "PASS",
        "root": str(tmp_path / "t2-receipt"),
        "receipt_id": "t2-receipt",
        "gate_sha256": t2_files["gate.json"],
        "source_evidence_file_sha256": t2_files["source_evidence.json"],
        "source_evidence_sha256": "1" * 64,
        "receipt_inventory_sha256": "2" * 64,
        "file_sha256": t2_files,
    }
    t3 = {
        "schema_version": "tastemolnet_t4_t3_binding_v2",
        "t3_root": str(tmp_path / "t3"),
        "t3_gate_sha256": "3" * 64,
        "t3_verification_sha256": "4" * 64,
        "checkpoint_dir": str(tmp_path / "t3" / "artifacts" / "checkpoint"),
        "checkpoint_id": "5" * 64,
        "checkpoint_inventory_sha256": "6" * 64,
        "checkpoint_sha256s_sha256": "7" * 64,
        "temperature_scaling_sha256": "8" * 64,
        "feature_schema_sha256": "9" * 64,
        "source_t2_receipt_id": t2["receipt_id"],
        "source_t2_gate_sha256": t2["gate_sha256"],
        "source_t2_evidence_sha256": t2["source_evidence_sha256"],
        "calibration_payload_loaded": False,
        "test_payload_loaded": False,
        "rf_oracle_used": False,
    }
    t4_science = {
        "t3_root": t3["t3_root"],
        "t3_gate_sha256": t3["t3_gate_sha256"],
        "t3_verification_sha256": t3["t3_verification_sha256"],
        "checkpoint_dir": t3["checkpoint_dir"],
        "checkpoint_id": t3["checkpoint_id"],
        "temperature_scaling_sha256": t3["temperature_scaling_sha256"],
        "feature_schema_sha256": t3["feature_schema_sha256"],
    }
    t4 = {
        "schema_version": "tastemolnet_t9_t4_managed_binding_v1",
        "status": "PASS",
        "root": str(tmp_path / "t4"),
        "science": t4_science,
    }
    checkpoint = {
        "checkpoint_dir": t3["checkpoint_dir"],
        "checkpoint_id": t3["checkpoint_id"],
        "checkpoint_inventory_sha256": t3["checkpoint_inventory_sha256"],
        "checkpoint_sha256s_sha256": t3["checkpoint_sha256s_sha256"],
        "payload_sha256": {
            name: str(index + 1) * 64
            for index, name in enumerate(
                (
                    "model.pt",
                    "config.yaml",
                    "model_card.json",
                    "feature_schema.json",
                    "label_map.json",
                    "split_manifest.json",
                    "test_evaluation_status.json",
                    "temperature_scaling.json",
                )
            )
        },
    }
    return {
        "schema_version": T9_INPUT_AUTHORITY_SCHEMA,
        "trust_model": TRUST_MODEL,
        "operator": {
            "run_id": "taste-t9-test",
            "task_id": "tastemolnet_t9_comrecgc_smoke",
        },
        "execution": {"commit": "a" * 40, "tree": "b" * 40},
        "config": {
            "path": str(PROJECT_ROOT / "configs/hpc.yaml"),
            "sha256": "c" * 64,
        },
        "gpu": {
            "physical_index": 1,
            "uuid": "GPU-test-1234",
            "logical_device": "cuda:0",
        },
        "t2_adoption_binding": t2,
        "t3_stage_evidence": t3,
        "t4_stage_evidence": t4,
        "checkpoint": checkpoint,
        "train": {
            "path": str(tmp_path / "train.csv"),
            "sha256": "d" * 64,
            "num_records": 128,
            "label_counts": {"0": 32, "1": 64, "2": 32},
            "graph_schema_sha256": "e" * 64,
            "sweet_source_pool_count": 64,
        },
        "official": {
            "schema_version": "comrecgc_held_official_sources_v1",
            "commit": "122f9341a360e9f06bb58a2f5823bb596021f6bf",
            "root": str(tmp_path / "official"),
            "root_identity": {
                "device": 1,
                "inode": 2,
                "mode": 16832,
                "uid": 501,
                "gid": 20,
            },
            "file_sha256": dict(OFFICIAL_SOURCE_SHA256),
            "module_names": [
                "util",
                "data",
                "neurosed.models",
                "distance",
                "gnn",
                "comrecgc",
                "common_recourse",
            ],
            "descriptor_loaded": True,
        },
        "data_access": {
            "train_loaded": True,
            "validation_loaded": False,
            "calibration_payload_loaded": False,
            "test_loaded": False,
            "rf_oracle_used": False,
            "dataset_redistributed": False,
        },
    }


def test_t9_worker_seals_and_only_independent_verifier_publishes(
    tmp_path: Path,
) -> None:
    stage_root = tmp_path / "stage"
    final_parent = tmp_path / "final-parent"
    stage_root.mkdir()
    final_parent.mkdir()
    final = final_parent / "t9-final"
    authority = _authority(tmp_path)
    science = _valid_result()
    with create_managed_attempt(
        stage_root=stage_root,
        controller_id="taste-t9-test",
        task_id="tastemolnet_t9_comrecgc_smoke",
        git_commit="a" * 40,
        config_hash="c" * 64,
        input_hashes=t9_managed_input_hashes(authority),
        boot_id="test-boot",
    ) as attempt, create_worker_staging(attempt) as staging:
        sealed = seal_t9_worker_evidence(
            staging,
            science=science,
            input_authority=authority,
            expected_final_path=final,
        )
        assert not (staging.path / "PASS").exists()
        assert not (staging.path / "gate.json").exists()
        assert (staging.path / SCIENCE_FILE).is_file()
        assert (staging.path / INPUT_AUTHORITY_FILE).is_file()

    calls = 0

    def revalidate() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return authority

    with open_t9_sealed(
        sealed.staging_path,
        expected_attempt_id=sealed.attempt_id,
        expected_generation_token=sealed.generation_token,
    ) as held:
        publication, verification = verify_and_publish_t9_sealed(
            held,
            final_path=final,
            expected_authority=authority,
            revalidate_inputs=revalidate,
        )
    assert publication.final_path == final
    assert calls == 2
    assert verification["domain_marker"] == PASS_MARKER
    assert verification["random_walk_steps"] == 500
    assert verification["checkpoint_step"] == 250
    assert verification["test_loaded"] is False
    assert load_verified_gate(final)["status"] == "PASS"
    nested = json.loads((final / "verification.json").read_text())["verification"]
    assert nested == verification


def test_t9_authority_rejects_gpu_or_test_drift(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    authority["gpu"]["physical_index"] = 2
    with pytest.raises(Exception, match="authority values changed"):
        t9_managed_input_hashes(authority)

    authority = _authority(tmp_path)
    authority["data_access"]["test_loaded"] = True
    with pytest.raises(Exception, match="authority values changed"):
        t9_managed_input_hashes(authority)


def test_t9_runner_and_verifier_have_static_slurm_pairs() -> None:
    for name in (
        "tastemolnet_t9_managed_runner_v2",
        "tastemolnet_t9_comrecgc_verifier_v2",
    ):
        python = PROJECT_ROOT / "scripts/autodl" / f"{name}.py"
        slurm = PROJECT_ROOT / "scripts/slurm" / f"{name}.sh"
        assert python.is_file()
        text = slurm.read_text(encoding="utf-8")
        for token in (
            "#SBATCH --partition=A800",
            "#SBATCH --gres=gpu:a800:1",
            "#SBATCH --output=logs/%j.out",
            "#SBATCH --error=logs/%j.err",
            "source ~/.bashrc",
            "conda activate smiles_pip118",
            "cd /share/home/u20526/czx/counterfactual-subgraph",
            "export PYTHONPATH=$PWD",
            "--config configs/hpc.yaml",
            "--set inference.fallback_to_heuristic=false",
            "exit 64",
        ):
            assert token in text
