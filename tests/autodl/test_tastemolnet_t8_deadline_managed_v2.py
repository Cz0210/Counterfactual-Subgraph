from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from src.baselines.tastemolnet_globalgce_full import validate_t8_pass
from src.baselines.tastemolnet_globalgce_smoke import (
    PASS_MARKER,
    TasteGlobalGCESmokeConfig,
    TasteGlobalGCESmokeError,
)
from src.utils.retained_output_directory import (
    FreshOutputDirectory,
    RetainedOutputTree,
    prepare_terminal_output,
)
from src.utils import tastemolnet_t8_deadline_managed_v2 as adapter
from src.utils.terminal_publisher_v2 import open_sealed_worker_artifact


DEADLINE_ATTEMPT = "70db9c1b-3f28-4ae9-bc21-d81027b2e53d"
RECOVERY_SOURCE = "4376be2b-42de-46d4-a3c6-ad291dd3f9f0"
MANAGED_ATTEMPT = "7567c307-3d6b-4aa1-a5a6-0eb2d9ec1ee3"


def _json_bytes(value: dict) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _make_path_set(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name in ("t3", "t4", "checkpoint", "official"):
        path = root / name
        path.mkdir()
        result[name] = path
    for name in ("config", "train"):
        path = root / f"{name}.data"
        path.write_bytes(name.encode())
        result[name] = path
    return result


def _preflight(*, epochs: int = 25) -> dict:
    recovery = {
        "schema_version": adapter.deadline.RECOVERY_SCHEMA,
        "enabled": True,
        "source_attempt_id": RECOVERY_SOURCE,
        "stop_reason": (
            "prior_native_generation_had_zero_valid_connected_candidates"
        ),
        "epochs": epochs,
    }
    return {
        "schema_version": adapter.deadline.SCHEMA,
        "status": "READY",
        "attempt_id": DEADLINE_ATTEMPT,
        "zero_candidate_recovery": recovery,
        "science_config": TasteGlobalGCESmokeConfig(epochs=epochs).to_dict(),
        "checkpoint_id": "a" * 64,
        "official_runtime_source_inventory_sha256": "b" * 64,
        "train_sha256": "c" * 64,
        "train_rows": 18,
        "t3_verification_sha256": "d" * 64,
        "t4_verification_sha256": "e" * 64,
        "calibration_loaded": False,
        "test_loaded": False,
        "rf_oracle_used": False,
        "gnn_ablation_started": False,
    }


def _publish_deadline(root: Path, *, preflight: dict, science: dict) -> None:
    science_bytes = _json_bytes(science)
    manifest, gate = adapter._expected_terminal_documents(
        preflight=preflight,
        science=science,
        science_document_sha256=hashlib.sha256(science_bytes).hexdigest(),
    )
    output = FreshOutputDirectory.create(root)
    output.write_new("science.json", science_bytes)
    output.write_new("manifest.json", _json_bytes(manifest))
    output.write_new("gate.json", _json_bytes(gate))
    prepared = prepare_terminal_output(
        output,
        marker_name="PASS",
        marker_payload=(PASS_MARKER + "\n").encode(),
    )
    # The production commit uses linkat on Linux.  Tests make the already-held
    # prepared inode visible under PASS without weakening the byte contract.
    os.rename(
        ".PASS.prepared",
        "PASS",
        src_dir_fd=output.descriptor,
        dst_dir_fd=output.descriptor,
    )
    output.committed = True
    prepared.close()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, epochs: int = 25):
    paths = _make_path_set(tmp_path)
    state_path = tmp_path / "deadline-state"
    state = FreshOutputDirectory.create(state_path)
    state.write_new("proof.bin", b"official-startup-evidence")
    tree = RetainedOutputTree.capture(state.descriptor)
    inventory = dict(tree.inventory)
    tree.close()
    state.close()

    preflight = _preflight(epochs=epochs)
    science = {
        "config": TasteGlobalGCESmokeConfig(epochs=epochs).to_dict(),
        "oracle_checkpoint_hash": "a" * 64,
        "target_branches": [0, 2],
        "rf_oracle_used": False,
        "gnn_ablation_started": False,
        "train_boundary": {
            "external_validation_loaded": False,
            "calibration_loaded": False,
            "test_loaded": False,
        },
        "private_state": {
            "inventory_sha256": inventory["inventory_sha256"],
            "file_count": len(inventory["files"]),
        },
        "strict_flip_validation": {
            "strict_flip_count": 2,
            "destination_distribution": {"0": 1, "2": 1},
        },
    }
    output_path = tmp_path / "deadline-output"
    _publish_deadline(output_path, preflight=preflight, science=science)

    monkeypatch.setattr(adapter, "validate_science_summary", lambda value: None)
    monkeypatch.setattr(
        adapter,
        "_derive_deadline_preflight",
        lambda _inputs: json.loads(json.dumps(preflight)),
    )
    monkeypatch.setattr(
        adapter,
        "collect_t8_official_startup_evidence",
        lambda **_kwargs: {
            "official_globalgce_commit": adapter.OFFICIAL_GLOBALGCE_COMMIT,
            "branches": {"0": {}, "2": {}},
        },
    )
    inputs = adapter.DeadlineRecoveryInputs(
        config=paths["config"],
        deadline_output_root=output_path,
        deadline_state_root=state_path,
        deadline_attempt_id=DEADLINE_ATTEMPT,
        recovery_source_attempt_id=RECOVERY_SOURCE,
        t3_output=paths["t3"],
        t4_output=paths["t4"],
        gnn_checkpoint=paths["checkpoint"],
        train_csv=paths["train"],
        official_root=paths["official"],
    )
    return inputs, output_path


def test_deadline_adoption_is_t13_consumable_and_worker_is_raw_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTO_TERMINATE_UNCONTROLLED_CHILDREN", "0")
    inputs, _source = _fixture(tmp_path, monkeypatch)
    stage = tmp_path / "managed-stage"
    stage.mkdir()
    final = tmp_path / "final" / "t8"
    final.parent.mkdir()
    sealed = adapter.create_deadline_adoption_sealed(
        inputs=inputs,
        stage_root=stage,
        final_path=final,
        managed_attempt_id=MANAGED_ATTEMPT,
        run_id="t8-deadline-adoption-test",
        execution_commit="f" * 40,
    )
    staging = Path(sealed["staging_path"])
    assert (staging / "SEALED.json").is_file()
    assert not (staging / "PASS").exists()
    assert not (staging / "gate.json").exists()
    assert not (staging / "verification.json").exists()

    with open_sealed_worker_artifact(
        staging,
        expected_attempt_id=MANAGED_ATTEMPT,
        expected_generation_token=sealed["generation_token"],
    ) as held:
        publication, typed = adapter.verify_and_publish_deadline_adoption(
            held,
            inputs=inputs,
            final_path=final,
            run_id="t8-deadline-adoption-test",
            execution_commit="f" * 40,
        )
    assert publication.final_path == final
    assert typed["schema_version"] == adapter.T8_VERIFICATION_SCHEMA
    assert typed["zero_candidate_recovery_epochs"] == 25
    assert typed["test_loaded"] is False
    assert typed["calibration_loaded"] is False
    reopened, adoption = validate_t8_pass(final)
    assert reopened == final
    assert adoption["typed_verification"]["deadline_attempt_id"] == (
        DEADLINE_ATTEMPT
    )


def test_verifier_reopens_and_rejects_changed_deadline_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTO_TERMINATE_UNCONTROLLED_CHILDREN", "0")
    inputs, source = _fixture(tmp_path, monkeypatch)
    stage = tmp_path / "managed-stage"
    stage.mkdir()
    final = tmp_path / "final" / "t8"
    final.parent.mkdir()
    sealed = adapter.create_deadline_adoption_sealed(
        inputs=inputs,
        stage_root=stage,
        final_path=final,
        managed_attempt_id=MANAGED_ATTEMPT,
        run_id="t8-deadline-adoption-test",
        execution_commit="f" * 40,
    )
    science_path = source / "science.json"
    science_path.write_bytes(science_path.read_bytes() + b" ")
    with open_sealed_worker_artifact(
        sealed["staging_path"],
        expected_attempt_id=MANAGED_ATTEMPT,
        expected_generation_token=sealed["generation_token"],
    ) as held:
        with pytest.raises(Exception):
            adapter.verify_and_publish_deadline_adoption(
                held,
                inputs=inputs,
                final_path=final,
                run_id="t8-deadline-adoption-test",
                execution_commit="f" * 40,
            )
    assert not final.exists()


def test_five_epoch_deadline_cannot_be_adopted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, _source = _fixture(tmp_path, monkeypatch, epochs=5)
    with pytest.raises(
        TasteGlobalGCESmokeError,
        match="fixed 25-epoch fresh recovery",
    ):
        with adapter.HeldDeadlineRecovery.open(inputs):
            pass


def test_cli_and_slurm_preserve_two_process_and_no_ablation_contract() -> None:
    project = Path(__file__).resolve().parents[2]
    cli = (
        project / "scripts/autodl/adopt_tastemolnet_t8_deadline_v2.py"
    ).read_text(encoding="utf-8")
    for required in (
        'choices=("run", "worker", "verifier", "validate")',
        '[sys.executable, "-I", "-B"',
        '"SEALED_PENDING_INDEPENDENT_VERIFICATION"',
        "validate_t8_pass(args.final_path)",
    ):
        assert required in cli
    slurm = (
        project / "scripts/slurm/adopt_tastemolnet_t8_deadline_v2.sh"
    ).read_text(encoding="utf-8")
    for required in (
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
        '[[ "${RUN_GNN_ABLATION:-0}" == "0" ]]',
    ):
        assert required in slurm
