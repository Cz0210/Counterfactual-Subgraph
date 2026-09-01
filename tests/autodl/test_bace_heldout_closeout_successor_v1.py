from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.autodl import run_bace_heldout_closeout_successor_v1 as successor


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> successor.Config:
    project = tmp_path / "project"
    project.mkdir()
    python = tmp_path / "env/python"
    python.parent.mkdir()
    python.write_text("python\n", encoding="utf-8")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    source = tmp_path / "source"
    test_split = tmp_path / "test.csv"
    test_split.write_text("id,smiles,label\n1,CC,1\n", encoding="utf-8")
    gnn = tmp_path / "gine"
    gnn.mkdir()
    model = gnn / "model.pt"
    model.write_bytes(b"frozen-bace-gine")
    oracle_hash = _sha(model)
    _write_json(
        gnn / "model_card.json",
        {
            "dataset": "bace",
            "backbone": "gine",
            "oracle_backend": "gnn",
            "classifier_type": "gnn",
            "rf_oracle_used": False,
            "num_classes": 2,
            "source_label": 1,
            "checkpoint_id": oracle_hash,
        },
    )
    _write_json(
        gnn / "split_manifest.json",
        {
            "dataset": "bace",
            "files": {
                "test": {
                    "path": str(test_split.resolve()),
                    "sha256": _sha(test_split),
                }
            },
        },
    )
    molclr_root = tmp_path / "molclr"
    molclr_root.mkdir()
    molclr = molclr_root / "model.pth"
    molclr.write_bytes(b"molclr")
    molclr_hash = _sha(molclr)
    threshold_hash = "b" * 64
    methods: dict[str, dict] = {}
    source_controllers: dict[str, dict] = {}
    for method, slug in successor.METHODS:
        calibration = source / slug / "calibration-merged"
        calibration.mkdir(parents=True)
        (calibration / "pair_matrix.jsonl").write_text("{}\n", encoding="utf-8")
        _write_json(calibration / "run_manifest.json", {"status": "PASS"})
        root = source / slug / "selection-shared"
        root.mkdir(parents=True)
        (root / "PASS").write_text("PASS\n", encoding="utf-8")
        _write_json(root / "_RUN_COMPLETE.json", {"test_loaded": False})
        _write_json(
            root / "frozen_selection_manifest.json",
            {
                "dataset": "bace",
                "method": method,
                "method_id": slug,
                "stage": "BASELINE_CALIBRATION_SELECTOR",
                "status": "FROZEN",
                "selection_frozen": True,
                "selector_fitted_on_calibration": True,
                "calibration_loaded": True,
                "test_loaded": False,
                "oracle_backend": "gnn",
                "rf_oracle_used": False,
                "effective_rule_count": 20,
                "oracle_checkpoint_hash": oracle_hash,
                "molclr_checkpoint_hash": molclr_hash,
                "threshold_config_hash": threshold_hash,
            },
        )
        _write_json(root / "selected_top20.json", {"candidate_ids": list(range(20))})
        inventory = {
            name: {"size": path.stat().st_size, "sha256": _sha(path)}
            for name in (
                "PASS",
                "_RUN_COMPLETE.json",
                "frozen_selection_manifest.json",
                "selected_top20.json",
            )
            if (path := root / name)
        }
        methods[slug] = {
            "root": str(root.resolve()),
            "source_inventory": inventory,
            "oracle_checkpoint_hash": oracle_hash,
            "molclr_checkpoint_hash": molclr_hash,
            "threshold_config_hash": threshold_hash,
            "calibration_pair_matrix": {
                "path": str((calibration / "pair_matrix.jsonl").resolve()),
                "size": (calibration / "pair_matrix.jsonl").stat().st_size,
                "sha256": _sha(calibration / "pair_matrix.jsonl"),
            },
            "calibration_run_manifest": {
                "path": str((calibration / "run_manifest.json").resolve()),
                "size": (calibration / "run_manifest.json").stat().st_size,
                "sha256": _sha(calibration / "run_manifest.json"),
            },
            "frozen_selection_manifest": {
                "path": str((root / "frozen_selection_manifest.json").resolve()),
                "size": (root / "frozen_selection_manifest.json").stat().st_size,
                "sha256": _sha(root / "frozen_selection_manifest.json"),
            },
            "selected_top20": {
                "path": str((root / "selected_top20.json").resolve()),
                "size": (root / "selected_top20.json").stat().st_size,
                "sha256": _sha(root / "selected_top20.json"),
            },
        }
        controller = tmp_path / "controllers" / f"{slug}.json"
        _write_json(controller, {"controller_id": slug})
        source_controllers[slug] = {
            "path": str(controller.resolve()),
            "sha256": _sha(controller),
        }
    receipt = tmp_path / "selection_adoption_receipt.json"
    _write_json(
        receipt,
        {
            "schema_version": successor.RECEIPT_SCHEMA,
            "status": "PASS",
            "selection_frozen_before_test": True,
            "test_loaded": False,
            "source_root": str(source.resolve()),
            "methods": methods,
            "source_controller_manifests": source_controllers,
        },
    )
    authority = tmp_path / "authority/state.json"
    _write_json(authority, {"schema_version": "fixture"})
    return successor.Config(
        project_root=project.resolve(),
        python=python.resolve(),
        runtime_root=runtime.resolve(),
        controller_id="bace-heldout-test",
        control_dir=(tmp_path / "control/current").resolve(),
        output_root=(tmp_path / "output").resolve(),
        source_root=source.resolve(),
        selection_receipt=receipt.resolve(),
        expected_receipt_sha256=_sha(receipt),
        gnn_checkpoint=gnn.resolve(),
        test_split=test_split.resolve(),
        molclr_root=molclr_root.resolve(),
        molclr_checkpoint=molclr.resolve(),
        matrix_authority_state=authority.resolve(),
        matrix_authority_lock=(tmp_path / "authority/publish.lock").resolve(),
        gpu_index=0,
        min_free_memory_mb=16000,
        poll_seconds=0.01,
    )


def test_reopens_both_frozen_selections_without_science_replay(tmp_path: Path) -> None:
    config = _fixture(tmp_path)

    evidence = successor.validate_selection_adoption(config)

    assert evidence["status"] == "PASS"
    assert set(evidence["methods"]) == {"globalgce", "comrecgc"}
    assert evidence["test_loaded"] is False
    assert evidence["generation_replayed"] is False
    assert evidence["calibration_replayed"] is False
    assert evidence["gnn_ablation_started"] is False


def test_selection_inventory_change_fails_closed(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    selected = config.source_root / "globalgce/selection-shared/selected_top20.json"
    selected.write_text("{}\n", encoding="utf-8")

    with pytest.raises(successor.BaceHeldoutCloseoutError, match="identity changed"):
        successor.validate_selection_adoption(config)


def test_runtime_gine_bytes_must_match_frozen_selection(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    (config.gnn_checkpoint / "model.pt").write_bytes(b"different-gine")

    with pytest.raises(successor.BaceHeldoutCloseoutError, match="model.pt differs"):
        successor.validate_selection_adoption(config)


def test_runtime_test_split_must_match_gine_split_manifest(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    other = tmp_path / "other-test.csv"
    other.write_text("id,smiles,label\n2,CCC,0\n", encoding="utf-8")
    config = successor.Config(**{**config.__dict__, "test_split": other.resolve()})

    with pytest.raises(successor.BaceHeldoutCloseoutError, match="split path differs"):
        successor.validate_selection_adoption(config)


def test_runtime_test_split_bytes_must_match_gine_split_manifest(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    config.test_split.write_text("id,smiles,label\n1,CC,0\n", encoding="utf-8")

    with pytest.raises(successor.BaceHeldoutCloseoutError, match="split bytes differ"):
        successor.validate_selection_adoption(config)


def test_selection_test_leakage_fails_closed_even_when_receipt_sha_is_rebound(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    complete = config.source_root / "comrecgc/selection-shared/_RUN_COMPLETE.json"
    _write_json(complete, {"test_loaded": True})
    receipt = json.loads(config.selection_receipt.read_text(encoding="utf-8"))
    identity = receipt["methods"]["comrecgc"]["source_inventory"]["_RUN_COMPLETE.json"]
    identity.update({"size": complete.stat().st_size, "sha256": _sha(complete)})
    _write_json(config.selection_receipt, receipt)
    config = successor.Config(
        **{
            **config.__dict__,
            "expected_receipt_sha256": _sha(config.selection_receipt),
        }
    )

    with pytest.raises(successor.BaceHeldoutCloseoutError, match="contract changed"):
        successor.validate_selection_adoption(config)


def test_stage_plan_starts_at_heldout_and_never_contains_train_or_calibration(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)

    stages = successor.build_method_stages(config, "GlobalGCE", "globalgce")

    assert [stage.kind for stage in stages] == [
        "shard",
        "shard",
        "shard",
        "shard",
        "merge",
        "final",
        "standardized",
    ]
    assert [stage.shard_index for stage in stages[:4]] == [0, 1, 2, 3]
    assert all(stage.gpu for stage in stages[:4])
    assert not any(stage.gpu for stage in stages[4:])
    serialized = json.dumps([stage.command for stage in stages]).lower()
    assert "baseline_test_eval" in serialized
    assert str(config.test_split).lower() in serialized
    assert "train-rules" not in serialized
    assert "train_generation" not in serialized
    assert "baseline_calibration_verify" not in serialized
    assert "gspan" not in serialized
    assert "ablation" not in serialized


def _write_terminal(root: Path, *, kind: str, method: str, shard: int | None = None) -> None:
    root.mkdir(parents=True)
    for name in successor._required_files(kind, method):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("x\n", encoding="utf-8")
    (root / "PASS").write_text("PASS\n", encoding="utf-8")
    dataset = "BACE" if kind == "standardized" else "bace"
    manifest = {
        "dataset": dataset,
        "method": method,
        "status": "PASS",
        "rf_oracle_used": False,
    }
    if kind in {"shard", "merge"}:
        manifest.update(
            {
                "stage": successor.TEST_STAGE,
                "test_loaded": True,
                "selection_frozen_before_test": True,
                "run_complete": True,
            }
        )
        if kind == "shard":
            manifest["shard_index"] = shard
    elif kind == "final":
        manifest.update(
            {
                "stage": successor.FINAL_STAGE,
                "selection_frozen_before_test": True,
                "test_used_only_after_freeze": True,
                "run_complete": True,
            }
        )
        _write_json(root / "FINAL_PASS.json", manifest)
    else:
        _write_json(
            root / "_FINALIZED.json",
            {"status": "PASS", "raw_test_opened": False},
        )
        _write_json(
            root / "final_artifact_audit.json",
            {
                "passed": True,
                "final_artifact_audit_passed": True,
                "raw_test_opened": False,
            },
        )
    _write_json(root / "run_manifest.json", manifest)


def test_restart_adopts_exact_terminal_and_skips_incomplete_attempt(tmp_path: Path) -> None:
    base = tmp_path / "shards"
    (base / "attempt-0").mkdir(parents=True)
    (base / "attempt-0/PASS").write_text("partial\n", encoding="utf-8")
    _write_terminal(
        base / "attempt-1", kind="shard", method="GlobalGCE", shard=0
    )

    selected, adopted = successor.choose_attempt(
        base, kind="shard", method="GlobalGCE", shard_index=0
    )

    assert selected == base / "attempt-1"
    assert adopted is True


def test_standardized_validator_uses_real_audit_boolean_schema(tmp_path: Path) -> None:
    root = tmp_path / "standardized"
    _write_terminal(root, kind="standardized", method="ComRecGC")

    assert successor.terminal_valid(
        root, kind="standardized", method="ComRecGC"
    )
    audit = json.loads((root / "final_artifact_audit.json").read_text(encoding="utf-8"))
    audit["raw_test_opened"] = True
    _write_json(root / "final_artifact_audit.json", audit)
    assert not successor.terminal_valid(
        root, kind="standardized", method="ComRecGC"
    )


def test_launcher_is_persistent_single_successor_and_ablation_disabled() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root / "scripts/autodl/launch_bace_heldout_closeout_successor_v1.sh"
    ).read_text(encoding="utf-8")
    slurm = (
        root / "scripts/slurm/run_bace_heldout_closeout_successor_v1.sh"
    ).read_text(encoding="utf-8")

    assert "ALLOW_BACE_HELDOUT_CLOSEOUT_SUCCESSOR=1" in launcher
    assert "RUN_GNN_ABLATION must remain 0" in launcher
    assert "nohup setsid" in launcher
    assert "preflight" in launcher
    assert "pkill" not in launcher
    assert "killall" not in launcher
    assert "SIGKILL" not in launcher
    assert "#SBATCH --partition=A800" in slurm
    assert "#SBATCH --gres=gpu:a800:1" in slurm
    assert "#SBATCH --output=logs/%j.out" in slurm
    assert "#SBATCH --error=logs/%j.err" in slurm
    assert "--config configs/hpc.yaml" in slurm
    assert "--set inference.fallback_to_heuristic=false" in slurm
