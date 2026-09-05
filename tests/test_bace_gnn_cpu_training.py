"""Tiny real CPU tests for scientific epoch continuation and bundle isolation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ablations.gnn.cpu_training import (
    bundle_file, classify_cpu_admission, effective_training_config, file_sha256, load_bundle,
    run_cpu_auto, run_cpu_training, trainer_arguments,
)


def _bundle(tmp_path: Path) -> Path:
    from src.data.molecular_graph_featurizer import default_molecular_feature_schema
    root = tmp_path / "bundle"
    root.mkdir()
    config = {
        "gnn": {"backbone": "gine", "hidden_dim": 8, "num_layers": 1,
                "dropout": 0.2, "normalization": "none", "readout_layers": 1},
        "training": {"optimizer": "adamw", "learning_rate": 0.001,
                     "weight_decay": 0.00001, "max_epochs": 3,
                     "early_stopping_patience": 20, "batch_size": 4,
                     "primary_seed": 7, "selection_metric": "macro_f1",
                     "class_weighted_loss": True, "weighted_sampler": False,
                     "gradient_clip_norm": 5.0,
                     "health_gate": {"enabled": True, "minimum_primary_metric": 0.65}},
        "calibration": {"fit_on_validation": True, "split": "validation", "max_iter": 3},
    }
    content = {
        "reference/gine/config.yaml": json.dumps(config),
        "configs/gin.yaml": json.dumps({"gnn": {**config["gnn"], "backbone": "gin"}}),
        "reference/gine/feature_schema.json": json.dumps(default_molecular_feature_schema().to_dict()),
        "data/train.csv": "id,smiles,label,split\na,CC,0,train\nb,CCC,1,train\nc,CCO,0,train\nd,CCN,1,train\n",
        "data/validation.csv": "id,smiles,label,split\ne,CCCC,0,val\nf,CCCO,1,val\n",
        # Malformed payloads deliberately prove they are hash-bound but not parsed.
        "data/calibration.csv": "not a parseable molecular dataset\n",
        "data/test.csv": "do not parse this heldout payload\n",
    }
    for name in ("model.pt", "model_card.json", "label_map.json", "split_manifest.json", "training_metrics.json",
                 "validation_predictions.csv", "test_evaluation_status.json", "temperature_scaling.json", "environment.json", "git_state.json"):
        content[f"reference/gine/{name}"] = "tiny hash-bound reference fixture\n"
    for relative, data in content.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
    checksums = "".join(f"{file_sha256(root / relative)}  {Path(relative).name}\n" for relative in sorted(content) if relative.startswith("reference/gine/"))
    (root / "reference/gine/sha256sums.txt").write_text(checksums)
    content["reference/gine/sha256sums.txt"] = checksums
    manifest = {
        "schema_version": "bace_gnn_cpu_bundle_v1", "dataset": "bace",
        "seed": 7, "num_classes": 2,
        "splits": {name: f"data/{name}.csv" for name in ("train", "validation", "calibration", "test")},
        "feature_schema_path": "reference/gine/feature_schema.json",
        "gine_reference_root": "reference/gine",
        "training_config_path": "reference/gine/config.yaml",
        "backbone_configs": {"gin": "configs/gin.yaml"},
        "files": {relative: {"sha256": file_sha256(root / relative), "size": (root / relative).stat().st_size} for relative in content},
    }
    from src.ablations.contracts import canonical_json_sha256
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    (root / "bundle_manifest.json").write_text(json.dumps(manifest))
    return root


def test_cpu_bundle_rejects_tampered_input_and_traversal(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    _, manifest = load_bundle(root)
    with pytest.raises(ValueError, match="relative"):
        bundle_file(root, manifest, "../data.csv")
    (root / "data/train.csv").write_text("tampered")
    with pytest.raises(ValueError, match="mismatch"):
        bundle_file(root, manifest, "data/train.csv")


def test_cpu_config_preserves_reference_optimizer_without_quality_censor(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    _, manifest = load_bundle(root)
    config = effective_training_config(root, manifest, "gin")
    assert config["training"]["max_epochs"] == 3
    assert config["training"]["health_gate"] == {"enabled": False}
    assert config["training"]["selection_metric"] == "macro_f1"
    with pytest.raises(ValueError, match="GINE is adopted"):
        effective_training_config(root, manifest, "gine")
    args = trainer_arguments(root=root, manifest=manifest, backbone="gin",
                             output_root=tmp_path / "run", effective_config_path=tmp_path / "effective.json", resume=True)
    assert args[args.index("--device") + 1] == "cpu"
    assert "--resume-training" in args
    assert "--max-epochs" not in args


@pytest.mark.parametrize("field,value", [("dropout", 0.5), ("pooling", "sum"), ("hidden_dim", 16), ("num_layers", 2)])
def test_backbone_cannot_change_matched_reference_fields(tmp_path: Path, field: str, value: object) -> None:
    root = _bundle(tmp_path)
    _, manifest = load_bundle(root)
    architecture = root / "configs/gin.yaml"
    payload = json.loads(architecture.read_text())
    payload["gnn"][field] = value
    architecture.write_text(json.dumps(payload))
    manifest["files"]["configs/gin.yaml"] = {"sha256": file_sha256(architecture), "size": architecture.stat().st_size}
    with pytest.raises(ValueError, match="(matched reference|hidden dimension)"):
        effective_training_config(root, manifest, "gin")


def test_gine_original_inventory_catches_repackaged_changed_weights(tmp_path: Path) -> None:
    from src.ablations.contracts import canonical_json_sha256
    root = _bundle(tmp_path)
    _, manifest = load_bundle(root)
    weights = root / "reference/gine/model.pt"
    weights.write_text("new mismatched weights")
    manifest["files"]["reference/gine/model.pt"] = {"sha256": file_sha256(weights), "size": weights.stat().st_size}
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    (root / "bundle_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Source GINE SHA inventory mismatch"):
        load_bundle(root)


def test_cpu_benchmark_gate_uses_runtime_and_unknown_eval_never_admits_full() -> None:
    assert classify_cpu_admission(100, 100) == "CPU_FULL_ELIGIBLE"
    assert classify_cpu_admission(100, None) == "CPU_TRAIN_ONLY_ELIGIBLE"
    assert classify_cpu_admission(100, 13 * 3600) == "CPU_TRAIN_ONLY_ELIGIBLE"
    assert classify_cpu_admission(13 * 3600, 1) == "GPU_FALLBACK_REQUIRED"
    assert classify_cpu_admission(None, 1) == "GPU_FALLBACK_REQUIRED"


def test_auto_resumes_the_same_benchmark_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ablations.gnn import cpu_training
    calls = []
    def fake_run(**values):
        calls.append(values)
        Path(values["output_root"]).mkdir(exist_ok=True)
        return {"cpu_admission": "CPU_TRAIN_ONLY_ELIGIBLE", "status": "PAUSED_AT_CHECKPOINT" if values["phase"] == "benchmark" else "TRAINING_PASS"}
    monkeypatch.setattr(cpu_training, "run_cpu_training", fake_run)
    result = run_cpu_auto(output_root=tmp_path / "attempt", resume=False)
    assert result["status"] == "TRAINING_PASS"
    assert [item["phase"] for item in calls] == ["benchmark", "train"]
    assert calls[0]["output_root"] == calls[1]["output_root"]
    assert calls[1]["resume"] is True


def test_cpu_benchmark_resume_matches_uninterrupted_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("rdkit")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    root = _bundle(tmp_path)
    interrupted = tmp_path / "resumed"
    common = dict(bundle_root=root, backbone="gin", config_path=root / "configs/gin.yaml", cpu_threads=1)
    first = run_cpu_training(**common, phase="benchmark", output_root=interrupted, benchmark_epochs=1)
    assert first["status"] == "PAUSED_AT_CHECKPOINT"
    assert first["completed_epoch"] == 1
    assert not (interrupted / "classifier").exists()
    resumed = run_cpu_training(**common, phase="train", output_root=interrupted, resume=True)
    continuous = tmp_path / "continuous"
    run_cpu_training(**common, phase="train", output_root=continuous)
    assert resumed["status"] == "TRAINING_PASS"
    assert resumed["test_split_loaded"] is False
    assert resumed["calibration_split_loaded"] is False
    assert resumed["scheduler"] == {"kind": "constant", "state": {}}
    snapshots = []
    for output in (interrupted, continuous):
        latest = json.loads((output / "training_state/latest_checkpoint.json").read_text())
        snapshots.append(torch.load(output / "training_state" / latest["checkpoint_file"], weights_only=False))
    left, right = snapshots
    for name, tensor in left["model_state"].items():
        assert torch.equal(tensor, right["model_state"][name]), name
    assert torch.equal(left["torch_cpu_rng_state"], right["torch_cpu_rng_state"])
    assert left["python_rng_state"] == right["python_rng_state"]
    assert left["history"] == right["history"]
    assert left["optimizer_state"]["param_groups"] == right["optimizer_state"]["param_groups"]
    for key, state in left["optimizer_state"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, right["optimizer_state"]["state"][key][name])
