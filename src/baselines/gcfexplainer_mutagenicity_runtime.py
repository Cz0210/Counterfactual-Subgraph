"""Runtime orchestration for strict Mutagenicity official GCFExplainer runs."""

from __future__ import annotations

import math
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.baselines.gcfexplainer_mutagenicity_adapter import (
    EXPECTED_GENERATION_SOURCE_ROWS,
    EXPECTED_MODEL_TRAIN_ROWS,
    EXPECTED_MODEL_VAL_ROWS,
    GCFExplainerEmptyCandidateSetError,
    GCFExplainerMutagenicityError,
    GraphRecordDataset,
    RunProfile,
    SEED,
    build_native_rank_candidate_universe,
    checkpoint_is_aids,
    cohort_hash,
    deterministic_balanced_prefix,
    graph_lineage_neighbor_wrapper,
    import_official_modules,
    load_dataset_artifacts,
    read_json,
    record_to_pyg,
    sha256_file,
    stable_graph_candidate_id,
    stable_json_sha256,
    validate_gnn_profile,
    validate_vrrw_profile,
    write_csv,
    write_json,
    write_jsonl,
)


OFFICIAL_VRRW_THETA = 0.05
OFFICIAL_SUMMARY_THETA = 0.1
OFFICIAL_TELEPORT = 0.1


def _runtime_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn.functional as functional
        try:
            from torch_geometric.loader import DataLoader
        except ImportError:  # PyG 1.x compatibility on the official environment
            from torch_geometric.data import DataLoader

        return torch, functional, DataLoader
    except Exception as exc:  # pragma: no cover - HPC dependency
        raise RuntimeError(
            "Official GCFExplainer runtime requires torch and torch_geometric."
        ) from exc


def _prepare_output(
    output_dir: Path,
    *,
    fingerprint: str,
    resume: bool,
    allow_progress: bool,
) -> None:
    if (output_dir / "_FINALIZED.json").exists():
        raise FileExistsError(f"Finalized output cannot be reused: {output_dir}")
    if (output_dir / "_RUN_COMPLETE.json").exists():
        raise FileExistsError(f"Completed output cannot be overwritten: {output_dir}")
    existing = output_dir.exists() and any(output_dir.iterdir())
    if existing and not resume:
        raise FileExistsError(f"Non-empty output with resume disabled: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "resolved_config.json"
    if existing:
        if not allow_progress or not config_path.is_file():
            raise ValueError(f"Output is not a resumable run: {output_dir}")
        previous = read_json(config_path)
        if previous.get("config_fingerprint") != fingerprint:
            raise ValueError("Resume configuration fingerprint mismatch.")


def _device_name(torch: Any, value: str) -> str:
    if value.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return value


def _load_official_gnn(
    modules: Mapping[str, Any],
    *,
    num_features: int,
    checkpoint: str | Path,
    device: str,
    num_layers: int = 3,
    dim: int = 20,
    dropout: float = 0.0,
) -> Any:
    torch, _functional, _loader = _runtime_stack()
    model = modules["gnn"].GNN(
        num_features=int(num_features),
        num_classes=2,
        num_layers=int(num_layers),
        dim=int(dim),
        dropout=float(dropout),
    ).to(device)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _evaluate_gnn(model: Any, loader: Any, device: str) -> dict[str, Any]:
    torch, functional, _loader = _runtime_stack()
    model.eval()
    total_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)[-1]
            loss = functional.nll_loss(logits, batch.y.long())
            predicted = torch.argmax(logits, dim=-1)
            total_loss += float(loss.item()) * int(batch.num_graphs)
            labels.extend(int(value) for value in batch.y.detach().cpu().tolist())
            predictions.extend(
                int(value) for value in predicted.detach().cpu().tolist()
            )
    correct = sum(
        int(expected == actual)
        for expected, actual in zip(labels, predictions, strict=True)
    )
    return {
        "loss": total_loss / len(labels) if labels else None,
        "accuracy": correct / len(labels) if labels else None,
        "rows": len(labels),
        "label_counts": dict(Counter(labels)),
        "prediction_counts": dict(Counter(predictions)),
    }


def train_official_gnn(
    *,
    dataset_dir: str | Path,
    official_root: str | Path,
    output_dir: str | Path,
    profile: str | RunProfile,
    epochs: int,
    train_limit: int,
    val_limit: int,
    batch_size: int,
    learning_rate: float,
    dropout: float,
    seed: int = SEED,
    device: str = "cuda:0",
    resume: bool = True,
) -> dict[str, Any]:
    schema, train_records, val_records, _generation, _dataset_summary = (
        load_dataset_artifacts(dataset_dir)
    )
    train_selected = deterministic_balanced_prefix(train_records, int(train_limit))
    val_selected = deterministic_balanced_prefix(val_records, int(val_limit))
    resolved_profile = validate_gnn_profile(
        profile,
        epochs=int(epochs),
        train_rows=len(train_selected),
        val_rows=len(val_selected),
    )
    if int(seed) != SEED:
        raise ValueError("Mutagenicity GNN requires seed=13.")
    root = Path(output_dir).expanduser().resolve()
    official = Path(official_root).expanduser().resolve()
    config = {
        "dataset": "Mutagenicity",
        "dataset_name": "mutagenicity",
        "profile": resolved_profile.value,
        "dataset_dir": str(Path(dataset_dir).expanduser().resolve()),
        "official_root": str(official),
        "epochs": int(epochs),
        "train_rows": len(train_selected),
        "val_rows": len(val_selected),
        "train_ids_hash": cohort_hash(train_selected),
        "val_ids_hash": cohort_hash(val_selected),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "dropout": float(dropout),
        "seed": int(seed),
        "device": str(device),
        "model_architecture": {
            "class": "official_gnn.GNN",
            "num_layers": 3,
            "dim": 20,
            "dropout": float(dropout),
            "num_features": schema.node_feature_dim,
            "num_classes": 2,
        },
        "checkpoint_selection": "minimum_validation_nll",
        "project_to_official_label_mapping": {"1": 0, "0": 1},
        "calibration_loaded": False,
        "test_loaded": False,
    }
    fingerprint = stable_json_sha256(config)
    config["config_fingerprint"] = fingerprint
    _prepare_output(
        root, fingerprint=fingerprint, resume=resume, allow_progress=True
    )
    for name in ("_RUN_FAILED.json", "failure_summary.json"):
        stale_failure = root / name
        if stale_failure.exists():
            stale_failure.unlink()
    write_json(root / "resolved_config.json", config)
    torch, functional, DataLoader = _runtime_stack()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    target_device = _device_name(torch, device)
    modules = import_official_modules(official)
    model = modules["gnn"].GNN(
        num_features=schema.node_feature_dim,
        num_classes=2,
        num_layers=3,
        dim=20,
        dropout=float(dropout),
    ).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    train_graphs = [record_to_pyg(row) for row in train_selected]
    val_graphs = [record_to_pyg(row) for row in val_selected]
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_graphs,
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_graphs,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
    )
    progress_path = root / "training_progress.json"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_val = math.inf
    best_epoch = 0
    state_path = root / "training_state.pt"
    checkpoint = root / "model_best.pth"
    if progress_path.is_file() and state_path.is_file() and resume:
        progress = read_json(progress_path)
        if progress.get("config_fingerprint") != fingerprint:
            raise ValueError("GNN resume fingerprint mismatch.")
        try:
            state = torch.load(state_path, map_location=target_device, weights_only=False)
        except TypeError:
            state = torch.load(state_path, map_location=target_device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_val = float(state["best_val"])
        best_epoch = int(state["best_epoch"])
        history = list(state.get("history", []))
        if state.get("data_loader_generator_state") is not None:
            generator.set_state(state["data_loader_generator_state"])
    for epoch in range(start_epoch, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch in train_loader:
            batch = batch.to(target_device)
            optimizer.zero_grad()
            logits = model(batch)[-1]
            loss = functional.nll_loss(logits, batch.y.long())
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(batch.num_graphs)
            total_rows += int(batch.num_graphs)
        val_metrics = _evaluate_gnn(model, val_loader, target_device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_rows,
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
        }
        history.append(row)
        if float(val_metrics["loss"]) < best_val:
            best_val = float(val_metrics["loss"])
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "data_loader_generator_state": generator.get_state(),
                "epoch": epoch,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "history": history,
            },
            state_path,
        )
        write_json(
            progress_path,
            {
                "config_fingerprint": fingerprint,
                "current_epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "run_complete": False,
            },
        )
    if not checkpoint.is_file():
        raise GCFExplainerMutagenicityError("GNN did not produce a checkpoint.")
    model = _load_official_gnn(
        modules,
        num_features=schema.node_feature_dim,
        checkpoint=checkpoint,
        device=target_device,
        dropout=float(dropout),
    )
    train_metrics = _evaluate_gnn(model, train_loader, target_device)
    val_metrics = _evaluate_gnn(model, val_loader, target_device)
    write_jsonl(root / "training_metrics.jsonl", history)
    write_json(root / "validation_metrics.json", val_metrics)
    checkpoint_manifest = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "best_epoch": best_epoch,
        "selection_metric": "validation_nll",
        "selection_metric_value": best_val,
        "AIDS_checkpoint_used": False,
    }
    write_json(root / "best_checkpoint_manifest.json", checkpoint_manifest)
    summary = {
        **config,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        **checkpoint_manifest,
        "model_training_performed": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "run_complete": True,
    }
    write_json(root / "run_manifest.json", summary)
    write_json(
        progress_path,
        {
            "config_fingerprint": fingerprint,
            "current_epoch": int(epochs),
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "run_complete": True,
        },
    )
    write_json(root / "_RUN_COMPLETE.json", {"run_complete": True, **checkpoint_manifest})
    return summary


def _predict_graphs(model: Any, graphs: Sequence[Any], device: str) -> list[int]:
    torch, _functional, DataLoader = _runtime_stack()
    predictions: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(list(graphs), batch_size=128, shuffle=False, num_workers=0):
            logits = model(batch.to(device))[-1]
            predictions.extend(
                int(value)
                for value in torch.argmax(logits, dim=-1).detach().cpu().tolist()
            )
    return predictions


def _reset_official_vrrw(vrrw: Any) -> None:
    vrrw.graph_map = {}
    vrrw.graph_index_map = {}
    vrrw.counterfactual_candidates = []
    vrrw.input_graphs_covered = []
    vrrw.covering_graphs = set()
    vrrw.transitions = {}
    vrrw.traversed_hashes = []
    vrrw.starting_step = 1


def _neurosed_input_dim(checkpoint: Path) -> int:
    torch, _functional, _loader = _runtime_stack()
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    value = state.get("embed_model.pre.weight") if isinstance(state, dict) else None
    if value is None or len(value.shape) != 2:
        raise ValueError("Cannot audit NeuroSED input feature dimension.")
    return int(value.shape[1])


def run_official_vrrw(
    *,
    dataset_dir: str | Path,
    official_root: str | Path,
    gnn_checkpoint: str | Path,
    neurosed_checkpoint: str | Path,
    output_dir: str | Path,
    profile: str | RunProfile,
    parent_limit: int,
    max_steps: int,
    alpha: float,
    theta: float,
    teleport: float,
    candidate_capacity: int,
    sample: bool,
    sample_size: int,
    seed: int = SEED,
    device1: str = "cuda:0",
    device2: str = "cuda:0",
    resume: bool = True,
) -> dict[str, Any]:
    schema, _train, _val, generation_records, dataset_summary = (
        load_dataset_artifacts(dataset_dir)
    )
    selected = sorted(generation_records, key=lambda row: str(row["molecule_id"]))[
        : int(parent_limit)
    ]
    resolved_profile = validate_vrrw_profile(
        profile,
        parent_count=len(selected),
        max_steps=max_steps,
        alpha=alpha,
        seed=seed,
    )
    if not math.isclose(float(theta), OFFICIAL_VRRW_THETA, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Official Mutagenicity VRRW theta must be 0.05.")
    if not math.isclose(float(teleport), OFFICIAL_TELEPORT, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Official VRRW teleport probability must be 0.1.")
    checkpoint = Path(gnn_checkpoint).expanduser().resolve()
    neurosed = Path(neurosed_checkpoint).expanduser().resolve()
    if checkpoint_is_aids(checkpoint):
        raise ValueError("Mutagenicity VRRW cannot use an AIDS/HIV checkpoint.")
    for path, label in ((checkpoint, "GNN"), (neurosed, "NeuroSED")):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{label} checkpoint missing: {path}")
    if _neurosed_input_dim(neurosed) != schema.node_feature_dim:
        raise ValueError(
            "NeuroSED checkpoint feature dimension does not match the frozen "
            "strict-train atom vocabulary."
        )
    config = {
        "dataset": "Mutagenicity",
        "dataset_name": "mutagenicity",
        "profile": resolved_profile.value,
        "dataset_dir": str(Path(dataset_dir).expanduser().resolve()),
        "official_root": str(Path(official_root).expanduser().resolve()),
        "gnn_checkpoint": str(checkpoint),
        "gnn_checkpoint_sha256": sha256_file(checkpoint),
        "neurosed_checkpoint": str(neurosed),
        "neurosed_checkpoint_sha256": sha256_file(neurosed),
        "parent_limit": int(parent_limit),
        "generation_source_parent_rows": EXPECTED_GENERATION_SOURCE_ROWS,
        "generation_parent_ids": [str(row["molecule_id"]) for row in selected],
        "generation_source_cohort_hash": cohort_hash(selected),
        "M": int(max_steps),
        "alpha": float(alpha),
        "theta": float(theta),
        "theta_source": "official_vrrw_mutagenicity_default",
        "distance_normalization": "neurosed_divided_by_sum_graph_element_counts",
        "teleport": float(teleport),
        "dynamic_teleportation": True,
        "candidate_capacity": int(candidate_capacity),
        "sample": bool(sample),
        "sample_size": int(sample_size),
        "seed": int(seed),
        "node_feature_dim": schema.node_feature_dim,
        "calibration_loaded": False,
        "test_loaded": False,
        "resume_mode": "deterministic_restart_from_seed",
    }
    fingerprint = stable_json_sha256(config)
    config["config_fingerprint"] = fingerprint
    root = Path(output_dir).expanduser().resolve()
    had_incomplete_output = root.exists() and any(root.iterdir())
    _prepare_output(root, fingerprint=fingerprint, resume=resume, allow_progress=True)
    if had_incomplete_output:
        # Official VRRW exposes no serializable per-step continuation API.  A
        # same-config resume therefore restarts only VRRW inference from the
        # frozen checkpoints and seed; GNN training and rule semantics are not
        # repeated or changed.
        runtime_root = root / "official_runtime"
        if runtime_root.exists():
            shutil.rmtree(runtime_root)
        for name in (
            "counterfactuals.pt",
            "visited_graph_universe.pt",
            "visit_counts.jsonl",
            "run_manifest.json",
            "_RUN_FAILED.json",
            "failure_summary.json",
        ):
            path = root / name
            if path.exists():
                path.unlink()
    write_json(root / "resolved_config.json", config)
    write_json(
        root / "vrrw_progress.json",
        {
            "config_fingerprint": fingerprint,
            "current_step": 0,
            "max_steps": int(max_steps),
            "run_complete": False,
            "resume_mode": "deterministic_restart_from_seed",
        },
    )
    torch, _functional, _loader = _runtime_stack()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Official VRRW requires NumPy.") from exc
    np.random.seed(seed)
    modules = import_official_modules(official_root)
    vrrw = modules["vrrw"]
    importance = modules["importance"]
    distance = modules["distance"]
    _reset_official_vrrw(vrrw)
    graphs = [
        record_to_pyg(record, origin_index=index)
        for index, record in enumerate(selected)
    ]
    dataset = GraphRecordDataset(graphs, schema.node_feature_dim)
    target_device1 = _device_name(torch, device1)
    target_device2 = _device_name(torch, device2)
    model = _load_official_gnn(
        modules,
        num_features=schema.node_feature_dim,
        checkpoint=checkpoint,
        device=target_device1,
    )
    internal_predictions = _predict_graphs(model, graphs, target_device1)
    prediction_counts = dict(Counter(internal_predictions))
    write_jsonl(
        root / "internal_gnn_predictions.jsonl",
        (
            {
                "molecule_id": record["molecule_id"],
                "source_graph_hash": record["source_graph_hash"],
                "project_label": int(record["label"]),
                "official_gnn_prediction": int(prediction),
            }
            for record, prediction in zip(selected, internal_predictions, strict=True)
        ),
    )
    original_load_neurosed = distance.load_neurosed

    def load_explicit_neurosed(original_graphs: Any, neurosed_model_path: Any, device: Any) -> Any:
        del neurosed_model_path
        return original_load_neurosed(
            original_graphs,
            neurosed_model_path=str(neurosed),
            device=device,
        )

    distance.load_neurosed = load_explicit_neurosed
    vrrw.neighbor_graph_access = graph_lineage_neighbor_wrapper(
        vrrw.neighbor_graph_access
    )
    vrrw.dataset_name = "mutagenicity"
    vrrw.alpha = float(alpha)
    vrrw.sample_size = int(sample_size)
    vrrw.is_sample = bool(sample)
    vrrw.MAX_COUNTERFACTUAL_SIZE = int(candidate_capacity)
    vrrw.input_graphs_covered = torch.zeros(len(graphs), dtype=torch.float)
    original_indices = np.arange(len(graphs))
    runtime_root = root / "official_runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    try:
        os.chdir(runtime_root)
        importance_args = importance.prepare_and_get(
            dataset,
            model,
            original_indices,
            float(alpha),
            float(theta),
            device1=target_device1,
            device2=target_device2,
            dataset_name="mutagenicity",
        )
        vrrw.importance_args = importance_args
        vrrw.counterfactual_summary_with_randomwalk(
            input_graphs=graphs,
            importance_args=importance_args,
            teleport_probability=float(teleport),
            max_steps=int(max_steps),
        )
    finally:
        os.chdir(old_cwd)
        distance.load_neurosed = original_load_neurosed
    official_result = (
        runtime_root / "results/mutagenicity/runs/counterfactuals.pt"
    )
    if not official_result.is_file():
        raise GCFExplainerMutagenicityError(
            "Official VRRW did not produce counterfactuals.pt."
        )
    result_path = root / "counterfactuals.pt"
    shutil.copy2(official_result, result_path)
    payload = _torch_load_compat(result_path)
    graph_map = dict(payload.get("graph_map", {}))
    candidates = list(payload.get("counterfactual_candidates", []))
    _torch_save_compat(
        {"graph_map": graph_map, "source_counterfactuals": str(result_path)},
        root / "visited_graph_universe.pt",
    )
    visit_rows = [
        {
            "native_frequency_rank": index + 1,
            "graph_hash": str(candidate.get("graph_hash")),
            "stable_candidate_id": stable_graph_candidate_id(
                graph_map[candidate.get("graph_hash")]
            ),
            "visit_count": int(candidate.get("frequency", 0)),
            "importance_prediction": float(candidate.get("importance_parts", [0.0])[0]),
        }
        for index, candidate in enumerate(candidates)
        if candidate.get("graph_hash") in graph_map
    ]
    write_jsonl(root / "visit_counts.jsonl", visit_rows)
    manifest = {
        **config,
        "internal_gnn_prediction_counts": prediction_counts,
        "internal_gnn_predictions_path": str(
            root / "internal_gnn_predictions.jsonl"
        ),
        "internal_gnn_predictions_sha256": sha256_file(
            root / "internal_gnn_predictions.jsonl"
        ),
        "visited_graph_count": len(graph_map),
        "counterfactual_candidate_count": len(candidates),
        "traversed_step_count": len(payload.get("traversed_hashes", [])),
        "counterfactuals_path": str(result_path),
        "counterfactuals_sha256": sha256_file(result_path),
        "official_algorithms_reused": [
            "vrrw.counterfactual_summary_with_randomwalk",
            "vrrw.move_to_next_graph",
            "vrrw.populate_counterfactual_candidates",
            "vrrw.dynamic_teleportation_probabilities",
            "importance.prepare_and_get",
            "distance.load_neurosed",
        ],
        "lineage_wrapper_changes_graph_tensors": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "run_complete": True,
    }
    write_json(root / "run_manifest.json", manifest)
    write_json(
        root / "vrrw_progress.json",
        {
            "config_fingerprint": fingerprint,
            "current_step": int(max_steps),
            "max_steps": int(max_steps),
            "visited_graph_count": len(graph_map),
            "run_complete": True,
            "resume_mode": "deterministic_restart_from_seed",
        },
    )
    write_json(
        root / "_RUN_COMPLETE.json",
        {"run_complete": True, "counterfactuals_sha256": sha256_file(result_path)},
    )
    return manifest


def _torch_load_compat(path: str | Path) -> Any:
    torch, _functional, _loader = _runtime_stack()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _torch_save_compat(payload: Any, path: str | Path) -> None:
    torch, _functional, _loader = _runtime_stack()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def build_native_summary(
    *,
    dataset_dir: str | Path,
    official_root: str | Path,
    vrrw_dir: str | Path,
    gnn_checkpoint: str | Path,
    neurosed_checkpoint: str | Path,
    output_dir: str | Path,
    profile: str | RunProfile,
    parent_limit: int,
    theta: float = OFFICIAL_SUMMARY_THETA,
    minimum_native_export: int = 100,
    device: str = "cuda:0",
) -> dict[str, Any]:
    resolved_profile = RunProfile(profile)
    expected_parents = 64 if resolved_profile is RunProfile.SMOKE else 1448
    if int(parent_limit) != expected_parents:
        raise ValueError(
            f"Summary {resolved_profile.value} requires {expected_parents} parents."
        )
    if not math.isclose(float(theta), OFFICIAL_SUMMARY_THETA, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Official native summary theta must be 0.1.")
    if int(minimum_native_export) < 100:
        raise ValueError("Native summary must export at least 100 ranked candidates.")
    root = Path(output_dir).expanduser().resolve()
    if (root / "_FINALIZED.json").exists() or (
        root.exists() and any(root.iterdir())
    ):
        raise FileExistsError(f"Summary output cannot be overwritten: {root}")
    root.mkdir(parents=True, exist_ok=True)
    schema, _train, _val, generation_records, _summary = load_dataset_artifacts(
        dataset_dir
    )
    checkpoint = Path(gnn_checkpoint).expanduser().resolve()
    neurosed = Path(neurosed_checkpoint).expanduser().resolve()
    if checkpoint_is_aids(checkpoint):
        raise ValueError("Mutagenicity summary cannot use an AIDS/HIV checkpoint.")
    for path, label in ((checkpoint, "GNN"), (neurosed, "NeuroSED")):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{label} checkpoint missing: {path}")
    selected_sources = sorted(
        generation_records, key=lambda row: str(row["molecule_id"])
    )[:parent_limit]
    source_graphs = [
        record_to_pyg(row, origin_index=index)
        for index, row in enumerate(selected_sources)
    ]
    vrrw_root = Path(vrrw_dir).expanduser().resolve()
    vrrw_manifest = read_json(vrrw_root / "run_manifest.json")
    if vrrw_manifest.get("run_complete") is not True:
        raise ValueError("VRRW run is not complete.")
    if str(vrrw_manifest.get("gnn_checkpoint_sha256")) != sha256_file(checkpoint):
        raise ValueError("Summary GNN checkpoint does not match the VRRW run.")
    if str(vrrw_manifest.get("neurosed_checkpoint_sha256")) != sha256_file(neurosed):
        raise ValueError("Summary NeuroSED checkpoint does not match the VRRW run.")
    if str(vrrw_manifest.get("generation_source_cohort_hash")) != cohort_hash(
        selected_sources
    ):
        raise ValueError("Summary source cohort does not match the VRRW run.")
    counterfactuals_path = vrrw_root / "counterfactuals.pt"
    payload = _torch_load_compat(counterfactuals_path)
    candidates = list(payload.get("counterfactual_candidates", []))
    graph_map = dict(payload.get("graph_map", {}))
    counterfactual_graphs: list[Any] = []
    candidate_meta: list[dict[str, Any]] = []
    target_native_candidates = max(len(source_graphs), int(minimum_native_export))
    index = 0
    while (
        len(counterfactual_graphs) < target_native_candidates
        and index < len(candidates)
    ):
        candidate = candidates[index]
        importance_parts = candidate.get("importance_parts", [0.0])
        prediction_importance = float(importance_parts[0])
        graph_hash = candidate.get("graph_hash")
        if prediction_importance >= 0.5 and graph_hash in graph_map:
            counterfactual_graphs.append(graph_map[graph_hash])
            candidate_meta.append(
                {
                    "graph_hash": str(graph_hash),
                    "frequency": int(candidate.get("frequency", 0)),
                    "importance_prediction": prediction_importance,
                    "source_candidate_index": index,
                }
            )
        index += 1
    if len(counterfactual_graphs) < minimum_native_export:
        raise GCFExplainerEmptyCandidateSetError(
            "Official VRRW produced fewer than 100 native counterfactual graphs."
        )
    modules = import_official_modules(official_root)
    torch, _functional, _loader = _runtime_stack()
    target_device = _device_name(torch, device)
    distance_model = modules["distance"].load_neurosed(
        source_graphs,
        neurosed_model_path=str(neurosed),
        device=target_device,
    )
    distance_matrix = distance_model.predict_outer_with_queries(
        counterfactual_graphs, batch_size=1000
    ).cpu()
    util = importlib_import_from_official(official_root, "util")
    original_counts = util.graph_element_counts(source_graphs)
    counterfactual_counts = util.graph_element_counts(counterfactual_graphs)
    normalization = torch.cartesian_prod(
        counterfactual_counts, original_counts
    ).sum(dim=1).view(len(counterfactual_graphs), len(source_graphs))
    distances = (distance_matrix / normalization).T
    close_sets = [
        set(torch.where(distances[parent_index] <= float(theta))[0].tolist())
        for parent_index in range(distances.shape[0])
    ]
    counterfactual_covering = {
        candidate_index: set() for candidate_index in range(distances.shape[1])
    }
    graphs_covered_by = {
        parent_index: set() for parent_index in range(distances.shape[0])
    }
    for parent_index, close in enumerate(close_sets):
        for candidate_index in close:
            counterfactual_covering[candidate_index].add(parent_index)
            graphs_covered_by[parent_index].add(candidate_index)
    coverings = modules["summary"].greedy_counterfactual_summary_from_covering_sets(
        counterfactual_covering={key: set(value) for key, value in counterfactual_covering.items()},
        graphs_covered_by={key: set(value) for key, value in graphs_covered_by.items()},
        k=len(counterfactual_covering),
    )
    order = [int(coverings[rank][0]) for rank in sorted(coverings)]
    native_rows: list[dict[str, Any]] = []
    selected_graphs: list[Any] = []
    for rank, candidate_index in enumerate(order, start=1):
        graph = counterfactual_graphs[candidate_index]
        selected_graphs.append(graph)
        native_rows.append(
            {
                "candidate_id": stable_graph_candidate_id(graph),
                "native_rank": rank,
                "candidate_index": candidate_index,
                "graph_hash": candidate_meta[candidate_index]["graph_hash"],
                "frequency": candidate_meta[candidate_index]["frequency"],
                "importance_prediction": candidate_meta[candidate_index]["importance_prediction"],
                "covered_parent_count_at_rank": int(coverings[rank][1]),
                "selection_method": "official_greedy_coverage_gain",
            }
        )
    # Keep the complete official greedy order.  The minimum of 100 is a gate,
    # not a truncation point, so later chemistry/RF filtering never re-ranks or
    # silently discards lower native ranks.
    exported_rows = native_rows
    exported_graphs = selected_graphs
    _torch_save_compat(
        {
            "selected_graphs": exported_graphs,
            "selected_records": exported_rows,
            "source_counterfactuals_path": str(counterfactuals_path),
            "selection_policy": "official_greedy_coverage_gain",
        },
        root / "selected_counterfactual_graphs.pt",
    )
    write_jsonl(root / "native_summary_rank.jsonl", exported_rows)
    write_csv(
        root / "native_summary_rank.csv",
        exported_rows,
        (
            "candidate_id",
            "native_rank",
            "candidate_index",
            "graph_hash",
            "frequency",
            "importance_prediction",
            "covered_parent_count_at_rank",
            "selection_method",
        ),
    )
    _torch_save_compat(distances, root / "native_distance_matrix.pt")
    config = {
        "dataset": "Mutagenicity",
        "profile": resolved_profile.value,
        "parent_count": len(source_graphs),
        "native_candidate_count": len(counterfactual_graphs),
        "native_rank_exported": len(exported_rows),
        "theta": float(theta),
        "theta_source": "official_summary_default",
        "distance_normalization": "neurosed_divided_by_sum_graph_element_counts",
        "selection_method": "official_greedy_coverage_gain",
        "gnn_checkpoint": str(checkpoint),
        "gnn_checkpoint_sha256": sha256_file(checkpoint),
        "neurosed_checkpoint": str(neurosed),
        "neurosed_checkpoint_sha256": sha256_file(neurosed),
        "source_cohort_hash": cohort_hash(selected_sources),
        "calibration_loaded": False,
        "test_loaded": False,
        "run_complete": True,
    }
    write_json(root / "resolved_config.json", config)
    write_json(root / "run_manifest.json", config)
    write_json(root / "_RUN_COMPLETE.json", {"run_complete": True, **config})
    return config


def importlib_import_from_official(official_root: str | Path, module_name: str) -> Any:
    root = str(Path(official_root).expanduser().resolve())
    sys.path.insert(0, root)
    try:
        return __import__(module_name)
    finally:
        try:
            sys.path.remove(root)
        except ValueError:
            pass


def export_rf_valid_native_top20(
    *,
    dataset_dir: str | Path,
    summary_dir: str | Path,
    teacher: Any,
    teacher_path: str | Path,
    output_dir: str | Path,
    profile: str | RunProfile,
    parent_limit: int,
    top_k: int = 20,
) -> dict[str, Any]:
    resolved_profile = RunProfile(profile)
    expected = 64 if resolved_profile is RunProfile.SMOKE else 1448
    if int(parent_limit) != expected:
        raise ValueError(
            f"Export {resolved_profile.value} requires parent_limit={expected}."
        )
    if int(top_k) != 20:
        raise ValueError("Mutagenicity GCF export requires top_k=20.")
    root = Path(output_dir).expanduser().resolve()
    if (root / "_FINALIZED.json").exists() or (
        root.exists() and any(root.iterdir())
    ):
        raise FileExistsError(f"Export output cannot be overwritten: {root}")
    root.mkdir(parents=True, exist_ok=True)
    schema, _train, _val, generation_records, _dataset_summary = (
        load_dataset_artifacts(dataset_dir)
    )
    source_records = sorted(
        generation_records, key=lambda row: str(row["molecule_id"])
    )[:parent_limit]
    summary_root = Path(summary_dir).expanduser().resolve()
    payload = _torch_load_compat(
        summary_root / "selected_counterfactual_graphs.pt"
    )
    graphs = list(payload.get("selected_graphs", []))
    native_rows = list(payload.get("selected_records", []))
    candidate_universe, skipped = build_native_rank_candidate_universe(
        native_rows,
        graphs,
        source_records,
        schema,
        teacher,
    )
    selected = candidate_universe[: int(top_k)]
    write_jsonl(
        root / "decoded_native_candidates.jsonl",
        sorted(
            [*candidate_universe, *skipped],
            key=lambda row: int(row.get("native_rank", 0)),
        ),
    )
    write_jsonl(
        root / "invalid_candidates.jsonl",
        [row for row in skipped if str(row.get("skip_reason", "")).startswith("generated_") or "lineage" in str(row.get("skip_reason", ""))],
    )
    write_jsonl(
        root / "non_target_candidates.jsonl",
        [row for row in skipped if row.get("skip_reason") == "rf_not_target_label_0"],
    )
    write_csv(
        root / "selected_top20.csv",
        selected,
        (
            "candidate_id",
            "native_rank",
            "smiles",
            "canonical_smiles",
            "rdkit_valid",
            "rf_pred",
            "rf_prob_0",
            "rf_prob_1",
            "source_method",
            "selection_method",
            "candidate_set_preselected",
            "selection_performed_in_eval",
            "projected_new_edge_count",
            "retained_edge_count",
            "source_parent_id",
        ),
    )
    write_jsonl(root / "candidate_universe.jsonl", candidate_universe)
    reason_counts = dict(Counter(str(row.get("skip_reason")) for row in skipped))
    manifest = {
        "dataset": "Mutagenicity",
        "profile": resolved_profile.value,
        "source_label": 1,
        "target_label": 0,
        "semantic_direction": "mutagenic_to_non_mutagenic",
        "strict_flip_definition": "source_teacher_pred_1_and_candidate_teacher_pred_0",
        "parent_limit": int(parent_limit),
        "native_rank_rows": len(native_rows),
        "rf_target_candidate_universe_rows": len(candidate_universe),
        "selected_top20_rows": len(selected),
        "top_k": int(top_k),
        "skipped_reason_counts": reason_counts,
        "selection_method": "native_gcf_summary_rank_filtered_by_validity",
        "native_rank_reordered": False,
        "candidate_set_preselected": True,
        "selection_performed_in_eval": False,
        "teacher_path": str(Path(teacher_path).expanduser().resolve()),
        "teacher_sha256": sha256_file(teacher_path),
        "teacher_used_only_for_target_validation": True,
        "selected_candidate_order_sha256": stable_json_sha256(
            [str(row["candidate_id"]) for row in selected]
        ),
        "calibration_loaded": False,
        "test_loaded": False,
        "run_complete": len(selected) == int(top_k),
    }
    write_json(root / "run_manifest.json", manifest)
    if len(selected) != int(top_k):
        write_json(
            root / "failure_summary.json",
            {**manifest, "failure": "fewer_than_20_rf_valid_native_candidates"},
        )
        write_json(
            root / "_RUN_FAILED.json",
            {**manifest, "run_complete": False},
        )
        raise GCFExplainerEmptyCandidateSetError(
            f"Native-rank filtering produced {len(selected)} / {top_k} candidates."
        )
    write_json(root / "_RUN_COMPLETE.json", manifest)
    return manifest


def audit_mutagenicity_run(
    *,
    dataset_dir: str | Path,
    gnn_dir: str | Path,
    vrrw_dir: str | Path,
    summary_dir: str | Path,
    export_dir: str | Path,
    profile: str | RunProfile,
) -> dict[str, Any]:
    resolved_profile = RunProfile(profile)
    expected_parents = 64 if resolved_profile is RunProfile.SMOKE else 1448
    dataset_summary = read_json(Path(dataset_dir) / "dataset_summary.json")
    gnn_manifest = read_json(Path(gnn_dir) / "run_manifest.json")
    vrrw_manifest = read_json(Path(vrrw_dir) / "run_manifest.json")
    summary_manifest = read_json(Path(summary_dir) / "run_manifest.json")
    export_manifest = read_json(Path(export_dir) / "run_manifest.json")
    if int(dataset_summary.get("train_rows", -1)) != EXPECTED_MODEL_TRAIN_ROWS:
        raise AssertionError("GNN model train count is not 2885.")
    if int(dataset_summary.get("val_rows", -1)) != EXPECTED_MODEL_VAL_ROWS:
        raise AssertionError("GNN validation count is not 355.")
    if int(dataset_summary.get("generation_source_rows", -1)) != EXPECTED_GENERATION_SOURCE_ROWS:
        raise AssertionError("Generation source count is not 1448.")
    for name, payload in (
        ("dataset", dataset_summary),
        ("gnn", gnn_manifest),
        ("vrrw", vrrw_manifest),
        ("summary", summary_manifest),
        ("export", export_manifest),
    ):
        if payload.get("calibration_loaded") is not False:
            raise AssertionError(f"{name} calibration_loaded must be false.")
        if payload.get("test_loaded") is not False:
            raise AssertionError(f"{name} test_loaded must be false.")
        if payload.get("run_complete") is not True:
            raise AssertionError(f"{name} is incomplete.")
    if checkpoint_is_aids(str(gnn_manifest.get("checkpoint_path", ""))):
        raise AssertionError("AIDS checkpoint leaked into Mutagenicity run.")
    checkpoint = Path(str(gnn_manifest["checkpoint_path"]))
    if sha256_file(checkpoint) != str(gnn_manifest["checkpoint_sha256"]):
        raise AssertionError("GNN checkpoint SHA256 mismatch.")
    if int(vrrw_manifest.get("parent_limit", -1)) != expected_parents:
        raise AssertionError("VRRW parent count mismatch.")
    parent_ids = [str(value) for value in vrrw_manifest.get("generation_parent_ids", [])]
    if len(parent_ids) != expected_parents or len(parent_ids) != len(set(parent_ids)):
        raise AssertionError("VRRW generation parent IDs are incomplete or duplicated.")
    if str(vrrw_manifest.get("gnn_checkpoint_sha256")) != str(
        gnn_manifest.get("checkpoint_sha256")
    ):
        raise AssertionError("VRRW did not use the frozen GNN checkpoint.")
    if resolved_profile is RunProfile.FULL:
        if int(vrrw_manifest.get("M", -1)) != 50000:
            raise AssertionError("Full VRRW M must be 50000.")
        if float(vrrw_manifest.get("alpha", -1)) != 1.0:
            raise AssertionError("Full VRRW alpha must be 1.0.")
    native_rows = read_jsonl(Path(summary_dir) / "native_summary_rank.jsonl")
    ranks = [int(row["native_rank"]) for row in native_rows]
    if ranks != list(range(1, len(ranks) + 1)) or len(set(ranks)) != len(ranks):
        raise AssertionError("Native summary rank is not unique and contiguous.")
    if len(native_rows) < 100:
        raise AssertionError("Native summary must export at least 100 ranks.")
    candidate_universe = read_jsonl(Path(export_dir) / "candidate_universe.jsonl")
    if len(candidate_universe) != int(
        export_manifest.get("rf_target_candidate_universe_rows", -1)
    ):
        raise AssertionError("Candidate universe row count does not match manifest.")
    universe_ranks = [int(row["native_rank"]) for row in candidate_universe]
    if universe_ranks != sorted(universe_ranks):
        raise AssertionError("Candidate universe no longer follows native rank.")
    universe_smiles = [str(row["canonical_smiles"]) for row in candidate_universe]
    if len(universe_smiles) != len(set(universe_smiles)):
        raise AssertionError("Candidate universe contains canonical duplicates.")
    if any(int(row["rf_pred"]) != 0 for row in candidate_universe):
        raise AssertionError("Candidate universe contains a non-target RF prediction.")
    from rdkit import Chem

    if any(Chem.MolFromSmiles(smiles) is None for smiles in universe_smiles):
        raise AssertionError("Candidate universe contains an RDKit-invalid molecule.")
    with (Path(export_dir) / "selected_top20.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        selected = [dict(row) for row in __import__("csv").DictReader(handle)]
    if len(selected) != 20:
        raise AssertionError("selected_top20.csv must contain exactly 20 rows.")
    selected_ranks = [int(row["native_rank"]) for row in selected]
    if selected_ranks != sorted(selected_ranks):
        raise AssertionError("RF filtering changed native candidate order.")
    if any(int(row["rf_pred"]) != 0 for row in selected):
        raise AssertionError("Selected candidate is not RF target label 0.")
    smiles = [row["canonical_smiles"] for row in selected]
    if len(smiles) != len(set(smiles)):
        raise AssertionError("Selected candidate canonical SMILES are duplicated.")
    if export_manifest.get("selection_performed_in_eval") is not False:
        raise AssertionError("Candidate selection was incorrectly deferred to evaluation.")
    selected_order_hash = stable_json_sha256(
        [str(row["candidate_id"]) for row in selected]
    )
    if selected_order_hash != str(
        export_manifest.get("selected_candidate_order_sha256")
    ):
        raise AssertionError("Frozen selected candidate order hash mismatch.")
    result = {
        "audit_passed": True,
        "run_complete": True,
        "profile": resolved_profile.value,
        "model_train_rows": EXPECTED_MODEL_TRAIN_ROWS,
        "model_val_rows": EXPECTED_MODEL_VAL_ROWS,
        "generation_source_rows": EXPECTED_GENERATION_SOURCE_ROWS,
        "selected_generation_parents": expected_parents,
        "selected_top20_rows": len(selected),
        "candidate_universe_rows": len(candidate_universe),
        "native_rank_unique": True,
        "native_rank_reordered": False,
        "selected_candidate_order_sha256": selected_order_hash,
        "all_candidates_rf_target_0": True,
        "calibration_rows_loaded": 0,
        "test_rows_loaded": 0,
    }
    write_json(Path(export_dir) / "audit.json", result)
    return result


__all__ = [
    "OFFICIAL_SUMMARY_THETA",
    "OFFICIAL_TELEPORT",
    "OFFICIAL_VRRW_THETA",
    "audit_mutagenicity_run",
    "build_native_summary",
    "export_rf_valid_native_top20",
    "run_official_vrrw",
    "train_official_gnn",
]
