"""Project-owned orchestration for official GCFExplainer on frozen BACE data."""

from __future__ import annotations

import hashlib
import math
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.baselines.gcfexplainer_bace_adapter import (
    DATASET,
    EXPECTED_GENERATION_SOURCE_ROWS,
    EXPECTED_MODEL_TRAIN_ROWS,
    EXPECTED_MODEL_VAL_ROWS,
    SEED,
    SOURCE_LABEL,
    TARGET_LABEL,
    load_bace_gcf_dataset,
    validate_bace_gnn_profile,
    validate_bace_vrrw_profile,
)
from src.baselines.gcfexplainer_mutagenicity_adapter import (
    GCFExplainerEmptyCandidateSetError,
    GCFExplainerMutagenicityCodecError,
    GraphRecordDataset,
    cohort_hash,
    decode_generated_fullgraph,
    deterministic_balanced_prefix,
    graph_lineage_neighbor_wrapper,
    import_official_modules,
    read_json,
    record_to_pyg,
    score_teacher_probabilities,
    sha256_file,
    stable_graph_candidate_id,
    stable_json_sha256,
    write_csv,
    write_json,
    write_jsonl,
)
from src.baselines.gcfexplainer_mutagenicity_runtime import (
    OFFICIAL_SUMMARY_THETA,
    OFFICIAL_TELEPORT,
    VRRW_ALPHA_ENDPOINT_PATCH,
    _alpha_endpoint_branch,
    _device_name,
    _evaluate_gnn,
    _load_official_gnn,
    _neurosed_input_dim,
    _official_vrrw_alpha_endpoint_patch,
    _predict_graphs,
    _prepare_output,
    _reset_official_vrrw,
    _runtime_stack,
    _torch_load_compat,
    _torch_save_compat,
    importlib_import_from_official,
)


def _reuse_complete(root: Path, *, resume: bool) -> dict[str, Any] | None:
    marker = root / "_RUN_COMPLETE.json"
    if not marker.is_file():
        return None
    if not resume:
        raise FileExistsError(f"Completed BACE GCF output cannot be overwritten: {root}")
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("run_complete") is not True:
        raise ValueError(f"Completion marker disagrees with manifest: {root}")
    return manifest


def train_bace_official_gnn(
    *,
    dataset_dir: str | Path,
    official_root: str | Path,
    output_dir: str | Path,
    profile: str,
    epochs: int,
    train_limit: int,
    val_limit: int,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    dropout: float = 0.0,
    seed: int = SEED,
    device: str = "cuda:0",
    resume: bool = True,
) -> dict[str, Any]:
    schema, train_records, val_records, _generation, _summary = load_bace_gcf_dataset(
        dataset_dir
    )
    train_selected = deterministic_balanced_prefix(train_records, int(train_limit))
    val_selected = deterministic_balanced_prefix(val_records, int(val_limit))
    validate_bace_gnn_profile(
        profile,
        epochs=int(epochs),
        train_rows=len(train_selected),
        val_rows=len(val_selected),
    )
    if int(seed) != SEED:
        raise ValueError("BACE GNN requires seed=13.")
    root = Path(output_dir).expanduser().resolve()
    complete = _reuse_complete(root, resume=resume)
    if complete is not None:
        return complete
    official = Path(official_root).expanduser().resolve()
    config = {
        "dataset": DATASET,
        "dataset_name": "bace",
        "profile": str(profile),
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
    _prepare_output(root, fingerprint=fingerprint, resume=resume, allow_progress=True)
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
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_graphs,
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_graphs, batch_size=int(batch_size), shuffle=False, num_workers=0
    )
    progress_path = root / "training_progress.json"
    state_path = root / "training_state.pt"
    checkpoint = root / "model_best.pth"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_val = math.inf
    best_epoch = 0
    if progress_path.is_file() and state_path.is_file() and resume:
        progress = read_json(progress_path)
        if progress.get("config_fingerprint") != fingerprint:
            raise ValueError("BACE GNN resume fingerprint mismatch.")
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
        validation = _evaluate_gnn(model, val_loader, target_device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_rows,
            "val_loss": float(validation["loss"]),
            "val_accuracy": float(validation["accuracy"]),
        }
        history.append(row)
        if float(validation["loss"]) < best_val:
            best_val = float(validation["loss"])
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
        raise RuntimeError("BACE official GNN did not produce a checkpoint.")
    model = _load_official_gnn(
        modules,
        num_features=schema.node_feature_dim,
        checkpoint=checkpoint,
        device=target_device,
        dropout=float(dropout),
    )
    checkpoint_manifest = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "best_epoch": best_epoch,
        "selection_metric": "validation_nll",
        "selection_metric_value": best_val,
        "AIDS_checkpoint_used": False,
        "Mutagenicity_GNN_checkpoint_used": False,
    }
    summary = {
        **config,
        "train_metrics": _evaluate_gnn(model, train_loader, target_device),
        "validation_metrics": _evaluate_gnn(model, val_loader, target_device),
        **checkpoint_manifest,
        "model_training_performed": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "run_complete": True,
    }
    write_jsonl(root / "training_metrics.jsonl", history)
    write_json(root / "best_checkpoint_manifest.json", checkpoint_manifest)
    write_json(root / "run_manifest.json", summary)
    write_json(progress_path, {**checkpoint_manifest, "run_complete": True})
    write_json(root / "_RUN_COMPLETE.json", {"run_complete": True, **checkpoint_manifest})
    return summary


def run_bace_official_vrrw(
    *,
    dataset_dir: str | Path,
    official_root: str | Path,
    gnn_checkpoint: str | Path,
    neurosed_checkpoint: str | Path,
    neurosed_manifest: str | Path,
    output_dir: str | Path,
    profile: str,
    parent_limit: int,
    m: int,
    alpha: float = 1.0,
    theta: float = 0.05,
    teleport: float = OFFICIAL_TELEPORT,
    candidate_capacity: int = 100000,
    sample: bool = False,
    sample_size: int = 10000,
    seed: int = SEED,
    device1: str = "cuda:0",
    device2: str = "cuda:0",
    resume: bool = True,
) -> dict[str, Any]:
    schema, _train, _val, generation, dataset_summary = load_bace_gcf_dataset(
        dataset_dir
    )
    validate_bace_vrrw_profile(
        profile,
        parent_limit=int(parent_limit),
        m=int(m),
        alpha=float(alpha),
        theta=float(theta),
        seed=int(seed),
    )
    if not math.isclose(float(teleport), OFFICIAL_TELEPORT, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Official BACE VRRW teleport must remain 0.1.")
    selected = sorted(generation, key=lambda row: str(row["molecule_id"]))[
        : int(parent_limit)
    ]
    if len(selected) != int(parent_limit):
        raise ValueError("BACE generation source cohort is smaller than parent_limit.")
    checkpoint = Path(gnn_checkpoint).expanduser().resolve()
    neurosed = Path(neurosed_checkpoint).expanduser().resolve()
    projection_manifest = Path(neurosed_manifest).expanduser().resolve()
    for path, label in (
        (checkpoint, "BACE GNN"),
        (neurosed, "BACE NeuroSED"),
        (projection_manifest, "BACE NeuroSED manifest"),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{label} missing: {path}")
    projection = read_json(projection_manifest)
    if projection.get("dataset") != DATASET:
        raise ValueError("NeuroSED projection manifest is not for BACE.")
    if projection.get("output_checkpoint_sha256") != sha256_file(neurosed):
        raise ValueError("BACE NeuroSED projection checkpoint hash mismatch.")
    if _neurosed_input_dim(neurosed) != schema.node_feature_dim:
        raise ValueError("BACE NeuroSED input dimension does not match BACE schema.")
    config = {
        "dataset": DATASET,
        "dataset_name": "bace",
        "profile": str(profile),
        "dataset_dir": str(Path(dataset_dir).expanduser().resolve()),
        "official_root": str(Path(official_root).expanduser().resolve()),
        "gnn_checkpoint": str(checkpoint),
        "gnn_checkpoint_sha256": sha256_file(checkpoint),
        "neurosed_checkpoint": str(neurosed),
        "neurosed_checkpoint_sha256": sha256_file(neurosed),
        "neurosed_manifest": str(projection_manifest),
        "neurosed_manifest_sha256": sha256_file(projection_manifest),
        "neurosed_adaptation": projection["adaptation"],
        "parent_limit": int(parent_limit),
        "generation_source_parent_rows": EXPECTED_GENERATION_SOURCE_ROWS,
        "generation_parent_ids": [str(row["molecule_id"]) for row in selected],
        "generation_parent_ids_sha256": stable_json_sha256(
            [str(row["molecule_id"]) for row in selected]
        ),
        "generation_source_cohort_hash": cohort_hash(selected),
        "M": int(m),
        "alpha": float(alpha),
        "alpha_endpoint_branch": _alpha_endpoint_branch(alpha),
        "theta": float(theta),
        "theta_source": "verified_mutagenicity_official_vrrw_protocol",
        "teleport": float(teleport),
        "dynamic_teleportation": True,
        "candidate_capacity": int(candidate_capacity),
        "sample": bool(sample),
        "sample_size": int(sample_size),
        "seed": int(seed),
        "node_feature_dim": schema.node_feature_dim,
        "dataset_fingerprint": dataset_summary.get("generation_source_cohort_hash"),
        "calibration_loaded": False,
        "test_loaded": False,
        "official_compatibility_patches": [VRRW_ALPHA_ENDPOINT_PATCH],
    }
    root = Path(output_dir).expanduser().resolve()
    complete = _reuse_complete(root, resume=resume)
    if complete is not None:
        return complete
    fingerprint = stable_json_sha256(config)
    config["config_fingerprint"] = fingerprint
    _prepare_output(root, fingerprint=fingerprint, resume=resume, allow_progress=False)
    write_json(root / "resolved_config.json", config)
    torch, _functional, _loader = _runtime_stack()
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Official BACE VRRW requires NumPy.") from exc
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
    original_neighbor_access = vrrw.neighbor_graph_access

    def load_explicit_neurosed(
        original_graphs: Any, neurosed_model_path: Any, device: Any
    ) -> Any:
        del neurosed_model_path
        return original_load_neurosed(
            original_graphs,
            neurosed_model_path=str(neurosed),
            device=device,
        )

    distance.load_neurosed = load_explicit_neurosed
    vrrw.neighbor_graph_access = graph_lineage_neighbor_wrapper(
        original_neighbor_access
    )
    vrrw.dataset_name = "bace"
    vrrw.alpha = float(alpha)
    vrrw.sample_size = int(sample_size)
    vrrw.is_sample = bool(sample)
    vrrw.MAX_COUNTERFACTUAL_SIZE = int(candidate_capacity)
    vrrw.input_graphs_covered = torch.zeros(len(graphs), dtype=torch.float)
    runtime_root = root / "official_runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    try:
        os.chdir(runtime_root)
        importance_args = importance.prepare_and_get(
            dataset,
            model,
            np.arange(len(graphs)),
            float(alpha),
            float(theta),
            device1=target_device1,
            device2=target_device2,
            dataset_name="bace",
        )
        vrrw.importance_args = importance_args
        print("[GCFEXPLAINER_OFFICIAL_COMPAT_PATCH]", flush=True)
        print(f"patch={VRRW_ALPHA_ENDPOINT_PATCH}", flush=True)
        print("dataset=bace", flush=True)
        print(f"alpha={float(alpha)}", flush=True)
        print("official_source_modified=false", flush=True)
        with _official_vrrw_alpha_endpoint_patch(vrrw):
            vrrw.counterfactual_summary_with_randomwalk(
                input_graphs=graphs,
                importance_args=importance_args,
                teleport_probability=float(teleport),
                max_steps=int(m),
            )
    finally:
        os.chdir(old_cwd)
        distance.load_neurosed = original_load_neurosed
        vrrw.neighbor_graph_access = original_neighbor_access
    official_result = runtime_root / "results/bace/runs/counterfactuals.pt"
    if not official_result.is_file():
        raise RuntimeError("Official BACE VRRW did not produce counterfactuals.pt.")
    result_path = root / "counterfactuals.pt"
    shutil.copy2(official_result, result_path)
    payload = _torch_load_compat(result_path)
    graph_map = dict(payload.get("graph_map", {}))
    candidates = list(payload.get("counterfactual_candidates", []))
    manifest = {
        **config,
        "internal_gnn_prediction_counts": dict(Counter(internal_predictions)),
        "internal_gnn_predictions_path": str(root / "internal_gnn_predictions.jsonl"),
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
    write_json(root / "_RUN_COMPLETE.json", {"run_complete": True})
    return manifest


def _resolve_parent_lineage(
    generation: Sequence[Mapping[str, Any]],
    vrrw_manifest: Mapping[str, Any],
    profile: str,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    if vrrw_manifest.get("run_complete") is not True:
        raise ValueError("BACE VRRW manifest is not complete.")
    if str(vrrw_manifest.get("profile")) != str(profile):
        raise ValueError("BACE summary profile differs from VRRW profile.")
    expected_parent_limit = 64 if profile == "smoke" else EXPECTED_GENERATION_SOURCE_ROWS
    if int(vrrw_manifest.get("parent_limit", -1)) != expected_parent_limit:
        raise ValueError("BACE VRRW parent_limit differs from profile contract.")
    if int(vrrw_manifest.get("generation_source_parent_rows", -1)) != len(generation):
        raise ValueError("BACE generation universe count differs from VRRW manifest.")
    parent_ids = [str(value) for value in vrrw_manifest.get("generation_parent_ids", [])]
    if len(parent_ids) != expected_parent_limit or len(parent_ids) != len(set(parent_ids)):
        raise ValueError("BACE VRRW generation parent lineage is incomplete or duplicated.")
    by_id = {str(record["molecule_id"]): record for record in generation}
    if len(by_id) != len(generation):
        raise ValueError("BACE dataset generation parent IDs are duplicated.")
    missing = [value for value in parent_ids if value not in by_id]
    if missing:
        raise ValueError(f"BACE VRRW parent IDs are missing from dataset: {missing[:5]}")
    selected = [by_id[value] for value in parent_ids]
    if cohort_hash(selected) != str(vrrw_manifest.get("generation_source_cohort_hash")):
        raise ValueError("BACE VRRW generation cohort hash mismatch.")
    return selected, {
        "generation_source_parent_rows": len(generation),
        "vrrw_parent_limit": expected_parent_limit,
        "vrrw_selected_parent_count": len(parent_ids),
        "summary_parent_count": len(selected),
        "generation_parent_ids": parent_ids,
        "generation_parent_ids_sha256": stable_json_sha256(parent_ids),
        "generation_source_cohort_hash": cohort_hash(selected),
        "parent_order_source": "vrrw_manifest_generation_parent_ids",
    }


def build_bace_native_summary(
    *,
    dataset_dir: str | Path,
    official_root: str | Path,
    vrrw_dir: str | Path,
    gnn_checkpoint: str | Path,
    neurosed_checkpoint: str | Path,
    output_dir: str | Path,
    profile: str,
    theta: float = OFFICIAL_SUMMARY_THETA,
    minimum_native_export: int = 100,
    device: str = "cuda:0",
) -> dict[str, Any]:
    if not math.isclose(float(theta), OFFICIAL_SUMMARY_THETA, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("BACE native summary must retain official theta=0.1.")
    if int(minimum_native_export) < 100:
        raise ValueError("BACE native summary must export at least 100 ranks.")
    root = Path(output_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"BACE summary output cannot be overwritten: {root}")
    root.mkdir(parents=True, exist_ok=True)
    _schema, _train, _val, generation, _dataset_summary = load_bace_gcf_dataset(
        dataset_dir
    )
    vrrw_root = Path(vrrw_dir).expanduser().resolve()
    vrrw_manifest_path = vrrw_root / "run_manifest.json"
    vrrw_manifest = read_json(vrrw_manifest_path)
    selected_sources, lineage = _resolve_parent_lineage(generation, vrrw_manifest, profile)
    checkpoint = Path(gnn_checkpoint).expanduser().resolve()
    neurosed = Path(neurosed_checkpoint).expanduser().resolve()
    if sha256_file(checkpoint) != str(vrrw_manifest.get("gnn_checkpoint_sha256")):
        raise ValueError("BACE summary GNN checkpoint differs from VRRW lineage.")
    if sha256_file(neurosed) != str(vrrw_manifest.get("neurosed_checkpoint_sha256")):
        raise ValueError("BACE summary NeuroSED checkpoint differs from VRRW lineage.")
    counterfactuals_path = (vrrw_root / "counterfactuals.pt").resolve()
    if sha256_file(counterfactuals_path) != str(
        vrrw_manifest.get("counterfactuals_sha256")
    ):
        raise ValueError("BACE summary counterfactual artifact hash mismatch.")
    payload = _torch_load_compat(counterfactuals_path)
    candidates = list(payload.get("counterfactual_candidates", []))
    graph_map = dict(payload.get("graph_map", {}))
    if len(candidates) != int(vrrw_manifest.get("counterfactual_candidate_count", -1)):
        raise ValueError("BACE VRRW candidate count differs from its manifest.")
    counterfactual_graphs: list[Any] = []
    candidate_meta: list[dict[str, Any]] = []
    target_count = max(len(selected_sources), int(minimum_native_export))
    for index, candidate in enumerate(candidates):
        parts = candidate.get("importance_parts", [0.0])
        prediction_importance = float(parts[0])
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
        if len(counterfactual_graphs) >= target_count:
            break
    if len(counterfactual_graphs) < int(minimum_native_export):
        raise GCFExplainerEmptyCandidateSetError(
            "BACE VRRW produced fewer than 100 model-counterfactual graphs."
        )
    source_graphs = [
        record_to_pyg(record, origin_index=index)
        for index, record in enumerate(selected_sources)
    ]
    modules = import_official_modules(official_root)
    torch, _functional, _loader = _runtime_stack()
    target_device = _device_name(torch, device)
    distance_model = modules["distance"].load_neurosed(
        source_graphs,
        neurosed_model_path=str(neurosed),
        device=target_device,
    )
    raw_distance = distance_model.predict_outer_with_queries(
        counterfactual_graphs, batch_size=1000
    ).cpu()
    util = importlib_import_from_official(official_root, "util")
    original_counts = util.graph_element_counts(source_graphs)
    candidate_counts = util.graph_element_counts(counterfactual_graphs)
    normalization = torch.cartesian_prod(candidate_counts, original_counts).sum(
        dim=1
    ).view(len(counterfactual_graphs), len(source_graphs))
    distances = (raw_distance / normalization).T
    counterfactual_covering = {
        index: set() for index in range(distances.shape[1])
    }
    graphs_covered_by = {index: set() for index in range(distances.shape[0])}
    for parent_index in range(distances.shape[0]):
        for candidate_index in torch.where(
            distances[parent_index] <= float(theta)
        )[0].tolist():
            counterfactual_covering[int(candidate_index)].add(parent_index)
            graphs_covered_by[parent_index].add(int(candidate_index))
    coverings = modules["summary"].greedy_counterfactual_summary_from_covering_sets(
        counterfactual_covering={
            key: set(value) for key, value in counterfactual_covering.items()
        },
        graphs_covered_by={key: set(value) for key, value in graphs_covered_by.items()},
        k=len(counterfactual_covering),
    )
    order = [int(coverings[rank][0]) for rank in sorted(coverings)]
    selected_graphs = [counterfactual_graphs[index] for index in order]
    native_rows = [
        {
            "candidate_id": stable_graph_candidate_id(counterfactual_graphs[index]),
            "native_rank": rank,
            "candidate_index": index,
            "graph_hash": candidate_meta[index]["graph_hash"],
            "frequency": candidate_meta[index]["frequency"],
            "importance_prediction": candidate_meta[index]["importance_prediction"],
            "covered_parent_count_at_rank": int(coverings[rank][1]),
            "selection_method": "official_greedy_coverage_gain",
        }
        for rank, index in enumerate(order, start=1)
    ]
    _torch_save_compat(
        {
            "selected_graphs": selected_graphs,
            "selected_records": native_rows,
            "source_counterfactuals_path": str(counterfactuals_path),
            "selection_policy": "official_greedy_coverage_gain",
        },
        root / "selected_counterfactual_graphs.pt",
    )
    write_jsonl(root / "native_summary_rank.jsonl", native_rows)
    write_csv(
        root / "native_summary_rank.csv",
        native_rows,
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
    manifest = {
        "schema_version": "gcfexplainer_bace_native_summary_v1",
        "dataset": DATASET,
        "profile": str(profile),
        "parent_count": len(source_graphs),
        **lineage,
        "native_candidate_count": len(counterfactual_graphs),
        "native_rank_exported": len(native_rows),
        "theta": float(theta),
        "theta_source": "official_summary_default",
        "selection_method": "official_greedy_coverage_gain",
        "gnn_checkpoint": str(checkpoint),
        "gnn_checkpoint_sha256": sha256_file(checkpoint),
        "neurosed_checkpoint": str(neurosed),
        "neurosed_checkpoint_sha256": sha256_file(neurosed),
        "vrrw_manifest_path": str(vrrw_manifest_path),
        "vrrw_manifest_sha256": sha256_file(vrrw_manifest_path),
        "counterfactuals_path": str(counterfactuals_path),
        "counterfactuals_sha256": sha256_file(counterfactuals_path),
        "native_rank_reordered": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "run_complete": True,
    }
    write_json(root / "resolved_config.json", manifest)
    write_json(root / "run_manifest.json", manifest)
    write_json(root / "audit.json", {"audit_passed": True, **manifest})
    write_json(root / "_RUN_COMPLETE.json", {"run_complete": True})
    return manifest


def _origin_index(graph: Any) -> int | None:
    value = getattr(graph, "gcf_origin_index", None)
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, list):
        return int(value[0]) if value else None
    return int(value)


def export_bace_rf_valid_top20(
    *,
    dataset_dir: str | Path,
    summary_dir: str | Path,
    teacher: Any,
    teacher_path: str | Path,
    output_dir: str | Path,
    profile: str,
    parent_limit: int,
    top_k: int = 20,
) -> dict[str, Any]:
    expected_parent_limit = 64 if profile == "smoke" else EXPECTED_GENERATION_SOURCE_ROWS
    if int(parent_limit) != expected_parent_limit:
        raise ValueError("BACE export parent_limit differs from profile contract.")
    if int(top_k) != 20:
        raise ValueError("BACE GCF export requires top_k=20.")
    root = Path(output_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"BACE export output cannot be overwritten: {root}")
    root.mkdir(parents=True, exist_ok=True)
    schema, _train, _val, generation, _summary = load_bace_gcf_dataset(dataset_dir)
    summary_root = Path(summary_dir).expanduser().resolve()
    summary_manifest = read_json(summary_root / "run_manifest.json")
    if summary_manifest.get("run_complete") is not True:
        raise ValueError("BACE native summary is incomplete.")
    parent_ids = [str(value) for value in summary_manifest["generation_parent_ids"]]
    by_id = {str(record["molecule_id"]): record for record in generation}
    source_records = [by_id[value] for value in parent_ids]
    if len(source_records) != expected_parent_limit:
        raise ValueError("BACE export parent lineage count mismatch.")
    payload = _torch_load_compat(summary_root / "selected_counterfactual_graphs.pt")
    graphs = list(payload.get("selected_graphs", []))
    native_rows = list(payload.get("selected_records", []))
    if len(graphs) != len(native_rows) or not native_rows:
        raise ValueError("BACE native summary graph/rank payload is incomplete.")
    ordered = sorted(
        zip(native_rows, graphs, strict=True),
        key=lambda item: int(item[0]["native_rank"]),
    )
    ranks = [int(row["native_rank"]) for row, _graph in ordered]
    if len(ranks) != len(set(ranks)):
        raise ValueError("BACE native ranks are duplicated.")
    audit_rows: list[dict[str, Any]] = []
    candidate_universe: list[dict[str, Any]] = []
    seen_smiles: set[str] = set()
    for native, graph in ordered:
        audit: dict[str, Any] = {
            "candidate_id": str(native["candidate_id"]),
            "native_rank": int(native["native_rank"]),
            "source_graph_index": _origin_index(graph),
            "decode_ok": False,
            "sanitize_ok": False,
            "canonical_smiles": "",
            "rf_inference_ok": False,
            "rf_pred": None,
            "rf_target_match": False,
            "selected": False,
            "rejection_stage": "",
            "rejection_reason": "",
        }
        origin = audit["source_graph_index"]
        if origin is None or not 0 <= int(origin) < len(source_records):
            audit.update(
                rejection_stage="graph_decode",
                rejection_reason="missing_or_invalid_source_lineage",
            )
            audit_rows.append(audit)
            continue
        decoded = decode_generated_fullgraph(
            graph, source_record=source_records[int(origin)], schema=schema
        )
        if not decoded.decode_ok:
            audit.update(
                rejection_stage="rdkit_sanitize"
                if "sanitize" in str(decoded.failure_reason)
                or "kekul" in str(decoded.failure_reason)
                or "valence" in str(decoded.failure_reason)
                else "graph_decode",
                rejection_reason=str(decoded.failure_reason),
            )
            audit_rows.append(audit)
            continue
        smiles = str(decoded.canonical_smiles).strip()
        audit.update(decode_ok=True, sanitize_ok=True, canonical_smiles=smiles)
        if not smiles or smiles in seen_smiles:
            audit.update(
                rejection_stage="canonicalization",
                rejection_reason="empty_or_duplicate_canonical_smiles",
            )
            audit_rows.append(audit)
            continue
        try:
            prediction, probability0, probability1 = score_teacher_probabilities(
                teacher, smiles
            )
        except Exception as exc:
            audit.update(
                rejection_stage="rf_inference",
                rejection_reason=f"rf_inference_failed:{type(exc).__name__}",
            )
            audit_rows.append(audit)
            continue
        audit.update(
            rf_inference_ok=True,
            rf_pred=int(prediction),
            rf_target_match=int(prediction) == TARGET_LABEL,
        )
        if int(prediction) != TARGET_LABEL:
            audit.update(
                rejection_stage="rf_target_filter",
                rejection_reason="rf_not_target_label_0",
            )
            audit_rows.append(audit)
            continue
        candidate_id = "GCFBACE_" + hashlib.sha256(smiles.encode("utf-8")).hexdigest()[
            :20
        ].upper()
        candidate_universe.append(
            {
                **dict(native),
                "candidate_id": candidate_id,
                "native_candidate_id": str(native["candidate_id"]),
                "smiles": smiles,
                "canonical_smiles": smiles,
                "rdkit_valid": True,
                "rf_pred": int(prediction),
                "rf_prob_0": float(probability0),
                "rf_prob_1": float(probability1),
                "source_method": "official_gcfexplainer_bace",
                "selection_method": "native_gcf_summary_rank_filtered_by_validity",
                "candidate_set_preselected": True,
                "selection_performed_in_eval": False,
                "source_parent_id": str(decoded.source_parent_id),
                "projected_new_edge_count": int(decoded.projected_new_edge_count),
                "retained_edge_count": int(decoded.retained_edge_count),
            }
        )
        seen_smiles.add(smiles)
        audit_rows.append(audit)
    selected = candidate_universe[: int(top_k)]
    selected_ranks = {int(row["native_rank"]) for row in selected}
    for audit in audit_rows:
        if audit["rf_target_match"]:
            if int(audit["native_rank"]) in selected_ranks:
                audit.update(selected=True, rejection_stage="selected")
            else:
                audit.update(
                    rejection_stage="selected",
                    rejection_reason="beyond_requested_top_k",
                )
    write_jsonl(root / "candidate_filter_audit.jsonl", audit_rows)
    write_jsonl(root / "candidate_universe.jsonl", candidate_universe)
    manifest = {
        "schema_version": "gcfexplainer_bace_frozen_top20_v1",
        "dataset": DATASET,
        "profile": str(profile),
        "source_label": SOURCE_LABEL,
        "target_label": TARGET_LABEL,
        "semantic_direction": "active_to_inactive",
        "generation_source_parent_rows": EXPECTED_GENERATION_SOURCE_ROWS,
        "summary_parent_count": len(source_records),
        "generation_parent_ids_sha256": stable_json_sha256(parent_ids),
        "native_rank_input_count": len(native_rows),
        "candidate_filter_audit_rows": len(audit_rows),
        "rf_target_candidate_universe_rows": len(candidate_universe),
        "selected_count": len(selected),
        "requested_top_k": int(top_k),
        "selection_method": "native_gcf_summary_rank_filtered_by_validity",
        "candidate_set_preselected": len(selected) == int(top_k),
        "selection_performed_in_eval": False,
        "rf_reranking_performed": False,
        "wnode_reranking_performed": False,
        "native_rank_reordered": False,
        "teacher_path": str(Path(teacher_path).expanduser().resolve()),
        "teacher_sha256": sha256_file(teacher_path),
        "selected_candidate_order_sha256": stable_json_sha256(
            [str(row["candidate_id"]) for row in selected]
        ),
        "calibration_loaded": False,
        "test_loaded": False,
        "run_complete": len(selected) == int(top_k),
    }
    write_json(root / "resolved_config.json", manifest)
    write_json(root / "run_manifest.json", manifest)
    if len(selected) < int(top_k):
        write_json(root / "failure_summary.json", manifest)
        write_json(root / "_RUN_FAILED.json", manifest)
        raise GCFExplainerEmptyCandidateSetError(
            f"BACE native-rank export yielded {len(selected)} / {top_k} candidates."
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
    write_json(root / "_RUN_COMPLETE.json", {"run_complete": True, **manifest})
    return manifest


__all__ = [
    "build_bace_native_summary",
    "export_bace_rf_valid_top20",
    "run_bace_official_vrrw",
    "train_bace_official_gnn",
]
