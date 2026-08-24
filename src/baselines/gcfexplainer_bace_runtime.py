"""Project-owned orchestration for official GCFExplainer on frozen BACE data."""

from __future__ import annotations

import csv
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import math
import os
import random
import resource
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from rdkit import Chem, rdBase

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
from src.baselines.gcfexplainer_acceleration import (
    BufferedVRRWLogging,
    GCFAccelerationConfig,
    LEGACY_ACCELERATION_MODE,
    LockstepVRRWTrace,
    ORDERED_ACCELERATION_MODE,
    OrderedImportanceAcceleration,
    OrderedNeighbourAcceleration,
    VRRWPhaseProfiler,
    build_vrrw_equivalence_trace,
    scientific_replay_contract_sha256,
    validate_full_acceleration_gate,
)
from src.baselines.bace_gine_native_adapter import (
    BACEFrozenGINENativeGraphAdapter,
)
from src.baselines.bace_gnn_native_candidates import (
    freeze_gine_candidate_universe,
)
from src.baselines.gcfexplainer_mutagenicity_adapter import (
    GCFExplainerEmptyCandidateSetError,
    GCFExplainerMutagenicityCodecError,
    GraphRecordDataset,
    StrictMolecule,
    cohort_hash,
    decode_generated_fullgraph,
    deterministic_balanced_prefix,
    encode_source_graph,
    graph_lineage_neighbor_wrapper,
    import_official_modules,
    read_json,
    record_to_pyg,
    reconstruct_source_graph,
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


def _bace_gnn_checkpoint_identity(path: str | Path) -> tuple[Path, str, str]:
    """Resolve an official checkpoint file or the frozen project GINE bundle."""

    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        model_path = resolved / "model.pt"
        if not model_path.is_file() or model_path.stat().st_size <= 0:
            raise FileNotFoundError(
                f"Frozen BACE GINE bundle has no model.pt: {resolved}"
            )
        return resolved, sha256_file(model_path), "frozen_project_gine_bundle"
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"BACE GNN checkpoint is missing: {resolved}")
    return resolved, sha256_file(resolved), "legacy_official_gnn_file"


def _load_bace_native_gnn(
    *,
    modules: Mapping[str, Any],
    checkpoint: Path,
    checkpoint_kind: str,
    schema: Any,
    source_records: Sequence[Mapping[str, Any]],
    device: str,
    acceleration: GCFAccelerationConfig | None = None,
    diagnostic_trace_enabled: bool = False,
) -> tuple[Any, dict[str, Any]]:
    if checkpoint_kind == "frozen_project_gine_bundle":
        model = BACEFrozenGINENativeGraphAdapter(
            checkpoint,
            source_records=source_records,
            graph_schema=schema,
            device=device,
            acceleration=acceleration,
            diagnostic_trace_enabled=diagnostic_trace_enabled,
        ).eval()
        return model, model.provenance()
    model = _load_official_gnn(
        modules,
        num_features=schema.node_feature_dim,
        checkpoint=checkpoint,
        device=device,
    )
    return model, {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "oracle_backend": "official_gnn",
        "classifier_family": "official_gcn",
        "rf_oracle_used": False,
        "eligible_for_bace_gnn_main_results": False,
    }


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
    acceleration_mode: str = LEGACY_ACCELERATION_MODE,
    gine_batch_size: int = 256,
    graph_cache_capacity: int = 0,
    cpu_neighbor_workers: int = 1,
    progress_every: int = 1000,
    acceleration_gate: str | Path | None = None,
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
    checkpoint, checkpoint_sha256, checkpoint_kind = _bace_gnn_checkpoint_identity(
        gnn_checkpoint
    )
    acceleration = GCFAccelerationConfig(
        mode=str(acceleration_mode),
        gine_batch_size=int(gine_batch_size),
        graph_cache_capacity=int(graph_cache_capacity),
        cpu_neighbor_workers=int(cpu_neighbor_workers),
        progress_every=int(progress_every),
    )
    if (
        acceleration.mode == ORDERED_ACCELERATION_MODE
        and checkpoint_kind != "frozen_project_gine_bundle"
    ):
        raise ValueError("ordered_v2 is supported only by the frozen BACE GINE adapter")
    acceleration_gate_payload: dict[str, Any] | None = None
    acceleration_gate_path: Path | None = None
    if profile == "full" and acceleration.mode == ORDERED_ACCELERATION_MODE:
        if acceleration_gate is None:
            raise ValueError(
                "Full ordered_v2 VRRW requires the 500/1000 equivalence and A/B gate"
            )
        acceleration_gate_path = Path(acceleration_gate).expanduser().resolve(strict=True)
    neurosed = Path(neurosed_checkpoint).expanduser().resolve()
    projection_manifest = Path(neurosed_manifest).expanduser().resolve()
    for path, label in (
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
        "diagnostic_only": str(profile) == "equivalence_quick",
        "dataset_dir": str(Path(dataset_dir).expanduser().resolve()),
        "official_root": str(Path(official_root).expanduser().resolve()),
        "gnn_checkpoint": str(checkpoint),
        "gnn_checkpoint_sha256": checkpoint_sha256,
        "gnn_checkpoint_kind": checkpoint_kind,
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
        "acceleration": {
            **asdict(acceleration),
            "fingerprint": acceleration.fingerprint,
            "full_gate_path": (
                str(acceleration_gate_path) if acceleration_gate_path is not None else None
            ),
            "full_gate_sha256": (
                sha256_file(acceleration_gate_path)
                if acceleration_gate_path is not None
                else None
            ),
            "full_gate_status": (
                acceleration_gate_payload.get("status")
                if acceleration_gate_payload is not None
                else None
            ),
            "mps_enabled": False,
            "random_stream_reordered": False,
            "transition_order_reordered": False,
        },
    }
    if acceleration_gate_path is not None:
        acceleration_gate_payload = validate_full_acceleration_gate(
            acceleration_gate_path,
            config=acceleration,
            scientific_contract_sha256=scientific_replay_contract_sha256(config),
        )
        config["acceleration"]["full_gate_status"] = acceleration_gate_payload.get(
            "status"
        )
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
    model, model_provenance = _load_bace_native_gnn(
        modules=modules,
        checkpoint=checkpoint,
        checkpoint_kind=checkpoint_kind,
        schema=schema,
        source_records=selected,
        device=target_device1,
        acceleration=acceleration,
        diagnostic_trace_enabled=str(profile) == "equivalence_quick",
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
    profiler = VRRWPhaseProfiler(
        vrrw=vrrw,
        importance=importance,
        progress_every=acceleration.progress_every,
    )
    importance_acceleration: OrderedImportanceAcceleration | None = None
    lockstep_trace: LockstepVRRWTrace | None = None
    random_walk_started: float | None = None
    random_walk_finished: float | None = None
    try:
        os.chdir(runtime_root)
        prepare_started = time.perf_counter()
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
        prepare_seconds = time.perf_counter() - prepare_started
        vrrw.importance_args = importance_args
        print("[GCFEXPLAINER_OFFICIAL_COMPAT_PATCH]", flush=True)
        print(f"patch={VRRW_ALPHA_ENDPOINT_PATCH}", flush=True)
        print("dataset=bace", flush=True)
        print(f"alpha={float(alpha)}", flush=True)
        print("official_source_modified=false", flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(target_device1)
        with ExitStack() as stack:
            if acceleration.cpu_neighbor_workers > 1:
                stack.enter_context(
                    OrderedNeighbourAcceleration(
                        vrrw, workers=acceleration.cpu_neighbor_workers
                    )
                )
            if acceleration.mode == ORDERED_ACCELERATION_MODE:
                stack.enter_context(BufferedVRRWLogging(vrrw))
                importance_acceleration = stack.enter_context(
                    OrderedImportanceAcceleration(
                        importance, capacity=acceleration.graph_cache_capacity
                    )
                )
            stack.enter_context(profiler)
            if str(profile) == "equivalence_quick":
                lockstep_trace = stack.enter_context(
                    LockstepVRRWTrace(
                        vrrw=vrrw,
                        importance=importance,
                        torch=torch,
                        np=np,
                        budget=int(m),
                    )
                )
            stack.enter_context(_official_vrrw_alpha_endpoint_patch(vrrw))
            random_walk_started = time.perf_counter()
            vrrw.counterfactual_summary_with_randomwalk(
                    input_graphs=graphs,
                    importance_args=importance_args,
                    teleport_probability=float(teleport),
                    max_steps=int(m),
                )
            random_walk_finished = time.perf_counter()
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
    equivalence_trace = build_vrrw_equivalence_trace(
        payload=payload,
        torch=torch,
        np=np,
        budget=int(m),
    )
    write_json(root / "equivalence_trace.json", equivalence_trace)
    if lockstep_trace is not None:
        write_json(root / "lockstep_trace.json", lockstep_trace.payload())
    graph_map = dict(payload.get("graph_map", {}))
    candidates = list(payload.get("counterfactual_candidates", []))
    gpu_total_bytes = 0
    peak_allocated_bytes = 0
    peak_reserved_bytes = 0
    if torch.cuda.is_available():
        device_object = torch.device(target_device1)
        device_index = (
            torch.cuda.current_device()
            if device_object.index is None
            else int(device_object.index)
        )
        gpu_total_bytes = int(torch.cuda.get_device_properties(device_index).total_memory)
        peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device_index))
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device_index))
    phase_profile = profiler.report()
    phase_profile.update(
        {
            "schema_version": 1,
            "prepare_importance_seconds": prepare_seconds,
            "random_walk_seconds": (
                random_walk_finished - random_walk_started
                if random_walk_started is not None and random_walk_finished is not None
                else None
            ),
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "gpu_uuid": os.environ.get("AUTODL_PHYSICAL_GPU_UUID"),
            "gpu_index": os.environ.get("AUTODL_PHYSICAL_GPU_INDEX"),
            "gpu_total_bytes": gpu_total_bytes,
            "peak_gpu_allocated_bytes": peak_allocated_bytes,
            "peak_gpu_reserved_bytes": peak_reserved_bytes,
            "peak_vram_fraction": (
                peak_reserved_bytes / gpu_total_bytes if gpu_total_bytes else 0.0
            ),
            "adapter": (
                model.provenance().get("acceleration", {})
                if hasattr(model, "provenance")
                else {}
            ),
            "importance_cache": (
                importance_acceleration.report()
                if importance_acceleration is not None
                else {
                    "calls": 0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "cache_entries": 0,
                }
            ),
        }
    )
    write_json(root / "performance_profile.json", phase_profile)
    manifest = {
        **config,
        "oracle_backend": model_provenance.get("oracle_backend"),
        "classifier_family": model_provenance.get("classifier_family"),
        "rf_oracle_used": False,
        "oracle_checkpoint_hash": model_provenance.get(
            "oracle_checkpoint_hash", checkpoint_sha256
        ),
        "model_provenance": model.provenance()
        if hasattr(model, "provenance")
        else model_provenance,
        "eligible_for_bace_gnn_main_results": (
            checkpoint_kind == "frozen_project_gine_bundle"
            and str(profile) != "equivalence_quick"
        ),
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
        "equivalence_trace_path": str(root / "equivalence_trace.json"),
        "equivalence_trace_sha256": sha256_file(root / "equivalence_trace.json"),
        "lockstep_trace_path": (
            str(root / "lockstep_trace.json") if lockstep_trace is not None else None
        ),
        "lockstep_trace_sha256": (
            sha256_file(root / "lockstep_trace.json")
            if lockstep_trace is not None
            else None
        ),
        "performance_profile_path": str(root / "performance_profile.json"),
        "performance_profile_sha256": sha256_file(root / "performance_profile.json"),
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


def official_greedy_coverage_order(
    counterfactual_covering: Mapping[int, set[int]],
    graphs_covered_by: Mapping[int, set[int]],
) -> list[tuple[int, int]]:
    """Return the official greedy order with an exact zero-gain fast path.

    The upstream implementation repeatedly selects the first insertion-ordered
    candidate having maximum uncovered-parent count.  Once that maximum is
    zero, every remaining set is empty and upstream deterministically emits the
    remaining dictionary keys in insertion order.  Fast-forwarding only that
    zero-gain tail preserves the exact upstream result while avoiding a
    quadratic scan over tens of thousands of saved VRRW candidates.
    """

    pending = {
        int(candidate): set(parents)
        for candidate, parents in counterfactual_covering.items()
    }
    covered: set[int] = set()
    order: list[tuple[int, int]] = []
    while pending:
        candidate, newly_covered = max(
            pending.items(), key=lambda item: len(item[1])
        )
        if not newly_covered:
            order.extend((int(value), len(covered)) for value in pending)
            break
        covered.update(newly_covered)
        pending.pop(candidate)
        for parent in newly_covered:
            for other in graphs_covered_by[int(parent)] - {candidate}:
                if other in pending:
                    pending[other].discard(int(parent))
        order.append((int(candidate), len(covered)))
    return order


def _model_counterfactual_candidates(
    candidates: Sequence[Mapping[str, Any]],
    graph_map: Mapping[Any, Any],
    *,
    limit: int | None,
) -> tuple[list[Any], list[dict[str, Any]], int]:
    if limit is not None and int(limit) < 0:
        raise ValueError("BACE native candidate limit cannot be negative.")
    graphs: list[Any] = []
    metadata: list[dict[str, Any]] = []
    available = 0
    for source_index, candidate in enumerate(candidates):
        parts = candidate.get("importance_parts", [0.0])
        prediction_importance = float(parts[0])
        graph_hash = candidate.get("graph_hash")
        if prediction_importance < 0.5 or graph_hash not in graph_map:
            continue
        available += 1
        if limit is not None and len(graphs) >= int(limit):
            continue
        graph = graph_map[graph_hash]
        graphs.append(graph)
        metadata.append(
            {
                "graph_hash": str(graph_hash),
                "frequency": int(candidate.get("frequency", 0)),
                "importance_prediction": prediction_importance,
                "source_candidate_index": int(source_index),
                "candidate_id": stable_graph_candidate_id(graph),
            }
        )
    return graphs, metadata, available


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
    native_candidate_limit: int | None = None,
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
    checkpoint, checkpoint_sha256, checkpoint_kind = _bace_gnn_checkpoint_identity(
        gnn_checkpoint
    )
    neurosed = Path(neurosed_checkpoint).expanduser().resolve()
    if checkpoint_sha256 != str(vrrw_manifest.get("gnn_checkpoint_sha256")):
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
    legacy_target_count = max(len(selected_sources), int(minimum_native_export))
    if native_candidate_limit is None:
        effective_limit: int | None = legacy_target_count
    elif int(native_candidate_limit) == 0:
        effective_limit = None
    elif int(native_candidate_limit) < int(minimum_native_export):
        raise ValueError(
            "BACE native candidate limit must be zero (all) or at least "
            f"minimum_native_export={minimum_native_export}."
        )
    else:
        effective_limit = int(native_candidate_limit)
    counterfactual_graphs, candidate_meta, available_model_counterfactuals = (
        _model_counterfactual_candidates(
            candidates,
            graph_map,
            limit=effective_limit,
        )
    )
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
    ranked = official_greedy_coverage_order(
        counterfactual_covering=counterfactual_covering,
        graphs_covered_by=graphs_covered_by,
    )
    order = [index for index, _covered in ranked]
    native_rows = [
        {
            "candidate_id": candidate_meta[index]["candidate_id"],
            "native_rank": rank,
            "candidate_index": index,
            "graph_hash": candidate_meta[index]["graph_hash"],
            "frequency": candidate_meta[index]["frequency"],
            "importance_prediction": candidate_meta[index]["importance_prediction"],
            "covered_parent_count_at_rank": int(covered_count),
            "selection_method": "official_greedy_coverage_gain",
        }
        for rank, (index, covered_count) in enumerate(ranked, start=1)
    ]
    inline_graphs = len(order) <= 1000
    _torch_save_compat(
        {
            "selected_graphs": (
                [counterfactual_graphs[index] for index in order]
                if inline_graphs
                else []
            ),
            "selected_graph_hashes": [
                candidate_meta[index]["graph_hash"] for index in order
            ],
            "selected_records": native_rows,
            "source_counterfactuals_path": str(counterfactuals_path),
            "source_counterfactuals_sha256": sha256_file(counterfactuals_path),
            "graph_storage_mode": (
                "inline_graphs"
                if inline_graphs
                else "vrrw_graph_hash_reference"
            ),
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
        "schema_version": "gcfexplainer_bace_native_summary_v2",
        "dataset": DATASET,
        "profile": str(profile),
        "parent_count": len(source_graphs),
        **lineage,
        "native_candidate_count": len(counterfactual_graphs),
        "available_model_counterfactual_count": available_model_counterfactuals,
        "native_candidate_limit": (
            0 if effective_limit is None else int(effective_limit)
        ),
        "native_candidate_pool_exhaustive": effective_limit is None
        or len(counterfactual_graphs) == available_model_counterfactuals,
        "legacy_native_candidate_limit": legacy_target_count,
        "native_rank_exported": len(native_rows),
        "theta": float(theta),
        "theta_source": "official_summary_default",
        "selection_method": "official_greedy_coverage_gain",
        "greedy_implementation": "official_semantics_zero_gain_tail_fast_forward",
        "official_greedy_semantics_preserved": True,
        "ranked_graph_storage_mode": (
            "inline_graphs" if inline_graphs else "vrrw_graph_hash_reference"
        ),
        "gnn_checkpoint": str(checkpoint),
        "gnn_checkpoint_sha256": checkpoint_sha256,
        "gnn_checkpoint_kind": checkpoint_kind,
        "oracle_backend": vrrw_manifest.get("oracle_backend"),
        "classifier_family": vrrw_manifest.get("classifier_family"),
        "rf_oracle_used": False,
        "oracle_checkpoint_hash": vrrw_manifest.get("oracle_checkpoint_hash"),
        "eligible_for_bace_gnn_main_results": bool(
            vrrw_manifest.get("eligible_for_bace_gnn_main_results")
        ),
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


def _load_ranked_summary_graphs(
    summary_root: Path,
    summary_manifest: Mapping[str, Any],
) -> list[tuple[dict[str, Any], Any]]:
    payload = _torch_load_compat(summary_root / "selected_counterfactual_graphs.pt")
    native_rows = [dict(value) for value in payload.get("selected_records", [])]
    if not native_rows:
        raise ValueError("BACE native summary rank payload is empty.")
    ranks = [int(row["native_rank"]) for row in native_rows]
    if ranks != list(range(1, len(native_rows) + 1)):
        raise ValueError(
            "BACE native summary rows are not in their contiguous native-rank order."
        )
    # ``candidate_id`` is a structural identity, not the ranked-row key.  The
    # VRRW pool can contain the same graph through distinct native records, so
    # native rank is the unique row identity and canonical dedup belongs to the
    # sequential export audit below.

    inline = list(payload.get("selected_graphs", []))
    if inline:
        if len(inline) != len(native_rows):
            raise ValueError("BACE inline summary graph/rank counts differ.")
        return list(zip(native_rows, inline, strict=True))

    graph_hashes = [str(value) for value in payload.get("selected_graph_hashes", [])]
    if len(graph_hashes) != len(native_rows):
        raise ValueError("BACE referenced summary graph/rank counts differ.")
    source_path = Path(
        str(payload.get("source_counterfactuals_path") or "")
    ).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"BACE referenced VRRW payload is missing: {source_path}")
    expected_path = Path(
        str(summary_manifest.get("counterfactuals_path") or source_path)
    ).expanduser().resolve()
    if source_path != expected_path:
        raise ValueError("BACE summary graph reference path differs from its manifest.")
    expected_sha = str(
        payload.get("source_counterfactuals_sha256")
        or summary_manifest.get("counterfactuals_sha256")
        or ""
    )
    if len(expected_sha) != 64 or sha256_file(source_path) != expected_sha:
        raise ValueError("BACE referenced VRRW payload SHA256 mismatch.")
    source_payload = _torch_load_compat(source_path)
    graph_map = {
        str(key): graph for key, graph in dict(source_payload.get("graph_map", {})).items()
    }
    missing = [value for value in graph_hashes if value not in graph_map]
    if missing:
        raise ValueError(
            "BACE native summary references missing VRRW graphs: "
            f"{missing[:5]}"
        )
    return [
        (row, graph_map[graph_hash])
        for row, graph_hash in zip(native_rows, graph_hashes, strict=True)
    ]


def _audit_bace_ranked_candidates(
    *,
    ranked: Sequence[tuple[Mapping[str, Any], Any]],
    source_records: Sequence[Mapping[str, Any]],
    schema: Any,
    teacher: Any,
    target_k: int,
    scan_limit: int,
    scan_all: bool = False,
    require_connected: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if int(target_k) <= 0:
        raise ValueError("BACE target_k must be positive.")
    if int(scan_limit) < 0:
        raise ValueError("BACE scan_limit cannot be negative.")
    maximum = len(ranked) if int(scan_limit) == 0 else min(len(ranked), int(scan_limit))
    audit_rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    seen_smiles: dict[str, str] = {}

    for native_value, graph in ranked[:maximum]:
        native = dict(native_value)
        graph_hash = str(native.get("graph_hash") or "")
        audit: dict[str, Any] = {
            "candidate_id": str(native["candidate_id"]),
            "native_rank": int(native["native_rank"]),
            "native_score": int(native.get("covered_parent_count_at_rank", 0)),
            "frequency": int(native.get("frequency", 0)),
            "graph_hash": graph_hash,
            "source_graph_index": _origin_index(graph),
            "decode_ok": False,
            "parse_ok": False,
            "sanitize_ok": False,
            "connected": False,
            "canonical_smiles": "",
            "duplicate_of": "",
            "rf_inference_ok": False,
            "teacher_prediction": None,
            "teacher_probability_0": None,
            "teacher_probability_1": None,
            "counterfactual_ok": False,
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
        with rdBase.BlockLogs():
            decoded = decode_generated_fullgraph(
                graph,
                source_record=source_records[int(origin)],
                schema=schema,
            )
        if not decoded.decode_ok:
            reason = str(decoded.failure_reason)
            audit.update(
                rejection_stage=(
                    "rdkit_sanitize"
                    if any(token in reason for token in ("sanitize", "kekul", "valence", "aromaticity"))
                    else "graph_decode"
                ),
                rejection_reason=reason,
            )
            audit_rows.append(audit)
            continue
        smiles = str(decoded.canonical_smiles).strip()
        audit.update(
            decode_ok=True,
            parse_ok=True,
            sanitize_ok=True,
            canonical_smiles=smiles,
        )
        if not smiles:
            audit.update(
                rejection_stage="canonicalization",
                rejection_reason="empty_canonical_smiles",
            )
            audit_rows.append(audit)
            continue
        with rdBase.BlockLogs():
            decoded_molecule = Chem.MolFromSmiles(smiles)
        connected = bool(
            decoded_molecule is not None
            and len(Chem.GetMolFrags(decoded_molecule)) == 1
            and "." not in smiles
        )
        audit["connected"] = connected
        if bool(require_connected) and not connected:
            audit.update(
                rejection_stage="connectivity",
                rejection_reason="disconnected_fullgraph_candidate",
            )
            audit_rows.append(audit)
            continue
        if smiles in seen_smiles:
            audit.update(
                duplicate_of=seen_smiles[smiles],
                rejection_stage="canonicalization",
                rejection_reason="duplicate_canonical_smiles",
            )
            audit_rows.append(audit)
            continue
        seen_smiles[smiles] = (
            f"{native['candidate_id']}@native_rank={int(native['native_rank'])}"
        )
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
        counterfactual_ok = int(prediction) == TARGET_LABEL
        audit.update(
            rf_inference_ok=True,
            teacher_prediction=int(prediction),
            teacher_probability_0=float(probability0),
            teacher_probability_1=float(probability1),
            counterfactual_ok=counterfactual_ok,
        )
        if not counterfactual_ok:
            audit.update(
                rejection_stage="rf_target_filter",
                rejection_reason="rf_not_target_label_0",
            )
            audit_rows.append(audit)
            continue
        candidate_id = "GCFBACE_" + hashlib.sha256(
            smiles.encode("utf-8")
        ).hexdigest()[:20].upper()
        if len(selected) < int(target_k):
            selected.append(
                {
                **native,
                "candidate_id": candidate_id,
                "native_candidate_id": str(native["candidate_id"]),
                "smiles": smiles,
                "canonical_smiles": smiles,
                "rdkit_valid": True,
                "connected": connected,
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
            audit.update(selected=True, rejection_stage="selected")
        else:
            audit.update(
                selected=False,
                rejection_stage="eligible_after_target",
                rejection_reason="valid_counterfactual_after_frozen_top_k",
            )
        audit_rows.append(audit)
        if len(selected) >= int(target_k) and not bool(scan_all):
            break

    reasons = Counter(
        str(row["rejection_reason"])
        for row in audit_rows
        if str(row["rejection_reason"])
    )
    graph_hashes = [str(native.get("graph_hash") or "") for native, _graph in ranked]
    native_candidate_ids = [
        str(native.get("candidate_id") or "") for native, _graph in ranked
    ]
    selected_candidate_ids = [str(row["candidate_id"]) for row in selected]
    selected_candidate_ids_unique = len(selected_candidate_ids) == len(
        set(selected_candidate_ids)
    )
    if not selected_candidate_ids_unique:
        raise ValueError("BACE selected candidate IDs are not unique.")
    attrition = {
        "num_raw_native_records": len(ranked),
        "num_unique_graph_hashes": len(set(graph_hashes)),
        "num_unique_native_candidate_ids": len(set(native_candidate_ids)),
        "num_duplicate_native_candidate_ids": len(native_candidate_ids)
        - len(set(native_candidate_ids)),
        "num_ranked_candidates": len(ranked),
        "num_scanned_candidates": len(audit_rows),
        "num_decode_success": sum(bool(row["decode_ok"]) for row in audit_rows),
        "num_decode_failed": sum(not bool(row["decode_ok"]) for row in audit_rows),
        "num_rdkit_parse_success": sum(bool(row["parse_ok"]) for row in audit_rows),
        "num_sanitize_success": sum(bool(row["sanitize_ok"]) for row in audit_rows),
        "num_sanitize_failed": sum(
            row["rejection_stage"] == "rdkit_sanitize" for row in audit_rows
        ),
        "num_connected_candidates": sum(bool(row["connected"]) for row in audit_rows),
        "num_unique_connected_candidates": len(
            {
                str(row["canonical_smiles"])
                for row in audit_rows
                if bool(row["connected"]) and str(row["canonical_smiles"])
            }
        ),
        "num_canonical_unique": len(seen_smiles),
        "num_teacher_evaluable": sum(
            bool(row["rf_inference_ok"]) for row in audit_rows
        ),
        "num_teacher_counterfactual": sum(
            bool(row["counterfactual_ok"]) for row in audit_rows
        ),
        "num_retained": len(selected),
        "requested_k": int(target_k),
        "scan_limit": int(scan_limit),
        "scan_all": bool(scan_all),
        "require_connected": bool(require_connected),
        "max_scanned_rank": (
            int(audit_rows[-1]["native_rank"]) if audit_rows else 0
        ),
        "scan_exhausted": len(audit_rows) == len(ranked),
        "scan_limit_exhausted": (
            int(scan_limit) > 0
            and len(audit_rows) == maximum
            and len(selected) < int(target_k)
        ),
        "target_reached": len(selected) == int(target_k),
        "selected_candidate_ids_unique": selected_candidate_ids_unique,
        "failure_reason_counts": dict(sorted(reasons.items())),
        "native_order_preserved": [
            int(row["native_rank"]) for row in audit_rows
        ] == sorted(int(row["native_rank"]) for row in audit_rows),
        "candidate_repair_performed": False,
        "candidate_copy_performed": False,
        "rank_backfill_performed": False,
        "rf_reranking_performed": False,
        "wnode_reranking_performed": False,
        "test_used_for_selection": False,
    }
    return audit_rows, selected, attrition


def _write_candidate_attrition_artifacts(
    root: Path,
    audit_rows: Sequence[Mapping[str, Any]],
    attrition: Mapping[str, Any],
) -> None:
    write_json(root / "candidate_attrition_audit.json", attrition)
    write_jsonl(root / "candidate_filter_audit.jsonl", audit_rows)
    write_csv(
        root / "candidate_attrition_rows.csv",
        audit_rows,
        (
            "candidate_id",
            "native_rank",
            "native_score",
            "frequency",
            "graph_hash",
            "source_graph_index",
            "decode_ok",
            "parse_ok",
            "sanitize_ok",
            "connected",
            "canonical_smiles",
            "duplicate_of",
            "rf_inference_ok",
            "teacher_prediction",
            "teacher_probability_0",
            "teacher_probability_1",
            "counterfactual_ok",
            "selected",
            "rejection_stage",
            "rejection_reason",
        ),
    )
    lines = [f"{key}={value}" for key, value in sorted(attrition.items())]
    destination = root / "candidate_attrition_report.txt"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def audit_bace_vrrw_candidate_sufficiency(
    *,
    dataset_dir: str | Path,
    vrrw_dir: str | Path,
    teacher: Any,
    teacher_path: str | Path,
    output_dir: str | Path,
    profile: str,
    parent_limit: int,
    target_k: int = 20,
    scan_limit: int = 0,
    scan_all: bool = False,
    require_connected: bool = False,
) -> dict[str, Any]:
    """Prove candidate-pool sufficiency without changing final native ranking.

    Rows are inspected in the immutable VRRW frequency order.  This audit can
    establish that the completed artifact contains at least ``target_k`` valid
    RF counterfactual graphs before the more expensive NeuroSED/greedy summary
    is rebuilt.  Its ranks are explicitly audit-only and are never exported as
    the final candidate sequence.
    """

    expected_parent_limit = 64 if profile == "smoke" else EXPECTED_GENERATION_SOURCE_ROWS
    if int(parent_limit) != expected_parent_limit:
        raise ValueError("BACE VRRW audit parent_limit differs from profile contract.")
    root = Path(output_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"BACE VRRW audit output cannot be overwritten: {root}")
    root.mkdir(parents=True, exist_ok=True)
    schema, _train, _val, generation, _summary = load_bace_gcf_dataset(dataset_dir)
    vrrw_root = Path(vrrw_dir).expanduser().resolve()
    manifest_path = vrrw_root / "run_manifest.json"
    manifest = read_json(manifest_path)
    source_records, lineage = _resolve_parent_lineage(generation, manifest, profile)
    counterfactuals_path = (vrrw_root / "counterfactuals.pt").resolve()
    expected_sha = str(manifest.get("counterfactuals_sha256") or "")
    if len(expected_sha) != 64 or sha256_file(counterfactuals_path) != expected_sha:
        raise ValueError("BACE VRRW audit artifact SHA256 mismatch.")
    payload = _torch_load_compat(counterfactuals_path)
    candidates = list(payload.get("counterfactual_candidates", []))
    graph_map = dict(payload.get("graph_map", {}))
    if len(candidates) != int(manifest.get("counterfactual_candidate_count", -1)):
        raise ValueError("BACE VRRW audit candidate count differs from its manifest.")
    graphs, metadata, available = _model_counterfactual_candidates(
        candidates,
        graph_map,
        limit=None,
    )
    audit_ranked = [
        (
            {
                "candidate_id": item["candidate_id"],
                "native_rank": rank,
                "candidate_index": item["source_candidate_index"],
                "graph_hash": item["graph_hash"],
                "frequency": item["frequency"],
                "importance_prediction": item["importance_prediction"],
                "covered_parent_count_at_rank": 0,
                "selection_method": "vrrw_frequency_model_cf_sufficiency_audit",
            },
            graph,
        )
        for rank, (item, graph) in enumerate(zip(metadata, graphs, strict=True), start=1)
    ]
    audit_rows, selected, attrition = _audit_bace_ranked_candidates(
        ranked=audit_ranked,
        source_records=source_records,
        schema=schema,
        teacher=teacher,
        target_k=int(target_k),
        scan_limit=int(scan_limit),
        scan_all=bool(scan_all),
        require_connected=bool(require_connected),
    )
    audit_summary = {
        "schema_version": "gcfexplainer_bace_vrrw_candidate_sufficiency_audit_v1",
        "dataset": DATASET,
        "profile": str(profile),
        **lineage,
        "vrrw_raw_candidate_count": len(candidates),
        "vrrw_graph_map_count": len(graph_map),
        "vrrw_model_counterfactual_count": available,
        "counterfactuals_path": str(counterfactuals_path),
        "counterfactuals_sha256": expected_sha,
        "vrrw_manifest_path": str(manifest_path),
        "vrrw_manifest_sha256": sha256_file(manifest_path),
        "teacher_path": str(Path(teacher_path).expanduser().resolve()),
        "teacher_sha256": sha256_file(teacher_path),
        "rank_semantics": "audit_only_vrrw_frequency_not_final_greedy_rank",
        "final_selection_performed": False,
        "candidate_pool_sufficient": len(selected) == int(target_k),
        "candidate_repair_performed": False,
        "test_used_for_selection": False,
        "calibration_loaded": False,
        "test_loaded": False,
        **attrition,
        "candidate_attrition": dict(attrition),
    }
    _write_candidate_attrition_artifacts(root, audit_rows, audit_summary)
    write_json(root / "candidate_attrition_audit.json", audit_summary)
    if len(selected) < int(target_k):
        write_json(root / "_AUDIT_FAILED.json", audit_summary)
        raise GCFExplainerEmptyCandidateSetError(
            "INSUFFICIENT_VALID_NATIVE_CANDIDATES: completed BACE VRRW pool "
            f"contains only {len(selected)} / {target_k} audit-valid candidates."
        )
    write_json(root / "_AUDIT_COMPLETE.json", {"audit_passed": True, **audit_summary})
    return audit_summary


def _external_roundtrip_records(
    path: str | Path,
    *,
    expected_split: str,
    sample_limit: int,
) -> list[StrictMolecule]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = sorted(rows, key=lambda row: str(row.get("molecule_id") or ""))[
        : int(sample_limit)
    ]
    result: list[StrictMolecule] = []
    for index, row in enumerate(selected):
        split = str(row.get("split") or "").strip().lower()
        if split != expected_split:
            raise ValueError(f"BACE round-trip split mismatch in {source}: {split}")
        smiles = str(row.get("smiles") or "").strip()
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"BACE round-trip source SMILES is invalid: {smiles!r}")
        Chem.SanitizeMol(molecule)
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        label = int(float(str(row.get("label"))))
        result.append(
            StrictMolecule(
                molecule_id=str(row["molecule_id"]),
                smiles=smiles,
                canonical_smiles=canonical,
                label=label,
                split=split,
                semantic_label=str(row.get("semantic_label") or ""),
                source_row_index=index,
                source_path=str(source),
            )
        )
    return result


def audit_bace_source_roundtrip(
    *,
    dataset_dir: str | Path,
    teacher: Any,
    output_dir: str | Path,
    calibration_source_csv: str | Path | None = None,
    test_source_csv: str | Path | None = None,
    external_sample_limit: int = 16,
) -> dict[str, Any]:
    schema, train, val, generation, _summary = load_bace_gcf_dataset(dataset_dir)
    groups: dict[str, Sequence[Mapping[str, Any]]] = {
        "train": train,
        "validation": val,
        "generation": generation,
    }
    for name, path, split in (
        ("calibration", calibration_source_csv, "calibration"),
        ("test", test_source_csv, "test"),
    ):
        if path is None:
            continue
        records = _external_roundtrip_records(
            path,
            expected_split=split,
            sample_limit=int(external_sample_limit),
        )
        groups[name] = [encode_source_graph(record, schema) for record in records]

    rows: list[dict[str, Any]] = []
    split_summary: dict[str, Any] = {}
    for split_name, records in groups.items():
        split_failed = 0
        for record in records:
            row: dict[str, Any] = {
                "split": split_name,
                "molecule_id": str(record["molecule_id"]),
                "canonical_smiles": str(record["canonical_smiles"]),
            }
            try:
                molecule, checks = reconstruct_source_graph(record, schema)
                reconstructed = Chem.MolToSmiles(
                    Chem.RemoveHs(molecule, sanitize=True),
                    canonical=True,
                    isomericSmiles=True,
                )
                prediction, _probability0, _probability1 = score_teacher_probabilities(
                    teacher, reconstructed
                )
                row.update(
                    checks,
                    reconstructed_canonical_smiles=reconstructed,
                    teacher_prediction=int(prediction),
                    teacher_prediction_exact=int(prediction) == int(record["label"]),
                    failure_reason="",
                )
                row["round_trip_passed"] = bool(
                    row["round_trip_passed"] and row["teacher_prediction_exact"]
                )
            except Exception as exc:
                row.update(
                    round_trip_passed=False,
                    teacher_prediction_exact=False,
                    failure_reason=f"{type(exc).__name__}:{exc}",
                )
            if row["round_trip_passed"] is not True:
                split_failed += 1
            rows.append(row)
        split_summary[split_name] = {
            "rows": len(records),
            "passed": len(records) - split_failed,
            "failed": split_failed,
        }
    summary = {
        "schema_version": "gcfexplainer_bace_source_roundtrip_audit_v1",
        "dataset": DATASET,
        "splits": split_summary,
        "round_trip_passed": all(value["failed"] == 0 for value in split_summary.values()),
        "mapping_bug_found": any(value["failed"] > 0 for value in split_summary.values()),
        "selection_performed": False,
        "test_used_for_selection": False,
    }
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "source_roundtrip_audit.json", summary)
    fieldnames = tuple(dict.fromkeys(key for row in rows for key in row))
    write_csv(root / "source_roundtrip_rows.csv", rows, fieldnames)
    if not summary["round_trip_passed"]:
        raise GCFExplainerMutagenicityCodecError(
            "BACE source graph round-trip audit failed."
        )
    return summary


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
    scan_limit: int = 0,
    scan_all: bool = False,
    require_connected: bool = False,
    validate_only: bool = False,
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
    ordered = _load_ranked_summary_graphs(summary_root, summary_manifest)
    audit_rows, selected, attrition = _audit_bace_ranked_candidates(
        ranked=ordered,
        source_records=source_records,
        schema=schema,
        teacher=teacher,
        target_k=int(top_k),
        scan_limit=int(scan_limit),
        scan_all=bool(scan_all),
        require_connected=bool(require_connected),
    )
    _write_candidate_attrition_artifacts(root, audit_rows, attrition)
    write_jsonl(root / "candidate_universe.jsonl", selected)
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
        "native_rank_input_count": len(ordered),
        "candidate_filter_audit_rows": len(audit_rows),
        "rf_target_candidate_universe_rows": len(selected),
        "selected_count": len(selected),
        "requested_top_k": int(top_k),
        "scan_limit": int(scan_limit),
        "scan_all": bool(scan_all),
        "require_connected": bool(require_connected),
        "max_scanned_rank": int(attrition["max_scanned_rank"]),
        "scan_exhausted": bool(attrition["scan_exhausted"]),
        "candidate_attrition": dict(attrition),
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
        "run_complete": len(selected) == int(top_k) and not bool(validate_only),
        "validation_passed": len(selected) == int(top_k),
        "validate_only": bool(validate_only),
    }
    write_json(root / "resolved_config.json", manifest)
    write_json(root / "run_manifest.json", manifest)
    if len(selected) < int(top_k):
        write_json(root / "failure_summary.json", manifest)
        write_json(root / "_RUN_FAILED.json", manifest)
        raise GCFExplainerEmptyCandidateSetError(
            "INSUFFICIENT_VALID_NATIVE_CANDIDATES: BACE native-rank export "
            f"yielded {len(selected)} / {top_k} candidates after scanning "
            f"{len(audit_rows)} / {len(ordered)} ranks."
        )
    if validate_only:
        write_json(root / "validation_summary.json", manifest)
        write_json(root / "_VALIDATION_COMPLETE.json", {"validation_passed": True})
        return manifest
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


def export_bace_gine_candidate_universe(
    *,
    dataset_dir: str | Path,
    summary_dir: str | Path,
    gnn_checkpoint: str | Path,
    output_dir: str | Path,
    profile: str,
    parent_limit: int,
    minimum_candidates: int = 20,
    scan_limit: int = 0,
    device: str = "cuda:0",
    oracle_batch_size: int = 256,
) -> dict[str, Any]:
    """Decode native GCF ranks and freeze a GINE-clean train universe.

    Unlike the historical RF exporter this function does not freeze the first
    twenty candidates as the final rule set.  It exports every valid strict
    target graph in official native order; calibration performs all ordering.
    """

    expected_parent_limit = (
        64 if profile == "smoke" else EXPECTED_GENERATION_SOURCE_ROWS
    )
    if int(parent_limit) != expected_parent_limit:
        raise ValueError("BACE GINE export parent_limit differs from profile contract.")
    if int(scan_limit) < 0:
        raise ValueError("BACE GINE export scan_limit cannot be negative.")
    schema, _train, _val, generation, _summary = load_bace_gcf_dataset(dataset_dir)
    summary_root = Path(summary_dir).expanduser().resolve(strict=True)
    summary_manifest_path = summary_root / "run_manifest.json"
    summary_manifest = read_json(summary_manifest_path)
    if summary_manifest.get("run_complete") is not True:
        raise ValueError("BACE native GCF summary is incomplete.")
    if summary_manifest.get("eligible_for_bace_gnn_main_results") is not True:
        raise ValueError("BACE native GCF summary is not frozen-GINE eligible.")
    parent_ids = [str(value) for value in summary_manifest["generation_parent_ids"]]
    by_id = {str(record["molecule_id"]): record for record in generation}
    try:
        source_records = [by_id[value] for value in parent_ids]
    except KeyError as exc:
        raise ValueError("BACE GCF summary references an unknown train parent.") from exc
    if len(source_records) != int(parent_limit):
        raise ValueError("BACE GCF GINE export parent lineage count mismatch.")
    ordered = _load_ranked_summary_graphs(summary_root, summary_manifest)
    if int(scan_limit):
        ordered = ordered[: int(scan_limit)]
    decoded_rows: list[dict[str, Any]] = []
    for native_value, graph in ordered:
        native = dict(native_value)
        origin = _origin_index(graph)
        if origin is None or not 0 <= int(origin) < len(source_records):
            decoded_rows.append(
                {
                    **native,
                    "decode_ok": False,
                    "canonical_smiles": "",
                    "decode_reason": "missing_or_invalid_source_lineage",
                }
            )
            continue
        with rdBase.BlockLogs():
            decoded = decode_generated_fullgraph(
                graph,
                source_record=source_records[int(origin)],
                schema=schema,
            )
        decoded_rows.append(
            {
                **native,
                "decode_ok": bool(decoded.decode_ok),
                "canonical_smiles": str(decoded.canonical_smiles or ""),
                "decode_reason": decoded.failure_reason,
                "source_parent_id": decoded.source_parent_id,
                "projected_new_edge_count": int(decoded.projected_new_edge_count),
                "retained_edge_count": int(decoded.retained_edge_count),
                "removed_source_edge_count": int(decoded.removed_source_edge_count),
            }
        )
    return freeze_gine_candidate_universe(
        method="gcfexplainer",
        decoded_candidates=decoded_rows,
        source_manifest=summary_manifest,
        source_manifest_path=summary_manifest_path,
        gnn_checkpoint=gnn_checkpoint,
        output_dir=output_dir,
        device=device,
        oracle_batch_size=int(oracle_batch_size),
        minimum_candidates=int(minimum_candidates),
    )


__all__ = [
    "audit_bace_source_roundtrip",
    "audit_bace_vrrw_candidate_sufficiency",
    "build_bace_native_summary",
    "export_bace_gine_candidate_universe",
    "export_bace_rf_valid_top20",
    "official_greedy_coverage_order",
    "run_bace_official_vrrw",
    "train_bace_official_gnn",
]
