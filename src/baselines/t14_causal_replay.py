"""T14-only 250→335 observation, executed through original scientific pins."""
from __future__ import annotations

from contextlib import contextmanager
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import time


def copy_prefix(source: Path, destination: Path, count: int) -> dict:
    """Copy only a committed immutable prefix, never modify/truncate source."""
    if source.is_symlink() or not source.is_file() or count < 0:
        raise ValueError("Physical source and nonnegative boundary required")
    before = source.stat()
    if before.st_size < count:
        raise ValueError("Committed prefix truncated")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open("rb") as reader, destination.open("xb") as writer:
        remaining = count
        while remaining:
            block = reader.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError("Short immutable prefix copy")
            writer.write(block)
            digest.update(block)
            remaining -= len(block)
        writer.flush()
        os.fsync(writer.fileno())
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed during committed-prefix clone")
    return {"source": str(source), "destination": str(destination), "bytes": count, "content_sha256": digest.hexdigest()}


def clone_route_state(source_root, output_root, loaded, diagnostic):
    import numpy as np
    from src.baselines.tastemolnet_t14_route_c_fresh import RouteCMMapCandidateState
    state = loaded.algorithm_state["route_c_state"]
    graph = state["graph_store"]
    candidate = state["candidates"]
    if loaded.algorithm_state["live_graph_state"]["store"] != graph:
        raise ValueError("Sealed duplicate graph-store bindings differ")
    source = source_root / "route_c_state"
    target = output_root / "route_c_state"
    receipts = []
    for relative, count in (
        ("graph_store/graphs.bin", int(graph["data_bytes"])),
        ("candidate_state/candidate_payloads.bin", int(candidate["payload_bytes"])),
        ("candidate_state/candidate_payload_index.jsonl", int(candidate["payload_index_bytes"])),
    ):
        receipts.append(copy_prefix(source / relative, target / relative, count))
    # This is the independently sealed SQLite snapshot, never an active WAL.
    snapshot = Path(loaded.sqlite_snapshot_path)
    receipts.append(copy_prefix(snapshot, target / "graph_store/graph_index.sqlite3", snapshot.stat().st_size))
    for filename, field, size, dtype in (
        ("frequency.i64", "frequency", int(candidate["record_capacity"]), "<i8"),
        ("order.i64", "order", int(candidate["capacity"]), "<i8"),
        ("metadata.bin", "metadata", int(candidate["record_capacity"]), RouteCMMapCandidateState._META_DTYPE),
    ):
        path = target / "candidate_state" / filename
        if path.exists():
            raise FileExistsError(path)
        array = np.memmap(path, mode="w+", shape=(size,), dtype=dtype)
        array[:] = 0
        values = np.asarray(candidate[field], dtype=dtype)
        array[:len(values)] = values
        array.flush()
        del array
    diagnostic.atomic_json(output_root / "committed_prefix_clone.json", {
        "schema_version": "t14_diagnostic_readonly_source_clone_v1",
        "source_checkpoint_digest": loaded.validation.checkpoint_digest,
        "source_unchanged": True, "diagnostic_only": True,
        "promotion_claimed": False, "copies": receipts,
        "restoration_gate": "ORIGINAL_UPDATER_CHECKPOINT_VALIDATOR_BEFORE_STEP251",
    })


def validate_campaign(campaign_path: Path, *, source_spec_path: Path, output_root: Path) -> list[Path]:
    value = json.loads(campaign_path.read_text())
    if value.get("total_new_transition_cap") != 170 or set(value.get("arms", {})) != {"reference", "lowmemory"}:
        raise ValueError("T14 campaign requires exactly two arms and total170")
    roots = []
    matched = 0
    for arm in value["arms"].values():
        if arm.get("start_step") != 250 or arm.get("end_step") != 335 or arm.get("new_transition_cap") != 85:
            raise ValueError("T14 arm must be exactly250→335/cap85")
        root = Path(arm["output_root"])
        if not root.is_absolute():
            raise ValueError("Campaign output must be absolute")
        roots.append(root)
        matched += int(root == output_root and arm["source_spec"] == str(source_spec_path))
    if matched != 1 or len(set(roots)) != 2:
        raise ValueError("Replay not bound to unique campaign slot")
    return roots


def execute_replay(*, source_worktree, source_spec_path, output_root, gpu_uuid, campaign_path, diagnostic):
    from src.baselines import tastemolnet_comrecgc_full as full
    from src.baselines.comrecgc.generation_checkpoint import load_generation_checkpoint, capture_rng_state, save_generation_checkpoint
    from src.baselines.comrecgc.generation_loop import restore_official_state, run_generation_loop
    from src.baselines.comrecgc.runtime import reset_official_state
    from src.baselines.gcfexplainer_mutagenicity_adapter import GraphRecordDataset
    from src.baselines.tastemolnet_t14_route_c_fresh import load_spec
    from src.utils.tastemolnet_t9_managed_v2 import hold_t9_inputs, require_gpu_runtime
    import torch

    campaign_roots = validate_campaign(campaign_path, source_spec_path=source_spec_path, output_root=output_root)
    spec = load_spec(source_spec_path)
    actual_commit = subprocess.check_output(["git", "-C", str(source_worktree), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != spec["execution_commit"] or spec["gpu_index"] != 2:
        raise ValueError("Original arm execution pin/GPU differs")
    source_root = Path(spec["output_root"])
    if output_root.exists() or output_root == source_root or not output_root.is_absolute():
        raise ValueError("Diagnostic output must be a fresh absolute root")
    output_root.mkdir(parents=True)
    environment = spec["science_environment"]
    if os.environ.get("AUTODL_PHYSICAL_GPU_UUID") != gpu_uuid or os.environ.get("AUTODL_PHYSICAL_GPU_INDEX") != "2":
        raise ValueError("Existing GPU2 owner identity not inherited")
    require_gpu_runtime(gpu_uuid, physical_gpu_index=2)
    identity = json.loads((source_root / "checkpoint_identity.json").read_text())
    diagnostic.atomic_json(output_root / "diagnostic_contract.json", {
        "start_step": 250, "end_step": 335, "arm_new_transition_cap": 85,
        "total_campaign_cap": 170, "source_spec": str(source_spec_path),
        "source_execution_commit": actual_commit, "original_checkpoint_identity": identity,
        "retry3": False, "formal_promotion": False, "old_root_modified": False,
        "sampling_backend_changed": False, "numerical_tolerances_changed": False,
        "campaign_contract": str(campaign_path),
    })
    with hold_t9_inputs(
        config_path=source_worktree / "configs/hpc.yaml", run_id=output_root.name,
        gpu_uuid=gpu_uuid, physical_gpu_index=2,
        t2_adoption_root=Path(environment["TASTEMOLNET_T2_ADOPTION_ROOT"]),
        t2_adoption_gate_sha256=environment["TASTEMOLNET_T2_ADOPTION_GATE_SHA256"],
        t2_adoption_receipt_sha256=environment["TASTEMOLNET_T2_ADOPTION_RECEIPT_SHA256"],
        t2_source_evidence_sha256=environment["TASTEMOLNET_T2_SOURCE_EVIDENCE_SHA256"],
        t3_output_root=Path(environment["TASTEMOLNET_T3_OUTPUT_ROOT"]),
        t4_output_root=Path(environment["TASTEMOLNET_T4_OUTPUT_ROOT"]),
        checkpoint_dir=Path(environment["TASTEMOLNET_T3_OUTPUT_ROOT"]) / "artifacts/checkpoint",
        train_csv=Path(environment["TASTEMOLNET_TRAIN_CSV"]),
        official_root=Path(environment["COMRECGC_OFFICIAL_ROOT"]),
    ) as inputs:
        authority = inputs.revalidate()
        train = authority["train"]
        loaded_train = full.load_train_rows(inputs.train_file.read_bytes(), source_path=Path(train["path"]), expected_num_records=train["num_records"], expected_label_counts=train["label_counts"])
        cohort = [json.loads(line) for line in (source_root / "cohort.jsonl").read_text().splitlines()]
        ids = [row["parent_id"] for row in cohort]
        selected = sorted((row for row in loaded_train.sweet_rows if row.molecule_id in set(ids)), key=lambda row: row.molecule_id)
        if [row.molecule_id for row in selected] != ids:
            raise ValueError("Original cohort row/order changed")
        payloads = {name: blob for name, blob in inputs.checkpoint_payloads.items() if name != "config.yaml"}
        graphs, _records, adapter, source_evidence = full._initialize_full_source_graphs(checkpoint_payloads=payloads, source_rows=selected, graph_schema=loaded_train.schema, device="cuda:0")
        adapter.enable_canonical_replay_cache()
        diagnostic.atomic_json(output_root / "source_cohort_replay.json", source_evidence)
        parameters = full.TasteComRecGCFullParameters(source_pool=len(graphs), source_count=len(graphs)).validate()
        dataset = GraphRecordDataset(graphs, num_features=len(loaded_train.schema.feature_atomic_numbers))
        module = inputs.official.modules["comrecgc"]
        reset_official_state(module, candidate_capacity=parameters.candidate_capacity, sample_size=parameters.sample_size)
        module.input_graphs_covered = torch.zeros(len(graphs), dtype=torch.float32)
        bridge = full.TasteComRecGCFullBridge(adapter=adapter, feature_atomic_numbers=loaded_train.schema.feature_atomic_numbers, cohort_count=len(graphs))
        importance_args = {"schema_version": "tastemolnet_comrecgc_gine_distance_v1", "classifier": "frozen_calibrated_three_class_gine", "distance_embedding": "frozen_gine_graph_hidden", "num_classes": 3, "source_label": 1}
        loaded = load_generation_checkpoint(source_root / "checkpoints/step-000000000250", expected_provenance=identity["provenance"], expected_scientific_argv=identity["scientific_argv"], expected_command_sha256=identity["command_sha256"], expected_total_steps=25000, expected_completed_step=250, single_pass=True)
        lowmemory = spec["storage_mode"] == "lowmemory"
        if lowmemory:
            clone_route_state(source_root, output_root, loaded, diagnostic)
            full._restore_route_c_official_state(module, loaded.algorithm_state.pop("official_state"))
        else:
            restore_official_state(module, loaded.algorithm_state.pop("official_state"), consume=True)
        store = full._prepare_runtime_store(output_root, loaded)
        with full._bounded_t14_runtime(module=module, bridge=bridge, graph_store_path=store, seed=parameters.seed, expanded_capacity=full.TRANSITION_EXPANDED_CAPACITY, route_c_root=output_root / "route_c_state" if lowmemory else None, route_c_resume=lowmemory) as handles:
            state = full._restore_checkpoint_state(module=module, bridge=bridge, loaded=loaded, handles=handles)
            started = completed = 0
            start_time = time.monotonic()
            with gzip.open(output_root / "raw_step_observations.pkl.gz", "wb") as observations, (output_root / "step_summary.jsonl").open("x") as summaries, diagnostic.SamplingObserver(module) as observer:
                original_move = module.move_to_next_graph
                def observed_move(*args, **kwargs):
                    nonlocal started
                    if started >= 85:
                        raise RuntimeError("T14 arm85/total170 cap exhausted")
                    other_started = 0
                    for other in campaign_roots:
                        if other != output_root and (other / "transition_budget.json").is_file():
                            other_started += int(json.loads((other / "transition_budget.json").read_text())["started_new_transitions"])
                    # GPU2's existing exclusive lease serializes this check.
                    # A failed partial transition remains counted in its arm.
                    if started + other_started >= 170:
                        raise RuntimeError("T14 total170 exhausted")
                    started += 1
                    diagnostic.atomic_json(output_root / "transition_budget.json", {"started_new_transitions": started, "completed_new_transitions": completed, "step_in_progress": 250 + started, "arm_cap": 85, "campaign_cap": 170, "auto_retry_allowed": False})
                    observer.events.clear()
                    pickle.dump({"phase": "BEFORE", "step": 250+started, "rng": capture_rng_state()}, observations, protocol=5)
                    observations.flush()
                    return original_move(*args, **kwargs)
                module.move_to_next_graph = observed_move
                def on_step(loop_state):
                    nonlocal completed
                    completed += 1
                    if loop_state.completed_step != 250 + completed or completed != started:
                        raise ValueError("T14 completed-step accounting drift")
                    row = {"phase": "AFTER", "step": loop_state.completed_step, "rng": capture_rng_state(), "actual_sampling_events": observer.events, "native_observation": handles.step_observation, "loop_state": loop_state.to_checkpoint_state()}
                    pickle.dump(row, observations, protocol=5)
                    observations.flush()
                    selected_actions = handles.step_observation.get("selected_transitions", ())
                    summaries.write(json.dumps({"completed_step": loop_state.completed_step, "elapsed_seconds": time.monotonic()-start_time, "selected_transitions": diagnostic.semantic(selected_actions), "sampling_draws": len(observer.events)}, sort_keys=True) + "\n")
                    summaries.flush()
                    diagnostic.atomic_json(output_root / "transition_budget.json", {"started_new_transitions": started, "completed_new_transitions": completed, "last_completed_step": loop_state.completed_step, "arm_cap": 85, "campaign_cap": 170, "auto_retry_allowed": False})
                try:
                    state = run_generation_loop(module, input_graphs=dataset, importance_args=importance_args, teleport_probability=parameters.teleport_probability, max_steps=335, heads=parameters.heads, initial_state=state, on_step_complete=on_step)
                finally:
                    module.move_to_next_graph = original_move
                os.fsync(summaries.fileno())
            # A diagnostic-only recovery checkpoint, never a formal promotion.
            checkpoint = save_generation_checkpoint(output_root / "diagnostic_checkpoints", completed_step=335, step_complete=True,
                algorithm_state=full._checkpoint_algorithm_state(module=module, bridge=bridge, loop_state=state, handles=handles),
                trace_state={"enabled": True, "policy": "bounded_causal_observation_diagnostic_only"},
                sqlite_source=handles.live_graph_state.store.checkpoint_connection,
                provenance_fingerprints=identity["provenance"], scientific_argv=identity["scientific_argv"], command_sha256=identity["command_sha256"], total_steps=25000, reload_after_write=False)
            result = {"status": "BOUNDED_DIAGNOSTIC_335_COMPLETE", "completed_step": 335, "started_new_transitions": started, "completed_new_transitions": completed, "arm_cap": 85, "total_cap": 170, "formal_dispatch_allowed": False, "parity_claimed": False, "checkpoint_digest": checkpoint.checkpoint_digest, "checkpoint_scope": "DIAGNOSTIC_ONLY_NOT_PROMOTABLE", "elapsed_seconds": time.monotonic()-start_time}
            diagnostic.atomic_json(output_root / "terminal.json", result)
            return result
