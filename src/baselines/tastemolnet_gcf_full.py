"""TasteMolNet T12 train-only 10k/20k production generation.

This is intentionally a narrow dataset-specific continuation of the reviewed
real T7/T12 path.  It does not perform calibration or test evaluation.  The
first process commits cursor 10,000 and exits; a distinct process reopens that
checkpoint, commits cursor 20,000, and materializes the complete ordered
native candidate pool for later calibration-only selection.
"""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping
import uuid

from src.baselines.tastemolnet_comrecgc_full import (
    build_full_train_correct_source_cohort,
)
from src.baselines.tastemolnet_comrecgc_smoke import canonical_attributed_graph
from src.baselines.tastemolnet_gcf_full_resume import (
    CANARY_GATE_SCHEMA,
    CANARY_PASS_MARKER,
    GRAPH_IDENTITY_CONTRACT,
    NEUROSED_QUERY_PERMUTATION_CONTRACT,
    PINNED_CANDIDATE_CAPACITY,
    PINNED_SAMPLE_SIZE,
    PRODUCTION_CHECKPOINT_CURSORS,
    PRODUCTION_TOTAL_STEPS,
    STAGE,
    T12ProductionCheckpointOrchestrator,
    T12StableGCFBridge,
    TasteGCFFullResumeError,
    production_transition_contract_sha256,
    reopen_checkpoint,
    restore_checkpoint_payload,
    validate_checkpoint_identity,
)
from src.baselines.tastemolnet_gcf_production_state import (
    T12CompactHistoryJournal,
    T12ProductionBounds,
)
from src.baselines.tastemolnet_gcf_replay_canary import (
    _BoundedNeuroSEDCoverage,
    _absolute,
    _canonical_bytes,
    _installed_bounded_neurosed_coverage,
    _official_vrrw_alpha_endpoint_patch,
    _reset_official_vrrw,
    _runtime_identity,
    _sha256_bytes,
    _sha256_file,
    _write_new,
    configure_exact_cuda_replay,
    load_threshold_authority,
    require_real_a800,
)
from src.baselines.tastemolnet_gcf_smoke import (
    SOURCE_LABEL,
    TasteFrozenGINENativeAdapter,
    _semantic_sha256,
    _installed_official_importance_args,
    _official_modules,
    _run_official_walk_segment,
    encode_taste_source_graph,
    load_train_rows,
    taste_record_to_pyg,
)
from src.baselines.tastemolnet_gcf_transition_store import (
    T12ExternalTransitionStore,
)


PRODUCTION_PARENT_COUNT = 3_778
PRODUCTION_SEED = 7
PRODUCTION_ALPHA = 1.0
PRODUCTION_TELEPORT = 0.1
PRODUCTION_RUN_SCHEMA = "tastemolnet_t12_gcf_generation_run_v1"
PRODUCTION_RECEIPT_SCHEMA = "tastemolnet_t12_gcf_generation_receipt_v1"


def _load_and_validate_native_result(
    *, runtime_root: Path, vrrw: Any, torch: Any
) -> tuple[Path, str, str]:
    """Bind the official terminal archive to the exact in-process state."""

    path = runtime_root / "results/tastemolnet/runs/counterfactuals.pt"
    if (
        not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
        or path.stat().st_size <= 0
    ):
        raise TasteGCFFullResumeError(
            "T12 official VRRW did not emit one physical native result"
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older supported Torch
        payload = torch.load(path, map_location="cpu")
    expected = {
        "graph_map": getattr(vrrw, "graph_map"),
        "graph_index_map": getattr(vrrw, "graph_index_map"),
        "counterfactual_candidates": getattr(
            vrrw, "counterfactual_candidates"
        ),
        "MAX_COUNTERFACTUAL_SIZE": getattr(vrrw, "MAX_COUNTERFACTUAL_SIZE"),
        "traversed_hashes": getattr(vrrw, "traversed_hashes"),
        "input_graphs_covered": getattr(vrrw, "input_graphs_covered"),
    }
    if (
        type(payload) is not dict
        or set(payload) != set(expected)
        or _semantic_sha256(payload) != _semantic_sha256(expected)
    ):
        raise TasteGCFFullResumeError(
            "T12 official native result differs from the terminal live state"
        )
    return path, _sha256_file(path), _semantic_sha256(payload)


def _load_replay_gate(path: str | Path) -> tuple[dict[str, Any], str]:
    gate_path = _absolute(path, field="T12 exact replay gate")
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteGCFFullResumeError("T12 exact replay gate is unreadable") from exc
    if (
        type(gate) is not dict
        or gate.get("schema_version") != CANARY_GATE_SCHEMA
        or gate.get("status") != "PASS"
        or gate.get("marker") != CANARY_PASS_MARKER
        or gate.get("stage") != STAGE
        or gate.get("graph_identity_contract") != GRAPH_IDENTITY_CONTRACT
        or gate.get("neurosed_query_permutation_contract")
        != NEUROSED_QUERY_PERMUTATION_CONTRACT
        or gate.get("cross_process") is not True
        or gate.get("cuda_used") is not True
        or gate.get("exact_equality") is not True
        or gate.get("scientific_exact_equality") is not True
        or gate.get("approximate_comparison_used") is not False
        or gate.get("native_result_approximate_comparison_used") is not False
        or gate.get("production_released") is not False
    ):
        raise TasteGCFFullResumeError(
            "T12 production requires the contract-bound cross-process replay gate v3"
        )
    return gate, _sha256_file(gate_path)


def _select_full_source_rows(
    *, sources: Any, loaded: Any, device: str
) -> tuple[list[Any], dict[str, Any], bytes]:
    """Freeze the same true-and-predicted Sweet train cohort used by T14."""

    import torch

    true_sweet = list(loaded.sweet_rows)
    records = [encode_taste_source_graph(row, loaded.schema) for row in true_sweet]
    graphs = [
        taste_record_to_pyg(record, origin_index=index)
        for index, record in enumerate(records)
    ]
    adapter = TasteFrozenGINENativeAdapter(
        sources.checkpoint_payloads,
        source_records=records,
        graph_schema=loaded.schema,
        device=device,
    )
    predictions: list[int] = []
    source_probabilities: list[float] = []
    graph_hashes: list[str] = []
    for offset in range(0, len(graphs), 128):
        chunk = graphs[offset : offset + 128]
        scored = adapter.score(chunk)
        if any(not value for value in scored.valid_fullgraphs):
            raise TasteGCFFullResumeError(
                "T12 source-cohort GINE replay produced an invalid graph"
            )
        predictions.extend(int(value) for value in scored.predictions)
        source_probabilities.extend(
            float(row[SOURCE_LABEL]) for row in scored.probabilities.tolist()
        )
        graph_hashes.extend(
            canonical_attributed_graph(
                graph,
                feature_atomic_numbers=loaded.schema.feature_atomic_numbers,
            ).graph_identity_sha256
            for graph in chunk
        )
    cohort, t14_manifest, cohort_bytes = build_full_train_correct_source_cohort(
        true_sweet_rows=true_sweet,
        predictions=predictions,
        source_probabilities=source_probabilities,
        canonical_graph_hashes=graph_hashes,
        train_csv_sha256=str(sources.train_contract["sha256"]),
        checkpoint_id=str(sources.authority.t3_checkpoint_id),
    )
    if len(cohort) != PRODUCTION_PARENT_COUNT:
        raise TasteGCFFullResumeError(
            "T12 frozen full source cohort is not the reviewed 3,778 parents"
        )
    by_id: dict[str, Any] = {}
    for row in true_sweet:
        key = str(row.molecule_id)
        if key in by_id:
            raise TasteGCFFullResumeError("T12 train parent ID is not unique")
        by_id[key] = row
    selected_rows = [by_id[str(row["parent_id"])] for row in cohort]
    manifest = {
        "schema_version": "tastemolnet_t12_full_train_cohort_v1",
        "status": "PASS",
        "dataset": "tastemolnet",
        "stage": STAGE,
        "selection": t14_manifest["selection"],
        "stable_sort": list(t14_manifest["stable_sort"]),
        "split": "train",
        "source_label": SOURCE_LABEL,
        "cohort_count": len(cohort),
        "train_csv_sha256": t14_manifest["train_csv_sha256"],
        "checkpoint_id": t14_manifest["checkpoint_id"],
        "cohort_jsonl_sha256": hashlib.sha256(cohort_bytes).hexdigest(),
        "shared_source_policy_with_t14": True,
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "result_conditioned_selection": False,
    }
    del adapter, graphs, records, predictions, source_probabilities, graph_hashes
    gc.collect()
    torch.cuda.empty_cache()
    return selected_rows, manifest, cohort_bytes


def _production_identity(
    *,
    sources: Any,
    attempt_id: str,
    generation_token: str,
    source_cohort_sha256: str,
    threshold_authority_sha256: str,
    runtime_identity_sha256: str,
    gpu_uuid: str,
) -> dict[str, Any]:
    model_config_sha = _sha256_bytes(
        _canonical_bytes(
            {
                name: _sha256_bytes(data)
                for name, data in sorted(sources.checkpoint_payloads.items())
            }
        )
    )
    return validate_checkpoint_identity(
        {
            "schema_version": "tastemolnet_t12_checkpoint_identity_v1",
            "stage": STAGE,
            "purpose": "production",
            "attempt_id": attempt_id,
            "generation_token": generation_token,
            "total_steps": PRODUCTION_TOTAL_STEPS,
            "checkpoint_cursor": min(PRODUCTION_CHECKPOINT_CURSORS),
            "source_cohort_sha256": source_cohort_sha256,
            "train_split_sha256": sources.pins.train_split_sha,
            "model_checkpoint_sha256": sources.pins.t3_calibrated_gine_sha,
            "model_config_sha256": model_config_sha,
            "neurosed_checkpoint_sha256": sources.pins.neurosed_model_sha,
            "neurosed_distance_threshold_hex": float(
                sources.authority.neurosed_distance_threshold
            ).hex(),
            "neurosed_threshold_authority_sha256": threshold_authority_sha256,
            "official_source_inventory_sha256": (
                sources.authority.official_gcf_inventory_sha256
            ),
            "execution_commit": sources.authority.implementation_commit,
            "execution_tree": sources.authority.implementation_tree,
            "runtime_identity_sha256": runtime_identity_sha256,
            "gpu_uuid": gpu_uuid,
            "device": "cuda:0",
            "graph_identity_contract": GRAPH_IDENTITY_CONTRACT,
            "seed": PRODUCTION_SEED,
            "alpha_hex": PRODUCTION_ALPHA.hex(),
            "teleport_hex": PRODUCTION_TELEPORT.hex(),
            "sample_size": PINNED_SAMPLE_SIZE,
            "candidate_capacity": PINNED_CANDIDATE_CAPACITY,
            "train_loaded": True,
            "calibration_loaded": False,
            "test_loaded": False,
            "rf_oracle_used": False,
        }
    )


def _run_identity(
    *,
    identity: Mapping[str, Any],
    cohort_manifest: Mapping[str, Any],
    replay_gate_sha256: str,
    threshold_path: Path,
    threshold_sha256: str,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": PRODUCTION_RUN_SCHEMA,
        "stage": STAGE,
        "purpose": "production",
        "identity_template": dict(identity),
        "transition_contract_sha256": production_transition_contract_sha256(
            identity
        ),
        "cohort_manifest": dict(cohort_manifest),
        "replay_gate_sha256": replay_gate_sha256,
        "threshold_authority_path": str(threshold_path),
        "threshold_authority_sha256": threshold_sha256,
        "runtime": dict(runtime),
        "train_loaded": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "rf_oracle_used": False,
        "production_parameters": {
            "M": PRODUCTION_TOTAL_STEPS,
            "checkpoint_cursors": sorted(PRODUCTION_CHECKPOINT_CURSORS),
            "sample_size": PINNED_SAMPLE_SIZE,
            "candidate_capacity": PINNED_CANDIDATE_CAPACITY,
            "expanded_transition_lru": 1,
        },
    }


def run_t12_generation_segment(
    *,
    mode: str,
    output_root: str | Path,
    checkpoint_manifest: str | Path | None,
    attempt_id: str,
    generation_token: str,
    gpu_uuid: str,
    managed_neurosed_root: str | Path,
    t3_root: str | Path,
    official_root: str | Path,
    threshold_authority_path: str | Path,
    replay_gate_path: str | Path,
) -> dict[str, Any]:
    """Run exactly fresh 1..10k or resumed 10001..20k generation."""

    if mode not in {"fresh", "resume"}:
        raise TasteGCFFullResumeError("T12 production mode must be fresh/resume")
    try:
        parsed = uuid.UUID(attempt_id)
    except (ValueError, AttributeError) as exc:
        raise TasteGCFFullResumeError("T12 production attempt ID is invalid") from exc
    if parsed.version != 4 or str(parsed) != attempt_id:
        raise TasteGCFFullResumeError("T12 production attempt ID is not UUIDv4")
    if (
        type(generation_token) is not str
        or len(generation_token) != 64
        or any(value not in "0123456789abcdef" for value in generation_token)
    ):
        raise TasteGCFFullResumeError("T12 production generation token is invalid")
    import numpy as np
    import torch

    determinism = configure_exact_cuda_replay(torch=torch)
    gpu = require_real_a800(gpu_uuid=gpu_uuid, torch=torch)
    replay_gate, replay_gate_sha = _load_replay_gate(replay_gate_path)
    random.seed(PRODUCTION_SEED)
    np.random.seed(PRODUCTION_SEED)
    torch.manual_seed(PRODUCTION_SEED)
    torch.cuda.manual_seed_all(PRODUCTION_SEED)
    from src.utils.tastemolnet_t7_typed_release_v1 import hold_t7_release_sources

    threshold_path = _absolute(
        threshold_authority_path, field="T12 threshold authority"
    )
    threshold_raw = json.loads(threshold_path.read_text(encoding="utf-8"))
    threshold_value = threshold_raw.get("neurosed_distance_threshold")
    if (
        isinstance(threshold_value, bool)
        or not isinstance(threshold_value, (int, float))
        or not math.isfinite(float(threshold_value))
        or float(threshold_value) < 0.0
    ):
        raise TasteGCFFullResumeError("T12 threshold authority has no usable value")
    with hold_t7_release_sources(
        managed_neurosed_root=managed_neurosed_root,
        t3_root=t3_root,
        official_gcf_root=official_root,
        neurosed_distance_threshold=float(threshold_value),
    ) as sources:
        threshold, threshold_sha = load_threshold_authority(
            threshold_path,
            expected_neurosed_checkpoint_sha256=sources.pins.neurosed_model_sha,
            expected_neurosed_feature_schema_sha256=sources.neurosed_evidence[
                "feature_schema_sha256"
            ],
            expected_t3_checkpoint_id=sources.authority.t3_checkpoint_id,
            expected_t3_temperature_sha256=sources.pins.t3_temperature_sha,
            expected_t3_gate_sha256=sources.authority.t3_gate_sha256,
            expected_t3_verification_sha256=(
                sources.authority.t3_verification_sha256
            ),
            expected_official_inventory_sha256=(
                sources.authority.official_gcf_inventory_sha256
            ),
            expected_managed_neurosed_root=(
                sources.authority.managed_neurosed_root
            ),
            expected_t3_root=sources.authority.t3_root,
            expected_official_root=sources.authority.official_gcf_root,
        )
        runtime, _unused_runtime_sha = _runtime_identity(
            sources=sources,
            threshold_authority_sha256=threshold_sha,
            gpu=gpu,
            determinism=determinism,
        )
        runtime = {
            **runtime,
            "exact_replay_gate_sha256": replay_gate_sha,
            "exact_replay_gate_semantic_sha256": replay_gate[
                "scientific_state_sha256"
            ],
            "external_transition_store": True,
            "transition_expanded_capacity": 1,
        }
        runtime_sha = _sha256_bytes(_canonical_bytes(runtime))
        loaded = load_train_rows(
            sources.train_bytes,
            source_path=Path(sources.train_contract["path"]),
            expected_num_records=sources.train_contract["num_records"],
            expected_label_counts=sources.train_contract["label_counts"],
        )
        selected_rows, cohort_manifest, cohort_bytes = _select_full_source_rows(
            sources=sources, loaded=loaded, device="cuda:0"
        )
        identity = _production_identity(
            sources=sources,
            attempt_id=attempt_id,
            generation_token=generation_token,
            source_cohort_sha256=cohort_manifest["cohort_jsonl_sha256"],
            threshold_authority_sha256=threshold_sha,
            runtime_identity_sha256=runtime_sha,
            gpu_uuid=gpu_uuid,
        )
        run_identity = _run_identity(
            identity=identity,
            cohort_manifest=cohort_manifest,
            replay_gate_sha256=replay_gate_sha,
            threshold_path=threshold_path,
            threshold_sha256=threshold_sha,
            runtime=runtime,
        )
        root = _absolute(
            output_root, field="T12 production output root", must_exist=False
        )
        if mode == "fresh":
            root.mkdir(mode=0o700, parents=True, exist_ok=False)
            if root.resolve(strict=True) != root or root.is_symlink():
                raise TasteGCFFullResumeError(
                    "T12 fresh production root is an alias"
                )
            _write_new(root / "run_identity.json", run_identity)
            _write_new(root / "cohort_manifest.json", cohort_manifest)
            cohort_path = root / "cohort.jsonl"
            with cohort_path.open("xb") as stream:
                stream.write(cohort_bytes)
                stream.flush()
                os.fsync(stream.fileno())
        else:
            if not root.is_dir() or root.resolve(strict=True) != root:
                raise TasteGCFFullResumeError("T12 resume root is invalid")
            if json.loads((root / "run_identity.json").read_text()) != run_identity:
                raise TasteGCFFullResumeError("T12 production run identity changed")
            if json.loads((root / "cohort_manifest.json").read_text()) != cohort_manifest:
                raise TasteGCFFullResumeError("T12 production cohort changed")
            if (root / "cohort.jsonl").read_bytes() != cohort_bytes:
                raise TasteGCFFullResumeError("T12 production cohort bytes changed")
        modules = _official_modules(sources.official_root)
        vrrw = modules["vrrw"]
        importance = modules["importance"]
        distance = modules["distance"]
        _reset_official_vrrw(vrrw)
        records = [
            encode_taste_source_graph(row, loaded.schema) for row in selected_rows
        ]
        input_graphs = [
            taste_record_to_pyg(record, origin_index=index)
            for index, record in enumerate(records)
        ]
        adapter = TasteFrozenGINENativeAdapter(
            sources.checkpoint_payloads,
            source_records=records,
            graph_schema=loaded.schema,
            device="cuda:0",
        )
        for offset in range(0, len(input_graphs), 128):
            scored = adapter.score(input_graphs[offset : offset + 128])
            if any(
                not valid or prediction != SOURCE_LABEL
                for valid, prediction in zip(
                    scored.valid_fullgraphs, scored.predictions, strict=True
                )
            ):
                raise TasteGCFFullResumeError(
                    "T12 frozen production source replay changed"
                )
        sources.revalidate()
        neurosed = distance.load_neurosed(
            input_graphs,
            neurosed_model_path=f"/proc/self/fd/{sources.neurosed_model.file_fd}",
            device="cuda:0",
        )
        sources.revalidate()
        original_counts = importance.util.graph_element_counts(input_graphs)
        original_neighbor = vrrw.neighbor_graph_access
        # The lineage-only wrapper is owned by the retained official-GCF
        # adapter.  The smoke module consumes it but intentionally does not
        # re-export it.
        from src.baselines.gcfexplainer_mutagenicity_adapter import (
            graph_lineage_neighbor_wrapper,
        )

        lineage_neighbor = graph_lineage_neighbor_wrapper(original_neighbor)
        action_counts: Counter[str] = Counter()

        def counted_neighbor(graph: Any, action: tuple[Any, ...]) -> Any:
            action_counts[str(action[0])] += 1
            return lineage_neighbor(graph, action)

        vrrw.neighbor_graph_access = counted_neighbor
        vrrw.dataset_name = "tastemolnet"
        vrrw.alpha = PRODUCTION_ALPHA
        vrrw.sample_size = PINNED_SAMPLE_SIZE
        vrrw.is_sample = True
        vrrw.MAX_COUNTERFACTUAL_SIZE = PINNED_CANDIDATE_CAPACITY
        vrrw.input_graphs_covered = torch.zeros(
            PRODUCTION_PARENT_COUNT, dtype=torch.float32
        )
        importance_args = {
            "schema_version": "tastemolnet_t12_gcf_neurosed_importance_v1",
            "alpha": PRODUCTION_ALPHA,
            "distance_status": "EVALUATED",
            "distance_threshold": float(threshold["neurosed_distance_threshold"]),
            "selector_status": "NOT_EVALUATED",
            "calibration_loaded": False,
            "test_loaded": False,
        }
        bounds = T12ProductionBounds.pinned(parent_count=PRODUCTION_PARENT_COUNT)
        orchestrator = T12ProductionCheckpointOrchestrator(
            checkpoint_root=root / "checkpoints",
            identity_template=identity,
            bounds=bounds,
        )
        configured_cursors = tuple(sorted(PRODUCTION_CHECKPOINT_CURSORS))
        if mode == "fresh":
            resume_cursor = 0
        else:
            try:
                resume_manifest = json.loads(
                    Path(str(checkpoint_manifest)).read_text(encoding="utf-8")
                )
                resume_cursor = int(resume_manifest["checkpoint_cursor"])
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise TasteGCFFullResumeError(
                    "T12 resume checkpoint cursor is unreadable"
                ) from exc
            if resume_cursor not in configured_cursors[:-1]:
                raise TasteGCFFullResumeError(
                    "T12 resume checkpoint cursor is not a nonterminal configured boundary"
                )
        plan = orchestrator.plan(resume_cursor=resume_cursor)
        runtime_root = root / (
            f"segment-{plan['segment_start']:05d}-{plan['segment_end']:05d}"
        )
        runtime_root.mkdir(mode=0o700, exist_ok=False)
        loaded_checkpoint = None
        if mode == "resume":
            if checkpoint_manifest is None:
                raise TasteGCFFullResumeError(
                    "T12 production resume requires the 10k checkpoint manifest"
                )
            loaded_checkpoint = reopen_checkpoint(
                checkpoint_manifest,
                expected_identity=orchestrator.identity_at(resume_cursor),
                torch=torch,
            )
            history_snapshot = loaded_checkpoint["state"]["bridge"]["history"]
        else:
            if checkpoint_manifest is not None:
                raise TasteGCFFullResumeError(
                    "T12 fresh production may not receive a checkpoint"
                )
            history_snapshot = None
        contract_sha = production_transition_contract_sha256(identity)
        history = T12CompactHistoryJournal(
            root=(root / "bridge_history").resolve(),
            index_root=(runtime_root / "history_index").resolve(),
            bounds=bounds,
            contract_sha256=contract_sha,
            attempt_id=attempt_id,
            generation_token=generation_token,
            resume_snapshot=history_snapshot,
        )
        coverage_runtime = _BoundedNeuroSEDCoverage(importance)
        bridge = T12StableGCFBridge(
            adapter=adapter,
            vrrw=vrrw,
            importance=importance,
            neurosed_model=neurosed,
            original_graph_element_counts=original_counts,
            distance_threshold=float(threshold["neurosed_distance_threshold"]),
            parent_count=PRODUCTION_PARENT_COUNT,
            feature_atomic_numbers=loaded.schema.feature_atomic_numbers,
            coverage_runtime=coverage_runtime,
            production_history=history,
        )
        transition_store = None
        if mode == "fresh":
            transition_store = T12ExternalTransitionStore(
                root=(root / "transition_store").resolve(),
                parent_count=PRODUCTION_PARENT_COUNT,
                sample_size=PINNED_SAMPLE_SIZE,
                candidate_capacity=PINNED_CANDIDATE_CAPACITY,
                contract_sha256=contract_sha,
                attempt_id=attempt_id,
                generation_token=generation_token,
                expanded_capacity=1,
            )
            vrrw.transitions = transition_store
        current_graph: str | None = None
        old_cwd = Path.cwd()
        science_body_failed = False
        try:
            os.chdir(runtime_root)
            with ExitStack() as stack:
                stack.enter_context(
                    _installed_bounded_neurosed_coverage(
                        importance, coverage_runtime
                    )
                )
                stack.enter_context(bridge.installed())
                stack.enter_context(_official_vrrw_alpha_endpoint_patch(vrrw))
                stack.enter_context(
                    _installed_official_importance_args(vrrw, importance_args)
                )
                if mode == "resume":
                    assert loaded_checkpoint is not None
                    current_graph = restore_checkpoint_payload(
                        loaded_checkpoint,
                        expected_identity=orchestrator.identity_at(resume_cursor),
                        vrrw=vrrw,
                        bridge=bridge,
                        adapter=adapter,
                        action_counts=action_counts,
                        np=np,
                        torch=torch,
                    )
                    transition_store = vrrw.transitions
                segment = _run_official_walk_segment(
                    vrrw=vrrw,
                    input_graphs=input_graphs,
                    importance_args=importance_args,
                    teleport_probability=PRODUCTION_TELEPORT,
                    start_step=plan["segment_start"],
                    end_step=plan["segment_end"],
                    **(
                        {"resume_graph_hash": current_graph}
                        if mode == "resume"
                        else {}
                    ),
                )
                current_graph = segment.current_graph_hash
                if mode == "resume" and not segment.resume_entry_used_saved_graph:
                    raise TasteGCFFullResumeError(
                        "T12 production resume did not consume the saved graph"
                    )
                sources.revalidate()
                native_result_path, native_result_sha, native_result_semantic_sha = (
                    _load_and_validate_native_result(
                        runtime_root=runtime_root, vrrw=vrrw, torch=torch
                    )
                )
                manifest = orchestrator.commit(
                    completed_steps=plan["checkpoint_cursor"],
                    vrrw=vrrw,
                    bridge=bridge,
                    adapter=adapter,
                    action_counts=action_counts,
                    current_graph_identity=current_graph,
                    np=np,
                    torch=torch,
                )
                candidate_manifest = None
                if plan["checkpoint_cursor"] == PRODUCTION_TOTAL_STEPS:
                    candidate_manifest = orchestrator.materialize_terminal_candidates(
                        vrrw=vrrw,
                        completed_steps=PRODUCTION_TOTAL_STEPS,
                        torch=torch,
                    )
                receipt = {
                    "schema_version": PRODUCTION_RECEIPT_SCHEMA,
                    "status": "GENERATION_CHECKPOINT_COMMITTED",
                    "stage": STAGE,
                    "attempt_id": attempt_id,
                    "generation_token": generation_token,
                    "checkpoint_cursor": plan["checkpoint_cursor"],
                    "checkpoint_manifest": str(manifest),
                    "checkpoint_manifest_sha256": _sha256_file(manifest),
                    "candidate_manifest": (
                        str(candidate_manifest) if candidate_manifest else None
                    ),
                    "candidate_manifest_sha256": (
                        _sha256_file(candidate_manifest)
                        if candidate_manifest
                        else None
                    ),
                    "official_native_result": str(native_result_path),
                    "official_native_result_sha256": native_result_sha,
                    "official_native_result_semantic_sha256": (
                        native_result_semantic_sha
                    ),
                    "transition_store_audit": transition_store.audit(),
                    "bridge_report": bridge.report(),
                    "action_counts": dict(sorted(action_counts.items())),
                    "source_cohort_sha256": cohort_manifest[
                        "cohort_jsonl_sha256"
                    ],
                    "exact_replay_gate_sha256": replay_gate_sha,
                    "train_loaded": True,
                    "calibration_loaded": False,
                    "test_loaded": False,
                    "rf_oracle_used": False,
                    "paper_cell_pass": False,
                }
                receipt_path = root / (
                    f"generation_receipt_{plan['checkpoint_cursor']:08d}.json"
                )
                _write_new(receipt_path, receipt)
                return {**receipt, "receipt_path": str(receipt_path)}
        except BaseException:
            science_body_failed = True
            raise
        finally:
            os.chdir(old_cwd)
            vrrw.neighbor_graph_access = original_neighbor
            try:
                history.close(commit_index=not science_body_failed)
            finally:
                if transition_store is not None:
                    transition_store.close()


__all__ = [
    "PRODUCTION_ALPHA",
    "PRODUCTION_PARENT_COUNT",
    "PRODUCTION_RECEIPT_SCHEMA",
    "PRODUCTION_RUN_SCHEMA",
    "PRODUCTION_SEED",
    "PRODUCTION_TELEPORT",
    "run_t12_generation_segment",
]
