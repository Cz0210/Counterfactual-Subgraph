"""TasteMolNet ComRecGC full generation with a train-only frozen cohort.

This module is intentionally dataset specific.  It extends the already
verified T9 native bridge to the preregistered T14 resource cap without
opening validation, calibration, or test payloads during generation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from src.baselines.tastemolnet_comrecgc_smoke import (
    SOURCE_LABEL,
    TasteComRecGCMulticlassBridge,
    TasteComRecGCSmokeError,
    _common_recourse_summary,
    _identity_graph_sha256,
    _restore_reload_checkpoint,
    _seed_all,
    _torch_load_handle,
    _write_reload_checkpoint,
    canonical_attributed_graph,
)
from src.baselines.tastemolnet_gcf_smoke import (
    TasteFrozenGINENativeAdapter,
    encode_taste_source_graph,
    load_train_rows,
    taste_record_to_pyg,
)


STAGE = "T14_COMRECGC_FULL"
DATASET = "tastemolnet"
METHOD = "comrecgc"
COHORT_POLICY = "FULL_TRAIN_CORRECT_SOURCE"
VALID_UNIQUE_POLICY = "TRAIN_SIDE_STRICT_FLIP_CANONICAL"
M_MAX = 20_000
M_FALLBACK_MAX = 25_000
MIN_VALID_UNIQUE_RULES = 10
CHECK_INTERVAL = 2_500
PASS_MARKER = "[TASTE_T14_COMRECGC_PASS]"


class TasteComRecGCFullError(TasteComRecGCSmokeError):
    """The fixed T14 scientific contract was violated."""


@dataclass(frozen=True, slots=True)
class TasteComRecGCFullParameters:
    """Parameters consumed by the shared native ComRecGC implementation."""

    steps: int = M_FALLBACK_MAX
    checkpoint_step: int = CHECK_INTERVAL
    source_pool: int = 0
    source_count: int = 0
    heads: int = 5
    candidate_capacity: int = 50_000
    sample_size: int = 10_000
    teleport_probability: float = 0.1
    theta: float = 0.1
    delta: float = 0.02
    cluster_size: int = 3
    recourse_size: int = 20
    seed: int = 7

    def validate(self) -> "TasteComRecGCFullParameters":
        if (
            self.steps != M_FALLBACK_MAX
            or self.checkpoint_step % CHECK_INTERVAL != 0
            or not CHECK_INTERVAL <= self.checkpoint_step <= M_FALLBACK_MAX
            or self.source_pool <= 0
            or self.source_count != self.source_pool
            or self.heads != 5
            or self.candidate_capacity != 50_000
            or self.sample_size != 10_000
            or self.teleport_probability != 0.1
            or self.theta != 0.1
            or self.delta != 0.02
            or self.cluster_size != 3
            or self.recourse_size != 20
            or self.seed != 7
        ):
            raise TasteComRecGCFullError("Taste T14 full parameters drifted")
        return self


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise TasteComRecGCFullError("Taste T14 atomic write was short")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, _canonical_bytes(dict(value)) + b"\n")


def _cohort_lines(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(dict(row)) + b"\n" for row in rows)


def build_full_train_correct_source_cohort(
    *,
    true_sweet_rows: Sequence[Any],
    predictions: Sequence[int],
    source_probabilities: Sequence[float],
    canonical_graph_hashes: Sequence[str],
    train_csv_sha256: str,
    checkpoint_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], bytes]:
    """Build the exact user-authorized train-only T14 source cohort."""

    sizes = {
        len(true_sweet_rows),
        len(predictions),
        len(source_probabilities),
        len(canonical_graph_hashes),
    }
    if len(sizes) != 1 or not true_sweet_rows:
        raise TasteComRecGCFullError("Taste T14 cohort inputs are unaligned")
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source, prediction, probability, graph_hash in zip(
        true_sweet_rows,
        predictions,
        source_probabilities,
        canonical_graph_hashes,
        strict=True,
    ):
        molecule_id = str(getattr(source, "molecule_id", "")).strip()
        label = getattr(source, "label", None)
        if label != SOURCE_LABEL or not molecule_id:
            raise TasteComRecGCFullError("Taste T14 source row is not true Sweet")
        if type(prediction) is not int or prediction not in (0, 1, 2):
            raise TasteComRecGCFullError("Taste T14 prediction is malformed")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not 0.0 <= float(probability) <= 1.0
        ):
            raise TasteComRecGCFullError("Taste T14 source probability is malformed")
        if (
            type(graph_hash) is not str
            or len(graph_hash) != 64
            or any(character not in "0123456789abcdef" for character in graph_hash)
        ):
            raise TasteComRecGCFullError("Taste T14 canonical graph hash is malformed")
        if prediction != SOURCE_LABEL:
            continue
        if molecule_id in seen_ids:
            raise TasteComRecGCFullError("Taste T14 parent identity is not unique")
        seen_ids.add(molecule_id)
        selected.append(
            {
                "parent_id": molecule_id,
                "canonical_graph_hash": graph_hash,
                "true_label": SOURCE_LABEL,
                "predicted_label": SOURCE_LABEL,
                "source_probability": float(probability),
                "split": "train",
            }
        )
    selected.sort(
        key=lambda row: (str(row["parent_id"]), str(row["canonical_graph_hash"]))
    )
    if not selected:
        raise TasteComRecGCFullError("Taste T14 full cohort is empty")
    lines = _cohort_lines(selected)
    manifest = {
        "schema_version": "tastemolnet_t14_full_train_cohort_v1",
        "status": "PASS",
        "dataset": DATASET,
        "stage": STAGE,
        "policy": COHORT_POLICY,
        "selection": "true_label == 1 and frozen_T3_GINE_prediction == 1",
        "stable_sort": ["molecule_id", "canonical_graph_hash"],
        "split": "train",
        "source_label": SOURCE_LABEL,
        "cohort_count": len(selected),
        "train_csv_sha256": train_csv_sha256,
        "checkpoint_id": checkpoint_id,
        "cohort_jsonl_sha256": _sha256_bytes(lines),
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "result_conditioned_selection": False,
    }
    return selected, manifest, lines


def count_train_side_valid_unique(bridge: TasteComRecGCMulticlassBridge) -> dict[str, Any]:
    """Apply the fixed canonical/train-side strict-flip retention policy."""

    bridge._assert_idle()  # type: ignore[attr-defined]
    retained: list[str] = []
    for graph_hash, record in sorted(bridge.records.items()):
        lineages = bridge.lineage_occurrences.get(graph_hash)
        collision = bridge.graph_collision_payloads.get(graph_hash)
        if (
            record.valid_fullgraph
            and record.candidate
            and record.prediction in (0, 2)
            and record.graph_identity_sha256 == graph_hash
            and record.canonical_graph
            and type(collision) is dict
            and _identity_graph_sha256(collision) == graph_hash
            and lineages
            and all(type(key) is str and len(key) == 64 and count > 0 for key, count in lineages.items())
        ):
            retained.append(graph_hash)
    return {
        "schema_version": "tastemolnet_t14_train_valid_unique_v1",
        "policy": VALID_UNIQUE_POLICY,
        "valid_unique_rule_count": len(retained),
        "valid_unique_rule_hashes_sha256": _sha256_bytes(_canonical_bytes(retained)),
        "lineage_error_count": 0,
        "pred_before": SOURCE_LABEL,
        "pred_after_condition": "pred_after != 1",
        "split": "train",
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }


def resource_cap_decision(*, completed_step: int, valid_unique_rule_count: int) -> dict[str, Any]:
    if completed_step not in {M_MAX, M_FALLBACK_MAX}:
        raise TasteComRecGCFullError("Taste T14 stop decision is off cadence")
    if valid_unique_rule_count >= MIN_VALID_UNIQUE_RULES:
        return {
            "state": "STOP_AND_POSTPROCESS",
            "m_configured_max": M_MAX,
            "m_fallback_max": M_FALLBACK_MAX,
            "m_effective": completed_step,
            "resource_cap_used": True,
            "early_stop_used": False,
            "stop_reason": (
                "RESOURCE_CAP_20K_VALID_UNIQUE_PASS"
                if completed_step == M_MAX
                else "FALLBACK_CAP_25K_VALID_UNIQUE_PASS"
            ),
        }
    if completed_step == M_MAX:
        return {
            "state": "EXTEND_ONCE_TO_25K",
            "m_configured_max": M_MAX,
            "m_fallback_max": M_FALLBACK_MAX,
            "m_effective": completed_step,
            "resource_cap_used": True,
            "early_stop_used": False,
            "stop_reason": "20K_INSUFFICIENT_VALID_UNIQUE_RULES",
        }
    return {
        "state": "SCIENTIFIC_FAILED_INSUFFICIENT_VALID_RULES",
        "m_configured_max": M_MAX,
        "m_fallback_max": M_FALLBACK_MAX,
        "m_effective": completed_step,
        "resource_cap_used": True,
        "early_stop_used": False,
        "stop_reason": "25K_INSUFFICIENT_VALID_UNIQUE_RULES",
    }


def _checkpoint_receipt_path(path: Path) -> Path:
    return path.with_suffix(".json")


def _write_checkpoint(
    *,
    module: Any,
    bridge: TasteComRecGCMulticlassBridge,
    loop_state: Any,
    parameters: TasteComRecGCFullParameters,
    path: Path,
) -> dict[str, Any]:
    step_parameters = replace(parameters, checkpoint_step=int(loop_state.completed_step))
    step_parameters.validate()
    loaded = _write_reload_checkpoint(
        module=module,
        bridge=bridge,
        loop_state=loop_state,
        parameters=step_parameters,  # structural protocol is shared with T9
        path=path,
    )
    evidence = {
        **dict(loaded["evidence"]),
        "schema_version": "tastemolnet_t14_checkpoint_v1",
        "checkpoint_path": str(path),
        "checkpoint_persisted_in_output": True,
        "written_at": _utc_now(),
    }
    _atomic_json(_checkpoint_receipt_path(path), evidence)
    return {"payload": loaded["payload"], "evidence": evidence}


def _load_latest_checkpoint(
    checkpoint_root: Path,
    *,
    parameters: TasteComRecGCFullParameters,
) -> dict[str, Any] | None:
    candidates = sorted(checkpoint_root.glob("checkpoint-*.pt"))
    if not candidates:
        return None
    path = candidates[-1]
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_size <= 0
    ):
        raise TasteComRecGCFullError("Taste T14 checkpoint is not a private file")
    receipt_path = _checkpoint_receipt_path(path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        type(receipt) is not dict
        or receipt.get("schema_version") != "tastemolnet_t14_checkpoint_v1"
        or receipt.get("checkpoint_path") != str(path)
        or receipt.get("checkpoint_sha256") != _sha256_file(path)
        or receipt.get("checkpoint_persisted_in_output") is not True
    ):
        raise TasteComRecGCFullError("Taste T14 checkpoint receipt changed")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            payload = _torch_load_handle(handle)
        if os.fstat(descriptor) != info:
            raise TasteComRecGCFullError("Taste T14 checkpoint identity changed")
    finally:
        os.close(descriptor)
    completed = payload.get("completed_step") if type(payload) is dict else None
    if (
        type(payload) is not dict
        or payload.get("schema_version") != "tastemolnet_comrecgc_smoke_checkpoint_v1"
        or type(completed) is not int
        or completed % CHECK_INTERVAL != 0
        or not CHECK_INTERVAL <= completed <= M_FALLBACK_MAX
        or payload.get("total_steps") != M_FALLBACK_MAX
        or payload.get("parameters")
        != asdict(replace(parameters, checkpoint_step=completed))
    ):
        raise TasteComRecGCFullError("Taste T14 checkpoint payload changed")
    return {"payload": payload, "evidence": receipt}


def _progress(
    output_root: Path,
    *,
    phase: str,
    completed_step: int,
    cohort_count: int,
    valid_unique_rule_count: int | None = None,
) -> None:
    _atomic_json(
        output_root / "progress.json",
        {
            "schema_version": "tastemolnet_t14_progress_v1",
            "status": "RUNNING",
            "phase": phase,
            "completed_step": completed_step,
            "m_configured_max": M_MAX,
            "m_fallback_max": M_FALLBACK_MAX,
            "cohort_count": cohort_count,
            "valid_unique_rule_count": valid_unique_rule_count,
            "pid": os.getpid(),
            "updated_at": _utc_now(),
        },
    )


def _initialize_full_source_graphs(
    *,
    checkpoint_payloads: Mapping[str, bytes],
    source_rows: Sequence[Any],
    graph_schema: Any,
    device: str,
) -> tuple[list[Any], list[Mapping[str, Any]], Any, dict[str, Any]]:
    """Reopen every frozen cohort row, including duplicate molecules."""

    records = [encode_taste_source_graph(row, graph_schema) for row in source_rows]
    graphs: list[Any] = []
    parent_ids: list[str] = []
    for index, (row, record) in enumerate(zip(source_rows, records, strict=True)):
        graph = taste_record_to_pyg(record, origin_index=index)
        graph.comrecgc_node_origin = graph.gcf_node_origin.clone()
        graph.comrecgc_source_index = index
        graph.comrecgc_parent_id = f"private-cohort-{index:06d}"
        graphs.append(graph)
        parent_ids.append(str(row.molecule_id))
    adapter = TasteFrozenGINENativeAdapter(
        checkpoint_payloads,
        source_records=records,
        graph_schema=graph_schema,
        device=device,
    )
    predictions: list[int] = []
    identities: list[str] = []
    for offset in range(0, len(graphs), 128):
        chunk = graphs[offset : offset + 128]
        scored = adapter.score(chunk)
        if any(not valid for valid in scored.valid_fullgraphs):
            raise TasteComRecGCFullError("Taste T14 cohort graph failed exact replay")
        predictions.extend(scored.predictions)
        identities.extend(
            canonical_attributed_graph(
                graph,
                feature_atomic_numbers=graph_schema.feature_atomic_numbers,
            ).graph_identity_sha256
            for graph in chunk
        )
    if len(predictions) != len(graphs) or any(
        prediction != SOURCE_LABEL for prediction in predictions
    ):
        raise TasteComRecGCFullError("Taste T14 cohort prediction changed on replay")
    evidence = {
        "schema_version": "tastemolnet_t14_source_cohort_v1",
        "source_split": "train",
        "source_label": SOURCE_LABEL,
        "source_count": len(graphs),
        "source_cohort_sha256": _sha256_bytes(
            _canonical_bytes(
                [
                    {"parent_id": parent_id, "canonical_graph_hash": graph_hash}
                    for parent_id, graph_hash in zip(parent_ids, identities, strict=True)
                ]
            )
        ),
        "parent_ids_unique": len(set(parent_ids)) == len(parent_ids),
        "canonical_graph_dedup_applied": False,
        "duplicate_graph_row_count": len(identities) - len(set(identities)),
        "duplicate_graph_rows_retained": True,
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    return graphs, records, adapter, evidence


def run_t14_full(
    *,
    inputs: Any,
    output_root: str | Path,
    resume: bool,
) -> dict[str, Any]:
    """Run or resume the train-only native full generation and postprocessing."""

    from src.baselines.comrecgc.generation_loop import run_generation_loop
    from src.baselines.comrecgc.runtime import lineage_neighbor_wrapper, reset_official_state
    from src.baselines.gcfexplainer_mutagenicity_adapter import (
        GraphRecordDataset,
        graph_lineage_neighbor_wrapper,
    )

    root = Path(output_root)
    if not root.is_absolute() or Path(os.path.abspath(root)) != root:
        raise TasteComRecGCFullError("Taste T14 output root must be normalized")
    if root.exists() and not resume:
        raise FileExistsError(f"Taste T14 output already exists: {root}")
    if not root.exists():
        root.mkdir(parents=True, mode=0o700)
        _fsync_dir(root.parent)
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(mode=0o700, exist_ok=True)
    authority = inputs.revalidate()
    train = authority["train"]
    loaded_train = load_train_rows(
        inputs.train_file.read_bytes(),
        source_path=Path(train["path"]),
        expected_num_records=int(train["num_records"]),
        expected_label_counts=train["label_counts"],
    )
    checkpoint_payloads = {
        name: inputs.checkpoint_payloads[name]
        for name in (
            "model.pt",
            "model_card.json",
            "feature_schema.json",
            "label_map.json",
            "split_manifest.json",
            "test_evaluation_status.json",
            "temperature_scaling.json",
        )
    }
    true_sweet = list(loaded_train.sweet_rows)
    encoded_all = [
        encode_taste_source_graph(row, loaded_train.schema) for row in true_sweet
    ]
    graphs_all = [
        taste_record_to_pyg(record, origin_index=index)
        for index, record in enumerate(encoded_all)
    ]
    cohort_adapter = TasteFrozenGINENativeAdapter(
        checkpoint_payloads,
        source_records=encoded_all,
        graph_schema=loaded_train.schema,
        device="cuda:0",
    )
    predictions: list[int] = []
    source_probabilities: list[float] = []
    graph_hashes: list[str] = []
    for offset in range(0, len(graphs_all), 128):
        graphs = graphs_all[offset : offset + 128]
        scored = cohort_adapter.score(graphs)
        predictions.extend(scored.predictions)
        source_probabilities.extend(
            float(row[SOURCE_LABEL]) for row in scored.probabilities.tolist()
        )
        for graph in graphs:
            graph_hashes.append(
                canonical_attributed_graph(
                    graph,
                    feature_atomic_numbers=loaded_train.schema.feature_atomic_numbers,
                ).graph_identity_sha256
            )
    cohort, cohort_manifest, cohort_bytes = build_full_train_correct_source_cohort(
        true_sweet_rows=true_sweet,
        predictions=predictions,
        source_probabilities=source_probabilities,
        canonical_graph_hashes=graph_hashes,
        train_csv_sha256=str(train["sha256"]),
        checkpoint_id=str(authority["checkpoint"]["checkpoint_id"]),
    )
    existing_cohort = root / "cohort.jsonl"
    existing_manifest = root / "cohort_manifest.json"
    if existing_cohort.exists() or existing_manifest.exists():
        if (
            existing_cohort.read_bytes() != cohort_bytes
            or json.loads(existing_manifest.read_text(encoding="utf-8"))
            != cohort_manifest
        ):
            raise TasteComRecGCFullError("Taste T14 cohort changed on resume")
    else:
        _atomic_write(existing_cohort, cohort_bytes)
        _atomic_json(existing_manifest, cohort_manifest)
    selected_ids = {row["parent_id"] for row in cohort}
    selected_rows = [row for row in true_sweet if row.molecule_id in selected_ids]
    selected_rows.sort(key=lambda row: row.molecule_id)
    if [row.molecule_id for row in selected_rows] != [row["parent_id"] for row in cohort]:
        raise TasteComRecGCFullError("Taste T14 cohort/source row order changed")
    parameters = TasteComRecGCFullParameters(
        source_pool=len(selected_rows), source_count=len(selected_rows)
    ).validate()
    _progress(root, phase="COHORT_FROZEN", completed_step=0, cohort_count=len(cohort))

    modules = inputs.official.modules
    module = modules["comrecgc"]
    source_graphs, _records, adapter, source_evidence = _initialize_full_source_graphs(
        checkpoint_payloads=checkpoint_payloads,
        source_rows=selected_rows,
        graph_schema=loaded_train.schema,
        device="cuda:0",
    )
    dataset = GraphRecordDataset(
        source_graphs, num_features=len(loaded_train.schema.feature_atomic_numbers)
    )
    reset_official_state(
        module,
        candidate_capacity=parameters.candidate_capacity,
        sample_size=parameters.sample_size,
    )
    import torch

    module.input_graphs_covered = torch.zeros(
        parameters.source_count, dtype=torch.float32
    )
    bridge = TasteComRecGCMulticlassBridge(
        adapter=adapter,
        feature_atomic_numbers=loaded_train.schema.feature_atomic_numbers,
    )

    def combined_lineage(original: Any) -> Any:
        return lineage_neighbor_wrapper(graph_lineage_neighbor_wrapper(original))

    importance_args = {
        "schema_version": "tastemolnet_comrecgc_gine_distance_v1",
        "classifier": "frozen_calibrated_three_class_gine",
        "distance_embedding": "frozen_gine_graph_hidden",
        "num_classes": 3,
        "source_label": SOURCE_LABEL,
    }
    latest = _load_latest_checkpoint(checkpoint_root, parameters=parameters)
    with bridge.installed(module, neighbor_wrapper=combined_lineage):
        state = None
        if latest is not None:
            state = _restore_reload_checkpoint(
                module=module, bridge=bridge, loaded=latest
            )
        else:
            _seed_all(parameters.seed)
        completed = int(state.completed_step) if state is not None else 0
        for target in range(
            ((completed // CHECK_INTERVAL) + 1) * CHECK_INTERVAL,
            M_MAX + 1,
            CHECK_INTERVAL,
        ):
            state = run_generation_loop(
                module,
                input_graphs=dataset,
                importance_args=importance_args,
                teleport_probability=parameters.teleport_probability,
                max_steps=target,
                heads=parameters.heads,
                initial_state=state,
            )
            checkpoint_path = checkpoint_root / f"checkpoint-{target:06d}.pt"
            _write_checkpoint(
                module=module,
                bridge=bridge,
                loop_state=state,
                parameters=parameters,
                path=checkpoint_path,
            )
            valid = count_train_side_valid_unique(bridge)
            _progress(
                root,
                phase="GENERATION",
                completed_step=target,
                cohort_count=len(cohort),
                valid_unique_rule_count=int(valid["valid_unique_rule_count"]),
            )
        if state is None:
            raise TasteComRecGCFullError("Taste T14 generation produced no state")
        completed_now = int(state.completed_step)
        valid = count_train_side_valid_unique(bridge)
        decision = resource_cap_decision(
            completed_step=completed_now,
            valid_unique_rule_count=int(valid["valid_unique_rule_count"]),
        )
        if decision["state"] == "EXTEND_ONCE_TO_25K":
            state = run_generation_loop(
                module,
                input_graphs=dataset,
                importance_args=importance_args,
                teleport_probability=parameters.teleport_probability,
                max_steps=M_FALLBACK_MAX,
                heads=parameters.heads,
                initial_state=state,
            )
            _write_checkpoint(
                module=module,
                bridge=bridge,
                loop_state=state,
                parameters=parameters,
                path=checkpoint_root / f"checkpoint-{M_FALLBACK_MAX:06d}.pt",
            )
            valid = count_train_side_valid_unique(bridge)
            decision = resource_cap_decision(
                completed_step=M_FALLBACK_MAX,
                valid_unique_rule_count=int(valid["valid_unique_rule_count"]),
            )
        _atomic_json(root / "valid_unique.json", valid)
        _atomic_json(root / "resource_cap_receipt.json", decision)
        if decision["state"] != "STOP_AND_POSTPROCESS":
            _progress(
                root,
                phase="SCIENTIFIC_FAILED_INSUFFICIENT_VALID_RULES",
                completed_step=int(decision["m_effective"]),
                cohort_count=len(cohort),
                valid_unique_rule_count=int(valid["valid_unique_rule_count"]),
            )
            raise TasteComRecGCFullError(
                "SCIENTIFIC_FAILED_INSUFFICIENT_VALID_RULES"
            )
        _progress(
            root,
            phase="POSTPROCESS",
            completed_step=int(decision["m_effective"]),
            cohort_count=len(cohort),
            valid_unique_rule_count=int(valid["valid_unique_rule_count"]),
        )
        common = _common_recourse_summary(
            modules=modules,
            module=module,
            bridge=bridge,
            source_graphs=source_graphs,
            adapter=adapter,
            parameters=parameters,
        )
        bridge_evidence = bridge.report()
    inputs.revalidate()
    final = {
        "schema_version": "tastemolnet_t14_comrecgc_full_v1",
        "status": "PASS",
        "dataset": DATASET,
        "method": METHOD,
        "stage": STAGE,
        "cohort_manifest_sha256": _sha256_file(existing_manifest),
        "cohort_jsonl_sha256": _sha256_bytes(cohort_bytes),
        "cohort_count": len(cohort),
        "source_cohort": source_evidence,
        "valid_unique": valid,
        "resource_cap": decision,
        "bridge": bridge_evidence,
        "common_recourse": common,
        "same_frozen_three_class_gine": True,
        "rf_oracle_used": False,
        "train_loaded": True,
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "completed_at": _utc_now(),
    }
    _atomic_json(root / "generation_manifest.json", final)
    _atomic_write(root / "PASS", f"{PASS_MARKER}\n".encode("utf-8"))
    _atomic_json(
        root / "progress.json",
        {
            "schema_version": "tastemolnet_t14_progress_v1",
            "status": "PASS",
            "phase": "GENERATION_AND_TRAIN_POSTPROCESS_COMPLETE",
            "completed_step": int(decision["m_effective"]),
            "m_configured_max": M_MAX,
            "m_fallback_max": M_FALLBACK_MAX,
            "cohort_count": len(cohort),
            "valid_unique_rule_count": int(valid["valid_unique_rule_count"]),
            "pid": os.getpid(),
            "updated_at": _utc_now(),
        },
    )
    return final


def validate_t14_full_output(output_root: str | Path) -> dict[str, Any]:
    """Independently reopen the bounded full-generation closure."""

    root = Path(output_root)
    if not root.is_absolute() or Path(os.path.abspath(root)) != root:
        raise TasteComRecGCFullError("Taste T14 verification root must be normalized")
    required = {
        "PASS",
        "cohort.jsonl",
        "cohort_manifest.json",
        "generation_manifest.json",
        "progress.json",
        "resource_cap_receipt.json",
        "valid_unique.json",
    }
    if not root.is_dir() or not required.issubset(
        {path.name for path in root.iterdir()}
    ):
        raise TasteComRecGCFullError("Taste T14 terminal closure is incomplete")
    for name in required:
        info = (root / name).lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise TasteComRecGCFullError(f"Taste T14 {name} is not a regular file")
    if (root / "PASS").read_bytes() != f"{PASS_MARKER}\n".encode("utf-8"):
        raise TasteComRecGCFullError("Taste T14 PASS marker changed")
    manifest = json.loads((root / "generation_manifest.json").read_text("utf-8"))
    cohort_manifest = json.loads((root / "cohort_manifest.json").read_text("utf-8"))
    resource = json.loads((root / "resource_cap_receipt.json").read_text("utf-8"))
    valid = json.loads((root / "valid_unique.json").read_text("utf-8"))
    progress = json.loads((root / "progress.json").read_text("utf-8"))
    cohort_bytes = (root / "cohort.jsonl").read_bytes()
    effective = resource.get("m_effective") if type(resource) is dict else None
    latest = root / "checkpoints" / f"checkpoint-{effective:06d}.pt" if type(effective) is int else root
    if (
        type(manifest) is not dict
        or manifest.get("schema_version") != "tastemolnet_t14_comrecgc_full_v1"
        or manifest.get("status") != "PASS"
        or manifest.get("stage") != STAGE
        or manifest.get("train_loaded") is not True
        or manifest.get("validation_loaded") is not False
        or manifest.get("calibration_loaded") is not False
        or manifest.get("test_loaded") is not False
        or manifest.get("rf_oracle_used") is not False
        or manifest.get("cohort_manifest_sha256") != _sha256_file(root / "cohort_manifest.json")
        or manifest.get("cohort_jsonl_sha256") != _sha256_bytes(cohort_bytes)
        or manifest.get("resource_cap") != resource
        or manifest.get("valid_unique") != valid
        or type(cohort_manifest) is not dict
        or cohort_manifest.get("status") != "PASS"
        or cohort_manifest.get("policy") != COHORT_POLICY
        or cohort_manifest.get("cohort_jsonl_sha256") != _sha256_bytes(cohort_bytes)
        or resource.get("state") != "STOP_AND_POSTPROCESS"
        or effective not in {M_MAX, M_FALLBACK_MAX}
        or valid.get("valid_unique_rule_count", 0) < MIN_VALID_UNIQUE_RULES
        or progress.get("status") != "PASS"
        or progress.get("completed_step") != effective
        or not latest.is_file()
        or _checkpoint_receipt_path(latest).is_file() is not True
    ):
        raise TasteComRecGCFullError("Taste T14 terminal science closure changed")
    inventory = {
        str(path.relative_to(root)): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    return {
        "schema_version": "tastemolnet_t14_independent_verification_v1",
        "status": "PASS",
        "marker": PASS_MARKER,
        "output_root": str(root),
        "m_effective": effective,
        "valid_unique_rule_count": valid["valid_unique_rule_count"],
        "inventory_sha256": _sha256_bytes(_canonical_bytes(inventory)),
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "verified_at": _utc_now(),
    }


__all__ = [
    "CHECK_INTERVAL",
    "COHORT_POLICY",
    "M_FALLBACK_MAX",
    "M_MAX",
    "MIN_VALID_UNIQUE_RULES",
    "PASS_MARKER",
    "STAGE",
    "TasteComRecGCFullError",
    "TasteComRecGCFullParameters",
    "VALID_UNIQUE_POLICY",
    "build_full_train_correct_source_cohort",
    "count_train_side_valid_unique",
    "resource_cap_decision",
    "run_t14_full",
    "validate_t14_full_output",
]
