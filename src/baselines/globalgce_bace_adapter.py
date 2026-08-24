"""BACE facade over the established project GlobalGCE molecular adapter."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .globalgce_mutagenicity_adapter import (
    NativeGeneratorProtocol,
    OfficialGlobalGCEMutagenicityGenerator,
    PoolBuildConfig,
    TrainParent,
    TeacherProtocol,
    _load_general_train_rows,
    audit_mutagenicity_train_pool,
    build_mutagenicity_train_pool,
)
from src.baselines.bace_gnn_baseline_contracts import (
    assert_gine_clean_manifest,
    oracle_provenance,
    validate_bace_frozen_gine,
)
from src.baselines.globalgce_bace_native_rules import (
    GlobalGCENativeRule,
    validate_official_globalgce_root,
)
from src.eval.bace_frozen_gnn_contracts import (
    atomic_json,
    atomic_jsonl,
    atomic_marker,
    file_identity,
    sha256_file,
    stable_sha256,
    utc_now,
)
from src.baselines.globalgce_resumable import (
    validate_exact_top_k_proof_identity,
)
from src.baselines.globalgce_mining_adoption import (
    validate_globalgce_gspan_adoption_proof,
)


DATASET_NAME = "BACE"
EXPECTED_TRAIN_SOURCE_COUNT = 360
EXPECTED_NATIVE_TRAIN_COUNT = 869
EXPECTED_NATIVE_INPUT_TRAIN_COUNT = 959
EXPECTED_TRAIN_TARGET_COUNT = 509
EXPECTED_VALIDATION_COUNT = 162
EXPECTED_VALIDATION_SOURCE_COUNT = 92
EXPECTED_VALIDATION_TARGET_COUNT = 70


@dataclass(frozen=True, slots=True)
class BACEGlobalGCETrainContract:
    source_parents: tuple[TrainParent, ...]
    native_train_parent_ids: tuple[str, ...]
    audit: dict[str, Any]


def _json_object(path: Path, *, description: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return payload


def _require_artifact(
    manifest: Mapping[str, Any],
    *,
    section: str,
    path: Path,
) -> dict[str, Any]:
    declared = dict((manifest.get(section) or {}).get(path.name) or {})
    declared_size = declared.get("bytes", declared.get("size", -1))
    if (
        declared.get("sha256") != sha256_file(path)
        or int(declared_size) != path.stat().st_size
    ):
        raise ValueError(f"BACE artifact manifest/hash closure failed: {path.name}")
    return file_identity(path)


def _read_source_manifest(path_like: str | Path) -> list[dict[str, Any]]:
    path = Path(path_like).expanduser().resolve(strict=True)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"BACE source manifest row {line_number} is not an object")
            parent_id = str(row.get("molecule_id") or "").strip()
            smiles = str(row.get("canonical_smiles") or "").strip()
            if not parent_id or parent_id in seen or not smiles:
                raise ValueError(
                    f"BACE source manifest has missing/duplicate identity at row {line_number}"
                )
            split = str(row.get("split") or "").strip().lower()
            label = int(row.get("label", -1))
            gnn_label = int(row.get("gnn_label", -1))
            if split not in {"train", "val"}:
                raise ValueError("BACE source manifest contains calibration/test data")
            if (
                label not in {0, 1}
                or gnn_label != (0 if label == 1 else 1)
                or int(row.get("source_label", -1)) != 1
                or int(row.get("target_label", -1)) != 0
            ):
                raise ValueError("BACE source manifest label mapping is not frozen")
            seen.add(parent_id)
            rows.append(row)
    return rows


def validate_bace_globalgce_terminal_artifacts(
    output_dir: str | Path, *, require_exact_top_k: bool
) -> dict[str, Any]:
    """Re-open the train publications and gate the final PASS marker."""

    root = Path(output_dir).expanduser().resolve(strict=True)
    training_path = root / "training_summary.json"
    summary_path = root / "summary.json"
    manifest_path = root / "run_manifest.json"
    complete_path = root / "_RUN_COMPLETE.json"
    training = _json_object(
        training_path, description="BACE GlobalGCE training summary"
    )
    summary = _json_object(summary_path, description="BACE GlobalGCE summary")
    manifest = _json_object(
        manifest_path, description="BACE GlobalGCE run manifest"
    )
    complete = _json_object(
        complete_path, description="BACE GlobalGCE completion gate"
    )
    training_identity = file_identity(training_path)
    summary_identity = file_identity(summary_path)
    manifest_identity = file_identity(manifest_path)
    if (
        complete.get("training_summary") != training_identity
        or complete.get("summary") != summary_identity
        or complete.get("run_manifest") != manifest_identity
        or manifest.get("training_summary") != training_identity
        or manifest.get("summary") != summary_identity
        or manifest.get("status") != "PASS"
        or manifest.get("run_complete") is not True
        or complete.get("status") != "PASS"
        or complete.get("run_complete") is not True
        or manifest.get("oracle_backend") != "gnn"
        or manifest.get("classifier_family") != "gine"
        or manifest.get("rf_oracle_used") is not False
        or summary.get("oracle_backend") != "gnn"
        or summary.get("classifier_family") != "gine"
        or summary.get("rf_oracle_used") is not False
    ):
        raise RuntimeError("BACE GlobalGCE terminal artifact/hash closure failed")
    exact_proof: dict[str, Any] | None = None
    if require_exact_top_k:
        exact_proof = validate_exact_top_k_proof_identity(
            training.get("gspan_exact_top_k_proof") or {}
        )
        for payload in (summary, manifest, complete):
            if validate_exact_top_k_proof_identity(
                payload.get("gspan_exact_top_k_proof") or {}
            ) != exact_proof:
                raise RuntimeError("BACE GlobalGCE exact proof binding changed")
    adoption_proof: dict[str, Any] | None = None
    if training.get("gspan_adoption_identity") is not None:
        raw_adoption = training.get("gspan_adoption_identity") or {}
        selected = raw_adoption.get("selected_top20") or {}
        proof_path = Path(str(selected.get("path") or "")).parent / "adoption_proof.json"
        adoption_proof = validate_globalgce_gspan_adoption_proof(proof_path)
        if adoption_proof != raw_adoption:
            raise RuntimeError("BACE GlobalGCE adoption proof binding changed")
        for payload in (summary, manifest, complete):
            if payload.get("gspan_adoption_identity") != adoption_proof:
                raise RuntimeError("BACE GlobalGCE adoption identity was not propagated")
    return {
        "status": "PASS",
        "training_summary": training_identity,
        "summary": summary_identity,
        "run_manifest": manifest_identity,
        "gspan_exact_top_k_proof": exact_proof,
        "gspan_adoption_identity": adoption_proof,
    }


def audit_bace_globalgce_train_contract(
    *,
    source_manifest: str | Path,
    native_train_csv: str | Path,
) -> BACEGlobalGCETrainContract:
    """Bind the 959-row processed train CSV to the frozen 869-row GCF view.

    The GCF dataset preparation already froze teacher-consistent train/val
    membership.  GlobalGCE may reuse its exact 869 train IDs as the native
    vocabulary, but it must not load the 162 validation IDs or any
    calibration/test rows.
    """

    source_path = Path(source_manifest).expanduser().resolve(strict=True)
    native_path = Path(native_train_csv).expanduser().resolve(strict=True)
    source_run_path = source_path.parent / "run_manifest.json"
    source_summary_path = source_path.parent / "dataset_summary.json"
    native_run_path = native_path.parent / "run_manifest.json"
    native_summary_path = native_path.parent / "bace_dataset_summary.json"
    for required in (
        source_run_path,
        source_summary_path,
        native_run_path,
        native_summary_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Missing BACE frozen dataset contract: {required}")

    source_run = _json_object(source_run_path, description="BACE GCF run manifest")
    source_summary = _json_object(
        source_summary_path, description="BACE GCF dataset summary"
    )
    source_identity = _require_artifact(
        source_run, section="artifacts", path=source_path
    )
    source_summary_identity = _require_artifact(
        source_run, section="artifacts", path=source_summary_path
    )
    required_source_contract = {
        "dataset": "BACE",
        "adapter": "official_gcfexplainer_bace_project_data",
        "train_rows": EXPECTED_NATIVE_TRAIN_COUNT,
        "train_source_rows": EXPECTED_TRAIN_SOURCE_COUNT,
        "train_target_rows": EXPECTED_TRAIN_TARGET_COUNT,
        "val_rows": EXPECTED_VALIDATION_COUNT,
        "val_source_rows": EXPECTED_VALIDATION_SOURCE_COUNT,
        "val_target_rows": EXPECTED_VALIDATION_TARGET_COUNT,
        "generation_source_rows": EXPECTED_TRAIN_SOURCE_COUNT,
        "calibration_loaded": False,
        "test_loaded": False,
        "run_complete": True,
    }
    for key, expected in required_source_contract.items():
        if source_summary.get(key) != expected or source_run.get(key) != expected:
            raise ValueError(
                f"BACE GCF dataset contract mismatch for {key}: "
                f"summary={source_summary.get(key)!r}, "
                f"manifest={source_run.get(key)!r}, expected={expected!r}"
            )
    if source_summary.get("gnn_label_mapping") != {
        "project_1": 0,
        "project_0": 1,
    }:
        raise ValueError("BACE GCF GINE label mapping changed")

    source_rows = _read_source_manifest(source_path)
    train_rows = [row for row in source_rows if row["split"] == "train"]
    val_rows = [row for row in source_rows if row["split"] == "val"]
    source_rows_360 = [
        row
        for row in train_rows
        if int(row["label"]) == 1 and int(row["gnn_label"]) == 0
    ]
    target_rows_509 = [
        row
        for row in train_rows
        if int(row["label"]) == 0 and int(row["gnn_label"]) == 1
    ]
    val_source = [row for row in val_rows if int(row["label"]) == 1]
    val_target = [row for row in val_rows if int(row["label"]) == 0]
    actual_counts = (
        len(train_rows),
        len(source_rows_360),
        len(target_rows_509),
        len(val_rows),
        len(val_source),
        len(val_target),
    )
    expected_counts = (
        EXPECTED_NATIVE_TRAIN_COUNT,
        EXPECTED_TRAIN_SOURCE_COUNT,
        EXPECTED_TRAIN_TARGET_COUNT,
        EXPECTED_VALIDATION_COUNT,
        EXPECTED_VALIDATION_SOURCE_COUNT,
        EXPECTED_VALIDATION_TARGET_COUNT,
    )
    if actual_counts != expected_counts:
        raise ValueError(
            "BACE GCF source/train/validation count mismatch: "
            f"actual={actual_counts}, expected={expected_counts}"
        )
    train_ids = [str(row["molecule_id"]) for row in train_rows]
    val_ids = [str(row["molecule_id"]) for row in val_rows]
    source_ids = [str(row["molecule_id"]) for row in source_rows_360]
    for key, values in (
        ("train_ids_hash", train_ids),
        ("val_ids_hash", val_ids),
        ("generation_source_cohort_hash", source_ids),
    ):
        observed = stable_sha256(values)
        if source_summary.get(key) != observed or source_run.get(key) != observed:
            raise ValueError(f"BACE GCF ordered cohort hash mismatch: {key}")
    if set(train_ids) & set(val_ids):
        raise ValueError("BACE GCF train/validation molecule ID overlap")

    native_run = _json_object(native_run_path, description="BACE processed manifest")
    native_summary = _json_object(
        native_summary_path, description="BACE processed dataset summary"
    )
    native_identity = _require_artifact(
        native_run, section="files", path=native_path
    )
    native_summary_identity = _require_artifact(
        native_run, section="files", path=native_summary_path
    )
    if (
        native_run.get("dataset") != "BACE"
        or native_run.get("schema_version") != "bace_processed_manifest_v1"
        or native_summary.get("schema_version") != "bace_processed_v1"
        or native_run.get("dataset_fingerprint")
        != native_summary.get("dataset_fingerprint")
        or native_summary.get("split_seed") != 13
        or native_summary.get("split_counts")
        != {"train": 959, "val": 187, "calibration": 129, "test": 238}
    ):
        raise ValueError("BACE processed dataset/split manifest closure failed")
    native_parents = _load_general_train_rows(native_path)
    if len(native_parents) != EXPECTED_NATIVE_INPUT_TRAIN_COUNT:
        raise ValueError(
            "BACE processed train row count mismatch: "
            f"actual={len(native_parents)}, expected={EXPECTED_NATIVE_INPUT_TRAIN_COUNT}"
        )
    native_by_id = {parent.parent_id: parent for parent in native_parents}
    missing = sorted(set(train_ids) - set(native_by_id))
    leaked_val = sorted(set(val_ids) & set(native_by_id))
    if missing or leaked_val:
        raise ValueError(
            "BACE frozen train-ID join failed: "
            f"missing_train={missing[:5]}, leaked_val={leaked_val[:5]}"
        )
    source_by_id = {str(row["molecule_id"]): row for row in train_rows}
    for parent_id in train_ids:
        native = native_by_id[parent_id]
        frozen = source_by_id[parent_id]
        if (
            native.label != int(frozen["label"])
            or native.smiles != str(frozen["canonical_smiles"])
        ):
            raise ValueError(f"BACE frozen train row drift: {parent_id}")
    selected_native = [native_by_id[parent_id] for parent_id in train_ids]
    if Counter(parent.label for parent in selected_native) != Counter({0: 509, 1: 360}):
        raise ValueError("BACE frozen native train label counts changed")
    excluded_ids = sorted(set(native_by_id) - set(train_ids))
    if len(excluded_ids) != 90:
        raise ValueError("BACE teacher-consistency exclusion count changed")

    parents = tuple(
        sorted(
            (
                TrainParent(
                    str(row["molecule_id"]),
                    str(row["canonical_smiles"]),
                    1,
                    "train",
                )
                for row in source_rows_360
            ),
            key=lambda parent: parent.parent_id,
        )
    )
    return BACEGlobalGCETrainContract(
        source_parents=parents,
        native_train_parent_ids=tuple(sorted(train_ids)),
        audit={
            "schema_version": "bace_globalgce_train_contract_v2",
            "source_manifest": source_identity,
            "source_dataset_manifest": file_identity(source_run_path),
            "source_dataset_summary": source_summary_identity,
            "native_train_csv": native_identity,
            "native_dataset_manifest": file_identity(native_run_path),
            "native_dataset_summary": native_summary_identity,
            "native_input_train_rows": len(native_parents),
            "native_selected_train_rows": len(train_ids),
            "native_excluded_train_rows": len(excluded_ids),
            "native_selected_label_counts": {"0": 509, "1": 360},
            "source_parent_rows": len(parents),
            "validation_manifest_rows_audited_not_loaded": len(val_rows),
            "calibration_loaded": False,
            "test_loaded": False,
            "native_train_parent_ids_hash": stable_sha256(sorted(train_ids)),
            "excluded_train_parent_ids_hash": stable_sha256(excluded_ids),
        },
    )


class OfficialGlobalGCEBACEGenerator(OfficialGlobalGCEMutagenicityGenerator):
    """Run unchanged official GlobalGCE components with BACE dataset identity."""

    def __init__(
        self,
        official_root: str | Path,
        *,
        native_train_csv: str | Path,
        min_freq: int,
        frozen_gine_checkpoint: str | Path | None = None,
        native_train_parent_ids: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            official_root,
            native_train_csv=native_train_csv,
            dataset_name=DATASET_NAME,
            min_freq=int(min_freq),
            frozen_gine_checkpoint=frozen_gine_checkpoint,
            source_label=1,
            target_label=0,
            native_train_parent_ids=native_train_parent_ids,
        )


def build_bace_train_pool(
    *,
    train_csv: str | Path,
    teacher_path: str | Path,
    official_root: str | Path,
    output_dir: str | Path,
    teacher: TeacherProtocol,
    generator: NativeGeneratorProtocol,
    config: PoolBuildConfig | None = None,
) -> dict[str, Any]:
    if config is not None and config.gspan_exact_top_k_pruning:
        raise ValueError(
            "BACE exact-top-k mining is forbidden on the historical RF route; "
            "use build_bace_frozen_gine_rule_pool instead."
        )
    return build_mutagenicity_train_pool(
        train_csv=train_csv,
        teacher_path=teacher_path,
        official_root=official_root,
        output_dir=output_dir,
        teacher=teacher,
        generator=generator,
        config=config,
        dataset_name=DATASET_NAME,
    )


def audit_bace_train_pool(
    run_dir: str | Path,
    *,
    train_csv: str | Path,
    expected_parent_count: int = EXPECTED_TRAIN_SOURCE_COUNT,
    expected_input_train_count: int | None = EXPECTED_TRAIN_SOURCE_COUNT,
    require_complete: bool = True,
) -> dict[str, Any]:
    return audit_mutagenicity_train_pool(
        run_dir,
        train_csv=train_csv,
        expected_parent_count=expected_parent_count,
        expected_input_train_count=expected_input_train_count,
        require_target_label_zero=True,
        require_unique_universe=True,
        forbid_calibration_test=True,
        require_complete=require_complete,
        dataset_name=DATASET_NAME,
    )


def build_bace_frozen_gine_rule_pool(
    *,
    source_manifest: str | Path,
    native_train_csv: str | Path,
    official_root: str | Path,
    gnn_checkpoint: str | Path,
    output_dir: str | Path,
    min_freq: int,
    config: PoolBuildConfig | None = None,
) -> dict[str, Any]:
    """Train and freeze native GlobalGCE rules against the one BACE GINE.

    This route deliberately does not call the historical RF-backed BACE pool
    builder.  The official LHS/RHS decoder sees a straight-through view of the
    exact frozen GINE, while final rules remain hard native transformations
    that are re-applied and re-scored during calibration/test verification.
    """

    resolved = config or PoolBuildConfig(expected_parent_count=EXPECTED_TRAIN_SOURCE_COUNT)
    if resolved.gspan_adoption_proof and resolved.gspan_exact_top_k_pruning:
        raise ValueError(
            "BACE GlobalGCE adoption and fresh exact-top-k mining are mutually exclusive"
        )
    if int(resolved.expected_parent_count) != EXPECTED_TRAIN_SOURCE_COUNT:
        raise ValueError("BACE GlobalGCE requires the frozen 360-parent train cohort")
    source_path = Path(source_manifest).expanduser().resolve(strict=True)
    native_path = Path(native_train_csv).expanduser().resolve(strict=True)
    if any(token in source_path.name.lower() for token in ("test", "calibration")):
        raise ValueError("BACE GlobalGCE source manifest must be train-only")
    if any(token in native_path.name.lower() for token in ("test", "calibration")):
        raise ValueError("BACE GlobalGCE native vocabulary CSV must be train-only")
    train_contract = audit_bace_globalgce_train_contract(
        source_manifest=source_path,
        native_train_csv=native_path,
    )
    parents = list(train_contract.source_parents)
    native_train_count = int(train_contract.audit["native_selected_train_rows"])
    source_dataset_manifest_path = source_path.parent / "run_manifest.json"
    if not source_dataset_manifest_path.is_file():
        raise FileNotFoundError(
            "BACE source cohort must be bound by its dataset run_manifest.json"
        )
    source_dataset_manifest = json.loads(
        source_dataset_manifest_path.read_text(encoding="utf-8")
    )
    declared_source = dict(
        (source_dataset_manifest.get("artifacts") or {}).get(
            source_path.name
        )
        or {}
    )
    if (
        source_dataset_manifest.get("run_complete") is not True
        or source_dataset_manifest.get("test_loaded") is not False
        or str(source_dataset_manifest.get("dataset") or "").lower() != "bace"
        or int(source_dataset_manifest.get("generation_source_rows") or -1)
        != EXPECTED_TRAIN_SOURCE_COUNT
        or declared_source.get("sha256") != sha256_file(source_path)
        or int(declared_source.get("bytes") or -1) != source_path.stat().st_size
    ):
        raise ValueError("BACE source cohort manifest/hash closure failed")
    checkpoint, card, _schema = validate_bace_frozen_gine(gnn_checkpoint)
    provenance = oracle_provenance(card, checkpoint)
    official_audit = validate_official_globalgce_root(official_root)
    generator = OfficialGlobalGCEBACEGenerator(
        official_root,
        native_train_csv=native_path,
        min_freq=int(min_freq),
        frozen_gine_checkpoint=checkpoint,
        native_train_parent_ids=train_contract.native_train_parent_ids,
    )
    adoption_identity = (
        validate_globalgce_gspan_adoption_proof(resolved.gspan_adoption_proof)
        if resolved.gspan_adoption_proof
        else None
    )
    root = Path(output_dir).expanduser().resolve(strict=False)
    fingerprint_payload = {
        "schema_version": "bace_globalgce_frozen_gine_rule_pool_v1",
        "source_manifest": file_identity(source_path),
        "source_dataset_manifest": file_identity(source_dataset_manifest_path),
        "native_train_csv": file_identity(native_path),
        "native_train_row_count": native_train_count,
        "native_input_train_row_count": int(
            train_contract.audit["native_input_train_rows"]
        ),
        "native_train_contract": train_contract.audit,
        "official_source_audit": official_audit,
        "native_generator": generator.config_identity(),
        "oracle": provenance,
        "source_parent_ids": [row.parent_id for row in parents],
        "config": {
            "seed": int(resolved.seed),
            "epochs": int(resolved.epochs),
            "top_k_native": int(resolved.top_k_native),
            "learning_rate": float(resolved.learning_rate),
            "dropout": float(resolved.dropout),
            "device": str(resolved.device),
            "min_freq": int(min_freq),
            "gspan_flush_every": int(resolved.gspan_flush_every),
            "gspan_max_in_memory_candidates": int(
                resolved.gspan_max_in_memory_candidates
            ),
            "gspan_exact_top_k_pruning": bool(
                resolved.gspan_exact_top_k_pruning
            ),
            "gspan_adoption_identity": adoption_identity,
        },
    }
    fingerprint = stable_sha256(fingerprint_payload)
    manifest_path = root / "run_manifest.json"
    if (root / "PASS").exists() or (root / "_RUN_COMPLETE.json").exists():
        raise FileExistsError(f"Completed BACE GlobalGCE rule pool is immutable: {root}")
    if root.exists() and any(root.iterdir()):
        if not bool(resolved.resume) or not manifest_path.is_file():
            raise FileExistsError("BACE GlobalGCE partial output requires explicit resume")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config_fingerprint") != fingerprint:
            raise ValueError("BACE GlobalGCE resume fingerprint mismatch")
    else:
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(
            manifest_path,
            {
                **fingerprint_payload,
                "dataset": "bace",
                "method": "GlobalGCE",
                "method_id": "globalgce",
                "stage": "TRAIN_CANDIDATE_GENERATION",
                "status": "RUNNING",
                "run_complete": False,
                "config_fingerprint": fingerprint,
                "oracle_backend": "gnn",
                "classifier_family": "gine",
                "rf_oracle_used": False,
                "calibration_loaded": False,
                "test_loaded": False,
                "created_at": utc_now(),
            },
        )

    before_checkpoint_hash = sha256_file(checkpoint / "model.pt")
    training_holder: dict[str, Any] = {}

    def record_training(summary: dict[str, Any]) -> None:
        if (
            summary.get("oracle_backend") != "gnn"
            or summary.get("classifier_family") != "gine"
            or summary.get("rf_oracle_used") is not False
        ):
            raise RuntimeError(
                "BACE GlobalGCE training summary is not frozen-GINE clean"
            )
        if resolved.gspan_exact_top_k_pruning:
            summary["gspan_exact_top_k_proof"] = (
                validate_exact_top_k_proof_identity(
                    summary.get("gspan_exact_top_k_proof") or {}
                )
            )
        if adoption_identity is not None:
            if summary.get("gspan_adoption_identity") != adoption_identity:
                raise RuntimeError(
                    "BACE GlobalGCE training adoption identity changed"
                )
        training_holder.clear()
        training_holder.update(summary)
        atomic_json(root / "training_summary.json", training_holder)

    result = generator.generate(
        parents,
        output_dir=root / "native",
        seed=int(resolved.seed),
        epochs=int(resolved.epochs),
        top_k_native=int(resolved.top_k_native),
        learning_rate=float(resolved.learning_rate),
        dropout=float(resolved.dropout),
        device=str(resolved.device),
        resume=bool(resolved.resume),
        generation_chunk_size=int(resolved.generation_chunk_size),
        generation_num_workers=int(resolved.generation_num_workers),
        memory_log_every_chunks=int(resolved.memory_log_every_chunks),
        gspan_flush_every=int(resolved.gspan_flush_every),
        gspan_max_in_memory_candidates=int(resolved.gspan_max_in_memory_candidates),
        gspan_adoption_proof=resolved.gspan_adoption_proof,
        **(
            {"gspan_exact_top_k_pruning": True}
            if resolved.gspan_exact_top_k_pruning
            else {}
        ),
        start_parent_offset=0,
        on_training_ready=record_training,
        on_chunk=None,
        rules_only=True,
    )
    training_holder.update(result.training_summary)
    if (
        training_holder.get("oracle_backend") != "gnn"
        or training_holder.get("classifier_family") != "gine"
        or training_holder.get("rf_oracle_used") is not False
    ):
        raise RuntimeError("BACE GlobalGCE result is not frozen-GINE clean")
    exact_top_k_proof: dict[str, Any] | None = None
    if resolved.gspan_exact_top_k_pruning:
        exact_top_k_proof = validate_exact_top_k_proof_identity(
            training_holder.get("gspan_exact_top_k_proof") or {}
        )
        training_holder["gspan_exact_top_k_proof"] = exact_top_k_proof
    if adoption_identity is not None:
        if training_holder.get("gspan_adoption_identity") != adoption_identity:
            raise RuntimeError("BACE GlobalGCE result lost adoption identity")
    catalog_path = root / "native" / "native_rule_catalog.jsonl"
    rows: list[dict[str, Any]] = []
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                GlobalGCENativeRule.from_payload(row)
                rows.append(row)
    if len(rows) < 20 or len({str(row["candidate_id"]) for row in rows}) != len(rows):
        raise RuntimeError("BACE GlobalGCE did not freeze twenty unique valid rules")
    after_checkpoint_hash = sha256_file(checkpoint / "model.pt")
    if after_checkpoint_hash != before_checkpoint_hash:
        raise RuntimeError("Frozen BACE GINE checkpoint bytes changed during GlobalGCE")
    model_state_path = root / "native" / "globalgce_model.pt"
    try:
        import torch

        state = torch.load(model_state_path, map_location="cpu")
        forbidden_state_keys = [
            str(key)
            for key in state
            if "gt_gnn" in str(key).lower() or "bridge" in str(key).lower()
        ]
    except ImportError as exc:  # pragma: no cover - AutoDL dependency
        raise RuntimeError("BACE GlobalGCE rule freeze requires PyTorch") from exc
    if forbidden_state_keys:
        raise RuntimeError(
            "GlobalGCE trainable checkpoint accidentally contains frozen classifier state"
        )
    atomic_jsonl(root / "candidate_universe.jsonl", rows)
    atomic_jsonl(
        root / "candidate_filter_audit.jsonl",
        [
            {
                "candidate_id": row["candidate_id"],
                "native_rule_index": row["rule"]["native_rule_index"],
                "accepted": True,
                "reason": "valid_native_lhs_rhs_rule",
                "oracle_backend": "gnn",
                "classifier_family": "gine",
                "rf_oracle_used": False,
            }
            for row in rows
        ],
    )
    training_holder.update(
        {
            "classifier_parameters_frozen": True,
            "classifier_checkpoint_hash_before": before_checkpoint_hash,
            "classifier_checkpoint_hash_after": after_checkpoint_hash,
            "classifier_checkpoint_unchanged": True,
            "trainable_checkpoint_classifier_keys": forbidden_state_keys,
        }
    )
    atomic_json(root / "training_summary.json", training_holder)
    training_summary_identity = file_identity(root / "training_summary.json")
    persisted_training = _json_object(
        root / "training_summary.json", description="BACE GlobalGCE training summary"
    )
    if resolved.gspan_exact_top_k_pruning:
        persisted_proof = validate_exact_top_k_proof_identity(
            persisted_training.get("gspan_exact_top_k_proof") or {}
        )
        if persisted_proof != exact_top_k_proof:
            raise RuntimeError("BACE GlobalGCE training proof binding changed")
    summary = {
        "status": "PASS",
        "run_complete": True,
        "source_parent_count": len(parents),
        "native_rule_count": int(training_holder.get("native_rule_count") or 0),
        "valid_native_rule_count": len(rows),
        "candidate_universe_hash": sha256_file(root / "candidate_universe.jsonl"),
        "oracle_checkpoint_hash": card["checkpoint_id"],
        "calibration_loaded": False,
        "test_loaded": False,
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "training_summary": training_summary_identity,
    }
    if exact_top_k_proof is not None:
        summary["gspan_exact_top_k_proof"] = exact_top_k_proof
    if adoption_identity is not None:
        summary["gspan_adoption_identity"] = adoption_identity
    atomic_json(root / "summary.json", summary)
    summary_identity = file_identity(root / "summary.json")
    manifest = {
        **fingerprint_payload,
        "dataset": "bace",
        "method": "GlobalGCE",
        "method_id": "globalgce",
        "stage": "TRAIN_CANDIDATE_GENERATION",
        "status": "PASS",
        "run_complete": True,
        "config_fingerprint": fingerprint,
        "action_kind": "lhs_rhs_graph_transformation_rule",
        "action_semantics": "native_lhs_to_rhs_attachment_aware_v1",
        **provenance,
        "candidate_universe_hash": summary["candidate_universe_hash"],
        "candidate_count": len(rows),
        "classifier_parameters_frozen": True,
        "classifier_checkpoint_unchanged": True,
        "selector_fitted_on_calibration": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "training_summary": training_summary_identity,
        "summary": summary_identity,
        "completed_at": utc_now(),
    }
    if exact_top_k_proof is not None:
        manifest["gspan_exact_top_k_proof"] = exact_top_k_proof
    if adoption_identity is not None:
        manifest["gspan_adoption_identity"] = adoption_identity
    assert_gine_clean_manifest(
        manifest, checkpoint_id=str(card["checkpoint_id"]), require_train_only=True
    )
    atomic_json(manifest_path, manifest)
    atomic_json(root / "oracle_provenance.json", provenance)
    run_manifest_identity = file_identity(manifest_path)
    complete = {
        **summary,
        "summary": summary_identity,
        "run_manifest": run_manifest_identity,
    }
    if adoption_identity is not None:
        complete["gspan_adoption_identity"] = adoption_identity
    atomic_json(root / "_RUN_COMPLETE.json", complete)

    # Re-open every upper-layer publication and revalidate the terminal exact
    # proof immediately before PASS.  Missing or modified proof bytes, or a
    # summary/manifest hash mismatch, therefore fail closed without PASS.
    validate_bace_globalgce_terminal_artifacts(
        root,
        require_exact_top_k=bool(resolved.gspan_exact_top_k_pruning),
    )
    atomic_marker(root / "PASS", "[BACE_GLOBALGCE_FROZEN_GINE_RULE_POOL_PASS]")
    return manifest


__all__ = [
    "BACEGlobalGCETrainContract",
    "DATASET_NAME",
    "EXPECTED_NATIVE_TRAIN_COUNT",
    "EXPECTED_TRAIN_SOURCE_COUNT",
    "OfficialGlobalGCEBACEGenerator",
    "audit_bace_globalgce_train_contract",
    "audit_bace_train_pool",
    "build_bace_frozen_gine_rule_pool",
    "build_bace_train_pool",
    "validate_bace_globalgce_terminal_artifacts",
]
