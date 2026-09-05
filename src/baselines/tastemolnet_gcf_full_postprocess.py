"""TasteMolNet T12 GCF paper-cell postprocess and terminal verifier.

This module is intentionally dataset specific.  The official train-only VRRW
generation is accepted only through its independent generation PASS.  Native
full graphs are then decoded with their source-lineage sidecars, calibration
alone freezes the global ordering, and held-out test bytes are opened only
after that durable freeze.  A separate invocation replays the frozen surface
and is the only writer of the paper-cell PASS marker.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from src.baselines.gcfexplainer_mutagenicity_adapter import (
    decode_generated_fullgraph,
)
from src.baselines.tastemolnet_gcf_candidate_store import (
    reopen_native_candidate_snapshot,
)
from src.baselines.tastemolnet_gcf_full_resume import (
    production_transition_contract_sha256,
    validate_checkpoint_identity,
)
from src.baselines.tastemolnet_gcf_full_verify import (
    GENERATION_PASS_MARKER,
    GENERATION_VERIFY_SCHEMA,
)
from src.utils.tastemolnet_t12_formal_profile_v1 import (
    FORMAL_PRODUCTION_CHECKPOINT_CURSORS,
)
from src.baselines.tastemolnet_gcf_smoke import (
    TasteFrozenGINENativeAdapter,
    _semantic_sha256,
    canonical_attributed_graph,
    encode_taste_source_graph,
    load_train_rows,
)
from src.baselines.tastemolnet_globalgce_full import (
    GINE_FILES,
    atomic_csv,
    atomic_json,
    atomic_jsonl,
    compute_standardized_metrics as _globalgce_metrics,
    read_json,
    read_json_value,
    read_jsonl,
    select_rules_on_calibration as _globalgce_selector,
    sha256_file,
    stable_sha256,
)
from src.baselines.tastemolnet_globalgce_smoke import FrozenTasteGINEScorer
from src.eval.four_by_four_registry import (
    PASS_STATUSES,
    audit_explicit_candidate,
)
from src.eval.node_wasserstein_distance import (
    MolCLRNodeWassersteinConfig,
    MolCLRNodeWassersteinDistance,
)
from src.eval.tastemolnet_ours_full import (
    ThresholdContract,
    load_prepared_split,
    load_threshold_contract,
)


STAGE = "T12_GCF_FULL"
DATASET = "TasteMolNet"
DATASET_ID = "tastemolnet"
METHOD = "GCFExplainer"
SOURCE_LABEL = 1
DESTINATION_LABELS = (0, 2)
NUM_CLASSES = 3
K_MAX = 20
MIN_CANDIDATES = 10
TABLE2_K = 10
DISTANCE_LINE = "MolCLR-Node-Wasserstein"
DISTANCE_NAMESPACE = "tastemolnet_gcf_full_wnode_v1"
CF_MODE = "strict_flip"
RUN_MANIFEST_SCHEMA = "tastemolnet_t12_final_run_manifest_v1"
VERIFY_SCHEMA = "tastemolnet_t12_terminal_verification_v1"
SELECTION_SCHEMA = "tastemolnet_t12_calibration_selection_v1"
CHECKPOINT_SCHEMA = "tastemolnet_t12_postprocess_checkpoint_v1"
CHUNK_MANIFEST_SCHEMA = "tastemolnet_t12_pair_chunk_manifest_v1"
CHUNK_INVENTORY_SCHEMA = "tastemolnet_t12_pair_chunk_inventory_v1"
CANDIDATE_MANIFEST_SCHEMA = "tastemolnet_t12_paper_candidate_pool_v1"
PASS_MARKER = "[TASTE_GCF_PASS]"
OFFICIAL_MAX_CANDIDATES = 3_778


class TasteGCFPostprocessError(RuntimeError):
    """T12 violated generation, split, full-graph, or terminal semantics."""


@dataclass(frozen=True, slots=True)
class InputAuthority:
    generation_root: Path
    generation_verification_root: Path
    generation_audit_sha256: str
    generation_pass_sha256: str
    candidate_manifest: Path
    candidate_manifest_sha256: str
    attempt_id: str
    generation_token: str
    transition_contract_sha256: str
    train_path: Path
    calibration_path: Path
    test_path: Path
    checkpoint_path: Path
    molclr_root: Path
    molclr_checkpoint: Path
    threshold_path: Path
    train_sha256: str
    calibration_sha256: str
    declared_test_sha256: str
    checkpoint_id: str
    dataset_hash: str
    split_manifest_sha256: str
    molclr_checkpoint_sha256: str
    temperature_calibration_hash: str
    feature_schema_hash: str
    feature_schema_file_sha256: str
    implementation_sha256: str
    threshold: ThresholdContract
    train_count: int
    train_label_counts: dict[str, int]

    def resume_identity(self) -> dict[str, Any]:
        return {
            "schema_version": "tastemolnet_t12_postprocess_identity_v1",
            "dataset": DATASET,
            "method": METHOD,
            "stage": STAGE,
            "generation_audit_sha256": self.generation_audit_sha256,
            "generation_pass_sha256": self.generation_pass_sha256,
            "candidate_manifest_sha256": self.candidate_manifest_sha256,
            "attempt_id": self.attempt_id,
            "generation_token": self.generation_token,
            "transition_contract_sha256": self.transition_contract_sha256,
            "train_sha256": self.train_sha256,
            "calibration_sha256": self.calibration_sha256,
            "declared_test_sha256": self.declared_test_sha256,
            "checkpoint_id": self.checkpoint_id,
            "dataset_hash": self.dataset_hash,
            "split_manifest_sha256": self.split_manifest_sha256,
            "molclr_checkpoint_sha256": self.molclr_checkpoint_sha256,
            "threshold_config_hash": self.threshold.config_hash,
            "temperature_calibration_hash": self.temperature_calibration_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "implementation_sha256": self.implementation_sha256,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_path(output: Path) -> Path:
    return output / "checkpoint.json"


def _write_checkpoint(
    output: Path,
    *,
    phase: str,
    identity: Mapping[str, Any],
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "stage": STAGE,
        "phase": phase,
        "resume_identity": dict(identity),
        "resume_identity_sha256": stable_sha256(dict(identity)),
        "detail": dict(detail or {}),
        "written_at": _utc_now(),
    }
    atomic_json(_checkpoint_path(output), payload)
    return payload


def _load_checkpoint(output: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    payload = read_json(_checkpoint_path(output))
    if (
        payload.get("schema_version") != CHECKPOINT_SCHEMA
        or payload.get("stage") != STAGE
        or payload.get("resume_identity") != dict(identity)
        or payload.get("resume_identity_sha256") != stable_sha256(dict(identity))
    ):
        raise TasteGCFPostprocessError("T12 postprocess resume identity changed")
    return payload


def _checkpoint_payloads(checkpoint: Path) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for name in GINE_FILES:
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise TasteGCFPostprocessError(f"frozen T3 GINE is missing {name}")
        payloads[name] = path.read_bytes()
    return payloads


def _threshold_payload(contract: ThresholdContract) -> dict[str, Any]:
    return {
        "thresholds": list(contract.values),
        "theta_star": contract.theta_star,
        "cost_cap": contract.cost_cap,
        "threshold_config_hash": contract.config_hash,
        "threshold_source": contract.source,
        "threshold_source_split": contract.source_split,
        "threshold_contract_file_sha256": contract.file_sha256,
        "threshold_shared_across_methods": True,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
    }


def _validate_generation_pass(
    generation_root: Path, verification_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    marker = verification_root / "GENERATION_PASS"
    audit_path = verification_root / "generation_verification.json"
    if marker.read_bytes() != (GENERATION_PASS_MARKER + "\n").encode():
        raise TasteGCFPostprocessError("T12 generation PASS bytes changed")
    audit = read_json(audit_path)
    run_path = generation_root / "run_identity.json"
    run = read_json(run_path)
    if (
        audit.get("schema_version") != GENERATION_VERIFY_SCHEMA
        or audit.get("status") != "GENERATION_PASS"
        or audit.get("passed") is not True
        or audit.get("stage") != STAGE
        or audit.get("marker") != GENERATION_PASS_MARKER
        or Path(str(audit.get("production_root"))) != generation_root
        or audit.get("run_identity_sha256") != sha256_file(run_path)
        or audit.get("checkpoint_cursors")
        not in (
            [10_000, 20_000],
            list(FORMAL_PRODUCTION_CHECKPOINT_CURSORS),
        )
        or audit.get("external_transition_store_exact_reopen") is not True
        or audit.get("compact_history_exact_reopen") is not True
        or audit.get("trace_10k_prefix_retained") is not True
        or audit.get("official_native_result_exact") is not True
        or audit.get("lossless_candidate_persistence") is not True
        or audit.get("generated_to_original_neurosed") is not True
        or audit.get("train_loaded") is not True
        or audit.get("calibration_loaded") is not False
        or audit.get("test_loaded") is not False
        or audit.get("test_used_for_selection") is not False
        or audit.get("rf_oracle_used") is not False
        or audit.get("independent_verifier") is not True
        or audit.get("paper_cell_pass") is not False
        or run.get("schema_version") != "tastemolnet_t12_gcf_generation_run_v1"
        or run.get("stage") != STAGE
        or run.get("purpose") != "production"
        or run.get("train_loaded") is not True
        or run.get("calibration_loaded") is not False
        or run.get("test_loaded") is not False
        or run.get("rf_oracle_used") is not False
    ):
        raise TasteGCFPostprocessError("T12 generation verification contract changed")
    return audit, run


def load_input_authority(
    *,
    generation_root: str | Path,
    generation_verification_root: str | Path,
    train_csv: str | Path,
    calibration_csv: str | Path,
    test_csv: str | Path,
    gnn_checkpoint: str | Path,
    molclr_root: str | Path,
    molclr_checkpoint: str | Path,
    threshold_contract: str | Path,
) -> InputAuthority:
    """Validate every pre-test identity without opening held-out test bytes."""

    generation = Path(generation_root).expanduser().resolve(strict=True)
    verification = Path(generation_verification_root).expanduser().resolve(
        strict=True
    )
    train = Path(train_csv).expanduser().resolve(strict=True)
    calibration = Path(calibration_csv).expanduser().resolve(strict=True)
    test = Path(test_csv).expanduser().absolute()  # lexical only until freeze
    checkpoint = Path(gnn_checkpoint).expanduser().resolve(strict=True)
    molclr_source = Path(molclr_root).expanduser().resolve(strict=True)
    molclr_ckpt = Path(molclr_checkpoint).expanduser().resolve(strict=True)
    threshold_path = Path(threshold_contract).expanduser().resolve(strict=True)
    threshold = load_threshold_contract(threshold_path)
    audit, run = _validate_generation_pass(generation, verification)
    identity = validate_checkpoint_identity(run.get("identity_template"))
    contract_sha = production_transition_contract_sha256(identity)
    candidate_manifest = Path(str(audit.get("candidate_manifest")))
    if (
        candidate_manifest.resolve(strict=True) != candidate_manifest
        or candidate_manifest.parent != generation / "native_candidates"
        or audit.get("candidate_manifest_sha256")
        != sha256_file(candidate_manifest)
        or run.get("transition_contract_sha256") != contract_sha
    ):
        raise TasteGCFPostprocessError("T12 generation candidate authority changed")

    payloads = _checkpoint_payloads(checkpoint)
    card = json.loads(payloads["model_card.json"].decode("utf-8"))
    checkpoint_id = sha256_file(checkpoint / "model.pt")
    if (
        card.get("dataset") not in {DATASET, DATASET_ID}
        or card.get("oracle_backend") != "gnn"
        or card.get("rf_oracle_used") is not False
        or str(card.get("backbone") or "").lower() != "gine"
        or card.get("num_classes") != NUM_CLASSES
        or card.get("source_label") != SOURCE_LABEL
        or card.get("checkpoint_id") != checkpoint_id
        or identity.get("model_checkpoint_sha256") != checkpoint_id
    ):
        raise TasteGCFPostprocessError("T12 T3 GINE authority changed")
    split = json.loads(payloads["split_manifest.json"].decode("utf-8"))
    files = split.get("files")
    roles = split.get("roles")
    if (
        split.get("schema_version") != "molecular_gnn_split_manifest_v1"
        or split.get("dataset") not in {DATASET, DATASET_ID}
        or type(files) is not dict
        or set(files) != {"train", "validation", "calibration", "test"}
        or type(roles) is not dict
        or roles.get("calibration") != "reserved_for_threshold_and_selector_only"
        or roles.get("test") != "frozen_model_final_quality_evaluation"
        or split.get("calibration_loaded_for_training") is not False
        or split.get("test_loaded_for_training") is not False
        or split.get("test_used_for_checkpoint_selection") is not False
    ):
        raise TasteGCFPostprocessError("T12 frozen split-role contract changed")
    declared: dict[str, str] = {}
    for role in ("train", "calibration", "test"):
        row = files.get(role)
        if type(row) is not dict or not _is_sha256(row.get("sha256")):
            raise TasteGCFPostprocessError(f"T12 split manifest lacks {role} SHA")
        declared[role] = str(row["sha256"])
    train_sha = sha256_file(train)
    calibration_sha = sha256_file(calibration)
    if (
        train_sha != declared["train"]
        or calibration_sha != declared["calibration"]
        or identity.get("train_split_sha256") != train_sha
    ):
        raise TasteGCFPostprocessError("T12 train/calibration bytes changed")
    manifest = split.get("train_manifest")
    if type(manifest) is not dict:
        raise TasteGCFPostprocessError("T12 split manifest lacks train authority")
    train_count = manifest.get("num_records")
    label_counts = manifest.get("label_counts")
    dataset_hash = str(manifest.get("dataset_fingerprint") or "")
    if (
        type(train_count) is not int
        or train_count <= 0
        or type(label_counts) is not dict
        or set(label_counts) != {"0", "1", "2"}
        or any(type(value) is not int or value <= 0 for value in label_counts.values())
        or sum(label_counts.values()) != train_count
        or not _is_sha256(dataset_hash)
    ):
        raise TasteGCFPostprocessError("T12 train authority is malformed")
    feature = json.loads(payloads["feature_schema.json"].decode("utf-8"))
    feature_schema_hash = str(feature.get("schema_sha256") or "")
    if not _is_sha256(feature_schema_hash):
        raise TasteGCFPostprocessError("T12 feature schema hash is absent")
    return InputAuthority(
        generation_root=generation,
        generation_verification_root=verification,
        generation_audit_sha256=sha256_file(
            verification / "generation_verification.json"
        ),
        generation_pass_sha256=sha256_file(verification / "GENERATION_PASS"),
        candidate_manifest=candidate_manifest,
        candidate_manifest_sha256=sha256_file(candidate_manifest),
        attempt_id=str(audit["attempt_id"]),
        generation_token=str(audit["generation_token"]),
        transition_contract_sha256=contract_sha,
        train_path=train,
        calibration_path=calibration,
        test_path=test,
        checkpoint_path=checkpoint,
        molclr_root=molclr_source,
        molclr_checkpoint=molclr_ckpt,
        threshold_path=threshold_path,
        train_sha256=train_sha,
        calibration_sha256=calibration_sha,
        declared_test_sha256=declared["test"],
        checkpoint_id=checkpoint_id,
        dataset_hash=dataset_hash,
        split_manifest_sha256=sha256_file(checkpoint / "split_manifest.json"),
        molclr_checkpoint_sha256=sha256_file(molclr_ckpt),
        temperature_calibration_hash=sha256_file(checkpoint / "temperature.json"),
        feature_schema_hash=feature_schema_hash,
        feature_schema_file_sha256=sha256_file(checkpoint / "feature_schema.json"),
        implementation_sha256=sha256_file(Path(__file__).resolve(strict=True)),
        threshold=threshold,
        train_count=train_count,
        train_label_counts={str(key): int(value) for key, value in label_counts.items()},
    )


def _origin_index(graph: Any, *, parent_count: int) -> int:
    value = getattr(graph, "gcf_origin_index", None)
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise TasteGCFPostprocessError("T12 native source lineage is malformed")
        value = value[0]
    if type(value) is not int or not 0 <= value < parent_count:
        raise TasteGCFPostprocessError("T12 native source lineage is out of range")
    return value


def _importance_parts(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TasteGCFPostprocessError("T12 native importance parts changed")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) for item in result):
        raise TasteGCFPostprocessError("T12 native importance is non-finite")
    return result


def _load_train_codec(authority: InputAuthority) -> tuple[Any, list[Any], list[Any]]:
    loaded = load_train_rows(
        authority.train_path.read_bytes(),
        source_path=authority.train_path,
        expected_num_records=authority.train_count,
        expected_label_counts=authority.train_label_counts,
    )
    cohort = read_jsonl(authority.generation_root / "cohort.jsonl")
    cohort_manifest = read_json(authority.generation_root / "cohort_manifest.json")
    if (
        sha256_file(authority.generation_root / "cohort.jsonl")
        != cohort_manifest.get("cohort_jsonl_sha256")
        or len(cohort) != cohort_manifest.get("cohort_count")
        or len(cohort) != OFFICIAL_MAX_CANDIDATES
    ):
        raise TasteGCFPostprocessError("T12 generation cohort bytes changed")
    by_id = {str(row.molecule_id): row for row in loaded.sweet_rows}
    if len(by_id) != len(loaded.sweet_rows):
        raise TasteGCFPostprocessError("T12 train Sweet identity is not unique")
    try:
        source_rows = [by_id[str(row["parent_id"])] for row in cohort]
    except KeyError as exc:
        raise TasteGCFPostprocessError("T12 cohort escaped train Sweet rows") from exc
    records = [encode_taste_source_graph(row, loaded.schema) for row in source_rows]
    return loaded, source_rows, records


def derive_candidate_pool(
    *, authority: InputAuthority, device: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Independently derive the official full-graph surface from the 20k snapshot."""

    import torch

    snapshot = reopen_native_candidate_snapshot(
        authority.candidate_manifest,
        expected_contract_sha256=authority.transition_contract_sha256,
        expected_attempt_id=authority.attempt_id,
        expected_generation_token=authority.generation_token,
        torch=torch,
    )
    loaded, source_rows, records = _load_train_codec(authority)
    adapter = TasteFrozenGINENativeAdapter(
        _checkpoint_payloads(authority.checkpoint_path),
        source_records=records,
        graph_schema=loaded.schema,
        device=device,
    )
    native_rows = snapshot["counterfactual_candidates"]
    graph_map = snapshot["graph_map"]
    selected: list[dict[str, Any]] = []
    scanned = 0
    for offset in range(0, len(native_rows), 128):
        if len(selected) >= len(source_rows):
            break
        candidate_batch: list[Mapping[str, Any]] = []
        graph_batch: list[Any] = []
        rank_batch: list[int] = []
        parts_batch: list[list[float]] = []
        for rank, candidate in enumerate(
            native_rows[offset : offset + 128], start=offset
        ):
            scanned += 1
            parts = _importance_parts(candidate.get("importance_parts"))
            # Exact official summary.py train-side candidate prefilter.
            if parts[0] < 0.5:
                continue
            graph_hash = str(candidate.get("graph_hash") or "")
            graph = graph_map.get(graph_hash)
            if graph is None:
                raise TasteGCFPostprocessError("T12 native candidate lost its graph")
            canonical = canonical_attributed_graph(
                graph,
                feature_atomic_numbers=loaded.schema.feature_atomic_numbers,
            )
            if canonical.graph_identity_sha256 != graph_hash:
                raise TasteGCFPostprocessError("T12 native graph identity changed")
            candidate_batch.append(candidate)
            graph_batch.append(graph)
            rank_batch.append(rank)
            parts_batch.append(parts)
        if not graph_batch:
            continue
        scores = adapter.score(graph_batch)
        probabilities = scores.probabilities.tolist()
        for candidate, graph, native_rank, parts, valid, predicted, score, flag, probs, identity_payload in zip(
            candidate_batch,
            graph_batch,
            rank_batch,
            parts_batch,
            scores.valid_fullgraphs,
            scores.predictions,
            scores.scores,
            scores.candidate_flags,
            probabilities,
            scores.identity_graph_payloads,
            strict=True,
        ):
            if not valid:
                raise TasteGCFPostprocessError(
                    "T12 official candidate prefilter retained an invalid full graph"
                )
            if (
                not flag
                or predicted not in DESTINATION_LABELS
                or identity_payload is None
                or not math.isclose(parts[0], score, rel_tol=0.0, abs_tol=1e-12)
            ):
                raise TasteGCFPostprocessError(
                    "T12 official importance and frozen GINE disagree"
                )
            graph_hash = str(candidate["graph_hash"])
            origin = _origin_index(graph, parent_count=len(source_rows))
            decoded = decode_generated_fullgraph(
                graph, source_record=records[origin], schema=loaded.schema
            )
            if (
                not decoded.decode_ok
                or decoded.canonical_smiles
                != str(identity_payload.get("canonical_graph") or "")
            ):
                raise TasteGCFPostprocessError("T12 full-graph decoding changed")
            probability_row = [float(value) for value in probs]
            content = {
                "graph_identity_sha256": graph_hash,
                "canonical_smiles": decoded.canonical_smiles,
                "probabilities": probability_row,
                "predicted_label": int(predicted),
            }
            selected.append(
                {
                    "schema_version": "tastemolnet_t12_paper_candidate_v1",
                    "dataset": DATASET,
                    "method": METHOD,
                    "stage": STAGE,
                    "candidate_id": graph_hash,
                    "rule_content_hash": stable_sha256(content),
                    "candidate_content_hash": stable_sha256(content),
                    "canonical_smiles": decoded.canonical_smiles,
                    "predicted_label": int(predicted),
                    "probabilities": probability_row,
                    "source_probability": probability_row[SOURCE_LABEL],
                    "native_rank": native_rank,
                    "frequency": int(candidate["frequency"]),
                    "native_importance": parts[0],
                    "native_neurosed_coverage": parts[1],
                    "native_coverage_sha256": _semantic_sha256(
                        candidate.get("input_graphs_covering_list")
                    ),
                    "source_parent_position": origin,
                    "source_parent_id": str(source_rows[origin].molecule_id),
                    "source_split": "train",
                    "official_summary_prefilter": "importance_parts[0] >= 0.5",
                    "full_graph_semantics": True,
                    "generated_to_original_neurosed": True,
                    "oracle_backend": "gnn",
                    "classifier_family": "gine",
                    "rf_oracle_used": False,
                    "oracle_checkpoint_hash": authority.checkpoint_id,
                }
            )
            if len(selected) >= len(source_rows):
                scanned = native_rank + 1
                break
    candidate_ids = [str(row["candidate_id"]) for row in selected]
    if (
        len(selected) < MIN_CANDIDATES
        or len(selected) > OFFICIAL_MAX_CANDIDATES
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        raise TasteGCFPostprocessError(
            "T12 official full-graph surface has fewer than ten unique candidates"
        )
    manifest = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA,
        "status": "PASS",
        "dataset": DATASET,
        "method": METHOD,
        "stage": STAGE,
        "source_split": "train",
        "native_candidate_count": len(native_rows),
        "native_candidates_scanned": scanned,
        "candidate_count": len(selected),
        "official_summary_limit": len(source_rows),
        "official_importance_threshold": 0.5,
        "decode_rejection_count": 0,
        "ordered_candidate_ids_sha256": stable_sha256(candidate_ids),
        "generation_candidate_manifest_sha256": authority.candidate_manifest_sha256,
        "generation_audit_sha256": authority.generation_audit_sha256,
        "train_only": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "full_graph_semantics": True,
        "postprocess_implementation_sha256": authority.implementation_sha256,
        "generated_to_original_neurosed": True,
        "rf_oracle_used": False,
        "adapter_report": adapter.report(),
    }
    return selected, manifest


def _pair_rows_for_parent(
    *,
    parent: Any,
    before: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    provider: MolCLRNodeWassersteinDistance,
    authority: InputAuthority,
    split: str,
) -> list[dict[str, Any]]:
    if parent.split != split or parent.label != SOURCE_LABEL:
        raise TasteGCFPostprocessError("T12 evaluation parent authority changed")
    rows: list[dict[str, Any]] = []
    pred_before = int(before["predicted_label"])
    p_before = [float(value) for value in before["probabilities"]]
    for candidate in candidates:
        pred_after = int(candidate["predicted_label"])
        p_after = [float(value) for value in candidate["probabilities"]]
        strict = pred_before == SOURCE_LABEL and pred_after in DESTINATION_LABELS
        distance: float | None = None
        failure: str | None = None
        if strict:
            observed = provider.distance(parent.smiles, str(candidate["canonical_smiles"]))
            value = observed.get("distance")
            if observed.get("ok") is True and isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0.0:
                distance = float(value)
            else:
                failure = str(observed.get("error") or "wnode_distance_failed")
        elif pred_before != SOURCE_LABEL:
            failure = "frozen_gine_parent_not_source"
        else:
            failure = "frozen_gine_candidate_not_strict_flip"
        rows.append(
            {
                "schema_version": "tastemolnet_t12_fullgraph_pair_v1",
                "dataset": DATASET,
                "method": METHOD,
                "stage": STAGE,
                "split": split,
                "parent_id": parent.parent_id,
                "parent_smiles": parent.smiles,
                "candidate_id": candidate["candidate_id"],
                "candidate_content_hash": candidate["candidate_content_hash"],
                "canonical_smiles": candidate["canonical_smiles"],
                "action_kind": "full_counterfactual_graph",
                "action_semantics": "nearest_full_counterfactual_graph",
                "full_graph_semantics": True,
                "generated_to_original_neurosed": True,
                "applicable": True,
                "pred_before": pred_before,
                "pred_after": pred_after,
                "p_before": p_before,
                "p_after": p_after,
                "p1_before": p_before[SOURCE_LABEL],
                "p1_after": p_after[SOURCE_LABEL],
                "cf_drop": p_before[SOURCE_LABEL] - p_after[SOURCE_LABEL],
                "cf_flip": strict,
                "pair_strict_flip": distance is not None,
                "destination_label": pred_after if distance is not None else None,
                "wnode_distance": distance,
                "distance_for_selection": distance if distance is not None else "+inf",
                "failure_reason": failure,
                "cf_mode": CF_MODE,
                "source_label": SOURCE_LABEL,
                "oracle_backend": "gnn",
                "classifier_family": "gine",
                "rf_oracle_used": False,
                "oracle_checkpoint_hash": authority.checkpoint_id,
                "temperature_calibration_hash": authority.temperature_calibration_hash,
                "feature_schema_hash": authority.feature_schema_hash,
                "molclr_checkpoint_hash": authority.molclr_checkpoint_sha256,
                "distance_namespace": DISTANCE_NAMESPACE,
            }
        )
    return rows


def _parent_cohort_sha256(parents: Sequence[Any]) -> str:
    return stable_sha256(
        [
            {
                "position": index,
                "parent_id": row.parent_id,
                "smiles": row.smiles,
                "label": row.label,
                "split": row.split,
            }
            for index, row in enumerate(parents)
        ]
    )


def _evaluation_identity(
    *, split: str, parents: Sequence[Any], candidates: Sequence[Mapping[str, Any]], authority: InputAuthority
) -> dict[str, Any]:
    identities = [
        {
            "candidate_id": str(row["candidate_id"]),
            "candidate_content_hash": str(row["candidate_content_hash"]),
        }
        for row in candidates
    ]
    return {
        "schema_version": "tastemolnet_t12_split_evaluation_identity_v1",
        "split": split,
        "parent_cohort_sha256": _parent_cohort_sha256(parents),
        "parent_count": len(parents),
        "ordered_candidates_sha256": stable_sha256(identities),
        "candidate_count": len(candidates),
        "oracle_checkpoint_hash": authority.checkpoint_id,
        "temperature_calibration_hash": authority.temperature_calibration_hash,
        "feature_schema_hash": authority.feature_schema_hash,
        "molclr_checkpoint_hash": authority.molclr_checkpoint_sha256,
        "threshold_config_hash": authority.threshold.config_hash,
        "distance_line": DISTANCE_LINE,
        "distance_namespace": DISTANCE_NAMESPACE,
        "cf_mode": CF_MODE,
        "num_classes": NUM_CLASSES,
        "source_label": SOURCE_LABEL,
        "full_graph_semantics": True,
    }


def evaluate_split_resumable(
    *,
    split: str,
    parents: Sequence[Any],
    candidates: Sequence[Mapping[str, Any]],
    scorer: FrozenTasteGINEScorer,
    provider: MolCLRNodeWassersteinDistance,
    authority: InputAuthority,
    output: Path,
    checkpoint_callback: Callable[[int], None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunks = output / "raw" / f"{split}_pair_chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    identity = _evaluation_identity(
        split=split, parents=parents, candidates=candidates, authority=authority
    )
    identity_sha = stable_sha256(identity)
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    all_rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for position, parent in enumerate(parents):
        chunk = chunks / f"{position:08d}.jsonl"
        manifest_path = chunks / f"{position:08d}.manifest.json"
        if chunk.is_file() and manifest_path.is_file():
            rows = read_jsonl(chunk)
            manifest = read_json(manifest_path)
            if (
                manifest.get("schema_version") != CHUNK_MANIFEST_SCHEMA
                or manifest.get("split") != split
                or manifest.get("position") != position
                or manifest.get("parent_id") != parent.parent_id
                or manifest.get("evaluation_identity_sha256") != identity_sha
                or manifest.get("candidate_ids_sha256") != stable_sha256(candidate_ids)
                or manifest.get("row_count") != len(candidates)
                or manifest.get("rows_sha256") != sha256_file(chunk)
                or [str(row.get("candidate_id")) for row in rows] != candidate_ids
            ):
                raise TasteGCFPostprocessError(f"T12 {split} resume chunk changed")
        elif chunk.exists() or manifest_path.exists():
            raise TasteGCFPostprocessError(f"T12 {split} has a partial chunk")
        else:
            before = scorer.score_smiles([parent.smiles])[0]
            rows = _pair_rows_for_parent(
                parent=parent,
                before=before,
                candidates=candidates,
                provider=provider,
                authority=authority,
                split=split,
            )
            atomic_jsonl(chunk, rows)
            atomic_json(
                manifest_path,
                {
                    "schema_version": CHUNK_MANIFEST_SCHEMA,
                    "split": split,
                    "position": position,
                    "parent_id": parent.parent_id,
                    "evaluation_identity_sha256": identity_sha,
                    "candidate_ids_sha256": stable_sha256(candidate_ids),
                    "row_count": len(rows),
                    "rows_sha256": sha256_file(chunk),
                },
            )
        all_rows.extend(rows)
        inventory.append(
            {
                "position": position,
                "parent_id": parent.parent_id,
                "chunk": chunk.relative_to(output).as_posix(),
                "chunk_sha256": sha256_file(chunk),
                "manifest": manifest_path.relative_to(output).as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
        checkpoint_callback(position + 1)
    details = output / "raw" / f"{split}_pair_details.jsonl"
    atomic_jsonl(details, all_rows)
    payload = {
        "schema_version": CHUNK_INVENTORY_SCHEMA,
        "split": split,
        "evaluation_identity": identity,
        "evaluation_identity_sha256": identity_sha,
        "parent_count": len(parents),
        "candidate_count": len(candidates),
        "pair_count": len(all_rows),
        "chunks": inventory,
        "chunks_sha256": stable_sha256(inventory),
        "pair_details_sha256": sha256_file(details),
    }
    inventory_path = output / "raw" / f"{split}_pair_chunk_inventory.json"
    atomic_json(inventory_path, payload)
    return all_rows, {
        "split": split,
        "parent_count": len(parents),
        "candidate_count": len(candidates),
        "pair_count": len(all_rows),
        "pair_details_sha256": sha256_file(details),
        "parent_ids_sha256": stable_sha256(sorted(row.parent_id for row in parents)),
        "candidate_ids_sha256": stable_sha256(candidate_ids),
        "resumable_parent_chunks": True,
        "checkpointed_parent_count": len(parents),
        "evaluation_identity_sha256": identity_sha,
        "chunk_inventory_sha256": sha256_file(inventory_path),
    }


def select_on_calibration(
    candidates: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    theta_star: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        selected, trace = _globalgce_selector(
            candidates, pair_rows, theta_star=theta_star
        )
    except Exception as exc:
        raise TasteGCFPostprocessError(str(exc).replace("T13", "T12")) from exc
    trace = dict(trace)
    ids = list(trace.pop("ordered_rule_ids"))
    trace.pop("ordered_rule_ids_sha256", None)
    trace["selector"] = (
        "calibration_greedy_marginal_theta_then_strict_then_total_coverage_"
        "then_mean_wnode_then_candidate_id_v1"
    )
    trace["ordered_candidate_ids"] = ids
    trace["ordered_candidate_ids_sha256"] = stable_sha256(ids)
    trace["calibration_only"] = True
    return selected, trace


def standardized_metrics(
    pair_rows: Sequence[Mapping[str, Any]],
    ordered_candidate_ids: Sequence[str],
    threshold: ThresholdContract,
) -> dict[str, Any]:
    # The shared numeric implementation is method-neutral; only its display
    # label is T13-specific.  Replace that label before any bytes are frozen.
    try:
        computed = _globalgce_metrics(
            pair_rows, ordered_candidate_ids, threshold  # type: ignore[arg-type]
        )
    except Exception as exc:
        raise TasteGCFPostprocessError(str(exc).replace("T13", "T12")) from exc
    for name in ("prefix", "parent_best", "figure3", "figure4", "table2", "destination"):
        for row in computed[name]:
            row["method"] = METHOD
    return computed


def _authorize_test(authority: InputAuthority, selection_path: Path) -> list[Any]:
    selection = read_json(selection_path)
    if (
        selection.get("schema_version") != SELECTION_SCHEMA
        or selection.get("status") != "FROZEN"
        or selection.get("selection_frozen") is not True
        or selection.get("selector_fitted_on_calibration") is not True
        or selection.get("calibration_only") is not True
        or selection.get("test_loaded") is not False
        or selection.get("test_used_for_selection") is not False
    ):
        raise TasteGCFPostprocessError("T12 test access requires frozen calibration order")
    receipt_path = selection_path.parent / "test_access_receipt.json"
    receipt = {
            "schema_version": "tastemolnet_t12_test_access_receipt_v1",
            "status": "AUTHORIZED_AFTER_FREEZE",
            "selection_manifest_sha256": sha256_file(selection_path),
            "selection_frozen_before_test": True,
            "test_used_for_selection": False,
            "authorized_at": _utc_now(),
        }
    if receipt_path.is_file():
        existing = read_json(receipt_path)
        if (
            existing.get("schema_version") != receipt["schema_version"]
            or existing.get("status") != receipt["status"]
            or existing.get("selection_manifest_sha256")
            != receipt["selection_manifest_sha256"]
            or existing.get("selection_frozen_before_test") is not True
            or existing.get("test_used_for_selection") is not False
            or not str(existing.get("authorized_at") or "")
        ):
            raise TasteGCFPostprocessError("T12 durable test-access receipt changed")
    else:
        atomic_json(receipt_path, receipt)
    return load_prepared_split(
        authority.test_path,
        expected_split="test",
        expected_sha256=authority.declared_test_sha256,
    )


def _immutable_inventory(output: Path) -> dict[str, dict[str, Any]]:
    required = {
        "figure3_coverage_vs_k.csv",
        "figure4_coverage_vs_threshold.csv",
        "prefix_metrics.csv",
        "prefix_metrics.json",
        "parent_best_distances.csv",
        "destination_distribution.csv",
        "table2_gcfexplainer_k10.csv",
        "summary.json",
        "oracle_manifest.json",
        "evaluation_manifest.json",
        "raw/candidate_pool.jsonl",
        "raw/candidate_pool_manifest.json",
        "raw/calibration_pair_details.jsonl",
        "raw/calibration_pair_chunk_inventory.json",
        "raw/selected_candidates.jsonl",
        "raw/selection_manifest.json",
        "raw/test_access_receipt.json",
        "raw/test_pair_details.jsonl",
        "raw/test_pair_chunk_inventory.json",
        "raw/test_evaluation_manifest.json",
    }
    names = set(required)
    for path in sorted((output / "raw").rglob("*")):
        if path.is_symlink():
            raise TasteGCFPostprocessError("T12 raw tree contains a symbolic link")
        if path.is_file():
            names.add(path.relative_to(output).as_posix())
    inventory: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        path = output / name
        if not path.is_file() or (name in required and path.stat().st_size <= 0):
            raise TasteGCFPostprocessError(f"T12 immutable artifact missing: {name}")
        inventory[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return inventory


def _common(
    authority: InputAuthority, output: Path, *, test_parent_ids_sha256: str
) -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "method": METHOD,
        "stage": STAGE,
        "num_classes": NUM_CLASSES,
        "source_label": SOURCE_LABEL,
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "oracle_checkpoint": str(authority.checkpoint_path),
        "oracle_hash": authority.checkpoint_id,
        "oracle_checkpoint_hash": authority.checkpoint_id,
        "dataset_hash": authority.dataset_hash,
        "test_parent_ids_sha256": test_parent_ids_sha256,
        "test_split_hash": authority.declared_test_sha256,
        "distance_line": DISTANCE_LINE,
        "distance_namespace": DISTANCE_NAMESPACE,
        "molclr_checkpoint_hash": authority.molclr_checkpoint_sha256,
        "temperature_calibration_hash": authority.temperature_calibration_hash,
        "feature_schema_hash": authority.feature_schema_hash,
        "feature_schema_file_sha256": authority.feature_schema_file_sha256,
        "cf_mode": CF_MODE,
        "threshold_config_hash": authority.threshold.config_hash,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "raw_output_root": str(output),
        "generation_root": str(authority.generation_root),
        "generation_verification_root": str(authority.generation_verification_root),
        "generation_audit_sha256": authority.generation_audit_sha256,
        "generation_pass_sha256": authority.generation_pass_sha256,
        "generation_candidate_manifest_sha256": authority.candidate_manifest_sha256,
        "postprocess_implementation_sha256": authority.implementation_sha256,
        "generated_to_original_neurosed": True,
        "full_graph_semantics": True,
    }


def run_t12_postprocess(
    *,
    authority: InputAuthority,
    output_dir: str | Path,
    resume: bool,
    device: str,
    wnode_cache_db: str | Path,
    node_embedding_cache_dir: str | Path,
) -> dict[str, Any]:
    if device != "cuda:0":
        raise TasteGCFPostprocessError("T12 postprocess is bound to logical cuda:0")
    output = Path(output_dir).expanduser().absolute()
    identity = authority.resume_identity()
    if resume:
        if (
            not output.is_dir()
            or output.resolve(strict=True) != output
            or output.is_symlink()
            or not _checkpoint_path(output).is_file()
        ):
            raise TasteGCFPostprocessError("T12 --resume requires a checkpoint")
        checkpoint = _load_checkpoint(output, identity)
        if checkpoint.get("phase") in {"SEALED", "PASS"}:
            return read_json(output / "run_manifest.json")
    else:
        if output.exists():
            raise TasteGCFPostprocessError("fresh T12 paper root already exists")
        output.mkdir(parents=True)
        if output.resolve(strict=True) != output or output.is_symlink():
            raise TasteGCFPostprocessError("T12 paper root is an alias")
        (output / "raw").mkdir()
        _write_checkpoint(output, phase="INITIALIZED", identity=identity)
    if (output / "PASS").exists():
        raise TasteGCFPostprocessError("T12 science cannot overwrite terminal PASS")

    pool_path = output / "raw" / "candidate_pool.jsonl"
    pool_manifest_path = output / "raw" / "candidate_pool_manifest.json"
    if pool_path.is_file() and pool_manifest_path.is_file():
        candidates = read_jsonl(pool_path)
        pool_manifest = read_json(pool_manifest_path)
        if (
            pool_manifest.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA
            or pool_manifest.get("candidate_pool_sha256") != sha256_file(pool_path)
            or pool_manifest.get("candidate_count") != len(candidates)
            or pool_manifest.get("generation_audit_sha256") != authority.generation_audit_sha256
        ):
            raise TasteGCFPostprocessError("T12 frozen candidate pool changed")
    elif pool_path.exists() or pool_manifest_path.exists():
        raise TasteGCFPostprocessError("T12 candidate pool is partially present")
    else:
        candidates, pool_manifest = derive_candidate_pool(
            authority=authority, device=device
        )
        atomic_jsonl(pool_path, candidates)
        pool_manifest = {
            **pool_manifest,
            "candidate_pool_sha256": sha256_file(pool_path),
            "materialized_at": _utc_now(),
        }
        atomic_json(pool_manifest_path, pool_manifest)
        _write_checkpoint(
            output,
            phase="CANDIDATE_POOL_COMPLETE",
            identity=identity,
            detail={"candidate_count": len(candidates), "candidate_pool_sha256": sha256_file(pool_path)},
        )

    payloads = _checkpoint_payloads(authority.checkpoint_path)
    scorer = FrozenTasteGINEScorer(payloads, device=device, batch_size=256)
    if scorer.checkpoint_id != authority.checkpoint_id:
        raise TasteGCFPostprocessError("T12 scorer checkpoint changed")
    provider = MolCLRNodeWassersteinDistance(
        MolCLRNodeWassersteinConfig(
            molclr_root=authority.molclr_root,
            molclr_ckpt=authority.molclr_checkpoint,
            cache_db=Path(wnode_cache_db).expanduser().absolute(),
            node_emb_cache_dir=Path(node_embedding_cache_dir).expanduser().absolute(),
            device=device,
            distance_namespace=DISTANCE_NAMESPACE,
        )
    )
    try:
        selection_path = output / "raw" / "selection_manifest.json"
        selected_path = output / "raw" / "selected_candidates.jsonl"
        if selection_path.is_file() and selected_path.is_file():
            selection = read_json(selection_path)
            selected = read_jsonl(selected_path)
            ordered_ids = [str(row.get("candidate_id") or "") for row in selected]
            if (
                selection.get("schema_version") != SELECTION_SCHEMA
                or selection.get("status") != "FROZEN"
                or selection.get("selection_frozen") is not True
                or selection.get("calibration_only") is not True
                or selection.get("test_loaded") is not False
                or selection.get("test_used_for_selection") is not False
                or selection.get("ordered_candidate_ids") != ordered_ids
                or selection.get("selected_candidates_sha256") != sha256_file(selected_path)
            ):
                raise TasteGCFPostprocessError("T12 frozen selection changed")
        elif selection_path.exists() or selected_path.exists():
            raise TasteGCFPostprocessError("T12 frozen selection is partially present")
        else:
            calibration_parents = load_prepared_split(
                authority.calibration_path,
                expected_split="calibration",
                expected_sha256=authority.calibration_sha256,
            )
            calibration_rows, calibration_manifest = evaluate_split_resumable(
                split="calibration",
                parents=calibration_parents,
                candidates=candidates,
                scorer=scorer,
                provider=provider,
                authority=authority,
                output=output,
                checkpoint_callback=lambda count: _write_checkpoint(
                    output,
                    phase="CALIBRATION_RUNNING",
                    identity=identity,
                    detail={"completed_parent_count": count, "parent_count": len(calibration_parents)},
                ),
            )
            selected, selector = select_on_calibration(
                candidates,
                calibration_rows,
                theta_star=authority.threshold.theta_star,
            )
            atomic_jsonl(selected_path, selected)
            selection = {
                "schema_version": SELECTION_SCHEMA,
                "dataset": DATASET,
                "method": METHOD,
                "stage": STAGE,
                "status": "FROZEN",
                "selection_frozen": True,
                "frozen_at": _utc_now(),
                **selector,
                **_threshold_payload(authority.threshold),
                "calibration_manifest": calibration_manifest,
                "candidate_pool_sha256": sha256_file(pool_path),
                "selected_candidates_sha256": sha256_file(selected_path),
                "oracle_checkpoint_hash": authority.checkpoint_id,
                "molclr_checkpoint_hash": authority.molclr_checkpoint_sha256,
                "selector_fitted_on_calibration": True,
                "test_loaded": False,
                "test_used_for_selection": False,
                "full_graph_semantics": True,
                "rf_oracle_used": False,
            }
            atomic_json(selection_path, selection)
            _write_checkpoint(
                output,
                phase="CALIBRATION_SELECTION_FROZEN",
                identity=identity,
                detail={"selection_manifest_sha256": sha256_file(selection_path)},
            )
        selection_sha = sha256_file(selection_path)

        test_parents = _authorize_test(authority, selection_path)
        test_started_at = str(
            read_json(output / "raw" / "test_access_receipt.json")["authorized_at"]
        )
        test_rows, test_manifest = evaluate_split_resumable(
            split="test",
            parents=test_parents,
            candidates=selected,
            scorer=scorer,
            provider=provider,
            authority=authority,
            output=output,
            checkpoint_callback=lambda count: _write_checkpoint(
                output,
                phase="TEST_RUNNING",
                identity=identity,
                detail={"completed_parent_count": count, "parent_count": len(test_parents)},
            ),
        )
        provider_stats = provider.stats_dict()
    finally:
        provider.close()
    test_manifest.update(
        {
            "started_at": test_started_at,
            "completed_at": _utc_now(),
            "selection_manifest_sha256": selection_sha,
            "test_access_receipt_sha256": sha256_file(output / "raw" / "test_access_receipt.json"),
            "selection_frozen_before_test": True,
            "test_used_for_selection": False,
        }
    )
    atomic_json(output / "raw" / "test_evaluation_manifest.json", test_manifest)
    ordered_ids = [str(row["candidate_id"]) for row in selected]
    metrics = standardized_metrics(test_rows, ordered_ids, authority.threshold)
    atomic_csv(output / "figure3_coverage_vs_k.csv", metrics["figure3"])
    atomic_csv(output / "figure4_coverage_vs_threshold.csv", metrics["figure4"])
    atomic_csv(output / "prefix_metrics.csv", metrics["prefix"])
    atomic_json(output / "prefix_metrics.json", metrics["prefix"])
    atomic_csv(output / "parent_best_distances.csv", metrics["parent_best"])
    atomic_csv(output / "destination_distribution.csv", metrics["destination"])
    atomic_csv(output / "table2_gcfexplainer_k10.csv", metrics["table2"])
    test_parent_hash = stable_sha256(sorted(row.parent_id for row in test_parents))
    common = _common(authority, output, test_parent_ids_sha256=test_parent_hash)
    summary = {
        "schema_version": "tastemolnet_t12_summary_v1",
        **common,
        "status": "SEALED",
        "frozen": True,
        "artifacts_frozen": True,
        "raw_output_complete": True,
        "raw_artifacts_complete": True,
        "selection_frozen_before_test": True,
        "calibration_loaded": True,
        "test_loaded": True,
        "candidate_count": len(candidates),
        "effective_rule_count": metrics["effective_rule_count"],
        "parent_count": metrics["parent_count"],
        "pair_count": metrics["pair_count"],
        "M_configured_max": 20_000,
        "M_effective": 20_000,
        "resource_cap_used": True,
        "early_stop_used": False,
        "stop_reason": "configured_resource_cap_20000",
        "K_MAX": K_MAX,
        "MIN_RULES_FOR_MAIN_TABLE": MIN_CANDIDATES,
        "distance_provider_stats": provider_stats,
        "threshold_contract": _threshold_payload(authority.threshold),
    }
    oracle = {
        "schema_version": "tastemolnet_t12_oracle_manifest_v1",
        **common,
        "temperature": scorer.temperature,
        "num_classes": scorer.num_classes,
        "source_label": scorer.source_label,
        "same_frozen_gine_for_generation_calibration_test": True,
        "calibration_loaded_for_training": False,
        "test_loaded_for_training": False,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "frozen": True,
    }
    evaluation = {
        "schema_version": "tastemolnet_t12_evaluation_manifest_v1",
        **common,
        "status": "SEALED",
        "selection_manifest_sha256": selection_sha,
        "test_evaluation_manifest_sha256": sha256_file(output / "raw" / "test_evaluation_manifest.json"),
        "selection_frozen_before_test": True,
        "calibration_only_selector": True,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "strict_flip_definition": "pred_before == 1 and pred_after in {0,2}",
        "destination_labels": [0, 2],
        "full_cartesian_test_pairs": True,
        "native_action": "nearest_full_counterfactual_graph",
        "frozen": True,
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "oracle_manifest.json", oracle)
    atomic_json(output / "evaluation_manifest.json", evaluation)
    inventory = _immutable_inventory(output)
    freeze = {
        "schema_version": "tastemolnet_t12_freeze_manifest_v1",
        **common,
        "status": "SEALED",
        "frozen": True,
        "artifacts_frozen": True,
        "files": inventory,
        "inventory_sha256": stable_sha256(inventory),
        "sealed_at": _utc_now(),
    }
    atomic_json(output / "freeze_manifest.json", freeze)
    run = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        **common,
        "status": "SEALED",
        "state": "SEALED",
        "run_complete": False,
        "raw_output_complete": True,
        "source_artifacts_complete": True,
        "frozen": True,
        "artifacts_frozen": True,
        "selection_frozen_before_test": True,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "freeze_manifest_sha256": sha256_file(output / "freeze_manifest.json"),
        "independent_terminal_verification_required": True,
        "worker_wrote_pass": False,
        "candidate_pool_sha256": sha256_file(pool_path),
        "selected_candidates_sha256": sha256_file(selected_path),
        "sealed_at": _utc_now(),
    }
    atomic_json(output / "run_manifest.json", run)
    _atomic_bytes(output / "SEALED", b"SEALED\n")
    _write_checkpoint(output, phase="SEALED", identity=identity)
    return run


def _verify_chunks(
    output: Path, *, split: str, pair_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    path = output / "raw" / f"{split}_pair_chunk_inventory.json"
    inventory = read_json(path)
    chunks = inventory.get("chunks")
    identity = inventory.get("evaluation_identity")
    if (
        inventory.get("schema_version") != CHUNK_INVENTORY_SCHEMA
        or inventory.get("split") != split
        or type(chunks) is not list
        or type(identity) is not dict
        or inventory.get("evaluation_identity_sha256") != stable_sha256(identity)
        or inventory.get("chunks_sha256") != stable_sha256(chunks)
        or inventory.get("pair_details_sha256") != sha256_file(output / "raw" / f"{split}_pair_details.jsonl")
    ):
        raise TasteGCFPostprocessError(f"T12 {split} chunk inventory changed")
    reconstructed: list[dict[str, Any]] = []
    for position, entry in enumerate(chunks):
        expected_chunk = f"raw/{split}_pair_chunks/{position:08d}.jsonl"
        expected_manifest = (
            f"raw/{split}_pair_chunks/{position:08d}.manifest.json"
        )
        if (
            entry.get("position") != position
            or entry.get("chunk") != expected_chunk
            or entry.get("manifest") != expected_manifest
        ):
            raise TasteGCFPostprocessError(f"T12 {split} chunk path changed")
        chunk = output / expected_chunk
        manifest_path = output / expected_manifest
        rows = read_jsonl(chunk)
        manifest = read_json(manifest_path)
        candidate_ids = [str(row.get("candidate_id") or "") for row in rows]
        if (
            sha256_file(chunk) != entry.get("chunk_sha256")
            or sha256_file(manifest_path) != entry.get("manifest_sha256")
            or manifest.get("schema_version") != CHUNK_MANIFEST_SCHEMA
            or manifest.get("split") != split
            or manifest.get("position") != position
            or manifest.get("parent_id") != entry.get("parent_id")
            or manifest.get("evaluation_identity_sha256")
            != inventory.get("evaluation_identity_sha256")
            or manifest.get("rows_sha256") != entry.get("chunk_sha256")
            or manifest.get("row_count") != len(rows)
            or manifest.get("candidate_ids_sha256")
            != stable_sha256(candidate_ids)
            or len(rows) != inventory.get("candidate_count")
            or any(row.get("parent_id") != entry.get("parent_id") or row.get("split") != split for row in rows)
        ):
            raise TasteGCFPostprocessError(f"T12 {split} chunk replay changed")
        reconstructed.extend(rows)
    if reconstructed != [dict(row) for row in pair_rows]:
        raise TasteGCFPostprocessError(f"T12 {split} pair chunks changed")
    if (
        inventory.get("parent_count") != len(chunks)
        or inventory.get("pair_count") != len(pair_rows)
    ):
        raise TasteGCFPostprocessError(f"T12 {split} chunk counts changed")
    return inventory


def _validate_fullgraph_pairs(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *, split: str,
) -> None:
    by_candidate = {str(row["candidate_id"]): row for row in candidates}
    pairs: set[tuple[str, str]] = set()
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        parent_id = str(row.get("parent_id") or "")
        candidate = by_candidate.get(candidate_id)
        if (
            candidate is None
            or not parent_id
            or (parent_id, candidate_id) in pairs
            or row.get("split") != split
            or row.get("action_kind") != "full_counterfactual_graph"
            or row.get("action_semantics") != "nearest_full_counterfactual_graph"
            or row.get("full_graph_semantics") is not True
            or row.get("generated_to_original_neurosed") is not True
            or row.get("applicable") is not True
            or row.get("canonical_smiles") != candidate.get("canonical_smiles")
            or row.get("pred_after") != candidate.get("predicted_label")
            or row.get("p_after") != candidate.get("probabilities")
            or row.get("rf_oracle_used") is not False
            or row.get("oracle_checkpoint_hash") != candidate.get("oracle_checkpoint_hash")
            or any(key in row for key in ("residual_smiles", "match_id", "deletion", "rule_id"))
        ):
            raise TasteGCFPostprocessError("T12 full-graph pair semantics changed")
        pairs.add((parent_id, candidate_id))
        strict = row.get("pred_before") == SOURCE_LABEL and row.get("pred_after") in DESTINATION_LABELS
        if row.get("cf_flip") is not strict:
            raise TasteGCFPostprocessError("T12 strict flip semantics changed")
        if row.get("pair_strict_flip") is True:
            value = row.get("wnode_distance")
            if not strict or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0 or row.get("destination_label") not in DESTINATION_LABELS:
                raise TasteGCFPostprocessError("T12 strict pair lacks WNode evidence")
        elif row.get("wnode_distance") is not None:
            raise TasteGCFPostprocessError("T12 non-strict pair has WNode evidence")


def _publish_terminal_pass(
    *, output: Path, verification: Path, audit: Mapping[str, Any]
) -> None:
    """Publish verifier proof first and the paper-cell marker last."""

    if (
        audit.get("schema_version") != VERIFY_SCHEMA
        or audit.get("status") != "PASS"
        or audit.get("passed") is not True
        or audit.get("audit_passed") is not True
        or audit.get("independent_verifier") is not True
        or (audit.get("checks") or {}).get(
            "calibration_only_selector_replayed"
        )
        is not True
        or (audit.get("checks") or {}).get("selection_frozen_before_test")
        is not True
    ):
        raise TasteGCFPostprocessError(
            "T12 terminal publication requires the exact independent audit"
        )
    atomic_json(verification / "terminal_verification.json", dict(audit))
    _atomic_bytes(verification / "PASS", (PASS_MARKER + "\n").encode())
    # This is intentionally last.  Matrix publication cannot observe a paper
    # PASS before the distinct verifier root is itself complete.
    _atomic_bytes(output / "PASS", (PASS_MARKER + "\n").encode())


def verify_t12_output(
    *,
    authority: InputAuthority,
    output_dir: str | Path,
    verification_dir: str | Path,
    device: str,
) -> dict[str, Any]:
    """Replay immutable artifacts and exclusively publish ``[TASTE_GCF_PASS]``."""

    output = Path(output_dir).expanduser().resolve(strict=True)
    verification = Path(verification_dir).expanduser().absolute()
    if verification == output or output in verification.parents:
        raise TasteGCFPostprocessError(
            "T12 independent verification root must be outside the paper root"
        )
    if verification.exists():
        raise TasteGCFPostprocessError("T12 terminal verification root must be fresh")
    verification.mkdir(parents=True)
    if verification.resolve(strict=True) != verification or verification.is_symlink():
        raise TasteGCFPostprocessError("T12 verification root is an alias")
    atomic_json(
        verification / "verification_input.json",
        {
            "schema_version": "tastemolnet_t12_terminal_verifier_input_v1",
            "stage": STAGE,
            "paper_output_root": str(output),
            "paper_run_manifest_sha256": sha256_file(output / "run_manifest.json"),
            "generation_audit_sha256": authority.generation_audit_sha256,
            "generation_pass_sha256": authority.generation_pass_sha256,
            "independent_verifier_process": True,
            "started_at": _utc_now(),
        },
    )
    if (output / "SEALED").read_bytes() != b"SEALED\n":
        raise TasteGCFPostprocessError("T12 verifier requires SEALED science output")
    pass_path = output / "PASS"
    if pass_path.exists():
        audit = read_json(output / "final_artifact_audit.json")
        if pass_path.read_bytes() == (PASS_MARKER + "\n").encode() and audit.get("passed") is True:
            _publish_terminal_pass(
                output=output, verification=verification, audit=audit
            )
            return audit
        raise TasteGCFPostprocessError("T12 PASS conflicts with terminal audit")
    run = read_json(output / "run_manifest.json")
    freeze = read_json(output / "freeze_manifest.json")
    if (
        run.get("schema_version") != RUN_MANIFEST_SCHEMA
        or run.get("status") not in {"SEALED", "PASS"}
        or run.get("state") != run.get("status")
        or run.get("worker_wrote_pass") is not False
        or freeze.get("status") != "SEALED"
        or freeze.get("frozen") is not True
        or run.get("freeze_manifest_sha256")
        != sha256_file(output / "freeze_manifest.json")
        or run.get("generation_audit_sha256") != authority.generation_audit_sha256
        or run.get("generation_pass_sha256") != authority.generation_pass_sha256
    ):
        raise TasteGCFPostprocessError("T12 verifier received a non-SEALED run")
    inventory = freeze.get("files")
    if type(inventory) is not dict or freeze.get("inventory_sha256") != stable_sha256(inventory):
        raise TasteGCFPostprocessError("T12 frozen inventory is malformed")
    for name, identity in inventory.items():
        path = output / name
        if not path.is_file() or path.stat().st_size != identity.get("bytes") or sha256_file(path) != identity.get("sha256"):
            raise TasteGCFPostprocessError(f"T12 frozen artifact changed: {name}")
    if _immutable_inventory(output) != inventory:
        raise TasteGCFPostprocessError("T12 frozen artifact closure changed")

    replayed_candidates, replayed_pool_manifest = derive_candidate_pool(
        authority=authority, device=device
    )
    candidates = read_jsonl(output / "raw" / "candidate_pool.jsonl")
    pool_manifest = read_json(output / "raw" / "candidate_pool_manifest.json")
    if (
        replayed_candidates != candidates
        or any(pool_manifest.get(key) != value for key, value in replayed_pool_manifest.items())
        or pool_manifest.get("candidate_pool_sha256") != sha256_file(output / "raw" / "candidate_pool.jsonl")
    ):
        raise TasteGCFPostprocessError("T12 native candidate derivation replay changed")
    selection = read_json(output / "raw" / "selection_manifest.json")
    test_manifest = read_json(output / "raw" / "test_evaluation_manifest.json")
    test_access = read_json(output / "raw" / "test_access_receipt.json")
    selected = read_jsonl(output / "raw" / "selected_candidates.jsonl")
    calibration_rows = read_jsonl(output / "raw" / "calibration_pair_details.jsonl")
    test_rows = read_jsonl(output / "raw" / "test_pair_details.jsonl")
    calibration_chunks = _verify_chunks(output, split="calibration", pair_rows=calibration_rows)
    test_chunks = _verify_chunks(output, split="test", pair_rows=test_rows)
    _validate_fullgraph_pairs(calibration_rows, candidates, split="calibration")
    _validate_fullgraph_pairs(test_rows, selected, split="test")
    if sha256_file(authority.test_path) != authority.declared_test_sha256:
        raise TasteGCFPostprocessError("T12 held-out test bytes changed")
    observed_test_parent_hash = stable_sha256(
        sorted({str(row.get("parent_id") or "") for row in test_rows})
    )
    expected_common = _common(
        authority, output, test_parent_ids_sha256=observed_test_parent_hash
    )
    if any(run.get(key) != value for key, value in expected_common.items()):
        raise TasteGCFPostprocessError("T12 sealed source identity changed")
    threshold_payload = _threshold_payload(authority.threshold)
    if (
        selection.get("schema_version") != SELECTION_SCHEMA
        or selection.get("status") != "FROZEN"
        or selection.get("selection_frozen") is not True
        or selection.get("selector_fitted_on_calibration") is not True
        or selection.get("calibration_only") is not True
        or selection.get("test_loaded") is not False
        or selection.get("test_used_for_selection") is not False
        or any(selection.get(key) != value for key, value in threshold_payload.items())
        or selection.get("oracle_checkpoint_hash") != authority.checkpoint_id
        or selection.get("molclr_checkpoint_hash")
        != authority.molclr_checkpoint_sha256
        or test_access.get("selection_manifest_sha256") != sha256_file(output / "raw" / "selection_manifest.json")
        or test_access.get("selection_frozen_before_test") is not True
        or test_manifest.get("selection_manifest_sha256") != sha256_file(output / "raw" / "selection_manifest.json")
        or test_manifest.get("test_access_receipt_sha256") != sha256_file(output / "raw" / "test_access_receipt.json")
        or str(selection.get("frozen_at")) > str(test_manifest.get("started_at"))
    ):
        raise TasteGCFPostprocessError("T12 calibration/test isolation replay failed")
    for chunks, manifest, label in (
        (calibration_chunks, selection.get("calibration_manifest") or {}, "calibration"),
        (test_chunks, test_manifest, "test"),
    ):
        if (
            manifest.get("split") != label
            or manifest.get("parent_count") != chunks.get("parent_count")
            or manifest.get("candidate_count") != chunks.get("candidate_count")
            or manifest.get("pair_count") != chunks.get("pair_count")
            or manifest.get("pair_details_sha256") != chunks.get("pair_details_sha256")
            or manifest.get("evaluation_identity_sha256") != chunks.get("evaluation_identity_sha256")
        ):
            raise TasteGCFPostprocessError(f"T12 frozen {label} manifest changed")
    for chunks, expected_candidates, label in (
        (calibration_chunks, candidates, "calibration"),
        (test_chunks, selected, "test"),
    ):
        split_identity = chunks.get("evaluation_identity") or {}
        candidate_identities = [
            {
                "candidate_id": str(row.get("candidate_id") or ""),
                "candidate_content_hash": str(
                    row.get("candidate_content_hash") or ""
                ),
            }
            for row in expected_candidates
        ]
        if (
            split_identity.get("schema_version")
            != "tastemolnet_t12_split_evaluation_identity_v1"
            or split_identity.get("split") != label
            or split_identity.get("candidate_count") != len(expected_candidates)
            or split_identity.get("ordered_candidates_sha256")
            != stable_sha256(candidate_identities)
            or split_identity.get("parent_count") != chunks.get("parent_count")
            or split_identity.get("oracle_checkpoint_hash")
            != authority.checkpoint_id
            or split_identity.get("molclr_checkpoint_hash")
            != authority.molclr_checkpoint_sha256
            or split_identity.get("threshold_config_hash")
            != authority.threshold.config_hash
            or split_identity.get("temperature_calibration_hash")
            != authority.temperature_calibration_hash
            or split_identity.get("feature_schema_hash")
            != authority.feature_schema_hash
            or split_identity.get("distance_namespace") != DISTANCE_NAMESPACE
            or split_identity.get("full_graph_semantics") is not True
        ):
            raise TasteGCFPostprocessError(
                f"T12 {label} evaluation identity changed"
            )
    ordered = [str(row["candidate_id"]) for row in selected]
    if ordered != selection.get("ordered_candidate_ids"):
        raise TasteGCFPostprocessError("T12 selected candidate order changed")
    replayed_selected, replayed_selector = select_on_calibration(
        candidates, calibration_rows, theta_star=authority.threshold.theta_star
    )
    if replayed_selected != selected or any(selection.get(key) != value for key, value in replayed_selector.items()):
        raise TasteGCFPostprocessError("T12 calibration-only selector replay changed")
    recomputed = standardized_metrics(test_rows, ordered, authority.threshold)
    expected = {
        "figure3_coverage_vs_k.csv": recomputed["figure3"],
        "figure4_coverage_vs_threshold.csv": recomputed["figure4"],
        "prefix_metrics.csv": recomputed["prefix"],
        "parent_best_distances.csv": recomputed["parent_best"],
        "destination_distribution.csv": recomputed["destination"],
        "table2_gcfexplainer_k10.csv": recomputed["table2"],
    }
    for name, rows in expected.items():
        with tempfile.TemporaryDirectory(prefix="t12-verify-") as temporary:
            candidate = Path(temporary) / name
            atomic_csv(candidate, rows)
            if candidate.read_bytes() != (output / name).read_bytes():
                raise TasteGCFPostprocessError(f"T12 standardized replay changed: {name}")
    if read_json_value(output / "prefix_metrics.json") != recomputed["prefix"]:
        raise TasteGCFPostprocessError("T12 prefix JSON replay changed")

    checks = {
        "generation_verifier_pass_reloaded": True,
        "lossless_candidate_snapshot_replayed": True,
        "official_summary_prefilter_replayed": True,
        "full_graph_semantics": True,
        "generated_to_original_neurosed": True,
        "calibration_only_selector_replayed": True,
        "selection_frozen_before_test": True,
        "held_out_test_cartesian_complete": True,
        "resumed_chunk_hashes_replayed": True,
        "standardized_metrics_recomputed": True,
        "same_gine_identity": True,
        "same_molclr_identity": True,
        "rf_oracle_absent": True,
        "destination_labels_0_or_2": True,
    }
    common_keys = (
        "dataset", "method", "stage", "oracle_backend", "classifier_family",
        "rf_oracle_used", "oracle_checkpoint", "oracle_hash", "oracle_checkpoint_hash",
        "dataset_hash", "test_parent_ids_sha256", "test_split_hash", "distance_line",
        "distance_namespace", "molclr_checkpoint_hash", "temperature_calibration_hash",
        "feature_schema_hash", "feature_schema_file_sha256", "cf_mode",
        "threshold_config_hash", "test_used_for_selection", "threshold_fitted_on_test",
        "raw_output_root", "generation_root", "generation_verification_root",
        "generation_audit_sha256", "generation_pass_sha256",
        "generation_candidate_manifest_sha256", "generated_to_original_neurosed",
        "full_graph_semantics", "postprocess_implementation_sha256",
    )
    common = {key: run[key] for key in common_keys}
    audit = {
        "schema_version": VERIFY_SCHEMA,
        **common,
        "status": "PASS",
        "passed": True,
        "audit_passed": True,
        "independent_verifier": True,
        "frozen": True,
        "artifacts_frozen": True,
        "raw_output_complete": True,
        "raw_artifacts_complete": True,
        "selection_frozen_before_test": True,
        "test_used_for_selection": False,
        "checks": checks,
        "freeze_manifest_sha256": sha256_file(output / "freeze_manifest.json"),
        "verified_at": _utc_now(),
    }
    atomic_json(output / "final_artifact_audit.json", audit)
    run.update(
        {
            "status": "PASS",
            "state": "PASS",
            "run_complete": True,
            "raw_output_complete": True,
            "source_artifacts_complete": True,
            "frozen": True,
            "artifacts_frozen": True,
            "finalized": True,
            "independent_terminal_verification_required": False,
            "independent_terminal_proof": True,
            "independent_verifier": True,
            "worker_wrote_pass": False,
            "terminal_verifier": "separate_verify_only_invocation",
            "final_artifact_audit_sha256": sha256_file(output / "final_artifact_audit.json"),
            "completed_at": _utc_now(),
        }
    )
    atomic_json(output / "run_manifest.json", run)
    registry = audit_explicit_candidate(output, dataset=DATASET, method=METHOD)
    if registry.status not in PASS_STATUSES:
        audit.update(
            {
                "status": "FAILED",
                "passed": False,
                "audit_passed": False,
                "registry_status": registry.status.value,
                "registry_reason_codes": registry.reason_codes,
            }
        )
        atomic_json(output / "final_artifact_audit.json", audit)
        run.update({"status": "FAILED", "state": "FAILED", "run_complete": False})
        atomic_json(output / "run_manifest.json", run)
        _atomic_bytes(output / "FAILED", b"FAILED\n")
        raise TasteGCFPostprocessError(
            "T12 registry gate failed: " + ";".join(registry.reason_codes)
        )
    audit["registry_status"] = registry.status.value
    audit["registry_reason_codes"] = []
    audit["independent_verification_root"] = str(verification)
    atomic_json(output / "final_artifact_audit.json", audit)
    run["final_artifact_audit_sha256"] = sha256_file(output / "final_artifact_audit.json")
    atomic_json(output / "run_manifest.json", run)
    _publish_terminal_pass(output=output, verification=verification, audit=audit)
    _write_checkpoint(
        output,
        phase="PASS",
        identity=authority.resume_identity(),
        detail={"final_artifact_audit_sha256": sha256_file(output / "final_artifact_audit.json")},
    )
    return audit


__all__ = [
    "PASS_MARKER",
    "RUN_MANIFEST_SCHEMA",
    "STAGE",
    "VERIFY_SCHEMA",
    "InputAuthority",
    "TasteGCFPostprocessError",
    "derive_candidate_pool",
    "evaluate_split_resumable",
    "load_input_authority",
    "run_t12_postprocess",
    "select_on_calibration",
    "standardized_metrics",
    "verify_t12_output",
]
