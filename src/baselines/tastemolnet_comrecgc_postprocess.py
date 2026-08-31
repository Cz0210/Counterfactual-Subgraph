"""TasteMolNet T14 ComRecGC paper-cell postprocessing.

The T14 generation worker deliberately stops at a train-only
``GENERATION_PASS``.  This dataset-specific continuation reopens that exact
checkpoint, materializes only the official common-recourse representatives,
orders them on calibration, and opens held-out test only after the order is
durably frozen.  A separate invocation verifies the sealed science tree and
atomically publishes the paper-cell root.
"""

from __future__ import annotations

import ctypes
import csv
from dataclasses import dataclass
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

from src.baselines.comrecgc.generation_checkpoint import load_generation_checkpoint
from src.baselines.tastemolnet_comrecgc_full import (
    M_FALLBACK_MAX,
    M_MAX,
    MIN_VALID_UNIQUE_RULES,
    RUNTIME_STATE_SCHEMA,
    validate_t14_full_output,
)
from src.baselines.tastemolnet_globalgce_full import (
    GINE_FILES,
    FrozenTasteGINEScorer,
    ThresholdContract,
    compute_standardized_metrics,
    load_prepared_split,
    load_threshold_contract,
    select_rules_on_calibration,
)
from src.eval.four_by_four_registry import (
    PASS_STATUSES,
    audit_explicit_candidate,
)
from src.eval.node_wasserstein_distance import (
    MolCLRNodeWassersteinConfig,
    MolCLRNodeWassersteinDistance,
)


STAGE = "T14_COMRECGC_FULL_POSTPROCESS"
DATASET = "TasteMolNet"
METHOD = "ComRecGC"
SOURCE_LABEL = 1
DESTINATION_LABELS = (0, 2)
NUM_CLASSES = 3
K_MAX = 20
TABLE2_K = 10
DISTANCE_LINE = "MolCLR-Node-Wasserstein"
DISTANCE_NAMESPACE = "tastemolnet_comrecgc_full_wnode_v1"
CF_MODE = "strict_flip"
SELECTION_SCHEMA = "tastemolnet_t14_calibration_selection_v1"
RUN_MANIFEST_SCHEMA = "tastemolnet_t14_postprocess_run_manifest_v1"
VERIFY_SCHEMA = "tastemolnet_t14_postprocess_terminal_verification_v1"
CHUNK_MANIFEST_SCHEMA = "tastemolnet_t14_pair_chunk_manifest_v1"
CHUNK_INVENTORY_SCHEMA = "tastemolnet_t14_pair_chunk_inventory_v1"
PASS_MARKER = "[TASTE_COMRECGC_PASS]"


class TasteComRecGCPostprocessError(RuntimeError):
    """T14 paper continuation violated a split, oracle, or output contract."""


@dataclass(frozen=True, slots=True)
class PostprocessAuthority:
    generation_root: Path
    calibration_path: Path
    test_path: Path
    checkpoint_path: Path
    molclr_root: Path
    molclr_checkpoint: Path
    threshold_path: Path
    generation_inventory_sha256: str
    generation_manifest_sha256: str
    generation_checkpoint_digest: str
    generation_effective_m: int
    generation_stop_reason: str
    generation_resource_cap_used: bool
    generation_early_stop_used: bool
    calibration_sha256: str
    declared_test_sha256: str
    checkpoint_id: str
    temperature_calibration_hash: str
    dataset_hash: str
    split_manifest_sha256: str
    molclr_checkpoint_sha256: str
    threshold: ThresholdContract

    def resume_identity(self) -> dict[str, Any]:
        return {
            "schema_version": "tastemolnet_t14_postprocess_resume_identity_v1",
            "dataset": DATASET,
            "method": METHOD,
            "stage": STAGE,
            "generation_inventory_sha256": self.generation_inventory_sha256,
            "generation_manifest_sha256": self.generation_manifest_sha256,
            "generation_checkpoint_digest": self.generation_checkpoint_digest,
            "M_configured_max": M_MAX,
            "M_fallback_max": M_FALLBACK_MAX,
            "M_effective": self.generation_effective_m,
            "stop_reason": self.generation_stop_reason,
            "calibration_sha256": self.calibration_sha256,
            "declared_test_sha256": self.declared_test_sha256,
            "oracle_checkpoint_hash": self.checkpoint_id,
            "temperature_calibration_hash": self.temperature_calibration_hash,
            "dataset_hash": self.dataset_hash,
            "split_manifest_sha256": self.split_manifest_sha256,
            "molclr_checkpoint_hash": self.molclr_checkpoint_sha256,
            "threshold_config_hash": self.threshold.config_hash,
        }


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path_like: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path_like).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(payload: Any) -> str:
    return sha256_bytes(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(
        path,
        (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(
                dict(row),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        )
    ).encode("utf-8")


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_bytes(path, jsonl_bytes(rows))


def csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise TasteComRecGCPostprocessError("cannot serialize an empty paper CSV")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_bytes(path, csv_bytes(rows))


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteComRecGCPostprocessError(f"invalid JSON: {path}") from exc
    if type(payload) is not dict:
        raise TasteComRecGCPostprocessError(f"JSON must be an object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise TasteComRecGCPostprocessError(f"cannot read JSONL: {path}") from exc
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TasteComRecGCPostprocessError(
                f"invalid JSONL row {number}: {path}"
            ) from exc
        if type(row) is not dict:
            raise TasteComRecGCPostprocessError(
                f"JSONL row {number} is not an object: {path}"
            )
        rows.append(row)
    return rows


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _checkpoint_payloads(checkpoint: Path) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for name in GINE_FILES:
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size <= 0 or path.is_symlink():
            raise TasteComRecGCPostprocessError(f"frozen GINE is missing {name}")
        payloads[name] = path.read_bytes()
    return payloads


def load_postprocess_authority(
    *,
    generation_root: str | Path,
    calibration_csv: str | Path,
    test_csv: str | Path,
    gnn_checkpoint: str | Path,
    molclr_root: str | Path,
    molclr_checkpoint: str | Path,
    threshold_contract: str | Path,
) -> PostprocessAuthority:
    """Reopen every pre-test authority without opening held-out test bytes."""

    generation = Path(generation_root).expanduser().resolve(strict=True)
    calibration = Path(calibration_csv).expanduser().resolve(strict=True)
    test = Path(test_csv).expanduser().absolute()
    checkpoint = Path(gnn_checkpoint).expanduser().resolve(strict=True)
    molclr_source = Path(molclr_root).expanduser().resolve(strict=True)
    molclr_ckpt = Path(molclr_checkpoint).expanduser().resolve(strict=True)
    threshold_path = Path(threshold_contract).expanduser().resolve(strict=True)
    generation_verification = validate_t14_full_output(generation)
    manifest = read_json(generation / "generation_manifest.json")
    resource = read_json(generation / "resource_cap_receipt.json")
    checkpoint_identity = read_json(generation / "checkpoint_identity.json")
    effective = resource.get("m_effective")
    if (
        generation_verification.get("status") != "PASS"
        or type(effective) is not int
        or effective not in {M_MAX, M_FALLBACK_MAX}
        or resource.get("state") != "STOP_AND_POSTPROCESS"
        or resource.get("resource_cap_used") is not True
        or type(resource.get("early_stop_used")) is not bool
        or not str(resource.get("stop_reason") or "")
        or manifest.get("same_frozen_three_class_gine") is not True
        or manifest.get("rf_oracle_used") is not False
        or manifest.get("validation_loaded") is not False
        or manifest.get("calibration_loaded") is not False
        or manifest.get("test_loaded") is not False
    ):
        raise TasteComRecGCPostprocessError("T14 generation is not postprocess eligible")
    identity_provenance = checkpoint_identity.get("provenance")
    if type(identity_provenance) is not dict:
        raise TasteComRecGCPostprocessError("T14 checkpoint identity is incomplete")
    loaded = load_generation_checkpoint(
        generation / "checkpoints",
        expected_provenance=identity_provenance,
        expected_scientific_argv=checkpoint_identity.get("scientific_argv"),
        expected_command_sha256=checkpoint_identity.get("command_sha256"),
        expected_total_steps=M_FALLBACK_MAX,
        expected_completed_step=effective,
    )
    payloads = _checkpoint_payloads(checkpoint)
    card = json.loads(payloads["model_card.json"].decode("utf-8"))
    checkpoint_id = sha256_bytes(payloads["model.pt"])
    if (
        card.get("dataset") not in {"TasteMolNet", "tastemolnet"}
        or card.get("oracle_backend") != "gnn"
        or card.get("rf_oracle_used") is not False
        or str(card.get("backbone") or "").lower() != "gine"
        or card.get("num_classes") != NUM_CLASSES
        or card.get("source_label") != SOURCE_LABEL
        or card.get("checkpoint_id") != checkpoint_id
        or identity_provenance.get("checkpoint_id") != checkpoint_id
    ):
        raise TasteComRecGCPostprocessError(
            "T14 generation/postprocess GINE identity differs"
        )
    split = json.loads(payloads["split_manifest.json"].decode("utf-8"))
    files = split.get("files")
    roles = split.get("roles")
    if (
        split.get("schema_version") != "molecular_gnn_split_manifest_v1"
        or split.get("dataset") not in {"TasteMolNet", "tastemolnet"}
        or type(files) is not dict
        or set(files) != {"train", "validation", "calibration", "test"}
        or type(roles) is not dict
        or roles.get("calibration") != "reserved_for_threshold_and_selector_only"
        or roles.get("test") != "frozen_model_final_quality_evaluation"
        or split.get("calibration_loaded_for_training") is not False
        or split.get("test_loaded_for_training") is not False
        or split.get("test_used_for_checkpoint_selection") is not False
    ):
        raise TasteComRecGCPostprocessError("Taste split-role contract changed")
    declared: dict[str, str] = {}
    for role in ("train", "calibration", "test"):
        row = files.get(role)
        if type(row) is not dict or not _is_sha256(row.get("sha256")):
            raise TasteComRecGCPostprocessError(f"split manifest lacks {role} SHA")
        declared[role] = str(row["sha256"]).lower()
    calibration_sha = sha256_file(calibration)
    if (
        calibration_sha != declared["calibration"]
        or identity_provenance.get("train_csv_sha256") != declared["train"]
    ):
        raise TasteComRecGCPostprocessError("T14 split bytes differ from generation")
    train_manifest = split.get("train_manifest")
    dataset_hash = (
        str((train_manifest or {}).get("dataset_fingerprint") or "").lower()
        if type(train_manifest) is dict
        else ""
    )
    if not _is_sha256(dataset_hash):
        raise TasteComRecGCPostprocessError("Taste dataset fingerprint is absent")
    threshold = load_threshold_contract(threshold_path)
    return PostprocessAuthority(
        generation_root=generation,
        calibration_path=calibration,
        test_path=test,
        checkpoint_path=checkpoint,
        molclr_root=molclr_source,
        molclr_checkpoint=molclr_ckpt,
        threshold_path=threshold_path,
        generation_inventory_sha256=str(generation_verification["inventory_sha256"]),
        generation_manifest_sha256=sha256_file(generation / "generation_manifest.json"),
        generation_checkpoint_digest=loaded.validation.checkpoint_digest,
        generation_effective_m=effective,
        generation_stop_reason=str(resource["stop_reason"]),
        generation_resource_cap_used=bool(resource["resource_cap_used"]),
        generation_early_stop_used=bool(resource["early_stop_used"]),
        calibration_sha256=calibration_sha,
        declared_test_sha256=declared["test"],
        checkpoint_id=checkpoint_id,
        temperature_calibration_hash=sha256_bytes(payloads["temperature_scaling.json"]),
        dataset_hash=dataset_hash,
        split_manifest_sha256=sha256_bytes(payloads["split_manifest.json"]),
        molclr_checkpoint_sha256=sha256_file(molclr_ckpt),
        threshold=threshold,
    )


def materialize_generation_candidates(
    *, generation_manifest: Mapping[str, Any], bridge_state: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Losslessly join official representatives to checkpointed graph payloads."""

    common = generation_manifest.get("common_recourse")
    records = bridge_state.get("records") if type(bridge_state) is dict else None
    lineages = (
        bridge_state.get("lineage_occurrences")
        if type(bridge_state) is dict
        else None
    )
    rows = common.get("selected_common_recourses") if type(common) is dict else None
    if (
        type(records) is not dict
        or type(lineages) is not dict
        or type(rows) is not list
        or not MIN_VALID_UNIQUE_RULES <= len(rows) <= K_MAX
        or common.get("selected_common_recourse_count") != len(rows)
    ):
        raise TasteComRecGCPostprocessError(
            "T14 official common-recourse set must contain 10..20 candidates"
        )
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for expected_rank, source in enumerate(rows, start=1):
        graph_hash = str(source.get("representative_graph_identity_sha256") or "")
        record = records.get(graph_hash)
        lineage = lineages.get(graph_hash)
        if (
            type(source) is not dict
            or source.get("rank") != expected_rank
            or not _is_sha256(graph_hash)
            or graph_hash in seen
            or type(record) is not dict
            or record.get("graph_identity_sha256") != graph_hash
            or record.get("candidate") is not True
            or record.get("valid_fullgraph") is not True
            or record.get("prediction") not in DESTINATION_LABELS
            or record.get("prediction") != source.get("destination_label")
            or type(record.get("canonical_graph")) is not str
            or not record["canonical_graph"]
            or type(lineage) is not dict
            or len(lineage) != source.get("lineage_count")
            or not lineage
            or any(not _is_sha256(key) or type(count) is not int or count <= 0 for key, count in lineage.items())
        ):
            raise TasteComRecGCPostprocessError(
                "T14 candidate graph/lineage checkpoint join changed"
            )
        try:
            from rdkit import Chem

            molecule = Chem.MolFromSmiles(record["canonical_graph"])
        except Exception as exc:
            raise TasteComRecGCPostprocessError(
                "T14 common recourse is not a parseable molecule"
            ) from exc
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            raise TasteComRecGCPostprocessError(
                "T14 common recourse is not one connected full graph"
            )
        roundtrip = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=False,
            allHsExplicit=True,
        )
        if not roundtrip:
            raise TasteComRecGCPostprocessError("T14 candidate canonicalization failed")
        # The checkpoint string was itself produced by the native attributed-
        # graph canonicalizer.  Keep those exact bytes as the GINE/MolCLR input
        # instead of letting a second serializer alter explicit-H spelling.
        canonical = str(record["canonical_graph"])
        seen.add(graph_hash)
        candidates.append(
            {
                "dataset": DATASET,
                "method": METHOD,
                "stage": STAGE,
                "candidate_id": f"comrecgc_{graph_hash}",
                "candidate_content_hash": graph_hash,
                "generation_rank": expected_rank,
                "cluster_id": int(source["cluster_id"]),
                "canonical_smiles": canonical,
                "rdkit_roundtrip_smiles": roundtrip,
                "destination_label": int(source["destination_label"]),
                "score": float(source["score"]),
                "frequency": int(source["frequency"]),
                "covered_parent_count": int(source["covered_parent_count"]),
                "cluster_size": int(source["cluster_size"]),
                "lineage_count": len(lineage),
                "lineage_sha256": stable_sha256(lineage),
                "source_split": "train",
                "action_kind": "full_graph_common_recourse",
                "action_semantics": "official_comrecgc_cluster_representative_v1",
                "oracle_backend": "gnn",
                "classifier_family": "gine",
                "rf_oracle_used": False,
            }
        )
    return candidates


def load_generation_candidates(authority: PostprocessAuthority) -> list[dict[str, Any]]:
    generation = authority.generation_root
    identity = read_json(generation / "checkpoint_identity.json")
    resource = read_json(generation / "resource_cap_receipt.json")
    loaded = load_generation_checkpoint(
        generation / "checkpoints",
        expected_provenance=identity.get("provenance"),
        expected_scientific_argv=identity.get("scientific_argv"),
        expected_command_sha256=identity.get("command_sha256"),
        expected_total_steps=M_FALLBACK_MAX,
        expected_completed_step=resource.get("m_effective"),
    )
    if (
        loaded.validation.checkpoint_digest != authority.generation_checkpoint_digest
        or loaded.algorithm_state.get("schema_version") != RUNTIME_STATE_SCHEMA
        or type(loaded.algorithm_state.get("bridge_state")) is not dict
    ):
        raise TasteComRecGCPostprocessError("T14 effective checkpoint changed")
    return materialize_generation_candidates(
        generation_manifest=read_json(generation / "generation_manifest.json"),
        bridge_state=loaded.algorithm_state["bridge_state"],
    )


def _prediction(row: Mapping[str, Any], *, checkpoint_id: str) -> dict[str, Any]:
    probabilities = row.get("probabilities")
    predicted = row.get("predicted_label")
    if (
        row.get("checkpoint_id") != checkpoint_id
        or row.get("num_classes") != NUM_CLASSES
        or row.get("source_label") != SOURCE_LABEL
        or str(row.get("backbone") or "").lower() != "gine"
        or type(predicted) is not int
        or predicted not in range(NUM_CLASSES)
        or type(probabilities) is not list
        or len(probabilities) != NUM_CLASSES
        or any(
            isinstance(value, bool) or not math.isfinite(float(value))
            for value in probabilities
        )
        or max(
            range(NUM_CLASSES), key=lambda index: float(probabilities[index])
        )
        != predicted
    ):
        raise TasteComRecGCPostprocessError(
            "T14 prediction differs from the frozen three-class GINE"
        )
    return {
        "predicted_label": predicted,
        "probabilities": [float(value) for value in probabilities],
    }


def _parent_cohort_sha256(parents: Sequence[Any]) -> str:
    return stable_sha256(
        [
            {
                "parent_id": str(parent.parent_id),
                "smiles": str(parent.smiles),
                "split": str(parent.split),
            }
            for parent in parents
        ]
    )


def build_split_evaluation_identity(
    *,
    split: str,
    parents: Sequence[Any],
    candidates: Sequence[Mapping[str, Any]],
    authority: PostprocessAuthority,
) -> dict[str, Any]:
    if (
        split not in {"calibration", "test"}
        or not parents
        or any(str(parent.split) != split for parent in parents)
        or not MIN_VALID_UNIQUE_RULES <= len(candidates) <= K_MAX
    ):
        raise TasteComRecGCPostprocessError("T14 split evaluation identity is invalid")
    candidate_identities = [
        {
            "candidate_id": str(row.get("candidate_id") or ""),
            "candidate_content_hash": str(row.get("candidate_content_hash") or ""),
        }
        for row in candidates
    ]
    if any(
        not row["candidate_id"] or not _is_sha256(row["candidate_content_hash"])
        for row in candidate_identities
    ):
        raise TasteComRecGCPostprocessError("T14 candidate identity is incomplete")
    return {
        "schema_version": "tastemolnet_t14_split_evaluation_identity_v1",
        "split": split,
        "parent_cohort_sha256": _parent_cohort_sha256(parents),
        "parent_count": len(parents),
        "ordered_candidates_sha256": stable_sha256(candidate_identities),
        "candidate_count": len(candidates),
        "oracle_checkpoint_hash": authority.checkpoint_id,
        "temperature_calibration_hash": authority.temperature_calibration_hash,
        "molclr_checkpoint_hash": authority.molclr_checkpoint_sha256,
        "threshold_config_hash": authority.threshold.config_hash,
        "generation_checkpoint_digest": authority.generation_checkpoint_digest,
        "distance_line": DISTANCE_LINE,
        "distance_namespace": DISTANCE_NAMESPACE,
        "cf_mode": CF_MODE,
        "num_classes": NUM_CLASSES,
        "source_label": SOURCE_LABEL,
    }


def evaluate_one_parent(
    *,
    parent: Any,
    candidates: Sequence[Mapping[str, Any]],
    scorer: Any,
    provider: Any,
    split: str,
    candidate_predictions: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate one parent against immutable full-graph common recourses."""

    before = _prediction(
        scorer.score_smiles([str(parent.smiles)])[0],
        checkpoint_id=scorer.checkpoint_id,
    )
    if candidate_predictions is None:
        scored = scorer.score_smiles(
            [str(row["canonical_smiles"]) for row in candidates]
        )
        if len(scored) != len(candidates):
            raise TasteComRecGCPostprocessError("T14 candidate score count changed")
        candidate_predictions = {
            str(candidate["candidate_id"]): _prediction(
                raw, checkpoint_id=scorer.checkpoint_id
            )
            for candidate, raw in zip(candidates, scored, strict=True)
        }
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        after = dict(candidate_predictions[candidate_id])
        strict = (
            before["predicted_label"] == SOURCE_LABEL
            and after["predicted_label"] in DESTINATION_LABELS
        )
        if after["predicted_label"] != int(candidate["destination_label"]):
            raise TasteComRecGCPostprocessError(
                "T14 checkpointed candidate changed frozen-GINE destination"
            )
        distance: float | None = None
        distance_failure: str | None = None
        if strict:
            result = provider.distance(
                str(parent.smiles), str(candidate["canonical_smiles"])
            )
            value = result.get("distance") if type(result) is dict else None
            if (
                type(result) is dict
                and result.get("ok") is True
                and value is not None
                and math.isfinite(float(value))
                and float(value) >= 0.0
            ):
                distance = float(value)
            else:
                distance_failure = str(
                    (result or {}).get("error")
                    if type(result) is dict
                    else "wnode_distance_failed"
                )
        source_drop = (
            before["probabilities"][SOURCE_LABEL]
            - after["probabilities"][SOURCE_LABEL]
        )
        rows.append(
            {
                "dataset": DATASET,
                "method": METHOD,
                "stage": STAGE,
                "split": split,
                "parent_id": str(parent.parent_id),
                "parent_smiles": str(parent.smiles),
                "candidate_id": candidate_id,
                "candidate_content_hash": candidate["candidate_content_hash"],
                "generation_rank": candidate["generation_rank"],
                "action_kind": "full_graph_common_recourse",
                "action_semantics": "official_comrecgc_cluster_representative_v1",
                "applicable": True,
                "canonical_smiles": candidate["canonical_smiles"],
                "pred_before": before["predicted_label"],
                "pred_after": after["predicted_label"],
                "p_before": before["probabilities"],
                "p_after": after["probabilities"],
                "p1_before": before["probabilities"][SOURCE_LABEL],
                "p1_after": after["probabilities"][SOURCE_LABEL],
                "cf_drop": source_drop,
                "cf_flip": strict,
                "pair_strict_flip": distance is not None,
                "destination_label": after["predicted_label"] if strict else None,
                "wnode_distance": distance,
                "distance_for_selection": distance if distance is not None else "+inf",
                "failure_reason": (
                    None
                    if distance is not None
                    else distance_failure
                    if distance_failure
                    else "frozen_gine_not_strict_flip"
                ),
                "cf_mode": CF_MODE,
                "source_label": SOURCE_LABEL,
                "oracle_backend": "gnn",
                "classifier_family": "gine",
                "rf_oracle_used": False,
                "oracle_checkpoint_hash": scorer.checkpoint_id,
            }
        )
    return rows


def _write_checkpoint(
    output: Path,
    *,
    phase: str,
    resume_identity: Mapping[str, Any],
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": "tastemolnet_t14_postprocess_checkpoint_v1",
        "phase": phase,
        "resume_identity": dict(resume_identity),
        "resume_identity_sha256": stable_sha256(resume_identity),
        "detail": dict(detail or {}),
        "updated_at": utc_now(),
    }
    atomic_json(output / "postprocess_checkpoint.json", payload)
    return payload


def evaluate_split_resumable(
    *,
    split: str,
    parents: Sequence[Any],
    candidates: Sequence[Mapping[str, Any]],
    scorer: Any,
    provider: Any,
    output: Path,
    authority: PostprocessAuthority,
    resume_identity: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunks = output / "raw" / f"{split}_pair_chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    identity = build_split_evaluation_identity(
        split=split,
        parents=parents,
        candidates=candidates,
        authority=authority,
    )
    identity_sha = stable_sha256(identity)
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    scored = scorer.score_smiles(
        [str(row["canonical_smiles"]) for row in candidates]
    )
    if len(scored) != len(candidates):
        raise TasteComRecGCPostprocessError("T14 candidate score count changed")
    candidate_predictions = {
        candidate_id: _prediction(raw, checkpoint_id=scorer.checkpoint_id)
        for candidate_id, raw in zip(candidate_ids, scored, strict=True)
    }
    all_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
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
                or manifest.get("parent_id") != str(parent.parent_id)
                or manifest.get("evaluation_identity_sha256") != identity_sha
                or manifest.get("candidate_ids_sha256") != stable_sha256(candidate_ids)
                or manifest.get("row_count") != len(candidates)
                or manifest.get("rows_sha256") != sha256_file(chunk)
                or len(rows) != len(candidates)
                or [str(row.get("candidate_id")) for row in rows] != candidate_ids
                or any(
                    row.get("parent_id") != str(parent.parent_id)
                    or row.get("split") != split
                    for row in rows
                )
            ):
                raise TasteComRecGCPostprocessError(
                    f"T14 {split} resume chunk changed"
                )
        elif chunk.exists() or manifest_path.exists():
            raise TasteComRecGCPostprocessError(
                f"T14 {split} has a partially committed chunk"
            )
        else:
            rows = evaluate_one_parent(
                parent=parent,
                candidates=candidates,
                scorer=scorer,
                provider=provider,
                split=split,
                candidate_predictions=candidate_predictions,
            )
            atomic_jsonl(chunk, rows)
            atomic_json(
                manifest_path,
                {
                    "schema_version": CHUNK_MANIFEST_SCHEMA,
                    "split": split,
                    "position": position,
                    "parent_id": str(parent.parent_id),
                    "evaluation_identity_sha256": identity_sha,
                    "candidate_ids_sha256": stable_sha256(candidate_ids),
                    "row_count": len(rows),
                    "rows_sha256": sha256_file(chunk),
                },
            )
        all_rows.extend(rows)
        inventory_rows.append(
            {
                "position": position,
                "parent_id": str(parent.parent_id),
                "chunk": chunk.relative_to(output).as_posix(),
                "chunk_sha256": sha256_file(chunk),
                "manifest": manifest_path.relative_to(output).as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
        _write_checkpoint(
            output,
            phase=f"{split.upper()}_RUNNING",
            resume_identity=resume_identity,
            detail={"completed_parent_count": position + 1, "parent_count": len(parents)},
        )
    details_path = output / "raw" / f"{split}_pair_details.jsonl"
    atomic_jsonl(details_path, all_rows)
    inventory = {
        "schema_version": CHUNK_INVENTORY_SCHEMA,
        "split": split,
        "evaluation_identity": identity,
        "evaluation_identity_sha256": identity_sha,
        "parent_count": len(parents),
        "candidate_count": len(candidates),
        "pair_count": len(all_rows),
        "chunks": inventory_rows,
        "chunks_sha256": stable_sha256(inventory_rows),
        "pair_details_sha256": sha256_file(details_path),
    }
    inventory_path = output / "raw" / f"{split}_pair_chunk_inventory.json"
    atomic_json(inventory_path, inventory)
    return all_rows, {
        "split": split,
        "parent_count": len(parents),
        "candidate_count": len(candidates),
        "pair_count": len(all_rows),
        "pair_details_sha256": sha256_file(details_path),
        "parent_ids_sha256": stable_sha256(
            sorted(str(parent.parent_id) for parent in parents)
        ),
        "candidate_ids_sha256": stable_sha256(candidate_ids),
        "resumable_parent_chunks": True,
        "checkpointed_parent_count": len(parents),
        "evaluation_identity_sha256": identity_sha,
        "chunk_inventory_sha256": sha256_file(inventory_path),
    }


def verify_pair_chunk_inventory(
    *, output: Path, split: str, pair_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    inventory_path = output / "raw" / f"{split}_pair_chunk_inventory.json"
    inventory = read_json(inventory_path)
    chunks = inventory.get("chunks")
    identity = inventory.get("evaluation_identity")
    if (
        inventory.get("schema_version") != CHUNK_INVENTORY_SCHEMA
        or inventory.get("split") != split
        or type(chunks) is not list
        or type(identity) is not dict
        or inventory.get("evaluation_identity_sha256") != stable_sha256(identity)
        or inventory.get("chunks_sha256") != stable_sha256(chunks)
        or inventory.get("pair_details_sha256")
        != sha256_file(output / "raw" / f"{split}_pair_details.jsonl")
    ):
        raise TasteComRecGCPostprocessError(f"T14 {split} inventory changed")
    reconstructed: list[dict[str, Any]] = []
    for position, entry in enumerate(chunks):
        if type(entry) is not dict or entry.get("position") != position:
            raise TasteComRecGCPostprocessError(f"T14 {split} chunk order changed")
        chunk = output / str(entry.get("chunk"))
        manifest_path = output / str(entry.get("manifest"))
        if (
            not chunk.is_file()
            or not manifest_path.is_file()
            or sha256_file(chunk) != entry.get("chunk_sha256")
            or sha256_file(manifest_path) != entry.get("manifest_sha256")
        ):
            raise TasteComRecGCPostprocessError(f"T14 {split} chunk bytes changed")
        manifest = read_json(manifest_path)
        rows = read_jsonl(chunk)
        if (
            manifest.get("schema_version") != CHUNK_MANIFEST_SCHEMA
            or manifest.get("split") != split
            or manifest.get("position") != position
            or manifest.get("rows_sha256") != sha256_file(chunk)
            or manifest.get("row_count") != len(rows)
        ):
            raise TasteComRecGCPostprocessError(
                f"T14 {split} chunk manifest changed"
            )
        reconstructed.extend(rows)
    if reconstructed != [dict(row) for row in pair_rows]:
        raise TasteComRecGCPostprocessError(
            f"T14 {split} chunks differ from pair details"
        )
    if (
        inventory.get("parent_count") != len(chunks)
        or inventory.get("pair_count") != len(pair_rows)
    ):
        raise TasteComRecGCPostprocessError(f"T14 {split} counts changed")
    return inventory


def select_candidates_on_calibration(
    candidates: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    theta_star: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        selected, source = select_rules_on_calibration(
            candidates, pair_rows, theta_star=theta_star
        )
    except Exception as exc:
        raise TasteComRecGCPostprocessError(
            "T14 calibration-only candidate ordering failed"
        ) from exc
    ordered = [str(row["candidate_id"]) for row in selected]
    result = dict(source)
    result.update(
        {
            "selector": (
                "calibration_greedy_marginal_theta_then_strict_then_total_coverage_"
                "then_mean_wnode_then_candidate_id_v1"
            ),
            "ordered_candidate_ids": ordered,
            "ordered_candidate_ids_sha256": stable_sha256(ordered),
        }
    )
    result.pop("ordered_rule_ids", None)
    result.pop("ordered_rule_ids_sha256", None)
    return selected, result


def compute_t14_standardized_metrics(
    pair_rows: Sequence[Mapping[str, Any]],
    ordered_candidate_ids: Sequence[str],
    threshold: ThresholdContract,
) -> dict[str, Any]:
    """Use the same cross-method prefix metric, changing only method identity."""

    try:
        result = compute_standardized_metrics(
            pair_rows, ordered_candidate_ids, threshold
        )
    except Exception as exc:
        raise TasteComRecGCPostprocessError(
            "T14 standardized metric replay failed"
        ) from exc
    for key in ("prefix", "parent_best", "figure3", "figure4", "table2", "destination"):
        for row in result[key]:
            row["method"] = METHOD
    return result


def _artifact_inventory(output: Path) -> dict[str, dict[str, Any]]:
    required = {
        "figure3_coverage_vs_k.csv",
        "figure4_coverage_vs_threshold.csv",
        "prefix_metrics.csv",
        "prefix_metrics.json",
        "parent_best_distances.csv",
        "destination_distribution.csv",
        "table2_comrecgc_k10.csv",
        "summary.json",
        "oracle_manifest.json",
        "evaluation_manifest.json",
        "raw/candidate_pool.jsonl",
        "raw/calibration_pair_details.jsonl",
        "raw/calibration_pair_chunk_inventory.json",
        "raw/selected_candidates.jsonl",
        "raw/selection_manifest.json",
        "raw/test_pair_details.jsonl",
        "raw/test_pair_chunk_inventory.json",
        "raw/test_evaluation_manifest.json",
    }
    names = set(required)
    raw = output / "raw"
    for path in sorted(raw.rglob("*")):
        if path.is_symlink():
            raise TasteComRecGCPostprocessError(
                "T14 postprocess raw tree contains a symbolic link"
            )
        if path.is_file():
            names.add(path.relative_to(output).as_posix())
    inventory: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        path = output / name
        if not path.is_file() or (name in required and path.stat().st_size <= 0):
            raise TasteComRecGCPostprocessError(
                f"T14 postprocess artifact is absent: {name}"
            )
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise TasteComRecGCPostprocessError(
                f"T14 postprocess artifact is not regular: {name}"
            )
        inventory[name] = {"bytes": info.st_size, "sha256": sha256_file(path)}
    return inventory


def _common_manifest(
    *,
    authority: PostprocessAuthority,
    science_root: Path,
    test_parent_ids_sha256: str,
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
        "temperature_calibration_hash": authority.temperature_calibration_hash,
        "dataset_hash": authority.dataset_hash,
        "test_parent_ids_sha256": test_parent_ids_sha256,
        "test_split_hash": authority.declared_test_sha256,
        "distance_line": DISTANCE_LINE,
        "molclr_checkpoint_hash": authority.molclr_checkpoint_sha256,
        "cf_mode": CF_MODE,
        "threshold_config_hash": authority.threshold.config_hash,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "raw_output_root": str(science_root),
        "source_generation_root": str(authority.generation_root),
        "generation_manifest_sha256": authority.generation_manifest_sha256,
        "generation_checkpoint_digest": authority.generation_checkpoint_digest,
        "generation_adopted": True,
        "M_configured_max": M_MAX,
        "M_fallback_max": M_FALLBACK_MAX,
        "M_effective": authority.generation_effective_m,
        "resource_cap_used": authority.generation_resource_cap_used,
        "early_stop_used": authority.generation_early_stop_used,
        "stop_reason": authority.generation_stop_reason,
    }


def run_t14_postprocess(
    *,
    authority: PostprocessAuthority,
    science_root: str | Path,
    resume: bool,
    device: str,
    wnode_cache_db: str | Path,
    node_embedding_cache_dir: str | Path,
) -> dict[str, Any]:
    """Materialize, calibration-freeze, test, export, and seal T14 science."""

    if device != "cuda:0":
        raise TasteComRecGCPostprocessError("T14 postprocess is bound to logical cuda:0")
    output = Path(science_root).expanduser().absolute()
    identity = authority.resume_identity()
    checkpoint_path = output / "postprocess_checkpoint.json"
    if resume:
        if not output.is_dir() or not checkpoint_path.is_file():
            raise TasteComRecGCPostprocessError(
                "T14 postprocess resume requires an existing checkpoint"
            )
        checkpoint = read_json(checkpoint_path)
        if (
            checkpoint.get("resume_identity") != identity
            or checkpoint.get("resume_identity_sha256") != stable_sha256(identity)
        ):
            raise TasteComRecGCPostprocessError("T14 postprocess resume identity changed")
        if checkpoint.get("phase") == "SEALED":
            return read_json(output / "run_manifest.json")
    else:
        if output.exists():
            raise FileExistsError(f"fresh T14 postprocess root exists: {output}")
        output.mkdir(parents=True, mode=0o700)
        (output / "raw").mkdir(mode=0o700)
        _fsync_dir(output.parent)
        _write_checkpoint(output, phase="INITIALIZED", resume_identity=identity)
    if (output / "PASS").exists():
        raise TasteComRecGCPostprocessError("science worker cannot write terminal PASS")

    candidates = load_generation_candidates(authority)
    candidate_path = output / "raw" / "candidate_pool.jsonl"
    if candidate_path.is_file():
        if read_jsonl(candidate_path) != candidates:
            raise TasteComRecGCPostprocessError("T14 materialized candidates changed")
    else:
        atomic_jsonl(candidate_path, candidates)
    _write_checkpoint(
        output,
        phase="TRAIN_CANDIDATES_FROZEN",
        resume_identity=identity,
        detail={"candidate_count": len(candidates), "candidate_pool_sha256": sha256_file(candidate_path)},
    )

    scorer = FrozenTasteGINEScorer(
        _checkpoint_payloads(authority.checkpoint_path),
        device=device,
        batch_size=256,
    )
    if scorer.checkpoint_id != authority.checkpoint_id:
        raise TasteComRecGCPostprocessError("T14 scorer checkpoint changed")
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
            output=output,
            authority=authority,
            resume_identity=identity,
        )
        selected, selection = select_candidates_on_calibration(
            candidates,
            calibration_rows,
            theta_star=authority.threshold.theta_star,
        )
        selected_path = output / "raw" / "selected_candidates.jsonl"
        selection_path = output / "raw" / "selection_manifest.json"
        frozen_at = utc_now()
        selection_manifest = {
            "schema_version": SELECTION_SCHEMA,
            "dataset": DATASET,
            "method": METHOD,
            "stage": STAGE,
            "status": "FROZEN",
            "selection_frozen": True,
            "frozen_at": frozen_at,
            **selection,
            **authority.threshold.to_dict(),
            "calibration_manifest": calibration_manifest,
            "selected_candidates_sha256": sha256_bytes(jsonl_bytes(selected)),
            "oracle_checkpoint_hash": authority.checkpoint_id,
            "molclr_checkpoint_hash": authority.molclr_checkpoint_sha256,
            "rf_oracle_used": False,
            "test_loaded": False,
            "test_used_for_selection": False,
        }
        if selected_path.is_file() or selection_path.is_file():
            if (
                not selected_path.is_file()
                or not selection_path.is_file()
                or read_jsonl(selected_path) != selected
                or read_json(selection_path) != selection_manifest
            ):
                # ``frozen_at`` is not reconstructed on resume.  Reopen the
                # existing receipt and compare every scientific field instead.
                existing = read_json(selection_path) if selection_path.is_file() else {}
                comparable = dict(selection_manifest)
                comparable["frozen_at"] = existing.get("frozen_at")
                if (
                    not selected_path.is_file()
                    or read_jsonl(selected_path) != selected
                    or existing != comparable
                ):
                    raise TasteComRecGCPostprocessError(
                        "T14 frozen calibration order changed"
                    )
                selection_manifest = existing
        else:
            atomic_jsonl(selected_path, selected)
            atomic_json(selection_path, selection_manifest)
        _write_checkpoint(
            output,
            phase="CALIBRATION_SELECTION_FROZEN",
            resume_identity=identity,
            detail={"selection_manifest_sha256": sha256_file(selection_path)},
        )

        # No held-out bytes are opened before the fsynced selection and stage
        # checkpoint above both exist.
        test_started_at = utc_now()
        if sha256_file(authority.test_path) != authority.declared_test_sha256:
            raise TasteComRecGCPostprocessError("held-out test bytes changed")
        test_parents = load_prepared_split(
            authority.test_path,
            expected_split="test",
            expected_sha256=authority.declared_test_sha256,
        )
        test_rows, test_manifest = evaluate_split_resumable(
            split="test",
            parents=test_parents,
            candidates=selected,
            scorer=scorer,
            provider=provider,
            output=output,
            authority=authority,
            resume_identity=identity,
        )
        provider_stats = provider.stats_dict()
    finally:
        provider.close()

    test_manifest.update(
        {
            "started_at": test_started_at,
            "completed_at": utc_now(),
            "selection_manifest_sha256": sha256_file(selection_path),
            "selection_frozen_before_test": True,
            "test_used_for_selection": False,
        }
    )
    atomic_json(output / "raw" / "test_evaluation_manifest.json", test_manifest)
    ordered = [str(row["candidate_id"]) for row in selected]
    metrics = compute_t14_standardized_metrics(test_rows, ordered, authority.threshold)
    outputs = {
        "figure3_coverage_vs_k.csv": metrics["figure3"],
        "figure4_coverage_vs_threshold.csv": metrics["figure4"],
        "prefix_metrics.csv": metrics["prefix"],
        "parent_best_distances.csv": metrics["parent_best"],
        "destination_distribution.csv": metrics["destination"],
        "table2_comrecgc_k10.csv": metrics["table2"],
    }
    for name, rows in outputs.items():
        atomic_csv(output / name, rows)
    atomic_json(output / "prefix_metrics.json", metrics["prefix"])
    test_parent_hash = stable_sha256(sorted(parent.parent_id for parent in test_parents))
    common = _common_manifest(
        authority=authority,
        science_root=output,
        test_parent_ids_sha256=test_parent_hash,
    )
    summary = {
        "schema_version": "tastemolnet_t14_summary_v1",
        **common,
        "status": "SEALED",
        "frozen": True,
        "artifacts_frozen": True,
        "raw_output_complete": True,
        "raw_artifacts_complete": True,
        "selection_frozen_before_test": True,
        "calibration_loaded": True,
        "test_loaded": True,
        "candidate_pool_count": len(candidates),
        "effective_rule_count": metrics["effective_rule_count"],
        "parent_count": metrics["parent_count"],
        "pair_count": metrics["pair_count"],
        "K_MAX": K_MAX,
        "MIN_RULES_FOR_MAIN_TABLE": MIN_VALID_UNIQUE_RULES,
        "distance_provider_stats": provider_stats,
        "threshold_contract": authority.threshold.to_dict(),
    }
    oracle_manifest = {
        "schema_version": "tastemolnet_t14_oracle_manifest_v1",
        **common,
        "temperature": scorer.temperature,
        "num_classes": scorer.num_classes,
        "source_label": scorer.source_label,
        "same_frozen_gine_for_generation_calibration_test": True,
        "calibration_loaded_for_generation": False,
        "test_loaded_for_generation": False,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "frozen": True,
    }
    evaluation_manifest = {
        "schema_version": "tastemolnet_t14_evaluation_manifest_v1",
        **common,
        "status": "SEALED",
        "selection_manifest_sha256": sha256_file(selection_path),
        "test_evaluation_manifest_sha256": sha256_file(
            output / "raw" / "test_evaluation_manifest.json"
        ),
        "selection_frozen_before_test": True,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "strict_flip_definition": "pred_before == 1 and pred_after != 1",
        "destination_labels": [0, 2],
        "full_cartesian_test_pairs": True,
        "frozen": True,
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "oracle_manifest.json", oracle_manifest)
    atomic_json(output / "evaluation_manifest.json", evaluation_manifest)
    inventory = _artifact_inventory(output)
    freeze = {
        "schema_version": "tastemolnet_t14_freeze_manifest_v1",
        **common,
        "status": "SEALED",
        "frozen": True,
        "artifacts_frozen": True,
        "files": inventory,
        "inventory_sha256": stable_sha256(inventory),
        "sealed_at": utc_now(),
    }
    atomic_json(output / "freeze_manifest.json", freeze)
    run_manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        **common,
        "status": "SEALED",
        "state": "SEALED",
        "run_complete": False,
        "raw_output_complete": True,
        "raw_artifacts_complete": True,
        "source_artifacts_complete": True,
        "frozen": True,
        "artifacts_frozen": True,
        "selection_frozen_before_test": True,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "freeze_manifest_sha256": sha256_file(output / "freeze_manifest.json"),
        "independent_terminal_verification_required": True,
        "worker_wrote_pass": False,
        "candidate_pool_count": len(candidates),
        "effective_rule_count": metrics["effective_rule_count"],
        "resume_identity": identity,
        "sealed_at": utc_now(),
    }
    atomic_json(output / "run_manifest.json", run_manifest)
    atomic_bytes(output / "SEALED", b"SEALED\n")
    _write_checkpoint(output, phase="SEALED", resume_identity=identity)
    return run_manifest


def _rename_noreplace(source: Path, destination: Path) -> None:
    if not sys.platform.startswith("linux"):
        if destination.exists():
            raise FileExistsError(f"terminal root exists: {destination}")
        os.rename(source, destination)
        return
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise TasteComRecGCPostprocessError("T14 publication requires renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if int(renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)) != 0:
        observed = ctypes.get_errno()
        if observed in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"terminal root exists: {destination}")
        raise OSError(observed, os.strerror(observed), str(destination))


def _copy_atomic(source: Path, destination: Path) -> None:
    atomic_bytes(destination, source.read_bytes())


def verify_and_publish_t14(
    *,
    authority: PostprocessAuthority,
    science_root: str | Path,
    final_root: str | Path,
) -> dict[str, Any]:
    """Independently replay T14 outputs and atomically publish PASS last."""

    science = Path(science_root).expanduser().resolve(strict=True)
    final = Path(final_root).expanduser().absolute()
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"T14 final root exists: {final}")
    if (science / "SEALED").read_bytes() != b"SEALED\n" or (science / "PASS").exists():
        raise TasteComRecGCPostprocessError("T14 verifier requires SEALED-only science")
    run_manifest = read_json(science / "run_manifest.json")
    freeze = read_json(science / "freeze_manifest.json")
    checkpoint = read_json(science / "postprocess_checkpoint.json")
    if (
        run_manifest.get("schema_version") != RUN_MANIFEST_SCHEMA
        or run_manifest.get("status") != "SEALED"
        or run_manifest.get("resume_identity") != authority.resume_identity()
        or checkpoint.get("phase") != "SEALED"
        or checkpoint.get("resume_identity") != authority.resume_identity()
        or freeze.get("status") != "SEALED"
        or freeze.get("frozen") is not True
    ):
        raise TasteComRecGCPostprocessError("T14 sealed identity changed")
    inventory = freeze.get("files")
    if type(inventory) is not dict or freeze.get("inventory_sha256") != stable_sha256(inventory):
        raise TasteComRecGCPostprocessError("T14 freeze inventory is malformed")
    for name, identity in inventory.items():
        path = science / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(identity.get("bytes", -1))
            or sha256_file(path) != identity.get("sha256")
        ):
            raise TasteComRecGCPostprocessError(f"T14 frozen artifact changed: {name}")
    if _artifact_inventory(science) != inventory:
        raise TasteComRecGCPostprocessError("T14 frozen artifact closure changed")
    expected_candidates = load_generation_candidates(authority)
    candidates = read_jsonl(science / "raw" / "candidate_pool.jsonl")
    selected = read_jsonl(science / "raw" / "selected_candidates.jsonl")
    calibration_rows = read_jsonl(science / "raw" / "calibration_pair_details.jsonl")
    test_rows = read_jsonl(science / "raw" / "test_pair_details.jsonl")
    selection = read_json(science / "raw" / "selection_manifest.json")
    test_manifest = read_json(science / "raw" / "test_evaluation_manifest.json")
    calibration_inventory = verify_pair_chunk_inventory(
        output=science, split="calibration", pair_rows=calibration_rows
    )
    test_inventory = verify_pair_chunk_inventory(
        output=science, split="test", pair_rows=test_rows
    )
    replayed_selected, replayed_selection = select_candidates_on_calibration(
        candidates,
        calibration_rows,
        theta_star=authority.threshold.theta_star,
    )
    expected_selection = dict(replayed_selection)
    expected_selection.update(
        {
            "schema_version": SELECTION_SCHEMA,
            "dataset": DATASET,
            "method": METHOD,
            "stage": STAGE,
            "status": "FROZEN",
            "selection_frozen": True,
            "frozen_at": selection.get("frozen_at"),
            **authority.threshold.to_dict(),
            "calibration_manifest": selection.get("calibration_manifest"),
            "selected_candidates_sha256": sha256_bytes(jsonl_bytes(replayed_selected)),
            "oracle_checkpoint_hash": authority.checkpoint_id,
            "molclr_checkpoint_hash": authority.molclr_checkpoint_sha256,
            "rf_oracle_used": False,
            "test_loaded": False,
            "test_used_for_selection": False,
        }
    )
    if (
        candidates != expected_candidates
        or selected != replayed_selected
        or selection != expected_selection
        or selection.get("calibration_manifest", {}).get("chunk_inventory_sha256")
        != sha256_file(science / "raw" / "calibration_pair_chunk_inventory.json")
        or test_manifest.get("chunk_inventory_sha256")
        != sha256_file(science / "raw" / "test_pair_chunk_inventory.json")
        or test_manifest.get("selection_manifest_sha256")
        != sha256_file(science / "raw" / "selection_manifest.json")
        or test_manifest.get("selection_frozen_before_test") is not True
        or str(selection.get("frozen_at")) > str(test_manifest.get("started_at"))
        or any(row.get("split") != "calibration" for row in calibration_rows)
        or any(row.get("split") != "test" for row in test_rows)
    ):
        raise TasteComRecGCPostprocessError("T14 calibration/test isolation changed")
    for observed, expected_split, expected_candidates_for_split in (
        (calibration_inventory, "calibration", candidates),
        (test_inventory, "test", selected),
    ):
        split_identity = observed.get("evaluation_identity") or {}
        if (
            split_identity.get("split") != expected_split
            or split_identity.get("candidate_count") != len(expected_candidates_for_split)
            or split_identity.get("oracle_checkpoint_hash") != authority.checkpoint_id
            or split_identity.get("molclr_checkpoint_hash")
            != authority.molclr_checkpoint_sha256
            or split_identity.get("threshold_config_hash")
            != authority.threshold.config_hash
            or split_identity.get("generation_checkpoint_digest")
            != authority.generation_checkpoint_digest
        ):
            raise TasteComRecGCPostprocessError(
                f"T14 {expected_split} evaluation identity changed"
            )
    ordered = [str(row["candidate_id"]) for row in selected]
    recomputed = compute_t14_standardized_metrics(test_rows, ordered, authority.threshold)
    expected_outputs = {
        "figure3_coverage_vs_k.csv": recomputed["figure3"],
        "figure4_coverage_vs_threshold.csv": recomputed["figure4"],
        "prefix_metrics.csv": recomputed["prefix"],
        "parent_best_distances.csv": recomputed["parent_best"],
        "destination_distribution.csv": recomputed["destination"],
        "table2_comrecgc_k10.csv": recomputed["table2"],
    }
    for name, rows in expected_outputs.items():
        if csv_bytes(rows) != (science / name).read_bytes():
            raise TasteComRecGCPostprocessError(
                f"T14 standardized metric replay differs: {name}"
            )
    if read_json(science / "summary.json").get("M_effective") != authority.generation_effective_m:
        raise TasteComRecGCPostprocessError("T14 resource-cap disclosure changed")
    prefix_json = json.loads((science / "prefix_metrics.json").read_text(encoding="utf-8"))
    if prefix_json != recomputed["prefix"]:
        raise TasteComRecGCPostprocessError("T14 prefix JSON replay differs")

    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{final.name}.publish-", dir=final.parent)
    )
    published = False
    try:
        standardized_names = (
            "figure3_coverage_vs_k.csv",
            "figure4_coverage_vs_threshold.csv",
            "prefix_metrics.csv",
            "prefix_metrics.json",
            "parent_best_distances.csv",
            "destination_distribution.csv",
            "table2_comrecgc_k10.csv",
            "summary.json",
            "oracle_manifest.json",
            "evaluation_manifest.json",
        )
        for name in standardized_names:
            _copy_atomic(science / name, temporary / name)
        common = {
            key: run_manifest[key]
            for key in (
                "dataset",
                "method",
                "stage",
                "num_classes",
                "source_label",
                "oracle_backend",
                "classifier_family",
                "rf_oracle_used",
                "oracle_checkpoint",
                "oracle_hash",
                "oracle_checkpoint_hash",
                "temperature_calibration_hash",
                "dataset_hash",
                "test_parent_ids_sha256",
                "test_split_hash",
                "distance_line",
                "molclr_checkpoint_hash",
                "cf_mode",
                "threshold_config_hash",
                "test_used_for_selection",
                "threshold_fitted_on_test",
                "raw_output_root",
                "source_generation_root",
                "generation_manifest_sha256",
                "generation_checkpoint_digest",
                "generation_adopted",
                "M_configured_max",
                "M_fallback_max",
                "M_effective",
                "resource_cap_used",
                "early_stop_used",
                "stop_reason",
            )
        }
        final_run = {
            "schema_version": RUN_MANIFEST_SCHEMA,
            **common,
            "status": "PASS",
            "state": "PASS",
            "run_complete": True,
            "raw_output_complete": True,
            "raw_artifacts_complete": True,
            "source_artifacts_complete": True,
            "frozen": True,
            "artifacts_frozen": True,
            "finalized": True,
            "selection_frozen_before_test": True,
            "independent_terminal_verification_required": False,
            "independent_terminal_verification_passed": True,
            "worker_wrote_pass": False,
            "terminal_verifier": "separate_verify_and_publish_invocation",
            "candidate_pool_count": len(candidates),
            "effective_rule_count": len(selected),
            "completed_at": utc_now(),
        }
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
            "source_artifacts_complete": True,
            "checks": {
                "generation_terminal_reopened": True,
                "effective_checkpoint_reopened": True,
                "candidate_lineage_rejoined": True,
                "minimum_ten_unique_candidates": True,
                "calibration_only_selector_replayed": True,
                "resumable_chunk_hashes_replayed": True,
                "selection_frozen_before_test": True,
                "held_out_test_cartesian_complete": True,
                "standardized_metrics_recomputed": True,
                "same_three_class_gine": True,
                "shared_wnode_threshold": True,
                "rf_oracle_absent": True,
                "resource_cap_disclosed": True,
            },
            "science_freeze_manifest_sha256": sha256_file(
                science / "freeze_manifest.json"
            ),
            "verified_at": utc_now(),
        }
        atomic_json(temporary / "run_manifest.json", final_run)
        atomic_json(temporary / "final_artifact_audit.json", audit)
        frozen_files = {
            name: {
                "bytes": (temporary / name).stat().st_size,
                "sha256": sha256_file(temporary / name),
            }
            for name in standardized_names
        }
        atomic_json(
            temporary / "freeze_manifest.json",
            {
                "schema_version": "tastemolnet_t14_final_freeze_manifest_v1",
                **common,
                "status": "PASS",
                "frozen": True,
                "artifacts_frozen": True,
                "files": frozen_files,
                "inventory_sha256": stable_sha256(frozen_files),
                "sealed_at": utc_now(),
            },
        )
        registry = audit_explicit_candidate(
            temporary, dataset=DATASET, method=METHOD
        )
        if registry.status not in PASS_STATUSES:
            raise TasteComRecGCPostprocessError(
                "T14 registry gate failed: " + ";".join(registry.reason_codes)
            )
        audit["registry_status"] = registry.status.value
        audit["registry_reason_codes"] = []
        atomic_json(temporary / "final_artifact_audit.json", audit)
        # Recheck after the exact audit bytes consumed by the final result are
        # present.  PASS is written only after both registry observations.
        registry = audit_explicit_candidate(
            temporary, dataset=DATASET, method=METHOD
        )
        if registry.status not in PASS_STATUSES:
            raise TasteComRecGCPostprocessError(
                "T14 final registry recheck failed: " + ";".join(registry.reason_codes)
            )
        atomic_bytes(temporary / "PASS", f"{PASS_MARKER}\n".encode("utf-8"))
        _fsync_dir(temporary)
        _rename_noreplace(temporary, final)
        _fsync_dir(final.parent)
        published = True
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)
    return read_json(final / "final_artifact_audit.json")


__all__ = [
    "DATASET",
    "METHOD",
    "PASS_MARKER",
    "STAGE",
    "PostprocessAuthority",
    "TasteComRecGCPostprocessError",
    "build_split_evaluation_identity",
    "compute_t14_standardized_metrics",
    "evaluate_one_parent",
    "evaluate_split_resumable",
    "load_generation_candidates",
    "load_postprocess_authority",
    "materialize_generation_candidates",
    "run_t14_postprocess",
    "select_candidates_on_calibration",
    "verify_and_publish_t14",
    "verify_pair_chunk_inventory",
]
