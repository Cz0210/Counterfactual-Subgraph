"""Held T2/T3 and split lineage for TasteMolNet NeuroSED.

The auxiliary model is allowed to read train and validation payloads, but the
paths, bytes, counts, labels, and graph fingerprints must come from the frozen
GINE lineage.  This module retains that lineage through worker/verifier use;
it never opens calibration or test payloads.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Mapping

from src.eval.tastemolnet_t3_calibration_v2 import (
    HeldT2Receipt,
    SCHEMA_VERSION as T3_SCHEMA_VERSION,
    STAGE as T3_STAGE,
    _validate_t2_authorities,
)
from src.utils.managed_execution_v2 import (
    GATE_SCHEMA,
    PASS_MARKER,
    VERIFICATION_SCHEMA,
)
from src.utils.retained_readonly_file import hold_readonly_file
from src.utils.tastemolnet_t2_adoption_v2 import HeldBundle
from src.utils.managed_final_consumer_v2 import hold_verified_managed_final


AUTHORITY_SCHEMA = "tastemolnet_gcf_neurosed_authority_v2"
SPLIT_LINEAGE_SCHEMA = "tastemolnet_gcf_neurosed_split_lineage_v2"
EXPECTED_LABEL_MAP = {"0": "Bitter", "1": "Sweet", "2": "Tasteless"}
_SHA256 = frozenset("0123456789abcdef")


class TasteNeuroSEDAuthorityError(RuntimeError):
    """The held T2/T3/data authority is incomplete or changed."""


def _json(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteNeuroSEDAuthorityError(f"{label} is not JSON") from exc
    if type(value) is not dict:
        raise TasteNeuroSEDAuthorityError(f"{label} is not one JSON object")
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA256 for character in value)
    )


def _absolute(path: str | Path, *, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute() or Path(os.path.abspath(value)) != value:
        raise TasteNeuroSEDAuthorityError(f"{label} must be normalized absolute")
    return value


def _sealed_file(held: Any, relative: str) -> Any:
    try:
        return held.file(relative)
    except Exception as exc:
        raise TasteNeuroSEDAuthorityError(
            f"T3 SEALED file is absent: {relative}"
        ) from exc


def _held_inventory_bytes(item: Any) -> bytes:
    return item.read_bytes()


def _split_manifest_contract(data: bytes) -> dict[str, Any]:
    split = _json(data, label="GINE split manifest")
    expected_keys = {
        "schema_version",
        "dataset",
        "roles",
        "files",
        "train_manifest",
        "validation_manifest",
        "calibration_loaded_for_training",
        "test_loaded_for_training",
        "test_evaluated_during_training",
        "test_used_for_checkpoint_selection",
    }
    if (
        set(split) != expected_keys
        or split.get("schema_version") != "molecular_gnn_split_manifest_v1"
        or split.get("dataset") != "tastemolnet"
        or split.get("calibration_loaded_for_training") is not False
        or split.get("test_loaded_for_training") is not False
        or split.get("test_evaluated_during_training") is not False
        or split.get("test_used_for_checkpoint_selection") is not False
        or split.get("roles")
        != {
            "train": "model_fitting",
            "validation": "checkpoint_selection_and_temperature_calibration",
            "calibration": "reserved_for_threshold_and_selector_only",
            "test": "frozen_model_final_quality_evaluation",
        }
    ):
        raise TasteNeuroSEDAuthorityError("GINE split manifest contract changed")
    files = split.get("files")
    if type(files) is not dict or set(files) != {
        "train",
        "validation",
        "calibration",
        "test",
    }:
        raise TasteNeuroSEDAuthorityError("GINE split file authority changed")
    for role in ("train", "validation", "calibration", "test"):
        row = files[role]
        if (
            type(row) is not dict
            or set(row) != {"path", "sha256"}
            or type(row.get("path")) is not str
            or not _is_sha256(row.get("sha256"))
        ):
            raise TasteNeuroSEDAuthorityError(f"GINE {role} split pin changed")
    return split


def _audit_csv(
    data: bytes,
    *,
    path: Path,
    split: str,
    manifest: Mapping[str, Any],
    feature_schema_data: bytes,
) -> dict[str, Any]:
    """Recompute the exact GINE dataset manifest from held CSV bytes."""

    from src.data.molecular_graph_featurizer import (
        MolecularFeatureSchema,
        MolecularGraphFeaturizer,
    )
    from src.data.tastemolnet_ppo import TASTEMOLNET_PREPARED_FIELDS

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise TasteNeuroSEDAuthorityError(f"held {split} CSV is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    if tuple(reader.fieldnames or ()) != TASTEMOLNET_PREPARED_FIELDS:
        raise TasteNeuroSEDAuthorityError(f"held {split} CSV columns changed")
    schema_payload = _json(feature_schema_data, label="T2 feature schema")
    schema = MolecularFeatureSchema.from_dict(schema_payload)
    featurizer = MolecularGraphFeaturizer(schema=schema)
    rows: list[dict[str, Any]] = []
    label_counts = {label: 0 for label in EXPECTED_LABEL_MAP}
    identifiers: set[str] = set()
    canonical_smiles: set[str] = set()
    normalized_split = "val" if split == "validation" else split
    for index, row in enumerate(reader):
        molecule_id = str(row.get("molecule_id") or "").strip()
        smiles = str(row.get("model_smiles") or "").strip()
        label = str(row.get("label") or "").strip()
        if (
            None in row
            or set(row) != set(TASTEMOLNET_PREPARED_FIELDS)
            or not molecule_id
            or molecule_id in identifiers
            or not smiles
            or label not in EXPECTED_LABEL_MAP
            or str(row.get("label_name") or "").strip() != EXPECTED_LABEL_MAP[label]
            or str(row.get("split") or "").strip() != split
            or str(row.get("exclusion_reason") or "").strip()
        ):
            raise TasteNeuroSEDAuthorityError(f"held {split} row {index} changed")
        graph = featurizer.featurize(smiles)
        if graph.canonical_smiles in canonical_smiles:
            raise TasteNeuroSEDAuthorityError(
                f"held {split} canonical SMILES are duplicated"
            )
        identifiers.add(molecule_id)
        canonical_smiles.add(graph.canonical_smiles)
        label_counts[label] += 1
        rows.append(
            {
                "molecule_id": molecule_id,
                "canonical_smiles": graph.canonical_smiles,
                "label": int(label),
                "split": normalized_split,
                "graph_sha256": graph.graph_sha256,
            }
        )
    expected = dict(manifest)
    observed = {
        "schema_version": "molecular_graph_dataset_v1",
        "num_records": len(rows),
        "num_classes": 3,
        "label_counts": label_counts,
        "split_counts": {normalized_split: len(rows)},
        "source_path": str(path),
        "source_sha256": _sha256_bytes(data),
        "dataset_fingerprint": _stable_sha256(rows),
        "feature_schema_sha256": schema.to_dict()["schema_sha256"],
    }
    if observed != expected:
        raise TasteNeuroSEDAuthorityError(
            f"held {split} CSV differs from authoritative GINE manifest"
        )
    return observed


@dataclass(slots=True)
class HeldTasteNeuroSEDDataAuthority:
    stack: ExitStack
    t2_receipt: HeldT2Receipt
    t2_bundle: HeldBundle
    t3_sealed: Any
    t3_gate: Any
    t3_verification: Any
    t3_pass: Any
    train_file: Any
    validation_file: Any
    evidence: Mapping[str, Any]
    train_bytes: bytes
    validation_bytes: bytes

    def revalidate(self) -> Mapping[str, Any]:
        current_t2 = _validate_t2_authorities(self.t2_receipt, self.t2_bundle)
        if current_t2 != self.evidence["t2_binding"]:
            raise TasteNeuroSEDAuthorityError("T2 authority changed while held")
        self.t3_sealed.revalidate()
        self.t3_gate.revalidate()
        self.t3_verification.revalidate()
        self.t3_pass.revalidate()
        self.train_file.revalidate()
        self.validation_file.revalidate()
        return self.evidence

    @property
    def input_hashes(self) -> dict[str, str]:
        return {
            "t2_receipt_gate": self.evidence["t2_receipt_gate_sha256"],
            "t2_source_evidence": self.evidence["t2_source_evidence_sha256"],
            "t2_source_sha256s": self.evidence["t2_source_sha256s_sha256"],
            "t2_source_split_manifest": self.evidence[
                "t2_source_split_manifest_sha256"
            ],
            "t3_gate": self.evidence["t3_gate_sha256"],
            "t3_verification": self.evidence["t3_verification_sha256"],
            "t3_checkpoint_split_manifest": self.evidence[
                "t3_checkpoint_split_manifest_sha256"
            ],
            "train_csv": self.evidence["train"]["sha256"],
            "validation_csv": self.evidence["validation"]["sha256"],
        }

    def close(self) -> None:
        self.stack.close()

    def __enter__(self) -> "HeldTasteNeuroSEDDataAuthority":
        self.revalidate()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def hold_tastemolnet_neurosed_data_authority(
    *,
    t2_receipt_root: str | Path,
    t2_source_bundle_root: str | Path,
    t3_final_root: str | Path,
    train_csv: str | Path,
    validation_csv: str | Path,
) -> HeldTasteNeuroSEDDataAuthority:
    """Retain and cross-bind the real T2/T3/GINE split authorities."""

    stack = ExitStack()
    try:
        receipt = HeldT2Receipt(_absolute(t2_receipt_root, label="T2 receipt root"))
        stack.callback(receipt.close)
        bundle = HeldBundle(_absolute(t2_source_bundle_root, label="T2 source bundle"))
        stack.callback(bundle.close)
        t2_binding = _validate_t2_authorities(receipt, bundle)
        t3_root = _absolute(t3_final_root, label="T3 final root")
        t3_sealed = stack.enter_context(
            hold_verified_managed_final(
                t3_root,
                required_relative_paths=(
                    "artifacts/checkpoint/split_manifest.json",
                ),
            )
        )
        t3_gate = t3_sealed.file("gate.json")
        t3_verification = t3_sealed.file("verification.json")
        t3_pass = t3_sealed.file(PASS_MARKER)
        if t3_pass.read_bytes() != b"[MANAGED_EXECUTION_V2_PASS]\n":
            raise TasteNeuroSEDAuthorityError("T3 generic PASS changed")
        gate = _json(t3_gate.read_bytes(), label="T3 generic gate")
        verification = _json(
            t3_verification.read_bytes(), label="T3 generic verification"
        )
        domain = verification.get("verification")
        if (
            gate.get("schema_version") != GATE_SCHEMA
            or gate.get("status") != "PASS"
            or gate.get("independent_verifier") is not True
            or gate.get("downstream_released") is not True
            or gate.get("verification_sha256") != t3_verification.sha256
            or verification.get("schema_version") != VERIFICATION_SCHEMA
            or verification.get("status") != "PASS"
            or verification.get("independent_verifier") is not True
            or verification.get("attempt_id") != gate.get("attempt_id")
            or verification.get("generation_token") != gate.get("generation_token")
            or verification.get("sealed_sha256") != t3_sealed.sealed.seal_sha256
            or verification.get("source_inventory_sha256")
            != t3_sealed.sealed.inventory_sha256
            or verification.get("published_inventory_sha256")
            != gate.get("published_inventory_sha256")
            or type(domain) is not dict
            or domain.get("schema_version") != T3_SCHEMA_VERSION
            or domain.get("status") != "PASS"
            or domain.get("stage") != T3_STAGE
            or domain.get("t2_receipt_gate_sha256")
            != t2_binding["receipt_gate_sha256"]
            or domain.get("source_evidence_sha256")
            != t2_binding["source_evidence_sha256"]
            or domain.get("source_bundle_root") != str(bundle.root)
            or domain.get("calibration_payload_loaded") is not False
            or domain.get("test_payload_loaded") is not False
        ):
            raise TasteNeuroSEDAuthorityError("T3 managed final authority changed")
        t2_split_bytes = bundle.files["split_manifest.json"].bytes()
        t3_split_item = _sealed_file(
            t3_sealed, "artifacts/checkpoint/split_manifest.json"
        )
        t3_split_bytes = _held_inventory_bytes(t3_split_item)
        if t3_split_bytes != t2_split_bytes:
            raise TasteNeuroSEDAuthorityError(
                "T3 copied split manifest differs byte-for-byte from T2 source"
            )
        split = _split_manifest_contract(t2_split_bytes)
        train_path = _absolute(train_csv, label="Taste train CSV")
        validation_path = _absolute(validation_csv, label="Taste validation CSV")
        if (
            str(train_path) != split["files"]["train"]["path"]
            or str(validation_path) != split["files"]["validation"]["path"]
        ):
            raise TasteNeuroSEDAuthorityError(
                "NeuroSED payload paths differ from authoritative GINE split"
            )
        train_file = stack.enter_context(
            hold_readonly_file(
                train_path,
                expected_sha256=split["files"]["train"]["sha256"],
            )
        )
        validation_file = stack.enter_context(
            hold_readonly_file(
                validation_path,
                expected_sha256=split["files"]["validation"]["sha256"],
            )
        )
        train_bytes = train_file.read_bytes()
        validation_bytes = validation_file.read_bytes()
        feature_schema_bytes = bundle.files["feature_schema.json"].bytes()
        train_evidence = _audit_csv(
            train_bytes,
            path=train_path,
            split="train",
            manifest=split["train_manifest"],
            feature_schema_data=feature_schema_bytes,
        )
        validation_evidence = _audit_csv(
            validation_bytes,
            path=validation_path,
            split="validation",
            manifest=split["validation_manifest"],
            feature_schema_data=feature_schema_bytes,
        )
        evidence = {
            "schema_version": AUTHORITY_SCHEMA,
            "split_lineage_schema": SPLIT_LINEAGE_SCHEMA,
            "t2_binding": t2_binding,
            "t2_receipt_gate_sha256": t2_binding["receipt_gate_sha256"],
            "t2_source_evidence_sha256": t2_binding["source_evidence_sha256"],
            "t2_source_sha256s_sha256": bundle.files["sha256sums.txt"].sha256,
            "t2_source_split_manifest_sha256": bundle.files[
                "split_manifest.json"
            ].sha256,
            "t3_final_root": str(t3_root),
            "t3_gate_sha256": t3_gate.sha256,
            "t3_verification_sha256": t3_verification.sha256,
            "t3_checkpoint_split_manifest_sha256": t3_split_item.sha256,
            "t3_split_manifest_byte_identical_to_t2": True,
            "train": train_evidence,
            "validation": validation_evidence,
            "calibration_loaded": False,
            "test_loaded": False,
        }
        result = HeldTasteNeuroSEDDataAuthority(
            stack=stack,
            t2_receipt=receipt,
            t2_bundle=bundle,
            t3_sealed=t3_sealed,
            t3_gate=t3_gate,
            t3_verification=t3_verification,
            t3_pass=t3_pass,
            train_file=train_file,
            validation_file=validation_file,
            evidence=evidence,
            train_bytes=train_bytes,
            validation_bytes=validation_bytes,
        )
        result.revalidate()
        return result
    except BaseException:
        stack.close()
        raise


__all__ = [
    "AUTHORITY_SCHEMA",
    "HeldTasteNeuroSEDDataAuthority",
    "SPLIT_LINEAGE_SCHEMA",
    "TasteNeuroSEDAuthorityError",
    "hold_tastemolnet_neurosed_data_authority",
]
