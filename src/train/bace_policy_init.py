"""Provenance-clean BACE policy initialization and train-only SFT data.

This module deliberately contains no classifier call.  The optional SFT target
builder is chemistry-only and may open only the frozen BACE train split.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from src.chem import parse_smiles
from src.data.hiv_dataset_utils import (
    HIVParentRecord,
    murcko_scaffold_smiles,
    parent_atom_count_bin,
)
from src.data.prompts import build_counterfactual_prompt
from src.data.schemas import MoleculeRecord
from src.data.sft_v3_builder import (
    SFTV3BuilderConfig,
    select_reference_candidate_for_parent,
)


POLICY_PROVENANCE_SCHEMA = "bace_policy_initializer_provenance_v1"
ORACLE_NEUTRAL_SFT_SCHEMA = "bace_oracle_neutral_sft_v1"
FORBIDDEN_RF_TOKENS = (
    "randomforestclassifier",
    "random_forest",
    "morgan-rf",
    "morgan_rf",
    "teacher_backend=rf",
    '"teacher_backend":"rf"',
    '"teacher_backend": "rf"',
    "oracle_backend=rf",
    '"oracle_backend":"rf"',
    '"oracle_backend": "rf"',
    "rf_model.pkl",
    "rf-ranked",
    "rf_ranked",
    "rf-filtered",
    "rf_filtered",
    "unknown_oracle_provenance",
)
_MANIFEST_NAMES = (
    "policy_initialization_manifest.json",
    "policy_provenance.json",
    "fresh_initialization_audit.json",
    "train_manifest.json",
    "trainer_state.json",
)


class InitializerClassification(str, Enum):
    CLEAN_CHEMLLM_BASE = "CLEAN_CHEMLLM_BASE"
    CLEAN_ORACLE_NEUTRAL_SFT = "CLEAN_ORACLE_NEUTRAL_SFT"
    RF_CONTAMINATED = "RF_CONTAMINATED"
    UNKNOWN = "UNKNOWN"
    MISSING = "MISSING"


@dataclass(frozen=True, slots=True)
class InitializerAuditRecord:
    path: str
    classification: str
    eligible: bool
    reason: str
    manifest_path: str | None
    policy_initializer_hash: str | None
    rf_evidence: tuple[str, ...]
    test_loaded: bool | None
    calibration_loaded: bool | None
    data_split_used: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rf_evidence"] = list(self.rf_evidence)
        payload["schema_version"] = POLICY_PROVENANCE_SCHEMA
        return payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def hash_path_inventory(path: str | Path, *, hash_file_bytes: bool = True) -> str:
    """Hash one file/tree once, including names, sizes, and file content.

    ``hash_file_bytes=False`` exists only for cheap preflight displays.  Formal
    initializer manifests always use the default content-bound form.
    """

    root = Path(path).expanduser().resolve(strict=True)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    digest = hashlib.sha256()
    for item in files:
        relative = item.name if root.is_file() else item.relative_to(root).as_posix()
        stat = item.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        if hash_file_bytes:
            with item.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_text(path: str | Path, text: str) -> Path:
    """Write a small terminal/status file with fsync then atomic replace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(text))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return target


def _manifest(path: Path) -> tuple[Path | None, dict[str, Any]]:
    if path.is_dir():
        roots: tuple[Path, ...] = (path,)
        if (path / "adapter_config.json").is_file():
            # A freshly built adapter lives at <run>/adapter while its formal
            # provenance lives exactly one level above.  Do not walk arbitrary
            # raw-base ancestors and accidentally consume an unrelated manifest.
            roots = (path, path.parent)
    else:
        roots = (path.parent,)
    for root in roots:
        for name in _MANIFEST_NAMES:
            candidate = root / name
            if not candidate.is_file():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                return candidate.resolve(), payload
    return None, {}


def _rf_evidence(path: Path, manifest: Mapping[str, Any]) -> tuple[str, ...]:
    haystacks = [path.as_posix().lower(), json.dumps(manifest, sort_keys=True).lower()]
    if path.is_dir():
        haystacks.extend(item.name.lower() for item in path.rglob("*") if item.is_file())
    return tuple(
        sorted(
            {
                token
                for token in FORBIDDEN_RF_TOKENS
                if any(token in value for value in haystacks)
            }
        )
    )


def _adapter_payload_files(path: Path) -> tuple[Path | None, Path | None]:
    root = path if path.is_dir() else path.parent
    config = root / "adapter_config.json"
    weights = next(
        (
            candidate
            for candidate in (root / "adapter_model.safetensors", root / "adapter_model.bin")
            if candidate.is_file()
        ),
        None,
    )
    return (config if config.is_file() else None), weights


def audit_policy_initializer(
    path: str | Path,
    *,
    kind_hint: str | None = None,
    content_hash: bool = True,
) -> InitializerAuditRecord:
    candidate = Path(path).expanduser()
    if not candidate.exists():
        return InitializerAuditRecord(
            path=str(candidate.absolute()),
            classification=InitializerClassification.MISSING.value,
            eligible=False,
            reason="initializer_path_missing",
            manifest_path=None,
            policy_initializer_hash=None,
            rf_evidence=(),
            test_loaded=None,
            calibration_loaded=None,
            data_split_used=None,
        )
    resolved = candidate.resolve()
    manifest_path, manifest = _manifest(resolved)
    evidence = _rf_evidence(resolved, manifest)
    digest = hash_path_inventory(resolved, hash_file_bytes=content_hash)
    if evidence:
        classification = InitializerClassification.RF_CONTAMINATED
        reason = "forbidden_rf_provenance:" + ",".join(evidence)
    else:
        init_type = str(manifest.get("policy_initialization_type") or "").strip().lower()
        clean_sft = bool(
            init_type in {"oracle_neutral_sft", "generic_molecular_sft"}
            and _safe_int(manifest.get("rf_reference_count")) == 0
            and manifest.get("gnn_reward_used") is False
            and manifest.get("calibration_loaded") is False
            and manifest.get("test_loaded") is False
            and manifest.get("data_split_used") == "train_only"
            and str(manifest.get("source_model_hash") or "").strip()
            and str(manifest.get("training_data_hash") or "").strip()
        )
        adapter_config, adapter_weights = _adapter_payload_files(resolved)
        if clean_sft and adapter_config is not None and adapter_weights is not None:
            classification = InitializerClassification.CLEAN_ORACLE_NEUTRAL_SFT
            reason = "explicit_train_only_oracle_neutral_sft_manifest"
        else:
            hint = str(kind_hint or "").strip().lower()
            config_path = resolved / "config.json" if resolved.is_dir() else None
            adapter_present = adapter_config is not None or adapter_weights is not None
            name_is_chemllm = "chemllm" in resolved.name.lower()
            manifest_base = init_type in {
                "chemllm_base",
                "chemllm_base_fresh_lora",
            }
            raw_checkpoint = bool(
                (hint in {"raw_base", "chemllm_base"} or name_is_chemllm or manifest_base)
                and config_path is not None
                and config_path.is_file()
                and not adapter_present
                and manifest.get("rf_reference_count", 0) == 0
                and manifest.get("calibration_loaded", False) is False
                and manifest.get("test_loaded", False) is False
            )
            fresh_raw_lora = bool(
                init_type == "chemllm_base_fresh_lora"
                and adapter_config is not None
                and adapter_weights is not None
                and manifest.get("data_split_used") == "none"
                and manifest.get("adapter_initialized_from_scratch") is True
                and _safe_int(manifest.get("optimizer_step_count")) == 0
                and _safe_int(manifest.get("rf_reference_count")) == 0
                and manifest.get("gnn_reward_used") is False
                and manifest.get("calibration_loaded") is False
                and manifest.get("test_loaded") is False
                and str(manifest.get("source_model_hash") or "").strip()
            )
            if raw_checkpoint or fresh_raw_lora:
                classification = InitializerClassification.CLEAN_CHEMLLM_BASE
                reason = "chemllm_base_without_rf_or_heldout_evidence"
            else:
                classification = InitializerClassification.UNKNOWN
                reason = "insufficient_explicit_clean_provenance"
    eligible = classification in {
        InitializerClassification.CLEAN_CHEMLLM_BASE,
        InitializerClassification.CLEAN_ORACLE_NEUTRAL_SFT,
    }
    return InitializerAuditRecord(
        path=str(resolved),
        classification=classification.value,
        eligible=eligible,
        reason=reason,
        manifest_path=None if manifest_path is None else str(manifest_path),
        policy_initializer_hash=digest,
        rf_evidence=evidence,
        test_loaded=manifest.get("test_loaded") if manifest else None,
        calibration_loaded=manifest.get("calibration_loaded") if manifest else None,
        data_split_used=manifest.get("data_split_used") if manifest else None,
    )


def qualification_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Predeclared clean/chemistry/oracle/flip/diversity lexicographic key."""

    classification = str(row.get("classification") or "")
    clean = classification in {
        InitializerClassification.CLEAN_CHEMLLM_BASE.value,
        InitializerClassification.CLEAN_ORACLE_NEUTRAL_SFT.value,
    }
    chemistry_ok = bool(
        float(row.get("parse_ok_rate", 0.0)) >= 0.05
        and float(row.get("direct_substructure_rate", 0.0)) >= 0.01
    )
    return (
        int(clean),
        int(chemistry_ok),
        float(row.get("oracle_evaluable_rate", 0.0)),
        float(row.get("strict_flip_rate", 0.0)),
        float(row.get("unique_fragment_rate", 0.0)),
        # Prefer the bounded train-only SFT only after all measured gates tie.
        int(classification == InitializerClassification.CLEAN_ORACLE_NEUTRAL_SFT.value),
        str(row.get("path") or ""),
    )


def select_policy_initializer(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No policy initializer candidates were supplied")
    eligible = [dict(row) for row in rows if bool(row.get("eligible"))]
    if not eligible:
        raise ValueError("No provenance-clean BACE policy initializer is eligible")
    return max(eligible, key=qualification_key)


def validate_policy_provenance_manifest(
    initializer: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    root = Path(initializer).expanduser().resolve(strict=True)
    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {
        POLICY_PROVENANCE_SCHEMA,
        ORACLE_NEUTRAL_SFT_SCHEMA,
    }:
        raise ValueError("Unsupported BACE policy provenance schema")
    forbidden = _rf_evidence(root, payload)
    if forbidden:
        raise ValueError(f"BACE initializer is RF contaminated: {forbidden}")
    required_false = ("gnn_reward_used", "calibration_loaded", "test_loaded")
    failures = [name for name in required_false if payload.get(name) is not False]
    if int(payload.get("rf_reference_count", -1)) != 0:
        failures.append("rf_reference_count")
    if failures:
        raise ValueError(f"BACE initializer clean-provenance gate failed: {failures}")
    initialization_type = str(
        payload.get("policy_initialization_type") or ""
    ).strip()
    if initialization_type == "chemllm_base_fresh_lora":
        if (
            payload.get("data_split_used") != "none"
            or payload.get("adapter_initialized_from_scratch") is not True
            or int(payload.get("optimizer_step_count", -1)) != 0
            or payload.get("training_data_hash") is not None
        ):
            raise ValueError("Fresh ChemLLM LoRA provenance contract failed")
    elif initialization_type in {"oracle_neutral_sft", "generic_molecular_sft"}:
        if (
            payload.get("data_split_used") != "train_only"
            or not str(payload.get("training_data_hash") or "").strip()
        ):
            raise ValueError("Clean molecular SFT must be train-only and data-bound")
    else:
        raise ValueError(
            f"Unsupported BACE policy initialization type: {initialization_type!r}"
        )
    if not str(payload.get("source_model_hash") or "").strip():
        raise ValueError("BACE policy initializer lacks a source-model identity")
    adapter_config, adapter_weights = _adapter_payload_files(root)
    if adapter_config is None or adapter_weights is None:
        raise ValueError("BACE PPO initializer must be a complete LoRA adapter")
    recorded = str(payload.get("policy_initializer_hash") or "")
    actual = stable_json_hash(
        {
            "adapter_config": sha256_file(adapter_config),
            "adapter_weights": sha256_file(adapter_weights),
            "source_model_hash": payload.get("source_model_hash"),
            "training_data_hash": payload.get("training_data_hash"),
        }
    )
    if recorded != actual:
        raise ValueError("BACE policy initializer hash differs from its adapter bytes")
    return {**payload, "policy_initializer_hash": actual, "initializer": str(root)}


def source_model_hash_from_passed_audit(
    selection_path: str | Path,
    *,
    expected_model_path: str | Path,
) -> str:
    """Reuse one formal base-model hash without scanning the 7B tree again."""

    selection_file = Path(selection_path).expanduser().resolve(strict=True)
    audit_root = selection_file.parent
    pass_path = audit_root / "PASS"
    audit_manifest_path = audit_root / "audit_manifest.json"
    if pass_path.read_text(encoding="utf-8") != (
        "[BACE_POLICY_PROVENANCE_AUDIT_PASS]\n"
    ):
        raise ValueError("BACE initializer audit PASS marker is missing or malformed")
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    if (
        audit_manifest.get("status") != "PASS"
        or Path(str(audit_manifest.get("selection_json") or ""))
        .expanduser()
        .resolve(strict=True)
        != selection_file
    ):
        raise ValueError("BACE initializer audit manifest does not bind this selection")
    audit_csv = Path(str(audit_manifest.get("output_csv") or "")).expanduser().resolve(
        strict=True
    )
    if not audit_csv.is_file():
        raise ValueError("BACE initializer audit CSV is missing")
    if str(audit_manifest.get("output_csv_sha256") or "") != sha256_file(
        audit_csv
    ):
        raise ValueError("BACE initializer audit CSV differs from its PASS manifest")
    if str(audit_manifest.get("selection_json_sha256") or "") != sha256_file(
        selection_file
    ):
        raise ValueError("BACE initializer selection differs from its PASS manifest")
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    if selection.get("schema_version") != "bace_policy_initializer_selection_v1":
        raise ValueError("Unsupported BACE initializer selection schema")
    selected = selection.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("BACE initializer selection has no selected record")
    expected = Path(expected_model_path).expanduser().resolve(strict=True)
    selected_path = Path(str(selected.get("path") or "")).expanduser().resolve(
        strict=True
    )
    if selected_path != expected:
        raise ValueError("BACE initializer audit selected a different model path")
    if (
        selected.get("classification")
        != InitializerClassification.CLEAN_CHEMLLM_BASE.value
        or selected.get("eligible") is not True
    ):
        raise ValueError("BACE initializer audit did not select a clean ChemLLM base")
    source_hash = str(selected.get("policy_initializer_hash") or "").strip().lower()
    if len(source_hash) != 64 or any(
        character not in "0123456789abcdef" for character in source_hash
    ):
        raise ValueError("BACE initializer audit source hash is malformed")
    return source_hash


def validate_frozen_train_contract(
    checkpoint_dir: Path,
    train_csv: Path,
) -> dict[str, Any]:
    manifest_path = checkpoint_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") not in (None, "bace"):
        raise ValueError("The GNN split manifest is not BACE")
    files = manifest.get("files")
    train = files.get("train") if isinstance(files, Mapping) else None
    if not isinstance(train, Mapping):
        raise ValueError("The frozen GNN split manifest has no train entry")
    expected_path = Path(str(train.get("path") or "")).expanduser().resolve(strict=True)
    if expected_path != train_csv:
        raise ValueError("BACE policy data path differs from frozen GNN train split")
    observed_sha = sha256_file(train_csv)
    if str(train.get("sha256") or "").lower() != observed_sha:
        raise ValueError("BACE policy train SHA differs from frozen GNN split manifest")
    roles = manifest.get("roles", {})
    if roles.get("calibration") != "reserved_for_threshold_and_selector_only":
        raise ValueError("Frozen BACE calibration role is missing or changed")
    return {
        "checkpoint_split_manifest": str(manifest_path.resolve(strict=True)),
        "checkpoint_split_manifest_sha256": sha256_file(manifest_path),
        "train_csv": str(train_csv),
        "train_csv_sha256": observed_sha,
        "calibration_loaded": False,
        "test_loaded": False,
    }


def _resolve_csv_columns(fieldnames: Sequence[str]) -> tuple[str, str, str | None]:
    available = {str(name).strip().lower(): str(name) for name in fieldnames}
    smiles = next(
        (
            available[name]
            for name in ("model_smiles", "canonical_smiles", "smiles")
            if name in available
        ),
        None,
    )
    label = next((available[name] for name in ("label", "target") if name in available), None)
    identifier = next(
        (
            available[name]
            for name in ("molecule_id", "id", "compound_id")
            if name in available
        ),
        None,
    )
    if smiles is None or label is None:
        raise ValueError(f"BACE train CSV requires SMILES and label columns: {fieldnames}")
    return smiles, label, identifier


def _internal_split(
    rows: Sequence[dict[str, Any]], *, seed: int, validation_ratio: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if not 0.0 < validation_ratio < 0.5:
        raise ValueError("policy-init validation ratio must be in (0, 0.5)")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["scaffold_smiles"]), []).append(row)
    target = max(1, min(len(rows) - 1, round(len(rows) * validation_ratio)))
    if len(groups) > 1:
        keys = sorted(groups)
        random.Random(seed).shuffle(keys)
        selected: set[str] = set()
        count = 0
        for key in sorted(keys, key=lambda item: (-len(groups[item]), item)):
            if count >= target and selected:
                break
            if len(rows) - (count + len(groups[key])) < 1:
                continue
            selected.add(key)
            count += len(groups[key])
        validation = [row for key in selected for row in groups[key]]
        training = [row for key, values in groups.items() if key not in selected for row in values]
        if training and validation:
            return training, validation, "scaffold_disjoint"
    ordered = sorted(rows, key=lambda row: str(row["molecule_id"]))
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    validation_ids = {str(row["molecule_id"]) for row in shuffled[:target]}
    return (
        [row for row in ordered if str(row["molecule_id"]) not in validation_ids],
        [row for row in ordered if str(row["molecule_id"]) in validation_ids],
        "parent_id_disjoint_fallback",
    )


def build_oracle_neutral_sft_dataset(
    *,
    train_csv: str | Path,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    source_model_hash: str,
    seed: int = 7,
    validation_ratio: float = 0.1,
    max_parents: int = 0,
) -> dict[str, Any]:
    """Build deterministic BACE SFT targets without an oracle or held-out split."""

    source = Path(train_csv).expanduser().resolve(strict=True)
    checkpoint = Path(checkpoint_dir).expanduser().resolve(strict=True)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"BACE oracle-neutral SFT output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frozen = validate_frozen_train_contract(checkpoint, source)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        smiles_col, label_col, id_col = _resolve_csv_columns(fieldnames)
        raw_rows = list(reader)
    if max_parents > 0:
        raw_rows = raw_rows[: int(max_parents)]
    config = SFTV3BuilderConfig(
        seed=int(seed),
        min_atom_ratio=0.10,
        max_atom_ratio=0.65,
        min_frag_atoms=2,
        max_frag_atoms=30,
        use_oracle_ranking=False,
        oracle_path=None,
        include_label_in_prompt=True,
    )
    examples: list[dict[str, Any]] = []
    dropped: dict[str, int] = {}
    for index, raw in enumerate(raw_rows):
        source_smiles = str(raw.get(smiles_col) or "").strip()
        try:
            label = int(str(raw.get(label_col)).strip())
        except (TypeError, ValueError):
            dropped["invalid_label"] = dropped.get("invalid_label", 0) + 1
            continue
        parsed = parse_smiles(source_smiles, sanitize=True, canonicalize=True)
        if not parsed.sanitized or parsed.mol is None or label not in (0, 1):
            dropped["invalid_parent"] = dropped.get("invalid_parent", 0) + 1
            continue
        parent_smiles = str(parsed.canonical_smiles or source_smiles)
        raw_identifier = raw.get(id_col) if id_col else None
        molecule_id = str(raw_identifier).strip() if raw_identifier is not None else ""
        if not molecule_id:
            molecule_id = f"bace:{index}"
        scaffold = murcko_scaffold_smiles(parsed.mol)
        parent = HIVParentRecord(
            sample_id=molecule_id,
            source_row_index=index,
            source_smiles=source_smiles,
            parent_smiles=parent_smiles,
            label=label,
            raw_label=label,
            parent_atom_count=int(parsed.atom_count or parsed.mol.GetNumAtoms()),
            scaffold_smiles=scaffold,
            size_bin=parent_atom_count_bin(int(parsed.atom_count or parsed.mol.GetNumAtoms())),
        )
        built = select_reference_candidate_for_parent(parent, config=config, oracle_scorer=None)
        candidate = built.selected_candidate
        if candidate is None:
            reason = built.drop_reason or "no_candidate"
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        prompt = build_counterfactual_prompt(
            MoleculeRecord(record_id=molecule_id, smiles=parent_smiles, label=label),
            include_label=True,
        )
        examples.append(
            {
                "id": molecule_id,
                "molecule_id": molecule_id,
                "parent_smiles": parent_smiles,
                "smiles": parent_smiles,
                "label": label,
                "prompt": prompt,
                "completion": candidate.core_fragment,
                "response": candidate.core_fragment,
                "reference_fragment": candidate.core_fragment,
                "scaffold_smiles": scaffold,
                "candidate_strategy": candidate.candidate_strategy,
                "fragment_atom_ratio": candidate.atom_ratio,
                "parseable": True,
                "connected": True,
                "direct_substructure": True,
                "oracle_scorer": None,
            }
        )
    if len(examples) < 2:
        raise ValueError("BACE oracle-neutral SFT produced fewer than two examples")
    training, validation, split_method = _internal_split(
        examples, seed=int(seed), validation_ratio=float(validation_ratio)
    )
    train_ids = {str(row["molecule_id"]) for row in training}
    val_ids = {str(row["molecule_id"]) for row in validation}
    if train_ids & val_ids:
        raise RuntimeError("BACE policy-init train/validation parent IDs overlap")
    if split_method == "scaffold_disjoint":
        train_scaffolds = {str(row["scaffold_smiles"]) for row in training}
        val_scaffolds = {str(row["scaffold_smiles"]) for row in validation}
        if train_scaffolds & val_scaffolds:
            raise RuntimeError("BACE policy-init scaffold split overlaps")
    train_path = output / "policy_init_train.jsonl"
    validation_path = output / "policy_init_validation.jsonl"
    _atomic_jsonl(train_path, training)
    _atomic_jsonl(validation_path, validation)
    training_data_hash = stable_json_hash(
        {"train": sha256_file(train_path), "validation": sha256_file(validation_path)}
    )
    manifest = {
        "schema_version": ORACLE_NEUTRAL_SFT_SCHEMA,
        "policy_initialization_type": "oracle_neutral_sft",
        "dataset": "bace",
        "data_split_used": "train_only",
        "formal_train_split_unchanged": True,
        "policy_internal_split_method": split_method,
        "policy_internal_parent_overlap": 0,
        "policy_internal_scaffold_overlap": 0 if split_method == "scaffold_disjoint" else None,
        "seed": int(seed),
        "train_examples": len(training),
        "validation_examples": len(validation),
        "input_rows": len(raw_rows),
        "dropped": dict(sorted(dropped.items())),
        "target_builder": "deterministic_molecular_fragment_builder_without_oracle",
        "oracle_scorer": None,
        "rf_reference_count": 0,
        "gnn_reward_used": False,
        "formal_validation_loaded": False,
        "policy_internal_validation_loaded": True,
        "policy_internal_validation_source": "train_only",
        "calibration_loaded": False,
        "test_loaded": False,
        "source_model_hash": str(source_model_hash),
        "training_data_hash": training_data_hash,
        "train_jsonl": str(train_path),
        "train_jsonl_sha256": sha256_file(train_path),
        "validation_jsonl": str(validation_path),
        "validation_jsonl_sha256": sha256_file(validation_path),
        **frozen,
    }
    _atomic_json(output / "policy_initialization_manifest.json", manifest)
    return manifest


def finalize_adapter_manifest(
    *,
    adapter_dir: str | Path,
    manifest: Mapping[str, Any],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(adapter_dir).expanduser().resolve(strict=True)
    config, weights = _adapter_payload_files(root)
    if config is None or weights is None:
        raise ValueError(f"Incomplete BACE clean adapter: {root}")
    payload = {
        **dict(manifest),
        "schema_version": POLICY_PROVENANCE_SCHEMA,
        "adapter_dir": str(root),
        "adapter_config_sha256": sha256_file(config),
        "adapter_weights_sha256": sha256_file(weights),
        "policy_initializer_hash": stable_json_hash(
            {
                "adapter_config": sha256_file(config),
                "adapter_weights": sha256_file(weights),
                "source_model_hash": manifest.get("source_model_hash"),
                "training_data_hash": manifest.get("training_data_hash"),
            }
        ),
        "rf_reference_count": 0,
        "gnn_reward_used": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    target = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else root / "policy_provenance.json"
    )
    _atomic_json(target, payload)
    if target.parent != root:
        _atomic_json(root / "policy_provenance.json", payload)
    return payload


__all__ = [
    "InitializerAuditRecord",
    "InitializerClassification",
    "ORACLE_NEUTRAL_SFT_SCHEMA",
    "POLICY_PROVENANCE_SCHEMA",
    "audit_policy_initializer",
    "atomic_text",
    "build_oracle_neutral_sft_dataset",
    "finalize_adapter_manifest",
    "hash_path_inventory",
    "qualification_key",
    "select_policy_initializer",
    "sha256_file",
    "stable_json_hash",
    "source_model_hash_from_passed_audit",
    "validate_frozen_train_contract",
    "validate_policy_provenance_manifest",
]
