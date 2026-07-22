"""Deterministic preprocessing utilities for the unified Mutagenicity benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:  # pragma: no cover - exercised by CLI dependency checks.
    Chem = None
    Descriptors = None
    MurckoScaffold = None


DATASET_NAME = "Mutagenicity"
DATASET_VERSION = "v1"
SEMANTIC_LABELS = {0: "non_mutagenic", 1: "mutagenic"}
SPLIT_NAMES = ("train", "val", "calibration", "test")
DEFAULT_SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.10,
    "calibration": 0.10,
    "test": 0.10,
}
REQUIRED_DOWNLOAD_FILES = (
    "smiles/smiles_mutagenicity_raw.csv",
    "smiles/smiles_mutagenicity_curated.csv",
    "smiles/smiles_mutagenicity_removed.csv",
    "tudataset/Mutagenicity/Mutagenicity_A.txt",
    "tudataset/Mutagenicity/Mutagenicity_edge_labels.txt",
    "tudataset/Mutagenicity/Mutagenicity_graph_indicator.txt",
    "tudataset/Mutagenicity/Mutagenicity_graph_labels.txt",
    "tudataset/Mutagenicity/Mutagenicity_node_labels.txt",
)
STANDARD_SPLIT_FIELDS = (
    "molecule_id",
    "smiles",
    "label",
    "semantic_label",
    "scaffold_smiles",
    "split",
)


@dataclass(frozen=True, slots=True)
class CsvSource:
    path: Path
    rows: list[dict[str, str]]
    smiles_col: str
    label_col: str


def require_rdkit() -> None:
    if Chem is None or Descriptors is None or MurckoScaffold is None:
        raise RuntimeError(
            "Mutagenicity preprocessing requires RDKit. Activate smiles_pip118 "
            "or install RDKit before running this command."
        )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    return value


def parse_sha256_manifest(path: str | Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    manifest = Path(path)
    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Malformed SHA256 manifest line {line_number}: {raw_line!r}")
        digest, filename = parts
        filename = filename.lstrip("*").strip()
        if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
            raise ValueError(f"Invalid SHA256 digest on line {line_number}: {digest!r}")
        if not filename:
            raise ValueError(f"Missing filename on SHA256 manifest line {line_number}")
        entries.append((digest.lower(), filename))
    if not entries:
        raise ValueError(f"SHA256 manifest is empty: {manifest}")
    return entries


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_download(root: str | Path, manifest: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    manifest_path = Path(manifest)
    entries = parse_sha256_manifest(manifest_path)
    manifest_names = {filename for _digest, filename in entries}
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for expected_digest, relative_name in entries:
        target = root_path / relative_name
        exists = target.is_file()
        size = target.stat().st_size if exists else 0
        actual_digest = sha256_file(target) if exists and size > 0 else None
        ok = bool(exists and size > 0 and actual_digest == expected_digest)
        results.append(
            {
                "path": relative_name,
                "exists": exists,
                "size_bytes": size,
                "expected_sha256": expected_digest,
                "actual_sha256": actual_digest,
                "ok": ok,
            }
        )
        if not ok:
            errors.append(f"checksum_or_file_failure:{relative_name}")
    for required in REQUIRED_DOWNLOAD_FILES:
        target = root_path / required
        if required not in manifest_names:
            errors.append(f"required_file_missing_from_manifest:{required}")
        if not target.is_file() or target.stat().st_size <= 0:
            errors.append(f"required_file_missing_or_empty:{required}")
    return {
        "root": str(root_path),
        "manifest": str(manifest_path),
        "num_manifest_entries": len(entries),
        "files": results,
        "errors": sorted(set(errors)),
        "passed": not errors,
    }


def infer_csv_columns(fieldnames: Sequence[str] | None) -> tuple[str, str]:
    if not fieldnames:
        raise ValueError("CSV has no header")
    normalized = {name.strip().lower(): name for name in fieldnames if name is not None}
    smiles_candidates = ("smiles", "canonical_smiles", "molecule_smiles")
    label_candidates = ("mutagenicity", "label", "target", "y", "class")
    smiles_matches = [normalized[name] for name in smiles_candidates if name in normalized]
    label_matches = [normalized[name] for name in label_candidates if name in normalized]
    if len(smiles_matches) != 1 or len(label_matches) != 1:
        raise ValueError(
            "Could not uniquely infer SMILES/label columns: "
            f"headers={list(fieldnames)}, smiles_matches={smiles_matches}, "
            f"label_matches={label_matches}"
        )
    return smiles_matches[0], label_matches[0]


def load_csv_source(path: str | Path) -> CsvSource:
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        smiles_col, label_col = infer_csv_columns(reader.fieldnames)
        rows = [dict(row) for row in reader]
    return CsvSource(source_path, rows, smiles_col, label_col)


def normalize_binary_label(value: Any) -> int:
    text = str(value).strip().lower()
    aliases = {
        "0": 0,
        "0.0": 0,
        "false": 0,
        "nonmutagen": 0,
        "non_mutagenic": 0,
        "non-mutagenic": 0,
        "1": 1,
        "1.0": 1,
        "true": 1,
        "mutagen": 1,
        "mutagenic": 1,
    }
    if text not in aliases:
        raise ValueError(f"Unsupported binary Mutagenicity label: {value!r}")
    return aliases[text]


def read_integer_lines(path: str | Path) -> list[int]:
    values: list[int] = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            values.append(int(line))
        except ValueError as exc:
            raise ValueError(f"Invalid integer at {path}:{line_number}: {line!r}") from exc
    return values


def evaluate_tu_label_mappings(
    csv_labels: Sequence[int], tu_labels: Sequence[int]
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    if len(csv_labels) != len(tu_labels):
        raise ValueError(
            f"CSV/TU label lengths differ: csv={len(csv_labels)}, tu={len(tu_labels)}"
        )
    mappings: dict[str, dict[int, int]] = {
        "identity_0_1": {0: 0, 1: 1},
        "inverted_0_1": {0: 1, 1: 0},
        "minus1_plus1_to_0_1": {-1: 0, 1: 1},
        "minus1_plus1_inverted": {-1: 1, 1: 0},
    }
    results: list[dict[str, Any]] = []
    transformed_by_name: dict[str, list[int | None]] = {}
    for name, mapping in mappings.items():
        transformed = [mapping.get(value) for value in tu_labels]
        valid = sum(value is not None for value in transformed)
        matches = sum(
            transformed_value is not None and transformed_value == csv_value
            for csv_value, transformed_value in zip(csv_labels, transformed)
        )
        result = {
            "mapping": name,
            "mapping_json": json.dumps(mapping, sort_keys=True),
            "num_rows": len(csv_labels),
            "num_mappable": valid,
            "num_matches": matches,
            "num_mismatches": len(csv_labels) - matches,
            "match_rate": matches / len(csv_labels) if csv_labels else 0.0,
        }
        results.append(result)
        transformed_by_name[name] = transformed
    best_matches = max(result["num_matches"] for result in results)
    best = [result for result in results if result["num_matches"] == best_matches]
    if len(best) != 1 or best_matches != len(csv_labels):
        raise ValueError(
            "Could not determine a unique complete TU label mapping: "
            f"best={best}, total={len(csv_labels)}"
        )
    selected_name = str(best[0]["mapping"])
    selected_values = transformed_by_name[selected_name]
    mismatches = [
        {
            "graph_id": index,
            "csv_label": csv_label,
            "tu_label": tu_label,
            "mapped_tu_label": mapped_label,
        }
        for index, (csv_label, tu_label, mapped_label) in enumerate(
            zip(csv_labels, tu_labels, selected_values), 1
        )
        if mapped_label != csv_label
    ]
    return results, selected_name, mismatches


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def audit_smiles(smiles: str) -> dict[str, Any]:
    require_rdkit()
    value = str(smiles or "").strip()
    result: dict[str, Any] = {
        "smiles": value,
        "parse_ok": False,
        "sanitize_ok": False,
        "parse_error": "",
        "sanitize_error": "",
        "num_components": 0,
        "contains_dot": "." in value,
        "formal_charge": None,
        "num_atoms": 0,
        "num_heavy_atoms": 0,
        "num_bonds": 0,
        "molecular_weight": None,
        "atom_types": "",
        "bond_types": "",
        "explicit_hydrogen": False,
        "radical": False,
        "dummy_atom": False,
        "canonical_isomeric_smiles": "",
        "canonical_nonisomeric_smiles": "",
        "scaffold_smiles": "",
    }
    if not value:
        result["parse_error"] = "empty_smiles"
        return result
    try:
        molecule = Chem.MolFromSmiles(value, sanitize=False)
    except Exception as exc:
        result["parse_error"] = _error_text(exc)
        return result
    if molecule is None:
        result["parse_error"] = "rdkit_parse_returned_none"
        return result
    result["parse_ok"] = True
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        result["sanitize_error"] = _error_text(exc)
        return result
    result["sanitize_ok"] = True
    atoms = list(molecule.GetAtoms())
    bonds = list(molecule.GetBonds())
    result.update(
        {
            "num_components": len(Chem.GetMolFrags(molecule)),
            "formal_charge": int(Chem.GetFormalCharge(molecule)),
            "num_atoms": int(molecule.GetNumAtoms()),
            "num_heavy_atoms": int(molecule.GetNumHeavyAtoms()),
            "num_bonds": int(molecule.GetNumBonds()),
            "molecular_weight": float(Descriptors.MolWt(molecule)),
            "atom_types": ";".join(sorted({atom.GetSymbol() for atom in atoms})),
            "bond_types": ";".join(sorted({str(bond.GetBondType()) for bond in bonds})),
            "explicit_hydrogen": any(
                atom.GetAtomicNum() == 1 or atom.GetNumExplicitHs() > 0 for atom in atoms
            ),
            "radical": any(atom.GetNumRadicalElectrons() > 0 for atom in atoms),
            "dummy_atom": any(atom.GetAtomicNum() == 0 for atom in atoms),
            "canonical_isomeric_smiles": Chem.MolToSmiles(
                molecule, canonical=True, isomericSmiles=True
            ),
            "canonical_nonisomeric_smiles": Chem.MolToSmiles(
                molecule, canonical=True, isomericSmiles=False
            ),
            "scaffold_smiles": MurckoScaffold.MurckoScaffoldSmiles(
                mol=molecule, includeChirality=True
            ),
        }
    )
    return result


RDKIT_AUDIT_FIELDS = (
    "source_row_id",
    "smiles",
    "label_original",
    "label",
    "parse_ok",
    "sanitize_ok",
    "parse_error",
    "sanitize_error",
    "num_components",
    "contains_dot",
    "formal_charge",
    "num_atoms",
    "num_heavy_atoms",
    "num_bonds",
    "molecular_weight",
    "atom_types",
    "bond_types",
    "explicit_hydrogen",
    "radical",
    "dummy_atom",
    "canonical_isomeric_smiles",
    "canonical_nonisomeric_smiles",
    "scaffold_smiles",
)


def audit_csv_source(source: CsvSource) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for row_id, row in enumerate(source.rows, 1):
        label = normalize_binary_label(row[source.label_col])
        details.append(
            {
                "source_row_id": row_id,
                "label_original": row[source.label_col],
                "label": label,
                **audit_smiles(row[source.smiles_col]),
            }
        )
    return details


def summarize_rdkit_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    count = lambda field: sum(bool(row.get(field)) for row in rows)
    return {
        "rows": total,
        "parse_rate": count("parse_ok") / total if total else 0.0,
        "sanitize_rate": count("sanitize_ok") / total if total else 0.0,
        "multicomponent_rate": (
            sum(int(row.get("num_components") or 0) > 1 for row in rows) / total
            if total
            else 0.0
        ),
        "dummy_atom_count": count("dummy_atom"),
        "radical_count": count("radical"),
        "explicit_hydrogen_count": count("explicit_hydrogen"),
    }


def deterministic_stratified_sample(
    rows: Sequence[Mapping[str, Any]], *, label_col: str, max_rows: int | None, seed: int
) -> list[dict[str, Any]]:
    copied = [dict(row) for row in rows]
    if max_rows is None or max_rows <= 0 or max_rows >= len(copied):
        return copied
    groups: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(copied):
        groups[normalize_binary_label(row[label_col])].append((index, row))
    exact = {label: max_rows * len(items) / len(copied) for label, items in groups.items()}
    quotas = {label: min(len(groups[label]), int(math.floor(value))) for label, value in exact.items()}
    remaining = max_rows - sum(quotas.values())
    order = sorted(groups, key=lambda label: (-(exact[label] - quotas[label]), label))
    while remaining > 0:
        progressed = False
        for label in order:
            if quotas[label] < len(groups[label]):
                quotas[label] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    rng = random.Random(seed)
    selected: list[tuple[int, dict[str, Any]]] = []
    for label in sorted(groups):
        items = list(groups[label])
        rng.shuffle(items)
        selected.extend(items[: quotas[label]])
    selected.sort(key=lambda item: item[0])
    return [row for _index, row in selected]


def stable_molecule_id(canonical_smiles: str) -> str:
    digest = hashlib.sha256(canonical_smiles.encode("utf-8")).hexdigest()[:20].upper()
    return f"MUT_{digest}"


def _drop_reason(audit: Mapping[str, Any]) -> str:
    if not audit.get("parse_ok"):
        return "parse_failed"
    if not audit.get("sanitize_ok"):
        return "sanitize_failed"
    if int(audit.get("num_components") or 0) > 1 or bool(audit.get("contains_dot")):
        return "multicomponent"
    if bool(audit.get("dummy_atom")):
        return "dummy_atom"
    if int(audit.get("num_atoms") or 0) <= 0:
        return "empty_molecule"
    if int(audit.get("num_heavy_atoms") or 0) <= 0:
        return "zero_heavy_atoms"
    return ""


MASTER_FIELDS = (
    "source_dataset",
    "source_row_id",
    "curated_row_id",
    "smiles_original",
    "mutagenicity_original",
    "label",
    "semantic_label",
    "molecule_id",
    "smiles",
    "canonical_nonisomeric_smiles",
    "scaffold_smiles",
    "formal_charge",
    "num_components",
    "num_atoms",
    "num_heavy_atoms",
    "num_bonds",
    "molecular_weight",
    "atom_types",
    "bond_types",
    "parse_ok",
    "sanitize_ok",
    "contains_dot",
    "dummy_atom",
    "keep",
    "drop_reason",
)

BENCHMARK_FIELDS = (
    "molecule_id",
    "source_dataset",
    "source_row_id",
    "source_row_ids",
    "smiles_original",
    "mutagenicity_original",
    "smiles",
    "canonical_nonisomeric_smiles",
    "label",
    "semantic_label",
    "scaffold_smiles",
    "formal_charge",
    "num_components",
    "num_atoms",
    "num_heavy_atoms",
    "num_bonds",
    "molecular_weight",
    "atom_types",
    "bond_types",
    "keep",
    "drop_reason",
)


def preprocess_curated_source(
    curated_source: CsvSource,
    *,
    raw_source: CsvSource | None = None,
    max_rows: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    indexed_rows = [
        {"__curated_row_id": index, **row}
        for index, row in enumerate(curated_source.rows, 1)
    ]
    selected = deterministic_stratified_sample(
        indexed_rows,
        label_col=curated_source.label_col,
        max_rows=max_rows,
        seed=seed,
    )
    # The curated source has no graph ID column. Its one-based CSV row is the
    # only unambiguous provenance ID; exact raw-SMILES matching can be ambiguous
    # when the raw file contains duplicate strings.
    _ = raw_source
    master: list[dict[str, Any]] = []
    canonical_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        curated_row_id = int(row["__curated_row_id"])
        smiles_original = str(row[curated_source.smiles_col]).strip()
        original_label = row[curated_source.label_col]
        label = normalize_binary_label(original_label)
        source_row_id = curated_row_id
        audit = audit_smiles(smiles_original)
        canonical = str(audit["canonical_isomeric_smiles"] or "")
        reason = _drop_reason(audit)
        molecule_id = stable_molecule_id(canonical) if canonical else ""
        record = {
            "source_dataset": "smiles_mutagenicity_curated",
            "source_row_id": source_row_id,
            "curated_row_id": curated_row_id,
            "smiles_original": smiles_original,
            "mutagenicity_original": original_label,
            "label": label,
            "semantic_label": SEMANTIC_LABELS[label],
            "molecule_id": molecule_id,
            "smiles": canonical,
            "canonical_nonisomeric_smiles": audit["canonical_nonisomeric_smiles"],
            "scaffold_smiles": audit["scaffold_smiles"],
            "formal_charge": audit["formal_charge"],
            "num_components": audit["num_components"],
            "num_atoms": audit["num_atoms"],
            "num_heavy_atoms": audit["num_heavy_atoms"],
            "num_bonds": audit["num_bonds"],
            "molecular_weight": audit["molecular_weight"],
            "atom_types": audit["atom_types"],
            "bond_types": audit["bond_types"],
            "parse_ok": audit["parse_ok"],
            "sanitize_ok": audit["sanitize_ok"],
            "contains_dot": audit["contains_dot"],
            "dummy_atom": audit["dummy_atom"],
            "keep": not reason,
            "drop_reason": reason,
        }
        master.append(record)
        if not reason:
            canonical_groups[canonical].append(record)

    duplicates: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    clean: list[dict[str, Any]] = []
    for canonical in sorted(canonical_groups):
        group = canonical_groups[canonical]
        labels = sorted({int(record["label"]) for record in group})
        source_ids = [str(record["source_row_id"]) for record in group]
        if len(labels) > 1:
            for record in group:
                record["keep"] = False
                record["drop_reason"] = "label_conflict"
            conflicts.append(
                {
                    "canonical_isomeric_smiles": canonical,
                    "molecule_id": stable_molecule_id(canonical),
                    "labels": ";".join(str(label) for label in labels),
                    "source_row_ids": ";".join(source_ids),
                    "conflict_count": len(group),
                }
            )
            continue
        representative = min(
            group,
            key=lambda record: (str(record["source_row_id"]), int(record["curated_row_id"])),
        )
        for record in group:
            if record is not representative:
                record["keep"] = False
                record["drop_reason"] = "duplicate_same_label"
        if len(group) > 1:
            duplicates.append(
                {
                    "canonical_isomeric_smiles": canonical,
                    "molecule_id": representative["molecule_id"],
                    "label": labels[0],
                    "source_row_ids": ";".join(source_ids),
                    "duplicate_count": len(group),
                    "representative_source_row_id": representative["source_row_id"],
                }
            )
        clean.append(
            {
                **{field: representative.get(field) for field in BENCHMARK_FIELDS},
                "source_row_ids": ";".join(source_ids),
                "keep": True,
                "drop_reason": "",
            }
        )

    clean.sort(key=lambda row: str(row["molecule_id"]))
    clean_ids = [str(row["molecule_id"]) for row in clean]
    if len(set(clean_ids)) != len(clean_ids):
        raise RuntimeError("Stable molecule_id hash collision detected")
    dropped = [record for record in master if not bool(record["keep"])]
    id_map = [
        {
            "source_dataset": record["source_dataset"],
            "source_row_id": record["source_row_id"],
            "curated_row_id": record["curated_row_id"],
            "smiles_original": record["smiles_original"],
            "canonical_isomeric_smiles": record["smiles"],
            "molecule_id": record["molecule_id"],
            "label": record["label"],
            "keep": record["keep"],
            "drop_reason": record["drop_reason"],
        }
        for record in master
    ]
    return {
        "master": master,
        "clean": clean,
        "dropped": dropped,
        "duplicates": duplicates,
        "conflicts": conflicts,
        "id_map": id_map,
        "selected_rows": len(selected),
        "source_rows": len(curated_source.rows),
        "max_rows": max_rows,
        "seed": seed,
    }


def preprocessing_summary(result: Mapping[str, Any], curated_path: Path) -> dict[str, Any]:
    clean = list(result["clean"])
    dropped = list(result["dropped"])
    return {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "source_curated_csv": str(curated_path),
        "source_rows": int(result["source_rows"]),
        "selected_rows": int(result["selected_rows"]),
        "max_rows": result["max_rows"],
        "seed": int(result["seed"]),
        "num_clean_molecules": len(clean),
        "num_dropped_rows": len(dropped),
        "num_duplicate_groups": len(result["duplicates"]),
        "num_conflict_groups": len(result["conflicts"]),
        "clean_label_counts": _json_counter(int(row["label"]) for row in clean),
        "drop_reason_counts": _json_counter(str(row["drop_reason"]) for row in dropped),
        "dedup_key": "canonical_isomeric_smiles",
        "canonical_isomeric": True,
        "neutralize": False,
        "tautomer_canonicalize": False,
        "drop_multicomponent": True,
        "preprocess_passed": bool(clean) and not any(not row.get("smiles") for row in clean),
    }


def _json_counter(values: Iterable[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items(), key=lambda x: str(x[0]))}


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _scaffold_group_key(row: Mapping[str, Any]) -> str:
    scaffold = str(row.get("scaffold_smiles") or "").strip()
    return scaffold if scaffold else "__ACYCLIC__"


def _stable_seed_value(seed: int, value: str, attempt: int) -> int:
    payload = f"{seed}|{attempt}|{value}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def _split_assignment_score(
    counts: Mapping[str, int],
    label_counts: Mapping[str, Counter[int]],
    *,
    targets: Mapping[str, float],
    label_targets: Mapping[str, Mapping[int, float]],
) -> float:
    score = 0.0
    for split in SPLIT_NAMES:
        target = max(float(targets[split]), 1.0)
        observed = float(counts.get(split, 0))
        relative_error = abs(observed - target) / target
        overfill = max(0.0, observed - target) / target
        score += relative_error + 4.0 * overfill
        for label in (0, 1):
            label_target = max(float(label_targets[split][label]), 1.0)
            label_observed = float(label_counts.get(split, Counter()).get(label, 0))
            score += 0.75 * abs(label_observed - label_target) / label_target
    return score


def _assignment_final_score(
    assignments: Mapping[str, str],
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    ratios: Mapping[str, float],
) -> float:
    total = sum(len(group) for group in groups.values())
    all_labels = Counter(int(row["label"]) for group in groups.values() for row in group)
    counts = Counter()
    label_counts: dict[str, Counter[int]] = {split: Counter() for split in SPLIT_NAMES}
    for scaffold, split in assignments.items():
        group = groups[scaffold]
        counts[split] += len(group)
        label_counts[split].update(int(row["label"]) for row in group)
    targets = {split: total * ratios[split] for split in SPLIT_NAMES}
    label_targets = {
        split: {label: all_labels[label] * ratios[split] for label in (0, 1)}
        for split in SPLIT_NAMES
    }
    score = _split_assignment_score(
        counts,
        label_counts,
        targets=targets,
        label_targets=label_targets,
    )
    for split in SPLIT_NAMES:
        if counts[split] == 0:
            score += 10000.0
        for label in (0, 1):
            if label_counts[split][label] == 0:
                score += 1000.0
    return score


def build_scaffold_splits(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    ratios: Mapping[str, float] | None = None,
    attempts: int = 128,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_ratios = dict(DEFAULT_SPLIT_RATIOS if ratios is None else ratios)
    if set(split_ratios) != set(SPLIT_NAMES):
        raise ValueError(f"Split ratios must define exactly {SPLIT_NAMES}: {split_ratios}")
    if not math.isclose(sum(split_ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Split ratios must sum to one: {split_ratios}")
    normalized_rows = [dict(row) for row in rows]
    if not normalized_rows:
        raise ValueError("Cannot split an empty Mutagenicity benchmark")
    molecule_ids = [str(row["molecule_id"]) for row in normalized_rows]
    smiles_values = [str(row["smiles"]) for row in normalized_rows]
    if len(set(molecule_ids)) != len(molecule_ids):
        raise ValueError("Benchmark contains duplicate molecule_id values before splitting")
    if len(set(smiles_values)) != len(smiles_values):
        raise ValueError("Benchmark contains duplicate canonical SMILES before splitting")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        groups[_scaffold_group_key(row)].append(row)
    total = len(normalized_rows)
    total_labels = Counter(int(row["label"]) for row in normalized_rows)
    targets = {split: total * split_ratios[split] for split in SPLIT_NAMES}
    label_targets = {
        split: {label: total_labels[label] * split_ratios[split] for label in (0, 1)}
        for split in SPLIT_NAMES
    }

    best_assignment: dict[str, str] | None = None
    best_score: float | None = None
    for attempt in range(max(1, int(attempts))):
        ordered_groups = sorted(
            groups,
            key=lambda scaffold: (
                -len(groups[scaffold]),
                _stable_seed_value(seed, scaffold, attempt),
                scaffold,
            ),
        )
        assignments: dict[str, str] = {}
        counts: Counter[str] = Counter()
        label_counts: dict[str, Counter[int]] = {split: Counter() for split in SPLIT_NAMES}
        split_tie_order = list(SPLIT_NAMES)
        random.Random(seed + attempt).shuffle(split_tie_order)
        tie_position = {split: index for index, split in enumerate(split_tie_order)}
        for scaffold in ordered_groups:
            group = groups[scaffold]
            group_labels = Counter(int(row["label"]) for row in group)
            candidates: list[tuple[float, int, str]] = []
            for split in SPLIT_NAMES:
                trial_counts = Counter(counts)
                trial_counts[split] += len(group)
                trial_labels = {name: Counter(values) for name, values in label_counts.items()}
                trial_labels[split].update(group_labels)
                score = _split_assignment_score(
                    trial_counts,
                    trial_labels,
                    targets=targets,
                    label_targets=label_targets,
                )
                candidates.append((score, tie_position[split], split))
            _score, _tie, selected_split = min(candidates)
            assignments[scaffold] = selected_split
            counts[selected_split] += len(group)
            label_counts[selected_split].update(group_labels)
        final_score = _assignment_final_score(assignments, groups, ratios=split_ratios)
        assignment_signature = tuple(sorted(assignments.items()))
        current_signature = tuple(sorted(best_assignment.items())) if best_assignment else None
        if (
            best_score is None
            or final_score < best_score - 1e-12
            or (
                math.isclose(final_score, best_score, rel_tol=0.0, abs_tol=1e-12)
                and (current_signature is None or assignment_signature < current_signature)
            )
        ):
            best_score = final_score
            best_assignment = assignments
    if best_assignment is None:
        raise RuntimeError("Failed to build a scaffold split assignment")

    manifest: list[dict[str, Any]] = []
    for row in normalized_rows:
        split = best_assignment[_scaffold_group_key(row)]
        manifest.append(
            {
                "molecule_id": str(row["molecule_id"]),
                "smiles": str(row["smiles"]),
                "label": int(row["label"]),
                "semantic_label": str(row["semantic_label"]),
                "scaffold_smiles": str(row.get("scaffold_smiles") or ""),
                "split": split,
            }
        )
    manifest.sort(key=lambda row: (SPLIT_NAMES.index(str(row["split"])), str(row["molecule_id"])))
    summary = validate_split_manifest(
        normalized_rows,
        manifest,
        ratios=split_ratios,
        seed=seed,
        assignment_score=float(best_score),
    )
    if not summary["split_validation_passed"]:
        raise ValueError(f"Generated split failed validation: {summary}")
    return manifest, summary


def _pairwise_overlap_count(values_by_split: Mapping[str, set[str]]) -> int:
    overlaps: set[str] = set()
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            overlaps.update(values_by_split[left] & values_by_split[right])
    return len(overlaps)


def validate_split_manifest(
    benchmark_rows: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
    *,
    ratios: Mapping[str, float],
    seed: int,
    assignment_score: float | None = None,
) -> dict[str, Any]:
    benchmark_ids = [str(row["molecule_id"]) for row in benchmark_rows]
    manifest_ids = [str(row["molecule_id"]) for row in manifest]
    split_counts = Counter(str(row["split"]) for row in manifest)
    split_label_counts = {
        split: Counter(int(row["label"]) for row in manifest if str(row["split"]) == split)
        for split in SPLIT_NAMES
    }
    ids_by_split = {
        split: {str(row["molecule_id"]) for row in manifest if str(row["split"]) == split}
        for split in SPLIT_NAMES
    }
    smiles_by_split = {
        split: {str(row["smiles"]) for row in manifest if str(row["split"]) == split}
        for split in SPLIT_NAMES
    }
    scaffold_by_split = {
        split: {
            str(row.get("scaffold_smiles") or "__ACYCLIC__")
            for row in manifest
            if str(row["split"]) == split
        }
        for split in SPLIT_NAMES
    }
    missing_ids = sorted(set(benchmark_ids) - set(manifest_ids))
    extra_ids = sorted(set(manifest_ids) - set(benchmark_ids))
    duplicate_count = len(manifest_ids) - len(set(manifest_ids))
    labels_present = {
        split: sorted(split_label_counts[split]) == [0, 1] for split in SPLIT_NAMES
    }
    total = len(benchmark_rows)
    scaffold_overlap = _pairwise_overlap_count(scaffold_by_split)
    smiles_overlap = _pairwise_overlap_count(smiles_by_split)
    molecule_overlap = _pairwise_overlap_count(ids_by_split)
    passed = bool(
        len(manifest) == total
        and not missing_ids
        and not extra_ids
        and duplicate_count == 0
        and scaffold_overlap == 0
        and smiles_overlap == 0
        and molecule_overlap == 0
        and all(labels_present.values())
        and set(split_counts) == set(SPLIT_NAMES)
    )
    return {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "seed": seed,
        "target_ratios": {split: float(ratios[split]) for split in SPLIT_NAMES},
        "total_rows": total,
        "split_counts": {split: int(split_counts[split]) for split in SPLIT_NAMES},
        "split_label_counts": {
            split: {str(label): int(split_label_counts[split][label]) for label in (0, 1)}
            for split in SPLIT_NAMES
        },
        "split_ratios_actual": {
            split: split_counts[split] / total if total else 0.0 for split in SPLIT_NAMES
        },
        "num_unique_scaffolds": len(
            {str(row.get("scaffold_smiles") or "__ACYCLIC__") for row in manifest}
        ),
        "scaffold_overlap_count": scaffold_overlap,
        "canonical_smiles_overlap_count": smiles_overlap,
        "molecule_id_overlap_count": molecule_overlap,
        "missing_molecule_count": len(missing_ids),
        "extra_molecule_count": len(extra_ids),
        "duplicate_molecule_count": duplicate_count,
        "each_split_contains_both_labels": labels_present,
        "assignment_score": assignment_score,
        "split_validation_passed": passed,
    }


def processed_validation(
    processed_dir: str | Path,
) -> tuple[dict[str, Any], list[str]]:
    require_rdkit()
    root = Path(processed_dir)
    benchmark_path = root / "mutagenicity_benchmark_clean.csv"
    benchmark = read_csv_rows(benchmark_path)
    split_rows: dict[str, list[dict[str, str]]] = {
        split: read_csv_rows(root / f"{split}.csv") for split in SPLIT_NAMES
    }
    errors: list[str] = []
    benchmark_ids = {row["molecule_id"] for row in benchmark}
    benchmark_smiles = {row["smiles"] for row in benchmark}
    invalid_smiles = 0
    dots = 0
    dummy_atoms = 0
    missing_smiles = 0
    invalid_labels = 0
    for row in benchmark:
        smiles = str(row.get("smiles") or "").strip()
        if not smiles:
            missing_smiles += 1
            continue
        if "." in smiles:
            dots += 1
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            invalid_smiles += 1
        elif any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
            dummy_atoms += 1
        try:
            label = int(row["label"])
        except (TypeError, ValueError):
            invalid_labels += 1
        else:
            invalid_labels += label not in (0, 1)
    all_split_rows = [row for split in SPLIT_NAMES for row in split_rows[split]]
    split_ids = [row["molecule_id"] for row in all_split_rows]
    split_id_set = set(split_ids)
    labels_by_split = {
        split: sorted({int(row["label"]) for row in rows}) for split, rows in split_rows.items()
    }
    ids_by_split = {split: {row["molecule_id"] for row in rows} for split, rows in split_rows.items()}
    smiles_by_split = {split: {row["smiles"] for row in rows} for split, rows in split_rows.items()}
    scaffolds_by_split = {
        split: {str(row.get("scaffold_smiles") or "__ACYCLIC__") for row in rows}
        for split, rows in split_rows.items()
    }
    conflicts_path = root / "mutagenicity_conflicts.csv"
    conflicts = read_csv_rows(conflicts_path) if conflicts_path.is_file() else []
    conflict_smiles = {row.get("canonical_isomeric_smiles", "") for row in conflicts}
    conflict_in_benchmark = len((conflict_smiles - {""}) & benchmark_smiles)
    metrics = {
        "processed_dir": str(root),
        "benchmark_rows": len(benchmark),
        "split_rows": {split: len(rows) for split, rows in split_rows.items()},
        "invalid_smiles_count": invalid_smiles,
        "missing_smiles_count": missing_smiles,
        "contains_dot_count": dots,
        "dummy_atom_count": dummy_atoms,
        "invalid_label_count": invalid_labels,
        "split_label_values": labels_by_split,
        "split_molecule_id_overlap_count": _pairwise_overlap_count(ids_by_split),
        "split_canonical_smiles_overlap_count": _pairwise_overlap_count(smiles_by_split),
        "split_scaffold_overlap_count": _pairwise_overlap_count(scaffolds_by_split),
        "missing_from_splits_count": len(benchmark_ids - split_id_set),
        "extra_in_splits_count": len(split_id_set - benchmark_ids),
        "duplicate_split_molecule_count": len(split_ids) - len(split_id_set),
        "label_conflict_in_benchmark_count": conflict_in_benchmark,
    }
    if invalid_smiles:
        errors.append(f"invalid_smiles:{invalid_smiles}")
    if missing_smiles:
        errors.append(f"missing_smiles:{missing_smiles}")
    if dots:
        errors.append(f"multicomponent_smiles:{dots}")
    if dummy_atoms:
        errors.append(f"dummy_atoms:{dummy_atoms}")
    if invalid_labels:
        errors.append(f"invalid_labels:{invalid_labels}")
    for split, labels in labels_by_split.items():
        if labels != [0, 1]:
            errors.append(f"split_missing_binary_class:{split}:{labels}")
    for key in (
        "split_molecule_id_overlap_count",
        "split_canonical_smiles_overlap_count",
        "split_scaffold_overlap_count",
        "missing_from_splits_count",
        "extra_in_splits_count",
        "duplicate_split_molecule_count",
        "label_conflict_in_benchmark_count",
    ):
        if metrics[key]:
            errors.append(f"{key}:{metrics[key]}")
    metrics["errors"] = errors
    metrics["validation_passed"] = not errors
    return metrics, errors


__all__ = [
    "BENCHMARK_FIELDS",
    "DATASET_NAME",
    "DATASET_VERSION",
    "DEFAULT_SPLIT_RATIOS",
    "MASTER_FIELDS",
    "RDKIT_AUDIT_FIELDS",
    "REQUIRED_DOWNLOAD_FILES",
    "SEMANTIC_LABELS",
    "SPLIT_NAMES",
    "STANDARD_SPLIT_FIELDS",
    "CsvSource",
    "audit_csv_source",
    "audit_smiles",
    "build_scaffold_splits",
    "deterministic_stratified_sample",
    "evaluate_tu_label_mappings",
    "infer_csv_columns",
    "load_csv_source",
    "normalize_binary_label",
    "parse_sha256_manifest",
    "preprocess_curated_source",
    "preprocessing_summary",
    "processed_validation",
    "read_csv_rows",
    "read_integer_lines",
    "require_rdkit",
    "sha256_file",
    "stable_molecule_id",
    "summarize_rdkit_audit",
    "validate_split_manifest",
    "verify_download",
    "write_csv",
    "write_json",
]
