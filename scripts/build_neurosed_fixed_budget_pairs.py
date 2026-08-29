#!/usr/bin/env python3
"""Build deterministic unlabeled Taste NeuroSED query-target pairs.

This command never calls a classifier and never writes a GED/SED label.  The
result is an independent pair universe for a later authenticated pyged/GEDLIB
labeling stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.tastemolnet_neurosed_fixed_budget import (  # noqa: E402
    FixedBudgetGraph,
    FixedBudgetPairError,
    bind_fixed_budget_pair_manifest,
    fixed_budget_pair_manifest,
    partition_disjoint_benchmarks,
    sample_fixed_budget_pairs,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _held_bytes(path: Path, expected_sha256: str, *, label: str) -> bytes:
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise FixedBudgetPairError(f"{label} path must be normalized absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise FixedBudgetPairError(f"{label} path contains a symlink")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or after.st_dev != named.st_dev
            or after.st_ino != named.st_ino
        ):
            raise FixedBudgetPairError(f"{label} changed while held")
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    if _sha256(data) != expected_sha256:
        raise FixedBudgetPairError(f"{label} SHA256 differs from held authority")
    return data


def _feature_schema(data: bytes) -> tuple[tuple[int, ...], str, str]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FixedBudgetPairError("feature schema is not one UTF-8 JSON object") from exc
    if type(payload) is not dict:
        raise FixedBudgetPairError("feature schema is not one JSON object")
    vocabulary = payload.get("feature_atomic_numbers")
    if (
        payload.get("schema_version")
        != "tastemolnet_gcf_neurosed_feature_schema_v1"
        or payload.get("dataset") != "tastemolnet"
        or payload.get("node_feature_semantics") != "one_hot_atomic_number"
        or payload.get("explicit_h_nodes") is not True
        or payload.get("native_adjacency_semantics")
        != "binary_connectivity_directed_both_ways"
        or payload.get("edge_features_used") is not False
        or payload.get("validation_unseen_atomic_numbers") != []
        or payload.get("train_derived_only") is not True
        or type(vocabulary) is not list
        or not vocabulary
        or any(type(value) is not int or value <= 0 for value in vocabulary)
        or vocabulary != sorted(set(vocabulary))
        or payload.get("input_dim") != len(vocabulary)
    ):
        raise FixedBudgetPairError("reviewed Taste NeuroSED feature schema changed")
    return tuple(vocabulary), "explicit", str(payload["schema_version"])


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in rows
    )


def _load_graphs(
    data: bytes,
    *,
    expected_split: str,
    atomic_number_vocabulary: tuple[int, ...],
    hydrogen_mode: str,
) -> list[FixedBudgetGraph]:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:
        raise FixedBudgetPairError("pair construction requires local RDKit") from exc
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise FixedBudgetPairError("split CSV is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    rows = [dict(row) for row in reader]
    if not rows:
        raise FixedBudgetPairError("split CSV is empty")
    lookup = {atomic_number: index for index, atomic_number in enumerate(atomic_number_vocabulary)}
    graphs: list[FixedBudgetGraph] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        split = str(row.get("split") or "").strip()
        graph_id = str(row.get("molecule_id") or "").strip()
        smiles = str(row.get("model_smiles") or row.get("canonical_smiles") or "").strip()
        label_text = str(row.get("label") or "").strip()
        if split != expected_split or not graph_id or graph_id in seen or not smiles:
            raise FixedBudgetPairError(f"invalid {expected_split} CSV row {index}")
        if label_text not in {"0", "1", "2"}:
            raise FixedBudgetPairError(f"Taste diagnostic label changed at row {index}")
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise FixedBudgetPairError(f"unparsable Taste SMILES at row {index}")
        Chem.SanitizeMol(molecule)
        if len(Chem.GetMolFrags(molecule)) != 1:
            raise FixedBudgetPairError(f"disconnected Taste graph at row {index}")
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        if not scaffold:
            scaffold = "ACYCLIC:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        graph_molecule = (
            Chem.AddHs(molecule, addCoords=False)
            if hydrogen_mode == "explicit"
            else molecule
        )
        node_labels: list[int] = []
        for atom in graph_molecule.GetAtoms():
            atomic_number = int(atom.GetAtomicNum())
            if atomic_number not in lookup:
                raise FixedBudgetPairError(
                    f"atomic number {atomic_number} is absent from the pinned vocabulary"
                )
            node_labels.append(lookup[atomic_number])
        edges: list[tuple[int, int]] = []
        for bond in graph_molecule.GetBonds():
            left = int(bond.GetBeginAtomIdx())
            right = int(bond.GetEndAtomIdx())
            edges.extend(((left, right), (right, left)))
        graphs.append(
            FixedBudgetGraph(
                graph_id=graph_id,
                split=split,
                node_labels=tuple(node_labels),
                directed_edges=tuple(sorted(edges)),
                scaffold=scaffold,
                class_label=int(label_text),
            )
        )
        seen.add(graph_id)
    return graphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--expected-split", choices=("train", "validation"), required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--feature-schema-json", type=Path, required=True)
    parser.add_argument("--expected-feature-schema-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, required=True)
    parser.add_argument("--seed", type=int, choices=(7,), default=7)
    parser.add_argument("--n-hops-query", type=int, required=True)
    parser.add_argument("--traversal-probability-query", type=float, required=True)
    parser.add_argument("--node-limit-query", type=int)
    parser.add_argument("--write-disjoint-benchmark-cohorts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.split_csv.name != f"{args.expected_split}.csv":
        raise FixedBudgetPairError("split CSV basename does not match expected split")
    if len(args.expected_split_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in args.expected_split_sha256
    ):
        raise FixedBudgetPairError("expected split SHA256 must be lowercase hexadecimal")
    if len(args.expected_feature_schema_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in args.expected_feature_schema_sha256
    ):
        raise FixedBudgetPairError(
            "expected feature-schema SHA256 must be lowercase hexadecimal"
        )
    schema_data = _held_bytes(
        args.feature_schema_json,
        args.expected_feature_schema_sha256,
        label="feature schema",
    )
    vocabulary, hydrogen_mode, feature_schema_version = _feature_schema(schema_data)
    data = _held_bytes(
        args.split_csv,
        args.expected_split_sha256,
        label=f"{args.expected_split} split CSV",
    )
    graphs = _load_graphs(
        data,
        expected_split=args.expected_split,
        atomic_number_vocabulary=vocabulary,
        hydrogen_mode=hydrogen_mode,
    )
    pairs = sample_fixed_budget_pairs(
        graphs,
        split=args.expected_split,
        pair_count=args.pair_count,
        seed=args.seed,
        n_hops_query=args.n_hops_query,
        traversal_probability_query=args.traversal_probability_query,
        node_limit_query=args.node_limit_query,
    )
    metadata_rows = [pair.metadata() for pair in pairs]
    pairs_text = _jsonl(metadata_rows)
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FixedBudgetPairError("pair output directory is not fresh")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_text(output / "pairs.jsonl", pairs_text)
    graph_rows = [
        {
            "graph_id": graph.graph_id,
            "split": graph.split,
            "node_labels": list(graph.node_labels),
            "directed_edges": [list(edge) for edge in graph.directed_edges],
            "num_nodes": graph.num_nodes,
            "num_undirected_edges": graph.num_undirected_edges,
            "scaffold": graph.scaffold,
            "class_label_sampling_diagnostic_only": graph.class_label,
            "graph_sha256": graph.graph_sha256,
            "canonical_graph_sha256": graph.canonical_graph_sha256,
        }
        for graph in graphs
    ]
    graph_inventory_text = _jsonl(graph_rows)
    _atomic_text(output / "graph_inventory.jsonl", graph_inventory_text)
    manifest = fixed_budget_pair_manifest(
        pairs,
        split=args.expected_split,
        seed=args.seed,
        n_hops_query=args.n_hops_query,
        traversal_probability_query=args.traversal_probability_query,
        node_limit_query=args.node_limit_query,
    )
    manifest.update(
        {
            "source_csv_sha256": args.expected_split_sha256,
            "feature_schema_sha256": args.expected_feature_schema_sha256,
            "source_graph_count": len(graphs),
            "feature_schema_version": feature_schema_version,
            "graph_hydrogen_mode": hydrogen_mode,
            "feature_atomic_number_vocabulary": list(vocabulary),
            "graph_inventory_sha256": _sha256(
                graph_inventory_text.encode("utf-8")
            ),
            "pairs_jsonl_sha256": _sha256(pairs_text.encode("utf-8")),
        }
    )
    if args.write_disjoint_benchmark_cohorts:
        cohorts = partition_disjoint_benchmarks(pairs)
        hashes: dict[str, str] = {}
        for budget, cohort in cohorts.items():
            cohort_text = _jsonl(pair.metadata() for pair in cohort)
            filename = f"benchmark_pairs_{budget}.jsonl"
            _atomic_text(output / filename, cohort_text)
            hashes[str(budget)] = _sha256(cohort_text.encode("utf-8"))
        manifest["benchmark_budgets"] = [100, 500, 1000]
        manifest["benchmark_cohorts_disjoint"] = True
        manifest["benchmark_cohort_file_sha256"] = hashes
    manifest = bind_fixed_budget_pair_manifest(manifest)
    _atomic_text(
        output / "pair_sampler_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
