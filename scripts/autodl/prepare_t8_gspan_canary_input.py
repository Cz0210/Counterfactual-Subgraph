#!/usr/bin/env python3
"""Export the exact train-only graph list consumed by Taste T8 gSpan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_mutagenicity_adapter import (  # noqa: E402
    OfficialGlobalGCEMutagenicityGenerator,
    TrainParent,
    _prepare_native_and_source_datasets,
)
from src.baselines.tastemolnet_globalgce_full import (  # noqa: E402
    NUM_CLASSES,
    SOURCE_LABEL,
    _checkpoint_payloads,
    select_full_sweet_train_cohort,
)
from src.baselines.tastemolnet_globalgce_smoke import (  # noqa: E402
    FrozenTasteGINEScorer,
)
from src.data.tastemolnet_ppo import TASTEMOLNET_PREPARED_FIELDS  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _load_train(path: Path) -> list[TrainParent]:
    resolved = path.expanduser().resolve(strict=True)
    rows: list[TrainParent] = []
    seen: set[str] = set()
    with resolved.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if tuple(reader.fieldnames or ()) != TASTEMOLNET_PREPARED_FIELDS:
            raise ValueError("Taste train schema differs from the prepared authority")
        for line_number, row in enumerate(reader, start=2):
            parent_id = str(row.get("molecule_id") or "").strip()
            smiles = str(row.get("model_smiles") or "").strip()
            label_text = str(row.get("label") or "").strip()
            if (
                None in row
                or set(row) != set(TASTEMOLNET_PREPARED_FIELDS)
                or not parent_id
                or parent_id in seen
                or not smiles
                or label_text not in {"0", "1", "2"}
                or str(row.get("split") or "") != "train"
                or str(row.get("exclusion_reason") or "").strip()
            ):
                raise ValueError(f"invalid prepared train row {line_number}")
            seen.add(parent_id)
            rows.append(TrainParent(parent_id, smiles, int(label_text), "train"))
    if {row.label for row in rows} != {0, 1, 2}:
        raise ValueError("Taste train input does not contain all three classes")
    return rows


def _edge_position(left: int, right: int) -> int:
    high, low = max(left, right), min(left, right)
    return (high - 1) * high // 2 + low


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--gnn-checkpoint", required=True, type=Path)
    parser.add_argument("--official-root", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--target-label", default=0, choices=(0, 2), type=int)
    parser.add_argument("--expected-selected-count", type=int)
    parser.add_argument("--expected-selected-cohort-sha256")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.config.expanduser().is_file():
        raise SystemExit("--config must identify an existing regular file")
    if args.set not in ([], ["inference.fallback_to_heuristic=false"]):
        raise SystemExit("only inference.fallback_to_heuristic=false is accepted")
    if args.seed != 7:
        raise SystemExit("the protected T8/T13 route is pinned to seed 7")
    output = args.output_jsonl.expanduser().absolute()
    manifest_path = args.manifest.expanduser().absolute()
    if output.exists() or manifest_path.exists():
        raise SystemExit("canary input outputs must be fresh")

    train = args.train_csv.expanduser().resolve(strict=True)
    checkpoint = args.gnn_checkpoint.expanduser().resolve(strict=True)
    official = args.official_root.expanduser().resolve(strict=True)
    payloads = _checkpoint_payloads(checkpoint)
    scorer = FrozenTasteGINEScorer(payloads, device=args.device, batch_size=256)
    selected, selection = select_full_sweet_train_cohort(
        _load_train(train), scorer=scorer, batch_size=256
    )
    if (
        args.expected_selected_count is not None
        and selection["selected_count"] != args.expected_selected_count
    ):
        raise SystemExit("selected T8 parent count differs from the protected route")
    expected_cohort = args.expected_selected_cohort_sha256
    if expected_cohort is not None and selection["selected_cohort_sha256"] != expected_cohort:
        raise SystemExit("selected T8 parent cohort differs from the protected route")

    generator = OfficialGlobalGCEMutagenicityGenerator(
        official,
        native_train_csv=train,
        dataset_name="TasteMolNet",
        min_freq=2,
        frozen_gine_checkpoint=checkpoint,
        source_label=SOURCE_LABEL,
        target_label=args.target_label,
        num_classes=NUM_CLASSES,
        require_isolated_imports=True,
        rules_only_min_valid_native_rules=0,
    )
    import torch

    prepared = _prepare_native_and_source_datasets(
        native_train_csv=train,
        parents=selected,
        seed=args.seed,
        torch_module=torch,
        dataset_name="TasteMolNet",
        num_classes=NUM_CLASSES,
        source_label=SOURCE_LABEL,
        frozen_gine_feature_schema=generator._frozen_feature_schema_for_native_codec(),
    )
    source_train_idx, source_dataset = prepared[4], prepared[6]
    lines: list[str] = []
    for graph_id, source_index in enumerate(source_train_idx):
        features = source_dataset.feat[source_index].argmax(-1)
        adjacency = source_dataset.adj[source_index]
        edge_labels = source_dataset.edge_attr[source_index].argmax(-1)
        active = [
            node
            for node in range(int(adjacency.shape[0]))
            if bool(adjacency[node].count_nonzero().item())
        ]
        if active != list(range(len(active))) or not active:
            raise RuntimeError("protected gSpan input contains a nonconsecutive/empty graph")
        nodes = [{"id": node, "label": int(features[node])} for node in active]
        edges = []
        for left in active:
            for right in active:
                if right <= left or not bool(adjacency[left, right].item()):
                    continue
                edges.append(
                    {
                        "source": left,
                        "target": right,
                        "label": int(edge_labels[_edge_position(left, right)]),
                    }
                )
        lines.append(
            json.dumps(
                {"graph_id": graph_id, "nodes": nodes, "edges": edges},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
    _atomic_text(output, "".join(lines))
    manifest = {
        "schema_version": "tastemolnet_t8_gspan_canary_input_v1",
        "status": "PASS",
        "train_only": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "seed": args.seed,
        "source_label": SOURCE_LABEL,
        "target_label": args.target_label,
        "selected_parent_count": len(selected),
        "selected_parent_cohort_sha256": selection["selected_cohort_sha256"],
        "gspan_graph_count": len(source_train_idx),
        "graph_jsonl": str(output),
        "graph_jsonl_sha256": _sha256(output),
        "frozen_gine_checkpoint_id": scorer.checkpoint_id,
        "native_train_csv_sha256": _sha256(train),
        "scientific_route_modified": False,
    }
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
