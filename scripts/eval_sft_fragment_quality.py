#!/usr/bin/env python3
"""Run SFT inference and score core-only fragment-quality metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_infer_sft import (  # type: ignore[import-not-found]
    InferenceExample,
    build_lora_model,
    build_tokenizer,
    find_latest_lora_checkpoint,
    generate_one,
)

from src.chem import match_core_fragment_to_parent, normalize_core_fragment
from src.data.prompts import build_counterfactual_prompt
from src.data.schemas import MoleculeRecord
from src.utils.io import read_jsonl, write_jsonl


DEFAULT_BASE_MODEL = REPO_ROOT / "pretrained_models" / "ChemLLM-7B-Chat"
DEFAULT_LORA_ROOT = REPO_ROOT / "ckpt" / "sft_v3_core_lora"
DEFAULT_EVAL_FILE = REPO_ROOT / "data" / "sft_v3_core_val.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "sft_v3_core_eval"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Accepted for HPC wrapper compatibility. The current evaluator uses explicit CLI overrides only.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Accepted for HPC wrapper compatibility. The current evaluator ignores runtime key=value overrides.",
    )
    parser.add_argument(
        "--base-model",
        default=str(DEFAULT_BASE_MODEL),
        help="Path to the base ChemLLM model.",
    )
    parser.add_argument(
        "--lora-root",
        default=str(DEFAULT_LORA_ROOT),
        help="Directory containing LoRA checkpoints such as checkpoint-500.",
    )
    parser.add_argument(
        "--checkpoint-path",
        help="Optional explicit LoRA checkpoint path. If omitted, the latest checkpoint under --lora-root is used.",
    )
    parser.add_argument(
        "--eval-file",
        default=str(DEFAULT_EVAL_FILE),
        help="Evaluation JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for summary/report outputs.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Optional cap on the number of eval rows. 0 means evaluate the full file.",
    )
    parser.add_argument(
        "--too-small-atoms",
        type=int,
        default=2,
        help="Atom-count threshold used for too_small_rate and bad-case bucketing.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_model_path = Path(args.base_model).expanduser().resolve()
    lora_root = Path(args.lora_root).expanduser().resolve()
    checkpoint_path = (
        Path(args.checkpoint_path).expanduser().resolve()
        if args.checkpoint_path
        else find_latest_lora_checkpoint(lora_root)
    )
    eval_file = Path(args.eval_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(eval_file)
    if args.max_examples and args.max_examples > 0:
        rows = rows[: args.max_examples]
    if not rows:
        raise SystemExit(f"No eval rows were loaded from {eval_file}.")

    tokenizer = build_tokenizer(base_model_path)
    model = build_lora_model(base_model_path, checkpoint_path)

    detailed_rows: list[dict[str, Any]] = []
    iterator = tqdm(rows, total=len(rows), desc="Evaluating core-only SFT outputs")
    for index, row in enumerate(iterator):
        record = _row_to_record(row, index=index)
        if record is None:
            continue
        result = generate_one(
            model,
            tokenizer,
            InferenceExample(
                index=index,
                parent_smiles=record["parent_smiles"],
                prompt=record["prompt"],
                reference_fragment=record["reference_fragment"],
            ),
        )
        evaluation = _evaluate_fragment_output(
            parent_smiles=record["parent_smiles"],
            raw_output=result.generated_fragment,
            too_small_atoms=args.too_small_atoms,
        )
        detailed_rows.append(
            {
                "id": record["id"],
                "graph_id": record["graph_id"],
                "label": record["label"],
                "parent_smiles": record["parent_smiles"],
                "prompt": record["prompt"],
                "reference_fragment": record["reference_fragment"],
                "raw_generation": result.raw_generation,
                "generated_fragment": result.generated_fragment,
                "error": result.error,
                **evaluation,
            }
        )

    if not detailed_rows:
        raise SystemExit("No usable eval rows remained after prompt normalization.")

    summary = _build_summary(detailed_rows, too_small_atoms=args.too_small_atoms)
    summary.update(
        {
            "base_model": str(base_model_path),
            "checkpoint_path": str(checkpoint_path),
            "eval_file": str(eval_file),
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    write_jsonl(output_dir / "details.jsonl", detailed_rows)
    _write_bad_cases(output_dir, detailed_rows)
    (output_dir / "report.txt").write_text(
        _render_report(summary),
        encoding="utf-8",
    )

    print("SFT core-only fragment-quality evaluation completed.")
    print(f"Base model: {base_model_path}")
    print(f"LoRA checkpoint: {checkpoint_path}")
    print(f"Eval file: {eval_file}")
    for key in (
        "nonempty_rate",
        "parse_rate",
        "sanitize_rate",
        "substructure_rate",
        "connected_rate",
        "dummy_output_rate",
        "full_parent_rate",
        "too_small_rate",
        "avg_atom_count",
        "avg_atom_ratio",
        "raw_parse_rate",
        "stripped_core_parse_rate",
        "stripped_core_substructure_rate",
    ):
        print(f"{key}: {summary[key]}")
    print(f"Summary JSON: {output_dir / 'summary.json'}")
    print(f"Details JSONL: {output_dir / 'details.jsonl'}")
    print(f"Bad cases: {output_dir}")


def _row_to_record(row: dict[str, Any], *, index: int) -> dict[str, Any] | None:
    parent_smiles = str(row.get("smiles") or row.get("parent_smiles") or "").strip()
    if not parent_smiles:
        return None
    prompt = str(row.get("prompt") or row.get("instruction") or "").strip()
    if not prompt:
        prompt = build_counterfactual_prompt(
            MoleculeRecord(
                record_id=row.get("id", index),
                smiles=parent_smiles,
                label=int(row.get("label", 1)),
            ),
            include_label=False,
        )
    reference_fragment = str(row.get("response") or row.get("output") or "").strip()
    return {
        "id": row.get("id", index),
        "graph_id": row.get("graph_id", row.get("id", index)),
        "label": int(row.get("label", 1)),
        "parent_smiles": parent_smiles,
        "prompt": prompt,
        "reference_fragment": reference_fragment,
    }


def _evaluate_fragment_output(
    *,
    parent_smiles: str,
    raw_output: str,
    too_small_atoms: int,
) -> dict[str, Any]:
    normalized = normalize_core_fragment(raw_output, keep_largest_component=True)
    match = (
        match_core_fragment_to_parent(parent_smiles, normalized.core_fragment_smiles)
        if normalized.core_parse_ok and normalized.core_fragment_smiles
        else None
    )
    raw_has_dummy = bool(normalized.raw_has_dummy)
    strict_parse_ok = bool(normalized.core_parse_ok and not raw_has_dummy)
    strict_substructure_ok = bool(match is not None and match.matched and not raw_has_dummy)
    atom_ratio = match.atom_ratio if match is not None else None
    full_parent = bool(match is not None and match.full_parent)
    too_small = bool(
        normalized.core_parse_ok and normalized.core_atom_count <= int(too_small_atoms)
    )

    return {
        "nonempty": bool(str(raw_output or "").strip()),
        "raw_output": str(raw_output or "").strip(),
        "raw_parse_ok": bool(normalized.raw_parse_ok),
        "raw_sanitize_ok": bool(normalized.raw_sanitize_ok),
        "raw_has_dummy": raw_has_dummy,
        "raw_dummy_count": int(normalized.raw_dummy_count),
        "core_fragment": normalized.core_fragment_smiles,
        "parse_ok": strict_parse_ok,
        "sanitize_ok": strict_parse_ok,
        "connected": bool(normalized.core_connected and not raw_has_dummy),
        "substructure_ok": strict_substructure_ok,
        "stripped_core_parse_ok": bool(normalized.core_parse_ok),
        "stripped_core_substructure_ok": bool(match is not None and match.matched),
        "full_parent": full_parent,
        "too_small": too_small,
        "atom_count": int(normalized.core_atom_count),
        "atom_ratio": atom_ratio,
        "parent_atom_indices": list(match.match_atom_indices) if match is not None else [],
        "boundary_bonds": match.boundary_bonds_as_dicts() if match is not None else [],
        "explanation_fragment_with_dummy": (
            match.explanation_fragment_with_dummy if match is not None else None
        ),
        "failure_tag": (
            normalized.failure_tag
            if normalized.failure_tag
            else (
                "dummy_output_in_core_mode"
                if raw_has_dummy
                else (
                    "not_substructure"
                    if not strict_substructure_ok
                    else ("full_parent" if full_parent else ("too_small" if too_small else None))
                )
            )
        ),
    }


def _build_summary(rows: list[dict[str, Any]], *, too_small_atoms: int) -> dict[str, Any]:
    total = max(1, len(rows))
    atom_counts = [int(row["atom_count"]) for row in rows if int(row["atom_count"]) > 0]
    atom_ratios = [
        float(row["atom_ratio"])
        for row in rows
        if row.get("atom_ratio") is not None
    ]
    return {
        "total_examples": len(rows),
        "too_small_atoms_threshold": int(too_small_atoms),
        "nonempty_rate": sum(bool(row["nonempty"]) for row in rows) / total,
        "parse_rate": sum(bool(row["parse_ok"]) for row in rows) / total,
        "sanitize_rate": sum(bool(row["sanitize_ok"]) for row in rows) / total,
        "substructure_rate": sum(bool(row["substructure_ok"]) for row in rows) / total,
        "connected_rate": sum(bool(row["connected"]) for row in rows) / total,
        "dummy_output_count": sum(bool(row["raw_has_dummy"]) for row in rows),
        "dummy_output_rate": sum(bool(row["raw_has_dummy"]) for row in rows) / total,
        "full_parent_rate": sum(bool(row["full_parent"]) for row in rows) / total,
        "too_small_rate": sum(bool(row["too_small"]) for row in rows) / total,
        "avg_atom_count": (sum(atom_counts) / len(atom_counts)) if atom_counts else 0.0,
        "avg_atom_ratio": (sum(atom_ratios) / len(atom_ratios)) if atom_ratios else 0.0,
        "raw_parse_rate": sum(bool(row["raw_parse_ok"]) for row in rows) / total,
        "stripped_core_parse_rate": sum(bool(row["stripped_core_parse_ok"]) for row in rows) / total,
        "stripped_core_substructure_rate": sum(
            bool(row["stripped_core_substructure_ok"]) for row in rows
        ) / total,
    }


def _write_bad_cases(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    bad_case_groups = {
        "parse_failed": [
            row for row in rows if not bool(row["stripped_core_parse_ok"])
        ],
        "dummy_output": [
            row for row in rows if bool(row["raw_has_dummy"])
        ],
        "parse_ok_but_not_substructure": [
            row
            for row in rows
            if bool(row["stripped_core_parse_ok"])
            and not bool(row["stripped_core_substructure_ok"])
        ],
        "full_parent": [
            row for row in rows if bool(row["full_parent"])
        ],
        "too_small": [
            row for row in rows if bool(row["too_small"])
        ],
    }
    for name, bad_rows in bad_case_groups.items():
        write_jsonl(output_dir / f"bad_cases_{name}.jsonl", bad_rows)


def _render_report(summary: dict[str, Any]) -> str:
    ordered_keys = [
        "nonempty_rate",
        "parse_rate",
        "sanitize_rate",
        "substructure_rate",
        "connected_rate",
        "dummy_output_rate",
        "full_parent_rate",
        "too_small_rate",
        "avg_atom_count",
        "avg_atom_ratio",
        "raw_parse_rate",
        "stripped_core_parse_rate",
        "stripped_core_substructure_rate",
    ]
    lines = [
        "SFT Core-Only Fragment Quality Evaluation",
        "=========================================",
    ]
    for key in ordered_keys:
        lines.append(f"{key}: {summary.get(key)}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
