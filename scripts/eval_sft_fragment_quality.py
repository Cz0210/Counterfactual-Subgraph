#!/usr/bin/env python3
"""Run SFT inference on a fixed eval set and score fragment-quality metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

from src.eval.fragment_quality import (
    FragmentSourceRecord,
    LEGACY_SFT_PROMPT_TEMPLATE,
    audit_fragment_records,
    build_detail_rows,
    build_summary_payload,
    format_role_summary_lines,
    load_fragment_source_records,
    write_detail_csv,
)


DEFAULT_BASE_MODEL = REPO_ROOT / "pretrained_models" / "ChemLLM-7B-Chat"
DEFAULT_LORA_ROOT = REPO_ROOT / "outputs" / "hpc" / "sft_checkpoints"
DEFAULT_EVAL_FILE = REPO_ROOT / "data" / "sft_val.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "hpc" / "eval" / "sft_fragment_quality"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Accepted for HPC wrapper compatibility. The current evaluator uses explicit CLI overrides only.",
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
        help="Evaluation jsonl path. The script auto-adapts the current repository SFT row format.",
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
        "--near-parent-threshold",
        type=float,
        default=0.8,
        help="Atom-ratio threshold used for near_parent_rate.",
    )
    parser.add_argument(
        "--tiny-fragment-threshold",
        type=float,
        default=0.08,
        help="Atom-ratio threshold used for tiny_fragment_rate.",
    )
    parser.add_argument(
        "--mid-size-min",
        type=float,
        default=0.1,
        help="Lower atom-ratio bound used for mid_size_rate.",
    )
    parser.add_argument(
        "--mid-size-max",
        type=float,
        default=0.6,
        help="Upper atom-ratio bound used for mid_size_rate.",
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
    summary_json = output_dir / "summary.json"
    details_csv = output_dir / "details.csv"
    report_txt = output_dir / "report.txt"

    source_records = load_fragment_source_records([eval_file])
    if args.max_examples and args.max_examples > 0:
        source_records = source_records[: args.max_examples]
    if not source_records:
        raise SystemExit(f"No eval records were loaded from {eval_file}.")

    tokenizer = build_tokenizer(base_model_path)
    model = build_lora_model(base_model_path, checkpoint_path)

    generated_records: list[FragmentSourceRecord] = []
    iterator = tqdm(source_records, total=len(source_records), desc="Evaluating SFT fragment quality")
    for index, source_record in enumerate(iterator):
        parent_smiles = str(source_record.parent_smiles or "").strip()
        if not parent_smiles:
            generated_records.append(
                FragmentSourceRecord(
                    sample_id=source_record.sample_id,
                    source_path=source_record.source_path,
                    source_kind="sft_eval_missing_parent",
                    parent_smiles=None,
                    instruction=source_record.instruction,
                    prompt=source_record.prompt,
                    label=source_record.label,
                    reference_fragment=source_record.reference_fragment,
                    generated_fragment=None,
                    raw_generation=None,
                    error="missing_parent_smiles",
                    raw_payload=source_record.raw_payload,
                )
            )
            continue

        prompt = (
            source_record.prompt
            or source_record.instruction
            or LEGACY_SFT_PROMPT_TEMPLATE.format(parent_smiles=parent_smiles)
        )
        inference_example = InferenceExample(
            index=index,
            parent_smiles=parent_smiles,
            prompt=prompt,
            reference_fragment=str(source_record.reference_fragment or ""),
        )
        result = generate_one(model, tokenizer, inference_example)
        generated_records.append(
            FragmentSourceRecord(
                sample_id=source_record.sample_id,
                source_path=source_record.source_path,
                source_kind="sft_eval_generated",
                parent_smiles=parent_smiles,
                instruction=source_record.instruction,
                prompt=prompt,
                label=source_record.label,
                reference_fragment=source_record.reference_fragment,
                generated_fragment=result.generated_fragment,
                raw_generation=result.raw_generation,
                error=result.error,
                raw_payload=source_record.raw_payload,
            )
        )

    audited_records = audit_fragment_records(generated_records)
    summary = build_summary_payload(
        audited_records,
        input_paths=[eval_file],
        near_parent_threshold=args.near_parent_threshold,
        tiny_fragment_threshold=args.tiny_fragment_threshold,
        mid_size_min=args.mid_size_min,
        mid_size_max=args.mid_size_max,
    )
    summary["base_model"] = str(base_model_path)
    summary["checkpoint_path"] = str(checkpoint_path)
    summary["eval_file"] = str(eval_file)

    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    detail_rows = build_detail_rows(audited_records)
    write_detail_csv(details_csv, detail_rows)
    report_txt.write_text(
        _render_report(
            base_model_path=base_model_path,
            checkpoint_path=checkpoint_path,
            eval_file=eval_file,
            summary=summary,
            audited_records=audited_records,
        ),
        encoding="utf-8",
    )

    print("SFT fragment-quality evaluation completed.")
    print(f"Base model: {base_model_path}")
    print(f"LoRA checkpoint: {checkpoint_path}")
    print(f"Eval file: {eval_file}")
    for role_name in ("reference", "generated"):
        for line in format_role_summary_lines(summary[role_name]):
            print(line)
    print(f"Summary JSON: {summary_json}")
    print(f"Detail CSV: {details_csv}")
    print(f"Text report: {report_txt}")


def _render_report(
    *,
    base_model_path: Path,
    checkpoint_path: Path,
    eval_file: Path,
    summary: dict[str, object],
    audited_records,
) -> str:
    lines = [
        "SFT Fragment Quality Evaluation",
        "===============================",
        f"Base model: {base_model_path}",
        f"LoRA checkpoint: {checkpoint_path}",
        f"Eval file: {eval_file}",
        f"Total evaluated: {summary['total_records']}",
        "",
    ]
    for role_name in ("reference", "generated"):
        for line in format_role_summary_lines(summary[role_name]):
            lines.append(line)
        lines.append("")

    for audited in audited_records:
        generated = audited.generated
        lines.extend(
            [
                f"[Sample {audited.record.sample_id}]",
                f"Parent SMILES: {audited.record.parent_smiles}",
                f"Prompt: {audited.record.prompt or audited.record.instruction or ''}",
                f"Reference Fragment: {audited.record.reference_fragment or ''}",
                f"Raw Generation: {audited.record.raw_generation or ''}",
                f"Generated Fragment: {audited.record.generated_fragment or ''}",
                f"Generated Core: {generated.core_fragment if generated else ''}",
                f"Generated atom_ratio: {generated.atom_ratio if generated else ''}",
                f"Generated valid: {generated.chemically_valid if generated else False}",
                f"Generated substructure: {generated.substructure_ok if generated else False}",
                f"Generated residual_nonempty: {generated.residual_nonempty if generated else False}",
                f"Error: {audited.record.error or ''}",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
