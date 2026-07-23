#!/usr/bin/env python3
"""Generate and audit Mutagenicity validation fragments from one SFT checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.mutagenicity_continued_sft import (  # noqa: E402
    EXPECTED_VAL_ROWS,
    GENERATION_FIELDS,
    deterministic_smoke_sample,
    load_continued_sft_records,
    score_generated_fragment,
    validate_peft_checkpoint,
    write_csv_atomic,
    write_json_atomic,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-model-path",
        type=Path,
        default=Path("pretrained_models/ChemLLM-7B-Chat"),
    )
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_VAL_ROWS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser


def _resolve(path: Path) -> Path:
    expanded = path.expanduser()
    return (REPO_ROOT / expanded).resolve() if not expanded.is_absolute() else expanded.resolve()


def generate_checkpoint_samples(
    *,
    model: Any,
    tokenizer: Any,
    records: Sequence[Any],
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - HPC runtime dependency
        raise RuntimeError("SFT checkpoint evaluation requires PyTorch") from exc

    rows: list[dict[str, Any]] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    tokenizer.padding_side = "left"
    for record in records:
        encoded = tokenizer(record.prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_length = int(encoded["input_ids"].shape[-1])
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
        generated_ids = generated[0][prompt_length:]
        generated_text = tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        rows.append(
            score_generated_fragment(
                record,
                generated_text=generated_text,
                generation_length=int(generated_ids.shape[-1]),
            )
        )
    total = len(rows)
    summary = {
        "num_samples": total,
        "parse_rate": (
            sum(bool(row["parse_ok"]) for row in rows) / total if total else 0.0
        ),
        "contains_dummy_rate": (
            sum(bool(row["contains_dummy"]) for row in rows) / total if total else 0.0
        ),
        "empty_output_rate": (
            sum(bool(row["empty_output"]) for row in rows) / total if total else 0.0
        ),
        "exact_match_rate": (
            sum(bool(row["exact_match"]) for row in rows) / total if total else 0.0
        ),
        "mean_generation_length": (
            sum(int(row["generation_length"]) for row in rows) / total if total else 0.0
        ),
    }
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = _resolve(args.checkpoint)
    data_csv = _resolve(args.data_csv)
    output_dir = _resolve(args.output_dir)
    base_model_path = _resolve(args.base_model_path)
    tokenizer_path = _resolve(args.tokenizer_path or base_model_path)
    checkpoint_audit = validate_peft_checkpoint(checkpoint)
    if not base_model_path.is_dir():
        raise FileNotFoundError(f"ChemLLM base model does not exist: {base_model_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Evaluation output directory is non-empty and will not be overwritten: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    records_all = load_continued_sft_records(
        data_csv,
        expected_split="val",
        expected_count=args.expected_rows if args.expected_rows > 0 else None,
    )
    sample_count = len(records_all) if args.max_samples <= 0 else args.max_samples
    records = deterministic_smoke_sample(
        records_all,
        max_rows=min(sample_count, len(records_all)),
        seed=args.seed,
    )
    try:
        from scripts.run_infer_sft import build_lora_model, build_tokenizer
    except ImportError as exc:  # pragma: no cover - HPC runtime dependency
        raise RuntimeError(
            "Checkpoint evaluation requires torch, transformers, peft, and RDKit "
            "inside smiles_pip118"
        ) from exc

    print("[MUTAGENICITY_SFT_CHECKPOINT_EVAL_CONFIG]")
    print(f"checkpoint={checkpoint}")
    print(f"base_model_path={base_model_path}")
    print(f"tokenizer_path={tokenizer_path}")
    print(f"data_csv={data_csv}")
    print(f"num_rows_full={len(records_all)}")
    print(f"num_rows_selected={len(records)}")
    print(f"output_dir={output_dir}")
    tokenizer = build_tokenizer(tokenizer_path)
    model = build_lora_model(base_model_path, checkpoint)
    rows, metrics = generate_checkpoint_samples(
        model=model,
        tokenizer=tokenizer,
        records=records,
        max_new_tokens=args.max_new_tokens,
    )
    write_csv_atomic(output_dir / "generation_samples.csv", rows, GENERATION_FIELDS)
    summary = {
        "dataset": "Mutagenicity",
        "split": "val",
        "source_label": 1,
        "target_label": 0,
        "checkpoint": str(checkpoint),
        "checkpoint_audit": checkpoint_audit,
        "base_model_path": str(base_model_path),
        "tokenizer_path": str(tokenizer_path),
        "data_csv": str(data_csv),
        "seed": int(args.seed),
        "max_new_tokens": int(args.max_new_tokens),
        **metrics,
    }
    write_json_atomic(output_dir / "eval_summary.json", summary)
    report = [
        "# Mutagenicity SFT Checkpoint Evaluation",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Validation CSV: `{data_csv}`",
        f"- Samples: {metrics['num_samples']}",
        f"- Parse rate: {metrics['parse_rate']:.6f}",
        f"- Dummy rate: {metrics['contains_dummy_rate']:.6f}",
        f"- Empty rate: {metrics['empty_output_rate']:.6f}",
        f"- Exact-match rate: {metrics['exact_match_rate']:.6f}",
        "",
    ]
    (output_dir / "evaluation_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print("[MUTAGENICITY_SFT_CHECKPOINT_EVAL_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
