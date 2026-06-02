#!/usr/bin/env python3
"""Add learned fragment embeddings to a candidate_pool.jsonl file."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.full_candidate_pool import (  # noqa: E402
    _build_base_model,
    _build_lora_model,
    _build_tokenizer,
    _is_no_adapter_path,
    inspect_checkpoint_directory,
    resolve_adapter_load_path,
)


DEFAULT_INPUT_JSONL = (
    REPO_ROOT
    / "outputs"
    / "hpc"
    / "full_candidate_pools"
    / "stable300_label1_merged_base_temp07"
    / "candidate_pool.jsonl"
)
DEFAULT_OUTPUT_JSONL = DEFAULT_INPUT_JSONL.with_name("candidate_pool_with_embeddings.jsonl")
DEFAULT_BASE_MODEL = REPO_ROOT / "pretrained_models" / "ChemLLM-7B-Chat"
DEFAULT_MODEL_PATH = (
    REPO_ROOT
    / "outputs"
    / "hpc"
    / "rl_checkpoints"
    / "decoded_chem_ppo_stable300_sftv3_projcf_dist03_projpen1_orig_shuffle13_ckpt500"
)
SOURCE_FALLBACK_FIELDS = (
    "core_fragment",
    "final_fragment_smiles",
    "candidate_smiles",
    "raw_fragment",
)


@dataclass(frozen=True, slots=True)
class SourceText:
    """Resolved candidate text used for one embedding."""

    text: str
    field_name: str


@dataclass(frozen=True, slots=True)
class PendingRow:
    """One parsed candidate row waiting for batched embedding."""

    row_index: int
    line_number: int
    row: dict[str, Any]
    source: SourceText


@dataclass(frozen=True, slots=True)
class LoadedEmbeddingModel:
    """Loaded tokenizer/model bundle plus checkpoint provenance."""

    tokenizer: Any
    model: Any
    model_device: Any
    load_mode: str
    adapter_path: str | None
    resolved_device: str
    checkpoint_inspection: dict[str, Any] | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config path for Slurm parity. This script uses explicit CLI paths.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Ignored dotted overrides kept for runtime wrapper parity.",
    )
    parser.add_argument("--input-jsonl", default=str(DEFAULT_INPUT_JSONL))
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="SFT/PPO adapter checkpoint path, or NONE/base for base ChemLLM only.",
    )
    parser.add_argument(
        "--base-model-path",
        default=str(DEFAULT_BASE_MODEL),
        help="Local ChemLLM base model path reused by the existing candidate-pool loader.",
    )
    parser.add_argument("--embedding-source", default="final_fragment")
    parser.add_argument("--embedding-field", default="final_fragment_embedding")
    parser.add_argument("--pooling", choices=("mean", "cls"), default="mean")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional summary JSON path. Defaults to embedding_summary.json beside the output JSONL.",
    )
    parser.add_argument(
        "--failed-jsonl",
        default=None,
        help="Optional failed rows JSONL path. Defaults to embedding_failed_rows.jsonl beside the output JSONL.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load Hugging Face artifacts from local files only.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forward trust_remote_code to the existing ChemLLM loader.",
    )
    return parser


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _source_fields(primary_field: str) -> list[str]:
    fields = [str(primary_field)]
    for fallback in SOURCE_FALLBACK_FIELDS:
        if fallback not in fields:
            fields.append(fallback)
    return fields


def resolve_embedding_source(row: dict[str, Any], primary_field: str) -> SourceText | None:
    """Resolve the subgraph SMILES field used for embedding this row."""

    for field_name in _source_fields(primary_field):
        value = _normalize_text(row.get(field_name))
        if value:
            return SourceText(text=value, field_name=field_name)
    return None


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false.")
    return device


def _build_cpu_model(
    *,
    base_model_path: Path,
    adapter_path: Path | None,
    trust_remote_code: bool,
    local_files_only: bool,
) -> Any:
    """CPU fallback for small/debug runs; CUDA uses the existing 4-bit loader."""

    import torch
    from transformers import AutoModelForCausalLM

    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
    else:
        model = base_model
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
    return model.to("cpu")


def load_embedding_model(
    *,
    base_model_path: Path,
    model_path: str | Path,
    device: str,
    trust_remote_code: bool,
    local_files_only: bool,
) -> LoadedEmbeddingModel:
    """Load ChemLLM plus optional SFT/PPO adapter using candidate-pool loader parity."""

    import torch
    from dataclasses import asdict

    resolved_device = _resolve_device(device)
    no_adapter = _is_no_adapter_path(model_path)
    adapter_path: Path | None = None
    inspection_dict: dict[str, Any] | None = None
    load_mode = "base"

    if not no_adapter:
        inspection = inspect_checkpoint_directory(model_path)
        inspection_dict = asdict(inspection)
        adapter_path = resolve_adapter_load_path(model_path)
        load_mode = "adapter"

    tokenizer = _build_tokenizer(
        base_model_path=base_model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    if resolved_device == "cpu":
        model = _build_cpu_model(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        model_device = torch.device("cpu")
    elif load_mode == "base":
        model = _build_base_model(
            base_model_path=base_model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        model_device = next(model.parameters()).device
    else:
        assert adapter_path is not None
        model = _build_lora_model(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        model_device = next(model.parameters()).device

    return LoadedEmbeddingModel(
        tokenizer=tokenizer,
        model=model,
        model_device=model_device,
        load_mode=load_mode,
        adapter_path=str(adapter_path) if adapter_path is not None else None,
        resolved_device=resolved_device,
        checkpoint_inspection=inspection_dict,
    )


def _extract_last_hidden_state(outputs: Any) -> Any:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states:
        return hidden_states[-1]
    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if last_hidden_state is not None:
        return last_hidden_state
    if isinstance(outputs, dict):
        if outputs.get("hidden_states"):
            return outputs["hidden_states"][-1]
        if outputs.get("last_hidden_state") is not None:
            return outputs["last_hidden_state"]
    raise RuntimeError(
        "Model forward output did not include hidden_states or last_hidden_state. "
        "The script calls the model with output_hidden_states=True; check checkpoint compatibility."
    )


def _pool_hidden_states(hidden: Any, attention_mask: Any, pooling: str) -> Any:
    import torch

    if pooling == "cls":
        first_token_indices = attention_mask.to(torch.int64).argmax(dim=1)
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_indices, first_token_indices, :]

    mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype, device=hidden.device)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (hidden * mask).sum(dim=1) / denom


def embed_text_batch(
    texts: list[str],
    *,
    tokenizer: Any,
    model: Any,
    model_device: Any,
    pooling: str,
    max_length: int,
) -> list[list[float]]:
    import torch

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_length),
    )
    encoded = {key: value.to(model_device) for key, value in encoded.items()}
    with torch.inference_mode():
        outputs = model(
            **encoded,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    hidden = _extract_last_hidden_state(outputs)
    pooled = _pool_hidden_states(hidden, encoded["attention_mask"], pooling)
    norms = torch.linalg.vector_norm(pooled, ord=2, dim=1, keepdim=True).clamp(min=1e-12)
    normalized = pooled / norms
    if not torch.all(torch.isfinite(normalized)):
        raise RuntimeError("Non-finite values appeared in normalized embeddings.")
    return normalized.detach().cpu().to(torch.float32).tolist()


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=False) + "\n"


def _write_failed_row(
    handle: Any,
    *,
    pending: PendingRow | None,
    row: dict[str, Any],
    row_index: int,
    line_number: int,
    failure_reason: str,
    source_field: str | None = None,
    source_text: str | None = None,
) -> None:
    payload = {
        "row_index": int(row_index),
        "line_number": int(line_number),
        "failure_reason": failure_reason,
        "source_field": source_field,
        "source_text": source_text,
        "row": row,
    }
    if pending is not None:
        payload["source_field"] = pending.source.field_name
        payload["source_text"] = pending.source.text
    handle.write(_json_dumps(payload))


def main() -> int:
    args = build_parser().parse_args()

    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    output_jsonl = Path(args.output_jsonl).expanduser().resolve()
    base_model_path = Path(args.base_model_path).expanduser().resolve()
    summary_json = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else output_jsonl.parent / "embedding_summary.json"
    )
    failed_jsonl = (
        Path(args.failed_jsonl).expanduser().resolve()
        if args.failed_jsonl
        else output_jsonl.parent / "embedding_failed_rows.jsonl"
    )

    if input_jsonl == output_jsonl:
        raise ValueError("--output-jsonl must differ from --input-jsonl to preserve the original pool.")
    if not input_jsonl.exists():
        raise FileNotFoundError(f"input candidate pool not found: {input_jsonl}")
    if not base_model_path.exists():
        raise FileNotFoundError(f"base model path not found: {base_model_path}")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if int(args.max_length) <= 0:
        raise ValueError("--max-length must be positive.")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    failed_jsonl.parent.mkdir(parents=True, exist_ok=True)

    loaded = load_embedding_model(
        base_model_path=base_model_path,
        model_path=args.model_path,
        device=str(args.device),
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )

    total_rows = 0
    rows_written = 0
    rows_missing_source = 0
    rows_failed_embedding = 0
    rows_embedded = 0
    embedding_dim: int | None = None
    dimension_counter: Counter[int] = Counter()
    source_field_counter: Counter[str] = Counter()
    failed_handle_opened = False

    def ensure_failed_handle() -> Any:
        nonlocal failed_handle_opened
        if not failed_handle_opened:
            failed_handle_opened = True
            return failed_jsonl.open("w", encoding="utf-8")
        return failed_jsonl.open("a", encoding="utf-8")

    def process_pending_batch(
        batch: list[PendingRow],
        *,
        output_handle: Any,
    ) -> None:
        nonlocal rows_written, rows_failed_embedding, rows_embedded, embedding_dim
        if not batch:
            return
        texts = [pending.source.text for pending in batch]
        try:
            embeddings = embed_text_batch(
                texts,
                tokenizer=loaded.tokenizer,
                model=loaded.model,
                model_device=loaded.model_device,
                pooling=str(args.pooling),
                max_length=int(args.max_length),
            )
            per_row_results = list(zip(batch, embeddings, [None] * len(batch)))
        except Exception as batch_exc:
            per_row_results = []
            for pending in batch:
                try:
                    single_embedding = embed_text_batch(
                        [pending.source.text],
                        tokenizer=loaded.tokenizer,
                        model=loaded.model,
                        model_device=loaded.model_device,
                        pooling=str(args.pooling),
                        max_length=int(args.max_length),
                    )[0]
                    per_row_results.append((pending, single_embedding, None))
                except Exception as single_exc:
                    reason = f"batch_error={batch_exc}; single_row_error={single_exc}"
                    per_row_results.append((pending, None, reason))

        for pending, embedding, failure_reason in per_row_results:
            row = dict(pending.row)
            if embedding is None:
                rows_failed_embedding += 1
                with ensure_failed_handle() as failed_handle:
                    _write_failed_row(
                        failed_handle,
                        pending=pending,
                        row=row,
                        row_index=pending.row_index,
                        line_number=pending.line_number,
                        failure_reason=str(failure_reason),
                    )
            else:
                if not embedding or any(not math.isfinite(float(value)) for value in embedding):
                    rows_failed_embedding += 1
                    with ensure_failed_handle() as failed_handle:
                        _write_failed_row(
                            failed_handle,
                            pending=pending,
                            row=row,
                            row_index=pending.row_index,
                            line_number=pending.line_number,
                            failure_reason="embedding was empty or contained NaN/Inf",
                        )
                else:
                    row[str(args.embedding_field)] = [float(value) for value in embedding]
                    rows_embedded += 1
                    embedding_dim = int(len(embedding))
                    dimension_counter[embedding_dim] += 1
                    source_field_counter[pending.source.field_name] += 1
            output_handle.write(_json_dumps(row))
            rows_written += 1

    pending_batch: list[PendingRow] = []
    with input_jsonl.open("r", encoding="utf-8") as input_handle, output_jsonl.open(
        "w",
        encoding="utf-8",
    ) as output_handle:
        for line_number, line in enumerate(input_handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            total_rows += 1
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse JSON at line {line_number}: {input_jsonl}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at line {line_number}: {input_jsonl}")

            source = resolve_embedding_source(row, str(args.embedding_source))
            if source is None:
                rows_missing_source += 1
                with ensure_failed_handle() as failed_handle:
                    _write_failed_row(
                        failed_handle,
                        pending=None,
                        row=row,
                        row_index=total_rows - 1,
                        line_number=line_number,
                        failure_reason=(
                            "missing embedding source field. "
                            f"Tried fields={_source_fields(str(args.embedding_source))}"
                        ),
                    )
                output_handle.write(_json_dumps(row))
                rows_written += 1
                continue

            pending_batch.append(
                PendingRow(
                    row_index=total_rows - 1,
                    line_number=line_number,
                    row=row,
                    source=source,
                )
            )
            if len(pending_batch) >= int(args.batch_size):
                process_pending_batch(pending_batch, output_handle=output_handle)
                pending_batch = []

        process_pending_batch(pending_batch, output_handle=output_handle)

    if not failed_handle_opened and failed_jsonl.exists():
        failed_jsonl.unlink()

    summary = {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "total_rows": int(total_rows),
        "rows_written": int(rows_written),
        "rows_embedded": int(rows_embedded),
        "rows_missing_source": int(rows_missing_source),
        "rows_failed_embedding": int(rows_failed_embedding),
        "embedding_field": str(args.embedding_field),
        "embedding_dim": int(embedding_dim) if embedding_dim is not None else None,
        "embedding_dimension_distribution": {
            str(dim): count for dim, count in sorted(dimension_counter.items())
        },
        "embedding_source": str(args.embedding_source),
        "embedding_source_fallbacks": list(SOURCE_FALLBACK_FIELDS),
        "embedding_source_field_counts": dict(sorted(source_field_counter.items())),
        "pooling": str(args.pooling),
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "model_path": str(Path(args.model_path).expanduser().resolve())
        if not _is_no_adapter_path(args.model_path)
        else str(args.model_path),
        "base_model_path": str(base_model_path),
        "load_mode": loaded.load_mode,
        "adapter_path": loaded.adapter_path,
        "device": loaded.resolved_device,
        "model_device": str(loaded.model_device),
        "summary_json": str(summary_json),
        "failed_rows_jsonl": str(failed_jsonl) if failed_handle_opened else None,
        "checkpoint_inspection": loaded.checkpoint_inspection,
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    if rows_missing_source > 0 or rows_failed_embedding > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
