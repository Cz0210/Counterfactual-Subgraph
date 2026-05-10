"""Compatibility helpers for SFT datasets consumed by TRL SFTTrainer."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


REQUIRED_SFT_COLUMNS = ("prompt", "completion")
SUPPORTED_SFT_FORMATS_MESSAGE = (
    "Supported SFT formats: "
    "{'prompt': '...', 'completion': '...'}, "
    "{'instruction': '...', 'output': '...', optional 'input': '...'}, "
    "and the legacy builder alias {'prompt': '...', 'response': '...'}."
)


def normalize_completion_text(value: Any) -> str:
    """Ensure the completion is separated from the prompt."""

    completion = str(value)
    if completion and not completion.startswith((" ", "\n")):
        completion = "\n" + completion
    return completion


def normalize_sft_example(example: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of one example with prompt/completion compatibility fields."""

    normalized = dict(example)

    if normalized.get("prompt") is None and normalized.get("instruction") is not None:
        prompt = str(normalized.get("instruction", ""))
        input_text = normalized.get("input")
        if input_text is not None and str(input_text).strip():
            if prompt:
                prompt = f"{prompt}\n{input_text}"
            else:
                prompt = str(input_text)
        normalized["prompt"] = prompt

    if normalized.get("completion") is None:
        for candidate_key in ("output", "response"):
            candidate_value = normalized.get(candidate_key)
            if candidate_value is not None:
                normalized["completion"] = normalize_completion_text(candidate_value)
                break

    return normalized


def build_missing_sft_fields_error(
    *,
    available_columns: Sequence[str],
    missing_fields: Sequence[str],
    split_name: str,
    row_index: int | None = None,
) -> str:
    """Build a clear compatibility error before TRL raises an internal KeyError."""

    location = f"split={split_name}"
    if row_index is not None:
        location = f"{location}, row={row_index}"
    return (
        "SFT dataset is missing required prompt/completion fields after normalization. "
        f"{location}; missing columns={list(missing_fields)}; "
        f"available columns={list(available_columns)}. "
        f"{SUPPORTED_SFT_FORMATS_MESSAGE}"
    )


def validate_required_sft_fields(
    example: Mapping[str, Any],
    *,
    available_columns: Sequence[str],
    split_name: str,
    row_index: int | None = None,
) -> None:
    """Raise a clear ValueError when prompt/completion are still unavailable."""

    missing_fields = [
        column_name for column_name in REQUIRED_SFT_COLUMNS if example.get(column_name) is None
    ]
    if missing_fields:
        raise ValueError(
            build_missing_sft_fields_error(
                available_columns=available_columns,
                missing_fields=missing_fields,
                split_name=split_name,
                row_index=row_index,
            )
        )
