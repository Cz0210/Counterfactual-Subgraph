"""Helpers for loading locally stored Hugging Face model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class LocalArtifacts:
    """Loaded model and tokenizer objects together with their local paths."""

    model: Any
    tokenizer: Any
    model_path: Path
    tokenizer_path: Path


def resolve_local_artifact_paths(
    model_path: str | Path,
    tokenizer_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve local model/tokenizer paths and ensure they exist."""

    resolved_model_path = Path(model_path).expanduser().resolve()
    resolved_tokenizer_path = (
        Path(tokenizer_path).expanduser().resolve()
        if tokenizer_path is not None
        else resolved_model_path
    )

    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Local model path does not exist: {resolved_model_path}")
    if not resolved_tokenizer_path.exists():
        raise FileNotFoundError(
            f"Local tokenizer path does not exist: {resolved_tokenizer_path}"
        )

    return resolved_model_path, resolved_tokenizer_path


def load_local_hf_artifacts(
    model_path: str | Path,
    tokenizer_path: str | Path | None = None,
    *,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
) -> LocalArtifacts:
    """Load a model and tokenizer from local filesystem paths only."""

    resolved_model_path, resolved_tokenizer_path = resolve_local_artifact_paths(
        model_path,
        tokenizer_path,
    )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "transformers is required to load local model artifacts."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        str(resolved_tokenizer_path),
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(resolved_model_path),
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    return LocalArtifacts(
        model=model,
        tokenizer=tokenizer,
        model_path=resolved_model_path,
        tokenizer_path=resolved_tokenizer_path,
    )
