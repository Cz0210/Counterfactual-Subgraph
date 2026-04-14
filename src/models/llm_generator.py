"""Local-only ChemLLM generation helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.chem import parse_smiles
from src.models.interfaces import FragmentGenerator, GenerationRequest, GenerationResult
from src.models.local_loader import resolve_local_artifact_paths
from src.models.prompt_builder import build_chemllm_messages, build_chemllm_prompt


_SMILES_CANDIDATE_PATTERN = re.compile(r"[\[\]\(\)@+\-#%=\\/.*A-Za-z0-9]+")


@dataclass(frozen=True, slots=True)
class ChemLLMAssets:
    """Resolved local ChemLLM assets used by the generator."""

    model_path: Path
    tokenizer_path: Path
    device: str


def clean_generated_smiles(raw_text: str) -> str:
    """Extract one likely SMILES token from a verbose model response."""

    text = str(raw_text or "").strip()
    if not text:
        return ""

    normalized = (
        text.replace("```smiles", "\n")
        .replace("```", "\n")
        .replace("\r", "\n")
        .strip()
    )

    keyword_patterns = (
        r"FRAGMENT_SMILES\s*[:=]\s*([^\s,;]+)",
        r"SMILES\s*[:=]\s*([^\s,;]+)",
        r"ANSWER\s*[:=]\s*([^\s,;]+)",
    )
    for pattern in keyword_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            candidate = _strip_wrapping_punctuation(match.group(1))
            if candidate:
                return candidate

    for line in normalized.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if ":" in stripped_line:
            _, _, suffix = stripped_line.partition(":")
            stripped_line = suffix.strip() or stripped_line
        candidate = _select_smiles_token(stripped_line)
        if candidate:
            return candidate

    candidate = _select_smiles_token(normalized)
    return candidate or _strip_wrapping_punctuation(normalized.split()[0])


class ChemLLMGenerator(FragmentGenerator):
    """A light local Hugging Face generator for ChemLLM checkpoints."""

    def __init__(
        self,
        *,
        model_name_or_path: str | Path,
        tokenizer_path: str | Path | None = None,
        device: str = "auto",
        local_files_only: bool = True,
        trust_remote_code: bool = True,
        max_new_tokens: int = 64,
        top_p: float = 1.0,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError(
                "ChemLLM inference requires transformers and torch to be installed."
            ) from exc

        resolved_model_path, resolved_tokenizer_path = resolve_local_artifact_paths(
            model_name_or_path,
            tokenizer_path,
        )
        resolved_device = _resolve_device(device, torch)

        tokenizer = AutoTokenizer.from_pretrained(
            str(resolved_tokenizer_path),
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(resolved_model_path),
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        if hasattr(model, "config"):
            model.config.use_cache = False
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            generation_config.use_cache = False
        model.to(resolved_device)
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._assets = ChemLLMAssets(
            model_path=resolved_model_path,
            tokenizer_path=resolved_tokenizer_path,
            device=resolved_device,
        )
        self._default_max_new_tokens = max_new_tokens
        self._default_top_p = top_p

        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @property
    def assets(self) -> ChemLLMAssets:
        return self._assets

    def generate_candidate(self, parent_smiles: str, temperature: float = 0.0) -> str:
        """Generate one capped fragment candidate from a parent SMILES."""

        request = GenerationRequest(
            parent_smiles=str(parent_smiles).strip(),
            prompt=None,
            max_new_tokens=self._default_max_new_tokens,
            temperature=temperature,
            top_p=self._default_top_p,
        )
        return self.generate(request).fragment_smiles

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate one capped fragment candidate with ChemLLM."""

        prompt = request.prompt or self._build_prompt(
            request.parent_smiles,
            label=request.label,
        )
        encoded = self._tokenizer(prompt, return_tensors="pt")
        encoded = {
            key: value.to(self._assets.device)
            for key, value in encoded.items()
        }

        do_sample = request.temperature > 0.0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(request.max_new_tokens or self._default_max_new_tokens),
            "do_sample": do_sample,
            "top_p": float(request.top_p or self._default_top_p),
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            # Disable use_cache to prevent KV cache shape mismatch between
            # transformers 4.5x and legacy modeling_internlm2.py.
            "use_cache": False,
        }
        if do_sample:
            generation_kwargs["temperature"] = float(request.temperature)

        with self._torch.no_grad():
            generated = self._model.generate(**encoded, **generation_kwargs)

        prompt_length = encoded["input_ids"].shape[-1]
        generated_tokens = generated[0][prompt_length:]
        raw_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        fragment_smiles = clean_generated_smiles(raw_text)

        return GenerationResult(
            fragment_smiles=fragment_smiles,
            raw_text=raw_text,
            finish_reason="eos_or_length",
            metadata={
                "device": self._assets.device,
                "model_path": str(self._assets.model_path),
                "tokenizer_path": str(self._assets.tokenizer_path),
                "prompt": prompt,
                "max_new_tokens": generation_kwargs["max_new_tokens"],
                "temperature": float(request.temperature),
                "top_p": float(request.top_p or self._default_top_p),
            },
        )

    def _build_prompt(self, parent_smiles: str, *, label: int | None = None) -> str:
        messages = build_chemllm_messages(parent_smiles, label=label)
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return build_chemllm_prompt(parent_smiles, label=label)


def _resolve_device(device: str, torch: Any) -> str:
    requested = str(device or "auto").strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but torch.cuda.is_available() is false.")
    return requested


def _strip_wrapping_punctuation(text: str) -> str:
    return str(text).strip().strip("`'\".,;")


def _select_smiles_token(text: str) -> str:
    candidates = [
        _strip_wrapping_punctuation(match.group(0))
        for match in _SMILES_CANDIDATE_PATTERN.finditer(text)
    ]
    ranked = sorted(
        (
            candidate
            for candidate in candidates
            if _looks_like_smiles(candidate)
        ),
        key=lambda candidate: (
            0 if _is_parseable_smiles(candidate) else 1,
            -len(candidate),
        ),
    )
    return ranked[0] if ranked else ""


def _looks_like_smiles(candidate: str) -> bool:
    if not candidate:
        return False
    if candidate.lower() in {"the", "smiles", "fragment", "answer", "is"}:
        return False
    if "*" in candidate:
        return True
    special_chars = set("[]=#()/\\%@+-0123456789")
    if any(char in special_chars for char in candidate):
        return True
    return re.fullmatch(r"(Br|Cl|Si|Na|Li|Ca|Mg|[BCNOFPSIKHbcnops])+", candidate) is not None


def _is_parseable_smiles(candidate: str) -> bool:
    parsed = parse_smiles(
        candidate,
        sanitize=True,
        canonicalize=False,
        allow_capped_fragments=True,
    )
    return parsed.sanitized
