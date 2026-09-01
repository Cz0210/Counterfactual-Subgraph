"""Typed contracts for the BACE LLM-proposer ablation.

This module intentionally validates provenance only.  It never imports a
transformer model, classifier, selector, evaluator, or held-out split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LLMAblationContractError(ValueError):
    """Raised when an LLM-ablation input is ambiguous or scientifically unsafe."""


class LLMProposerVariant(str, Enum):
    """The four attempt-matched proposer variants."""

    BRICS_FIXED = "BRICS_FIXED"
    CHEMLLM_PRETRAINED = "CHEMLLM_PRETRAINED"
    CHEMLLM_SFT = "CHEMLLM_SFT"
    CHEMLLM_SFT_PPO = "CHEMLLM_SFT_PPO"


AVAILABLE = "AVAILABLE"
BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT = "BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT"


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Return a stable digest for a JSON object."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_sha256(value: object, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise LLMAblationContractError(f"{field} must be one lowercase SHA256 digest")
    return normalized


def artifact_sha256(path: str | Path) -> str:
    """Recompute one deterministic file or directory identity.

    Files use their ordinary byte SHA256.  Directories reuse the exact
    content-bound ``hash_path_inventory`` algorithm that produced the BACE
    main-policy ``source_model_hash``.  Symlinks and special files are rejected
    first so that the identity cannot depend on the traversal environment.
    """

    resolved = Path(path).resolve(strict=True)
    if resolved.is_file():
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not resolved.is_dir():
        raise LLMAblationContractError(
            f"artifact path must be a regular file or directory: {resolved}"
        )

    for entry in sorted(
        resolved.rglob("*"),
        key=lambda item: item.relative_to(resolved).as_posix(),
    ):
        if entry.is_symlink():
            raise LLMAblationContractError(
                f"directory artifact may not contain symlinks: {entry}"
            )
        if not entry.is_dir() and not entry.is_file():
            raise LLMAblationContractError(
                f"directory artifact contains a special file: {entry}"
            )
    from src.train.bace_policy_init import hash_path_inventory

    return hash_path_inventory(resolved)


def _resolve_existing_path(value: object, *, field: str) -> tuple[str, str]:
    raw = str(value or "").strip()
    if not raw:
        raise LLMAblationContractError(f"{field} is required")
    path = Path(raw)
    if not path.is_absolute():
        raise LLMAblationContractError(f"{field} must be an exact absolute path")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise LLMAblationContractError(f"{field} does not exist: {path}") from exc
    return str(path), str(resolved)


@dataclass(frozen=True, slots=True)
class ArtifactPin:
    """One exact local path plus a precomputed artifact identity."""

    path: str
    sha256: str
    role: str
    resolved_path: str = ""

    def __post_init__(self) -> None:
        if not str(self.role).strip():
            raise LLMAblationContractError("artifact role must be non-empty")
        lexical, resolved = _resolve_existing_path(self.path, field=f"{self.role}.path")
        object.__setattr__(self, "path", lexical)
        object.__setattr__(self, "resolved_path", resolved)
        object.__setattr__(
            self,
            "sha256",
            require_sha256(self.sha256, field=f"{self.role}.sha256"),
        )
        actual_sha256 = artifact_sha256(resolved)
        if self.sha256 != actual_sha256:
            raise LLMAblationContractError(
                f"{self.role}.sha256 mismatch: configured={self.sha256}, "
                f"actual={actual_sha256}"
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, role: str) -> "ArtifactPin":
        if not isinstance(payload, Mapping):
            raise LLMAblationContractError(f"{role} must be an object")
        if set(payload) != {"path", "sha256"}:
            raise LLMAblationContractError(f"{role} must contain only path and sha256")
        return cls(
            path=str(payload.get("path") or ""),
            sha256=str(payload.get("sha256") or ""),
            role=role,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GeneratorAssets:
    """Exact model lineage for one proposer without loading model weights.

    ``CHEMLLM_SFT_PPO`` records both the SFT lineage and the final PPO adapter.
    The final load adapter is the PPO path; the SFT path is mandatory provenance.
    """

    variant: LLMProposerVariant
    base_model: ArtifactPin | None = None
    tokenizer: ArtifactPin | None = None
    sft_adapter: ArtifactPin | None = None
    ppo_adapter: ArtifactPin | None = None
    availability: str = AVAILABLE
    blocker_reason: str | None = None
    schema_version: str = "llm_generator_assets_v1"

    def __post_init__(self) -> None:
        variant = self.variant
        if not isinstance(variant, LLMProposerVariant):
            variant = LLMProposerVariant(str(variant))
            object.__setattr__(self, "variant", variant)

        availability = str(self.availability)
        if availability == BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT:
            if variant not in (
                LLMProposerVariant.CHEMLLM_SFT,
                LLMProposerVariant.CHEMLLM_SFT_PPO,
            ):
                raise LLMAblationContractError(
                    "matched-SFT blocker is valid only for SFT/SFT_PPO"
                )
            if any(
                pin is not None
                for pin in (
                    self.base_model,
                    self.tokenizer,
                    self.sft_adapter,
                    self.ppo_adapter,
                )
            ):
                raise LLMAblationContractError(
                    "matched-SFT-blocked variant may not bind model or adapter assets"
                )
            if self.blocker_reason != BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT:
                raise LLMAblationContractError(
                    "blocked variant must repeat the exact blocker_reason"
                )
            return
        if availability != AVAILABLE or self.blocker_reason is not None:
            raise LLMAblationContractError("available variant has invalid availability metadata")

        if variant is LLMProposerVariant.BRICS_FIXED:
            expected = (False, False, False, False)
        elif variant is LLMProposerVariant.CHEMLLM_PRETRAINED:
            expected = (True, True, False, False)
        elif variant is LLMProposerVariant.CHEMLLM_SFT:
            expected = (True, True, True, False)
        else:
            expected = (True, True, True, True)
        actual = (
            self.base_model is not None,
            self.tokenizer is not None,
            self.sft_adapter is not None,
            self.ppo_adapter is not None,
        )
        if actual != expected:
            raise LLMAblationContractError(
                f"{variant.value} requires base/tokenizer/sft/ppo presence "
                f"{expected}, got {actual}"
            )

        pins = [
            pin
            for pin in (
                self.base_model,
                self.tokenizer,
                self.sft_adapter,
                self.ppo_adapter,
            )
            if pin
        ]
        resolved = [pin.resolved_path for pin in pins]
        if len(resolved) != len(set(resolved)):
            raise LLMAblationContractError(
                f"{variant.value} base/SFT/PPO paths must be physically distinct"
            )

    @property
    def load_adapter(self) -> ArtifactPin | None:
        if self.availability != AVAILABLE:
            return None
        if self.variant is LLMProposerVariant.CHEMLLM_SFT:
            return self.sft_adapter
        if self.variant is LLMProposerVariant.CHEMLLM_SFT_PPO:
            return self.ppo_adapter
        return None

    @classmethod
    def from_mapping(
        cls,
        variant: LLMProposerVariant | str,
        payload: Mapping[str, Any],
    ) -> "GeneratorAssets":
        parsed_variant = (
            variant if isinstance(variant, LLMProposerVariant) else LLMProposerVariant(str(variant))
        )
        if not isinstance(payload, Mapping):
            raise LLMAblationContractError(f"assets for {parsed_variant.value} must be an object")
        expected_keys = {
            "availability",
            "blocker_reason",
            "base_model",
            "tokenizer",
            "sft_adapter",
            "ppo_adapter",
        }
        if set(payload) != expected_keys:
            raise LLMAblationContractError(
                f"assets for {parsed_variant.value} must use exact keys {sorted(expected_keys)}"
            )

        def pin(name: str) -> ArtifactPin | None:
            value = payload.get(name)
            if value is None:
                return None
            return ArtifactPin.from_mapping(value, role=f"{parsed_variant.value}.{name}")

        return cls(
            variant=parsed_variant,
            base_model=pin("base_model"),
            tokenizer=pin("tokenizer"),
            sft_adapter=pin("sft_adapter"),
            ppo_adapter=pin("ppo_adapter"),
            availability=str(payload.get("availability") or ""),
            blocker_reason=(
                str(payload["blocker_reason"])
                if payload.get("blocker_reason") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "variant": self.variant.value,
            "base_model": self.base_model.to_dict() if self.base_model else None,
            "tokenizer": self.tokenizer.to_dict() if self.tokenizer else None,
            "sft_adapter": self.sft_adapter.to_dict() if self.sft_adapter else None,
            "ppo_adapter": self.ppo_adapter.to_dict() if self.ppo_adapter else None,
            "availability": self.availability,
            "blocker_reason": self.blocker_reason,
            "load_adapter_role": self.load_adapter.role if self.load_adapter else None,
        }
        payload["assets_sha256"] = canonical_json_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class ProposalRequest:
    """One deterministic attempt in the shared proposer budget."""

    parent_id: str
    parent_smiles: str
    source_label: int
    regime: str
    regime_attempt_index: int
    attempt_index: int
    attempt_seed: int
    slot_id: str
    attempt_budget_sha256: str
    max_new_tokens: int
    temperature: float
    top_p: float
    prompt: str | None = None

    def __post_init__(self) -> None:
        if not str(self.parent_id).strip() or not str(self.parent_smiles).strip():
            raise LLMAblationContractError("parent_id and parent_smiles must be non-empty")
        if isinstance(self.source_label, bool) or not isinstance(self.source_label, int):
            raise LLMAblationContractError("source_label must be an integer, not bool")
        if not str(self.regime).strip():
            raise LLMAblationContractError("regime must be non-empty")
        if (
            isinstance(self.regime_attempt_index, bool)
            or int(self.regime_attempt_index) < 0
        ):
            raise LLMAblationContractError(
                "regime_attempt_index must be a non-negative integer"
            )
        if isinstance(self.attempt_index, bool) or int(self.attempt_index) < 0:
            raise LLMAblationContractError("attempt_index must be a non-negative integer")
        if isinstance(self.attempt_seed, bool) or int(self.attempt_seed) < 0:
            raise LLMAblationContractError("attempt_seed must be a non-negative integer")
        normalized_slot_id = require_sha256(self.slot_id, field="slot_id")
        expected_slot_id = canonical_json_sha256(
            {
                "parent_id": self.parent_id,
                "regime": self.regime,
                "regime_attempt_index": self.regime_attempt_index,
                "attempt_index": self.attempt_index,
                "attempt_seed": self.attempt_seed,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
        )
        if normalized_slot_id != expected_slot_id:
            raise LLMAblationContractError("slot_id does not match proposal request fields")
        object.__setattr__(self, "slot_id", normalized_slot_id)
        object.__setattr__(
            self,
            "attempt_budget_sha256",
            require_sha256(
                self.attempt_budget_sha256,
                field="attempt_budget_sha256",
            ),
        )
        if int(self.max_new_tokens) <= 0:
            raise LLMAblationContractError("max_new_tokens must be positive")
        if float(self.temperature) < 0.0:
            raise LLMAblationContractError("temperature must be non-negative")
        if not 0.0 < float(self.top_p) <= 1.0:
            raise LLMAblationContractError("top_p must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class ProposalResult:
    """Generator-neutral result before GINE, selector, or evaluator science."""

    variant: LLMProposerVariant
    fragment_smiles: str
    raw_text: str
    finish_reason: str | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.variant, LLMProposerVariant):
            object.__setattr__(self, "variant", LLMProposerVariant(str(self.variant)))
        object.__setattr__(self, "metadata", dict(self.metadata))


__all__ = [
    "ArtifactPin",
    "AVAILABLE",
    "BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT",
    "GeneratorAssets",
    "LLMAblationContractError",
    "LLMProposerVariant",
    "ProposalRequest",
    "ProposalResult",
    "artifact_sha256",
    "canonical_json_sha256",
    "require_sha256",
]
