"""Unified proposer protocol and adapter for the existing ChemLLM interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from src.models.interfaces import FragmentGenerator

from .contracts import (
    AVAILABLE,
    GeneratorAssets,
    LLMAblationContractError,
    LLMProposerVariant,
    ProposalRequest,
    ProposalResult,
    canonical_json_sha256,
)


class ProposalGenerator(Protocol):
    """One shared surface for BRICS and every ChemLLM policy state."""

    variant: LLMProposerVariant

    def generate(self, request: ProposalRequest) -> ProposalResult:
        """Generate exactly one candidate for one matched attempt."""


@dataclass(frozen=True, slots=True)
class RuntimeGeneratorIdentity:
    """Runtime-attested assets and generation capability.

    Constructing the embedded :class:`GeneratorAssets` recomputes every file
    or directory digest.  A runtime that cannot expose this identity remains
    config-only; caller-supplied labels are not accepted as runtime evidence.
    """

    assets: GeneratorAssets
    generation_api: str
    explicit_seed_supported: bool
    num_return_sequences_supported: bool
    schema_version: str = "llm_ablation_runtime_generator_identity_v1"

    def __post_init__(self) -> None:
        if self.assets.availability != AVAILABLE:
            raise LLMAblationContractError("runtime generator assets must be AVAILABLE")
        if self.assets.variant is LLMProposerVariant.BRICS_FIXED:
            raise LLMAblationContractError("BRICS does not use a ChemLLM runtime identity")
        for field in ("explicit_seed_supported", "num_return_sequences_supported"):
            if not isinstance(getattr(self, field), bool):
                raise LLMAblationContractError(f"{field} must be bool")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "assets": self.assets.to_dict(),
            "generation_api": self.generation_api,
            "explicit_seed_supported": self.explicit_seed_supported,
            "num_return_sequences_supported": self.num_return_sequences_supported,
        }
        payload["runtime_identity_sha256"] = canonical_json_sha256(payload)
        return payload


def _assert_runtime_assets_match(
    declared: GeneratorAssets,
    runtime: GeneratorAssets,
) -> None:
    if runtime.variant is not declared.variant:
        raise LLMAblationContractError(
            f"runtime variant mismatch: {runtime.variant.value} != {declared.variant.value}"
        )
    for field in ("base_model", "tokenizer", "sft_adapter", "ppo_adapter"):
        declared_pin = getattr(declared, field)
        runtime_pin = getattr(runtime, field)
        if (declared_pin is None) != (runtime_pin is None):
            raise LLMAblationContractError(f"runtime {field} topology mismatch")
        if declared_pin is not None and runtime_pin is not None:
            if (
                declared_pin.resolved_path != runtime_pin.resolved_path
                or declared_pin.sha256 != runtime_pin.sha256
            ):
                raise LLMAblationContractError(f"runtime {field} identity mismatch")


class ChemLLMGeneratorAdapter:
    """Adapt the repository's existing ``FragmentGenerator`` without loading it.

    BACE main generated four sequences in one model call for each decoding
    regime.  The repository's ordinary ``FragmentGenerator.generate`` API is
    single-result and has no explicit seed, so this adapter deliberately does
    not emulate the main run with four independent calls.  A runtime must expose
    both ``llm_ablation_runtime_identity`` and ``generate_ablation_regime``;
    older generators remain fail-closed/config-only.
    """

    def __init__(self, *, generator: FragmentGenerator, assets: GeneratorAssets) -> None:
        if assets.variant is LLMProposerVariant.BRICS_FIXED:
            raise LLMAblationContractError("BRICS_FIXED cannot wrap ChemLLM")
        if assets.availability != AVAILABLE:
            raise LLMAblationContractError(
                f"{assets.variant.value} is fail-closed: {assets.blocker_reason}"
            )
        if not callable(getattr(generator, "generate", None)):
            raise LLMAblationContractError(
                "generator must implement src.models.interfaces.FragmentGenerator"
            )
        identity_provider = getattr(generator, "llm_ablation_runtime_identity", None)
        identity = identity_provider() if callable(identity_provider) else identity_provider
        if not isinstance(identity, RuntimeGeneratorIdentity):
            raise LLMAblationContractError(
                "CONFIG_ONLY: generator lacks verifiable llm_ablation_runtime_identity"
            )
        _assert_runtime_assets_match(assets, identity.assets)
        if (
            identity.generation_api != "single_call_num_return_sequences_v1"
            or identity.explicit_seed_supported is not True
            or identity.num_return_sequences_supported is not True
            or not callable(getattr(generator, "generate_ablation_regime", None))
        ):
            raise LLMAblationContractError(
                "CONFIG_ONLY: runtime cannot preserve one-call four-sequence seed semantics"
            )
        self._generator = generator
        self._assets = assets
        self._runtime_identity = identity
        self.variant = assets.variant

    @property
    def assets(self) -> GeneratorAssets:
        return self._assets

    @property
    def runtime_identity(self) -> RuntimeGeneratorIdentity:
        return self._runtime_identity

    def generate(self, request: ProposalRequest) -> ProposalResult:
        raise LLMAblationContractError(
            "ChemLLM ablation requires generate_regime(): four sequences in one seeded call"
        )

    def generate_regime(
        self,
        requests: Sequence[ProposalRequest],
    ) -> tuple[ProposalResult, ...]:
        """Generate one frozen four-slot regime with exactly one runtime call."""

        materialized = tuple(requests)
        if len(materialized) != 4:
            raise LLMAblationContractError("ChemLLM regime must contain exactly four slots")
        first = materialized[0]
        expected_by_regime = {
            "base": (7, 0.3, 0.9, 96),
            "high_temperature": (13, 0.7, 0.9, 96),
        }
        expected = expected_by_regime.get(first.regime)
        if expected is None:
            raise LLMAblationContractError(f"unexpected BACE regime: {first.regime}")
        shared = (
            first.parent_id,
            first.parent_smiles,
            first.source_label,
            first.regime,
            first.attempt_seed,
            first.temperature,
            first.top_p,
            first.max_new_tokens,
            first.attempt_budget_sha256,
            first.prompt,
        )
        for request in materialized:
            current = (
                request.parent_id,
                request.parent_smiles,
                request.source_label,
                request.regime,
                request.attempt_seed,
                request.temperature,
                request.top_p,
                request.max_new_tokens,
                request.attempt_budget_sha256,
                request.prompt,
            )
            if current != shared:
                raise LLMAblationContractError(
                    "ChemLLM regime slots do not share one call contract"
                )
        if tuple(request.regime_attempt_index for request in materialized) != (0, 1, 2, 3):
            raise LLMAblationContractError("ChemLLM regime slot order must be 0,1,2,3")
        expected_attempt_indices = (0, 1, 2, 3) if first.regime == "base" else (4, 5, 6, 7)
        if tuple(request.attempt_index for request in materialized) != expected_attempt_indices:
            raise LLMAblationContractError(
                f"{first.regime} global attempt indices must be {expected_attempt_indices}"
            )
        if len({request.slot_id for request in materialized}) != 4:
            raise LLMAblationContractError("ChemLLM regime slot IDs must be unique")
        actual = (
            first.attempt_seed,
            float(first.temperature),
            float(first.top_p),
            first.max_new_tokens,
        )
        if actual != expected:
            raise LLMAblationContractError(
                f"{first.regime} decoding contract mismatch: {actual} != {expected}"
            )

        generated = tuple(
            self._generator.generate_ablation_regime(
                parent_smiles=first.parent_smiles,
                label=first.source_label,
                prompt=first.prompt,
                num_return_sequences=4,
                seed=first.attempt_seed,
                max_new_tokens=first.max_new_tokens,
                temperature=first.temperature,
                top_p=first.top_p,
            )
        )
        if len(generated) != 4:
            raise LLMAblationContractError(
                f"runtime returned {len(generated)} sequences; exactly four are required"
            )

        wrapped: list[ProposalResult] = []
        for request, result in zip(materialized, generated, strict=True):
            if not all(
                hasattr(result, field)
                for field in ("fragment_smiles", "raw_text", "finish_reason", "metadata")
            ):
                raise LLMAblationContractError("runtime returned a non-GenerationResult value")
            metadata = dict(result.metadata)
            metadata.update(
                {
                    "llm_ablation_variant": self.variant.value,
                    "generator_assets_sha256": self._assets.to_dict()["assets_sha256"],
                    "runtime_generation_api": self._runtime_identity.generation_api,
                    "runtime_identity_sha256": self._runtime_identity.to_dict()[
                        "runtime_identity_sha256"
                    ],
                    "regime": request.regime,
                    "regime_attempt_index": request.regime_attempt_index,
                    "attempt_index": request.attempt_index,
                    "attempt_seed": request.attempt_seed,
                    "slot_id": request.slot_id,
                    "adapter_load_role": (
                        self._assets.load_adapter.role if self._assets.load_adapter else None
                    ),
                }
            )
            wrapped.append(
                ProposalResult(
                    variant=self.variant,
                    fragment_smiles=result.fragment_smiles,
                    raw_text=result.raw_text,
                    finish_reason=result.finish_reason,
                    metadata=metadata,
                )
            )
        return tuple(wrapped)


def assert_generator_variant(
    generator: ProposalGenerator,
    expected_variant: LLMProposerVariant | str,
) -> None:
    """Fail if a proposer is registered under a different policy state."""

    expected = (
        expected_variant
        if isinstance(expected_variant, LLMProposerVariant)
        else LLMProposerVariant(str(expected_variant))
    )
    if not hasattr(generator, "variant") or not callable(getattr(generator, "generate", None)):
        raise LLMAblationContractError("object does not implement ProposalGenerator")
    if generator.variant is not expected:
        raise LLMAblationContractError(
            f"generator variant mismatch: expected {expected.value}, got {generator.variant.value}"
        )


__all__ = [
    "ChemLLMGeneratorAdapter",
    "ProposalGenerator",
    "RuntimeGeneratorIdentity",
    "assert_generator_variant",
]
