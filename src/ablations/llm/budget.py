"""Attempt-matched proposal budgets for the LLM ablation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    LLMAblationContractError,
    LLMProposerVariant,
    ProposalRequest,
    canonical_json_sha256,
)


@dataclass(frozen=True, slots=True)
class ParentInput:
    parent_id: str
    parent_smiles: str
    source_label: int

    def __post_init__(self) -> None:
        if not str(self.parent_id).strip() or not str(self.parent_smiles).strip():
            raise LLMAblationContractError("parent input ID/SMILES must be non-empty")
        if isinstance(self.source_label, bool) or not isinstance(self.source_label, int):
            raise LLMAblationContractError("source_label must be an integer")


@dataclass(frozen=True, slots=True)
class AttemptRegime:
    """One frozen main-contract decoding regime."""

    name: str
    attempts_per_parent: int
    seed: int
    temperature: float
    top_p: float
    max_new_tokens: int

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise LLMAblationContractError("attempt regime name must be non-empty")
        for field in ("attempts_per_parent", "seed", "max_new_tokens"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise LLMAblationContractError(f"{field} must be an integer")
        if self.attempts_per_parent <= 0 or self.seed < 0 or self.max_new_tokens <= 0:
            raise LLMAblationContractError("attempt regime counts/seeds must be valid")
        if float(self.temperature) < 0.0 or not 0.0 < float(self.top_p) <= 1.0:
            raise LLMAblationContractError("invalid attempt regime temperature/top_p")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AttemptRegime":
        expected = {
            "name",
            "attempts_per_parent",
            "seed",
            "temperature",
            "top_p",
            "max_new_tokens",
        }
        if set(payload) != expected:
            raise LLMAblationContractError("attempt regime fields are not exact")
        return cls(**{field: payload[field] for field in expected})


@dataclass(frozen=True, slots=True)
class AttemptSlot:
    parent_id: str
    regime: str
    regime_attempt_index: int
    attempt_index: int
    attempt_seed: int
    max_new_tokens: int
    temperature: float
    top_p: float
    slot_id: str


@dataclass(frozen=True, slots=True)
class MatchedAttemptBudget:
    """One exact schedule reused byte-for-byte by every proposer variant."""

    parents: tuple[ParentInput, ...]
    regimes: tuple[AttemptRegime, ...]
    expected_parent_count: int
    schema_version: str = "llm_attempt_matched_budget_v2"

    def __post_init__(self) -> None:
        if not self.parents:
            raise LLMAblationContractError("attempt budget requires at least one parent")
        ids = [str(parent.parent_id) for parent in self.parents]
        if len(ids) != len(set(ids)):
            raise LLMAblationContractError("attempt budget parent IDs must be unique")
        if isinstance(self.expected_parent_count, bool) or not isinstance(
            self.expected_parent_count, int
        ):
            raise LLMAblationContractError("expected_parent_count must be an integer")
        if len(self.parents) != self.expected_parent_count:
            raise LLMAblationContractError(
                f"expected {self.expected_parent_count} train parents, got {len(self.parents)}"
            )
        if not self.regimes:
            raise LLMAblationContractError("attempt budget requires at least one regime")
        names = [regime.name for regime in self.regimes]
        if len(names) != len(set(names)):
            raise LLMAblationContractError("attempt regime names must be unique")

    @property
    def attempts_per_parent(self) -> int:
        return sum(regime.attempts_per_parent for regime in self.regimes)

    def slots(self) -> tuple[AttemptSlot, ...]:
        slots: list[AttemptSlot] = []
        for parent in self.parents:
            global_attempt_index = 0
            for regime in self.regimes:
                for regime_attempt_index in range(regime.attempts_per_parent):
                    identity = {
                        "parent_id": parent.parent_id,
                        "regime": regime.name,
                        "regime_attempt_index": regime_attempt_index,
                        "attempt_index": global_attempt_index,
                        # The main BACE artifact made one four-sequence call
                        # with the regime seed.  Do not invent per-slot seeds.
                        "attempt_seed": regime.seed,
                        "max_new_tokens": regime.max_new_tokens,
                        "temperature": regime.temperature,
                        "top_p": regime.top_p,
                    }
                    slots.append(AttemptSlot(slot_id=canonical_json_sha256(identity), **identity))
                    global_attempt_index += 1
        return tuple(slots)

    def requests(self) -> tuple[ProposalRequest, ...]:
        by_id = {parent.parent_id: parent for parent in self.parents}
        budget_sha256 = self.to_dict()["budget_sha256"]
        return tuple(
            ProposalRequest(
                parent_id=slot.parent_id,
                parent_smiles=by_id[slot.parent_id].parent_smiles,
                source_label=by_id[slot.parent_id].source_label,
                regime=slot.regime,
                regime_attempt_index=slot.regime_attempt_index,
                attempt_index=slot.attempt_index,
                attempt_seed=slot.attempt_seed,
                slot_id=slot.slot_id,
                attempt_budget_sha256=budget_sha256,
                max_new_tokens=slot.max_new_tokens,
                temperature=slot.temperature,
                top_p=slot.top_p,
            )
            for slot in self.slots()
        )

    def variant_schedules(self) -> dict[str, tuple[AttemptSlot, ...]]:
        slots = self.slots()
        return {variant.value: slots for variant in LLMProposerVariant}

    def to_dict(self) -> dict[str, Any]:
        slots = [asdict(slot) for slot in self.slots()]
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "budget_unit": "proposal_attempt",
            "attempt_matched": True,
            "valid_candidate_matched": False,
            "parent_order": [parent.parent_id for parent in self.parents],
            "parent_count": len(self.parents),
            "expected_parent_count": self.expected_parent_count,
            "attempts_per_parent": self.attempts_per_parent,
            "total_attempts_per_variant": len(slots),
            "regimes": [asdict(regime) for regime in self.regimes],
            "slots": slots,
            "variant_slot_digests": {
                variant.value: canonical_json_sha256({"slots": slots})
                for variant in LLMProposerVariant
            },
        }
        payload["budget_sha256"] = canonical_json_sha256(payload)
        return payload


def validate_attempt_matched_schedules(
    schedules: Mapping[str, Sequence[AttemptSlot | Mapping[str, Any]]],
) -> None:
    """Fail if any variant receives a different attempt set or order."""

    expected_names = {variant.value for variant in LLMProposerVariant}
    if set(schedules) != expected_names:
        raise LLMAblationContractError(
            f"variant schedules must be exactly {sorted(expected_names)}"
        )
    reference: tuple[tuple[Any, ...], ...] | None = None
    fields = tuple(AttemptSlot.__dataclass_fields__)
    for variant in LLMProposerVariant:
        normalized = tuple(
            tuple(
                (asdict(item) if isinstance(item, AttemptSlot) else dict(item)).get(field)
                for field in fields
            )
            for item in schedules[variant.value]
        )
        if reference is None:
            reference = normalized
        elif normalized != reference:
            raise LLMAblationContractError(
                f"{variant.value} proposal attempts do not match the shared schedule"
            )


def parents_from_records(records: Iterable[Mapping[str, Any]]) -> tuple[ParentInput, ...]:
    """Create ordered parent inputs from train rows without reading oracle fields."""

    parents: list[ParentInput] = []
    for row in records:
        raw_label = row.get("label")
        if isinstance(raw_label, bool):
            raise LLMAblationContractError("source label must not be bool")
        parents.append(
            ParentInput(
                parent_id=str(row.get("parent_id") or row.get("molecule_id") or ""),
                parent_smiles=str(row.get("parent_smiles") or row.get("smiles") or ""),
                source_label=int(str(raw_label)),
            )
        )
    return tuple(parents)


__all__ = [
    "AttemptRegime",
    "AttemptSlot",
    "MatchedAttemptBudget",
    "ParentInput",
    "parents_from_records",
    "validate_attempt_matched_schedules",
]
