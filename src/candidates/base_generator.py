"""Common, lineage-complete candidate generator contract."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class CandidateRequest:
    parent_id: str
    parent_smiles: str
    parent_split: str
    label: int
    candidates_per_parent: int
    size_targets: tuple[int, ...]
    seed: int
    max_attempts: int = 200

    def validate(self) -> None:
        if not self.parent_id or not self.parent_smiles:
            raise ValueError("Candidate request requires parent ID and SMILES.")
        if self.parent_split not in {"train", "val", "calibration"}:
            raise ValueError(
                "Candidate generation is forbidden on test or unknown splits: "
                f"{self.parent_split!r}"
            )
        if int(self.label) not in (0, 1):
            raise ValueError("Candidate parent label must be binary.")
        if int(self.candidates_per_parent) <= 0:
            raise ValueError("candidates_per_parent must be positive.")
        if not self.size_targets or any(int(value) <= 0 for value in self.size_targets):
            raise ValueError("Candidate size targets must be positive.")
        if int(self.max_attempts) <= 0:
            raise ValueError("max_attempts must be positive.")


@dataclass(frozen=True, slots=True)
class CandidateBatch:
    rows: tuple[dict[str, Any], ...]
    requested_count: int
    generated_count: int
    shortfall_count: int
    shortfall_reason_counts: dict[str, int]


class CandidateGenerator(Protocol):
    source_name: str

    def generate(self, request: CandidateRequest) -> CandidateBatch:
        """Generate one deterministic candidate batch."""


def stable_candidate_id(
    *,
    source: str,
    parent_id: str,
    rank: int,
    fragment: str,
) -> str:
    digest = hashlib.sha256(
        f"{source}\0{parent_id}\0{int(rank)}\0{fragment}".encode("utf-8")
    ).hexdigest()[:20]
    return f"{source.upper()}_{digest}"


def candidate_row(
    *,
    request: CandidateRequest,
    source: str,
    source_variant: str,
    rank: int,
    raw_fragment: str,
    canonical_fragment: str,
    num_fragment_atoms: int,
    num_parent_atoms: int,
) -> dict[str, Any]:
    atom_ratio = float(num_fragment_atoms / num_parent_atoms) if num_parent_atoms else None
    return {
        "candidate_id": stable_candidate_id(
            source=source,
            parent_id=request.parent_id,
            rank=rank,
            fragment=canonical_fragment,
        ),
        "candidate_source": source,
        "candidate_source_variant": source_variant,
        "parent_id": request.parent_id,
        "parent_smiles": request.parent_smiles,
        "parent_split": request.parent_split,
        "label": int(request.label),
        "raw_fragment": raw_fragment,
        "core_fragment": canonical_fragment,
        "final_fragment": canonical_fragment,
        "valid": True,
        "parse_ok": True,
        "connected": True,
        "direct_substructure": True,
        "final_substructure": True,
        "projection_used": False,
        "oracle_ok": False,
        "cf_flip": False,
        "cf_drop": None,
        "p_before": None,
        "p_after": None,
        "atom_ratio": atom_ratio,
        "reward_total": None,
        "generation_seed": int(request.seed),
        "generation_rank": int(rank),
        "checkpoint_path": None,
        "checkpoint_kind": "random_control",
        "source_git_commit": None,
        "num_fragment_atoms": int(num_fragment_atoms),
        "candidate_set_preselected": False,
        "selection_performed_in_eval": False,
    }


__all__ = [
    "CandidateBatch",
    "CandidateGenerator",
    "CandidateRequest",
    "candidate_row",
    "stable_candidate_id",
]
