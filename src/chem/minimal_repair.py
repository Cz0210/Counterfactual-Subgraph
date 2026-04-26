"""Minimal syntax repair helpers for decoded fragment SMILES.

The functions here only make local text edits such as suffix trimming and
closure balancing. They intentionally do not attempt unconstrained molecule
repair or nearest-valid-molecule search.
"""

from __future__ import annotations

from collections import Counter
import re

from src.chem.smiles_utils import is_rdkit_available, parse_smiles, sanitize_molecule
from src.chem.types import FragmentSyntaxRepairResult

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - depends on local environment
    Chem = None


_RING_TOKEN_PATTERN = re.compile(r"%\d{2}|\d")
_TRAILING_DANGLING_CHARS = set("(-=#:/\\.[")


def repair_minimal_fragment_syntax(
    raw_fragment: str,
    *,
    parse_failed_reason: str | None = None,
    max_edits: int = 4,
    min_atoms: int = 3,
    allow_parentheses_fix: bool = True,
    allow_ring_fix: bool = True,
    allow_tail_trim: bool = True,
    allow_balanced_prefix_salvage: bool = True,
    prefer_prefix_salvage: bool = True,
    max_suffix_trim: int = 8,
    max_added_closures: int = 2,
) -> FragmentSyntaxRepairResult:
    """Try a small set of deterministic syntax repairs.

    The returned string is only a parseable candidate. Callers must still run
    dummy-aware normalization, connectedness, and strict parent-subgraph checks.
    """

    normalized = str(raw_fragment or "").strip()
    if not normalized:
        return FragmentSyntaxRepairResult(
            raw_fragment_smiles=normalized,
            attempted=False,
            success=False,
            reason="empty_fragment",
        )
    if not is_rdkit_available() or Chem is None:
        return FragmentSyntaxRepairResult(
            raw_fragment_smiles=normalized,
            attempted=False,
            success=False,
            reason="rdkit_unavailable",
        )
    if max_edits <= 0:
        return FragmentSyntaxRepairResult(
            raw_fragment_smiles=normalized,
            attempted=False,
            success=False,
            reason="invalid_max_edits",
        )

    candidates: list[tuple[str, str, str, int, int, int]] = []

    def add_candidate(
        candidate: str,
        method: str,
        reason: str,
        *,
        suffix_trim_count: int = 0,
        added_parentheses: int = 0,
        added_ring_closures: int = 0,
    ) -> None:
        candidate = str(candidate or "").strip()
        if not candidate or candidate == normalized:
            return
        edit_distance = int(suffix_trim_count + added_parentheses + added_ring_closures)
        if edit_distance <= 0:
            return
        if suffix_trim_count <= 0 and edit_distance > int(max_edits):
            return
        if suffix_trim_count > int(max_suffix_trim):
            return
        if added_parentheses + added_ring_closures > int(max_added_closures):
            return
        candidates.append(
            (
                candidate,
                method,
                reason,
                int(suffix_trim_count),
                int(added_parentheses),
                int(added_ring_closures),
            )
        )

    if allow_tail_trim:
        tail_trim_candidates = _tail_trim_candidates(
            normalized,
            max_suffix_trim=max_suffix_trim,
        )
        for candidate, trim_count in tail_trim_candidates:
            add_candidate(
                candidate,
                "tail_trim",
                "minimal_tail_trim",
                suffix_trim_count=trim_count,
            )
            if allow_parentheses_fix:
                for repaired, added_parentheses in _parentheses_closure_variants(
                    candidate,
                    max_added_closures=max_added_closures,
                ):
                    add_candidate(
                        repaired,
                        "tail_trim_parentheses_closure",
                        "trimmed_tail_and_added_missing_right_parentheses",
                        suffix_trim_count=trim_count,
                        added_parentheses=added_parentheses,
                    )
            if allow_ring_fix:
                for repaired, added_rings in _ring_closure_variants(
                    candidate,
                    max_added_closures=max_added_closures,
                ):
                    add_candidate(
                        repaired,
                        "tail_trim_ring_closure",
                        "trimmed_tail_and_added_unclosed_ring_digits",
                        suffix_trim_count=trim_count,
                        added_ring_closures=added_rings,
                    )

    if allow_parentheses_fix:
        for repaired, added_parentheses in _parentheses_closure_variants(
            normalized,
            max_added_closures=max_added_closures,
        ):
            add_candidate(
                repaired,
                "parentheses_closure",
                "added_missing_right_parentheses",
                added_parentheses=added_parentheses,
            )

    if allow_ring_fix:
        for repaired, added_rings in _ring_closure_variants(
            normalized,
            max_added_closures=max_added_closures,
        ):
            add_candidate(
                repaired,
                "ring_closure",
                "added_unclosed_ring_digits",
                added_ring_closures=added_rings,
            )

    if allow_balanced_prefix_salvage:
        prefix_candidates = _prefix_salvage_candidates(
            normalized,
            max_suffix_trim=max_suffix_trim,
            allow_parentheses_fix=allow_parentheses_fix,
            allow_ring_fix=allow_ring_fix,
            max_added_closures=max_added_closures,
        )
        prefix_items = [
            (
                candidate,
                method,
                reason,
                trim_count,
                added_parentheses,
                added_rings,
            )
            for (
                candidate,
                trim_count,
                method,
                reason,
                added_parentheses,
                added_rings,
            ) in prefix_candidates
        ]
        if prefer_prefix_salvage:
            candidates = prefix_items + candidates
        else:
            candidates.extend(prefix_items)

    seen: set[str] = set()
    unique_candidates: list[tuple[str, str, str, int, int, int]] = []
    for candidate, method, reason, suffix_trim, added_parens, added_rings in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(
            (candidate, method, reason, suffix_trim, added_parens, added_rings)
        )

    candidate_count = len(unique_candidates)
    rejection_counts: Counter[str] = Counter()
    for (
        candidate,
        method,
        reason,
        suffix_trim,
        added_parens,
        added_rings,
    ) in unique_candidates:
        parsed = parse_smiles(
            candidate,
            sanitize=True,
            canonicalize=True,
            allow_capped_fragments=True,
        )
        if not parsed.parseable or parsed.mol is None:
            rejection_counts["repair_candidate_parse_failed"] += 1
            continue
        atom_count = _non_dummy_atom_count(parsed.mol)
        if atom_count < int(min_atoms):
            rejection_counts["repair_candidate_too_small"] += 1
            continue
        return FragmentSyntaxRepairResult(
            raw_fragment_smiles=normalized,
            attempted=True,
            success=True,
            repaired_fragment_smiles=str(parsed.canonical_smiles or candidate),
            repair_method=method,
            reason=reason,
            edit_distance=suffix_trim + added_parens + added_rings,
            suffix_trim_count=suffix_trim,
            added_parentheses=added_parens,
            added_ring_closures=added_rings,
            repaired_atom_count=atom_count,
            candidate_count=candidate_count,
            candidate_accepted=True,
        )

    failure_reason, failure_stage = _repair_failure_from_rejections(
        candidate_count=candidate_count,
        rejection_counts=rejection_counts,
        fallback_reason=parse_failed_reason,
    )
    return FragmentSyntaxRepairResult(
        raw_fragment_smiles=normalized,
        attempted=True,
        success=False,
        reason=failure_reason,
        failure_reason=failure_reason,
        failure_stage=failure_stage,
        candidate_count=candidate_count,
        candidate_accepted=False,
        candidate_rejected_reason=failure_reason,
    )


def _tail_trim_candidates(
    text: str,
    *,
    max_suffix_trim: int,
) -> tuple[tuple[str, int], ...]:
    candidates: list[tuple[str, int]] = []
    max_trim = min(int(max_suffix_trim), max(0, len(text) - 1))
    for trim_count in range(1, max_trim + 1):
        candidate = text[:-trim_count].strip()
        if not candidate:
            continue
        if _looks_less_dangling(candidate) or _has_balanced_simple_syntax(candidate):
            candidates.append((candidate, trim_count))
    return tuple(candidates)


def _prefix_salvage_candidates(
    text: str,
    *,
    max_suffix_trim: int,
    allow_parentheses_fix: bool,
    allow_ring_fix: bool,
    max_added_closures: int,
) -> tuple[tuple[str, int, str, str, int, int], ...]:
    candidates: list[tuple[str, int, str, str, int, int]] = []
    max_trim = min(int(max_suffix_trim), max(0, len(text) - 1))
    for trim_count in range(1, max_trim + 1):
        prefix = text[:-trim_count].strip()
        if not prefix:
            continue
        if _has_balanced_simple_syntax(prefix):
            candidates.append(
                (
                    prefix,
                    trim_count,
                    "balanced_prefix",
                    "longest_parseable_balanced_prefix",
                    0,
                    0,
                )
            )
        if allow_parentheses_fix:
            for repaired, added_parentheses in _parentheses_closure_variants(
                prefix,
                max_added_closures=max_added_closures,
            ):
                if _ends_with_dangling_token(repaired):
                    continue
                candidates.append(
                    (
                        repaired,
                        trim_count,
                        "balanced_prefix_parentheses_closure",
                        "prefix_salvage_added_missing_right_parentheses",
                        added_parentheses,
                        0,
                    )
                )
        if allow_ring_fix:
            for repaired, added_rings in _ring_closure_variants(
                prefix,
                max_added_closures=max_added_closures,
            ):
                if _ends_with_dangling_token(repaired):
                    continue
                candidates.append(
                    (
                        repaired,
                        trim_count,
                        "balanced_prefix_ring_closure",
                        "prefix_salvage_added_unclosed_ring_digits",
                        0,
                        added_rings,
                    )
                )
    return tuple(candidates)


def _parentheses_closure_variants(
    text: str,
    *,
    max_added_closures: int,
) -> tuple[tuple[str, int], ...]:
    missing_right = str(text or "").count("(") - str(text or "").count(")")
    if 0 < missing_right <= int(max_added_closures):
        return ((str(text or "").strip() + ")" * missing_right, missing_right),)
    return ()


def _ring_closure_variants(
    text: str,
    *,
    max_added_closures: int,
) -> tuple[tuple[str, int], ...]:
    unclosed_ring_tokens = _unclosed_ring_tokens(text)
    if not unclosed_ring_tokens:
        return ()
    if len(unclosed_ring_tokens) <= int(max_added_closures):
        return (
            (
                str(text or "").strip() + "".join(unclosed_ring_tokens),
                len(unclosed_ring_tokens),
            ),
        )
    return ()


def _repair_failure_from_rejections(
    *,
    candidate_count: int,
    rejection_counts: Counter[str],
    fallback_reason: str | None,
) -> tuple[str, str]:
    if candidate_count <= 0:
        return "repair_no_candidate_generated", "candidate_generation"
    if rejection_counts:
        reason, _ = rejection_counts.most_common(1)[0]
        stage = (
            "parse"
            if reason == "repair_candidate_parse_failed"
            else "candidate_filter"
        )
        return reason, stage
    return fallback_reason or "repair_candidate_other", "candidate_filter"


def _has_balanced_simple_syntax(text: str) -> bool:
    return (
        text.count("(") == text.count(")")
        and text.count("[") == text.count("]")
        and not _unclosed_ring_tokens(text)
        and not _ends_with_dangling_token(text)
    )


def _looks_less_dangling(text: str) -> bool:
    return not _ends_with_dangling_token(text)


def _ends_with_dangling_token(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return True
    if normalized[-1] in _TRAILING_DANGLING_CHARS:
        return True
    if normalized.endswith("%"):
        return True
    if normalized.endswith("%0") or normalized.endswith("%1"):
        return True
    return False


def _unclosed_ring_tokens(smiles: str) -> tuple[str, ...]:
    counts = Counter(_collect_ring_tokens_outside_brackets(smiles))
    return tuple(sorted(token for token, count in counts.items() if count % 2 == 1))


def _collect_ring_tokens_outside_brackets(smiles: str) -> list[str]:
    tokens: list[str] = []
    bracket_depth = 0
    text = str(smiles or "")
    index = 0
    while index < len(text):
        char = text[index]
        if char == "[":
            bracket_depth += 1
            index += 1
            continue
        if char == "]":
            bracket_depth = max(0, bracket_depth - 1)
            index += 1
            continue
        if bracket_depth == 0:
            match = _RING_TOKEN_PATTERN.match(text, index)
            if match:
                tokens.append(match.group(0))
                index = match.end()
                continue
        index += 1
    return tokens


def _non_dummy_atom_count(mol: object | None) -> int:
    if Chem is None or mol is None:
        return 0
    try:
        sanitized, _, _, _ = sanitize_molecule(mol, allow_capped_fragments=True)
        target = sanitized or mol
        return sum(1 for atom in target.GetAtoms() if atom.GetAtomicNum() != 0)
    except Exception:  # pragma: no cover - defensive around RDKit objects
        return 0


__all__ = ["repair_minimal_fragment_syntax"]
