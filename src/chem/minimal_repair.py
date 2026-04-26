"""Minimal syntax repair helpers for decoded fragment SMILES.

The functions here only make local text edits such as suffix trimming and
closure balancing. They intentionally do not attempt unconstrained molecule
repair or nearest-valid-molecule search.
"""

from __future__ import annotations

from collections import Counter
import re

from src.chem.smiles_utils import is_rdkit_available, parse_smiles, sanitize_molecule
from src.chem.types import FragmentSyntaxRepairCandidate, FragmentSyntaxRepairResult

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - depends on local environment
    Chem = None


_RING_TOKEN_PATTERN = re.compile(r"%\d{2}|\d")
_TRAILING_DANGLING_CHARS = set("(-=#:/\\.[")
_TRAILING_CLOSURE_CHARS = set(")]")


def generate_minimal_syntax_repair_candidates(
    raw_fragment: str,
    *,
    parse_failed_reason: str | None = None,
    max_edits: int = 4,
    allow_parentheses_fix: bool = True,
    allow_ring_fix: bool = True,
    allow_tail_trim: bool = True,
    allow_balanced_prefix_salvage: bool = True,
    prefer_prefix_salvage: bool = True,
    max_suffix_trim: int = 8,
    max_added_closures: int = 2,
) -> tuple[FragmentSyntaxRepairCandidate, ...]:
    """Generate deterministic local repair candidates in priority order."""

    normalized = str(raw_fragment or "").strip()
    if not normalized or max_edits <= 0:
        return ()

    candidates: list[FragmentSyntaxRepairCandidate] = []
    failure_reason = str(parse_failed_reason or "")

    def add_candidate(
        candidate: str,
        method: str,
        reason: str,
        *,
        suffix_trim_count: int = 0,
        added_parentheses: int = 0,
        added_ring_closures: int = 0,
    ) -> None:
        repaired = str(candidate or "").strip()
        if not repaired or repaired == normalized:
            return
        edit_distance = int(suffix_trim_count + added_parentheses + added_ring_closures)
        if edit_distance <= 0 or edit_distance > int(max_edits):
            return
        if suffix_trim_count > int(max_suffix_trim):
            return
        if added_parentheses + added_ring_closures > int(max_added_closures):
            return
        candidates.append(
            FragmentSyntaxRepairCandidate(
                fragment_smiles=repaired,
                repair_method=method,
                reason=reason,
                edit_distance=edit_distance,
                suffix_trim_count=int(suffix_trim_count),
                added_parentheses=int(added_parentheses),
                added_ring_closures=int(added_ring_closures),
            )
        )

    def add_tail_trim_bundle(text: str) -> None:
        if not allow_tail_trim:
            return
        for trimmed, trim_count in _tail_trim_candidates(
            text,
            max_suffix_trim=max_suffix_trim,
        ):
            add_candidate(
                trimmed,
                "tail_trim",
                "minimal_tail_trim",
                suffix_trim_count=trim_count,
            )
            if allow_parentheses_fix:
                for repaired, added_parentheses in _direct_parentheses_closure_variants(
                    trimmed,
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
                for repaired, added_rings in _safe_ring_completion_variants(
                    trimmed,
                    max_added_closures=max_added_closures,
                ):
                    add_candidate(
                        repaired,
                        "tail_trim_ring_closure",
                        "trimmed_tail_and_closed_safe_ring_digit",
                        suffix_trim_count=trim_count,
                        added_ring_closures=added_rings,
                    )

    if allow_parentheses_fix and "parse_failed_unbalanced_parentheses" in failure_reason:
        for repaired, added_parentheses in _direct_parentheses_closure_variants(
            normalized,
            max_added_closures=max_added_closures,
        ):
            add_candidate(
                repaired,
                "parentheses_closure",
                "added_missing_right_parentheses",
                added_parentheses=added_parentheses,
            )
        add_tail_trim_bundle(normalized)
        for repaired, trim_count in _dangling_suffix_cleanup_candidates(
            normalized,
            max_suffix_trim=max_suffix_trim,
        ):
            add_candidate(
                repaired,
                "dangling_suffix_cleanup",
                "removed_trailing_dangling_branch_or_bond",
                suffix_trim_count=trim_count,
            )
        if allow_balanced_prefix_salvage:
            _add_prefix_candidates(
                normalized,
                candidates=candidates,
                max_suffix_trim=max_suffix_trim,
                allow_parentheses_fix=allow_parentheses_fix,
                allow_ring_fix=allow_ring_fix,
                max_added_closures=max_added_closures,
            )
    elif allow_ring_fix and "parse_failed_unclosed_ring" in failure_reason:
        add_tail_trim_bundle(normalized)
        for repaired, trim_count in _remove_last_unclosed_ring_variants(normalized):
            add_candidate(
                repaired,
                "ring_digit_trim",
                "removed_last_unclosed_ring_digit",
                suffix_trim_count=trim_count,
            )
        for repaired, added_rings in _safe_ring_completion_variants(
            normalized,
            max_added_closures=max_added_closures,
        ):
            add_candidate(
                repaired,
                "ring_closure",
                "closed_safe_unbalanced_ring_digit",
                added_ring_closures=added_rings,
            )
        if allow_balanced_prefix_salvage:
            _add_prefix_candidates(
                normalized,
                candidates=candidates,
                max_suffix_trim=max_suffix_trim,
                allow_parentheses_fix=False,
                allow_ring_fix=allow_ring_fix,
                max_added_closures=max_added_closures,
            )
    else:
        add_tail_trim_bundle(normalized)
        for repaired, trim_count in _dangling_suffix_cleanup_candidates(
            normalized,
            max_suffix_trim=max_suffix_trim,
        ):
            add_candidate(
                repaired,
                "dangling_suffix_cleanup",
                "removed_trailing_dangling_branch_or_bond",
                suffix_trim_count=trim_count,
            )
        if allow_parentheses_fix:
            for repaired, added_parentheses in _direct_parentheses_closure_variants(
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
            for repaired, added_rings in _safe_ring_completion_variants(
                normalized,
                max_added_closures=max_added_closures,
            ):
                add_candidate(
                    repaired,
                    "ring_closure",
                    "closed_safe_unbalanced_ring_digit",
                    added_ring_closures=added_rings,
                )
            for repaired, trim_count in _remove_last_unclosed_ring_variants(normalized):
                add_candidate(
                    repaired,
                    "ring_digit_trim",
                    "removed_last_unclosed_ring_digit",
                    suffix_trim_count=trim_count,
                )
        if allow_balanced_prefix_salvage:
            _add_prefix_candidates(
                normalized,
                candidates=candidates,
                max_suffix_trim=max_suffix_trim,
                allow_parentheses_fix=allow_parentheses_fix,
                allow_ring_fix=allow_ring_fix,
                max_added_closures=max_added_closures,
            )

    deduped = _dedupe_candidates(
        candidates,
        prefer_prefix_salvage=prefer_prefix_salvage,
    )
    return tuple(
        candidate
        for candidate in deduped
        if 0 < candidate.edit_distance <= int(max_edits)
    )


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
    """Try a deterministic set of local syntax repairs.

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

    candidates = generate_minimal_syntax_repair_candidates(
        normalized,
        parse_failed_reason=parse_failed_reason,
        max_edits=max_edits,
        allow_parentheses_fix=allow_parentheses_fix,
        allow_ring_fix=allow_ring_fix,
        allow_tail_trim=allow_tail_trim,
        allow_balanced_prefix_salvage=allow_balanced_prefix_salvage,
        prefer_prefix_salvage=prefer_prefix_salvage,
        max_suffix_trim=max_suffix_trim,
        max_added_closures=max_added_closures,
    )
    candidate_count = len(candidates)
    rejection_counts: Counter[str] = Counter()
    parse_ok_count = 0
    best_candidate: str | None = None
    for candidate in candidates:
        best_candidate = candidate.fragment_smiles
        parsed = parse_smiles(
            candidate.fragment_smiles,
            sanitize=True,
            canonicalize=True,
            allow_capped_fragments=True,
        )
        if not parsed.parseable or parsed.mol is None:
            rejection_counts["repair_candidate_parse_failed"] += 1
            continue
        parse_ok_count += 1
        atom_count = _non_dummy_atom_count(parsed.mol)
        if atom_count < int(min_atoms):
            rejection_counts["repair_candidate_too_small"] += 1
            continue
        return FragmentSyntaxRepairResult(
            raw_fragment_smiles=normalized,
            attempted=True,
            success=True,
            repaired_fragment_smiles=str(parsed.canonical_smiles or candidate.fragment_smiles),
            repair_method=candidate.repair_method,
            reason=candidate.reason,
            edit_distance=candidate.edit_distance,
            suffix_trim_count=candidate.suffix_trim_count,
            added_parentheses=candidate.added_parentheses,
            added_ring_closures=candidate.added_ring_closures,
            repaired_atom_count=atom_count,
            candidate_count=candidate_count,
            candidates_parse_ok=parse_ok_count,
            best_candidate=candidate.fragment_smiles,
            accept_stage="parse",
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
        candidates_parse_ok=parse_ok_count,
        best_candidate=best_candidate,
        candidate_accepted=False,
        candidate_rejected_reason=failure_reason,
    )


def _add_prefix_candidates(
    text: str,
    *,
    candidates: list[FragmentSyntaxRepairCandidate],
    max_suffix_trim: int,
    allow_parentheses_fix: bool,
    allow_ring_fix: bool,
    max_added_closures: int,
) -> None:
    for (
        candidate,
        trim_count,
        method,
        reason,
        added_parentheses,
        added_rings,
    ) in _prefix_salvage_candidates(
        text,
        max_suffix_trim=max_suffix_trim,
        allow_parentheses_fix=allow_parentheses_fix,
        allow_ring_fix=allow_ring_fix,
        max_added_closures=max_added_closures,
    ):
        candidates.append(
            FragmentSyntaxRepairCandidate(
                fragment_smiles=candidate,
                repair_method=method,
                reason=reason,
                edit_distance=int(trim_count + added_parentheses + added_rings),
                suffix_trim_count=int(trim_count),
                added_parentheses=int(added_parentheses),
                added_ring_closures=int(added_rings),
            )
        )


def _dedupe_candidates(
    candidates: list[FragmentSyntaxRepairCandidate],
    *,
    prefer_prefix_salvage: bool,
) -> list[FragmentSyntaxRepairCandidate]:
    if prefer_prefix_salvage:
        sort_key = lambda item: (0 if "prefix" in str(item.repair_method or "") else 1)
        ordered = sorted(
            enumerate(candidates),
            key=lambda item: (sort_key(item[1]), item[0]),
        )
        candidate_iter = (candidate for _, candidate in ordered)
    else:
        candidate_iter = iter(candidates)

    seen: set[str] = set()
    deduped: list[FragmentSyntaxRepairCandidate] = []
    for candidate in candidate_iter:
        if candidate.fragment_smiles in seen:
            continue
        seen.add(candidate.fragment_smiles)
        deduped.append(candidate)
    return deduped


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


def _dangling_suffix_cleanup_candidates(
    text: str,
    *,
    max_suffix_trim: int,
) -> tuple[tuple[str, int], ...]:
    candidates: list[tuple[str, int]] = []
    current = str(text or "").strip()
    trimmed = 0
    while current and trimmed < int(max_suffix_trim):
        last_char = current[-1]
        if last_char not in _TRAILING_DANGLING_CHARS and last_char not in _TRAILING_CLOSURE_CHARS:
            if not current.endswith("%"):
                break
        current = current[:-1].strip()
        trimmed += 1
        if current and not _ends_with_dangling_token(current):
            candidates.append((current, trimmed))
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
            for repaired, added_parentheses in _direct_parentheses_closure_variants(
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
            for repaired, added_rings in _safe_ring_completion_variants(
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
                        "prefix_salvage_closed_safe_ring_digit",
                        0,
                        added_rings,
                    )
                )
    return tuple(candidates)


def _direct_parentheses_closure_variants(
    text: str,
    *,
    max_added_closures: int,
) -> tuple[tuple[str, int], ...]:
    variants: list[tuple[str, int]] = []
    normalized = str(text or "").strip()
    missing_right = max(0, normalized.count("(") - normalized.count(")"))
    limit = max(0, int(max_added_closures))
    for added_parentheses in range(1, limit + 1):
        if missing_right > 0 and added_parentheses > missing_right:
            continue
        variants.append((normalized + ")" * added_parentheses, added_parentheses))
    return tuple(variants)


def _safe_ring_completion_variants(
    text: str,
    *,
    max_added_closures: int,
) -> tuple[tuple[str, int], ...]:
    unclosed_ring_tokens = _unclosed_ring_tokens(text)
    if len(unclosed_ring_tokens) != 1:
        return ()
    token = unclosed_ring_tokens[0]
    if int(max_added_closures) < 1:
        return ()
    return ((str(text or "").strip() + token, 1),)


def _remove_last_unclosed_ring_variants(text: str) -> tuple[tuple[str, int], ...]:
    odd_tokens = set(_unclosed_ring_tokens(text))
    if not odd_tokens:
        return ()
    matches = list(_ring_token_matches_outside_brackets(text))
    for start, end, token in reversed(matches):
        if token not in odd_tokens:
            continue
        candidate = (text[:start] + text[end:]).strip()
        if candidate:
            return ((candidate, len(text) - len(candidate)),)
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
    counts = Counter(token for _, _, token in _ring_token_matches_outside_brackets(smiles))
    return tuple(sorted(token for token, count in counts.items() if count % 2 == 1))


def _ring_token_matches_outside_brackets(smiles: str) -> list[tuple[int, int, str]]:
    matches: list[tuple[int, int, str]] = []
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
                token = match.group(0)
                matches.append((match.start(), match.end(), token))
                index = match.end()
                continue
        index += 1
    return matches


def _non_dummy_atom_count(mol: object | None) -> int:
    if Chem is None or mol is None:
        return 0
    try:
        sanitized, _, _, _ = sanitize_molecule(mol, allow_capped_fragments=True)
        target = sanitized or mol
        return sum(1 for atom in target.GetAtoms() if atom.GetAtomicNum() != 0)
    except Exception:  # pragma: no cover - defensive around RDKit objects
        return 0


__all__ = [
    "generate_minimal_syntax_repair_candidates",
    "repair_minimal_fragment_syntax",
]
