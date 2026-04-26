"""Unified PPO reward wrapper for counterfactual molecular fragment generation.

This module intentionally keeps the reward interface small and defensive:

1. Step A checks whether the generated fragment is parseable and connected.
2. Step B checks whether the fragment is a genuine parent subgraph.
3. Step C computes the semantic term with a deletion-based counterfactual
   teacher oracle on the parent and residual molecules.

Important alignment note:
The repository's source-of-truth documents define the task as deletion-based
counterfactual generation. Fragment-level teacher scores are retained only as
diagnostic fields; the semantic term that enters the decoded PPO reward is the
counterfactual score computed after deleting one matched fragment instance from
the parent molecule.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
import re
from typing import Any, Sequence

from src.chem import (
    delete_fragment_from_parent,
    generate_minimal_syntax_repair_candidates,
    is_parent_substructure,
    is_rdkit_available,
    parse_smiles,
    project_fragment_to_parent_subgraph,
    repair_minimal_fragment_syntax,
    repair_fragment_to_parent_subgraph,
    salvage_connected_component,
    sanitize_molecule,
)
from src.rewards.reward_calculator import (
    load_oracle_bundle,
)
from src.rewards.counterfactual_oracle import CounterfactualTeacherScorer
from src.rewards.teacher_semantic import TeacherSemanticScorer
from src.chem.types import FragmentSyntaxRepairResult

try:
    from rdkit import Chem, RDLogger
except ImportError:  # pragma: no cover - depends on local runtime
    Chem = None
    RDLogger = None


_LOGGER = logging.getLogger(__name__)
_RING_TOKEN_PATTERN = re.compile(r"%\d{2}|\d")


def shape_probability_reward(
    target_probability: float,
    *,
    success_threshold: float = 0.5,
    success_base_reward: float = 5.0,
    probability_scale: float = 5.0,
) -> float:
    """Map one target-label probability to a smooth PPO reward.

    The shaping follows the user's requested monotonic structure:

    - successful flips receive a strong positive bonus;
    - unsuccessful attempts still receive a small dense reward equal to the
      target probability, which gives PPO a direction for exploration.
    """

    probability = float(max(0.0, min(1.0, target_probability)))
    if probability > float(success_threshold):
        return float(success_base_reward + probability * probability_scale)
    return probability


@dataclass(frozen=True, slots=True, kw_only=True)
class RewardTrace:
    """Structured debug record for one PPO reward computation."""

    parent_smiles: str
    generated_smiles: str
    normalized_generated_smiles: str
    raw_fragment_smiles: str | None = None
    core_fragment_smiles: str | None = None
    original_label: int
    target_label: int
    reward: float
    valid_smiles: bool = False
    connected_fragment: bool = False
    is_subgraph: bool = False
    deletion_success: bool = False
    counterfactual_evaluated: bool = False
    flip_success: bool = False
    target_probability: float | None = None
    inactive_probability: float | None = None
    residual_smiles: str | None = None
    empty_response: bool = False
    full_parent: bool = False
    empty_residual: bool = False
    oracle_ok: bool = False
    raw_parse_ok: bool = False
    core_parse_ok: bool = False
    has_dummy_atoms: bool = False
    dummy_count: int = 0
    raw_has_dummy: bool = False
    raw_dummy_count: int = 0
    parse_stage: str | None = None
    parsed_raw_with_dummy: bool = False
    parsed_core: bool = False
    dummy_removed_before_parse: bool = False
    parse_failed_reason: str | None = None
    core_atom_count: int = 0
    teacher_input_smiles: str | None = None
    teacher_available: bool = False
    teacher_called: bool = False
    teacher_probability: float | None = None
    teacher_predicted_label: int | None = None
    teacher_reason: str | None = None
    fragment_teacher_sem: float | None = None
    parent_without_fragment_smiles: str | None = None
    counterfactual_teacher_available: bool = False
    counterfactual_teacher_called: bool = False
    counterfactual_teacher_reason: str | None = None
    p_before: float | None = None
    p_after: float | None = None
    pred_before: int | None = None
    pred_after: int | None = None
    cf_drop: float | None = None
    cf_flip: bool = False
    failure_stage: str | None = None
    failure_tag: str | None = None
    invalid_detail: str | None = None
    generated_char_count: int = 0
    repair_attempted: bool = False
    repair_success: bool = False
    repaired_fragment_smiles: str | None = None
    repair_source: str | None = None
    repair_similarity: float | None = None
    repair_reason: str | None = None
    repair_method: str | None = None
    repair_edit_distance: int = 0
    repair_suffix_trim_count: int = 0
    repair_added_parentheses: int = 0
    repair_added_ring_closures: int = 0
    repaired_raw_fragment: str | None = None
    repaired_fragment_chars: int = 0
    repaired_parse_stage: str | None = None
    repaired_parsed_raw: bool = False
    repaired_parsed_core: bool = False
    repair_failure_reason: str | None = None
    repair_failure_stage: str | None = None
    repair_candidate_count: int = 0
    repair_candidates_parse_ok: int = 0
    repair_candidates_core_ok: int = 0
    repair_candidates_parent_ok: int = 0
    repair_candidates_projection_ok: int = 0
    repair_best_candidate: str | None = None
    repair_accept_stage: str | None = None
    repair_candidate_accepted: bool = False
    repair_candidate_rejected_reason: str | None = None
    component_salvage_attempted: bool = False
    component_salvage_success: bool = False
    component_count: int = 0
    raw_component_count: int = 0
    core_component_count: int = 0
    salvage_method: str | None = None
    salvaged_fragment: str | None = None
    salvaged_atom_count: int | None = None
    component_salvage_failure_reason: str | None = None
    component_salvage_stage: str | None = None
    component_salvage_candidate_count: int = 0
    component_salvage_best_candidate: str | None = None
    multi_dummy_hard_fail: bool = False
    dummy_salvage_attempted: bool = False
    dummy_salvage_success: bool = False
    dummy_salvage_method: str | None = None
    dummy_salvaged_fragment: str | None = None
    near_parent_hard_fail: bool = False
    residual_atom_count: int | None = None
    residual_atom_ratio: float | None = None
    tiny_fragment_hard_fail: bool = False
    fragment_atom_count: int = 0
    min_fragment_atoms: int = 0
    projection_attempted: bool = False
    projection_success: bool = False
    projection_method: str | None = None
    projection_score: float | None = None
    projection_source: str | None = None
    projected_fragment_smiles: str | None = None
    projection_atom_count: int | None = None
    projection_atom_ratio: float | None = None
    projection_penalty: float = 0.0
    num_projection_candidates: int = 0
    projection_reason: str | None = None
    size_window_reward: float = 0.0
    size_window_bucket: str | None = None
    size_window_low: float | None = None
    size_window_high: float | None = None
    final_fragment_atom_count: int = 0
    final_fragment_atom_ratio: float | None = None
    error_message: str | None = None
    breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary for logging."""

        return asdict(self)


def has_dummy_atom(mol: object | None) -> bool:
    """Return whether the RDKit molecule contains at least one dummy atom."""

    if Chem is None or mol is None:
        return False
    try:
        return any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms())
    except Exception:  # pragma: no cover - defensive around RDKit objects
        return False


def _dummy_atom_count(mol: object | None) -> int:
    if Chem is None or mol is None:
        return 0
    try:
        return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)
    except Exception:  # pragma: no cover - defensive around RDKit objects
        return 0


def _non_dummy_atom_count(mol: object | None) -> int:
    if Chem is None or mol is None:
        return 0
    try:
        return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 0)
    except Exception:  # pragma: no cover - defensive around RDKit objects
        return 0


def _component_count(mol: object | None) -> int:
    if Chem is None or mol is None:
        return 0
    try:
        return len(Chem.GetMolFrags(mol, asMols=False))
    except Exception:  # pragma: no cover - defensive around RDKit objects
        return 0


def remove_dummy_atoms_from_mol(mol: object | None) -> object | None:
    """Remove atomic-number-0 atoms and return a sanitized core molecule."""

    if Chem is None or mol is None:
        return None

    editable = Chem.RWMol(Chem.Mol(mol))
    dummy_indices = sorted(
        (atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0),
        reverse=True,
    )
    for atom_index in dummy_indices:
        editable.RemoveAtom(atom_index)

    core_mol = editable.GetMol()
    if core_mol.GetNumAtoms() == 0:
        return None

    sanitized_core, _, _, _ = sanitize_molecule(
        core_mol,
        allow_capped_fragments=False,
    )
    return sanitized_core


def mol_to_smiles_safe(mol: object | None) -> str | None:
    """Canonicalize a molecule defensively."""

    if Chem is None or mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:  # pragma: no cover - depends on RDKit internals
        return None


def preprocess_generated_fragment(generated_smiles: str) -> tuple[str, int]:
    """Apply the light decoded-fragment cleanup before chemistry rewarding."""

    normalized = str(generated_smiles or "").strip()
    if not normalized:
        return "", 0
    first_line = normalized.splitlines()[0].strip()
    return first_line, len(first_line)


def _collect_ring_tokens_outside_brackets(smiles: str) -> list[str]:
    tokens: list[str] = []
    bracket_depth = 0
    index = 0
    text = str(smiles or "")
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
            if (
                char == "%"
                and index + 2 < len(text)
                and text[index + 1 : index + 3].isdigit()
            ):
                tokens.append(text[index : index + 3])
                index += 3
                continue
            if char.isdigit():
                tokens.append(char)
        index += 1
    return tokens


def detect_obvious_parse_failure_detail(smiles: str) -> str | None:
    """Return a coarse reason when the fragment likely failed due to missing closure."""

    normalized = str(smiles or "").strip()
    if not normalized:
        return None
    if normalized.count("(") != normalized.count(")"):
        return "parse_failed_unbalanced_parentheses"
    if normalized.count("[") != normalized.count("]"):
        return "parse_failed_unbalanced_brackets"
    ring_tokens = _collect_ring_tokens_outside_brackets(normalized)
    if any(count % 2 == 1 for count in Counter(ring_tokens).values()):
        return "parse_failed_unclosed_ring"
    return None


def classify_parse_failed_reason(
    *,
    raw_fragment_smiles: str,
    raw_has_dummy: bool,
    parse_stage: str | None,
) -> str:
    """Bucket parse failures without rewriting the fragment string."""

    closure_reason = detect_obvious_parse_failure_detail(raw_fragment_smiles)
    if closure_reason is not None:
        return closure_reason
    if parse_stage == "core_after_dummy_removal":
        return "parse_failed_after_dummy_removal"
    if raw_has_dummy:
        return "parse_failed_raw_with_dummy"
    return "parse_failed_raw_without_dummy"


def normalize_fragment_with_dummy_atoms(fragment_smiles: str) -> dict[str, Any]:
    """Build one raw/core fragment view while preserving dummy-atom semantics."""

    normalized = str(fragment_smiles or "").strip()
    raw_dummy_count = normalized.count("*")
    raw_has_dummy = raw_dummy_count > 0
    normalized_fragment: dict[str, Any] = {
        "raw": normalized,
        "raw_parse_ok": False,
        "raw_sanitized": False,
        "raw_mol": None,
        "raw_canonical_smiles": None,
        "has_dummy": raw_has_dummy,
        "raw_has_dummy": raw_has_dummy,
        "raw_dummy_count": raw_dummy_count,
        "core_mol": None,
        "core_smiles": None,
        "core_parse_ok": False,
        "dummy_count": raw_dummy_count,
        "parse_stage": None,
        "parsed_raw_with_dummy": False,
        "parsed_core": False,
        "dummy_removed_before_parse": False,
        "parse_failed_reason": None,
        "core_atom_count": 0,
        "raw_component_count": 0,
        "core_component_count": 0,
    }
    if not normalized or not is_rdkit_available() or Chem is None:
        return normalized_fragment

    normalized_fragment["parse_stage"] = "raw_with_dummy" if raw_has_dummy else "raw_without_dummy"
    parsed_raw = parse_smiles(
        normalized,
        sanitize=True,
        canonicalize=True,
        allow_capped_fragments=True,
    )
    normalized_fragment["parsed_raw_with_dummy"] = bool(parsed_raw.parseable)
    normalized_fragment["raw_parse_ok"] = bool(parsed_raw.parseable)
    normalized_fragment["raw_sanitized"] = bool(parsed_raw.sanitized)
    normalized_fragment["raw_mol"] = parsed_raw.mol
    normalized_fragment["raw_canonical_smiles"] = parsed_raw.canonical_smiles
    normalized_fragment["raw_component_count"] = _component_count(parsed_raw.mol)
    if parsed_raw.mol is None:
        normalized_fragment["parse_failed_reason"] = classify_parse_failed_reason(
            raw_fragment_smiles=normalized,
            raw_has_dummy=raw_has_dummy,
            parse_stage=normalized_fragment["parse_stage"],
        )
        return normalized_fragment

    parsed_dummy_count = _dummy_atom_count(parsed_raw.mol)
    normalized_fragment["has_dummy"] = raw_has_dummy or has_dummy_atom(parsed_raw.mol)
    normalized_fragment["dummy_count"] = max(raw_dummy_count, parsed_dummy_count)

    if normalized_fragment["has_dummy"]:
        normalized_fragment["parse_stage"] = "core_after_dummy_removal"
        core_mol = remove_dummy_atoms_from_mol(parsed_raw.mol)
        normalized_fragment["core_mol"] = core_mol
        normalized_fragment["core_smiles"] = mol_to_smiles_safe(core_mol)
        normalized_fragment["parsed_core"] = core_mol is not None
        normalized_fragment["core_parse_ok"] = core_mol is not None
        normalized_fragment["core_atom_count"] = _non_dummy_atom_count(core_mol)
        normalized_fragment["core_component_count"] = _component_count(core_mol)
        if core_mol is None:
            normalized_fragment["parse_failed_reason"] = classify_parse_failed_reason(
                raw_fragment_smiles=normalized,
                raw_has_dummy=raw_has_dummy,
                parse_stage=normalized_fragment["parse_stage"],
            )
        return normalized_fragment

    normalized_fragment["parse_stage"] = "core_without_dummy_removal"
    sanitized_core = None
    if parsed_raw.mol is not None:
        sanitized_core, _, _, _ = sanitize_molecule(
            parsed_raw.mol,
            allow_capped_fragments=False,
        )
    normalized_fragment["core_mol"] = sanitized_core
    normalized_fragment["core_smiles"] = (
        parsed_raw.canonical_smiles if sanitized_core is not None else None
    )
    normalized_fragment["parsed_core"] = sanitized_core is not None
    normalized_fragment["core_parse_ok"] = sanitized_core is not None
    normalized_fragment["core_atom_count"] = _non_dummy_atom_count(sanitized_core)
    normalized_fragment["core_component_count"] = _component_count(sanitized_core)
    return normalized_fragment


class ChemRLRewarder:
    """Unified reward computer for PPO-stage molecular fragment generation.

    Public API:
    - ``calculate_rewards(parent_smiles, generated_smiles)`` matches the
      user-requested minimal interface and assumes the configured default parent
      label when no explicit labels are supplied.
    - ``calculate_rewards_with_labels(...)`` keeps the implementation aligned
      with the repository's binary counterfactual objective.
    """

    def __init__(
        self,
        oracle_path: str | Path = Path("outputs/hpc/oracle/aids_rf_model.pkl"),
        *,
        default_parent_label: int = 1,
        minimum_reward: float = -5.0,
        format_pass_reward: float = 0.25,
        format_penalty: float = -0.25,
        valid_pass_reward: float = 1.0,
        partial_valid_reward: float = 0.3,
        invalid_smiles_penalty: float = -2.0,
        subgraph_pass_reward: float = 1.0,
        invalid_subgraph_penalty: float = -2.0,
        compactness_bonus: float = 0.25,
        compact_atom_target: int = 12,
        compact_atom_penalty_scale: float = 0.05,
        teacher_scorer: TeacherSemanticScorer | None = None,
        counterfactual_teacher_scorer: CounterfactualTeacherScorer | None = None,
        teacher_sem_scale: float = 1.0,
        teacher_sem_missing_penalty: float = -5.0,
        full_parent_penalty: float = -6.0,
        empty_residual_penalty: float = -4.0,
        max_generation_chars: int = 96,
        enable_parent_aware_repair: bool = False,
        repair_min_similarity: float = 0.35,
        repair_max_candidates: int = 24,
        enable_parent_projection: bool = False,
        projection_min_score: float = 0.35,
        projection_max_candidates: int = 128,
        projection_min_atoms: int = 3,
        projection_max_atom_ratio: float = 0.70,
        projection_penalty: float = 0.5,
        projection_enable_khop3: bool = False,
        projection_mcs_timeout: int = 1,
        enable_minimal_syntax_repair: bool = True,
        syntax_repair_max_edits: int = 4,
        syntax_repair_min_atoms: int = 3,
        syntax_repair_allow_parentheses_fix: bool = True,
        syntax_repair_allow_ring_fix: bool = True,
        syntax_repair_allow_tail_trim: bool = True,
        syntax_repair_allow_balanced_prefix_salvage: bool = True,
        syntax_repair_prefer_prefix_salvage: bool = True,
        syntax_repair_max_suffix_trim: int = 8,
        syntax_repair_max_added_closures: int = 2,
        enable_component_salvage: bool = True,
        component_salvage_method: str = "largest_then_best_parent_match",
        component_salvage_min_atoms: int = 3,
        multi_dummy_hard_fail_threshold: int = 3,
        enable_light_dummy_salvage: bool = False,
        near_parent_hard_ratio: float = 0.85,
        min_residual_atoms: int = 3,
        min_residual_ratio: float = 0.10,
        min_fragment_atoms: int = 0,
        tiny_fragment_hard_fail_penalty: float = -6.0,
        enable_size_window_reward: bool = True,
        size_window_low: float = 0.15,
        size_window_high: float = 0.65,
        size_window_bonus: float = 0.4,
        size_window_small_penalty: float = -0.4,
        size_window_large_penalty: float = -0.4,
        require_teacher_sem: bool = False,
        disable_counterfactual_teacher: bool = False,
        success_threshold: float = 0.5,
        success_base_reward: float = 5.0,
        probability_scale: float = 5.0,
        quiet_rdkit: bool = True,
    ) -> None:
        if not is_rdkit_available() or Chem is None:
            raise RuntimeError(
                "ChemRLRewarder requires RDKit. Please install RDKit in the PPO runtime."
            )

        if int(default_parent_label) not in (0, 1):
            raise ValueError("default_parent_label must be 0 or 1.")

        self.oracle_path = Path(oracle_path).expanduser().resolve()
        self.default_parent_label = int(default_parent_label)
        self.minimum_reward = float(minimum_reward)
        self.format_pass_reward = float(format_pass_reward)
        self.format_penalty = float(format_penalty)
        self.valid_pass_reward = float(valid_pass_reward)
        self.partial_valid_reward = float(partial_valid_reward)
        self.invalid_smiles_penalty = float(invalid_smiles_penalty)
        self.subgraph_pass_reward = float(subgraph_pass_reward)
        self.invalid_subgraph_penalty = float(invalid_subgraph_penalty)
        self.compactness_bonus = float(compactness_bonus)
        self.compact_atom_target = int(compact_atom_target)
        self.compact_atom_penalty_scale = float(compact_atom_penalty_scale)
        self.teacher_scorer = teacher_scorer
        self.counterfactual_teacher_scorer = counterfactual_teacher_scorer
        self.teacher_sem_scale = float(teacher_sem_scale)
        self.teacher_sem_missing_penalty = float(teacher_sem_missing_penalty)
        self.full_parent_penalty = float(full_parent_penalty)
        self.empty_residual_penalty = float(empty_residual_penalty)
        self.max_generation_chars = int(max_generation_chars)
        self.enable_parent_aware_repair = bool(enable_parent_aware_repair)
        self.repair_min_similarity = float(repair_min_similarity)
        self.repair_max_candidates = int(repair_max_candidates)
        self.enable_parent_projection = bool(enable_parent_projection)
        self.projection_min_score = float(projection_min_score)
        self.projection_max_candidates = int(projection_max_candidates)
        self.projection_min_atoms = int(projection_min_atoms)
        self.projection_max_atom_ratio = float(projection_max_atom_ratio)
        self.projection_penalty = float(projection_penalty)
        self.projection_enable_khop3 = bool(projection_enable_khop3)
        self.projection_mcs_timeout = int(projection_mcs_timeout)
        self.enable_minimal_syntax_repair = bool(enable_minimal_syntax_repair)
        self.syntax_repair_max_edits = int(syntax_repair_max_edits)
        self.syntax_repair_min_atoms = int(syntax_repair_min_atoms)
        self.syntax_repair_allow_parentheses_fix = bool(
            syntax_repair_allow_parentheses_fix
        )
        self.syntax_repair_allow_ring_fix = bool(syntax_repair_allow_ring_fix)
        self.syntax_repair_allow_tail_trim = bool(syntax_repair_allow_tail_trim)
        self.syntax_repair_allow_balanced_prefix_salvage = bool(
            syntax_repair_allow_balanced_prefix_salvage
        )
        self.syntax_repair_prefer_prefix_salvage = bool(
            syntax_repair_prefer_prefix_salvage
        )
        self.syntax_repair_max_suffix_trim = int(syntax_repair_max_suffix_trim)
        self.syntax_repair_max_added_closures = int(
            syntax_repair_max_added_closures
        )
        self.enable_component_salvage = bool(enable_component_salvage)
        self.component_salvage_method = str(component_salvage_method)
        self.component_salvage_min_atoms = int(component_salvage_min_atoms)
        self.multi_dummy_hard_fail_threshold = int(multi_dummy_hard_fail_threshold)
        self.enable_light_dummy_salvage = bool(enable_light_dummy_salvage)
        self.near_parent_hard_ratio = float(near_parent_hard_ratio)
        self.min_residual_atoms = int(min_residual_atoms)
        self.min_residual_ratio = float(min_residual_ratio)
        self.min_fragment_atoms = int(min_fragment_atoms)
        self.tiny_fragment_hard_fail_penalty = float(tiny_fragment_hard_fail_penalty)
        self.enable_size_window_reward = bool(enable_size_window_reward)
        self.size_window_low = float(size_window_low)
        self.size_window_high = float(size_window_high)
        self.size_window_bonus = float(size_window_bonus)
        self.size_window_small_penalty = float(size_window_small_penalty)
        self.size_window_large_penalty = float(size_window_large_penalty)
        self.require_teacher_sem = bool(require_teacher_sem)
        self.disable_counterfactual_teacher = bool(disable_counterfactual_teacher)
        self.success_threshold = float(success_threshold)
        self.success_base_reward = float(success_base_reward)
        self.probability_scale = float(probability_scale)
        if self.max_generation_chars <= 0:
            raise ValueError("max_generation_chars must be positive.")
        if self.repair_min_similarity < 0.0 or self.repair_min_similarity > 1.0:
            raise ValueError("repair_min_similarity must be in [0, 1].")
        if self.repair_max_candidates <= 0:
            raise ValueError("repair_max_candidates must be positive.")
        if self.projection_min_score < 0.0:
            raise ValueError("projection_min_score must be non-negative.")
        if self.projection_max_candidates <= 0:
            raise ValueError("projection_max_candidates must be positive.")
        if self.projection_min_atoms <= 0:
            raise ValueError("projection_min_atoms must be positive.")
        if self.projection_max_atom_ratio <= 0.0:
            raise ValueError("projection_max_atom_ratio must be positive.")
        if self.projection_mcs_timeout <= 0:
            raise ValueError("projection_mcs_timeout must be positive.")
        if self.syntax_repair_max_edits <= 0:
            raise ValueError("syntax_repair_max_edits must be positive.")
        if self.syntax_repair_min_atoms <= 0:
            raise ValueError("syntax_repair_min_atoms must be positive.")
        if self.syntax_repair_max_suffix_trim <= 0:
            raise ValueError("syntax_repair_max_suffix_trim must be positive.")
        if self.syntax_repair_max_added_closures <= 0:
            raise ValueError("syntax_repair_max_added_closures must be positive.")
        if self.component_salvage_min_atoms <= 0:
            raise ValueError("component_salvage_min_atoms must be positive.")
        if self.multi_dummy_hard_fail_threshold <= 0:
            raise ValueError("multi_dummy_hard_fail_threshold must be positive.")
        if self.near_parent_hard_ratio <= 0.0:
            raise ValueError("near_parent_hard_ratio must be positive.")
        if self.min_residual_atoms < 0:
            raise ValueError("min_residual_atoms must be non-negative.")
        if self.min_residual_ratio < 0.0:
            raise ValueError("min_residual_ratio must be non-negative.")
        if self.min_fragment_atoms < 0:
            raise ValueError("min_fragment_atoms must be non-negative.")
        if self.size_window_low < 0.0 or self.size_window_low > 1.0:
            raise ValueError("size_window_low must be in [0, 1].")
        if self.size_window_high < 0.0 or self.size_window_high > 1.0:
            raise ValueError("size_window_high must be in [0, 1].")
        if self.size_window_low > self.size_window_high:
            raise ValueError("size_window_low must be <= size_window_high.")

        if self.require_teacher_sem and self.disable_counterfactual_teacher:
            raise ValueError(
                "require_teacher_sem=True is incompatible with disable_counterfactual_teacher=True."
            )

        if quiet_rdkit and RDLogger is not None:
            try:
                RDLogger.DisableLog("rdApp.*")
            except Exception:
                _LOGGER.debug("Could not disable RDKit logging.", exc_info=True)

        bundle = load_oracle_bundle(self.oracle_path)
        self._oracle_bundle = bundle
        self._oracle_model = bundle["model"]
        self._fingerprint_radius = int(bundle["fingerprint_radius"])
        self._fingerprint_bits = int(bundle["fingerprint_bits"])
        raw_labels = bundle.get("class_labels", getattr(self._oracle_model, "classes_", (0, 1)))
        self._class_labels = tuple(int(label) for label in raw_labels)

    def calculate_rewards(
        self,
        prompts_smiles: list[str],
        generated_smiles: list[str],
    ) -> list[float]:
        """Return only scalar rewards using the configured default label.

        This keeps the user-requested signature intact and is appropriate when
        the PPO dataset contains only active parent molecules and the target is
        to flip them toward the inactive class.
        """

        traces = self.calculate_reward_details_batch(
            prompts_smiles,
            generated_smiles,
            parent_labels=None,
        )
        return [trace.reward for trace in traces]

    def calculate_rewards_with_labels(
        self,
        prompts_smiles: Sequence[str],
        generated_smiles: Sequence[str],
        parent_labels: Sequence[int],
    ) -> list[float]:
        """Return scalar rewards while respecting one label per parent molecule."""

        traces = self.calculate_reward_details_batch(
            prompts_smiles,
            generated_smiles,
            parent_labels=parent_labels,
        )
        return [trace.reward for trace in traces]

    def calculate_reward_details_batch(
        self,
        prompts_smiles: Sequence[str],
        generated_smiles: Sequence[str],
        parent_labels: Sequence[int] | None = None,
    ) -> list[RewardTrace]:
        """Return one rich reward trace per parent / fragment pair."""

        if len(prompts_smiles) != len(generated_smiles):
            raise ValueError("prompts_smiles and generated_smiles must have the same length.")
        if parent_labels is not None and len(parent_labels) != len(prompts_smiles):
            raise ValueError("parent_labels must have the same length as prompts_smiles.")

        traces: list[RewardTrace] = []
        for index, (parent_smiles, fragment_smiles) in enumerate(
            zip(prompts_smiles, generated_smiles, strict=True)
        ):
            label = (
                int(parent_labels[index])
                if parent_labels is not None
                else self.default_parent_label
            )
            traces.append(
                self._calculate_single_reward(
                    parent_smiles=str(parent_smiles),
                    generated_smiles=str(fragment_smiles),
                    original_label=label,
                )
            )
        return traces

    def compute_rewards_from_decoded(
        self,
        *,
        parent_smiles: Sequence[str],
        generated_fragments: Sequence[str],
        labels: Sequence[int] | None = None,
        metas: Sequence[dict[str, Any]] | None = None,
        device: Any = None,
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Return a reward tensor and structured per-sample logs for decoded PPO.

        This keeps the chemistry-reward path explicit:

        decoded fragment strings -> rich reward traces -> tensor for PPO update.
        """

        traces = self.calculate_reward_details_batch(
            parent_smiles,
            generated_fragments,
            parent_labels=labels,
        )

        reward_logs: list[dict[str, Any]] = []
        for index, trace in enumerate(traces):
            meta = metas[index] if metas is not None and index < len(metas) else {}
            breakdown = dict(trace.breakdown)
            reward_logs.append(
                {
                    "id": meta.get("id", meta.get("index", index)),
                    "parent_smiles": trace.parent_smiles,
                    "fragment": trace.raw_fragment_smiles or trace.normalized_generated_smiles,
                    "raw_fragment": trace.raw_fragment_smiles or trace.normalized_generated_smiles,
                    "core_fragment": trace.core_fragment_smiles,
                    "format": breakdown.get("format_r"),
                    "valid": breakdown.get("valid_r"),
                    "substructure": breakdown.get("subgraph_r"),
                    "length": breakdown.get("length_r"),
                    "semantic": breakdown.get("sem_r"),
                    "teacher_sem": breakdown.get("teacher_sem_r"),
                    "fragment_teacher_sem": breakdown.get("fragment_teacher_sem_r"),
                    "counterfactual_sem": breakdown.get("cf_r"),
                    "total": float(trace.reward),
                    "parse_ok": bool(trace.raw_parse_ok),
                    "core_parse_ok": bool(trace.core_parse_ok),
                    "raw_has_dummy": bool(trace.raw_has_dummy),
                    "raw_dummy_count": int(trace.raw_dummy_count),
                    "parse_stage": trace.parse_stage,
                    "parsed_raw_with_dummy": bool(trace.parsed_raw_with_dummy),
                    "parsed_core": bool(trace.parsed_core),
                    "dummy_removed_before_parse": bool(trace.dummy_removed_before_parse),
                    "parse_failed_reason": trace.parse_failed_reason,
                    "substructure_ok": bool(trace.is_subgraph),
                    "connected_ok": bool(trace.connected_fragment),
                    "deletion_ok": bool(trace.deletion_success),
                    "has_dummy": bool(trace.has_dummy_atoms),
                    "dummy_count": int(trace.dummy_count),
                    "core_atom_count": int(trace.core_atom_count),
                    "teacher_input_smiles": trace.teacher_input_smiles,
                    "teacher_available": bool(trace.teacher_available),
                    "teacher_called": bool(trace.teacher_called),
                    "teacher_prob": trace.teacher_probability,
                    "teacher_label": trace.teacher_predicted_label,
                    "teacher_reason": trace.teacher_reason,
                    "parent_without_fragment_smiles": trace.parent_without_fragment_smiles,
                    "counterfactual_teacher_available": bool(trace.counterfactual_teacher_available),
                    "counterfactual_called": bool(trace.counterfactual_teacher_called),
                    "counterfactual_reason": trace.counterfactual_teacher_reason,
                    "empty_response": bool(trace.empty_response),
                    "full_parent": bool(trace.full_parent),
                    "empty_residual": bool(trace.empty_residual),
                    "oracle_ok": bool(trace.oracle_ok),
                    "p_before": trace.p_before,
                    "p_after": trace.p_after,
                    "pred_before": trace.pred_before,
                    "pred_after": trace.pred_after,
                    "cf_drop": trace.cf_drop,
                    "cf_flip": bool(trace.cf_flip),
                    "reward_total": float(trace.reward),
                    "target_probability": trace.target_probability,
                    "failure_stage": trace.failure_stage,
                    "failure_tag": trace.failure_tag,
                    "invalid_detail": trace.invalid_detail,
                    "generated_char_count": int(trace.generated_char_count),
                    "repair_attempted": bool(trace.repair_attempted),
                    "repair_success": bool(trace.repair_success),
                    "repaired_fragment": trace.repaired_fragment_smiles,
                    "repair_source": trace.repair_source,
                    "repair_similarity": trace.repair_similarity,
                    "repair_reason": trace.repair_reason,
                    "repair_method": trace.repair_method,
                    "repair_edit_distance": int(trace.repair_edit_distance),
                    "repair_suffix_trim_count": int(trace.repair_suffix_trim_count),
                    "repair_added_parentheses": int(trace.repair_added_parentheses),
                    "repair_added_ring_closures": int(trace.repair_added_ring_closures),
                    "repaired_raw_fragment": trace.repaired_raw_fragment,
                    "repaired_fragment_chars": int(trace.repaired_fragment_chars),
                    "repaired_parse_stage": trace.repaired_parse_stage,
                    "repaired_parsed_raw": bool(trace.repaired_parsed_raw),
                    "repaired_parsed_core": bool(trace.repaired_parsed_core),
                    "repair_failure_reason": trace.repair_failure_reason,
                    "repair_failure_stage": trace.repair_failure_stage,
                    "repair_candidate_count": int(trace.repair_candidate_count),
                    "repair_candidates_parse_ok": int(trace.repair_candidates_parse_ok),
                    "repair_candidates_core_ok": int(trace.repair_candidates_core_ok),
                    "repair_candidates_parent_ok": int(trace.repair_candidates_parent_ok),
                    "repair_candidates_projection_ok": int(trace.repair_candidates_projection_ok),
                    "repair_best_candidate": trace.repair_best_candidate,
                    "repair_accept_stage": trace.repair_accept_stage,
                    "repair_candidate_accepted": bool(trace.repair_candidate_accepted),
                    "repair_candidate_rejected_reason": trace.repair_candidate_rejected_reason,
                    "component_salvage_attempted": bool(trace.component_salvage_attempted),
                    "component_salvage_success": bool(trace.component_salvage_success),
                    "component_count": int(trace.component_count),
                    "raw_component_count": int(trace.raw_component_count),
                    "core_component_count": int(trace.core_component_count),
                    "salvage_method": trace.salvage_method,
                    "salvaged_fragment": trace.salvaged_fragment,
                    "salvaged_atom_count": trace.salvaged_atom_count,
                    "component_salvage_failure_reason": trace.component_salvage_failure_reason,
                    "component_salvage_stage": trace.component_salvage_stage,
                    "component_salvage_candidate_count": int(trace.component_salvage_candidate_count),
                    "component_salvage_best_candidate": trace.component_salvage_best_candidate,
                    "multi_dummy_hard_fail": bool(trace.multi_dummy_hard_fail),
                    "dummy_salvage_attempted": bool(trace.dummy_salvage_attempted),
                    "dummy_salvage_success": bool(trace.dummy_salvage_success),
                    "dummy_salvage_method": trace.dummy_salvage_method,
                    "dummy_salvaged_fragment": trace.dummy_salvaged_fragment,
                    "near_parent_hard_fail": bool(trace.near_parent_hard_fail),
                    "residual_atom_count": trace.residual_atom_count,
                    "residual_atom_ratio": trace.residual_atom_ratio,
                    "tiny_fragment_hard_fail": bool(trace.tiny_fragment_hard_fail),
                    "fragment_atom_count": int(trace.fragment_atom_count),
                    "min_fragment_atoms": int(trace.min_fragment_atoms),
                    "projection_attempted": bool(trace.projection_attempted),
                    "projection_success": bool(trace.projection_success),
                    "projection_method": trace.projection_method,
                    "projection_score": trace.projection_score,
                    "projection_source": trace.projection_source,
                    "projected_fragment": trace.projected_fragment_smiles,
                    "projection_atom_count": trace.projection_atom_count,
                    "projection_atom_ratio": trace.projection_atom_ratio,
                    "projection_penalty": trace.projection_penalty,
                    "num_projection_candidates": int(trace.num_projection_candidates),
                    "projection_reason": trace.projection_reason,
                    "size_window_reward": trace.size_window_reward,
                    "size_window_bucket": trace.size_window_bucket,
                    "size_window_low": trace.size_window_low,
                    "size_window_high": trace.size_window_high,
                    "final_fragment_atom_count": int(trace.final_fragment_atom_count),
                    "final_fragment_atom_ratio": trace.final_fragment_atom_ratio,
                    "error_message": trace.error_message,
                    "original_label": trace.original_label,
                }
            )

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on training env
            raise RuntimeError(
                "compute_rewards_from_decoded requires torch in the PPO runtime."
            ) from exc

        reward_tensor = torch.tensor(
            [float(trace.reward) for trace in traces],
            dtype=torch.float32,
            device=device,
        )
        return reward_tensor, reward_logs

    def _calculate_single_reward(
        self,
        *,
        parent_smiles: str,
        generated_smiles: str,
        original_label: int,
    ) -> RewardTrace:
        """Compute one reward with step-by-step early stopping."""

        normalized_parent = str(parent_smiles or "").strip()
        normalized_generated, generated_char_count = preprocess_generated_fragment(
            generated_smiles
        )
        if int(original_label) not in (0, 1):
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=self.default_parent_label,
                failure_stage="label",
                error_message=f"Unsupported original label: {original_label}",
                generated_char_count=generated_char_count,
                breakdown=self._build_breakdown(
                    format_reward=0.0,
                    valid_reward=0.0,
                    subgraph_reward=0.0,
                    length_reward=0.0,
                    semantic_reward=0.0,
                    fragment_teacher_reward=0.0,
                    counterfactual_reward=self.minimum_reward,
                ),
                teacher_reason="invalid_or_missing_label",
                counterfactual_teacher_reason="invalid_or_missing_label",
            )

        target_label = 1 - int(original_label)

        if not normalized_parent:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="parent",
                error_message="Parent SMILES is empty.",
                generated_char_count=generated_char_count,
                breakdown=self._build_breakdown(
                    format_reward=0.0,
                    valid_reward=0.0,
                    subgraph_reward=0.0,
                    length_reward=0.0,
                    semantic_reward=0.0,
                    fragment_teacher_reward=0.0,
                    counterfactual_reward=self.minimum_reward,
                ),
                teacher_reason="invalid_or_missing_label",
                counterfactual_teacher_reason="invalid_or_missing_label",
            )

        if not normalized_generated:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="validity",
                error_message="Generated fragment is empty.",
                generated_char_count=generated_char_count,
                breakdown=self._build_breakdown(
                    format_reward=self.format_penalty,
                    valid_reward=self.invalid_smiles_penalty,
                    subgraph_reward=0.0,
                    length_reward=0.0,
                    semantic_reward=self.teacher_sem_missing_penalty,
                    fragment_teacher_reward=self.teacher_sem_missing_penalty,
                    counterfactual_reward=self.teacher_sem_missing_penalty,
                ),
                empty_response=True,
                teacher_reason="empty_response",
                counterfactual_teacher_reason="empty_response",
            )

        if generated_char_count > self.max_generation_chars:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="validity",
                error_message=(
                    "Generated fragment exceeded the reward-side character limit: "
                    f"{generated_char_count} > {self.max_generation_chars}"
                ),
                generated_char_count=generated_char_count,
                failure_tag="invalid_generation_too_long",
                invalid_detail=(
                    f"fragment_length_exceeds_limit:{generated_char_count}>{self.max_generation_chars}"
                ),
                breakdown=self._build_breakdown(
                    format_reward=self.format_penalty,
                    valid_reward=self.invalid_smiles_penalty,
                    subgraph_reward=0.0,
                    length_reward=0.0,
                    semantic_reward=self.teacher_sem_missing_penalty,
                    fragment_teacher_reward=self.teacher_sem_missing_penalty,
                    counterfactual_reward=self.teacher_sem_missing_penalty,
                ),
                raw_fragment_smiles=normalized_generated,
                teacher_input_smiles=normalized_generated,
                teacher_reason="invalid_generation_too_long",
                counterfactual_teacher_reason="invalid_generation_too_long",
            )

        # Step A: parseability and connectivity.
        parent = parse_smiles(
            normalized_parent,
            sanitize=True,
            canonicalize=False,
            allow_capped_fragments=False,
        )
        if not parent.sanitized or parent.mol is None:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="parent",
                error_message=(
                    "Parent SMILES could not be parsed by RDKit."
                    if not parent.failure_reason
                    else f"Parent SMILES is not usable: {parent.failure_reason}"
                ),
                generated_char_count=generated_char_count,
                breakdown=self._build_breakdown(
                    format_reward=0.0,
                    valid_reward=0.0,
                    subgraph_reward=0.0,
                    length_reward=0.0,
                    semantic_reward=0.0,
                    fragment_teacher_reward=0.0,
                    counterfactual_reward=self.minimum_reward,
                ),
                teacher_reason="invalid_or_missing_label",
            )
        parent_canonical_smiles = mol_to_smiles_safe(parent.mol)

        decoded_raw_fragment = normalized_generated
        effective_fragment = normalized_generated
        effective_generated_char_count = generated_char_count
        repair_trace_kwargs = self._repair_trace_kwargs()
        repair_attempt_consumed = False
        syntax_repair_attempt_consumed = False
        component_salvage_trace_kwargs = self._component_salvage_trace_kwargs()
        component_salvage_attempt_consumed = False
        dummy_trace_kwargs = self._dummy_trace_kwargs()
        residual_guard_trace_kwargs = self._residual_guard_trace_kwargs()
        projection_trace_kwargs = self._projection_trace_kwargs()
        size_window_trace_kwargs = self._size_window_trace_kwargs()
        projection_attempt_consumed = False
        projection_penalty_applied = 0.0

        while True:
            fragment_info = normalize_fragment_with_dummy_atoms(effective_fragment)
            format_reward = self._compute_format_reward(effective_fragment, fragment_info)
            valid_reward = self._compute_valid_reward(fragment_info)
            length_reward = self._compute_length_reward(fragment_info["core_atom_count"])
            base_trace_kwargs = self._fragment_trace_kwargs(
                fragment_info,
                raw_fragment_smiles_override=decoded_raw_fragment,
                teacher_input_smiles=fragment_info["core_smiles"],
                teacher_reason="invalid_or_not_substructure",
            )
            base_trace_kwargs.update(
                self._counterfactual_trace_kwargs(
                    counterfactual_teacher_available=bool(
                        self.counterfactual_teacher_scorer is not None
                        and self.counterfactual_teacher_scorer.available
                    ),
                    counterfactual_teacher_called=False,
                    counterfactual_teacher_reason="invalid_or_not_substructure",
                )
            )
            base_trace_kwargs.update(repair_trace_kwargs)
            base_trace_kwargs.update(component_salvage_trace_kwargs)
            base_trace_kwargs.update(dummy_trace_kwargs)
            base_trace_kwargs.update(residual_guard_trace_kwargs)
            base_trace_kwargs.update(projection_trace_kwargs)
            base_trace_kwargs.update(size_window_trace_kwargs)

            if not fragment_info["raw_parse_ok"]:
                parse_failure_reason = str(
                    fragment_info.get("parse_failed_reason")
                    or classify_parse_failed_reason(
                        raw_fragment_smiles=effective_fragment,
                        raw_has_dummy=bool(fragment_info.get("raw_has_dummy")),
                        parse_stage=fragment_info.get("parse_stage"),
                    )
                )
                if (
                    not syntax_repair_attempt_consumed
                    and self.enable_minimal_syntax_repair
                ):
                    syntax_repair_result, repair_trace_kwargs = (
                        self._attempt_minimal_syntax_repair(
                            parent_smiles=normalized_parent,
                            raw_fragment_smiles=effective_fragment,
                            parse_failed_reason=parse_failure_reason,
                        )
                    )
                    syntax_repair_attempt_consumed = True
                    if (
                        syntax_repair_result is not None
                        and syntax_repair_result.success
                        and syntax_repair_result.repaired_fragment_smiles
                        and syntax_repair_result.repaired_fragment_smiles
                        != effective_fragment
                    ):
                        effective_fragment = str(
                            syntax_repair_result.repaired_fragment_smiles
                        ).strip()
                        effective_generated_char_count = len(effective_fragment)
                        continue
                    base_trace_kwargs.update(repair_trace_kwargs)

                return self._fail(
                    parent_smiles=normalized_parent,
                    generated_smiles=generated_smiles,
                    normalized_generated=effective_fragment,
                    original_label=int(original_label),
                    failure_stage="validity",
                    error_message=(
                        "Generated fragment could not be parsed by RDKit."
                        if parse_failure_reason is None
                        else (
                            "Generated fragment could not be parsed by RDKit. "
                            f"Likely cause: {parse_failure_reason}."
                        )
                    ),
                    generated_char_count=effective_generated_char_count,
                    failure_tag=(
                        "parse_failed_after_minimal_repair"
                        if bool(base_trace_kwargs.get("repair_attempted"))
                        else self._parse_failure_tag(parse_failure_reason)
                    ),
                    invalid_detail=(
                        f"{parse_failure_reason}:minimal_syntax_repair_failed"
                        if bool(base_trace_kwargs.get("repair_attempted"))
                        else parse_failure_reason
                    ),
                    breakdown=self._build_breakdown(
                        format_reward=format_reward,
                        valid_reward=valid_reward,
                        subgraph_reward=0.0,
                        length_reward=0.0,
                        semantic_reward=self.teacher_sem_missing_penalty,
                        fragment_teacher_reward=self.teacher_sem_missing_penalty,
                        counterfactual_reward=self.teacher_sem_missing_penalty,
                    ),
                    **base_trace_kwargs,
                )

            raw_dummy_count = int(fragment_info.get("raw_dummy_count", 0) or 0)
            if raw_dummy_count >= self.multi_dummy_hard_fail_threshold:
                dummy_trace_kwargs = self._dummy_trace_kwargs(
                    multi_dummy_hard_fail=True,
                    dummy_salvage_attempted=bool(self.enable_light_dummy_salvage),
                    dummy_salvage_success=False,
                )
                base_trace_kwargs.update(dummy_trace_kwargs)
                return self._fail(
                    parent_smiles=normalized_parent,
                    generated_smiles=generated_smiles,
                    normalized_generated=effective_fragment,
                    original_label=int(original_label),
                    failure_stage="validity",
                    error_message=(
                        "Generated fragment contains too many dummy atoms: "
                        f"{raw_dummy_count} >= {self.multi_dummy_hard_fail_threshold}"
                    ),
                    generated_char_count=effective_generated_char_count,
                    failure_tag="parse_failed_after_dummy_removal",
                    invalid_detail="multi_dummy_hard_fail",
                    breakdown=self._build_breakdown(
                        format_reward=format_reward,
                        valid_reward=self.invalid_smiles_penalty,
                        subgraph_reward=0.0,
                        length_reward=0.0,
                        semantic_reward=self.teacher_sem_missing_penalty,
                        fragment_teacher_reward=self.teacher_sem_missing_penalty,
                        counterfactual_reward=self.teacher_sem_missing_penalty,
                    ),
                    **base_trace_kwargs,
                )

            raw_component_count = int(fragment_info.get("raw_component_count", 0) or 0)
            core_component_count = int(fragment_info.get("core_component_count", 0) or 0)
            raw_disconnected = raw_component_count > 1
            core_smiles = fragment_info["core_smiles"]
            core_mol = fragment_info["core_mol"]
            core_usable = bool(
                core_mol is not None and fragment_info["core_parse_ok"] and core_smiles
            )
            core_disconnected = bool(core_usable and core_component_count > 1)

            if raw_disconnected or core_disconnected:
                salvage_stage = "raw" if raw_disconnected else "core"
                component_fragment = (
                    effective_fragment
                    if salvage_stage == "raw"
                    else str(fragment_info.get("core_smiles") or effective_fragment)
                )
                if (
                    not component_salvage_attempt_consumed
                    and self.enable_component_salvage
                ):
                    salvage_result, component_salvage_trace_kwargs = (
                        self._attempt_component_salvage(
                            parent_smiles=normalized_parent,
                            raw_fragment_smiles=effective_fragment,
                            component_fragment_smiles=component_fragment,
                            raw_component_count=raw_component_count,
                            core_component_count=core_component_count,
                            component_salvage_stage=salvage_stage,
                        )
                    )
                    component_salvage_attempt_consumed = True
                    if (
                        salvage_result is not None
                        and salvage_result.success
                        and salvage_result.salvaged_fragment_smiles
                        and salvage_result.salvaged_fragment_smiles != effective_fragment
                    ):
                        effective_fragment = str(
                            salvage_result.salvaged_fragment_smiles
                        ).strip()
                        effective_generated_char_count = len(effective_fragment)
                        continue
                    base_trace_kwargs.update(component_salvage_trace_kwargs)

                return self._fail(
                    parent_smiles=normalized_parent,
                    generated_smiles=generated_smiles,
                    normalized_generated=effective_fragment,
                    original_label=int(original_label),
                    failure_stage="validity",
                    error_message="Generated fragment is not connected.",
                    generated_char_count=effective_generated_char_count,
                    failure_tag="fragment_not_connected",
                    invalid_detail="fragment_not_connected",
                    breakdown=self._build_breakdown(
                        format_reward=format_reward,
                        valid_reward=valid_reward,
                        subgraph_reward=0.0,
                        length_reward=length_reward,
                        semantic_reward=self.teacher_sem_missing_penalty,
                        fragment_teacher_reward=self.teacher_sem_missing_penalty,
                        counterfactual_reward=self.teacher_sem_missing_penalty,
                    ),
                    **base_trace_kwargs,
                )

            # Step B: subgraph match against the parent molecule.
            if not core_usable:
                dummy_trace_kwargs = self._dummy_trace_kwargs(
                    dummy_salvage_attempted=bool(self.enable_light_dummy_salvage),
                    dummy_salvage_success=False,
                    dummy_salvage_method=(
                        "disabled" if not self.enable_light_dummy_salvage else None
                    ),
                )
                base_trace_kwargs.update(dummy_trace_kwargs)
                if not repair_attempt_consumed and self.enable_parent_aware_repair:
                    repair_result, repair_trace_kwargs = self._attempt_parent_aware_repair(
                        parent_smiles=normalized_parent,
                        raw_fragment_smiles=decoded_raw_fragment,
                    )
                    repair_attempt_consumed = True
                    if (
                        repair_result is not None
                        and repair_result.success
                        and repair_result.repaired_fragment_smiles
                        and repair_result.repaired_fragment_smiles != effective_fragment
                    ):
                        effective_fragment = str(repair_result.repaired_fragment_smiles).strip()
                        effective_generated_char_count = len(effective_fragment)
                        continue
                    base_trace_kwargs.update(repair_trace_kwargs)

                if (
                    bool(repair_trace_kwargs.get("repair_success"))
                    and repair_trace_kwargs.get("repair_source") == "minimal_syntax"
                ):
                    repair_trace_kwargs = dict(repair_trace_kwargs)
                    repair_trace_kwargs.update(
                        {
                            "repair_failure_reason": "repair_candidate_core_unusable",
                            "repair_failure_stage": "core_normalization",
                            "repair_candidate_accepted": False,
                            "repair_candidate_rejected_reason": "repair_candidate_core_unusable",
                        }
                    )
                    base_trace_kwargs.update(repair_trace_kwargs)

                invalid_detail = (
                    "core_fragment_unusable_after_dummy_normalization"
                    if bool(fragment_info.get("raw_has_dummy"))
                    else "core_fragment_unusable_after_normalization"
                )
                return self._fail(
                    parent_smiles=normalized_parent,
                    generated_smiles=generated_smiles,
                    normalized_generated=effective_fragment,
                    original_label=int(original_label),
                    failure_stage="subgraph",
                    error_message="Generated fragment did not yield a usable core fragment after dummy-atom normalization.",
                    generated_char_count=effective_generated_char_count,
                    failure_tag=(
                        "parse_failed_after_dummy_removal"
                        if bool(fragment_info.get("raw_has_dummy"))
                        else "parse_ok_but_not_substructure"
                    ),
                    invalid_detail=invalid_detail,
                    breakdown=self._build_breakdown(
                        format_reward=format_reward,
                        valid_reward=valid_reward,
                        subgraph_reward=self.invalid_subgraph_penalty,
                        length_reward=length_reward,
                        semantic_reward=self.teacher_sem_missing_penalty,
                        fragment_teacher_reward=self.teacher_sem_missing_penalty,
                        counterfactual_reward=self.teacher_sem_missing_penalty,
                    ),
                    valid_smiles=True,
                    connected_fragment=not raw_disconnected,
                    **base_trace_kwargs,
                )

            is_full_parent = bool(
                core_smiles
                and parent_canonical_smiles
                and core_smiles == parent_canonical_smiles
            )
            if is_full_parent:
                full_parent_trace_kwargs = self._merge_reward_fields(
                    base_trace_kwargs,
                    repair_trace_kwargs,
                    component_salvage_trace_kwargs,
                    dummy_trace_kwargs,
                    residual_guard_trace_kwargs,
                    projection_trace_kwargs,
                    size_window_trace_kwargs,
                    raw_fragment_smiles=decoded_raw_fragment,
                    core_fragment_smiles=core_smiles,
                    teacher_input_smiles=core_smiles,
                    teacher_reason="full_parent_fragment",
                    parent_without_fragment_smiles="",
                    counterfactual_teacher_available=bool(
                        self.counterfactual_teacher_scorer is not None
                        and self.counterfactual_teacher_scorer.available
                    ),
                    counterfactual_teacher_called=False,
                    counterfactual_teacher_reason="full_parent_fragment",
                    full_parent=True,
                    empty_residual=True,
                    residual_atom_count=0,
                    residual_atom_ratio=0.0,
                )
                return self._fail(
                    parent_smiles=normalized_parent,
                    generated_smiles=generated_smiles,
                    normalized_generated=effective_fragment,
                    original_label=int(original_label),
                    failure_stage="counterfactual",
                    error_message="Generated fragment matches the full parent molecule.",
                    generated_char_count=effective_generated_char_count,
                    breakdown=self._build_breakdown(
                        format_reward=format_reward,
                        valid_reward=valid_reward,
                        subgraph_reward=self.subgraph_pass_reward,
                        length_reward=length_reward,
                        semantic_reward=self.full_parent_penalty,
                        fragment_teacher_reward=self.teacher_sem_missing_penalty,
                        counterfactual_reward=self.full_parent_penalty,
                    ),
                    valid_smiles=True,
                    connected_fragment=True,
                    is_subgraph=True,
                    residual_smiles="",
                    **full_parent_trace_kwargs,
                )

            try:
                has_precise_match = bool(
                    parent.mol.HasSubstructMatch(core_mol, useChirality=True)
                    and is_parent_substructure(normalized_parent, core_smiles)
                )
            except Exception as exc:
                return self._fail(
                    parent_smiles=normalized_parent,
                    generated_smiles=generated_smiles,
                    normalized_generated=effective_fragment,
                    original_label=int(original_label),
                    failure_stage="subgraph",
                    error_message=f"Subgraph check failed: {exc}",
                    generated_char_count=effective_generated_char_count,
                    failure_tag="parse_ok_but_not_substructure",
                    invalid_detail="subgraph_check_failed",
                    breakdown=self._build_breakdown(
                        format_reward=format_reward,
                        valid_reward=valid_reward,
                        subgraph_reward=self.invalid_subgraph_penalty,
                        length_reward=length_reward,
                        semantic_reward=self.teacher_sem_missing_penalty,
                        fragment_teacher_reward=self.teacher_sem_missing_penalty,
                        counterfactual_reward=self.teacher_sem_missing_penalty,
                    ),
                    **base_trace_kwargs,
                )

            if has_precise_match and not bool(
                projection_trace_kwargs.get("projection_success", False)
            ):
                parent_atom_count = max(1, int(parent.mol.GetNumAtoms()))
                projection_trace_kwargs = self._projection_trace_kwargs(
                    projection_attempted=False,
                    projection_success=True,
                    projection_method="identity",
                    projected_fragment_smiles=core_smiles,
                    projection_source="identity",
                    projection_score=1.0,
                    projection_reason="already_strict_parent_substructure",
                    projected_atom_count=int(fragment_info["core_atom_count"]),
                    projected_atom_ratio=(
                        int(fragment_info["core_atom_count"]) / parent_atom_count
                    ),
                    projection_penalty=0.0,
                    num_projection_candidates=0,
                )
                base_trace_kwargs.update(projection_trace_kwargs)

            if not has_precise_match:
                projection_invalid_detail = "not_parent_substructure"
                if not projection_attempt_consumed and self.enable_parent_projection:
                    projection_result, projection_trace_kwargs = self._attempt_parent_projection(
                        parent_smiles=normalized_parent,
                        raw_fragment_smiles=effective_fragment,
                    )
                    projection_attempt_consumed = True
                    projection_invalid_detail = (
                        getattr(projection_result, "reason", None)
                        or "projection_failed"
                    )
                    if (
                        projection_result is not None
                        and projection_result.success
                        and projection_result.projected_fragment_smiles
                        and projection_result.projected_fragment_smiles != effective_fragment
                    ):
                        effective_fragment = str(
                            projection_result.projected_fragment_smiles
                        ).strip()
                        effective_generated_char_count = len(effective_fragment)
                        projection_penalty_applied = (
                            self.projection_penalty
                            if projection_result.projection_method == "retrieval"
                            else 0.0
                        )
                        projection_trace_kwargs["projection_penalty"] = (
                            projection_penalty_applied
                        )
                        continue
                    base_trace_kwargs.update(projection_trace_kwargs)

                if not repair_attempt_consumed and self.enable_parent_aware_repair:
                    repair_result, repair_trace_kwargs = self._attempt_parent_aware_repair(
                        parent_smiles=normalized_parent,
                        raw_fragment_smiles=decoded_raw_fragment,
                    )
                    repair_attempt_consumed = True
                    if (
                        repair_result is not None
                        and repair_result.success
                        and repair_result.repaired_fragment_smiles
                        and repair_result.repaired_fragment_smiles != effective_fragment
                    ):
                        effective_fragment = str(repair_result.repaired_fragment_smiles).strip()
                        effective_generated_char_count = len(effective_fragment)
                        continue
                    base_trace_kwargs.update(repair_trace_kwargs)

                if (
                    bool(repair_trace_kwargs.get("repair_success"))
                    and repair_trace_kwargs.get("repair_source") == "minimal_syntax"
                ):
                    repair_trace_kwargs = dict(repair_trace_kwargs)
                    repair_trace_kwargs.update(
                        {
                            "repair_failure_reason": "repair_candidate_core_unusable",
                            "repair_failure_stage": "core_normalization",
                            "repair_candidate_accepted": False,
                            "repair_candidate_rejected_reason": "repair_candidate_core_unusable",
                        }
                    )
                    base_trace_kwargs.update(repair_trace_kwargs)

                return self._fail(
                    parent_smiles=normalized_parent,
                    generated_smiles=generated_smiles,
                    normalized_generated=effective_fragment,
                    original_label=int(original_label),
                    failure_stage="subgraph",
                    error_message="Generated fragment is not a valid parent subgraph.",
                    generated_char_count=effective_generated_char_count,
                    failure_tag=(
                        "projection_failed_low_score"
                        if projection_invalid_detail == "projection_failed_low_score"
                        else "parse_ok_but_not_substructure"
                    ),
                    invalid_detail=projection_invalid_detail,
                    breakdown=self._build_breakdown(
                        format_reward=format_reward,
                        valid_reward=valid_reward,
                        subgraph_reward=self.invalid_subgraph_penalty,
                        length_reward=length_reward,
                        semantic_reward=self.teacher_sem_missing_penalty,
                        fragment_teacher_reward=self.teacher_sem_missing_penalty,
                        counterfactual_reward=self.teacher_sem_missing_penalty,
                    ),
                    valid_smiles=True,
                    connected_fragment=True,
                    **base_trace_kwargs,
                )

            parent_atom_count = max(1, int(parent.mol.GetNumAtoms()))
            core_atom_count = int(fragment_info["core_atom_count"])
            if (
                self.min_fragment_atoms > 0
                and core_atom_count < self.min_fragment_atoms
            ):
                residual_guard_trace_kwargs = self._residual_guard_trace_kwargs(
                    tiny_fragment_hard_fail=True,
                    fragment_atom_count=core_atom_count,
                )
                size_window_trace_kwargs = self._size_window_trace_kwargs(
                    size_window_reward=0.0,
                    size_window_bucket="hard_failed_tiny",
                    final_fragment_atom_count=core_atom_count,
                    final_fragment_atom_ratio=(core_atom_count / parent_atom_count),
                )
                base_trace_kwargs.update(residual_guard_trace_kwargs)
                base_trace_kwargs.update(size_window_trace_kwargs)
                tiny_guard_trace_kwargs = dict(base_trace_kwargs)
                tiny_guard_trace_kwargs.update(
                    {
                        "raw_fragment_smiles": decoded_raw_fragment,
                        "core_fragment_smiles": core_smiles,
                        "teacher_input_smiles": core_smiles,
                        "teacher_reason": "tiny_fragment_hard_fail",
                        "counterfactual_teacher_available": bool(
                            self.counterfactual_teacher_scorer is not None
                            and self.counterfactual_teacher_scorer.available
                        ),
                        "counterfactual_teacher_called": False,
                        "counterfactual_teacher_reason": "tiny_fragment_hard_fail",
                    }
                )
                return self._fail(
                    parent_smiles=normalized_parent,
                    generated_smiles=generated_smiles,
                    normalized_generated=effective_fragment,
                    original_label=int(original_label),
                    failure_stage="counterfactual",
                    error_message=(
                        "Generated fragment is below the configured minimum "
                        f"atom count: {core_atom_count} < {self.min_fragment_atoms}."
                    ),
                    generated_char_count=effective_generated_char_count,
                    failure_tag="tiny_fragment_hard_fail",
                    invalid_detail="tiny_fragment_hard_fail",
                    breakdown=self._build_breakdown(
                        format_reward=0.0,
                        valid_reward=0.0,
                        subgraph_reward=0.0,
                        length_reward=0.0,
                        semantic_reward=self.tiny_fragment_hard_fail_penalty,
                        fragment_teacher_reward=self.teacher_sem_missing_penalty,
                        counterfactual_reward=self.tiny_fragment_hard_fail_penalty,
                    ),
                    valid_smiles=True,
                    connected_fragment=True,
                    is_subgraph=True,
                    **tiny_guard_trace_kwargs,
                )
            atom_ratio = core_atom_count / parent_atom_count
            deletion_for_guard = delete_fragment_from_parent(
                normalized_parent,
                core_smiles,
                max_matches=1,
            )
            residual_atom_count = deletion_for_guard.residual_atom_count
            residual_atom_ratio = (
                residual_atom_count / parent_atom_count
                if residual_atom_count is not None
                else None
            )
            residual_guard_trace_kwargs = self._residual_guard_trace_kwargs(
                residual_atom_count=residual_atom_count,
                residual_atom_ratio=residual_atom_ratio,
                fragment_atom_count=core_atom_count,
            )
            base_trace_kwargs.update(residual_guard_trace_kwargs)
            hard_guard_tag: str | None = None
            hard_guard_penalty = self.full_parent_penalty
            near_parent_hard_fail = False
            if atom_ratio >= self.near_parent_hard_ratio:
                hard_guard_tag = "near_parent_fragment"
                near_parent_hard_fail = True
                hard_guard_penalty = self.full_parent_penalty
            elif residual_atom_count is not None and residual_atom_count < self.min_residual_atoms:
                hard_guard_tag = "tiny_residual_fragment"
                hard_guard_penalty = self.empty_residual_penalty
            elif (
                residual_atom_ratio is not None
                and residual_atom_ratio <= self.min_residual_ratio
            ):
                hard_guard_tag = "tiny_residual_fragment"
                hard_guard_penalty = self.empty_residual_penalty

            if hard_guard_tag is not None:
                residual_guard_trace_kwargs = self._residual_guard_trace_kwargs(
                    near_parent_hard_fail=near_parent_hard_fail,
                    residual_atom_count=residual_atom_count,
                    residual_atom_ratio=residual_atom_ratio,
                    fragment_atom_count=core_atom_count,
                )
                size_window_trace_kwargs = self._size_window_trace_kwargs(
                    size_window_reward=0.0,
                    size_window_bucket=(
                        "hard_failed_near_parent" if near_parent_hard_fail else "unknown"
                    ),
                    final_fragment_atom_count=core_atom_count,
                    final_fragment_atom_ratio=atom_ratio,
                )
                base_trace_kwargs.update(residual_guard_trace_kwargs)
                base_trace_kwargs.update(size_window_trace_kwargs)
                hard_guard_trace_kwargs = dict(base_trace_kwargs)
                hard_guard_trace_kwargs.update(
                    {
                        "raw_fragment_smiles": decoded_raw_fragment,
                        "core_fragment_smiles": core_smiles,
                        "teacher_input_smiles": core_smiles,
                        "teacher_reason": hard_guard_tag,
                        "counterfactual_teacher_available": bool(
                            self.counterfactual_teacher_scorer is not None
                            and self.counterfactual_teacher_scorer.available
                        ),
                        "counterfactual_teacher_called": False,
                        "counterfactual_teacher_reason": hard_guard_tag,
                    }
                )
                return self._fail(
                    parent_smiles=normalized_parent,
                    generated_smiles=generated_smiles,
                    normalized_generated=effective_fragment,
                    original_label=int(original_label),
                    failure_stage="counterfactual",
                    error_message=(
                        "Generated fragment is too close to the full parent or leaves a tiny residual."
                    ),
                    generated_char_count=effective_generated_char_count,
                    failure_tag=hard_guard_tag,
                    invalid_detail=hard_guard_tag,
                    breakdown=self._build_breakdown(
                        format_reward=0.0,
                        valid_reward=0.0,
                        subgraph_reward=0.0,
                        length_reward=0.0,
                        semantic_reward=hard_guard_penalty,
                        fragment_teacher_reward=self.teacher_sem_missing_penalty,
                        counterfactual_reward=hard_guard_penalty,
                    ),
                    valid_smiles=True,
                    connected_fragment=True,
                    is_subgraph=True,
                    residual_smiles=deletion_for_guard.residual_smiles,
                    **hard_guard_trace_kwargs,
                )

            size_window_reward, size_window_bucket = self._compute_size_window_reward(
                atom_ratio=atom_ratio,
            )
            size_window_trace_kwargs = self._size_window_trace_kwargs(
                size_window_reward=size_window_reward,
                size_window_bucket=size_window_bucket,
                final_fragment_atom_count=core_atom_count,
                final_fragment_atom_ratio=atom_ratio,
            )
            base_trace_kwargs.update(size_window_trace_kwargs)
            normalized_generated = effective_fragment
            generated_char_count = effective_generated_char_count
            break

        fragment_teacher_reward, teacher_trace_kwargs = self._score_teacher_semantic(
            core_smiles=core_smiles,
            original_label=int(original_label),
            parent_smiles=normalized_parent,
            fragment_info=fragment_info,
        )
        counterfactual_reward, counterfactual_trace_kwargs = (
            self._score_counterfactual_semantic(
                parent_smiles=normalized_parent,
                core_smiles=core_smiles,
                raw_fragment_smiles=normalized_generated,
                original_label=int(original_label),
                fragment_info=fragment_info,
                valid_reward=valid_reward,
                substructure_ok=has_precise_match,
            )
        )
        success_trace_kwargs = {
            **base_trace_kwargs,
            **teacher_trace_kwargs,
            **counterfactual_trace_kwargs,
            "teacher_input_smiles": core_smiles,
            "fragment_teacher_sem": fragment_teacher_reward,
        }
        breakdown = self._build_breakdown(
            format_reward=format_reward,
            valid_reward=valid_reward,
            subgraph_reward=self.subgraph_pass_reward,
            length_reward=length_reward,
            size_window_reward=size_window_reward,
            semantic_reward=counterfactual_reward,
            fragment_teacher_reward=fragment_teacher_reward,
            counterfactual_reward=counterfactual_reward,
        )

        return RewardTrace(
            parent_smiles=normalized_parent,
            generated_smiles=str(generated_smiles),
            normalized_generated_smiles=normalized_generated,
            raw_fragment_smiles=decoded_raw_fragment,
            core_fragment_smiles=core_smiles,
            original_label=int(original_label),
            target_label=target_label,
            reward=self._reward_from_breakdown(breakdown) - projection_penalty_applied,
            valid_smiles=True,
            connected_fragment=True,
            is_subgraph=True,
            deletion_success=bool(
                success_trace_kwargs.get("parent_without_fragment_smiles")
            ),
            counterfactual_evaluated=bool(
                success_trace_kwargs.get("counterfactual_teacher_called", False)
            ),
            flip_success=bool(success_trace_kwargs.get("cf_flip", False)),
            target_probability=success_trace_kwargs.get("p_after"),
            inactive_probability=None,
            residual_smiles=success_trace_kwargs.get("parent_without_fragment_smiles"),
            empty_response=False,
            full_parent=bool(success_trace_kwargs.get("full_parent", False)),
            empty_residual=bool(success_trace_kwargs.get("empty_residual", False)),
            oracle_ok=bool(
                success_trace_kwargs.get("counterfactual_teacher_called", False)
                and success_trace_kwargs.get("counterfactual_teacher_reason") == "ok"
            ),
            raw_parse_ok=bool(fragment_info["raw_parse_ok"]),
            core_parse_ok=bool(fragment_info["core_parse_ok"]),
            has_dummy_atoms=bool(fragment_info["has_dummy"]),
            dummy_count=int(fragment_info["dummy_count"]),
            raw_has_dummy=bool(fragment_info["raw_has_dummy"]),
            raw_dummy_count=int(fragment_info["raw_dummy_count"]),
            parse_stage=fragment_info.get("parse_stage"),
            parsed_raw_with_dummy=bool(fragment_info["parsed_raw_with_dummy"]),
            parsed_core=bool(fragment_info["parsed_core"]),
            dummy_removed_before_parse=bool(fragment_info["dummy_removed_before_parse"]),
            parse_failed_reason=fragment_info.get("parse_failed_reason"),
            core_atom_count=int(fragment_info["core_atom_count"]),
            teacher_input_smiles=core_smiles,
            teacher_available=bool(success_trace_kwargs.get("teacher_available", False)),
            teacher_called=bool(success_trace_kwargs.get("teacher_called", False)),
            teacher_probability=success_trace_kwargs.get("teacher_probability"),
            teacher_predicted_label=success_trace_kwargs.get("teacher_predicted_label"),
            teacher_reason=success_trace_kwargs.get("teacher_reason"),
            fragment_teacher_sem=fragment_teacher_reward,
            parent_without_fragment_smiles=success_trace_kwargs.get(
                "parent_without_fragment_smiles"
            ),
            counterfactual_teacher_available=bool(
                success_trace_kwargs.get("counterfactual_teacher_available", False)
            ),
            counterfactual_teacher_called=bool(
                success_trace_kwargs.get("counterfactual_teacher_called", False)
            ),
            counterfactual_teacher_reason=success_trace_kwargs.get(
                "counterfactual_teacher_reason"
            ),
            p_before=success_trace_kwargs.get("p_before"),
            p_after=success_trace_kwargs.get("p_after"),
            pred_before=success_trace_kwargs.get("pred_before"),
            pred_after=success_trace_kwargs.get("pred_after"),
            cf_drop=success_trace_kwargs.get("cf_drop"),
            cf_flip=bool(success_trace_kwargs.get("cf_flip", False)),
            failure_stage=None,
            generated_char_count=generated_char_count,
            repair_attempted=bool(success_trace_kwargs.get("repair_attempted", False)),
            repair_success=bool(success_trace_kwargs.get("repair_success", False)),
            repaired_fragment_smiles=success_trace_kwargs.get("repaired_fragment_smiles"),
            repair_source=success_trace_kwargs.get("repair_source"),
            repair_similarity=success_trace_kwargs.get("repair_similarity"),
            repair_reason=success_trace_kwargs.get("repair_reason"),
            repair_method=success_trace_kwargs.get("repair_method"),
            repair_edit_distance=int(
                success_trace_kwargs.get("repair_edit_distance", 0) or 0
            ),
            repair_suffix_trim_count=int(
                success_trace_kwargs.get("repair_suffix_trim_count", 0) or 0
            ),
            repair_added_parentheses=int(
                success_trace_kwargs.get("repair_added_parentheses", 0) or 0
            ),
            repair_added_ring_closures=int(
                success_trace_kwargs.get("repair_added_ring_closures", 0) or 0
            ),
            repaired_raw_fragment=success_trace_kwargs.get("repaired_raw_fragment"),
            repaired_fragment_chars=int(
                success_trace_kwargs.get("repaired_fragment_chars", 0) or 0
            ),
            repaired_parse_stage=success_trace_kwargs.get("repaired_parse_stage"),
            repaired_parsed_raw=bool(
                success_trace_kwargs.get("repaired_parsed_raw", False)
            ),
            repaired_parsed_core=bool(
                success_trace_kwargs.get("repaired_parsed_core", False)
            ),
            repair_failure_reason=success_trace_kwargs.get("repair_failure_reason"),
            repair_failure_stage=success_trace_kwargs.get("repair_failure_stage"),
            repair_candidate_count=int(
                success_trace_kwargs.get("repair_candidate_count", 0) or 0
            ),
            repair_candidates_parse_ok=int(
                success_trace_kwargs.get("repair_candidates_parse_ok", 0) or 0
            ),
            repair_candidates_core_ok=int(
                success_trace_kwargs.get("repair_candidates_core_ok", 0) or 0
            ),
            repair_candidates_parent_ok=int(
                success_trace_kwargs.get("repair_candidates_parent_ok", 0) or 0
            ),
            repair_candidates_projection_ok=int(
                success_trace_kwargs.get("repair_candidates_projection_ok", 0) or 0
            ),
            repair_best_candidate=success_trace_kwargs.get("repair_best_candidate"),
            repair_accept_stage=success_trace_kwargs.get("repair_accept_stage"),
            repair_candidate_accepted=bool(
                success_trace_kwargs.get("repair_candidate_accepted", False)
            ),
            repair_candidate_rejected_reason=success_trace_kwargs.get(
                "repair_candidate_rejected_reason"
            ),
            component_salvage_attempted=bool(
                success_trace_kwargs.get("component_salvage_attempted", False)
            ),
            component_salvage_success=bool(
                success_trace_kwargs.get("component_salvage_success", False)
            ),
            component_count=int(success_trace_kwargs.get("component_count", 0) or 0),
            raw_component_count=int(
                success_trace_kwargs.get("raw_component_count", 0) or 0
            ),
            core_component_count=int(
                success_trace_kwargs.get("core_component_count", 0) or 0
            ),
            salvage_method=success_trace_kwargs.get("salvage_method"),
            salvaged_fragment=success_trace_kwargs.get("salvaged_fragment"),
            salvaged_atom_count=success_trace_kwargs.get("salvaged_atom_count"),
            component_salvage_failure_reason=success_trace_kwargs.get(
                "component_salvage_failure_reason"
            ),
            component_salvage_stage=success_trace_kwargs.get(
                "component_salvage_stage"
            ),
            component_salvage_candidate_count=int(
                success_trace_kwargs.get("component_salvage_candidate_count", 0) or 0
            ),
            component_salvage_best_candidate=success_trace_kwargs.get(
                "component_salvage_best_candidate"
            ),
            multi_dummy_hard_fail=bool(
                success_trace_kwargs.get("multi_dummy_hard_fail", False)
            ),
            dummy_salvage_attempted=bool(
                success_trace_kwargs.get("dummy_salvage_attempted", False)
            ),
            dummy_salvage_success=bool(
                success_trace_kwargs.get("dummy_salvage_success", False)
            ),
            dummy_salvage_method=success_trace_kwargs.get("dummy_salvage_method"),
            dummy_salvaged_fragment=success_trace_kwargs.get("dummy_salvaged_fragment"),
            near_parent_hard_fail=bool(
                success_trace_kwargs.get("near_parent_hard_fail", False)
            ),
            residual_atom_count=success_trace_kwargs.get("residual_atom_count"),
            residual_atom_ratio=success_trace_kwargs.get("residual_atom_ratio"),
            tiny_fragment_hard_fail=bool(
                success_trace_kwargs.get("tiny_fragment_hard_fail", False)
            ),
            fragment_atom_count=int(
                success_trace_kwargs.get("fragment_atom_count", 0) or 0
            ),
            min_fragment_atoms=int(
                success_trace_kwargs.get("min_fragment_atoms", self.min_fragment_atoms)
                or 0
            ),
            projection_attempted=bool(
                success_trace_kwargs.get("projection_attempted", False)
            ),
            projection_success=bool(
                success_trace_kwargs.get("projection_success", False)
            ),
            projection_method=success_trace_kwargs.get("projection_method"),
            projection_score=success_trace_kwargs.get("projection_score"),
            projection_source=success_trace_kwargs.get("projection_source"),
            projected_fragment_smiles=success_trace_kwargs.get(
                "projected_fragment_smiles"
            ),
            projection_atom_count=success_trace_kwargs.get("projection_atom_count"),
            projection_atom_ratio=success_trace_kwargs.get("projection_atom_ratio"),
            projection_penalty=float(
                success_trace_kwargs.get(
                    "projection_penalty",
                    projection_penalty_applied,
                )
                or 0.0
            ),
            num_projection_candidates=int(
                success_trace_kwargs.get("num_projection_candidates", 0) or 0
            ),
            projection_reason=success_trace_kwargs.get("projection_reason"),
            size_window_reward=float(
                success_trace_kwargs.get("size_window_reward", 0.0) or 0.0
            ),
            size_window_bucket=success_trace_kwargs.get("size_window_bucket"),
            size_window_low=success_trace_kwargs.get("size_window_low"),
            size_window_high=success_trace_kwargs.get("size_window_high"),
            final_fragment_atom_count=int(
                success_trace_kwargs.get("final_fragment_atom_count", core_atom_count)
                or 0
            ),
            final_fragment_atom_ratio=success_trace_kwargs.get(
                "final_fragment_atom_ratio"
            ),
            error_message=None,
            breakdown=breakdown,
        )

    def _compute_format_reward(
        self,
        fragment_smiles: str,
        fragment_info: dict[str, Any],
    ) -> float:
        normalized = str(fragment_smiles or "").strip()
        if not normalized:
            return self.format_penalty
        if any(separator in normalized for separator in ("\n", "\r", "\t", ";")):
            return self.format_penalty
        if fragment_info.get("raw_parse_ok"):
            return self.format_pass_reward
        return self.format_penalty

    def _compute_valid_reward(self, fragment_info: dict[str, Any]) -> float:
        if fragment_info.get("raw_parse_ok") and fragment_info.get("core_parse_ok"):
            return self.valid_pass_reward
        if fragment_info.get("raw_parse_ok"):
            return self.partial_valid_reward
        return self.invalid_smiles_penalty

    def _compute_length_reward(self, core_atom_count: int) -> float:
        if core_atom_count <= 0:
            return 0.0
        if core_atom_count <= self.compact_atom_target:
            return self.compactness_bonus
        overflow = core_atom_count - self.compact_atom_target
        return float(
            max(
                -self.compactness_bonus,
                self.compactness_bonus - overflow * self.compact_atom_penalty_scale,
            )
        )

    def _compute_size_window_reward(
        self,
        *,
        atom_ratio: float | None,
    ) -> tuple[float, str]:
        if atom_ratio is None:
            return 0.0, "unknown"
        if not self.enable_size_window_reward:
            return 0.0, "unknown"
        if atom_ratio < self.size_window_low:
            return self.size_window_small_penalty, "too_small"
        if atom_ratio > self.size_window_high:
            return self.size_window_large_penalty, "too_large"
        return self.size_window_bonus, "in_window"

    def _infer_failure_tag(
        self,
        *,
        explicit_failure_tag: str | None,
        failure_stage: str,
        empty_response: bool,
        full_parent: bool,
        empty_residual: bool,
    ) -> str | None:
        if explicit_failure_tag:
            return explicit_failure_tag
        if empty_response:
            return "empty_response"
        if full_parent:
            return "full_parent_fragment"
        if empty_residual:
            return "empty_residual"
        if failure_stage in {"validity", "subgraph", "counterfactual"}:
            return "invalid_or_not_substructure"
        return failure_stage or None

    def _parse_failure_tag(self, parse_failure_detail: str | None) -> str:
        return parse_failure_detail or "parse_failed"

    def _merge_reward_fields(
        self,
        *field_groups: dict[str, Any] | None,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Merge trace-field dictionaries so later values override earlier ones."""

        merged: dict[str, Any] = {}
        for field_group in field_groups:
            if field_group:
                merged.update(field_group)
        merged.update(overrides)
        return merged

    def _build_breakdown(
        self,
        *,
        format_reward: float,
        valid_reward: float,
        subgraph_reward: float,
        length_reward: float,
        size_window_reward: float = 0.0,
        semantic_reward: float,
        fragment_teacher_reward: float,
        counterfactual_reward: float,
    ) -> dict[str, float]:
        return {
            "format_r": float(format_reward),
            "valid_r": float(valid_reward),
            "subgraph_r": float(subgraph_reward),
            "length_r": float(length_reward),
            "size_window_r": float(size_window_reward),
            "sem_r": float(semantic_reward),
            "teacher_sem_r": float(semantic_reward),
            "cf_r": float(counterfactual_reward),
            "fragment_teacher_sem_r": float(fragment_teacher_reward),
        }

    def _reward_from_breakdown(self, breakdown: dict[str, float]) -> float:
        return float(
            breakdown.get("format_r", 0.0)
            + breakdown.get("valid_r", 0.0)
            + breakdown.get("subgraph_r", 0.0)
            + breakdown.get("length_r", 0.0)
            + breakdown.get("size_window_r", 0.0)
            + breakdown.get("sem_r", 0.0)
        )

    def _fragment_trace_kwargs(
        self,
        fragment_info: dict[str, Any],
        *,
        raw_fragment_smiles_override: str | None = None,
        teacher_input_smiles: str | None,
        teacher_available: bool = False,
        teacher_called: bool = False,
        teacher_probability: float | None = None,
        teacher_predicted_label: int | None = None,
        teacher_reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "raw_fragment_smiles": (
                raw_fragment_smiles_override
                if raw_fragment_smiles_override is not None
                else fragment_info.get("raw")
            ),
            "core_fragment_smiles": fragment_info.get("core_smiles"),
            "raw_parse_ok": bool(fragment_info.get("raw_parse_ok")),
            "core_parse_ok": bool(fragment_info.get("core_parse_ok")),
            "has_dummy_atoms": bool(fragment_info.get("has_dummy")),
            "dummy_count": int(fragment_info.get("dummy_count", 0)),
            "raw_has_dummy": bool(fragment_info.get("raw_has_dummy")),
            "raw_dummy_count": int(fragment_info.get("raw_dummy_count", 0)),
            "parse_stage": fragment_info.get("parse_stage"),
            "parsed_raw_with_dummy": bool(fragment_info.get("parsed_raw_with_dummy")),
            "parsed_core": bool(fragment_info.get("parsed_core")),
            "dummy_removed_before_parse": bool(fragment_info.get("dummy_removed_before_parse")),
            "parse_failed_reason": fragment_info.get("parse_failed_reason"),
            "core_atom_count": int(fragment_info.get("core_atom_count", 0)),
            "raw_component_count": int(fragment_info.get("raw_component_count", 0)),
            "core_component_count": int(fragment_info.get("core_component_count", 0)),
            "fragment_atom_count": int(fragment_info.get("core_atom_count", 0)),
            "min_fragment_atoms": int(self.min_fragment_atoms),
            "teacher_input_smiles": teacher_input_smiles,
            "teacher_available": bool(teacher_available),
            "teacher_called": bool(teacher_called),
            "teacher_probability": teacher_probability,
            "teacher_predicted_label": teacher_predicted_label,
            "teacher_reason": teacher_reason,
        }

    def _counterfactual_trace_kwargs(
        self,
        *,
        parent_without_fragment_smiles: str | None = None,
        counterfactual_teacher_available: bool = False,
        counterfactual_teacher_called: bool = False,
        counterfactual_teacher_reason: str | None = None,
        p_before: float | None = None,
        p_after: float | None = None,
        pred_before: int | None = None,
        pred_after: int | None = None,
        cf_drop: float | None = None,
        cf_flip: bool = False,
        empty_residual: bool = False,
    ) -> dict[str, Any]:
        return {
            "parent_without_fragment_smiles": parent_without_fragment_smiles,
            "counterfactual_teacher_available": bool(counterfactual_teacher_available),
            "counterfactual_teacher_called": bool(counterfactual_teacher_called),
            "counterfactual_teacher_reason": counterfactual_teacher_reason,
            "p_before": p_before,
            "p_after": p_after,
            "pred_before": pred_before,
            "pred_after": pred_after,
            "cf_drop": cf_drop,
            "cf_flip": bool(cf_flip),
            "empty_residual": bool(empty_residual),
        }

    def _repair_trace_kwargs(
        self,
        *,
        repair_attempted: bool = False,
        repair_success: bool = False,
        repaired_fragment_smiles: str | None = None,
        repair_source: str | None = None,
        repair_similarity: float | None = None,
        repair_reason: str | None = None,
        repair_method: str | None = None,
        repair_edit_distance: int = 0,
        repair_suffix_trim_count: int = 0,
        repair_added_parentheses: int = 0,
        repair_added_ring_closures: int = 0,
        repaired_raw_fragment: str | None = None,
        repaired_fragment_chars: int = 0,
        repaired_parse_stage: str | None = None,
        repaired_parsed_raw: bool = False,
        repaired_parsed_core: bool = False,
        repair_failure_reason: str | None = None,
        repair_failure_stage: str | None = None,
        repair_candidate_count: int = 0,
        repair_candidates_parse_ok: int = 0,
        repair_candidates_core_ok: int = 0,
        repair_candidates_parent_ok: int = 0,
        repair_candidates_projection_ok: int = 0,
        repair_best_candidate: str | None = None,
        repair_accept_stage: str | None = None,
        repair_candidate_accepted: bool = False,
        repair_candidate_rejected_reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "repair_attempted": bool(repair_attempted),
            "repair_success": bool(repair_success),
            "repaired_fragment_smiles": repaired_fragment_smiles,
            "repair_source": repair_source,
            "repair_similarity": repair_similarity,
            "repair_reason": repair_reason,
            "repair_method": repair_method,
            "repair_edit_distance": int(repair_edit_distance),
            "repair_suffix_trim_count": int(repair_suffix_trim_count),
            "repair_added_parentheses": int(repair_added_parentheses),
            "repair_added_ring_closures": int(repair_added_ring_closures),
            "repaired_raw_fragment": repaired_raw_fragment,
            "repaired_fragment_chars": int(repaired_fragment_chars),
            "repaired_parse_stage": repaired_parse_stage,
            "repaired_parsed_raw": bool(repaired_parsed_raw),
            "repaired_parsed_core": bool(repaired_parsed_core),
            "repair_failure_reason": repair_failure_reason,
            "repair_failure_stage": repair_failure_stage,
            "repair_candidate_count": int(repair_candidate_count),
            "repair_candidates_parse_ok": int(repair_candidates_parse_ok),
            "repair_candidates_core_ok": int(repair_candidates_core_ok),
            "repair_candidates_parent_ok": int(repair_candidates_parent_ok),
            "repair_candidates_projection_ok": int(repair_candidates_projection_ok),
            "repair_best_candidate": repair_best_candidate,
            "repair_accept_stage": repair_accept_stage,
            "repair_candidate_accepted": bool(repair_candidate_accepted),
            "repair_candidate_rejected_reason": repair_candidate_rejected_reason,
        }

    def _component_salvage_trace_kwargs(
        self,
        *,
        component_salvage_attempted: bool = False,
        component_salvage_success: bool = False,
        component_count: int = 0,
        raw_component_count: int = 0,
        core_component_count: int = 0,
        salvage_method: str | None = None,
        salvaged_fragment: str | None = None,
        salvaged_atom_count: int | None = None,
        component_salvage_failure_reason: str | None = None,
        component_salvage_stage: str | None = None,
        component_salvage_candidate_count: int = 0,
        component_salvage_best_candidate: str | None = None,
    ) -> dict[str, Any]:
        return {
            "component_salvage_attempted": bool(component_salvage_attempted),
            "component_salvage_success": bool(component_salvage_success),
            "component_count": int(component_count),
            "raw_component_count": int(raw_component_count),
            "core_component_count": int(core_component_count),
            "salvage_method": salvage_method,
            "salvaged_fragment": salvaged_fragment,
            "salvaged_atom_count": salvaged_atom_count,
            "component_salvage_failure_reason": component_salvage_failure_reason,
            "component_salvage_stage": component_salvage_stage,
            "component_salvage_candidate_count": int(component_salvage_candidate_count),
            "component_salvage_best_candidate": component_salvage_best_candidate,
        }

    def _dummy_trace_kwargs(
        self,
        *,
        multi_dummy_hard_fail: bool = False,
        dummy_salvage_attempted: bool = False,
        dummy_salvage_success: bool = False,
        dummy_salvage_method: str | None = None,
        dummy_salvaged_fragment: str | None = None,
    ) -> dict[str, Any]:
        return {
            "multi_dummy_hard_fail": bool(multi_dummy_hard_fail),
            "dummy_salvage_attempted": bool(dummy_salvage_attempted),
            "dummy_salvage_success": bool(dummy_salvage_success),
            "dummy_salvage_method": dummy_salvage_method,
            "dummy_salvaged_fragment": dummy_salvaged_fragment,
        }

    def _residual_guard_trace_kwargs(
        self,
        *,
        near_parent_hard_fail: bool = False,
        residual_atom_count: int | None = None,
        residual_atom_ratio: float | None = None,
        tiny_fragment_hard_fail: bool = False,
        fragment_atom_count: int = 0,
    ) -> dict[str, Any]:
        return {
            "near_parent_hard_fail": bool(near_parent_hard_fail),
            "residual_atom_count": residual_atom_count,
            "residual_atom_ratio": residual_atom_ratio,
            "tiny_fragment_hard_fail": bool(tiny_fragment_hard_fail),
            "fragment_atom_count": int(fragment_atom_count),
            "min_fragment_atoms": int(self.min_fragment_atoms),
        }

    def _size_window_trace_kwargs(
        self,
        *,
        size_window_reward: float = 0.0,
        size_window_bucket: str | None = None,
        final_fragment_atom_count: int = 0,
        final_fragment_atom_ratio: float | None = None,
    ) -> dict[str, Any]:
        return {
            "size_window_reward": float(size_window_reward),
            "size_window_bucket": size_window_bucket,
            "size_window_low": float(self.size_window_low),
            "size_window_high": float(self.size_window_high),
            "final_fragment_atom_count": int(final_fragment_atom_count),
            "final_fragment_atom_ratio": final_fragment_atom_ratio,
        }

    def _projection_trace_kwargs(
        self,
        *,
        projection_attempted: bool = False,
        projection_success: bool = False,
        projection_method: str | None = None,
        projected_fragment_smiles: str | None = None,
        projection_source: str | None = None,
        projection_score: float | None = None,
        projection_reason: str | None = None,
        projected_atom_count: int | None = None,
        projected_atom_ratio: float | None = None,
        projection_penalty: float = 0.0,
        num_projection_candidates: int = 0,
    ) -> dict[str, Any]:
        return {
            "projection_attempted": bool(projection_attempted),
            "projection_success": bool(projection_success),
            "projection_method": projection_method,
            "projected_fragment_smiles": projected_fragment_smiles,
            "projection_source": projection_source,
            "projection_score": projection_score,
            "projection_reason": projection_reason,
            "projection_atom_count": projected_atom_count,
            "projection_atom_ratio": projected_atom_ratio,
            "projection_penalty": float(projection_penalty),
            "num_projection_candidates": int(num_projection_candidates),
        }

    def _attempt_parent_projection(
        self,
        *,
        parent_smiles: str,
        raw_fragment_smiles: str,
    ) -> tuple[Any, dict[str, Any]]:
        if not self.enable_parent_projection:
            return None, self._projection_trace_kwargs()

        projection_result = project_fragment_to_parent_subgraph(
            parent_smiles,
            raw_fragment_smiles,
            min_score=self.projection_min_score,
            max_candidates=self.projection_max_candidates,
            min_atoms=self.projection_min_atoms,
            max_atom_ratio=self.projection_max_atom_ratio,
            enable_khop3=self.projection_enable_khop3,
            mcs_timeout=self.projection_mcs_timeout,
        )
        applied_penalty = (
            self.projection_penalty
            if projection_result.success
            and projection_result.projection_method == "retrieval"
            else 0.0
        )
        return projection_result, self._projection_trace_kwargs(
            projection_attempted=projection_result.attempted,
            projection_success=projection_result.success,
            projection_method=projection_result.projection_method,
            projected_fragment_smiles=projection_result.projected_fragment_smiles,
            projection_source=projection_result.projection_source,
            projection_score=projection_result.projection_score,
            projection_reason=projection_result.reason,
            projected_atom_count=projection_result.projected_atom_count,
            projected_atom_ratio=projection_result.projected_atom_ratio,
            projection_penalty=applied_penalty,
            num_projection_candidates=projection_result.candidate_count,
        )

    def _attempt_minimal_syntax_repair(
        self,
        *,
        parent_smiles: str,
        raw_fragment_smiles: str,
        parse_failed_reason: str | None,
    ) -> tuple[Any, dict[str, Any]]:
        if not self.enable_minimal_syntax_repair:
            return None, self._repair_trace_kwargs()

        candidates = generate_minimal_syntax_repair_candidates(
            raw_fragment_smiles,
            parse_failed_reason=parse_failed_reason,
            max_edits=self.syntax_repair_max_edits,
            allow_parentheses_fix=self.syntax_repair_allow_parentheses_fix,
            allow_ring_fix=self.syntax_repair_allow_ring_fix,
            allow_tail_trim=self.syntax_repair_allow_tail_trim,
            allow_balanced_prefix_salvage=self.syntax_repair_allow_balanced_prefix_salvage,
            prefer_prefix_salvage=self.syntax_repair_prefer_prefix_salvage,
            max_suffix_trim=self.syntax_repair_max_suffix_trim,
            max_added_closures=self.syntax_repair_max_added_closures,
        )
        rejection_counts: Counter[str] = Counter()
        parse_ok_count = 0
        core_ok_count = 0
        parent_ok_count = 0
        projection_ok_count = 0
        best_candidate: str | None = None
        best_stage_rank = -1
        best_failure_reason: str | None = None
        accepted_result: FragmentSyntaxRepairResult | None = None
        accepted_info: dict[str, Any] | None = None

        def update_best(candidate_smiles: str, stage_rank: int, failure_reason: str | None) -> None:
            nonlocal best_candidate, best_stage_rank, best_failure_reason
            if stage_rank >= best_stage_rank:
                best_candidate = candidate_smiles
                best_stage_rank = stage_rank
                best_failure_reason = failure_reason

        for candidate in candidates:
            parsed = parse_smiles(
                candidate.fragment_smiles,
                sanitize=True,
                canonicalize=True,
                allow_capped_fragments=True,
            )
            if not parsed.parseable or parsed.mol is None:
                rejection_counts["repair_candidate_parse_failed"] += 1
                update_best(
                    candidate.fragment_smiles,
                    0,
                    "repair_candidate_parse_failed",
                )
                continue

            parse_ok_count += 1
            atom_count = _non_dummy_atom_count(parsed.mol)
            if atom_count < self.syntax_repair_min_atoms:
                rejection_counts["repair_candidate_too_small"] += 1
                update_best(
                    candidate.fragment_smiles,
                    1,
                    "repair_candidate_too_small",
                )
                continue

            repaired_info = normalize_fragment_with_dummy_atoms(candidate.fragment_smiles)
            core_smiles = repaired_info.get("core_smiles")
            if (
                not repaired_info.get("core_parse_ok")
                or repaired_info.get("core_mol") is None
                or not core_smiles
            ):
                rejection_counts["repair_candidate_core_unusable"] += 1
                update_best(
                    candidate.fragment_smiles,
                    2,
                    "repair_candidate_core_unusable",
                )
                continue

            core_ok_count += 1
            if is_parent_substructure(parent_smiles, str(core_smiles)):
                parent_ok_count += 1
                accepted_result = FragmentSyntaxRepairResult(
                    raw_fragment_smiles=str(raw_fragment_smiles or "").strip(),
                    attempted=True,
                    success=True,
                    repaired_fragment_smiles=candidate.fragment_smiles,
                    repair_method=candidate.repair_method,
                    reason=candidate.reason,
                    edit_distance=candidate.edit_distance,
                    suffix_trim_count=candidate.suffix_trim_count,
                    added_parentheses=candidate.added_parentheses,
                    added_ring_closures=candidate.added_ring_closures,
                    repaired_atom_count=int(repaired_info.get("core_atom_count", atom_count) or 0),
                    candidate_count=len(candidates),
                    candidates_parse_ok=parse_ok_count,
                    candidates_core_ok=core_ok_count,
                    candidates_parent_ok=parent_ok_count,
                    candidates_projection_ok=projection_ok_count,
                    best_candidate=candidate.fragment_smiles,
                    accept_stage="strict_parent",
                    candidate_accepted=True,
                )
                accepted_info = repaired_info
                break

            projection_result = project_fragment_to_parent_subgraph(
                parent_smiles,
                candidate.fragment_smiles,
                min_score=self.projection_min_score,
                max_candidates=self.projection_max_candidates,
                min_atoms=self.projection_min_atoms,
                max_atom_ratio=self.projection_max_atom_ratio,
                enable_khop3=self.projection_enable_khop3,
                mcs_timeout=self.projection_mcs_timeout,
            )
            if projection_result.success and projection_result.projected_fragment_smiles:
                projection_ok_count += 1
                accepted_result = FragmentSyntaxRepairResult(
                    raw_fragment_smiles=str(raw_fragment_smiles or "").strip(),
                    attempted=True,
                    success=True,
                    repaired_fragment_smiles=candidate.fragment_smiles,
                    repair_method=candidate.repair_method,
                    reason=candidate.reason,
                    edit_distance=candidate.edit_distance,
                    suffix_trim_count=candidate.suffix_trim_count,
                    added_parentheses=candidate.added_parentheses,
                    added_ring_closures=candidate.added_ring_closures,
                    repaired_atom_count=int(repaired_info.get("core_atom_count", atom_count) or 0),
                    candidate_count=len(candidates),
                    candidates_parse_ok=parse_ok_count,
                    candidates_core_ok=core_ok_count,
                    candidates_parent_ok=parent_ok_count,
                    candidates_projection_ok=projection_ok_count,
                    best_candidate=candidate.fragment_smiles,
                    accept_stage="projection",
                    candidate_accepted=True,
                )
                accepted_info = repaired_info
                break

            rejection_reason = (
                "repair_candidate_projection_failed_low_score"
                if projection_result.reason == "projection_failed_low_score"
                else "repair_candidate_not_parent_subgraph"
            )
            rejection_counts[rejection_reason] += 1
            update_best(candidate.fragment_smiles, 3, rejection_reason)

        if accepted_result is None:
            fallback_result = repair_minimal_fragment_syntax(
                raw_fragment_smiles,
                parse_failed_reason=parse_failed_reason,
                max_edits=self.syntax_repair_max_edits,
                min_atoms=self.syntax_repair_min_atoms,
                allow_parentheses_fix=self.syntax_repair_allow_parentheses_fix,
                allow_ring_fix=self.syntax_repair_allow_ring_fix,
                allow_tail_trim=self.syntax_repair_allow_tail_trim,
                allow_balanced_prefix_salvage=self.syntax_repair_allow_balanced_prefix_salvage,
                prefer_prefix_salvage=self.syntax_repair_prefer_prefix_salvage,
                max_suffix_trim=self.syntax_repair_max_suffix_trim,
                max_added_closures=self.syntax_repair_max_added_closures,
            )
            failure_reason = fallback_result.failure_reason or best_failure_reason or "repair_candidate_other"
            failure_stage = fallback_result.failure_stage or (
                "candidate_generation"
                if not candidates
                else "candidate_filter"
            )
            repair_result = FragmentSyntaxRepairResult(
                raw_fragment_smiles=str(raw_fragment_smiles or "").strip(),
                attempted=True,
                success=False,
                reason=failure_reason,
                failure_reason=failure_reason,
                failure_stage=failure_stage,
                candidate_count=len(candidates),
                candidates_parse_ok=parse_ok_count,
                candidates_core_ok=core_ok_count,
                candidates_parent_ok=parent_ok_count,
                candidates_projection_ok=projection_ok_count,
                best_candidate=best_candidate,
                candidate_accepted=False,
                candidate_rejected_reason=(
                    best_failure_reason
                    or fallback_result.candidate_rejected_reason
                    or failure_reason
                ),
            )
            return repair_result, self._repair_trace_kwargs(
                repair_attempted=repair_result.attempted,
                repair_success=False,
                repair_source="minimal_syntax",
                repair_reason=repair_result.reason,
                repair_failure_reason=repair_result.failure_reason,
                repair_failure_stage=repair_result.failure_stage,
                repair_candidate_count=repair_result.candidate_count,
                repair_candidates_parse_ok=repair_result.candidates_parse_ok,
                repair_candidates_core_ok=repair_result.candidates_core_ok,
                repair_candidates_parent_ok=repair_result.candidates_parent_ok,
                repair_candidates_projection_ok=repair_result.candidates_projection_ok,
                repair_best_candidate=repair_result.best_candidate,
                repair_candidate_accepted=False,
                repair_candidate_rejected_reason=repair_result.candidate_rejected_reason,
            )

        repaired_info = accepted_info
        repaired_fragment_smiles = accepted_result.repaired_fragment_smiles
        return accepted_result, self._repair_trace_kwargs(
            repair_attempted=True,
            repair_success=True,
            repaired_fragment_smiles=repaired_fragment_smiles,
            repair_source="minimal_syntax",
            repair_reason=accepted_result.reason,
            repair_method=accepted_result.repair_method,
            repair_edit_distance=accepted_result.edit_distance,
            repair_suffix_trim_count=accepted_result.suffix_trim_count,
            repair_added_parentheses=accepted_result.added_parentheses,
            repair_added_ring_closures=accepted_result.added_ring_closures,
            repaired_raw_fragment=accepted_result.best_candidate,
            repaired_fragment_chars=len(repaired_fragment_smiles or ""),
            repaired_parse_stage=(
                repaired_info.get("parse_stage") if repaired_info is not None else None
            ),
            repaired_parsed_raw=(
                bool(repaired_info.get("raw_parse_ok"))
                if repaired_info is not None
                else False
            ),
            repaired_parsed_core=(
                bool(repaired_info.get("core_parse_ok"))
                if repaired_info is not None
                else False
            ),
            repair_candidate_count=accepted_result.candidate_count,
            repair_candidates_parse_ok=accepted_result.candidates_parse_ok,
            repair_candidates_core_ok=accepted_result.candidates_core_ok,
            repair_candidates_parent_ok=accepted_result.candidates_parent_ok,
            repair_candidates_projection_ok=accepted_result.candidates_projection_ok,
            repair_best_candidate=accepted_result.best_candidate,
            repair_accept_stage=accepted_result.accept_stage,
            repair_candidate_accepted=True,
        )

    def _attempt_component_salvage(
        self,
        *,
        parent_smiles: str,
        raw_fragment_smiles: str,
        component_fragment_smiles: str | None = None,
        raw_component_count: int = 0,
        core_component_count: int = 0,
        component_salvage_stage: str | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if not self.enable_component_salvage:
            return None, self._component_salvage_trace_kwargs()

        fragment_for_salvage = str(
            component_fragment_smiles or raw_fragment_smiles or ""
        ).strip()
        salvage_result = salvage_connected_component(
            parent_smiles,
            fragment_for_salvage,
            method=self.component_salvage_method,
            min_atoms=self.component_salvage_min_atoms,
            projection_min_score=self.projection_min_score,
            projection_max_candidates=self.projection_max_candidates,
            projection_max_atom_ratio=self.projection_max_atom_ratio,
            projection_enable_khop3=self.projection_enable_khop3,
            projection_mcs_timeout=self.projection_mcs_timeout,
            salvage_stage=component_salvage_stage,
            raw_component_count=raw_component_count,
            core_component_count=core_component_count,
        )
        return salvage_result, self._component_salvage_trace_kwargs(
            component_salvage_attempted=salvage_result.attempted,
            component_salvage_success=salvage_result.success,
            component_count=salvage_result.component_count,
            raw_component_count=salvage_result.raw_component_count or raw_component_count,
            core_component_count=salvage_result.core_component_count or core_component_count,
            salvage_method=salvage_result.salvage_method,
            salvaged_fragment=salvage_result.salvaged_fragment_smiles,
            salvaged_atom_count=salvage_result.salvaged_atom_count,
            component_salvage_failure_reason=(
                None
                if salvage_result.success
                else salvage_result.failure_reason or salvage_result.reason
            ),
            component_salvage_stage=salvage_result.salvage_stage or component_salvage_stage,
            component_salvage_candidate_count=salvage_result.candidate_count,
            component_salvage_best_candidate=salvage_result.best_candidate,
        )

    def _attempt_parent_aware_repair(
        self,
        *,
        parent_smiles: str,
        raw_fragment_smiles: str,
    ) -> tuple[Any, dict[str, Any]]:
        if not self.enable_parent_aware_repair:
            return None, self._repair_trace_kwargs()

        repair_result = repair_fragment_to_parent_subgraph(
            parent_smiles,
            raw_fragment_smiles,
            min_similarity=self.repair_min_similarity,
            max_candidates=self.repair_max_candidates,
        )
        return repair_result, self._repair_trace_kwargs(
            repair_attempted=repair_result.attempted,
            repair_success=repair_result.success,
            repaired_fragment_smiles=repair_result.repaired_fragment_smiles,
            repair_source=repair_result.repair_source,
            repair_similarity=repair_result.repair_similarity,
            repair_reason=repair_result.reason,
        )

    def _score_teacher_semantic(
        self,
        *,
        core_smiles: str | None,
        original_label: int,
        parent_smiles: str,
        fragment_info: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        teacher_input_smiles = core_smiles
        if not core_smiles or not fragment_info.get("core_parse_ok"):
            return self.teacher_sem_missing_penalty, self._fragment_trace_kwargs(
                fragment_info,
                teacher_input_smiles=teacher_input_smiles,
                teacher_reason="invalid_or_not_substructure",
            )

        if self.teacher_scorer is None:
            return self.teacher_sem_missing_penalty, self._fragment_trace_kwargs(
                fragment_info,
                teacher_input_smiles=teacher_input_smiles,
                teacher_reason="teacher_sem_disabled",
            )

        teacher_result = self.teacher_scorer.score_smiles(
            core_smiles,
            label=int(original_label),
            parent_smiles=parent_smiles,
            meta={"raw_fragment": fragment_info.get("raw")},
        )
        teacher_result_ok = bool(teacher_result.get("teacher_result_ok"))
        teacher_available = bool(teacher_result.get("teacher_available"))
        teacher_reward = (
            self.teacher_sem_scale * float(teacher_result.get("teacher_sem", 0.0))
            if teacher_result_ok
            else self.teacher_sem_missing_penalty
        )
        return teacher_reward, self._fragment_trace_kwargs(
            fragment_info,
            teacher_input_smiles=teacher_result.get("teacher_input_smiles", teacher_input_smiles),
            teacher_available=teacher_available,
            teacher_called=teacher_result_ok,
            teacher_probability=teacher_result.get("teacher_prob"),
            teacher_predicted_label=teacher_result.get("teacher_label"),
            teacher_reason=teacher_result.get("teacher_reason"),
        )

    def _score_counterfactual_semantic(
        self,
        *,
        parent_smiles: str,
        core_smiles: str | None,
        raw_fragment_smiles: str,
        original_label: int,
        fragment_info: dict[str, Any],
        valid_reward: float,
        substructure_ok: bool,
    ) -> tuple[float, dict[str, Any]]:
        if (
            not core_smiles
            or not fragment_info.get("core_parse_ok")
            or valid_reward <= 0.0
            or not substructure_ok
            or not parent_smiles
        ):
            _LOGGER.info(
                "[CF_ORACLE_SKIPPED] reason=%s parent=%s fragment=%s",
                "invalid_or_not_substructure",
                parent_smiles,
                core_smiles,
            )
            return self.teacher_sem_missing_penalty, self._counterfactual_trace_kwargs(
                counterfactual_teacher_available=bool(
                    self.counterfactual_teacher_scorer is not None
                    and self.counterfactual_teacher_scorer.available
                ),
                counterfactual_teacher_called=False,
                counterfactual_teacher_reason="invalid_or_not_substructure",
            )

        if self.disable_counterfactual_teacher:
            _LOGGER.warning(
                "[CF_ORACLE_UNAVAILABLE] reason=%s parent=%s fragment=%s",
                "counterfactual_teacher_disabled",
                parent_smiles,
                core_smiles,
            )
            return self.teacher_sem_missing_penalty, self._counterfactual_trace_kwargs(
                counterfactual_teacher_available=False,
                counterfactual_teacher_called=False,
                counterfactual_teacher_reason="counterfactual_teacher_disabled",
            )

        if self.counterfactual_teacher_scorer is None:
            _LOGGER.warning(
                "[CF_ORACLE_UNAVAILABLE] reason=%s parent=%s fragment=%s",
                "counterfactual_teacher_unavailable",
                parent_smiles,
                core_smiles,
            )
            return self.teacher_sem_missing_penalty, self._counterfactual_trace_kwargs(
                counterfactual_teacher_available=False,
                counterfactual_teacher_called=False,
                counterfactual_teacher_reason="counterfactual_teacher_unavailable",
            )

        _LOGGER.info(
            "[CF_ORACLE_CALLED] parent=%s fragment=%s label=%s",
            parent_smiles,
            core_smiles,
            original_label,
        )
        counterfactual_result = self.counterfactual_teacher_scorer.score_counterfactual(
            parent_smiles=parent_smiles,
            core_fragment_smiles=core_smiles,
            label=int(original_label),
            raw_fragment_smiles=raw_fragment_smiles,
            meta={"raw_fragment": raw_fragment_smiles},
        )
        counterfactual_result_ok = bool(counterfactual_result.get("teacher_result_ok"))
        if counterfactual_result_ok:
            _LOGGER.info(
                "[CF_ORACLE_RESULT] parent_without_fragment=%s p_before=%s p_after=%s cf_drop=%s cf_flip=%s counterfactual_sem=%s reason=%s",
                counterfactual_result.get("parent_without_fragment_smiles"),
                counterfactual_result.get("p_before"),
                counterfactual_result.get("p_after"),
                counterfactual_result.get("cf_drop"),
                counterfactual_result.get("cf_flip"),
                counterfactual_result.get("counterfactual_sem"),
                counterfactual_result.get("teacher_reason"),
            )
        else:
            reason = str(counterfactual_result.get("teacher_reason"))
            if "deletion" in reason or "residual" in reason or "substructure" in reason:
                _LOGGER.warning(
                    "[CF_ORACLE_DELETION_FAILED] reason=%s parent=%s fragment=%s",
                    reason,
                    parent_smiles,
                    core_smiles,
                )
            else:
                _LOGGER.warning(
                    "[CF_ORACLE_UNAVAILABLE] reason=%s parent=%s fragment=%s",
                    reason,
                    parent_smiles,
                    core_smiles,
                )
        failure_reason = str(counterfactual_result.get("teacher_reason"))
        empty_residual = failure_reason in {
            "empty_residual_after_deletion",
            "empty_residual_smiles",
        }
        counterfactual_reward = (
            self.teacher_sem_scale
            * float(counterfactual_result.get("counterfactual_sem", 0.0))
            if counterfactual_result_ok
            else (
                self.empty_residual_penalty
                if empty_residual
                else self.teacher_sem_missing_penalty
            )
        )
        return counterfactual_reward, self._counterfactual_trace_kwargs(
            parent_without_fragment_smiles=counterfactual_result.get(
                "parent_without_fragment_smiles"
            ),
            counterfactual_teacher_available=bool(
                counterfactual_result.get("teacher_available")
            ),
            counterfactual_teacher_called=counterfactual_result_ok,
            counterfactual_teacher_reason=counterfactual_result.get("teacher_reason"),
            p_before=counterfactual_result.get("p_before"),
            p_after=counterfactual_result.get("p_after"),
            pred_before=counterfactual_result.get("pred_before"),
            pred_after=counterfactual_result.get("pred_after"),
            cf_drop=counterfactual_result.get("cf_drop"),
            cf_flip=bool(counterfactual_result.get("cf_flip", False)),
            empty_residual=empty_residual,
        )

    def _delete_to_residual_smiles(
        self,
        parent_smiles: str,
        fragment_smiles: str,
    ) -> tuple[str | None, str | None]:
        """Delete one fragment and return a sanitized residual SMILES.

        The primary path uses ``Chem.DeleteSubstructs``.
        If RDKit removes an unexpected number of atoms, the method falls back to
        ``delete_fragment_from_parent`` so PPO still sees a deterministic residual.
        """

        parent = parse_smiles(
            parent_smiles,
            sanitize=True,
            canonicalize=False,
            allow_capped_fragments=False,
        )
        fragment = parse_smiles(
            fragment_smiles,
            sanitize=True,
            canonicalize=False,
            allow_capped_fragments=True,
        )
        if not parent.sanitized or parent.mol is None or not fragment.sanitized or fragment.mol is None:
            return None, "Parent or fragment could not be parsed for deletion."

        parent_mol = Chem.Mol(parent.mol)
        fragment_mol = Chem.Mol(fragment.mol)
        deletion_query, expected_removed_atoms = self._build_deletion_query(fragment_mol)
        if deletion_query is None or expected_removed_atoms <= 0:
            return None, "Could not build a deletion query from the generated fragment."

        try:
            try:
                residual_mol = Chem.DeleteSubstructs(
                    Chem.Mol(parent_mol),
                    deletion_query,
                    onlyFrags=False,
                    useChirality=True,
                )
            except TypeError:
                residual_mol = Chem.DeleteSubstructs(
                    Chem.Mol(parent_mol),
                    deletion_query,
                    onlyFrags=False,
                )
        except Exception as exc:
            return self._fallback_delete_fragment(
                parent_smiles,
                fragment_smiles,
                f"DeleteSubstructs raised: {exc}",
            )

        if residual_mol is None:
            return self._fallback_delete_fragment(
                parent_smiles,
                fragment_smiles,
                "DeleteSubstructs returned None.",
            )

        removed_atom_count = parent_mol.GetNumAtoms() - residual_mol.GetNumAtoms()
        if removed_atom_count != expected_removed_atoms:
            return self._fallback_delete_fragment(
                parent_smiles,
                fragment_smiles,
                (
                    "DeleteSubstructs removed an unexpected number of atoms "
                    f"(expected {expected_removed_atoms}, got {removed_atom_count})."
                ),
            )

        sanitized_smiles, sanitize_error = self._sanitize_residual_molecule(residual_mol)
        if sanitized_smiles is None:
            return self._fallback_delete_fragment(
                parent_smiles,
                fragment_smiles,
                sanitize_error or "Residual sanitization failed after DeleteSubstructs.",
            )

        return sanitized_smiles, None

    def _build_deletion_query(
        self,
        fragment_mol: Any,
    ) -> tuple[Any | None, int]:
        """Build the DeleteSubstructs query and count real fragment atoms."""

        if fragment_mol is None:
            return None, 0

        real_atom_indices = [
            atom.GetIdx()
            for atom in fragment_mol.GetAtoms()
            if atom.GetAtomicNum() != 0
        ]
        if not real_atom_indices:
            return None, 0

        if len(real_atom_indices) == fragment_mol.GetNumAtoms():
            return fragment_mol, len(real_atom_indices)

        editable = Chem.RWMol(fragment_mol)
        dummy_indices = sorted(
            (
                atom.GetIdx()
                for atom in fragment_mol.GetAtoms()
                if atom.GetAtomicNum() == 0
            ),
            reverse=True,
        )
        for atom_index in dummy_indices:
            editable.RemoveAtom(atom_index)

        core_mol = editable.GetMol()
        if core_mol.GetNumAtoms() == 0:
            return None, 0
        try:
            Chem.SanitizeMol(core_mol)
        except Exception:
            return None, 0
        return core_mol, len(real_atom_indices)

    def _sanitize_residual_molecule(
        self,
        residual_mol: Any,
    ) -> tuple[str | None, str | None]:
        """Sanitize a residual molecule before Oracle featurization."""

        if residual_mol is None:
            return None, "Residual molecule is None."
        if residual_mol.GetNumAtoms() == 0:
            return "", None

        try:
            Chem.SanitizeMol(residual_mol)
            return Chem.MolToSmiles(residual_mol, canonical=True), None
        except Exception as sanitize_error:
            try:
                hydrogenated = Chem.AddHs(residual_mol, addCoords=False)
                Chem.SanitizeMol(hydrogenated)
                recovered = Chem.RemoveHs(hydrogenated)
                Chem.SanitizeMol(recovered)
                return Chem.MolToSmiles(recovered, canonical=True), None
            except Exception as hydrogen_error:
                return None, (
                    "Residual sanitization failed after DeleteSubstructs. "
                    f"SanitizeMol error: {sanitize_error}; AddHs recovery error: {hydrogen_error}"
                )

    def _fallback_delete_fragment(
        self,
        parent_smiles: str,
        fragment_smiles: str,
        fallback_reason: str,
    ) -> tuple[str | None, str | None]:
        """Fallback to the repository's deterministic single-match deletion helper."""

        deletion_result = delete_fragment_from_parent(parent_smiles, fragment_smiles)
        if not deletion_result.success or deletion_result.residual_smiles is None:
            return None, (
                f"{fallback_reason} Fallback deletion also failed: "
                f"{deletion_result.failure_reason or 'unknown reason'}"
            )
        return deletion_result.residual_smiles, None

    def _fail(
        self,
        *,
        parent_smiles: str,
        generated_smiles: str,
        normalized_generated: str,
        original_label: int,
        failure_stage: str,
        error_message: str,
        breakdown: dict[str, float],
        valid_smiles: bool = False,
        connected_fragment: bool = False,
        is_subgraph: bool = False,
        deletion_success: bool = False,
        residual_smiles: str | None = None,
        raw_fragment_smiles: str | None = None,
        core_fragment_smiles: str | None = None,
        raw_parse_ok: bool = False,
        core_parse_ok: bool = False,
        has_dummy_atoms: bool = False,
        dummy_count: int = 0,
        raw_has_dummy: bool = False,
        raw_dummy_count: int = 0,
        parse_stage: str | None = None,
        parsed_raw_with_dummy: bool = False,
        parsed_core: bool = False,
        dummy_removed_before_parse: bool = False,
        parse_failed_reason: str | None = None,
        core_atom_count: int = 0,
        teacher_input_smiles: str | None = None,
        teacher_available: bool = False,
        teacher_called: bool = False,
        teacher_probability: float | None = None,
        teacher_predicted_label: int | None = None,
        teacher_reason: str | None = None,
        fragment_teacher_sem: float | None = None,
        parent_without_fragment_smiles: str | None = None,
        counterfactual_teacher_available: bool = False,
        counterfactual_teacher_called: bool = False,
        counterfactual_teacher_reason: str | None = None,
        p_before: float | None = None,
        p_after: float | None = None,
        pred_before: int | None = None,
        pred_after: int | None = None,
        cf_drop: float | None = None,
        cf_flip: bool = False,
        failure_tag: str | None = None,
        invalid_detail: str | None = None,
        generated_char_count: int = 0,
        repair_attempted: bool = False,
        repair_success: bool = False,
        repaired_fragment_smiles: str | None = None,
        repair_source: str | None = None,
        repair_similarity: float | None = None,
        repair_reason: str | None = None,
        repair_method: str | None = None,
        repair_edit_distance: int = 0,
        repair_suffix_trim_count: int = 0,
        repair_added_parentheses: int = 0,
        repair_added_ring_closures: int = 0,
        repaired_raw_fragment: str | None = None,
        repaired_fragment_chars: int = 0,
        repaired_parse_stage: str | None = None,
        repaired_parsed_raw: bool = False,
        repaired_parsed_core: bool = False,
        repair_failure_reason: str | None = None,
        repair_failure_stage: str | None = None,
        repair_candidate_count: int = 0,
        repair_candidates_parse_ok: int = 0,
        repair_candidates_core_ok: int = 0,
        repair_candidates_parent_ok: int = 0,
        repair_candidates_projection_ok: int = 0,
        repair_best_candidate: str | None = None,
        repair_accept_stage: str | None = None,
        repair_candidate_accepted: bool = False,
        repair_candidate_rejected_reason: str | None = None,
        component_salvage_attempted: bool = False,
        component_salvage_success: bool = False,
        component_count: int = 0,
        raw_component_count: int = 0,
        core_component_count: int = 0,
        salvage_method: str | None = None,
        salvaged_fragment: str | None = None,
        salvaged_atom_count: int | None = None,
        component_salvage_failure_reason: str | None = None,
        component_salvage_stage: str | None = None,
        component_salvage_candidate_count: int = 0,
        component_salvage_best_candidate: str | None = None,
        multi_dummy_hard_fail: bool = False,
        dummy_salvage_attempted: bool = False,
        dummy_salvage_success: bool = False,
        dummy_salvage_method: str | None = None,
        dummy_salvaged_fragment: str | None = None,
        near_parent_hard_fail: bool = False,
        residual_atom_count: int | None = None,
        residual_atom_ratio: float | None = None,
        tiny_fragment_hard_fail: bool = False,
        fragment_atom_count: int = 0,
        min_fragment_atoms: int = 0,
        projection_attempted: bool = False,
        projection_success: bool = False,
        projection_method: str | None = None,
        projection_score: float | None = None,
        projection_source: str | None = None,
        projected_fragment_smiles: str | None = None,
        projection_atom_count: int | None = None,
        projection_atom_ratio: float | None = None,
        projection_penalty: float = 0.0,
        num_projection_candidates: int = 0,
        projection_reason: str | None = None,
        size_window_reward: float = 0.0,
        size_window_bucket: str | None = None,
        size_window_low: float | None = None,
        size_window_high: float | None = None,
        final_fragment_atom_count: int = 0,
        final_fragment_atom_ratio: float | None = None,
        empty_response: bool = False,
        full_parent: bool = False,
        empty_residual: bool = False,
    ) -> RewardTrace:
        """Build one failure trace consistently."""

        return RewardTrace(
            parent_smiles=parent_smiles,
            generated_smiles=str(generated_smiles),
            normalized_generated_smiles=normalized_generated,
            raw_fragment_smiles=raw_fragment_smiles,
            core_fragment_smiles=core_fragment_smiles,
            original_label=int(original_label),
            target_label=1 - int(original_label),
            reward=self._reward_from_breakdown(breakdown),
            valid_smiles=bool(valid_smiles),
            connected_fragment=bool(connected_fragment),
            is_subgraph=bool(is_subgraph),
            deletion_success=bool(deletion_success),
            counterfactual_evaluated=False,
            flip_success=False,
            target_probability=None,
            inactive_probability=None,
            residual_smiles=residual_smiles,
            empty_response=bool(empty_response),
            full_parent=bool(full_parent),
            empty_residual=bool(empty_residual),
            oracle_ok=False,
            raw_parse_ok=bool(raw_parse_ok),
            core_parse_ok=bool(core_parse_ok),
            has_dummy_atoms=bool(has_dummy_atoms),
            dummy_count=int(dummy_count),
            raw_has_dummy=bool(raw_has_dummy),
            raw_dummy_count=int(raw_dummy_count),
            parse_stage=parse_stage,
            parsed_raw_with_dummy=bool(parsed_raw_with_dummy),
            parsed_core=bool(parsed_core),
            dummy_removed_before_parse=bool(dummy_removed_before_parse),
            parse_failed_reason=parse_failed_reason,
            core_atom_count=int(core_atom_count),
            teacher_input_smiles=teacher_input_smiles,
            teacher_available=bool(teacher_available),
            teacher_called=bool(teacher_called),
            teacher_probability=teacher_probability,
            teacher_predicted_label=teacher_predicted_label,
            teacher_reason=teacher_reason,
            fragment_teacher_sem=fragment_teacher_sem,
            parent_without_fragment_smiles=parent_without_fragment_smiles,
            counterfactual_teacher_available=bool(counterfactual_teacher_available),
            counterfactual_teacher_called=bool(counterfactual_teacher_called),
            counterfactual_teacher_reason=counterfactual_teacher_reason,
            p_before=p_before,
            p_after=p_after,
            pred_before=pred_before,
            pred_after=pred_after,
            cf_drop=cf_drop,
            cf_flip=bool(cf_flip),
            failure_stage=failure_stage,
            failure_tag=self._infer_failure_tag(
                explicit_failure_tag=failure_tag,
                failure_stage=failure_stage,
                empty_response=bool(empty_response),
                full_parent=bool(full_parent),
                empty_residual=bool(empty_residual),
            ),
            invalid_detail=invalid_detail,
            generated_char_count=int(generated_char_count),
            repair_attempted=bool(repair_attempted),
            repair_success=bool(repair_success),
            repaired_fragment_smiles=repaired_fragment_smiles,
            repair_source=repair_source,
            repair_similarity=repair_similarity,
            repair_reason=repair_reason,
            repair_method=repair_method,
            repair_edit_distance=int(repair_edit_distance),
            repair_suffix_trim_count=int(repair_suffix_trim_count),
            repair_added_parentheses=int(repair_added_parentheses),
            repair_added_ring_closures=int(repair_added_ring_closures),
            repaired_raw_fragment=repaired_raw_fragment,
            repaired_fragment_chars=int(repaired_fragment_chars),
            repaired_parse_stage=repaired_parse_stage,
            repaired_parsed_raw=bool(repaired_parsed_raw),
            repaired_parsed_core=bool(repaired_parsed_core),
            repair_failure_reason=repair_failure_reason,
            repair_failure_stage=repair_failure_stage,
            repair_candidate_count=int(repair_candidate_count),
            repair_candidates_parse_ok=int(repair_candidates_parse_ok),
            repair_candidates_core_ok=int(repair_candidates_core_ok),
            repair_candidates_parent_ok=int(repair_candidates_parent_ok),
            repair_candidates_projection_ok=int(repair_candidates_projection_ok),
            repair_best_candidate=repair_best_candidate,
            repair_accept_stage=repair_accept_stage,
            repair_candidate_accepted=bool(repair_candidate_accepted),
            repair_candidate_rejected_reason=repair_candidate_rejected_reason,
            component_salvage_attempted=bool(component_salvage_attempted),
            component_salvage_success=bool(component_salvage_success),
            component_count=int(component_count),
            raw_component_count=int(raw_component_count),
            core_component_count=int(core_component_count),
            salvage_method=salvage_method,
            salvaged_fragment=salvaged_fragment,
            salvaged_atom_count=salvaged_atom_count,
            component_salvage_failure_reason=component_salvage_failure_reason,
            component_salvage_stage=component_salvage_stage,
            component_salvage_candidate_count=int(component_salvage_candidate_count),
            component_salvage_best_candidate=component_salvage_best_candidate,
            multi_dummy_hard_fail=bool(multi_dummy_hard_fail),
            dummy_salvage_attempted=bool(dummy_salvage_attempted),
            dummy_salvage_success=bool(dummy_salvage_success),
            dummy_salvage_method=dummy_salvage_method,
            dummy_salvaged_fragment=dummy_salvaged_fragment,
            near_parent_hard_fail=bool(near_parent_hard_fail),
            residual_atom_count=residual_atom_count,
            residual_atom_ratio=residual_atom_ratio,
            tiny_fragment_hard_fail=bool(tiny_fragment_hard_fail),
            fragment_atom_count=int(fragment_atom_count or core_atom_count),
            min_fragment_atoms=int(min_fragment_atoms or self.min_fragment_atoms),
            projection_attempted=bool(projection_attempted),
            projection_success=bool(projection_success),
            projection_method=projection_method,
            projection_score=projection_score,
            projection_source=projection_source,
            projected_fragment_smiles=projected_fragment_smiles,
            projection_atom_count=projection_atom_count,
            projection_atom_ratio=projection_atom_ratio,
            projection_penalty=float(projection_penalty),
            num_projection_candidates=int(num_projection_candidates),
            projection_reason=projection_reason,
            size_window_reward=float(size_window_reward),
            size_window_bucket=size_window_bucket,
            size_window_low=(
                float(size_window_low)
                if size_window_low is not None
                else float(self.size_window_low)
            ),
            size_window_high=(
                float(size_window_high)
                if size_window_high is not None
                else float(self.size_window_high)
            ),
            final_fragment_atom_count=int(final_fragment_atom_count or fragment_atom_count or core_atom_count),
            final_fragment_atom_ratio=final_fragment_atom_ratio,
            error_message=error_message,
            breakdown=dict(breakdown),
        )


__all__ = [
    "ChemRLRewarder",
    "RewardTrace",
    "has_dummy_atom",
    "mol_to_smiles_safe",
    "normalize_fragment_with_dummy_atoms",
    "remove_dummy_atoms_from_mol",
    "shape_probability_reward",
]
