"""Unified PPO reward wrapper for counterfactual molecular fragment generation.

This module intentionally keeps the reward interface small and defensive:

1. Step A checks whether the generated fragment is parseable and connected.
2. Step B checks whether the fragment is a genuine parent subgraph.
3. Step C evaluates the counterfactual effect after deleting the fragment.

Important alignment note:
The repository's source-of-truth documents define the task as deletion-based
counterfactual generation, so the semantic reward is computed on the residual
molecule after deletion rather than on the fragment alone.

Implementation detail:
The rewarder first attempts ``Chem.DeleteSubstructs`` as requested, then checks
whether RDKit removed the expected number of real fragment atoms. If symmetry or
query ambiguity causes over-deletion or under-deletion, the rewarder falls back
to the repository's deterministic single-match deletion helper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
from typing import Any, Sequence

from src.chem import (
    delete_fragment_from_parent,
    is_connected_fragment,
    is_parent_substructure,
    is_rdkit_available,
    parse_smiles,
    sanitize_molecule,
)
from src.rewards.reward_calculator import (
    load_oracle_bundle,
    prepare_smiles_for_oracle,
    smiles_to_morgan_array,
)
from src.rewards.teacher_semantic import TeacherSemanticScorer

try:
    from rdkit import Chem, RDLogger
except ImportError:  # pragma: no cover - depends on local runtime
    Chem = None
    RDLogger = None


_LOGGER = logging.getLogger(__name__)


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
    raw_parse_ok: bool = False
    core_parse_ok: bool = False
    has_dummy_atoms: bool = False
    dummy_count: int = 0
    core_atom_count: int = 0
    teacher_input_smiles: str | None = None
    teacher_available: bool = False
    teacher_called: bool = False
    teacher_probability: float | None = None
    teacher_predicted_label: int | None = None
    teacher_reason: str | None = None
    failure_stage: str | None = None
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


def normalize_fragment_with_dummy_atoms(fragment_smiles: str) -> dict[str, Any]:
    """Build one raw/core fragment view while preserving dummy-atom semantics."""

    normalized = str(fragment_smiles or "").strip()
    normalized_fragment: dict[str, Any] = {
        "raw": normalized,
        "raw_parse_ok": False,
        "raw_sanitized": False,
        "raw_mol": None,
        "raw_canonical_smiles": None,
        "has_dummy": False,
        "core_mol": None,
        "core_smiles": None,
        "core_parse_ok": False,
        "dummy_count": 0,
        "core_atom_count": 0,
    }
    if not normalized or not is_rdkit_available() or Chem is None:
        return normalized_fragment

    parsed_raw = parse_smiles(
        normalized,
        sanitize=True,
        canonicalize=True,
        allow_capped_fragments=True,
    )
    normalized_fragment["raw_parse_ok"] = bool(parsed_raw.parseable)
    normalized_fragment["raw_sanitized"] = bool(parsed_raw.sanitized)
    normalized_fragment["raw_mol"] = parsed_raw.mol
    normalized_fragment["raw_canonical_smiles"] = parsed_raw.canonical_smiles
    if parsed_raw.mol is None:
        return normalized_fragment

    normalized_fragment["has_dummy"] = has_dummy_atom(parsed_raw.mol)
    normalized_fragment["dummy_count"] = _dummy_atom_count(parsed_raw.mol)

    if normalized_fragment["has_dummy"]:
        core_mol = remove_dummy_atoms_from_mol(parsed_raw.mol)
        normalized_fragment["core_mol"] = core_mol
        normalized_fragment["core_smiles"] = mol_to_smiles_safe(core_mol)
        normalized_fragment["core_parse_ok"] = core_mol is not None
        normalized_fragment["core_atom_count"] = _non_dummy_atom_count(core_mol)
        return normalized_fragment

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
    normalized_fragment["core_parse_ok"] = sanitized_core is not None
    normalized_fragment["core_atom_count"] = _non_dummy_atom_count(sanitized_core)
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
        teacher_sem_scale: float = 1.0,
        teacher_sem_missing_penalty: float = -5.0,
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
        self.teacher_sem_scale = float(teacher_sem_scale)
        self.teacher_sem_missing_penalty = float(teacher_sem_missing_penalty)
        self.success_threshold = float(success_threshold)
        self.success_base_reward = float(success_base_reward)
        self.probability_scale = float(probability_scale)

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
                    "semantic": breakdown.get("teacher_sem_r"),
                    "teacher_sem": breakdown.get("teacher_sem_r"),
                    "counterfactual_sem": breakdown.get("cf_r"),
                    "total": float(trace.reward),
                    "parse_ok": bool(trace.raw_parse_ok),
                    "core_parse_ok": bool(trace.core_parse_ok),
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
                    "target_probability": trace.target_probability,
                    "failure_stage": trace.failure_stage,
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
        normalized_generated = str(generated_smiles or "").strip()
        if int(original_label) not in (0, 1):
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=self.default_parent_label,
                failure_stage="label",
                error_message=f"Unsupported original label: {original_label}",
                breakdown=self._build_breakdown(
                    format_reward=0.0,
                    valid_reward=0.0,
                    subgraph_reward=0.0,
                    length_reward=0.0,
                    teacher_reward=0.0,
                    counterfactual_reward=self.minimum_reward,
                ),
                teacher_reason="invalid_or_missing_label",
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
                breakdown=self._build_breakdown(
                    format_reward=0.0,
                    valid_reward=0.0,
                    subgraph_reward=0.0,
                    length_reward=0.0,
                    teacher_reward=0.0,
                    counterfactual_reward=self.minimum_reward,
                ),
                teacher_reason="invalid_or_missing_label",
            )

        if not normalized_generated:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="validity",
                error_message="Generated fragment is empty.",
                breakdown=self._build_breakdown(
                    format_reward=self.format_penalty,
                    valid_reward=self.invalid_smiles_penalty,
                    subgraph_reward=0.0,
                    length_reward=0.0,
                    teacher_reward=self.teacher_sem_missing_penalty,
                    counterfactual_reward=0.0,
                ),
                teacher_reason="invalid_or_not_substructure",
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
                breakdown=self._build_breakdown(
                    format_reward=0.0,
                    valid_reward=0.0,
                    subgraph_reward=0.0,
                    length_reward=0.0,
                    teacher_reward=0.0,
                    counterfactual_reward=self.minimum_reward,
                ),
                teacher_reason="invalid_or_missing_label",
            )

        fragment_info = normalize_fragment_with_dummy_atoms(normalized_generated)
        format_reward = self._compute_format_reward(normalized_generated, fragment_info)
        valid_reward = self._compute_valid_reward(fragment_info)
        length_reward = self._compute_length_reward(fragment_info["core_atom_count"])
        base_trace_kwargs = self._fragment_trace_kwargs(
            fragment_info,
            teacher_input_smiles=fragment_info["core_smiles"],
            teacher_reason="invalid_or_not_substructure",
        )
        if not fragment_info["raw_parse_ok"]:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="validity",
                error_message="Generated fragment could not be parsed by RDKit.",
                breakdown=self._build_breakdown(
                    format_reward=format_reward,
                    valid_reward=valid_reward,
                    subgraph_reward=0.0,
                    length_reward=0.0,
                    teacher_reward=self.teacher_sem_missing_penalty,
                    counterfactual_reward=0.0,
                ),
                **base_trace_kwargs,
            )

        try:
            connected_fragment = bool(is_connected_fragment(normalized_generated))
        except Exception as exc:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="validity",
                error_message=f"Fragment connectivity check failed: {exc}",
                breakdown=self._build_breakdown(
                    format_reward=format_reward,
                    valid_reward=valid_reward,
                    subgraph_reward=0.0,
                    length_reward=length_reward,
                    teacher_reward=self.teacher_sem_missing_penalty,
                    counterfactual_reward=0.0,
                ),
                **base_trace_kwargs,
            )

        if not connected_fragment:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="validity",
                error_message="Generated fragment is not connected.",
                breakdown=self._build_breakdown(
                    format_reward=format_reward,
                    valid_reward=valid_reward,
                    subgraph_reward=0.0,
                    length_reward=length_reward,
                    teacher_reward=self.teacher_sem_missing_penalty,
                    counterfactual_reward=0.0,
                ),
                **base_trace_kwargs,
            )

        # Step B: subgraph match against the parent molecule.
        core_smiles = fragment_info["core_smiles"]
        core_mol = fragment_info["core_mol"]
        if core_mol is None or not fragment_info["core_parse_ok"] or not core_smiles:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="subgraph",
                error_message="Generated fragment did not yield a usable core fragment after dummy-atom normalization.",
                breakdown=self._build_breakdown(
                    format_reward=format_reward,
                    valid_reward=valid_reward,
                    subgraph_reward=self.invalid_subgraph_penalty,
                    length_reward=length_reward,
                    teacher_reward=self.teacher_sem_missing_penalty,
                    counterfactual_reward=0.0,
                ),
                valid_smiles=True,
                connected_fragment=True,
                **base_trace_kwargs,
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
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="subgraph",
                error_message=f"Subgraph check failed: {exc}",
                breakdown=self._build_breakdown(
                    format_reward=format_reward,
                    valid_reward=valid_reward,
                    subgraph_reward=self.invalid_subgraph_penalty,
                    length_reward=length_reward,
                    teacher_reward=self.teacher_sem_missing_penalty,
                    counterfactual_reward=0.0,
                ),
                **base_trace_kwargs,
            )

        if not has_precise_match:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="subgraph",
                error_message="Generated fragment is not a valid parent subgraph.",
                breakdown=self._build_breakdown(
                    format_reward=format_reward,
                    valid_reward=valid_reward,
                    subgraph_reward=self.invalid_subgraph_penalty,
                    length_reward=length_reward,
                    teacher_reward=self.teacher_sem_missing_penalty,
                    counterfactual_reward=0.0,
                ),
                valid_smiles=True,
                connected_fragment=True,
                **base_trace_kwargs,
            )

        teacher_reward, teacher_trace_kwargs = self._score_teacher_semantic(
            core_smiles=core_smiles,
            original_label=int(original_label),
            parent_smiles=normalized_parent,
            fragment_info=fragment_info,
        )
        success_trace_kwargs = {
            **base_trace_kwargs,
            **teacher_trace_kwargs,
            "teacher_input_smiles": core_smiles,
        }

        # Step C: deletion-based counterfactual scoring on the residual molecule.
        try:
            residual_smiles, deletion_error = self._delete_to_residual_smiles(
                normalized_parent,
                normalized_generated,
            )
        except Exception as exc:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="counterfactual",
                error_message=f"Deletion step failed: {exc}",
                breakdown=self._build_breakdown(
                    format_reward=format_reward,
                    valid_reward=valid_reward,
                    subgraph_reward=self.subgraph_pass_reward,
                    length_reward=length_reward,
                    teacher_reward=teacher_reward,
                    counterfactual_reward=self.minimum_reward,
                ),
                valid_smiles=True,
                connected_fragment=True,
                is_subgraph=True,
                **success_trace_kwargs,
            )

        if residual_smiles is None:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="counterfactual",
                error_message=deletion_error or "Residual deletion failed.",
                breakdown=self._build_breakdown(
                    format_reward=format_reward,
                    valid_reward=valid_reward,
                    subgraph_reward=self.subgraph_pass_reward,
                    length_reward=length_reward,
                    teacher_reward=teacher_reward,
                    counterfactual_reward=self.minimum_reward,
                ),
                valid_smiles=True,
                connected_fragment=True,
                is_subgraph=True,
                **success_trace_kwargs,
            )

        try:
            residual_for_oracle = prepare_smiles_for_oracle(residual_smiles)
        except Exception as exc:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="counterfactual",
                error_message=f"Residual cleanup failed: {exc}",
                breakdown=self._build_breakdown(
                    format_reward=format_reward,
                    valid_reward=valid_reward,
                    subgraph_reward=self.subgraph_pass_reward,
                    length_reward=length_reward,
                    teacher_reward=teacher_reward,
                    counterfactual_reward=self.minimum_reward,
                ),
                valid_smiles=True,
                connected_fragment=True,
                is_subgraph=True,
                deletion_success=True,
                residual_smiles=residual_smiles,
                **success_trace_kwargs,
            )

        if residual_for_oracle in (None, ""):
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="counterfactual",
                error_message="Residual molecule is empty or unusable after dummy-atom cleanup.",
                breakdown=self._build_breakdown(
                    format_reward=format_reward,
                    valid_reward=valid_reward,
                    subgraph_reward=self.subgraph_pass_reward,
                    length_reward=length_reward,
                    teacher_reward=teacher_reward,
                    counterfactual_reward=self.minimum_reward,
                ),
                valid_smiles=True,
                connected_fragment=True,
                is_subgraph=True,
                deletion_success=True,
                residual_smiles=residual_smiles,
                **success_trace_kwargs,
            )

        try:
            fingerprint = smiles_to_morgan_array(
                residual_for_oracle,
                radius=self._fingerprint_radius,
                n_bits=self._fingerprint_bits,
                clean_dummy_atoms=False,
            )
        except Exception as exc:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="counterfactual",
                error_message=f"Morgan fingerprint generation failed: {exc}",
                breakdown=self._build_breakdown(
                    format_reward=format_reward,
                    valid_reward=valid_reward,
                    subgraph_reward=self.subgraph_pass_reward,
                    length_reward=length_reward,
                    teacher_reward=teacher_reward,
                    counterfactual_reward=self.minimum_reward,
                ),
                valid_smiles=True,
                connected_fragment=True,
                is_subgraph=True,
                deletion_success=True,
                residual_smiles=residual_smiles,
                **success_trace_kwargs,
            )

        if fingerprint is None:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="counterfactual",
                error_message="Morgan fingerprint generation returned None.",
                breakdown=self._build_breakdown(
                    format_reward=format_reward,
                    valid_reward=valid_reward,
                    subgraph_reward=self.subgraph_pass_reward,
                    length_reward=length_reward,
                    teacher_reward=teacher_reward,
                    counterfactual_reward=self.minimum_reward,
                ),
                valid_smiles=True,
                connected_fragment=True,
                is_subgraph=True,
                deletion_success=True,
                residual_smiles=residual_smiles,
                **success_trace_kwargs,
            )

        try:
            probabilities = self._oracle_model.predict_proba(
                fingerprint.reshape(1, -1)
            )[0]
        except Exception as exc:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="counterfactual",
                error_message=f"Oracle prediction failed: {exc}",
                breakdown=self._build_breakdown(
                    format_reward=format_reward,
                    valid_reward=valid_reward,
                    subgraph_reward=self.subgraph_pass_reward,
                    length_reward=length_reward,
                    teacher_reward=teacher_reward,
                    counterfactual_reward=self.minimum_reward,
                ),
                valid_smiles=True,
                connected_fragment=True,
                is_subgraph=True,
                deletion_success=True,
                residual_smiles=residual_smiles,
                **success_trace_kwargs,
            )

        probability_map = {
            int(label): float(probability)
            for label, probability in zip(self._class_labels, probabilities, strict=True)
        }
        target_probability = float(probability_map.get(target_label, 0.0))
        inactive_probability = float(probability_map.get(0, 0.0))
        reward = shape_probability_reward(
            target_probability,
            success_threshold=self.success_threshold,
            success_base_reward=self.success_base_reward,
            probability_scale=self.probability_scale,
        )
        breakdown = self._build_breakdown(
            format_reward=format_reward,
            valid_reward=valid_reward,
            subgraph_reward=self.subgraph_pass_reward,
            length_reward=length_reward,
            teacher_reward=teacher_reward,
            counterfactual_reward=float(reward),
        )

        return RewardTrace(
            parent_smiles=normalized_parent,
            generated_smiles=str(generated_smiles),
            normalized_generated_smiles=normalized_generated,
            raw_fragment_smiles=normalized_generated,
            core_fragment_smiles=core_smiles,
            original_label=int(original_label),
            target_label=target_label,
            reward=float(sum(breakdown.values())),
            valid_smiles=True,
            connected_fragment=True,
            is_subgraph=True,
            deletion_success=True,
            counterfactual_evaluated=True,
            flip_success=target_probability > self.success_threshold,
            target_probability=target_probability,
            inactive_probability=inactive_probability,
            residual_smiles=residual_for_oracle,
            raw_parse_ok=bool(fragment_info["raw_parse_ok"]),
            core_parse_ok=bool(fragment_info["core_parse_ok"]),
            has_dummy_atoms=bool(fragment_info["has_dummy"]),
            dummy_count=int(fragment_info["dummy_count"]),
            core_atom_count=int(fragment_info["core_atom_count"]),
            teacher_input_smiles=core_smiles,
            teacher_available=bool(success_trace_kwargs.get("teacher_available", False)),
            teacher_called=bool(success_trace_kwargs.get("teacher_called", False)),
            teacher_probability=success_trace_kwargs.get("teacher_probability"),
            teacher_predicted_label=success_trace_kwargs.get("teacher_predicted_label"),
            teacher_reason=success_trace_kwargs.get("teacher_reason"),
            failure_stage=None,
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

    def _build_breakdown(
        self,
        *,
        format_reward: float,
        valid_reward: float,
        subgraph_reward: float,
        length_reward: float,
        teacher_reward: float,
        counterfactual_reward: float,
    ) -> dict[str, float]:
        return {
            "format_r": float(format_reward),
            "valid_r": float(valid_reward),
            "subgraph_r": float(subgraph_reward),
            "length_r": float(length_reward),
            "teacher_sem_r": float(teacher_reward),
            "cf_r": float(counterfactual_reward),
        }

    def _fragment_trace_kwargs(
        self,
        fragment_info: dict[str, Any],
        *,
        teacher_input_smiles: str | None,
        teacher_available: bool = False,
        teacher_called: bool = False,
        teacher_probability: float | None = None,
        teacher_predicted_label: int | None = None,
        teacher_reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "raw_fragment_smiles": fragment_info.get("raw"),
            "core_fragment_smiles": fragment_info.get("core_smiles"),
            "raw_parse_ok": bool(fragment_info.get("raw_parse_ok")),
            "core_parse_ok": bool(fragment_info.get("core_parse_ok")),
            "has_dummy_atoms": bool(fragment_info.get("has_dummy")),
            "dummy_count": int(fragment_info.get("dummy_count", 0)),
            "core_atom_count": int(fragment_info.get("core_atom_count", 0)),
            "teacher_input_smiles": teacher_input_smiles,
            "teacher_available": bool(teacher_available),
            "teacher_called": bool(teacher_called),
            "teacher_probability": teacher_probability,
            "teacher_predicted_label": teacher_predicted_label,
            "teacher_reason": teacher_reason,
        }

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
        core_atom_count: int = 0,
        teacher_input_smiles: str | None = None,
        teacher_available: bool = False,
        teacher_called: bool = False,
        teacher_probability: float | None = None,
        teacher_predicted_label: int | None = None,
        teacher_reason: str | None = None,
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
            reward=float(sum(breakdown.values())),
            valid_smiles=bool(valid_smiles),
            connected_fragment=bool(connected_fragment),
            is_subgraph=bool(is_subgraph),
            deletion_success=bool(deletion_success),
            counterfactual_evaluated=False,
            flip_success=False,
            target_probability=None,
            inactive_probability=None,
            residual_smiles=residual_smiles,
            raw_parse_ok=bool(raw_parse_ok),
            core_parse_ok=bool(core_parse_ok),
            has_dummy_atoms=bool(has_dummy_atoms),
            dummy_count=int(dummy_count),
            core_atom_count=int(core_atom_count),
            teacher_input_smiles=teacher_input_smiles,
            teacher_available=bool(teacher_available),
            teacher_called=bool(teacher_called),
            teacher_probability=teacher_probability,
            teacher_predicted_label=teacher_predicted_label,
            teacher_reason=teacher_reason,
            failure_stage=failure_stage,
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
