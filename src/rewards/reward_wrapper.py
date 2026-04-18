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
    is_valid_capped_subgraph,
)
from src.rewards.reward_calculator import (
    load_oracle_bundle,
    prepare_smiles_for_oracle,
    smiles_to_morgan_array,
)

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


@dataclass(frozen=True, slots=True)
class RewardTrace:
    """Structured debug record for one PPO reward computation."""

    parent_smiles: str
    generated_smiles: str
    normalized_generated_smiles: str
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
    failure_stage: str | None = None
    error_message: str | None = None
    breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary for logging."""

        return asdict(self)


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
                    "fragment": trace.normalized_generated_smiles,
                    "format": None,
                    "valid": breakdown.get("valid_r"),
                    "substructure": breakdown.get("subgraph_r"),
                    "length": None,
                    "semantic": breakdown.get("cf_r"),
                    "teacher_sem": None,
                    "total": float(trace.reward),
                    "parse_ok": bool(trace.valid_smiles),
                    "substructure_ok": bool(trace.is_subgraph),
                    "connected_ok": bool(trace.connected_fragment),
                    "deletion_ok": bool(trace.deletion_success),
                    "target_probability": trace.target_probability,
                    "failure_stage": trace.failure_stage,
                    "error_message": trace.error_message,
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
                breakdown={"valid_r": 0.0, "subgraph_r": 0.0, "cf_r": self.minimum_reward},
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
                breakdown={"valid_r": 0.0, "subgraph_r": 0.0, "cf_r": self.minimum_reward},
            )

        if not normalized_generated:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="validity",
                error_message="Generated fragment is empty.",
                breakdown={"valid_r": self.minimum_reward, "subgraph_r": 0.0, "cf_r": 0.0},
            )

        # Step A: parseability and connectivity.
        try:
            parent_mol = Chem.MolFromSmiles(normalized_parent, sanitize=True)
            fragment_mol = Chem.MolFromSmiles(normalized_generated, sanitize=True)
        except Exception as exc:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="validity",
                error_message=f"RDKit parse failure: {exc}",
                breakdown={"valid_r": self.minimum_reward, "subgraph_r": 0.0, "cf_r": 0.0},
            )

        if parent_mol is None:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="parent",
                error_message="Parent SMILES could not be parsed by RDKit.",
                breakdown={"valid_r": 0.0, "subgraph_r": 0.0, "cf_r": self.minimum_reward},
            )

        if fragment_mol is None:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="validity",
                error_message="Generated fragment could not be parsed by RDKit.",
                breakdown={"valid_r": self.minimum_reward, "subgraph_r": 0.0, "cf_r": 0.0},
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
                breakdown={"valid_r": self.minimum_reward, "subgraph_r": 0.0, "cf_r": 0.0},
            )

        if not connected_fragment:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="validity",
                error_message="Generated fragment is not connected.",
                breakdown={"valid_r": self.minimum_reward, "subgraph_r": 0.0, "cf_r": 0.0},
            )

        # Step B: subgraph match against the parent molecule.
        try:
            fragment_has_dummy = any(atom.GetAtomicNum() == 0 for atom in fragment_mol.GetAtoms())
            if fragment_has_dummy:
                query_mol = Chem.MolFromSmarts(normalized_generated)
                has_raw_match = bool(
                    query_mol is not None
                    and parent_mol.HasSubstructMatch(query_mol, useChirality=True)
                )
                has_precise_match = bool(
                    has_raw_match and is_valid_capped_subgraph(normalized_parent, normalized_generated)
                )
            else:
                has_precise_match = bool(
                    parent_mol.HasSubstructMatch(fragment_mol, useChirality=True)
                    and is_parent_substructure(normalized_parent, normalized_generated)
                )
        except Exception as exc:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="subgraph",
                error_message=f"Subgraph check failed: {exc}",
                breakdown={"valid_r": 0.0, "subgraph_r": self.minimum_reward, "cf_r": 0.0},
            )

        if not has_precise_match:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="subgraph",
                error_message="Generated fragment is not a valid parent subgraph.",
                breakdown={"valid_r": 0.0, "subgraph_r": self.minimum_reward, "cf_r": 0.0},
                valid_smiles=True,
                connected_fragment=True,
            )

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
                breakdown={"valid_r": 0.0, "subgraph_r": 0.0, "cf_r": self.minimum_reward},
                valid_smiles=True,
                connected_fragment=True,
                is_subgraph=True,
            )

        if residual_smiles is None:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="counterfactual",
                error_message=deletion_error or "Residual deletion failed.",
                breakdown={"valid_r": 0.0, "subgraph_r": 0.0, "cf_r": self.minimum_reward},
                valid_smiles=True,
                connected_fragment=True,
                is_subgraph=True,
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
                breakdown={"valid_r": 0.0, "subgraph_r": 0.0, "cf_r": self.minimum_reward},
                valid_smiles=True,
                connected_fragment=True,
                is_subgraph=True,
                deletion_success=True,
                residual_smiles=residual_smiles,
            )

        if residual_for_oracle in (None, ""):
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="counterfactual",
                error_message="Residual molecule is empty or unusable after dummy-atom cleanup.",
                breakdown={"valid_r": 0.0, "subgraph_r": 0.0, "cf_r": self.minimum_reward},
                valid_smiles=True,
                connected_fragment=True,
                is_subgraph=True,
                deletion_success=True,
                residual_smiles=residual_smiles,
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
                breakdown={"valid_r": 0.0, "subgraph_r": 0.0, "cf_r": self.minimum_reward},
                valid_smiles=True,
                connected_fragment=True,
                is_subgraph=True,
                deletion_success=True,
                residual_smiles=residual_smiles,
            )

        if fingerprint is None:
            return self._fail(
                parent_smiles=normalized_parent,
                generated_smiles=generated_smiles,
                normalized_generated=normalized_generated,
                original_label=int(original_label),
                failure_stage="counterfactual",
                error_message="Morgan fingerprint generation returned None.",
                breakdown={"valid_r": 0.0, "subgraph_r": 0.0, "cf_r": self.minimum_reward},
                valid_smiles=True,
                connected_fragment=True,
                is_subgraph=True,
                deletion_success=True,
                residual_smiles=residual_smiles,
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
                breakdown={"valid_r": 0.0, "subgraph_r": 0.0, "cf_r": self.minimum_reward},
                valid_smiles=True,
                connected_fragment=True,
                is_subgraph=True,
                deletion_success=True,
                residual_smiles=residual_smiles,
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

        return RewardTrace(
            parent_smiles=normalized_parent,
            generated_smiles=str(generated_smiles),
            normalized_generated_smiles=normalized_generated,
            original_label=int(original_label),
            target_label=target_label,
            reward=float(reward),
            valid_smiles=True,
            connected_fragment=True,
            is_subgraph=True,
            deletion_success=True,
            counterfactual_evaluated=True,
            flip_success=target_probability > self.success_threshold,
            target_probability=target_probability,
            inactive_probability=inactive_probability,
            residual_smiles=residual_for_oracle,
            failure_stage=None,
            error_message=None,
            breakdown={
                "valid_r": 0.0,
                "subgraph_r": 0.0,
                "cf_r": float(reward),
            },
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

        parent_mol = Chem.MolFromSmiles(parent_smiles, sanitize=True)
        fragment_mol = Chem.MolFromSmiles(fragment_smiles, sanitize=True)
        if parent_mol is None or fragment_mol is None:
            return None, "Parent or fragment could not be parsed for deletion."

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
    ) -> RewardTrace:
        """Build one failure trace consistently."""

        return RewardTrace(
            parent_smiles=parent_smiles,
            generated_smiles=str(generated_smiles),
            normalized_generated_smiles=normalized_generated,
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
            failure_stage=failure_stage,
            error_message=error_message,
            breakdown=dict(breakdown),
        )


__all__ = [
    "ChemRLRewarder",
    "RewardTrace",
    "shape_probability_reward",
]
