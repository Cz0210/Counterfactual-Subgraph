"""Deletion-based teacher-oracle scoring for counterfactual PPO rewards."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from src.chem.smiles_utils import is_rdkit_available, parse_smiles
from src.rewards.teacher_semantic import TeacherSemanticScorer

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - depends on local runtime
    Chem = None


@dataclass(frozen=True, slots=True)
class CounterfactualTeacherResult:
    teacher_available: bool
    teacher_input_parent_smiles: str
    teacher_input_fragment_smiles: str
    parent_parse_ok: bool
    fragment_parse_ok: bool
    substructure_ok: bool
    deletion_ok: bool
    parent_without_fragment_smiles: str | None
    p_before: float | None
    p_after: float | None
    pred_before: int | None
    pred_after: int | None
    cf_drop: float | None
    cf_flip: bool
    counterfactual_sem: float
    teacher_sem: float
    teacher_reason: str
    teacher_result_ok: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "teacher_available": self.teacher_available,
            "teacher_input_parent_smiles": self.teacher_input_parent_smiles,
            "teacher_input_fragment_smiles": self.teacher_input_fragment_smiles,
            "parent_parse_ok": self.parent_parse_ok,
            "fragment_parse_ok": self.fragment_parse_ok,
            "substructure_ok": self.substructure_ok,
            "deletion_ok": self.deletion_ok,
            "parent_without_fragment_smiles": self.parent_without_fragment_smiles,
            "p_before": self.p_before,
            "p_after": self.p_after,
            "pred_before": self.pred_before,
            "pred_after": self.pred_after,
            "cf_drop": self.cf_drop,
            "cf_flip": self.cf_flip,
            "counterfactual_sem": self.counterfactual_sem,
            "teacher_sem": self.teacher_sem,
            "teacher_reason": self.teacher_reason,
            "teacher_result_ok": self.teacher_result_ok,
        }


def _clear_broken_aromatic_flags(mol: object) -> None:
    if Chem is None:
        return
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic() and not atom.IsInRing():
            atom.SetIsAromatic(False)
    for bond in mol.GetBonds():
        if bond.GetIsAromatic() and not bond.IsInRing():
            bond.SetIsAromatic(False)
            bond.SetBondType(Chem.BondType.SINGLE)


def delete_one_substructure(parent_smiles: str, fragment_smiles: str) -> dict[str, Any]:
    """Delete exactly one fragment match from the parent molecule."""

    result: dict[str, Any] = {
        "parent_parse_ok": False,
        "fragment_parse_ok": False,
        "substructure_ok": False,
        "deletion_ok": False,
        "parent_without_fragment_smiles": None,
        "match_atom_indices": [],
        "reason": "uninitialized",
    }

    if not is_rdkit_available() or Chem is None:
        result["reason"] = "rdkit_unavailable"
        return result

    normalized_parent = str(parent_smiles or "").strip()
    normalized_fragment = str(fragment_smiles or "").strip()
    parent = parse_smiles(
        normalized_parent,
        sanitize=True,
        canonicalize=False,
        allow_capped_fragments=False,
    )
    fragment = parse_smiles(
        normalized_fragment,
        sanitize=True,
        canonicalize=False,
        allow_capped_fragments=False,
    )
    result["parent_parse_ok"] = bool(parent.sanitized and parent.mol is not None)
    result["fragment_parse_ok"] = bool(fragment.sanitized and fragment.mol is not None)
    if not result["parent_parse_ok"]:
        result["reason"] = (
            f"parent_parse_failed:{parent.failure_reason}"
            if parent.failure_reason
            else "parent_parse_failed"
        )
        return result
    if not result["fragment_parse_ok"]:
        result["reason"] = (
            f"fragment_parse_failed:{fragment.failure_reason}"
            if fragment.failure_reason
            else "fragment_parse_failed"
        )
        return result

    matches = parent.mol.GetSubstructMatches(
        fragment.mol,
        useChirality=True,
        uniquify=True,
    )
    if not matches:
        result["reason"] = "fragment_not_substructure"
        return result

    selected_match = tuple(int(index) for index in matches[0])
    result["substructure_ok"] = True
    result["match_atom_indices"] = list(selected_match)

    rw_mol = Chem.RWMol(parent.mol)
    for atom_index in sorted(selected_match, reverse=True):
        rw_mol.RemoveAtom(int(atom_index))

    residual_mol = rw_mol.GetMol()
    if residual_mol.GetNumAtoms() == 0:
        result["reason"] = "empty_residual_after_deletion"
        return result

    try:
        Chem.SanitizeMol(residual_mol)
    except Exception:
        _clear_broken_aromatic_flags(residual_mol)
        try:
            Chem.SanitizeMol(residual_mol)
        except Exception as exc:
            result["reason"] = f"residual_sanitize_failed:{exc}"
            return result

    try:
        residual_smiles = Chem.MolToSmiles(residual_mol, canonical=True)
    except Exception as exc:
        result["reason"] = f"residual_smiles_failed:{exc}"
        return result

    if not residual_smiles:
        result["reason"] = "empty_residual_smiles"
        return result

    result["deletion_ok"] = True
    result["parent_without_fragment_smiles"] = residual_smiles
    result["reason"] = "ok"
    return result


class CounterfactualTeacherScorer:
    """Deletion-based semantic scorer using parent and residual teacher scores."""

    def __init__(
        self,
        teacher_path: str | Path | None,
        device: str | object = "cpu",
        logger: logging.Logger | None = None,
        flip_bonus: float = 1.0,
        missing_penalty: float = -5.0,
        teacher_scorer: TeacherSemanticScorer | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.flip_bonus = float(flip_bonus)
        self.missing_penalty = float(missing_penalty)
        self.teacher_scorer = teacher_scorer or TeacherSemanticScorer(
            teacher_path=teacher_path,
            device=device,
            logger=self.logger,
        )
        self.available = bool(self.teacher_scorer.available)
        self.availability_reason = self.teacher_scorer.availability_reason

    def score_counterfactual(
        self,
        parent_smiles: str,
        core_fragment_smiles: str,
        label: int,
        raw_fragment_smiles: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del raw_fragment_smiles, meta

        normalized_parent = str(parent_smiles or "").strip()
        normalized_fragment = str(core_fragment_smiles or "").strip()
        if not self.available:
            return CounterfactualTeacherResult(
                teacher_available=False,
                teacher_input_parent_smiles=normalized_parent,
                teacher_input_fragment_smiles=normalized_fragment,
                parent_parse_ok=False,
                fragment_parse_ok=False,
                substructure_ok=False,
                deletion_ok=False,
                parent_without_fragment_smiles=None,
                p_before=None,
                p_after=None,
                pred_before=None,
                pred_after=None,
                cf_drop=None,
                cf_flip=False,
                counterfactual_sem=self.missing_penalty,
                teacher_sem=self.missing_penalty,
                teacher_reason=self.availability_reason,
                teacher_result_ok=False,
            ).to_dict()

        if int(label) not in (0, 1):
            return CounterfactualTeacherResult(
                teacher_available=True,
                teacher_input_parent_smiles=normalized_parent,
                teacher_input_fragment_smiles=normalized_fragment,
                parent_parse_ok=False,
                fragment_parse_ok=False,
                substructure_ok=False,
                deletion_ok=False,
                parent_without_fragment_smiles=None,
                p_before=None,
                p_after=None,
                pred_before=None,
                pred_after=None,
                cf_drop=None,
                cf_flip=False,
                counterfactual_sem=self.missing_penalty,
                teacher_sem=self.missing_penalty,
                teacher_reason=f"unsupported_teacher_label:{label}",
                teacher_result_ok=False,
            ).to_dict()

        deletion_result = delete_one_substructure(
            normalized_parent,
            normalized_fragment,
        )
        if not deletion_result["deletion_ok"]:
            return CounterfactualTeacherResult(
                teacher_available=True,
                teacher_input_parent_smiles=normalized_parent,
                teacher_input_fragment_smiles=normalized_fragment,
                parent_parse_ok=bool(deletion_result["parent_parse_ok"]),
                fragment_parse_ok=bool(deletion_result["fragment_parse_ok"]),
                substructure_ok=bool(deletion_result["substructure_ok"]),
                deletion_ok=False,
                parent_without_fragment_smiles=deletion_result["parent_without_fragment_smiles"],
                p_before=None,
                p_after=None,
                pred_before=None,
                pred_after=None,
                cf_drop=None,
                cf_flip=False,
                counterfactual_sem=self.missing_penalty,
                teacher_sem=self.missing_penalty,
                teacher_reason=str(deletion_result["reason"]),
                teacher_result_ok=False,
            ).to_dict()

        before_result = self.teacher_scorer.score_smiles(
            normalized_parent,
            label=int(label),
            parent_smiles=normalized_parent,
            meta={"counterfactual_role": "parent"},
        )
        if not before_result.get("teacher_result_ok"):
            return CounterfactualTeacherResult(
                teacher_available=bool(before_result.get("teacher_available")),
                teacher_input_parent_smiles=normalized_parent,
                teacher_input_fragment_smiles=normalized_fragment,
                parent_parse_ok=bool(deletion_result["parent_parse_ok"]),
                fragment_parse_ok=bool(deletion_result["fragment_parse_ok"]),
                substructure_ok=bool(deletion_result["substructure_ok"]),
                deletion_ok=bool(deletion_result["deletion_ok"]),
                parent_without_fragment_smiles=deletion_result["parent_without_fragment_smiles"],
                p_before=None,
                p_after=None,
                pred_before=None,
                pred_after=None,
                cf_drop=None,
                cf_flip=False,
                counterfactual_sem=self.missing_penalty,
                teacher_sem=self.missing_penalty,
                teacher_reason=f"parent_teacher_failed:{before_result.get('teacher_reason')}",
                teacher_result_ok=False,
            ).to_dict()

        residual_smiles = str(deletion_result["parent_without_fragment_smiles"] or "")
        after_result = self.teacher_scorer.score_smiles(
            residual_smiles,
            label=int(label),
            parent_smiles=normalized_parent,
            meta={"counterfactual_role": "residual"},
        )
        if not after_result.get("teacher_result_ok"):
            return CounterfactualTeacherResult(
                teacher_available=bool(after_result.get("teacher_available")),
                teacher_input_parent_smiles=normalized_parent,
                teacher_input_fragment_smiles=normalized_fragment,
                parent_parse_ok=bool(deletion_result["parent_parse_ok"]),
                fragment_parse_ok=bool(deletion_result["fragment_parse_ok"]),
                substructure_ok=bool(deletion_result["substructure_ok"]),
                deletion_ok=bool(deletion_result["deletion_ok"]),
                parent_without_fragment_smiles=residual_smiles,
                p_before=float(before_result.get("teacher_prob"))
                if before_result.get("teacher_prob") is not None
                else None,
                p_after=None,
                pred_before=int(before_result.get("teacher_label"))
                if before_result.get("teacher_label") is not None
                else None,
                pred_after=None,
                cf_drop=None,
                cf_flip=False,
                counterfactual_sem=self.missing_penalty,
                teacher_sem=self.missing_penalty,
                teacher_reason=f"residual_teacher_failed:{after_result.get('teacher_reason')}",
                teacher_result_ok=False,
            ).to_dict()

        p_before = float(before_result["teacher_prob"])
        p_after = float(after_result["teacher_prob"])
        pred_before = int(before_result["teacher_label"])
        pred_after = int(after_result["teacher_label"])
        cf_drop = float(p_before - p_after)
        cf_flip = bool(pred_after != int(label))
        counterfactual_sem = float(cf_drop + self.flip_bonus * float(cf_flip))
        return CounterfactualTeacherResult(
            teacher_available=True,
            teacher_input_parent_smiles=normalized_parent,
            teacher_input_fragment_smiles=normalized_fragment,
            parent_parse_ok=bool(deletion_result["parent_parse_ok"]),
            fragment_parse_ok=bool(deletion_result["fragment_parse_ok"]),
            substructure_ok=bool(deletion_result["substructure_ok"]),
            deletion_ok=bool(deletion_result["deletion_ok"]),
            parent_without_fragment_smiles=residual_smiles,
            p_before=p_before,
            p_after=p_after,
            pred_before=pred_before,
            pred_after=pred_after,
            cf_drop=cf_drop,
            cf_flip=cf_flip,
            counterfactual_sem=counterfactual_sem,
            teacher_sem=counterfactual_sem,
            teacher_reason="ok",
            teacher_result_ok=True,
        ).to_dict()


__all__ = [
    "CounterfactualTeacherResult",
    "CounterfactualTeacherScorer",
    "delete_one_substructure",
]
