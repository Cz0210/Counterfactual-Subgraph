"""Teacher-semantic scoring for decoded fragment rewards.

The repository currently ships one readily available classifier artifact:

- a scikit-learn style bundle containing ``model.predict_proba(...)`` plus
  Morgan fingerprint settings.

This module wraps that format explicitly and provides a narrow compatibility
layer for torch models only when the checkpoint itself exposes the required
fingerprint metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.rewards.reward_calculator import load_oracle_bundle, smiles_to_morgan_array

try:
    import torch
except ImportError:  # pragma: no cover - depends on local runtime
    torch = None


@dataclass(frozen=True, slots=True)
class TeacherSemanticResult:
    teacher_available: bool
    teacher_input_smiles: str
    teacher_parse_ok: bool
    teacher_prob: float | None
    teacher_label: int | None
    teacher_sem: float
    teacher_reason: str
    teacher_result_ok: bool = False
    teacher_format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "teacher_available": self.teacher_available,
            "teacher_input_smiles": self.teacher_input_smiles,
            "teacher_parse_ok": self.teacher_parse_ok,
            "teacher_prob": self.teacher_prob,
            "teacher_label": self.teacher_label,
            "teacher_sem": self.teacher_sem,
            "teacher_reason": self.teacher_reason,
            "teacher_result_ok": self.teacher_result_ok,
            "teacher_format": self.teacher_format,
        }


class TeacherSemanticScorer:
    """Score one core fragment with a teacher classifier."""

    def __init__(
        self,
        teacher_path: str | Path | None,
        device: str | object = "cpu",
        logger: logging.Logger | None = None,
    ) -> None:
        self.teacher_path = (
            Path(teacher_path).expanduser().resolve()
            if teacher_path not in (None, "")
            else None
        )
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.available = False
        self.availability_reason = "teacher_path_not_provided"
        self.teacher_format: str | None = None
        self.model: Any = None
        self.fingerprint_radius: int | None = None
        self.fingerprint_bits: int | None = None
        self._load()

    def _load(self) -> None:
        if self.teacher_path is None:
            self.available = False
            self.availability_reason = "teacher_path_not_provided"
            return
        if not self.teacher_path.exists():
            self.available = False
            self.availability_reason = f"teacher_file_not_found:{self.teacher_path}"
            return

        bundle_error: Exception | None = None
        try:
            bundle = load_oracle_bundle(self.teacher_path)
        except Exception as exc:
            bundle = None
            bundle_error = exc

        if bundle is not None:
            model = bundle.get("model")
            if hasattr(model, "predict_proba"):
                self.model = model
                self.fingerprint_radius = int(bundle["fingerprint_radius"])
                self.fingerprint_bits = int(bundle["fingerprint_bits"])
                self.teacher_format = "sklearn_bundle"
                self.available = True
                self.availability_reason = "ok"
                return

        if torch is None:
            self.available = False
            if bundle_error is not None:
                self.availability_reason = (
                    f"unsupported_teacher_format_without_torch:{bundle_error}"
                )
            else:
                self.availability_reason = "unsupported_teacher_format_without_torch"
            return

        try:
            payload = torch.load(str(self.teacher_path), map_location=self.device)
        except Exception as exc:
            self.available = False
            self.availability_reason = f"teacher_load_failed:{exc}"
            return

        self._configure_from_payload(payload)
        if not self.available and bundle_error is not None:
            self.availability_reason = (
                f"{self.availability_reason}; bundle_probe_failed:{bundle_error}"
            )

    def _configure_from_payload(self, payload: Any) -> None:
        candidate_model = None
        radius = None
        n_bits = None

        if isinstance(payload, dict):
            if {"model", "fingerprint_radius", "fingerprint_bits"}.issubset(payload):
                candidate_model = payload.get("model")
                radius = payload.get("fingerprint_radius")
                n_bits = payload.get("fingerprint_bits")
            else:
                self.available = False
                self.availability_reason = (
                    "unsupported_teacher_payload_dict_missing_model_or_fingerprint_config"
                )
                return
        else:
            candidate_model = payload
            radius = getattr(payload, "fingerprint_radius", None)
            n_bits = getattr(payload, "fingerprint_bits", None)

        if hasattr(candidate_model, "predict_proba"):
            if radius is None or n_bits is None:
                self.available = False
                self.availability_reason = (
                    "teacher_predict_proba_model_missing_fingerprint_config"
                )
                return
            self.model = candidate_model
            self.fingerprint_radius = int(radius)
            self.fingerprint_bits = int(n_bits)
            self.teacher_format = "predict_proba_model"
            self.available = True
            self.availability_reason = "ok"
            return

        if torch is not None and isinstance(candidate_model, torch.nn.Module):
            if radius is None or n_bits is None:
                self.available = False
                self.availability_reason = (
                    "torch_teacher_model_missing_fingerprint_config"
                )
                return
            candidate_model.to(self.device)
            candidate_model.eval()
            self.model = candidate_model
            self.fingerprint_radius = int(radius)
            self.fingerprint_bits = int(n_bits)
            self.teacher_format = "torch_logits_model"
            self.available = True
            self.availability_reason = "ok"
            return

        self.available = False
        self.availability_reason = "unsupported_teacher_payload"

    def score_smiles(
        self,
        smiles: str,
        label: int | None = None,
        parent_smiles: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del parent_smiles, meta

        normalized_smiles = str(smiles or "").strip()
        if not self.available:
            return TeacherSemanticResult(
                teacher_available=False,
                teacher_input_smiles=normalized_smiles,
                teacher_parse_ok=False,
                teacher_prob=None,
                teacher_label=None,
                teacher_sem=0.0,
                teacher_reason=self.availability_reason,
                teacher_result_ok=False,
                teacher_format=self.teacher_format,
            ).to_dict()

        if label not in (0, 1):
            return TeacherSemanticResult(
                teacher_available=True,
                teacher_input_smiles=normalized_smiles,
                teacher_parse_ok=False,
                teacher_prob=None,
                teacher_label=None,
                teacher_sem=0.0,
                teacher_reason=f"unsupported_teacher_label:{label}",
                teacher_result_ok=False,
                teacher_format=self.teacher_format,
            ).to_dict()

        if not normalized_smiles:
            return TeacherSemanticResult(
                teacher_available=True,
                teacher_input_smiles=normalized_smiles,
                teacher_parse_ok=False,
                teacher_prob=None,
                teacher_label=None,
                teacher_sem=0.0,
                teacher_reason="empty_teacher_input",
                teacher_result_ok=False,
                teacher_format=self.teacher_format,
            ).to_dict()

        fingerprint = smiles_to_morgan_array(
            normalized_smiles,
            radius=int(self.fingerprint_radius or 2),
            n_bits=int(self.fingerprint_bits or 2048),
            clean_dummy_atoms=False,
        )
        if fingerprint is None:
            return TeacherSemanticResult(
                teacher_available=True,
                teacher_input_smiles=normalized_smiles,
                teacher_parse_ok=False,
                teacher_prob=None,
                teacher_label=None,
                teacher_sem=0.0,
                teacher_reason="teacher_input_fingerprint_failed",
                teacher_result_ok=False,
                teacher_format=self.teacher_format,
            ).to_dict()

        try:
            probabilities = self._predict_probabilities(fingerprint)
        except Exception as exc:
            return TeacherSemanticResult(
                teacher_available=True,
                teacher_input_smiles=normalized_smiles,
                teacher_parse_ok=True,
                teacher_prob=None,
                teacher_label=None,
                teacher_sem=0.0,
                teacher_reason=f"teacher_forward_failed:{exc}",
                teacher_result_ok=False,
                teacher_format=self.teacher_format,
            ).to_dict()

        if probabilities.ndim != 1 or probabilities.size < 2:
            return TeacherSemanticResult(
                teacher_available=True,
                teacher_input_smiles=normalized_smiles,
                teacher_parse_ok=True,
                teacher_prob=None,
                teacher_label=None,
                teacher_sem=0.0,
                teacher_reason="teacher_probabilities_invalid_shape",
                teacher_result_ok=False,
                teacher_format=self.teacher_format,
            ).to_dict()

        teacher_prob = float(probabilities[int(label)])
        teacher_label = int(np.argmax(probabilities))
        teacher_sem = float(2.0 * teacher_prob - 1.0)
        return TeacherSemanticResult(
            teacher_available=True,
            teacher_input_smiles=normalized_smiles,
            teacher_parse_ok=True,
            teacher_prob=teacher_prob,
            teacher_label=teacher_label,
            teacher_sem=teacher_sem,
            teacher_reason="ok",
            teacher_result_ok=True,
            teacher_format=self.teacher_format,
        ).to_dict()

    def _predict_probabilities(self, fingerprint: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(fingerprint.reshape(1, -1))[0]
            return np.asarray(probabilities, dtype=np.float32)

        if torch is None or not isinstance(self.model, torch.nn.Module):
            raise RuntimeError("Teacher model does not expose a supported prediction API.")

        features = torch.tensor(
            fingerprint,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(features)
        logits = self._extract_logits(outputs)
        if logits.ndim == 0:
            positive_probability = float(torch.sigmoid(logits).item())
            return np.asarray([1.0 - positive_probability, positive_probability], dtype=np.float32)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if logits.shape[-1] == 1:
            positive_probability = float(torch.sigmoid(logits[0, 0]).item())
            return np.asarray([1.0 - positive_probability, positive_probability], dtype=np.float32)
        probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        return np.asarray(probabilities, dtype=np.float32)

    def _extract_logits(self, outputs: Any) -> Any:
        if torch is None:
            raise RuntimeError("Torch is unavailable for teacher-logit extraction.")
        if torch.is_tensor(outputs):
            return outputs
        if hasattr(outputs, "logits"):
            return outputs.logits
        if isinstance(outputs, dict) and "logits" in outputs:
            return outputs["logits"]
        raise RuntimeError("Torch teacher output did not expose logits.")


def require_teacher_semantic_scorer(
    scorer: TeacherSemanticScorer | None,
    *,
    teacher_path: str | Path | None,
) -> TeacherSemanticScorer:
    """Raise a clear error when teacher semantics are required but unavailable."""

    if scorer is None or not scorer.available:
        path_text = str(teacher_path) if teacher_path not in (None, "") else "<unset>"
        reason = (
            scorer.availability_reason
            if scorer is not None
            else "teacher_scorer_not_created"
        )
        raise RuntimeError(
            "Teacher semantic reward was required but unavailable. "
            f"path={path_text} reason={reason}"
        )
    return scorer


__all__ = [
    "TeacherSemanticResult",
    "TeacherSemanticScorer",
    "require_teacher_semantic_scorer",
]
