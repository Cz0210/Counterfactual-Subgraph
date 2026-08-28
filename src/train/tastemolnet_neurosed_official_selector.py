"""Pure state machine for GREED's batch-interleaved checkpoint selector.

Pinned ``neuro.train.train_full`` evaluates exactly one validation batch before
each training batch.  A strict loss improvement snapshots the current model;
ties count as non-improvements.  Training stops before the paired train update
when the consecutive counter becomes greater than
``cycle_patience * (step_size_up + step_size_down)``.

This module records that ordering without importing PyTorch.  A future trainer
must bind every checkpoint candidate to the bytes captured before its paired
training update; this state machine alone is not a trained-model PASS.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any


SELECTOR_TRACE_SCHEMA = "tastemolnet_neurosed_official_selector_trace_v1"


class OfficialSelectorError(RuntimeError):
    """The caller departed from pinned GREED validation/train ordering."""


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise OfficialSelectorError(f"{label} must be a positive integer")
    return value


def _checkpoint_sha256(value: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise OfficialSelectorError("checkpoint SHA256 is invalid")
    return digest


@dataclass(frozen=True, slots=True)
class OfficialSelectorDecision:
    validation_event_index: int
    training_batch_index: int
    validation_metric: float
    previous_best_validation_metric: float | None
    strict_improvement: bool
    checkpoint_candidate: bool
    consecutive_non_improvement_count: int
    stop_before_training_batch: bool
    paired_training_batch_allowed: bool


class OfficialBatchInterleavedSelector:
    """Enforce the exact selector event ordering in pinned ``train_full``."""

    def __init__(
        self,
        *,
        cycle_patience: int,
        step_size_up: int,
        step_size_down: int,
    ) -> None:
        self.cycle_patience = _positive_int(
            cycle_patience, label="cycle_patience"
        )
        self.step_size_up = _positive_int(step_size_up, label="step_size_up")
        self.step_size_down = _positive_int(
            step_size_down, label="step_size_down"
        )
        self.stop_threshold = self.cycle_patience * (
            self.step_size_up + self.step_size_down
        )
        self._best: float | None = None
        self._non_improvements = 0
        self._completed_train_updates = 0
        self._phase = "EXPECT_VALIDATION"
        self._trace: list[dict[str, Any]] = []
        self._checkpoint_bindings: dict[int, str] = {}
        self._selected_event_index: int | None = None

    @property
    def stopped(self) -> bool:
        return self._phase == "STOPPED"

    def observe_validation(
        self,
        validation_metric: float,
        *,
        training_batch_index: int,
    ) -> OfficialSelectorDecision:
        if self._phase != "EXPECT_VALIDATION":
            raise OfficialSelectorError(
                "validation must occur exactly once before each training batch"
            )
        if (
            type(training_batch_index) is not int
            or training_batch_index != self._completed_train_updates
        ):
            raise OfficialSelectorError("training batch index is not contiguous")
        if isinstance(validation_metric, bool):
            raise OfficialSelectorError("validation metric must be numeric")
        try:
            metric = float(validation_metric)
        except (TypeError, ValueError) as exc:
            raise OfficialSelectorError("validation metric must be numeric") from exc
        if not math.isfinite(metric):
            raise OfficialSelectorError("validation metric must be finite")
        previous = self._best
        improvement = previous is None or metric < previous
        if improvement:
            self._best = metric
            self._non_improvements = 0
            self._selected_event_index = len(self._trace)
        else:
            self._non_improvements += 1
        stop = self._non_improvements > self.stop_threshold
        decision = OfficialSelectorDecision(
            validation_event_index=len(self._trace),
            training_batch_index=training_batch_index,
            validation_metric=metric,
            previous_best_validation_metric=previous,
            strict_improvement=improvement,
            checkpoint_candidate=improvement,
            consecutive_non_improvement_count=self._non_improvements,
            stop_before_training_batch=stop,
            paired_training_batch_allowed=not stop,
        )
        row = asdict(decision)
        row.update(
            {
                "checkpoint_sha256": None,
                "training_update_completed": False,
                "optimizer_step_completed": False,
                "cyclic_lr_step_completed": False,
                "gradient_clip_norm": None,
            }
        )
        self._trace.append(row)
        self._phase = "STOPPED" if stop else "EXPECT_TRAIN_UPDATE"
        return decision

    def bind_checkpoint_candidate(
        self,
        *,
        validation_event_index: int,
        checkpoint_sha256: str,
    ) -> None:
        if type(validation_event_index) is not int or not (
            0 <= validation_event_index < len(self._trace)
        ):
            raise OfficialSelectorError("checkpoint validation event is invalid")
        row = self._trace[validation_event_index]
        if row["checkpoint_candidate"] is not True:
            raise OfficialSelectorError("non-improvement cannot bind a checkpoint")
        if validation_event_index in self._checkpoint_bindings:
            raise OfficialSelectorError("checkpoint candidate was already bound")
        digest = _checkpoint_sha256(checkpoint_sha256)
        self._checkpoint_bindings[validation_event_index] = digest
        row["checkpoint_sha256"] = digest

    def record_training_update(
        self,
        *,
        training_batch_index: int,
        optimizer_step_completed: bool,
        cyclic_lr_step_completed: bool,
        gradient_clip_norm: float,
    ) -> None:
        if self._phase != "EXPECT_TRAIN_UPDATE":
            raise OfficialSelectorError(
                "training update lacks its immediately preceding validation event"
            )
        if (
            type(training_batch_index) is not int
            or training_batch_index != self._completed_train_updates
        ):
            raise OfficialSelectorError("training update index is not contiguous")
        if optimizer_step_completed is not True or cyclic_lr_step_completed is not True:
            raise OfficialSelectorError("official optimizer/scheduler step was skipped")
        if isinstance(gradient_clip_norm, bool):
            raise OfficialSelectorError("official gradient clipping must equal 0.1")
        try:
            clip_norm = float(gradient_clip_norm)
        except (TypeError, ValueError) as exc:
            raise OfficialSelectorError(
                "official gradient clipping must equal 0.1"
            ) from exc
        if clip_norm != 0.1:
            raise OfficialSelectorError("official gradient clipping must equal 0.1")
        row = self._trace[-1]
        if row["training_batch_index"] != training_batch_index:
            raise OfficialSelectorError("validation/train pair index changed")
        if (
            row["checkpoint_candidate"] is True
            and row["validation_event_index"] not in self._checkpoint_bindings
        ):
            raise OfficialSelectorError(
                "checkpoint candidate must bind pre-update bytes before training"
            )
        row.update(
            {
                "training_update_completed": True,
                "optimizer_step_completed": True,
                "cyclic_lr_step_completed": True,
                "gradient_clip_norm": 0.1,
            }
        )
        self._completed_train_updates += 1
        self._phase = "EXPECT_VALIDATION"

    def trace_manifest(self) -> dict[str, Any]:
        if not self.stopped:
            raise OfficialSelectorError("official selector has not reached its stop event")
        if self._selected_event_index is None:
            raise OfficialSelectorError("official selector never selected a checkpoint")
        candidates = {
            index for index, row in enumerate(self._trace) if row["checkpoint_candidate"]
        }
        if candidates != set(self._checkpoint_bindings):
            raise OfficialSelectorError("one or more checkpoint candidates are unbound")
        stop_row = self._trace[-1]
        if (
            stop_row["stop_before_training_batch"] is not True
            or stop_row["training_update_completed"] is not False
            or self._selected_event_index not in self._checkpoint_bindings
        ):
            raise OfficialSelectorError("official stop/checkpoint state is incomplete")
        payload = {
            "schema_version": SELECTOR_TRACE_SCHEMA,
            "status": "READY_FOR_INDEPENDENT_VERIFICATION",
            "selector_contract": "neuro.train.train_full_batch_interleaved_validation",
            "validation_before_every_training_batch": True,
            "strictly_lower_validation_loss_selects_checkpoint": True,
            "validation_tie_is_non_improvement": True,
            "stop_comparison": "consecutive_non_improvements > threshold",
            "cycle_patience": self.cycle_patience,
            "step_size_up": self.step_size_up,
            "step_size_down": self.step_size_down,
            "non_improvement_stop_threshold": self.stop_threshold,
            "validation_event_count": len(self._trace),
            "completed_training_batch_count": self._completed_train_updates,
            "stopping_validation_event_index": stop_row["validation_event_index"],
            "stopped_before_paired_training_batch": True,
            "selected_validation_event_index": self._selected_event_index,
            "selected_checkpoint_sha256": self._checkpoint_bindings[
                self._selected_event_index
            ],
            "tie_break": "none",
            "epoch_end_validation_used": False,
            "trace": [dict(row) for row in self._trace],
        }
        payload["trace_sha256"] = _stable_sha256(payload)
        return payload


__all__ = [
    "OfficialBatchInterleavedSelector",
    "OfficialSelectorDecision",
    "OfficialSelectorError",
    "SELECTOR_TRACE_SCHEMA",
]
