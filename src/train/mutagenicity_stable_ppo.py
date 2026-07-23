"""Mutagenicity-specific contracts and audit hooks for the shared stable PPO loop."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.train_ppo import PromptExample
from src.data.mutagenicity_continued_sft import (
    audit_single_trainable_lora_adapter,
    validate_peft_checkpoint,
    write_csv_atomic,
    write_json_atomic,
)
from src.rewards.counterfactual_oracle import CounterfactualTeacherScorer


SOURCE_LABEL = 1
TARGET_LABEL = 0
EXPECTED_TRAIN_ROWS = 1448
EXPECTED_VAL_ROWS = 260
EXPECTED_POLICY_CHECKPOINT_STEP = 200

REQUIRED_DATA_FIELDS = (
    "molecule_id",
    "parent_smiles",
    "label",
    "source_label",
    "target_label",
    "split",
    "scaffold_smiles",
    "teacher_pred",
    "teacher_prob_0",
    "teacher_prob_1",
    "teacher_correct",
    "prompt",
)

REQUIRED_CANDIDATE_FIELDS = (
    "molecule_id",
    "parent_smiles",
    "prompt",
    "generated_text",
    "raw_fragment",
    "core_fragment",
    "final_fragment",
    "residual_smiles",
    "parse_ok",
    "connected",
    "direct_substructure",
    "final_substructure",
    "projection_used",
    "projection_failed",
    "oracle_ok",
    "pred_before",
    "pred_after",
    "prob_before_0",
    "prob_before_1",
    "prob_after_0",
    "prob_after_1",
    "cf_drop",
    "cf_flip",
    "target_prob_gain",
    "atom_ratio",
    "reward_total",
    "reward_components",
    "global_step",
)


@dataclass(frozen=True, slots=True)
class MutagenicityPPORecord:
    row_index: int
    molecule_id: str
    parent_smiles: str
    canonical_parent_smiles: str
    prompt: str
    label: int
    source_label: int
    target_label: int
    split: str
    scaffold_smiles: str
    teacher_pred: int
    teacher_prob_0: float
    teacher_prob_1: float
    teacher_correct: bool

    def to_prompt_example(self) -> PromptExample:
        return PromptExample(
            index=int(self.row_index),
            prompt=self.prompt,
            parent_smiles=self.parent_smiles,
            original_label=int(self.label),
            molecule_id=self.molecule_id,
        )


@dataclass(frozen=True, slots=True)
class ParentCoveragePlan:
    num_dataset_rows: int
    num_unique_dataset_parents: int
    rollout_batch_size: int
    samples_per_update: int
    updates_per_epoch: int
    max_updates: int
    planned_epochs_equivalent: float
    sampler_seed: int
    shuffle_enabled: bool
    sampling_with_replacement: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
        }


def _parse_bool(value: Any, *, field: str, row_index: int) -> bool:
    token = str(value).strip().lower()
    if token in {"1", "true", "t", "yes", "y"}:
        return True
    if token in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"row={row_index} has invalid {field}={value!r}")


def _parse_binary(value: Any, *, field: str, row_index: int) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row={row_index} has invalid {field}={value!r}") from exc
    if parsed not in (0, 1):
        raise ValueError(f"row={row_index} has non-binary {field}={parsed}")
    return parsed


def _parse_probability(value: Any, *, field: str, row_index: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row={row_index} has invalid {field}={value!r}") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"row={row_index} has out-of-range {field}={value!r}")
    return parsed


def _reject_nontrain_path(path: Path, *, expected_split: str) -> None:
    if expected_split not in {"train", "val"}:
        raise ValueError(f"PPO accepts only train/val, received {expected_split!r}")
    path_tokens = {part.lower() for part in path.parts}
    if "calibration" in path_tokens or "test" in path_tokens:
        raise ValueError(f"PPO must not read calibration/test data: {path}")
    stem = path.stem.lower()
    if "calibration" in stem or stem == "test" or "_test" in stem:
        raise ValueError(f"PPO must not read calibration/test data: {path}")


def load_mutagenicity_ppo_records(
    path: str | Path,
    *,
    expected_split: str,
    expected_count: int | None,
) -> list[MutagenicityPPORecord]:
    """Load the fixed teacher-consistent source-label prompt view."""

    csv_path = Path(path).expanduser().resolve()
    _reject_nontrain_path(csv_path, expected_split=expected_split)
    if not csv_path.is_file() or csv_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Mutagenicity PPO CSV is missing or empty: {csv_path}")

    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - HPC dependency
        raise RuntimeError("Mutagenicity PPO data validation requires RDKit") from exc

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = sorted(set(REQUIRED_DATA_FIELDS) - set(fieldnames))
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")
        rows = [dict(row) for row in reader]

    if expected_count is not None and len(rows) != int(expected_count):
        raise ValueError(
            f"{expected_split} row count mismatch: expected={expected_count} actual={len(rows)}"
        )
    if not rows:
        raise ValueError(f"Mutagenicity PPO CSV has no rows: {csv_path}")

    records: list[MutagenicityPPORecord] = []
    seen_ids: set[str] = set()
    seen_smiles: set[str] = set()
    for row_index, row in enumerate(rows):
        molecule_id = str(row.get("molecule_id") or "").strip()
        parent_smiles = str(row.get("parent_smiles") or "").strip()
        prompt = str(row.get("prompt") or "").strip()
        split = str(row.get("split") or "").strip().lower()
        scaffold = str(row.get("scaffold_smiles") or "").strip()
        if not molecule_id:
            raise ValueError(f"row={row_index} has empty molecule_id")
        if molecule_id in seen_ids:
            raise ValueError(f"duplicate molecule_id in {csv_path}: {molecule_id}")
        molecule = Chem.MolFromSmiles(parent_smiles)
        if molecule is None or molecule.GetNumAtoms() <= 0:
            raise ValueError(f"row={row_index} has invalid parent_smiles")
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        if canonical in seen_smiles:
            raise ValueError(f"duplicate canonical parent SMILES in {csv_path}: {canonical}")
        if not prompt:
            raise ValueError(f"row={row_index} has empty prompt")
        if split != expected_split:
            raise ValueError(
                f"row={row_index} split mismatch: expected={expected_split} actual={split}"
            )

        label = _parse_binary(row.get("label"), field="label", row_index=row_index)
        source_label = _parse_binary(
            row.get("source_label"), field="source_label", row_index=row_index
        )
        target_label = _parse_binary(
            row.get("target_label"), field="target_label", row_index=row_index
        )
        teacher_pred = _parse_binary(
            row.get("teacher_pred"), field="teacher_pred", row_index=row_index
        )
        teacher_correct = _parse_bool(
            row.get("teacher_correct"), field="teacher_correct", row_index=row_index
        )
        if (label, source_label, target_label, teacher_pred, teacher_correct) != (
            SOURCE_LABEL,
            SOURCE_LABEL,
            TARGET_LABEL,
            SOURCE_LABEL,
            True,
        ):
            raise ValueError(
                "Mutagenicity PPO requires teacher-correct source parents with "
                f"label/source/target/pred/correct=1/1/0/1/true; row={row_index}"
            )

        records.append(
            MutagenicityPPORecord(
                row_index=row_index,
                molecule_id=molecule_id,
                parent_smiles=parent_smiles,
                canonical_parent_smiles=canonical,
                prompt=prompt,
                label=label,
                source_label=source_label,
                target_label=target_label,
                split=split,
                scaffold_smiles=scaffold,
                teacher_pred=teacher_pred,
                teacher_prob_0=_parse_probability(
                    row.get("teacher_prob_0"),
                    field="teacher_prob_0",
                    row_index=row_index,
                ),
                teacher_prob_1=_parse_probability(
                    row.get("teacher_prob_1"),
                    field="teacher_prob_1",
                    row_index=row_index,
                ),
                teacher_correct=teacher_correct,
            )
        )
        seen_ids.add(molecule_id)
        seen_smiles.add(canonical)
    return records


def validate_train_val_isolation(
    train_records: Sequence[MutagenicityPPORecord],
    val_records: Sequence[MutagenicityPPORecord],
) -> dict[str, Any]:
    train_ids = {record.molecule_id for record in train_records}
    val_ids = {record.molecule_id for record in val_records}
    train_smiles = {record.canonical_parent_smiles for record in train_records}
    val_smiles = {record.canonical_parent_smiles for record in val_records}
    train_scaffolds = {record.scaffold_smiles for record in train_records}
    val_scaffolds = {record.scaffold_smiles for record in val_records}
    overlaps = {
        "molecule_id_overlap": sorted(train_ids & val_ids),
        "canonical_smiles_overlap": sorted(train_smiles & val_smiles),
        "scaffold_overlap": sorted(train_scaffolds & val_scaffolds),
    }
    if any(overlaps.values()):
        raise ValueError(
            "Mutagenicity PPO train/val leakage detected: "
            f"id={len(overlaps['molecule_id_overlap'])} "
            f"smiles={len(overlaps['canonical_smiles_overlap'])} "
            f"scaffold={len(overlaps['scaffold_overlap'])}"
        )
    return {
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "train_val_molecule_id_overlap_count": 0,
        "train_val_canonical_smiles_overlap_count": 0,
        "train_val_scaffold_overlap_count": 0,
        "calibration_loaded": False,
        "test_loaded": False,
        "isolation_passed": True,
    }


def deterministically_order_records(
    records: Sequence[MutagenicityPPORecord],
    *,
    seed: int,
    limit: int = 0,
) -> list[MutagenicityPPORecord]:
    def key(record: MutagenicityPPORecord) -> tuple[str, str]:
        digest = hashlib.sha256(
            f"{int(seed)}:{record.molecule_id}".encode("utf-8")
        ).hexdigest()
        return digest, record.molecule_id

    ordered = sorted(records, key=key)
    return ordered[: int(limit)] if int(limit) > 0 else ordered


def build_parent_coverage_plan(
    *,
    num_dataset_rows: int,
    rollout_batch_size: int,
    sampler_seed: int,
    max_updates: int | None = None,
) -> ParentCoveragePlan:
    if num_dataset_rows <= 0 or rollout_batch_size <= 0:
        raise ValueError("Coverage planning requires positive dataset and batch sizes")
    samples_per_update = min(int(rollout_batch_size), int(num_dataset_rows))
    updates_per_epoch = int(math.ceil(num_dataset_rows / samples_per_update))
    resolved_max_updates = (
        updates_per_epoch if max_updates is None or int(max_updates) <= 0 else int(max_updates)
    )
    return ParentCoveragePlan(
        num_dataset_rows=int(num_dataset_rows),
        num_unique_dataset_parents=int(num_dataset_rows),
        rollout_batch_size=samples_per_update,
        samples_per_update=samples_per_update,
        updates_per_epoch=updates_per_epoch,
        max_updates=resolved_max_updates,
        planned_epochs_equivalent=min(
            1.0,
            resolved_max_updates * samples_per_update / num_dataset_rows,
        ),
        sampler_seed=int(sampler_seed),
        shuffle_enabled=True,
        sampling_with_replacement=False,
    )


def validate_policy_adapter_checkpoint(
    path: str | Path,
    *,
    expected_step: int = EXPECTED_POLICY_CHECKPOINT_STEP,
) -> dict[str, Any]:
    checkpoint = Path(path).expanduser().resolve()
    expected_name = f"checkpoint-{int(expected_step)}"
    if checkpoint.name != expected_name:
        raise ValueError(
            "Mutagenicity PPO policy must initialize from continued-SFT "
            f"{expected_name}; resolved={checkpoint}"
        )
    payload = validate_peft_checkpoint(checkpoint)
    payload["expected_checkpoint_step"] = int(expected_step)
    payload["checkpoint_step_verified"] = True
    return payload


def _active_adapters(model: Any) -> list[str]:
    active = getattr(model, "active_adapters", None)
    if callable(active):
        active = active()
    if active is None:
        active = getattr(model, "active_adapter", None)
        if callable(active):
            active = active()
    if active is None:
        return []
    if isinstance(active, str):
        return [active]
    return [str(value) for value in active]


def audit_mutagenicity_ppo_models(
    *,
    policy_model: Any,
    reference_model: Any,
    value_model: Any,
    base_model_path: str | Path,
    policy_adapter_checkpoint: str | Path,
) -> dict[str, Any]:
    policy = audit_single_trainable_lora_adapter(
        policy_model,
        base_model_name_or_path=base_model_path,
        source_adapter_checkpoint=policy_adapter_checkpoint,
    )

    reference_config = getattr(reference_model, "peft_config", None)
    if not isinstance(reference_config, Mapping) or len(reference_config) != 1:
        raise ValueError("Reference policy must contain exactly one PEFT adapter")
    reference_adapter_names = [str(value) for value in reference_config]
    reference_active = _active_adapters(reference_model)
    if reference_active != reference_adapter_names:
        raise ValueError(
            "Reference policy active adapter mismatch: "
            f"configured={reference_adapter_names} active={reference_active}"
        )
    reference_named = list(reference_model.named_parameters())
    reference_trainable = [
        name for name, parameter in reference_named if bool(parameter.requires_grad)
    ]
    if reference_trainable:
        raise ValueError(
            "Reference policy must be frozen; trainable examples="
            f"{reference_trainable[:10]}"
        )

    value_named = list(value_model.named_parameters())
    value_trainable = [
        (name, parameter)
        for name, parameter in value_named
        if bool(parameter.requires_grad)
    ]
    value_head_trainable = [
        (name, parameter)
        for name, parameter in value_trainable
        if "v_head" in name
    ]
    non_value_trainable = [
        name for name, _ in value_trainable if "v_head" not in name
    ]
    if not value_head_trainable:
        raise ValueError("PPO value head has no trainable parameters")
    if non_value_trainable:
        raise ValueError(
            "Value-model base parameters are unexpectedly trainable: "
            f"{non_value_trainable[:10]}"
        )

    audit = {
        "base_model_path": str(Path(base_model_path).expanduser().resolve()),
        "policy_adapter_checkpoint": str(
            Path(policy_adapter_checkpoint).expanduser().resolve()
        ),
        "policy_adapter_names": policy["adapter_names"],
        "active_adapters": policy["active_adapters"],
        "policy_trainable_params": policy["trainable_parameter_count"],
        "policy_total_params": policy["total_parameter_count"],
        "policy_trainable_percent": policy["trainable_percent"],
        "policy_trainable_parameter_name_examples": policy[
            "trainable_parameter_name_examples"
        ],
        "reference_adapter_names": reference_adapter_names,
        "reference_active_adapters": reference_active,
        "reference_trainable_params": 0,
        "base_params_trainable": policy["base_parameter_trainable_count"],
        "value_head_trainable_params": sum(
            int(parameter.numel()) for _, parameter in value_head_trainable
        ),
        "value_head_trainable_parameter_name_examples": [
            name for name, _ in value_head_trainable[:20]
        ],
        "model_audit_passed": True,
    }
    return audit


class MutagenicityCounterfactualTeacherScorer(CounterfactualTeacherScorer):
    """Apply the source=1, target=0 teacher semantics before reward aggregation."""

    source_label = SOURCE_LABEL
    target_label = TARGET_LABEL

    def score_counterfactual(
        self,
        parent_smiles: str,
        core_fragment_smiles: str,
        label: int,
        raw_fragment_smiles: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if int(label) != SOURCE_LABEL:
            raise ValueError(
                "Mutagenicity PPO counterfactual scorer accepts source label 1 only"
            )
        result = super().score_counterfactual(
            parent_smiles=parent_smiles,
            core_fragment_smiles=core_fragment_smiles,
            label=SOURCE_LABEL,
            raw_fragment_smiles=raw_fragment_smiles,
            meta=meta,
        )
        updated = dict(result)
        p_before_1 = updated.get("p_before")
        p_after_1 = updated.get("p_after")
        pred_before = updated.get("pred_before")
        pred_after = updated.get("pred_after")
        strict_flip = bool(
            updated.get("teacher_result_ok")
            and pred_before is not None
            and pred_after is not None
            and int(pred_before) == SOURCE_LABEL
            and int(pred_after) == TARGET_LABEL
        )
        updated["teacher_parent_ok"] = bool(
            pred_before is not None and int(pred_before) == SOURCE_LABEL
        )
        updated["cf_flip"] = strict_flip
        updated["strict_flip_definition"] = "pred_before==1_and_pred_after==0"
        updated["cf_drop_definition"] = "p1_before_minus_p1_after"
        if p_before_1 is not None:
            updated["prob_before_1"] = float(p_before_1)
            updated["prob_before_0"] = float(1.0 - float(p_before_1))
        else:
            updated["prob_before_1"] = None
            updated["prob_before_0"] = None
        if p_after_1 is not None:
            updated["prob_after_1"] = float(p_after_1)
            updated["prob_after_0"] = float(1.0 - float(p_after_1))
        else:
            updated["prob_after_1"] = None
            updated["prob_after_0"] = None
        if p_before_1 is not None and p_after_1 is not None:
            cf_drop = float(float(p_before_1) - float(p_after_1))
            updated["cf_drop"] = cf_drop
            updated["target_prob_gain"] = cf_drop
            counterfactual_sem = float(
                cf_drop + self.flip_bonus * float(strict_flip)
            )
            updated["counterfactual_sem"] = counterfactual_sem
            updated["teacher_sem"] = counterfactual_sem
        else:
            updated["target_prob_gain"] = None
        return updated


def enrich_mutagenicity_candidate_row(
    row: Mapping[str, Any],
    *,
    molecule_id: str | None,
    prompt: str | None,
    generated_text: str | None,
    generated_fragment: str | None,
    global_step: int,
    parent_smiles: str | None = None,
) -> dict[str, Any]:
    output = dict(row)
    p_before_1 = output.get("p_before")
    p_after_1 = output.get("p_after")
    pred_before = output.get("pred_before")
    pred_after = output.get("pred_after")
    teacher_ok = bool(
        output.get("oracle_ok")
        or output.get("counterfactual_teacher_called")
        or output.get("counterfactual_called")
    )
    strict_flip = bool(
        teacher_ok
        and pred_before is not None
        and pred_after is not None
        and int(pred_before) == SOURCE_LABEL
        and int(pred_after) == TARGET_LABEL
    )
    output.update(
        {
            "molecule_id": str(molecule_id or output.get("molecule_id") or ""),
            "parent_smiles": str(
                parent_smiles
                if parent_smiles is not None
                else output.get("parent_smiles", "")
            ),
            "prompt": str(prompt or output.get("prompt") or ""),
            "generated_text": str(
                generated_text
                if generated_text is not None
                else output.get("raw_response", output.get("raw_output", ""))
            ),
            "raw_fragment": output.get("raw_fragment") or generated_fragment,
            "core_fragment": output.get("core_fragment"),
            "final_fragment": (
                output.get("final_fragment")
                or output.get("projected_fragment")
                or output.get("core_fragment")
            ),
            "residual_smiles": output.get("parent_without_fragment_smiles"),
            "parse_ok": bool(
                output.get("parse_ok")
                or output.get("core_parse_ok")
                or output.get("raw_parse_ok")
            ),
            "connected": bool(
                output.get("connected") or output.get("connected_fragment")
            ),
            "direct_substructure": bool(
                output.get("direct_substructure")
                or output.get("direct_substructure_success")
            ),
            "final_substructure": bool(
                output.get("final_substructure")
                or output.get("is_subgraph")
            ),
            "projection_used": bool(
                output.get("projection_used")
                or output.get("used_projected_subgraph_for_reward")
            ),
            "projection_failed": bool(
                output.get("projection_attempted")
                and not output.get("projection_success")
            ),
            "oracle_ok": bool(output.get("oracle_ok")),
            "pred_before": pred_before,
            "pred_after": pred_after,
            "cf_drop": (
                float(float(p_before_1) - float(p_after_1))
                if p_before_1 is not None and p_after_1 is not None
                else output.get("cf_drop")
            ),
            "teacher_parent_ok": bool(
                pred_before is not None and int(pred_before) == SOURCE_LABEL
            ),
            "cf_flip": strict_flip,
            "strict_flip_definition": "pred_before==1_and_pred_after==0",
            "cf_drop_definition": "p1_before_minus_p1_after",
            "prob_before_1": (
                float(p_before_1) if p_before_1 is not None else None
            ),
            "prob_before_0": (
                float(1.0 - float(p_before_1))
                if p_before_1 is not None
                else None
            ),
            "prob_after_1": float(p_after_1) if p_after_1 is not None else None,
            "prob_after_0": (
                float(1.0 - float(p_after_1))
                if p_after_1 is not None
                else None
            ),
            "target_prob_gain": (
                float(float(p_before_1) - float(p_after_1))
                if p_before_1 is not None and p_after_1 is not None
                else None
            ),
            "atom_ratio": output.get(
                "atom_ratio", output.get("final_fragment_atom_ratio")
            ),
            "reward_total": output.get("reward_total", output.get("total")),
            "reward_components": output.get(
                "reward_components", output.get("breakdown", {})
            ),
            "global_step": int(global_step),
            "source_label": SOURCE_LABEL,
            "target_label": TARGET_LABEL,
        }
    )
    return output


class MutagenicityPPORunObserver:
    """Persist coverage and validation artifacts around the shared PPO algorithm."""

    def __init__(
        self,
        *,
        output_root: str | Path,
        dataset_parent_ids: Sequence[str],
        coverage_plan: ParentCoveragePlan,
        resolved_config: Mapping[str, Any],
        dataset_manifest: Mapping[str, Any],
        require_full_coverage: bool,
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.dataset_parent_ids = frozenset(str(value) for value in dataset_parent_ids)
        self.coverage_plan = coverage_plan
        self.resolved_config = dict(resolved_config)
        self.dataset_manifest = dict(dataset_manifest)
        self.require_full_coverage = bool(require_full_coverage)
        self.samples_processed = 0
        self.parents_seen: set[str] = set()
        self.update_rows: list[dict[str, Any]] = []
        self.validation_rows: list[dict[str, Any]] = []
        self.validation_metrics: list[dict[str, Any]] = []
        self.checkpoints: list[dict[str, Any]] = []
        self.best_checkpoint: str | None = None

    def enrich_reward_logs(
        self,
        *,
        step_index: int,
        batch_ids: Sequence[str],
        parent_smiles: Sequence[str],
        prompts: Sequence[str],
        generated_texts: Sequence[str],
        fragments: Sequence[str],
        reward_logs: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if not (
            len(batch_ids)
            == len(prompts)
            == len(generated_texts)
            == len(fragments)
            == len(reward_logs)
        ):
            raise ValueError("Mutagenicity PPO observer received misaligned batch metadata")
        return [
            enrich_mutagenicity_candidate_row(
                row,
                molecule_id=str(batch_ids[index]),
                parent_smiles=str(parent_smiles[index]),
                prompt=str(prompts[index]),
                generated_text=str(generated_texts[index]),
                generated_fragment=str(fragments[index]),
                global_step=step_index,
            )
            for index, row in enumerate(reward_logs)
        ]

    def on_update(
        self,
        *,
        step_index: int,
        batch_ids: Sequence[str],
        reward_logs: Sequence[Mapping[str, Any]],
        metrics: Mapping[str, Any],
    ) -> None:
        del reward_logs
        normalized = [str(value) for value in batch_ids]
        unknown = sorted(set(normalized) - self.dataset_parent_ids)
        if unknown:
            raise ValueError(f"PPO observer received unknown molecule IDs: {unknown[:10]}")
        repeated = sorted(set(normalized) & self.parents_seen)
        if repeated:
            raise ValueError(
                "Mutagenicity first-epoch PPO sampling must be without replacement; "
                f"repeated={repeated[:10]}"
            )
        self.samples_processed += len(normalized)
        self.parents_seen.update(normalized)
        row = dict(metrics)
        row.update(
            {
                "global_step": int(step_index),
                "num_samples_processed": self.samples_processed,
                "num_unique_parents_seen": len(self.parents_seen),
                "unique_parent_coverage": (
                    len(self.parents_seen) / len(self.dataset_parent_ids)
                    if self.dataset_parent_ids
                    else 0.0
                ),
                "epochs_equivalent": (
                    self.samples_processed / len(self.dataset_parent_ids)
                    if self.dataset_parent_ids
                    else 0.0
                ),
            }
        )
        self.update_rows.append(row)
        _write_jsonl_atomic(self.output_root / "ppo_metrics.jsonl", self.update_rows)
        write_json_atomic(
            self.output_root / "parent_coverage.json",
            self.coverage_summary(global_step=step_index),
        )

    def on_validation(
        self,
        *,
        step_index: int,
        summary: Mapping[str, Any],
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        metric_row = {"global_step": int(step_index), **dict(summary)}
        self.validation_metrics.append(metric_row)
        _write_jsonl_atomic(
            self.output_root / "validation_metrics.jsonl",
            self.validation_metrics,
        )
        for row in rows:
            self.validation_rows.append(
                enrich_mutagenicity_candidate_row(
                    row,
                    molecule_id=str(row.get("molecule_id") or ""),
                    parent_smiles=str(row.get("parent_smiles") or ""),
                    prompt=str(row.get("prompt") or ""),
                    generated_text=str(
                        row.get("raw_response", row.get("raw_output", ""))
                    ),
                    generated_fragment=str(row.get("raw_fragment") or ""),
                    global_step=step_index,
                )
            )
        _write_dynamic_csv_atomic(
            self.output_root / "validation_samples.csv",
            self.validation_rows,
        )

    def on_checkpoint(
        self,
        *,
        step_index: int,
        checkpoint_dir: str | Path,
        checkpoint_kind: str,
    ) -> None:
        path = str(Path(checkpoint_dir).expanduser().resolve())
        row = {
            "global_step": int(step_index),
            "checkpoint_path": path,
            "checkpoint_kind": str(checkpoint_kind),
        }
        self.checkpoints.append(row)
        if checkpoint_kind == "best_validation":
            self.best_checkpoint = path

    def coverage_summary(self, *, global_step: int) -> dict[str, Any]:
        total = len(self.dataset_parent_ids)
        return {
            **self.coverage_plan.to_dict(),
            "num_samples_processed": int(self.samples_processed),
            "num_unique_parents_seen": len(self.parents_seen),
            "unique_parent_coverage": (
                len(self.parents_seen) / total if total else 0.0
            ),
            "epochs_equivalent": self.samples_processed / total if total else 0.0,
            "global_step": int(global_step),
        }

    def on_finish(
        self,
        *,
        final_output_dir: str | Path,
        candidate_pool_path: str | Path,
        candidate_count: int,
        global_step: int,
        validation_state: Any,
        early_stop_reason: str | None,
    ) -> None:
        coverage = self.coverage_summary(global_step=global_step)
        write_json_atomic(self.output_root / "parent_coverage.json", coverage)
        if self.require_full_coverage and not math.isclose(
            float(coverage["unique_parent_coverage"]),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "Mutagenicity PPO ended before one no-replacement pass: "
                f"coverage={coverage['unique_parent_coverage']}"
            )
        checkpoint_manifest = {
            "initial_policy_adapter": self.resolved_config[
                "policy_adapter_checkpoint"
            ],
            "final_checkpoint": str(Path(final_output_dir).expanduser().resolve()),
            "best_checkpoint": self.best_checkpoint,
            "checkpoints": self.checkpoints,
            "num_checkpoints": len(self.checkpoints),
        }
        write_json_atomic(
            self.output_root / "checkpoint_manifest.json",
            checkpoint_manifest,
        )
        write_json_atomic(
            self.output_root / "best_checkpoint.json",
            {
                "best_checkpoint": self.best_checkpoint,
                "best_step": getattr(validation_state, "best_step", None),
                "best_val_score": getattr(validation_state, "best_val_score", None),
                "selection_data": "validation_only",
            },
        )
        report = _training_report(
            resolved=self.resolved_config,
            manifest=self.dataset_manifest,
            coverage=coverage,
            candidate_pool_path=candidate_pool_path,
            candidate_count=candidate_count,
            early_stop_reason=early_stop_reason,
            checkpoint_manifest=checkpoint_manifest,
        )
        (self.output_root / "training_report.md").write_text(
            report, encoding="utf-8"
        )
        write_json_atomic(
            self.output_root / "_RUN_COMPLETE.json",
            {
                "completed": True,
                "global_step": int(global_step),
                "candidate_pool_path": str(
                    Path(candidate_pool_path).expanduser().resolve()
                ),
                "candidate_count": int(candidate_count),
                "unique_parent_coverage": coverage["unique_parent_coverage"],
                "early_stop_reason": early_stop_reason,
            },
        )


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "\n".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str)
        for row in rows
    )
    if payload:
        payload += "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _write_dynamic_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    write_csv_atomic(path, rows, fieldnames)


def _training_report(
    *,
    resolved: Mapping[str, Any],
    manifest: Mapping[str, Any],
    coverage: Mapping[str, Any],
    candidate_pool_path: str | Path,
    candidate_count: int,
    early_stop_reason: str | None,
    checkpoint_manifest: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# Mutagenicity Stable PPO Report",
            "",
            "## Initialization",
            "",
            f"- Base model: `{resolved['base_model_path']}`",
            f"- Policy/reference adapter: `{resolved['policy_adapter_checkpoint']}`",
            f"- RF teacher: `{resolved['teacher_path']}`",
            "- Reference policy starts from the same adapter and is frozen.",
            "",
            "## Counterfactual Direction",
            "",
            "- Source: label 1 (mutagenic)",
            "- Target: label 0 (non-mutagenic)",
            "- strict flip: `pred_before == 1 and pred_after == 0`",
            "- cf_drop: `p1_before - p1_after`",
            "",
            "## Data",
            "",
            f"- Train rows: {manifest['num_train_rows_selected']}",
            f"- Validation rows: {manifest['num_val_rows_selected']}",
            "- Calibration/test loaded: false",
            "",
            "## Sampling and Coverage",
            "",
            f"- Samples per update: {coverage['samples_per_update']}",
            f"- Updates per epoch: {coverage['updates_per_epoch']}",
            f"- Global step: {coverage['global_step']}",
            f"- Samples processed: {coverage['num_samples_processed']}",
            f"- Unique parents seen: {coverage['num_unique_parents_seen']}",
            f"- Unique parent coverage: {coverage['unique_parent_coverage']:.8f}",
            f"- Equivalent epochs: {coverage['epochs_equivalent']:.8f}",
            "- Deterministic shuffle: true",
            "- Sampling with replacement: false",
            "",
            "## Artifacts",
            "",
            f"- Candidate pool: `{Path(candidate_pool_path).expanduser().resolve()}` ({candidate_count} rows)",
            f"- Best checkpoint: `{checkpoint_manifest.get('best_checkpoint')}`",
            f"- Early stop reason: `{early_stop_reason}`",
            "",
        ]
    )


def validate_candidate_pool_schema(rows: Sequence[Mapping[str, Any]]) -> None:
    for index, row in enumerate(rows):
        missing = sorted(set(REQUIRED_CANDIDATE_FIELDS) - set(row))
        if missing:
            raise ValueError(
                f"candidate_pool row={index} is missing required fields: {missing}"
            )


__all__ = [
    "EXPECTED_POLICY_CHECKPOINT_STEP",
    "EXPECTED_TRAIN_ROWS",
    "EXPECTED_VAL_ROWS",
    "MutagenicityCounterfactualTeacherScorer",
    "MutagenicityPPORecord",
    "MutagenicityPPORunObserver",
    "ParentCoveragePlan",
    "REQUIRED_CANDIDATE_FIELDS",
    "SOURCE_LABEL",
    "TARGET_LABEL",
    "audit_mutagenicity_ppo_models",
    "build_parent_coverage_plan",
    "deterministically_order_records",
    "enrich_mutagenicity_candidate_row",
    "load_mutagenicity_ppo_records",
    "validate_candidate_pool_schema",
    "validate_policy_adapter_checkpoint",
    "validate_train_val_isolation",
]
