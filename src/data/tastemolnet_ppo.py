"""Train-only TasteMolNet prompt loading from already-retained CSV bytes."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
from typing import Any, Mapping

from src.data.prompts import build_counterfactual_prompt
from src.data.schemas import MoleculeRecord


TASTEMOLNET_PREPARED_FIELDS = (
    "molecule_id",
    "raw_smiles",
    "canonical_smiles",
    "model_smiles",
    "label",
    "label_name",
    "scaffold",
    "num_atoms",
    "num_bonds",
    "formal_charge",
    "fragment_count",
    "supported_atom_check",
    "molclr_compatible",
    "chemlm_compatible",
    "has_stereochemistry",
    "source",
    "source_row_id",
    "standardization_version",
    "component_policy",
    "split",
    "exclusion_reason",
)
LABEL_MAP = {0: "Bitter", 1: "Sweet", 2: "Tasteless"}


@dataclass(frozen=True, slots=True)
class TastePPOPromptExample:
    index: int
    prompt: str
    parent_smiles: str
    original_label: int
    molecule_id: str

    def to_dataset_row(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "query": self.prompt,
            "parent_smiles": self.parent_smiles,
            "original_label": self.original_label,
            "molecule_id": self.molecule_id,
        }


def load_tastemolnet_train_prompts(
    data: bytes,
    *,
    expected_num_records: int,
    expected_label_counts: Mapping[str, Any],
    max_prompt_examples: int,
) -> tuple[list[TastePPOPromptExample], dict[str, Any]]:
    """Parse the full frozen train CSV, returning only bounded Sweet prompts.

    The caller supplies bytes read through a retained descriptor.  This
    function intentionally has no path argument and therefore cannot reopen a
    renamed or substituted dataset path.
    """

    if type(data) is not bytes or not data:
        raise ValueError("Taste PPO train CSV bytes must be nonempty")
    if type(expected_num_records) is not int or expected_num_records <= 0:
        raise ValueError("Taste PPO expected train record count is invalid")
    if (
        type(expected_label_counts) is not dict
        or set(expected_label_counts) != {"0", "1", "2"}
        or any(
            type(expected_label_counts[label]) is not int
            or expected_label_counts[label] <= 0
            for label in ("0", "1", "2")
        )
        or sum(expected_label_counts.values()) != expected_num_records
    ):
        raise ValueError("Taste PPO expected train label counts are invalid")
    if (
        type(max_prompt_examples) is not int
        or max_prompt_examples <= 0
        or max_prompt_examples > expected_num_records
    ):
        raise ValueError("Taste PPO prompt bound is invalid")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Taste PPO train CSV is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    if tuple(reader.fieldnames or ()) != TASTEMOLNET_PREPARED_FIELDS:
        raise ValueError("Taste PPO train CSV columns differ from prepared schema")
    prompts: list[TastePPOPromptExample] = []
    observed_counts = {"0": 0, "1": 0, "2": 0}
    observed_ids: set[str] = set()
    row_count = 0
    for row_index, row in enumerate(reader):
        row_count += 1
        if None in row or set(row) != set(TASTEMOLNET_PREPARED_FIELDS):
            raise ValueError("Taste PPO train CSV row width changed")
        molecule_id = str(row.get("molecule_id") or "").strip()
        model_smiles = str(row.get("model_smiles") or "").strip()
        label_text = str(row.get("label") or "").strip()
        if (
            not molecule_id
            or molecule_id in observed_ids
            or not model_smiles
            or label_text not in observed_counts
            or str(row.get("label_name") or "").strip()
            != LABEL_MAP[int(label_text)]
            or str(row.get("split") or "").strip() != "train"
            or str(row.get("exclusion_reason") or "").strip()
        ):
            raise ValueError("Taste PPO train CSV row authority changed")
        observed_ids.add(molecule_id)
        observed_counts[label_text] += 1
        if label_text == "1" and len(prompts) < max_prompt_examples:
            prompts.append(
                TastePPOPromptExample(
                    index=row_index,
                    prompt=build_counterfactual_prompt(
                        MoleculeRecord(
                            record_id=row_index,
                            smiles=model_smiles,
                            label=1,
                        ),
                        include_label=True,
                    ),
                    parent_smiles=model_smiles,
                    original_label=1,
                    molecule_id=molecule_id,
                )
            )
    if row_count != expected_num_records or observed_counts != dict(
        expected_label_counts
    ):
        raise ValueError("Taste PPO train CSV count authority changed")
    if len(prompts) != max_prompt_examples:
        raise ValueError("Taste PPO train split lacks the requested Sweet prompt pool")
    return prompts, {
        "schema_version": "tastemolnet_train_ppo_prompt_pool_v1",
        "total_train_records": row_count,
        "label_counts": observed_counts,
        "source_label": 1,
        "source_prompt_count": len(prompts),
        "max_prompt_examples": max_prompt_examples,
        "train_loaded": True,
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }


__all__ = [
    "LABEL_MAP",
    "TASTEMOLNET_PREPARED_FIELDS",
    "TastePPOPromptExample",
    "load_tastemolnet_train_prompts",
]
