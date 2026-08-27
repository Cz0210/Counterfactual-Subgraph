from __future__ import annotations

import csv
import io

import pytest

from src.data.tastemolnet_ppo import (
    TASTEMOLNET_PREPARED_FIELDS,
    load_tastemolnet_train_prompts,
)


def _csv_bytes(*, bad_split: bool = False) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=TASTEMOLNET_PREPARED_FIELDS)
    writer.writeheader()
    for index, (label, name) in enumerate(
        ((0, "Bitter"), (1, "Sweet"), (1, "Sweet"), (2, "Tasteless"))
    ):
        row = {field: "value" for field in TASTEMOLNET_PREPARED_FIELDS}
        row.update(
            {
                "molecule_id": f"taste-{index}",
                "model_smiles": "CCO" if index % 2 else "CCC",
                "label": str(label),
                "label_name": name,
                "split": "validation" if bad_split and index == 0 else "train",
                "exclusion_reason": "",
            }
        )
        writer.writerow(row)
    return handle.getvalue().encode("utf-8")


def test_train_prompt_loader_uses_model_smiles_and_only_sweet_rows() -> None:
    from scripts.prepare_tastemolnet import CLEAN_FIELDS

    assert tuple(CLEAN_FIELDS) == TASTEMOLNET_PREPARED_FIELDS
    prompts, evidence = load_tastemolnet_train_prompts(
        _csv_bytes(),
        expected_num_records=4,
        expected_label_counts={"0": 1, "1": 2, "2": 1},
        max_prompt_examples=2,
    )
    assert [row.molecule_id for row in prompts] == ["taste-1", "taste-2"]
    assert all(row.original_label == 1 for row in prompts)
    assert all("ORIGINAL_LABEL: 1" in row.prompt for row in prompts)
    assert evidence["label_counts"] == {"0": 1, "1": 2, "2": 1}
    assert evidence["validation_loaded"] is False
    assert evidence["calibration_loaded"] is False
    assert evidence["test_loaded"] is False


def test_train_prompt_loader_rejects_split_and_count_drift() -> None:
    with pytest.raises(ValueError, match="row authority changed"):
        load_tastemolnet_train_prompts(
            _csv_bytes(bad_split=True),
            expected_num_records=4,
            expected_label_counts={"0": 1, "1": 2, "2": 1},
            max_prompt_examples=2,
        )
    with pytest.raises(ValueError, match="count authority changed"):
        load_tastemolnet_train_prompts(
            _csv_bytes(),
            expected_num_records=5,
            expected_label_counts={"0": 1, "1": 3, "2": 1},
            max_prompt_examples=2,
        )


@pytest.mark.parametrize(
    ("records", "counts", "limit"),
    (
        (4.0, {"0": 1, "1": 2, "2": 1}, 2),
        (4, {"0": True, "1": 2, "2": 1}, 2),
        (4, {"0": 1, "1": 2, "2": 1}, 2.0),
    ),
)
def test_train_prompt_loader_rejects_non_native_contract_types(
    records: object,
    counts: object,
    limit: object,
) -> None:
    with pytest.raises(ValueError):
        load_tastemolnet_train_prompts(
            _csv_bytes(),
            expected_num_records=records,  # type: ignore[arg-type]
            expected_label_counts=counts,  # type: ignore[arg-type]
            max_prompt_examples=limit,  # type: ignore[arg-type]
        )
