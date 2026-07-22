from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402

from src.data.mutagenicity_sft_ppo import (  # noqa: E402
    MutagenicityParent,
    MutagenicitySFTPPOConfig,
    build_mutagenicity_sft_ppo_data,
    deterministic_parent_sample,
    load_teacher_consistent_parents,
    validate_source_isolation,
)


FIELDNAMES = [
    "molecule_id",
    "smiles",
    "label",
    "source_label",
    "target_label",
    "semantic_label",
    "split",
    "scaffold_smiles",
    "teacher_pred",
    "teacher_prob_0",
    "teacher_prob_1",
    "teacher_correct",
]


def _scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=True)


def _row(molecule_id: str, smiles: str, split: str, **updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "molecule_id": molecule_id,
        "smiles": smiles,
        "label": 1,
        "source_label": 1,
        "target_label": 0,
        "semantic_label": "mutagenic",
        "split": split,
        "scaffold_smiles": _scaffold(smiles),
        "teacher_pred": 1,
        "teacher_prob_0": 0.1,
        "teacher_prob_1": 0.9,
        "teacher_correct": True,
    }
    row.update(updates)
    return row


def _write_rows(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _valid_four_split_inputs(root: Path) -> dict[str, Path]:
    return {
        "train": _write_rows(
            root / "train.csv",
            [_row("MUT_TRAIN", "CC(=O)Oc1ccccc1C(=O)O", "train")],
        ),
        "val": _write_rows(
            root / "val.csv",
            [_row("MUT_VAL", "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "val")],
        ),
        "calibration": _write_rows(
            root / "calibration.csv", [_row("MUT_CAL", "CCN", "calibration")]
        ),
        "test": _write_rows(
            root / "test.csv", [_row("MUT_TEST", "C1CCNCC1", "test")]
        ),
    }


def test_source_label_and_teacher_filter_accepts_only_teacher_correct_label1(tmp_path: Path) -> None:
    source = _write_rows(tmp_path / "train.csv", [_row("MUT_OK", "CCO", "train")])
    rows = load_teacher_consistent_parents(source, expected_split="train", expected_count=1)
    assert len(rows) == 1
    assert rows[0].label == rows[0].teacher_pred == 1
    assert rows[0].teacher_correct is True


@pytest.mark.parametrize(
    "updates",
    [
        {"label": 0},
        {"teacher_pred": 0},
        {"teacher_correct": False},
        {"source_label": 0},
        {"target_label": 1},
    ],
)
def test_source_label_or_teacher_contract_violation_is_rejected(
    tmp_path: Path, updates: dict[str, object]
) -> None:
    source = _write_rows(
        tmp_path / "train.csv", [_row("MUT_BAD", "CCO", "train", **updates)]
    )
    with pytest.raises(ValueError):
        load_teacher_consistent_parents(source, expected_split="train", expected_count=1)


def test_train_val_have_no_molecule_or_scaffold_overlap(tmp_path: Path) -> None:
    paths = _valid_four_split_inputs(tmp_path)
    loaded = {
        split: load_teacher_consistent_parents(path, expected_split=split, expected_count=1)
        for split, path in paths.items()
    }
    audit = validate_source_isolation(
        loaded["train"], loaded["val"], loaded["calibration"], loaded["test"]
    )
    assert audit["leakage_free"] is True
    assert audit["pair_audits"]["train_vs_val"]["molecule_id_overlap"] == []
    assert audit["pair_audits"]["train_vs_val"]["scaffold_overlap"] == []


def test_train_val_scaffold_overlap_is_rejected(tmp_path: Path) -> None:
    train_path = _write_rows(
        tmp_path / "train.csv", [_row("MUT_TRAIN", "Cc1ccccc1", "train")]
    )
    val_path = _write_rows(
        tmp_path / "val.csv", [_row("MUT_VAL", "Oc1ccccc1", "val")]
    )
    calibration_path = _write_rows(
        tmp_path / "cal.csv", [_row("MUT_CAL", "CCN", "calibration")]
    )
    test_path = _write_rows(
        tmp_path / "test.csv", [_row("MUT_TEST", "C1CCNCC1", "test")]
    )
    with pytest.raises(ValueError, match="scaffold_overlap"):
        validate_source_isolation(
            load_teacher_consistent_parents(train_path, expected_split="train", expected_count=1),
            load_teacher_consistent_parents(val_path, expected_split="val", expected_count=1),
            load_teacher_consistent_parents(
                calibration_path, expected_split="calibration", expected_count=1
            ),
            load_teacher_consistent_parents(test_path, expected_split="test", expected_count=1),
        )


def test_calibration_or_test_leakage_is_detected(tmp_path: Path) -> None:
    shared = "CC(=O)Oc1ccccc1C(=O)O"
    train_path = _write_rows(
        tmp_path / "train.csv", [_row("MUT_SHARED", shared, "train")]
    )
    val_path = _write_rows(
        tmp_path / "val.csv", [_row("MUT_VAL", "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "val")]
    )
    calibration_path = _write_rows(
        tmp_path / "cal.csv", [_row("MUT_SHARED", shared, "calibration")]
    )
    test_path = _write_rows(
        tmp_path / "test.csv", [_row("MUT_TEST", "C1CCNCC1", "test")]
    )
    with pytest.raises(ValueError, match="train_vs_calibration"):
        validate_source_isolation(
            load_teacher_consistent_parents(train_path, expected_split="train", expected_count=1),
            load_teacher_consistent_parents(val_path, expected_split="val", expected_count=1),
            load_teacher_consistent_parents(
                calibration_path, expected_split="calibration", expected_count=1
            ),
            load_teacher_consistent_parents(test_path, expected_split="test", expected_count=1),
        )


def test_invalid_smiles_is_rejected(tmp_path: Path) -> None:
    invalid_row = _row("MUT_BAD", "CCO", "train")
    invalid_row["smiles"] = "not-a-smiles"
    invalid_row["scaffold_smiles"] = ""
    source = _write_rows(
        tmp_path / "train.csv",
        [invalid_row],
    )
    with pytest.raises(ValueError, match="invalid SMILES"):
        load_teacher_consistent_parents(source, expected_split="train", expected_count=1)


def test_duplicate_parent_id_is_rejected_before_ppo_output(tmp_path: Path) -> None:
    source = _write_rows(
        tmp_path / "train.csv",
        [
            _row("MUT_DUP", "CCO", "train"),
            _row("MUT_DUP", "CCN", "train"),
        ],
    )
    with pytest.raises(ValueError, match="duplicate molecule_id"):
        load_teacher_consistent_parents(source, expected_split="train", expected_count=2)


def test_stable_molecule_id_is_preserved(tmp_path: Path) -> None:
    source = _write_rows(
        tmp_path / "train.csv", [_row("MUT_STABLE_ID", "CCO", "train")]
    )
    loaded = load_teacher_consistent_parents(source, expected_split="train", expected_count=1)
    assert loaded[0].molecule_id == "MUT_STABLE_ID"


def test_smoke_scaffold_size_sampling_is_deterministic_and_not_head() -> None:
    parents = [
        MutagenicityParent(
            molecule_id=f"MUT_{index}",
            source_row_index=index,
            source_smiles="C" * (index + 2),
            parent_smiles="C" * (index + 2),
            label=1,
            semantic_label="mutagenic",
            split="train",
            scaffold_smiles=f"scaffold_{index % 3}",
            teacher_pred=1,
            teacher_prob_0=0.1,
            teacher_prob_1=0.9,
            teacher_correct=True,
            parent_atom_count=index + 2,
        )
        for index in range(12)
    ]
    first = deterministic_parent_sample(parents, max_parents=5, seed=42)
    second = deterministic_parent_sample(parents, max_parents=5, seed=42)
    assert [row.molecule_id for row in first] == [row.molecule_id for row in second]
    assert [row.molecule_id for row in first] != [row.molecule_id for row in parents[:5]]


def test_end_to_end_outputs_unique_ppo_parents_targets_and_correct_summary(tmp_path: Path) -> None:
    paths = _valid_four_split_inputs(tmp_path / "inputs")
    output_dir = tmp_path / "outputs"
    summary = build_mutagenicity_sft_ppo_data(
        train_input=paths["train"],
        val_input=paths["val"],
        calibration_exclusion_input=paths["calibration"],
        test_exclusion_input=paths["test"],
        teacher_path=None,
        output_dir=output_dir,
        config=MutagenicitySFTPPOConfig(use_teacher_ranking=False),
        expected_counts={"train": 1, "val": 1, "calibration": 1, "test": 1},
    )

    train_ppo = _read_csv(output_dir / "mutagenicity_ppo_prompts_train_label1.csv")
    train_sft = _read_csv(output_dir / "mutagenicity_sft_train.csv")
    assert len(train_ppo) == len({row["molecule_id"] for row in train_ppo}) == 1
    assert train_ppo[0]["molecule_id"] == "MUT_TRAIN"
    assert "ORIGINAL_LABEL: 1" in train_ppo[0]["prompt"]
    assert train_sft[0]["completion"].strip() == train_sft[0]["core_fragment"]
    assert "*" not in train_sft[0]["core_fragment"]
    assert train_sft[0]["raw_fragment"]
    assert summary["splits"]["train"]["num_source_parents"] == 1
    assert summary["splits"]["train"]["num_ppo_rows"] == 1
    assert summary["splits"]["train"]["num_sft_rows"] == 1
    assert summary["leakage_audit_passed"] is True
    leakage = json.loads((output_dir / "leakage_audit.json").read_text(encoding="utf-8"))
    assert leakage["calibration_and_test_usage"] == "exclusion_audit_only"
