from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.baselines.globalgce_mutagenicity_adapter import (
    NativeGenerationResult,
    PoolBuildConfig,
    attach_globalgce_generation_dataset,
    audit_mutagenicity_train_pool,
    build_mutagenicity_train_pool,
    load_strict_train_parents,
    stable_candidate_id,
    validate_globalgce_generation_loader,
)


def _write_train(path: Path) -> None:
    rows = [
        {
            "molecule_id": "p3",
            "smiles": "CCCl",
            "label": 1,
            "split": "train",
            "teacher_pred": 1,
            "teacher_correct": "true",
        },
        {
            "molecule_id": "p1",
            "smiles": "CCO",
            "label": 1,
            "split": "train",
            "teacher_pred": 1,
            "teacher_correct": "true",
        },
        {
            "molecule_id": "p2",
            "smiles": "CCN",
            "label": 1,
            "split": "train",
            "teacher_pred": 1,
            "teacher_correct": "true",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_official_root(root: Path) -> None:
    for relative in ("main.py", "models/GlobalGCE.py", "data/data_preprocess.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fake official layout\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class _FakeTeacher:
    def score_smiles(self, smiles: str, label: int | None = None, **_kwargs):
        del label
        prediction = 1 if smiles == "CO" else 0
        return {
            "teacher_result_ok": True,
            "teacher_label": prediction,
        }


class _FakeGenerator:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def generate(self, parents, **_kwargs):
        if self.fail:
            raise RuntimeError("intentional generation interruption")
        assert [parent.parent_id for parent in parents] == ["p1", "p2", "p3"]
        records = [
            {
                "source_parent_id": "p1",
                "source_parent_smiles": "CCO",
                "source_split": "train",
                "raw_smiles": "CC",
                "generator_rank": 1,
                "generator_score": 3.0,
                "native_rule_id": "r1",
                "native_run_id": "fake",
            },
            {
                "source_parent_id": "p1",
                "source_parent_smiles": "CCO",
                "source_split": "train",
                "raw_smiles": "C-C",
                "generator_rank": 2,
                "generator_score": 2.0,
                "native_rule_id": "r2",
                "native_run_id": "fake",
            },
            {
                "source_parent_id": "p1",
                "source_parent_smiles": "CCO",
                "source_split": "train",
                "raw_smiles": "not_a_smiles",
                "generator_rank": 3,
                "native_rule_id": "r3",
                "native_run_id": "fake",
            },
            {
                "source_parent_id": "p1",
                "source_parent_smiles": "CCO",
                "source_split": "train",
                "raw_smiles": "CO",
                "generator_rank": 4,
                "native_rule_id": "r4",
                "native_run_id": "fake",
            },
            {
                "source_parent_id": "p2",
                "source_parent_smiles": "CCN",
                "source_split": "train",
                "raw_smiles": "CC",
                "generator_rank": 1,
                "native_rule_id": "r1",
                "native_run_id": "fake",
            },
            {
                "source_parent_id": "p2",
                "source_parent_smiles": "CCN",
                "source_split": "train",
                "raw_smiles": "CCC",
                "generator_rank": 2,
                "native_rule_id": "r5",
                "native_run_id": "fake",
            },
        ]
        return NativeGenerationResult(
            records,
            {
                "internal_train_ids_hash": "train-hash",
                "internal_val_ids_hash": "val-hash",
                "native_gnn_required": True,
            },
        )


class _FakeNativeDataset:
    def __init__(self, max_num_nodes: int = 17) -> None:
        self.max_num_nodes = max_num_nodes


class _FakeAugmentedDataset:
    def __init__(self) -> None:
        self.rows = [
            {"parent_id": "p2", "value": 2},
            {"parent_id": "p1", "value": 1},
            {"parent_id": "p3", "value": 3},
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return dict(self.rows[index])


@pytest.fixture()
def fixture_paths(tmp_path: Path) -> dict[str, Path]:
    train = tmp_path / "train.csv"
    teacher = tmp_path / "teacher.pkl"
    official = tmp_path / "official"
    output = tmp_path / "output"
    _write_train(train)
    teacher.write_bytes(b"fake-teacher")
    _write_official_root(official)
    return {
        "train": train,
        "teacher": teacher,
        "official": official,
        "output": output,
    }


def _build(paths: dict[str, Path], **config_overrides):
    defaults = {
        "expected_parent_count": 3,
        "device": "cpu",
        "epochs": 1,
        "top_k_native": 2,
    }
    defaults.update(config_overrides)
    return build_mutagenicity_train_pool(
        train_csv=paths["train"],
        teacher_path=paths["teacher"],
        official_root=paths["official"],
        output_dir=paths["output"],
        teacher=_FakeTeacher(),
        generator=_FakeGenerator(),
        config=PoolBuildConfig(**defaults),
    )


def test_train_only_parent_loading_and_deterministic_parent_limit(
    fixture_paths: dict[str, Path],
) -> None:
    all_parents, selected = load_strict_train_parents(
        fixture_paths["train"],
        parent_limit=2,
        expected_parent_count=3,
    )
    assert [parent.parent_id for parent in all_parents] == ["p1", "p2", "p3"]
    assert [parent.parent_id for parent in selected] == ["p1", "p2"]


def test_generation_loader_retains_real_native_dataset_contract() -> None:
    torch = pytest.importorskip("torch")
    native = _FakeNativeDataset(max_num_nodes=17)
    augmented = _FakeAugmentedDataset()
    rows_before = [augmented[index] for index in range(len(augmented))]
    length_before = len(augmented)

    returned = attach_globalgce_generation_dataset(augmented, native)
    loader = torch.utils.data.DataLoader(
        augmented,
        batch_size=2,
        shuffle=False,
    )

    assert returned is augmented
    assert loader.dataset is augmented
    assert loader.dataset.dataset is native
    assert loader.dataset.dataset is not loader.dataset
    assert loader.dataset.dataset.max_num_nodes == 17
    assert validate_globalgce_generation_loader(loader) == 17
    assert len(augmented) == length_before
    assert [augmented[index] for index in range(len(augmented))] == rows_before

    # Identity, rather than a copied/fabricated max_num_nodes, is observable.
    native.max_num_nodes = 19
    assert loader.dataset.dataset.max_num_nodes == 19
    assert validate_globalgce_generation_loader(loader) == 19


def test_fake_official_generate_cfs_can_use_nested_dataset_contract() -> None:
    torch = pytest.importorskip("torch")
    native = _FakeNativeDataset(max_num_nodes=11)
    augmented = attach_globalgce_generation_dataset(
        _FakeAugmentedDataset(),
        native,
    )
    loader = torch.utils.data.DataLoader(augmented, batch_size=1, shuffle=False)

    def fake_generate_cfs(dataloader):
        max_num_nodes = dataloader.dataset.dataset.max_num_nodes
        parent_order = [
            parent_id
            for batch in dataloader
            for parent_id in batch["parent_id"]
        ]
        return max_num_nodes, parent_order

    max_num_nodes, parent_order = fake_generate_cfs(loader)
    assert max_num_nodes == native.max_num_nodes
    assert parent_order == ["p2", "p1", "p3"]


def test_generation_loader_validator_reports_missing_underlying_dataset() -> None:
    class LoaderWithoutDataset:
        pass

    with pytest.raises(
        ValueError,
        match=r"LoaderWithoutDataset.*missing attribute dataset",
    ):
        validate_globalgce_generation_loader(LoaderWithoutDataset())

    class Loader:
        dataset = _FakeAugmentedDataset()

    with pytest.raises(
        ValueError,
        match=r"_FakeAugmentedDataset.*missing underlying attribute dataset",
    ):
        validate_globalgce_generation_loader(Loader())


def test_generation_loader_validator_reports_missing_max_num_nodes() -> None:
    class NativeWithoutMaximum:
        pass

    class Loader:
        dataset = _FakeAugmentedDataset()

    Loader.dataset.dataset = NativeWithoutMaximum()
    with pytest.raises(
        ValueError,
        match=r"NativeWithoutMaximum.*missing required attribute max_num_nodes",
    ):
        validate_globalgce_generation_loader(Loader())


def test_generation_loader_rejects_self_reference() -> None:
    augmented = _FakeAugmentedDataset()
    augmented.dataset = augmented

    class Loader:
        dataset = augmented

    with pytest.raises(ValueError, match="self-referential"):
        validate_globalgce_generation_loader(Loader())
    with pytest.raises(ValueError, match="cannot reference itself"):
        attach_globalgce_generation_dataset(augmented, augmented)


@pytest.mark.parametrize("name", ["calibration_source.csv", "test_source.csv"])
def test_forbidden_cohort_paths_are_rejected(
    fixture_paths: dict[str, Path],
    name: str,
) -> None:
    forbidden = fixture_paths["train"].with_name(name)
    forbidden.write_bytes(fixture_paths["train"].read_bytes())
    with pytest.raises(ValueError, match="forbidden|train-only"):
        load_strict_train_parents(forbidden, expected_parent_count=3)


def test_build_isolates_invalid_and_non_target_and_deduplicates(
    fixture_paths: dict[str, Path],
) -> None:
    summary = _build(fixture_paths)
    root = fixture_paths["output"]
    raw = _read_jsonl(root / "raw_generated_candidates.jsonl")
    pool = _read_jsonl(root / "candidate_pool.jsonl")
    universe = _read_jsonl(root / "candidate_universe.jsonl")
    invalid = _read_jsonl(root / "invalid_candidates.jsonl")
    non_target = _read_jsonl(root / "non_target_candidates.jsonl")

    assert len(raw) == 6
    assert len(invalid) == 1
    assert len(non_target) == 1
    assert {(row["source_parent_id"], row["canonical_smiles"]) for row in pool} == {
        ("p1", "CC"),
        ("p2", "CC"),
        ("p2", "CCC"),
    }
    assert [row["canonical_smiles"] for row in universe] == ["CC", "CCC"]
    cc = next(row for row in universe if row["canonical_smiles"] == "CC")
    assert cc["source_parent_count"] == 2
    assert cc["source_occurrence_count"] == 2
    assert cc["candidate_id"] == stable_candidate_id("CC")
    assert summary["raw_generated_rows"] == 6
    assert summary["rdkit_valid_rows"] == 5
    assert summary["teacher_target_rows"] == 4
    assert summary["candidate_pool_rows"] == 3
    assert summary["canonical_unique_candidates"] == 2
    assert summary["calibration_loaded"] is False
    assert summary["test_loaded"] is False

    audit = audit_mutagenicity_train_pool(
        root,
        train_csv=fixture_paths["train"],
        expected_parent_count=3,
    )
    assert audit["audit_passed"] is True


def test_generator_cannot_return_parent_outside_train(
    fixture_paths: dict[str, Path],
) -> None:
    class BadGenerator(_FakeGenerator):
        def generate(self, parents, **kwargs):
            result = super().generate(parents, **kwargs)
            result.records[0]["source_parent_id"] = "not-in-train"
            return result

    with pytest.raises(ValueError, match="non-train source parent"):
        build_mutagenicity_train_pool(
            train_csv=fixture_paths["train"],
            teacher_path=fixture_paths["teacher"],
            official_root=fixture_paths["official"],
            output_dir=fixture_paths["output"],
            teacher=_FakeTeacher(),
            generator=BadGenerator(),
            config=PoolBuildConfig(
                expected_parent_count=3,
                device="cpu",
                epochs=1,
                top_k_native=2,
            ),
        )


def test_completed_run_refuses_overwrite(fixture_paths: dict[str, Path]) -> None:
    _build(fixture_paths)
    with pytest.raises(FileExistsError, match="cannot be rerun"):
        _build(fixture_paths)


def test_resume_rejects_configuration_mismatch(
    fixture_paths: dict[str, Path],
) -> None:
    with pytest.raises(RuntimeError, match="intentional"):
        build_mutagenicity_train_pool(
            train_csv=fixture_paths["train"],
            teacher_path=fixture_paths["teacher"],
            official_root=fixture_paths["official"],
            output_dir=fixture_paths["output"],
            teacher=_FakeTeacher(),
            generator=_FakeGenerator(fail=True),
            config=PoolBuildConfig(
                expected_parent_count=3,
                device="cpu",
                epochs=1,
                top_k_native=2,
                seed=13,
            ),
        )
    with pytest.raises(ValueError, match="Resume configuration mismatch"):
        build_mutagenicity_train_pool(
            train_csv=fixture_paths["train"],
            teacher_path=fixture_paths["teacher"],
            official_root=fixture_paths["official"],
            output_dir=fixture_paths["output"],
            teacher=_FakeTeacher(),
            generator=_FakeGenerator(),
            config=PoolBuildConfig(
                expected_parent_count=3,
                device="cpu",
                epochs=1,
                top_k_native=2,
                seed=99,
            ),
        )
