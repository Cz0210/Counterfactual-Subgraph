from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem

from src.baselines.clear_mutagenicity_adapter import (
    ATOM_SIDECAR_SCHEMA_VERSION,
    PreparedMolecule,
    build_train_schema,
    molecule_to_clear_graph,
)
from src.baselines.clear_mutagenicity_train_pool import (
    GeneratedGraph,
    TrainPoolConfig,
    audit_train_pool,
    candidate_universe_from_pool,
    cohort_hash,
    read_json,
    read_jsonl,
    run_streaming_generation,
    select_generation_parents,
    stable_candidate_id,
    validate_phase_a_data,
    validate_phase_a_splits,
    write_json,
)


class FakeTeacher:
    available = True

    def score_smiles(self, smiles, label=None, **_kwargs):
        prediction = 0 if smiles == "C" else 1
        probabilities = (0.9, 0.1) if prediction == 0 else (0.1, 0.9)
        return {
            "teacher_result_ok": True,
            "teacher_label": prediction,
            "teacher_prob": probabilities[int(label)],
            "teacher_reason": "ok",
        }


def _prepared(index: int) -> PreparedMolecule:
    smiles = "CC"
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return PreparedMolecule(
        molecule_id=f"MUT_{index:04d}",
        original_smiles=smiles,
        canonical_smiles=smiles,
        label=1,
        split="train",
        source_row_index=index,
        mol=mol,
        atom_categories=(6, 6),
        charge_categories=(0, 0),
        aromatic_categories=(False, False),
        bond_categories=("SINGLE",),
        explicit_h_categories=(0, 0),
        implicit_h_categories=(3, 3),
        no_implicit_categories=(False, False),
        chiral_tag_categories=(0, 0),
    )


def _fixture_data(count: int = 64):
    parents = [_prepared(index) for index in range(count)]
    schema = build_train_schema(parents)
    features = []
    adjacency = []
    sidecars = []
    for parent in parents:
        adj, feat, sidecar = molecule_to_clear_graph(parent, schema=schema)
        features.append(feat)
        adjacency.append(adj)
        sidecars.append(sidecar)
    data = SimpleNamespace(
        molecule_id_all=[row.molecule_id for row in parents],
        canonical_smiles_all=[row.canonical_smiles for row in parents],
        source_split_all=["train"] * count,
        molecule_sidecar_all=sidecars,
        feature_all=features,
        adj_all=adjacency,
        u_all=[np.asarray([0.0])] * count,
        labels_all=[np.asarray([1.0])] * count,
        feature_schema=schema.to_dict(),
        atom_sidecar_schema_version=ATOM_SIDECAR_SCHEMA_VERSION,
        max_num_nodes=2,
    )
    return parents, schema, data


def _generated(parent, feature, adjacency) -> GeneratedGraph:
    generated_feature = np.asarray(feature).copy()
    generated_feature[1, :] = 0.0
    generated_adjacency = np.zeros_like(adjacency)
    generated_adjacency[0, 0] = 1.0
    return GeneratedGraph(
        parent_id=parent.molecule_id,
        features=generated_feature,
        adjacency=generated_adjacency,
        official_pred_before=1,
        official_pred_after=0,
        official_prob_before=(0.1, 0.9),
        official_prob_after=(0.8, 0.2),
    )


def test_deterministic_generation_parent_selection() -> None:
    rows = [_prepared(index) for index in reversed(range(70))]
    selected = select_generation_parents(rows, 64)
    assert [row.molecule_id for row in selected] == [
        f"MUT_{index:04d}" for index in range(64)
    ]


def test_phase_a_contract_checks_train_val_and_sidecar_v2() -> None:
    data = SimpleNamespace(
        molecule_id_all=["a", "b", "c"],
        canonical_smiles_all=["C", "CC", "CCC"],
        source_split_all=["train", "train", "val"],
        molecule_sidecar_all=[{}, {}, {}],
        feature_schema={
            "atom_vocabulary": [6],
            "formal_charge_vocabulary": [0],
            "aromaticity_vocabulary": [False],
            "bond_type_vocabulary": ["SINGLE"],
            "atom_feature_start": 0,
            "atom_feature_end": 1,
            "charge_feature_start": 1,
            "charge_feature_end": 2,
            "aromatic_feature_index": 2,
            "node_present_feature_index": 3,
            "feature_dim": 4,
        },
        atom_sidecar_schema_version=ATOM_SIDECAR_SCHEMA_VERSION,
        max_num_nodes=99,
        labels_all=[0, 1, 1],
    )
    result = validate_phase_a_data(
        data, expected_train_rows=2, expected_val_rows=1
    )
    assert result["model_train_rows"] == 2
    assert result["model_val_rows"] == 1
    assert result["max_num_nodes"] == 99
    assert result["calibration_loaded"] is False
    assert result["test_loaded"] is False
    split_result = validate_phase_a_splits(
        data,
        {
            "idx_train_list": [np.asarray([0, 1])],
            "idx_val_list": [np.asarray([2])],
            "idx_test_list": [np.asarray([2])],
            "calibration_loaded": False,
            "test_loaded": False,
        },
    )
    assert split_result["official_train_indices_exact"] is True
    assert split_result["official_test_loader_is_validation_alias"] is True


def test_phase_a_split_rejects_validation_in_official_train() -> None:
    data = SimpleNamespace(source_split_all=["train", "val"])
    with pytest.raises(Exception, match="train split"):
        validate_phase_a_splits(
            data,
            {
                "idx_train_list": [np.asarray([0, 1])],
                "idx_val_list": [np.asarray([1])],
                "idx_test_list": [np.asarray([1])],
                "calibration_loaded": False,
                "test_loaded": False,
            },
        )


def test_streaming_generation_uses_four_chunks_and_lightweight_records(
    tmp_path: Path,
) -> None:
    parents, schema, data = _fixture_data()
    chunks: list[list[str]] = []

    def generate(chunk, indices, _chunk_index):
        chunks.append([row.molecule_id for row in chunk])
        return [
            _generated(row, data.feature_all[index], data.adj_all[index])
            for row, index in zip(chunk, indices, strict=True)
        ]

    result = run_streaming_generation(
        output_dir=tmp_path,
        parents=parents,
        data=data,
        schema=schema,
        teacher=FakeTeacher(),
        generate_chunk=generate,
        config=TrainPoolConfig(),
        config_fingerprint="config",
        model_checkpoint_hash="model",
    )
    assert [len(chunk) for chunk in chunks] == [16, 16, 16, 16]
    assert [item for chunk in chunks for item in chunk] == [
        row.molecule_id for row in parents
    ]
    assert result["raw_generated_rows"] == 64
    assert result["candidate_pool_rows"] == 64
    assert result["canonical_unique_candidates"] == 1
    raw = read_jsonl(tmp_path / "raw_generated_candidates.jsonl")
    assert all("features" not in row and "adjacency" not in row for row in raw)
    assert all(row["strict_flip"] is True for row in raw)
    assert all(row["teacher_pred"] == 0 for row in raw)
    universe = read_jsonl(tmp_path / "candidate_universe.jsonl")
    assert universe[0]["canonical_smiles"] == "C"
    assert universe[0]["source_parent_count"] == 64
    assert universe[0]["occurrence_count"] == 64


def test_generation_resume_does_not_duplicate_rows(tmp_path: Path) -> None:
    parents, schema, data = _fixture_data()
    calls = 0

    def fail_after_first(chunk, indices, _chunk_index):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return [
            _generated(row, data.feature_all[index], data.adj_all[index])
            for row, index in zip(chunk, indices, strict=True)
        ]

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_streaming_generation(
            output_dir=tmp_path,
            parents=parents,
            data=data,
            schema=schema,
            teacher=FakeTeacher(),
            generate_chunk=fail_after_first,
            config=TrainPoolConfig(),
            config_fingerprint="config",
            model_checkpoint_hash="model",
        )
    progress = read_json(tmp_path / "generation_progress.json")
    assert progress["next_parent_offset"] == 16

    resumed_calls: list[list[str]] = []

    def resume(chunk, indices, _chunk_index):
        resumed_calls.append([row.molecule_id for row in chunk])
        return [
            _generated(row, data.feature_all[index], data.adj_all[index])
            for row, index in zip(chunk, indices, strict=True)
        ]

    run_streaming_generation(
        output_dir=tmp_path,
        parents=parents,
        data=data,
        schema=schema,
        teacher=FakeTeacher(),
        generate_chunk=resume,
        config=TrainPoolConfig(),
        config_fingerprint="config",
        model_checkpoint_hash="model",
    )
    assert resumed_calls[0][0] == "MUT_0016"
    raw = read_jsonl(tmp_path / "raw_generated_candidates.jsonl")
    assert len(raw) == 64
    assert len({row["source_parent_id"] for row in raw}) == 64


def test_completed_generation_resume_reuses_final_jsonl(
    tmp_path: Path,
) -> None:
    parents, schema, data = _fixture_data()

    def generate(chunk, indices, _chunk_index):
        return [
            _generated(row, data.feature_all[index], data.adj_all[index])
            for row, index in zip(chunk, indices, strict=True)
        ]

    first = run_streaming_generation(
        output_dir=tmp_path,
        parents=parents,
        data=data,
        schema=schema,
        teacher=FakeTeacher(),
        generate_chunk=generate,
        config=TrainPoolConfig(),
        config_fingerprint="config",
        model_checkpoint_hash="model",
    )

    def must_not_generate(*_args):
        raise AssertionError("completed generation must not run another chunk")

    resumed = run_streaming_generation(
        output_dir=tmp_path,
        parents=parents,
        data=data,
        schema=schema,
        teacher=FakeTeacher(),
        generate_chunk=must_not_generate,
        config=TrainPoolConfig(),
        config_fingerprint="config",
        model_checkpoint_hash="model",
    )
    assert resumed == first
    assert len(read_jsonl(tmp_path / "raw_generated_candidates.jsonl")) == 64


def test_resume_rejects_model_or_config_mismatch(tmp_path: Path) -> None:
    parents, schema, data = _fixture_data()

    def interrupt(chunk, indices, _chunk_index):
        rows = [
            _generated(row, data.feature_all[index], data.adj_all[index])
            for row, index in zip(chunk, indices, strict=True)
        ]
        if chunk[0].molecule_id != "MUT_0000":
            raise RuntimeError("stop")
        return rows

    with pytest.raises(RuntimeError):
        run_streaming_generation(
            output_dir=tmp_path,
            parents=parents,
            data=data,
            schema=schema,
            teacher=FakeTeacher(),
            generate_chunk=interrupt,
            config=TrainPoolConfig(),
            config_fingerprint="config-a",
            model_checkpoint_hash="model-a",
        )
    with pytest.raises(Exception, match="resume mismatch"):
        run_streaming_generation(
            output_dir=tmp_path,
            parents=parents,
            data=data,
            schema=schema,
            teacher=FakeTeacher(),
            generate_chunk=lambda *_args: [],
            config=TrainPoolConfig(),
            config_fingerprint="config-b",
            model_checkpoint_hash="model-a",
        )


def test_candidate_universe_is_canonical_dedup_with_support() -> None:
    pool = [
        {
            "canonical_smiles": "C",
            "source_parent_id": "p2",
            "generator_rank": 1,
            "projection_provenance": {},
            "teacher_prob_0": 0.8,
            "teacher_prob_1": 0.2,
            "num_atoms": 1,
            "num_bonds": 0,
        },
        {
            "canonical_smiles": "C",
            "source_parent_id": "p1",
            "generator_rank": 1,
            "projection_provenance": {},
            "teacher_prob_0": 0.9,
            "teacher_prob_1": 0.1,
            "num_atoms": 1,
            "num_bonds": 0,
        },
    ]
    universe = candidate_universe_from_pool(pool)
    assert len(universe) == 1
    assert universe[0]["candidate_id"] == stable_candidate_id("C")
    assert universe[0]["source_parent_ids"] == ["p1", "p2"]
    assert universe[0]["source_parent_count"] == 2
    assert universe[0]["occurrence_count"] == 2


def _write_generation_csv(path: Path, count: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "molecule_id",
                "smiles",
                "label",
                "split",
                "teacher_pred",
                "teacher_correct",
            ),
        )
        writer.writeheader()
        for index in range(count):
            writer.writerow(
                {
                    "molecule_id": f"MUT_{index:04d}",
                    "smiles": "CC",
                    "label": 1,
                    "split": "train",
                    "teacher_pred": 1,
                    "teacher_correct": "true",
                }
            )


def test_independent_audit_accepts_partial_source_parent_coverage(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    generation_csv = tmp_path / "train_source.csv"
    _write_generation_csv(generation_csv, 3)
    canonical = "C"
    pool = [
        {
            "source_parent_id": "MUT_0000",
            "canonical_smiles": canonical,
        },
        {
            "source_parent_id": "MUT_0001",
            "canonical_smiles": canonical,
        },
    ]
    universe = [
        {
            "candidate_id": stable_candidate_id(canonical),
            "canonical_smiles": canonical,
            "source_parent_ids": ["MUT_0000", "MUT_0001"],
            "source_parent_count": 2,
            "occurrence_count": 2,
            "teacher_pred": 0,
            "teacher_target_ok": True,
        }
    ]
    raw = [
        {"source_parent_id": f"MUT_{index:04d}"} for index in range(3)
    ]
    for name, rows in (
        ("candidate_pool.jsonl", pool),
        ("candidate_universe.jsonl", universe),
        ("raw_generated_candidates.jsonl", raw),
    ):
        (run_dir / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    write_json(
        run_dir / "summary.json",
        {
            "model_train_rows": 4,
            "model_val_rows": 2,
            "selected_generation_parents": 3,
            "source_parent_coverage": 2 / 3,
            "calibration_loaded": False,
            "test_loaded": False,
            "run_complete": True,
        },
    )
    write_json(
        run_dir / "run_manifest.json",
        {
            "inputs": {"generation_csv": str(generation_csv)},
            "calibration_loaded": False,
            "test_loaded": False,
            "run_complete": True,
        },
    )
    write_json(run_dir / "generation_progress.json", {"run_complete": True})
    write_json(run_dir / "_RUN_COMPLETE.json", {"run_complete": True})
    result = audit_train_pool(
        run_dir=run_dir,
        generation_csv=generation_csv,
        expected_model_train_rows=4,
        expected_model_val_rows=2,
        expected_generation_parent_rows=3,
        expected_selected_parents=3,
    )
    assert result["candidate_source_parent_rows"] == 2
    assert result["source_parent_coverage_recomputed"] == pytest.approx(2 / 3)
    assert result["calibration_rows_loaded"] == 0
    assert result["test_rows_loaded"] == 0


def test_train_pool_config_rejects_full_parent_run() -> None:
    with pytest.raises(ValueError, match="64-parent smoke"):
        TrainPoolConfig(parent_limit=1448).validate_smoke()


def test_cohort_hash_changes_with_parent_order() -> None:
    rows = [_prepared(0), _prepared(1)]
    assert cohort_hash(rows) != cohort_hash(list(reversed(rows)))
