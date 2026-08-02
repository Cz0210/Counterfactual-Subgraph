from __future__ import annotations

import csv
import hashlib
import importlib.util
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
    ClearMutagenicityEmptyPoolError,
    GeneratedGraph,
    GenerationProfile,
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
    feature_audit = read_json(tmp_path / "feature_decoding_summary.json")
    assert feature_audit["node_feature_dim"] == schema.feature_dim
    assert feature_audit["feature_dimension_matches_schema"] is True
    assert feature_audit["generated_row_count"] == 64
    assert feature_audit["padded_node_count"] == 64
    assert feature_audit["raw_atom_argmax_index_distribution"] == {"0": 128}
    assert feature_audit["decoded_atomic_number_distribution"] == {"6": 64}


def test_full_generation_profile_streams_all_parents_in_91_chunks(
    tmp_path: Path,
) -> None:
    parents, schema, data = _fixture_data(1448)
    chunk_sizes: list[int] = []

    def generate(chunk, indices, _chunk_index):
        chunk_sizes.append(len(chunk))
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
        config=TrainPoolConfig(parent_limit=1448),
        config_fingerprint="full-config",
        model_checkpoint_hash="full-model",
        generation_profile=GenerationProfile.FULL,
    )
    assert chunk_sizes == ([16] * 90) + [8]
    assert result["generation_profile"] == "full"
    assert result["selected_generation_parents"] == 1448
    assert result["completed_chunk_count"] == 91
    progress = read_json(tmp_path / "generation_progress.json")
    assert progress["generation_profile"] == "full"
    assert progress["selected_parent_count"] == 1448
    assert progress["completed_chunk_count"] == 91


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
    _write_generation_csv(generation_csv, 64)
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
        {"source_parent_id": f"MUT_{index:04d}"} for index in range(64)
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
            "generation_source_parent_rows": 64,
            "selected_generation_parents": 64,
            "generation_chunk_size": 16,
            "completed_chunk_count": 4,
            "generation_profile": "smoke",
            "source_parent_coverage": 2 / 64,
            "calibration_loaded": False,
            "test_loaded": False,
            "run_complete": True,
        },
    )
    write_json(
        run_dir / "run_manifest.json",
        {
            "inputs": {"generation_csv": str(generation_csv)},
            "generation_profile": "smoke",
            "generation_only": False,
            "calibration_loaded": False,
            "test_loaded": False,
            "run_complete": True,
        },
    )
    write_json(
        run_dir / "generation_progress.json",
        {
            "generation_profile": "smoke",
            "selected_parent_count": 64,
            "completed_chunk_count": 4,
            "run_complete": True,
        },
    )
    write_json(run_dir / "_RUN_COMPLETE.json", {"run_complete": True})
    result = audit_train_pool(
        run_dir=run_dir,
        generation_csv=generation_csv,
        expected_model_train_rows=4,
        expected_model_val_rows=2,
        expected_generation_parent_rows=64,
        expected_selected_parents=64,
    )
    assert result["candidate_source_parent_rows"] == 2
    assert result["source_parent_coverage_recomputed"] == pytest.approx(2 / 64)
    assert result["calibration_rows_loaded"] == 0
    assert result["test_rows_loaded"] == 0
    assert result["generation_profile"] == "smoke"


@pytest.mark.parametrize(
    ("profile", "parent_limit"),
    (("smoke", 64), ("full", 1448)),
)
def test_generation_profile_accepts_only_preregistered_parent_count(
    profile: str, parent_limit: int
) -> None:
    resolved = TrainPoolConfig(
        parent_limit=parent_limit
    ).validate_generation_contract(profile)
    assert resolved.value == profile


@pytest.mark.parametrize(
    ("profile", "parent_limit", "required"),
    (("smoke", 1448, 64), ("full", 64, 1448)),
)
def test_generation_profile_rejects_cross_profile_parent_count(
    profile: str, parent_limit: int, required: int
) -> None:
    with pytest.raises(ValueError, match=f"parent_limit={required}"):
        TrainPoolConfig(
            parent_limit=parent_limit
        ).validate_generation_contract(profile)


def test_legacy_validate_smoke_still_rejects_full_parent_run() -> None:
    with pytest.raises(ValueError, match="64-parent smoke"):
        TrainPoolConfig(parent_limit=1448).validate_smoke()


def test_cohort_hash_changes_with_parent_order() -> None:
    rows = [_prepared(0), _prepared(1)]
    assert cohort_hash(rows) != cohort_hash(list(reversed(rows)))


def test_empty_pool_writes_diagnostics_and_failed_marker(tmp_path: Path) -> None:
    parents, schema, data = _fixture_data()

    def invalid_generation(chunk, indices, _chunk_index):
        rows = []
        for parent, index in zip(chunk, indices, strict=True):
            features = np.zeros_like(data.feature_all[index])
            rows.append(
                GeneratedGraph(
                    parent_id=parent.molecule_id,
                    features=features,
                    adjacency=np.zeros_like(data.adj_all[index]),
                    official_pred_before=1,
                    official_pred_after=1,
                )
            )
        return rows

    checkpoint_provenance = {
        "graphpred_checkpoint_path": "/run/graphpred.pt",
        "graphpred_checkpoint_sha256": "a" * 64,
        "graphcfe_checkpoint_path": "/run/graphcfe.pt",
        "graphcfe_checkpoint_sha256": "b" * 64,
    }
    with pytest.raises(ClearMutagenicityEmptyPoolError):
        run_streaming_generation(
            output_dir=tmp_path,
            parents=parents,
            data=data,
            schema=schema,
            teacher=FakeTeacher(),
            generate_chunk=invalid_generation,
            config=TrainPoolConfig(),
            config_fingerprint="empty-config",
            model_checkpoint_hash="empty-model",
            checkpoint_provenance=checkpoint_provenance,
        )
    for name in (
        "raw_generated_candidates.jsonl",
        "invalid_candidates.jsonl",
        "non_target_candidates.jsonl",
        "candidate_pool.jsonl",
        "candidate_universe.jsonl",
        "generation_progress.json",
        "feature_decoding_summary.json",
        "failure_summary.json",
        "_RUN_FAILED.json",
    ):
        assert (tmp_path / name).is_file()
    assert not (tmp_path / "_RUN_COMPLETE.json").exists()
    assert len(read_jsonl(tmp_path / "raw_generated_candidates.jsonl")) == 64
    assert len(read_jsonl(tmp_path / "invalid_candidates.jsonl")) == 64
    assert read_jsonl(tmp_path / "candidate_pool.jsonl") == []
    failure = read_json(tmp_path / "failure_summary.json")
    assert failure["run_complete"] is False
    assert failure["invalid_reason_counts"] == {
        "generated_invalid_active_node_mask": 64
    }
    assert failure["graphpred_checkpoint_sha256"] == "a" * 64
    assert failure["graphcfe_checkpoint_sha256"] == "b" * 64
    assert failure["calibration_loaded"] is False
    assert failure["test_loaded"] is False


def test_all_non_target_pool_reports_rf_after_counts(tmp_path: Path) -> None:
    parents, schema, data = _fixture_data()

    def valid_non_target_generation(chunk, indices, _chunk_index):
        return [
            GeneratedGraph(
                parent_id=parent.molecule_id,
                features=np.asarray(data.feature_all[index]).copy(),
                adjacency=np.asarray(data.adj_all[index]).copy(),
                official_pred_before=1,
                official_pred_after=1,
            )
            for parent, index in zip(chunk, indices, strict=True)
        ]

    with pytest.raises(ClearMutagenicityEmptyPoolError):
        run_streaming_generation(
            output_dir=tmp_path,
            parents=parents,
            data=data,
            schema=schema,
            teacher=FakeTeacher(),
            generate_chunk=valid_non_target_generation,
            config=TrainPoolConfig(),
            config_fingerprint="non-target-config",
            model_checkpoint_hash="non-target-model",
        )
    failure = read_json(tmp_path / "failure_summary.json")
    assert failure["invalid_candidate_rows"] == 0
    assert failure["non_target_candidate_rows"] == 64
    assert failure["rf_before_prediction_counts"] == {"1": 64}
    assert failure["rf_after_prediction_counts"] == {"1": 64}
    assert failure["candidate_pool_rows"] == 0


def _load_build_script_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/baselines/clear/build_mutagenicity_train_pool.py"
    )
    spec = importlib.util.spec_from_file_location("clear_build_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generation_profile_cli_is_explicit_and_defaults_to_smoke() -> None:
    module = _load_build_script_module()
    parser = module.build_parser()
    args = parser.parse_args(["--output-dir", "unused"])
    assert args.generation_profile == "smoke"
    profile_action = next(
        action
        for action in parser._actions
        if action.dest == "generation_profile"
    )
    assert tuple(profile_action.choices) == ("smoke", "full")


def test_full_profile_requires_generation_only() -> None:
    module = _load_build_script_module()
    config = TrainPoolConfig(parent_limit=1448)
    with pytest.raises(ValueError, match="only valid with --generation-only"):
        module._validate_run_profile(
            config=config,
            generation_profile="full",
            generation_only=False,
        )
    assert module._validate_run_profile(
        config=config,
        generation_profile="full",
        generation_only=True,
    ) is GenerationProfile.FULL


def test_generation_only_uses_explicit_checkpoints_without_training(
    tmp_path: Path,
) -> None:
    module = _load_build_script_module()
    source_run = tmp_path / "failed_run_2021625"
    checkpoints = source_run / "checkpoints"
    checkpoints.mkdir(parents=True)
    graphpred = checkpoints / "graphpred.pt"
    graphcfe = checkpoints / "graphcfe.pt"
    graphpred.write_bytes(b"graphpred-2021625")
    graphcfe.write_bytes(b"graphcfe-2021625")
    training_calls = 0
    profile = module._validate_run_profile(
        config=TrainPoolConfig(parent_limit=1448),
        generation_profile="full",
        generation_only=True,
    )
    assert profile is GenerationProfile.FULL

    def train_models():
        nonlocal training_calls
        training_calls += 1
        raise AssertionError("generation-only replay must not train")

    resolved_graphpred, resolved_graphcfe, resolved_source, trained = (
        module._resolve_generation_checkpoints(
            generation_only=True,
            graphpred_checkpoint=graphpred,
            graphcfe_checkpoint=graphcfe,
            source_run_root=source_run,
            train_models=train_models,
        )
    )
    assert training_calls == 0
    assert trained is False
    assert resolved_source == source_run.resolve()
    assert resolved_graphpred == graphpred.resolve()
    assert resolved_graphcfe == graphcfe.resolve()
    provenance = module._checkpoint_provenance(
        resolved_graphpred, resolved_graphcfe
    )
    assert provenance["graphpred_checkpoint_sha256"] == hashlib.sha256(
        graphpred.read_bytes()
    ).hexdigest()
    assert provenance["graphcfe_checkpoint_sha256"] == hashlib.sha256(
        graphcfe.read_bytes()
    ).hexdigest()


def test_generation_only_rejects_checkpoint_outside_source_run(
    tmp_path: Path,
) -> None:
    module = _load_build_script_module()
    source_run = tmp_path / "source"
    source_run.mkdir()
    graphpred = tmp_path / "graphpred.pt"
    graphcfe = source_run / "graphcfe.pt"
    graphpred.write_bytes(b"x")
    graphcfe.write_bytes(b"y")
    with pytest.raises(ValueError, match="inside the explicit source run root"):
        module._resolve_generation_checkpoints(
            generation_only=True,
            graphpred_checkpoint=graphpred,
            graphcfe_checkpoint=graphcfe,
            source_run_root=source_run,
            train_models=lambda: (_ for _ in ()).throw(AssertionError()),
        )
