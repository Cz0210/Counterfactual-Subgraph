from __future__ import annotations

import csv
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest
from rdkit import Chem

from src.baselines.globalgce_mutagenicity_adapter import (
    FROZEN_GINE_IN_MEMORY_FILES,
    GlobalGCECodecMetadata,
    GlobalGCEEmptyCandidateUniverseError,
    GlobalGCEMutagenicityCodecError,
    NativeGenerationResult,
    NativeGeneratorProtocol,
    OFFICIAL_GLOBALGCE_API_SIGNATURE_SCHEMA,
    OfficialGlobalGCEMutagenicityGenerator,
    PINNED_OFFICIAL_GLOBALGCE_API_SIGNATURES,
    PoolBuildConfig,
    TrainParent,
    _add_bond_once,
    _build_dense_dataset,
    _build_training_resume_identity,
    _hash_open_regular_source,
    _import_official_modules,
    _load_general_train_rows,
    _read_csv_payload,
    attach_globalgce_generation_dataset,
    audit_mutagenicity_train_pool,
    build_mutagenicity_train_pool,
    capture_globalgce_module_provenance,
    decode_globalgce_molecule,
    globalgce_tensors_to_graph_record,
    load_strict_train_parents,
    log_globalgce_phase_memory,
    probe_source_graph_codec,
    require_source_codec_gate,
    stable_candidate_id,
    validate_official_globalgce_api_signatures,
    validate_globalgce_generation_loader,
)


def test_descriptor_csv_payload_loader_uses_in_memory_bytes() -> None:
    assert _read_csv_payload(b"molecule_id,label\na,1\n") == [
        {"molecule_id": "a", "label": "1"}
    ]


def test_taste_generator_binds_exact_descriptor_authorized_gine_payloads(
    tmp_path: Path,
) -> None:
    official = tmp_path / "official"
    _write_official_root(official)
    train = tmp_path / "train.csv"
    _write_three_class_train(train)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.pt").write_bytes(b"held-model")
    payloads = {
        name: f"held:{name}".encode("utf-8")
        for name in FROZEN_GINE_IN_MEMORY_FILES
    }
    generator = OfficialGlobalGCEMutagenicityGenerator(
        official,
        native_train_csv=train,
        dataset_name="tastemolnet",
        min_freq=2,
        frozen_gine_checkpoint=checkpoint,
        frozen_gine_payloads=payloads,
        native_train_payload=train.read_bytes(),
        source_label=1,
        target_label=2,
        num_classes=3,
    )
    identity = generator.config_identity()
    assert identity["prediction_backend"] == "frozen_gine_differentiable_bridge"
    assert identity["num_classes"] == 3
    assert identity["source_label"] == 1
    assert identity["target_label"] == 2
    assert identity["native_train_csv"]["kind"] == (
        "descriptor_authorized_file_payload"
    )
    assert identity["frozen_gine_payload_sha256"] == {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in sorted(payloads.items())
    }

    incomplete = dict(payloads)
    incomplete.pop("label_map.json")
    with pytest.raises(ValueError, match="exact seven-file set"):
        OfficialGlobalGCEMutagenicityGenerator(
            official,
            native_train_csv=train,
            dataset_name="tastemolnet",
            frozen_gine_checkpoint=checkpoint,
            frozen_gine_payloads=incomplete,
            native_train_payload=train.read_bytes(),
            source_label=1,
            target_label=0,
            num_classes=3,
        )

    coerced = dict(payloads)
    coerced["model.pt"] = bytearray(coerced["model.pt"])
    with pytest.raises(ValueError, match="native bytes"):
        OfficialGlobalGCEMutagenicityGenerator(
            official,
            native_train_csv=train,
            dataset_name="tastemolnet",
            frozen_gine_checkpoint=checkpoint,
            frozen_gine_payloads=coerced,
            native_train_payload=train.read_bytes(),
            source_label=1,
            target_label=0,
            num_classes=3,
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


def _write_ten_parent_train(path: Path) -> None:
    smiles_values = [
        "CC",
        "CCC",
        "CCCC",
        "CCO",
        "CCN",
        "CCCl",
        "CCBr",
        "COC",
        "CNC",
        "CCF",
    ]
    rows = [
        {
            "molecule_id": f"p{index:02d}",
            "smiles": smiles,
            "label": 1,
            "split": "train",
            "teacher_pred": 1,
            "teacher_correct": "true",
        }
        for index, smiles in enumerate(smiles_values)
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_many_parent_train(path: Path, count: int) -> None:
    rows = [
        {
            "molecule_id": f"p{index:03d}",
            "smiles": "C" * (index + 1),
            "label": 1,
            "split": "train",
            "teacher_pred": 1,
            "teacher_correct": "true",
        }
        for index in range(int(count))
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_three_class_train(path: Path) -> None:
    rows = [
        {
            "molecule_id": f"taste-{label}-{index}",
            "smiles": ("C" * (2 + label + index)) + ("O" if label == 0 else ""),
            "label": label,
            "split": "train",
        }
        for label in range(3)
        for index in range(2)
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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


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
                "source_codec_checked_rows": len(parents),
                "source_codec_rdkit_valid_rows": len(parents),
                "source_codec_structure_match_rows": len(parents),
                "source_codec_invalid_valence_rows": 0,
                "source_codec_attribute_mapping_failed_rows": 0,
                "source_codec_passed": True,
                "source_codec_failure_examples": [],
                "atom_attribute_source": "source_anchored",
                "formal_charge_encoded_by_native": False,
                "source_atom_mapping_method": (
                    "rdkit_atom_index_preserved_by_dense_builder"
                ),
                "source_atom_mapping_unique": True,
                "source_formal_charge_nonzero_atom_count": 0,
            },
        )


def test_official_generator_accepts_two_taste_target_branches_on_one_three_class_gine(
    tmp_path: Path,
) -> None:
    official = tmp_path / "official"
    _write_official_root(official)
    train_csv = tmp_path / "taste_train.csv"
    _write_three_class_train(train_csv)
    checkpoint = tmp_path / "gine"
    checkpoint.mkdir()
    (checkpoint / "model.pt").write_bytes(b"frozen-three-class-gine")

    branches = [
        OfficialGlobalGCEMutagenicityGenerator(
            official,
            native_train_csv=train_csv,
            dataset_name="TasteMolNet",
            frozen_gine_checkpoint=checkpoint,
            num_classes=3,
            source_label=1,
            target_label=target,
        )
        for target in (0, 2)
    ]
    assert [branch.target_label for branch in branches] == [0, 2]
    assert all(branch.source_label == 1 for branch in branches)
    assert all(branch.num_classes == 3 for branch in branches)
    assert [branch.config_identity()["target_label"] for branch in branches] == [0, 2]
    assert all(branch.config_identity()["num_classes"] == 3 for branch in branches)
    loaded = _load_general_train_rows(train_csv, num_classes=3)
    assert {row.label for row in loaded} == {0, 1, 2}


@pytest.mark.parametrize("bad_label", ("0.9", "1.5", "NaN", "True", "+1", "01"))
def test_multiclass_native_train_rejects_noncanonical_integer_labels(
    tmp_path: Path,
    bad_label: str,
) -> None:
    train_csv = tmp_path / "taste_train.csv"
    _write_three_class_train(train_csv)
    with train_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["label"] = bad_label
    with train_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="exact integer label"):
        _load_general_train_rows(train_csv, num_classes=3)


@pytest.mark.parametrize("bad_split", ("", "TRAIN", " train", "validation"))
def test_multiclass_native_train_requires_explicit_exact_train_split(
    tmp_path: Path,
    bad_split: str,
) -> None:
    train_csv = tmp_path / "taste_train.csv"
    _write_three_class_train(train_csv)
    with train_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["split"] = bad_split
    with train_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="non-train split"):
        _load_general_train_rows(train_csv, num_classes=3)


def test_taste_multiclass_generator_requires_the_frozen_gine(tmp_path: Path) -> None:
    official = tmp_path / "official"
    _write_official_root(official)
    train_csv = tmp_path / "taste_train.csv"
    _write_three_class_train(train_csv)
    with pytest.raises(ValueError, match="requires one frozen GINE"):
        OfficialGlobalGCEMutagenicityGenerator(
            official,
            native_train_csv=train_csv,
            dataset_name="TasteMolNet",
            num_classes=3,
            source_label=1,
            target_label=2,
        )


@pytest.mark.parametrize(
    "dataset_alias",
    ("tastemolnet", "taste", "bst", "bitter_sweet_tasteless"),
)
def test_every_registered_taste_alias_rejects_binary_gtgnn_fallback(
    tmp_path: Path,
    dataset_alias: str,
) -> None:
    official = tmp_path / "official"
    _write_official_root(official)
    train_csv = tmp_path / "taste_train.csv"
    _write_three_class_train(train_csv)
    with pytest.raises(
        ValueError,
        match="TasteMolNet GlobalGCE requires the frozen three-class GINE",
    ):
        OfficialGlobalGCEMutagenicityGenerator(
            official,
            native_train_csv=train_csv,
            dataset_name=dataset_alias,
            num_classes=2,
            source_label=1,
            target_label=0,
            frozen_gine_checkpoint=None,
        )


@pytest.mark.parametrize(
    "dataset_alias",
    ("tastemolnet", "taste", "bst", "bitter_sweet_tasteless"),
)
def test_every_registered_taste_alias_uses_same_three_class_gine_route(
    tmp_path: Path,
    dataset_alias: str,
) -> None:
    official = tmp_path / "official"
    _write_official_root(official)
    train_csv = tmp_path / "taste_train.csv"
    _write_three_class_train(train_csv)
    checkpoint = tmp_path / "gine"
    checkpoint.mkdir()
    (checkpoint / "model.pt").write_bytes(b"frozen-three-class-gine")
    generator = OfficialGlobalGCEMutagenicityGenerator(
        official,
        native_train_csv=train_csv,
        dataset_name=dataset_alias,
        num_classes=3,
        source_label=1,
        target_label=2,
        frozen_gine_checkpoint=checkpoint,
    )
    assert generator.dataset_id == "tastemolnet"
    assert generator.dataset_name == "TasteMolNet"
    assert generator.num_classes == 3
    assert generator.source_label == 1
    assert generator.target_label == 2


def test_bace_binary_native_gtgnn_constructor_remains_available(tmp_path: Path) -> None:
    official = tmp_path / "official"
    _write_official_root(official)
    train_csv = tmp_path / "binary_train.csv"
    _write_train(train_csv)
    generator = OfficialGlobalGCEMutagenicityGenerator(
        official,
        native_train_csv=train_csv,
        dataset_name="BACE",
        num_classes=2,
        source_label=1,
        target_label=0,
    )
    assert generator.frozen_gine_checkpoint is None
    assert generator.num_classes == 2


def test_binary_bace_v2_identity_keeps_native_route_and_ordered_cohorts(
    tmp_path: Path,
) -> None:
    official = tmp_path / "official"
    _write_official_root(official)
    train_csv = tmp_path / "binary_train.csv"
    _write_train(train_csv)
    native = [
        TrainParent("n0a", "CC", 0, "train"),
        TrainParent("n0b", "CCC", 0, "train"),
        TrainParent("n1a", "CCO", 1, "train"),
        TrainParent("n1b", "CCN", 1, "train"),
    ]
    source = [
        TrainParent("s1", "CCO", 1, "train"),
        TrainParent("s2", "CCN", 1, "train"),
        TrainParent("s3", "CCC", 1, "train"),
        TrainParent("s4", "CCCl", 1, "train"),
        TrainParent("filtered-out", "CCBr", 1, "train"),
    ]

    def _identity(rows: list[TrainParent]) -> tuple[dict, str]:
        return _build_training_resume_identity(
            dataset_name="BACE",
            num_classes=2,
            source_label=1,
            target_label=0,
            official_src=official,
            native_train_csv=train_csv,
            native_parents=native,
            native_train_idx=[0, 2],
            native_val_idx=[1, 3],
            source_parents=rows,
            source_train_idx=[0, 2],
            source_val_idx=[1, 3],
            frozen_gine_checkpoint=None,
            seed=7,
            epochs=1,
            top_k_native=20,
            learning_rate=0.001,
            dropout=0.5,
            min_freq=7,
            gspan_flush_every=256,
            gspan_max_in_memory_candidates=256,
            gspan_exact_top_k_pruning=False,
            gspan_adoption_identity=None,
        )

    identity, digest = _identity(source)
    reordered, reordered_digest = _identity(list(reversed(source)))
    assert identity["oracle_identity"]["backend"] == "official_native_gtgnn"
    assert identity["oracle_identity"]["num_classes"] == 2
    assert identity["source_train_cohort"]["count"] == 4
    assert len(digest) == 64
    assert reordered["source_train_cohort"] != identity["source_train_cohort"]
    assert reordered_digest != digest


@pytest.mark.parametrize(
    ("num_classes", "source_label", "target_label"),
    ((True, 1, 0), (3.0, 1, 0), (3, True, 0), (3, 1, False), (3, 1, 1), (3, 1, 3)),
)
def test_official_generator_rejects_untyped_or_invalid_multiclass_contract(
    tmp_path: Path,
    num_classes: object,
    source_label: object,
    target_label: object,
) -> None:
    official = tmp_path / "official"
    _write_official_root(official)
    train_csv = tmp_path / "taste_train.csv"
    _write_three_class_train(train_csv)
    with pytest.raises(ValueError, match="num_classes|exact integers|distinct frozen"):
        OfficialGlobalGCEMutagenicityGenerator(
            official,
            native_train_csv=train_csv,
            num_classes=num_classes,  # type: ignore[arg-type]
            source_label=source_label,  # type: ignore[arg-type]
            target_label=target_label,  # type: ignore[arg-type]
        )


class _LimitedCoverageGenerator:
    def generate(self, parents, **_kwargs):
        records = [
            {
                "source_parent_id": parent.parent_id,
                "source_parent_smiles": parent.smiles,
                "source_split": "train",
                "raw_smiles": "C",
                "generator_rank": 1,
                "native_codec_decoded": True,
                "native_conversion_ok": True,
            }
            for parent in parents[:2]
        ]
        return NativeGenerationResult(
            records,
            {
                "internal_train_ids_hash": "train-hash",
                "internal_val_ids_hash": "val-hash",
                "source_codec_checked_rows": len(parents),
                "source_codec_rdkit_valid_rows": len(parents),
                "source_codec_structure_match_rows": len(parents),
                "source_codec_invalid_valence_rows": 0,
                "source_codec_attribute_mapping_failed_rows": 0,
                "source_codec_passed": True,
                "source_codec_failure_examples": [],
                "atom_attribute_source": "source_anchored",
                "formal_charge_encoded_by_native": False,
                "source_atom_mapping_method": (
                    "rdkit_atom_index_preserved_by_dense_builder"
                ),
                "source_atom_mapping_unique": True,
                "source_formal_charge_nonzero_atom_count": 0,
            },
        )


class _ChunkedGenerator:
    def __init__(self, *, fail_after_total_chunks: int | None = None) -> None:
        self.fail_after_total_chunks = fail_after_total_chunks
        self.train_calls = 0
        self.rule_selection_calls = 0
        self.chunk_ranges: list[tuple[int, int]] = []
        self.parent_order: list[str] = []
        self.raw_part_row_counts: list[int] = []

    def config_identity(self) -> dict:
        return {"generator_class": "_ChunkedGenerator", "version": 1}

    @staticmethod
    def _summary(parent_count: int) -> dict:
        return {
            "internal_train_ids_hash": "train-hash",
            "internal_val_ids_hash": "val-hash",
            "native_gnn_required": True,
            "source_codec_checked_rows": parent_count,
            "source_codec_rdkit_valid_rows": parent_count,
            "source_codec_structure_match_rows": parent_count,
            "source_codec_invalid_valence_rows": 0,
            "source_codec_attribute_mapping_failed_rows": 0,
            "source_codec_passed": True,
            "source_codec_failure_examples": [],
            "atom_attribute_source": "source_anchored",
            "formal_charge_encoded_by_native": False,
            "source_atom_mapping_method": (
                "rdkit_atom_index_preserved_by_dense_builder"
            ),
            "source_atom_mapping_unique": True,
            "source_formal_charge_nonzero_atom_count": 0,
            "generation_uses_inference_mode": True,
            "generation_requires_gradients": False,
        }

    def generate(
        self,
        parents,
        *,
        output_dir,
        generation_chunk_size,
        start_parent_offset,
        on_training_ready,
        on_chunk,
        **_kwargs,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        if int(start_parent_offset) == 0:
            self.train_calls += 1
            self.rule_selection_calls += 1
            on_training_ready(self._summary(len(parents)))
        for parent_start in range(
            int(start_parent_offset),
            len(parents),
            int(generation_chunk_size),
        ):
            parent_end = min(
                len(parents),
                parent_start + int(generation_chunk_size),
            )
            records = []
            for parent in parents[parent_start:parent_end]:
                self.parent_order.append(parent.parent_id)
                records.append(
                    {
                        "source_parent_id": parent.parent_id,
                        "source_parent_smiles": parent.smiles,
                        "source_split": "train",
                        "raw_smiles": "C",
                        "generator_rank": 1,
                        "native_codec_decoded": True,
                        "native_conversion_ok": True,
                    }
                )
            chunk_index = parent_start // int(generation_chunk_size)
            on_chunk(chunk_index, parent_start, parent_end, records)
            self.chunk_ranges.append((parent_start, parent_end))
            raw_part = (
                output_dir.parent
                / "raw_generated_candidates.jsonl.part"
            )
            self.raw_part_row_counts.append(len(_read_jsonl(raw_part)))
            if (
                self.fail_after_total_chunks is not None
                and len(self.chunk_ranges) >= self.fail_after_total_chunks
            ):
                self.fail_after_total_chunks = None
                raise RuntimeError("intentional chunk interruption")
            del records
        return NativeGenerationResult([], self._summary(len(parents)))


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


def _native_graph_tensors(
    atom_labels: list[int],
    edges: list[tuple[int, int, int]],
    *,
    node_class_count: int,
    edge_class_count: int = 4,
    adjacency_value: float = 1.0,
):
    torch = pytest.importorskip("torch")
    size = len(atom_labels)
    features = torch.zeros((size, node_class_count), dtype=torch.float32)
    for index, label in enumerate(atom_labels):
        features[index, int(label)] = 1.0
    adjacency = torch.zeros((size, size), dtype=torch.float32)
    edge_attr = torch.zeros(
        (size * (size - 1) // 2, edge_class_count),
        dtype=torch.float32,
    )
    edge_attr[:, 0] = 1.0
    for left, right, edge_label in edges:
        adjacency[left, right] = adjacency_value
        adjacency[right, left] = adjacency_value
        high, low = max(left, right), min(left, right)
        position = (high - 1) * high // 2 + low
        edge_attr[position, 0] = 0.0
        edge_attr[position, edge_label] = 1.0
    return features, adjacency, edge_attr


def _atom_sidecar(
    atom_labels: list[int],
    metadata: GlobalGCECodecMetadata,
    *,
    formal_charges: dict[int, int] | None = None,
    explicit_hs: dict[int, int] | None = None,
) -> list[dict[str, object]]:
    periodic_table = Chem.GetPeriodicTable()
    charges = formal_charges or {}
    hydrogens = explicit_hs or {}
    rows = []
    for index, label in enumerate(atom_labels):
        if label <= 0:
            continue
        symbol = metadata.atom_symbols[label - 1]
        rows.append(
            {
                "native_node_index": index,
                "source_atom_index": index,
                "atomic_num": periodic_table.GetAtomicNumber(symbol),
                "formal_charge": charges.get(index, 0),
                "is_aromatic": False,
                "num_explicit_hs": hydrogens.get(index, 0),
                "isotope": 0,
                "chiral_tag": int(Chem.ChiralType.CHI_UNSPECIFIED),
                "no_implicit": bool(hydrogens.get(index, 0)),
                "attribute_source": "source_anchored",
            }
        )
    return rows


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


def _build_ten_parent_run(
    tmp_path: Path,
    *,
    parent_limit: int,
) -> dict[str, Path]:
    paths = {
        "train": tmp_path / "train_10.csv",
        "teacher": tmp_path / "teacher.pkl",
        "official": tmp_path / "official",
        "output": tmp_path / "output",
    }
    _write_ten_parent_train(paths["train"])
    paths["teacher"].write_bytes(b"fake-teacher")
    _write_official_root(paths["official"])
    build_mutagenicity_train_pool(
        train_csv=paths["train"],
        teacher_path=paths["teacher"],
        official_root=paths["official"],
        output_dir=paths["output"],
        teacher=_FakeTeacher(),
        generator=_LimitedCoverageGenerator(),
        config=PoolBuildConfig(
            parent_limit=parent_limit,
            expected_parent_count=10,
            device="cpu",
            epochs=1,
            top_k_native=2,
        ),
    )
    return paths


def _many_parent_paths(tmp_path: Path, count: int = 65) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": tmp_path / f"train_{count}.csv",
        "teacher": tmp_path / "teacher.pkl",
        "official": tmp_path / "official",
        "output": tmp_path / "output",
    }
    _write_many_parent_train(paths["train"], count)
    paths["teacher"].write_bytes(b"fake-teacher")
    _write_official_root(paths["official"])
    return paths


def _build_many_parent_run(
    paths: dict[str, Path],
    generator: _ChunkedGenerator,
    *,
    count: int = 65,
    chunk_size: int = 32,
) -> dict:
    return build_mutagenicity_train_pool(
        train_csv=paths["train"],
        teacher_path=paths["teacher"],
        official_root=paths["official"],
        output_dir=paths["output"],
        teacher=_FakeTeacher(),
        generator=generator,
        config=PoolBuildConfig(
            expected_parent_count=count,
            device="cpu",
            epochs=1,
            top_k_native=2,
            generation_chunk_size=chunk_size,
            generation_num_workers=0,
            memory_log_every_chunks=1,
        ),
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


def test_source_graph_round_trip_uses_native_metadata() -> None:
    torch = pytest.importorskip("torch")
    parents = [
        TrainParent("benzene", "c1ccccc1", 1, "train"),
        TrainParent("bromoethane", "CCBr", 1, "train"),
    ]
    dataset = _build_dense_dataset(
        parents,
        train_idx=[0],
        val_idx=[1],
        test_idx=[],
        torch_module=torch,
    )
    summary = probe_source_graph_codec(dataset, parents)

    assert summary["source_codec_checked_rows"] == 2
    assert summary["source_codec_rdkit_valid_rows"] == 2
    assert summary["source_codec_structure_match_rows"] == 2
    assert summary["source_codec_passed"] is True
    assert summary["source_codec_failure_examples"] == []
    node_mapping = summary["codec_metadata"]["node_label_mapping"]
    edge_mapping = summary["codec_metadata"]["edge_label_mapping"]
    assert node_mapping == {"0": "padding", "1": "C", "2": "Br"}
    assert edge_mapping == {
        "0": "no_edge",
        "1": "single",
        "2": "double",
        "3": "triple",
    }


@pytest.mark.parametrize(
    "smiles",
    [
        "[N+](=O)[O-]",
        "O=[N+]([O-])c1ccccc1",
        "Nc1ccc(O)c([N+](=O)[O-])c1",
    ],
)
def test_nitro_source_graph_round_trip_restores_formal_charge(
    smiles: str,
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    parents = [TrainParent("nitro", smiles, 1, "train")]
    dataset = _build_dense_dataset(
        parents,
        train_idx=[0],
        val_idx=[],
        test_idx=[],
        torch_module=torch,
    )
    audit_path = tmp_path / "source_atom_attribute_audit.jsonl"
    summary = probe_source_graph_codec(
        dataset,
        parents,
        atom_attribute_audit_path=audit_path,
    )

    assert summary["formal_charge_encoded_by_native"] is False
    assert summary["atom_attribute_source"] == "source_anchored"
    assert summary["source_atom_mapping_unique"] is True
    assert summary["source_codec_invalid_valence_rows"] == 0
    assert summary["source_codec_attribute_mapping_failed_rows"] == 0
    assert summary["source_codec_structure_match_rows"] == 1
    assert summary["source_codec_passed"] is True
    charges = {
        int(row["formal_charge"])
        for row in map(json.loads, audit_path.read_text().splitlines())
    }
    assert charges == {-1, 1}


def test_source_sidecar_preserves_charged_atoms_and_neutral_amine() -> None:
    torch = pytest.importorskip("torch")
    parents = [
        TrainParent("nitro", "O=[N+]([O-])c1ccccc1", 1, "train"),
        TrainParent("amine", "CCN", 1, "train"),
    ]
    dataset = _build_dense_dataset(
        parents,
        train_idx=[0],
        val_idx=[1],
        test_idx=[],
        torch_module=torch,
    )
    nitro_sidecar = dataset.source_atom_attributes[0]
    amine_sidecar = dataset.source_atom_attributes[1]

    assert any(
        row["atomic_num"] == 7 and row["formal_charge"] == 1
        for row in nitro_sidecar
    )
    assert any(
        row["atomic_num"] == 8 and row["formal_charge"] == -1
        for row in nitro_sidecar
    )
    assert all(
        row["formal_charge"] == 0
        for row in amine_sidecar
        if row["atomic_num"] == 7
    )


def test_source_atom_mapping_ambiguity_is_rejected() -> None:
    metadata = GlobalGCECodecMetadata(
        atom_symbols=("C",),
        bond_names=("no_edge", "single", "double", "triple"),
    )
    tensors = _native_graph_tensors(
        [1, 1],
        [(0, 1, 1)],
        node_class_count=2,
    )
    ambiguous = _atom_sidecar([1, 1], metadata)
    ambiguous[1]["source_atom_index"] = 0
    result = decode_globalgce_molecule(
        *tensors,
        metadata=metadata,
        graph_role="source",
        source_atom_attributes=ambiguous,
    )

    assert result.ok is False
    assert result.error_type == "source_codec_attribute_mapping_failed"
    assert "not unique" in str(result.error_message)


def test_generated_same_atom_type_inherits_source_charge() -> None:
    torch = pytest.importorskip("torch")
    parents = [
        TrainParent("nitro", "O=[N+]([O-])c1ccccc1", 1, "train")
    ]
    dataset = _build_dense_dataset(
        parents,
        train_idx=[0],
        val_idx=[],
        test_idx=[],
        torch_module=torch,
    )
    graph = dataset[0]
    result = decode_globalgce_molecule(
        graph["feature"],
        graph["adj"],
        graph["edge_attr"],
        metadata=GlobalGCECodecMetadata.from_dataset(dataset),
        num_nodes=int(graph["num_nodes"]),
        graph_role="generated",
        source_atom_attributes=dataset.source_atom_attributes[0],
    )

    assert result.ok
    assert sorted(
        atom.GetFormalCharge()
        for atom in result.mol.GetAtoms()
        if atom.GetFormalCharge()
    ) == [-1, 1]


def test_generated_atom_type_change_does_not_inherit_charge() -> None:
    torch = pytest.importorskip("torch")
    parents = [
        TrainParent("nitro", "O=[N+]([O-])c1ccccc1", 1, "train")
    ]
    dataset = _build_dense_dataset(
        parents,
        train_idx=[0],
        val_idx=[],
        test_idx=[],
        torch_module=torch,
    )
    graph = dataset[0]
    metadata = GlobalGCECodecMetadata.from_dataset(dataset)
    feature = graph["feature"].clone()
    nitrogen_index = next(
        int(row["native_node_index"])
        for row in dataset.source_atom_attributes[0]
        if row["formal_charge"] == 1
    )
    carbon_label = metadata.atom_symbols.index("C") + 1
    feature[nitrogen_index].zero_()
    feature[nitrogen_index, carbon_label] = 1.0
    result = decode_globalgce_molecule(
        feature,
        graph["adj"],
        graph["edge_attr"],
        metadata=metadata,
        num_nodes=int(graph["num_nodes"]),
        graph_role="generated",
        source_atom_attributes=dataset.source_atom_attributes[0],
    )

    assert result.ok is False
    assert result.error_type == "generated_attribute_ambiguous"
    changed = result.graph_record["atom_attribute_audit"][-1]
    assert changed["attributes_inherited"] is False
    assert changed["formal_charge"] is None
    assert (
        changed["ambiguity_reason"]
        == "generated_atom_type_differs_from_source"
    )


def test_generated_new_node_without_source_identity_is_ambiguous() -> None:
    metadata = GlobalGCECodecMetadata(
        atom_symbols=("C",),
        bond_names=("no_edge", "single", "double", "triple"),
    )
    tensors = _native_graph_tensors(
        [1, 1, 1],
        [(0, 1, 1), (1, 2, 1)],
        node_class_count=2,
    )
    result = decode_globalgce_molecule(
        *tensors,
        metadata=metadata,
        graph_role="generated",
        source_atom_attributes=_atom_sidecar([1, 1], metadata),
    )

    assert result.ok is False
    assert result.error_type == "generated_attribute_ambiguous"
    assert (
        result.graph_record["atom_attribute_audit"][-1][
            "ambiguity_reason"
        ]
        == "new_native_node_without_source_identity"
    )


def test_source_and_generated_valence_errors_use_distinct_types() -> None:
    metadata = GlobalGCECodecMetadata(
        atom_symbols=("C", "Br"),
        bond_names=("no_edge", "single", "double", "triple"),
    )
    labels = [2, 1, 1, 1, 1, 1, 1]
    tensors = _native_graph_tensors(
        labels,
        [(0, index, 1) for index in range(1, 7)],
        node_class_count=3,
    )
    sidecar = _atom_sidecar(labels, metadata)
    source_result = decode_globalgce_molecule(
        *tensors,
        metadata=metadata,
        graph_role="source",
        source_atom_attributes=sidecar,
    )
    generated_result = decode_globalgce_molecule(
        *tensors,
        metadata=metadata,
        graph_role="generated",
        source_atom_attributes=sidecar,
    )

    assert source_result.error_type == "source_codec_invalid_valence"
    assert generated_result.error_type == "generated_invalid_valence"
    assert all(
        row["formal_charge"] == 0
        for row in generated_result.graph_record["atom_attribute_audit"]
    )


def test_adjacency_is_only_edge_mask_and_edge_class_controls_bond_type() -> None:
    metadata = GlobalGCECodecMetadata(
        atom_symbols=("C",),
        bond_names=("no_edge", "single", "double", "triple"),
    )
    single = _native_graph_tensors(
        [1, 1],
        [(0, 1, 1)],
        node_class_count=2,
        adjacency_value=9.0,
    )
    double = _native_graph_tensors(
        [1, 1],
        [(0, 1, 2)],
        node_class_count=2,
        adjacency_value=9.0,
    )
    sidecar = _atom_sidecar([1, 1], metadata)
    single_result = decode_globalgce_molecule(
        *single,
        metadata=metadata,
        graph_role="generated",
        source_atom_attributes=sidecar,
    )
    double_result = decode_globalgce_molecule(
        *double,
        metadata=metadata,
        graph_role="generated",
        source_atom_attributes=sidecar,
    )

    assert single_result.ok and double_result.ok
    assert single_result.mol.GetBondWithIdx(0).GetBondType() == Chem.BondType.SINGLE
    assert double_result.mol.GetBondWithIdx(0).GetBondType() == Chem.BondType.DOUBLE


def test_padding_self_loops_and_undirected_edges_are_handled_once() -> None:
    metadata = GlobalGCECodecMetadata(
        atom_symbols=("C",),
        bond_names=("no_edge", "single", "double", "triple"),
    )
    feature, adjacency, edge_attr = _native_graph_tensors(
        [1, 1, 0],
        [(0, 1, 1)],
        node_class_count=2,
    )
    adjacency[0, 0] = 1.0
    record = globalgce_tensors_to_graph_record(
        feature,
        adjacency,
        edge_attr,
        metadata=metadata,
    )
    result = decode_globalgce_molecule(
        feature,
        adjacency,
        edge_attr,
        metadata=metadata,
        graph_role="generated",
        source_atom_attributes=_atom_sidecar([1, 1, 0], metadata),
    )

    assert record["padding_node_indices"] == [2]
    assert record["self_loop_count_ignored"] == 1
    assert result.ok
    assert result.num_atoms == 2
    assert result.num_bonds == 1


def test_duplicate_bond_is_rejected_explicitly() -> None:
    molecule = Chem.RWMol()
    molecule.AddAtom(Chem.Atom("C"))
    molecule.AddAtom(Chem.Atom("C"))
    seen: set[tuple[int, int]] = set()
    _add_bond_once(molecule, 0, 1, Chem.BondType.SINGLE, seen)
    with pytest.raises(GlobalGCEMutagenicityCodecError, match="Duplicate"):
        _add_bond_once(molecule, 1, 0, Chem.BondType.SINGLE, seen)


def test_aromatic_bond_metadata_sets_consistent_aromatic_atoms() -> None:
    metadata = GlobalGCECodecMetadata(
        atom_symbols=("C",),
        bond_names=("no_edge", "single", "double", "triple", "aromatic"),
    )
    edges = [
        (0, 1, 4),
        (1, 2, 4),
        (2, 3, 4),
        (3, 4, 4),
        (4, 5, 4),
        (5, 0, 4),
    ]
    tensors = _native_graph_tensors(
        [1] * 6,
        edges,
        node_class_count=2,
        edge_class_count=5,
    )
    result = decode_globalgce_molecule(
        *tensors,
        metadata=metadata,
        graph_role="generated",
        source_atom_attributes=_atom_sidecar([1] * 6, metadata),
    )

    assert result.ok
    assert result.smiles == "c1ccccc1"
    assert all(atom.GetIsAromatic() for atom in result.mol.GetAtoms())
    assert all(bond.GetIsAromatic() for bond in result.mol.GetBonds())


def test_bromine_single_bond_is_valid_but_overvalent_bromine_is_not_repaired() -> None:
    metadata = GlobalGCECodecMetadata(
        atom_symbols=("C", "Br"),
        bond_names=("no_edge", "single", "double", "triple"),
    )
    valid = _native_graph_tensors(
        [1, 2],
        [(0, 1, 1)],
        node_class_count=3,
    )
    valid_result = decode_globalgce_molecule(
        *valid,
        metadata=metadata,
        graph_role="generated",
        source_atom_attributes=_atom_sidecar([1, 2], metadata),
    )
    assert valid_result.ok
    assert valid_result.smiles == "CBr"

    invalid = _native_graph_tensors(
        [2, 1, 1, 1, 1, 1, 1],
        [(0, index, 1) for index in range(1, 7)],
        node_class_count=3,
    )
    invalid_result = decode_globalgce_molecule(
        *invalid,
        metadata=metadata,
        graph_role="generated",
        source_atom_attributes=_atom_sidecar(
            [2, 1, 1, 1, 1, 1, 1],
            metadata,
        ),
    )
    assert invalid_result.ok is False
    assert invalid_result.codec_decoded is True
    assert invalid_result.error_type == "generated_invalid_valence"
    assert invalid_result.smiles is None
    assert invalid_result.num_bonds == 6


def test_source_codec_gate_hard_fails_on_round_trip_mismatch() -> None:
    summary = {
        "source_codec_passed": False,
        "source_codec_failure_examples": [{"parent_id": "p1"}],
    }
    with pytest.raises(
        GlobalGCEMutagenicityCodecError,
        match="before training",
    ):
        require_source_codec_gate(summary)


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


def test_audit_distinguishes_input_train_from_selected_cohort(
    tmp_path: Path,
) -> None:
    paths = _build_ten_parent_run(tmp_path, parent_limit=3)

    all_parents, selected_parents = load_strict_train_parents(
        paths["train"],
        parent_limit=3,
        expected_parent_count=10,
    )
    assert len(all_parents) == 10
    assert [parent.parent_id for parent in selected_parents] == [
        "p00",
        "p01",
        "p02",
    ]

    audit = audit_mutagenicity_train_pool(
        paths["output"],
        train_csv=paths["train"],
        expected_parent_count=3,
        expected_input_train_count=10,
    )

    assert audit["input_train_rows"] == 10
    assert audit["expected_input_train_rows"] == 10
    assert audit["selected_train_rows"] == 3
    assert audit["expected_selected_parent_rows"] == 3
    assert audit["selected_cohort_hash_matches"] is True
    assert audit["candidate_source_parent_rows"] == 2
    assert audit["candidate_source_parent_subset_of_selected"] is True
    assert audit["source_parent_coverage_recomputed"] == pytest.approx(2 / 3)
    assert audit["parent_limit"] == 3
    assert (
        audit["deterministic_parent_selection_method"]
        == "parent_id_ascending_prefix_v1"
    )


def test_audit_rejects_wrong_complete_input_train_count(
    tmp_path: Path,
) -> None:
    paths = _build_ten_parent_run(tmp_path, parent_limit=3)

    with pytest.raises(
        ValueError,
        match=r"Strict train row count mismatch: expected=9, found=10",
    ):
        audit_mutagenicity_train_pool(
            paths["output"],
            train_csv=paths["train"],
            expected_parent_count=3,
            expected_input_train_count=9,
        )


def test_audit_rejects_selected_cohort_hash_mismatch(
    tmp_path: Path,
) -> None:
    paths = _build_ten_parent_run(tmp_path, parent_limit=3)
    summary_path = paths["output"] / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["selected_train_cohort_hash"] = "incorrect-hash"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="Selected train cohort hash mismatch"):
        audit_mutagenicity_train_pool(
            paths["output"],
            train_csv=paths["train"],
            expected_parent_count=3,
            expected_input_train_count=10,
        )


def test_audit_rejects_candidate_parent_outside_selected_cohort(
    tmp_path: Path,
) -> None:
    paths = _build_ten_parent_run(tmp_path, parent_limit=3)
    pool_path = paths["output"] / "candidate_pool.jsonl"
    rows = _read_jsonl(pool_path)
    rows[0]["source_parent_id"] = "p09"
    _write_jsonl(pool_path, rows)

    with pytest.raises(
        AssertionError,
        match="outside the deterministically selected cohort",
    ):
        audit_mutagenicity_train_pool(
            paths["output"],
            train_csv=paths["train"],
            expected_parent_count=3,
            expected_input_train_count=10,
        )


def test_audit_rejects_candidate_parent_outside_complete_train(
    tmp_path: Path,
) -> None:
    paths = _build_ten_parent_run(tmp_path, parent_limit=3)
    pool_path = paths["output"] / "candidate_pool.jsonl"
    rows = _read_jsonl(pool_path)
    rows[0]["source_parent_id"] = "not-in-train"
    _write_jsonl(pool_path, rows)

    with pytest.raises(AssertionError, match="non-train parent"):
        audit_mutagenicity_train_pool(
            paths["output"],
            train_csv=paths["train"],
            expected_parent_count=3,
            expected_input_train_count=10,
        )


def test_audit_full_mode_uses_complete_train_as_selected_cohort(
    tmp_path: Path,
) -> None:
    paths = _build_ten_parent_run(tmp_path, parent_limit=0)

    audit = audit_mutagenicity_train_pool(
        paths["output"],
        train_csv=paths["train"],
        expected_parent_count=10,
        expected_input_train_count=10,
    )

    assert audit["input_train_rows"] == 10
    assert audit["selected_train_rows"] == 10
    assert audit["candidate_source_parent_rows"] == 2
    assert audit["source_parent_coverage_recomputed"] == pytest.approx(0.2)


def test_chunked_generation_uses_three_chunks_and_preserves_parent_order(
    tmp_path: Path,
) -> None:
    paths = _many_parent_paths(tmp_path)
    generator = _ChunkedGenerator()

    summary = _build_many_parent_run(paths, generator)

    assert generator.chunk_ranges == [(0, 32), (32, 64), (64, 65)]
    assert generator.parent_order == [
        f"p{index:03d}" for index in range(65)
    ]
    assert generator.raw_part_row_counts == [32, 64, 65]
    assert generator.train_calls == 1
    assert generator.rule_selection_calls == 1
    assert summary["generation_completed_chunks"] == 3
    assert summary["generation_num_workers"] == 0
    assert len(_read_jsonl(paths["output"] / "candidate_pool.jsonl")) == 65
    assert not (
        paths["output"] / "raw_generated_candidates.jsonl.part"
    ).exists()
    assert not (paths["output"] / "invalid_candidates.jsonl.part").exists()
    assert not (
        paths["output"] / "non_target_candidates.jsonl.part"
    ).exists()


def test_chunked_and_single_chunk_outputs_are_identical(
    tmp_path: Path,
) -> None:
    chunked_paths = _many_parent_paths(tmp_path / "chunked")
    single_paths = _many_parent_paths(tmp_path / "single")

    _build_many_parent_run(
        chunked_paths,
        _ChunkedGenerator(),
        chunk_size=32,
    )
    _build_many_parent_run(
        single_paths,
        _ChunkedGenerator(),
        chunk_size=65,
    )

    for filename in (
        "raw_generated_candidates.jsonl",
        "candidate_pool.jsonl",
        "candidate_universe.jsonl",
        "invalid_candidates.jsonl",
        "non_target_candidates.jsonl",
    ):
        assert _read_jsonl(
            chunked_paths["output"] / filename
        ) == _read_jsonl(single_paths["output"] / filename)


def test_chunk_resume_continues_without_duplicate_rows_or_retraining(
    tmp_path: Path,
) -> None:
    paths = _many_parent_paths(tmp_path)
    generator = _ChunkedGenerator(fail_after_total_chunks=1)

    with pytest.raises(RuntimeError, match="intentional chunk interruption"):
        _build_many_parent_run(paths, generator)

    output = paths["output"]
    generation_checkpoint = json.loads(
        (output / "generation_resume_checkpoint.json").read_text(
            encoding="utf-8"
        )
    )
    assert generation_checkpoint["next_parent_offset"] == 32
    assert generation_checkpoint["completed_chunk_count"] == 1
    assert len(
        _read_jsonl(output / "raw_generated_candidates.jsonl.part")
    ) == 32
    assert not (output / "_RUN_COMPLETE.json").exists()
    interrupted_rows = _read_jsonl(
        output / "raw_generated_candidates.jsonl.part"
    )
    with (
        output / "raw_generated_candidates.jsonl.part"
    ).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(interrupted_rows[-1]) + "\n")

    summary = _build_many_parent_run(paths, generator)

    raw_rows = _read_jsonl(output / "raw_generated_candidates.jsonl")
    assert len(raw_rows) == 65
    assert len({row["source_parent_id"] for row in raw_rows}) == 65
    assert generator.chunk_ranges == [(0, 32), (32, 64), (64, 65)]
    assert generator.train_calls == 1
    assert generator.rule_selection_calls == 1
    assert summary["raw_generated_rows"] == 65
    assert (output / "_RUN_COMPLETE.json").is_file()


def test_chunk_resume_rejects_generation_config_mismatch(
    tmp_path: Path,
) -> None:
    paths = _many_parent_paths(tmp_path)
    generator = _ChunkedGenerator(fail_after_total_chunks=1)
    with pytest.raises(RuntimeError, match="intentional chunk interruption"):
        _build_many_parent_run(paths, generator, chunk_size=32)

    with pytest.raises(ValueError, match="Resume configuration mismatch"):
        _build_many_parent_run(paths, generator, chunk_size=16)


def test_memory_log_contains_rss_and_cuda_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = log_globalgce_phase_memory(
        phase="unit_test",
        chunk_index=2,
        processed_parent_count=65,
        raw_generated_count=65,
    )

    output = capsys.readouterr().out
    assert "[GLOBALGCE_MEMORY]" in output
    assert "rss_gb=" in output
    assert payload["rss_gb"] > 0
    assert payload["cuda_allocated_gb"] == 0.0
    assert payload["cuda_reserved_gb"] == 0.0


def test_generation_worker_default_is_zero() -> None:
    assert PoolBuildConfig().generation_num_workers == 0
    assert PoolBuildConfig().generation_chunk_size == 32
    assert PoolBuildConfig().memory_log_every_chunks == 1


def test_official_generation_releases_dense_chunk_outputs() -> None:
    source = inspect.getsource(
        OfficialGlobalGCEMutagenicityGenerator.generate
    )
    assert "with torch.inference_mode()" in source
    assert "del cf_feat, cf_adj, cf_edge, graph_idx" in source
    assert "del chunk_loader, chunk_dataset" in source
    assert "if on_chunk is not None:" in source
    assert "all_records.extend(chunk_records)" in source


def test_real_generator_resume_completion_signature_matches_protocol() -> None:
    concrete = inspect.signature(OfficialGlobalGCEMutagenicityGenerator.generate)
    protocol = inspect.signature(NativeGeneratorProtocol.generate)
    for name in (
        "expected_resume_checkpoint",
        "on_resume_checkpoint",
        "after_epoch_checkpoint",
        "on_generation_complete",
    ):
        assert name in concrete.parameters
        assert concrete.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert concrete.parameters[name].default is None
        assert name in protocol.parameters
        assert protocol.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert protocol.parameters[name].default is None
    source = inspect.getsource(OfficialGlobalGCEMutagenicityGenerator.generate)
    assert source.count("on_generation_complete()") == 1
    # One local definition plus the successful rules-only and generation paths.
    assert source.count("_notify_generation_complete()") == 3
    assert "if rules_only:" in source
    assert "except TypeError" not in source
    assert "**kwargs" not in source


def _exact_official_api_modules() -> dict[str, ModuleType]:
    class GTGNN:
        def __init__(
            self, x_dim, h_dim, n_out, num_edge_attr, device, save_model_path
        ):
            pass

    class GlobalGCE:
        def __init__(
            self,
            x_dim,
            h_dim,
            z_dim,
            edge_attr_dim,
            dropout,
            fs_min_nodes,
            fs_max_nodes,
            topk_fs,
            random_subgraph,
            save_fs_path,
            device,
            gt_gnn,
        ):
            pass

        def get_fs_expanded_data(
            self, train_loader, crop_expansion=False
        ):
            pass

        def get_rules(self, fss):
            pass

        def run_one_batch(self, rules, data):
            pass

    def generate_cfs(dataloader, rules, device):
        pass

    def concate_inputs_with_local_recourse(
        features,
        adj,
        edge_attrs,
        fs_index_list,
        mask_idx_list,
        rules,
        device,
    ):
        pass

    class FrequentSubgraphGenerator:
        def __init__(
            self,
            fs_min_vertices,
            fs_max_vertices,
            save_fs_path,
            topk,
            random_subgraph=False,
        ):
            pass

    class gSpan:
        def __init__(
            self,
            nx_graph_list,
            min_support=10,
            min_num_vertices=1,
            max_num_vertices=float("inf"),
            max_ngraphs=float("inf"),
            is_undirected=True,
            verbose=False,
            visualize=False,
            where=False,
        ):
            pass

    def train_globalgce(
        epochs,
        pred_model,
        model,
        lr,
        train_loader,
        val_loader,
        save_rule_path,
        save_model_path,
    ):
        pass

    def test_globalgce(data_loader, model, gnn_model, rules):
        pass

    gtgnn = ModuleType("models.GTGNN")
    gtgnn.GTGNN = GTGNN
    globalgce = ModuleType("models.GlobalGCE")
    globalgce.GlobalGCE = GlobalGCE
    globalgce.generate_cfs = generate_cfs
    globalgce.concate_inputs_with_local_recourse = (
        concate_inputs_with_local_recourse
    )
    fsg = ModuleType("models.fsg")
    fsg.FrequentSubgraphGenerator = FrequentSubgraphGenerator
    gspan = ModuleType("models.gSpan.gSpan")
    gspan.gSpan = gSpan
    models_utils = ModuleType("models.models_utils")
    models_utils.train_globalgce = train_globalgce
    models_utils.test_globalgce = test_globalgce
    return {
        "gtgnn_module": gtgnn,
        "globalgce_module": globalgce,
        "fsg_module": fsg,
        "gspan_module": gspan,
        "models_utils_module": models_utils,
    }


def test_official_api_signature_is_exact_and_variadics_fail_closed() -> None:
    modules = _exact_official_api_modules()
    evidence = validate_official_globalgce_api_signatures(modules)
    assert evidence["schema_version"] == OFFICIAL_GLOBALGCE_API_SIGNATURE_SCHEMA
    assert evidence["signatures"] == PINNED_OFFICIAL_GLOBALGCE_API_SIGNATURES

    def variadic_generate_cfs(*args, **kwargs):
        raise AssertionError((args, kwargs))

    modules["globalgce_module"].generate_cfs = variadic_generate_cfs
    with pytest.raises(RuntimeError, match="BLOCKED_OFFICIAL_API_SIGNATURE_DRIFT"):
        validate_official_globalgce_api_signatures(modules)


@pytest.mark.parametrize(
    "field",
    (
        "on_resume_checkpoint",
        "after_epoch_checkpoint",
        "on_generation_complete",
    ),
)
def test_real_generator_rejects_noncallable_resume_callbacks_before_import(
    tmp_path: Path,
    field: str,
) -> None:
    official = tmp_path / "official"
    _write_official_root(official)
    generator = OfficialGlobalGCEMutagenicityGenerator(official)
    kwargs = {
        "output_dir": tmp_path / "output",
        "seed": 7,
        "epochs": 1,
        "top_k_native": 20,
        "learning_rate": 0.1,
        "dropout": 0.5,
        "device": "cpu",
        "resume": True,
        field: object(),
    }
    with pytest.raises(TypeError, match=field):
        generator.generate([], **kwargs)


def test_real_generator_rejects_malformed_expected_checkpoint_before_import(
    tmp_path: Path,
) -> None:
    official = tmp_path / "official"
    _write_official_root(official)
    generator = OfficialGlobalGCEMutagenicityGenerator(official)
    with pytest.raises(ValueError, match="checkpoint-file evidence"):
        generator.generate(
            [],
            output_dir=tmp_path / "output",
            seed=7,
            epochs=1,
            top_k_native=20,
            learning_rate=0.1,
            dropout=0.5,
            device="cpu",
            resume=True,
            expected_resume_checkpoint={"sha256": "a" * 64},
        )


def _write_importable_official_modules(root: Path) -> dict[str, dict]:
    payloads = {
        "models/__init__.py": "# exact official package\n",
        "models/models_utils.py": (
            "def train_globalgce(*args, **kwargs): return None\n"
            "def test_globalgce(*args, **kwargs): return None\n"
        ),
        "models/GTGNN.py": "class GTGNN: pass\n",
        "models/GlobalGCE.py": (
            "class GlobalGCE: pass\n"
            "def generate_cfs(*args, **kwargs): return []\n"
        ),
        "models/fsg.py": "MIN_FREQ = {}\n",
        "models/gSpan/__init__.py": "# exact gSpan package\n",
        "models/gSpan/gSpan.py": "class gSpan: pass\n",
    }
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    return {
        relative: _hash_open_regular_source(root / relative)
        for relative in payloads
    }


def _displace_model_module_closure() -> dict[str, ModuleType]:
    displaced = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "models"
        or name.startswith("models.")
        or name == "data"
        or name.startswith("data.")
        or name == "utils"
    }
    for name in displaced:
        sys.modules.pop(name, None)
    return displaced


def _restore_model_module_closure(displaced: dict[str, ModuleType]) -> None:
    for name in tuple(sys.modules):
        if (
            name == "models"
            or name.startswith("models.")
            or name == "data"
            or name.startswith("data.")
            or name == "utils"
        ):
            sys.modules.pop(name, None)
    sys.modules.update(displaced)


def test_official_import_rejects_preloaded_models_outside_held_authority(
    tmp_path: Path,
) -> None:
    official = tmp_path / "official" / "src"
    authority = _write_importable_official_modules(official)
    displaced = _displace_model_module_closure()
    try:
        attacker = tmp_path / "attacker.py"
        attacker.write_text("# attacker\n", encoding="utf-8")
        sentinel = ModuleType("models")
        sentinel.__file__ = str(attacker)
        sys.modules["models"] = sentinel
        with pytest.raises(RuntimeError, match="outside the held checkout"):
            _import_official_modules(
                official,
                expected_source_authority=authority,
            )
    finally:
        _restore_model_module_closure(displaced)


def test_official_import_rejects_preloaded_utils_outside_held_authority(
    tmp_path: Path,
) -> None:
    official = tmp_path / "official" / "src"
    authority = _write_importable_official_modules(official)
    displaced = _displace_model_module_closure()
    try:
        attacker = tmp_path / "utils.py"
        attacker.write_text("# attacker utils\n", encoding="utf-8")
        sentinel = ModuleType("utils")
        sentinel.__file__ = str(attacker)
        sys.modules["utils"] = sentinel
        with pytest.raises(RuntimeError, match="outside the held checkout"):
            _import_official_modules(
                official,
                expected_source_authority=authority,
            )
    finally:
        _restore_model_module_closure(displaced)


def test_module_provenance_requires_isolated_python_before_import(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="BLOCKED_PYTHON_MODULE_PROVENANCE"):
        capture_globalgce_module_provenance(
            official_src=tmp_path,
            expected_source_authority=None,
            require_isolated=True,
        )


def test_official_import_reloads_valid_prior_closure_and_rejects_pyc(
    tmp_path: Path,
) -> None:
    official = tmp_path / "official" / "src"
    authority = _write_importable_official_modules(official)
    displaced = _displace_model_module_closure()
    try:
        first = _import_official_modules(
            official,
            expected_source_authority=authority,
        )
        prior_globalgce = sys.modules["models.GlobalGCE"]
        second = _import_official_modules(
            official,
            expected_source_authority=authority,
        )
        assert first["GlobalGCE"] is not second["GlobalGCE"]
        assert sys.modules["models.GlobalGCE"] is not prior_globalgce

        cache = official / "models" / "__pycache__"
        cache.mkdir()
        (cache / "GlobalGCE.cpython-311.pyc").write_bytes(b"malicious")
        with pytest.raises(RuntimeError, match="__pycache__"):
            _import_official_modules(
                official,
                expected_source_authority=authority,
            )
    finally:
        _restore_model_module_closure(displaced)


def test_official_import_rejects_same_byte_source_inode_replacement(
    tmp_path: Path,
) -> None:
    official = tmp_path / "official" / "src"
    authority = _write_importable_official_modules(official)
    displaced = _displace_model_module_closure()
    try:
        source = official / "models/GlobalGCE.py"
        replacement = source.with_name(".GlobalGCE.py.replacement")
        replacement.write_bytes(source.read_bytes())
        replacement.replace(source)
        with pytest.raises(RuntimeError, match="named source differs"):
            _import_official_modules(
                official,
                expected_source_authority=authority,
            )
    finally:
        _restore_model_module_closure(displaced)


def test_generated_attribute_ambiguity_is_excluded_from_candidate_pool(
    fixture_paths: dict[str, Path],
) -> None:
    class AmbiguousGenerator:
        def generate(self, parents, **_kwargs):
            return NativeGenerationResult(
                [
                    {
                        "source_parent_id": parents[0].parent_id,
                        "source_parent_smiles": parents[0].smiles,
                        "source_split": "train",
                        "raw_smiles": None,
                        "generator_rank": 1,
                        "native_codec_decoded": False,
                        "native_conversion_ok": False,
                        "native_conversion_error_type": (
                            "generated_attribute_ambiguous"
                        ),
                        "native_conversion_error": (
                            "new node has no source identity"
                        ),
                    },
                    {
                        "source_parent_id": parents[0].parent_id,
                        "source_parent_smiles": parents[0].smiles,
                        "source_split": "train",
                        "raw_smiles": "CC",
                        "generator_rank": 2,
                        "native_codec_decoded": True,
                        "native_conversion_ok": True,
                    },
                ],
                {
                    "source_codec_checked_rows": len(parents),
                    "source_codec_rdkit_valid_rows": len(parents),
                    "source_codec_structure_match_rows": len(parents),
                    "source_codec_invalid_valence_rows": 0,
                    "source_codec_attribute_mapping_failed_rows": 0,
                    "source_codec_passed": True,
                    "source_codec_failure_examples": [],
                    "atom_attribute_source": "source_anchored",
                    "formal_charge_encoded_by_native": False,
                    "source_atom_mapping_method": (
                        "rdkit_atom_index_preserved_by_dense_builder"
                    ),
                    "source_atom_mapping_unique": True,
                    "source_formal_charge_nonzero_atom_count": 0,
                },
            )

    summary = build_mutagenicity_train_pool(
        train_csv=fixture_paths["train"],
        teacher_path=fixture_paths["teacher"],
        official_root=fixture_paths["official"],
        output_dir=fixture_paths["output"],
        teacher=_FakeTeacher(),
        generator=AmbiguousGenerator(),
        config=PoolBuildConfig(
            expected_parent_count=3,
            device="cpu",
            epochs=1,
            top_k_native=2,
        ),
    )
    invalid = _read_jsonl(
        fixture_paths["output"] / "invalid_candidates.jsonl"
    )
    pool = _read_jsonl(fixture_paths["output"] / "candidate_pool.jsonl")

    assert summary["generated_attribute_ambiguous_rows"] == 1
    assert len(invalid) == 1
    assert invalid[0]["invalid_reason"] == "generated_attribute_ambiguous"
    assert len(pool) == 1
    assert pool[0]["canonical_smiles"] == "CC"


def test_empty_generated_universe_is_incomplete_and_has_no_completion_marker(
    fixture_paths: dict[str, Path],
) -> None:
    class AllInvalidGenerator:
        def generate(self, parents, **_kwargs):
            return NativeGenerationResult(
                [
                    {
                        "source_parent_id": parents[0].parent_id,
                        "source_parent_smiles": parents[0].smiles,
                        "source_split": "train",
                        "raw_smiles": None,
                        "generator_rank": 1,
                        "native_codec_decoded": True,
                        "native_conversion_ok": False,
                        "native_conversion_error_type": (
                            "generated_invalid_valence"
                        ),
                        "native_conversion_error": "explicit valence failure",
                    }
                ],
                {
                    "source_codec_checked_rows": 3,
                    "source_codec_rdkit_valid_rows": 3,
                    "source_codec_structure_match_rows": 3,
                    "source_codec_passed": True,
                    "source_codec_failure_examples": [],
                },
            )

    with pytest.raises(
        GlobalGCEEmptyCandidateUniverseError,
        match=r"source_codec_passed=True.*generated_invalid_valence_rows=1",
    ):
        build_mutagenicity_train_pool(
            train_csv=fixture_paths["train"],
            teacher_path=fixture_paths["teacher"],
            official_root=fixture_paths["official"],
            output_dir=fixture_paths["output"],
            teacher=_FakeTeacher(),
            generator=AllInvalidGenerator(),
            config=PoolBuildConfig(
                expected_parent_count=3,
                device="cpu",
                epochs=1,
                top_k_native=2,
            ),
        )

    summary = json.loads(
        (fixture_paths["output"] / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["raw_generated_rows"] == 1
    assert summary["generated_graph_rows"] == 1
    assert summary["generated_codec_decoded_rows"] == 1
    assert summary["generated_rdkit_valid_rows"] == 0
    assert summary["generated_invalid_valence_rows"] == 1
    assert summary["generated_invalid_other_rows"] == 0
    assert summary["candidate_pool_rows"] == 0
    assert summary["canonical_unique_candidates"] == 0
    assert summary["source_codec_passed"] is True
    assert summary["run_complete"] is False
    assert not (fixture_paths["output"] / "_RUN_COMPLETE.json").exists()


def test_codec_probe_only_does_not_construct_teacher_or_call_training(
    fixture_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.baselines.globalgce import (
        build_mutagenicity_train_pool as build_cli,
    )

    calls = {"probe": 0, "train": 0}

    class ProbeGenerator:
        def __init__(self, _official_root, *, native_train_csv=None):
            assert native_train_csv == fixture_paths["train"].resolve()

        def probe_codec(self, parents, *, seed, output_path):
            calls["probe"] += 1
            assert [parent.parent_id for parent in parents] == ["p1", "p2"]
            assert seed == 13
            payload = {
                "source_codec_checked_rows": 2,
                "source_codec_rdkit_valid_rows": 2,
                "source_codec_structure_match_rows": 2,
                "source_codec_passed": True,
                "source_codec_failure_examples": [],
                "calibration_loaded": False,
                "test_loaded": False,
            }
            Path(output_path).write_text(json.dumps(payload), encoding="utf-8")
            return payload

    def forbidden_teacher(*_args, **_kwargs):
        raise AssertionError("Codec probe must not load the RF teacher.")

    def forbidden_train(*_args, **_kwargs):
        calls["train"] += 1
        raise AssertionError("Codec probe must not invoke training.")

    monkeypatch.setattr(
        build_cli,
        "OfficialGlobalGCEMutagenicityGenerator",
        ProbeGenerator,
    )
    monkeypatch.setattr(build_cli, "TeacherSemanticScorer", forbidden_teacher)
    monkeypatch.setattr(
        build_cli,
        "build_mutagenicity_train_pool",
        forbidden_train,
    )
    exit_code = build_cli.main(
        [
            "--train-csv",
            str(fixture_paths["train"]),
            "--teacher-path",
            str(
                fixture_paths["output"].parent
                / "teacher-must-not-be-read.pkl"
            ),
            "--official-root",
            str(fixture_paths["official"]),
            "--native-train-csv",
            str(fixture_paths["train"]),
            "--output-dir",
            str(fixture_paths["output"]),
            "--expected-parent-count",
            "3",
            "--parent-limit",
            "2",
            "--codec-probe-only",
        ]
    )
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert calls == {"probe": 1, "train": 0}
    assert "[MUTAGENICITY_GLOBALGCE_CODEC_PROBE_OK]" in captured
    assert "[MUTAGENICITY_GLOBALGCE_TRAIN_POOL_BUILD_OK]" not in captured
    assert (fixture_paths["output"] / "codec_probe_summary.json").is_file()


def test_build_cli_does_not_print_success_after_empty_pool_error(
    fixture_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.baselines.globalgce import (
        build_mutagenicity_train_pool as build_cli,
    )

    class AvailableTeacher:
        available = True
        availability_reason = "ok"

        def __init__(self, _path):
            pass

    class Generator:
        def __init__(self, _official_root, *, native_train_csv=None):
            del native_train_csv

    def empty_build(**_kwargs):
        raise GlobalGCEEmptyCandidateUniverseError("empty")

    monkeypatch.setattr(build_cli, "TeacherSemanticScorer", AvailableTeacher)
    monkeypatch.setattr(
        build_cli,
        "OfficialGlobalGCEMutagenicityGenerator",
        Generator,
    )
    monkeypatch.setattr(build_cli, "build_mutagenicity_train_pool", empty_build)
    with pytest.raises(GlobalGCEEmptyCandidateUniverseError):
        build_cli.main(
            [
                "--train-csv",
                str(fixture_paths["train"]),
                "--teacher-path",
                str(fixture_paths["teacher"]),
                "--official-root",
                str(fixture_paths["official"]),
                "--output-dir",
                str(fixture_paths["output"]),
                "--expected-parent-count",
                "3",
            ]
        )
    assert (
        "[MUTAGENICITY_GLOBALGCE_TRAIN_POOL_BUILD_OK]"
        not in capsys.readouterr().out
    )


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
