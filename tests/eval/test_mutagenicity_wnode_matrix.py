from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from src.eval.mutagenicity_wnode_matrix import (
    CalibrationParent,
    MatrixBuildConfig,
    audit_calibration_matrix_run,
    build_calibration_matrix_run,
    build_candidate_universe,
    build_cartesian_rows,
    evaluate_parent_candidate_pair,
    load_calibration_parents,
    stable_candidate_id,
)


class FakeTeacher:
    available = True
    availability_reason = "ok"

    def __init__(self, predictions: dict[str, tuple[int, float]]) -> None:
        self.predictions = predictions

    def score_smiles(
        self,
        smiles: str,
        label: int | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        pred, p1 = self.predictions.get(smiles, (1, 0.9))
        assert label == 1
        return {
            "teacher_result_ok": True,
            "teacher_label": pred,
            "teacher_prob": p1,
            "teacher_reason": "ok",
        }


class FakeMolCLREncoder:
    checkpoint_identity = "fake-molclr"


class FakeDistance:
    def __init__(self, distances: dict[tuple[str, str], float] | None = None) -> None:
        self.distances = distances or {}
        self.encoder = FakeMolCLREncoder()
        self.calls: list[tuple[str, str]] = []

    def distance(self, smiles_a: str, smiles_b: str) -> dict[str, Any]:
        self.calls.append((smiles_a, smiles_b))
        if smiles_a == smiles_b:
            value = 0.0
        else:
            value = self.distances.get(
                (smiles_a, smiles_b),
                self.distances.get((smiles_b, smiles_a), 0.25),
            )
        return {
            "distance": value,
            "ok": True,
            "cache_hit": False,
            "error": None,
        }

    def stats_dict(self) -> dict[str, Any]:
        return {
            "pair_distance_cache_hit_rate": 0.0,
            "node_embedding_cache_hit_rate": 0.0,
        }


def _eligible_row(
    fragment: str,
    parent_id: str,
    *,
    cf_drop: float = 0.5,
) -> dict[str, Any]:
    return {
        "final_fragment": fragment,
        "parent_smiles": "CCO",
        "parent_index": parent_id,
        "parse_ok": True,
        "final_substructure": True,
        "oracle_ok": True,
        "cf_flip": True,
        "cf_drop": cf_drop,
        "reward_total": 1.0,
        "atom_ratio": 0.3,
    }


def _candidate(fragment: str, candidate_id: str = "candidate") -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "canonical_fragment": fragment,
    }


def test_source_eligible_filter_and_canonical_fragment_dedup() -> None:
    rows = [
        _eligible_row("CC", "p1", cf_drop=0.4),
        _eligible_row("C(C)", "p2", cf_drop=0.8),
        {**_eligible_row("N", "p3"), "final_substructure": False},
        {**_eligible_row("O", "p4"), "oracle_ok": False},
        {**_eligible_row("F", "p5"), "cf_flip": False},
    ]

    universe, stats = build_candidate_universe(rows)

    assert stats == {
        "input_pool_rows": 5,
        "source_eligible_rows": 2,
        "source_eligible_raw_unique_fragments": 2,
        "canonical_unique_candidates": 1,
    }
    assert len(universe) == 1
    assert universe[0]["canonical_fragment"] == "CC"
    assert universe[0]["source_row_count"] == 2
    assert universe[0]["source_parent_count"] == 2
    assert universe[0]["source_parent_ids"] == ["p1", "p2"]


def test_candidate_id_is_stable_sha256_identity() -> None:
    first = stable_candidate_id("C[C@H](O)N")
    second = stable_candidate_id("C[C@H](O)N")

    assert first == second
    assert first.startswith("MUT_WNODE_")
    assert stable_candidate_id("CC(O)N") != first


def test_cartesian_rows_keep_inapplicable_pairs() -> None:
    parents = [
        CalibrationParent("p1", "CCO", 1, "calibration"),
        CalibrationParent("p2", "CCN", 1, "calibration"),
    ]
    candidates = [_candidate("C", "c1"), _candidate("N", "c2")]
    teacher = FakeTeacher({"CCO": (1, 0.9), "CCN": (1, 0.8)})
    distance = FakeDistance()

    pair_rows, match_rows = build_cartesian_rows(
        parents,
        candidates,
        teacher=teacher,
        distance_provider=distance,
        deletion_fn=lambda _parent, _fragment: [],
    )

    assert len(pair_rows) == 4
    assert match_rows == []
    assert all(row["applicable"] is False for row in pair_rows)
    assert all(row["num_matches"] == 0 for row in pair_rows)
    assert all(row["wnode_distance"] is None for row in pair_rows)


def test_all_matches_are_enumerated_and_nonflip_does_not_enter_wnode() -> None:
    parent = CalibrationParent("p1", "CCCO", 1, "calibration")
    teacher = FakeTeacher(
        {
            "CCCO": (1, 0.9),
            "CCC": (1, 0.7),
            "CCO": (0, 0.2),
        }
    )
    distance = FakeDistance({("CCCO", "CCO"): 0.3})
    deletions = [
        {
            "match_index": 0,
            "match_atoms": [3],
            "delete_valid": True,
            "residual_smiles": "CCC",
            "error": None,
        },
        {
            "match_index": 1,
            "match_atoms": [0],
            "delete_valid": True,
            "residual_smiles": "CCO",
            "error": None,
        },
    ]

    pair, matches = evaluate_parent_candidate_pair(
        parent,
        _candidate("C"),
        teacher=teacher,
        distance_provider=distance,
        deletion_fn=lambda _parent, _fragment: deletions,
    )

    assert len(matches) == 2
    assert pair["num_matches"] == 2
    assert pair["num_strict_flip_matches"] == 1
    assert pair["best_match_index"] == 1
    assert distance.calls == [("CCCO", "CCO")]
    assert matches[0]["teacher_strict_flip"] is False
    assert matches[0]["wnode_distance"] is None


def test_multiple_strict_matches_choose_distance_then_cf_drop_then_index() -> None:
    parent = CalibrationParent("p1", "CCCO", 1, "calibration")
    teacher = FakeTeacher(
        {
            "CCCO": (1, 0.9),
            "CCN": (0, 0.1),
            "CCO": (0, 0.2),
            "CCC": (0, 0.2),
        }
    )
    distance = FakeDistance(
        {
            ("CCCO", "CCN"): 0.4,
            ("CCCO", "CCO"): 0.2,
            ("CCCO", "CCC"): 0.2,
        }
    )
    deletions = [
        {"match_index": 0, "match_atoms": [0], "delete_valid": True, "residual_smiles": "CCN"},
        {"match_index": 2, "match_atoms": [2], "delete_valid": True, "residual_smiles": "CCO"},
        {"match_index": 1, "match_atoms": [1], "delete_valid": True, "residual_smiles": "CCC"},
    ]

    pair, matches = evaluate_parent_candidate_pair(
        parent,
        _candidate("C"),
        teacher=teacher,
        distance_provider=distance,
        deletion_fn=lambda _parent, _fragment: deletions,
    )

    assert len(matches) == 3
    assert pair["pair_strict_flip"] is True
    assert pair["wnode_distance"] == pytest.approx(0.2)
    assert pair["best_match_index"] == 1
    assert pair["residual_smiles"] == "CCC"


@pytest.mark.parametrize(
    ("pred_before", "pred_after", "expected"),
    [
        (1, 0, True),
        (0, 0, False),
        (1, 1, False),
        (0, 1, False),
    ],
)
def test_strict_flip_definition_is_exact(
    pred_before: int,
    pred_after: int,
    expected: bool,
) -> None:
    parent = CalibrationParent("p1", "CCO", 1, "calibration")
    teacher = FakeTeacher({"CCO": (pred_before, 0.9), "CO": (pred_after, 0.1)})
    distance = FakeDistance({("CCO", "CO"): 0.1})

    pair, matches = evaluate_parent_candidate_pair(
        parent,
        _candidate("C"),
        teacher=teacher,
        distance_provider=distance,
        deletion_fn=lambda _parent, _fragment: [
            {
                "match_index": 0,
                "match_atoms": [0],
                "delete_valid": True,
                "residual_smiles": "CO",
            }
        ],
    )

    assert matches[0]["teacher_strict_flip"] is expected
    assert pair["pair_strict_flip"] is expected
    assert (pair["wnode_distance"] is not None) is expected


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_test_path_and_test_split_are_rejected(tmp_path: Path) -> None:
    forbidden_path = tmp_path / "test.csv"
    _write_csv(
        forbidden_path,
        [{"molecule_id": "p1", "smiles": "CCO", "label": 1}],
    )
    with pytest.raises(ValueError, match="Test input path"):
        load_calibration_parents(forbidden_path)

    calibration_path = tmp_path / "calibration.csv"
    _write_csv(
        calibration_path,
        [
            {
                "molecule_id": "p1",
                "smiles": "CCO",
                "label": 1,
                "split": "test",
            }
        ],
    )
    with pytest.raises(ValueError, match="forbidden split"):
        load_calibration_parents(calibration_path)


def test_resume_produces_no_duplicate_or_missing_pairs(tmp_path: Path) -> None:
    pool_path = tmp_path / "pool.jsonl"
    pool_rows = [_eligible_row("C", "source1"), _eligible_row("N", "source2")]
    pool_path.write_text(
        "".join(json.dumps(row) + "\n" for row in pool_rows),
        encoding="utf-8",
    )
    calibration_path = tmp_path / "calibration.csv"
    _write_csv(
        calibration_path,
        [
            {
                "molecule_id": "p1",
                "smiles": "CCO",
                "label": 1,
                "split": "calibration",
            }
        ],
    )
    teacher_path = tmp_path / "teacher.pkl"
    teacher_path.write_bytes(b"fake teacher")
    molclr_root = tmp_path / "molclr"
    molclr_root.mkdir()
    molclr_checkpoint = tmp_path / "model.pth"
    molclr_checkpoint.write_bytes(b"fake checkpoint")
    output_dir = tmp_path / "run"
    teacher = FakeTeacher({"CCO": (1, 0.9), "O": (0, 0.1)})
    distance = FakeDistance({("CCO", "O"): 0.15})

    def deletion(_parent: str, fragment: str) -> list[dict[str, Any]]:
        if fragment == "C":
            return [
                {
                    "match_index": 0,
                    "match_atoms": [0],
                    "delete_valid": True,
                    "residual_smiles": "O",
                }
            ]
        return []

    kwargs = {
        "candidate_pool": pool_path,
        "calibration_csv": calibration_path,
        "output_dir": output_dir,
        "teacher_path": teacher_path,
        "molclr_root": molclr_root,
        "molclr_checkpoint": molclr_checkpoint,
        "wnode_cache_db": tmp_path / "cache.sqlite",
        "teacher": teacher,
        "distance_provider": distance,
        "config": MatrixBuildConfig(
            expected_parent_count=1,
            flush_every=1,
            resume=True,
        ),
        "deletion_fn": deletion,
    }
    first = build_calibration_matrix_run(**kwargs)
    with (output_dir / "pair_matrix.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"parent_id":"interrupted')
    with (output_dir / "match_instances.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"parent_id":"interrupted')
    second = build_calibration_matrix_run(**kwargs)
    pairs = [
        json.loads(line)
        for line in (output_dir / "pair_matrix.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert first["actual_pair_rows"] == 2
    assert second["actual_pair_rows"] == 2
    assert len(pairs) == 2
    assert len({(row["parent_id"], row["candidate_id"]) for row in pairs}) == 2
    audit = audit_calibration_matrix_run(
        output_dir,
        expected_parent_count=1,
        expected_candidate_count=2,
        expected_pair_count=2,
        expected_source_eligible_rows=2,
        expected_source_eligible_raw_unique=2,
        require_complete_cartesian=True,
        require_strict_flip_pair=True,
        forbid_test=True,
    )
    assert audit["audit_passed"] is True
