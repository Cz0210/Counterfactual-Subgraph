from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.eval.mutagenicity_wnode_matrix import CalibrationParent
from src.eval.wnode_action_matrix import (
    BACE_CANDIDATE_ID_PREFIX,
    ActionMatrixConfig,
    audit_bace_action_matrix,
    bace_matrix_config,
    build_bace_action_matrix,
    evaluate_parent_candidate_pair,
)


class Teacher:
    def score_smiles(self, smiles: str, label: int | None = None, **_: Any) -> dict[str, Any]:
        prediction = 1 if smiles == "CCCO" else 0
        return {
            "teacher_result_ok": True,
            "teacher_label": prediction,
            "teacher_prob": 0.9 if prediction else 0.1,
        }


class Distance:
    def distance(self, _left: str, right: str) -> dict[str, Any]:
        return {"ok": True, "distance": {"CCO": 0.2, "CCC": 0.4}[right]}

    def stats_dict(self) -> dict[str, Any]:
        return {}


class BuildDistance:
    def distance(self, left: str, right: str) -> dict[str, Any]:
        return {"ok": True, "distance": 0.0 if left == right else 0.2}

    def stats_dict(self) -> dict[str, Any]:
        return {"cache_hit_rate": 0.0}


def test_bace_matrix_config_has_dataset_specific_candidate_identity() -> None:
    config = bace_matrix_config()
    assert config.dataset_name == "BACE"
    assert config.candidate_id_prefix == BACE_CANDIDATE_ID_PREFIX


def test_all_matches_use_minimum_valid_strict_flip_distance() -> None:
    deletions = [
        {"match_index": 0, "match_atoms": [0], "delete_valid": True, "residual_smiles": "CCC"},
        {"match_index": 1, "match_atoms": [1], "delete_valid": True, "residual_smiles": "CCO"},
    ]
    pair, matches = evaluate_parent_candidate_pair(
        CalibrationParent("p", "CCCO", 1, "calibration"),
        {"candidate_id": "c", "canonical_fragment": "C"},
        teacher=Teacher(),
        distance_provider=Distance(),
        deletion_fn=lambda _parent, _fragment: deletions,
    )
    assert len(matches) == 2
    assert pair["num_strict_flip_matches"] == 2
    assert pair["best_match_index"] == 1
    assert pair["wnode_distance"] == 0.2


def test_matrix_build_audit_and_resume_are_calibration_only(tmp_path: Path) -> None:
    pool = tmp_path / "candidate_pool.jsonl"
    pool.write_text(
        "".join(
            json.dumps(
                {
                        "molecule_id": f"source{index}",
                        "label": 1,
                        "parent_smiles": "CCCO",
                    "final_fragment": fragment,
                    "parse_ok": True,
                    "valid": True,
                    "connected": True,
                    "final_substructure": True,
                    "oracle_ok": True,
                    "cf_flip": True,
                    "cf_drop": 0.4 + index / 10,
                    "failure_tag": None,
                    "full_parent": False,
                    "near_parent_hard_fail": False,
                    "tiny_fragment_hard_fail": False,
                    "projection_used": False,
                    "direct_substructure": True,
                }
            )
            + "\n"
            for index, fragment in enumerate(("C", "N"))
        ),
        encoding="utf-8",
    )
    parents = tmp_path / "calibration.csv"
    with parents.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["molecule_id", "smiles", "label", "split"]
        )
        writer.writeheader()
        writer.writerow(
            {"molecule_id": "p0", "smiles": "CCCO", "label": 1, "split": "calibration"}
        )
    teacher_path = tmp_path / "teacher.pkl"
    checkpoint = tmp_path / "model.pth"
    teacher_path.write_bytes(b"teacher")
    checkpoint.write_bytes(b"molclr")
    molclr_root = tmp_path / "MolCLR"
    molclr_root.mkdir()
    output = tmp_path / "matrix"

    def delete(_parent: str, _fragment: str) -> list[dict[str, Any]]:
        return [
            {
                "match_index": 0,
                "match_atoms": [0],
                "delete_valid": True,
                "residual_smiles": "CCO",
            }
        ]

    kwargs = {
        "candidate_pool": pool,
        "calibration_csv": parents,
        "output_dir": output,
        "teacher_path": teacher_path,
        "molclr_root": molclr_root,
        "molclr_checkpoint": checkpoint,
        "wnode_cache_db": tmp_path / "cache.sqlite3",
        "teacher": Teacher(),
        "distance_provider": BuildDistance(),
        "config": ActionMatrixConfig(expected_parent_count=1, flush_every=1),
        "deletion_fn": delete,
    }
    first = build_bace_action_matrix(**kwargs)
    first_pairs = (output / "pair_matrix.jsonl").read_text(encoding="utf-8")
    second = build_bace_action_matrix(**kwargs)
    second_pairs = (output / "pair_matrix.jsonl").read_text(encoding="utf-8")

    assert first["actual_pair_rows"] == second["actual_pair_rows"] == 2
    assert first_pairs == second_pairs
    assert all(
        json.loads(line)["candidate_id"].startswith(f"{BACE_CANDIDATE_ID_PREFIX}_")
        for line in first_pairs.splitlines()
    )
    audit = audit_bace_action_matrix(
        output, expected_parent_count=1, expected_candidate_count=2
    )
    assert audit["complete_cartesian"] is True
    assert audit["required_pair_schema_pass"] is True
    assert json.loads((output / "run_manifest.json").read_text())["test_loaded"] is False
