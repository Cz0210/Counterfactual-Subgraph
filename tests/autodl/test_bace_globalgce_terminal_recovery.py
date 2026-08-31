from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.autodl.recover_bace_globalgce_terminal import build_parser
from scripts.autodl.run_four_gpu_recovery_controller import load_controller_manifest
from scripts.autodl.standardize_bace_frozen_cell import (
    build_parser as build_standardize_parser,
)
from src.baselines.bace_globalgce_terminal_recovery import (
    MIN_RULES,
    PASS_MARKER,
    build_recovery_controller_fragment,
)
from src.baselines.globalgce_bace_native_rules import GlobalGCENativeRule
from src.baselines.globalgce_mutagenicity_adapter import (
    OFFICIAL_AFFINE_EDGE_HARD_DECODE,
    materialize_frozen_gine_native_rule_rows,
)
from src.eval.bace_native_baseline_gnn import _minimum_candidate_count


def _absolute_paths(tmp_path: Path) -> dict[str, Path]:
    slug = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]
    root = Path("/persistent") / slug
    return {
        "python": root / "env/bin/python",
        "project_root": root / "project",
        "output_root": root / "recovery",
        "failed_controller_root": root / "failed-controller",
        "source_round_root": root / "failed-controller/rounds/round-1-seed-7",
        "source_manifest": root / "source.jsonl",
        "native_train_csv": root / "train.csv",
        "official_root": root / "official",
        "gnn_checkpoint": root / "gine",
        "dataset_dir": root / "dataset",
        "calibration_split": root / "calibration.csv",
        "test_split": root / "test.csv",
        "molclr_root": root / "molclr",
        "molclr_checkpoint": root / "molclr.pt",
        "neurosed_checkpoint": root / "neurosed.pt",
    }


def _production_manifest(tmp_path: Path, tasks: list[dict]) -> Path:
    path = tmp_path / "controller.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "controller_id": "bace-globalgce-terminal-recovery-test",
                "paper_frozen": True,
                "runtime": {
                    "max_gpus": 4,
                    "stable_idle_seconds": 60,
                    "sample_interval_seconds": 5,
                    "poll_seconds": 60,
                    "max_transient_retries": 1,
                },
                "resource_gates": {},
                "tasks": tasks,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_affine_edge_scores_are_typed_before_unrelaxed_native_validation() -> None:
    torch = pytest.importorskip("torch")
    lhs_feature = torch.tensor(
        [[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32
    )
    lhs_adjacency = torch.tensor(
        [[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32
    )
    lhs_edge = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]], dtype=torch.float32)
    # These are pinned-official affine class scores, deliberately not values
    # in [0, 1].  The argmax is the categorical ``single`` bond label.
    rhs_affine_edge = torch.tensor([[[-9.0, 7.0, 2.0, 1.0]]], dtype=torch.float32)
    rows, rejected = materialize_frozen_gine_native_rule_rows(
        rules={
            "feat": lhs_feature,
            "adj": lhs_adjacency,
            "edge_attr": lhs_edge,
            "features_reconst": lhs_feature.clone(),
            "adj_reconst": lhs_adjacency.clone(),
            "edge_attrs_reconst": rhs_affine_edge,
        },
        atom_symbols=("C", "O"),
        bond_names=("no_edge", "single", "double", "triple"),
        oracle_checkpoint_hash="a" * 64,
    )
    assert rejected == []
    assert len(rows) == 1
    assert rows[0]["edge_score_hard_decode"] == OFFICIAL_AFFINE_EDGE_HARD_DECODE
    assert rows[0]["rule"]["rhs_edge_attr"] == [[0.0, 1.0, 0.0, 0.0]]
    # Reopen through the original hard validator; recovery did not broaden its
    # numeric domain or otherwise special-case the row.
    GlobalGCENativeRule.from_payload(rows[0]).validate()


def test_recovery_fragment_reuses_candidate_dependency_chain_and_never_trains(
    tmp_path: Path,
) -> None:
    fragment = build_recovery_controller_fragment(**_absolute_paths(tmp_path))
    tasks = {task["id"]: task for task in fragment["tasks"]}
    assert fragment["root_task_ids"] == ["bace_globalgce_train_candidates"]
    assert fragment["terminal_task_ids"] == ["bace_globalgce_standardized"]
    assert fragment["MIN_RULES_FOR_MAIN_TABLE"] == MIN_RULES == 10
    assert "bace_globalgce_preflight" not in tasks
    assert "bace_globalgce_bridge_smoke" not in tasks
    recovery = tasks["bace_globalgce_train_candidates"]
    assert recovery["resource"] == "cpu"
    assert recovery["data_splits"] == ["train"]
    assert recovery["required_log_marker"] == PASS_MARKER
    command = " ".join(recovery["command"])
    assert "recover_bace_globalgce_terminal.py" in command
    assert "globalgce-train-rules" not in command
    assert recovery["read_only_adoption"] is True
    assert recovery["retraining_forbidden"] is True
    calibration = tasks["bace_globalgce_calibration_shard_0"]
    assert recovery["id"] in calibration["depends_on"]
    manifest = load_controller_manifest(
        _production_manifest(tmp_path, list(tasks.values()))
    )
    assert manifest.by_id[recovery["id"]].resource == "cpu"


def test_globalgce_resource_cap_and_cli_contracts_are_enabled() -> None:
    assert _minimum_candidate_count("globalgce") == 10
    assert _minimum_candidate_count("comrecgc") == 10
    assert _minimum_candidate_count("gcfexplainer") == 20
    assert (
        build_standardize_parser().parse_args(
            [
                "--method",
                "GlobalGCE",
                "--source-final-root",
                "/source",
                "--gnn-checkpoint",
                "/gine",
                "--output-dir",
                "/output",
            ]
        ).method
        == "GlobalGCE"
    )
    parsed = build_parser().parse_args(
        [
            "recover",
            "--failed-controller-root",
            "/failed",
            "--source-round-root",
            "/failed/rounds/round-1-seed-7",
            "--source-manifest",
            "/source.jsonl",
            "--native-train-csv",
            "/train.csv",
            "--official-root",
            "/official",
            "--gnn-checkpoint",
            "/gine",
            "--output-dir",
            "/recovered",
        ]
    )
    assert parsed.command == "recover"
