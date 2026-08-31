from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.baselines import tastemolnet_globalgce_full as full
from src.baselines.globalgce_mutagenicity_adapter import TrainParent
from src.data.tastemolnet_ppo import TASTEMOLNET_PREPARED_FIELDS


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _threshold(path: Path) -> full.ThresholdContract:
    values = [0.1, 0.2, 0.4]
    _json(
        path,
        {
            "schema_version": "four_by_four_frozen_threshold_contract_v1",
            "dataset": "TasteMolNet",
            "thresholds": values,
            "theta_star": 0.2,
            "cost_cap": 0.4,
            "threshold_source": "preregistered calibration protocol",
            "threshold_source_split": "frozen_protocol",
            "threshold_config_hash": full.stable_json_sha256(values),
            "test_used_for_selection": False,
        },
    )
    return full.load_threshold_contract(path)


def _prepared_csv(path: Path, *, split: str, labels: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TASTEMOLNET_PREPARED_FIELDS)
        writer.writeheader()
        for index, label in enumerate(labels):
            row = {field: "" for field in TASTEMOLNET_PREPARED_FIELDS}
            row.update(
                {
                    "molecule_id": f"{split}-{index}",
                    "raw_smiles": "CC",
                    "canonical_smiles": "CC",
                    "model_smiles": "CC",
                    "label": str(label),
                    "label_name": {0: "Bitter", 1: "Sweet", 2: "Tasteless"}[label],
                    "split": split,
                }
            )
            writer.writerow(row)


def _pair(parent: str, candidate: str, *, distance: float | None, destination: int = 0):
    strict = distance is not None
    return {
        "dataset": "TasteMolNet",
        "method": "GlobalGCE",
        "split": "test",
        "parent_id": parent,
        "candidate_id": candidate,
        "applicable": True,
        "pair_strict_flip": strict,
        "wnode_distance": distance,
        "pred_before": 1,
        "pred_after": destination if strict else 1,
        "destination_label": destination if strict else None,
        "cf_drop": 0.5 if strict else 0.0,
        "rf_oracle_used": False,
    }


def test_threshold_contract_is_exact_grid_identity(tmp_path: Path) -> None:
    contract = _threshold(tmp_path / "threshold.json")
    assert contract.values == (0.1, 0.2, 0.4)
    assert contract.config_hash == full.stable_json_sha256([0.1, 0.2, 0.4])

    payload = json.loads((tmp_path / "threshold.json").read_text())
    payload["thresholds"] = [0.1, 0.2, 0.5]
    _json(tmp_path / "threshold.json", payload)
    with pytest.raises(full.TasteGlobalGCEFullError, match="threshold_config_hash"):
        full.load_threshold_contract(tmp_path / "threshold.json")


def test_prepared_split_is_exact_and_keeps_only_sweet(tmp_path: Path) -> None:
    path = tmp_path / "calibration.csv"
    _prepared_csv(path, split="calibration", labels=(0, 1, 2, 1))
    parents = full.load_prepared_split(
        path,
        expected_split="calibration",
        expected_sha256=full.sha256_file(path),
    )
    assert [row.parent_id for row in parents] == ["calibration-1", "calibration-3"]
    assert all(row.label == 1 and row.split == "calibration" for row in parents)

    with pytest.raises(full.TasteGlobalGCEFullError, match="row authority"):
        full.load_prepared_split(
            path,
            expected_split="test",
            expected_sha256=full.sha256_file(path),
        )


def test_test_file_is_not_opened_before_frozen_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = tmp_path / "selection_manifest.json"
    _json(
        selection,
        {
            "schema_version": full.SELECTION_SCHEMA,
            "status": "RUNNING",
            "selection_frozen": False,
            "selector_fitted_on_calibration": True,
            "test_loaded": False,
            "test_used_for_selection": False,
        },
    )
    opened = []
    monkeypatch.setattr(full, "load_prepared_split", lambda *args, **kwargs: opened.append(True))
    authority = SimpleNamespace(test_path=tmp_path / "never-open.csv", declared_test_sha256="a" * 64)
    with pytest.raises(full.TasteGlobalGCEFullError, match="requires frozen"):
        full.authorize_and_load_test_after_freeze(
            authority=authority, selection_manifest_path=selection
        )
    assert opened == []


def test_calibration_selector_is_deterministic_and_test_free() -> None:
    rules = [
        {"candidate_id": "r1"},
        {"candidate_id": "r2"},
        {"candidate_id": "r3"},
    ]
    rows = [
        {"split": "calibration", "parent_id": "p1", "candidate_id": "r1", "pair_strict_flip": True, "wnode_distance": 0.1},
        {"split": "calibration", "parent_id": "p2", "candidate_id": "r1", "pair_strict_flip": False, "wnode_distance": None},
        {"split": "calibration", "parent_id": "p1", "candidate_id": "r2", "pair_strict_flip": False, "wnode_distance": None},
        {"split": "calibration", "parent_id": "p2", "candidate_id": "r2", "pair_strict_flip": True, "wnode_distance": 0.1},
        {"split": "calibration", "parent_id": "p1", "candidate_id": "r3", "pair_strict_flip": True, "wnode_distance": 0.3},
        {"split": "calibration", "parent_id": "p2", "candidate_id": "r3", "pair_strict_flip": True, "wnode_distance": 0.3},
    ]
    # The production route has >=10 rules.  Repeat inert, deterministic rules
    # so this unit fixture exercises the same minimum without faking a lower gate.
    for index in range(4, 11):
        candidate = f"r{index}"
        rules.append({"candidate_id": candidate})
        rows.extend(
            {
                "split": "calibration",
                "parent_id": parent,
                "candidate_id": candidate,
                "pair_strict_flip": False,
                "wnode_distance": None,
            }
            for parent in ("p1", "p2")
        )
    selected, manifest = full.select_rules_on_calibration(rules, rows, theta_star=0.2)
    assert [row["candidate_id"] for row in selected[:2]] == ["r1", "r2"]
    assert manifest["selector_fitted_on_calibration"] is True
    assert manifest["test_loaded"] is False
    assert manifest["test_used_for_selection"] is False


def test_standardized_metrics_plateau_without_copying_rules(tmp_path: Path) -> None:
    threshold = _threshold(tmp_path / "threshold.json")
    ordered = [f"r{index}" for index in range(10)]
    rows = []
    for parent in ("p1", "p2"):
        for index, candidate in enumerate(ordered):
            distance = 0.1 if (parent == "p1" and index == 0) else None
            if parent == "p2" and index == 1:
                distance = 0.2
            rows.append(_pair(parent, candidate, distance=distance, destination=index % 2 * 2))
    metrics = full.compute_standardized_metrics(rows, ordered, threshold)
    assert len(metrics["figure3"]) == 20
    assert metrics["effective_rule_count"] == 10
    assert metrics["figure3"][9]["coverage"] == metrics["figure3"][19]["coverage"]
    assert metrics["prefix"][19]["plateau_after_effective_k"] is True
    assert {row["destination_label"] for row in metrics["destination"]} == {0, 2}


class _FakeBranchGenerator:
    def __init__(self, target_label: int) -> None:
        self.target_label = target_label
        self.calls = 0

    def generate(self, parents, **kwargs):  # noqa: ANN001, ANN003
        self.calls += 1
        root = Path(kwargs["output_dir"])
        files = (
            "native_rule_catalog.jsonl",
            "native_rule_rejections.jsonl",
            "globalgce_model.pt",
            "globalgce_rules.pt",
            "training_core_summary.json",
            "globalgce_training_checkpoints/training_checkpoint.pt",
            "globalgce_training_checkpoints/training_heartbeat.json",
        )
        for name in files:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x\n")
        return SimpleNamespace(
            training_summary={
                "prediction_backend": "frozen_gine_differentiable_bridge",
                "classifier_family": "gine",
                "oracle_backend": "gnn",
                "rf_oracle_used": False,
                "num_classes": 3,
                "frozen_source_label": 1,
                "frozen_target_label": self.target_label,
                "generation_input_split": "train",
                "calibration_loaded": False,
                "test_loaded": False,
                "valid_native_rule_count": 20,
                "training_resume_identity_sha256": "a" * 64,
            }
        )


def test_native_branch_resume_adopts_completed_manifest(tmp_path: Path) -> None:
    generator = _FakeBranchGenerator(0)
    config = full.TasteGlobalGCEFullConfig(epochs=25)
    parents = [
        TrainParent("a", "CC", 1, "train"),
        TrainParent("b", "CCC", 1, "train"),
    ]
    first = full.run_native_branch(
        target_label=0,
        generator=generator,
        parents=parents,
        branch_root=tmp_path / "target_0",
        config=config,
    )
    second = full.run_native_branch(
        target_label=0,
        generator=generator,
        parents=parents,
        branch_root=tmp_path / "target_0",
        config=config,
    )
    assert generator.calls == 1
    assert first == second
    assert second["official_epoch_checkpoint_resume"] is True


def test_pair_evaluation_resume_reuses_durable_parent_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    rules = [{"candidate_id": f"r{index}"} for index in range(10)]

    def fake_evaluate(*, parent, rules, scorer, provider, split):  # noqa: ANN001
        del scorer, provider
        calls.append(parent.parent_id)
        return [
            {
                "parent_id": parent.parent_id,
                "candidate_id": row["candidate_id"],
                "split": split,
            }
            for row in rules
        ]

    monkeypatch.setattr(full, "evaluate_one_parent", fake_evaluate)
    parents = [
        TrainParent("a", "CC", 1, "calibration"),
        TrainParent("b", "CCC", 1, "calibration"),
    ]
    checkpoints: list[int] = []
    first, _ = full.evaluate_split_resumable(
        split="calibration",
        parents=parents,
        rules=rules,
        scorer=object(),
        provider=object(),
        output=tmp_path,
        checkpoint_callback=checkpoints.append,
    )
    second, _ = full.evaluate_split_resumable(
        split="calibration",
        parents=parents,
        rules=rules,
        scorer=object(),
        provider=object(),
        output=tmp_path,
        checkpoint_callback=checkpoints.append,
    )
    assert calls == ["a", "b"]
    assert first == second
    assert checkpoints == [1, 2, 1, 2]


def test_sealed_resume_returns_without_reopening_science(tmp_path: Path) -> None:
    output = tmp_path / "sealed"
    output.mkdir()
    config = full.TasteGlobalGCEFullConfig(epochs=25)
    identity = {"schema_version": "test", "config": config.to_dict()}

    class Authority:
        @staticmethod
        def resume_identity(observed):  # noqa: ANN001
            assert observed == config
            return identity

    full.write_checkpoint(output, phase="SEALED", resume_identity=identity)
    full.atomic_json(output / "run_manifest.json", {"status": "SEALED"})
    result = full.run_t13_full(
        authority=Authority(),
        output_dir=output,
        config=config,
        resume=True,
        device="cuda:0",
        wnode_cache_db=tmp_path / "unused.sqlite",
        node_embedding_cache_dir=tmp_path / "unused-cache",
    )
    assert result == {"status": "SEALED"}


def test_slurm_wrapper_runs_science_then_independent_verifier() -> None:
    text = Path("scripts/slurm/run_tastemolnet_globalgce_full.sh").read_text()
    assert "#SBATCH --partition=A800" in text
    assert "#SBATCH --gres=gpu:a800:1" in text
    assert "export PYTHONPATH=$PWD" in text
    assert text.count("python scripts/run_tastemolnet_globalgce_full.py") == 2
    assert "--verify-only" in text
    assert "--set inference.fallback_to_heuristic=false" in text


def test_terminal_verifier_replays_metrics_and_passes_registry(tmp_path: Path) -> None:
    output = tmp_path / "t13"
    raw = output / "raw"
    raw.mkdir(parents=True)
    threshold = _threshold(tmp_path / "threshold.json")
    ordered = [f"r{index}" for index in range(10)]
    selected_rules = [{"candidate_id": candidate} for candidate in ordered]
    full.atomic_jsonl(raw / "selected_rules.jsonl", selected_rules)
    full.atomic_jsonl(raw / "merged_rules.jsonl", selected_rules)
    test_rows = []
    for parent in ("p1", "p2"):
        for index, candidate in enumerate(ordered):
            distance = 0.1 if (parent == "p1" and index == 0) else None
            if parent == "p2" and index == 1:
                distance = 0.2
            test_rows.append(
                _pair(parent, candidate, distance=distance, destination=index % 2 * 2)
            )
    calibration_rows = [dict(row, split="calibration") for row in test_rows]
    full.atomic_jsonl(raw / "calibration_pair_details.jsonl", calibration_rows)
    full.atomic_jsonl(raw / "test_pair_details.jsonl", test_rows)
    selection = {
        "schema_version": full.SELECTION_SCHEMA,
        "dataset": full.DATASET,
        "method": full.METHOD,
        "stage": full.STAGE,
        "status": "FROZEN",
        "selection_frozen": True,
        "selector_fitted_on_calibration": True,
        "test_loaded": False,
        "test_used_for_selection": False,
        "frozen_at": "2026-08-31T00:00:00+00:00",
        "ordered_rule_ids": ordered,
        **threshold.to_dict(),
    }
    full.atomic_json(raw / "selection_manifest.json", selection)
    full.atomic_json(
        raw / "test_evaluation_manifest.json",
        {
            "selection_manifest_sha256": full.sha256_file(
                raw / "selection_manifest.json"
            ),
            "selection_frozen_before_test": True,
            "test_used_for_selection": False,
            "started_at": "2026-08-31T00:01:00+00:00",
        },
    )
    metrics = full.compute_standardized_metrics(test_rows, ordered, threshold)
    full.atomic_csv(output / "figure3_coverage_vs_k.csv", metrics["figure3"])
    full.atomic_csv(output / "figure4_coverage_vs_threshold.csv", metrics["figure4"])
    full.atomic_csv(output / "prefix_metrics.csv", metrics["prefix"])
    full.atomic_json(output / "prefix_metrics.json", metrics["prefix"])
    full.atomic_csv(output / "parent_best_distances.csv", metrics["parent_best"])
    full.atomic_csv(output / "destination_distribution.csv", metrics["destination"])
    full.atomic_csv(output / "table2_globalgce_k10.csv", metrics["table2"])
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    common = {
        "dataset": full.DATASET,
        "method": full.METHOD,
        "stage": full.STAGE,
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "oracle_checkpoint": str(checkpoint),
        "oracle_hash": "a" * 64,
        "oracle_checkpoint_hash": "a" * 64,
        "dataset_hash": "b" * 64,
        "test_parent_ids_sha256": "c" * 64,
        "test_split_hash": "d" * 64,
        "distance_line": full.DISTANCE_LINE,
        "molclr_checkpoint_hash": "e" * 64,
        "cf_mode": full.CF_MODE,
        "threshold_config_hash": threshold.config_hash,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "raw_output_root": str(output),
    }
    full.atomic_json(
        output / "summary.json",
        {
            "schema_version": "test",
            **common,
            "status": "SEALED",
            "frozen": True,
            "raw_output_complete": True,
        },
    )
    full.atomic_json(
        output / "oracle_manifest.json",
        {"schema_version": "test", **common, "frozen": True},
    )
    full.atomic_json(
        output / "evaluation_manifest.json",
        {"schema_version": "test", **common, "status": "SEALED", "frozen": True},
    )
    inventory = full._immutable_artifact_inventory(output)
    full.atomic_json(
        output / "freeze_manifest.json",
        {
            "schema_version": "test",
            **common,
            "status": "SEALED",
            "frozen": True,
            "files": inventory,
            "inventory_sha256": full.stable_sha256(inventory),
        },
    )
    full.atomic_json(
        output / "run_manifest.json",
        {
            "schema_version": full.RUN_MANIFEST_SCHEMA,
            **common,
            "status": "SEALED",
            "state": "SEALED",
            "run_complete": False,
            "raw_output_complete": True,
            "source_artifacts_complete": True,
            "frozen": True,
        },
    )
    resume_identity = {"test": True}
    full.write_checkpoint(
        output, phase="SEALED", resume_identity=resume_identity
    )
    (output / "SEALED").write_text("SEALED\n", encoding="utf-8")

    audit = full.verify_t13_output(output)
    assert audit["passed"] is True
    assert audit["registry_status"] in {"FROZEN_PASS", "ADOPTABLE_PASS"}
    assert (output / "PASS").read_text(encoding="utf-8") == "PASS\n"
    assert full.read_json(output / "run_manifest.json")["run_complete"] is True
