from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import train_tastemolnet_ours_full as train_full
from src.eval import tastemolnet_ours_full as ours
from src.eval.four_by_four_registry import PASS_STATUSES, audit_explicit_candidate


def _threshold(tmp_path: Path) -> tuple[Path, ours.ThresholdContract]:
    values = [0.5, 1.0]
    path = tmp_path / "threshold.json"
    path.write_text(
        json.dumps(
            {
                "dataset": "TasteMolNet",
                "thresholds": values,
                "theta_star": 0.5,
                "cost_cap": 1.0,
                "threshold_source": "frozen_shared_calibration",
                "threshold_source_split": "calibration",
                "threshold_config_hash": ours.stable_sha256(values),
                "test_used_for_selection": False,
            }
        ),
        encoding="utf-8",
    )
    return path, ours.load_threshold_contract(path)


def _pairs(candidate_ids: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for parent_index, parent in enumerate(("P0", "P1")):
        for index, candidate in enumerate(candidate_ids):
            strict = index == parent_index
            rows.append(
                {
                    "dataset": "TasteMolNet",
                    "method": "Ours",
                    "split": "test",
                    "parent_id": parent,
                    "candidate_id": candidate,
                    "pair_strict_flip": strict,
                    "wnode_distance": 0.25 + parent_index * 0.1 if strict else None,
                    "destination_label": (0, 2)[parent_index] if strict else None,
                    "cf_drop": 0.7 if strict else None,
                    "applicable": True,
                    "rf_oracle_used": False,
                }
            )
    return rows


def test_full_ppo_schedule_is_real_and_restartable() -> None:
    assert train_full.UPDATES == 300
    assert train_full.CHECKPOINT_STEPS == (50, 100, 150, 200, 250, 300)
    source = Path(train_full.__file__).read_text(encoding="utf-8")
    assert "run_stable_decoded_chem_ppo_loop(" in source
    assert "resume_from_checkpoint=resume_checkpoint" in source
    assert "calibration_loaded\": False" in source
    assert "test_loaded\": False" in source


def test_shared_threshold_is_required_and_excludes_test(tmp_path: Path) -> None:
    path, contract = _threshold(tmp_path)
    assert contract.theta_star == 0.5
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["test_used_for_selection"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ours.TasteOursFullError, match="exclude test"):
        ours.load_threshold_contract(path)


def test_base_high_merge_is_canonical_and_train_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ours, "canonicalize_smiles", lambda value: value or None)
    rows = []
    for index in range(20):
        for mode in ("base", "high_temp"):
            rows.append(
                {
                    "canonical_fragment": f"F{index}",
                    "parent_id": f"P{index % 3}",
                    "stage": mode,
                    "parse_ok": True,
                    "connected": True,
                    "direct_substructure": True,
                    "oracle_ok": True,
                    "cf_flip": index % 2 == 0,
                    "cf_drop": float(index) / 20,
                }
            )
    universe = ours.merge_candidate_modes(rows)
    assert len(universe) == 20
    assert all(row["source_modes"] == ["base", "high_temp"] for row in universe)
    assert len({row["candidate_id"] for row in universe}) == 20


def test_calibration_selector_and_test_metrics_use_frozen_prefix(tmp_path: Path) -> None:
    candidate_ids = [f"C{index}" for index in range(10)]
    candidates = [{"candidate_id": value, "canonical_fragment": value} for value in candidate_ids]
    calibration = []
    for parent in ("A", "B"):
        for index, candidate in enumerate(candidate_ids):
            strict = index < 2
            calibration.append(
                {
                    "split": "calibration",
                    "parent_id": parent,
                    "candidate_id": candidate,
                    "pair_strict_flip": strict,
                    "wnode_distance": 0.2 + 0.01 * index if strict else None,
                }
            )
    selected, trace = ours.select_on_calibration(candidates, calibration, theta_star=0.5)
    assert len(selected) == 10
    assert len(trace) == 10
    _path, threshold = _threshold(tmp_path)
    metrics = ours.standardized_metrics(_pairs([row["candidate_id"] for row in selected]), [row["candidate_id"] for row in selected], threshold)
    assert [row["k"] for row in metrics["figure3"]] == list(range(1, 21))
    assert metrics["table2"][0]["k"] == 10
    assert {row["destination_label"] for row in metrics["destination"]} == {0, 2}


def test_independent_verifier_replays_and_publishes_fresh_root(tmp_path: Path) -> None:
    threshold_path, threshold = _threshold(tmp_path)
    science = tmp_path / "science"
    raw = science / "raw"
    raw.mkdir(parents=True)
    candidate_ids = [f"C{index}" for index in range(10)]
    universe = [{"candidate_id": value, "canonical_fragment": value} for value in candidate_ids]
    calibration = []
    for parent in ("A", "B"):
        for index, candidate in enumerate(candidate_ids):
            strict = index < 2
            calibration.append(
                {
                    "split": "calibration",
                    "parent_id": parent,
                    "candidate_id": candidate,
                    "pair_strict_flip": strict,
                    "wnode_distance": 0.2 + index * 0.01 if strict else None,
                }
            )
    selected, selector_trace = ours.select_on_calibration(
        universe, calibration, theta_star=threshold.theta_star
    )
    candidate_ids = [row["candidate_id"] for row in selected]
    pairs = _pairs(candidate_ids)
    metrics = ours.standardized_metrics(pairs, candidate_ids, threshold)
    ours.atomic_jsonl(raw / "candidate_universe.jsonl", universe)
    ours.atomic_jsonl(raw / "calibration_pair_details.jsonl", calibration)
    ours.atomic_jsonl(raw / "selected_rules.jsonl", selected)
    ours.atomic_jsonl(raw / "test_pair_details.jsonl", pairs)
    selection = {
        "status": "FROZEN",
        "selection_frozen": True,
        "selector_fitted_on_calibration": True,
        "test_loaded": False,
        "test_used_for_selection": False,
        "threshold_config_hash": threshold.config_hash,
        "threshold_contract_file_sha256": threshold.file_sha256,
        "theta_star": threshold.theta_star,
        "cost_cap": threshold.cost_cap,
        "oracle_checkpoint_hash": "a" * 64,
        "molclr_checkpoint_hash": "e" * 64,
        "frozen_at": "2026-08-31T00:00:00+00:00",
        "ordered_rule_ids": candidate_ids,
        "trace": selector_trace,
        "selected_rules_sha256": ours.sha256_file(raw / "selected_rules.jsonl"),
        "candidate_universe_sha256": ours.sha256_file(raw / "candidate_universe.jsonl"),
        "calibration_pair_details_sha256": ours.sha256_file(raw / "calibration_pair_details.jsonl"),
    }
    ours.atomic_json(raw / "selection_manifest.json", selection)
    ours.atomic_json(
        raw / "test_access_receipt.json",
        {
            "started_at": "2026-08-31T00:01:00+00:00",
            "selection_manifest_sha256": ours.sha256_file(raw / "selection_manifest.json"),
            "selection_frozen_before_test": True,
            "test_used_for_selection": False,
        },
    )
    ours.atomic_json(
        raw / "test_evaluation_manifest.json",
        {
            "started_at": "2026-08-31T00:01:00+00:00",
            "selection_manifest_sha256": ours.sha256_file(raw / "selection_manifest.json"),
            "selection_frozen_before_test": True,
        },
    )
    for name, rows in ours._artifact_rows(metrics).items():
        ours.atomic_csv(science / name, rows)
    ours.atomic_json(science / "prefix_metrics.json", metrics["prefix"])
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    common = {
        "dataset": "TasteMolNet",
        "method": "Ours",
        "stage": "T11_OURS_FULL",
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "oracle_checkpoint": str(checkpoint),
        "oracle_hash": "a" * 64,
        "oracle_checkpoint_hash": "a" * 64,
        "dataset_hash": "b" * 64,
        "test_parent_ids_sha256": "c" * 64,
        "test_split_hash": "d" * 64,
        "distance_line": ours.DISTANCE_LINE,
        "molclr_checkpoint_hash": "e" * 64,
        "cf_mode": "strict_flip",
        "threshold_config_hash": threshold.config_hash,
        "threshold_contract_file_sha256": threshold.file_sha256,
        "theta_star": threshold.theta_star,
        "cost_cap": threshold.cost_cap,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "raw_output_root": str(science),
    }
    ours.atomic_json(
        science / "summary.json",
        {
            **common,
            "status": "SEALED",
            "raw_output_complete": True,
            "frozen": True,
        },
    )
    ours.atomic_json(science / "oracle_manifest.json", {**common, "frozen": True})
    ours.atomic_json(science / "evaluation_manifest.json", {**common, "frozen": True})
    names = [
        *ours._artifact_rows(metrics),
        "prefix_metrics.json",
        "summary.json",
        "oracle_manifest.json",
        "evaluation_manifest.json",
        "raw/candidate_universe.jsonl",
        "raw/calibration_pair_details.jsonl",
        "raw/selected_rules.jsonl",
        "raw/test_pair_details.jsonl",
        "raw/selection_manifest.json",
        "raw/test_access_receipt.json",
        "raw/test_evaluation_manifest.json",
    ]
    inventory = {
        name: {
            "sha256": ours.sha256_file(science / name),
            "bytes": (science / name).stat().st_size,
        }
        for name in names
    }
    ours.atomic_json(science / "freeze_manifest.json", {"files": inventory})
    ours.atomic_json(
        science / "run_manifest.json",
        {
            **common,
            "status": "SEALED",
            "worker_wrote_pass": False,
            "freeze_manifest_sha256": ours.sha256_file(science / "freeze_manifest.json"),
            "raw_output_complete": True,
            "source_artifacts_complete": True,
            "frozen": True,
        },
    )
    (science / "SEALED").write_text("SEALED\n", encoding="utf-8")
    final = tmp_path / "final"
    manifest = ours.verify_and_publish(
        science_root=science,
        final_root=final,
        threshold_contract=threshold_path,
    )
    assert manifest["status"] == "PASS"
    assert (final / "PASS").read_text(encoding="utf-8").strip() == ours.PASS_MARKER
    audit = ours.read_json(final / "final_artifact_audit.json")
    assert audit["audit_passed"] is True
    assert ours.read_json(final / "summary.json")["status"] == "PASS"
    registry = audit_explicit_candidate(final, dataset="TasteMolNet", method="Ours")
    assert registry.status in PASS_STATUSES, registry.reason_codes


def test_worker_and_verifier_are_distinct_and_not_disabled() -> None:
    wrapper = Path("scripts/autodl/run_tastemolnet_ours_full.sh").read_text(encoding="utf-8")
    assert "train_tastemolnet_ours_full.py" in wrapper
    assert wrapper.count("run_tastemolnet_ours_full.py") == 2
    assert "--verify-only" in wrapper
    assert "DISABLED" not in wrapper
