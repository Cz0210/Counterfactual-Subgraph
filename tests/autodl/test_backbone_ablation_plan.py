from __future__ import annotations

import argparse
import json

from scripts.autodl.run_backbone_ablation import BACKBONES, build


def _args(tmp_path, *, dataset="bace", enable=False, license_gate=None):
    split = tmp_path / "splits"
    split.mkdir()
    (split / "split_manifest.json").write_text("{}\n")
    return argparse.Namespace(
        dataset=dataset,
        backbone=list(BACKBONES),
        seed=7,
        split_root=split,
        output_root=tmp_path / "runs",
        output=tmp_path / "plan.json",
        license_gate=license_gate,
        candidate_pool_variant="primary",
        selector_variant="primary",
        reward_variant="primary",
        distance_variant="wnode",
        enable=enable,
    )


def test_ablation_plan_is_dormant_by_default(tmp_path):
    payload = build(_args(tmp_path))
    assert payload["enabled"] is False
    assert payload["tasks"] == []
    assert payload["axes"]["gnn_backbone"] == list(BACKBONES)


def test_enabled_bace_plan_preserves_test_leakage_boundary(tmp_path):
    payload = build(_args(tmp_path, enable=True))
    assert len(payload["tasks"]) == 4
    assert all(task["data_splits"] == ["train", "validation"] for task in payload["tasks"])
    assert all("attempt-{attempt}" in task["expected_output"] for task in payload["tasks"])


def test_taste_plan_stays_blocked_without_license(tmp_path):
    payload = build(_args(tmp_path, dataset="tastemolnet", enable=True))
    assert payload["enabled"] is False
    assert payload["blocker"] == "BLOCKED_LICENSE_REVIEW"
