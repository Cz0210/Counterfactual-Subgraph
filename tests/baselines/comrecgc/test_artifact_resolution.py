from __future__ import annotations

import json
from pathlib import Path

from src.baselines.comrecgc.artifact_resolution import resolve_recovery_artifacts
from src.baselines.comrecgc.contracts import UPSTREAM_COMMIT, sha256_file


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_resolution_uses_exact_statistics_and_hashes(tmp_path: Path) -> None:
    aids = tmp_path / "native/aids/run"
    aids.mkdir(parents=True)
    (aids / "counterfactuals.pt").write_bytes(b"aids")
    dump(
        aids / "native_common_recourse_failure.json",
        {
            "model_counterfactual_candidate_count": 31,
            "distance_pair_count": 1984,
            "theta_eligible_pair_count": 28,
            "dbscan_cluster_count": 0,
            "selected_common_recourse_count": 0,
        },
    )
    base = tmp_path / "mutagenicity/smoke/run"
    generation = base / "generation"
    generation.mkdir(parents=True)
    counterfactuals = generation / "counterfactuals.pt"
    counterfactuals.write_bytes(b"mut")
    dump(
        generation / "run_manifest.json",
        {
            "dataset": "mutagenicity",
            "route": "project_adapted",
            "mode": "smoke",
            "run_complete": True,
            "parent_limit": 64,
            "counterfactual_candidate_count": 164,
            "upstream_commit": UPSTREAM_COMMIT,
            "counterfactuals_path": str(counterfactuals),
            "counterfactuals_sha256": sha256_file(counterfactuals),
        },
    )
    common = base / "common_recourse"
    common.mkdir()
    (common / "selected_common_recourses.json").write_text("[]", encoding="utf-8")
    (common / "representative_counterfactuals.pt").write_bytes(b"graphs")
    dump(
        common / "run_manifest.json",
        {
            "dataset": "mutagenicity",
            "mode": "smoke",
            "run_complete": True,
            "model_counterfactual_candidate_count": 70,
            "distance_pair_count": 4480,
            "theta_eligible_pair_count": 90,
            "dbscan_cluster_count": 4,
            "common_recourse_count": 4,
        },
    )

    output = tmp_path / "resolution.json"
    result = resolve_recovery_artifacts(outputs_root=tmp_path, output_path=output)

    assert result["resolution_passed"] is True
    assert result["selected"]["aids_native"]["counterfactuals_sha256"] == sha256_file(
        aids / "counterfactuals.pt"
    )
    assert result["selected"]["mutagenicity_generation"]["counterfactuals_sha256"] == sha256_file(
        counterfactuals
    )
