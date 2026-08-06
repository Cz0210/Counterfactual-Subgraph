"""Resolve frozen COMRECGC blockers by manifest statistics, never by name alone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import UPSTREAM_COMMIT, sha256_file, write_json


def _json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def resolve_recovery_artifacts(
    *, outputs_root: str | Path, output_path: str | Path
) -> dict[str, Any]:
    root = Path(outputs_root).expanduser().resolve()
    aids_matches: list[dict[str, Any]] = []
    mut_generation_matches: list[dict[str, Any]] = []
    mut_common_matches: list[dict[str, Any]] = []
    for path in root.rglob("*.json"):
        value = _json(path)
        if value is None:
            continue
        if (
            int(value.get("model_counterfactual_candidate_count", -1)) == 31
            and int(value.get("distance_pair_count", -1)) == 1984
            and int(value.get("theta_eligible_pair_count", -1)) == 28
            and int(value.get("dbscan_cluster_count", -1)) == 0
            and int(value.get("selected_common_recourse_count", -1)) == 0
        ):
            artifact = path.parent / "counterfactuals.pt"
            if artifact.is_file():
                aids_matches.append(
                    {
                        "evidence_path": str(path),
                        "counterfactuals_path": str(artifact),
                        "counterfactuals_bytes": artifact.stat().st_size,
                        "counterfactuals_sha256": sha256_file(artifact),
                        "statistics": value,
                    }
                )
        if (
            value.get("dataset") == "mutagenicity"
            and value.get("route") == "project_adapted"
            and value.get("mode") == "smoke"
            and value.get("run_complete") is True
            and int(value.get("parent_limit", -1)) == 64
            and int(value.get("counterfactual_candidate_count", -1)) == 164
            and value.get("upstream_commit") == UPSTREAM_COMMIT
        ):
            artifact = Path(str(value.get("counterfactuals_path", "")))
            if not artifact.is_absolute():
                artifact = path.parent / artifact
            if artifact.is_file() and sha256_file(artifact) == value.get("counterfactuals_sha256"):
                mut_generation_matches.append(
                    {
                        "manifest_path": str(path),
                        "counterfactuals_path": str(artifact.resolve()),
                        "counterfactuals_bytes": artifact.stat().st_size,
                        "counterfactuals_sha256": sha256_file(artifact),
                        "project_commit": value.get("project_commit"),
                        "dataset_fingerprint": (value.get("dataset_audit") or {}).get(
                            "dataset_fingerprint"
                        ),
                        "generation_parent_ids_sha256": value.get(
                            "generation_parent_ids_sha256"
                        ),
                    }
                )
        if (
            value.get("dataset") == "mutagenicity"
            and value.get("mode") == "smoke"
            and value.get("run_complete") is True
            and int(value.get("model_counterfactual_candidate_count", -1)) == 70
            and int(value.get("distance_pair_count", -1)) == 4480
            and int(value.get("theta_eligible_pair_count", -1)) == 90
            and int(value.get("dbscan_cluster_count", -1)) == 4
            and int(value.get("common_recourse_count", -1)) == 4
        ):
            selected = path.parent / "selected_common_recourses.json"
            representatives = path.parent / "representative_counterfactuals.pt"
            if selected.is_file() and representatives.is_file():
                mut_common_matches.append(
                    {
                        "manifest_path": str(path),
                        "common_recourse_dir": str(path.parent),
                        "selected_common_recourses_sha256": sha256_file(selected),
                        "representative_counterfactuals_sha256": sha256_file(representatives),
                    }
                )
    aids_matches.sort(key=lambda row: row["evidence_path"])
    mut_generation_matches.sort(key=lambda row: row["manifest_path"])
    mut_common_matches.sort(key=lambda row: row["manifest_path"])
    if len(aids_matches) != 1:
        raise ValueError(f"AIDS blocker resolution is not unique: {len(aids_matches)} matches")
    compatible_pairs = [
        (generation, common)
        for generation in mut_generation_matches
        for common in mut_common_matches
        if str(common["manifest_path"]).startswith(str(Path(generation["manifest_path"]).parent.parent))
    ]
    if len(compatible_pairs) != 1:
        raise ValueError(
            "Mutagenicity blocker resolution is not unique: "
            f"generation={len(mut_generation_matches)}, common={len(mut_common_matches)}, "
            f"compatible={len(compatible_pairs)}"
        )
    generation, common = compatible_pairs[0]
    result = {
        "schema_version": 1,
        "resolution_passed": True,
        "selection_basis": "exact_statistics_plus_manifest_lineage_plus_sha256",
        "aids_candidates": aids_matches,
        "mutagenicity_generation_candidates": mut_generation_matches,
        "mutagenicity_common_recourse_candidates": mut_common_matches,
        "selected": {
            "aids_native": aids_matches[0],
            "mutagenicity_generation": generation,
            "mutagenicity_common_recourse": common,
        },
    }
    write_json(output_path, result)
    return result
