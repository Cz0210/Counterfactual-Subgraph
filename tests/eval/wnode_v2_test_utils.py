from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.mutagenicity_wnode_selector import MatrixData


def matrix_data(tmp_path: Path, *, parents: int = 10, candidates: int = 20) -> MatrixData:
    root = tmp_path / "calibration_matrix"
    root.mkdir(parents=True, exist_ok=True)
    parent_ids = tuple(f"p{index:02d}" for index in range(parents))
    fragments = ["C" * (index + 1) for index in range(candidates)]
    candidate_rows = tuple(
        {
            "candidate_id": f"c{index:02d}",
            "canonical_fragment": fragment,
            "source_parent_count": 1,
            "source_parent_ids": [f"source{index:02d}"],
            "source_cf_drop_mean": 0.5,
            "source_atom_ratio_mean": 0.2,
            "source_projection_used_rate": 0.0,
            "source_direct_substructure_rate": 1.0,
        }
        for index, fragment in enumerate(fragments)
    )
    distances = np.full((parents, candidates), np.inf, dtype=np.float64)
    for parent in range(parents):
        distances[parent, parent % candidates] = 0.01
        distances[parent, (parent + 1) % candidates] = 0.03
    cf_drops = np.where(np.isfinite(distances), 0.6, np.nan)
    applicable = np.isfinite(distances).copy()
    pair_rows = []
    for parent_index, parent_id in enumerate(parent_ids):
        for candidate_index, candidate in enumerate(candidate_rows):
            value = distances[parent_index, candidate_index]
            pair_rows.append(
                {
                    "parent_id": parent_id,
                    "parent_smiles": "CCO",
                    "candidate_id": candidate["candidate_id"],
                    "applicable": bool(applicable[parent_index, candidate_index]),
                    "pair_strict_flip": bool(np.isfinite(value)),
                    "wnode_distance": float(value) if np.isfinite(value) else None,
                    "cf_drop": 0.6 if np.isfinite(value) else None,
                }
            )
    (root / "pair_matrix.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in pair_rows), encoding="utf-8"
    )
    (root / "selected_candidate_universe.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in candidate_rows), encoding="utf-8"
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "parent_count": parents,
                "selected_candidate_count": candidates,
                "strict_flip_pair_count": int(np.count_nonzero(np.isfinite(distances))),
                "test_loaded": False,
            }
        ),
        encoding="utf-8",
    )
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "test_loaded": False,
                "inputs": {
                    "cohort_name": "calibration",
                    "candidate_pool": {"path": "/frozen/pool.jsonl", "sha256": "a" * 64},
                    "teacher_path": {"path": "/frozen/teacher.pkl", "sha256": "b" * 64},
                    "molclr_checkpoint": {"path": "/frozen/model.pth", "sha256": "c" * 64},
                },
            }
        ),
        encoding="utf-8",
    )
    return MatrixData(
        matrix_run_dir=root,
        parent_ids=parent_ids,
        candidate_rows=candidate_rows,
        distances=distances,
        cf_drops=cf_drops,
        applicable=applicable,
        full_finite_distances=distances[np.isfinite(distances)],
        full_parent_count=parents,
        full_candidate_count=candidates,
        full_pair_count=parents * candidates,
        full_strict_flip_pair_count=int(np.count_nonzero(np.isfinite(distances))),
        summary={"test_loaded": False},
        manifest={"test_loaded": False, "inputs": {"cohort_name": "calibration"}},
        full_candidate_rows=candidate_rows,
    )


def threshold_manifest(path: Path) -> Path:
    payload = {
        "schema_version": "bace_wnode_thresholds_v1",
        "dataset": "BACE",
        "distance_line": "MolCLR-Node-Wasserstein",
        "distance_type": "node_wasserstein",
        "cf_mode": "strict_flip",
        "quantiles": [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90],
        "thresholds": [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05],
        "theta_star_quantile": 0.30,
        "theta_star": 0.02,
        "cost_cap_quantile": 0.90,
        "cost_cap": 0.05,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
