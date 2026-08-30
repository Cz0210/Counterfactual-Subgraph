from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.autodl import run_mutagenicity_exact_multicomponent_subset as subset
from src.baselines.comrecgc.external_memory_dbscan import _rss_bytes
from src.baselines.comrecgc.external_memory_recourse import ExternalPairStore


sklearn = pytest.importorskip("sklearn")


def test_production_derived_subset_uses_terminal_pair_store_and_exact_reload(
    tmp_path: Path,
) -> None:
    identity = {
        "dataset": "mutagenicity",
        "candidate_count": 6,
        "parent_count": 2,
    }
    store = ExternalPairStore(
        root=tmp_path / "pair-store",
        scientific_identity=identity,
        max_rss_bytes=_rss_bytes() + 512 * 1024**2,
    )
    values = np.zeros((12, 2), dtype=np.float32)
    values[:6, 0] = np.asarray(
        [0.0, 0.005, 0.010, 1.0, 1.005, 1.010], dtype=np.float32
    )
    values[6:, 0] = values[:6, 0]
    store.append(
        chunk_index=0,
        pairs=np.asarray(
            [[parent, candidate] for candidate in range(6) for parent in range(2)],
            dtype=np.int64,
        ),
        vectors=values,
        chunk_identity={"candidate_start": 0, "candidate_stop": 6},
    )
    source = store.finalize()

    result = subset.run_subset_gate(
        pair_store_manifest=source.manifest_path,
        output_dir=tmp_path / "subset-gate",
        subset_count=2,
        subset_size=6,
        expected_sklearn_version=sklearn.__version__,
    )

    assert result["status"] == "PASS"
    assert result["route"] == "sklearn_float64_exact_multi_component_v1"
    assert result["source_row_count"] == 12
    assert result["single_component_shortcut_used"] is False
    assert result["failure_cap_used"] is False
    assert all(row["labels_equal_sklearn_float64"] for row in result["subsets"])
    assert all(row["terminal_reload_equal"] for row in result["subsets"])
