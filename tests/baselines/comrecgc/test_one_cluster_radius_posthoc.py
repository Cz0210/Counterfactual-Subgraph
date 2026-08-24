from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
torch = pytest.importorskip("torch")

from src.baselines.comrecgc import external_memory_recourse  # noqa: E402
from src.baselines.comrecgc.external_memory_dbscan import (  # noqa: E402
    ALL_CORE_ONE_COMPONENT_SHORTCUT,
    ExternalDBSCANContract,
    ExternalMemoryDBSCANError,
    _rss_bytes,
    fit_external_memory_dbscan,
)
from src.baselines.comrecgc.external_memory_recourse import (  # noqa: E402
    summarize_proven_one_cluster_external,
)
from src.baselines.comrecgc.one_cluster_radius_posthoc import (  # noqa: E402
    run_one_cluster_radius_posthoc_audit,
)


def _save(path: Path, values: np.ndarray) -> Path:
    with path.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    return path


def _greedy(counterfactual_covering, graphs_covered_by, k):
    del graphs_covered_by
    selected = {}
    covered = set()
    for rank in range(1, min(k, len(counterfactual_covering)) + 1):
        label = min(
            counterfactual_covering,
            key=lambda value: (
                -len(counterfactual_covering[value] - covered),
                value,
            ),
        )
        covered.update(counterfactual_covering.pop(label))
        selected[rank] = (label, len(covered))
    return selected


def _terminal_summary(
    tmp_path: Path,
    values: np.ndarray,
    *,
    monkeypatch: pytest.MonkeyPatch | None = None,
    historical_widened: bool,
):
    vectors_path = _save(tmp_path / "vectors.npy", values)
    pairs_path = _save(
        tmp_path / "pairs.npy",
        np.column_stack(
            (
                np.arange(len(values), dtype=np.int64),
                np.arange(len(values), dtype=np.int64) + 10,
            )
        ),
    )
    dbscan = fit_external_memory_dbscan(
        vectors_path=vectors_path,
        work_dir=tmp_path / "dbscan",
        contract=ExternalDBSCANContract(
            eps=0.1,
            min_samples=3,
            query_block_size=2,
            checkpoint_interval_blocks=1,
            max_rss_bytes=_rss_bytes() + 1024**3,
            expected_sklearn_version=sklearn.__version__,
            shortcut_mode=ALL_CORE_ONE_COMPONENT_SHORTCUT,
            shortcut_anchor_count=len(values),
            shortcut_query_block_size=2,
            exact_fallback_max_samples=0,
        ),
    )
    if historical_widened:
        assert monkeypatch is not None

        def widened(values, *, center, radius):
            distances = np.linalg.norm(values - center, axis=1)
            return distances.astype(np.float64, copy=False) < float(radius)

        monkeypatch.setattr(
            external_memory_recourse, "_numpy_trace_membership", widened
        )
    summary = summarize_proven_one_cluster_external(
        work_dir=tmp_path / "summary",
        dbscan_manifest_path=dbscan.manifest_path,
        dbscan_manifest_sha256=dbscan.manifest_sha256,
        recourse_vectors=np.load(vectors_path, mmap_mode="r", allow_pickle=False),
        pair_indices=np.load(pairs_path, mmap_mode="r", allow_pickle=False),
        pairs_sha256=external_memory_recourse._sha256_file(pairs_path),
        radius=0.02,
        theta=0.1,
        recourse_size=100,
        official_greedy=_greedy,
        torch_module=torch,
        max_rss_bytes=_rss_bytes() + 1024**3,
        block_size=2,
    )
    if historical_widened:
        monkeypatch.undo()
    return summary


def test_one_cluster_membership_excludes_exact_float32_delta() -> None:
    values = np.asarray([[-0.02], [0.02]], dtype=np.float32)
    center = np.mean(values, axis=0)
    assert external_memory_recourse._numpy_trace_membership(
        values, center=center, radius=0.02
    ).tolist() == [False, False]


def test_posthoc_exact_delta_blocks_live_adoption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = _terminal_summary(
        tmp_path,
        np.asarray([[-0.02], [-0.02], [0.02], [0.02]], dtype=np.float32),
        monkeypatch=monkeypatch,
        historical_widened=True,
    )
    output = tmp_path / "posthoc"
    result = run_one_cluster_radius_posthoc_audit(
        terminal_manifest_path=summary.manifest_path,
        expected_terminal_manifest_sha256=summary.manifest_sha256,
        output_dir=output,
    )
    assert result["status"] == "BLOCKED_BOUNDARY_DIFF"
    assert result["old_vs_dtype_cast_diff_count"] == 4
    assert result["old_widened"]["retained_count"] == 4
    assert result["dtype_cast"]["retained_count"] == 0
    assert result["official_torch"]["retained_count"] == 0
    assert result["old_vs_dtype_cast_parent_sets_equal"] is False
    assert result["old_vs_dtype_cast_candidate_sets_equal"] is False
    assert result["old_vs_dtype_cast_medoid_equal"] is False
    assert result["old_vs_dtype_cast_selected_trace_equal"] is False
    assert result["live_output_adoptable"] is False
    assert result["recommended_action"] == (
        "fresh_downstream_only_replay_from_existing_dbscan_manifest"
    )
    assert not (output / "PASS").exists()
    assert (output / "BLOCKED").read_text() == "BLOCKED_BOUNDARY_DIFF\n"
    corrected = json.loads((output / "corrected_downstream_trace.json").read_text())
    assert corrected["selected"] == []
    assert corrected["dbscan_recomputed"] is False


def test_posthoc_no_diff_adopts_without_dbscan_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = _terminal_summary(
        tmp_path,
        np.asarray([[-0.01], [-0.01], [0.01], [0.01]], dtype=np.float32),
        monkeypatch=monkeypatch,
        historical_widened=True,
    )
    output = tmp_path / "posthoc"
    result = run_one_cluster_radius_posthoc_audit(
        terminal_manifest_path=summary.manifest_path,
        expected_terminal_manifest_sha256=summary.manifest_sha256,
        output_dir=output,
    )
    assert result["status"] == "PASS"
    assert result["old_vs_dtype_cast_diff_count"] == 0
    assert result["live_output_adoptable"] is True
    assert result["old_vs_dtype_cast_parent_sets_equal"] is True
    assert result["old_vs_dtype_cast_candidate_sets_equal"] is True
    assert result["old_vs_dtype_cast_medoid_equal"] is True
    assert result["old_vs_dtype_cast_selected_trace_equal"] is True
    assert result["recommended_action"] == (
        "adopt_live_terminal_without_dbscan_rerun"
    )
    assert (output / "PASS").read_text() == "PASS\n"
    assert not (output / "BLOCKED").exists()


def test_posthoc_rejects_terminal_manifest_hash_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = _terminal_summary(
        tmp_path,
        np.asarray([[-0.01], [-0.01], [0.01], [0.01]], dtype=np.float32),
        monkeypatch=monkeypatch,
        historical_widened=True,
    )
    with pytest.raises(
        ExternalMemoryDBSCANError,
        match="terminal one-cluster manifest SHA mismatch",
    ):
        run_one_cluster_radius_posthoc_audit(
            terminal_manifest_path=summary.manifest_path,
            expected_terminal_manifest_sha256="0" * 64,
            output_dir=tmp_path / "posthoc",
        )
    assert not (tmp_path / "posthoc").exists()
