from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.baselines.comrecgc.close_pair_scan import (
    ClosePairScanError,
    cartesian_row,
    close_mask,
    normalize_distance_block,
    pair_for_cartesian_row,
    recourse_vector_formula,
    scan_theta_close_pairs,
)


def _chunks(*, parent_count: int, candidate_stops: tuple[int, ...]) -> list[dict]:
    values: list[dict] = []
    start = 0
    for index, stop in enumerate(candidate_stops):
        values.append(
            {
                "chunk_index": index,
                "row_count": (stop - start) * parent_count,
                "first_pair": [0, start],
                "last_pair": [parent_count - 1, stop - 1],
                "scientific_identity": {
                    "candidate_start": start,
                    "candidate_stop": stop,
                },
            }
        )
        start = stop
    return values


def _write_pairs(
    path: Path, *, parent_count: int, candidate_count: int, corrupt_row: int | None = None
) -> np.ndarray:
    values = np.asarray(
        [
            (parent, candidate)
            for candidate in range(candidate_count)
            for parent in range(parent_count)
        ],
        dtype=np.int64,
    )
    if corrupt_row is not None:
        values[corrupt_row] = values[corrupt_row, ::-1]
    np.save(path, values, allow_pickle=False)
    return values


def test_cartesian_pair_axis_contract_round_trips() -> None:
    for candidate in range(5):
        for parent in range(3):
            row = cartesian_row(
                parent_index=parent,
                candidate_index=candidate,
                parent_count=3,
            )
            assert pair_for_cartesian_row(row, parent_count=3) == (
                parent,
                candidate,
            )


def test_distance_normalization_preserves_frozen_float32_arithmetic() -> None:
    raw = np.asarray([[1.0, 2.0], [7.0, 11.0]], dtype=np.float32)
    candidate_counts = np.asarray([2, 5], dtype=np.int64)
    parent_counts = np.asarray([1, 3], dtype=np.int64)

    actual = normalize_distance_block(
        raw,
        candidate_element_counts=candidate_counts,
        parent_element_counts=parent_counts,
    )
    expected = raw / np.asarray([[3, 5], [6, 8]], dtype=np.float32)

    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected)


def test_close_filter_is_inclusive_self_contained_and_fail_closed() -> None:
    values = np.asarray(
        [[0.0, 0.1, np.nextafter(np.float32(0.1), np.float32(np.inf))],
         [np.nan, np.inf, 0.01]],
        dtype=np.float32,
    )

    assert close_mask(values, theta=0.1).tolist() == [
        [True, True, False],
        [False, False, True],
    ]
    with pytest.raises(ClosePairScanError, match="nonnegative"):
        close_mask(np.asarray([[-1e-6]], dtype=np.float32), theta=0.1)


def test_recourse_formula_uses_candidate_minus_parent_and_count_sum() -> None:
    candidate = np.asarray([4.0, 1.0], dtype=np.float32)
    parent = np.asarray([1.0, 7.0], dtype=np.float32)
    result = recourse_vector_formula(
        candidate,
        parent,
        candidate_element_count=2,
        parent_element_count=4,
    )
    assert np.array_equal(result, np.asarray([0.5, -1.0], dtype=np.float32))


def test_scan_materializes_exact_close_view_and_resumes(tmp_path: Path) -> None:
    parent_count = 3
    candidate_count = 4
    pair_path = tmp_path / "pairs.npy"
    _write_pairs(
        pair_path,
        parent_count=parent_count,
        candidate_count=candidate_count,
    )
    chunks = _chunks(parent_count=parent_count, candidate_stops=(2, 4))
    all_distances = np.asarray(
        [
            [0.0, 0.1, np.nextafter(np.float32(0.1), np.float32(np.inf))],
            [np.nan, 0.05, np.inf],
            [0.2, 0.03, 0.1],
            [0.09, 0.11, 0.04],
        ],
        dtype=np.float32,
    )

    def provider(start: int, stop: int) -> np.ndarray:
        return all_distances[start:stop]

    progress: list[dict] = []
    first = scan_theta_close_pairs(
        output_dir=tmp_path / "scan",
        pair_indices_path=pair_path,
        pair_chunks=chunks,
        parent_count=parent_count,
        candidate_count=candidate_count,
        theta=0.1,
        scientific_identity={"distance": "frozen"},
        distance_provider=provider,
        max_chunks=1,
        boundary_sample_size=4,
        progress_callback=lambda value: progress.append(dict(value)),
    )
    assert first is None
    assert progress[-1]["rows_processed"] == 6
    assert progress[-1]["logical_close_pair_count"] == 3

    result = scan_theta_close_pairs(
        output_dir=tmp_path / "scan",
        pair_indices_path=pair_path,
        pair_chunks=chunks,
        parent_count=parent_count,
        candidate_count=candidate_count,
        theta=0.1,
        scientific_identity={"distance": "frozen"},
        distance_provider=provider,
        resume=True,
        boundary_sample_size=4,
    )
    assert result is not None
    assert result.logical_close_pair_count == 7
    assert result.close_bitmap_path.name == "close_pair_bitmap.greed.uint8.npy"
    assert result.distance_path.name == "normalized_distances.greed.float32.npy"
    assert np.array_equal(
        np.load(result.close_bitmap_path, allow_pickle=False),
        np.isfinite(all_distances).reshape(-1)
        & (all_distances.reshape(-1) <= np.float32(0.1)),
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["physical_pair_count"] == 12
    assert manifest["logical_close_pair_count"] == 7
    assert manifest["distance_statistics"]["count_distance_eq_theta"] == 2
    assert manifest["distance_statistics"]["nonfinite_count"] == 2
    assert manifest["pair_axis"]["all_rows_checked"] is True
    assert len(manifest["boundary_rows"]) == 4

    terminal_resume = scan_theta_close_pairs(
        output_dir=tmp_path / "scan",
        pair_indices_path=pair_path,
        pair_chunks=chunks,
        parent_count=parent_count,
        candidate_count=candidate_count,
        theta=0.1,
        scientific_identity={"distance": "frozen"},
        distance_provider=lambda _start, _stop: pytest.fail(
            "terminal resume must not call the distance provider"
        ),
        resume=True,
        boundary_sample_size=4,
    )
    assert terminal_resume == result


def test_resume_rejects_corrupted_materialized_prefix(tmp_path: Path) -> None:
    pair_path = tmp_path / "pairs.npy"
    _write_pairs(pair_path, parent_count=2, candidate_count=2)
    chunks = _chunks(parent_count=2, candidate_stops=(1, 2))
    values = np.asarray([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32)
    assert (
        scan_theta_close_pairs(
            output_dir=tmp_path / "scan",
            pair_indices_path=pair_path,
            pair_chunks=chunks,
            parent_count=2,
            candidate_count=2,
            theta=0.1,
            scientific_identity={"checkpoint": "immutable"},
            distance_provider=lambda start, stop: values[start:stop],
            max_chunks=1,
        )
        is None
    )
    partial = np.load(
        tmp_path / "scan/normalized_distances.greed.float32.partial.npy",
        mmap_mode="r+",
        allow_pickle=False,
    )
    partial[0] = np.float32(0.09)
    partial.flush()

    with pytest.raises(ClosePairScanError, match="resume prefix differs"):
        scan_theta_close_pairs(
            output_dir=tmp_path / "scan",
            pair_indices_path=pair_path,
            pair_chunks=chunks,
            parent_count=2,
            candidate_count=2,
            theta=0.1,
            scientific_identity={"checkpoint": "immutable"},
            distance_provider=lambda start, stop: values[start:stop],
            resume=True,
        )


def test_scan_rejects_reversed_pair_columns_before_advancing_checkpoint(
    tmp_path: Path,
) -> None:
    pair_path = tmp_path / "pairs.npy"
    _write_pairs(
        pair_path,
        parent_count=3,
        candidate_count=2,
        corrupt_row=3,
    )
    chunks = _chunks(parent_count=3, candidate_stops=(2,))

    with pytest.raises(ClosePairScanError, match="physical pair axes differ"):
        scan_theta_close_pairs(
            output_dir=tmp_path / "scan",
            pair_indices_path=pair_path,
            pair_chunks=chunks,
            parent_count=3,
            candidate_count=2,
            theta=0.1,
            scientific_identity={"axis": "parent,candidate"},
            distance_provider=lambda _start, _stop: np.zeros(
                (2, 3), dtype=np.float32
            ),
        )
    checkpoint = json.loads(
        (tmp_path / "scan/checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["next_chunk_index"] == 0
    assert checkpoint["rows_processed"] == 0
