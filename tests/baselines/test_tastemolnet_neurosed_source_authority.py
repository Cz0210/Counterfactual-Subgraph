from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.tastemolnet_neurosed_gedlib_build import (
    PINNED_GEDLIB_COMMIT,
    TasteGEDLIBBuildError,
    audit_preprovisioned_dependencies,
)


def test_gedlib_v1_commit_is_frozen_before_dependency_discovery(
    tmp_path: Path,
) -> None:
    assert PINNED_GEDLIB_COMMIT == "120856f670e013f080b116c0be4cc6bd72fc935d"
    with pytest.raises(TasteGEDLIBBuildError, match="v1.0 commit"):
        audit_preprovisioned_dependencies(
            gedlib_root=tmp_path / "missing-gedlib",
            expected_gedlib_commit="c" * 40,
            pybind11_cmake_dir=tmp_path / "missing-pybind11",
            cmake_executable="missing-cmake",
            cxx_executable="missing-cxx",
            python_executable="missing-python",
        )
