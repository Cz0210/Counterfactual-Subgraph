from pathlib import Path

import pytest

from src.utils.tastemolnet_neurosed_gedlib_build import (
    TasteGEDLIBBuildError,
    _offline_cmake_text,
)


def test_offline_overlay_counts_only_the_unique_official_root_include() -> None:
    official_shape = """\
target_link_directories(pyged PUBLIC
\t${CMAKE_SOURCE_DIR}/ext/gedlib/ext/nomad.3.8.1/lib
\t${CMAKE_SOURCE_DIR}/ext/gedlib/ext/libsvm.3.22
\t${CMAKE_SOURCE_DIR}/ext/gedlib/ext/fann.2.2.0/lib
\t${CMAKE_SOURCE_DIR}/ext/gurobi911/linux64/lib
)
# The official standalone GEDLIB include follows linked subpaths.
target_include_directories(pyged PUBLIC
\t${CMAKE_SOURCE_DIR}/ext/gedlib
\t${CMAKE_SOURCE_DIR}/ext/gedlib/ext/boost_1_82_0
\t${CMAKE_SOURCE_DIR}/ext/gedlib/ext/eigen.3.3.4/Eigen
\t${CMAKE_SOURCE_DIR}/ext/gedlib/ext/nomad.3.8.1/src
\t${CMAKE_SOURCE_DIR}/ext/gedlib/ext/nomad.3.8.1/ext/sgtelib/src
\t${CMAKE_SOURCE_DIR}/ext/gedlib/ext/lsape.5/include
\t${CMAKE_SOURCE_DIR}/ext/gedlib/ext/libsvm.3.22
\t${CMAKE_SOURCE_DIR}/ext/gedlib/ext/fann.2.2.0/include
\t${CMAKE_SOURCE_DIR}/ext/gurobi911/linux64/include
)
target_link_libraries(pyged PUBLIC
\tdoublefann
\tsvm
\tnomad
\tgurobi_c++
\tgurobi91
)
"""
    root_prefix = "\t${CMAKE_SOURCE_DIR}/ext/gedlib"
    assert official_shape.count(root_prefix) == 11
    assert sum(line == root_prefix for line in official_shape.splitlines()) == 1

    gedlib = Path("/opt/taste/pinned-gedlib")
    overlay = _offline_cmake_text(official_shape, gedlib_root=gedlib)

    assert f"\t{gedlib}\n" in overlay
    for relative in (
        "ext/boost_1_82_0",
        "ext/eigen.3.3.4/Eigen",
        "ext/nomad.3.8.1/src",
        "ext/nomad.3.8.1/ext/sgtelib/src",
        "ext/lsape.5/include",
        "ext/libsvm.3.22",
        "ext/fann.2.2.0/include",
    ):
        assert f"\t{gedlib}/{relative}\n" in overlay
    for relative in (
        "ext/nomad.3.8.1/lib",
        "ext/libsvm.3.22",
        "ext/fann.2.2.0/lib",
    ):
        assert f"\t{gedlib}/{relative}\n" in overlay
    assert "${CMAKE_SOURCE_DIR}/ext/gedlib" not in overlay
    assert "gurobi" not in overlay.lower()


@pytest.mark.parametrize(
    "source",
    [
        "\t${CMAKE_SOURCE_DIR}/ext/gedlib/ext/eigen.3.3.4/Eigen\n",
        "\t${CMAKE_SOURCE_DIR}/ext/gedlib\n\t${CMAKE_SOURCE_DIR}/ext/gedlib\n",
    ],
)
def test_offline_overlay_rejects_missing_or_duplicate_exact_root_line(source: str) -> None:
    with pytest.raises(TasteGEDLIBBuildError, match="include anchor changed"):
        _offline_cmake_text(source, gedlib_root=Path("/opt/taste/pinned-gedlib"))


def test_offline_overlay_rejects_legacy_root_hidden_in_a_preceding_comment() -> None:
    source = """\
# previous anchor was \t${CMAKE_SOURCE_DIR}/ext/gedlib
target_include_directories(pyged PUBLIC
\t${CMAKE_SOURCE_DIR}/ext/gedlib
\t${CMAKE_SOURCE_DIR}/ext/gedlib/ext/eigen.3.3.4/Eigen
)
"""

    with pytest.raises(
        TasteGEDLIBBuildError,
        match="legacy GEDLIB include reference remains",
    ):
        _offline_cmake_text(source, gedlib_root=Path("/opt/taste/pinned-gedlib"))
