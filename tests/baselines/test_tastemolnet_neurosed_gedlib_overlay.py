from pathlib import Path

import pytest

from src.utils.tastemolnet_neurosed_gedlib_build import (
    NON_MIP_METHOD_CONFIGS,
    TasteGEDLIBBuildError,
    _non_mip_cpp_text,
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


def test_cpp_overlay_removes_gurobi_only_methods_and_retains_deterministic_candidates() -> None:
    source = """\
#define GUROBI
ged::Options::GEDMethod method_name_to_option(std::string name)
{
    if (name == "anchor_aware_ged") {
        return ged::Options::GEDMethod::ANCHOR_AWARE_GED;
    } else if (name == "blp_no_edge_labels") {
        return ged::Options::GEDMethod::BLP_NO_EDGE_LABELS;
    } else if (name == "branch") {
        return ged::Options::GEDMethod::BRANCH;
    } else if (name == "f2") {
        return ged::Options::GEDMethod::F2;
    } else if (name == "ipfp") {
        return ged::Options::GEDMethod::IPFP;
    }
}
void choose(std::vector<std::string> method_name, Env& env) {
    if (method_name[0] == "ged_f2") {
        env.set_edit_costs(new GEDEditCosts());
        method_name[0] = "f2";
    } else if (method_name[0] == "ged_branch") {
        env.set_edit_costs(new GEDEditCosts());
        method_name[0] = "branch";
    } else {
        env.set_edit_costs(new SEDEditCosts());
    }
}
"""

    overlay = _non_mip_cpp_text(source)

    assert "#define GUROBI" not in overlay
    assert "BLP_NO_EDGE_LABELS" not in overlay
    assert "Options::GEDMethod::F2" not in overlay
    assert 'method_name[0] = "f2"' not in overlay
    for method in NON_MIP_METHOD_CONFIGS:
        assert overlay.count(f'name == "{method}"') == 1


def test_cpp_overlay_fails_closed_when_one_gurobi_mapper_is_missing() -> None:
    with pytest.raises(TasteGEDLIBBuildError, match="method mapper changed"):
        _non_mip_cpp_text(
            "#define GUROBI\n"
            'if (name == "anchor_aware_ged") { return A; }\n'
            'else if (name == "branch") { return B; }\n'
            'else if (name == "f2") { return ged::Options::GEDMethod::F2; }\n'
            'else if (name == "ipfp") { return C; }\n'
        )
