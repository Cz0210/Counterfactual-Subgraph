"""Fail-closed isolated build smoke for the pinned GREED pyged wrapper.

The builder consumes only pre-provisioned, commit-pinned sources.  It never
clones, downloads, invokes pip/conda, or mutates the active project environment.
Missing GEDLIB/pybind11/compiler assets produce ``BLOCKED_GEDLIB_BUILD`` and
can never produce the build PASS marker.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


BUILD_SCHEMA = "tastemolnet_neurosed_gedlib_build_v2"
PINNED_GREED_COMMIT = "1c756f49625abb62c9f6de5b0059876a4c7499c1"
PINNED_GREED_EXPTS_COMMIT = "e85423dc943fda1979811e7449846efffec2a1e1"
PINNED_GEDLIB_COMMIT = "120856f670e013f080b116c0be4cc6bd72fc935d"
PINNED_SOURCE_SHA256 = {
    "neuro/datasets.py": "aa1bab19394b2fcad4d6f1c45c5206f0485cc098dbd4742bf1396d229c0fa1ad",
    "neuro/train.py": "8e4d425d9d63e0aa56d5a1e6e25738f511ca7b52b08ac297fcf2c1678bdf9e28",
    "neuro/models.py": "c5653dd9eeec1add8d6ae6253c30908df5ab8962ea0d9f9a6f25d32c393e0e70",
    "neuro/config.py": "cb34333a497c9627ee2f728cf45734162b78a6924e596b7cde88ef2788f66050",
    "pyged/src/pyged.cpp": "55b35f952ea4070fad430d0911d29bfca21b4e10926e9bd7d56d2515d6499b16",
    "pyged/CMakeLists.txt": "597f2f23252b0681d8de0d4c48cd4d10fad59d5c9130262fe2e7d3753737a010",
}
OFFICIAL_METHOD_NAME = ("f2",)
OFFICIAL_METHOD_ARGS_SINGLE_WORKER = ("--threads 1 --time-limit 1",)
GED_LABEL_BACKEND_VARIANT = "NON_MIP_GEDLIB"
NON_MIP_METHOD_CONFIGS: dict[str, str] = {
    "branch": "--threads 1",
}


class TasteGEDLIBBuildError(RuntimeError):
    """The isolated real-pyged build cannot be authenticated."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    value = result.stdout.strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise TasteGEDLIBBuildError(f"{path} does not expose a full Git commit")
    return value


def audit_pinned_greed_sources(
    greed_root: str | Path,
    greed_expts_root: str | Path,
) -> dict[str, Any]:
    """Authenticate the exact official sources used by this project."""

    greed = Path(greed_root).resolve()
    expts = Path(greed_expts_root).resolve()
    if _git_commit(greed) != PINNED_GREED_COMMIT:
        raise TasteGEDLIBBuildError("official GREED commit changed")
    if _git_commit(expts) != PINNED_GREED_EXPTS_COMMIT:
        raise TasteGEDLIBBuildError("official GREED experiments commit changed")
    observed: dict[str, str] = {}
    for relative, expected in PINNED_SOURCE_SHA256.items():
        source = greed / relative
        if not source.is_file() or source.is_symlink():
            raise TasteGEDLIBBuildError(f"pinned GREED source is absent: {relative}")
        observed[relative] = sha256_file(source)
        if observed[relative] != expected:
            raise TasteGEDLIBBuildError(f"pinned GREED source changed: {relative}")
    notebook = expts / "nbs_train" / "AIDS.ipynb"
    notebook_sha = sha256_file(notebook)
    if notebook_sha != "49a7bc0095d879bf49454cd6c18e42bb687c149a32e425b59c2acbe6c2df0114":
        raise TasteGEDLIBBuildError("official AIDS training notebook changed")
    return {
        "official_greed_commit": PINNED_GREED_COMMIT,
        "official_greed_expts_commit": PINNED_GREED_EXPTS_COMMIT,
        "source_sha256": observed,
        "aids_training_notebook_sha256": notebook_sha,
        "official_pair_builder_signature": (
            "neuro.datasets.make_inner_dataset(graphs,n_pairs,n_hops_query,"
            "trav_prob_query,node_lim_query=None,n_hops_target=None,targets=None)"
        ),
        "official_selector_contract": (
            "validation_batch_before_each_training_batch; consecutive batch "
            "non-improvement stop > cycle_patience*(step_size_up+step_size_down)"
        ),
        "official_ged_method": "f2",
        "official_ged_method_args_single_worker": "--threads 1 --time-limit 1",
    }


def _require_git_commit(path: Path, expected: str, *, label: str) -> str:
    if len(expected) != 40 or any(character not in "0123456789abcdef" for character in expected):
        raise TasteGEDLIBBuildError(f"{label} expected commit must be a full SHA")
    observed = _git_commit(path)
    if observed != expected:
        raise TasteGEDLIBBuildError(f"{label} commit changed")
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if status.stdout:
        raise TasteGEDLIBBuildError(f"{label} checkout is dirty")
    return observed


def audit_preprovisioned_dependencies(
    *,
    gedlib_root: str | Path,
    expected_gedlib_commit: str,
    pybind11_cmake_dir: str | Path,
    cmake_executable: str,
    cxx_executable: str,
    python_executable: str,
) -> dict[str, Any]:
    """Check all offline dependencies before creating an isolated build root."""

    if expected_gedlib_commit != PINNED_GEDLIB_COMMIT:
        raise TasteGEDLIBBuildError(
            "GEDLIB must match the v1.0 commit prescribed by pinned GREED"
        )
    gedlib = Path(gedlib_root).resolve()
    pybind_cmake = Path(pybind11_cmake_dir).resolve()
    missing: list[str] = []
    for relative in (
        "src/env/ged_env.hpp",
        "ext/boost_1_82_0",
        "ext/eigen.3.3.4/Eigen",
        "ext/nomad.3.8.1/src",
        "ext/nomad.3.8.1/ext/sgtelib/src",
        "ext/nomad.3.8.1/lib",
        "ext/lsape.5/include",
        "ext/libsvm.3.22",
        "ext/fann.2.2.0/include",
        "ext/fann.2.2.0/lib",
    ):
        if not (gedlib / relative).exists():
            missing.append(f"gedlib/{relative}")
    pybind_config = pybind_cmake / "pybind11Config.cmake"
    pybind_version = pybind_cmake / "pybind11ConfigVersion.cmake"
    if not pybind_cmake.is_dir() or not pybind_config.is_file() or not pybind_version.is_file():
        missing.append("pybind11_cmake_dir")
    executables: dict[str, str] = {}
    for label, requested in (
        ("cmake", cmake_executable),
        ("cxx", cxx_executable),
        ("python", python_executable),
    ):
        resolved = shutil.which(requested) if not Path(requested).is_absolute() else requested
        if not resolved or not Path(resolved).is_file():
            missing.append(f"{label}_executable")
        else:
            executables[label] = str(Path(resolved).resolve())
    if missing:
        raise TasteGEDLIBBuildError(
            "pre-provisioned offline build dependencies are absent: " + ", ".join(missing)
        )
    gedlib_commit = _require_git_commit(
        gedlib, expected_gedlib_commit, label="GEDLIB"
    )
    versions: dict[str, str] = {}
    for label, argv in (
        ("cmake", [executables["cmake"], "--version"]),
        ("cxx", [executables["cxx"], "--version"]),
        ("python", [executables["python"], "--version"]),
    ):
        result = subprocess.run(
            argv,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        versions[label] = (result.stdout or result.stderr).splitlines()[0]
    return {
        "gedlib_root": str(gedlib),
        "gedlib_commit": gedlib_commit,
        "gedlib_version": f"v1.0@{PINNED_GEDLIB_COMMIT}",
        "pybind11_cmake_dir": str(pybind_cmake),
        "pybind11_cmake_sha256": {
            "pybind11Config.cmake": sha256_file(pybind_config),
            "pybind11ConfigVersion.cmake": sha256_file(pybind_version),
        },
        "executables": executables,
        "versions": versions,
        "network_install_performed": False,
    }


def _offline_cmake_text(original: str, *, gedlib_root: Path) -> str:
    """Create a Gurobi-free build overlay for the pinned non-MIP methods."""

    source_lines = original.splitlines()
    old_include = "\t${CMAKE_SOURCE_DIR}/ext/gedlib"
    root_indexes = [
        index for index, line in enumerate(source_lines) if line == old_include
    ]
    if len(root_indexes) != 1:
        raise TasteGEDLIBBuildError("official CMake GEDLIB include anchor changed")
    source_lines[root_indexes[0]] = f"\t{gedlib_root}"
    text = "\n".join(source_lines) + "\n"
    text = text.replace("${CMAKE_SOURCE_DIR}/ext/gedlib/", f"{gedlib_root}/")
    lines = []
    for line in text.splitlines():
        if "gurobi911" in line or line.strip() in {"gurobi_c++", "gurobi91"}:
            continue
        lines.append(line)
    result = "\n".join(lines) + "\n"
    if "${CMAKE_SOURCE_DIR}/ext/gedlib" in result:
        raise TasteGEDLIBBuildError(
            "legacy GEDLIB include reference remains in offline overlay"
        )
    if "gurobi" in result.lower():
        raise TasteGEDLIBBuildError("Gurobi link reference remains in offline overlay")
    return result


def _non_mip_cpp_text(original: str) -> str:
    """Remove Gurobi-only mappings and retain deterministic pinned candidates.

    The pinned wrapper defines ``GUROBI`` itself and consequently exposes F2
    and BLP enum members even on hosts with no licensed runtime.  The build
    overlay must remove that define *and* the two mapper branches; deleting the
    define alone leaves uncompilable enum references behind.
    """

    if original.count("#define GUROBI") != 1:
        raise TasteGEDLIBBuildError("official pyged Gurobi define anchor changed")
    result = original.replace(
        "#define GUROBI",
        "// GUROBI intentionally disabled: NON_MIP_GEDLIB label backend",
    )
    for official_method in (
        "anchor_aware_ged",
        "blp_no_edge_labels",
        "branch",
        "f2",
        "ipfp",
    ):
        if original.count(f'name == "{official_method}"') != 1:
            raise TasteGEDLIBBuildError("official pyged method mapper changed")
    mapper = re.compile(
        r"ged::Options::GEDMethod method_name_to_option\(std::string name\)\s*"
        r"\{.*?\n\}",
        re.DOTALL,
    )
    result, replacements = mapper.subn(
        """ged::Options::GEDMethod method_name_to_option(std::string name)
{
\tif (name == \"branch\") {
\t\treturn ged::Options::GEDMethod::BRANCH;
\t} else {
\t\tthrow std::invalid_argument(\"unknown method\");
\t}
}""",
        result,
        count=1,
    )
    if replacements != 1:
        raise TasteGEDLIBBuildError("official pyged method mapper changed")
    f2_quick_fix = re.compile(
        r'if \(method_name\[0\] == "ged_f2"\) \{\s*'
        r'env\.set_edit_costs\(new GEDEditCosts\(\)\);\s*'
        r'method_name\[0\] = "f2";\s*'
        r'\} else if \(method_name\[0\] == "ged_branch"\) \{'
    )
    result, replacements = f2_quick_fix.subn(
        'if (method_name[0] == "ged_branch") {', result, count=1
    )
    if replacements != 1:
        raise TasteGEDLIBBuildError("official pyged GED/F2 quick-fix anchor changed")
    forbidden = (
        "#define GUROBI",
        "Options::GEDMethod::F2",
        "Options::GEDMethod::BLP_NO_EDGE_LABELS",
        "Options::GEDMethod::ANCHOR_AWARE_GED",
        "Options::GEDMethod::IPFP",
        'method_name[0] = "f2"',
    )
    if any(token in result for token in forbidden):
        raise TasteGEDLIBBuildError("Gurobi-only pyged method remains in overlay")
    for method in NON_MIP_METHOD_CONFIGS:
        if result.count(f'name == "{method}"') != 1:
            raise TasteGEDLIBBuildError(
                f"pinned non-MIP pyged method changed: {method}"
            )
    return result


def _run(
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout: int,
) -> dict[str, Any]:
    result = subprocess.run(
        list(argv),
        cwd=cwd,
        env=dict(environment),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    payload = {
        "argv": list(argv),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-8000:],
        "stderr_tail": result.stderr[-8000:],
    }
    if result.returncode != 0:
        raise TasteGEDLIBBuildError(
            f"isolated build command failed: {argv[0]} rc={result.returncode}"
        )
    return payload


def _real_pyged_smoke(lib_dir: Path) -> dict[str, Any]:
    candidates = sorted(lib_dir.glob("pyged*.so"))
    if len(candidates) != 1:
        raise TasteGEDLIBBuildError("isolated build did not produce exactly one pyged .so")
    module_path = candidates[0].resolve()
    previous = sys.modules.pop("pyged", None)
    sys.path.insert(0, str(lib_dir))
    try:
        module = importlib.import_module("pyged")
        imported = Path(module.__file__).resolve()
        if imported != module_path:
            raise TasteGEDLIBBuildError("pyged imported outside isolated build root")
        method_smokes: dict[str, dict[str, Any]] = {}
        for method, method_args in NON_MIP_METHOD_CONFIGS.items():
            equal_lb, equal_ub = module.sed(
                ([0], []), ([0], []), [method], [method_args]
            )
            small_lb, small_ub = module.sed(
                ([0], []),
                ([0, 0], [(0, 1), (1, 0)]),
                [method],
                [method_args],
            )
            reverse_lb, reverse_ub = module.sed(
                ([0, 0], [(0, 1), (1, 0)]),
                ([0], []),
                [method],
                [method_args],
            )
            values = [equal_lb, equal_ub, small_lb, small_ub, reverse_lb, reverse_ub]
            if any(not math.isfinite(float(value)) for value in values):
                raise TasteGEDLIBBuildError(
                    f"pyged {method} smoke returned a non-finite bound"
                )
            if not (
                float(equal_lb) == 0.0
                and float(equal_ub) == 0.0
                and 0.0 <= float(small_lb) <= float(small_ub)
                and float(small_ub) == 0.0
                and 0.0 <= float(reverse_lb) <= float(reverse_ub)
                and float(reverse_ub) > 0.0
            ):
                raise TasteGEDLIBBuildError(
                    f"pyged {method} smoke violates pinned SED direction/bounds"
                )
            method_smokes[method] = {
                "method_args": method_args,
                "equal_bounds": [float(equal_lb), float(equal_ub)],
                "zero_insertion_bounds": [float(small_lb), float(small_ub)],
                "positive_deletion_bounds": [float(reverse_lb), float(reverse_ub)],
                "finite_lower_le_upper": True,
                "sed_cost_direction_authenticated": True,
            }
    finally:
        sys.path.pop(0)
        sys.modules.pop("pyged", None)
        if previous is not None:
            sys.modules["pyged"] = previous
    return {
        "module_path": str(module_path),
        "module_sha256": sha256_file(module_path),
        "candidate_methods": list(NON_MIP_METHOD_CONFIGS),
        "method_smokes": method_smokes,
        "all_candidates_finite_lower_le_upper": True,
        "sed_cost_direction_authenticated": True,
    }


def isolated_build_smoke(
    *,
    greed_root: str | Path,
    greed_expts_root: str | Path,
    gedlib_root: str | Path,
    expected_gedlib_commit: str,
    pybind11_cmake_dir: str | Path,
    output_root: str | Path,
    cmake_executable: str = "cmake",
    cxx_executable: str = "c++",
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Build and execute the pinned real-pyged smoke in a fresh directory."""

    destination = Path(output_root)
    if not destination.is_absolute() or Path(os.path.abspath(destination)) != destination:
        raise TasteGEDLIBBuildError("isolated build root must be normalized absolute")
    if destination.exists():
        raise TasteGEDLIBBuildError("isolated build root already exists")
    source_authority = audit_pinned_greed_sources(greed_root, greed_expts_root)
    dependencies = audit_preprovisioned_dependencies(
        gedlib_root=gedlib_root,
        expected_gedlib_commit=expected_gedlib_commit,
        pybind11_cmake_dir=pybind11_cmake_dir,
        cmake_executable=cmake_executable,
        cxx_executable=cxx_executable,
        python_executable=python_executable,
    )
    destination.mkdir(parents=True, mode=0o700)
    source = destination / "source"
    build = source / "build"
    lib = destination / "lib"
    source.mkdir()
    build.mkdir()
    lib.mkdir()
    greed = Path(greed_root).resolve()
    cpp = (greed / "pyged" / "src" / "pyged.cpp").read_text(encoding="utf-8")
    patched_cpp = _non_mip_cpp_text(cpp)
    (source / "src").mkdir()
    (source / "src" / "pyged.cpp").write_text(patched_cpp, encoding="utf-8")
    cmake_text = (greed / "pyged" / "CMakeLists.txt").read_text(encoding="utf-8")
    overlay = _offline_cmake_text(
        cmake_text,
        gedlib_root=Path(dependencies["gedlib_root"]),
    )
    overlay = overlay.replace(
        "${CMAKE_SOURCE_DIR}/lib/", f"{lib}/"
    )
    (source / "CMakeLists.txt").write_text(overlay, encoding="utf-8")
    environment = dict(os.environ)
    environment.update(
        {
            "PIP_NO_INDEX": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "CXX": dependencies["executables"]["cxx"],
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    configure = _run(
        [
            dependencies["executables"]["cmake"],
            "-S",
            str(source),
            "-B",
            str(build),
            f"-Dpybind11_DIR={Path(pybind11_cmake_dir).resolve()}",
            f"-DPYTHON_EXECUTABLE={dependencies['executables']['python']}",
            f"-DPython_EXECUTABLE={dependencies['executables']['python']}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=destination,
        environment=environment,
        timeout=300,
    )
    compile_result = _run(
        [dependencies["executables"]["cmake"], "--build", str(build), "--parallel", "1"],
        cwd=destination,
        environment=environment,
        timeout=1800,
    )
    smoke = _real_pyged_smoke(lib)
    return {
        "schema_version": BUILD_SCHEMA,
        "status": "PASS",
        "marker": "[TASTE_NEUROSED_GEDLIB_BUILD_PASS]",
        "source_authority": source_authority,
        "dependencies": dependencies,
        "build_root": str(destination),
        "build_isolated_from_smiles_pip118": True,
        "network_install_performed": False,
        "GED_LABEL_BACKEND_VARIANT": GED_LABEL_BACKEND_VARIANT,
        "F2_BLP_USED": False,
        "GUROBI_USED": False,
        "ged_label_backend_variant": GED_LABEL_BACKEND_VARIANT,
        "f2_blp_used": False,
        "gurobi_used": False,
        "ged_method": None,
        "selected_ged_backend": None,
        "candidate_ged_backends": list(NON_MIP_METHOD_CONFIGS),
        "ged_method_switched_from_official": True,
        "build_flags": {
            "cmake_build_type": "Release",
            "target_compile_options": ["-fpermissive"],
            "parallel_compile_jobs": 1,
            "omp_num_threads": 1,
            "mkl_num_threads": 1,
            "openblas_num_threads": 1,
            "tokenizers_parallelism": False,
        },
        "build_overlay": {
            "mode": "pinned_pyged_deterministic_non_mip_methods_gurobi_disabled_v2",
            "patched_pyged_cpp_sha256": hashlib.sha256(
                patched_cpp.encode("utf-8")
            ).hexdigest(),
            "patched_cmake_sha256": hashlib.sha256(overlay.encode("utf-8")).hexdigest(),
        },
        "configure": configure,
        "compile": compile_result,
        "smoke": smoke,
    }


def blocked_build_manifest(
    *,
    error: BaseException,
    greed_root: str | Path,
    greed_expts_root: str | Path,
    gedlib_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Create a non-PASS diagnostic that cannot be mistaken for a build."""

    return {
        "schema_version": BUILD_SCHEMA,
        "status": "BLOCKED_GEDLIB_BUILD",
        "marker": None,
        "error_type": type(error).__name__,
        "error": str(error),
        "greed_root": str(greed_root),
        "greed_expts_root": str(greed_expts_root),
        "gedlib_root": str(gedlib_root),
        "requested_output_root": str(output_root),
        "real_pyged_smoke_passed": False,
        "GED_LABEL_BACKEND_VARIANT": GED_LABEL_BACKEND_VARIANT,
        "F2_BLP_USED": False,
        "GUROBI_USED": False,
        "approximate_or_neural_fallback_used": False,
        "network_install_performed": False,
    }


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "BUILD_SCHEMA",
    "GED_LABEL_BACKEND_VARIANT",
    "NON_MIP_METHOD_CONFIGS",
    "OFFICIAL_METHOD_ARGS_SINGLE_WORKER",
    "OFFICIAL_METHOD_NAME",
    "PINNED_GREED_COMMIT",
    "PINNED_GREED_EXPTS_COMMIT",
    "PINNED_GEDLIB_COMMIT",
    "PINNED_SOURCE_SHA256",
    "TasteGEDLIBBuildError",
    "atomic_write_json",
    "audit_pinned_greed_sources",
    "audit_preprovisioned_dependencies",
    "blocked_build_manifest",
    "isolated_build_smoke",
    "sha256_file",
]
