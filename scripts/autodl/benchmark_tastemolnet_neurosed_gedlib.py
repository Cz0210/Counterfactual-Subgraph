#!/usr/bin/env python3
"""Run one disjoint Taste pair cohort through authenticated pyged/GEDLIB."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.tastemolnet_neurosed_fixed_budget import (  # noqa: E402
    FixedBudgetGraph,
    sample_official_style_query,
)
from src.eval.tastemolnet_neurosed_fixed_budget import (  # noqa: E402
    OFFICIAL_SED_EDIT_COSTS,
    summarize_real_gedlib_observations,
)
from src.utils.tastemolnet_neurosed_gedlib_build import (  # noqa: E402
    BUILD_SCHEMA,
    PINNED_GREED_COMMIT,
    sha256_file,
)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory = os.open(path.parent, os.O_RDONLY)
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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise RuntimeError(f"{path} is not one JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if type(row) is not dict:
                raise RuntimeError(f"{path}:{line_number} is not one object")
            rows.append(row)
    return rows


def _worker(
    module_dir: str,
    query_data: tuple[list[int], list[tuple[int, int]]],
    target_data: tuple[list[int], list[tuple[int, int]]],
    method_args: list[str],
) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, module_dir)
    started = time.perf_counter()
    try:
        pyged = importlib.import_module("pyged")
        lower, upper = pyged.sed(query_data, target_data, ["f2"], method_args)
        elapsed = time.perf_counter() - started
        lower_value = float(lower)
        upper_value = float(upper)
        if (
            not math.isfinite(lower_value)
            or not math.isfinite(upper_value)
            or lower_value < 0
            or lower_value > upper_value
        ):
            raise RuntimeError("pyged returned invalid bounds")
        return {
            "status": "SUCCESS",
            "latency_seconds": elapsed,
            "lower_bound": lower_value,
            "upper_bound": upper_value,
            "exact_bound": lower_value == upper_value,
            "error": None,
        }
    except BaseException as exc:
        return {
            "status": "GEDLIB_ERROR",
            "latency_seconds": time.perf_counter() - started,
            "lower_bound": None,
            "upper_bound": None,
            "exact_bound": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        sys.path.pop(0)


def _graphs(rows: Iterable[Mapping[str, Any]]) -> dict[str, FixedBudgetGraph]:
    result: dict[str, FixedBudgetGraph] = {}
    for row in rows:
        graph = FixedBudgetGraph(
            graph_id=str(row["graph_id"]),
            split=str(row["split"]),
            node_labels=tuple(int(value) for value in row["node_labels"]),
            directed_edges=tuple(
                (int(edge[0]), int(edge[1])) for edge in row["directed_edges"]
            ),
            scaffold=str(row["scaffold"]),
            class_label=int(row["class_label_sampling_diagnostic_only"]),
        )
        if (
            graph.graph_sha256 != row.get("graph_sha256")
            or graph.canonical_graph_sha256 != row.get("canonical_graph_sha256")
            or graph.graph_id in result
        ):
            raise RuntimeError("graph inventory authority changed")
        result[graph.graph_id] = graph
    return result


def _iowait_ticks() -> int | None:
    path = Path("/proc/stat")
    if not path.is_file():
        return None
    first = path.read_text(encoding="utf-8").splitlines()[0].split()
    if first[0] != "cpu" or len(first) < 6:
        return None
    return int(first[5])


def _physical_core_count() -> int:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        sockets: set[tuple[str, str]] = set()
        physical = "0"
        core = ""
        for line in cpuinfo.read_text(encoding="utf-8").splitlines() + [""]:
            if not line.strip():
                if core:
                    sockets.add((physical, core))
                physical, core = "0", ""
            elif line.startswith("physical id"):
                physical = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core = line.split(":", 1)[1].strip()
        if sockets:
            return len(sockets)
    return int(os.cpu_count() or 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--pair-sampler-manifest", type=Path, required=True)
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--graph-inventory-jsonl", type=Path, required=True)
    parser.add_argument("--benchmark-budget", type=int, choices=(100, 500, 1000), required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--gedlib-time-limit-seconds", type=int, default=1)
    parser.add_argument("--hard-wall-seconds", type=float, required=True)
    parser.add_argument("--bace-legacy-throughput-drop-percent", type=float, required=True)
    parser.add_argument("--aids-exact-throughput-drop-percent", type=float, required=True)
    parser.add_argument("--host-load-gate-pass", action="store_true")
    parser.add_argument("--iowait-gate-pass", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gedlib_time_limit_seconds <= 0 or args.hard_wall_seconds <= 0:
        raise RuntimeError("GEDLIB time limits must be positive")
    if args.workers > _physical_core_count():
        raise RuntimeError("GEDLIB workers exceed physical core count")
    build = _load_json(args.build_manifest)
    smoke = build.get("smoke")
    source = build.get("source_authority")
    dependencies = build.get("dependencies")
    if (
        build.get("schema_version") != BUILD_SCHEMA
        or build.get("status") != "PASS"
        or build.get("marker") != "[TASTE_NEUROSED_GEDLIB_BUILD_PASS]"
        or build.get("network_install_performed") is not False
        or build.get("gurobi_used") is not False
        or build.get("ged_method") != "f2"
        or build.get("ged_method_switched_from_official") is not False
        or type(smoke) is not dict
        or type(source) is not dict
        or type(dependencies) is not dict
        or source.get("official_greed_commit") != PINNED_GREED_COMMIT
    ):
        raise RuntimeError("real pyged/GEDLIB build authority is not PASS")
    module = Path(str(smoke["module_path"])).resolve()
    if not module.is_file() or sha256_file(module) != smoke.get("module_sha256"):
        raise RuntimeError("isolated pyged module changed after build smoke")
    pair_manifest = _load_json(args.pair_sampler_manifest)
    pairs = _load_jsonl(args.pairs_jsonl)
    if (
        pair_manifest.get("schema_version")
        != "tastemolnet_neurosed_fixed_budget_pairs_v1"
        or pair_manifest.get("independent_query_target_pairs") is not True
        or pair_manifest.get("parent_own_subgraph_shortcut") is not False
        or pair_manifest.get("cartesian_product_materialized") is not False
        or pair_manifest.get("ged_labels_present") is not False
        or len(pairs) != args.benchmark_budget
        or len({row.get("pair_id") for row in pairs}) != len(pairs)
    ):
        raise RuntimeError("fixed-budget pair cohort authority changed")
    expected_cohort_sha = pair_manifest.get("benchmark_cohort_file_sha256", {}).get(
        str(args.benchmark_budget)
    )
    if expected_cohort_sha != sha256_file(args.pairs_jsonl):
        raise RuntimeError("disjoint benchmark pair file SHA256 changed")
    graph_rows = _load_jsonl(args.graph_inventory_jsonl)
    if sha256_file(args.graph_inventory_jsonl) != pair_manifest.get("graph_inventory_sha256"):
        raise RuntimeError("graph inventory file SHA256 changed")
    graph_map = _graphs(graph_rows)
    query_config = pair_manifest["query_generation"]
    prepared: list[tuple[dict[str, Any], Any, Any]] = []
    for row in pairs:
        source_graph = graph_map[str(row["query_graph_id"])]
        target_graph = graph_map[str(row["target_graph_id"])]
        query = sample_official_style_query(
            source_graph,
            n_hops=int(query_config["n_hops_query"]),
            traversal_probability=float(query_config["traversal_probability_query"]),
            node_limit=query_config["node_limit_query"],
            sampling_seed=int(row["query_sampling_seed"]),
        )
        if (
            query.graph_sha256 != row.get("query_instance_sha256")
            or target_graph.graph_sha256 != row.get("target_graph_sha256")
            or source_graph.graph_id == target_graph.graph_id
        ):
            raise RuntimeError("pair reconstruction changed")
        prepared.append((row, query.pyged_data(), target_graph.pyged_data()))
    method_args = [f"--threads 1 --time-limit {args.gedlib_time_limit_seconds}"]
    before_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    before_iowait = _iowait_ticks()
    started = time.perf_counter()
    executor = ProcessPoolExecutor(max_workers=args.workers)
    futures = [
        executor.submit(_worker, str(module.parent), query_data, target_data, method_args)
        for _row, query_data, target_data in prepared
    ]
    done, pending = wait(futures, timeout=args.hard_wall_seconds)
    elapsed = time.perf_counter() - started
    if pending:
        for future in pending:
            future.cancel()
        processes = list(getattr(executor, "_processes", {}).values())
        for process in processes:
            process.terminate()
        for process in processes:
            process.join(timeout=5)
        executor.shutdown(wait=False, cancel_futures=True)
    else:
        executor.shutdown(wait=True)
    observations: list[dict[str, Any]] = []
    for (row, _query_data, _target_data), future in zip(prepared, futures):
        if future in pending:
            result = {
                "status": "TIMEOUT",
                "latency_seconds": float(args.hard_wall_seconds),
                "lower_bound": None,
                "upper_bound": None,
                "exact_bound": None,
                "error": "controlled GEDLIB benchmark worker exceeded hard wall",
            }
        else:
            try:
                result = future.result()
            except BaseException as exc:
                result = {
                    "status": "GEDLIB_ERROR",
                    "latency_seconds": 0.0,
                    "lower_bound": None,
                    "upper_bound": None,
                    "exact_bound": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        observations.append(
            {
                "pair_id": row["pair_id"],
                "query_graph_id": row["query_graph_id"],
                "target_graph_id": row["target_graph_id"],
                "query_split": row["query_split"],
                "target_split": row["target_split"],
                "status": result["status"],
                "latency_seconds": result["latency_seconds"],
                "lower_bound": result["lower_bound"],
                "upper_bound": result["upper_bound"],
                "exact_bound": result["exact_bound"],
                "error": result["error"],
                "query_num_nodes": row["query_num_nodes"],
                "target_num_nodes": row["target_num_nodes"],
                "query_num_edges": row["query_num_edges"],
                "target_num_edges": row["target_num_edges"],
                "query_canonical_graph_sha256": row[
                    "query_canonical_graph_sha256"
                ],
                "target_canonical_graph_sha256": row[
                    "target_canonical_graph_sha256"
                ],
            }
        )
    after_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    after_iowait = _iowait_ticks()
    bace_drop = float(args.bace_legacy_throughput_drop_percent)
    aids_drop = float(args.aids_exact_throughput_drop_percent)
    aggregate_cpu_percent = 100.0 * (
        (after_usage.ru_utime - before_usage.ru_utime)
        + (after_usage.ru_stime - before_usage.ru_stime)
    ) / max(elapsed, 1e-12)
    resources = {
        "load_average": list(os.getloadavg()),
        "child_user_cpu_seconds": after_usage.ru_utime - before_usage.ru_utime,
        "child_system_cpu_seconds": after_usage.ru_stime - before_usage.ru_stime,
        "aggregate_child_cpu_utilization_percent": aggregate_cpu_percent,
        "selected_worker_capacity_utilization_percent": (
            aggregate_cpu_percent / args.workers
        ),
        "maximum_child_rss": after_usage.ru_maxrss,
        "maximum_child_rss_unit": "kilobytes_on_linux_getrusage",
        "iowait_ticks_delta": (
            after_iowait - before_iowait
            if before_iowait is not None and after_iowait is not None
            else None
        ),
        "host_load_gate_pass": args.host_load_gate_pass,
        "iowait_gate_pass": args.iowait_gate_pass,
        "bace_legacy_throughput_drop_percent": bace_drop,
        "aids_exact_throughput_drop_percent": aids_drop,
        "bace_legacy_throughput_drop_le_10pct": bace_drop <= 10.0,
        "aids_exact_throughput_drop_le_10pct": aids_drop <= 10.0,
    }
    config_sha = hashlib.sha256(
        json.dumps(
            {
                "method": "f2",
                "method_args": method_args,
                "edit_costs": OFFICIAL_SED_EDIT_COSTS,
                "build_overlay": build["build_overlay"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    summary = summarize_real_gedlib_observations(
        observations,
        benchmark_budget=args.benchmark_budget,
        worker_count=args.workers,
        wall_seconds=elapsed,
        pyged_module_sha256=str(smoke["module_sha256"]),
        gedlib_commit=str(dependencies["gedlib_commit"]),
        gedlib_config_sha256=config_sha,
        feature_schema_sha256=str(pair_manifest["feature_schema_sha256"]),
        resource_metrics=resources,
    )
    output = args.output_dir
    observations_path = output / (
        f"gedlib_benchmark_{args.benchmark_budget}_observations.jsonl"
    )
    report_path = output / f"gedlib_benchmark_{args.benchmark_budget}.json"
    if observations_path.exists() or report_path.exists():
        raise RuntimeError("GEDLIB benchmark output already exists")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        observations_path,
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in observations),
    )
    _atomic_text(
        report_path,
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    healthy = summary["timeout_rate"] <= 0.05 and summary["failure_count"] == 0
    if not healthy:
        print("BLOCKED_GEDLIB_THROUGHPUT", file=sys.stderr)
        return 78
    print(f"[TASTE_NEUROSED_GED_BENCHMARK_{args.benchmark_budget}_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
