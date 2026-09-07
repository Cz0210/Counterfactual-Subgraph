"""AIDS-only continuation from a completed RF pair store, without pair replay.

The DBSCAN and native summary implementations are the existing exact engines.
Only their storage/lifetime and resource admission are changed here.  A waiting
owner holds the existing recourse writer lock; no GPU lock or registry is added.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping

from .rf_aligned_pool import atomic_json, digest, file_sha

GIB = 1024 ** 3
PAIR_ROWS = 37_342_977
PAIR_IDENTITY = "8952fe854cf5c4639f4809aa3c5b7f3548557b579c4ef99f5fb971a306028a32"


def sealed_pairs(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Bind small manifests and array headers, not all 266 chunk payloads again."""
    import numpy as np
    manifest_path = root / "pair_store/run_manifest.json"
    manifest_sha = file_sha(manifest_path)
    if manifest_sha != config["phase_pair_manifest_sha256"]:
        raise ValueError("Completed pair manifest changed from this fresh phase binding")
    manifest = json.loads(manifest_path.read_text())
    identity = json.loads((root / "universe_manifest.json").read_text())
    count = json.loads((root / "exact_count/manifest.json").read_text())
    expected = {"schema": "aids_rf_aligned_native_recourse_v2", "source_denominator": 1283,
                "theta": .1, "eps": .02, "min_samples": 3, "metric": "euclidean",
                "pair_order": "candidate_major_parent_minor", "old_dbscan_labels_reused": False}
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError("Frozen RF-aligned native scientific contract differs")
    if (identity.get("rf_sha") != config["rf_sha256"] or identity.get("greed_sha") != config["greed_sha256"]
            or len(identity["source_positions"]) != 1097 or len(identity["candidate_graphs"]) != 34041
            or manifest.get("scientific_identity") != identity or digest(identity) != PAIR_IDENTITY
            or manifest.get("scientific_identity_sha256") != PAIR_IDENTITY
            or count.get("identity_sha") != PAIR_IDENTITY or count.get("pair_count") != PAIR_ROWS
            or count.get("state") != "COUNT_COMPLETE" or manifest.get("run_complete") is not True
            or manifest.get("row_count") != PAIR_ROWS or len(manifest.get("chunks", [])) != 266):
        raise ValueError("Completed RF pool/count/pair mapping is not the bound 266-chunk universe")
    arrays = {}
    for key, shape, dtype in (("vectors", (PAIR_ROWS, 64), "float32"), ("pairs", (PAIR_ROWS, 2), "int64")):
        path = Path(manifest[key + "_path"])
        if path.is_symlink() or path.parent.resolve() != manifest_path.parent.resolve():
            raise ValueError("Pair source escapes completed pair-store root")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape != shape or str(values.dtype) != dtype or values.flags.writeable:
            raise ValueError("Completed array header differs: " + key)
        stat = path.stat()
        arrays[key] = {"path": str(path), "sha256": manifest[key + "_sha256"],
                       "size": stat.st_size, "inode": stat.st_ino, "device": stat.st_dev,
                       "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
                       "shape": list(shape), "dtype": dtype}
        del values
    return {"manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
            "scientific_identity_sha256": PAIR_IDENTITY, "rows": PAIR_ROWS, "chunks": 266,
            "arrays": arrays, "source_count": 1283, "rf_source1_count": 1097,
            "candidate_count": 34041, "old_gnn_pair_store_reused": False,
            "pair_vectors_recomputed": False, "pair_hash_receipts_reused": True}


def phase_memory_plan(binding: Mapping[str, Any], *, phase: str) -> dict[str, Any]:
    """Conservative charged-memory bounds; no discount of existing file cache."""
    rows = int(binding["rows"])
    if phase not in ("CERTIFIED_EXACT_DBSCAN", "NATIVE_SUMMARY", "RF_WNODE_RELEASE"):
        raise ValueError("Unknown phase has no memory bound")
    # Float32 adaptive engine: at most 4096 failed anchors + three seeds.
    # rows/adjacency/edge tuples and ndarray transition: 3 GiB reserve.  The
    # separate float64 Mut recheck (A*A*64 tensor) is NOT this execution path.
    components = {"full_vector_file_pages": int(binding["arrays"]["vectors"]["size"]),
                  "full_length_memmap_state_pages": rows * 17,
                  "bounded_query_workspace": 256 * 1024**2,
                  "owner_and_sampler": 128 * 1024**2}
    if phase == "CERTIFIED_EXACT_DBSCAN":
        components.update(anchor_graph_max_4099=3 * GIB, numpy_sklearn_runtime=GIB)
    else:
        components.update(pair_index_file_pages=int(binding["arrays"]["pairs"]["size"]),
                          torch_rf_runtime_and_selected_records=2 * GIB,
                          bounded_summary_metadata=512 * 1024**2)
    bound = 14 * GIB
    if sum(components.values()) > bound:
        raise ValueError("Proven phase components exceed 14 GiB; require a fresh resource review")
    return {"schema": "aids_completed_pairs_phase_memory_v1", "phase": phase,
            "components_bytes": components, "component_sum_bytes": sum(components.values()),
            "charged_peak_bound_bytes": bound, "experiment_total_ceiling_bytes": 64 * GIB,
            "workers": 1, "worker_limit": 4, "query_rows_max": 1024,
            "original_14gib_guard_lowered": False, "pagecache_counted": True,
            "shared_cache_discounted": False, "max_anchor_count": 4099,
            "large_quadratic_fallback_max_samples": 100000,
            "scope": "existing_float32_adaptive_exact_then_existing_streamed_summary"}


def resource_sample(config: Mapping[str, Any], plan: Mapping[str, Any], *, running: bool) -> dict[str, Any]:
    from src.utils.stage_file_policy import load_stage_policy, config_file_admission
    resource = json.loads(Path(config["autodl_cpu_resource_config"]).read_text())
    policy = load_stage_policy(resource["stage_file_policy"], resource["persistent_root"])
    fs = os.statvfs(resource["persistent_root"])
    files = config_file_admission(resource, fs.f_favail, stage_id="aids_rf_cpu_recourse", policy=policy)
    cg = Path("/sys/fs/cgroup/memory")
    limit = int((cg / "memory.limit_in_bytes").read_text())
    usage = int((cg / "memory.usage_in_bytes").read_text())
    memory_stat = dict(line.split() for line in (cg / "memory.stat").read_text().splitlines())
    owned = process_tree_snapshot(os.getpid())
    other = int(resource["other_tasks_headroom_reserve_bytes"])
    if other < 384 * GIB:
        raise ValueError("Existing 384 GiB concurrent-task reserve may not be lowered")
    # Start reserves the whole prospective charge, including source pages.
    # While running, actual cgroup usage already charges owned pages.  Do not
    # add the same full peak twice and do not subtract shared pages from usage.
    private_lower = sum(max(0, p["anonymous_bytes"]-p["shared_clean_bytes"]-p["shared_dirty_bytes"])
                        for p in owned["processes"])
    prospective = max(0, int(plan["charged_peak_bound_bytes"])-private_lower)
    required = other + (0 if running else prospective)
    admitted = (files["admitted"] and limit - usage >= required
                and owned["tree_rss_bytes"] <= plan["charged_peak_bound_bytes"]
                and len(owned["processes"]) == 1
                and fs.f_bavail * fs.f_frsize >= int(resource["minimum_persistent_free_bytes"]))
    return {"sampled_at_unix": time.time(), "state": "PASS" if admitted else "WAITING_RESOURCE",
            "phase": plan["phase"], "running_charge_already_in_usage": running,
            "cgroup_limit_bytes": limit, "cgroup_usage_bytes": usage, "cgroup_headroom_bytes": limit-usage,
            "required_headroom_bytes": required, "shortfall_bytes": max(0, required-(limit-usage)),
            "other_tasks_reserve_bytes": other, "task_charged_bound_bytes": plan["charged_peak_bound_bytes"],
            "owned_private_anon_lower_bound_bytes": private_lower,
            "file_cache_discounted_bytes": 0,
            "cgroup_cache_bytes": int(memory_stat.get("total_cache", memory_stat.get("cache", -1))),
            "cgroup_rss_bytes": int(memory_stat.get("total_rss", memory_stat.get("rss", -1))),
            "process_tree": owned, "file_admission": files,
            "persistent_available_bytes": fs.f_bavail * fs.f_frsize,
            "isolation": "MONITORED_SHARED_CGROUP_NOT_CHILD_CGROUP_HARD_LIMIT",
            "gpu_requested": False, "unknown_memory_treated_as_zero": False}


def process_tree_snapshot(pid: int) -> dict[str, Any]:
    result, pending = [], [int(pid)]
    while pending:
        current = pending.pop()
        proc = Path("/proc") / str(current)
        smaps = {}
        for line in (proc / "smaps_rollup").read_text().splitlines():
            if ":" in line and line.strip().endswith("kB"):
                key, value = line.split(":", 1); smaps[key] = int(value.split()[0])*1024
        children = set()
        for task in (proc / "task").iterdir():
            children.update(int(x) for x in (task / "children").read_text().split())
        pending.extend(children)
        result.append({"pid": current, "start_ticks": (proc / "stat").read_text().split(") ",1)[1].split()[19],
                       "rss_bytes": smaps["Rss"], "anonymous_bytes": smaps["Anonymous"],
                       "shared_clean_bytes": smaps["Shared_Clean"], "shared_dirty_bytes": smaps["Shared_Dirty"]})
    return {"processes": result, "tree_rss_bytes": sum(p["rss_bytes"] for p in result),
            "tree_rss_double_counts_shared_pages_conservatively": True}


class PhaseObserver:
    """One compact sampler plus resource checks after exact engine checkpoints."""
    def __init__(self, config, plan, root):
        self.config, self.plan, self.root = config, plan, Path(root)
        self.stop = threading.Event(); self.last = None; self.error = None
        self.peak_rss = self.peak_usage = self.peak_cache = 0
        self.minimum_headroom = None
        self.sample_lock = threading.RLock()

    def sample(self):
        with self.sample_lock:
            return self._sample_locked()

    def _sample_locked(self):
        value = resource_sample(self.config, self.plan, running=True)
        self.peak_rss = max(self.peak_rss, value["process_tree"]["tree_rss_bytes"])
        self.peak_usage = max(self.peak_usage, value["cgroup_usage_bytes"])
        self.peak_cache = max(self.peak_cache, value["cgroup_cache_bytes"])
        headroom = value["cgroup_headroom_bytes"]
        self.minimum_headroom = headroom if self.minimum_headroom is None else min(self.minimum_headroom, headroom)
        value.update(peak_tree_rss_bytes=self.peak_rss, peak_cgroup_usage_bytes=self.peak_usage,
                     peak_cgroup_cache_bytes=self.peak_cache, minimum_cgroup_headroom_bytes=self.minimum_headroom)
        self.last = value
        atomic_json(self.root / "runtime_memory_latest.json", value)
        return value

    def _loop(self):
        while not self.stop.wait(5):
            try: self.sample()
            except Exception as exc: self.error = exc; return

    def __enter__(self):
        self.sample(); self.thread = threading.Thread(target=self._loop, daemon=True); self.thread.start(); return self

    def boundary(self):
        if self.error: raise RuntimeError("Live resource evidence unavailable") from self.error
        while self.sample()["state"] != "PASS":
            atomic_json(self.root / "science_boundary.json", {"state": "CHECKPOINT_COMMITTED_WAITING_RESOURCE",
                        "pid": os.getpid(), "sampled_at_unix": time.time(), "GPU_requested": False})
            time.sleep(60)

    def __exit__(self, *args):
        self.stop.set(); self.thread.join(timeout=10); self.sample()


@contextmanager
def checkpoint_observer(observer):
    """Instrument only this new AIDS process, after original atomic commits."""
    from . import external_memory_dbscan as engine, external_component_summary as component, external_memory_recourse as summary
    originals = [(engine, "_checkpoint", engine._checkpoint),
                 (component, "_write_checkpoint", component._write_checkpoint),
                 (summary, "_summary_checkpoint", summary._summary_checkpoint)]
    def wrapped(original):
        def after_commit(*args, **kwargs):
            result = original(*args, **kwargs)
            observer.boundary()
            return result
        return after_commit
    for module, name, original in originals: setattr(module, name, wrapped(original))
    try: yield
    finally:
        for module, name, original in originals: setattr(module, name, original)


def require_start_admission(config, plan, evidence_root):
    evidence = resource_sample(config, plan, running=False)
    atomic_json(evidence_root / "start_resource_admission.json", evidence)
    if evidence["state"] != "PASS":
        raise RuntimeError("CPU phase start resource changed; no science started")


def run_cluster_only(config, *, recourse_root: Path, evidence_root: Path):
    from .external_memory_dbscan import ExternalDBSCANContract, fit_external_memory_dbscan, ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT
    binding = sealed_pairs(config, recourse_root)
    plan = phase_memory_plan(binding, phase="CERTIFIED_EXACT_DBSCAN")
    atomic_json(evidence_root / "pair_adoption.json", binding)
    atomic_json(evidence_root / "phase_memory_plan.json", plan)
    require_start_admission(config, plan, evidence_root)
    contract = ExternalDBSCANContract(eps=.02, min_samples=3, query_block_size=8,
        checkpoint_interval_blocks=64, max_rss_bytes=14*GIB-128*1024**2,
        expected_sklearn_version=config["expected_sklearn_version"],
        shortcut_mode=ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
        shortcut_query_block_size=1024, exact_fallback_max_samples=100000)
    with PhaseObserver(config, plan, evidence_root) as observer, checkpoint_observer(observer):
        result = fit_external_memory_dbscan(vectors_path=binding["arrays"]["vectors"]["path"],
            work_dir=recourse_root / "dbscan", contract=contract,
            expected_vectors_sha256=binding["arrays"]["vectors"]["sha256"], resume=True)
    terminal = {"state": "EXACT_DBSCAN_COMPLETE", "cluster_count": result.cluster_count,
                "dbscan_manifest": str(result.manifest_path), "dbscan_manifest_sha256": result.manifest_sha256,
                "pair_rows": PAIR_ROWS, "source_count": 1283, "rf_source1_count": 1097,
                "pair_recomputed_count": 0, "selected_summary_complete": False}
    atomic_json(evidence_root / "terminal.json", terminal)
    return terminal


def run_summary_only(config, *, pool_root: Path, recourse_root: Path, evidence_root: Path):
    import numpy as np
    import torch
    from .external_memory_dbscan import ExternalDBSCANContract, fit_external_memory_dbscan
    from .external_memory_recourse import summarize_proven_one_cluster_external, trace_external_cluster_order
    from .external_component_summary import summarize_proven_all_core_components_external
    from .upstream import imported_upstream
    from .rf_aligned_pool import rf_predict
    from src.rewards.reward_calculator import load_oracle_bundle
    binding = sealed_pairs(config, recourse_root)
    plan = phase_memory_plan(binding, phase="NATIVE_SUMMARY")
    require_start_admission(config, plan, evidence_root)
    identity = json.loads((recourse_root/"universe_manifest.json").read_text())
    pool_terminal = json.loads((pool_root/"terminal.json").read_text())
    if (file_sha(pool_root/"terminal.json") != identity["pool_terminal_sha"]
            or pool_terminal.get("state") != "POOL_SCREEN_COMPLETE"
            or pool_terminal.get("contract_sha") != identity["pool_contract"]):
        raise ValueError("Summary pool terminal differs from completed pair universe")
    manifest_path = recourse_root / "dbscan/run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    contract = ExternalDBSCANContract(**manifest["scientific_identity"]["contract"])
    cluster = fit_external_memory_dbscan(vectors_path=binding["arrays"]["vectors"]["path"],
        work_dir=manifest_path.parent, contract=contract,
        expected_vectors_sha256=binding["arrays"]["vectors"]["sha256"], resume=True)
    vectors = np.load(binding["arrays"]["vectors"]["path"], mmap_mode="r", allow_pickle=False)
    pairs = np.load(binding["arrays"]["pairs"]["path"], mmap_mode="r", allow_pickle=False)
    labels = np.load(cluster.labels_path, mmap_mode="r", allow_pickle=False)
    torch.set_num_threads(1)
    runtime = dict(config, **config.get("runtime_paths", {}))
    with PhaseObserver(config, plan, evidence_root) as observer, checkpoint_observer(observer), imported_upstream(runtime["upstream_root"]) as modules:
        common = dict(work_dir=recourse_root / "streamed_native_summary", dbscan_manifest_path=manifest_path,
            dbscan_manifest_sha256=cluster.manifest_sha256, recourse_vectors=vectors, pair_indices=pairs,
            pairs_sha256=binding["arrays"]["pairs"]["sha256"], pair_authority_manifest_path=Path(binding["manifest_path"]),
            pair_authority_manifest_sha256=binding["manifest_sha256"], radius=.02, theta=.1, recourse_size=100,
            official_greedy=modules["common_recourse"].greedy_counterfactual_summary_from_covering_sets,
            torch_module=torch, max_rss_bytes=14*GIB-128*1024**2, block_size=4096, resume=True)
        if manifest["clustering_path"] == "all_core_adaptive_anchor_component_recovery_v1":
            result = summarize_proven_all_core_components_external(labels=labels, **common)
        elif cluster.shortcut_proof_path is not None and cluster.cluster_count == 1:
            result = summarize_proven_one_cluster_external(**common)
        else:
            raise RuntimeError("Large RF universe needs certified exact streamed summary; no unbounded legacy fallback")
        selected = result.selected
        audit = {"streaming_summary_manifest": str(result.manifest_path), "manifest_sha256": result.manifest_sha256}
    # Only selected medoids are retained. Full native graphs / GREED are never loaded.
    needed = {int(x["representative_counterfactual_index"]) for x in selected}
    records, cursor = {}, 0
    for path in sorted((pool_root / "segments").glob("segment-*.json")):
        for row in json.loads(path.read_text())["rows"]:
            if row["state"] == "RF_TARGET0":
                if cursor >= len(identity["candidate_graphs"]) or row["stable_graph_sha256"] != identity["candidate_graphs"][cursor]:
                    raise ValueError("Pool sequence changed from completed native pair mapping")
                if cursor in needed: records[cursor] = row
                cursor += 1
    if cursor != 34041 or set(records) != needed: raise ValueError("Selected RF pool row mapping differs")
    rows = []
    with PhaseObserver(config, plan, evidence_root/"rf-medoid-validation"):
        rf = load_oracle_bundle(runtime["rf_path"])
        for item in selected:
            record = records[int(item["representative_counterfactual_index"])]
            score = rf_predict([record["canonical_smiles"]], rf)[0]
            if score["prediction"] != 0: raise ValueError("Selected medoid no longer predicts RF0")
            rows.append(dict(item, canonical_smiles=record["canonical_smiles"], rf=score,
                             original_candidate_index=record["candidate_index"], original_parent_id=record["parent_id"], graph=record["graph"]))
    atomic_json(recourse_root / "selected_native_recourses.json", rows)
    terminal = {"state": "RF_ALIGNED_NATIVE_SUMMARY_COMPLETE" if rows else "POOL_NO_VALID_COMMON_SUMMARY",
        "selected_count": len(rows), "source_denominator": 1283, "source1_count": 1097, "candidate_count": 34041,
        "pair_rows": PAIR_ROWS, "dbscan_cluster_count": cluster.cluster_count, "summary_audit": audit,
        "old_cluster_labels_reused": False, "evaluation_complete": False, "test_loaded": False,
        "pair_chunks_recomputed": 0, "phase_only_execution": True}
    atomic_json(recourse_root / "terminal.json", terminal)
    atomic_json(evidence_root / "terminal.json", terminal)
    return terminal
