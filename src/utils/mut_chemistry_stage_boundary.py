"""One Mut chemistry boundary and budget monitor; no scheduler or science changes."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping

from src.baselines.comrecgc.contracts import sha256_file, stable_json_sha256, write_json

GIB = 1024**3
CHEMISTRY_STAGE = "mut_chemistry"
EVALUATION_STAGE = "mut_unified_evaluation"
CHEMISTRY_NEW_INODE_BOUND = 128
BOUNDARY_SCHEMA = "mut_train_only_chemistry_stage_boundary_v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stat(path: Path) -> dict[str, Any]:
    value = path.stat()
    return dict(path=str(path.resolve()), size=value.st_size, inode=value.st_ino,
                device=value.st_dev, mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)


def chemistry_contract(inputs, commands, historical_path: Path, project_commit: str) -> dict:
    # Existing receipts supply content hashes. A phase restart checks immutable
    # stat identities rather than hashing large source/model inputs again.
    paths = [inputs.distance_checkpoint, inputs.dataset_csv, inputs.teacher_path,
             inputs.molclr_checkpoint, inputs.thresholds_path,
             inputs.dataset_dir / "generation_source_graphs.pt",
             inputs.dataset_dir / "dataset_summary.json", historical_path]
    values = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(inputs).items()}
    return dict(schema_version=BOUNDARY_SCHEMA, inputs=values,
                project_commit=project_commit, commands=[[s, a, str(m), f] for s, a, m, f in commands],
                historical_adoption_sha256=sha256_file(historical_path),
                immutable_inputs=[_stat(p) for p in paths])


def validate_boundary(root: Path, contract: Mapping[str, Any]) -> dict:
    receipt = json.loads((root / "chemistry_stage_boundary.json").read_text())
    body = {k: v for k, v in receipt.items() if k != "self_sha256"}
    if (receipt.get("self_sha256") != stable_json_sha256(body)
            or receipt.get("schema_version") != BOUNDARY_SCHEMA
            or receipt.get("state") != "SEALED_CHEMISTRY_WAITING_EVALUATION_ADMISSION"
            or receipt.get("contract") != contract
            or receipt.get("test_evaluation_started") is not False):
        raise ValueError("MUT_CHEMISTRY_BOUNDARY_CONTRACT_CHANGED")
    for row in receipt["sealed_files"]:
        path = Path(row["path"])
        if _stat(path) != row["stat"] or sha256_file(path) != row["sha256"]:
            raise ValueError("MUT_CHEMISTRY_BOUNDARY_ARTIFACT_CHANGED:" + str(path))
    if any((root / name).exists() for name in ("unified_eval", "full_gate", "standardized", "PASS")):
        raise ValueError("MUT_EVALUATION_ALREADY_STARTED_REQUIRES_STAGE_RECOVERY")
    marker = json.loads((root / "chemistry/_RUN_COMPLETE.json").read_text())
    if marker.get("run_complete") is not True:
        raise ValueError("MUT_CHEMISTRY_NOT_COMPLETE")
    return receipt


def seal_boundary(root: Path, contract: Mapping[str, Any]) -> dict:
    # Small scientific closure manifests already bind chemistry's newly
    # generated containers. Never reread all event/graph containers here.
    names = ("chemistry/_RUN_COMPLETE.json", "chemistry/run_manifest.json",
             "chemistry/final_artifact_audit.json", "generation_adoption_manifest.json",
             "historical_adoption_manifest.json", "upstream_checkout_audit.json")
    files = [dict(path=str(root / n), stat=_stat(root / n), sha256=sha256_file(root / n)) for n in names]
    receipt = dict(schema_version=BOUNDARY_SCHEMA,
                   state="SEALED_CHEMISTRY_WAITING_EVALUATION_ADMISSION",
                   chemistry_complete=True, test_evaluation_started=False,
                   final_cell_pass=False, contract=dict(contract), sealed_files=files,
                   next_stage=EVALUATION_STAGE, completed_at=_now(),
                   source_generation_rerun=False, common_recourse_rerun=False)
    receipt["self_sha256"] = stable_json_sha256(receipt)
    write_json(root / "chemistry_stage_boundary.json", receipt)
    return receipt


def _resources(root: Path, cgroup: Path) -> dict[str, int]:
    limit = int((cgroup / "memory.limit_in_bytes").read_text())
    usage = int((cgroup / "memory.usage_in_bytes").read_text())
    if limit <= 0 or limit >= 2**60 or usage < 0:
        raise ValueError("MUT_CGROUP_EFFECTIVE_LIMIT_UNAVAILABLE")
    stat = os.statvfs(root)
    return dict(cgroup_limit_bytes=limit, cgroup_usage_bytes=usage,
                headroom_bytes=limit - usage,
                failcnt=int((cgroup / "memory.failcnt").read_text()),
                free_bytes=stat.f_frsize * stat.f_bavail, free_inodes=stat.f_favail)


class ChemistryResourceMonitor:
    """Observe the existing stage child, using its existing SIGTERM cleanup."""

    def __init__(self, root: Path, persistent_root: Path, config_path: Path, *,
                 stage_id: str, max_rss_gib: int = 96, other_main_reserve_gib: int = 192,
                 transient_reserve_gib: int = 32, min_free_gib: int = 100,
                 cgroup: Path = Path("/sys/fs/cgroup/memory")):
        from src.utils.stage_file_policy import load_stage_policy
        if (max_rss_gib <= 0 or max_rss_gib > 96 or other_main_reserve_gib < 192
                or transient_reserve_gib < 32 or min_free_gib < 100):
            raise ValueError("MUT_CHEMISTRY_RESOURCE_BUDGET_WEAKENED")
        self.root, self.persistent_root, self.cgroup = root, persistent_root, cgroup
        self.stage_id = stage_id
        descriptor = json.loads(config_path.read_text())["stage_file_policy"]
        self.policy = load_stage_policy(descriptor, persistent_root, stage_id=stage_id)
        self.limits = dict(max_process_tree_rss_bytes=max_rss_gib * GIB,
                           other_main_reserve_bytes=other_main_reserve_gib * GIB,
                           transient_reserve_bytes=transient_reserve_gib * GIB,
                           minimum_persistent_free_bytes=min_free_gib * GIB,
                           memory_limits_are_budgets_not_measured_peaks=True)
        self.baseline = None
        self.peak = 0
        self.minimum_headroom = None
        self.root_identity = None

    def _gate(self, now):
        from src.utils.stage_file_policy import stage_file_admission
        return stage_file_admission(self.policy, now["free_inodes"], stage_id=self.stage_id,
                                    baseline_available=(self.baseline or now)["free_inodes"])

    def admission(self):
        self.baseline = _resources(self.persistent_root, self.cgroup)
        gate = self._gate(self.baseline)
        required = (self.limits["max_process_tree_rss_bytes"]
                    + self.limits["other_main_reserve_bytes"] + self.limits["transient_reserve_bytes"])
        admitted = (gate["admitted"] and self.baseline["headroom_bytes"] >= required
                    and self.baseline["free_bytes"] >= self.limits["minimum_persistent_free_bytes"])
        receipt = dict(stage_id=self.stage_id, observed_at=_now(), resources=self.baseline,
                       limits=self.limits, required_start_headroom_bytes=required,
                       inode_admission=gate, admitted=bool(admitted), science_started=False)
        write_json(self.root / (self.stage_id + "_resource_admission.json"), receipt)
        if not admitted:
            raise ValueError("MUT_STAGE_RESOURCE_ADMISSION_BLOCKED:" + json.dumps(receipt, sort_keys=True))
        return receipt

    def sample(self, pid: int):
        from scripts.autodl.run_t14_route_c_owner import _process_tree_snapshot
        rows = _process_tree_snapshot(pid)
        root_rows = [r for r in rows if r["pid"] == pid]
        if not root_rows:
            return  # Child may have just exited; existing runner verifies exit/marker.
        identity = (pid, root_rows[0]["start_ticks"])
        if self.root_identity is None:
            self.root_identity = identity
        if identity != self.root_identity:
            raise ValueError("MUT_STAGE_CHILD_IDENTITY_CHANGED")
        now = _resources(self.persistent_root, self.cgroup)
        rss = sum(r["rss_bytes"] for r in rows)
        self.peak = max(self.peak, rss)
        self.minimum_headroom = min(now["headroom_bytes"], self.minimum_headroom or now["headroom_bytes"])
        gate = self._gate(now)
        pressure = (rss > self.limits["max_process_tree_rss_bytes"]
                    or now["headroom_bytes"] < self.limits["other_main_reserve_bytes"]
                    or now["free_bytes"] < self.limits["minimum_persistent_free_bytes"]
                    or now["failcnt"] > self.baseline["failcnt"]
                    or gate.get("pause_requested") is True or not gate["admitted"])
        timestamp = _now()
        write_json(self.root / (self.stage_id + "_resource_progress.json"),
                   dict(stage_id=self.stage_id, observed_at=timestamp, updated_at=timestamp, owner_pid=os.getpid(),
                        science_pid=pid, process_tree=rows, resources=now, limits=self.limits,
                        rss_bytes=rss, peak_rss_bytes=self.peak,
                        minimum_headroom_bytes=self.minimum_headroom,
                        inode_admission=gate, resource_stop_requested=bool(pressure)))
        if pressure:
            raise ValueError("MUT_STAGE_RESOURCE_PRESSURE_SIGTERM_REQUIRED")
