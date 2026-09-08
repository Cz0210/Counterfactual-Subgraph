"""Bounded T12 missing-evidence recovery, not another scheduler.

Only fresh, explicitly launched shadow processes install the observer.  Missing
raw values stay missing; neither probabilities nor binary masks are inverted.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Callable, Mapping

from src.utils.main_ready_task_specs import atomic_json, stable_sha256

SCHEMA = "t12_missing_ledger_shadow_plan_v1"
BASE_BUDGET = 520
HARD_CAP = 540
PER_TASK_CACHE_CAP = 1 << 30
ALL_CACHE_CAP = 2 << 30


def tensor_value(value: Any) -> Any:
    """Lossless value snapshot, never pickle identity or mutable references."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if getattr(value, "is_sparse", False):
            value = value.to_dense()
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "values": value.tolist()}
    if hasattr(value, "dtype") and hasattr(value, "tolist"):
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "values": value.tolist()}
    if isinstance(value, Mapping):
        return {str(k): tensor_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_value(v) for v in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"T12_UNSUPPORTED_SCIENCE_VALUE:{type(value).__name__}")


def rng_snapshot(np: Any, torch: Any) -> dict[str, Any]:
    return {"python": tensor_value(random.getstate()),
            "numpy": tensor_value(np.random.get_state()),
            "torch_cpu": tensor_value(torch.get_rng_state()),
            "torch_cuda": [tensor_value(x) for x in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized() else []}


def build_shadow_plan(*, run_id: str, reference_root: str, output_root: str,
                      source_bindings: Mapping[str, Any],
                      existing_continuous_ledgers: Mapping[str, Any],
                      activation_plan: str) -> dict[str, Any]:
    """No implicit extra transitions: optional continuous tails are predeclared."""
    stages = []
    extra = 0
    for arm in ("reference", "accelerated"):
        stages.append({"stage_id": f"{arm}_251_500", "arm": arm,
                       "start": 251, "end": 500, "transitions": 250,
                       "restore_cursor": 250, "kind": "SHADOW"})
        stages.append({"stage_id": f"{arm}_reload_501_510", "arm": arm,
                       "start": 501, "end": 510, "transitions": 10,
                       "restore_cursor": 500, "kind": "INDEPENDENT_RELOAD"})
        evidence = existing_continuous_ledgers.get(arm)
        if evidence is None:
            extra += 10
            stages.append({"stage_id": f"{arm}_continuous_501_510", "arm": arm,
                           "start": 501, "end": 510, "transitions": 10,
                           "restore_cursor": None, "kind": "SAME_PROCESS_TAIL",
                           "reason": "NO_SAME_IMPLEMENTATION_CONTINUOUS_LEDGER"})
        elif not isinstance(evidence, dict) or evidence.get("complete") is not True:
            raise ValueError("T12_EXISTING_CONTINUOUS_EVIDENCE_UNBOUND")
    if extra > 20 or sum(x["transitions"] for x in stages) > HARD_CAP:
        raise ValueError("T12_TRANSITION_BUDGET_EXCEEDED")
    for value in (reference_root, output_root, activation_plan):
        if not Path(value).is_absolute():
            raise ValueError("T12_ABSOLUTE_PATH_REQUIRED")
    if not source_bindings:
        raise ValueError("T12_SOURCE_BINDINGS_REQUIRED")
    plan = {"schema_version": SCHEMA, "run_id": run_id,
            "reference_root": reference_root, "output_root": output_root,
            "source_bindings": copy.deepcopy(dict(source_bindings)),
            "required_natural_510": str(Path(reference_root) / "reference_500_receipt.json"),
            "stages": stages, "base_budget": BASE_BUDGET,
            "optional_extra_transitions": extra, "hard_cap": HARD_CAP,
            "transitions_budgeted": BASE_BUDGET + extra,
            "existing_continuous_ledgers": copy.deepcopy(dict(existing_continuous_ledgers)),
            "no_steps_0_250": True, "current_reader_modified": False,
            "diagnostic_checkpoint_promotable": False,
            "fresh_zero_plan": activation_plan,
            "cache": {"future_only": True, "optional": True,
                      "per_task_max_bytes": PER_TASK_CACHE_CAP,
                      "aggregate_max_bytes": ALL_CACHE_CAP,
                      "persistent_originals_required": True,
                      "cgroup_accounted": True},
            "status": "SEALED_WAITING_NATURAL_510", "science_started": False}
    plan["plan_sha256"] = stable_sha256(plan)
    return plan


def validate_plan(plan: Mapping[str, Any]) -> None:
    payload = dict(plan)
    expected = payload.pop("plan_sha256", None)
    if payload.get("schema_version") != SCHEMA or stable_sha256(payload) != expected:
        raise ValueError("T12_SHADOW_PLAN_BINDING_CHANGED")
    total = sum(s["transitions"] for s in plan["stages"])
    if total != plan["transitions_budgeted"] or total > HARD_CAP:
        raise ValueError("T12_TRANSITION_BUDGET_EXCEEDED")
    if any(s["start"] < 251 or s["end"] - s["start"] + 1 != s["transitions"]
           for s in plan["stages"]):
        raise ValueError("T12_HIDDEN_TRANSITIONS")


def require_natural_510(plan: Mapping[str, Any], *, process_alive: Callable[[int, int], bool]) -> dict:
    validate_plan(plan)
    identity = plan["source_bindings"]["active_reader"]
    if process_alive(identity["pid"], identity["start_ticks"]):
        raise ValueError("T12_ACTIVE_READER_MUST_FINISH_NATURALLY")
    receipt = json.loads(Path(plan["required_natural_510"]).read_text())
    checkpoint = Path(reference_checkpoint_path(plan, 510))
    if (receipt.get("status") != "PASS" or receipt.get("reference_steps") != 500
            or receipt.get("reload_steps") != [501, 510]
            or receipt.get("checkpoint_510") != str(checkpoint)
            or receipt.get("test_loaded") is not False
            or receipt.get("calibration_loaded") is not False):
        raise ValueError("T12_NATURAL_510_RECEIPT_INCOMPLETE")
    manifest = json.loads(checkpoint.read_text())
    if manifest.get("checkpoint_cursor") != 510:
        raise ValueError("T12_NATURAL_510_CHECKPOINT_INCOMPLETE")
    return receipt


def reference_checkpoint_path(plan: Mapping[str, Any], cursor: int) -> str:
    return str(Path(plan["reference_root"]) / "checkpoints" / f"checkpoint-{cursor:08d}.manifest.json")


def cache_admission(*, new_bytes: int, existing_volatile_bytes: int,
                    cgroup_headroom: int, required_other_headroom: int) -> dict:
    if min(new_bytes, existing_volatile_bytes, cgroup_headroom, required_other_headroom) < 0:
        raise ValueError("T12_CACHE_RESOURCE_VALUE_INVALID")
    # The copy and verification stream do not create a second full cache copy.
    permitted = (new_bytes <= PER_TASK_CACHE_CAP
                 and existing_volatile_bytes + new_bytes <= ALL_CACHE_CAP
                 and cgroup_headroom >= required_other_headroom + new_bytes + (2 << 20))
    return {"allowed": permitted, "new_bytes": new_bytes,
            "aggregate_bytes": existing_volatile_bytes + new_bytes,
            "accounted_additional_memory_bytes": new_bytes + (2 << 20),
            "persistent_checkpoint_in_tmpfs": False}


class JointLedger:
    """One compressed append segment; only a checkpoint-bound prefix is adoptable."""
    def __init__(self, root: Path, *, start: int, end: int, binding_sha: str):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"steps-{start:05d}-{end:05d}.jsonl.gz"
        self.raw = self.path.open("xb")
        self.stream = gzip.GzipFile(fileobj=self.raw, mode="wb", mtime=0)
        self.start, self.end, self.next = start, end, start
        self.binding_sha = binding_sha
        self.chain = "0" * 64

    def append(self, row: Mapping[str, Any]) -> None:
        if row.get("step") != self.next or self.next > self.end:
            raise ValueError("T12_LEDGER_GAP_DUPLICATE_OR_BUDGET_EXCEEDED")
        payload = {"binding_sha": self.binding_sha, "previous_sha": self.chain,
                   "record": dict(row)}
        self.chain = stable_sha256(payload)
        payload["sha256"] = self.chain
        self.stream.write(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode() + b"\n")
        self.next += 1

    def flush_before_checkpoint(self) -> None:
        """Call before the existing checkpoint writer; does not publish a boundary."""
        self.stream.flush()
        self.raw.flush()
        os.fsync(self.raw.fileno())

    def seal(self, checkpoint_manifest: Path) -> dict:
        if self.next != self.end + 1:
            raise ValueError("T12_LEDGER_INCOMPLETE")
        self.stream.close()
        self.raw.flush()
        os.fsync(self.raw.fileno())
        self.raw.close()
        content = checkpoint_manifest.read_bytes()
        checkpoint = json.loads(content)
        if checkpoint.get("checkpoint_cursor") != self.end:
            raise ValueError("T12_JOINT_CHECKPOINT_CURSOR_MISMATCH")
        receipt = {"schema_version": "t12_joint_ledger_checkpoint_v1",
                   "ledger": str(self.path), "first_step": self.start,
                   "last_step": self.end, "record_count": self.end - self.start + 1,
                   "ledger_chain_sha256": self.chain, "binding_sha": self.binding_sha,
                   "checkpoint_manifest": str(checkpoint_manifest),
                   "checkpoint_manifest_sha256": hashlib.sha256(content).hexdigest(),
                   "joint_boundary_complete": True}
        atomic_json(self.path.with_suffix(".joint.json"), receipt)
        return receipt

    def close_failed(self) -> None:
        self.stream.close()
        self.raw.flush()
        os.fsync(self.raw.fileno())
        self.raw.close()


def read_ledger(path: Path, *, binding_sha: str, start: int, end: int) -> list[dict]:
    rows, chain = [], "0" * 64
    with gzip.open(path, "rt") as stream:
        for line in stream:
            raw = json.loads(line)
            digest = raw.pop("sha256")
            if (raw["binding_sha"] != binding_sha or raw["previous_sha"] != chain
                    or stable_sha256(raw) != digest):
                raise ValueError("T12_LEDGER_CHAIN_CHANGED")
            if raw["record"]["step"] != start + len(rows):
                raise ValueError("T12_LEDGER_GAP_OR_DUPLICATE")
            rows.append(raw["record"])
            chain = digest
    if len(rows) != end - start + 1:
        raise ValueError("T12_LEDGER_RANGE_INCOMPLETE")
    return rows


class SelectedStepObserver:
    """Read the actual official function frames; no extra oracle/RNG calls.

    Raw pre-temperature logits are collected at the real classifier forward hook.
    Cached probabilities or masks never masquerade as missing raw evidence.
    """
    def __init__(self, ledger: JointLedger, *, np: Any, torch: Any):
        self.ledger, self.np, self.torch = ledger, np, torch
        self.pending = None
        self.query_events: list[dict] = []
        self.handles = []
        self.bound_models: set[int] = set()

    def _science_state(self, globals_: dict) -> dict:
        return {"candidate_order_frequency": [
            {"graph_hash": str(r["graph_hash"]), "frequency": r["frequency"],
             "importance_parts": tensor_value(r["importance_parts"]),
             "coverage": tensor_value(r["input_graphs_covering_list"])}
            for r in globals_["counterfactual_candidates"]],
            "graph_index_map": tensor_value(globals_["graph_index_map"]),
            "input_graphs_covered": tensor_value(globals_["input_graphs_covered"]),
            "rng": rng_snapshot(self.np, self.torch)}

    def _flush(self, globals_: dict) -> None:
        if self.pending is not None:
            if self.pending.get("exceptional_return"):
                raise ValueError("T12_EXCEPTIONAL_TRANSITION_NOT_COMMITTED")
            self.pending["query_events"] = self.query_events.copy()
            kinds = {x["kind"] for x in self.query_events}
            self.pending["missing_raw_fields"] = [
                k for k in ("RAW_CLASSIFIER_LOGITS", "RAW_NEUROSED") if k not in kinds]
            self.pending["after"] = self._science_state(globals_)
            self.ledger.append(self.pending)
            self.pending = None

    def callback(self, frame: Any, event: str, result: Any):
        name, filename = frame.f_code.co_name, frame.f_code.co_filename
        local, global_ = frame.f_locals, frame.f_globals
        if name == "score" and filename.endswith("frozen_gine_batch_scorer.py") and event == "call":
            scorer = local["self"]
            if id(scorer.model) not in self.bound_models:
                self.bound_models.add(id(scorer.model))
                self.handles.append(scorer.model.classifier.register_forward_hook(
                    lambda _module, _input, output: self.query_events.append(
                        {"kind": "RAW_CLASSIFIER_LOGITS", "tensor": tensor_value(output)})))
        if name == "score" and filename.endswith("frozen_gine_batch_scorer.py") and event == "return" and result is not None:
            self.query_events.append({"kind": "CALIBRATED_LOGITS", "tensor": tensor_value(result.project_logits)})
        if name == "__call__" and filename.endswith("tastemolnet_gcf_replay_canary.py") and event == "return":
            if "distances" in local:
                # Keep value evidence only under a hard per-call bound. Refuse, do not truncate.
                size = local["distances"].numel() * local["distances"].element_size()
                if size > 64 * 1024 * 1024:
                    self.query_events.append({"kind": "RAW_DISTANCE_TOO_LARGE", "bytes": size})
                else:
                    self.query_events.append({"kind": "RAW_NEUROSED", "distances": tensor_value(local["distances"]),
                                              "normalizer": tensor_value(local["sums"]), "threshold": local["threshold"]})
        if (filename.endswith("/vrrw.py") and name == "move_from_known_graph"
                and event == "return" and self.pending is not None and result is not None):
            self.pending["sampling_probabilities"] = tensor_value(local["probabilities"])
            self.pending["sampling_target_hashes"] = tensor_value(local["hashes"])
        if filename.endswith("/vrrw.py") and name == "move_to_next_graph":
            if event == "call":
                self._flush(global_)
                self.query_events = []
                self.pending = {"step": len(global_["traversed_hashes"]),
                                "before_graph": str(local["graph_hash"]),
                                "before_rng": rng_snapshot(self.np, self.torch)}
            elif event == "return" and self.pending is not None:
                if result is None:
                    self.pending["exceptional_return"] = True
                    return
                self.pending.update(after_graph=None if result[0] is None else str(result[0]),
                                    teleported=result[1],
                                    selected_action=tensor_value(local.get("selected_action")),
                                    selected_index=local.get("selected_hash_idx"),
                                    importance=tensor_value(local.get("selected_importance_parts")),
                                    query_events=self.query_events.copy())
                graph = local.get("selected_graph")
                self.pending["selected_graph"] = None if graph is None else {
                    key: tensor_value(getattr(graph, key, None))
                    for key in ("x", "edge_index", "edge_attr", "num_nodes")}
        if name == "counterfactual_summary_with_randomwalk" and filename.endswith("/vrrw.py") and event == "return":
            self._flush(global_)
        if name == "commit" and filename.endswith("tastemolnet_gcf_full_resume.py") and event == "call":
            self.ledger.flush_before_checkpoint()

    @contextmanager
    def installed(self):
        if sys.getprofile() is not None:
            raise ValueError("T12_OBSERVER_EXISTING_PROFILE_CONFLICT")
        before = rng_snapshot(self.np, self.torch)
        sys.setprofile(self.callback)
        try:
            yield self
        finally:
            sys.setprofile(None)
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
        # Installation/removal never reseeds; RNG values remain scientific outputs.
        self.initial_rng = before


def compare_ledgers(left: list[dict], right: list[dict]) -> dict:
    """Exact is the existing default; absent numerical contract is not widened."""
    if [x["step"] for x in left] != [x["step"] for x in right]:
        return {"status": "FAILED", "first_difference": "STEP_COVERAGE"}
    for a, b in zip(left, right, strict=True):
        if a.get("missing_raw_fields") or b.get("missing_raw_fields"):
            return {"status": "EVIDENCE_INCOMPLETE", "step": a["step"],
                    "missing_raw_fields": {"left": a.get("missing_raw_fields"), "right": b.get("missing_raw_fields")}}
        if a != b:
            field = next(k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k))
            return {"status": "FAILED", "step": a["step"], "first_difference": field,
                    "numerical_contract": "EXACT_NO_NEW_TOLERANCE"}
    return {"status": "PASS", "steps": len(left), "numerical_contract": "EXACT_NO_NEW_TOLERANCE"}


def validate_full_parity(receipt: Mapping[str, Any]) -> None:
    required = {"reference_500_binding", "accelerated_500_binding", "natural_510_binding",
                "reference_accelerated_251_500", "reference_reload_501_510",
                "accelerated_reload_501_510"}
    value = dict(receipt)
    digest = value.pop("self_sha256", None)
    comparisons = value.get("comparisons", {})
    if (stable_sha256(value) != digest or value.get("status") != "T12_DIAGNOSTIC_PARITY_PASS"
            or set(comparisons) != required or any(x.get("status") != "PASS" for x in comparisons.values())
            or value.get("transitions_used", HARD_CAP + 1) > HARD_CAP
            or value.get("test_loaded") is not False or value.get("checkpoint_promotion_allowed") is not False):
        raise ValueError("T12_FULL_PARITY_EVIDENCE_NOT_COMPLETE")
    # A small PASS JSON is not an independent parity proof. Every component is
    # bound to an existing immutable comparison artifact with the same outcome.
    for name, row in comparisons.items():
        proof = row.get("evidence_path")
        if not isinstance(proof, str) or not Path(proof).is_absolute():
            raise ValueError(f"T12_COMPARISON_ARTIFACT_MISSING:{name}")
        path = Path(proof)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 << 20:
            raise ValueError(f"T12_COMPARISON_ARTIFACT_INVALID:{name}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row.get("evidence_sha256"):
            raise ValueError(f"T12_COMPARISON_ARTIFACT_CHANGED:{name}")
        component = json.loads(raw)
        if component.get("status") != "PASS" or component.get("comparison") != name:
            raise ValueError(f"T12_COMPARISON_ARTIFACT_CONFLICT:{name}")
        if component.get("raw_evidence_complete") is not True:
            raise ValueError(f"T12_COMPARISON_RAW_EVIDENCE_MISSING:{name}")


def finite_fresh_zero(plan: Mapping[str, Any], *, parity: Mapping[str, Any],
                      output: Path, run_stage: Callable[[dict], int],
                      transfer_owner: Callable[[], dict],
                      stage_admission: Callable[[dict], dict]) -> dict:
    """Finite execution of the *existing* plan. Callbacks are original owner/lease interfaces."""
    validate_full_parity(parity)
    if plan.get("plan_sha256") != stable_sha256({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ValueError("T12_FORMAL_PLAN_CHANGED")
    if plan.get("fresh_from_zero") is not True or plan.get("source_checkpoint") is not None:
        raise ValueError("T12_DIAGNOSTIC_PROMOTION_FORBIDDEN")
    stages = plan["stages"]
    if len(stages) != 15 or not stages[0]["stage_id"].startswith("T12_FRESH_FROM_ZERO"):
        raise ValueError("T12_FORMAL_STAGE_CHAIN_CHANGED")
    previous = None
    for row in stages:
        if row.get("stage_sha256") != stable_sha256({k: v for k, v in row.items() if k != "stage_sha256"}):
            raise ValueError("T12_FORMAL_STAGE_CHANGED")
        if row["required_predecessor"] != previous or row.get("matrix_write_allowed") is not False:
            raise ValueError("T12_FORMAL_PREDECESSOR_OR_AUTHORITY_CHANGED")
        if not row["command"] or any(x in {"TODO", "<path>"} for x in row["command"]):
            raise ValueError("T12_FORMAL_COMMAND_INCOMPLETE")
        previous = row["stage_id"]
    if output.exists():
        raise FileExistsError("T12_ACTIVATION_ALREADY_STARTED_USE_ORIGINAL_CHECKPOINT_RESUME")
    first_command = stages[0]["command"]
    if "--output-root" not in first_command:
        raise ValueError("T12_FORMAL_OUTPUT_ROOT_NOT_BOUND")
    production_root = Path(first_command[first_command.index("--output-root") + 1])
    if production_root.exists():
        raise FileExistsError("T12_PRODUCTION_ROOT_ALREADY_EXISTS_NO_DUPLICATE_FRESH")
    # Never claim an observer became an executor by changing a status string.
    transfer = transfer_owner()
    if transfer.get("exclusive_owner_transfer") is not True:
        raise ValueError("T12_CANONICAL_OWNER_TRANSFER_REQUIRED")
    completed = []
    output.mkdir(parents=True, exist_ok=False)
    for row in stages:
        admission = stage_admission(row)
        if admission.get("allowed") is not True:
            atomic_json(output / "status.json", {"status": "WAITING_RESOURCE", "stage": row["stage_id"], "admission": admission})
            return {"status": "WAITING_RESOURCE", "completed_stages": completed}
        code = run_stage(row)
        receipt = {"stage_id": row["stage_id"], "returncode": code,
                   "status": "COMMAND_EXIT_ZERO_OUTPUTS_NOT_YET_VERIFIED" if code == 0 else "FAILED"}
        atomic_json(output / (row["stage_id"] + ".json"), receipt)
        if code != 0:
            return {"status": "FAILED", "stage": row["stage_id"], "completed_stages": completed}
        if any(not Path(p).exists() for p in row["outputs"].values()):
            raise ValueError("T12_FORMAL_STAGE_OUTPUT_ABSENT")
        if "checkpoint" in row["outputs"]:
            checkpoint = json.loads(Path(row["outputs"]["checkpoint"]).read_text())
            if checkpoint.get("checkpoint_cursor") != int(row["stage_id"].rsplit("_", 1)[1]):
                raise ValueError("T12_FORMAL_CHECKPOINT_CURSOR_CHANGED")
        completed.append(row["stage_id"])
    return {"status": "FORMAL_CHAIN_COMPLETE_WAITING_CANONICAL_PUBLISHER",
            "completed_stages": completed, "matrix_written": False,
            "publisher_locator": plan["publisher_handoff"]["canonical_locator"]}


def subprocess_stage(row: dict) -> int:
    """Real command invocation; caller must already hold the existing lease."""
    return subprocess.run(row["command"], cwd=row["cwd"], check=False).returncode


def activate_inherited_owner(*, plan: dict, parity: dict, binding: dict, output: Path) -> dict:
    """Execute once as a child of the canonical owner, never become another owner.

    The predecessor must perform its existing registry CAS before invoking this
    function. There is no default lease, owner, provider command, or GPU ordinal.
    """
    import fcntl
    from src.utils.final16_owner_registry_v1 import process_start_ticks

    validate_full_parity(parity)  # Before opening outputs or admitting science.
    unsigned = {k: v for k, v in binding.items() if k != "self_sha256"}
    if binding.get("self_sha256") != stable_sha256(unsigned):
        raise ValueError("T12_ACTIVATION_BINDING_CHANGED")
    if binding.get("plan_sha256") != plan.get("plan_sha256"):
        raise ValueError("T12_ACTIVATION_PLAN_CHANGED")
    descriptor = int(os.environ.get("T12_OWNER_HELD_GPU_FD", "-1"))
    if descriptor < 0:
        raise ValueError("T12_EXISTING_OWNER_FD_HANDOFF_MISSING")
    owner_pid, owner_ticks = int(binding["owner_pid"]), int(binding["owner_start_ticks"])
    if os.getppid() != owner_pid or process_start_ticks("/proc", owner_pid) != owner_ticks:
        raise ValueError("T12_CANONICAL_PARENT_OWNER_CHANGED")
    registry = json.loads(Path(binding["registry_path"]).read_text())
    if registry.get("self_sha256") != stable_sha256({k: v for k, v in registry.items() if k != "self_sha256"}):
        raise ValueError("T12_CANONICAL_REGISTRY_CHANGED")
    rows = [x for x in registry["tasks"] if x["task_id"] == binding["task_id"]]
    if (len(rows) != 1 or rows[0]["owner_pid"] != owner_pid
            or rows[0]["owner_start_ticks"] != owner_ticks
            or rows[0]["owner_state"] not in {"RUNNING", "ADOPTED_RUNNING"}
            or rows[0]["output_root"] != binding["science_output_root"]):
        raise ValueError("T12_CANONICAL_REGISTRY_OWNER_HANDOFF_MISSING")
    lease = Path(binding["lease_path"])
    opened, named = os.fstat(descriptor), lease.lstat()
    if lease.is_symlink() or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
        raise ValueError("T12_INHERITED_LEASE_IDENTITY_CHANGED")
    # A different process and a different open description must lose contention.
    probe = subprocess.run([sys.executable, "-I", "-c",
        "import fcntl,sys\nf=open(sys.argv[1], 'rb')\ntry:\n"
        " fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
        "except BlockingIOError:\n sys.exit(73)\n", str(lease)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if probe.returncode != 73:
        raise ValueError("T12_INHERITED_LEASE_NOT_EXCLUSIVE")
    # After the contender loses, this descriptor must share the actual holder's
    # open description; an unrelated open descriptor would lose too.
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu = binding["gpu_uuid"]
    if not gpu.startswith("GPU-") or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
        raise ValueError("T12_GPU_UUID_MAPPING_CHANGED")
    # Keep this duplicated descriptor until every child stage has exited. Do not
    # leak it to the science subprocess or grandchildren: this process holds it.
    held = os.dup(descriptor)
    try:
        def admission(row: dict) -> dict:
            if process_start_ticks("/proc", owner_pid) != owner_ticks:
                return {"allowed": False, "reason": "CANONICAL_PARENT_OWNER_EXITED"}
            command = binding["stage_admission_commands"].get(row["stage_id"])
            if not command:
                raise ValueError("T12_EXISTING_RESOURCE_PROVIDER_BINDING_MISSING")
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                return {"allowed": False, "reason": "EXISTING_PROVIDER_FAILED", "returncode": result.returncode}
            measured = json.loads(result.stdout)
            if measured.get("stage_id") != row["stage_id"] or measured.get("actual_resources_resampled") is not True:
                raise ValueError("T12_FRESH_PROVIDER_EVIDENCE_REQUIRED")
            age = time.time() - float(measured.get("measured_at_unix_seconds", 0))
            if not 0 <= age <= 120:
                raise ValueError("T12_STALE_PROVIDER_EVIDENCE")
            return measured
        return finite_fresh_zero(plan, parity=parity, output=output,
            run_stage=subprocess_stage, stage_admission=admission,
            transfer_owner=lambda: {"exclusive_owner_transfer": True,
                                    "owner_pid": owner_pid, "registry_already_bound": True})
    finally:
        os.close(held)
