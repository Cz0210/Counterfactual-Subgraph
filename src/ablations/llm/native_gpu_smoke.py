"""Bounded real-loader/call-boundary resume check, under the existing GPU owner.

No new GPU lease or scheduler. Diagnostic attempts are excluded from the formal
pool. The same full task specification is used, but only two complete calls run.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

from src.ablations.llm.bace_native_runtime import BACEHFNativeRuntime, run_generation, verified_file
from src.ablations.llm.contracts import canonical_json_sha256
from src.eval.bace_frozen_gnn_contracts import atomic_json, sha256_file


def semantic_equal(a, b):
    """Compare state values, not pickle bytes or Python object identities."""
    import torch
    if type(a) is not type(b):
        return False
    if torch.is_tensor(a):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(semantic_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(semantic_equal(x, y) for x, y in zip(a, b))
    return a == b


def read_state(root):
    import torch
    latest = json.loads((Path(root) / "latest_checkpoint.json").read_text())
    name = latest["checkpoint_file"]
    if Path(name).name != name:
        raise ValueError("SMOKE_CHECKPOINT_PATH_ESCAPE")
    path = verified_file({"path": str((Path(root) / name).absolute()), "sha256": latest["sha256"]})
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["next_call"] != latest["next_call"] or state["spec_sha256"] != latest["spec_sha256"]:
        raise ValueError("SMOKE_CHECKPOINT_CURSOR_BINDING")
    return state


def accepted_smoke(path, spec):
    receipt = json.loads(Path(path).read_text())
    body = {k: v for k, v in receipt.items() if k != "self_sha256"}
    if (receipt.get("self_sha256") != canonical_json_sha256(body)
            or receipt.get("state") != "GPU_LOAD_GENERATE_CALL_RESUME_PASS"
            or receipt.get("task_spec_sha256") != canonical_json_sha256(spec)
            or receipt.get("actual_cuda") is not True
            or receipt.get("finite_forward") is not True
            or receipt.get("resume_parity") is not True):
        raise ValueError("NATIVE_GPU_SMOKE_REQUIRED")
    return receipt


def run_smoke(*, spec, output_root, continue_guard, runtime_factory=BACEHFNativeRuntime):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("SMOKE_REQUIRES_ACTUAL_SINGLE_LEASED_GPU")
    if len(spec["calls"]) < 2:
        raise ValueError("SMOKE_REQUIRES_TWO_ORIGINAL_PARENT_CALLS")
    root = Path(output_root).absolute()
    root.mkdir(parents=True, exist_ok=False)
    state = {"state": "RUNNING", "task_spec_sha256": canonical_json_sha256(spec),
             "variant": spec["variant"], "actual_cuda": True,
             "formal_candidates_adopted": False, "training_performed": False,
             "test_loaded": False, "calibration_loaded": False,
             "fresh_runtime_instances": 0, "reload_process_scope": "FRESH_MODEL_SAME_LEASED_PROCESS"}
    def boundary():
        if not continue_guard():
            raise ValueError("SMOKE_PAUSED_MAIN_RESOURCE_PRIORITY")
    try:
        # Separate loaded model instances also exclude adapter/KV cache carryover.
        for name, resume, calls in (("continuous", False, 2), ("interrupted", False, 1), ("interrupted", True, 1)):
            boundary()
            runtime = runtime_factory(spec)
            state["fresh_runtime_instances"] += 1
            try:
                state["finite_forward"] = runtime.finite_forward(spec["calls"][0]) is True
                if not state["finite_forward"]:
                    raise ValueError("SMOKE_NONFINITE_FORWARD")
                result = run_generation(spec=spec, output_root=root / name, resume=resume,
                                        runtime=runtime, max_calls=calls, continue_guard=continue_guard)
                expected = 1 if name == "interrupted" and not resume else 2
                if result["next_call"] != expected:
                    raise ValueError("SMOKE_PAUSED_BEFORE_COMPLETE_BOUNDARY")
                state["peak_allocated_gpu_bytes"] = max(state.get("peak_allocated_gpu_bytes", 0), torch.cuda.max_memory_allocated())
                state["peak_reserved_gpu_bytes"] = max(state.get("peak_reserved_gpu_bytes", 0), torch.cuda.max_memory_reserved())
            finally:
                del runtime
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            atomic_json(root / "progress.json", {**state, "completed_part": name, "resumed": resume})
        continuous, resumed = read_state(root / "continuous"), read_state(root / "interrupted")
        if not semantic_equal(continuous, resumed):
            atomic_json(root / "first_difference.json", {"differing_components": [
                k for k in continuous if not semantic_equal(continuous[k], resumed.get(k))]})
            raise ValueError("SMOKE_PARENT_CALL_RESUME_DIVERGENCE")
        state.update(state="GPU_LOAD_GENERATE_CALL_RESUME_PASS", resume_parity=True,
                     compared_calls=2, compared_attempts=8, actual_device=str(torch.cuda.get_device_name(0)))
    except BaseException as exc:
        state.update(state="FAILED", error=f"{type(exc).__name__}:{exc}")
        raise
    finally:
        state["self_sha256"] = canonical_json_sha256(state)
        atomic_json(root / "gpu_smoke.json", state)
    return state
