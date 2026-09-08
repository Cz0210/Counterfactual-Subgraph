"""Bounded, observational T14 diagnostics. Never grants formal promotion."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import resource
import signal
import sys
import threading
import time
from typing import Any

TOTAL_TRANSITION_CAP = 170
START_STEP = 250
END_STEP = 335


class ResourceSampler:
    """Bound this new CPU inspection, never alters another task's policy."""
    def __init__(self, root: Path, max_rss_bytes: int = 12 * 1024**3):
        self.root = root
        self.max_rss_bytes = max_rss_bytes
        self.peak_rss = 0
        self.peak_cgroup = 0
        self.minimum_headroom = None
        self.stop = threading.Event()

    def sample(self):
        status = Path("/proc/self/status").read_text()
        rss = int(next(line.split()[1] for line in status.splitlines() if line.startswith("VmRSS:"))) * 1024
        domain = Path("/sys/fs/cgroup/memory")
        current = int((domain / "memory.usage_in_bytes").read_text())
        limit = int((domain / "memory.limit_in_bytes").read_text())
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_cgroup = max(self.peak_cgroup, current)
        headroom = limit - current
        self.minimum_headroom = headroom if self.minimum_headroom is None else min(self.minimum_headroom, headroom)
        return {"time": time.time(), "pid": os.getpid(), "rss_bytes": rss, "cgroup_usage_bytes": current, "cgroup_limit_bytes": limit, "headroom_bytes": headroom}

    def __enter__(self):
        self.old_signal = signal.getsignal(signal.SIGTERM)
        def terminate(*_):
            raise RuntimeError("T14_DIAGNOSTIC_RESOURCE_BOUND_EXCEEDED")
        signal.signal(signal.SIGTERM, terminate)
        initial = self.sample()
        if initial["headroom_bytes"] < 96 * 1024**3 + self.max_rss_bytes:
            raise RuntimeError("T14 diagnostic existing runtime reserve plus own bound unavailable")
        atomic_json(self.root / "resource_admission.json", {**initial, "own_max_rss_bytes": self.max_rss_bytes, "existing_runtime_headroom_bytes": 96 * 1024**3, "status": "PASS", "scope": "THIS_CPU_INSPECTION_ONLY"})
        def monitor():
            with (self.root / "resource_samples.jsonl").open("x") as stream:
                while not self.stop.is_set():
                    row = self.sample()
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    if row["rss_bytes"] > self.max_rss_bytes or row["headroom_bytes"] < 96 * 1024**3:
                        os.kill(os.getpid(), signal.SIGTERM)
                        return
                    self.stop.wait(1.0)
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=2)
        signal.signal(signal.SIGTERM, self.old_signal)
        atomic_json(self.root / "resource_summary.json", {"peak_rss_bytes": self.peak_rss, "peak_cgroup_usage_bytes": self.peak_cgroup, "minimum_headroom_bytes": self.minimum_headroom, "sampling_seconds": 1})


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def semantic(value: Any) -> Any:
    """Values, not pickle bytes/object IDs, define comparison identity."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, dict):
        return {str(key): semantic(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [semantic(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((semantic(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        import numpy as np
        array = np.asarray(value)
        if array.ndim == 0:
            return semantic(array.item())
        if array.size <= 32:
            return {"dtype": str(array.dtype), "shape": list(array.shape), "values": semantic(array.tolist())}
        return {"dtype": str(array.dtype), "shape": list(array.shape), "content_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}
    raise TypeError(f"Unsupported diagnostic semantic type: {type(value).__module__}.{type(value).__name__}")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(semantic(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def first_difference(left: Any, right: Any, path: str = "$") -> dict[str, Any] | None:
    if type(left) is not type(right):
        return {"path": path, "reason": "type", "left_type": type(left).__name__, "right_type": type(right).__name__}
    if isinstance(left, dict):
        if set(left) != set(right):
            return {"path": path, "reason": "keys", "left_only": sorted(set(left)-set(right))[:10], "right_only": sorted(set(right)-set(left))[:10]}
        for key in sorted(left):
            difference = first_difference(left[key], right[key], path + "." + str(key))
            if difference:
                return difference
    elif isinstance(left, list):
        if len(left) != len(right):
            return {"path": path, "reason": "length", "left": len(left), "right": len(right)}
        for index, (a, b) in enumerate(zip(left, right)):
            difference = first_difference(a, b, f"{path}[{index}]")
            if difference:
                return difference
    elif left != right:
        return {"path": path, "reason": "value", "left": left, "right": right}
    return None


class SamplingObserver:
    """Call original RNG once; observe actual native locals at function return."""

    def __init__(self, module: Any, *, follower_code=None, action_lookup=None):
        self.module = module
        self.events: list[dict[str, Any]] = []
        self._profile = None
        self.capture_follower = False
        self.follower_code=follower_code
        self.action_lookup=action_lookup

    def __enter__(self):
        self._random = random._inst.random
        self._getrandbits = random._inst.getrandbits
        self._profile = sys.getprofile()
        if self._profile is not None:
            raise RuntimeError("Refuse to replace an active profiler")
        def draw():
            value = self._random()
            self.events.append({"api": "Random.random", "u": value})
            return value
        def bits(count):
            value = self._getrandbits(count)
            self.events.append({"api": "Random.getrandbits", "bits": count, "value": value})
            return value
        self._target = self.module.move_from_known_graph.__code__
        self._follower_target = self.follower_code or getattr(getattr(self.module,'move_to_next_graph',None),'__code__',None)
        self._np=getattr(self.module,'np',None)
        self._argmin = self._np.argmin if self._np is not None else None
        def observed_argmin(*args, **kwargs):
            # Call the native implementation once with untouched arguments.
            result = self._argmin(*args, **kwargs)
            frame = sys._getframe(1)
            if self.capture_follower and frame.f_code is self._follower_target:
                row=follower_snapshot(frame.f_locals,args[0],result)
                if self.action_lookup is not None:
                    entry=self.action_lookup(row['source_hash'])
                    if entry is None or tuple(entry.target_hashes)!=tuple(row['candidate_order']):
                        raise ValueError('FOLLOWER_NATIVE_ACTION_ORDER_UNBOUND')
                    row['ordered_actions']=tuple(entry.actions)
                    row['selected_action']=tuple(entry.actions[row['selected_index']])
                self.events.append(row)
            return result
        random._inst.random = draw
        random._inst.getrandbits = bits
        if self._np is not None:self._np.argmin = observed_argmin
        self._choices_target = random._inst.choices.__func__.__code__
        sys.setprofile(self._observe)
        return self

    def _observe(self, frame, event, result):
        if event != "return":
            return
        if frame.f_code is self._choices_target:
            local = frame.f_locals
            self.events.append({
                "api": "Random.choices.return", "caller": frame.f_back.f_code.co_name if frame.f_back else None,
                "population": semantic(list(local["population"])),
                "actual_cumulative_weights": semantic(local.get("cum_weights")),
                "actual_total": semantic(local.get("total")),
                "k": int(local["k"]), "selected": semantic(result),
            })
            return
        if frame.f_code is not self._target:
            return
        local = frame.f_locals
        hashes = local.get("hashes", ())
        parent = frame.f_back.f_locals if frame.f_back and frame.f_back.f_code.co_name == "move_to_next_graph" else {}
        frequency = []
        for key in hashes:
            if key in self.module.graph_index_map:
                frequency.append(int(self.module.counterfactual_candidates[self.module.graph_index_map[key]]["frequency"]))
            else:
                # The native loop's last local frequency is not sufficient for
                # all candidates. Mark absent rows without extra native calls.
                frequency.append(None)
        probabilities = local.get("probabilities")
        self.events.append({
            "api": "move_from_known_graph.return", "candidate_order": semantic(hashes),
            "raw_importances": semantic([row.tolist() if hasattr(row, "tolist") else row for row in local.get("importances", ())]),
            "importance_values": semantic(local.get("importance_values")),
            "actual_probabilities": semantic(probabilities.tolist() if hasattr(probabilities, "tolist") else probabilities),
            "existing_candidate_frequency": frequency,
            "selected_index": int(result) if result is not None else None,
            "actual_lead_head": semantic(parent.get("select")),
            "actual_source_hash": semantic(parent.get("graph_hash")),
        })

    def __exit__(self, *_):
        sys.setprofile(self._profile)
        random._inst.random = self._random
        random._inst.getrandbits = self._getrandbits
        if self._np is not None:self._np.argmin = self._argmin


def follower_snapshot(local, argmin_input, selected):
    """Copy actual follower locals, never reconstruct them from the selected min."""
    import numpy as np
    names=('recourse','matching_recourses','difference','target_graphs_embedding',
           'start_embedding','selected_elements','start_elements','target_graphs_importance_parts')
    missing=[name for name in (*names,'k','i','select','target_graphs_hashes') if name not in local]
    if missing:raise ValueError('FOLLOWER_RAW_LOCALS_MISSING:'+','.join(missing))
    difference=np.asarray(argmin_input)
    if difference.ndim!=1 or not np.array_equal(difference,np.asarray(local['difference']),equal_nan=True):
        raise ValueError('FOLLOWER_ARGMIN_INPUT_BINDING')
    ids=list(local['target_graphs_hashes'])
    if len(ids)!=len(difference) or len(set(ids))!=len(ids) or not np.isfinite(difference).all():
        raise ValueError('FOLLOWER_CANDIDATES_OR_NONFINITE')
    arrays={}
    for name in names:
        a=np.asarray(local[name])
        if a.dtype.hasobject or a.nbytes>128*1024**2:
            raise ValueError('FOLLOWER_RAW_ARRAY_BOUND_OR_OBJECT:'+name)
        arrays[name]={'dtype':str(a.dtype),'shape':list(a.shape),'strides':list(a.strides),
                      'values':np.array(a,copy=True,order='K')}
    best=float(difference[int(selected)])
    sorted_values=np.sort(difference,kind='stable')
    return dict(api='follower_argmin',head=int(local['k']),source_hash=int(local['i']),
        lead_head=int(local['select']),candidate_order=ids,selected_index=int(selected),
        selected_hash=ids[int(selected)],exact_minimum_indices=np.flatnonzero(difference==best).tolist(),
        min_value=best,second_min_value=float(sorted_values[1]) if len(sorted_values)>1 else None,
        arrays=arrays,original_argmin_called_once=True,extra_rng_calls=0)


def follower_field_preflight(output_root):
    """Full raw-array save/reopen test before charging any replay transition."""
    import numpy as np
    root=Path(output_root);root.mkdir(parents=True,exist_ok=False)
    local=dict(k=1,i=11,select=0,target_graphs_hashes=list(range(100,140)),
        difference=np.arange(40,dtype=np.float32),recourse=np.zeros(64,dtype=np.float32),
        matching_recourses=np.arange(2560,dtype=np.float32).reshape(40,64),
        target_graphs_embedding=np.arange(2560,dtype=np.float32).reshape(40,64),
        start_embedding=np.zeros(64,dtype=np.float32),selected_elements=np.ones(40,dtype=np.int64),
        start_elements=np.int64(3),target_graphs_importance_parts=np.ones((40,3),dtype=np.float32))
    row=follower_snapshot(local,local['difference'],0)
    path=root/'raw_fields.pkl.gz'
    with gzip.open(path,'wb') as f:pickle.dump(row,f,protocol=5)
    with gzip.open(path,'rb') as f:loaded=pickle.load(f)
    for name,info in row['arrays'].items():
        observed=loaded['arrays'][name]
        if info['dtype']!=observed['dtype'] or info['shape']!=observed['shape'] or not np.array_equal(info['values'],observed['values']):
            raise ValueError('RAW_FIELD_ROUNDTRIP:'+name)
    result=dict(state='FULL_FOLLOWER_RAW_SAVE_REOPEN_PASS',new_transitions=0,
        array_fields=list(row['arrays']),raw_file=str(path),raw_bytes=path.stat().st_size,
        captured_candidates=len(loaded['candidate_order']),hash_only_fields=False)
    atomic_json(root/'terminal.json',result)
    return result


def compare_followers(reference, lowmemory, output_root):
    import numpy as np
    def read(root):
        root=Path(root)
        terminal=json.loads((root/'terminal.json').read_text())
        if terminal.get('follower_records_335')!=4:
            raise ValueError('COMPLETE_STEP335_FOLLOWERS_REQUIRED')
        with gzip.open(root/'raw_step_observations.pkl.gz','rb') as f:
            while True:
                try:row=pickle.load(f)
                except EOFError:break
                if row['phase']=='AFTER' and row['step']==335:
                    return {e['head']:e for e in row['actual_sampling_events'] if e['api']=='follower_argmin'}
        raise ValueError('RAW_STEP335_MISSING')
    left,right=read(reference),read(lowmemory)
    if set(left)!=set(right):raise ValueError('FOLLOWER_HEAD_SET_DIFFERS')
    rows=[]
    for head in sorted(left):
        a,b=left[head],right[head];differences={}
        for name,info in a['arrays'].items():
            x=info['values'];y=b['arrays'][name]['values']
            same_shape=x.shape==y.shape
            same=np.array_equal(x,y) if same_shape else False
            first=np.argwhere(x!=y)[0].tolist() if same_shape and not same else None
            differences[name]=dict(exact_equal=same,left_dtype=str(x.dtype),right_dtype=str(y.dtype),
                left_shape=list(x.shape),right_shape=list(y.shape),first_difference_index=first,
                max_abs=float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))) if same_shape and x.size else None)
        rows.append(dict(head=head,candidate_order_equal=a['candidate_order']==b['candidate_order'],
            actions_equal=a['ordered_actions']==b['ordered_actions'],left_selected_action=a['selected_action'],
            right_selected_action=b['selected_action'],left_argmin=a['selected_index'],right_argmin=b['selected_index'],
            left_exact_ties=a['exact_minimum_indices'],right_exact_ties=b['exact_minimum_indices'],
            left_min=a['min_value'],right_min=b['min_value'],left_second=a['second_min_value'],right_second=b['second_min_value'],
            arrays=differences))
    root=Path(output_root);root.mkdir(parents=True,exist_ok=False)
    result=dict(state='FOLLOWER_335_RAW_COMPARISON_COMPLETE',followers=rows,
        root_cause_classification='EVIDENCE_ONLY_REQUIRES_SEMANTIC_REVIEW',
        parity_claimed=False,formal_dispatch_allowed=False,new_transitions=0)
    atomic_json(root/'follower_comparison.json',result)
    return result


def candidate_snapshot(algorithm: dict, source_root: Path) -> list:
    official = algorithm["official_state"]
    if "route_c_state" not in algorithm:
        candidates = official["counterfactual_candidates"]
        return [semantic(dict(row)) for row in candidates]
    state = algorithm["route_c_state"]["candidates"]
    path = source_root / "route_c_state/candidate_state"
    with (path / "candidate_payload_index.jsonl").open("rb") as stream:
        index = [json.loads(line) for line in stream.read(int(state["payload_index_bytes"])).splitlines()]
    if len(index) != int(state["record_count"]):
        raise ValueError("Candidate checkpoint index prefix is incomplete")
    result = []
    with (path / "candidate_payloads.bin").open("rb") as stream:
        for record_id in state["order"]:
            record_id = int(record_id)
            locator = index[record_id]
            stream.seek(int(locator["offset"]))
            blob = stream.read(int(locator["length"]))
            if hashlib.sha256(blob).hexdigest() != locator["payload_sha256"]:
                raise ValueError("Candidate immutable payload source mismatch")
            row = dict(pickle.loads(blob))
            row["frequency"] = int(state["frequency"][record_id])
            result.append(semantic(row))
    return result


def inspect_checkpoint(*, source_root: Path, output_root: Path) -> dict:
    """CPU-only, no oracle inference, no transition, immutable input only."""
    from src.baselines.comrecgc.generation_checkpoint import load_generation_checkpoint
    identity = json.loads((source_root / "checkpoint_identity.json").read_text())
    checkpoint = source_root / "checkpoints/step-000000000250"
    output_root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    atomic_json(output_root / "progress.json", {"status": "RUNNING", "stage": "LOAD_SEALED_CHECKPOINT250", "pid": os.getpid(), "new_transitions": 0})
    loaded = load_generation_checkpoint(
        checkpoint, expected_provenance=identity["provenance"],
        expected_scientific_argv=identity["scientific_argv"],
        expected_command_sha256=identity["command_sha256"],
        expected_total_steps=25000, expected_completed_step=250, single_pass=True,
    )
    state = loaded.algorithm_state
    # Save raw RNG for independent value comparison, not only opaque digests.
    with gzip.open(output_root / "rng250.pkl.gz", "wb") as stream:
        pickle.dump(loaded.rng_state, stream, protocol=5)
    official = state["official_state"]
    excluded = {"graph_map", "counterfactual_candidates", "transitions", "schema_version", "graph_objects_saved", "full_python_candidate_list_saved"}
    bridge = state["bridge_state"]
    record_metadata = {
        str(key): semantic({name: value for name, value in row.items() if name != "embedding_values"})
        for key, row in bridge["records"].items()
    }
    components = {
        "loop_state": semantic(state["loop_state"]),
        "rng": semantic(loaded.rng_state),
        "official": semantic({key: value for key, value in official.items() if key not in excluded}),
        "candidate_order_and_frequency": candidate_snapshot(state, source_root),
        "bridge_records": record_metadata,
        "bridge_counters": semantic({key: bridge[key] for key in ("call_count", "evaluated_graph_count", "calculate_hash_count") if key in bridge}),
        "transition_state": semantic(state["transition_state"]),
    }
    for name, value in components.items():
        atomic_json(output_root / (name + ".json"), value)
    receipt = {
        "status": "CHECKPOINT250_COMPONENTS_RECORDED", "source_root": str(source_root),
        "checkpoint_digest": loaded.validation.checkpoint_digest,
        "source_execution_commit": identity["provenance"]["execution_commit"],
        "storage_schema_recorded_not_compared_as_science": official.get("schema_version"),
        "completed_step": 250, "new_transitions": 0, "total_transition_cap": TOTAL_TRANSITION_CAP,
        "formal_dispatch_allowed": False, "raw_rng": str(output_root / "rng250.pkl.gz"),
        "components": {name: {"path": str(output_root / (name + ".json")), "semantic_sha256": digest(value)} for name, value in components.items()},
        "elapsed_seconds": time.monotonic() - start,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1024 if sys.platform != "darwin" else 1),
    }
    atomic_json(output_root / "terminal.json", receipt)
    return receipt


def compare_checkpoints(left: Path, right: Path, output: Path) -> dict:
    receipts = [json.loads((root / "terminal.json").read_text()) for root in (left, right)]
    if any(row["status"] != "CHECKPOINT250_COMPONENTS_RECORDED" for row in receipts):
        raise ValueError("Both checkpoint component inspections must be complete")
    differences = {}
    for name in receipts[0]["components"]:
        values = [json.loads(Path(row["components"][name]["path"]).read_text()) for row in receipts]
        differences[name] = first_difference(*values)
    result = {"status": "STARTING_STATE_DIFFERENT" if any(differences.values()) else "STARTING_COMPONENTS_EQUAL", "component_first_differences": differences, "raw_rng_first_differences": compare_saved_rng(left, right), "new_transitions": 0, "formal_dispatch_allowed": False, "causal_source_of_step335_proven": False, "scope": "CHECKPOINT250_SAVED_COMPONENTS_NOT_FULL_EXECUTION_PARITY"}
    atomic_json(output, result)
    return result


def compare_saved_rng(left: Path, right: Path) -> dict:
    """Expand only the tiny saved RNG states; report the first actual byte."""
    import numpy as np
    states = []
    for root in (left, right):
        with gzip.open(root / "rng250.pkl.gz", "rb") as stream:
            states.append(pickle.load(stream))
    def expanded(value):
        if hasattr(value, "detach"):
            return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.cpu().tolist()}
        if isinstance(value, np.ndarray):
            return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
        if isinstance(value, dict):
            return {key: expanded(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [expanded(item) for item in value]
        return semantic(value)
    return {key: first_difference(expanded(states[0][key]), expanded(states[1][key]), "$." + key) for key in states[0]}
