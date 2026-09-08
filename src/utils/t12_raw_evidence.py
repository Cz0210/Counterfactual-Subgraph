"""T12 selected-action raw evidence, collected only from real executed calls.

This is not the scientific cache. It never supplies a value to the walker, never
inverts probabilities/masks, and never evaluates a model to fill a missing row.
"""
from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
from typing import Any

from src.utils.main_ready_task_specs import stable_sha256
from src.utils.t12_shadow_recovery import SelectedStepObserver, tensor_value


class RawEvidenceResolver:
    def __init__(self, contract_sha: str, *, max_bytes: int = 64 << 20):
        if len(contract_sha) != 64 or not 0 < max_bytes <= 1 << 30:
            raise ValueError("T12_RAW_EVIDENCE_CONTRACT_OR_BOUND_INVALID")
        self.contract_sha, self.max_bytes = contract_sha, max_bytes
        self.rows: dict[str, dict] = {}
        self.bytes = 0

    def remember(self, row: dict) -> None:
        required = {"graph_identity", "model_graph", "canonical_query_sha",
                    "canonical_probabilities", "observed_probabilities", "valid_fullgraph", "raw_classifier",
                    "raw_neurosed", "raw_normalizer", "threshold"}
        if set(row) != required:
            raise ValueError("T12_RAW_ROW_SCHEMA_CHANGED")
        value = copy.deepcopy(row)
        # First observed here is not necessarily the original pre-checkpoint
        # first-seen value. Canonical probabilities must separately match below.
        value["capture_scope"] = "ACTUAL_SHADOW_CALL_NOT_HISTORICAL_FIRST_SEEN"
        value["contract_sha"] = self.contract_sha
        value["content_sha"] = stable_sha256(value)
        key = value["graph_identity"]
        if key in self.rows:
            return
        size = len(json.dumps(value, separators=(",", ":"), allow_nan=False).encode())
        if self.bytes + size > self.max_bytes:
            raise ValueError("T12_RAW_EVIDENCE_BOUND_REACHED_SAVE_BEFORE_MORE_WORK")
        self.rows[key] = value
        self.bytes += size

    def resolve(self, graph_identity: str, *, canonical_probabilities=None) -> dict:
        row = self.rows.get(graph_identity)
        if row is None:
            return {"status": "CACHE_RAW_EVIDENCE_MISSING", "graph_identity": graph_identity}
        if row["contract_sha"] != self.contract_sha or stable_sha256(
                {k: v for k, v in row.items() if k != "content_sha"}) != row["content_sha"]:
            raise ValueError("T12_RAW_EVIDENCE_BINDING_CHANGED")
        if (canonical_probabilities is not None
                and tensor_value(canonical_probabilities) != row["canonical_probabilities"]):
            return {"status": "CACHE_CANONICAL_PROBABILITY_BINDING_MISMATCH",
                    "graph_identity": graph_identity}
        missing = []
        if row["valid_fullgraph"] and row["raw_classifier"] is None:
            missing.append("RAW_CLASSIFIER_LOGITS")
        if row["raw_neurosed"] is None or row["raw_normalizer"] is None:
            missing.append("RAW_NEUROSED")
        return {"status": "PASS" if not missing else "CACHE_RAW_EVIDENCE_MISSING",
                "missing": missing, "evidence": copy.deepcopy(row)}

    def save(self, path: Path) -> None:
        with path.open("xb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
                for row in self.rows.values():
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False).encode() + b"\n")
            raw.flush()
            import os
            os.fsync(raw.fileno())

    def load(self, path: Path) -> None:
        with gzip.open(path, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("contract_sha") != self.contract_sha or stable_sha256(
                        {k: v for k, v in row.items() if k != "content_sha"}) != row.get("content_sha"):
                    raise ValueError("T12_RAW_INPUT_BINDING_CHANGED")
                key = row["graph_identity"]
                if key in self.rows and self.rows[key] != row:
                    raise ValueError("T12_RAW_INPUT_CONFLICT")
                size = len(line.encode())
                if self.bytes + size > self.max_bytes:
                    raise ValueError("T12_RAW_EVIDENCE_BOUND_REACHED_SAVE_BEFORE_MORE_WORK")
                self.rows[key] = row
                self.bytes += size


class BoundSelectedStepObserver(SelectedStepObserver):
    """Add exact graph/query-to-raw bindings to the existing fresh observer."""
    def __init__(self, *args, resolver: RawEvidenceResolver, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolver = resolver
        self.raw_by_position = {}
        self.distance_by_query = {}
        self.bridge_event_start = None

    def _flush(self, globals_: dict) -> None:
        if self.pending is not None:
            identity = self.pending.get("after_graph")
            evidence = self.resolver.resolve(identity) if identity else {
                "status": "NO_SELECTED_GRAPH_AT_TELEPORT_BOUNDARY"}
            self.pending["selected_raw_evidence"] = evidence
            # No-query cache hits are valid only with real bound raw evidence.
            self.pending["query_events"] = self.query_events.copy()
            self.pending["missing_raw_fields"] = (
                [] if evidence["status"] == "PASS" else [evidence["status"]])
            if self.pending.get("exceptional_return"):
                raise ValueError("T12_EXCEPTIONAL_TRANSITION_NOT_COMMITTED")
            self.pending["after"] = self._science_state(globals_)
            self.ledger.append(self.pending)
            self.pending = None

    def callback(self, frame: Any, event: str, result: Any):
        name, filename = frame.f_code.co_name, frame.f_code.co_filename
        local = frame.f_locals
        if (name == "call" and filename.endswith("tastemolnet_gcf_full_resume.py")
                and event == "call"):
            self.bridge_event_start = len(self.query_events)
            self.raw_by_position, self.distance_by_query = {}, {}
        super().callback(frame, event, result)
        if (name == "score" and filename.endswith("tastemolnet_gcf_smoke.py")
                and event == "return" and "valid_positions" in local
                and self.bridge_event_start is not None):
            # The production adapter explicitly uses one complete, uncached
            # ordered batch. Do not guess row alignment for another adapter.
            if local["self"].canonical_replay_cache_enabled:
                raise ValueError("T12_RAW_CAPTURE_UNSUPPORTED_CANONICAL_ADAPTER_CACHE")
            batches = [e["tensor"] for e in self.query_events[self.bridge_event_start:]
                       if e["kind"] == "RAW_CLASSIFIER_LOGITS"]
            positions = local["valid_positions"]
            if positions:
                if len(batches) != 1 or len(batches[0]["values"]) != len(positions):
                    raise ValueError("T12_RAW_CLASSIFIER_ROW_ALIGNMENT_MISSING")
                batch = batches[0]
                for index, values in zip(positions, batch["values"], strict=True):
                    self.raw_by_position[index] = {"dtype": batch["dtype"],
                        "shape": batch["shape"][1:], "values": values}
        if (name == "__call__" and filename.endswith("tastemolnet_gcf_replay_canary.py")
                and event == "return" and "distances" in local
                and self.bridge_event_start is not None):
            from src.baselines.tastemolnet_gcf_full_resume import _t12_neurosed_query_sha256
            if local["distances"].numel() * local["distances"].element_size() > 64 << 20:
                raise ValueError("T12_RAW_DISTANCE_CAPTURE_BOUND_EXCEEDED")
            for index, graph in enumerate(local["dataset"]):
                key = _t12_neurosed_query_sha256(graph)
                self.distance_by_query[key] = (tensor_value(local["distances"][index]),
                    tensor_value(local["sums"][index]), float(local["threshold"]))
        if (name == "call" and filename.endswith("tastemolnet_gcf_full_resume.py")
                and event == "return" and result is not None and "graph_hashes" in local):
            bridge = local["self"]
            for index, identity in enumerate(local["graph_hashes"]):
                query = local["canonical_query_hashes"][index]
                distance, normalizer, threshold = self.distance_by_query.get(query, (None, None, None))
                record = bridge.records[identity]
                self.resolver.remember({"graph_identity": identity,
                    "model_graph": local["model_payloads"][index],
                    "canonical_query_sha": query,
                    "canonical_probabilities": tensor_value(record.probabilities),
                    "observed_probabilities": tensor_value(local["batch"].probabilities[index]),
                    "valid_fullgraph": bool(record.valid_fullgraph),
                    "raw_classifier": self.raw_by_position.get(index),
                    "raw_neurosed": distance, "raw_normalizer": normalizer,
                    "threshold": threshold})
            self.bridge_event_start = None
