"""Bounded engineering-only comparison with the immutable T13 bridge source.

No campaign model, dataset, optimizer or output is opened.  The fixed chain,
ring, hole-padding and invalid-chemistry fixtures exercise storage/decoding,
not scientific coverage.  Passing this does not authorize a live handover.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import resource
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import torch

from src.baselines import globalgce_frozen_gine_bridge as optimized
from src.data.molecular_graph_featurizer import default_molecular_feature_schema
from src.models.molecular_gnn import MolecularGNN, MolecularGNNConfig

BASE_COMMIT = "c0eb892dd13ef05a5891c4acf1c5f4fef3966f67"
SOURCE_PATH = "src/baselines/globalgce_frozen_gine_bridge.py"


def reference_module(repo: Path):
    source = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{BASE_COMMIT}:{SOURCE_PATH}"]
    )
    name = "_t13_c0eb892d_reference"
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, f"git:{BASE_COMMIT}:{SOURCE_PATH}", "exec"), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def fixture(nodes=12, *, hole=False, ring=False, invalid=False, device="cpu"):
    """No RNG: positive class weights, unrestricted finite edge scores."""
    size = nodes + int(hole)
    active = [i for i in range(size) if not (hole and i == 1)]
    features = torch.full((1, size, 3), 0.1, dtype=torch.float32, device=device)
    features[:, :, 1] = 1.0
    if hole:
        features[:, 1] = torch.tensor([1.0, 0.0, 0.0], device=device)
    adjacency = torch.zeros((1, size, size), dtype=torch.float32, device=device)
    edges = torch.full((1, size * (size - 1) // 2, 4), -1.5, device=device)
    edges[:, :, 0] = 1.0
    pairs = list(zip(active, active[1:]))
    if ring and nodes > 2:
        pairs.append((active[-1], active[0]))
    if invalid:
        pairs = [(active[0], i) for i in active[1:]]
    for left, right in pairs:
        adjacency[0, left, right] = adjacency[0, right, left] = 0.9
        pos = optimized._edge_position(left, right)
        edges[0, pos, 0] = -1.5
        edges[0, pos, 1] = 2.0
    return features, adjacency, edges


def model(device="cpu"):
    schema = default_molecular_feature_schema()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        result = MolecularGNN(
            MolecularGNNConfig(backbone="gine", num_classes=3, num_layers=2,
                               hidden_dim=32, dropout=0.0, pooling="mean",
                               normalization="layer_norm", readout_layers=1),
            node_cardinalities=schema.node_cardinalities,
            edge_cardinalities=schema.edge_cardinalities,
        )
    return result.to(device)


def bridge(module, frozen, device="cpu"):
    return module.FrozenGINEDifferentiableBridge(
        copy.deepcopy(frozen), feature_schema=default_molecular_feature_schema(),
        atom_symbols=("C", "O"), bond_names=("no_edge", "single", "double", "triple"),
        checkpoint_id="synthetic-fixture-not-campaign-weights", temperature=1.7,
        device=device, expected_num_classes=3,
    )


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _rng():
    return {"cpu": torch.get_rng_state().clone(),
            "cuda": [v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def training_arm(call, values, target, reload_after_one=False):
    parameters = [torch.nn.Parameter(v.detach().clone()) for v in values]
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    records = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = call(*parameters)
        loss = torch.nn.functional.nll_loss(result["y_pred"], torch.tensor([target], device=parameters[0].device))
        loss.backward()
        record = {"logits": result["logits"].detach().clone(),
                  "loss": loss.detach().clone(),
                  "gradients": [None if v.grad is None else v.grad.detach().clone() for v in parameters],
                  "audit": result["bridge_audit"]}
        optimizer.step()
        scheduler.step()
        record.update(parameters=[v.detach().clone() for v in parameters],
                      optimizer=copy.deepcopy(optimizer.state_dict()),
                      scheduler=copy.deepcopy(scheduler.state_dict()), rng=_rng())
        records.append(record)
        if step == 0 and reload_after_one:
            buffer = io.BytesIO()
            torch.save({"parameters": parameters, "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "rng": _rng()}, buffer)
            buffer.seek(0)
            device = parameters[0].device
            # Adam's non-capturable step counter remains CPU even when its
            # parameters are CUDA. Map the container to CPU, then let the
            # optimizer's loader restore per-state device semantics.
            saved = torch.load(buffer, map_location="cpu", weights_only=False)
            parameters = [torch.nn.Parameter(v.detach().clone().to(device)) for v in saved["parameters"]]
            optimizer = torch.optim.Adam(parameters, lr=1e-4)
            optimizer.load_state_dict(saved["optimizer"])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            scheduler.load_state_dict(saved["scheduler"])
            torch.set_rng_state(saved["rng"]["cpu"].cpu())
            if saved["rng"]["cuda"]:
                torch.cuda.set_rng_state_all([v.cpu() for v in saved["rng"]["cuda"]])
    return records


def compare_records(left, right):
    for step, (a, b) in enumerate(zip(left, right), start=1):
        for field in a:
            if not _equal(a[field], b[field]):
                info = {"step": step, "component": field}
                if isinstance(a[field], torch.Tensor):
                    info["max_abs_error"] = float((a[field] - b[field]).abs().max())
                if field == "gradients":
                    info["max_abs_errors"] = [None if x is None else float((x-y).abs().max()) for x,y in zip(a[field], b[field])]
                return {"exact": False, "first_divergence": info}
    return {"exact": True, "first_divergence": None}


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def benchmark(repo, device="cpu", repeats=3):
    reference, source_sha = reference_module(Path(repo))
    frozen = model(device)
    old, new = bridge(reference, frozen, device), bridge(optimized, frozen, device)
    checks = []
    for options in ({"nodes": 1}, {"nodes": 7, "hole": True},
                    {"nodes": 9, "ring": True}, {"nodes": 7, "invalid": True}):
        values = fixture(device=device, **options)
        for target in (0, 2):
            before = _rng()
            a = training_arm(old, values, target)
            b = training_arm(new, values, target)
            reload = training_arm(new, values, target, reload_after_one=True)
            checks.append({"fixture": options, "target": target,
                           "reference_vs_optimized": compare_records(a, b),
                           "optimized_reload": compare_records(b, reload),
                           "rng_unchanged": _equal(before, _rng())})
    measurements = []
    for nodes in (16, 32, 64):
        values = fixture(nodes=nodes, hole=True, device=device)
        row = {"nodes": nodes, "graphs_per_repeat": 4, "repeats": repeats}
        for name, call in (("reference", old), ("optimized", new)):
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            times = []
            with torch.no_grad():
                call(*values)
                for _ in range(repeats):
                    _sync(device)
                    start = time.perf_counter()
                    for _ in range(4):
                        call(*values)
                    _sync(device)
                    times.append(time.perf_counter() - start)
            row[name] = {"wall_seconds": times, "median_seconds": statistics.median(times),
                         "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else None,
                         "process_peak_rss_native": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        row["median_speedup"] = row["reference"]["median_seconds"] / row["optimized"]["median_seconds"]
        measurements.append(row)
    return {"scope": "BOUNDED_SYNTHETIC_BRIDGE_NOT_FULL_T13_TRAINING",
            "base_commit": BASE_COMMIT, "base_source_sha256": source_sha,
            "optimized_source_sha256": hashlib.sha256(Path(optimized.__file__).read_bytes()).hexdigest(),
            "device": str(device), "torch_version": torch.__version__,
            "checks": checks, "measurements": measurements,
            "all_exact": all(v["reference_vs_optimized"]["exact"] and v["optimized_reload"]["exact"] and v["rng_unchanged"] for v in checks),
            "active_handover_ready": False,
            "remaining_gate": "real frozen GINE / pinned official generator fixed train workload and boundary-resume proof"}
