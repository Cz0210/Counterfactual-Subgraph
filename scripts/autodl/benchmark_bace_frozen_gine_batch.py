#!/usr/bin/env python3
"""Bounded CPU/GPU benchmark for the exact ordered frozen-GINE scorer."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import time

from src.baselines.frozen_gine_batch_scorer import FrozenGINEBatchScorer
from src.baselines.gcfexplainer_acceleration import write_fresh_json
from src.baselines.gcfexplainer_bace_adapter import load_bace_gcf_dataset
from src.data.molecular_graph_dataset import MolecularGraphData, collate_molecular_graphs
from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer
from src.oracles.gnn_oracle import load_gnn_checkpoint_bundle, sha256_file


def _record_smiles(record: dict) -> str:
    for key in ("smiles", "canonical_smiles", "original_smiles"):
        value = str(record.get(key, "")).strip()
        if value:
            return value
    raise ValueError("BACE benchmark record has no molecular SMILES field.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args(argv)


def _portable_rows(records: list[dict], featurizer: MolecularGraphFeaturizer) -> list[MolecularGraphData]:
    output: list[MolecularGraphData] = []
    for index, record in enumerate(records):
        features = featurizer.featurize(_record_smiles(record))
        output.append(
            MolecularGraphData(
                x=features.node_features,
                edge_index=features.edge_index,
                edge_attr=features.edge_features,
                y=int(record["label"]),
                molecule_id=str(record.get("molecule_id", index)),
                smiles=features.canonical_smiles,
                split="frozen_gine_benchmark",
                graph_sha256=features.graph_sha256,
            )
        )
    return output


def _tensor_sha256(value: object) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


def _timed(torch: object, scorer: FrozenGINEBatchScorer, rows: list[MolecularGraphData], repeats: int, device: str) -> tuple[object, float, dict]:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    score = None
    hidden_sha256: list[str] = []
    logits_sha256: list[str] = []
    with torch.inference_mode():
        for _ in range(repeats):
            score = scorer.score(rows, context={"benchmark": "cold_no_cache"})
            hidden_sha256.append(_tensor_sha256(score.graph_hidden))
            logits_sha256.append(_tensor_sha256(score.project_logits))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    assert score is not None
    return (
        score,
        time.perf_counter() - started,
        {
            "graph_hidden_sha256": hidden_sha256,
            "project_logits_sha256": logits_sha256,
            "graph_hidden_repeat_exact": len(set(hidden_sha256)) == 1,
            "project_logits_repeat_exact": len(set(logits_sha256)) == 1,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rows <= 0 or args.repeats <= 0:
        raise ValueError("--rows and --repeats must be positive")
    root = args.output_dir.expanduser().resolve(strict=False)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Benchmark output must be fresh: {root}")
    root.mkdir(parents=True)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Frozen-GINE CPU/GPU benchmark requires one CUDA GPU.")
    checkpoint = args.checkpoint_dir.expanduser().resolve(strict=True)
    _schema, _train, _val, generation, _summary = load_bace_gcf_dataset(
        args.dataset_dir
    )
    selected = [dict(row) for row in generation[: args.rows]]
    if len(selected) != args.rows:
        raise ValueError("BACE generation cohort is smaller than --rows.")

    cpu_model, cpu_metadata = load_gnn_checkpoint_bundle(checkpoint, device="cpu")
    gpu_model, gpu_metadata = load_gnn_checkpoint_bundle(checkpoint, device="cuda:0")
    for metadata in (cpu_metadata, gpu_metadata):
        card = metadata["model_card"]
        if (
            str(card.get("dataset", "")).lower() != "bace"
            or str(card.get("backbone", "")).lower() != "gine"
            or card.get("rf_oracle_used") is not False
        ):
            raise ValueError("Benchmark requires the frozen BACE GINE, never RF.")
    if cpu_metadata["checkpoint_id"] != gpu_metadata["checkpoint_id"]:
        raise ValueError("CPU/GPU checkpoint identities differ.")
    feature_schema = cpu_metadata["feature_schema"]
    rows = _portable_rows(selected, MolecularGraphFeaturizer(feature_schema))
    edge_dim = len(feature_schema.edge_fields)
    collate = lambda values: collate_molecular_graphs(
        values, edge_feature_dim=edge_dim
    )
    temperature = float(
        cpu_metadata["temperature_scaling"].get("temperature", 1.0)
    )
    common = {
        "temperature": temperature,
        "checkpoint_id": str(cpu_metadata["checkpoint_id"]),
        "collate_fn": collate,
        "cache_capacity": 0,
        "diagnostic_trace": True,
    }
    cpu_scorer = FrozenGINEBatchScorer(model=cpu_model, device="cpu", **common)
    gpu_scorer = FrozenGINEBatchScorer(model=gpu_model, device="cuda:0", **common)
    torch.cuda.reset_peak_memory_stats(0)
    cpu_score, cpu_seconds, cpu_repeat = _timed(
        torch, cpu_scorer, rows, args.repeats, "cpu"
    )
    gpu_score, gpu_seconds, gpu_repeat = _timed(
        torch, gpu_scorer, rows, args.repeats, "cuda:0"
    )
    cpu_logits = cpu_score.project_logits.detach().cpu()
    gpu_logits = gpu_score.project_logits.detach().cpu()
    finite = bool(torch.isfinite(cpu_logits).all() and torch.isfinite(gpu_logits).all())
    labels_equal = bool(torch.equal(cpu_logits.argmax(-1), gpu_logits.argmax(-1)))
    close = bool(torch.allclose(cpu_logits, gpu_logits, atol=args.atol, rtol=args.rtol))
    max_abs = float(torch.max(torch.abs(cpu_logits - gpu_logits)).item())
    cross_device_hidden_exact = bool(
        torch.equal(
            cpu_score.graph_hidden.detach().cpu(),
            gpu_score.graph_hidden.detach().cpu(),
        )
    )
    cross_device_logits_exact = bool(torch.equal(cpu_logits, gpu_logits))

    cached = FrozenGINEBatchScorer(
        model=cpu_model,
        device="cpu",
        temperature=temperature,
        checkpoint_id=str(cpu_metadata["checkpoint_id"]),
        collate_fn=collate,
        cache_capacity=1,
    )
    with torch.inference_mode():
        uncached_score = cached.score(rows, context={"cache_probe": 1})
        cached_score = cached.score(rows, context={"cache_probe": 1})
    cache_exact = bool(
        torch.equal(uncached_score.graph_hidden, cached_score.graph_hidden)
        and torch.equal(uncached_score.project_logits, cached_score.project_logits)
    )
    cpu_repeat_exact = bool(
        cpu_repeat["graph_hidden_repeat_exact"]
        and cpu_repeat["project_logits_repeat_exact"]
    )
    gpu_repeat_exact = bool(
        gpu_repeat["graph_hidden_repeat_exact"]
        and gpu_repeat["project_logits_repeat_exact"]
    )
    passed = (
        finite
        and labels_equal
        and close
        and cache_exact
        and cpu_repeat_exact
        and gpu_repeat_exact
    )
    payload = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAILED",
        "diagnostic_only": True,
        "authorizes_vrrw_replacement": False,
        "checkpoint_dir": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint / "model.pt"),
        "checkpoint_id": cpu_metadata["checkpoint_id"],
        "rows": len(rows),
        "repeats": args.repeats,
        "cpu_seconds": cpu_seconds,
        "gpu_seconds": gpu_seconds,
        "cpu_rows_per_second": len(rows) * args.repeats / cpu_seconds,
        "gpu_rows_per_second": len(rows) * args.repeats / gpu_seconds,
        "gpu_speedup_over_cpu": cpu_seconds / gpu_seconds,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "finite": finite,
        "labels_equal": labels_equal,
        "allclose": close,
        "atol": args.atol,
        "rtol": args.rtol,
        "max_abs_logit_difference": max_abs,
        "cpu_repeat": cpu_repeat,
        "gpu_repeat": gpu_repeat,
        "cpu_repeat_exact": cpu_repeat_exact,
        "gpu_repeat_exact": gpu_repeat_exact,
        "cross_device_graph_hidden_exact": cross_device_hidden_exact,
        "cross_device_project_logits_exact": cross_device_logits_exact,
        "raw_embedding_identity_is_device_specific": not cross_device_hidden_exact,
        "cache_exact": cache_exact,
        "cache_report": cached.report(),
        "full_order_preserved": True,
        "deduplication": False,
        "chunking": False,
    }
    if not math.isfinite(cpu_seconds + gpu_seconds + max_abs):
        payload["status"] = "FAILED"
    write_fresh_json(root / "benchmark.json", payload)
    if payload["status"] == "PASS":
        write_fresh_json(root / "_BENCHMARK_COMPLETE.json", payload)
        print("[BACE_FROZEN_GINE_CPU_GPU_BENCHMARK_PASS]")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
