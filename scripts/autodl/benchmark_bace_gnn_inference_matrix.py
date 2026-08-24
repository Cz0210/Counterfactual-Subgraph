#!/usr/bin/env python3
"""Benchmark frozen BACE GINE inference across a fixed CPU/GPU batch matrix.

The diagnostic times ordered batching/device transfer, prepared-batch model
inference, and collation-to-logits end-to-end inference.  Numerical/argmax
agreement is deliberately separate from raw-byte repeatability: an allclose
PASS never authorizes exact VRRW replay when CUDA embeddings change bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Sequence

from src.baselines.gcfexplainer_acceleration import write_fresh_json
from src.baselines.gcfexplainer_bace_adapter import load_bace_gcf_dataset
from src.data.molecular_graph_dataset import MolecularGraphData, collate_molecular_graphs
from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer
from src.oracles.gnn_oracle import load_gnn_checkpoint_bundle, sha256_file


DEFAULT_BATCH_SIZES = (1, 8, 32, 128, 512)


def _parse_batch_sizes(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "batch sizes must be comma-separated integers"
        ) from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("batch sizes must contain positive integers")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("batch sizes must not contain duplicates")
    if tuple(sorted(parsed)) != parsed:
        raise argparse.ArgumentTypeError("batch sizes must be strictly increasing")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--batch-sizes",
        type=_parse_batch_sizes,
        default=DEFAULT_BATCH_SIZES,
        help="strictly increasing comma-separated sizes",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args(argv)


def _record_smiles(record: dict[str, Any]) -> str:
    for key in ("smiles", "canonical_smiles", "original_smiles"):
        value = str(record.get(key, "")).strip()
        if value:
            return value
    raise ValueError("BACE benchmark record has no molecular SMILES field.")


def _portable_rows(
    records: Sequence[dict[str, Any]], featurizer: MolecularGraphFeaturizer
) -> list[MolecularGraphData]:
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
                split="frozen_gine_inference_benchmark",
                graph_sha256=features.graph_sha256,
            )
        )
    return output


def _tensor_sha256(value: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    array = tensor.numpy()
    identity = hashlib.sha256()
    identity.update(str(array.dtype).encode("ascii"))
    identity.update(json.dumps(list(array.shape)).encode("ascii"))
    identity.update(array.tobytes(order="C"))
    return identity.hexdigest()


def _percentile_nearest_rank(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _timing_summary(samples: Sequence[float], *, batch_size: int) -> dict[str, Any]:
    if not samples or any(value <= 0 or not math.isfinite(value) for value in samples):
        raise ValueError("timing samples must be finite and positive")
    median = float(statistics.median(samples))
    return {
        "repeats": len(samples),
        "total_seconds": float(sum(samples)),
        "min_seconds": float(min(samples)),
        "mean_seconds": float(statistics.fmean(samples)),
        "median_seconds": median,
        "p95_seconds": _percentile_nearest_rank(samples, 0.95),
        "median_rows_per_second": float(batch_size / median),
    }


def _synchronize(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _timed_calls(
    *,
    torch: Any,
    device: str,
    warmups: int,
    repeats: int,
    batch_size: int,
    function: Callable[[], Any],
    capture_model_output: bool,
) -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
    result: Any = None
    with torch.inference_mode():
        for _ in range(warmups):
            result = function()
        _synchronize(torch, device)
        samples: list[float] = []
        hidden_digests: list[str] = []
        logits_digests: list[str] = []
        for _ in range(repeats):
            _synchronize(torch, device)
            started = time.perf_counter()
            result = function()
            _synchronize(torch, device)
            samples.append(time.perf_counter() - started)
            if capture_model_output:
                hidden, logits = result
                hidden_digests.append(_tensor_sha256(hidden))
                logits_digests.append(_tensor_sha256(logits))
    repeat_identity = None
    if capture_model_output:
        repeat_identity = {
            "graph_hidden_sha256": hidden_digests,
            "project_logits_sha256": logits_digests,
            "graph_hidden_repeat_exact": len(set(hidden_digests)) == 1,
            "project_logits_repeat_exact": len(set(logits_digests)) == 1,
        }
    return result, _timing_summary(samples, batch_size=batch_size), repeat_identity


def _forward(model: Any, batch: Any, temperature: float) -> tuple[Any, Any]:
    hidden = model.encode_graph(batch)
    return hidden, model.classifier(hidden) / temperature


def _correctness(
    torch: Any, cpu_result: Any, gpu_result: Any, *, atol: float, rtol: float
) -> dict[str, Any]:
    cpu_hidden, cpu_logits = (value.detach().cpu() for value in cpu_result)
    gpu_hidden, gpu_logits = (value.detach().cpu() for value in gpu_result)
    finite = bool(
        torch.isfinite(cpu_hidden).all()
        and torch.isfinite(cpu_logits).all()
        and torch.isfinite(gpu_hidden).all()
        and torch.isfinite(gpu_logits).all()
    )
    return {
        "finite": finite,
        "argmax_equal": bool(torch.equal(cpu_logits.argmax(-1), gpu_logits.argmax(-1))),
        "logits_allclose": bool(
            torch.allclose(cpu_logits, gpu_logits, atol=atol, rtol=rtol)
        ),
        "hidden_allclose": bool(
            torch.allclose(cpu_hidden, gpu_hidden, atol=atol, rtol=rtol)
        ),
        "max_abs_logit_difference": float(
            torch.max(torch.abs(cpu_logits - gpu_logits)).item()
        ),
        "max_abs_hidden_difference": float(
            torch.max(torch.abs(cpu_hidden - gpu_hidden)).item()
        ),
        "cross_device_graph_hidden_exact": bool(torch.equal(cpu_hidden, gpu_hidden)),
        "cross_device_project_logits_exact": bool(torch.equal(cpu_logits, gpu_logits)),
    }


def _same_device_phase_correctness(
    torch: Any, left: Any, right: Any, *, atol: float, rtol: float
) -> dict[str, Any]:
    left_hidden, left_logits = (value.detach().cpu() for value in left)
    right_hidden, right_logits = (value.detach().cpu() for value in right)
    return {
        "argmax_equal": bool(torch.equal(left_logits.argmax(-1), right_logits.argmax(-1))),
        "logits_allclose": bool(
            torch.allclose(left_logits, right_logits, atol=atol, rtol=rtol)
        ),
        "hidden_allclose": bool(
            torch.allclose(left_hidden, right_hidden, atol=atol, rtol=rtol)
        ),
        "max_abs_logit_difference": float(
            torch.max(torch.abs(left_logits - right_logits)).item()
        ),
        "max_abs_hidden_difference": float(
            torch.max(torch.abs(left_hidden - right_hidden)).item()
        ),
    }


def _validate_checkpoint(metadata: dict[str, Any]) -> None:
    card = metadata["model_card"]
    if (
        str(card.get("dataset", "")).lower() != "bace"
        or str(card.get("backbone", "")).lower() != "gine"
        or card.get("rf_oracle_used") is not False
    ):
        raise ValueError("Benchmark requires the frozen BACE GINE, never RF.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("--warmups must be non-negative and --repeats positive")
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("--atol and --rtol must be non-negative")
    if tuple(args.batch_sizes) != DEFAULT_BATCH_SIZES:
        raise ValueError(
            "formal BACE benchmark requires batch sizes exactly 1,8,32,128,512"
        )

    root = args.output_dir.expanduser().resolve(strict=False)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Benchmark output must be fresh: {root}")
    root.mkdir(parents=True)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("BACE GINE inference matrix requires one CUDA GPU.")
    checkpoint = args.checkpoint_dir.expanduser().resolve(strict=True)
    _schema, train, _val, _generation, dataset_summary = load_bace_gcf_dataset(
        args.dataset_dir
    )
    if dataset_summary.get("test_loaded") is not False:
        raise ValueError("Inference benchmark must not load held-out BACE test data.")
    maximum = max(args.batch_sizes)
    # The formal matrix needs 512 rows while the strict-flip generation cohort
    # intentionally contains only 360 source-label parents.  Use the frozen
    # train prefix (869 rows), never held-out test data, and bind its order.
    selected = [dict(row) for row in train[:maximum]]
    if len(selected) != maximum:
        raise ValueError(
            f"BACE frozen train cohort has {len(selected)} rows; {maximum} required."
        )

    cpu_model, cpu_metadata = load_gnn_checkpoint_bundle(checkpoint, device="cpu")
    gpu_model, gpu_metadata = load_gnn_checkpoint_bundle(checkpoint, device="cuda:0")
    _validate_checkpoint(cpu_metadata)
    _validate_checkpoint(gpu_metadata)
    if cpu_metadata["checkpoint_id"] != gpu_metadata["checkpoint_id"]:
        raise ValueError("CPU/GPU checkpoint identities differ.")
    feature_schema = cpu_metadata["feature_schema"]
    rows = _portable_rows(selected, MolecularGraphFeaturizer(feature_schema))
    edge_dim = len(feature_schema.edge_fields)
    temperature = float(
        cpu_metadata["temperature_scaling"].get("temperature", 1.0)
    )

    def collate(values: Sequence[MolecularGraphData], device: str) -> Any:
        return collate_molecular_graphs(
            values, edge_feature_dim=edge_dim
        ).to(device)

    matrix: list[dict[str, Any]] = []
    all_correct = True
    cpu_raw_exact = True
    gpu_raw_exact = True
    for batch_size in args.batch_sizes:
        batch_rows = rows[:batch_size]
        torch.cuda.reset_peak_memory_stats(0)

        _cpu_batch, cpu_batching, _ = _timed_calls(
            torch=torch,
            device="cpu",
            warmups=args.warmups,
            repeats=args.repeats,
            batch_size=batch_size,
            function=lambda values=batch_rows: collate(values, "cpu"),
            capture_model_output=False,
        )
        _gpu_batch, gpu_batching, _ = _timed_calls(
            torch=torch,
            device="cuda:0",
            warmups=args.warmups,
            repeats=args.repeats,
            batch_size=batch_size,
            function=lambda values=batch_rows: collate(values, "cuda:0"),
            capture_model_output=False,
        )
        prepared_cpu = collate(batch_rows, "cpu")
        prepared_gpu = collate(batch_rows, "cuda:0")

        cpu_pure, cpu_pure_timing, cpu_pure_repeat = _timed_calls(
            torch=torch,
            device="cpu",
            warmups=args.warmups,
            repeats=args.repeats,
            batch_size=batch_size,
            function=lambda batch=prepared_cpu: _forward(
                cpu_model, batch, temperature
            ),
            capture_model_output=True,
        )
        gpu_pure, gpu_pure_timing, gpu_pure_repeat = _timed_calls(
            torch=torch,
            device="cuda:0",
            warmups=args.warmups,
            repeats=args.repeats,
            batch_size=batch_size,
            function=lambda batch=prepared_gpu: _forward(
                gpu_model, batch, temperature
            ),
            capture_model_output=True,
        )
        cpu_e2e, cpu_e2e_timing, cpu_e2e_repeat = _timed_calls(
            torch=torch,
            device="cpu",
            warmups=args.warmups,
            repeats=args.repeats,
            batch_size=batch_size,
            function=lambda values=batch_rows: _forward(
                cpu_model, collate(values, "cpu"), temperature
            ),
            capture_model_output=True,
        )
        gpu_e2e, gpu_e2e_timing, gpu_e2e_repeat = _timed_calls(
            torch=torch,
            device="cuda:0",
            warmups=args.warmups,
            repeats=args.repeats,
            batch_size=batch_size,
            function=lambda values=batch_rows: _forward(
                gpu_model, collate(values, "cuda:0"), temperature
            ),
            capture_model_output=True,
        )

        pure_correctness = _correctness(
            torch, cpu_pure, gpu_pure, atol=args.atol, rtol=args.rtol
        )
        e2e_correctness = _correctness(
            torch, cpu_e2e, gpu_e2e, atol=args.atol, rtol=args.rtol
        )
        cpu_phase = _same_device_phase_correctness(
            torch, cpu_pure, cpu_e2e, atol=args.atol, rtol=args.rtol
        )
        gpu_phase = _same_device_phase_correctness(
            torch, gpu_pure, gpu_e2e, atol=args.atol, rtol=args.rtol
        )
        row_correct = all(
            bool(check["finite"])
            and bool(check["argmax_equal"])
            and bool(check["logits_allclose"])
            and bool(check["hidden_allclose"])
            for check in (pure_correctness, e2e_correctness)
        ) and all(
            bool(check["argmax_equal"])
            and bool(check["logits_allclose"])
            and bool(check["hidden_allclose"])
            for check in (cpu_phase, gpu_phase)
        )
        assert cpu_pure_repeat is not None
        assert gpu_pure_repeat is not None
        assert cpu_e2e_repeat is not None
        assert gpu_e2e_repeat is not None
        cpu_row_exact = all(
            bool(repeat["graph_hidden_repeat_exact"])
            and bool(repeat["project_logits_repeat_exact"])
            for repeat in (cpu_pure_repeat, cpu_e2e_repeat)
        )
        gpu_row_exact = all(
            bool(repeat["graph_hidden_repeat_exact"])
            and bool(repeat["project_logits_repeat_exact"])
            for repeat in (gpu_pure_repeat, gpu_e2e_repeat)
        )
        all_correct = all_correct and row_correct
        cpu_raw_exact = cpu_raw_exact and cpu_row_exact
        gpu_raw_exact = gpu_raw_exact and gpu_row_exact
        matrix.append(
            {
                "batch_size": batch_size,
                "status": "PASS" if row_correct else "FAILED",
                "timing": {
                    "batching": {"cpu": cpu_batching, "gpu": gpu_batching},
                    "pure_model": {
                        "cpu": cpu_pure_timing,
                        "gpu": gpu_pure_timing,
                        "gpu_speedup_over_cpu_median": (
                            cpu_pure_timing["median_seconds"]
                            / gpu_pure_timing["median_seconds"]
                        ),
                    },
                    "end_to_end": {
                        "cpu": cpu_e2e_timing,
                        "gpu": gpu_e2e_timing,
                        "gpu_speedup_over_cpu_median": (
                            cpu_e2e_timing["median_seconds"]
                            / gpu_e2e_timing["median_seconds"]
                        ),
                    },
                },
                "correctness": {
                    "pure_model_cpu_vs_gpu": pure_correctness,
                    "end_to_end_cpu_vs_gpu": e2e_correctness,
                    "cpu_pure_vs_end_to_end": cpu_phase,
                    "gpu_pure_vs_end_to_end": gpu_phase,
                },
                "repeat_identity": {
                    "pure_model": {
                        "cpu": cpu_pure_repeat,
                        "gpu": gpu_pure_repeat,
                    },
                    "end_to_end": {
                        "cpu": cpu_e2e_repeat,
                        "gpu": gpu_e2e_repeat,
                    },
                    "cpu_raw_byte_repeat_exact": cpu_row_exact,
                    "gpu_raw_byte_repeat_exact": gpu_row_exact,
                },
                "gpu_peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(0)
                ),
                "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            }
        )

    payload = {
        "schema_version": "bace_gnn_inference_benchmark_v1",
        "status": "PASS" if all_correct and cpu_raw_exact else "FAILED",
        "benchmark_completed": True,
        "diagnostic_only": True,
        "paper_eligible": False,
        "authorizes_vrrw_replacement": False,
        "authorizes_gpu_raw_byte_identity": False,
        "exact_replay_status": (
            "PASS" if cpu_raw_exact and gpu_raw_exact else "FAILED_GPU_RAW_BYTE_REPEAT"
        ),
        "checkpoint_dir": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint / "model.pt"),
        "checkpoint_id": cpu_metadata["checkpoint_id"],
        "gpu_uuid": os.environ.get("AUTODL_PHYSICAL_GPU_UUID"),
        "gpu_index": os.environ.get("AUTODL_PHYSICAL_GPU_INDEX"),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "cohort": {
            "source": "frozen_bace_train_prefix_no_test_v1",
            "available_train_rows": len(train),
            "selected_rows": len(selected),
            "test_loaded": bool(dataset_summary.get("test_loaded", False)),
            "ordered_molecule_ids_sha256": hashlib.sha256(
                json.dumps(
                    [str(row["molecule_id"]) for row in selected],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
        "batch_sizes": list(args.batch_sizes),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "atol": args.atol,
        "rtol": args.rtol,
        "all_argmax_and_allclose_checks_pass": all_correct,
        "cpu_raw_byte_repeat_exact_all_batches": cpu_raw_exact,
        "gpu_raw_byte_repeat_exact_all_batches": gpu_raw_exact,
        "not_an_order_rng_or_batch_bug": bool(all_correct and not gpu_raw_exact),
        "matrix": matrix,
    }
    output = root / "bace_gnn_inference_benchmark.json"
    write_fresh_json(output, payload)
    completion = {
        "schema_version": "bace_gnn_inference_benchmark_complete_v1",
        "status": payload["status"],
        "result_path": str(output),
        "result_sha256": sha256_file(output),
        "benchmark_completed": True,
        "exact_replay_status": payload["exact_replay_status"],
    }
    write_fresh_json(root / "_BENCHMARK_COMPLETE.json", completion)
    if payload["status"] == "PASS":
        print("[BACE_GNN_INFERENCE_MATRIX_BENCHMARK_PASS]")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
