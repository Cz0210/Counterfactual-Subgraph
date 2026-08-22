"""Immutable A/B evidence gate for AutoDL ``shared_lowmem`` GPU slots.

The scheduler's two low-memory slots are a release feature, not a utilization
knob.  A task may opt into one only after two isolated single-task profiles and
one two-task same-GPU profile prove a measured aggregate throughput gain of at
least 20 percent without changing canonical results, OOMs, errors, sustained
CPU saturation, disk instability, or MPS.  The published gate embeds the
normalized profiles and hashes every source/result manifest so launch-time
validation never has to trust a mutable status label.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence


PROFILE_SCHEMA = "gpu_colocation_profile_v1"
GATE_SCHEMA = "gpu_colocation_benchmark_gate_v1"
PASS_MARKER = "[GPU_COLOCATION_BENCHMARK_PASS]"
MINIMUM_PROFILE_SECONDS = 600.0
MAXIMUM_PROFILE_SECONDS = 900.0
MINIMUM_THROUGHPUT_IMPROVEMENT = 0.20
MAXIMUM_SHARED_TASKS_PER_GPU = 2
MAXIMUM_VRAM_FRACTION = 0.70
ALLOWED_WORKLOAD_CLASSES = frozenset(
    {
        "bace_gcfexplainer_vrrw",
        "bace_comrecgc_generation",
        "verified_candidate_scoring",
    }
)
_HEX64 = re.compile(r"[0-9a-f]{64}")


class GPUColocationGateError(ValueError):
    """The requested shared-lowmem launch lacks valid measured evidence."""


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _physical_file(path: str | Path, *, label: str) -> Path:
    logical = Path(path).expanduser()
    if not logical.is_absolute() or logical.is_symlink():
        raise GPUColocationGateError(f"{label} must be an absolute physical file")
    resolved = logical.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise GPUColocationGateError(f"{label} is missing or empty: {resolved}")
    return resolved


def _read_object(path: str | Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    source = _physical_file(path, label=label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GPUColocationGateError(f"{label} is not valid JSON: {source}") from exc
    if not isinstance(value, dict):
        raise GPUColocationGateError(f"{label} must contain one JSON object")
    return source, value


def _positive_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GPUColocationGateError(f"{label} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise GPUColocationGateError(f"{label} must be a positive finite number")
    return result


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GPUColocationGateError(f"{label} must be a nonnegative integer")
    return int(value)


def _normalized_task(
    raw: Any, *, profile_label: str, verify_result_manifest: bool
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise GPUColocationGateError(f"{profile_label}.tasks entries must be objects")
    benchmark_key = str(raw.get("benchmark_key") or "").strip()
    workload_class = str(raw.get("workload_class") or "").strip()
    run_id = str(raw.get("run_id") or "").strip()
    if not benchmark_key or not run_id:
        raise GPUColocationGateError(
            f"{profile_label}.tasks require benchmark_key and run_id"
        )
    if workload_class not in ALLOWED_WORKLOAD_CLASSES:
        raise GPUColocationGateError(
            f"{profile_label} has unsupported workload_class={workload_class!r}"
        )
    science_pid = _nonnegative_int(
        raw.get("science_pid"), label=f"{profile_label}.{benchmark_key}.science_pid"
    )
    if science_pid <= 0:
        raise GPUColocationGateError(f"{profile_label} science_pid must be positive")
    duration = _positive_number(
        raw.get("duration_seconds"),
        label=f"{profile_label}.{benchmark_key}.duration_seconds",
    )
    if not MINIMUM_PROFILE_SECONDS <= duration <= MAXIMUM_PROFILE_SECONDS:
        raise GPUColocationGateError(
            f"{profile_label} task duration must be within 10--15 minutes"
        )
    throughput = _positive_number(
        raw.get("throughput_per_second"),
        label=f"{profile_label}.{benchmark_key}.throughput_per_second",
    )
    peak_vram = _positive_number(
        raw.get("peak_vram_mb"),
        label=f"{profile_label}.{benchmark_key}.peak_vram_mb",
    )
    canonical_hash = str(raw.get("canonical_result_sha256") or "")
    scientific_config_hash = str(raw.get("scientific_config_sha256") or "")
    expected_manifest_hash = str(raw.get("result_manifest_sha256") or "")
    if _HEX64.fullmatch(canonical_hash) is None:
        raise GPUColocationGateError(
            f"{profile_label}.{benchmark_key} lacks canonical result SHA256"
        )
    if _HEX64.fullmatch(scientific_config_hash) is None:
        raise GPUColocationGateError(
            f"{profile_label}.{benchmark_key} lacks scientific config SHA256"
        )
    if _HEX64.fullmatch(expected_manifest_hash) is None:
        raise GPUColocationGateError(
            f"{profile_label}.{benchmark_key} lacks result manifest SHA256"
        )
    manifest_value = str(raw.get("result_manifest") or "")
    if not manifest_value:
        raise GPUColocationGateError(
            f"{profile_label}.{benchmark_key} lacks result_manifest"
        )
    manifest = Path(manifest_value).expanduser()
    if not manifest.is_absolute() or manifest.is_symlink():
        raise GPUColocationGateError(
            f"{profile_label}.{benchmark_key} result_manifest must be absolute/physical"
        )
    if verify_result_manifest:
        resolved_manifest = _physical_file(
            manifest, label=f"{profile_label}.{benchmark_key}.result_manifest"
        )
        if sha256_file(resolved_manifest) != expected_manifest_hash:
            raise GPUColocationGateError(
                f"{profile_label}.{benchmark_key} result manifest SHA256 mismatch"
            )
        manifest_value = str(resolved_manifest)
    output_root = Path(str(raw.get("output_root") or "")).expanduser()
    if not output_root.is_absolute() or output_root.is_symlink():
        raise GPUColocationGateError(
            f"{profile_label}.{benchmark_key} output_root must be absolute/physical"
        )
    if verify_result_manifest:
        resolved_output = output_root.resolve(strict=True)
        if not resolved_output.is_dir():
            raise GPUColocationGateError(
                f"{profile_label}.{benchmark_key} output_root is not a directory"
            )
        try:
            Path(manifest_value).relative_to(resolved_output)
        except ValueError as exc:
            raise GPUColocationGateError(
                f"{profile_label}.{benchmark_key} result manifest escapes output_root"
            ) from exc
        output_root = resolved_output
    return {
        "benchmark_key": benchmark_key,
        "workload_class": workload_class,
        "run_id": run_id,
        "science_pid": science_pid,
        "duration_seconds": duration,
        "throughput_per_second": throughput,
        "peak_vram_mb": peak_vram,
        "output_root": str(output_root),
        "result_manifest": manifest_value,
        "result_manifest_sha256": expected_manifest_hash,
        "canonical_result_sha256": canonical_hash,
        "scientific_config_sha256": scientific_config_hash,
    }


def _normalized_profile(
    raw: Mapping[str, Any],
    *,
    expected_mode: str,
    expected_concurrency: int,
    expected_task_count: int,
    verify_result_manifests: bool,
) -> dict[str, Any]:
    label = f"{expected_mode} profile"
    if (
        raw.get("schema_version") != PROFILE_SCHEMA
        or raw.get("status") != "PASS"
        or raw.get("mode") != expected_mode
    ):
        raise GPUColocationGateError(f"{label} header is invalid")
    if raw.get("mps_enabled") is not False:
        raise GPUColocationGateError(f"{label} must prove MPS disabled")
    if raw.get("cpu_sustained_saturation") is not False:
        raise GPUColocationGateError(f"{label} reports sustained CPU saturation")
    if raw.get("disk_io_instability") is not False:
        raise GPUColocationGateError(f"{label} reports disk I/O instability")
    if _nonnegative_int(raw.get("oom_count"), label=f"{label}.oom_count") != 0:
        raise GPUColocationGateError(f"{label} reports an OOM")
    if _nonnegative_int(raw.get("error_count"), label=f"{label}.error_count") != 0:
        raise GPUColocationGateError(f"{label} reports an error")
    if raw.get("result_equivalence_checked") is not True:
        raise GPUColocationGateError(f"{label} lacks result-equivalence evidence")
    concurrency = _nonnegative_int(
        raw.get("max_concurrent_tasks"), label=f"{label}.max_concurrent_tasks"
    )
    if concurrency != expected_concurrency:
        raise GPUColocationGateError(
            f"{label} concurrency must be exactly {expected_concurrency}"
        )
    duration = _positive_number(
        raw.get("duration_seconds"), label=f"{label}.duration_seconds"
    )
    if not MINIMUM_PROFILE_SECONDS <= duration <= MAXIMUM_PROFILE_SECONDS:
        raise GPUColocationGateError(f"{label} duration must be within 10--15 minutes")
    interval = _positive_number(
        raw.get("sample_interval_seconds"), label=f"{label}.sample_interval_seconds"
    )
    if interval > 1.0:
        raise GPUColocationGateError(f"{label} must sample GPU utilization every second")
    sample_count = _nonnegative_int(
        raw.get("sample_count"), label=f"{label}.sample_count"
    )
    if float(sample_count) * interval < duration * 0.90:
        raise GPUColocationGateError(f"{label} has insufficient time-series samples")
    gpu_uuid = str(raw.get("gpu_uuid") or "").strip()
    gpu_name = str(raw.get("gpu_name") or "").strip()
    throughput_metric = str(raw.get("throughput_metric") or "").strip()
    if not gpu_uuid or not gpu_name or not throughput_metric:
        raise GPUColocationGateError(
            f"{label} requires GPU UUID/name and a common throughput metric"
        )
    total_memory = _positive_number(
        raw.get("gpu_total_memory_mb"), label=f"{label}.gpu_total_memory_mb"
    )
    aggregate = _positive_number(
        raw.get("aggregate_throughput_per_second"),
        label=f"{label}.aggregate_throughput_per_second",
    )
    peak_vram = _positive_number(
        raw.get("peak_vram_mb"), label=f"{label}.peak_vram_mb"
    )
    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or len(tasks_raw) != expected_task_count:
        raise GPUColocationGateError(
            f"{label} must describe exactly {expected_task_count} task(s)"
        )
    tasks = [
        _normalized_task(
            value,
            profile_label=label,
            verify_result_manifest=verify_result_manifests,
        )
        for value in tasks_raw
    ]
    keys = [row["benchmark_key"] for row in tasks]
    if len(set(keys)) != len(keys):
        raise GPUColocationGateError(f"{label} benchmark keys must be unique")
    measured_aggregate = sum(row["throughput_per_second"] for row in tasks)
    if not math.isclose(
        aggregate, measured_aggregate, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise GPUColocationGateError(
            f"{label} aggregate throughput differs from per-task measurements"
        )
    if peak_vram + _positive_number(
        raw.get("safety_margin_mb"), label=f"{label}.safety_margin_mb"
    ) >= MAXIMUM_VRAM_FRACTION * total_memory:
        raise GPUColocationGateError(f"{label} violates the strict <70% VRAM contract")
    return {
        "schema_version": PROFILE_SCHEMA,
        "status": "PASS",
        "mode": expected_mode,
        "gpu_uuid": gpu_uuid,
        "gpu_name": gpu_name,
        "gpu_total_memory_mb": total_memory,
        "duration_seconds": duration,
        "sample_interval_seconds": interval,
        "sample_count": sample_count,
        "max_concurrent_tasks": concurrency,
        "throughput_metric": throughput_metric,
        "aggregate_throughput_per_second": aggregate,
        "peak_vram_mb": peak_vram,
        "safety_margin_mb": float(raw["safety_margin_mb"]),
        "oom_count": 0,
        "error_count": 0,
        "cpu_sustained_saturation": False,
        "disk_io_instability": False,
        "result_equivalence_checked": True,
        "mps_enabled": False,
        "tasks": sorted(tasks, key=lambda row: row["benchmark_key"]),
    }


def _publish_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise GPUColocationGateError(f"refusing to replace gate evidence: {path}")
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{random.SystemRandom().randrange(1 << 32):08x}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_gpu_colocation_gate(
    *,
    single_profile_paths: Sequence[str | Path],
    colocated_profile_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build one immutable gate from two single-task and one paired profile."""

    if len(single_profile_paths) != MAXIMUM_SHARED_TASKS_PER_GPU:
        raise GPUColocationGateError("exactly two single-task profiles are required")
    source_rows: list[dict[str, str]] = []
    singles: list[dict[str, Any]] = []
    for index, value in enumerate(single_profile_paths):
        source, raw = _read_object(value, label=f"single profile {index}")
        singles.append(
            _normalized_profile(
                raw,
                expected_mode="single_task",
                expected_concurrency=1,
                expected_task_count=1,
                verify_result_manifests=True,
            )
        )
        source_rows.append({"path": str(source), "sha256": sha256_file(source)})
    colocated_source, colocated_raw = _read_object(
        colocated_profile_path, label="colocated profile"
    )
    colocated = _normalized_profile(
        colocated_raw,
        expected_mode="shared_lowmem_pair",
        expected_concurrency=2,
        expected_task_count=2,
        verify_result_manifests=True,
    )
    source_rows.append(
        {"path": str(colocated_source), "sha256": sha256_file(colocated_source)}
    )
    all_profiles = [*singles, colocated]
    if len({row["gpu_uuid"] for row in all_profiles}) != 1:
        raise GPUColocationGateError("A/B profiles were not measured on one physical GPU")
    if len({row["gpu_name"] for row in all_profiles}) != 1 or len(
        {row["gpu_total_memory_mb"] for row in all_profiles}
    ) != 1:
        raise GPUColocationGateError("A/B GPU model/memory identities differ")
    if len({row["throughput_metric"] for row in all_profiles}) != 1:
        raise GPUColocationGateError("A/B profiles use different throughput metrics")
    single_tasks = {
        row["tasks"][0]["benchmark_key"]: row["tasks"][0] for row in singles
    }
    colocated_tasks = {
        row["benchmark_key"]: row for row in colocated["tasks"]
    }
    if len(single_tasks) != 2 or set(single_tasks) != set(colocated_tasks):
        raise GPUColocationGateError("A/B profiles do not cover the same two workloads")
    mismatches = [
        key
        for key in sorted(single_tasks)
        if (
            single_tasks[key]["workload_class"]
            != colocated_tasks[key]["workload_class"]
            or single_tasks[key]["scientific_config_sha256"]
            != colocated_tasks[key]["scientific_config_sha256"]
            or single_tasks[key]["canonical_result_sha256"]
            != colocated_tasks[key]["canonical_result_sha256"]
        )
    ]
    if mismatches:
        raise GPUColocationGateError(
            "co-location changed canonical result or scientific config identity: "
            + ",".join(mismatches)
        )
    single_reference = sum(
        row["aggregate_throughput_per_second"] for row in singles
    ) / float(len(singles))
    colocated_throughput = float(colocated["aggregate_throughput_per_second"])
    improvement = colocated_throughput / single_reference - 1.0
    if improvement + 1e-12 < MINIMUM_THROUGHPUT_IMPROVEMENT:
        raise GPUColocationGateError("aggregate co-location throughput gain is below 20%")
    pair = sorted(row["workload_class"] for row in colocated["tasks"])
    peak_by_class: dict[str, float] = {}
    for row in colocated["tasks"]:
        key = row["workload_class"]
        peak_by_class[key] = max(peak_by_class.get(key, 0.0), row["peak_vram_mb"])
    root = Path(output_dir).expanduser()
    if not root.is_absolute() or root.is_symlink() or root.exists():
        raise GPUColocationGateError("gate output directory must be fresh, absolute, physical")
    root.mkdir(parents=True, exist_ok=False)
    payload: dict[str, Any] = {
        "schema_version": GATE_SCHEMA,
        "status": "PASS",
        "marker": PASS_MARKER,
        "minimum_profile_seconds": MINIMUM_PROFILE_SECONDS,
        "maximum_profile_seconds": MAXIMUM_PROFILE_SECONDS,
        "minimum_throughput_improvement_fraction": MINIMUM_THROUGHPUT_IMPROVEMENT,
        "maximum_shared_tasks_per_gpu": MAXIMUM_SHARED_TASKS_PER_GPU,
        "maximum_vram_fraction": MAXIMUM_VRAM_FRACTION,
        "same_gpu_uuid": True,
        "gpu_uuid": colocated["gpu_uuid"],
        "gpu_name": colocated["gpu_name"],
        "gpu_total_memory_mb": colocated["gpu_total_memory_mb"],
        "throughput_metric": colocated["throughput_metric"],
        "single_task_reference_throughput_per_second": single_reference,
        "colocated_aggregate_throughput_per_second": colocated_throughput,
        "aggregate_throughput_improvement_fraction": improvement,
        "result_equivalence": True,
        "result_mismatch_count": 0,
        "oom_count": 0,
        "error_count": 0,
        "cpu_sustained_saturation": False,
        "disk_io_instability": False,
        "mps_enabled": False,
        "authorized_workload_pair": pair,
        "authorized_workload_pair_sha256": _stable_hash(pair),
        "peak_vram_mb_by_workload": dict(sorted(peak_by_class.items())),
        "single_profiles": singles,
        "colocated_profile": colocated,
        "source_profiles": source_rows,
    }
    payload["gate_payload_sha256"] = _stable_hash(payload)
    _publish_json(root / "GPU_COLOCATION_BENCHMARK_GATE.json", payload)
    _publish_json(root / "PASS.json", {"status": "PASS", "marker": PASS_MARKER})
    return payload


def _validate_gate_payload(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": GATE_SCHEMA,
        "status": "PASS",
        "marker": PASS_MARKER,
        "minimum_profile_seconds": MINIMUM_PROFILE_SECONDS,
        "maximum_profile_seconds": MAXIMUM_PROFILE_SECONDS,
        "minimum_throughput_improvement_fraction": MINIMUM_THROUGHPUT_IMPROVEMENT,
        "maximum_shared_tasks_per_gpu": MAXIMUM_SHARED_TASKS_PER_GPU,
        "maximum_vram_fraction": MAXIMUM_VRAM_FRACTION,
        "same_gpu_uuid": True,
        "result_equivalence": True,
        "result_mismatch_count": 0,
        "oom_count": 0,
        "error_count": 0,
        "cpu_sustained_saturation": False,
        "disk_io_instability": False,
        "mps_enabled": False,
    }
    failures = [key for key, value in expected.items() if payload.get(key) != value]
    if failures:
        raise GPUColocationGateError(
            "co-location gate contract failed: " + ",".join(failures)
        )
    sources = payload.get("source_profiles")
    if not isinstance(sources, list) or len(sources) != 3:
        raise GPUColocationGateError("co-location gate lacks three source profiles")
    rebuilt: list[dict[str, Any]] = []
    for index, source_row in enumerate(sources):
        if not isinstance(source_row, Mapping):
            raise GPUColocationGateError("co-location source profile row is invalid")
        source, raw = _read_object(
            str(source_row.get("path") or ""),
            label=f"co-location source profile {index}",
        )
        if source_row.get("sha256") != sha256_file(source):
            raise GPUColocationGateError("co-location source profile SHA256 changed")
        rebuilt.append(
            _normalized_profile(
                raw,
                expected_mode=("single_task" if index < 2 else "shared_lowmem_pair"),
                expected_concurrency=(1 if index < 2 else 2),
                expected_task_count=(1 if index < 2 else 2),
                verify_result_manifests=True,
            )
        )
    if rebuilt[:2] != payload.get("single_profiles") or rebuilt[2] != payload.get(
        "colocated_profile"
    ):
        raise GPUColocationGateError(
            "co-location normalized source profiles changed after publication"
        )
    if len({row["gpu_uuid"] for row in rebuilt}) != 1:
        raise GPUColocationGateError("co-location source profiles are not same-GPU")
    if len({row["throughput_metric"] for row in rebuilt}) != 1:
        raise GPUColocationGateError("co-location throughput metrics differ")
    single_tasks = {
        row["tasks"][0]["benchmark_key"]: row["tasks"][0] for row in rebuilt[:2]
    }
    colocated_tasks = {
        row["benchmark_key"]: row for row in rebuilt[2]["tasks"]
    }
    if set(single_tasks) != set(colocated_tasks):
        raise GPUColocationGateError("co-location benchmark keys changed")
    for key in single_tasks:
        if any(
            single_tasks[key][field] != colocated_tasks[key][field]
            for field in (
                "workload_class",
                "scientific_config_sha256",
                "canonical_result_sha256",
            )
        ):
            raise GPUColocationGateError(
                "co-location result/config identity changed after publication"
            )
    recomputed_reference = sum(
        row["aggregate_throughput_per_second"] for row in rebuilt[:2]
    ) / 2.0
    recomputed_colocated = float(rebuilt[2]["aggregate_throughput_per_second"])
    recomputed_improvement = recomputed_colocated / recomputed_reference - 1.0
    numeric_fields = {
        "single_task_reference_throughput_per_second": recomputed_reference,
        "colocated_aggregate_throughput_per_second": recomputed_colocated,
        "aggregate_throughput_improvement_fraction": recomputed_improvement,
    }
    if any(
        not math.isclose(
            float(payload.get(key, math.nan)), value, rel_tol=1e-9, abs_tol=1e-12
        )
        for key, value in numeric_fields.items()
    ):
        raise GPUColocationGateError("co-location throughput aggregate was altered")
    improvement = _positive_number(
        payload.get("aggregate_throughput_improvement_fraction"),
        label="gate.aggregate_throughput_improvement_fraction",
    )
    if improvement + 1e-12 < MINIMUM_THROUGHPUT_IMPROVEMENT:
        raise GPUColocationGateError("co-location gate speedup is below 20%")
    pair = payload.get("authorized_workload_pair")
    measured_pair = sorted(row["workload_class"] for row in rebuilt[2]["tasks"])
    if (
        not isinstance(pair, list)
        or len(pair) != 2
        or any(value not in ALLOWED_WORKLOAD_CLASSES for value in pair)
        or pair != sorted(pair)
        or pair != measured_pair
        or payload.get("authorized_workload_pair_sha256") != _stable_hash(pair)
    ):
        raise GPUColocationGateError("co-location gate workload pair is invalid")
    peaks = payload.get("peak_vram_mb_by_workload")
    if not isinstance(peaks, Mapping):
        raise GPUColocationGateError("co-location gate lacks workload VRAM peaks")
    for workload_class in set(pair):
        _positive_number(peaks.get(workload_class), label=f"peak:{workload_class}")
    measured_peaks: dict[str, float] = {}
    for row in rebuilt[2]["tasks"]:
        workload_class = row["workload_class"]
        measured_peaks[workload_class] = max(
            measured_peaks.get(workload_class, 0.0), row["peak_vram_mb"]
        )
    if dict(peaks) != measured_peaks:
        raise GPUColocationGateError("co-location workload VRAM peaks were altered")
    recorded_payload_hash = str(payload.get("gate_payload_sha256") or "")
    unsigned = dict(payload)
    unsigned.pop("gate_payload_sha256", None)
    if (
        _HEX64.fullmatch(recorded_payload_hash) is None
        or recorded_payload_hash != _stable_hash(unsigned)
    ):
        raise GPUColocationGateError("co-location gate payload digest is invalid")


def validate_gpu_colocation_gate(
    path: str | Path,
    *,
    expected_sha256: str,
    workload_class: str,
    memory_reservation_mb: int,
    gpu_name: str | None = None,
    gpu_total_memory_mb: int | None = None,
) -> dict[str, Any]:
    """Validate gate bytes plus one task reservation at schema or launch time."""

    source, payload = _read_object(path, label="GPU co-location gate")
    actual_sha256 = sha256_file(source)
    if (
        _HEX64.fullmatch(str(expected_sha256)) is None
        or actual_sha256 != str(expected_sha256)
    ):
        raise GPUColocationGateError("GPU co-location gate SHA256 mismatch")
    _validate_gate_payload(payload)
    pair = list(payload["authorized_workload_pair"])
    if workload_class not in pair:
        raise GPUColocationGateError(
            f"workload {workload_class!r} was not benchmarked by this gate"
        )
    if (
        isinstance(memory_reservation_mb, bool)
        or not isinstance(memory_reservation_mb, int)
        or memory_reservation_mb <= 0
    ):
        raise GPUColocationGateError("shared-lowmem reservation must be positive")
    measured_peak = float(payload["peak_vram_mb_by_workload"][workload_class])
    if float(memory_reservation_mb) + 1e-9 < measured_peak:
        raise GPUColocationGateError(
            "shared-lowmem reservation is below the measured task peak VRAM"
        )
    if gpu_name is not None and str(gpu_name) != str(payload.get("gpu_name")):
        raise GPUColocationGateError("launch GPU model differs from benchmark GPU")
    if gpu_total_memory_mb is not None and int(gpu_total_memory_mb) < int(
        float(payload.get("gpu_total_memory_mb", 0))
    ):
        raise GPUColocationGateError("launch GPU has less memory than benchmark GPU")
    return {
        "path": str(source),
        "sha256": actual_sha256,
        "status": "PASS",
        "marker": PASS_MARKER,
        "workload_class": workload_class,
        "measured_peak_vram_mb": measured_peak,
        "memory_reservation_mb": int(memory_reservation_mb),
        "authorized_workload_pair": pair,
        "authorized_workload_pair_sha256": payload[
            "authorized_workload_pair_sha256"
        ],
        "aggregate_throughput_improvement_fraction": payload[
            "aggregate_throughput_improvement_fraction"
        ],
        "gpu_name": payload["gpu_name"],
        "gpu_total_memory_mb": payload["gpu_total_memory_mb"],
    }


def validate_authorized_pair(
    gate_evidence: Mapping[str, Any], workload_classes: Sequence[str]
) -> None:
    """Require an actual two-slot class multiset to equal the benchmarked pair."""

    expected = Counter(gate_evidence.get("authorized_workload_pair") or ())
    actual = Counter(str(value) for value in workload_classes)
    if actual != expected:
        raise GPUColocationGateError(
            f"shared-lowmem pair differs from benchmark: actual={dict(actual)}, "
            f"expected={dict(expected)}"
        )


__all__ = [
    "ALLOWED_WORKLOAD_CLASSES",
    "GATE_SCHEMA",
    "GPUColocationGateError",
    "MAXIMUM_SHARED_TASKS_PER_GPU",
    "MINIMUM_THROUGHPUT_IMPROVEMENT",
    "PASS_MARKER",
    "PROFILE_SCHEMA",
    "build_gpu_colocation_gate",
    "sha256_file",
    "validate_authorized_pair",
    "validate_gpu_colocation_gate",
]
