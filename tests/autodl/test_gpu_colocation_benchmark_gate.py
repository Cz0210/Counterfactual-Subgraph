from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from src.utils.autodl_gpu_colocation_gate import (
    GPUColocationGateError,
    build_gpu_colocation_gate,
    sha256_file,
    validate_authorized_pair,
    validate_gpu_colocation_gate,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _profile(
    tmp_path: Path,
    *,
    name: str,
    mode: str,
    rows: list[tuple[str, str, float, str]],
    aggregate: float,
    **overrides: object,
) -> Path:
    tasks = []
    for key, workload, throughput, canonical_hash in rows:
        output = tmp_path / "outputs" / name / key
        manifest = output / "result_manifest.json"
        _write(manifest, {"key": key, "canonical": canonical_hash})
        tasks.append(
            {
                "benchmark_key": key,
                "workload_class": workload,
                "run_id": f"{name}-{key}",
                "science_pid": 1000 + len(tasks),
                "duration_seconds": 600,
                "throughput_per_second": throughput,
                "peak_vram_mb": 8000 + len(tasks) * 1000,
                "output_root": str(output.resolve()),
                "result_manifest": str(manifest.resolve()),
                "result_manifest_sha256": sha256_file(manifest),
                "canonical_result_sha256": canonical_hash,
                "scientific_config_sha256": hashlib.sha256(
                    f"config:{key}".encode("utf-8")
                ).hexdigest(),
            }
        )
    payload = {
        "schema_version": "gpu_colocation_profile_v1",
        "status": "PASS",
        "mode": mode,
        "gpu_uuid": "GPU-same",
        "gpu_name": "NVIDIA A800 80GB PCIe",
        "gpu_total_memory_mb": 81920,
        "duration_seconds": 600,
        "sample_interval_seconds": 1,
        "sample_count": 600,
        "max_concurrent_tasks": 1 if mode == "single_task" else 2,
        "throughput_metric": "normalized_work_units_per_second",
        "aggregate_throughput_per_second": aggregate,
        "peak_vram_mb": 9000 if mode == "single_task" else 18000,
        "safety_margin_mb": 8192,
        "oom_count": 0,
        "error_count": 0,
        "cpu_sustained_saturation": False,
        "disk_io_instability": False,
        "result_equivalence_checked": True,
        "mps_enabled": False,
        "tasks": tasks,
        **overrides,
    }
    path = tmp_path / f"{name}.json"
    _write(path, payload)
    return path


def build_test_gate(tmp_path: Path) -> tuple[Path, str]:
    gcf = "a" * 64
    comrec = "b" * 64
    single_gcf = _profile(
        tmp_path,
        name="single-gcf",
        mode="single_task",
        rows=[("gcf", "bace_gcfexplainer_vrrw", 10.0, gcf)],
        aggregate=10.0,
    )
    single_comrec = _profile(
        tmp_path,
        name="single-comrec",
        mode="single_task",
        rows=[("comrec", "bace_comrecgc_generation", 10.0, comrec)],
        aggregate=10.0,
    )
    colocated = _profile(
        tmp_path,
        name="colocated",
        mode="shared_lowmem_pair",
        rows=[
            ("gcf", "bace_gcfexplainer_vrrw", 6.5, gcf),
            ("comrec", "bace_comrecgc_generation", 6.5, comrec),
        ],
        aggregate=13.0,
    )
    build_gpu_colocation_gate(
        single_profile_paths=[single_gcf, single_comrec],
        colocated_profile_path=colocated,
        output_dir=tmp_path / "gate",
    )
    path = tmp_path / "gate/GPU_COLOCATION_BENCHMARK_GATE.json"
    return path, sha256_file(path)


def test_gate_requires_real_profiles_and_authorizes_exact_pair(tmp_path: Path) -> None:
    path, digest = build_test_gate(tmp_path)
    evidence = validate_gpu_colocation_gate(
        path,
        expected_sha256=digest,
        workload_class="bace_gcfexplainer_vrrw",
        memory_reservation_mb=9000,
        gpu_name="NVIDIA A800 80GB PCIe",
        gpu_total_memory_mb=81920,
    )
    assert evidence["aggregate_throughput_improvement_fraction"] == pytest.approx(
        0.30
    )
    validate_authorized_pair(
        evidence,
        ["bace_gcfexplainer_vrrw", "bace_comrecgc_generation"],
    )
    with pytest.raises(GPUColocationGateError, match="differs from benchmark"):
        validate_authorized_pair(
            evidence,
            ["bace_gcfexplainer_vrrw", "bace_gcfexplainer_vrrw"],
        )


def test_gate_rejects_result_difference_and_sub_twenty_percent_gain(
    tmp_path: Path,
) -> None:
    first = _profile(
        tmp_path,
        name="first",
        mode="single_task",
        rows=[("one", "bace_gcfexplainer_vrrw", 10.0, "a" * 64)],
        aggregate=10.0,
    )
    second = _profile(
        tmp_path,
        name="second",
        mode="single_task",
        rows=[("two", "bace_comrecgc_generation", 10.0, "b" * 64)],
        aggregate=10.0,
    )
    changed = _profile(
        tmp_path,
        name="changed",
        mode="shared_lowmem_pair",
        rows=[
            ("one", "bace_gcfexplainer_vrrw", 6.5, "c" * 64),
            ("two", "bace_comrecgc_generation", 6.5, "b" * 64),
        ],
        aggregate=13.0,
    )
    with pytest.raises(GPUColocationGateError, match="changed canonical"):
        build_gpu_colocation_gate(
            single_profile_paths=[first, second],
            colocated_profile_path=changed,
            output_dir=tmp_path / "changed-gate",
        )


def test_profile_rejects_inflated_aggregate_and_cross_mode_config_drift(
    tmp_path: Path,
) -> None:
    first = _profile(
        tmp_path,
        name="first-aggregate",
        mode="single_task",
        rows=[("one", "bace_gcfexplainer_vrrw", 10.0, "a" * 64)],
        aggregate=100.0,
    )
    second = _profile(
        tmp_path,
        name="second-aggregate",
        mode="single_task",
        rows=[("two", "bace_comrecgc_generation", 10.0, "b" * 64)],
        aggregate=10.0,
    )
    colocated = _profile(
        tmp_path,
        name="pair-aggregate",
        mode="shared_lowmem_pair",
        rows=[
            ("one", "bace_gcfexplainer_vrrw", 6.5, "a" * 64),
            ("two", "bace_comrecgc_generation", 6.5, "b" * 64),
        ],
        aggregate=13.0,
    )
    with pytest.raises(GPUColocationGateError, match="aggregate throughput differs"):
        build_gpu_colocation_gate(
            single_profile_paths=[first, second],
            colocated_profile_path=colocated,
            output_dir=tmp_path / "inflated-gate",
        )

    # Restore an honest first profile, then change only the paired scientific
    # config identity while leaving its result digest unchanged.
    first = _profile(
        tmp_path,
        name="first-config",
        mode="single_task",
        rows=[("one", "bace_gcfexplainer_vrrw", 10.0, "a" * 64)],
        aggregate=10.0,
    )
    second = _profile(
        tmp_path,
        name="second-config",
        mode="single_task",
        rows=[("two", "bace_comrecgc_generation", 10.0, "b" * 64)],
        aggregate=10.0,
    )
    drifted = _profile(
        tmp_path,
        name="pair-config",
        mode="shared_lowmem_pair",
        rows=[
            ("one", "bace_gcfexplainer_vrrw", 6.5, "a" * 64),
            ("two", "bace_comrecgc_generation", 6.5, "b" * 64),
        ],
        aggregate=13.0,
    )
    payload = json.loads(drifted.read_text(encoding="utf-8"))
    payload["tasks"][0]["scientific_config_sha256"] = "f" * 64
    _write(drifted, payload)
    with pytest.raises(GPUColocationGateError, match="changed canonical"):
        build_gpu_colocation_gate(
            single_profile_paths=[first, second],
            colocated_profile_path=drifted,
            output_dir=tmp_path / "config-gate",
        )
    slow = _profile(
        tmp_path,
        name="slow",
        mode="shared_lowmem_pair",
        rows=[
            ("one", "bace_gcfexplainer_vrrw", 5.5, "a" * 64),
            ("two", "bace_comrecgc_generation", 5.5, "b" * 64),
        ],
        aggregate=11.0,
    )
    with pytest.raises(GPUColocationGateError, match="below 20%"):
        build_gpu_colocation_gate(
            single_profile_paths=[first, second],
            colocated_profile_path=slow,
            output_dir=tmp_path / "slow-gate",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("oom_count", 1, "OOM"),
        ("cpu_sustained_saturation", True, "CPU saturation"),
        ("disk_io_instability", True, "disk I/O"),
        ("mps_enabled", True, "MPS"),
    ),
)
def test_profile_health_contract_is_fail_closed(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    bad = _profile(
        tmp_path,
        name=f"bad-{field}",
        mode="single_task",
        rows=[("one", "bace_gcfexplainer_vrrw", 10.0, "a" * 64)],
        aggregate=10.0,
        **{field: value},
    )
    good = _profile(
        tmp_path,
        name=f"good-{field}",
        mode="single_task",
        rows=[("two", "bace_comrecgc_generation", 10.0, "b" * 64)],
        aggregate=10.0,
    )
    colocated = _profile(
        tmp_path,
        name=f"colocated-{field}",
        mode="shared_lowmem_pair",
        rows=[
            ("one", "bace_gcfexplainer_vrrw", 6.5, "a" * 64),
            ("two", "bace_comrecgc_generation", 6.5, "b" * 64),
        ],
        aggregate=13.0,
    )
    with pytest.raises(GPUColocationGateError, match=message):
        build_gpu_colocation_gate(
            single_profile_paths=[bad, good],
            colocated_profile_path=colocated,
            output_dir=tmp_path / f"gate-{field}",
        )


def test_launch_validator_rejects_gate_or_reservation_drift(tmp_path: Path) -> None:
    path, digest = build_test_gate(tmp_path)
    with pytest.raises(GPUColocationGateError, match="SHA256 mismatch"):
        validate_gpu_colocation_gate(
            path,
            expected_sha256="0" * 64,
            workload_class="bace_gcfexplainer_vrrw",
            memory_reservation_mb=9000,
        )
    with pytest.raises(GPUColocationGateError, match="below the measured"):
        validate_gpu_colocation_gate(
            path,
            expected_sha256=digest,
            workload_class="bace_gcfexplainer_vrrw",
            memory_reservation_mb=7000,
        )
    gate = json.loads(path.read_text(encoding="utf-8"))
    source = Path(gate["source_profiles"][0]["path"])
    source.write_text(source.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(GPUColocationGateError, match="source profile SHA256 changed"):
        validate_gpu_colocation_gate(
            path,
            expected_sha256=digest,
            workload_class="bace_gcfexplainer_vrrw",
            memory_reservation_mb=9000,
        )
