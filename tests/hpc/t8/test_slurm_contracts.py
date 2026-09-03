from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
HPC = ROOT / "scripts" / "hpc" / "t8"


@pytest.mark.parametrize("name", ["slurm_canary.sh", "slurm_array.sh", "slurm_merge.sh"])
def test_hpc_jobs_are_cpu_only_and_hash_bound(name: str) -> None:
    script = HPC / name
    text = script.read_text(encoding="utf-8")
    assert "#SBATCH --partition=" in text
    assert "#SBATCH --cpus-per-task=" in text
    assert "#SBATCH --mem=" in text
    assert "#SBATCH --time=" in text
    assert "#SBATCH --gres" not in text
    assert 'export CUDA_VISIBLE_DEVICES=""' in text
    assert "SLURM_TMPDIR" in text
    assert "T8_EXPECTED_COMMIT" in text
    assert "T8_EXPECTED_INPUT_MANIFEST_SHA256" in text
    assert "T8_EXPECTED_CONFIG_SHA256" in text
    assert "T8_EXPECTED_HPC_CONFIG_SHA256" in text
    assert '$1 == "intel"' in text
    assert "HPC_FALLBACK_PARTITION:-intel" in text
    assert "#SBATCH --partition=intel" in text
    assert "--config configs/hpc.yaml" in text
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_only_boundary_resumable_array_requests_requeue() -> None:
    array = (HPC / "slurm_array.sh").read_text(encoding="utf-8")
    assert "#SBATCH --requeue" in array
    assert "#SBATCH --signal=B:USR1@120" in array
    for name in ("slurm_canary.sh", "slurm_merge.sh"):
        text = (HPC / name).read_text(encoding="utf-8")
        assert "#SBATCH --requeue" not in text
        assert "scontrol requeue" not in text


def test_array_has_bounded_dynamic_concurrency_and_boundary_resume() -> None:
    text = (HPC / "slurm_array.sh").read_text(encoding="utf-8")
    assert "T8_ARRAY_CONCURRENCY" in text
    assert '--array="0-${last_index}%${concurrency}"' in text
    assert "T8_EXPECTED_PARTITION_MANIFEST_SHA256" in text
    assert "T8_CANARY_PARITY_RECEIPT" in text
    assert "T8_EXPECTED_CANARY_PARITY_SHA256" in text
    assert 'c["status"]=="PASS"' in text
    assert 'c["scientific_input_sha256"]==f["scientific_input_sha256"]' in text
    assert "SLURM_ARRAY_TASK_ID" in text
    assert "shard-%03d" in text
    assert "run_exact_mining_shard.py" in text
    assert "scontrol requeue" in text
    assert '--scratch-root "$job_tmp/active"' in text


def test_canary_is_one_hour_64g_and_requires_bounded_real_prefix() -> None:
    text = (HPC / "slurm_canary.sh").read_text(encoding="utf-8")
    assert "#SBATCH --partition=intel" in text
    assert "#SBATCH --mem=64G" in text
    assert "#SBATCH --time=01:00:00" in text
    assert "96G" not in text
    assert "06:00:00" not in text
    assert "T8_CANARY_PREFIX_UNIT_ID" in text
    assert "T8_CANARY_SHARD_COUNT must be 1 or 2" in text
    assert 'canary_prefix_unit_id="${T8_CANARY_PREFIX_UNIT_ID:-}"' in text
    assert "root0-prefix-catalog.json" in text
    assert 'x["root_index"] == 0' in text
    assert 'x["partition_type"] == "PREFIX_SUBTREE"' in text
    assert 'key=lambda x:(int(x["support_hint"]),x["partition_id"])' in text
    assert '--included-unit-id "$canary_prefix_unit_id"' in text
    assert "canary_prefix_selection.json" in text
    assert "SUPPORT_HINT_ASC_PARTITION_ID_ASC" in text
    assert "selection_receipt_sha256" in text
    assert '--included-root-indices "22"' in text
    assert '--canary-root-indices "0,22"' in text
    assert '--input-manifest "$T8_INPUT_MANIFEST"' in text
    assert '--expected-commit "$T8_EXPECTED_COMMIT"' in text
    assert '--scratch-root "$job_tmp/reference"' in text
    assert '--scratch-root "$job_tmp/$shard_name"' in text
    assert '--scratch-root "$job_tmp/merge"' in text


def test_merge_is_afterok_and_never_has_matrix_arguments() -> None:
    text = (HPC / "slurm_merge.sh").read_text(encoding="utf-8")
    assert '--dependency="afterok:${T8_FULL_ARRAY_JOB_ID}"' in text
    assert "merge_exact_shards.py" in text
    assert "build_result_bundle.py" in text
    assert "T8_CANARY_PARITY_RECEIPT" in text
    assert "T8_EXPECTED_CANARY_PARITY_SHA256" in text
    assert '--scratch-root "$job_tmp/merge"' in text
    assert "T8_ENVIRONMENT_MANIFEST" in text
    assert "T8_SLURM_INVENTORY" in text
    assert "T8_RESOURCE_METRICS" in text
    assert '--environment-manifest "$T8_ENVIRONMENT_MANIFEST"' in text
    assert '--slurm-inventory "$T8_SLURM_INVENTORY"' in text
    assert '--resource-metrics "$T8_RESOURCE_METRICS"' in text
    assert "matrix-authority" not in text
    assert "matrix_root" not in text.lower()


@pytest.mark.parametrize("name", ["slurm_canary.sh", "slurm_array.sh", "slurm_merge.sh"])
def test_paired_slurm_wrapper_documents_cpu_override(name: str) -> None:
    script = ROOT / "scripts" / "slurm" / name
    text = script.read_text(encoding="utf-8")
    assert "CPU-only" in text
    assert "#SBATCH --partition=intel" in text
    assert "#SBATCH --gres" not in text
    assert f"scripts/hpc/t8/{name}" in text
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_bundle_has_paired_cpu_wrapper() -> None:
    text = (ROOT / "scripts" / "slurm" / "build_input_bundle.sh").read_text(encoding="utf-8")
    assert "CPU-only" in text
    assert "#SBATCH --gres" not in text
    assert "build_input_bundle.py --config configs/hpc.yaml" in text
    subprocess.run(["bash", "-n", str(ROOT / "scripts" / "slurm" / "build_input_bundle.sh")], check=True)


@pytest.mark.parametrize(
    ("script_name", "entrypoint"),
    [
        ("build_partition_manifest.sh", "build_partition_manifest.py"),
        ("run_exact_reference_canary.sh", "run_exact_reference_canary.py"),
        ("run_exact_mining_shard.sh", "run_exact_mining_shard.py"),
        ("verify_exact_parity.sh", "verify_exact_parity.py"),
        ("merge_exact_shards.sh", "merge_exact_shards.py"),
        ("build_result_bundle.sh", "build_result_bundle.py"),
    ],
)
def test_every_hpc_python_entrypoint_has_a_cpu_slurm_wrapper(
    script_name: str, entrypoint: str
) -> None:
    script = ROOT / "scripts" / "slurm" / script_name
    text = script.read_text(encoding="utf-8")
    assert "CPU-only override" in text
    assert "#SBATCH --partition=intel" in text
    assert "#SBATCH --gres" not in text
    assert f"scripts/hpc/t8/{entrypoint} --config configs/hpc.yaml" in text
    subprocess.run(["bash", "-n", str(script)], check=True)
