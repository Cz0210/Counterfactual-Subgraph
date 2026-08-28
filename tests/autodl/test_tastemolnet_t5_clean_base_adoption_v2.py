from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct

import pytest

from src.train import tastemolnet_t5_clean_base_adoption_v2 as t5
from src.utils.managed_execution_v2 import (
    create_managed_attempt,
    create_worker_staging,
    load_verified_gate,
    write_worker_exit,
    write_worker_raw_evidence,
)
from src.utils.terminal_publisher_v2 import seal_worker_staging


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_safetensors(path: Path, tensor_name: str) -> int:
    tensor_bytes = b"\x00\x00\x00\x00"
    header = json.dumps(
        {
            tensor_name: {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, len(tensor_bytes)],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + tensor_bytes)
    return len(tensor_bytes)


def _make_source(parent: Path) -> Path:
    root = parent / "ChemLLM-7B-Chat"
    root.mkdir(mode=0o700)
    _write_json(
        root / "config.json",
        {
            "architectures": ["InternLM2ForCausalLM"],
            "hidden_size": 8,
            "model_type": "internlm2",
            "vocab_size": 32,
        },
    )
    _write_json(
        root / "tokenizer_config.json",
        {"tokenizer_class": "InternLM2Tokenizer"},
    )
    (root / "tokenizer.model").write_bytes(b"tiny-tokenizer-model\n")
    shard_a = "model-00001-of-00002.safetensors"
    shard_b = "model-00002-of-00002.safetensors"
    size_a = _write_safetensors(root / shard_a, "model.embed_tokens.weight")
    size_b = _write_safetensors(root / shard_b, "lm_head.weight")
    _write_json(
        root / "model.safetensors.index.json",
        {
            "metadata": {"total_size": size_a + size_b},
            "weight_map": {
                "lm_head.weight": shard_b,
                "model.embed_tokens.weight": shard_a,
            },
        },
    )
    return root


def test_inspect_closes_full_hf_inventory(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    evidence = t5.inspect_clean_chemllm_base(source)
    assert evidence["semantic_state"] == "ADOPTED_CLEAN_GENERIC_BASE"
    assert evidence["source_model_file_count"] == 6
    assert set(evidence["source_model_files"]) == {
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.model",
        "tokenizer_config.json",
    }
    closure = evidence["hf_closure"]
    assert closure["index_closes_exact_shards"] is True
    assert closure["index_matches_all_shard_headers"] is True
    assert closure["safetensors_shard_count"] == 2
    assert closure["indexed_tensor_count"] == 2
    assert evidence["optimizer_steps"] == 0
    assert evidence["taste_splits_loaded"] == []
    assert evidence["matrix_method_cell"] is False


def test_inspect_rejects_peft_or_dataset_payload(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    (source / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(t5.TasteT5CleanBaseAdoptionError, match="adapter"):
        t5.inspect_clean_chemllm_base(source)

    (source / "adapter_config.json").unlink()
    (source / "train.csv").write_text("smiles,label\nCC,1\n", encoding="utf-8")
    with pytest.raises(t5.TasteT5CleanBaseAdoptionError, match="dataset payload"):
        t5.inspect_clean_chemllm_base(source)


def test_inspect_rejects_unclosed_safetensors_index(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"].pop("lm_head.weight")
    _write_json(index_path, index)
    with pytest.raises(
        t5.TasteT5CleanBaseAdoptionError,
        match="exact shard inventory|tensor map",
    ):
        t5.inspect_clean_chemllm_base(source)


def test_managed_worker_candidate_and_independent_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTO_TERMINATE_UNCONTROLLED_CHILDREN", "0")
    source = _make_source(tmp_path)
    evidence = t5.inspect_clean_chemllm_base(source)
    source_digest = evidence["source_model_inventory_sha256"]
    config_digest = "b" * 64
    commit = "c" * 40
    controller = "tastemolnet-main-v2-test"
    stage_root = tmp_path / "control" / "T5_CLEAN_BASE_ADOPTION"
    stage_root.mkdir(parents=True, mode=0o700)
    final_parent = tmp_path / "outputs" / "clean-policy"
    final_parent.mkdir(parents=True, mode=0o700)
    final_path = final_parent / "adopted-clean-base-test"

    with create_managed_attempt(
        stage_root=stage_root,
        controller_id=controller,
        task_id=t5.TASK_ID,
        git_commit=commit,
        config_hash=config_digest,
        input_hashes={"source_model_inventory": source_digest},
        hostname="test-host",
        boot_id="test-boot",
    ) as attempt:
        with create_worker_staging(attempt) as staging:
            result = t5.build_clean_base_candidate(
                source_model=source,
                artifact_root=staging.artifact_root,
                attempt_id=attempt.attempt_id,
                generation_token=staging.generation_token,
                config_sha256=config_digest,
                expected_source_inventory_sha256=source_digest,
            )
            assert result["status"] == "SEALED_CANDIDATE"
            assert not (staging.path / "PASS").exists()
            command = [
                "python",
                "scripts/autodl/tastemolnet_t5_clean_base_worker_v2.py",
                "build",
                "--config",
                "configs/hpc.yaml",
                "--source-model",
                str(source),
                "--expected-source-inventory-sha256",
                source_digest,
            ]
            raw = write_worker_raw_evidence(
                staging,
                {
                    "attempt_manifest": dict(attempt.manifest.payload),
                    "process_lineage": {
                        "controller_id": controller,
                        "attempt_id": attempt.attempt_id,
                    },
                    "scientific_command": command,
                    "artifact_root": str(staging.artifact_root),
                },
            )
            raw.close()
            exited = write_worker_exit(
                staging,
                {
                    "exit_code": 0,
                    "process_audit": {
                        "state": "EXITED",
                        "controller_id": controller,
                        "attempt_id": attempt.attempt_id,
                    },
                    "worker_closed_artifact_writers": True,
                },
            )
            exited.close()
            sealed = seal_worker_staging(staging)
            assert not (staging.path / "gate.json").exists()
            publication, verification = t5.verify_and_publish_clean_base_adoption(
                sealed_path=sealed.staging_path,
                final_path=final_path,
                source_model=source,
                expected_attempt_id=sealed.attempt_id,
                expected_generation_token=sealed.generation_token,
                expected_controller_id=controller,
                expected_git_commit=commit,
                expected_config_sha256=config_digest,
                expected_source_inventory_sha256=source_digest,
            )
    assert publication.final_path == final_path
    assert verification["semantic_state"] == t5.SEMANTIC_STATE
    assert verification["optimizer_steps"] == 0
    assert verification["taste_splits_loaded"] == []
    assert verification["matrix_method_cell"] is False
    assert (final_path / "PASS").read_bytes() == b"[MANAGED_EXECUTION_V2_PASS]\n"
    gate = load_verified_gate(final_path)
    assert gate["status"] == "PASS"
    verifier_document = json.loads(
        (final_path / "verification.json").read_text(encoding="utf-8")
    )
    assert verifier_document["verification"]["marker"] == t5.PASS_MARKER
    assert not any(
        path.suffix == ".safetensors" for path in final_path.rglob("*")
    )


def test_source_mutation_after_worker_blocks_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTO_TERMINATE_UNCONTROLLED_CHILDREN", "0")
    source = _make_source(tmp_path)
    evidence = t5.inspect_clean_chemllm_base(source)
    source_digest = evidence["source_model_inventory_sha256"]
    (source / "tokenizer.model").write_bytes(b"changed-tokenizer\n")
    with pytest.raises(
        t5.TasteT5CleanBaseAdoptionError,
        match="launcher pin",
    ):
        with t5.HeldCleanChemLLMSource.open(source) as held:
            if held.evidence["source_model_inventory_sha256"] != source_digest:
                raise t5.TasteT5CleanBaseAdoptionError(
                    "independent source inventory differs from launcher pin"
                )


def test_cli_wrapper_and_slurm_contracts() -> None:
    repository = Path(__file__).resolve().parents[2]
    worker = (
        repository / "scripts/autodl/tastemolnet_t5_clean_base_worker_v2.py"
    ).read_text(encoding="utf-8")
    verifier = (
        repository / "scripts/autodl/tastemolnet_t5_clean_base_verifier_v2.py"
    ).read_text(encoding="utf-8")
    wrapper = (
        repository / "scripts/autodl/run_tastemolnet_t5_clean_base_adoption_v2.sh"
    ).read_text(encoding="utf-8")
    assert t5.PASS_MARKER not in worker
    assert "print(PASS_MARKER" in verifier
    assert "AUTO_TERMINATE_UNCONTROLLED_CHILDREN" in wrapper
    assert "managed_worker_v2.py" in wrapper
    assert "T5_CLEAN_BASE_ADOPTION" in wrapper
    for name in (
        "tastemolnet_t5_clean_base_worker_v2.sh",
        "tastemolnet_t5_clean_base_verifier_v2.sh",
    ):
        slurm = (repository / "scripts/slurm" / name).read_text(encoding="utf-8")
        assert "#SBATCH --partition=A800" in slurm
        assert "#SBATCH --gres=gpu:a800:1" in slurm
        assert "#SBATCH --output=logs/%j.out" in slurm
        assert "#SBATCH --error=logs/%j.err" in slurm
        assert "source ~/.bashrc" in slurm
        assert "conda activate smiles_pip118" in slurm
        assert "cd /share/home/u20526/czx/counterfactual-subgraph" in slurm
        assert "export PYTHONPATH=$PWD" in slurm
        assert "--config configs/hpc.yaml" in slurm
        assert "REFUSING_HPC_EXECUTION" in slurm
