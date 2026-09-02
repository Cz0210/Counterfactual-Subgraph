from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.ablations.llm.build_brics_train_vocab_v2 import build
from src.ablations.llm.contracts import (
    LLMAblationContractError,
    canonical_json_sha256,
)
from src.eval.bace_frozen_gnn_contracts import stable_sha256


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path, *, split_override: str | None = None):
    pytest.importorskip("rdkit")
    root = tmp_path / "inputs"
    root.mkdir()
    train = root / "train.csv"
    fieldnames = [
        "molecule_id",
        "parent_id",
        "smiles",
        "label",
        "split",
        "oracle_score",
    ]
    rows = []
    for index in range(387):
        parent_id = f"BACE_{index:04d}"
        rows.append(
            {
                "molecule_id": parent_id,
                "parent_id": parent_id,
                "smiles": "CCOC(=O)NCC" if index % 2 == 0 else "CCN(CC)CCOC(=O)c1ccccc1",
                "label": "1" if index < 3 else "0",
                "split": split_override if index == 0 and split_override else "train",
                "oracle_score": str(1000 - index),
            }
        )
    with train.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    train_sha = _sha(train)
    parent_ids = ["BACE_0000", "BACE_0001"]
    cohort = root / "train_parent_ids.frozen.json"
    cohort.write_text(
        json.dumps(
            {
                "schema_version": "bace_frozen_parent_ids_v1",
                "status": "FROZEN",
                "dataset": "bace",
                "split": "train",
                "source_label": 1,
                "num_classes": 2,
                "parent_ids": parent_ids,
                "parent_count": len(parent_ids),
                "parent_ids_sha256": stable_sha256(parent_ids),
                "shard_count": 4,
                "shard_rule": "sorted(parent_id)_position_mod_4",
                "split_identity": {
                    "path": str(train),
                    "sha256": train_sha,
                    "size": train.stat().st_size,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    cohort_sha = _sha(cohort)
    reference = root / "bace_ours_llm_reference_v2.json"
    reference_payload = {
        "schema_version": "bace_ours_llm_reference_v2",
        "status": "PASS",
        "dataset": "bace",
        "method": "ours",
        "source_label": 1,
        "scientific_values_inferred": False,
        "candidate_generation": {
            "train_only": True,
            "test_loaded": False,
            "parent_count": len(parent_ids),
            "attempts_per_parent": 20,
            "parent_manifest": {
                "path": str(cohort),
                "sha256": cohort_sha,
                "size": cohort.stat().st_size,
            },
            "base_regime": {
                "batch_size": 1,
                "num_return_sequences": 4,
                "oracle_batch_size": 256,
                "seed": 7,
                "stage": "B8_POOL_BASE",
                "temperature": 0.3,
                "top_p": 0.9,
                "max_new_tokens": 96,
            },
            "high_temperature_regime": {
                "batch_size": 1,
                "num_return_sequences": 4,
                "oracle_batch_size": 256,
                "seed": 13,
                "stage": "B9_POOL_HIGHTEMP",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 96,
            },
        },
        "frozen_downstream": {
            "dataset_split_paths": {
                "train": str(train),
                "calibration": "/frozen/BACE/calibration.csv",
                "test": "/frozen/BACE/test.csv",
            },
            "dataset_split_hashes": {
                "train": train_sha,
                "calibration": "a" * 64,
                "test": "b" * 64,
            },
        },
        "stage_variants": {
            "A0_BRICS_FIXED": {
                "model": None,
                "status": "CPU_FRAMEWORK_AVAILABLE",
            }
        },
    }
    reference_payload["reference_contract_sha256"] = canonical_json_sha256(
        reference_payload
    )
    reference.write_text(
        json.dumps(reference_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return train, train_sha, cohort, cohort_sha, reference, _sha(reference)


def _args(tmp_path: Path, **overrides):
    train, train_sha, cohort, cohort_sha, reference, reference_sha = _inputs(
        tmp_path, split_override=overrides.pop("split_override", None)
    )
    values = {
        "config": "configs/hpc.yaml",
        "train_csv": str(train),
        "train_csv_sha256": train_sha,
        "expected_train_rows": 387,
        "proposal_cohort_manifest": str(cohort),
        "proposal_cohort_sha256": cohort_sha,
        "expected_proposal_parents": 2,
        "reference_contract": str(reference),
        "reference_contract_sha256": reference_sha,
        "attempts_per_parent": 20,
        "workers": 2,
        "output_root": str(tmp_path / "output"),
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_full_train_vocab_and_source_cohort_proposals_are_separate(tmp_path: Path) -> None:
    result = build(_args(tmp_path))
    output = Path(result["output_root"])
    vocab = _jsonl(output / "brics_vocab.jsonl")
    pool = _jsonl(output / "brics_proposal_pool.jsonl")
    attempts = _jsonl(output / "brics_proposal_attempts.jsonl")
    manifest = json.loads((output / "brics_vocab_manifest.json").read_text())
    receipt = json.loads(
        (output / "brics_proposal_shortfall_receipt.json").read_text()
    )

    assert manifest["observed_train_rows"] == 387
    assert manifest["vocabulary_size"] == len(vocab) > 0
    assert manifest["oracle_fields_read"] == []
    assert "oracle_score" not in manifest["input_columns_read"]
    assert manifest["calibration_loaded"] is False
    assert manifest["test_loaded"] is False
    assert manifest["requested_workers"] == 2
    assert manifest["effective_workers"] == 1

    assert len(attempts) == 2 * 20
    assert len(pool) + receipt["proposal_shortfall"] == len(attempts)
    assert receipt["candidate_duplication_used"] is False
    assert receipt["shortfall_is_not_backfilled"] is True
    assert receipt["oracle_ranking_used"] is False
    assert all(row["parent_id"] in {"BACE_0000", "BACE_0001"} for row in pool)
    assert all(row["proposal_shortfall"] is False for row in pool)
    assert len({(row["parent_id"], row["fragment_smiles"]) for row in pool}) == len(pool)
    assert (output / "brics_vocab_sha256s.txt").is_file()
    assert result["gpu_used"] is False


def test_cli_rejects_source_cohort_as_full_train_claim(tmp_path: Path) -> None:
    with pytest.raises(LLMAblationContractError, match="exceed the 386-parent"):
        build(_args(tmp_path, expected_train_rows=386))


def test_cli_rejects_non_train_rows_before_output(tmp_path: Path) -> None:
    args = _args(tmp_path, split_override="test")
    with pytest.raises(LLMAblationContractError, match="split='test'"):
        build(args)
    assert not Path(args.output_root).exists()


def test_cli_caps_cpu_workers_at_two(tmp_path: Path) -> None:
    with pytest.raises(LLMAblationContractError, match="workers"):
        build(_args(tmp_path, workers=3))


def test_cli_rejects_cohort_hash_drift(tmp_path: Path) -> None:
    with pytest.raises(LLMAblationContractError, match="cohort manifest SHA256"):
        build(_args(tmp_path, proposal_cohort_sha256="f" * 64))


def test_cli_rejects_reference_contract_hash_drift(tmp_path: Path) -> None:
    with pytest.raises(LLMAblationContractError, match="reference contract file SHA"):
        build(_args(tmp_path, reference_contract_sha256="f" * 64))


def test_cli_rejects_self_consistent_reference_budget_drift(tmp_path: Path) -> None:
    args = _args(tmp_path)
    reference = Path(args.reference_contract)
    payload = json.loads(reference.read_text(encoding="utf-8"))
    payload["candidate_generation"]["base_regime"]["seed"] = 8
    payload.pop("reference_contract_sha256")
    payload["reference_contract_sha256"] = canonical_json_sha256(payload)
    reference.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.reference_contract_sha256 = _sha(reference)
    with pytest.raises(LLMAblationContractError, match="base_regime changed"):
        build(args)
