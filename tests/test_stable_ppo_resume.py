from __future__ import annotations

import json
from pathlib import Path
import random

import pytest
import torch
from safetensors.torch import save_file

from src.train.stable_ppo_resume import (
    MANIFEST_NAME,
    adopt_stable_ppo_checkpoint_prefix,
    find_latest_stable_ppo_resume_checkpoint,
    load_stable_ppo_resume_checkpoint,
    read_stable_ppo_resume_manifest,
    restore_stable_ppo_training_state,
    save_stable_ppo_resume_checkpoint,
)


class _TinyValueModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.v_head = torch.nn.Linear(2, 1)


def _resume_contract(*, max_steps: int = 3) -> dict:
    return {
        "schema_version": "bace_b7_stable_ppo_resume_contract_v1",
        "stage": "B7_PPO_FULL",
        "dataset": "bace",
        "max_steps": max_steps,
        "actual_batch_size": 2,
        "git_commit": "a" * 40,
    }


def _write_checkpoint(
    root: Path,
    *,
    step: int,
    optimizer: torch.optim.Optimizer,
    value_model: _TinyValueModel,
    contract: dict,
) -> None:
    root.mkdir(parents=True)
    (root / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 2}) + "\n",
        encoding="utf-8",
    )
    save_file(
        {"lora_weight": torch.tensor([float(step)])},
        root / "adapter_model.safetensors",
    )
    torch.save(value_model.v_head.state_dict(), root / "decoded_chem_value_head.pt")
    save_stable_ppo_resume_checkpoint(
        checkpoint_dir=root,
        torch_module=torch,
        optimizer=optimizer,
        completed_steps=step,
        current_kl_penalty=0.2,
        validation_state={
            "best_val_score": None,
            "best_step": None,
            "stale_eval_count": 0,
        },
        last_validation_step=None,
        candidate_pool_rows=[
            {
                "step_index": index,
                "reward_total": float(index),
                "parent_id": f"p{index}",
            }
            for index in range(1, step + 1)
        ],
        observer_state={
            "schema_version": "bace_ppo_observer_v1",
            "updates": [
                {"step_index": index, "metrics": {}, "batch_ids": []}
                for index in range(1, step + 1)
            ],
            "checkpoints": [],
            "finish": None,
        },
        resume_contract=contract,
    )


def test_resume_checkpoint_restores_optimizer_value_and_rng_fail_closed(
    tmp_path: Path,
) -> None:
    policy = torch.nn.Linear(2, 2)
    value = _TinyValueModel()
    optimizer = torch.optim.AdamW(
        [*policy.parameters(), *value.v_head.parameters()], lr=0.01
    )
    loss = policy(torch.ones(1, 2)).sum() + value.v_head(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    contract = _resume_contract(max_steps=3)
    root = tmp_path / "checkpoint-1"
    random.seed(17)
    torch.manual_seed(19)
    _write_checkpoint(
        root,
        step=1,
        optimizer=optimizer,
        value_model=value,
        contract=contract,
    )
    expected_python_random = random.random()
    expected_torch_random = torch.rand(1)

    restored_policy = torch.nn.Linear(2, 2)
    restored_value = _TinyValueModel()
    restored_optimizer = torch.optim.AdamW(
        [*restored_policy.parameters(), *restored_value.v_head.parameters()], lr=0.01
    )
    bundle = load_stable_ppo_resume_checkpoint(
        checkpoint_dir=root,
        torch_module=torch,
        expected_contract=contract,
        map_location="cpu",
    )
    restore_stable_ppo_training_state(
        bundle=bundle,
        optimizer=restored_optimizer,
        value_model=restored_value,
        torch_module=torch,
    )
    assert bundle.completed_steps == 1
    assert len(restored_optimizer.state) == len(optimizer.state)
    for original, restored in zip(
        value.v_head.parameters(), restored_value.v_head.parameters(), strict=True
    ):
        assert torch.equal(original, restored)
    assert random.random() == expected_python_random
    assert torch.equal(torch.rand(1), expected_torch_random)

    with pytest.raises(ValueError, match="contract differs"):
        load_stable_ppo_resume_checkpoint(
            checkpoint_dir=root,
            torch_module=torch,
            expected_contract={**contract, "actual_batch_size": 1},
            map_location="cpu",
        )
    with (root / "candidate_pool.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"tampered":true}\n')
    with pytest.raises(ValueError, match="candidate_pool"):
        read_stable_ppo_resume_manifest(root)


def test_resume_prefix_adoption_rebinds_only_canonical_manifest_path(
    tmp_path: Path,
) -> None:
    policy = torch.nn.Linear(2, 2)
    value = _TinyValueModel()
    optimizer = torch.optim.AdamW(
        [*policy.parameters(), *value.v_head.parameters()], lr=0.01
    )
    contract = _resume_contract(max_steps=3)
    source = tmp_path / "attempt-0"
    _write_checkpoint(
        source / "checkpoint-1",
        step=1,
        optimizer=optimizer,
        value_model=value,
        contract=contract,
    )
    _write_checkpoint(
        source / "checkpoint-2",
        step=2,
        optimizer=optimizer,
        value_model=value,
        contract=contract,
    )
    destination = tmp_path / "attempt-1"
    destination.mkdir()
    adopted = adopt_stable_ppo_checkpoint_prefix(
        resume_checkpoint=source / "checkpoint-2",
        output_dir=destination,
        checkpoint_steps=(1, 2, 3),
    )
    assert [row["step"] for row in adopted] == [1, 2]
    copied = read_stable_ppo_resume_manifest(destination / "checkpoint-2")
    assert copied["checkpoint_dir"] == str(
        (destination / "checkpoint-2").resolve()
    )
    assert copied["resume_contract"] == contract
    assert adopted[-1]["source_manifest_sha256"] != adopted[-1][
        "copied_manifest_sha256"
    ]
    assert find_latest_stable_ppo_resume_checkpoint(source) == (
        source / "checkpoint-2"
    ).resolve()

    # A complete checkpoint is a finalization artifact, not a legal resume
    # point.  The finder must fall back to the latest incomplete checkpoint.
    complete_contract = _resume_contract(max_steps=2)
    complete = tmp_path / "complete-attempt"
    _write_checkpoint(
        complete / "checkpoint-1",
        step=1,
        optimizer=optimizer,
        value_model=value,
        contract=complete_contract,
    )
    _write_checkpoint(
        complete / "checkpoint-2",
        step=2,
        optimizer=optimizer,
        value_model=value,
        contract=complete_contract,
    )
    assert find_latest_stable_ppo_resume_checkpoint(complete) == (
        complete / "checkpoint-1"
    ).resolve()

    # The manifest itself is the last-published atomic readiness boundary.
    (complete / "checkpoint-1" / MANIFEST_NAME).unlink()
    assert find_latest_stable_ppo_resume_checkpoint(complete) is None

    mixed = tmp_path / "mixed-attempt"
    _write_checkpoint(
        mixed / "checkpoint-1",
        step=1,
        optimizer=optimizer,
        value_model=value,
        contract=contract,
    )
    _write_checkpoint(
        mixed / "checkpoint-2",
        step=2,
        optimizer=optimizer,
        value_model=value,
        contract={**contract, "git_commit": "b" * 40},
    )
    mixed_destination = tmp_path / "mixed-destination"
    mixed_destination.mkdir()
    with pytest.raises(ValueError, match="different run contract"):
        adopt_stable_ppo_checkpoint_prefix(
            resume_checkpoint=mixed / "checkpoint-2",
            output_dir=mixed_destination,
            checkpoint_steps=(1, 2, 3),
        )
