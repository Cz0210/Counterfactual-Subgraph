#!/usr/bin/env python3
"""Run Fresh Mutagenicity PPO through the shared stable PPO implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_mutagenicity_ppo_stable import main as stable_main  # noqa: E402
from src.train.mutagenicity_fresh_sft import write_json_atomic  # noqa: E402


REWARD_CLI_KEYS = {
    "reward_profile": "--reward-profile",
    "strict_flip_bonus": "--strict-flip-bonus",
    "non_flip_penalty": "--non-flip-penalty",
    "cf_drop_weight": "--cf-drop-weight",
    "validity_weight": "--validity-weight",
    "substructure_weight": "--substructure-weight",
    "size_weight": "--size-weight",
    "projection_penalty": "--projection-penalty",
    "non_flip_aux_reward_cap": "--non-flip-aux-reward-cap",
    "strict_flip_reward_margin": "--strict-flip-reward-margin",
    "reward_clip_min": "--reward-clip-min",
    "reward_clip_max": "--reward-clip-max",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "All remaining options are forwarded to "
            "scripts/train_mutagenicity_ppo_stable.py."
        ),
    )
    parser.add_argument("--reward-config-json", type=Path, required=True)
    parser.add_argument("--fresh-initialization-audit", type=Path, default=None)
    return parser


def _option_value(argv: Sequence[str], option: str) -> str | None:
    for index, token in enumerate(argv):
        if token == option and index + 1 < len(argv):
            return str(argv[index + 1])
        prefix = f"{option}="
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def find_fresh_initialization_audit(checkpoint: Path) -> Path:
    for root in (checkpoint, *checkpoint.parents[:6]):
        candidate = root / "fresh_initialization_audit.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Fresh PPO policy checkpoint has no nearby fresh_initialization_audit.json: "
        f"{checkpoint}"
    )


def validate_fresh_policy_provenance(
    checkpoint: str | Path,
    *,
    explicit_audit: str | Path | None = None,
) -> dict[str, Any]:
    resolved_checkpoint = _resolve(checkpoint)
    if not resolved_checkpoint.is_dir():
        raise FileNotFoundError(f"Fresh SFT checkpoint does not exist: {resolved_checkpoint}")
    audit_path = (
        _resolve(explicit_audit)
        if explicit_audit is not None
        else find_fresh_initialization_audit(resolved_checkpoint)
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if not bool(audit.get("adapter_initialized_from_scratch")):
        errors.append("adapter_initialized_from_scratch is false")
    if audit.get("source_adapter_checkpoint") is not None:
        errors.append("source_adapter_checkpoint is not null")
    if bool(audit.get("aids_adapter_weights_loaded")):
        errors.append("aids_adapter_weights_loaded is true")
    if audit.get("adapter_names") != ["default"]:
        errors.append(f"adapter_names={audit.get('adapter_names')!r}")
    if audit.get("active_adapters") != ["default"]:
        errors.append(f"active_adapters={audit.get('active_adapters')!r}")
    if int(audit.get("base_parameter_trainable_count", -1)) != 0:
        errors.append("base_parameter_trainable_count is not zero")
    if not bool(audit.get("initialization_audit_passed")):
        errors.append("initialization_audit_passed is false")
    if errors:
        raise ValueError(f"Fresh PPO policy provenance failed: {errors}")
    return {
        "policy_adapter_checkpoint": str(resolved_checkpoint),
        "fresh_initialization_audit": str(audit_path),
        "adapter_initialized_from_scratch": True,
        "source_adapter_checkpoint": None,
        "aids_adapter_weights_loaded": False,
        "fresh_policy_provenance_passed": True,
    }


def load_reward_config(path: str | Path) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Audited reward config does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    missing = sorted(set(REWARD_CLI_KEYS) - set(payload))
    if missing:
        raise ValueError(f"Reward config is missing required fields: {missing}")
    if payload.get("reward_profile") != "mutagenicity_flip_dominant":
        raise ValueError("Fresh PPO requires reward_profile=mutagenicity_flip_dominant")
    return {"path": str(resolved), "values": payload}


def merge_reward_config_cli(
    argv: Sequence[str],
    reward_config: Mapping[str, Any],
) -> list[str]:
    """Append audited values only when an explicit CLI override is absent."""

    output = list(argv)
    present = {
        token.split("=", 1)[0]
        for token in output
        if token.startswith("--")
    }
    for key, option in REWARD_CLI_KEYS.items():
        if option in present:
            continue
        output.extend((option, str(reward_config[key])))
    return output


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    fresh_args, delegated = build_parser().parse_known_args(raw)
    checkpoint_raw = _option_value(delegated, "--policy-adapter-checkpoint")
    output_raw = _option_value(delegated, "--output-dir")
    if not checkpoint_raw:
        raise ValueError("Fresh PPO requires explicit --policy-adapter-checkpoint")
    if not output_raw:
        raise ValueError("Fresh PPO requires explicit --output-dir")
    provenance = validate_fresh_policy_provenance(
        checkpoint_raw,
        explicit_audit=fresh_args.fresh_initialization_audit,
    )
    reward = load_reward_config(fresh_args.reward_config_json)
    delegated = merge_reward_config_cli(delegated, reward["values"])
    if _option_value(delegated, "--expected-policy-checkpoint-step") is None:
        delegated.extend(("--expected-policy-checkpoint-step", "0"))
    exit_code = stable_main(delegated)
    output_root = _resolve(output_raw)
    payload = {
        **provenance,
        "reward_config_path": reward["path"],
        "reward_profile": reward["values"]["reward_profile"],
        "shared_trainer": "scripts/train_mutagenicity_ppo_stable.py",
        "shared_algorithm_reimplemented": False,
        "fresh_ppo_provenance_passed": True,
    }
    write_json_atomic(output_root / "fresh_ppo_provenance.json", payload)
    mode = _option_value(delegated, "--mode") or "full"
    print(
        {
            "smoke": "[MUTAGENICITY_FRESH_PPO_SMOKE_OK]",
            "medium": "[MUTAGENICITY_FRESH_PPO_MEDIUM_OK]",
            "full": "[MUTAGENICITY_FRESH_PPO_FULL_OK]",
        }[mode],
        flush=True,
    )
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
