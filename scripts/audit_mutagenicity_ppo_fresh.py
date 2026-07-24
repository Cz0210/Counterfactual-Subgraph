#!/usr/bin/env python3
"""Audit Fresh Mutagenicity PPO provenance, reward, and parent coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train.mutagenicity_fresh_sft import write_json_atomic  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-mode", choices=("smoke", "medium", "full"), required=True)
    return parser


def _resolve(path: Path) -> Path:
    value = path.expanduser()
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _read(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _resolve(args.run_dir)
    resolved = _read(root, "resolved_config.json")
    model = _read(root, "model_audit.json")
    reward = _read(root, "reward_config.json")
    coverage = _read(root, "parent_coverage.json")
    provenance = _read(root, "fresh_ppo_provenance.json")
    errors: list[str] = []
    if resolved.get("mode") != args.expected_mode:
        errors.append(f"mode={resolved.get('mode')!r}")
    if resolved.get("source_label") != 1 or resolved.get("target_label") != 0:
        errors.append("source/target label mismatch")
    if reward.get("profile_name") != "mutagenicity_flip_dominant":
        errors.append(f"reward profile={reward.get('profile_name')!r}")
    if not bool(reward.get("enabled")):
        errors.append("flip-dominant reward is disabled")
    if int(model.get("reference_trainable_params", -1)) != 0:
        errors.append("reference model is trainable")
    if int(model.get("base_params_trainable", -1)) != 0:
        errors.append("policy base parameters are trainable")
    if int(model.get("value_head_trainable_params", 0)) <= 0:
        errors.append("value head has no trainable parameters")
    if not bool(provenance.get("fresh_policy_provenance_passed")):
        errors.append("fresh policy provenance failed")
    if bool(provenance.get("aids_adapter_weights_loaded")):
        errors.append("AIDS adapter weights were loaded")
    expected = {"smoke": (5, 5), "medium": (256, 16), "full": (1448, 91)}
    expected_parents, expected_updates = expected[args.expected_mode]
    if int(coverage.get("num_dataset_rows", -1)) != expected_parents:
        errors.append(
            f"parent count={coverage.get('num_dataset_rows')} expected={expected_parents}"
        )
    if int(coverage.get("max_updates", -1)) != expected_updates:
        errors.append(
            f"max updates={coverage.get('max_updates')} expected={expected_updates}"
        )
    if float(coverage.get("unique_parent_coverage", 0.0)) < 1.0 - 1e-12:
        errors.append(
            f"unique parent coverage={coverage.get('unique_parent_coverage')}"
        )
    if bool(coverage.get("sampling_with_replacement", True)):
        errors.append("sampling_with_replacement is true")
    payload = {
        "run_dir": str(root),
        "mode": args.expected_mode,
        "expected_parents": expected_parents,
        "expected_updates": expected_updates,
        "audit_errors": errors,
        "fresh_ppo_audit_passed": not errors,
    }
    write_json_atomic(root / "fresh_ppo_audit.json", payload)
    (root / "fresh_ppo_audit.md").write_text(
        "\n".join(
            [
                "# Mutagenicity Fresh PPO Audit",
                "",
                f"- Mode: `{args.expected_mode}`",
                f"- Expected parents: {expected_parents}",
                f"- Expected updates: {expected_updates}",
                f"- Errors: {errors}",
                f"- Passed: {str(not errors).lower()}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    if errors:
        raise SystemExit(2)
    print(
        {
            "smoke": "[MUTAGENICITY_FRESH_PPO_SMOKE_OK]",
            "medium": "[MUTAGENICITY_FRESH_PPO_MEDIUM_OK]",
            "full": "[MUTAGENICITY_FRESH_PPO_FULL_OK]",
        }[args.expected_mode]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
