#!/usr/bin/env python3
"""Audit existing Mutagenicity PPO reward separation without retraining."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train.mutagenicity_fresh_sft import write_json_atomic  # noqa: E402


DEFAULT_RUN = Path("outputs/hpc/mutagenicity/ppo_stable_v1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hpc/mutagenicity/audits/ppo_reward_components_v1"),
    )
    parser.add_argument("--strict-flip-reward-margin", type=float, default=0.5)
    return parser


def _resolve(path: Path) -> Path:
    value = path.expanduser()
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _read_jsonl(path: Path, *, required: bool) -> list[dict[str, Any]]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _components(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("reward_components", row.get("breakdown", {}))
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _component(row: Mapping[str, Any], *keys: str) -> float:
    components = _components(row)
    for key in keys:
        value = _number(row.get(key))
        if value is None:
            value = _number(components.get(key))
        if value is not None:
            return value
    return 0.0


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * float(q)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: Iterable[Any]) -> dict[str, Any]:
    finite = [value for raw in values if (value := _number(raw)) is not None]
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
        "p10": _quantile(finite, 0.10),
        "p90": _quantile(finite, 0.90),
    }


def _confidence_bin(row: Mapping[str, Any]) -> str:
    value = _number(row.get("prob_before_1", row.get("p_before")))
    if value is None:
        return "missing"
    if value < 0.70:
        return "low"
    if value < 0.90:
        return "medium"
    return "high"


def _ratio_bin(row: Mapping[str, Any]) -> str:
    value = _number(row.get("atom_ratio"))
    if value is None:
        return "missing"
    if value < 0.25:
        return "low"
    if value <= 0.50:
        return "medium"
    return "high"


def _drop_bin(row: Mapping[str, Any]) -> str:
    value = _number(row.get("cf_drop"))
    if value is None:
        return "missing"
    if value <= 0.0:
        return "non_positive"
    if value < 0.30:
        return "low_positive"
    if value < 0.60:
        return "medium_positive"
    return "high_positive"


def _is_teacher_strict_flip(row: Mapping[str, Any]) -> bool:
    pred_before = _number(row.get("pred_before"))
    pred_after = _number(row.get("pred_after"))
    if pred_before is not None and pred_after is not None:
        return int(pred_before) == 1 and int(pred_after) == 0
    return bool(row.get("cf_flip"))


def _group_labels(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        (
            "strict_flip",
            "strict" if _is_teacher_strict_flip(row) else "non_flip",
        ),
        ("projection", "projected" if bool(row.get("projection_used")) else "direct_or_invalid"),
        ("validity", "valid" if bool(row.get("valid") or row.get("parse_ok")) else "invalid"),
        ("teacher_confidence", _confidence_bin(row)),
        ("atom_ratio", _ratio_bin(row)),
        ("cf_drop", _drop_bin(row)),
        ("strategy", str(row.get("candidate_strategy") or row.get("strategy") or "unknown")),
        ("reward_clipping", "clipped" if bool(row.get("reward_clipped")) else "not_clipped"),
    ]


def _row_components(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "total_reward": _component(row, "reward_total", "total"),
        "ppo_reward": _component(row, "ppo_reward", "reward_total", "total"),
        "strict_flip_bonus": _component(row, "strict_flip_bonus"),
        "cf_drop_component": _component(row, "counterfactual_sem", "cf_r"),
        "validity_component": _component(row, "valid_r", "valid_component"),
        "projection_component": -abs(_component(row, "projection_penalty")),
        "substructure_distance_component": _component(row, "subdist_contribution"),
        "size_component": _component(row, "size_window_r", "size_window_reward"),
        "syntax_component": _component(row, "format_r", "length_r", "dummy_r"),
        "kl_penalty": _component(row, "kl_penalty", "non_score_reward"),
    }


def _profile_raw_terms(row: Mapping[str, Any]) -> dict[str, float]:
    """Mirror the unweighted inputs consumed by the flip-dominant profile."""

    return {
        "validity": _component(
            row,
            "format_r",
        )
        + _component(row, "valid_r", "valid_component"),
        "substructure": _component(
            row,
            "subgraph_r",
            "substructure_component",
        )
        + _component(row, "subdist_contribution"),
        "size": _component(row, "size_window_r", "size_window_reward"),
    }


def summarize_groups(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for key in _group_labels(row):
            grouped[key].append(row)
    output: list[dict[str, Any]] = []
    for (dimension, value), members in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "group_dimension": dimension,
            "group_value": value,
            "count": len(members),
            "reward_clipping_rate": sum(bool(row.get("reward_clipped")) for row in members) / len(members),
        }
        for component in _row_components(members[0]):
            component_stats = _stats(
                _row_components(row)[component] for row in members
            )
            for stat, number in component_stats.items():
                if stat != "count":
                    summary[f"{component}_{stat}"] = number
        output.append(summary)
    return output


def recommend_config(
    rows: Sequence[Mapping[str, Any]],
    *,
    margin: float,
) -> dict[str, Any]:
    """Scale auxiliary terms from observed p90 magnitudes, then derive flip bonus."""

    if not any(_is_teacher_strict_flip(row) for row in rows):
        raise ValueError("Reward recommendation requires at least one strict-flip row")
    if not any(not _is_teacher_strict_flip(row) for row in rows):
        raise ValueError("Reward recommendation requires at least one non-flip row")

    def scale(component: str, target: float) -> float:
        values = [abs(_profile_raw_terms(row)[component]) for row in rows]
        denominator = _quantile(values, 0.90) or 1.0
        return target / max(denominator, 1e-6)

    validity_weight = scale("validity", 0.25)
    substructure_weight = scale("substructure", 0.25)
    size_weight = scale("size", 0.15)
    positive_drops = [
        value
        for row in rows
        if (value := _number(row.get("cf_drop"))) is not None and value > 0.0
    ]
    cf_drop_weight = 1.0 / max(_quantile(positive_drops, 0.90) or 1.0, 1e-6)
    non_flip_cap = 0.75
    non_flip_penalty = -0.5
    projection_penalty = min(
        0.25,
        abs(
            statistics.median(
                [
                    _row_components(row)["projection_component"]
                    for row in rows
                    if bool(row.get("projection_used"))
                ]
                or [0.0]
            )
        ),
    )

    def provisional(row: Mapping[str, Any], *, include_bonus: float = 0.0) -> float:
        terms = _profile_raw_terms(row)
        positive = (
            max(0.0, validity_weight * terms["validity"])
            + max(0.0, substructure_weight * terms["substructure"])
            + max(0.0, size_weight * terms["size"])
            + max(0.0, cf_drop_weight * max(_number(row.get("cf_drop")) or 0.0, 0.0))
        )
        strict = _is_teacher_strict_flip(row)
        if not strict:
            positive = min(positive, non_flip_cap)
        return (
            positive
            + (include_bonus if strict else non_flip_penalty)
            - (projection_penalty if bool(row.get("projection_used")) else 0.0)
        )

    strict_base = [
        provisional(row) for row in rows if _is_teacher_strict_flip(row)
    ]
    non_flip = [
        provisional(row) for row in rows if not _is_teacher_strict_flip(row)
    ]
    strict_p10 = _quantile(strict_base, 0.10) or 0.0
    non_flip_p90 = _quantile(non_flip, 0.90) or 0.0
    strict_bonus = max(0.0, non_flip_p90 + float(margin) - strict_p10)
    strict_rewards = [value + strict_bonus for value in strict_base]
    clip_max = max(5.0, (_quantile(strict_rewards, 0.99) or 4.0) + 1.0)
    return {
        "reward_profile": "mutagenicity_flip_dominant",
        "strict_flip_bonus": strict_bonus,
        "non_flip_penalty": non_flip_penalty,
        "cf_drop_weight": cf_drop_weight,
        "validity_weight": validity_weight,
        "substructure_weight": substructure_weight,
        "size_weight": size_weight,
        "projection_penalty": projection_penalty,
        "non_flip_aux_reward_cap": non_flip_cap,
        "strict_flip_reward_margin": float(margin),
        "reward_clip_min": -5.0,
        "reward_clip_max": clip_max,
        "derivation": {
            "method": "observed_component_p90_scale_and_margin_constraint",
            "strict_base_p10": strict_p10,
            "non_flip_p90": non_flip_p90,
            "projected_strict_p10_after_bonus": strict_p10 + strict_bonus,
            "target_margin": float(margin),
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = _resolve(args.run_dir)
    output = _resolve(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Reward audit output is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    candidates = _read_jsonl(run_dir / "candidate_pool.jsonl", required=True)
    ppo_metrics = _read_jsonl(run_dir / "ppo_metrics.jsonl", required=False)
    validation_metrics = _read_jsonl(
        run_dir / "validation_metrics.jsonl", required=False
    )
    if not candidates:
        raise ValueError("candidate_pool.jsonl contains no rows")
    resolved_config = _read_json_object(run_dir / "resolved_config.json")
    clip_min = _number(resolved_config.get("reward_clip_min"))
    clip_max = _number(resolved_config.get("reward_clip_max"))
    for row in candidates:
        reward = _number(row.get("reward_total", row.get("total")))
        inferred_boundary_clip = bool(
            reward is not None
            and (
                (clip_min is not None and reward <= clip_min + 1e-12)
                or (clip_max is not None and reward >= clip_max - 1e-12)
            )
        )
        row["reward_clipped"] = bool(
            row.get("reward_clipped") or inferred_boundary_clip
        )
    strict_rewards = [
        _row_components(row)["total_reward"]
        for row in candidates
        if _is_teacher_strict_flip(row)
    ]
    non_flip_rewards = [
        _row_components(row)["total_reward"]
        for row in candidates
        if not _is_teacher_strict_flip(row)
    ]
    strict_p10 = _quantile(strict_rewards, 0.10)
    strict_p50 = _quantile(strict_rewards, 0.50)
    non_flip_p90 = _quantile(non_flip_rewards, 0.90)
    overlap = {
        "num_candidates": len(candidates),
        "num_strict_flip": len(strict_rewards),
        "num_non_flip": len(non_flip_rewards),
        "recorded_cf_flip_vs_teacher_strict_mismatch_count": sum(
            bool(row.get("cf_flip")) != _is_teacher_strict_flip(row)
            for row in candidates
            if _number(row.get("pred_before")) is not None
            and _number(row.get("pred_after")) is not None
        ),
        "p10_total_reward_strict_flip": strict_p10,
        "p50_total_reward_strict_flip": strict_p50,
        "p90_total_reward_non_flip": non_flip_p90,
        "reward_margin": (
            strict_p10 - non_flip_p90
            if strict_p10 is not None and non_flip_p90 is not None
            else None
        ),
        "reward_clipping_rate": sum(
            bool(row.get("reward_clipped")) for row in candidates
        )
        / len(candidates),
        "strict_flip_clip_rate": (
            sum(
                bool(row.get("reward_clipped"))
                for row in candidates
                if _is_teacher_strict_flip(row)
            )
            / len(strict_rewards)
            if strict_rewards
            else 0.0
        ),
    }
    component_summary = []
    for component in _row_components(candidates[0]):
        stats = _stats(_row_components(row)[component] for row in candidates)
        component_summary.append({"component": component, **stats})
    group_summary = summarize_groups(candidates)
    recommendation = recommend_config(
        candidates,
        margin=float(args.strict_flip_reward_margin),
    )
    _write_csv(output / "reward_component_summary.csv", component_summary)
    _write_csv(output / "reward_group_summary.csv", group_summary)
    write_json_atomic(output / "reward_overlap.json", overlap)
    write_json_atomic(output / "recommended_reward_config.json", recommendation)
    write_json_atomic(
        output / "audit_inputs.json",
        {
            "run_dir": str(run_dir),
            "candidate_pool_rows": len(candidates),
            "ppo_metric_rows": len(ppo_metrics),
            "validation_metric_rows": len(validation_metrics),
            "reward_clip_min": clip_min,
            "reward_clip_max": clip_max,
            "clip_status_inferred_from_boundary_when_missing": True,
            "distance_or_teacher_recomputed": False,
        },
    )
    report = [
        "# Mutagenicity PPO Reward Component Audit",
        "",
        f"- Candidate rows: {len(candidates)}",
        f"- Strict flip rows: {len(strict_rewards)}",
        f"- Non-flip rows: {len(non_flip_rewards)}",
        f"- p10(strict): {strict_p10}",
        f"- p50(strict): {strict_p50}",
        f"- p90(non-flip): {non_flip_p90}",
        f"- Existing reward margin: {overlap['reward_margin']}",
        f"- Strict clipping rate: {overlap['strict_flip_clip_rate']:.6f}",
        "",
        "The recommended profile is derived from observed component scales and "
        "the requested strict-vs-non-flip margin. It is a training configuration "
        "proposal, not evidence of improved validation performance.",
        "",
    ]
    (output / "reward_audit_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print("[MUTAGENICITY_PPO_REWARD_COMPONENT_AUDIT_OK]")
    print("[MUTAGENICITY_PPO_FLIP_MARGIN_AUDIT_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
