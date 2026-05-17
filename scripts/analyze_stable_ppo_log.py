#!/usr/bin/env python3
"""Analyze stable PPO logs by training segments and emit a concise report."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


TAG_LINE_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} [^|]+)\s+\|\s+"
    r"(?P<level>[A-Z]+)\s+\|\s+"
    r"(?P<logger>[^|]+)\s+\|\s+"
    r"\[(?P<tag>[A-Z0-9_]+)\]\s*"
    r"(?P<body>.*)$"
)
KV_RE = re.compile(r"(^|\s)(?P<key>[A-Za-z_][A-Za-z0-9_]*)=")

STABLE_UPDATE_TAG = "STABLE_PPO_UPDATE"
STABLE_GATE_TAG = "STABLE_PPO_TEACHER_CONF_GATE"
LEGACY_UPDATE_TAG = "DECODED_CHEM_PPO_UPDATE"
LEGACY_BATCH_SIZE_TAG = "CHEM_REWARD_CALLED"
LEGACY_CF_TAG = "CHEM_REWARD_CF_ORACLE"
LEGACY_FAILURE_TAG = "CHEM_REWARD_FAILURE_STATS"
LEGACY_PROJECTION_TAG = "CHEM_REWARD_PROJECTION_STATS"

STEP_RANGES = (
    ("1-50", 1, 50),
    ("51-100", 51, 100),
    ("101-150", 101, 150),
    ("151-200", 151, 200),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--err-log", required=True, help="Stable PPO stderr log path.")
    parser.add_argument("--out-json", required=True, help="Output summary JSON path.")
    parser.add_argument("--out-txt", required=True, help="Output readable TXT report path.")
    return parser


def parse_kv_body(body: str) -> dict[str, str]:
    matches = list(KV_RE.finditer(body))
    parsed: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = match.group("key")
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        parsed[key] = body[value_start:value_end].rstrip()
    return parsed


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except Exception:
        return default
    if math.isnan(numeric):
        return default
    return numeric


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _extract_step_metrics(log_path: Path) -> dict[int, dict[str, Any]]:
    per_step: dict[int, dict[str, Any]] = {}
    pending_legacy: dict[str, Any] = {}

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            match = TAG_LINE_RE.match(line)
            if not match:
                continue
            tag = match.group("tag")
            parsed = parse_kv_body(match.group("body"))

            if tag == STABLE_GATE_TAG:
                step = int(parsed.get("step") or 0)
                if step <= 0:
                    continue
                metrics = per_step.setdefault(step, {})
                if str(parsed.get("applied") or "").strip().lower() == "true":
                    metrics["teacher_conf_gate_applied_count"] = int(
                        metrics.get("teacher_conf_gate_applied_count", 0)
                    ) + 1
                continue

            if tag == STABLE_UPDATE_TAG:
                step = int(parsed.get("step") or 0)
                if step <= 0:
                    continue
                metrics = per_step.setdefault(step, {})
                for key in (
                    "reward_mean",
                    "parse_ok_rate",
                    "valid_rate",
                    "direct_substructure_rate",
                    "final_substructure_rate",
                    "projection_used_rate",
                    "oracle_ok_rate",
                    "cf_flip_rate",
                    "approx_kl",
                    "atom_ratio_mean",
                ):
                    metrics[key] = _safe_float(parsed.get(key))
                metrics["core_unusable_count"] = int(
                    _safe_float(parsed.get("core_unusable_count"))
                )
                metrics["parse_failed_count"] = int(
                    _safe_float(parsed.get("parse_failed_count"))
                )
                metrics.setdefault("teacher_conf_gate_applied_count", 0)
                continue

            if tag == LEGACY_BATCH_SIZE_TAG:
                pending_legacy["batch_size"] = int(_safe_float(parsed.get("batch_size"), 0.0))
                continue
            if tag == LEGACY_CF_TAG:
                pending_legacy["cf_flip_rate"] = _safe_float(parsed.get("cf_flip_rate"))
                pending_legacy["cf_oracle_called_rate"] = _safe_rate(
                    int(_safe_float(parsed.get("cf_oracle_called"), 0.0)),
                    int(pending_legacy.get("batch_size", 0) or 0),
                )
                pending_legacy["cf_drop_mean"] = _safe_float(parsed.get("cf_drop_mean"))
                continue
            if tag == LEGACY_FAILURE_TAG:
                batch_size = int(pending_legacy.get("batch_size", 0) or 0)
                parse_failed_count = int(_safe_float(parsed.get("parse_failed"), 0.0))
                sanitize_failed_count = int(_safe_float(parsed.get("sanitize_failed"), 0.0))
                pending_legacy["parse_failed_count"] = parse_failed_count
                pending_legacy["core_unusable_count"] = parse_failed_count + sanitize_failed_count
                pending_legacy["parse_ok_rate"] = 1.0 - _safe_rate(parse_failed_count, batch_size)
                continue
            if tag == LEGACY_PROJECTION_TAG:
                batch_size = int(pending_legacy.get("batch_size", 0) or 0)
                direct_count = int(_safe_float(parsed.get("direct_substructure_success"), 0.0))
                projection_success = int(_safe_float(parsed.get("success"), 0.0))
                pending_legacy["direct_substructure_rate"] = _safe_rate(direct_count, batch_size)
                pending_legacy["final_substructure_rate"] = _safe_rate(
                    direct_count + projection_success,
                    batch_size,
                )
                pending_legacy["projection_used_rate"] = _safe_rate(projection_success, batch_size)
                continue
            if tag == LEGACY_UPDATE_TAG:
                step = int(parsed.get("step") or 0)
                if step <= 0:
                    continue
                metrics = per_step.setdefault(step, {})
                metrics["reward_mean"] = _safe_float(parsed.get("reward_mean"))
                metrics["approx_kl"] = _safe_float(parsed.get("approx_kl"))
                for key in (
                    "parse_ok_rate",
                    "direct_substructure_rate",
                    "final_substructure_rate",
                    "projection_used_rate",
                    "cf_flip_rate",
                    "parse_failed_count",
                    "core_unusable_count",
                ):
                    if key in pending_legacy:
                        metrics[key] = pending_legacy[key]
                metrics.setdefault("oracle_ok_rate", pending_legacy.get("cf_oracle_called_rate", 0.0))
                metrics.setdefault("teacher_conf_gate_applied_count", 0)
                pending_legacy = {}

    return per_step


def _summarize_ranges(per_step: dict[int, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for label, start, end in STEP_RANGES:
        selected_steps = [step for step in sorted(per_step) if start <= step <= end]
        rows = [per_step[step] for step in selected_steps]
        reward_values = [_safe_float(row.get("reward_mean")) for row in rows]
        direct_values = [_safe_float(row.get("direct_substructure_rate")) for row in rows]
        final_values = [_safe_float(row.get("final_substructure_rate")) for row in rows]
        projection_values = [_safe_float(row.get("projection_used_rate")) for row in rows]
        oracle_values = [_safe_float(row.get("oracle_ok_rate")) for row in rows]
        cf_flip_values = [_safe_float(row.get("cf_flip_rate")) for row in rows]
        parse_ok_values = [_safe_float(row.get("parse_ok_rate")) for row in rows]
        approx_kl_values = [_safe_float(row.get("approx_kl")) for row in rows]
        atom_ratio_values = [_safe_float(row.get("atom_ratio_mean")) for row in rows]
        summary[label] = {
            "num_steps": len(selected_steps),
            "step_start": start,
            "step_end": end,
            "reward_mean": _mean(reward_values),
            "direct_substructure_rate": _mean(direct_values),
            "final_substructure_rate": _mean(final_values),
            "projection_used_rate": _mean(projection_values),
            "oracle_ok_rate": _mean(oracle_values),
            "cf_flip_rate": _mean(cf_flip_values),
            "parse_ok_rate": _mean(parse_ok_values),
            "core_unusable_count": int(
                sum(int(_safe_float(row.get("core_unusable_count"))) for row in rows)
            ),
            "parse_failed_count": int(
                sum(int(_safe_float(row.get("parse_failed_count"))) for row in rows)
            ),
            "approx_kl_mean": _mean(approx_kl_values),
            "approx_kl_max": max(approx_kl_values) if approx_kl_values else 0.0,
            "teacher_conf_gate_applied_count": int(
                sum(int(row.get("teacher_conf_gate_applied_count", 0)) for row in rows)
            ),
            "atom_ratio_mean": _mean(atom_ratio_values),
        }
    return summary


def _build_judgment(range_summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    last_range = range_summary["151-200"]
    first_range = range_summary["1-50"]
    drift_relieved = (
        last_range["reward_mean"] >= 3.3
        and last_range["cf_flip_rate"] >= 0.85
        and last_range["direct_substructure_rate"] >= 0.60
        and last_range["projection_used_rate"] <= 0.40
        and last_range["approx_kl_mean"] <= 0.50
        and last_range["core_unusable_count"] <= max(1, first_range["core_unusable_count"])
    )
    should_continue_stable300 = bool(drift_relieved)
    should_stop_and_move_to_full_pool = not bool(drift_relieved)
    return {
        "stable200_relieves_drift": bool(drift_relieved),
        "better_than_origin_shuffle100": "unknown_from_single_log",
        "worth_continue_stable300": bool(should_continue_stable300),
        "should_stop_ppo_and_move_to_full_candidate_pool": bool(
            should_stop_and_move_to_full_pool
        ),
    }


def _render_txt(
    *,
    log_path: Path,
    per_step: dict[int, dict[str, Any]],
    range_summary: dict[str, dict[str, Any]],
    judgment: dict[str, Any],
) -> str:
    lines = [
        f"log: {log_path}",
        f"num_steps_parsed: {len(per_step)}",
        "",
    ]
    for label, _start, _end in STEP_RANGES:
        item = range_summary[label]
        lines.extend(
            [
                f"[{label}]",
                f"reward_mean={item['reward_mean']:.4f}",
                f"direct_substructure_rate={item['direct_substructure_rate']:.4f}",
                f"final_substructure_rate={item['final_substructure_rate']:.4f}",
                f"projection_used_rate={item['projection_used_rate']:.4f}",
                f"oracle_ok_rate={item['oracle_ok_rate']:.4f}",
                f"cf_flip_rate={item['cf_flip_rate']:.4f}",
                f"parse_ok_rate={item['parse_ok_rate']:.4f}",
                f"core_unusable_count={item['core_unusable_count']}",
                f"parse_failed_count={item['parse_failed_count']}",
                f"approx_kl_mean={item['approx_kl_mean']:.4f}",
                f"approx_kl_max={item['approx_kl_max']:.4f}",
                f"teacher_conf_gate_applied_count={item['teacher_conf_gate_applied_count']}",
                "",
            ]
        )
    lines.extend(
        [
            "[judgment]",
            f"stable200_relieves_drift={judgment['stable200_relieves_drift']}",
            f"better_than_origin_shuffle100={judgment['better_than_origin_shuffle100']}",
            f"worth_continue_stable300={judgment['worth_continue_stable300']}",
            f"should_stop_ppo_and_move_to_full_candidate_pool={judgment['should_stop_ppo_and_move_to_full_candidate_pool']}",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = build_parser().parse_args()
    log_path = Path(args.err_log).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    per_step = _extract_step_metrics(log_path)
    range_summary = _summarize_ranges(per_step)
    judgment = _build_judgment(range_summary)
    payload = {
        "log_path": str(log_path),
        "num_steps_parsed": len(per_step),
        "ranges": range_summary,
        "judgment": judgment,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_txt.write_text(
        _render_txt(
            log_path=log_path,
            per_step=per_step,
            range_summary=range_summary,
            judgment=judgment,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
