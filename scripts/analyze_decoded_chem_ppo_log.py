#!/usr/bin/env python3
"""Analyze decoded chemistry PPO logs and export PPT-ready cases."""

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
ATOM_TOKEN_RE = re.compile(r"\[[^\]]+\]|Br|Cl|[A-Z][a-z]?|[bcnops]")

GENERATION_TAG = "DECODED_CHEM_GENERATION_SAMPLE"
FRAGMENT_TAG = "DECODED_CHEM_FRAGMENT_SAMPLE"
NORMALIZE_TAG = "DUMMY_FRAGMENT_NORMALIZED"
TEACHER_CALLED_TAG = "TEACHER_SEM_CALLED"
TEACHER_RESULT_TAG = "TEACHER_SEM_RESULT"
CF_CALLED_TAG = "CF_ORACLE_CALLED"
CF_RESULT_TAG = "CF_ORACLE_RESULT"
COMPONENTS_TAG = "CHEM_REWARD_COMPONENTS"
UPDATE_TAG = "DECODED_CHEM_PPO_UPDATE"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="Path to one decoded chemistry PPO log file.")
    parser.add_argument("--top-k", type=int, default=8, help="How many top cases to export.")
    parser.add_argument("--out-md", required=True, help="Markdown output path.")
    parser.add_argument("--out-jsonl", required=True, help="JSONL output path.")
    return parser


def parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def parse_number(value: str | None) -> int | float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        return float(text)
    except ValueError:
        return None


def format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def parse_kv_body(body: str) -> dict[str, str]:
    matches = list(KV_RE.finditer(body))
    parsed: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = match.group("key")
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        parsed[key] = body[value_start:value_end].rstrip()
    return parsed


def estimate_atom_count(smiles: str | None) -> int:
    if not smiles:
        return 0
    return len(ATOM_TOKEN_RE.findall(str(smiles)))


def is_trivial_fragment(smiles: str | None) -> bool:
    return estimate_atom_count(smiles) <= 2


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric):
        return default
    return numeric


def safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    parsed = parse_bool(str(value))
    return parsed


def normalize_case(case: dict[str, Any], *, source: str) -> dict[str, Any]:
    normalized = dict(case)

    for float_key in (
        "valid",
        "substructure",
        "length_reward",
        "teacher_prob",
        "fragment_teacher_sem",
        "p_before",
        "p_after",
        "cf_drop",
        "counterfactual_sem",
        "total_reward",
        "reward_mean",
        "policy_loss",
        "value_loss",
        "total_loss",
        "approx_kl",
    ):
        normalized[float_key] = parse_number(normalized.get(float_key))

    for int_key in ("step", "label", "dummy_count"):
        normalized[int_key] = (
            None if normalized.get(int_key) is None else safe_int(normalized.get(int_key), default=0)
        )

    for bool_key in ("raw_parse_ok", "core_parse_ok", "cf_flip"):
        normalized[bool_key] = to_bool_or_none(normalized.get(bool_key))

    normalized["source"] = source
    normalized["case_id"] = build_case_id(normalized)
    normalized["core_atom_count"] = estimate_atom_count(normalized.get("core_fragment"))
    normalized["selection_score"] = compute_selection_score(normalized)
    normalized["why_selected"] = build_why_selected(normalized)
    return normalized


def compute_selection_score(case: dict[str, Any]) -> float:
    valid = 1 if safe_float(case.get("valid")) == 1.0 else 0
    sub = 1 if safe_float(case.get("substructure")) == 1.0 else 0
    cf_flip = 1 if case.get("cf_flip") is True else 0
    cf_drop = max(safe_float(case.get("cf_drop")), 0.0)
    counterfactual_sem = max(safe_float(case.get("counterfactual_sem")), 0.0)
    trivial_penalty = 1 if is_trivial_fragment(case.get("core_fragment")) else 0
    total_reward = max(safe_float(case.get("total_reward")), 0.0)
    return (
        3.0 * cf_flip
        + 2.0 * cf_drop
        + 1.5 * counterfactual_sem
        + 1.0 * valid
        + 1.0 * sub
        + 0.25 * total_reward
        - 0.5 * trivial_penalty
    )


def sort_key(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        1 if case.get("cf_flip") is True else 0,
        1 if safe_float(case.get("counterfactual_sem")) > 0.0 else 0,
        1 if safe_float(case.get("cf_drop")) > 0.0 else 0,
        safe_float(case.get("counterfactual_sem")),
        safe_float(case.get("cf_drop")),
        safe_float(case.get("total_reward")),
        1 if safe_float(case.get("valid")) == 1.0 else 0,
        1 if safe_float(case.get("substructure")) == 1.0 else 0,
        0 if is_trivial_fragment(case.get("core_fragment")) else 1,
        safe_int(case.get("step"), default=0),
        safe_float(case.get("selection_score")),
    )


def build_case_id(case: dict[str, Any]) -> str:
    step = case.get("step")
    case_id = case.get("id")
    if step is not None and case_id is not None:
        return f"hpc_1398298_step{step}_id{case_id}"
    if step is not None:
        return f"hpc_1398298_step{step}"
    if case_id is not None:
        return f"hpc_1398298_id{case_id}"
    return "hpc_1398298_unknown"


def build_why_selected(case: dict[str, Any]) -> str:
    valid_ok = safe_float(case.get("valid")) == 1.0
    sub_ok = safe_float(case.get("substructure")) == 1.0
    cf_flip = case.get("cf_flip") is True
    cf_drop = safe_float(case.get("cf_drop"))
    counterfactual_sem = safe_float(case.get("counterfactual_sem"))
    dummy_count = safe_int(case.get("dummy_count"), default=0)

    labels: list[str] = []
    if dummy_count > 0:
        labels.append("raw-to-core normalization")
    if cf_flip:
        labels.append("cf_flip-positive")
    elif cf_drop > 0 and counterfactual_sem > 0:
        labels.append("cf_drop-positive but no flip")
    elif valid_ok and sub_ok:
        labels.append("pipeline-good but reward-negative")
    else:
        labels.append("best available case in log")

    if valid_ok and sub_ok:
        labels.append("valid/substructure-good")
    if safe_float(case.get("total_reward")) > 0:
        labels.append("positive total reward")

    return "; ".join(labels)


def parse_log(log_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts = {
        "generation_samples": 0,
        "reward_components": 0,
        "cf_oracle_result_lines": 0,
        "teacher_result_lines": 0,
        "ppo_update_lines": 0,
    }

    cases: list[dict[str, Any]] = []
    working: dict[str, Any] = {}
    pending_case_indices: list[int] = []

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            match = TAG_LINE_RE.match(line)
            if not match:
                continue

            tag = match.group("tag")
            body = match.group("body")
            parsed = parse_kv_body(body)

            if tag == GENERATION_TAG:
                counts["generation_samples"] += 1
                working = {
                    "parent_smiles": parsed.get("parent", "").strip() or None,
                    "generated_response": parsed.get("response"),
                }
                continue

            if tag == FRAGMENT_TAG:
                working["generated_response"] = (
                    working.get("generated_response")
                    if working.get("generated_response") is not None
                    else parsed.get("raw")
                )
                working["raw_fragment"] = (parsed.get("fragment") or "").strip() or None
                continue

            if tag == TEACHER_CALLED_TAG:
                working["teacher_input_smiles"] = (parsed.get("input") or "").strip() or None
                working["label"] = parse_number(parsed.get("label"))
                continue

            if tag == TEACHER_RESULT_TAG:
                counts["teacher_result_lines"] += 1
                working["teacher_input_smiles"] = (parsed.get("input") or "").strip() or None
                working["label"] = parse_number(parsed.get("label"))
                working["teacher_prob"] = parsed.get("prob")
                working["fragment_teacher_sem"] = parsed.get("teacher_sem")
                continue

            if tag == CF_CALLED_TAG:
                working["parent_smiles"] = (parsed.get("parent") or "").strip() or working.get("parent_smiles")
                working["core_fragment"] = (parsed.get("fragment") or "").strip() or working.get("core_fragment")
                working["label"] = parse_number(parsed.get("label")) if parsed.get("label") is not None else working.get("label")
                continue

            if tag == CF_RESULT_TAG:
                counts["cf_oracle_result_lines"] += 1
                working["parent_without_fragment_smiles"] = (
                    (parsed.get("parent_without_fragment") or "").strip() or None
                )
                working["p_before"] = parsed.get("p_before")
                working["p_after"] = parsed.get("p_after")
                working["cf_drop"] = parsed.get("cf_drop")
                working["cf_flip"] = parsed.get("cf_flip")
                working["counterfactual_sem"] = parsed.get("counterfactual_sem")
                continue

            if tag == NORMALIZE_TAG:
                working["raw_fragment"] = (
                    working.get("raw_fragment") or ((parsed.get("raw") or "").strip() or None)
                )
                working["core_fragment"] = (parsed.get("core") or "").strip() or None
                working["dummy_count"] = parsed.get("dummy_count")
                working["raw_parse_ok"] = parsed.get("raw_parse_ok")
                working["core_parse_ok"] = parsed.get("core_parse_ok")
                continue

            if tag == COMPONENTS_TAG:
                counts["reward_components"] += 1
                case = dict(working)
                case.update(
                    {
                        "id": parsed.get("id"),
                        "parent_smiles": (parsed.get("parent") or "").strip() or case.get("parent_smiles"),
                        "raw_fragment": (parsed.get("raw_fragment") or "").strip() or case.get("raw_fragment"),
                        "core_fragment": (parsed.get("core_fragment") or "").strip() or case.get("core_fragment"),
                        "valid": parsed.get("valid"),
                        "substructure": parsed.get("sub"),
                        "length_reward": parsed.get("len"),
                        "teacher_input_smiles": (parsed.get("teacher_input_smiles") or "").strip()
                        or case.get("teacher_input_smiles"),
                        "teacher_prob": parsed.get("teacher_prob") or case.get("teacher_prob"),
                        "fragment_teacher_sem": parsed.get("fragment_teacher_sem")
                        or case.get("fragment_teacher_sem"),
                        "parent_without_fragment_smiles": (parsed.get("parent_without_fragment_smiles") or "").strip()
                        or case.get("parent_without_fragment_smiles"),
                        "p_before": parsed.get("p_before") or case.get("p_before"),
                        "p_after": parsed.get("p_after") or case.get("p_after"),
                        "cf_drop": parsed.get("cf_drop") or case.get("cf_drop"),
                        "cf_flip": parsed.get("cf_flip") or case.get("cf_flip"),
                        "counterfactual_sem": parsed.get("counterfactual_sem")
                        or case.get("counterfactual_sem"),
                        "total_reward": parsed.get("total"),
                        "dummy_count": parsed.get("dummy_count") or case.get("dummy_count"),
                        "raw_parse_ok": parsed.get("raw_parse_ok") or case.get("raw_parse_ok"),
                        "core_parse_ok": parsed.get("core_parse_ok") or case.get("core_parse_ok"),
                    }
                )
                cases.append(case)
                pending_case_indices.append(len(cases) - 1)
                working = {}
                continue

            if tag == UPDATE_TAG:
                counts["ppo_update_lines"] += 1
                step = parse_number(parsed.get("step"))
                update_fields = {
                    "step": step,
                    "reward_mean": parsed.get("reward_mean"),
                    "policy_loss": parsed.get("policy_loss"),
                    "value_loss": parsed.get("value_loss"),
                    "total_loss": parsed.get("total_loss"),
                    "approx_kl": parsed.get("approx_kl"),
                }
                for case_index in pending_case_indices:
                    cases[case_index].update(update_fields)
                pending_case_indices = []

    return cases, counts


def build_summary(cases: list[dict[str, Any]], counts: dict[str, int]) -> dict[str, int]:
    summary = dict(counts)
    summary["parsed_cases"] = len(cases)
    summary["valid_one_count"] = sum(1 for case in cases if safe_float(case.get("valid")) == 1.0)
    summary["sub_one_count"] = sum(1 for case in cases if safe_float(case.get("substructure")) == 1.0)
    summary["cf_oracle_success_count"] = sum(
        1
        for case in cases
        if case.get("parent_without_fragment_smiles") not in (None, "")
        and case.get("p_before") is not None
        and case.get("p_after") is not None
    )
    summary["cf_drop_positive_count"] = sum(1 for case in cases if safe_float(case.get("cf_drop")) > 0.0)
    summary["cf_flip_true_count"] = sum(1 for case in cases if case.get("cf_flip") is True)
    summary["counterfactual_sem_positive_count"] = sum(
        1 for case in cases if safe_float(case.get("counterfactual_sem")) > 0.0
    )
    return summary


def render_markdown(
    *,
    log_path: Path,
    summary: dict[str, int],
    top_cases: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append(f"# Decoded Chemistry PPO Best Cases from `{log_path.name}`")
    lines.append("")
    lines.append("## Overall Statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Parsed generation samples | {summary['generation_samples']} |")
    lines.append(f"| Parsed reward components | {summary['reward_components']} |")
    lines.append(f"| Parsed cases | {summary['parsed_cases']} |")
    lines.append(f"| `valid=1.0` count | {summary['valid_one_count']} |")
    lines.append(f"| `sub=1.0` count | {summary['sub_one_count']} |")
    lines.append(f"| CF oracle success count | {summary['cf_oracle_success_count']} |")
    lines.append(f"| `cf_drop > 0` count | {summary['cf_drop_positive_count']} |")
    lines.append(f"| `cf_flip=True` count | {summary['cf_flip_true_count']} |")
    lines.append(
        f"| `counterfactual_sem > 0` count | {summary['counterfactual_sem_positive_count']} |"
    )
    lines.append("")
    lines.append("## Top Cases")
    lines.append("")
    lines.append(
        "| case_rank | step | id | parent_smiles | label | generated_response | raw_fragment | core_fragment | "
        "valid | substructure | p_before | p_after | cf_drop | cf_flip | counterfactual_sem | total_reward | PPO update info | why_selected |"
    )
    lines.append(
        "|---:|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|"
    )
    for rank, case in enumerate(top_cases, start=1):
        update_text = (
            f"reward_mean={format_value(case.get('reward_mean'))}; "
            f"policy_loss={format_value(case.get('policy_loss'))}; "
            f"value_loss={format_value(case.get('value_loss'))}; "
            f"total_loss={format_value(case.get('total_loss'))}; "
            f"approx_kl={format_value(case.get('approx_kl'))}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    format_value(case.get("step")),
                    format_value(case.get("id")),
                    format_value(case.get("parent_smiles")),
                    format_value(case.get("label")),
                    format_value(case.get("generated_response")),
                    format_value(case.get("raw_fragment")),
                    format_value(case.get("core_fragment")),
                    format_value(case.get("valid")),
                    format_value(case.get("substructure")),
                    format_value(case.get("p_before")),
                    format_value(case.get("p_after")),
                    format_value(case.get("cf_drop")),
                    format_value(case.get("cf_flip")),
                    format_value(case.get("counterfactual_sem")),
                    format_value(case.get("total_reward")),
                    update_text,
                    format_value(case.get("why_selected")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## PPT-ready Explanation")
    lines.append("")
    for rank, case in enumerate(top_cases, start=1):
        lines.append(f"### Case {rank}: `{case['case_id']}`")
        lines.append("")
        lines.append(
            f"- 这个案例来自 step {format_value(case.get('step'))}，"
            f"`raw_fragment={format_value(case.get('raw_fragment'))}`，"
            f"`core_fragment={format_value(case.get('core_fragment'))}`。"
        )
        lines.append(
            f"- 适合展示：{case['why_selected']}。"
            f" 其中 `p_before={format_value(case.get('p_before'))}`、"
            f"`p_after={format_value(case.get('p_after'))}`、"
            f"`cf_drop={format_value(case.get('cf_drop'))}`、"
            f"`cf_flip={format_value(case.get('cf_flip'))}`、"
            f"`counterfactual_sem={format_value(case.get('counterfactual_sem'))}`、"
            f"`total_reward={format_value(case.get('total_reward'))}`。"
        )
        lines.append("")
    return "\n".join(lines)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            serializable = {key: value for key, value in row.items() if key != "selection_score"}
            handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    log_path = Path(args.log).expanduser().resolve()
    out_md = Path(args.out_md).expanduser().resolve()
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()

    raw_cases, counts = parse_log(log_path)
    normalized_cases = [
        normalize_case(case, source=f"hpc_{log_path.stem}_err" if log_path.suffix == ".err" else f"hpc_{log_path.stem}")
        for case in raw_cases
    ]
    top_cases = sorted(normalized_cases, key=sort_key, reverse=True)[: max(0, int(args.top_k))]
    summary = build_summary(normalized_cases, counts)

    markdown = render_markdown(log_path=log_path, summary=summary, top_cases=top_cases)
    write_text(out_md, markdown)
    write_jsonl(out_jsonl, top_cases)


if __name__ == "__main__":
    main()
