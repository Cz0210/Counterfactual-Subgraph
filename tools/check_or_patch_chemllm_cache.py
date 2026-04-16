#!/usr/bin/env python3
"""Inspect and optionally patch a ChemLLM InternLM2 cache module.

This helper is meant to be synced to HPC and executed there against the real
Hugging Face dynamic-module cache file used at runtime.
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import shutil
import sys
from typing import Iterable


DANGEROUS_EXPR = "past_key_values[0][0].shape[2]"
HELPER_NAME = "_has_valid_past_key_values"
HELPER_GUARD = "if _has_valid_past_key_values(past_key_values):"

HELPER_SNIPPET = '''
def _has_valid_past_key_values(past_key_values) -> bool:
    """Return True only when the first cached key/value tensor is really present."""

    try:
        if past_key_values is None:
            return False
        if past_key_values[0] is None:
            return False
        if past_key_values[0][0] is None:
            return False
    except (IndexError, TypeError, KeyError):
        return False
    return True


'''.lstrip("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="目标 modeling_internlm2.py 路径。")
    parser.add_argument(
        "--patch",
        action="store_true",
        help="对目标文件执行稳健补丁，并在原文件旁创建 .bak 备份。",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=4,
        help="打印命中位置附近的上下文行数。",
    )
    return parser


def stripped(line: str) -> str:
    return line.strip()


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def indent_text(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def is_significant(line: str) -> bool:
    text = stripped(line)
    return bool(text) and not text.startswith("#")


def ensure_trailing_newline(line: str) -> str:
    return line if line.endswith("\n") else line + "\n"


def render_context(lines: list[str], *, line_number: int, radius: int) -> str:
    start = max(0, line_number - radius - 1)
    end = min(len(lines), line_number + radius)
    rendered: list[str] = []
    for current in range(start, end):
        marker = ">>" if current + 1 == line_number else "  "
        rendered.append(f"{marker} {current + 1:4d}: {lines[current].rstrip()}")
    return "\n".join(rendered)


def classify_occurrences(lines: list[str]) -> list[dict[str, object]]:
    """Classify dangerous accesses as guarded or unguarded by indentation scope."""

    occurrences: list[dict[str, object]] = []
    active_guard_indents: list[int] = []

    for index, line in enumerate(lines):
        if is_significant(line):
            current_indent = indent_of(line)
            while active_guard_indents and current_indent <= active_guard_indents[-1]:
                active_guard_indents.pop()

            if stripped(line) == HELPER_GUARD:
                active_guard_indents.append(current_indent)

        if DANGEROUS_EXPR in line:
            occurrences.append(
                {
                    "line_number": index + 1,
                    "guarded": bool(active_guard_indents),
                }
            )

    return occurrences


def summarize_file(text: str) -> dict[str, object]:
    lines = text.splitlines()
    occurrences = classify_occurrences([ensure_trailing_newline(line) for line in lines])
    guarded_lines = [item["line_number"] for item in occurrences if item["guarded"]]
    unguarded_lines = [item["line_number"] for item in occurrences if not item["guarded"]]
    return {
        "has_helper": HELPER_NAME in text,
        "guarded_count": len(guarded_lines),
        "unguarded_count": len(unguarded_lines),
        "guarded_lines": guarded_lines,
        "unguarded_lines": unguarded_lines,
    }


def print_summary(path: Path, text: str, *, context: int) -> None:
    summary = summarize_file(text)
    lines = [ensure_trailing_newline(line) for line in text.splitlines()]

    print(f"[summary] path={path}")
    print(f"[summary] has_{HELPER_NAME}={summary['has_helper']}")
    print(f"[summary] unguarded_count={summary['unguarded_count']}")
    print(f"[summary] guarded_count={summary['guarded_count']}")
    print(f"[summary] unguarded_lines={summary['unguarded_lines']}")
    print(f"[summary] guarded_lines={summary['guarded_lines']}")

    occurrences = classify_occurrences(lines)
    for index, occurrence in enumerate(occurrences, start=1):
        line_number = int(occurrence["line_number"])
        label = "guarded" if occurrence["guarded"] else "unguarded"
        print("")
        print(f"[{label} #{index}] line={line_number}")
        print(render_context(lines, line_number=line_number, radius=context))


def ensure_helper(text: str) -> tuple[str, str]:
    if HELPER_NAME in text:
        return text, "skipped existing helper"

    anchor = '_CONFIG_FOR_DOC = "InternLM2Config"\n'
    if anchor in text:
        return text.replace(anchor, anchor + "\n" + HELPER_SNIPPET, 1), "added helper"

    doc_anchor = "__all__ ="
    if doc_anchor in text:
        return text.replace(doc_anchor, HELPER_SNIPPET + doc_anchor, 1), "added helper"

    return HELPER_SNIPPET + text, "added helper"


def patch_forward_blocks(lines: list[str]) -> tuple[list[str], dict[str, int]]:
    stats = {"patched": 0, "skipped": 0}
    result: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]

        if (
            stripped(line) == HELPER_GUARD
            and index + 2 < len(lines)
            and stripped(lines[index + 1]) == f"past_key_values_length = {DANGEROUS_EXPR}"
            and stripped(lines[index + 2]) == "seq_length_with_past = seq_length_with_past + past_key_values_length"
        ):
            stats["skipped"] += 1
            result.append(line)
            index += 1
            continue

        if (
            stripped(line) == "if past_key_values is not None:"
            and index + 2 < len(lines)
            and stripped(lines[index + 1]) == f"past_key_values_length = {DANGEROUS_EXPR}"
            and stripped(lines[index + 2]) == "seq_length_with_past = seq_length_with_past + past_key_values_length"
        ):
            base_indent = indent_text(line)
            body_indent = base_indent + " " * 4
            result.extend(
                [
                    f"{base_indent}{HELPER_GUARD}\n",
                    f"{body_indent}past_key_values_length = {DANGEROUS_EXPR}\n",
                    f"{body_indent}seq_length_with_past = seq_length_with_past + past_key_values_length\n",
                    f"{base_indent}else:\n",
                    f"{body_indent}past_key_values = None\n",
                ]
            )
            stats["patched"] += 1
            index += 3
            continue

        result.append(line)
        index += 1

    return result, stats


def find_block_end(lines: list[str], start_index: int, base_indent: int) -> int:
    end_index = start_index
    index = start_index + 1
    while index < len(lines):
        candidate = lines[index]
        if not is_significant(candidate):
            end_index = index
            index += 1
            continue
        if indent_of(candidate) <= base_indent:
            break
        end_index = index
        index += 1
    return end_index


def patch_prepare_blocks(lines: list[str]) -> tuple[list[str], dict[str, int]]:
    stats = {"patched": 0, "skipped": 0}
    result: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]

        if (
            stripped(line) == HELPER_GUARD
            and index + 1 < len(lines)
            and stripped(lines[index + 1]) == f"past_length = {DANGEROUS_EXPR}"
        ):
            stats["skipped"] += 1
            result.append(line)
            index += 1
            continue

        if (
            stripped(line) == "if past_key_values is not None:"
            and index + 1 < len(lines)
            and stripped(lines[index + 1]) == f"past_length = {DANGEROUS_EXPR}"
        ):
            base_indent = indent_of(line)
            base_indent_text = indent_text(line)
            body_indent_text = base_indent_text + " " * 4
            block_end = find_block_end(lines, index, base_indent)
            block_lines = lines[index + 1 : block_end + 1]
            has_target_line = any(
                stripped(candidate).startswith("input_ids = input_ids[:, remove_prefix_length")
                for candidate in block_lines
            )
            if not has_target_line:
                result.append(line)
                index += 1
                continue

            result.append(f"{base_indent_text}{HELPER_GUARD}\n")
            result.extend(block_lines)
            result.extend(
                [
                    f"{base_indent_text}else:\n",
                    f"{body_indent_text}past_key_values = None\n",
                    f"{body_indent_text}past_length = 0\n",
                ]
            )
            stats["patched"] += 1
            index = block_end + 1
            continue

        result.append(line)
        index += 1

    return result, stats


def apply_patch(text: str) -> tuple[str, dict[str, str]]:
    patched_text, helper_status = ensure_helper(text)
    lines = patched_text.splitlines(keepends=True)

    lines, forward_stats = patch_forward_blocks(lines)
    lines, prepare_stats = patch_prepare_blocks(lines)
    patched_text = "".join(lines)

    summary = {
        "helper": helper_status,
        "forward": (
            f"patched x{forward_stats['patched']}"
            if forward_stats["patched"]
            else f"skipped existing x{forward_stats['skipped']}" if forward_stats["skipped"] else "no match"
        ),
        "prepare_inputs_for_generation": (
            f"patched x{prepare_stats['patched']}"
            if prepare_stats["patched"]
            else f"skipped existing x{prepare_stats['skipped']}" if prepare_stats["skipped"] else "no match"
        ),
    }
    return patched_text, summary


def write_backup_if_needed(path: Path) -> Path:
    backup_path = Path(f"{path}.bak")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
    return backup_path


def unified_diff(before: str, after: str, *, path: Path) -> str:
    diff_iter: Iterable[str] = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=str(path),
        tofile=f"{path} (patched)",
    )
    return "".join(diff_iter)


def main() -> int:
    args = build_parser().parse_args()
    target_path = Path(args.path).expanduser()
    if not target_path.exists():
        print(f"[error] file does not exist: {target_path}", file=sys.stderr)
        return 1

    original_text = target_path.read_text(encoding="utf-8")
    print_summary(target_path, original_text, context=args.context)

    if not args.patch:
        return 0

    patched_text, patch_summary = apply_patch(original_text)
    if patched_text == original_text:
        print("")
        print(f"[patch] helper={patch_summary['helper']}")
        print(f"[patch] forward={patch_summary['forward']}")
        print(f"[patch] prepare_inputs_for_generation={patch_summary['prepare_inputs_for_generation']}")
        print("[patch] no textual changes were necessary.")
        return 0

    backup_path = write_backup_if_needed(target_path)
    target_path.write_text(patched_text, encoding="utf-8")

    print("")
    print(f"[patch] backup={backup_path}")
    print(f"[patch] helper={patch_summary['helper']}")
    print(f"[patch] forward={patch_summary['forward']}")
    print(f"[patch] prepare_inputs_for_generation={patch_summary['prepare_inputs_for_generation']}")
    print("[patch] diff:")
    print(unified_diff(original_text, patched_text, path=target_path))
    print("")
    print_summary(target_path, patched_text, context=args.context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
