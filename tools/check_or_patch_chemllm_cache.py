#!/usr/bin/env python3
"""Inspect and optionally patch a ChemLLM InternLM2 cache module.

This helper is meant to be copied to / executed on HPC after the repository has
been synced there. It focuses on the cache-related crash pattern:

    past_key_values[0][0].shape[2]

where the outer ``past_key_values`` object is non-null but the first cached
key/value entry is still ``None``.
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import re
import shutil
import sys
from typing import Iterable


DANGEROUS_EXPR = "past_key_values[0][0].shape[2]"
HELPER_NAME = "_has_valid_past_key_values"
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


FORWARD_PATTERN = re.compile(
    r"(?P<indent>[ \t]*)seq_length_with_past = seq_length\n"
    r"(?P=indent)past_key_values_length = 0\n"
    r"(?P=indent)if past_key_values is not None:\n"
    r"(?P=indent)    past_key_values_length = past_key_values\[0\]\[0\]\.shape\[2\]\n"
    r"(?P=indent)    seq_length_with_past = seq_length_with_past \+ past_key_values_length",
    re.MULTILINE,
)

PREPARE_PATTERN = re.compile(
    r"(?P<indent>[ \t]*)if past_key_values is not None:\n"
    r"(?P=indent)    past_length = past_key_values\[0\]\[0\]\.shape\[2\]\n"
    r"(?P<body>(?:\n(?P=indent)    .*)*?\n(?P=indent)    input_ids = input_ids\[:, remove_prefix_length\])",
    re.MULTILINE,
)


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
        help="打印危险命中附近的上下文行数。",
    )
    return parser


def count_occurrences(text: str, needle: str) -> list[int]:
    positions: list[int] = []
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            return positions
        positions.append(index)
        start = index + len(needle)


def byte_offset_to_line_number(text: str, offset: int) -> int:
    return text[:offset].count("\n") + 1


def render_context(text: str, *, line_number: int, radius: int) -> str:
    lines = text.splitlines()
    start = max(0, line_number - radius - 1)
    end = min(len(lines), line_number + radius)
    rendered: list[str] = []
    for current in range(start, end):
        marker = ">>" if current + 1 == line_number else "  "
        rendered.append(f"{marker} {current + 1:4d}: {lines[current]}")
    return "\n".join(rendered)


def summarize_file(text: str) -> dict[str, object]:
    occurrences = count_occurrences(text, DANGEROUS_EXPR)
    return {
        "has_helper": HELPER_NAME in text,
        "dangerous_count": len(occurrences),
        "dangerous_lines": [byte_offset_to_line_number(text, offset) for offset in occurrences],
    }


def print_summary(path: Path, text: str, *, context: int) -> None:
    summary = summarize_file(text)
    print(f"[summary] path={path}")
    print(f"[summary] has_{HELPER_NAME}={summary['has_helper']}")
    print(f"[summary] dangerous_count={summary['dangerous_count']}")
    if summary["dangerous_lines"]:
        print(f"[summary] dangerous_lines={summary['dangerous_lines']}")
    else:
        print("[summary] dangerous_lines=[]")

    occurrences = count_occurrences(text, DANGEROUS_EXPR)
    for index, offset in enumerate(occurrences, start=1):
        line_number = byte_offset_to_line_number(text, offset)
        print("")
        print(f"[context #{index}] line={line_number}")
        print(render_context(text, line_number=line_number, radius=context))


def ensure_helper(text: str) -> tuple[str, bool]:
    if HELPER_NAME in text:
        return text, False

    anchor = '_CONFIG_FOR_DOC = "InternLM2Config"\n'
    if anchor in text:
        return text.replace(anchor, anchor + "\n" + HELPER_SNIPPET, 1), True

    doc_anchor = "__all__ ="
    if doc_anchor in text:
        return text.replace(doc_anchor, HELPER_SNIPPET + doc_anchor, 1), True

    return HELPER_SNIPPET + text, True


def patch_forward_block(text: str) -> tuple[str, int]:
    def _replace(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}seq_length_with_past = seq_length\n"
            f"{indent}past_key_values_length = 0\n"
            f"{indent}if _has_valid_past_key_values(past_key_values):\n"
            f"{indent}    past_key_values_length = past_key_values[0][0].shape[2]\n"
            f"{indent}    seq_length_with_past = seq_length_with_past + past_key_values_length\n"
            f"{indent}else:\n"
            f"{indent}    past_key_values = None"
        )

    return FORWARD_PATTERN.subn(_replace, text)


def patch_prepare_block(text: str) -> tuple[str, int]:
    def _replace(match: re.Match[str]) -> str:
        indent = match.group("indent")
        body = match.group("body")
        return (
            f"{indent}if _has_valid_past_key_values(past_key_values):\n"
            f"{indent}    past_length = past_key_values[0][0].shape[2]"
            f"{body}\n"
            f"{indent}else:\n"
            f"{indent}    past_key_values = None\n"
            f"{indent}    past_length = 0"
        )

    return PREPARE_PATTERN.subn(_replace, text)


def apply_patch(text: str) -> tuple[str, list[str]]:
    patched_text = text
    changes: list[str] = []

    patched_text, helper_changed = ensure_helper(patched_text)
    if helper_changed:
        changes.append("added _has_valid_past_key_values helper")

    patched_text, forward_changes = patch_forward_block(patched_text)
    if forward_changes:
        changes.append(f"patched forward past_key_values_length block x{forward_changes}")

    patched_text, prepare_changes = patch_prepare_block(patched_text)
    if prepare_changes:
        changes.append(f"patched prepare_inputs_for_generation block x{prepare_changes}")

    return patched_text, changes


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

    patched_text, changes = apply_patch(original_text)
    if patched_text == original_text:
        print("[patch] no textual changes were necessary.")
        return 0

    backup_path = write_backup_if_needed(target_path)
    target_path.write_text(patched_text, encoding="utf-8")

    print("")
    print(f"[patch] backup={backup_path}")
    print(f"[patch] changes={changes}")
    print("[patch] diff:")
    print(unified_diff(original_text, patched_text, path=target_path))
    print("")
    print_summary(target_path, patched_text, context=args.context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
