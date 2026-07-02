#!/usr/bin/env python3
"""Export GlobalGCE official artifacts into project-owned JSON/JSONL files."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_adapter import (  # noqa: E402
    label_alignment_audit,
    load_globalgce_cfs,
    load_globalgce_rules,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GlobalGCE official result files for unified evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-root", default="outputs/hpc/globalgce/aids_official_top30")
    parser.add_argument("--dataset", default="AIDS")
    parser.add_argument("--out-dir", default="outputs/hpc/globalgce/aids_official_top30_exported")
    parser.add_argument("--config", default=None, help="Ignored compatibility hook for HPC wrappers.")
    parser.add_argument("--set", action="append", default=[], help="Ignored compatibility hook for HPC wrappers.")
    return parser.parse_args()


def load_pickle_or_torch(path: Path) -> Any:
    try:
        import torch  # type: ignore

        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
        except Exception:
            pass
    except Exception:
        pass
    with path.open("rb") as handle:
        return pickle.load(handle)


def shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(item) for item in shape]
    except Exception:
        return None


def describe_object(value: Any, *, depth: int = 0, max_depth: int = 4) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "type": type(value).__name__,
    }
    shape = shape_of(value)
    if shape is not None:
        summary["shape"] = shape
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        summary["dtype"] = str(dtype)
    try:
        summary["len"] = len(value)
    except Exception:
        pass

    if depth >= max_depth:
        summary["raw_repr_summary"] = repr(value)[:500]
        return summary

    if isinstance(value, dict):
        keys = list(value.keys())
        summary["keys"] = [str(key) for key in keys[:50]]
        summary["items"] = {
            str(key): describe_object(value[key], depth=depth + 1, max_depth=max_depth)
            for key in keys[:20]
        }
    elif isinstance(value, (list, tuple)):
        summary["items"] = [
            describe_object(item, depth=depth + 1, max_depth=max_depth)
            for item in list(value)[:5]
        ]
    else:
        summary["raw_repr_summary"] = repr(value)[:500]
    return summary


def find_files(run_root: Path, dataset: str) -> dict[str, list[str]]:
    run_src = run_root / "GlobalGCE_src"
    saved = run_src / "saved_results"
    patterns = {
        "exp_results": [f"saved_exp_res/GlobalGCE/*{dataset}*exp_res.txt"],
        "rules": [
            f"saved_rules/GlobalGCE/*{dataset}*rules.pt",
            f"saved_rules/GlobalGCE/*{dataset}*rules.pkl",
            f"saved_rules/*{dataset}*rules.pt",
            f"saved_rules/*{dataset}*rules.pkl",
        ],
        "cfs": [
            f"saved_cfs/GlobalGCE/*{dataset}*cfs.pkl",
            f"saved_cfs/GlobalGCE/*{dataset}*cfs.pt",
            f"saved_cfs/*{dataset}*cfs.pkl",
            f"saved_cfs/*{dataset}*cfs.pt",
        ],
        "gnn_models": [f"saved_models/gnn_model/{dataset}.pt"],
    }
    found: dict[str, list[str]] = {}
    for key, key_patterns in patterns.items():
        values: list[Path] = []
        for pattern in key_patterns:
            values.extend(saved.glob(pattern))
        found[key] = [str(path.resolve()) for path in sorted(set(values))]
    return found


def parse_metric_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    experiments: list[dict[str, Any]] = []
    summary_line = None
    for line in text.splitlines():
        if line.startswith("Experiment "):
            experiments.append({"line": line})
        if line.startswith("Summary:"):
            summary_line = line

    parsed_summary: dict[str, Any] = {}
    if summary_line:
        numbers = [float(item) for item in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", summary_line)]
        parsed_summary = {
            "line": summary_line,
            "numbers": numbers,
        }

    return {
        "source_path": str(path),
        "num_experiment_lines": len(experiments),
        "experiments": experiments,
        "summary": parsed_summary,
        "raw_text": text,
    }


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_root": str(run_root),
        "dataset": args.dataset,
        "files": find_files(run_root, args.dataset),
        "label_alignment_audit": label_alignment_audit(),
    }
    (out_dir / "globalgce_files_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    metrics = [
        parse_metric_file(Path(path))
        for path in manifest["files"].get("exp_results", [])
    ]
    (out_dir / "globalgce_official_metrics.json").write_text(
        json.dumps({"dataset": args.dataset, "metrics": metrics}, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    introspection: list[dict[str, Any]] = []
    all_rules: list[dict[str, Any]] = []
    for path_str in manifest["files"].get("rules", []):
        path = Path(path_str)
        try:
            payload = load_pickle_or_torch(path)
            introspection.append(
                {
                    "source_path": str(path),
                    "ok": True,
                    "summary": describe_object(payload),
                }
            )
            all_rules.extend(load_globalgce_rules(path))
        except Exception as exc:
            introspection.append(
                {
                    "source_path": str(path),
                    "ok": False,
                    "error": str(exc),
                }
            )
    (out_dir / "globalgce_rules_introspection.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "num_rule_files": len(manifest["files"].get("rules", [])),
                "num_exported_rules": len(all_rules),
                "files": introspection,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_jsonl(out_dir / "globalgce_rules.jsonl", all_rules)

    all_cfs: list[dict[str, Any]] = []
    cf_files: list[dict[str, Any]] = []
    for path_str in manifest["files"].get("cfs", []):
        path = Path(path_str)
        try:
            records = load_globalgce_cfs(path)
            all_cfs.extend(records)
            cf_files.append(
                {
                    "source_path": str(path),
                    "ok": True,
                    "num_records": len(records),
                }
            )
        except Exception as exc:
            cf_files.append(
                {
                    "source_path": str(path),
                    "ok": False,
                    "error": str(exc),
                }
            )
    (out_dir / "globalgce_cfs_manifest.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "num_cf_files": len(manifest["files"].get("cfs", [])),
                "num_exported_cfs": len(all_cfs),
                "files": cf_files,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_jsonl(out_dir / "globalgce_cfs_graphs.jsonl", all_cfs)

    print("[GLOBALGCE_EXPORT]")
    print(f"out_dir={out_dir}")
    print(f"num_rules={len(all_rules)}")
    print(f"num_cfs={len(all_cfs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
