#!/usr/bin/env python3
"""Convert selected official GCFExplainer graphs to SMILES when possible."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.gcf_graph_to_smiles import convert_selected_graphs_to_smiles  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--selected-graphs-path", required=True)
    parser.add_argument("--out-csv", default="outputs/hpc/gcfexplainer_official/graph_to_smiles/gcf_graph_smiles_candidates.csv")
    parser.add_argument("--out-report", default="outputs/hpc/gcfexplainer_official/graph_to_smiles/gcf_graph_to_smiles_report.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = convert_selected_graphs_to_smiles(
        selected_graphs_path=args.selected_graphs_path,
        out_csv=args.out_csv,
        out_report=args.out_report,
    )
    print("[GCF_GRAPH_TO_SMILES_SUMMARY]", flush=True)
    print(f"smiles_convert_ok_rate={report['smiles_convert_ok_rate']}", flush=True)
    print(f"sanitize_ok_rate={report['sanitize_ok_rate']}", flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

