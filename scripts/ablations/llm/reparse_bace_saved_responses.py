#!/usr/bin/env python3
"""Extract saved train-only responses with the corrected common parser, no model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ablations.llm.parser_correction import reparse_scored_diagnostic
from src.eval.bace_frozen_gnn_contracts import atomic_json, atomic_jsonl, read_jsonl, sha256_file


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/hpc.yaml')
    parser.add_argument('--set',action='append',default=[])
    parser.add_argument('--scored-jsonl',required=True)
    parser.add_argument('--output-root',required=True)
    args=parser.parse_args()
    if set(args.set)-{'inference.fallback_to_heuristic=false'}:
        parser.error('No science overrides are supported')
    source=Path(args.scored_jsonl).resolve(strict=True)
    output=Path(args.output_root).resolve()
    if output == source.parent or source.parent in output.parents or output in source.parents:
        parser.error('Diagnostic output must be separate from its input directory')
    output.mkdir(parents=True,exist_ok=False)
    report=reparse_scored_diagnostic(read_jsonl(source))
    rows=report.pop('rows')
    report.update(source_path=str(source),source_bytes=source.stat().st_size,source_sha256=sha256_file(source))
    atomic_jsonl(output/'reparsed_diagnostic_rows.jsonl',rows)
    atomic_json(output/'parser_diagnostic.json',report)
    print(json.dumps(report,sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
