#!/usr/bin/env python3
"""Finite CPU-only export/union; requires the actual new A+ freeze first."""
from pathlib import Path
import argparse
import functools
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--action", choices=("export", "union"), required=True)
    parser.add_argument("--binding", required=True, help="New immutable source/union descriptors JSON")
    parser.add_argument("--binding-sha", required=True)
    parser.add_argument("--experiment-spec", required=True)
    parser.add_argument("--experiment-spec-sha", required=True, help="Actual copied spec file SHA, not semantic digest")
    parser.add_argument("--evidence-root", required=True, help="Actual copied new contract.json and selection_freeze.json")
    parser.add_argument("--new-freeze-sha", required=True, help="Actual new selection_freeze.json file SHA")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not Path(args.config).is_file():
        parser.error("config absent")
    from src.experiments.bace_gin_reach_test_raw import (
        bound_json, export_test_raw, union_test_indexes, validate_aplus_freeze,
    )
    spec = bound_json({"path": args.experiment_spec, "sha256": args.experiment_spec_sha})
    binding = bound_json({"path": args.binding, "sha256": args.binding_sha})
    kwargs = dict(repo=Path(__file__).resolve().parents[2],
        new_freeze_path=Path(args.evidence_root) / "selection_freeze.json",
        new_freeze_sha=args.new_freeze_sha,
        validate_new_freeze=functools.partial(validate_aplus_freeze, spec=spec, evidence_root=args.evidence_root))
    if args.action == "export":
        result = export_test_raw(binding, Path(args.output), **kwargs)
    else:
        result = union_test_indexes(binding["original_index"], binding["ours_index"], Path(args.output), **kwargs)
    print(json.dumps({k: result[k] for k in ("state", "raw_cost_count", "source_parent_units",
        "source_finite_match_records", "new_test_freeze_sha256", "self_sha256", "ot_recomputed")}, sort_keys=True))


if __name__ == "__main__":
    main()
