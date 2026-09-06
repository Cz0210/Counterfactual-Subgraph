#!/usr/bin/env python3
"""Resume completed BACE BRICS train artifacts into at-most-K evaluation only."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ablations.llm import bace_l0_successor as route


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    phases = parser.add_subparsers(dest="phase", required=True)
    prepare = phases.add_parser("prepare")
    for key in ("source-train-root", "corrected-package-receipt", "output-root"):
        prepare.add_argument("--" + key, required=True)
    run = phases.add_parser("evaluate")
    for key in ("protocol-overlay", "portable-input-bundle", "gnn-input-bundle", "registry-root", "output-root"):
        run.add_argument("--" + key, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--cpu-threads", type=int, default=8)
    run.add_argument("--batch-size", type=int, default=256)
    package = phases.add_parser("package")
    for key in ("science-root", "gnn-input-bundle", "output-root"):
        package.add_argument("--" + key, required=True)
    importer = phases.add_parser("import-result")
    for key in ("archive", "package-receipt", "output-root", "registry-root"):
        importer.add_argument("--" + key, required=True)
    args = vars(parser.parse_args(argv))
    if args.pop("config") != "configs/hpc.yaml" or set(args.pop("set")) - {"inference.fallback_to_heuristic=false"}:
        parser.error("Frozen reference only; arbitrary science overrides forbidden")
    phase = args.pop("phase")
    # This deployment is explicitly scoped to czx on HPC. Never write elsewhere.
    allowed = Path("/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/llm") if phase == "import-result" else Path("/share/home/u20526/czx")
    for key in ("output_root", "registry_root"):
        if key in args:
            Path(args[key]).resolve().relative_to(allowed)
    if phase == "evaluate" and not 1 <= args["cpu_threads"] <= 8:
        parser.error("CPU threads must be 1..8")
    result = {"prepare": route.prepare, "evaluate": route.run, "package": route.package,
              "import-result": route.import_package}[phase](**args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if phase == "prepare" or result.get("state") == "PASS" else 75


if __name__ == "__main__":
    raise SystemExit(main())
