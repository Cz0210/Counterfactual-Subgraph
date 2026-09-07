#!/usr/bin/env python3
"""CPU-only dataset-specific frozen pool screening entrypoint."""
import argparse
import json
from pathlib import Path
from src.baselines.comrecgc.rf_aligned_pool import screen_pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Project config, recorded only; no inference fallback")
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--action", choices=["screen-pool", "repair-gaps", "recourse", "freeze-summary", "evaluate", "status"], required=True)
    parser.add_argument("--pool-root")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = Path(args.output_root)
    if args.action == "status":
        path = root / "terminal.json"
        if not path.exists():
            candidates = [root / name for name in ["progress.json", "exact_count/progress.json", "resource_admission.json", "input_ram_admission.json", "graph_ram_admission.json"]]
            candidates = [p for p in candidates if p.exists()]
            path = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else path
        print(path.read_text() if path.exists() else json.dumps({"state": "NOT_STARTED"}))
    elif args.action in ('freeze-summary', 'evaluate'):
        from src.baselines.comrecgc.rf_aligned_release import freeze_summary, evaluate_frozen_summary
        if not args.pool_root:
            parser.error('--pool-root must identify the completed native recourse root')
        operation = freeze_summary if args.action == 'freeze-summary' else evaluate_frozen_summary
        print(json.dumps(operation(json.loads(Path(args.run_manifest).read_text()), recourse_root=Path(args.pool_root), output_root=root), sort_keys=True))
    elif args.action == "repair-gaps":
        from src.baselines.comrecgc.rf_aligned_pool import repair_screen_gaps
        if not args.pool_root:
            parser.error("--pool-root is required for repair-gaps")
        config = json.loads(Path(args.run_manifest).read_text())
        print(json.dumps(repair_screen_gaps(config, source_root=Path(args.pool_root), output_root=root), sort_keys=True))
    elif args.action == "recourse":
        import fcntl
        from src.baselines.comrecgc.rf_aligned_recourse import run_recourse
        if not args.pool_root:
            parser.error("--pool-root is required for recourse")
        config = json.loads(Path(args.run_manifest).read_text())
        root.mkdir(parents=True, exist_ok=True)
        with (root / 'writer.lock').open('a+') as writer:
            fcntl.flock(writer.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(json.dumps(run_recourse(config, pool_root=Path(args.pool_root), output_root=root), sort_keys=True))
    else:
        config = json.loads(Path(args.run_manifest).read_text())
        print(json.dumps(screen_pool(config, root), sort_keys=True))


if __name__ == "__main__":
    main()
