#!/usr/bin/env python3
"""AIDS-only completed RF pair-store CPU continuation and phase admission."""
import argparse
import json
import hashlib
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--recourse-root", required=True)
    parser.add_argument("--pool-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--action", choices=["plan", "owner", "cluster-existing", "summary-existing", "global-witness", "status"], required=True)
    parser.add_argument("--writer-fd", type=int)
    args = parser.parse_args()
    root = Path(args.output_root); recourse = Path(args.recourse_root); pool = Path(args.pool_root)
    if args.action == "status":
        for name in ("terminal.json", "failed.json", "heartbeat.json"):
            if (root/name).exists(): print((root/name).read_text()); return
        print(json.dumps({"state": "NOT_STARTED"})); return
    config = json.loads(Path(args.run_manifest).read_text())
    if config.get("phase_overlay_files"):
        # An explicit immutable code overlay, not a claim that base HEAD contains
        # the new driver. Only three named AIDS files can override base imports.
        expected = {"scripts/continue_aids_rf_pairs.py", "src/baselines/comrecgc/rf_aligned_cluster_phase.py",
                    "src/baselines/comrecgc/rf_aligned_phase_owner.py"}
        if config.get("phase_route") == "GLOBAL_RADIUS_WITNESS_ONLY":
            expected.add("src/baselines/comrecgc/aids_global_witness.py")
        overlay = Path(config["phase_overlay_root"])
        if set(config["phase_overlay_files"]) != expected or Path(__file__).resolve() != overlay/"scripts/continue_aids_rf_pairs.py":
            raise ValueError("Unrecognized phase-only code overlay")
        for relative, sha in config["phase_overlay_files"].items():
            path = overlay/relative
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError("Immutable phase driver overlay changed: "+relative)
        sys.path.insert(0, config["phase_execution_worktree"])
        import src.baselines.comrecgc
        src.baselines.comrecgc.__path__.insert(0,str(overlay/"src/baselines/comrecgc"))
    from src.baselines.comrecgc.rf_aligned_cluster_phase import sealed_pairs, phase_memory_plan, resource_sample, run_cluster_only, run_summary_only
    from src.baselines.comrecgc.rf_aligned_phase_owner import run_owner, validate_writer_fd
    from src.baselines.comrecgc.rf_aligned_pool import atomic_json
    root.mkdir(parents=True, exist_ok=True)
    if args.action == "plan":
        binding = sealed_pairs(config, recourse)
        plans = {phase: phase_memory_plan(binding, phase=phase) for phase in ("CERTIFIED_EXACT_DBSCAN", "NATIVE_SUMMARY", "RF_WNODE_RELEASE")}
        value = {"pair_adoption": binding, "phase_plans": plans,
                 "resource": resource_sample(config, plans["CERTIFIED_EXACT_DBSCAN"], running=False), "science_started": False}
        atomic_json(root/"plan.json", value); print(json.dumps(value, sort_keys=True)); return
    try:
        if args.action == "owner":
            run_owner(config, manifest=Path(args.run_manifest), recourse_root=recourse, pool_root=pool, output_root=root)
        else:
            if args.writer_fd is None: parser.error("science child requires the actual inherited original writer FD")
            validate_writer_fd(args.writer_fd, recourse)
            if args.action == "global-witness":
                from src.baselines.comrecgc.aids_global_witness import run_witness
                run_witness(config, recourse_root=recourse, evidence_root=root)
            elif args.action == "cluster-existing": run_cluster_only(config, recourse_root=recourse, evidence_root=root)
            else: run_summary_only(config, pool_root=pool, recourse_root=recourse, evidence_root=root)
    except Exception as exc:
        atomic_json(root/"failed.json", {"state": "FAILED_STAGE", "error_type": type(exc).__name__, "error": str(exc),
                                       "pair_chunks_recomputed": False, "automatic_retry": False})
        raise


if __name__ == "__main__": main()
