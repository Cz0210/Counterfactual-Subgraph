#!/usr/bin/env python3
"""Prepare or read-verify a future-only immutable T12 history cache; no science."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.t12_future_history_cache import load_cache, stage_closed_snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--action", choices=("prepare", "verify-read"), required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--producer-pid", type=int)
    parser.add_argument("--producer-start-ticks", type=int)
    parser.add_argument("--min-free-bytes", type=int)
    parser.add_argument("--min-free-inodes", type=int)
    parser.add_argument("--cache-manifest-sha256")
    parser.add_argument("--disposable-index-root", type=Path)
    args = parser.parse_args(argv)
    if not args.config.is_file() or args.set != ["inference.fallback_to_heuristic=false"]:
        parser.error("existing config and explicit fail-closed inference override required")
    snapshot = json.loads(args.snapshot.read_text())
    if args.action == "prepare":
        if any(v is None for v in (args.producer_pid, args.producer_start_ticks,
                                    args.min_free_bytes, args.min_free_inodes)):
            parser.error("prepare requires producer identity and unchanged local resource reserves")
        result = stage_closed_snapshot(snapshot, expected_snapshot_sha256=args.snapshot_sha256,
            cache_root=args.cache_root, producer_pid=args.producer_pid,
            producer_start_ticks=args.producer_start_ticks,
            min_free_bytes=args.min_free_bytes, min_free_inodes=args.min_free_inodes)
    else:
        if not args.cache_manifest_sha256 or args.disposable_index_root is None:
            parser.error("verify-read requires cache manifest SHA and fresh disposable index root")
        from src.baselines.tastemolnet_gcf_production_state import T12CompactHistoryJournal, T12ProductionBounds
        from src.utils.main_ready_task_specs import stable_sha256
        if stable_sha256(snapshot) != args.snapshot_sha256:
            raise ValueError("T12_CACHE_SNAPSHOT_BINDING_REQUIRED")
        if args.disposable_index_root.exists():
            raise ValueError("T12_CACHE_DISPOSABLE_INDEX_MUST_BE_FRESH")
        cache = load_cache(args.cache_root, expected_manifest_sha256=args.cache_manifest_sha256)
        started = time.monotonic()
        journal = T12CompactHistoryJournal(root=snapshot["history_root"],
            index_root=args.disposable_index_root, bounds=T12ProductionBounds.from_dict(snapshot["bounds"]),
            contract_sha256=snapshot["contract_sha256"], attempt_id=snapshot["attempt_id"],
            generation_token=snapshot["generation_token"], resume_snapshot=snapshot,
            open_writer=False, history_read_cache=cache)
        try:
            observed = journal.checkpoint_state()
            if observed != snapshot:
                raise ValueError("T12_CACHE_ORIGINAL_CODEC_STATE_CHANGED")
            result = dict(state="FUTURE_CACHE_CODEC_REPLAY_PASS_NOT_PRODUCTION_PARITY",
                original_snapshot_unchanged=True, observations=journal.observation_count,
                local_read_and_index_seconds=time.monotonic()-started,
                kernel_page_cache_state="UNCONTROLLED_NO_DROP_CACHES", science_started=False)
        finally:
            journal.close()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
