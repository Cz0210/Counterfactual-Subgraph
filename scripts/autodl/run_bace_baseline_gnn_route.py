#!/usr/bin/env python3
"""Foreground stages for BACE native baselines against one frozen GINE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.bace_gnn_baseline_contracts import (  # noqa: E402
    baseline_spec,
    write_route_preflight,
)
from src.baselines.bace_gnn_baseline_tasks import (  # noqa: E402
    build_bace_baseline_controller_fragment,
)
from src.baselines.bace_gnn_baseline_generic_adapter import (  # noqa: E402
    atomic_write_generic_fragment,
    build_bace_baseline_generic_controller_fragment,
)
from src.baselines.comrecgc.exporter import (  # noqa: E402
    export_bace_gine_representatives,
)
from src.baselines.gcfexplainer_bace_runtime import (  # noqa: E402
    export_bace_gine_candidate_universe,
)
from src.baselines.globalgce_bace_adapter import (  # noqa: E402
    EXPECTED_TRAIN_SOURCE_COUNT,
    build_bace_frozen_gine_rule_pool,
)
from src.baselines.globalgce_mutagenicity_adapter import PoolBuildConfig  # noqa: E402
from src.baselines.globalgce_min_freq import resolve_globalgce_min_freq  # noqa: E402
from src.eval.bace_native_baseline_gnn import (  # noqa: E402
    freeze_native_baseline_final,
    merge_fullgraph_verification_shards,
    run_fullgraph_verification_shard,
    run_native_baseline_selector,
)
from src.eval.bace_globalgce_native_gine import (  # noqa: E402
    run_native_gine_forward_canary,
)
from src.eval.bace_globalgce_frozen_gine_bridge import (  # noqa: E402
    run_frozen_gine_bridge_smoke,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", required=True)
    parser.add_argument("--gnn-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)


def _fragment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gnn-checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--calibration-split", required=True)
    parser.add_argument("--test-split", required=True)
    parser.add_argument("--molclr-root", required=True)
    parser.add_argument("--molclr-checkpoint", required=True)
    parser.add_argument("--neurosed-checkpoint", required=True)
    parser.add_argument("--official-root")
    parser.add_argument("--neurosed-manifest")
    parser.add_argument("--globalgce-source-manifest")
    parser.add_argument("--globalgce-native-train-csv")
    parser.add_argument("--omp-threads", type=int, default=4)


def _fragment_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "method": args.method,
        "python": args.python,
        "project_root": args.project_root,
        "output_root": args.output_dir,
        "gnn_checkpoint": args.gnn_checkpoint,
        "dataset_dir": args.dataset_dir,
        "calibration_split": args.calibration_split,
        "test_split": args.test_split,
        "molclr_root": args.molclr_root,
        "molclr_checkpoint": args.molclr_checkpoint,
        "neurosed_checkpoint": args.neurosed_checkpoint,
        "official_root": args.official_root,
        "neurosed_manifest": args.neurosed_manifest,
        "globalgce_source_manifest": args.globalgce_source_manifest,
        "globalgce_native_train_csv": args.globalgce_native_train_csv,
        "omp_threads": args.omp_threads,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[])
    sub = parser.add_subparsers(dest="stage", required=True)

    describe = sub.add_parser("describe")
    describe.add_argument("--method", required=True)

    fragment = sub.add_parser("task-fragment")
    _fragment_arguments(fragment)

    generic_fragment = sub.add_parser("generic-task-fragment")
    _fragment_arguments(generic_fragment)
    generic_fragment.add_argument(
        "--fragment-output",
        required=True,
        help="Fresh absolute JSON path consumed by build_four_by_four_manifest.py",
    )

    preflight = sub.add_parser("preflight")
    _common(preflight)
    preflight.add_argument("--official-root")

    globalgce_forward = sub.add_parser("globalgce-forward-canary")
    _common(globalgce_forward)
    globalgce_forward.add_argument("--rule-json", required=True)
    globalgce_forward.add_argument("--parent-id", required=True)
    globalgce_forward.add_argument("--parent-smiles", required=True)
    globalgce_forward.add_argument("--device", default="cpu")
    globalgce_forward.add_argument("--oracle-batch-size", type=int, default=256)

    globalgce_bridge = sub.add_parser("globalgce-bridge-smoke")
    _common(globalgce_bridge)
    globalgce_bridge.add_argument("--parent-smiles", required=True)
    globalgce_bridge.add_argument("--atom-symbol", action="append", required=True)
    globalgce_bridge.add_argument(
        "--bond-name",
        action="append",
        default=None,
    )
    globalgce_bridge.add_argument("--device", default="cuda:0")

    globalgce_train = sub.add_parser("globalgce-train-rules")
    _common(globalgce_train)
    globalgce_train.add_argument("--source-manifest", required=True)
    globalgce_train.add_argument("--native-train-csv", required=True)
    globalgce_train.add_argument("--official-root", required=True)
    globalgce_train.add_argument("--expected-parent-count", type=int, default=360)
    globalgce_train.add_argument("--seed", type=int, default=13)
    globalgce_train.add_argument("--epochs", type=int, default=100)
    globalgce_train.add_argument("--top-k-native", type=int, default=20)
    globalgce_train.add_argument("--learning-rate", type=float, default=0.1)
    globalgce_train.add_argument("--dropout", type=float, default=0.5)
    globalgce_train.add_argument("--device", default="cuda:0")
    globalgce_train.add_argument("--min-freq", type=int, default=None)
    globalgce_train.add_argument("--min-freq-manifest")
    globalgce_train.add_argument("--gspan-flush-every", type=int, default=256)
    globalgce_train.add_argument(
        "--gspan-max-in-memory-candidates", type=int, default=256
    )
    globalgce_train.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )

    gcf = sub.add_parser("gcf-export")
    _common(gcf)
    gcf.add_argument("--dataset-dir", required=True)
    gcf.add_argument("--summary-dir", required=True)
    gcf.add_argument("--profile", choices=("smoke", "full"), required=True)
    gcf.add_argument("--parent-limit", type=int, required=True)
    gcf.add_argument("--minimum-candidates", type=int, default=20)
    gcf.add_argument("--scan-limit", type=int, default=0)
    gcf.add_argument("--device", default="cuda:0")
    gcf.add_argument("--oracle-batch-size", type=int, default=256)

    comrecgc = sub.add_parser("comrecgc-export")
    _common(comrecgc)
    comrecgc.add_argument("--common-recourse-dir", required=True)
    comrecgc.add_argument("--dataset-summary-json", required=True)
    comrecgc.add_argument("--minimum-candidates", type=int, default=20)
    comrecgc.add_argument("--device", default="cuda:0")
    comrecgc.add_argument("--oracle-batch-size", type=int, default=256)

    verify = sub.add_parser("verify-shard")
    _common(verify)
    verify.add_argument(
        "--verification-stage",
        choices=("BASELINE_CALIBRATION_VERIFY", "BASELINE_TEST_EVAL"),
        required=True,
    )
    verify.add_argument("--split-path", required=True)
    verify.add_argument("--predecessor-output", required=True)
    verify.add_argument("--molclr-root", required=True)
    verify.add_argument("--molclr-checkpoint", required=True)
    verify.add_argument("--shard-index", type=int, choices=range(4), required=True)
    verify.add_argument("--wnode-cache-db", required=True)
    verify.add_argument("--node-embedding-cache-dir", required=True)
    verify.add_argument("--device", default="cuda:0")
    verify.add_argument("--oracle-batch-size", type=int, default=256)

    merge = sub.add_parser("merge")
    merge.add_argument("--method", required=True)
    merge.add_argument(
        "--verification-stage",
        choices=("BASELINE_CALIBRATION_VERIFY", "BASELINE_TEST_EVAL"),
        required=True,
    )
    merge.add_argument("--shard-dir", action="append", required=True)
    merge.add_argument("--predecessor-output", required=True)
    merge.add_argument("--output-dir", required=True)

    select = sub.add_parser("select")
    select.add_argument("--method", required=True)
    select.add_argument("--matrix-output", required=True)
    select.add_argument("--output-dir", required=True)
    select.add_argument("--seed", type=int, default=13)

    freeze = sub.add_parser("freeze")
    freeze.add_argument("--method", required=True)
    freeze.add_argument("--selection-output", required=True)
    freeze.add_argument("--test-output", required=True)
    freeze.add_argument("--output-dir", required=True)
    return parser


def _atom_vocabulary(path: str | Path) -> list[str | int]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    values = payload.get("feature_atomic_numbers") if isinstance(payload, dict) else payload
    if isinstance(values, dict):
        return [key for key, _value in sorted(values.items(), key=lambda item: int(item[1]))]
    if isinstance(values, list):
        return values
    raise ValueError("dataset summary has no feature_atomic_numbers list/mapping")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "describe":
        spec = baseline_spec(args.method)
        result = {
            "method": spec.method,
            "method_id": spec.method_id,
            "route_available": spec.native_route_available,
            "action_kind": spec.action_kind,
            "action_semantics": spec.action_semantics,
            "resources": {
                "generation": spec.generation_resource,
                "verification": spec.verification_resource,
                "selection": spec.selector_resource,
            },
            "fresh_output_required": True,
            "terminal_markers": ["PASS", "BLOCKED", "BLOCKED_CODE"],
            "blocker_code": spec.blocker_code,
            "blocker_reason": spec.blocker_reason,
        }
    elif args.stage == "task-fragment":
        result = build_bace_baseline_controller_fragment(**_fragment_kwargs(args))
    elif args.stage == "generic-task-fragment":
        fragment = build_bace_baseline_generic_controller_fragment(
            **_fragment_kwargs(args)
        )
        destination = atomic_write_generic_fragment(args.fragment_output, fragment)
        result = {
            "status": "PASS",
            "schema_version": fragment["schema_version"],
            "method": fragment["method"],
            "method_id": fragment["method_id"],
            "task_count": len(fragment["tasks"]),
            "fragment_output": str(destination),
        }
    elif args.stage == "preflight":
        result = write_route_preflight(
            method=args.method,
            checkpoint_dir=args.gnn_checkpoint,
            output_dir=args.output_dir,
            official_root=args.official_root,
        )
    elif args.stage == "globalgce-forward-canary":
        if baseline_spec(args.method).method_id != "globalgce":
            raise ValueError("globalgce-forward-canary requires method=GlobalGCE")
        result = run_native_gine_forward_canary(
            parent_id=args.parent_id,
            parent_smiles=args.parent_smiles,
            rule_json=args.rule_json,
            gnn_checkpoint=args.gnn_checkpoint,
            output_dir=args.output_dir,
            device=args.device,
            oracle_batch_size=args.oracle_batch_size,
        )
    elif args.stage == "globalgce-bridge-smoke":
        if baseline_spec(args.method).method_id != "globalgce":
            raise ValueError("globalgce-bridge-smoke requires method=GlobalGCE")
        result = run_frozen_gine_bridge_smoke(
            gnn_checkpoint=args.gnn_checkpoint,
            parent_smiles=args.parent_smiles,
            atom_symbols=tuple(args.atom_symbol),
            bond_names=tuple(
                args.bond_name or ("no_edge", "single", "double", "triple")
            ),
            output_dir=args.output_dir,
            device=args.device,
        )
    elif args.stage == "globalgce-train-rules":
        if baseline_spec(args.method).method_id != "globalgce":
            raise ValueError("globalgce-train-rules requires method=GlobalGCE")
        if int(args.expected_parent_count) != EXPECTED_TRAIN_SOURCE_COUNT:
            raise ValueError("GlobalGCE full route requires all 360 BACE train parents")
        minimum_frequency = resolve_globalgce_min_freq(
            "BACE",
            explicit_min_freq=args.min_freq,
            calibration_manifest=args.min_freq_manifest,
        )
        result = build_bace_frozen_gine_rule_pool(
            source_manifest=args.source_manifest,
            native_train_csv=args.native_train_csv,
            official_root=args.official_root,
            gnn_checkpoint=args.gnn_checkpoint,
            output_dir=args.output_dir,
            min_freq=minimum_frequency.value,
            config=PoolBuildConfig(
                expected_parent_count=int(args.expected_parent_count),
                seed=int(args.seed),
                epochs=int(args.epochs),
                top_k_native=int(args.top_k_native),
                learning_rate=float(args.learning_rate),
                dropout=float(args.dropout),
                device=str(args.device),
                resume=bool(args.resume),
                forbid_calibration_test=True,
                gspan_flush_every=int(args.gspan_flush_every),
                gspan_max_in_memory_candidates=int(
                    args.gspan_max_in_memory_candidates
                ),
            ),
        )
    elif args.stage == "gcf-export":
        if baseline_spec(args.method).method_id != "gcfexplainer":
            raise ValueError("gcf-export requires method=GCFExplainer")
        result = export_bace_gine_candidate_universe(
            dataset_dir=args.dataset_dir,
            summary_dir=args.summary_dir,
            gnn_checkpoint=args.gnn_checkpoint,
            output_dir=args.output_dir,
            profile=args.profile,
            parent_limit=args.parent_limit,
            minimum_candidates=args.minimum_candidates,
            scan_limit=args.scan_limit,
            device=args.device,
            oracle_batch_size=args.oracle_batch_size,
        )
    elif args.stage == "comrecgc-export":
        if baseline_spec(args.method).method_id != "comrecgc":
            raise ValueError("comrecgc-export requires method=ComRecGC")
        result = export_bace_gine_representatives(
            common_recourse_dir=args.common_recourse_dir,
            gnn_checkpoint=args.gnn_checkpoint,
            atom_vocabulary=_atom_vocabulary(args.dataset_summary_json),
            output_dir=args.output_dir,
            minimum_candidates=args.minimum_candidates,
            device=args.device,
            oracle_batch_size=args.oracle_batch_size,
        )
    elif args.stage == "verify-shard":
        result = run_fullgraph_verification_shard(
            method=args.method,
            stage=args.verification_stage,
            split_path=args.split_path,
            predecessor_output=args.predecessor_output,
            gnn_checkpoint=args.gnn_checkpoint,
            molclr_root=args.molclr_root,
            molclr_checkpoint=args.molclr_checkpoint,
            output_dir=args.output_dir,
            shard_index=args.shard_index,
            wnode_cache_db=args.wnode_cache_db,
            node_embedding_cache_dir=args.node_embedding_cache_dir,
            device=args.device,
            oracle_batch_size=args.oracle_batch_size,
        )
    elif args.stage == "merge":
        result = merge_fullgraph_verification_shards(
            method=args.method,
            stage=args.verification_stage,
            shard_dirs=args.shard_dir,
            predecessor_output=args.predecessor_output,
            output_dir=args.output_dir,
        )
    elif args.stage == "select":
        result = run_native_baseline_selector(
            method=args.method,
            matrix_output=args.matrix_output,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    elif args.stage == "freeze":
        result = freeze_native_baseline_final(
            method=args.method,
            selection_output=args.selection_output,
            test_output=args.test_output,
            output_dir=args.output_dir,
        )
    else:  # pragma: no cover - argparse makes this unreachable.
        raise AssertionError(args.stage)
    print(json.dumps(result, sort_keys=True), flush=True)
    # The controller's PASS-last contract checks a human-readable log marker in
    # addition to the on-disk BRIDGE_PASS sentinel.  Keep that evidence in the
    # thin CLI rather than making the reusable evaluator print to stdout.
    if args.stage == "globalgce-bridge-smoke":
        print("[BACE_GLOBALGCE_BRIDGE_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
