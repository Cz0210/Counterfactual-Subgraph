#!/usr/bin/env python3
"""Run or strictly consume the bounded TasteMolNet T8 GlobalGCE smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.globalgce_mutagenicity_adapter import (  # noqa: E402
    OfficialGlobalGCEMutagenicityGenerator,
)
from src.baselines.tastemolnet_globalgce_smoke import (  # noqa: E402
    DATASET,
    FrozenTasteGINEScorer,
    NUM_CLASSES,
    SOURCE_LABEL,
    STAGE,
    TasteGlobalGCESmokeConfig,
    TasteGlobalGCESmokeError,
    run_t8_science,
)
from src.utils.retained_output_directory import (  # noqa: E402
    FreshOutputDirectory,
)
from src.utils.managed_execution_v2 import (  # noqa: E402
    create_managed_attempt,
    create_worker_staging,
)
from src.utils.tastemolnet_t8_managed_v2 import (  # noqa: E402
    seal_t8_worker_evidence,
    t8_managed_config_hash,
    t8_managed_input_hashes,
)
from src.utils.tastemolnet_t8_globalgce_release import (  # noqa: E402
    TasteT8ReleaseDisabled,
    assert_execution_released,
    hold_tastemolnet_t8_inputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=(STAGE,), default=STAGE)
    parser.add_argument("--t2-adoption", type=Path)
    parser.add_argument("--t3-output", type=Path)
    parser.add_argument("--t4-output", type=Path)
    parser.add_argument("--gnn-checkpoint", type=Path)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--official-root", type=Path)
    parser.add_argument("--downstream-policy", type=Path)
    parser.add_argument("--base-policy", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    return parser


def _require_one_logical_cuda() -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - AutoDL dependency.
        raise TasteGlobalGCESmokeError("T8 requires PyTorch with CUDA") from exc
    if (
        not bool(torch.cuda.is_available())
        or int(torch.cuda.device_count()) != 1
        or int(torch.cuda.current_device()) != 0
    ):
        raise TasteGlobalGCESmokeError(
            "T8 requires exactly physical GPU2 exposed as logical cuda:0"
        )


def run(
    args: argparse.Namespace,
    *,
    managed_v2_authority: Any | None = None,
) -> int:
    assert_execution_released()
    required = (
        "t2_adoption",
        "t3_output",
        "t4_output",
        "gnn_checkpoint",
        "train_csv",
        "official_root",
        "downstream_policy",
        "base_policy",
        "state_dir",
    )
    missing = [field for field in required if getattr(args, field, None) is None]
    if missing:
        raise TasteGlobalGCESmokeError(
            "T8 run is missing required inputs: " + ", ".join(missing)
        )
    config = TasteGlobalGCESmokeConfig()
    config.validate()
    inputs = hold_tastemolnet_t8_inputs(
        output_dir=args.output_dir,
        state_dir=args.state_dir,
        config_path=args.config,
        t2_adoption=args.t2_adoption,
        t3_output=args.t3_output,
        t4_output=args.t4_output,
        checkpoint_dir=args.gnn_checkpoint,
        train_csv=args.train_csv,
        official_root=args.official_root,
        downstream_policy=args.downstream_policy,
        base_policy=args.base_policy,
        managed_v2_authority=managed_v2_authority,
    )
    state_root: FreshOutputDirectory | None = None
    state_tree: Any | None = None
    attempt: Any | None = None
    staging: Any | None = None
    try:
        _require_one_logical_cuda()
        authority = inputs.terminal_authority()
        attempt = create_managed_attempt(
            stage_root=inputs.state_root.parent,
            controller_id=authority["managed_execution"]["run_id"],
            task_id=authority["managed_execution"]["task_id"],
            git_commit=authority["execution"]["commit"],
            config_hash=t8_managed_config_hash(),
            input_hashes=t8_managed_input_hashes(authority),
        )
        staging = create_worker_staging(attempt)
        state_root = FreshOutputDirectory.create(inputs.state_root)
        scorer = FrozenTasteGINEScorer(
            inputs.checkpoint_payloads,
            device="cuda:0",
            batch_size=config.oracle_batch_size,
        )
        if scorer.checkpoint_id != inputs.checkpoint_evidence["checkpoint_id"]:
            raise TasteGlobalGCESmokeError(
                "T8 scorer differs from the held frozen checkpoint"
            )

        def generator_factory(target_label: int) -> OfficialGlobalGCEMutagenicityGenerator:
            return OfficialGlobalGCEMutagenicityGenerator(
                inputs.official.runtime_root,
                native_train_csv=inputs.train.path,
                dataset_name=DATASET,
                min_freq=config.min_freq,
                frozen_gine_checkpoint=Path(
                    f"/proc/self/fd/{inputs.checkpoint.directory.descriptor}"
                ),
                source_label=SOURCE_LABEL,
                target_label=target_label,
                num_classes=NUM_CLASSES,
                frozen_gine_payloads=inputs.checkpoint_payloads,
                native_train_payload=inputs.train_bytes,
                official_source_authority=inputs.official.import_authority(),
                require_isolated_imports=True,
            )

        science, state_tree = run_t8_science(
            train_payload=inputs.train_bytes,
            expected_train_row_count=inputs.train_contract["row_count"],
            expected_train_label_counts=inputs.train_contract["label_counts"],
            scorer=scorer,
            generator_factory=generator_factory,
            state_root=state_root,
            config=config,
        )
        inputs.revalidate()
        if inputs.terminal_authority() != authority:
            raise TasteGlobalGCESmokeError(
                "T8 input authority changed before worker SEALED"
            )
        sealed = seal_t8_worker_evidence(
            staging,
            science=science,
            state_tree=state_tree,
            input_authority=authority,
            expected_final_path=inputs.output_root,
        )
        print(
            json.dumps(
                {
                    "state": "SEALED",
                    "attempt_id": sealed.attempt_id,
                    "generation_token": sealed.generation_token,
                    "sealed_path": str(sealed.seal_path),
                    "sealed_sha256": sealed.seal_sha256,
                    "inventory_sha256": sealed.inventory_sha256,
                    "independent_t8_verification_required": True,
                    "worker_wrote_pass": False,
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        # A worker never publishes PASS, so ordinary best-effort close is safe.
        for retained in (state_tree, state_root, staging, attempt):
            if retained is None:
                continue
            try:
                retained.close()
            except BaseException:
                pass
        inputs.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_only:
        raise TasteGlobalGCESmokeError(
            "T8 validate-only requires an injected independent managed-v2 "
            "verifier authority adapter and remains release-disabled"
        )
    try:
        # This tracked gate runs before reading any caller-selected science
        # path.  run() repeats it at the execution boundary.
        assert_execution_released()
    except TasteT8ReleaseDisabled as exc:
        print(str(exc), file=sys.stderr)
        return 78
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise TasteGlobalGCESmokeError(
            "T8 requires exactly --set inference.fallback_to_heuristic=false"
        )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
