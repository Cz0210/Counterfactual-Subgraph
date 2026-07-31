#!/usr/bin/env python3
"""Train CLEAR on strict train/val and build a 64-parent train-only pool."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.clear_mutagenicity_adapter import (  # noqa: E402
    ATOM_SIDECAR_SCHEMA_VERSION,
    GENERATED_CODEC_VERSION,
    feature_decoding_schema_summary,
    load_strict_cohort,
)
from src.baselines.clear_mutagenicity_train_pool import (  # noqa: E402
    ClearMutagenicityEmptyPoolError,
    EXPECTED_GENERATION_PARENT_ROWS,
    EXPECTED_MODEL_TRAIN_ROWS,
    EXPECTED_MODEL_VAL_ROWS,
    GeneratedGraph,
    TrainPoolConfig,
    audit_train_pool,
    cohort_hash,
    run_streaming_generation,
    schema_from_mapping,
    select_generation_parents,
    sha256_file,
    validate_phase_a_data,
    validate_phase_a_splits,
    write_json,
)
from src.rewards.teacher_semantic import TeacherSemanticScorer  # noqa: E402


DEFAULT_PHASE_A_ROOT = (
    "outputs/hpc/mutagenicity/final/clear_phase_a_dataset_codec_best"
)
DEFAULT_GENERATION_CSV = (
    "outputs/hpc/datasets/mutagenicity_v1_teacher_consistent/"
    "train_source_label1_teacher_correct.csv"
)
DEFAULT_TEACHER = (
    "outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phase-a-root", default=DEFAULT_PHASE_A_ROOT)
    parser.add_argument("--generation-csv", default=DEFAULT_GENERATION_CSV)
    parser.add_argument("--teacher-path", default=DEFAULT_TEACHER)
    parser.add_argument(
        "--official-root", default="baselines/clear_official"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-limit", type=int, default=64)
    parser.add_argument("--graphpred-epochs", type=int, default=5)
    parser.add_argument("--cfe-epochs", type=int, default=5)
    parser.add_argument("--generation-chunk-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--graphpred-learning-rate", type=float, default=0.001)
    parser.add_argument("--cfe-learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--graphpred-checkpoint", default=None)
    parser.add_argument("--graphcfe-checkpoint", default=None)
    parser.add_argument("--source-run-root", default=None)
    parser.add_argument(
        "--forbid-calibration-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    return parser


def _required_file(path_like: str | Path, label: str) -> Path:
    path = Path(path_like).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")
    return path


def _required_dir(path_like: str | Path, label: str) -> Path:
    path = Path(path_like).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _phase_a_summary_path(root: Path) -> Path:
    candidates = (
        root / "dataset_summary.json",
        root / "dataset_v1" / "dataset_summary.json",
        root / "summary" / "dataset_summary.json",
    )
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(
        "Frozen Phase A root does not contain dataset_summary.json: "
        f"{root}"
    )


def _load_phase_a_data(official_root: Path) -> tuple[Any, dict[str, Any]]:
    official_src = official_root / "src"
    if str(official_src) not in sys.path:
        sys.path.insert(0, str(official_src))
    full_pickle = _required_file(
        official_root / "dataset" / "mutagenicity_full.pickle",
        "Phase A Mutagenicity full pickle",
    )
    split_pickle = _required_file(
        official_root / "dataset" / "mutagenicity_datasplit.pickle",
        "Phase A Mutagenicity split pickle",
    )
    with full_pickle.open("rb") as handle:
        payload = pickle.load(handle)
    with split_pickle.open("rb") as handle:
        split_payload = pickle.load(handle)
    data = payload["data"]
    return data, {
        "full_pickle": str(full_pickle),
        "full_pickle_sha256": sha256_file(full_pickle),
        "split_pickle": str(split_pickle),
        "split_pickle_sha256": sha256_file(split_pickle),
        "split_payload": split_payload,
    }


def _run_logged(
    command: Sequence[str], *, cwd: Path, log_path: Path
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(command))


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unknown"


def _config_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _apply_official_patches() -> None:
    subprocess.run(
        ["bash", "scripts/baselines/clear/apply_clear_patches.sh"],
        cwd=REPO_ROOT,
        check=True,
    )


def _train_official_models(
    *,
    official_root: Path,
    output_dir: Path,
    config: TrainPoolConfig,
    graphpred_learning_rate: float,
    cfe_learning_rate: float,
    dropout: float,
) -> tuple[Path, Path]:
    source_dir = official_root / "src"
    checkpoint_dir = output_dir / "checkpoints"
    graphpred_checkpoint = (
        checkpoint_dir / "prediction" / "weights_graphPred__mutagenicity.pt"
    )
    cfe_checkpoint = (
        checkpoint_dir
        / (
            "weights_graphCFE_CLEAR_mutagenicity_exp0_epoch"
            f"{config.cfe_epochs}.pt"
        )
    )
    if not graphpred_checkpoint.is_file():
        _run_logged(
            (
                sys.executable,
                "train_pred.py",
                "--dataset",
                "mutagenicity",
                "--epochs",
                str(config.graphpred_epochs),
                "--lr",
                str(graphpred_learning_rate),
                "--dropout",
                str(dropout),
                "--batch_size",
                str(config.batch_size),
                "--num_workers",
                str(config.num_workers),
                "--seed",
                str(config.seed),
                "--model_dir",
                str(checkpoint_dir),
            ),
            cwd=source_dir,
            log_path=output_dir / "graphpred" / "train.log",
        )
    if not graphpred_checkpoint.is_file():
        raise FileNotFoundError(
            f"Official graph predictor checkpoint missing: {graphpred_checkpoint}"
        )
    print("[MUTAGENICITY_CLEAR_GRAPHPRED_SMOKE_OK]", flush=True)

    if not cfe_checkpoint.is_file():
        _run_logged(
            (
                sys.executable,
                "main.py",
                "--dataset",
                "mutagenicity",
                "--experiment_type",
                "train",
                "--epochs",
                str(config.cfe_epochs),
                "--lr",
                str(cfe_learning_rate),
                "--dropout",
                str(dropout),
                "--batch_size",
                str(config.batch_size),
                "--num_workers",
                str(config.num_workers),
                "--seed",
                str(config.seed),
                "--model_dir",
                str(checkpoint_dir),
                "--num_experiments",
                "1",
            ),
            cwd=source_dir,
            log_path=output_dir / "graphcfe" / "train.log",
        )
    if not cfe_checkpoint.is_file():
        raise FileNotFoundError(
            f"Official GraphCFE checkpoint missing: {cfe_checkpoint}"
        )
    print("[MUTAGENICITY_CLEAR_GRAPHCFE_SMOKE_OK]", flush=True)
    return graphpred_checkpoint, cfe_checkpoint


def _resolve_generation_checkpoints(
    *,
    generation_only: bool,
    graphpred_checkpoint: str | Path | None,
    graphcfe_checkpoint: str | Path | None,
    source_run_root: str | Path | None,
    train_models: Callable[[], tuple[Path, Path]],
) -> tuple[Path, Path, Path | None, bool]:
    """Resolve checkpoints without ever guessing a replay checkpoint."""

    if not generation_only:
        if any(
            value is not None
            for value in (
                graphpred_checkpoint,
                graphcfe_checkpoint,
                source_run_root,
            )
        ):
            raise ValueError(
                "Explicit checkpoint/source-run options require "
                "--generation-only."
            )
        graphpred, graphcfe = train_models()
        return graphpred.resolve(), graphcfe.resolve(), None, True

    if source_run_root is None:
        raise ValueError("--source-run-root is required for generation-only.")
    source_root = _required_dir(source_run_root, "source failed run root")
    if graphpred_checkpoint is None or graphcfe_checkpoint is None:
        raise ValueError(
            "--graphpred-checkpoint and --graphcfe-checkpoint are both "
            "required for generation-only."
        )
    graphpred = _required_file(
        graphpred_checkpoint, "explicit GraphPred checkpoint"
    )
    graphcfe = _required_file(
        graphcfe_checkpoint, "explicit GraphCFE checkpoint"
    )
    for checkpoint in (graphpred, graphcfe):
        try:
            checkpoint.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(
                "Generation-only checkpoints must be inside the explicit "
                f"source run root: checkpoint={checkpoint}, "
                f"source_run_root={source_root}"
            ) from exc
    return graphpred, graphcfe, source_root, False


def _checkpoint_provenance(
    graphpred_checkpoint: Path,
    graphcfe_checkpoint: Path,
) -> dict[str, Any]:
    return {
        "graphpred_checkpoint_path": str(graphpred_checkpoint.resolve()),
        "graphpred_checkpoint_sha256": sha256_file(graphpred_checkpoint),
        "graphcfe_checkpoint_path": str(graphcfe_checkpoint.resolve()),
        "graphcfe_checkpoint_sha256": sha256_file(graphcfe_checkpoint),
    }


def _load_generation_models(
    *,
    official_root: Path,
    data: Any,
    graphpred_checkpoint: Path,
    cfe_checkpoint: Path,
    config: TrainPoolConfig,
    dropout: float,
) -> tuple[Any, Any, Any]:
    import torch

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CLEAR Phase B smoke.")
    source_dir = official_root / "src"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    import models

    device = torch.device(config.device)
    x_dim = int(data.feature_all[0].shape[1])
    num_class = len(
        np.unique(np.asarray(data.labels_all, dtype=np.int64).reshape(-1))
    )
    graphpred = models.Graph_pred_model(
        x_dim, 32, num_class, int(data.max_num_nodes), "mutagenicity"
    ).to(device)
    graphpred.load_state_dict(
        torch.load(str(graphpred_checkpoint), map_location=device)
    )
    graphpred.eval()
    cfe_args = SimpleNamespace(
        dim_h=16,
        dim_z=16,
        dropout=float(dropout),
        disable_u=0,
    )
    cfe = models.GraphCFE(
        init_params={
            "vae_type": "graphVAE",
            "x_dim": x_dim,
            "u_dim": 1,
            "max_num_nodes": int(data.max_num_nodes),
        },
        args=cfe_args,
    ).to(device)
    cfe.load_state_dict(torch.load(str(cfe_checkpoint), map_location=device))
    cfe.eval()
    return torch, graphpred, cfe


def _generation_callable(
    *,
    torch_module: Any,
    graphpred: Any,
    cfe: Any,
    data: Any,
    device: str,
):
    torch = torch_module
    target_device = torch.device(device)

    def generate(
        parents: Sequence[Any],
        indices: Sequence[int],
        _chunk_index: int,
    ) -> list[GeneratedGraph]:
        features = torch.as_tensor(
            np.stack([data.feature_all[index] for index in indices]),
            dtype=torch.float32,
            device=target_device,
        )
        adjacency = torch.as_tensor(
            np.stack([data.adj_all[index] for index in indices]),
            dtype=torch.float32,
            device=target_device,
        )
        u_values = torch.as_tensor(
            np.stack([data.u_all[index] for index in indices]),
            dtype=torch.float32,
            device=target_device,
        )
        target = torch.zeros(
            (len(indices), 1), dtype=torch.float32, device=target_device
        )
        with torch.no_grad():
            generated = cfe(features, u_values, adjacency, target)
            generated_features = generated["features_reconst"]
            generated_adjacency = torch.where(
                generated["adj_reconst"] > 0.5,
                torch.ones_like(generated["adj_reconst"]),
                torch.zeros_like(generated["adj_reconst"]),
            )
            before_logits = graphpred(features, adjacency)["y_pred"]
            after_logits = graphpred(
                generated_features, generated_adjacency
            )["y_pred"]
            before_prob = torch.softmax(before_logits, dim=-1)
            after_prob = torch.softmax(after_logits, dim=-1)
            before_pred = before_prob.argmax(dim=-1)
            after_pred = after_prob.argmax(dim=-1)
            generated_features_np = (
                generated_features.detach().cpu().numpy()
            )
            generated_adjacency_np = (
                generated_adjacency.detach().cpu().numpy()
            )
            before_prob_np = before_prob.detach().cpu().numpy()
            after_prob_np = after_prob.detach().cpu().numpy()
            before_pred_np = before_pred.detach().cpu().numpy()
            after_pred_np = after_pred.detach().cpu().numpy()
        rows = [
            GeneratedGraph(
                parent_id=parent.molecule_id,
                features=generated_features_np[index],
                adjacency=generated_adjacency_np[index],
                official_pred_before=int(before_pred_np[index]),
                official_pred_after=int(after_pred_np[index]),
                official_prob_before=tuple(
                    float(value) for value in before_prob_np[index]
                ),
                official_prob_after=tuple(
                    float(value) for value in after_prob_np[index]
                ),
                generator_rank=1,
            )
            for index, parent in enumerate(parents)
        ]
        del (
            features,
            adjacency,
            u_values,
            target,
            generated,
            generated_features,
            generated_adjacency,
            before_logits,
            after_logits,
            before_prob,
            after_prob,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return rows

    return generate


def _prepare_output(
    output_dir: Path,
    *,
    resume: bool,
    fingerprint: str,
    manifest: dict[str, Any],
    create_training_directories: bool = True,
) -> None:
    if (output_dir / "_RUN_COMPLETE.json").is_file():
        raise FileExistsError(
            f"Completed CLEAR Phase B run cannot be overwritten: {output_dir}"
        )
    if (output_dir / "_RUN_FAILED.json").is_file():
        raise FileExistsError(
            "Failed CLEAR replay is immutable; choose a new output directory: "
            f"{output_dir}"
        )
    existing = output_dir.exists() and any(output_dir.iterdir())
    if existing and not resume:
        raise FileExistsError(
            f"Output directory is non-empty and resume is disabled: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if create_training_directories:
        for directory in ("graphpred", "graphcfe", "checkpoints"):
            (output_dir / directory).mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    if existing:
        if not manifest_path.is_file():
            raise ValueError("Resume requires run_manifest.json.")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config_fingerprint") != fingerprint:
            raise ValueError("Resume configuration mismatch.")
    else:
        write_json(manifest_path, manifest)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.forbid_calibration_test:
        raise ValueError("Phase B requires --forbid-calibration-test.")
    config = TrainPoolConfig(
        parent_limit=int(args.parent_limit),
        graphpred_epochs=int(args.graphpred_epochs),
        cfe_epochs=int(args.cfe_epochs),
        generation_chunk_size=int(args.generation_chunk_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        device=str(args.device),
        resume=bool(args.resume),
    )
    config.validate_smoke()
    phase_a_root = _required_dir(args.phase_a_root, "frozen Phase A root")
    phase_a_summary_path = _phase_a_summary_path(phase_a_root)
    phase_a_summary = json.loads(
        phase_a_summary_path.read_text(encoding="utf-8")
    )
    if phase_a_summary.get("atom_sidecar_schema_version") != (
        ATOM_SIDECAR_SCHEMA_VERSION
    ):
        raise ValueError("Frozen Phase A root is not sidecar schema v2.")
    if int(phase_a_summary.get("max_num_nodes", -1)) != 99:
        raise ValueError("Frozen Phase A max_num_nodes must be 99.")
    generation_csv = _required_file(
        args.generation_csv, "strict train source generation CSV"
    )
    teacher_path = _required_file(
        args.teacher_path, "Mutagenicity RF teacher"
    )
    official_root = _required_dir(args.official_root, "official CLEAR root")
    replay_paths = (
        args.source_run_root,
        args.graphpred_checkpoint,
        args.graphcfe_checkpoint,
    )
    if any(
        token in str(path).lower()
        for path in (
            generation_csv,
            teacher_path,
            phase_a_root,
            *(path for path in replay_paths if path is not None),
        )
        for token in ("calibration", "test_source", "test_target")
    ):
        raise ValueError("Phase B input path contains calibration/test token.")
    if not args.generation_only:
        _apply_official_patches()
    data, phase_a_pickle_info = _load_phase_a_data(official_root)
    data_audit = validate_phase_a_data(
        data,
        expected_train_rows=EXPECTED_MODEL_TRAIN_ROWS,
        expected_val_rows=EXPECTED_MODEL_VAL_ROWS,
    )
    data_audit.update(
        validate_phase_a_splits(
            data, dict(phase_a_pickle_info["split_payload"])
        )
    )
    parents = load_strict_cohort(
        generation_csv,
        expected_split="train",
        expected_label=1,
        expected_rows=EXPECTED_GENERATION_PARENT_ROWS,
    )
    selected = select_generation_parents(parents, config.parent_limit)
    output_dir = Path(args.output_dir).expanduser().resolve()
    replay_graphpred: Path | None = None
    replay_graphcfe: Path | None = None
    source_run_root: Path | None = None
    if args.generation_only:
        replay_graphpred, replay_graphcfe, source_run_root, _ = (
            _resolve_generation_checkpoints(
                generation_only=True,
                graphpred_checkpoint=args.graphpred_checkpoint,
                graphcfe_checkpoint=args.graphcfe_checkpoint,
                source_run_root=args.source_run_root,
                train_models=lambda: (_ for _ in ()).throw(
                    AssertionError("generation-only must not train models")
                ),
            )
        )
        try:
            output_dir.relative_to(source_run_root)
            output_overlaps_source = True
        except ValueError:
            output_overlaps_source = False
        if output_overlaps_source:
            raise ValueError(
                "Generation replay output must be outside the source failed run."
            )
    replay_checkpoint_identity = (
        _checkpoint_provenance(replay_graphpred, replay_graphcfe)
        if replay_graphpred is not None and replay_graphcfe is not None
        else {}
    )
    identity = {
        "config": config.identity(),
        "generation_only": bool(args.generation_only),
        "source_failed_run_root": (
            str(source_run_root) if source_run_root is not None else None
        ),
        "codec_version": GENERATED_CODEC_VERSION,
        "phase_a_summary_sha256": sha256_file(phase_a_summary_path),
        "phase_a_full_pickle_sha256": phase_a_pickle_info[
            "full_pickle_sha256"
        ],
        "phase_a_split_pickle_sha256": phase_a_pickle_info[
            "split_pickle_sha256"
        ],
        "generation_csv_sha256": sha256_file(generation_csv),
        "teacher_sha256": sha256_file(teacher_path),
        "selected_parent_hash": cohort_hash(selected),
        "graphpred_learning_rate": float(args.graphpred_learning_rate),
        "cfe_learning_rate": float(args.cfe_learning_rate),
        "dropout": float(args.dropout),
        "explicit_replay_checkpoints": replay_checkpoint_identity,
    }
    fingerprint = _config_fingerprint(identity)
    manifest = {
        "dataset": "Mutagenicity",
        "phase": (
            "clear_graphcfe_generation_only_replay"
            if args.generation_only
            else "clear_graphcfe_train_pool_smoke"
        ),
        "config_fingerprint": fingerprint,
        "inputs": {
            "phase_a_root": str(phase_a_root),
            "phase_a_summary": str(phase_a_summary_path),
            "generation_csv": str(generation_csv),
            "teacher_path": str(teacher_path),
            "official_root": str(official_root),
        },
        "config": identity,
        "source_failed_run_root": (
            str(source_run_root) if source_run_root is not None else None
        ),
        "generation_parent_ids": [row.molecule_id for row in selected],
        "parent_limit": int(config.parent_limit),
        "generation_chunk_size": int(config.generation_chunk_size),
        "seed": int(config.seed),
        "model_training_performed": not bool(args.generation_only),
        "codec_version": GENERATED_CODEC_VERSION,
        "generation_input_split": "train",
        "model_train_split": "train",
        "model_validation_split": "val",
        "candidate_selection_performed": False,
        "teacher_used_only_for_final_target_validation": True,
        "official_predictor_role": "generation_diagnostic_and_graphcfe_objective",
        "source_label": 1,
        "target_label": 0,
        "calibration_loaded": False,
        "test_loaded": False,
        "git_commit": _git_commit(),
        "run_complete": False,
    }
    _prepare_output(
        output_dir,
        resume=config.resume,
        fingerprint=fingerprint,
        manifest=manifest,
        create_training_directories=not bool(args.generation_only),
    )
    if args.generation_only:
        assert replay_graphpred is not None and replay_graphcfe is not None
        graphpred_checkpoint = replay_graphpred
        cfe_checkpoint = replay_graphcfe
        trained_models = False
    else:
        (
            graphpred_checkpoint,
            cfe_checkpoint,
            source_run_root,
            trained_models,
        ) = _resolve_generation_checkpoints(
            generation_only=False,
            graphpred_checkpoint=None,
            graphcfe_checkpoint=None,
            source_run_root=None,
            train_models=lambda: _train_official_models(
                official_root=official_root,
                output_dir=output_dir,
                config=config,
                graphpred_learning_rate=float(args.graphpred_learning_rate),
                cfe_learning_rate=float(args.cfe_learning_rate),
                dropout=float(args.dropout),
            ),
        )
    checkpoint_provenance = _checkpoint_provenance(
        graphpred_checkpoint, cfe_checkpoint
    )
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    manifest.update(
        {
            **checkpoint_provenance,
            "source_failed_run_root": (
                str(source_run_root) if source_run_root is not None else None
            ),
            "model_training_performed": bool(trained_models),
        }
    )
    write_json(output_dir / "run_manifest.json", manifest)
    if args.generation_only:
        print(
            "[MUTAGENICITY_CLEAR_GENERATION_REPLAY_CHECKPOINTS_OK]",
            flush=True,
        )
    torch, graphpred, cfe = _load_generation_models(
        official_root=official_root,
        data=data,
        graphpred_checkpoint=graphpred_checkpoint,
        cfe_checkpoint=cfe_checkpoint,
        config=config,
        dropout=float(args.dropout),
    )
    teacher = TeacherSemanticScorer(teacher_path, device="cpu")
    if not teacher.available:
        raise RuntimeError(
            f"Mutagenicity RF teacher unavailable: {teacher.availability_reason}"
        )
    combined_checkpoint_hash = hashlib.sha256(
        (
            checkpoint_provenance["graphpred_checkpoint_sha256"]
            + checkpoint_provenance["graphcfe_checkpoint_sha256"]
        ).encode("utf-8")
    ).hexdigest()
    schema = schema_from_mapping(dict(data.feature_schema))
    if int(data.feature_all[0].shape[1]) != int(schema.feature_dim):
        raise ValueError(
            "Phase A tensor feature dimension does not match feature schema: "
            f"{data.feature_all[0].shape[1]} != {schema.feature_dim}"
        )
    summary_feature_dim = phase_a_summary.get("feature_dim")
    if summary_feature_dim is not None and int(summary_feature_dim) != int(
        schema.feature_dim
    ):
        raise ValueError(
            "Frozen Phase A summary feature dimension does not match pickle "
            f"schema: {summary_feature_dim} != {schema.feature_dim}"
        )
    print(
        "[MUTAGENICITY_CLEAR_FEATURE_DECODING_SCHEMA] "
        + json.dumps(
            feature_decoding_schema_summary(schema), sort_keys=True
        ),
        flush=True,
    )
    try:
        generation_summary = run_streaming_generation(
            output_dir=output_dir,
            parents=selected,
            data=data,
            schema=schema,
            teacher=teacher,
            generate_chunk=_generation_callable(
                torch_module=torch,
                graphpred=graphpred,
                cfe=cfe,
                data=data,
                device=config.device,
            ),
            config=config,
            config_fingerprint=fingerprint,
            model_checkpoint_hash=combined_checkpoint_hash,
            checkpoint_provenance=checkpoint_provenance,
        )
    except ClearMutagenicityEmptyPoolError:
        failure_summary = json.loads(
            (output_dir / "failure_summary.json").read_text(encoding="utf-8")
        )
        failure_summary.update(
            {
                **data_audit,
                **checkpoint_provenance,
                "source_failed_run_root": (
                    str(source_run_root)
                    if source_run_root is not None
                    else None
                ),
                "generation_parent_ids": [
                    row.molecule_id for row in selected
                ],
                "parent_limit": int(config.parent_limit),
                "generation_chunk_size": int(config.generation_chunk_size),
                "seed": int(config.seed),
                "model_training_performed": bool(trained_models),
                "calibration_loaded": False,
                "test_loaded": False,
                "run_complete": False,
            }
        )
        write_json(output_dir / "failure_summary.json", failure_summary)
        write_json(output_dir / "summary.json", failure_summary)
        manifest["run_complete"] = False
        manifest["failure"] = "empty_candidate_pool"
        write_json(output_dir / "run_manifest.json", manifest)
        raise
    print("[MUTAGENICITY_CLEAR_GENERATION_SMOKE_OK]", flush=True)
    summary = {
        **data_audit,
        **generation_summary,
        "generation_source_parent_rows": len(parents),
        "graphpred_epochs": (
            None if args.generation_only else int(config.graphpred_epochs)
        ),
        "cfe_epochs": None if args.generation_only else int(config.cfe_epochs),
        "batch_size": int(config.batch_size),
        "seed": int(config.seed),
        "graphpred_checkpoint": str(graphpred_checkpoint),
        "graphpred_checkpoint_sha256": checkpoint_provenance[
            "graphpred_checkpoint_sha256"
        ],
        "graphcfe_checkpoint": str(cfe_checkpoint),
        "graphcfe_checkpoint_sha256": checkpoint_provenance[
            "graphcfe_checkpoint_sha256"
        ],
        "source_failed_run_root": (
            str(source_run_root) if source_run_root is not None else None
        ),
        "generation_parent_ids": [row.molecule_id for row in selected],
        "parent_limit": int(config.parent_limit),
        "generation_chunk_size": int(config.generation_chunk_size),
        "model_training_performed": bool(trained_models),
        "codec_version": GENERATED_CODEC_VERSION,
        "selected_generation_cohort_hash": cohort_hash(selected),
        "source_label": 1,
        "target_label": 0,
        "strict_flip_definition": "source_rf_pred_1_and_candidate_rf_pred_0",
        "official_clear_flip_role": "diagnostic_only",
        "calibration_rows_loaded": 0,
        "test_rows_loaded": 0,
        "calibration_loaded": False,
        "test_loaded": False,
        "run_complete": False,
    }
    write_json(output_dir / "summary.json", summary)
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    manifest.update(
        {
            "graphpred_checkpoint": str(graphpred_checkpoint),
            "graphcfe_checkpoint": str(cfe_checkpoint),
            **checkpoint_provenance,
            "source_failed_run_root": (
                str(source_run_root) if source_run_root is not None else None
            ),
            "generation_parent_ids": [row.molecule_id for row in selected],
            "parent_limit": int(config.parent_limit),
            "generation_chunk_size": int(config.generation_chunk_size),
            "seed": int(config.seed),
            "model_training_performed": bool(trained_models),
            "codec_version": GENERATED_CODEC_VERSION,
            "selected_generation_cohort_hash": cohort_hash(selected),
            "calibration_loaded": False,
            "test_loaded": False,
            "run_complete": False,
        }
    )
    write_json(output_dir / "run_manifest.json", manifest)
    audit_train_pool(
        run_dir=output_dir,
        generation_csv=generation_csv,
        expected_selected_parents=64,
        require_complete=False,
        teacher=teacher,
    )
    summary["run_complete"] = True
    manifest["run_complete"] = True
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "run_manifest.json", manifest)
    write_json(
        output_dir / "_RUN_COMPLETE.json",
        {
            "run_complete": True,
            "config_fingerprint": fingerprint,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print("[MUTAGENICITY_CLEAR_TRAIN_POOL_SMOKE_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
