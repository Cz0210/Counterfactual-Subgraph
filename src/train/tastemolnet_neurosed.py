"""Train the TasteMolNet-specific GCF auxiliary NeuroSED model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
from typing import Any, Mapping, Sequence
import uuid

from src.data.tastemolnet_neurosed_pairs import (
    TastePairDataset,
    build_connected_bfs_pairs,
    derive_feature_schema,
    pair_manifest,
    parse_taste_split_rows_bytes,
    rows_to_graphs,
    split_boundary_manifest,
)
from src.eval.tastemolnet_neurosed_gate import checkpoint_health
from src.models.tastemolnet_neurosed import (
    DEFAULT_MAX_GRAD_NORM,
    build_training_model,
    interval_loss,
    model_contract,
    runtime_stack,
)
from src.eval.tastemolnet_neurosed_gate import STRICT_OFFICIAL_PROVENANCE


@dataclass(frozen=True, slots=True)
class TasteNeuroSEDTrainConfig:
    seed: int = 7
    train_pairs: int = 50_000
    validation_pairs: int = 5_000
    batch_size: int = 128
    max_epochs: int = 200
    early_stopping_patience: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    cyclic_step_size_up: int = 2_000
    cyclic_step_size_down: int = 2_000
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    num_workers: int = 0
    require_cuda_health_gate: bool = True
    pair_semantics: str = "PENDING_SCIENTIFIC_REVIEW"

    def validate(self) -> None:
        if self.pair_semantics != "directional_exact_deletion_v1":
            raise ValueError(
                "Taste NeuroSED pair semantics require explicit reviewed selection"
            )
        if self.seed != 7:
            raise ValueError("formal Taste NeuroSED seed must be 7")
        integer_fields = {
            "train_pairs": self.train_pairs,
            "validation_pairs": self.validation_pairs,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "cyclic_step_size_up": self.cyclic_step_size_up,
            "cyclic_step_size_down": self.cyclic_step_size_down,
        }
        if any(isinstance(value, bool) or int(value) <= 0 for value in integer_fields.values()):
            raise ValueError(f"Taste NeuroSED integer config is invalid: {integer_fields}")
        if isinstance(self.num_workers, bool) or int(self.num_workers) < 0:
            raise ValueError("Taste NeuroSED num_workers must be a non-negative integer")
        for name, value in {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
        }.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"Taste NeuroSED {name} must be positive finite")
        if not math.isclose(self.max_grad_norm, 0.1, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("official GREED gradient clipping must remain 0.1")
        if not math.isclose(self.learning_rate, 1e-3, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("official GREED learning rate must remain 1e-3")
        if not math.isclose(self.weight_decay, 1e-3, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("official GREED weight decay must remain 1e-3")
        if self.cyclic_step_size_up != 2_000 or self.cyclic_step_size_down != 2_000:
            raise ValueError("official GREED CyclicLR step sizes must remain 2000/2000")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short NeuroSED artifact write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    _write_exclusive(path, _canonical_bytes(payload))


def _save_state_exclusive(path: Path, state: Mapping[str, Any]) -> None:
    torch, _tg, _gcf_models = runtime_stack()
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(os.dup(descriptor), "wb") as handle:
            torch.save(dict(state), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for position in order[cursor:end]:
            result[position] = rank
        cursor = end
    return result


def _spearman(predictions: Sequence[float], targets: Sequence[float]) -> float:
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("Spearman inputs must be non-empty and aligned")
    left = _average_ranks(predictions)
    right = _average_ranks(targets)
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(numerator / (left_norm * right_norm))


def _loader(dataset: TastePairDataset, *, config: TasteNeuroSEDTrainConfig, shuffle: bool) -> Any:
    torch, _tg, _gcf_models = runtime_stack()
    from torch_geometric.loader import DataLoader

    generator = torch.Generator()
    generator.manual_seed(config.seed + (1 if shuffle else 2))
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        generator=generator,
    )


def _evaluate(model: Any, loader: Any, *, device: str) -> dict[str, Any]:
    torch, _tg, _gcf_models = runtime_stack()
    predictions: list[float] = []
    targets: list[float] = []
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for ordered_source, ordered_target, lb, ub in loader:
            ordered_source = ordered_source.to(device)
            ordered_target = ordered_target.to(device)
            lb = lb.to(device=device, dtype=torch.float32)
            ub = ub.to(device=device, dtype=torch.float32)
            prediction = model(ordered_source, ordered_target)
            loss = interval_loss(lb, ub, prediction)
            losses.append(float(loss.detach().cpu().item()))
            predictions.extend(float(value) for value in prediction.detach().cpu().tolist())
            targets.extend(float(value) for value in lb.detach().cpu().tolist())
    if not predictions or not all(
        math.isfinite(value) for value in predictions + targets + losses
    ):
        raise RuntimeError("NeuroSED validation produced non-finite or empty values")
    errors = [prediction - target for prediction, target in zip(predictions, targets)]
    mse = sum(error * error for error in errors) / len(errors)
    return {
        "pair_count": len(predictions),
        "interval_loss": sum(losses) / len(losses),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(mse),
        "spearman_rank": _spearman(predictions, targets),
        "minimum_distance": min(predictions),
        "maximum_distance": max(predictions),
        "finite_distances": True,
    }


def _cpu_state(model: Any) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _new_checkpoint(
    root: Path,
    *,
    model: Any,
    epoch: int,
    validation_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_id = str(uuid.uuid4())
    checkpoint_root = root / checkpoint_id
    checkpoint_root.mkdir(mode=0o700, exist_ok=False)
    state_path = checkpoint_root / "model.pt"
    _save_state_exclusive(state_path, _cpu_state(model))
    manifest = {
        "schema_version": "tastemolnet_gcf_neurosed_checkpoint_v1",
        "checkpoint_uuid": checkpoint_id,
        "epoch": int(epoch),
        "selected_using": "validation_interval_loss_then_validation_mae",
        "validation_metrics": dict(validation_metrics),
        "model_sha256": _sha256_file(state_path),
        "created_at": _utc_now(),
        "path_reuse_forbidden": True,
    }
    _write_json_exclusive(checkpoint_root / "checkpoint.json", manifest)
    return {
        "checkpoint_uuid": checkpoint_id,
        "relative_path": f"checkpoints/{checkpoint_id}/model.pt",
        "manifest_relative_path": f"checkpoints/{checkpoint_id}/checkpoint.json",
        "model_sha256": manifest["model_sha256"],
        "epoch": int(epoch),
        "validation_metrics": dict(validation_metrics),
    }


def _git_state(*, commit: str, tree: str) -> dict[str, Any]:
    for label, value in (("commit", commit), ("tree", tree)):
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"execution git {label} is invalid")
    return {
        "schema_version": "tastemolnet_gcf_neurosed_git_state_v1",
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "cleanliness_verified_by_launcher_before_managed_attempt": True,
    }


def _environment(device: str) -> dict[str, Any]:
    torch, tg, _gcf_models = runtime_stack()
    try:
        from rdkit import rdBase

        rdkit_version = rdBase.rdkitVersion
    except Exception:  # pragma: no cover - dependency already checked upstream.
        rdkit_version = "UNKNOWN"
    return {
        "schema_version": "tastemolnet_gcf_neurosed_environment_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_geometric": str(tg.__version__),
        "rdkit": str(rdkit_version),
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "auto_terminate_uncontrolled_children": False,
    }


def _write_sha256sums(root: Path) -> None:
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"sha256sums.txt", ".generation_token.json"}:
            continue
        relative = path.relative_to(root).as_posix()
        rows.append(f"{_sha256_file(path)}  {relative}")
    _write_exclusive(root / "sha256sums.txt", ("\n".join(rows) + "\n").encode("utf-8"))


def _prepare_output_root(output_root: Path) -> None:
    if output_root.exists():
        unexpected = [
            path.name
            for path in output_root.iterdir()
            if path.name != ".generation_token.json"
        ]
        if unexpected:
            raise FileExistsError(
                f"NeuroSED output root is not fresh managed staging: {unexpected}"
            )
    else:
        output_root.mkdir(parents=True, mode=0o700)


def train_tastemolnet_neurosed(
    *,
    train_csv: str | Path,
    validation_csv: str | Path,
    train_csv_bytes: bytes,
    validation_csv_bytes: bytes,
    output_root: str | Path,
    execution_git_commit: str,
    execution_git_tree: str,
    source_execution_config_sha256: str,
    authoritative_lineage: Mapping[str, Any],
    controller_authority: Mapping[str, Any],
    config: TasteNeuroSEDTrainConfig,
    device: str,
) -> dict[str, Any]:
    """Run one fresh train-only-fit/validation-only-select NeuroSED worker."""

    config.validate()
    if (
        len(source_execution_config_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_execution_config_sha256
        )
    ):
        raise ValueError("source execution config SHA256 is invalid")
    if os.environ.get("AUTO_TERMINATE_UNCONTROLLED_CHILDREN", "0") != "0":
        raise RuntimeError("AUTO_TERMINATE_UNCONTROLLED_CHILDREN must be 0")
    if os.environ.get("RUN_GNN_ABLATION", "0") != "0":
        raise RuntimeError("Taste NeuroSED may not start a GNN backbone ablation")
    torch, _tg, _gcf_models = runtime_stack()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("formal Taste NeuroSED requested CUDA but CUDA is unavailable")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    destination = Path(output_root).absolute()
    _prepare_output_root(destination)
    train_path = Path(train_csv).absolute()
    validation_path = Path(validation_csv).absolute()
    if train_path.parent != validation_path.parent:
        raise RuntimeError(
            "Taste train/validation payloads must share one physical split root"
        )
    train_rows, train_evidence = parse_taste_split_rows_bytes(
        train_csv_bytes, expected_split="train"
    )
    validation_rows, validation_evidence = parse_taste_split_rows_bytes(
        validation_csv_bytes, expected_split="validation"
    )
    train_evidence = {
        **train_evidence,
        "source_csv_sha256": hashlib.sha256(train_csv_bytes).hexdigest(),
    }
    validation_evidence = {
        **validation_evidence,
        "source_csv_sha256": hashlib.sha256(validation_csv_bytes).hexdigest(),
    }
    train_ids = {row.molecule_id for row in train_rows}
    validation_ids = {row.molecule_id for row in validation_rows}
    if train_ids & validation_ids:
        raise RuntimeError("Taste train and validation molecule IDs overlap")
    train_authority = authoritative_lineage.get("train")
    validation_authority = authoritative_lineage.get("validation")
    if (
        authoritative_lineage.get("schema_version")
        != "tastemolnet_gcf_neurosed_authority_v2"
        or authoritative_lineage.get("t3_split_manifest_byte_identical_to_t2")
        is not True
        or authoritative_lineage.get("calibration_loaded") is not False
        or authoritative_lineage.get("test_loaded") is not False
        or type(train_authority) is not dict
        or type(validation_authority) is not dict
        or train_authority.get("source_path") != str(train_path)
        or validation_authority.get("source_path") != str(validation_path)
        or train_authority.get("source_sha256")
        != train_evidence["source_csv_sha256"]
        or validation_authority.get("source_sha256")
        != validation_evidence["source_csv_sha256"]
        or train_authority.get("num_records") != len(train_rows)
        or validation_authority.get("num_records") != len(validation_rows)
    ):
        raise RuntimeError("Taste NeuroSED authoritative split lineage changed")
    worker_latest = controller_authority.get("worker_latest")
    worker_initial = controller_authority.get("worker_initial_heartbeat")
    if (
        controller_authority.get("schema_version")
        != "tastemolnet_gcf_neurosed_controller_binding_v2"
        or type(worker_latest) is not dict
        or type(worker_initial) is not dict
        or worker_latest.get("controller_id") is None
        or worker_latest.get("state") != "RUNNING"
        or worker_latest.get("controller_state") != "MONITORING"
        or worker_latest.get("git_commit") != execution_git_commit
        or worker_latest.get("git_tree") != execution_git_tree
        or worker_initial.get("receipt_sha256")
        != worker_latest.get("receipt_sha256")
        or not isinstance(worker_initial.get("heartbeat_sha256"), str)
        or type(worker_initial.get("sequence")) is not int
        or type(worker_latest.get("sequence")) is not int
        or int(worker_latest["sequence"]) < int(worker_initial["sequence"])
    ):
        raise RuntimeError("Taste main-v2 controller authority changed")
    feature_schema = derive_feature_schema(train_rows, validation_rows)
    train_graphs = rows_to_graphs(train_rows, feature_schema)
    validation_graphs = rows_to_graphs(validation_rows, feature_schema)
    train_pairs = build_connected_bfs_pairs(
        train_graphs,
        split="train",
        num_pairs=config.train_pairs,
        seed=config.seed,
    )
    validation_pairs = build_connected_bfs_pairs(
        validation_graphs,
        split="validation",
        num_pairs=config.validation_pairs,
        seed=config.seed + 1,
    )
    train_pair_manifest = pair_manifest(train_pairs, split="train")
    validation_pair_manifest = pair_manifest(validation_pairs, split="validation")
    split_manifest = split_boundary_manifest(
        train_evidence=train_evidence,
        validation_evidence=validation_evidence,
        preparation_manifest={
            "sha256": authoritative_lineage[
                "t2_source_split_manifest_sha256"
            ]
        },
        train_validation_intersection_empty=not bool(train_ids & validation_ids),
    )
    split_manifest.update(
        {
            "authority_schema": authoritative_lineage["schema_version"],
            "authority_manifest_sha256": hashlib.sha256(
                _canonical_bytes(authoritative_lineage)
            ).hexdigest(),
            "t2_receipt_gate_sha256": authoritative_lineage[
                "t2_receipt_gate_sha256"
            ],
            "t2_source_evidence_sha256": authoritative_lineage[
                "t2_source_evidence_sha256"
            ],
            "t3_gate_sha256": authoritative_lineage["t3_gate_sha256"],
            "t3_verification_sha256": authoritative_lineage[
                "t3_verification_sha256"
            ],
            "train_authoritative_num_records": train_authority["num_records"],
            "train_authoritative_label_counts": train_authority["label_counts"],
            "train_authoritative_dataset_fingerprint": train_authority[
                "dataset_fingerprint"
            ],
            "validation_authoritative_num_records": validation_authority[
                "num_records"
            ],
            "validation_authoritative_label_counts": validation_authority[
                "label_counts"
            ],
            "validation_authoritative_dataset_fingerprint": validation_authority[
                "dataset_fingerprint"
            ],
        }
    )

    train_loader = _loader(TastePairDataset(train_pairs), config=config, shuffle=True)
    validation_loader = _loader(
        TastePairDataset(validation_pairs), config=config, shuffle=False
    )
    input_dim = int(feature_schema["input_dim"])
    model = build_training_model(input_dim=input_dim, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=0.0,
        max_lr=config.learning_rate,
        step_size_up=config.cyclic_step_size_up,
        step_size_down=config.cyclic_step_size_down,
        cycle_momentum=False,
    )
    checkpoints_root = destination / "checkpoints"
    checkpoints_root.mkdir(mode=0o700, exist_ok=False)
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    stale_epochs = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses: list[float] = []
        gradient_norms: list[float] = []
        for ordered_source, ordered_target, lb, ub in train_loader:
            ordered_source = ordered_source.to(device)
            ordered_target = ordered_target.to(device)
            lb = lb.to(device=device, dtype=torch.float32)
            ub = ub.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(ordered_source, ordered_target)
            loss = interval_loss(lb, ub, prediction)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Taste NeuroSED training loss is non-finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu().item()))
            gradient_norms.append(float(torch.as_tensor(gradient_norm).detach().cpu().item()))
        validation_metrics = _evaluate(model, validation_loader, device=device)
        train_loss = sum(losses) / len(losses)
        if not math.isfinite(train_loss) or not all(
            math.isfinite(value) for value in gradient_norms
        ):
            raise RuntimeError("Taste NeuroSED epoch diagnostics are non-finite")
        epoch_row = {
            "epoch": epoch,
            "train_interval_loss": train_loss,
            "maximum_unclipped_gradient_norm": max(gradient_norms),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "validation": validation_metrics,
        }
        history.append(epoch_row)
        selection_key = (
            float(validation_metrics["interval_loss"]),
            float(validation_metrics["mae"]),
        )
        best_key = (
            (
                float(best["validation_metrics"]["interval_loss"]),
                float(best["validation_metrics"]["mae"]),
            )
            if best is not None
            else (float("inf"), float("inf"))
        )
        if selection_key < best_key:
            checkpoint = _new_checkpoint(
                checkpoints_root,
                model=model,
                epoch=epoch,
                validation_metrics=validation_metrics,
            )
            checkpoints.append(checkpoint)
            best = checkpoint
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(
            "[TASTE_GCF_NEUROSED_EPOCH] "
            f"epoch={epoch} train_interval_loss={train_loss:.8g} "
            f"validation_interval_loss={validation_metrics['interval_loss']:.8g} "
            f"validation_mae={validation_metrics['mae']:.8g}",
            flush=True,
        )
        if stale_epochs >= config.early_stopping_patience:
            break
    if best is None:
        raise RuntimeError("Taste NeuroSED did not create a validation-selected checkpoint")

    selected_path = destination / str(best["relative_path"])
    _write_exclusive(destination / "best.pt", selected_path.read_bytes())
    final_state = _cpu_state(model)
    _save_state_exclusive(destination / "model.pt", final_state)
    best_health = checkpoint_health(
        destination / "best.pt",
        input_dim=input_dim,
        require_cuda_tolerance=config.require_cuda_health_gate,
    )
    pair_payload = {
        "schema_version": "tastemolnet_gcf_neurosed_pair_bundle_v1",
        "train": train_pair_manifest,
        "validation": validation_pair_manifest,
        "train_pair_count": len(train_pairs),
        "validation_pair_count": len(validation_pairs),
        "calibration_pair_count": 0,
        "test_pair_count": 0,
        "source_label_independent": True,
        "labels_used": False,
    }
    training_metrics = {
        "schema_version": "tastemolnet_gcf_neurosed_training_metrics_v1",
        "optimizer": "AdamW",
        "scheduler": "CyclicLR",
        "gradient_clip_norm": config.max_grad_norm,
        "criterion": "GREED interval loss",
        "epochs_completed": len(history),
        "best_epoch": int(best["epoch"]),
        "best_validation_interval_loss": float(
            best["validation_metrics"]["interval_loss"]
        ),
        "selection_split": "validation",
        "selection_metric": "interval_loss",
        "selection_tiebreak": "mae",
        "history": history,
        "finite_loss": True,
        "test_metrics_computed": False,
    }
    validation_metrics = {
        "schema_version": "tastemolnet_gcf_neurosed_validation_metrics_v1",
        **dict(best["validation_metrics"]),
        "selected_checkpoint_uuid": best["checkpoint_uuid"],
        "checkpoint_selection_only": True,
        "test_used_for_selection": False,
    }
    checkpoint_manifest = {
        "schema_version": "tastemolnet_gcf_neurosed_checkpoint_manifest_v1",
        "selected_checkpoint_uuid": best["checkpoint_uuid"],
        "selected_checkpoint_relative_path": best["relative_path"],
        "selected_checkpoint_sha256": best["model_sha256"],
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "uuid_paths_never_reused": True,
        "fixed_checkpoint_directory_used": False,
    }
    model_card = {
        "schema_version": "tastemolnet_gcf_neurosed_model_card_v1",
        "dataset": "tastemolnet",
        "role": "GCF_AUXILIARY_DISTANCE_MODEL",
        "classifier": False,
        "oracle": False,
        "teacher_label_model": False,
        "source_label_independent": True,
        "train_only_fit": True,
        "validation_only_selection": True,
        "training_loop_authority": "reviewed_taste_epoch_level_adaptation_v1",
        "upstream_greed_batch_interleaved_selection_loop_unchanged": False,
        "adaptation_note": (
            "Taste keeps the pinned model/forward/loss/optimizer/scheduler but "
            "uses reviewed epoch-level validation selection and patience."
        ),
        "calibration_loaded": False,
        "test_loaded": False,
        "rf_oracle_used": False,
        "frozen_classifier_replaced": False,
        "architecture": model_contract(input_dim),
        "downstream_checkpoint": "best.pt",
        "vrrw_semantics_verified_by_neurosed_training": False,
        "vrrw_semantics_scope": "T7_CONSUMER_ONLY",
        "taste_pair_direction_adaptation": (
            "parent_to_connected_induced_bfs_subgraph_for_exact_directional_deletions"
        ),
        "pair_semantics": config.pair_semantics,
        "upstream_greed_pair_sampling_unchanged": False,
        "upstream_greed_pair_sampling": (
            "independent_query_subgraph_to_random_target_with_pyged_bounds"
        ),
        "gcf_runtime_direction": "generated_query_to_original_parent_target",
        "training_direction_matches_gcf_runtime": False,
        "full_official_neurosed_semantics_claimed": False,
        "scientific_release_eligible": False,
        "strict_official_pair_builder_implemented": False,
        "strict_official_pyged_bounds_authenticated": False,
        "strict_official_batch_interleaved_selector_implemented": False,
        "strict_official_provenance": dict(STRICT_OFFICIAL_PROVENANCE),
        "reverse_subgraph_to_parent_cost_under_pinned_greed_is_zero": True,
        "data_redistribution_allowed": False,
    }
    health_gate = {
        "schema_version": "tastemolnet_gcf_neurosed_worker_health_v1",
        "status": "READY_FOR_INDEPENDENT_VERIFICATION",
        "worker_is_not_independent_verifier": True,
        "worker_wrote_pass": False,
        "finite_loss": True,
        "finite_validation_rank_error": True,
        "no_train_test_leakage": True,
        "feature_schema_compatibility": True,
        "checkpoint_health": best_health,
    }
    scrubbed_config = {
        "schema_version": "tastemolnet_gcf_neurosed_config_v1",
        **asdict(config),
        "source_execution_config_sha256": source_execution_config_sha256,
        "device": device,
        "dataset": "tastemolnet",
        "role": "GCF_AUXILIARY_DISTANCE_MODEL",
        "architecture": model_contract(input_dim),
        "optimizer": "AdamW",
        "scheduler": "CyclicLR",
        "criterion": "GREED interval loss",
        "pair_builder": "taste_connected_induced_bfs_nested_pair_v1",
        "pair_direction": "parent_to_connected_induced_bfs_subgraph",
        "opened_payload_splits": ["train", "validation"],
        "calibration_loaded": False,
        "test_loaded": False,
    }
    documents = {
        "config.yaml": scrubbed_config,
        "model_card.json": model_card,
        "pair_manifest.json": pair_payload,
        "split_manifest.json": split_manifest,
        "training_metrics.json": training_metrics,
        "validation_metrics.json": validation_metrics,
        "feature_schema.json": feature_schema,
        "environment.json": _environment(device),
        "git_state.json": _git_state(
            commit=execution_git_commit,
            tree=execution_git_tree,
        ),
        "checkpoint_manifest.json": checkpoint_manifest,
        "health_gate.json": health_gate,
        "authority_manifest.json": dict(authoritative_lineage),
        "controller_authority.json": dict(controller_authority),
    }
    for name, payload in documents.items():
        _write_json_exclusive(destination / name, payload)
    _write_sha256sums(destination)
    return {
        "state": "WORKER_ARTIFACT_READY_FOR_SEAL",
        "dataset": "tastemolnet",
        "role": "GCF_AUXILIARY_DISTANCE_MODEL",
        "output_root": str(destination),
        "selected_checkpoint": str(destination / "best.pt"),
        "selected_checkpoint_sha256": _sha256_file(destination / "best.pt"),
        "selected_checkpoint_uuid": best["checkpoint_uuid"],
        "train_graph_ids_hash": split_manifest["neurosed_train_graph_ids_hash"],
        "validation_graph_ids_hash": split_manifest[
            "neurosed_validation_graph_ids_hash"
        ],
        "calibration_loaded": False,
        "test_loaded": False,
        "independent_verification_required": True,
    }


__all__ = ["TasteNeuroSEDTrainConfig", "train_tastemolnet_neurosed"]
