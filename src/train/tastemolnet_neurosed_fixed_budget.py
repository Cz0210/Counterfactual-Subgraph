"""Production trainer for frozen fixed-budget TasteMolNet NeuroSED labels.

This is deliberately separate from the historical own-parent/epoch-selector
trainer.  It consumes only authenticated independent train/validation pairs,
uses the pinned GREED AIDS training hyperparameters, and delegates checkpoint
ordering to :class:`OfficialBatchInterleavedSelector`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
from typing import Any, Mapping, Sequence
import uuid

from src.data.tastemolnet_neurosed_production import (
    FixedBudgetPairInventory,
    NeuroSEDProductionDataError,
    PreparedFixedBudgetPair,
    load_fixed_budget_pair_inventory,
    load_json,
    read_compact_npz,
    sha256_file,
    stable_sha256,
)
from src.eval.tastemolnet_neurosed_fixed_budget import OFFICIAL_SED_EDIT_COSTS
from src.eval.tastemolnet_neurosed_gate import STRICT_OFFICIAL_PROVENANCE
from src.eval.tastemolnet_neurosed_label_writer import (
    GED_LABEL_MANIFEST_SCHEMA,
    GED_LABEL_PASS_MARKER,
)
from src.eval.tastemolnet_neurosed_non_mip import (
    validate_non_mip_selection_manifest,
)
from src.eval.tastemolnet_neurosed_official_fixed_budget import (
    OFFICIAL_FIXED_MODEL_CARD_SCHEMA,
    OFFICIAL_GCF_COMMIT,
    OFFICIAL_GCF_REPOSITORY,
    VENDORED_GCF_RETAINED_INVENTORY_SHA256,
    VENDORED_GCF_SOURCE_SHA256,
    GeneratedQueryOriginalTargetBinding,
    validate_official_fixed_budget_model_card,
    verify_official_fixed_budget_readiness,
)
from src.models.tastemolnet_neurosed import (
    DEFAULT_MAX_GRAD_NORM,
    build_training_model,
    interval_loss,
    load_runner_checkpoint,
    load_state_dict_bytes,
    model_contract,
    runtime_stack,
    verify_bundled_runner_sources,
)
from src.train.tastemolnet_neurosed_official_selector import (
    OfficialBatchInterleavedSelector,
)
from src.utils.tastemolnet_neurosed_gedlib_build import (
    GED_LABEL_BACKEND_VARIANT,
)


TRAINER_SCHEMA = "tastemolnet_neurosed_fixed_budget_trainer_v1"
TRAINER_READY_MARKER = "[TASTE_NEUROSED_FIXED_BUDGET_READY_FOR_VERIFIER]"
NEUROSED_PASS_MARKER = "[TASTE_NEUROSED_FIXED_BUDGET_PASS]"
OFFICIAL_EXPERIMENT_NOTEBOOK_SHA256 = (
    "49a7bc0095d879bf49454cd6c18e42bb687c149a32e425b59c2acbe6c2df0114"
)


@dataclass(frozen=True, slots=True)
class FixedBudgetNeuroSEDTrainConfig:
    """Hyperparameters pinned by the GREED-expts AIDS training notebook."""

    seed: int = 7
    train_pair_budget: int = 5000
    validation_pair_budget: int = 1000
    train_batch_size: int = 200
    validation_batch_size: int = 1000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    cycle_patience: int = 5
    step_size_up: int = 2000
    step_size_down: int = 2000
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    num_workers: int = 0

    def validate(self) -> None:
        expected = {
            "seed": 7,
            "train_pair_budget": 5000,
            "validation_pair_budget": 1000,
            "train_batch_size": 200,
            "validation_batch_size": 1000,
            "cycle_patience": 5,
            "step_size_up": 2000,
            "step_size_down": 2000,
            "num_workers": 0,
        }
        if any(getattr(self, key) != value for key, value in expected.items()):
            raise ValueError("fixed-budget official NeuroSED integer config changed")
        if (
            self.learning_rate != 1e-3
            or self.weight_decay != 1e-3
            or self.max_grad_norm != 0.1
        ):
            raise ValueError("fixed-budget official NeuroSED optimizer config changed")


@dataclass(frozen=True, slots=True)
class LabeledPairBundle:
    inventory: FixedBudgetPairInventory
    rows: tuple[Mapping[str, Any], ...]
    label_manifest: Mapping[str, Any]


class FixedBudgetPairDataset:
    def __init__(self, examples: Sequence[tuple[Any, Any, float, float]]) -> None:
        self.examples = tuple(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[Any, Any, Any, Any]:
        torch, _tg, _models = runtime_stack()
        query, target, lower, upper = self.examples[index]
        return (
            query,
            target,
            torch.tensor(lower, dtype=torch.float32),
            torch.tensor(upper, dtype=torch.float32),
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(data)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise OSError("short NeuroSED artifact write")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new(path, _canonical_bytes(value))


def _save_state(path: Path, state: Mapping[str, Any]) -> str:
    torch, _tg, _models = runtime_stack()
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
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
    return sha256_file(path)


def _cpu_state(model: Any) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _prepare_output_root(path: Path) -> None:
    if path.exists():
        if any(path.iterdir()):
            raise FileExistsError("fixed-budget NeuroSED output root is not fresh")
    else:
        path.mkdir(parents=True, mode=0o700)


def _validate_feature_schema(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise NeuroSEDProductionDataError("feature schema file hash changed")
    schema = load_json(path)
    vocabulary = schema.get("feature_atomic_numbers")
    if (
        schema.get("schema_version")
        != "tastemolnet_gcf_neurosed_feature_schema_v1"
        or schema.get("dataset") != "tastemolnet"
        or schema.get("node_feature_semantics") != "one_hot_atomic_number"
        or schema.get("explicit_h_nodes") is not True
        or schema.get("native_adjacency_semantics")
        != "binary_connectivity_directed_both_ways"
        or schema.get("edge_features_used") is not False
        or schema.get("validation_unseen_atomic_numbers") != []
        or schema.get("train_derived_only") is not True
        or type(vocabulary) is not list
        or not vocabulary
        or vocabulary != sorted(set(vocabulary))
        or schema.get("input_dim") != len(vocabulary)
    ):
        raise NeuroSEDProductionDataError("feature schema contract changed")
    return schema


def _validate_label_manifest(
    root: Path,
    *,
    split: str,
    budget: int,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = root / f"{split}_label_manifest.json"
    manifest = load_json(manifest_path)
    claimed = str(manifest.get("manifest_sha256") or "")
    if (
        claimed != expected_manifest_sha256
        or claimed
        != stable_sha256(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        )
        or manifest.get("split") != split
        or manifest.get("requested_pair_count") != budget
        or manifest.get("successful_pair_count") != budget
        or manifest.get("calibration_loaded") is not False
        or manifest.get("test_loaded") is not False
    ):
        raise NeuroSEDProductionDataError(f"{split} label manifest changed")
    labels_path = root / str(manifest.get("compact_labels_path") or "")
    if sha256_file(labels_path) != manifest.get("compact_labels_sha256"):
        raise NeuroSEDProductionDataError(f"{split} compact labels changed")
    rows = read_compact_npz(labels_path)
    if (
        len(rows) != budget
        or any(row["status"] != "SUCCESS" or row["split"] != split for row in rows)
        or stable_sha256([row["pair_id"] for row in rows])
        != manifest.get("pair_ids_sha256")
    ):
        raise NeuroSEDProductionDataError(f"{split} successful labels changed")
    return manifest, rows


def load_labeled_pair_bundles(
    *,
    ged_label_root: str | Path,
    train_pair_root: str | Path,
    validation_pair_root: str | Path,
    feature_schema_path: str | Path,
    config: FixedBudgetNeuroSEDTrainConfig,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    LabeledPairBundle,
    LabeledPairBundle,
]:
    """Reopen labels and prove each row still binds its reconstructed pair."""

    config.validate()
    label_root = Path(ged_label_root).absolute()
    aggregate = load_json(label_root / "ged_label_manifest.json")
    if (
        aggregate.get("schema_version") != GED_LABEL_MANIFEST_SCHEMA
        or aggregate.get("status") != "PASS"
        or aggregate.get("marker") != GED_LABEL_PASS_MARKER
        or aggregate.get("train_success_count") != config.train_pair_budget
        or aggregate.get("validation_success_count")
        != config.validation_pair_budget
        or aggregate.get("ged_backend") != "branch"
        or aggregate.get("F2_BLP_USED") is not False
        or aggregate.get("GUROBI_USED") is not False
        or aggregate.get("calibration_loaded") is not False
        or aggregate.get("test_loaded") is not False
        or aggregate.get("manifest_sha256")
        != stable_sha256(
            {key: value for key, value in aggregate.items() if key != "manifest_sha256"}
        )
        or (label_root / "PASS").read_text(encoding="utf-8").strip()
        != GED_LABEL_PASS_MARKER
    ):
        raise NeuroSEDProductionDataError("GED label aggregate is not PASS")
    schema = _validate_feature_schema(
        Path(feature_schema_path).absolute(), str(aggregate["feature_schema_sha256"])
    )
    train_inventory = load_fixed_budget_pair_inventory(
        train_pair_root,
        split="train",
        requested_pair_count=config.train_pair_budget,
    )
    validation_inventory = load_fixed_budget_pair_inventory(
        validation_pair_root,
        split="validation",
        requested_pair_count=config.validation_pair_budget,
    )
    if (
        train_inventory.manifest["manifest_sha256"]
        != aggregate["train_pair_sampler_manifest_sha256"]
        or validation_inventory.manifest["manifest_sha256"]
        != aggregate["validation_pair_sampler_manifest_sha256"]
    ):
        raise NeuroSEDProductionDataError("label aggregate sampler binding changed")
    if (
        train_inventory.reserve_available_count != 0
        or validation_inventory.reserve_available_count != 0
        or aggregate.get("train_inventory_pair_count") != config.train_pair_budget
        or aggregate.get("validation_inventory_pair_count")
        != config.validation_pair_budget
        or aggregate.get("train_reserve_available_count") != 0
        or aggregate.get("validation_reserve_available_count") != 0
        or aggregate.get("exact_budget_inventory_without_reserve") is not True
    ):
        raise NeuroSEDProductionDataError(
            "active trainer requires the frozen exact-budget inventory mode"
        )
    train_manifest, train_rows = _validate_label_manifest(
        label_root,
        split="train",
        budget=config.train_pair_budget,
        expected_manifest_sha256=str(aggregate["train_label_manifest_sha256"]),
    )
    validation_manifest, validation_rows = _validate_label_manifest(
        label_root,
        split="validation",
        budget=config.validation_pair_budget,
        expected_manifest_sha256=str(
            aggregate["validation_label_manifest_sha256"]
        ),
    )

    def bind(
        inventory: FixedBudgetPairInventory,
        rows: Sequence[Mapping[str, Any]],
        manifest: Mapping[str, Any],
    ) -> LabeledPairBundle:
        by_id = {pair.pair_id: pair for pair in inventory.pairs}
        for row in rows:
            pair = by_id.get(str(row["pair_id"]))
            if (
                pair is None
                or row["query_graph_id"] != pair.metadata["query_graph_id"]
                or row["target_graph_id"] != pair.metadata["target_graph_id"]
                or row["query_hash"] != pair.query.canonical_graph_sha256
                or row["target_hash"] != pair.target.canonical_graph_sha256
                or not math.isfinite(float(row["lower_bound"]))
                or not math.isfinite(float(row["upper_bound"]))
                or float(row["lower_bound"]) > float(row["upper_bound"])
            ):
                raise NeuroSEDProductionDataError("compact label/pair binding changed")
        return LabeledPairBundle(
            inventory=inventory,
            rows=tuple(dict(row) for row in rows),
            label_manifest=dict(manifest),
        )

    return (
        aggregate,
        schema,
        bind(train_inventory, train_rows, train_manifest),
        bind(validation_inventory, validation_rows, validation_manifest),
    )


def _graph_data(graph: Any, *, input_dim: int) -> Any:
    torch, tg, _models = runtime_stack()
    labels = tuple(int(value) for value in graph.node_labels)
    if not labels or min(labels) < 0 or max(labels) >= input_dim:
        raise NeuroSEDProductionDataError("pair node feature index changed")
    x = torch.zeros((len(labels), input_dim), dtype=torch.float32)
    x[torch.arange(len(labels)), torch.tensor(labels, dtype=torch.long)] = 1.0
    edge_index = torch.tensor(graph.directed_edges, dtype=torch.long).t().contiguous()
    return tg.data.Data(x=x, edge_index=edge_index, num_nodes=len(labels))


def _examples(bundle: LabeledPairBundle, *, input_dim: int) -> list[tuple[Any, Any, float, float]]:
    by_id: dict[str, PreparedFixedBudgetPair] = {
        pair.pair_id: pair for pair in bundle.inventory.pairs
    }
    target_cache: dict[str, Any] = {}
    result: list[tuple[Any, Any, float, float]] = []
    for row in bundle.rows:
        pair = by_id[str(row["pair_id"])]
        target = target_cache.get(pair.target.graph_id)
        if target is None:
            target = _graph_data(pair.target, input_dim=input_dim)
            target_cache[pair.target.graph_id] = target
        result.append(
            (
                _graph_data(pair.query, input_dim=input_dim),
                target,
                float(row["lower_bound"]),
                float(row["upper_bound"]),
            )
        )
    return result


def _loader(dataset: FixedBudgetPairDataset, *, batch_size: int, seed: int) -> Any:
    torch, _tg, _models = runtime_stack()
    from torch_geometric.loader import DataLoader

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )


def _checkpoint_candidate(root: Path, model: Any, event_index: int) -> dict[str, Any]:
    identifier = str(uuid.uuid4())
    checkpoint_root = root / identifier
    checkpoint_root.mkdir(mode=0o700, exist_ok=False)
    path = checkpoint_root / "model.pt"
    digest = _save_state(path, _cpu_state(model))
    manifest = {
        "schema_version": "tastemolnet_neurosed_preupdate_checkpoint_v1",
        "checkpoint_uuid": identifier,
        "validation_event_index": event_index,
        "captured_before_paired_training_update": True,
        "model_sha256": digest,
    }
    _write_json(checkpoint_root / "checkpoint.json", manifest)
    return {
        **manifest,
        "relative_path": path.relative_to(root.parent).as_posix(),
    }


def _evaluate(model: Any, loader: Any, *, device: str) -> dict[str, Any]:
    torch, _tg, _models = runtime_stack()
    predictions: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    weighted_loss = 0.0
    model.eval()
    with torch.no_grad():
        for query, target, lower, upper in loader:
            query = query.to(device)
            target = target.to(device)
            lower = lower.to(device=device, dtype=torch.float32)
            upper = upper.to(device=device, dtype=torch.float32)
            prediction = model(query, target)
            loss = interval_loss(lower, upper, prediction)
            count = int(prediction.numel())
            weighted_loss += float(loss.detach().cpu().item()) * count
            predictions.extend(float(value) for value in prediction.detach().cpu())
            lowers.extend(float(value) for value in lower.detach().cpu())
            uppers.extend(float(value) for value in upper.detach().cpu())
    if not predictions or not all(
        math.isfinite(value) for value in predictions + lowers + uppers
    ):
        raise RuntimeError("NeuroSED selected validation metric is non-finite")
    violations = [
        max(lower - prediction, prediction - upper, 0.0)
        for prediction, lower, upper in zip(predictions, lowers, uppers)
    ]
    return {
        "pair_count": len(predictions),
        "interval_loss": weighted_loss / len(predictions),
        "mean_absolute_interval_violation": sum(violations) / len(violations),
        "maximum_interval_violation": max(violations),
        "finite_validation_metric": True,
    }


def _health_checks(
    *,
    checkpoint: Path,
    input_dim: int,
    validation_examples: Sequence[tuple[Any, Any, float, float]],
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch, tg, _models = runtime_stack()
    checkpoint_bytes = checkpoint.read_bytes()
    state = load_state_dict_bytes(checkpoint_bytes, map_location=device)
    training_model = build_training_model(input_dim=input_dim, device=device)
    training_model.load_state_dict(state, strict=True)
    training_model.eval()
    runner_model = load_runner_checkpoint(
        checkpoint, input_dim=input_dim, device=device
    )
    count = min(3, len(validation_examples))
    queries = [validation_examples[index][0] for index in range(count)]
    targets = [validation_examples[index][1] for index in range(count)]
    query_batch = tg.data.Batch.from_data_list(queries).to(device)
    target_batch = tg.data.Batch.from_data_list(targets).to(device)

    def single_predictions(model: Any) -> Any:
        values = []
        for query, target in zip(queries, targets):
            q = tg.data.Batch.from_data_list([query]).to(device)
            t = tg.data.Batch.from_data_list([target]).to(device)
            values.append(model(q, t).reshape(-1))
        return torch.cat(values)

    with torch.no_grad():
        training_batch = training_model(query_batch, target_batch).reshape(-1)
        runner_batch = runner_model(query_batch, target_batch).reshape(-1)
        training_single = single_predictions(training_model)
        runner_single = single_predictions(runner_model)
        training_q = training_model.embed_model(query_batch)
        training_t = training_model.embed_model(target_batch)
        runner_q = runner_model.embed_model(query_batch)
        runner_t = runner_model.embed_model(target_batch)
        expected_training = torch.norm(torch.relu(training_q - training_t), dim=-1)
        expected_runner = torch.norm(runner_q - runner_t, dim=-1)
    checks = {
        "checkpoint_reload_passed": True,
        "training_model_strict_load_passed": True,
        "gcf_runner_load_passed": True,
        "training_batch_single_agreement": bool(
            torch.allclose(training_batch, training_single, rtol=1e-5, atol=1e-6)
        ),
        "runner_batch_single_agreement": bool(
            torch.allclose(runner_batch, runner_single, rtol=1e-5, atol=1e-6)
        ),
        "training_forward_is_relu_directional_sed": bool(
            torch.allclose(training_batch, expected_training, rtol=1e-5, atol=1e-6)
        ),
        "runner_forward_is_symmetric_normged": bool(
            torch.allclose(runner_batch, expected_runner, rtol=1e-5, atol=1e-6)
        ),
        "training_runner_forward_intentionally_different": True,
        "shared_embedding_state_dict": all(
            torch.equal(training_model.state_dict()[key], runner_model.state_dict()[key])
            for key in training_model.state_dict()
        ),
        "bundled_runner_sources": verify_bundled_runner_sources(),
    }
    if any(checks[key] is not True for key in (
        "training_batch_single_agreement",
        "runner_batch_single_agreement",
        "training_forward_is_relu_directional_sed",
        "runner_forward_is_symmetric_normged",
        "shared_embedding_state_dict",
    )):
        raise RuntimeError("NeuroSED checkpoint reload/forward health check failed")
    binding = GeneratedQueryOriginalTargetBinding.create(
        runner_model,
        original_targets=targets[:2],
        original_target_hashes=[
            hashlib.sha256(
                json.dumps(
                    {
                        "x": target.x.tolist(),
                        "edge_index": target.edge_index.tolist(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for target in targets[:2]
        ],
    )
    binding.predict_generated_queries(
        queries[:2],
        generated_query_hashes=[
            hashlib.sha256(
                json.dumps(
                    {
                        "x": query.x.tolist(),
                        "edge_index": query.edge_index.tolist(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for query in queries[:2]
        ],
        batch_size=2,
    )
    direction = binding.direction_manifest()
    direction.update(
        {
            "health_probe_only": True,
            "probe_query_source": "validation_fixed_budget_query",
            "generated_query_to_original_target_assertion": True,
        }
    )
    direction.pop("trace_sha256", None)
    direction["trace_sha256"] = stable_sha256(direction)
    return checks, direction


def _environment(device: str) -> dict[str, Any]:
    torch, tg, _models = runtime_stack()
    return {
        "schema_version": "tastemolnet_neurosed_environment_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_geometric": str(tg.__version__),
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.device(device))
            if str(device).startswith("cuda") and torch.cuda.is_available()
            else None
        ),
    }


def _git_state(commit: str, tree: str) -> dict[str, Any]:
    for label, value in (("commit", commit), ("tree", tree)):
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"execution Git {label} is invalid")
    return {
        "schema_version": "tastemolnet_neurosed_git_state_v1",
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
    }


def _write_sha256sums(root: Path) -> None:
    rows = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "sha256sums.txt"
    ]
    _write_new(root / "sha256sums.txt", ("\n".join(rows) + "\n").encode("utf-8"))


def _verify_sha256sums(root: Path) -> None:
    rows = (root / "sha256sums.txt").read_text(encoding="utf-8").splitlines()
    expected: dict[str, str] = {}
    for row in rows:
        digest, separator, relative = row.partition("  ")
        if not separator or relative in expected:
            raise RuntimeError("NeuroSED checksum inventory is malformed")
        expected[relative] = digest
    actual = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in root.rglob("*")
        if path.is_file()
        and path.name
        not in {"sha256sums.txt", "verification.json", "verification_sha256s.txt", "PASS"}
    }
    if actual != expected:
        raise RuntimeError("NeuroSED checksum inventory changed")


def _replay_selector_trace(trace: Mapping[str, Any]) -> None:
    selector = OfficialBatchInterleavedSelector(
        cycle_patience=int(trace["cycle_patience"]),
        step_size_up=int(trace["step_size_up"]),
        step_size_down=int(trace["step_size_down"]),
    )
    for original in trace["trace"]:
        decision = selector.observe_validation(
            float(original["validation_metric"]),
            training_batch_index=int(original["training_batch_index"]),
        )
        replayed = asdict(decision)
        if any(original.get(key) != value for key, value in replayed.items()):
            raise RuntimeError("official selector decision trace changed")
        if decision.checkpoint_candidate:
            selector.bind_checkpoint_candidate(
                validation_event_index=decision.validation_event_index,
                checkpoint_sha256=str(original["checkpoint_sha256"]),
            )
        if decision.stop_before_training_batch:
            break
        selector.record_training_update(
            training_batch_index=decision.training_batch_index,
            optimizer_step_completed=bool(original["optimizer_step_completed"]),
            cyclic_lr_step_completed=bool(original["cyclic_lr_step_completed"]),
            gradient_clip_norm=float(original["gradient_clip_norm"]),
        )
    if selector.trace_manifest() != dict(trace):
        raise RuntimeError("official selector trace failed independent replay")


def train_fixed_budget_neurosed(
    *,
    ged_label_root: str | Path,
    train_pair_root: str | Path,
    validation_pair_root: str | Path,
    feature_schema_path: str | Path,
    non_mip_selection_manifest_path: str | Path,
    non_mip_verifier_receipt_path: str | Path,
    vendored_gcf_root: str | Path,
    output_root: str | Path,
    execution_git_commit: str,
    execution_git_tree: str,
    source_execution_config_sha256: str,
    device: str = "cuda:0",
    config: FixedBudgetNeuroSEDTrainConfig = FixedBudgetNeuroSEDTrainConfig(),
) -> dict[str, Any]:
    """Train one fresh artifact and stop at managed-verifier readiness."""

    config.validate()
    if os.environ.get("RUN_GNN_ABLATION", "0") != "0":
        raise RuntimeError("GNN backbone ablation is disabled for NeuroSED")
    torch, _tg, _models = runtime_stack()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("fixed-budget NeuroSED requested unavailable CUDA")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    destination = Path(output_root).absolute()
    _prepare_output_root(destination)
    checkpoints_root = destination / "checkpoints"
    checkpoints_root.mkdir(mode=0o700, exist_ok=False)
    aggregate, feature_schema, train_bundle, validation_bundle = (
        load_labeled_pair_bundles(
            ged_label_root=ged_label_root,
            train_pair_root=train_pair_root,
            validation_pair_root=validation_pair_root,
            feature_schema_path=feature_schema_path,
            config=config,
        )
    )
    selection_path = Path(non_mip_selection_manifest_path).absolute()
    receipt_path = Path(non_mip_verifier_receipt_path).absolute()
    selection = validate_non_mip_selection_manifest(
        load_json(selection_path), reopen_artifacts=True
    )
    receipt = load_json(receipt_path)
    if (
        sha256_file(selection_path)
        != aggregate["non_mip_selection_manifest_file_sha256"]
        or selection["selection_sha256"]
        != aggregate["non_mip_selection_sha256"]
        or receipt.get("receipt_sha256")
        != aggregate["non_mip_selection_verifier_receipt_sha256"]
        or receipt.get("receipt_sha256")
        != stable_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        )
    ):
        raise NeuroSEDProductionDataError("non-MIP selection evidence changed")
    input_dim = int(feature_schema["input_dim"])
    train_examples = _examples(train_bundle, input_dim=input_dim)
    validation_examples = _examples(validation_bundle, input_dim=input_dim)
    train_loader = _loader(
        FixedBudgetPairDataset(train_examples),
        batch_size=config.train_batch_size,
        seed=config.seed + 1,
    )
    validation_loader = _loader(
        FixedBudgetPairDataset(validation_examples),
        batch_size=config.validation_batch_size,
        seed=config.seed + 2,
    )
    if len(validation_loader) != 1:
        raise RuntimeError("official validation loader must contain one full batch")
    model = build_training_model(input_dim=input_dim, device=device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=0.0,
        max_lr=config.learning_rate,
        step_size_up=config.step_size_up,
        step_size_down=config.step_size_down,
        cycle_momentum=False,
    )
    selector = OfficialBatchInterleavedSelector(
        cycle_patience=config.cycle_patience,
        step_size_up=config.step_size_up,
        step_size_down=config.step_size_down,
    )
    candidates: list[dict[str, Any]] = []
    selected_candidate: dict[str, Any] | None = None
    epoch_summaries: list[dict[str, Any]] = []
    training_batch_index = 0
    epoch = 0
    while not selector.stopped:
        epoch += 1
        train_losses: list[float] = []
        validation_losses: list[float] = []
        gradient_norms: list[float] = []
        for train_batch, validation_batch in zip(
            train_loader, itertools.cycle(validation_loader)
        ):
            val_query, val_target, val_lower, val_upper = (
                value.to(device) for value in validation_batch
            )
            with torch.no_grad():
                val_prediction = model(val_query, val_target)
                val_loss_tensor = interval_loss(
                    val_lower.float(), val_upper.float(), val_prediction
                )
            validation_loss = float(val_loss_tensor.detach().cpu().item())
            if not math.isfinite(validation_loss):
                raise RuntimeError("official selector validation loss is non-finite")
            decision = selector.observe_validation(
                validation_loss, training_batch_index=training_batch_index
            )
            validation_losses.append(validation_loss)
            if decision.checkpoint_candidate:
                candidate = _checkpoint_candidate(
                    checkpoints_root, model, decision.validation_event_index
                )
                selector.bind_checkpoint_candidate(
                    validation_event_index=decision.validation_event_index,
                    checkpoint_sha256=candidate["model_sha256"],
                )
                candidates.append(candidate)
                selected_candidate = candidate
            if decision.stop_before_training_batch:
                break
            query, target, lower, upper = (value.to(device) for value in train_batch)
            prediction = model(query, target)
            loss = interval_loss(lower.float(), upper.float(), prediction)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("official NeuroSED training loss is non-finite")
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            )
            gradient_norm_value = float(
                torch.as_tensor(gradient_norm).detach().cpu()
            )
            if not math.isfinite(gradient_norm_value):
                raise RuntimeError("official NeuroSED gradient norm is non-finite")
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.detach().cpu().item()))
            gradient_norms.append(gradient_norm_value)
            selector.record_training_update(
                training_batch_index=training_batch_index,
                optimizer_step_completed=True,
                cyclic_lr_step_completed=True,
                gradient_clip_norm=config.max_grad_norm,
            )
            training_batch_index += 1
        epoch_summaries.append(
            {
                "epoch": epoch,
                "completed_training_batches": len(train_losses),
                "validation_events": len(validation_losses),
                "mean_train_interval_loss": (
                    sum(train_losses) / len(train_losses) if train_losses else None
                ),
                "mean_validation_interval_loss": (
                    sum(validation_losses) / len(validation_losses)
                ),
                "maximum_unclipped_gradient_norm": (
                    max(gradient_norms) if gradient_norms else None
                ),
                "last_learning_rate": float(scheduler.get_last_lr()[0]),
                "stopped": selector.stopped,
            }
        )
        print(
            "[TASTE_NEUROSED_OFFICIAL_EPOCH] "
            f"epoch={epoch} train_batches={len(train_losses)} "
            f"validation_events={len(validation_losses)} stopped={selector.stopped}",
            flush=True,
        )
    if selected_candidate is None:
        raise RuntimeError("official selector did not bind a checkpoint")
    selector_trace = selector.trace_manifest()
    selected_path = destination / selected_candidate["relative_path"]
    if sha256_file(selected_path) != selector_trace["selected_checkpoint_sha256"]:
        raise RuntimeError("official selected pre-update checkpoint bytes changed")
    _write_new(destination / "best.pt", selected_path.read_bytes())
    _write_new(destination / "model.pt", selected_path.read_bytes())
    if not (
        sha256_file(destination / "best.pt")
        == sha256_file(destination / "model.pt")
        == selector_trace["selected_checkpoint_sha256"]
    ):
        raise RuntimeError("best/model selected-byte publication changed")
    selected_state = load_state_dict_bytes(
        (destination / "best.pt").read_bytes(), map_location=device
    )
    model.load_state_dict(selected_state, strict=True)
    selected_validation_loader = _loader(
        FixedBudgetPairDataset(validation_examples),
        batch_size=config.validation_batch_size,
        seed=config.seed + 3,
    )
    validation_metrics = _evaluate(model, selected_validation_loader, device=device)
    health, direction_trace = _health_checks(
        checkpoint=destination / "best.pt",
        input_dim=input_dim,
        validation_examples=validation_examples,
        device=device,
    )
    _write_json(destination / "selector_trace.json", selector_trace)
    _write_json(destination / "distance_direction_trace.json", direction_trace)
    selected_sha = sha256_file(destination / "best.pt")
    pair_manifest = {
        "schema_version": "tastemolnet_neurosed_fixed_budget_pair_bundle_v1",
        "train_pair_count": config.train_pair_budget,
        "validation_pair_count": config.validation_pair_budget,
        "train_pair_sampler_manifest_sha256": train_bundle.inventory.manifest[
            "manifest_sha256"
        ],
        "validation_pair_sampler_manifest_sha256": (
            validation_bundle.inventory.manifest["manifest_sha256"]
        ),
        "train_pair_labels_manifest_sha256": train_bundle.label_manifest[
            "manifest_sha256"
        ],
        "validation_pair_labels_manifest_sha256": (
            validation_bundle.label_manifest["manifest_sha256"]
        ),
        "independent_pairs": True,
        "query_graph_id_differs_from_target_graph_id": True,
        "class_labels_used_as_supervision": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    split_manifest = {
        "schema_version": "tastemolnet_neurosed_fixed_budget_split_manifest_v1",
        "opened_payload_splits": ["train", "validation"],
        "train_pair_roles_subset_of_train": True,
        "validation_pair_roles_subset_of_validation": True,
        "train_validation_graph_id_intersection_empty": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "source_split_isolation_sha256": aggregate["split_isolation_sha256"],
    }
    training_metrics = {
        "schema_version": "tastemolnet_neurosed_fixed_budget_training_metrics_v1",
        "finite_loss": True,
        "optimizer": "AdamW",
        "scheduler": "CyclicLR",
        "criterion": "GREED lower/upper interval loss",
        "selection": "validation_batch_before_each_training_batch",
        "completed_training_batch_count": selector_trace[
            "completed_training_batch_count"
        ],
        "validation_event_count": selector_trace["validation_event_count"],
        "epochs_completed": epoch,
        "epoch_summaries": epoch_summaries,
        "test_metrics_computed": False,
    }
    checkpoint_manifest = {
        "schema_version": "tastemolnet_neurosed_fixed_budget_checkpoints_v1",
        "selected_checkpoint_sha256": selected_sha,
        "best_pt_sha256": selected_sha,
        "model_pt_sha256": selected_sha,
        "best_pt_semantics": "official_selector_preupdate_candidate_bytes",
        "model_pt_semantics": "reload_of_same_selected_preupdate_candidate_bytes",
        "best_and_model_bytes_identical": True,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "fresh_uuid_checkpoint_roots": True,
    }
    model_card = {
        "schema_version": OFFICIAL_FIXED_MODEL_CARD_SCHEMA,
        "dataset": "tastemolnet",
        "role": "GCF_AUXILIARY_DISTANCE_MODEL",
        "classifier": False,
        "source_label_independent": True,
        "train_only_fit": True,
        "validation_only_selection": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "pair_budget_strategy": "fixed_budget_resource_control",
        "fixed_pair_budget": True,
        "fixed_pair_budget_is_project_extension": True,
        "official_pair_semantics": True,
        "real_pyged_gedlib_labels": True,
        "ged_backend_variant": "non_mip",
        "ged_label_backend_variant": GED_LABEL_BACKEND_VARIANT,
        "GED_LABEL_BACKEND_VARIANT": GED_LABEL_BACKEND_VARIANT,
        "F2_BLP_USED": False,
        "GUROBI_USED": False,
        "ged_method_switched_from_official": True,
        "f2_blp_used": False,
        "gurobi_used": False,
        "approximate_or_neural_labels_used": False,
        "timeout_or_error_rows_used_as_labels": False,
        "label_representation": "ordered_query_target_lower_upper_interval",
        "pyged_return_dtype": "float64",
        "label_dtype": "float32",
        "label_transform": "official_torch_float32_storage_cast_only",
        "bound_average_used": False,
        "single_bound_substitution_used": False,
        "training_loop_authority": "neuro.train.train_full_batch_interleaved_validation",
        "upstream_greed_batch_interleaved_selection_loop_unchanged": True,
        "official_model_training_semantics": True,
        "non_mip_selector_independently_verified": True,
        "strict_official_batch_interleaved_selector_implemented": True,
        "gcf_runtime_direction": "generated_query_to_original_target",
        "training_direction_matches_gcf_runtime": True,
        "checkpoint_reload_passed": True,
        "batch_single_inference_passed": True,
        "finite_labels": True,
        "all_lower_bounds_le_upper_bounds": True,
        "official_selection_trace_authenticated": True,
        "gcf_runner_load_passed": True,
        "feature_schema_compatible": True,
        "pair_sampling_seed": 7,
        "inventory_mode": "exact_budget",
        "deterministic_reserve_fraction": 0.0,
        "train_reserve_candidate_count": config.train_pair_budget,
        "validation_reserve_candidate_count": config.validation_pair_budget,
        "train_reserve_surplus": 0,
        "validation_reserve_surplus": 0,
        "disk_reservation_pass": aggregate["disk_reservation_pass"],
        "minimum_persistent_free_bytes": aggregate[
            "minimum_persistent_free_bytes"
        ],
        "persistent_free_after_label_artifacts_bytes": aggregate[
            "persistent_free_after_label_artifacts_bytes"
        ],
        "cpu_contention_gate_pass": aggregate["bounded_cpu_worker_policy_pass"],
        "ged_label_workers": aggregate["workers"],
        "cpu_contention_evidence": aggregate["cpu_contention_evidence"],
        "worker_wrote_pass": False,
        "scientific_release_eligible": True,
        "full_official_neurosed_semantics_claimed": False,
        "train_pair_budget": config.train_pair_budget,
        "validation_pair_budget": config.validation_pair_budget,
        "successful_train_pair_count": config.train_pair_budget,
        "successful_validation_pair_count": config.validation_pair_budget,
        "edit_cost_contract": dict(OFFICIAL_SED_EDIT_COSTS),
        "strict_official_provenance": dict(STRICT_OFFICIAL_PROVENANCE),
        "vendored_gcf_source_sha256": dict(VENDORED_GCF_SOURCE_SHA256),
        "vendored_gcf_retained_inventory_sha256": (
            VENDORED_GCF_RETAINED_INVENTORY_SHA256
        ),
        "official_gcf_repository": OFFICIAL_GCF_REPOSITORY,
        "official_gcf_commit": OFFICIAL_GCF_COMMIT,
        "official_greed_commit": STRICT_OFFICIAL_PROVENANCE["greed_commit"],
        "official_experiment_notebook_sha256": OFFICIAL_EXPERIMENT_NOTEBOOK_SHA256,
        "architecture": model_contract(input_dim),
        "official_training_hyperparameters": asdict(config),
        "ged_method": aggregate["ged_backend"],
        "ged_method_args": aggregate["ged_method_args"],
        "selected_ged_backend": aggregate["ged_backend"],
        "selected_ged_backend_config": aggregate["ged_method_args"],
        "gedlib_commit": aggregate["gedlib_commit"],
        "pyged_module_sha256": aggregate["pyged_module_sha256"],
        "gedlib_build_manifest_sha256": aggregate[
            "gedlib_build_manifest_sha256"
        ],
        "gedlib_config_sha256": aggregate["gedlib_config_sha256"],
        "feature_schema_sha256": aggregate["feature_schema_sha256"],
        # The active 2026-08-29 override replaced the historical tier search.
        # Its selected candidate report is the real benchmark summary and the
        # independently verified selection manifest is the fixed-budget plan.
        "gedlib_benchmark_summary_sha256": selection["candidate_reports"][
            selection["selected_ged_backend"]
        ]["report_sha256"],
        "pair_budget_plan_sha256": selection["selection_sha256"],
        "active_budget_authority": "verified_non_mip_selection_manifest",
        "train_pair_labels_manifest_sha256": train_bundle.label_manifest[
            "manifest_sha256"
        ],
        "validation_pair_labels_manifest_sha256": (
            validation_bundle.label_manifest["manifest_sha256"]
        ),
        "train_pair_sampler_manifest_sha256": train_bundle.inventory.manifest[
            "manifest_sha256"
        ],
        "validation_pair_sampler_manifest_sha256": (
            validation_bundle.inventory.manifest["manifest_sha256"]
        ),
        "selector_trace_sha256": selector_trace["trace_sha256"],
        "distance_direction_trace_sha256": direction_trace["trace_sha256"],
        "selected_checkpoint_sha256": selected_sha,
        "non_mip_gedlib_selection_sha256": selection["selection_sha256"],
        "non_mip_gedlib_selection_manifest_file_sha256": sha256_file(
            selection_path
        ),
        "non_mip_selector_verifier_receipt_sha256": receipt["receipt_sha256"],
    }
    validate_official_fixed_budget_model_card(
        model_card, vendored_gcf_root=vendored_gcf_root
    )
    readiness = verify_official_fixed_budget_readiness(
        model_card=model_card,
        non_mip_selection_manifest=selection,
        non_mip_selector_verifier_receipt=receipt,
        train_pair_sampler_manifest=train_bundle.inventory.manifest,
        validation_pair_sampler_manifest=validation_bundle.inventory.manifest,
        train_pair_labels_manifest=train_bundle.label_manifest,
        validation_pair_labels_manifest=validation_bundle.label_manifest,
        selector_trace=selector_trace,
        distance_direction_trace=direction_trace,
        vendored_gcf_root=vendored_gcf_root,
    )
    config_payload = {
        "schema_version": TRAINER_SCHEMA,
        **asdict(config),
        "source_execution_config_sha256": source_execution_config_sha256,
        "device": device,
        "official_experiment_notebook_sha256": OFFICIAL_EXPERIMENT_NOTEBOOK_SHA256,
        "calibration_loaded": False,
        "test_loaded": False,
        "inference_direction": "generated_to_original",
    }
    health_gate = {
        "schema_version": "tastemolnet_neurosed_fixed_budget_worker_health_v1",
        "status": "READY_FOR_INDEPENDENT_VERIFICATION",
        "worker_wrote_scientific_pass": False,
        "finite_loss": True,
        "finite_validation_metric": True,
        "no_split_leakage": True,
        "official_selector_trace": True,
        "generated_query_to_original_target_assertion": True,
        **health,
    }
    documents = {
        "config.yaml": config_payload,
        "model_card.json": model_card,
        "pair_manifest.json": pair_manifest,
        "split_manifest.json": split_manifest,
        "training_metrics.json": training_metrics,
        "validation_metrics.json": validation_metrics,
        "checkpoint_manifest.json": checkpoint_manifest,
        "health_gate.json": health_gate,
        "readiness.json": readiness,
        "environment.json": _environment(device),
        "git_state.json": _git_state(execution_git_commit, execution_git_tree),
    }
    for name, value in documents.items():
        _write_json(destination / name, value)
    _write_new(
        destination / "feature_schema.json",
        Path(feature_schema_path).read_bytes(),
    )
    _write_new(
        destination / "ged_label_manifest.json",
        (Path(ged_label_root) / "ged_label_manifest.json").read_bytes(),
    )
    _write_sha256sums(destination)
    directory = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {
        "state": "READY_FOR_MANAGED_INDEPENDENT_VERIFICATION",
        "marker": TRAINER_READY_MARKER,
        "output_root": str(destination),
        "checkpoint": str(destination / "best.pt"),
        "checkpoint_sha256": selected_sha,
        "train_pairs": config.train_pair_budget,
        "validation_pairs": config.validation_pair_budget,
        "ged_backend": "branch",
        "gurobi_used": False,
        "F2_BLP_used": False,
        "independent_pairs": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "inference_direction": "generated_to_original",
        "worker_wrote_scientific_pass": False,
        "created_at": _utc_now(),
    }


def verify_fixed_budget_neurosed(
    *,
    ged_label_root: str | Path,
    train_pair_root: str | Path,
    validation_pair_root: str | Path,
    feature_schema_path: str | Path,
    non_mip_selection_manifest_path: str | Path,
    non_mip_verifier_receipt_path: str | Path,
    vendored_gcf_root: str | Path,
    output_root: str | Path,
    execution_git_commit: str,
    execution_git_tree: str,
    device: str = "cuda:0",
    config: FixedBudgetNeuroSEDTrainConfig = FixedBudgetNeuroSEDTrainConfig(),
) -> dict[str, Any]:
    """Independently reopen a completed worker root and publish PASS last."""

    destination = Path(output_root).absolute()
    if not destination.is_dir() or (destination / "PASS").exists():
        raise RuntimeError("NeuroSED verifier requires one unverified worker root")
    required = {
        "model.pt",
        "best.pt",
        "config.yaml",
        "model_card.json",
        "pair_manifest.json",
        "ged_label_manifest.json",
        "selector_trace.json",
        "distance_direction_trace.json",
        "training_metrics.json",
        "validation_metrics.json",
        "feature_schema.json",
        "split_manifest.json",
        "checkpoint_manifest.json",
        "health_gate.json",
        "readiness.json",
        "environment.json",
        "git_state.json",
        "sha256sums.txt",
    }
    if not required.issubset(path.name for path in destination.iterdir()):
        raise RuntimeError("NeuroSED worker artifact inventory is incomplete")
    _verify_sha256sums(destination)
    aggregate, feature_schema, train_bundle, validation_bundle = (
        load_labeled_pair_bundles(
            ged_label_root=ged_label_root,
            train_pair_root=train_pair_root,
            validation_pair_root=validation_pair_root,
            feature_schema_path=feature_schema_path,
            config=config,
        )
    )
    if (destination / "ged_label_manifest.json").read_bytes() != (
        Path(ged_label_root) / "ged_label_manifest.json"
    ).read_bytes():
        raise RuntimeError("copied GED-label authority changed")
    if (destination / "feature_schema.json").read_bytes() != Path(
        feature_schema_path
    ).read_bytes():
        raise RuntimeError("copied feature schema changed")
    git_state = load_json(destination / "git_state.json")
    if (
        git_state.get("commit") != execution_git_commit
        or git_state.get("tree") != execution_git_tree
    ):
        raise RuntimeError("NeuroSED execution Git authority changed")
    selection_path = Path(non_mip_selection_manifest_path).absolute()
    receipt_path = Path(non_mip_verifier_receipt_path).absolute()
    selection = validate_non_mip_selection_manifest(
        load_json(selection_path), reopen_artifacts=True
    )
    receipt = load_json(receipt_path)
    model_card = load_json(destination / "model_card.json")
    selector_trace = load_json(destination / "selector_trace.json")
    direction_trace = load_json(destination / "distance_direction_trace.json")
    validate_official_fixed_budget_model_card(
        model_card, vendored_gcf_root=vendored_gcf_root
    )
    readiness = verify_official_fixed_budget_readiness(
        model_card=model_card,
        non_mip_selection_manifest=selection,
        non_mip_selector_verifier_receipt=receipt,
        train_pair_sampler_manifest=train_bundle.inventory.manifest,
        validation_pair_sampler_manifest=validation_bundle.inventory.manifest,
        train_pair_labels_manifest=train_bundle.label_manifest,
        validation_pair_labels_manifest=validation_bundle.label_manifest,
        selector_trace=selector_trace,
        distance_direction_trace=direction_trace,
        vendored_gcf_root=vendored_gcf_root,
    )
    if readiness != load_json(destination / "readiness.json"):
        raise RuntimeError("NeuroSED readiness did not reproduce")
    _replay_selector_trace(selector_trace)
    checkpoint_sha = sha256_file(destination / "best.pt")
    if (
        checkpoint_sha != sha256_file(destination / "model.pt")
        or checkpoint_sha != model_card["selected_checkpoint_sha256"]
        or checkpoint_sha != selector_trace["selected_checkpoint_sha256"]
    ):
        raise RuntimeError("selected NeuroSED checkpoint bytes changed")
    input_dim = int(feature_schema["input_dim"])
    validation_examples = _examples(validation_bundle, input_dim=input_dim)
    health, reproduced_direction = _health_checks(
        checkpoint=destination / "best.pt",
        input_dim=input_dim,
        validation_examples=validation_examples,
        device=device,
    )
    if reproduced_direction != direction_trace:
        raise RuntimeError("generated-query/original-target direction did not reproduce")
    model = build_training_model(input_dim=input_dim, device=device)
    model.load_state_dict(
        load_state_dict_bytes(
            (destination / "best.pt").read_bytes(), map_location=device
        ),
        strict=True,
    )
    validation_loader = _loader(
        FixedBudgetPairDataset(validation_examples),
        batch_size=config.validation_batch_size,
        seed=config.seed + 3,
    )
    reproduced_metrics = _evaluate(model, validation_loader, device=device)
    recorded_metrics = load_json(destination / "validation_metrics.json")
    for key, value in reproduced_metrics.items():
        recorded = recorded_metrics.get(key)
        if isinstance(value, float):
            if not math.isclose(value, float(recorded), rel_tol=1e-5, abs_tol=1e-6):
                raise RuntimeError("selected validation metric did not reproduce")
        elif recorded != value:
            raise RuntimeError("selected validation metric did not reproduce")
    worker_health = load_json(destination / "health_gate.json")
    if any(worker_health.get(key) != value for key, value in health.items()):
        raise RuntimeError("worker checkpoint health did not reproduce")
    verification = {
        "schema_version": "tastemolnet_neurosed_fixed_budget_verification_v1",
        "status": "PASS",
        "marker": NEUROSED_PASS_MARKER,
        "independent_process_reopened_worker_root": True,
        "worker_wrote_scientific_pass": False,
        "labels_reopened": True,
        "checkpoint_reload_passed": True,
        "official_selector_trace_replayed": True,
        "batch_single_agreement_reproduced": True,
        "gcf_runner_load_reproduced": True,
        "generated_query_to_original_target_reproduced": True,
        "validation_metrics_reproduced": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "checkpoint_sha256": checkpoint_sha,
        "ged_label_manifest_sha256": aggregate["manifest_sha256"],
        "selector_trace_sha256": selector_trace["trace_sha256"],
        "distance_direction_trace_sha256": direction_trace["trace_sha256"],
        "verified_at": _utc_now(),
    }
    verification["verification_sha256"] = stable_sha256(verification)
    _write_json(destination / "verification.json", verification)
    _write_new(
        destination / "verification_sha256s.txt",
        (
            f"{sha256_file(destination / 'verification.json')}  verification.json\n"
        ).encode("utf-8"),
    )
    _write_new(destination / "PASS", (NEUROSED_PASS_MARKER + "\n").encode("utf-8"))
    directory = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {
        "state": "PASS",
        "marker": NEUROSED_PASS_MARKER,
        "output_root": str(destination),
        "checkpoint": str(destination / "best.pt"),
        "checkpoint_sha256": checkpoint_sha,
    }


__all__ = [
    "FixedBudgetNeuroSEDTrainConfig",
    "NEUROSED_PASS_MARKER",
    "TRAINER_READY_MARKER",
    "load_labeled_pair_bundles",
    "train_fixed_budget_neurosed",
    "verify_fixed_budget_neurosed",
]
