"""Independent health and bundle verification for TasteMolNet NeuroSED."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from types import ModuleType
from typing import Any, Mapping
import uuid

from src.data.tastemolnet_neurosed_pairs import (
    TastePairDataset,
    build_connected_bfs_pairs,
    derive_feature_schema,
    pair_manifest as build_pair_manifest,
    parse_taste_split_rows_bytes,
    rows_to_graphs,
)
from src.models.tastemolnet_neurosed import (
    build_runner_model,
    build_training_model,
    interval_loss,
    load_runner_checkpoint,
    model_contract,
    runtime_stack,
    verify_bundled_runner_sources,
)


REQUIRED_BUNDLE_FILES = frozenset(
    {
        "model.pt",
        "best.pt",
        "config.yaml",
        "model_card.json",
        "pair_manifest.json",
        "split_manifest.json",
        "training_metrics.json",
        "validation_metrics.json",
        "feature_schema.json",
        "environment.json",
        "git_state.json",
        "checkpoint_manifest.json",
        "health_gate.json",
        "authority_manifest.json",
        "controller_authority.json",
        "sha256sums.txt",
    }
)

STRICT_OFFICIAL_PROVENANCE = {
    "greed_commit": "1c756f49625abb62c9f6de5b0059876a4c7499c1",
    "datasets_py_sha256": (
        "aa1bab19394b2fcad4d6f1c45c5206f0485cc098dbd4742bf1396d229c0fa1ad"
    ),
    "train_py_sha256": (
        "8e4d425d9d63e0aa56d5a1e6e25738f511ca7b52b08ac297fcf2c1678bdf9e28"
    ),
    "pyged_cpp_sha256": (
        "55b35f952ea4070fad430d0911d29bfca21b4e10926e9bd7d56d2515d6499b16"
    ),
    "pair_builder": "neuro.datasets.make_inner_dataset",
    "sed_bounds": "neuro.datasets.inner_sed_via_pyged.sed",
    "selection_loop": "neuro.train.train_full_batch_interleaved_validation",
    "runtime_direction": "generated_query_to_original_parent_target",
}


class TasteNeuroSEDGateError(RuntimeError):
    """Raised when the independent verifier cannot close the bundle."""


def require_release_eligible_official_semantics(
    model_card: Mapping[str, Any],
) -> None:
    """Prevent a research-only Taste adaptation from receiving generic PASS."""

    if (
        model_card.get("scientific_release_eligible") is not True
        or model_card.get("full_official_neurosed_semantics_claimed") is not True
        or model_card.get("upstream_greed_pair_sampling_unchanged") is not True
        or model_card.get("training_direction_matches_gcf_runtime") is not True
        or model_card.get(
            "upstream_greed_batch_interleaved_selection_loop_unchanged"
        )
        is not True
        or model_card.get("strict_official_pair_builder_implemented") is not True
        or model_card.get("strict_official_pyged_bounds_authenticated") is not True
        or model_card.get(
            "strict_official_batch_interleaved_selector_implemented"
        )
        is not True
        or model_card.get("strict_official_provenance")
        != STRICT_OFFICIAL_PROVENANCE
    ):
        raise TasteNeuroSEDGateError(
            "strict official NeuroSED pair/pyged/selector semantics are not "
            "implemented; the Taste directional adaptation is not release eligible"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteNeuroSEDGateError(f"{path.name} is not JSON") from exc
    if not isinstance(payload, dict):
        raise TasteNeuroSEDGateError(f"{path.name} is not one JSON object")
    return payload


def _synthetic_graphs(input_dim: int) -> tuple[list[Any], list[Any]]:
    torch, _tg, _gcf_models = runtime_stack()
    from torch_geometric.data import Data

    if input_dim <= 0:
        raise TasteNeuroSEDGateError("NeuroSED input_dim must be positive")

    def one_hot(indices: list[int]) -> Any:
        x = torch.zeros((len(indices), input_dim), dtype=torch.float32)
        for row, index in enumerate(indices):
            x[row, index % input_dim] = 1.0
        return x

    query_a = Data(
        x=one_hot([0, 1]),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        num_nodes=2,
    )
    parent_a = Data(
        x=one_hot([0, 1, 2]),
        edge_index=torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        ),
        num_nodes=3,
    )
    query_b = Data(
        x=one_hot([1, 2, 0]),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
        ),
        num_nodes=3,
    )
    parent_b = Data(
        x=one_hot([1, 2, 0, 1]),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        ),
        num_nodes=4,
    )
    return [query_a, query_b], [parent_a, parent_b]


def _predict_pairs(model: Any, queries: list[Any], parents: list[Any], device: str) -> Any:
    torch, tg, _gcf_models = runtime_stack()
    query_batch = tg.data.Batch.from_data_list(queries).to(device)
    parent_batch = tg.data.Batch.from_data_list(parents).to(device)
    with torch.no_grad():
        return model(query_batch, parent_batch).detach().cpu()


def _load_with_bundled_distance_runner(
    checkpoint: Path, *, original_graphs: list[Any], device: str
) -> Any:
    """Call the reviewed fork's actual ``distance.py::load_neurosed``.

    The upstream file uses the historical top-level import name
    ``neurosed.models``.  Install that alias only for the duration of this
    held-source import, then restore the interpreter exactly as it was.
    """

    _torch, _tg, gcf_models = runtime_stack()
    distance_path = Path(gcf_models.__file__).resolve(strict=True).parents[1] / "distance.py"
    package = ModuleType("neurosed")
    package.__path__ = []  # type: ignore[attr-defined]
    package.models = gcf_models  # type: ignore[attr-defined]
    displaced_package = sys.modules.get("neurosed")
    displaced_models = sys.modules.get("neurosed.models")
    probe_name = f"_taste_neurosed_distance_probe_{uuid.uuid4().hex}"
    try:
        sys.modules["neurosed"] = package
        sys.modules["neurosed.models"] = gcf_models
        spec = importlib.util.spec_from_file_location(probe_name, distance_path)
        if spec is None or spec.loader is None:
            raise TasteNeuroSEDGateError("cannot load bundled GCF distance.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.load_neurosed(
            original_graphs,
            neurosed_model_path=str(checkpoint),
            device=device,
        )
    finally:
        sys.modules.pop(probe_name, None)
        if displaced_package is None:
            sys.modules.pop("neurosed", None)
        else:
            sys.modules["neurosed"] = displaced_package
        if displaced_models is None:
            sys.modules.pop("neurosed.models", None)
        else:
            sys.modules["neurosed.models"] = displaced_models


def checkpoint_health(
    checkpoint: str | Path,
    *,
    input_dim: int,
    require_cuda_tolerance: bool,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Independently exercise training reload and GCF runner load semantics."""

    torch, _tg, _gcf_models = runtime_stack()
    runner_sources = verify_bundled_runner_sources()
    checkpoint_path = Path(checkpoint)
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older AutoDL torch.
        state = torch.load(checkpoint_path, map_location="cpu")
    training_model = build_training_model(input_dim=input_dim, device="cpu")
    training_result = training_model.load_state_dict(state, strict=True)
    if training_result.missing_keys or training_result.unexpected_keys:
        raise TasteNeuroSEDGateError("training checkpoint reload is not strict")
    training_model.eval()

    queries, parents = _synthetic_graphs(input_dim)
    runner = _load_with_bundled_distance_runner(
        checkpoint_path, original_graphs=parents, device="cpu"
    )
    with torch.no_grad():
        runner_outer = runner.predict_outer_with_queries(
            queries, batch_size=len(queries)
        ).detach().cpu()
    if runner_outer.shape != (len(queries), len(parents)) or not bool(
        torch.isfinite(runner_outer).all().item()
    ):
        raise TasteNeuroSEDGateError(
            "GCF load/embed_targets/predict_outer_with_queries contract failed"
        )
    batch_prediction = _predict_pairs(runner, queries, parents, "cpu")
    single_predictions = torch.cat(
        [
            _predict_pairs(runner, [query], [parent], "cpu")
            for query, parent in zip(queries, parents)
        ]
    )
    batch_single_max_abs = float(
        torch.max(torch.abs(batch_prediction - single_predictions)).item()
    )
    finite_distances = bool(torch.isfinite(batch_prediction).all().item())
    if batch_single_max_abs > tolerance or not finite_distances:
        raise TasteNeuroSEDGateError(
            "GCF runner batch/single or finite-distance health check failed"
        )

    training_prediction = _predict_pairs(training_model, queries, parents, "cpu")
    if not bool(torch.isfinite(training_prediction).all().item()):
        raise TasteNeuroSEDGateError("NormSED training reload produced non-finite values")

    cpu_gpu_status = "CUDA_NOT_AVAILABLE"
    cpu_gpu_max_abs: float | None = None
    if torch.cuda.is_available():
        gpu_runner = build_runner_model(input_dim=input_dim, device="cuda:0")
        gpu_runner.load_state_dict(state, strict=True)
        gpu_runner.eval()
        gpu_prediction = _predict_pairs(gpu_runner, queries, parents, "cuda:0")
        cpu_gpu_max_abs = float(
            torch.max(torch.abs(batch_prediction - gpu_prediction)).item()
        )
        if not math.isfinite(cpu_gpu_max_abs) or cpu_gpu_max_abs > tolerance:
            raise TasteNeuroSEDGateError("NeuroSED CPU/GPU tolerance check failed")
        cpu_gpu_status = "PASS"
    elif require_cuda_tolerance:
        raise TasteNeuroSEDGateError(
            "CUDA is required for the formal NeuroSED CPU/GPU health gate"
        )

    raw = [float(value) for value in batch_prediction.tolist()]
    denominators = [
        float(
            int(query.num_nodes)
            + int(query.num_edges) / 2
            + int(parent.num_nodes)
            + int(parent.num_edges) / 2
        )
        for query, parent in zip(queries, parents)
    ]
    normalized = [value / denominator for value, denominator in zip(raw, denominators)]
    if not all(math.isfinite(value) for value in normalized):
        raise TasteNeuroSEDGateError("normalized GCF distances are non-finite")

    return {
        "schema_version": "tastemolnet_gcf_neurosed_checkpoint_health_v1",
        "checkpoint_reload": True,
        "training_normsed_strict_load": True,
        "gcf_runner_normged_strict_load": True,
        "gcf_runner_can_load": True,
        "gcf_distance_py_load_neurosed": True,
        "gcf_runner_embed_targets": True,
        "gcf_runner_predict_outer_with_queries": True,
        "state_dict_isomorphic": True,
        "finite_distances": finite_distances,
        "finite_training_forward": True,
        "batch_single_agreement": True,
        "batch_single_max_abs": batch_single_max_abs,
        "cpu_gpu_numeric_tolerance": cpu_gpu_status,
        "cpu_gpu_max_abs": cpu_gpu_max_abs,
        "numeric_tolerance": float(tolerance),
        "distance_normalization": "divide_by_sum_graph_element_counts",
        "normalized_distances_finite": True,
        "probe_uses_synthetic_graphs_only": True,
        "bundled_runner_sources": runner_sources,
    }


def _parse_sha256sums(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise TasteNeuroSEDGateError("sha256sums.txt is malformed")
        candidate = PurePosixPath(relative)
        if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
            raise TasteNeuroSEDGateError("sha256sums.txt contains an unsafe path")
        if relative in values:
            raise TasteNeuroSEDGateError("sha256sums.txt repeats a path")
        values[relative] = digest
    return values


def _scientific_files(root: Path) -> set[str]:
    paths: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode):
            raise TasteNeuroSEDGateError(f"bundle contains symlink: {relative}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise TasteNeuroSEDGateError(
                f"bundle contains unsafe file identity: {relative}"
            )
        if relative == "sha256sums.txt" or path.name == ".generation_token.json":
            continue
        paths.add(relative)
    return paths


def _finite_metric(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TasteNeuroSEDGateError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TasteNeuroSEDGateError(f"{label} is not finite")
    return result


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def _spearman(left_values: list[float], right_values: list[float]) -> float:
    left = _average_ranks(left_values)
    right = _average_ranks(right_values)
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


def recompute_validation_metrics(
    checkpoint: str | Path,
    *,
    validation_csv_bytes: bytes,
    feature_schema: Mapping[str, Any],
    config: Mapping[str, Any],
    published_pair_manifest: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    """Rebuild held validation pairs and independently rescore ``best.pt``."""

    torch, _tg, _gcf_models = runtime_stack()
    from torch_geometric.loader import DataLoader

    rows, row_evidence = parse_taste_split_rows_bytes(
        validation_csv_bytes, expected_split="validation"
    )
    graphs = rows_to_graphs(rows, feature_schema)
    pairs = build_connected_bfs_pairs(
        graphs,
        split="validation",
        num_pairs=int(config["validation_pairs"]),
        seed=int(config["seed"]) + 1,
    )
    rebuilt_pair_manifest = build_pair_manifest(pairs, split="validation")
    for name in (
        "pair_count",
        "pair_ids_hash",
        "parent_graph_ids_hash",
        "minimum_edit_count",
        "maximum_edit_count",
        "mean_edit_count",
    ):
        observed = rebuilt_pair_manifest[name]
        expected = published_pair_manifest.get(name)
        if isinstance(observed, float):
            if not isinstance(expected, (int, float)) or not math.isclose(
                observed, float(expected), rel_tol=0.0, abs_tol=1e-12
            ):
                raise TasteNeuroSEDGateError(
                    f"rebuilt validation pair manifest differs: {name}"
                )
        elif observed != expected:
            raise TasteNeuroSEDGateError(
                f"rebuilt validation pair manifest differs: {name}"
            )
    loader = DataLoader(
        TastePairDataset(pairs),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    model = build_training_model(
        input_dim=int(feature_schema["input_dim"]), device=device
    )
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - older AutoDL torch.
        state = torch.load(checkpoint, map_location=device)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise TasteNeuroSEDGateError("selected checkpoint NormSED reload changed")
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    losses: list[float] = []
    with torch.no_grad():
        for ordered_source, ordered_target, lb, ub in loader:
            ordered_source = ordered_source.to(device)
            ordered_target = ordered_target.to(device)
            lb = lb.to(device=device, dtype=torch.float32)
            ub = ub.to(device=device, dtype=torch.float32)
            prediction = model(ordered_source, ordered_target)
            loss = interval_loss(lb, ub, prediction)
            losses.append(float(loss.detach().cpu().item()))
            predictions.extend(
                float(value) for value in prediction.detach().cpu().tolist()
            )
            targets.extend(float(value) for value in lb.detach().cpu().tolist())
    values = predictions + targets + losses
    if not predictions or not all(math.isfinite(value) for value in values):
        raise TasteNeuroSEDGateError("independent validation replay is non-finite")
    errors = [prediction - target for prediction, target in zip(predictions, targets)]
    mse = sum(error * error for error in errors) / len(errors)
    return {
        "schema_version": "tastemolnet_gcf_neurosed_validation_replay_v1",
        "pair_count": len(predictions),
        "interval_loss": sum(losses) / len(losses),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(mse),
        "spearman_rank": _spearman(predictions, targets),
        "minimum_distance": min(predictions),
        "maximum_distance": max(predictions),
        "validation_csv_sha256": hashlib.sha256(validation_csv_bytes).hexdigest(),
        "validation_graph_ids_hash": row_evidence["graph_ids_hash"],
        "rebuilt_pair_ids_hash": rebuilt_pair_manifest["pair_ids_hash"],
        "selected_best_checkpoint_reloaded": True,
        "deterministic_validation_pairs_rebuilt": True,
        "calibration_loaded": False,
        "test_loaded": False,
    }


def _require_uuid4(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise TasteNeuroSEDGateError(f"{label} is not a UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise TasteNeuroSEDGateError(f"{label} is invalid") from exc
    if parsed.version != 4 or parsed.variant != uuid.RFC_4122 or str(parsed) != value:
        raise TasteNeuroSEDGateError(f"{label} is not canonical RFC-4122 UUIDv4")
    return value


_CONTROLLER_STABLE_FIELDS = (
    "controller_id",
    "controller_uuid",
    "pid",
    "pid_start_ticks",
    "boot_id",
    "exe",
    "command_hash",
    "cwd",
    "cgroup",
    "git_commit",
    "git_tree",
    "receipt_path",
    "receipt_sha256",
)


def validate_controller_heartbeat_progression(
    bundled_controller: Mapping[str, Any],
    verifier_authority: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate H1 -> worker-latest -> verifier-terminal without equating hashes.

    The immutable worker-attempt input is H1.  The worker holder may legally
    advance to a later generation while training, and the verifier holder may
    advance again.  Receipt/process identity must remain stable and heartbeat
    sequences must be monotonic; heartbeat hashes are deliberately *not*
    required to be equal.
    """

    worker_latest = bundled_controller.get("worker_latest")
    worker_initial = bundled_controller.get("worker_initial_heartbeat")
    if (
        bundled_controller.get("schema_version")
        != "tastemolnet_gcf_neurosed_controller_binding_v2"
        or type(worker_latest) is not dict
        or type(worker_initial) is not dict
        or any(
            worker_latest.get(field) != verifier_authority.get(field)
            for field in _CONTROLLER_STABLE_FIELDS
        )
        or worker_latest.get("state") != "RUNNING"
        or worker_latest.get("controller_state") != "MONITORING"
        or verifier_authority.get("state") != "RUNNING"
        or verifier_authority.get("controller_state") != "MONITORING"
        or type(worker_initial.get("sequence")) is not int
        or type(worker_latest.get("sequence")) is not int
        or type(verifier_authority.get("sequence")) is not int
        or int(worker_latest["sequence"]) < int(worker_initial["sequence"])
        or int(verifier_authority["sequence"]) < int(worker_latest["sequence"])
        or worker_initial.get("receipt_sha256")
        != worker_latest.get("receipt_sha256")
        or not isinstance(worker_initial.get("heartbeat_sha256"), str)
        or not isinstance(worker_latest.get("heartbeat_sha256"), str)
        or not isinstance(verifier_authority.get("heartbeat_sha256"), str)
    ):
        raise TasteNeuroSEDGateError("held controller receipt/heartbeat changed")
    return dict(worker_initial), dict(worker_latest)


def verify_bundle(
    bundle_root: str | Path,
    *,
    require_cuda_tolerance: bool,
    train_csv_bytes: bytes,
    validation_csv_bytes: bytes,
    authoritative_lineage: Mapping[str, Any],
    controller_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently verify one SEALED scientific bundle before publication."""

    root = Path(bundle_root)
    if not root.is_absolute() or root.resolve(strict=True) != root:
        raise TasteNeuroSEDGateError("bundle root must be one physical absolute directory")
    names = {path.name for path in root.iterdir() if path.is_file()}
    missing = sorted(REQUIRED_BUNDLE_FILES - names)
    if missing:
        raise TasteNeuroSEDGateError(f"NeuroSED bundle files are missing: {missing}")
    top_level_names = {path.name for path in root.iterdir()}
    allowed_top_level = set(REQUIRED_BUNDLE_FILES) | {
        "checkpoints",
        ".generation_token.json",
    }
    if not top_level_names <= allowed_top_level or "checkpoints" not in top_level_names:
        raise TasteNeuroSEDGateError("NeuroSED top-level bundle inventory changed")

    checksums = _parse_sha256sums(root / "sha256sums.txt")
    scientific_files = _scientific_files(root)
    if set(checksums) != scientific_files:
        raise TasteNeuroSEDGateError("sha256sums inventory differs from scientific files")
    for relative, expected in checksums.items():
        if _sha256_file(root / relative) != expected:
            raise TasteNeuroSEDGateError(f"bundle SHA256 changed: {relative}")

    model_card = _read_json(root / "model_card.json")
    feature_schema = _read_json(root / "feature_schema.json")
    split_manifest = _read_json(root / "split_manifest.json")
    pair_manifest = _read_json(root / "pair_manifest.json")
    training = _read_json(root / "training_metrics.json")
    validation = _read_json(root / "validation_metrics.json")
    checkpoint_manifest = _read_json(root / "checkpoint_manifest.json")
    worker_health = _read_json(root / "health_gate.json")
    config = _read_json(root / "config.yaml")
    git_state = _read_json(root / "git_state.json")
    bundled_authority = _read_json(root / "authority_manifest.json")
    bundled_controller = _read_json(root / "controller_authority.json")

    if (
        model_card.get("dataset") != "tastemolnet"
        or model_card.get("role") != "GCF_AUXILIARY_DISTANCE_MODEL"
        or model_card.get("classifier") is not False
        or model_card.get("source_label_independent") is not True
        or model_card.get("train_only_fit") is not True
        or model_card.get("validation_only_selection") is not True
        or model_card.get("training_loop_authority")
        != "reviewed_taste_epoch_level_adaptation_v1"
        or model_card.get(
            "upstream_greed_batch_interleaved_selection_loop_unchanged"
        )
        is not False
        or model_card.get("upstream_greed_pair_sampling_unchanged") is not False
        or model_card.get("upstream_greed_pair_sampling")
        != "independent_query_subgraph_to_random_target_with_pyged_bounds"
        or model_card.get("gcf_runtime_direction")
        != "generated_query_to_original_parent_target"
        or model_card.get("training_direction_matches_gcf_runtime") is not False
        or model_card.get("full_official_neurosed_semantics_claimed") is not False
        or model_card.get("scientific_release_eligible") is not False
        or model_card.get("strict_official_pair_builder_implemented") is not False
        or model_card.get("strict_official_pyged_bounds_authenticated") is not False
        or model_card.get(
            "strict_official_batch_interleaved_selector_implemented"
        )
        is not False
        or model_card.get("strict_official_provenance")
        != STRICT_OFFICIAL_PROVENANCE
        or model_card.get("calibration_loaded") is not False
        or model_card.get("test_loaded") is not False
        or split_manifest.get("opened_payload_splits") != ["train", "validation"]
        or split_manifest.get("calibration_loaded") is not False
        or split_manifest.get("test_loaded") is not False
        or split_manifest.get("calibration_intersection_empty") is not True
        or split_manifest.get("test_intersection_empty") is not True
        or split_manifest.get("calibration_graph_hashes_observed") is not False
        or split_manifest.get("test_graph_hashes_observed") is not False
        or feature_schema.get("train_derived_only") is not True
        or feature_schema.get("validation_unseen_atomic_numbers") != []
        or pair_manifest.get("calibration_pair_count") != 0
        or pair_manifest.get("test_pair_count") != 0
        or config.get("seed") != 7
        or config.get("learning_rate") != 0.001
        or config.get("weight_decay") != 0.001
        or config.get("cyclic_step_size_up") != 2000
        or config.get("cyclic_step_size_down") != 2000
        or config.get("max_grad_norm") != 0.1
        or training.get("optimizer") != "AdamW"
        or training.get("scheduler") != "CyclicLR"
        or training.get("gradient_clip_norm") != 0.1
        or training.get("criterion") != "GREED interval loss"
        or git_state.get("worktree_clean") is not True
        or git_state.get("cleanliness_verified_by_launcher_before_managed_attempt")
        is not True
    ):
        raise TasteNeuroSEDGateError("NeuroSED model/data boundary contract changed")
    for name in ("commit", "tree"):
        value = git_state.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise TasteNeuroSEDGateError(f"NeuroSED execution git {name} is invalid")
    train_pairs = pair_manifest.get("train")
    validation_pairs = pair_manifest.get("validation")
    if (
        not isinstance(train_pairs, dict)
        or not isinstance(validation_pairs, dict)
        or train_pairs.get("split") != "train"
        or validation_pairs.get("split") != "validation"
        or train_pairs.get("pair_builder")
        != "taste_connected_induced_bfs_nested_pair_v1"
        or validation_pairs.get("pair_builder")
        != "taste_connected_induced_bfs_nested_pair_v1"
        or train_pairs.get("pair_direction")
        != "parent_to_connected_induced_bfs_subgraph"
        or validation_pairs.get("pair_direction")
        != "parent_to_connected_induced_bfs_subgraph"
        or train_pairs.get("upstream_greed_pair_sampling_unchanged") is not False
        or validation_pairs.get("upstream_greed_pair_sampling_unchanged")
        is not False
        or train_pairs.get("upstream_greed_pair_sampling")
        != "independent_query_subgraph_to_random_target_with_pyged_bounds"
        or validation_pairs.get("upstream_greed_pair_sampling")
        != train_pairs.get("upstream_greed_pair_sampling")
        or train_pairs.get("gcf_runtime_direction")
        != "generated_query_to_original_parent_target"
        or validation_pairs.get("gcf_runtime_direction")
        != train_pairs.get("gcf_runtime_direction")
        or train_pairs.get("training_direction_matches_gcf_runtime") is not False
        or validation_pairs.get("training_direction_matches_gcf_runtime")
        is not False
        or train_pairs.get("full_official_pair_semantics_claimed") is not False
        or validation_pairs.get("full_official_pair_semantics_claimed") is not False
        or train_pairs.get("reverse_subgraph_to_parent_cost_is_zero") is not True
        or validation_pairs.get("reverse_subgraph_to_parent_cost_is_zero")
        is not True
        or train_pairs.get("edit_cost_contract")
        != {
            "node_insertion": 0,
            "node_deletion": 1,
            "edge_insertion": 0,
            "edge_deletion": 1,
            "node_relabel": 1,
            "edge_relabel": 0,
        }
        or validation_pairs.get("edit_cost_contract")
        != train_pairs.get("edit_cost_contract")
        or train_pairs.get("interval_bounds_exact") is not True
        or validation_pairs.get("interval_bounds_exact") is not True
        or train_pairs.get("connected_queries") is not True
        or validation_pairs.get("connected_queries") is not True
        or train_pairs.get("cross_parent_pairs") is not False
        or validation_pairs.get("cross_parent_pairs") is not False
        or pair_manifest.get("labels_used") is not False
        or pair_manifest.get("train_pair_count") != config.get("train_pairs")
        or pair_manifest.get("validation_pair_count")
        != config.get("validation_pairs")
        or train_pairs.get("pair_count") != config.get("train_pairs")
        or validation_pairs.get("pair_count") != config.get("validation_pairs")
    ):
        raise TasteNeuroSEDGateError("NeuroSED exact pair-builder contract changed")

    input_dim = feature_schema.get("input_dim")
    if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
        raise TasteNeuroSEDGateError("NeuroSED feature input_dim is invalid")
    atom_vocabulary = feature_schema.get("feature_atomic_numbers")
    if (
        not isinstance(atom_vocabulary, list)
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in atom_vocabulary)
        or atom_vocabulary != sorted(set(atom_vocabulary))
        or len(atom_vocabulary) != input_dim
        or feature_schema.get("node_feature_semantics") != "one_hot_atomic_number"
        or feature_schema.get("explicit_h_nodes") is not True
        or feature_schema.get("edge_features_used") is not False
    ):
        raise TasteNeuroSEDGateError("NeuroSED feature schema is invalid")
    if model_card.get("architecture") != model_contract(input_dim):
        raise TasteNeuroSEDGateError("NeuroSED official architecture authority changed")
    if (
        config.get("architecture") != model_contract(input_dim)
        or config.get("optimizer") != "AdamW"
        or config.get("scheduler") != "CyclicLR"
        or config.get("criterion") != "GREED interval loss"
        or config.get("pair_builder")
        != "taste_connected_induced_bfs_nested_pair_v1"
        or config.get("pair_semantics") != "directional_exact_deletion_v1"
        or model_card.get("pair_semantics")
        != "directional_exact_deletion_v1"
    ):
        raise TasteNeuroSEDGateError("NeuroSED resolved execution config changed")

    _finite_metric(training.get("best_validation_interval_loss"), label="best validation loss")
    _finite_metric(validation.get("interval_loss"), label="validation interval loss")
    _finite_metric(validation.get("mae"), label="validation MAE")
    _finite_metric(validation.get("rmse"), label="validation RMSE")
    _finite_metric(validation.get("spearman_rank"), label="validation rank metric")
    if worker_health.get("status") != "READY_FOR_INDEPENDENT_VERIFICATION":
        raise TasteNeuroSEDGateError("worker health evidence is not raw-ready")

    selected_id = _require_uuid4(
        checkpoint_manifest.get("selected_checkpoint_uuid"),
        label="selected checkpoint UUID",
    )
    checkpoints = checkpoint_manifest.get("checkpoints")
    if (
        not isinstance(checkpoints, list)
        or checkpoint_manifest.get("checkpoint_count") != len(checkpoints)
        or not checkpoints
        or checkpoint_manifest.get("uuid_paths_never_reused") is not True
        or checkpoint_manifest.get("fixed_checkpoint_directory_used") is not False
    ):
        raise TasteNeuroSEDGateError("checkpoint manifest inventory is invalid")
    observed_checkpoint_ids: set[str] = set()
    observed_checkpoint_epochs: list[int] = []
    observed_selection_keys: list[tuple[float, float]] = []
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, dict):
            raise TasteNeuroSEDGateError("checkpoint inventory row is not a mapping")
        checkpoint_id = _require_uuid4(
            checkpoint.get("checkpoint_uuid"), label=f"checkpoint[{index}] UUID"
        )
        if checkpoint_id in observed_checkpoint_ids:
            raise TasteNeuroSEDGateError("checkpoint UUID is reused")
        observed_checkpoint_ids.add(checkpoint_id)
        epoch = checkpoint.get("epoch")
        validation_row = checkpoint.get("validation_metrics")
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch <= 0
            or not isinstance(validation_row, dict)
        ):
            raise TasteNeuroSEDGateError("checkpoint epoch/validation row is invalid")
        observed_checkpoint_epochs.append(epoch)
        observed_selection_keys.append(
            (
                _finite_metric(
                    validation_row.get("interval_loss"),
                    label=f"checkpoint[{index}] validation interval loss",
                ),
                _finite_metric(
                    validation_row.get("mae"),
                    label=f"checkpoint[{index}] validation MAE",
                ),
            )
        )
        relative = f"checkpoints/{checkpoint_id}/model.pt"
        manifest_relative = f"checkpoints/{checkpoint_id}/checkpoint.json"
        if (
            checkpoint.get("relative_path") != relative
            or checkpoint.get("manifest_relative_path") != manifest_relative
            or relative not in checksums
            or manifest_relative not in checksums
            or checkpoint.get("model_sha256") != checksums[relative]
        ):
            raise TasteNeuroSEDGateError("checkpoint UUID/path/hash binding changed")
        leaf = _read_json(root / manifest_relative)
        if (
            leaf.get("checkpoint_uuid") != checkpoint_id
            or leaf.get("model_sha256") != checksums[relative]
            or leaf.get("path_reuse_forbidden") is not True
            or leaf.get("selected_using")
            != "validation_interval_loss_then_validation_mae"
        ):
            raise TasteNeuroSEDGateError("checkpoint leaf manifest changed")
        leaf_names = {
            path.name for path in (root / "checkpoints" / checkpoint_id).iterdir()
        }
        if leaf_names != {"model.pt", "checkpoint.json"}:
            raise TasteNeuroSEDGateError("checkpoint leaf file inventory changed")
    checkpoint_directories = {
        path.name for path in (root / "checkpoints").iterdir() if path.is_dir()
    }
    if checkpoint_directories != observed_checkpoint_ids:
        raise TasteNeuroSEDGateError("checkpoint directory inventory differs")
    if selected_id not in observed_checkpoint_ids:
        raise TasteNeuroSEDGateError("selected checkpoint is absent from inventory")
    if (
        observed_checkpoint_epochs != sorted(set(observed_checkpoint_epochs))
        or selected_id != checkpoints[-1].get("checkpoint_uuid")
        or any(
            current >= previous
            for previous, current in zip(
                observed_selection_keys, observed_selection_keys[1:]
            )
        )
    ):
        raise TasteNeuroSEDGateError("validation-selected checkpoint order changed")
    selected_validation = checkpoints[-1]["validation_metrics"]
    if (
        validation.get("selected_checkpoint_uuid") != selected_id
        or validation.get("checkpoint_selection_only") is not True
        or validation.get("test_used_for_selection") is not False
        or training.get("selection_split") != "validation"
        or training.get("test_metrics_computed") is not False
        or training.get("best_epoch") != checkpoints[-1].get("epoch")
        or float(training["best_validation_interval_loss"])
        != float(selected_validation["interval_loss"])
        or any(
            float(validation[name]) != float(selected_validation[name])
            for name in (
                "interval_loss",
                "mae",
                "rmse",
                "spearman_rank",
            )
        )
    ):
        raise TasteNeuroSEDGateError("selected validation metrics binding changed")
    selected_relative = checkpoint_manifest.get("selected_checkpoint_relative_path")
    expected_relative = f"checkpoints/{selected_id}/model.pt"
    if selected_relative != expected_relative or selected_relative not in checksums:
        raise TasteNeuroSEDGateError("selected UUID checkpoint binding changed")
    if checksums["best.pt"] != checksums[selected_relative]:
        raise TasteNeuroSEDGateError("best.pt differs from selected UUID checkpoint")

    lineage_train = authoritative_lineage.get("train")
    lineage_validation = authoritative_lineage.get("validation")
    if (
        bundled_authority != dict(authoritative_lineage)
        or checksums["authority_manifest.json"]
        != hashlib.sha256(
            (root / "authority_manifest.json").read_bytes()
        ).hexdigest()
        or authoritative_lineage.get("schema_version")
        != "tastemolnet_gcf_neurosed_authority_v2"
        or authoritative_lineage.get("t3_split_manifest_byte_identical_to_t2")
        is not True
        or authoritative_lineage.get("calibration_loaded") is not False
        or authoritative_lineage.get("test_loaded") is not False
        or type(lineage_train) is not dict
        or type(lineage_validation) is not dict
        or lineage_train.get("source_sha256")
        != split_manifest.get("train_source_csv_sha256")
        or hashlib.sha256(train_csv_bytes).hexdigest()
        != lineage_train.get("source_sha256")
        or lineage_validation.get("source_sha256")
        != split_manifest.get("validation_source_csv_sha256")
        or hashlib.sha256(validation_csv_bytes).hexdigest()
        != lineage_validation.get("source_sha256")
    ):
        raise TasteNeuroSEDGateError("held authoritative validation lineage changed")
    replay_train_rows, _replay_train_rows_evidence = parse_taste_split_rows_bytes(
        train_csv_bytes, expected_split="train"
    )
    replay_validation_rows, _replay_validation_rows_evidence = (
        parse_taste_split_rows_bytes(
            validation_csv_bytes, expected_split="validation"
        )
    )
    replay_feature_schema = derive_feature_schema(
        replay_train_rows, replay_validation_rows
    )
    if replay_feature_schema != feature_schema:
        raise TasteNeuroSEDGateError(
            "held train/validation feature vocabulary differs from worker"
        )
    replay_train_graphs = rows_to_graphs(replay_train_rows, replay_feature_schema)
    replay_train_pairs = build_connected_bfs_pairs(
        replay_train_graphs,
        split="train",
        num_pairs=int(config["train_pairs"]),
        seed=int(config["seed"]),
    )
    if build_pair_manifest(replay_train_pairs, split="train") != train_pairs:
        raise TasteNeuroSEDGateError(
            "independently rebuilt train pair manifest differs from worker"
        )
    bundled_controller_initial, bundled_controller_latest = (
        validate_controller_heartbeat_progression(
            bundled_controller,
            controller_authority,
        )
    )
    replay_device = "cuda:0" if require_cuda_tolerance else "cpu"
    validation_replay = recompute_validation_metrics(
        root / "best.pt",
        validation_csv_bytes=validation_csv_bytes,
        feature_schema=feature_schema,
        config=config,
        published_pair_manifest=validation_pairs,
        device=replay_device,
    )
    metric_tolerance = {"relative": 1e-6, "absolute": 1e-7}
    for name in (
        "interval_loss",
        "mae",
        "rmse",
        "spearman_rank",
        "minimum_distance",
        "maximum_distance",
    ):
        if not math.isclose(
            float(validation_replay[name]),
            float(validation[name]),
            rel_tol=metric_tolerance["relative"],
            abs_tol=metric_tolerance["absolute"],
        ):
            raise TasteNeuroSEDGateError(
                f"independent validation replay differs from worker metric: {name}"
            )
    if validation_replay["pair_count"] != validation.get("pair_count"):
        raise TasteNeuroSEDGateError("independent validation replay count changed")

    for name in ("best.pt", "model.pt"):
        model = load_runner_checkpoint(root / name, input_dim=input_dim, device="cpu")
        del model
    health = checkpoint_health(
        root / "best.pt",
        input_dim=input_dim,
        require_cuda_tolerance=require_cuda_tolerance,
    )
    verification = {
        "schema_version": "tastemolnet_gcf_neurosed_independent_verification_v1",
        "status": "PASS",
        "independent_scientific_verifier": True,
        "dataset": "tastemolnet",
        "role": "GCF_AUXILIARY_DISTANCE_MODEL",
        "classifier": False,
        "bundle_inventory_closed": True,
        "bundle_sha256sums_sha256": _sha256_file(root / "sha256sums.txt"),
        "best_checkpoint_sha256": checksums["best.pt"],
        "selected_checkpoint_uuid": selected_id,
        "finite_validation_error_and_rank": True,
        "no_train_test_leakage": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "feature_schema_compatible": True,
        "checkpoint_health": health,
        "validation_replay": validation_replay,
        "validation_metric_tolerance": metric_tolerance,
        "worker_validation_metrics_reproduced": True,
        "authoritative_split_lineage": dict(authoritative_lineage),
        "controller_authority": {
            field: bundled_controller_latest[field]
            for field in _CONTROLLER_STABLE_FIELDS
        },
        "controller_worker_initial_heartbeat": dict(
            bundled_controller_initial
        ),
        "controller_worker_latest_heartbeat_sha256": bundled_controller_latest[
            "heartbeat_sha256"
        ],
        "controller_worker_latest_heartbeat_sequence": bundled_controller_latest[
            "sequence"
        ],
        "controller_verifier_heartbeat_sha256": controller_authority[
            "heartbeat_sha256"
        ],
        "controller_verifier_heartbeat_sequence": controller_authority[
            "sequence"
        ],
        "worker_did_not_self_sign_pass": True,
        "managed_input_binding": {
            "source_execution_config_sha256": config[
                "source_execution_config_sha256"
            ],
            "input_hashes": {
                "controller_receipt": bundled_controller_initial[
                    "receipt_sha256"
                ],
                "worker_initial_heartbeat": bundled_controller_initial[
                    "heartbeat_sha256"
                ],
                "t2_receipt_gate": authoritative_lineage[
                    "t2_receipt_gate_sha256"
                ],
                "t2_source_evidence": authoritative_lineage[
                    "t2_source_evidence_sha256"
                ],
                "t2_source_sha256s": authoritative_lineage[
                    "t2_source_sha256s_sha256"
                ],
                "t2_source_split_manifest": authoritative_lineage[
                    "t2_source_split_manifest_sha256"
                ],
                "t3_gate": authoritative_lineage["t3_gate_sha256"],
                "t3_verification": authoritative_lineage[
                    "t3_verification_sha256"
                ],
                "t3_checkpoint_split_manifest": authoritative_lineage[
                    "t3_checkpoint_split_manifest_sha256"
                ],
                "train_csv": split_manifest["train_source_csv_sha256"],
                "validation_csv": split_manifest[
                    "validation_source_csv_sha256"
                ],
            },
            "execution_git_commit": git_state["commit"],
        },
        "t7_consumer": {
            "schema_version": "tastemolnet_gcf_neurosed_t7_consumer_v1",
            "dataset": "tastemolnet",
            "role": "GCF_AUXILIARY_DISTANCE_MODEL",
            "classifier": False,
            "source_label_independent": True,
            "train_only_fit": True,
            "validation_only_selection": True,
            "calibration_loaded": False,
            "test_loaded": False,
            "health_gate_status": "PASS",
            "checkpoint_relative_path": "artifacts/best.pt",
            "checkpoint_sha256": checksums["best.pt"],
            "feature_schema_relative_path": "artifacts/feature_schema.json",
            "feature_schema_sha256": checksums["feature_schema.json"],
            "feature_atomic_numbers": list(atom_vocabulary),
            "feature_input_dim": input_dim,
            "sha256s_relative_path": "artifacts/sha256sums.txt",
            "sha256s_sha256": _sha256_file(root / "sha256sums.txt"),
            "neurosed_train_graph_ids_hash": split_manifest[
                "neurosed_train_graph_ids_hash"
            ],
            "neurosed_validation_graph_ids_hash": split_manifest[
                "neurosed_validation_graph_ids_hash"
            ],
        },
    }
    require_release_eligible_official_semantics(model_card)
    return verification


__all__ = [
    "REQUIRED_BUNDLE_FILES",
    "STRICT_OFFICIAL_PROVENANCE",
    "TasteNeuroSEDGateError",
    "checkpoint_health",
    "recompute_validation_metrics",
    "require_release_eligible_official_semantics",
    "validate_controller_heartbeat_progression",
    "verify_bundle",
]
