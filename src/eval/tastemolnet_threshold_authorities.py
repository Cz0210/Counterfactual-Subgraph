"""Calibration-only TasteMolNet distance-threshold authorities.

This dataset-specific selector replays the already-published T4 calibration
cohort and measures the same strict-flip residual->parent pairs with both
distance lines used downstream:

* official normalized NeuroSED for the T7 GCFExplainer coverage threshold;
* MolCLR node-Wasserstein for the method-shared T11/T12/T13/T14 contract.

No train, validation, or held-out test payload is opened by this selector.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from src.eval.frozen_threshold_manifest import load_shared_frozen_thresholds
from src.eval.mutagenicity_wnode_selector import (
    DEFAULT_COST_CAP_QUANTILE,
    DEFAULT_THRESHOLD_QUANTILES,
    DEFAULT_THRESHOLD_WEIGHTS,
    DEFAULT_THETA_STAR_QUANTILE,
    derive_thresholds,
)
from src.eval.node_wasserstein_distance import (
    MolCLRNodeWassersteinConfig,
    MolCLRNodeWassersteinDistance,
)


DATASET = "tastemolnet"
SOURCE_LABEL = 1
NUM_CLASSES = 3
NEUROSED_MARKER = "[TASTE_T7_NEUROSED_THRESHOLD_AUTHORITY_PASS]"
WNODE_MARKER = "[TASTE_SHARED_WNODE_THRESHOLD_AUTHORITY_PASS]"
SELECTOR_MARKER = "[TASTE_THRESHOLD_AUTHORITIES_PASS]"
NEUROSED_DISTANCE_LINE = "official_normged_generated_query_to_original_target_v1"
WNODE_DISTANCE_LINE = "MolCLR-Node-Wasserstein"
WNODE_DISTANCE_NAMESPACE = "tastemolnet_ours_full_wnode_v1"
CALIBRATION_OBJECTIVE = (
    "method_independent_empirical_distance_quantiles_over_all_finite_"
    "t4_calibration_strict_flip_residual_to_parent_pairs"
)


class TasteThresholdAuthorityError(RuntimeError):
    """A frozen input, split boundary, distance, or receipt is invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteThresholdAuthorityError(f"invalid JSON authority: {path}") from exc
    if type(value) is not dict:
        raise TasteThresholdAuthorityError(f"JSON authority is not an object: {path}")
    return value


def _write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new(path, _canonical_bytes(dict(value)) + b"\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_new(
        path,
        b"".join(_canonical_bytes(dict(row)) + b"\n" for row in rows),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _finite_distances(values: Sequence[float], *, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if (
        result.ndim != 1
        or result.size <= 0
        or not np.isfinite(result).all()
        or bool(np.any(result < 0.0))
    ):
        raise TasteThresholdAuthorityError(
            f"{label} distances must be one non-empty finite nonnegative vector"
        )
    return result


def derive_t7_neurosed_threshold(
    distances: Sequence[float],
    *,
    selection_split: str = "calibration",
    test_loaded: bool = False,
) -> dict[str, Any]:
    """Derive the fixed primary q30 NeuroSED threshold from calibration only."""

    if selection_split != "calibration" or test_loaded is not False:
        raise TasteThresholdAuthorityError(
            "NeuroSED threshold selection must be calibration-only and test-isolated"
        )
    values = _finite_distances(distances, label="NeuroSED")
    quantiles = np.asarray(DEFAULT_THRESHOLD_QUANTILES, dtype=np.float64)
    measured = np.quantile(values, quantiles, method="linear").astype(np.float64)
    theta = float(
        np.quantile(
            values,
            np.float64(DEFAULT_THETA_STAR_QUANTILE),
            method="linear",
        )
    )
    return {
        "schema_version": "tastemolnet_t7_neurosed_threshold_authority_v1",
        "status": "PASS",
        "marker": NEUROSED_MARKER,
        "dataset": DATASET,
        "method_consumer": "GCFExplainer",
        "distance_line": NEUROSED_DISTANCE_LINE,
        "inference_direction": "generated_query_to_original_target",
        "distance_normalization": "divide_by_sum_graph_element_counts",
        "selection_split": "calibration",
        "threshold_source_split": "calibration",
        "threshold_source": "tastemolnet_t4_strict_flip_neurosed_q30_v1",
        "objective": CALIBRATION_OBJECTIVE,
        "quantile_method": "linear",
        "dtype": "float64",
        "requested_quantiles": list(DEFAULT_THRESHOLD_QUANTILES),
        "raw_quantile_thresholds": [
            {"quantile": float(quantile), "threshold": float(threshold)}
            for quantile, threshold in zip(quantiles.tolist(), measured.tolist())
        ],
        "theta_star_quantile": float(DEFAULT_THETA_STAR_QUANTILE),
        "neurosed_distance_threshold": theta,
        "finite_strict_flip_distance_count": int(values.size),
        "tie_break": (
            "numpy_float64_linear_interpolation; equal_adjacent_order_"
            "statistics_retain_the_identical_smaller_threshold"
        ),
        "shared_across_t7_training_and_evaluation": True,
        "threshold_fitted_on_test": False,
        "selection_used_test": False,
        "test_used_for_selection": False,
        "train_payload_loaded": False,
        "validation_payload_loaded": False,
        "test_payload_loaded": False,
        "cf_mode": "strict_flip",
    }


def derive_shared_wnode_contract(
    distances: Sequence[float],
    *,
    selection_split: str = "calibration",
    test_loaded: bool = False,
) -> dict[str, Any]:
    """Derive the existing q-grid as one shared four-method Taste contract."""

    if selection_split != "calibration" or test_loaded is not False:
        raise TasteThresholdAuthorityError(
            "WNode threshold selection must be calibration-only and test-isolated"
        )
    values = _finite_distances(distances, label="WNode")
    bundle = derive_thresholds(
        values,
        quantiles=DEFAULT_THRESHOLD_QUANTILES,
        weights=DEFAULT_THRESHOLD_WEIGHTS,
        theta_star_quantile=DEFAULT_THETA_STAR_QUANTILE,
        cost_cap_quantile=DEFAULT_COST_CAP_QUANTILE,
    )
    thresholds = [float(level.threshold) for level in bundle.levels]
    return {
        "schema_version": "tastemolnet_shared_wnode_threshold_contract_v1",
        "status": "PASS",
        "marker": WNODE_MARKER,
        "dataset": DATASET,
        "methods": ["Ours", "GCFExplainer", "GlobalGCE", "ComRecGC"],
        "distance_line": WNODE_DISTANCE_LINE,
        "distance_namespace": WNODE_DISTANCE_NAMESPACE,
        "selection_split": "calibration",
        "threshold_source_split": "calibration",
        "threshold_source": "tastemolnet_t4_strict_flip_wnode_quantiles_v1",
        "objective": CALIBRATION_OBJECTIVE,
        "quantile_method": "linear",
        "dtype": "float64",
        "requested_quantiles": list(bundle.requested_quantiles),
        "requested_weights": list(bundle.requested_weights),
        "raw_quantile_thresholds": [
            {
                "quantile": float(quantile),
                "quantile_label": label,
                "threshold": float(threshold),
                "weight": float(weight),
            }
            for quantile, label, threshold, weight in zip(
                bundle.requested_quantiles,
                bundle.quantile_labels,
                bundle.raw_thresholds,
                bundle.requested_weights,
            )
        ],
        "thresholds": thresholds,
        "theta_star_quantile": float(bundle.theta_star_quantile),
        "theta_star": float(bundle.theta_star),
        "cost_cap_quantile": float(bundle.cost_cap_quantile),
        "cost_cap": float(bundle.cost_cap),
        "threshold_config_hash": _stable_sha256(thresholds),
        "finite_strict_flip_distance_count": int(bundle.finite_distance_count),
        "duplicate_thresholds_merged": len(thresholds)
        < len(bundle.requested_quantiles),
        "tie_break": (
            "identical_quantile_values_merge_into_one_level; earliest_lower_"
            "quantile_names_the_level_and_weights_are_summed"
        ),
        "shared_across_methods": True,
        "method_shared": True,
        "threshold_fitted_on_test": False,
        "selection_used_test": False,
        "test_used_for_selection": False,
        "train_payload_loaded": False,
        "validation_payload_loaded": False,
        "test_payload_loaded": False,
        "cf_mode": "strict_flip",
    }


def _validate_t4_authority(
    root: Path, *, t3_binding: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not root.is_absolute() or root.resolve(strict=True) != root:
        raise TasteThresholdAuthorityError("T4 root must be one physical absolute path")
    if (root / "PASS").read_bytes() != b"[MANAGED_EXECUTION_V2_PASS]\n":
        raise TasteThresholdAuthorityError("T4 managed PASS is absent")
    verification = _json(root / "verification.json")
    science = verification.get("verification")
    smoke = _json(root / "artifacts/oracle_smoke.json")
    if (
        verification.get("status") != "PASS"
        or verification.get("independent_verifier") is not True
        or type(science) is not dict
        or science.get("status") != "PASS"
        or science.get("marker") != "[TASTE_T4_ORACLE_SMOKE_PASS]"
        or science.get("independent_scientific_verifier") is not True
        or science.get("strict_flip_gate_pass") is not True
        or science.get("calibration_payload_loaded") is not True
        or science.get("train_payload_loaded") is not False
        or science.get("validation_payload_loaded") is not False
        or science.get("test_payload_loaded") is not False
        or science.get("rf_oracle_used") is not False
        or science.get("checkpoint_id") != t3_binding.get("checkpoint_id")
        or science.get("t3_gate_sha256") != t3_binding.get("t3_gate_sha256")
        or science.get("t3_verification_sha256")
        != t3_binding.get("t3_verification_sha256")
        or smoke.get("status") != "PASS"
        or smoke.get("terminal_round") != science.get("terminal_round")
        or smoke.get("selected_count") != science.get("selected_count")
        or smoke.get("valid_deletion_count") != science.get("valid_deletion_count")
        or smoke.get("strict_flip_count") != science.get("strict_flip_count")
        or smoke.get("test_payload_loaded") is not False
    ):
        raise TasteThresholdAuthorityError("frozen T4 authority semantics changed")
    inventory = verification.get("published_inventory")
    files = inventory.get("files") if type(inventory) is dict else None
    expected = {
        str(row.get("relative_path")): str(row.get("sha256"))
        for row in files or []
        if type(row) is dict
    }
    for relative in ("artifacts/oracle_smoke.json", "artifacts/t3_binding.json"):
        if expected.get(relative) != _sha256_file(root / relative):
            raise TasteThresholdAuthorityError(f"T4 published file changed: {relative}")
    return dict(science), smoke


def _replay_t4_pairs(
    *,
    dataset: Any,
    oracle: Any,
    feature_schema: Any,
    smoke: Mapping[str, Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer
    from src.eval.tastemolnet_gnn_stages import (
        _cohort_digest,
        _graph_from_smiles,
        _real_connected_deletions,
    )

    graphs = [dataset[index] for index in range(len(dataset))]
    before_rows = oracle.predict_records(graphs, batch_size=batch_size)
    eligible = [
        (index, graph, record)
        for index, (graph, record) in enumerate(zip(graphs, before_rows, strict=True))
        if int(graph.y) == SOURCE_LABEL
        and int(record["predicted_label"]) == SOURCE_LABEL
    ]
    terminal_round = int(smoke["terminal_round"])
    rounds = smoke.get("rounds_executed")
    if type(rounds) is not list or terminal_round <= 0 or terminal_round > len(rounds):
        raise TasteThresholdAuthorityError("T4 terminal round is malformed")
    terminal = rounds[terminal_round - 1]
    parent_limit = int(terminal["parent_limit"])
    deletion_cap = int(terminal["deletion_cap_per_parent"])
    selected_source = eligible[:parent_limit]
    selected = [
        (
            source_index,
            graph,
            _real_connected_deletions(
                graph.smiles,
                parent_id=graph.molecule_id,
                maximum=deletion_cap,
            ),
        )
        for source_index, graph, _before in selected_source
    ]
    if (
        len(selected) != int(smoke["selected_count"])
        or _cohort_digest(selected) != smoke.get("selected_cohort_digest")
    ):
        raise TasteThresholdAuthorityError("T4 selected calibration cohort changed")

    featurizer = MolecularGraphFeaturizer(feature_schema)
    residual_graphs: list[Any] = []
    positions: list[tuple[int, int, str, Any]] = []
    for parent_position, (_source_index, graph, actions) in enumerate(selected):
        for action_index, (fragment, outcome) in enumerate(actions):
            residual_graphs.append(
                _graph_from_smiles(
                    featurizer,
                    str(outcome.residual_smiles),
                    f"threshold-calibration-p{parent_position}-a{action_index}",
                )
            )
            positions.append((parent_position, action_index, fragment, outcome))
    if len(residual_graphs) != int(smoke["valid_deletion_count"]):
        raise TasteThresholdAuthorityError("T4 connected-deletion inventory changed")
    after_rows = oracle.predict_records(residual_graphs, batch_size=batch_size)
    result: list[dict[str, Any]] = []
    for after, residual_graph, (parent_position, action_index, fragment, outcome) in zip(
        after_rows, residual_graphs, positions, strict=True
    ):
        destination = int(after["predicted_label"])
        if destination == SOURCE_LABEL:
            continue
        _source_index, graph, _actions = selected[parent_position]
        before = selected_source[parent_position][2]
        identity = {
            "source_index": int(_source_index),
            "parent_graph_sha256": str(graph.graph_sha256),
            "residual_graph_sha256": str(residual_graph.graph_sha256),
            "fragment_sha256": hashlib.sha256(fragment.encode("utf-8")).hexdigest(),
            "action_index": action_index,
            "match_atom_indices": list(outcome.match_atom_indices),
            "destination_label": destination,
        }
        result.append(
            {
                "pair_id": "taste-threshold-" + _stable_sha256(identity)[:24],
                "identity": identity,
                "parent_smiles": str(graph.smiles),
                "residual_smiles": str(outcome.residual_smiles),
                "cf_drop": float(before["probabilities"][SOURCE_LABEL])
                - float(after["probabilities"][SOURCE_LABEL]),
            }
        )
    if len(result) != int(smoke["strict_flip_count"]):
        raise TasteThresholdAuthorityError("T4 strict-flip count did not replay")
    destinations = {0: 0, 2: 0}
    parent_positions: set[int] = set()
    for row in result:
        destination = int(row["identity"]["destination_label"])
        if destination not in destinations:
            raise TasteThresholdAuthorityError("T4 replay escaped three-class semantics")
        destinations[destination] += 1
        parent_positions.add(int(row["identity"]["source_index"]))
    if (
        destinations[0] != int(smoke["destination_0_count"])
        or destinations[2] != int(smoke["destination_2_count"])
        or len(parent_positions) != int(smoke["distinct_flipped_parent_count"])
    ):
        raise TasteThresholdAuthorityError("T4 strict-flip distribution changed")
    return result


def _neurosed_graph(smiles: str, *, feature_atomic_numbers: Sequence[int]) -> Any:
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - AutoDL dependency gate.
        raise TasteThresholdAuthorityError("RDKit is required") from exc
    from src.models.tastemolnet_neurosed import runtime_stack

    torch, tg, _models = runtime_stack()
    molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    if molecule is None:
        raise TasteThresholdAuthorityError("strict-flip SMILES no longer parses")
    molecule = Chem.AddHs(molecule, addCoords=False)
    Chem.SanitizeMol(molecule)
    vocabulary = {int(value): index for index, value in enumerate(feature_atomic_numbers)}
    labels = [vocabulary.get(int(atom.GetAtomicNum())) for atom in molecule.GetAtoms()]
    if not labels or any(label is None for label in labels):
        raise TasteThresholdAuthorityError("calibration graph escaped NeuroSED vocabulary")
    x = torch.zeros((len(labels), len(vocabulary)), dtype=torch.float32)
    x[
        torch.arange(len(labels)),
        torch.tensor([int(value) for value in labels], dtype=torch.long),
    ] = 1.0
    edges: list[tuple[int, int]] = []
    for bond in molecule.GetBonds():
        left = int(bond.GetBeginAtomIdx())
        right = int(bond.GetEndAtomIdx())
        edges.extend(((left, right), (right, left)))
    edges.sort()
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return tg.data.Data(x=x, edge_index=edge_index, num_nodes=len(labels))


def _measure_neurosed(
    pairs: Sequence[Mapping[str, Any]],
    *,
    checkpoint: Path,
    feature_schema: Mapping[str, Any],
    device: str,
    batch_size: int,
) -> list[float]:
    from src.models.tastemolnet_neurosed import load_runner_checkpoint, runtime_stack

    torch, tg, _models = runtime_stack()
    vocabulary = feature_schema.get("feature_atomic_numbers")
    input_dim = feature_schema.get("input_dim")
    if (
        type(vocabulary) is not list
        or not vocabulary
        or vocabulary != sorted(set(vocabulary))
        or type(input_dim) is not int
        or input_dim != len(vocabulary)
        or feature_schema.get("explicit_h_nodes") is not True
        or feature_schema.get("native_adjacency_semantics")
        != "binary_connectivity_directed_both_ways"
    ):
        raise TasteThresholdAuthorityError("NeuroSED feature schema changed")
    queries = [
        _neurosed_graph(str(row["residual_smiles"]), feature_atomic_numbers=vocabulary)
        for row in pairs
    ]
    targets = [
        _neurosed_graph(str(row["parent_smiles"]), feature_atomic_numbers=vocabulary)
        for row in pairs
    ]
    model = load_runner_checkpoint(checkpoint, input_dim=input_dim, device=device)
    predictions: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            query_batch = tg.data.Batch.from_data_list(
                queries[start : start + batch_size]
            ).to(device)
            target_batch = tg.data.Batch.from_data_list(
                targets[start : start + batch_size]
            ).to(device)
            raw = model(query_batch, target_batch).reshape(-1).detach().cpu()
            for offset, value in enumerate(raw.tolist()):
                query = queries[start + offset]
                target = targets[start + offset]
                denominator = (
                    int(query.num_nodes)
                    + int(query.edge_index.shape[1]) / 2.0
                    + int(target.num_nodes)
                    + int(target.edge_index.shape[1]) / 2.0
                )
                normalized = float(value) / float(denominator)
                if not math.isfinite(normalized) or normalized < 0.0:
                    raise TasteThresholdAuthorityError("NeuroSED returned invalid distance")
                predictions.append(normalized)
    if len(predictions) != len(pairs):
        raise TasteThresholdAuthorityError("NeuroSED prediction count changed")
    return predictions


def _measure_wnode(
    pairs: Sequence[Mapping[str, Any]],
    *,
    molclr_root: Path,
    molclr_checkpoint: Path,
    cache_db: Path,
    node_embedding_cache_dir: Path,
    device: str,
) -> tuple[list[float], dict[str, Any]]:
    provider = MolCLRNodeWassersteinDistance(
        MolCLRNodeWassersteinConfig(
            molclr_root=molclr_root,
            molclr_ckpt=molclr_checkpoint,
            cache_db=cache_db,
            node_emb_cache_dir=node_embedding_cache_dir,
            device=device,
            distance_namespace=WNODE_DISTANCE_NAMESPACE,
        )
    )
    try:
        distances: list[float] = []
        for row in pairs:
            measured = provider.distance(
                str(row["residual_smiles"]), str(row["parent_smiles"])
            )
            value = measured.get("distance")
            if (
                measured.get("ok") is not True
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise TasteThresholdAuthorityError(
                    f"WNode distance failed: {measured.get('error')}"
                )
            distances.append(float(value))
        return distances, provider.stats_dict()
    finally:
        provider.close()


def run_tastemolnet_threshold_authority_selector(
    *,
    t3_root: str | Path,
    t4_root: str | Path,
    graph_cache_root: str | Path,
    managed_neurosed_root: str | Path,
    official_gcf_root: str | Path,
    molclr_root: str | Path,
    molclr_checkpoint: str | Path,
    output_root: str | Path,
    wnode_cache_db: str | Path,
    node_embedding_cache_dir: str | Path,
    device: str = "cuda:0",
    batch_size: int = 32,
) -> dict[str, Any]:
    """Replay T4 once and atomically publish two non-paper authorities."""

    if type(batch_size) is not int or batch_size <= 0:
        raise TasteThresholdAuthorityError("batch_size must be a positive integer")
    output = Path(output_root)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise TasteThresholdAuthorityError("output_root must be fresh and absolute")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4()}"
    staging.mkdir(mode=0o700)
    t3 = None
    cache = None
    try:
        from src.data.molecular_graph_featurizer import MolecularFeatureSchema
        from src.eval.tastemolnet_gnn_stages import _load_gnn_oracle_anchored
        from src.eval.tastemolnet_neurosed_fixed_budget_adoption import (
            inspect_fixed_budget_neurosed_pass,
        )
        from src.eval.tastemolnet_neurosed_official_fixed_budget import (
            verify_vendored_gcf_retained_inventory,
        )
        from src.eval.tastemolnet_t4_oracle_smoke_v2 import (
            HeldCalibrationCache,
            HeldPublishedT3,
        )

        t3_path = Path(t3_root)
        t4_path = Path(t4_root)
        graph_cache_path = Path(graph_cache_root)
        neurosed_path = Path(managed_neurosed_root)
        official_path = Path(official_gcf_root).resolve(strict=True)
        molclr_root_path = Path(molclr_root).resolve(strict=True)
        molclr_checkpoint_path = Path(molclr_checkpoint).resolve(strict=True)
        if any(
            not path.is_absolute()
            for path in (
                t3_path,
                t4_path,
                graph_cache_path,
                neurosed_path,
                official_path,
                molclr_root_path,
                molclr_checkpoint_path,
            )
        ):
            raise TasteThresholdAuthorityError("all scientific inputs must be absolute")

        t3 = HeldPublishedT3(t3_path)
        t4_science, t4_smoke = _validate_t4_authority(
            t4_path, t3_binding=t3.binding
        )
        cache = HeldCalibrationCache(
            graph_cache_path,
            expected_manifest_sha256=str(t3.binding["graph_cache_manifest_sha256"]),
        )
        feature_schema = MolecularFeatureSchema.from_dict(
            t3.files["artifacts/checkpoint/feature_schema.json"].json()
        )
        dataset = cache.load(feature_schema)
        oracle = _load_gnn_oracle_anchored(
            t3.checkpoint_directory,
            feature_schema=feature_schema,
            device=device,
            batch_size=batch_size,
        )
        if (
            oracle.checkpoint_id != t3.binding["checkpoint_id"]
            or oracle.num_classes != NUM_CLASSES
            or oracle.source_label != SOURCE_LABEL
        ):
            raise TasteThresholdAuthorityError("loaded GINE differs from T3 authority")
        pairs = _replay_t4_pairs(
            dataset=dataset,
            oracle=oracle,
            feature_schema=feature_schema,
            smoke=t4_smoke,
            batch_size=batch_size,
        )

        # Release the GINE graph tensors before loading the two distance models.
        del oracle, dataset
        try:
            import torch

            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

        neurosed_artifacts = neurosed_path / "artifacts"
        fixed = inspect_fixed_budget_neurosed_pass(
            neurosed_artifacts,
            vendored_gcf_root=official_path,
            allow_managed_generation_token=True,
        )
        official = verify_vendored_gcf_retained_inventory(official_path)
        neurosed_checkpoint_path = neurosed_artifacts / "best.pt"
        neurosed_feature_path = neurosed_artifacts / "feature_schema.json"
        neurosed_model_card_path = neurosed_artifacts / "model_card.json"
        neurosed_feature = _json(neurosed_feature_path)
        neurosed_model_card = _json(neurosed_model_card_path)
        if (
            fixed.get("checkpoint_sha256") != _sha256_file(neurosed_checkpoint_path)
            or neurosed_model_card.get("selected_checkpoint_sha256")
            != fixed.get("checkpoint_sha256")
            or neurosed_model_card.get("calibration_loaded") is not False
            or neurosed_model_card.get("test_loaded") is not False
            or neurosed_model_card.get("gcf_runtime_direction")
            != "generated_query_to_original_target"
        ):
            raise TasteThresholdAuthorityError("managed NeuroSED authority changed")
        neurosed_distances = _measure_neurosed(
            pairs,
            checkpoint=neurosed_checkpoint_path,
            feature_schema=neurosed_feature,
            device=device,
            batch_size=batch_size,
        )
        try:
            import torch

            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass
        wnode_distances, wnode_stats = _measure_wnode(
            pairs,
            molclr_root=molclr_root_path,
            molclr_checkpoint=molclr_checkpoint_path,
            cache_db=Path(wnode_cache_db),
            node_embedding_cache_dir=Path(node_embedding_cache_dir),
            device=device,
        )

        neurosed = derive_t7_neurosed_threshold(neurosed_distances)
        wnode = derive_shared_wnode_contract(wnode_distances)
        input_authority = {
            "schema_version": "tastemolnet_threshold_selector_inputs_v1",
            "t3_root": str(t3_path),
            "t3_gate_sha256": t3.binding["t3_gate_sha256"],
            "t3_verification_sha256": t3.binding["t3_verification_sha256"],
            "t3_checkpoint_sha256": t3.binding["checkpoint_id"],
            "t3_temperature_scaling_sha256": t3.binding[
                "temperature_scaling_sha256"
            ],
            "graph_cache_root": str(graph_cache_path),
            "graph_cache_manifest_sha256": cache.manifest.sha256,
            "calibration_cache_sha256": cache.calibration.sha256,
            "t4_root": str(t4_path),
            "t4_verification_sha256": _sha256_file(t4_path / "verification.json"),
            "t4_oracle_smoke_sha256": _sha256_file(
                t4_path / "artifacts/oracle_smoke.json"
            ),
            "t4_terminal_round": t4_science["terminal_round"],
            "t4_selected_count": t4_science["selected_count"],
            "t4_valid_deletion_count": t4_science["valid_deletion_count"],
            "t4_strict_flip_count": t4_science["strict_flip_count"],
            "managed_neurosed_root": str(neurosed_path),
            "neurosed_checkpoint_sha256": fixed["checkpoint_sha256"],
            "neurosed_feature_schema_sha256": _sha256_file(neurosed_feature_path),
            "official_gcf_root": str(official_path),
            "official_gcf_inventory_sha256": official["inventory_sha256"],
            "molclr_root": str(molclr_root_path),
            "molclr_checkpoint": str(molclr_checkpoint_path),
            "molclr_checkpoint_sha256": _sha256_file(molclr_checkpoint_path),
            "opened_payload_splits": ["calibration"],
            "train_payload_loaded": False,
            "validation_payload_loaded": False,
            "test_payload_loaded": False,
        }
        input_authority["input_authority_sha256"] = _stable_sha256(input_authority)
        pairs_public = [
            {
                "pair_id": row["pair_id"],
                **dict(row["identity"]),
                "cf_drop": float(row["cf_drop"]),
                "neurosed_distance": float(neurosed_distance),
                "wnode_distance": float(wnode_distance),
                "inference_direction": "generated_query_to_original_target",
                "selection_split": "calibration",
            }
            for row, neurosed_distance, wnode_distance in zip(
                pairs, neurosed_distances, wnode_distances, strict=True
            )
        ]
        pair_digest = _stable_sha256(pairs_public)
        for payload in (neurosed, wnode):
            payload.update(
                {
                    "pair_inventory_sha256": pair_digest,
                    "input_authority_sha256": input_authority[
                        "input_authority_sha256"
                    ],
                    "selected_at": _utc_now(),
                }
            )
        _write_json(staging / "input_authority.json", input_authority)
        _write_jsonl(staging / "calibration_distance_rows.jsonl", pairs_public)
        _write_json(staging / "t7_neurosed_threshold_authority.json", neurosed)
        _write_json(staging / "tastemolnet.json", wnode)
        # Re-open the exact downstream WNode loader before publishing PASS.
        loaded_wnode = load_shared_frozen_thresholds(staging / "tastemolnet.json")
        if (
            loaded_wnode["thresholds"] != wnode["thresholds"]
            or loaded_wnode["theta_star"] != wnode["theta_star"]
            or loaded_wnode["cost_cap"] != wnode["cost_cap"]
        ):
            raise TasteThresholdAuthorityError("downstream WNode loader disagrees")
        receipt = {
            "schema_version": "tastemolnet_threshold_authority_selector_receipt_v1",
            "status": "PASS",
            "marker": SELECTOR_MARKER,
            "dataset": DATASET,
            "selection_split": "calibration",
            "opened_payload_splits": ["calibration"],
            "train_payload_loaded": False,
            "validation_payload_loaded": False,
            "test_payload_loaded": False,
            "test_used_for_selection": False,
            "strict_flip_pair_count": len(pairs_public),
            "pair_inventory_sha256": pair_digest,
            "neurosed_authority_sha256": _sha256_file(
                staging / "t7_neurosed_threshold_authority.json"
            ),
            "wnode_contract_sha256": _sha256_file(staging / "tastemolnet.json"),
            "distance_rows_sha256": _sha256_file(
                staging / "calibration_distance_rows.jsonl"
            ),
            "input_authority_sha256": input_authority["input_authority_sha256"],
            "wnode_runtime_stats": wnode_stats,
            "paper_cell_published": False,
            "selected_at": _utc_now(),
        }
        replayed_t4_science, replayed_t4_smoke = _validate_t4_authority(
            t4_path, t3_binding=t3.binding
        )
        if replayed_t4_science != t4_science or replayed_t4_smoke != t4_smoke:
            raise TasteThresholdAuthorityError("T4 authority changed during selection")
        if (
            _sha256_file(neurosed_checkpoint_path) != fixed["checkpoint_sha256"]
            or _sha256_file(neurosed_feature_path)
            != input_authority["neurosed_feature_schema_sha256"]
            or _sha256_file(molclr_checkpoint_path)
            != input_authority["molclr_checkpoint_sha256"]
        ):
            raise TasteThresholdAuthorityError(
                "distance-model input changed during selection"
            )
        _write_json(staging / "selection_receipt.json", receipt)
        checksums = [
            f"{_sha256_file(path)}  {path.name}"
            for path in sorted(staging.iterdir())
            if path.is_file()
        ]
        _write_new(staging / "sha256sums.txt", ("\n".join(checksums) + "\n").encode())
        _write_new(staging / "PASS", (SELECTOR_MARKER + "\n").encode())
        _fsync_directory(staging)
        t3.verify()
        cache.verify()
        if output.exists() or output.is_symlink():
            raise TasteThresholdAuthorityError("output_root appeared during selection")
        os.rename(staging, output)
        _fsync_directory(output.parent)
        return {
            "status": "PASS",
            "marker": SELECTOR_MARKER,
            "output_root": str(output),
            "t7_neurosed_threshold": neurosed["neurosed_distance_threshold"],
            "shared_wnode_theta_star": wnode["theta_star"],
            "shared_wnode_cost_cap": wnode["cost_cap"],
            "strict_flip_pair_count": len(pairs_public),
            "test_payload_loaded": False,
            "paper_cell_published": False,
        }
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        if cache is not None:
            cache.close()
        if t3 is not None:
            t3.close()


__all__ = [
    "NEUROSED_MARKER",
    "SELECTOR_MARKER",
    "TasteThresholdAuthorityError",
    "WNODE_MARKER",
    "derive_shared_wnode_contract",
    "derive_t7_neurosed_threshold",
    "run_tastemolnet_threshold_authority_selector",
]
