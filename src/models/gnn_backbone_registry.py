"""Registry of molecular message-passing backbones.

The registry is intentionally declarative: every backbone consumes the same
encoded atom and bond information, while the recorded ``edge_feature_mode``
documents how that information enters its message function.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TEMPERATURE_BASE_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "selection_split",
        "test_used_for_fit",
        "temperature",
        "num_examples",
        "num_classes",
        "nll_before",
        "nll_after",
        "ece_before",
        "ece_after",
        "brier_before",
        "brier_after",
        "argmax_invariant",
    }
)
_TEMPERATURE_PROVENANCE_FIELDS = frozenset(
    {
        "dataset",
        "validation_split_sha256",
        "ordered_parent_ids_sha256",
        "ordered_labels_sha256",
        "selected_checkpoint_sha256",
        "feature_schema_sha256",
        "temperature_contract_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class GNNBackboneSpec:
    name: str
    display_name: str
    edge_feature_mode: str
    description: str
    aliases: tuple[str, ...] = ()


_BACKBONES: dict[str, GNNBackboneSpec] = {}
_ALIASES: dict[str, str] = {}


def register_gnn_backbone(spec: GNNBackboneSpec) -> None:
    name = str(spec.name).strip().lower()
    if not name or name != spec.name:
        raise ValueError("Backbone registry names must be normalized lowercase strings.")
    if name in _BACKBONES:
        raise ValueError(f"Molecular GNN backbone is already registered: {name}")
    aliases = tuple(str(alias).strip().lower() for alias in spec.aliases)
    collisions = [alias for alias in (name, *aliases) if alias in _ALIASES]
    if collisions:
        raise ValueError(f"Molecular GNN backbone aliases collide: {collisions}")
    _BACKBONES[name] = spec
    for alias in (name, *aliases):
        _ALIASES[alias] = name


def normalize_gnn_backbone(name: str) -> str:
    normalized = str(name or "").strip().lower().replace("-", "").replace("_", "")
    canonical = _ALIASES.get(normalized)
    if canonical is None:
        raise ValueError(
            f"Unknown molecular GNN backbone {name!r}; "
            f"available={','.join(available_gnn_backbones())}"
        )
    return canonical


def get_gnn_backbone_spec(name: str) -> GNNBackboneSpec:
    return _BACKBONES[normalize_gnn_backbone(name)]


def available_gnn_backbones() -> tuple[str, ...]:
    return tuple(sorted(_BACKBONES))


def iter_gnn_backbone_specs() -> Iterable[GNNBackboneSpec]:
    for name in available_gnn_backbones():
        yield _BACKBONES[name]


def validate_backbone_feature_schema(
    payload: Any,
    *,
    expected_sha256: str | None = None,
) -> Any:
    """Validate the shared atom/bond schema used by every backbone.

    The import is lazy because :mod:`molecular_gnn` itself consumes this
    registry.  Returning the project's typed schema keeps feature handling
    identical for GINE, GIN, GCN, GATv2, and GatedGCN+.
    """

    from src.data.molecular_graph_featurizer import MolecularFeatureSchema

    schema = (
        payload
        if isinstance(payload, MolecularFeatureSchema)
        else MolecularFeatureSchema.from_dict(payload)
    )
    actual = str(schema.to_dict()["schema_sha256"])
    if expected_sha256 is not None and actual != str(expected_sha256).lower():
        raise ValueError("Molecular feature schema differs from the frozen contract.")
    return schema


def build_backbone(
    name: str,
    config: Mapping[str, Any],
    *,
    feature_schema: Any | None = None,
    expected_feature_schema_sha256: str | None = None,
    num_classes: int | None = None,
) -> Any:
    """Build one molecular classifier through the common public interface.

    Checked-in model YAML may keep architecture under ``gnn`` while the
    dataset supplies ``num_classes`` and the shared full feature schema.  Flat
    legacy configurations with explicit node/edge schemas remain supported.
    """

    from src.models.molecular_gnn import build_molecular_gnn

    if not isinstance(config, Mapping):
        raise ValueError("GNN build configuration must be a mapping.")
    root = dict(config)
    source = root.get("gnn", root)
    if not isinstance(source, Mapping):
        raise ValueError("config.gnn must be a mapping.")
    values = dict(source)
    configured = values.pop("backbone", name)
    canonical = normalize_gnn_backbone(name)
    if normalize_gnn_backbone(str(configured)) != canonical:
        raise ValueError("Requested and configured GNN backbones differ.")
    shared_schema = feature_schema
    if shared_schema is None:
        shared_schema = values.pop("feature_schema", root.get("feature_schema"))
    if shared_schema is not None:
        typed_schema = validate_backbone_feature_schema(
            shared_schema,
            expected_sha256=expected_feature_schema_sha256,
        )
        node_schema = typed_schema
        edge_schema = typed_schema
    else:
        node_schema = values.pop(
            "node_feature_schema", root.get("node_feature_schema")
        )
        edge_schema = values.pop(
            "edge_feature_schema", root.get("edge_feature_schema")
        )
        if node_schema is None or edge_schema is None:
            raise ValueError(
                "build_backbone requires one shared feature_schema or explicit "
                "node_feature_schema and edge_feature_schema"
            )
        if expected_feature_schema_sha256 is not None:
            raise ValueError(
                "expected_feature_schema_sha256 requires one full shared schema"
            )
    configured_classes = values.pop("num_classes", root.get("num_classes", num_classes))
    if configured_classes is None:
        raise ValueError("build_backbone requires dataset num_classes")
    if num_classes is not None and int(configured_classes) != int(num_classes):
        raise ValueError("Configured and requested num_classes differ.")
    allowed = {
        "num_classes",
        "num_layers",
        "hidden_dim",
        "dropout",
        "pooling",
        "readout_layers",
        "normalization",
        "residual",
    }
    if canonical == "gps":
        allowed.update(
            {
                "rwpe_walk_length",
                "attention_heads",
                "local_mpnn",
                "global_attention",
                "backend",
            }
        )
    if canonical == "gatedgcn_plus":
        allowed.update(
            {
                "ffn",
                "rwpe_walk_length",
                "rwpe_dim",
                "rwpe_raw_normalization",
            }
        )
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unsupported GNN build configuration fields: {unknown}")
    return build_molecular_gnn(
        backbone=canonical,
        num_classes=int(configured_classes),
        node_feature_schema=node_schema,
        edge_feature_schema=edge_schema,
        **values,
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    try:
        serialized = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature contract is not canonical JSON") from exc
    return hashlib.sha256(serialized).hexdigest()


def _sha256(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HEX64.fullmatch(normalized):
        raise ValueError(f"{field} must be one lowercase SHA-256")
    return normalized


def _validation_authority(split_manifest: Any) -> tuple[str, str]:
    if not isinstance(split_manifest, Mapping):
        raise ValueError("split_manifest must be one mapping")
    dataset = str(split_manifest.get("dataset", "")).strip().lower()
    files = split_manifest.get("files")
    validation = files.get("validation") if isinstance(files, Mapping) else None
    if not dataset or not isinstance(validation, Mapping):
        raise ValueError("split_manifest must bind dataset and validation file")
    return dataset, _sha256(
        validation.get("sha256"), field="split_manifest.files.validation.sha256"
    )


def _prediction_authority(
    rows: Any,
) -> tuple[str, str, int]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Iterable):
        raise ValueError("validation_predictions must be an ordered iterable")
    parent_ids: list[str] = []
    labels: list[int] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("validation prediction rows must be mappings")
        parent_id = str(row.get("parent_id", row.get("molecule_id", ""))).strip()
        label = row.get("true_label", row.get("label"))
        if not parent_id or type(label) is not int:
            raise ValueError(
                f"validation prediction row {index} lacks parent identity/true label"
            )
        parent_ids.append(parent_id)
        labels.append(label)
    if not parent_ids or len(parent_ids) != len(set(parent_ids)):
        raise ValueError("validation prediction parent IDs are empty or duplicated")
    return (
        _canonical_sha256({"ordered_parent_ids": parent_ids}),
        _canonical_sha256({"ordered_true_labels": labels}),
        len(parent_ids),
    )


def _validate_temperature_contract(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("temperature_scaling must be one mapping")
    normalized = dict(payload)
    required = _TEMPERATURE_BASE_FIELDS | _TEMPERATURE_PROVENANCE_FIELDS
    if set(normalized) != required:
        raise ValueError("temperature_scaling fields differ from the closed schema")
    temperature = normalized.get("temperature")
    if (
        normalized.get("schema_version") != "temperature_scaling_v1"
        or normalized.get("status") != "fit"
        or normalized.get("selection_split") != "validation"
        or normalized.get("test_used_for_fit") is not False
        or type(temperature) not in (int, float)
        or isinstance(temperature, bool)
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
        or normalized.get("argmax_invariant") is not True
        or type(normalized.get("num_examples")) is not int
        or int(normalized["num_examples"]) <= 0
        or type(normalized.get("num_classes")) is not int
        or int(normalized["num_classes"]) < 2
    ):
        raise ValueError(
            "temperature scaling must be a validation-only fitted positive scalar"
        )
    for field in (
        "nll_before",
        "nll_after",
        "ece_before",
        "ece_after",
        "brier_before",
        "brier_after",
    ):
        value = normalized[field]
        if (
            type(value) not in (int, float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"temperature_scaling.{field} must be finite")
    from src.data.dataset_registry import normalize_dataset_id

    normalized["dataset"] = normalize_dataset_id(
        normalized.get("dataset"), allow_historical=False
    )
    for field in (
        "validation_split_sha256",
        "ordered_parent_ids_sha256",
        "ordered_labels_sha256",
        "selected_checkpoint_sha256",
        "feature_schema_sha256",
        "temperature_contract_sha256",
    ):
        normalized[field] = _sha256(normalized.get(field), field=field)
    body = dict(normalized)
    claimed = body.pop("temperature_contract_sha256")
    if claimed != _canonical_sha256(body):
        raise ValueError("temperature contract SHA differs from its payload")
    return normalized


def _close_temperature_to_bundle(
    *,
    temperature: Mapping[str, Any],
    feature_schema_sha256: str,
    model_card: Any,
    split_manifest: Any,
    validation_predictions: Any | None,
    actual_checkpoint_sha256: str | None = None,
) -> None:
    if not isinstance(model_card, Mapping):
        raise ValueError("model_card must be one mapping")
    model_dataset = str(model_card.get("dataset", "")).strip().lower()
    selected_checkpoint = _sha256(
        model_card.get("selected_checkpoint_sha256"),
        field="model_card.selected_checkpoint_sha256",
    )
    split_dataset, validation_split_sha = _validation_authority(split_manifest)
    if (
        temperature["dataset"] != model_dataset
        or temperature["dataset"] != split_dataset
        or temperature["feature_schema_sha256"] != feature_schema_sha256
        or temperature["validation_split_sha256"] != validation_split_sha
        or temperature["selected_checkpoint_sha256"] != selected_checkpoint
    ):
        raise ValueError(
            "temperature provenance differs from model/split/feature authority"
        )
    if validation_predictions is not None:
        parent_sha, label_sha, count = _prediction_authority(validation_predictions)
        if (
            temperature["ordered_parent_ids_sha256"] != parent_sha
            or temperature["ordered_labels_sha256"] != label_sha
            or temperature["num_examples"] != count
        ):
            raise ValueError(
                "temperature provenance differs from validation prediction rows"
            )
    if (
        actual_checkpoint_sha256 is not None
        and temperature["selected_checkpoint_sha256"]
        != _sha256(actual_checkpoint_sha256, field="actual model.pt checkpoint_id")
    ):
        raise ValueError(
            "temperature selected checkpoint differs from actual model.pt checkpoint_id"
        )


def save_backbone_bundle(
    *,
    expected_backbone: str | None = None,
    expected_feature_schema_sha256: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Publish only a schema- and validation-calibration-closed bundle."""

    from src.oracles.gnn_oracle import save_gnn_checkpoint_bundle

    model = kwargs.get("model")
    if model is None or not hasattr(model, "config"):
        raise ValueError("save_backbone_bundle requires a molecular GNN model")
    actual_backbone = normalize_gnn_backbone(model.config.backbone)
    model_card = kwargs.get("model_card")
    if (
        isinstance(model_card, Mapping)
        and str(model_card.get("dataset", "")).strip().lower() == "tastemolnet"
    ):
        raise RuntimeError(
            "BLOCKED_UNIMPLEMENTED_FULL_CLOSURE: registry saver cannot atomically "
            "publish the six required TasteMolNet closure files"
        )
    if (
        expected_backbone is not None
        and actual_backbone != normalize_gnn_backbone(expected_backbone)
    ):
        raise ValueError("Model backbone differs from the save contract.")
    schema = validate_backbone_feature_schema(
        kwargs.get("feature_schema"),
        expected_sha256=expected_feature_schema_sha256,
    )
    if kwargs.get("defer_tastemolnet_closure", False) is not False:
        raise ValueError("GNN ablation bundles may not defer TasteMolNet closure")
    kwargs["defer_tastemolnet_closure"] = False
    kwargs["feature_schema"] = schema
    temperature = _validate_temperature_contract(
        kwargs.get("temperature_scaling")
    )
    _close_temperature_to_bundle(
        temperature=temperature,
        feature_schema_sha256=schema.to_dict()["schema_sha256"],
        model_card=kwargs.get("model_card"),
        split_manifest=kwargs.get("split_manifest"),
        validation_predictions=kwargs.get("validation_predictions"),
    )
    kwargs["temperature_scaling"] = temperature
    result = save_gnn_checkpoint_bundle(**kwargs)
    _close_temperature_to_bundle(
        temperature=temperature,
        feature_schema_sha256=schema.to_dict()["schema_sha256"],
        model_card=kwargs.get("model_card"),
        split_manifest=kwargs.get("split_manifest"),
        validation_predictions=kwargs.get("validation_predictions"),
        actual_checkpoint_sha256=result.get("checkpoint_id"),
    )
    return result


def load_backbone_bundle(
    checkpoint_dir: str | Any,
    *,
    device: str | Any = "cpu",
    verify_hashes: bool = True,
    require_taste_closure: bool = True,
    expected_backbone: str | None = None,
    expected_feature_schema_sha256: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a bundle and close schema, calibration, and edge-mode authority."""

    from src.oracles.gnn_oracle import load_gnn_checkpoint_bundle

    if verify_hashes is not True:
        raise ValueError("GNN ablation bundle loading requires hash verification")
    if require_taste_closure is not True:
        raise ValueError("GNN ablation bundle loading requires Taste closure")
    model, metadata = load_gnn_checkpoint_bundle(
        checkpoint_dir,
        device=device,
        verify_hashes=verify_hashes,
        require_taste_closure=require_taste_closure,
    )
    actual_backbone = normalize_gnn_backbone(model.config.backbone)
    if (
        expected_backbone is not None
        and actual_backbone != normalize_gnn_backbone(expected_backbone)
    ):
        raise ValueError("Loaded backbone differs from the expected contract.")
    schema = validate_backbone_feature_schema(
        metadata.get("feature_schema"),
        expected_sha256=expected_feature_schema_sha256,
    )
    temperature = _validate_temperature_contract(
        metadata.get("temperature_scaling")
    )
    expected_mode = get_gnn_backbone_spec(actual_backbone).edge_feature_mode
    model_card = metadata.get("model_card")
    if (
        not isinstance(model_card, Mapping)
        or model_card.get("backbone") != actual_backbone
        or model_card.get("edge_feature_mode") != expected_mode
        or model_card.get("feature_schema_sha256")
        != schema.to_dict()["schema_sha256"]
    ):
        raise ValueError("Loaded model-card edge/schema disclosure differs from model.")
    dataset = str(model_card.get("dataset", "")).strip().lower()
    if dataset == "tastemolnet":
        if str(model_card.get("profile", "")).strip().lower() != "full":
            raise ValueError("TasteMolNet ablation bundle must declare profile=full")
        required = required_backbone_bundle_files(dataset)
        root = Path(checkpoint_dir).expanduser()
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise ValueError(f"TasteMolNet ablation bundle is incomplete: {missing}")
    prediction_path = Path(checkpoint_dir).expanduser() / "validation_predictions.csv"
    with prediction_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    normalized_rows = [
        {**row, "label": int(row["label"])}
        for row in rows
    ]
    _close_temperature_to_bundle(
        temperature=temperature,
        feature_schema_sha256=schema.to_dict()["schema_sha256"],
        model_card=model_card,
        split_manifest=metadata.get("split_manifest"),
        validation_predictions=normalized_rows,
        actual_checkpoint_sha256=metadata.get("checkpoint_id"),
    )
    metadata = dict(metadata)
    metadata["feature_schema"] = schema
    metadata["temperature_scaling"] = temperature
    metadata["edge_feature_mode"] = expected_mode
    return model, metadata


def fit_backbone_temperature(
    logits: Any,
    labels: Any,
    *,
    split: str,
    dataset: str,
    validation_split_sha256: str,
    ordered_parent_ids: Iterable[str],
    selected_checkpoint_sha256: str,
    feature_schema_sha256: str,
    max_iter: int = 100,
) -> dict[str, Any]:
    """Fit the existing scalar calibrator, restricted to validation data."""

    if str(split).strip().lower() != "validation":
        raise ValueError("Backbone temperature calibration is validation-only.")
    from src.oracles.gnn_oracle import fit_temperature_scaling
    from src.data.dataset_registry import normalize_dataset_id

    result = fit_temperature_scaling(logits, labels, max_iter=max_iter)
    label_values = labels.tolist() if hasattr(labels, "tolist") else list(labels)
    parent_values = [str(value).strip() for value in ordered_parent_ids]
    if (
        not parent_values
        or any(not value for value in parent_values)
        or len(parent_values) != len(set(parent_values))
        or len(parent_values) != len(label_values)
    ):
        raise ValueError("temperature parent identities do not match validation labels")
    if any(type(value) is not int for value in label_values):
        raise ValueError("temperature validation labels must be native integers")
    result.update(
        {
            "dataset": normalize_dataset_id(dataset, allow_historical=False),
            "validation_split_sha256": _sha256(
                validation_split_sha256, field="validation_split_sha256"
            ),
            "ordered_parent_ids_sha256": _canonical_sha256(
                {"ordered_parent_ids": parent_values}
            ),
            "ordered_labels_sha256": _canonical_sha256(
                {"ordered_true_labels": label_values}
            ),
            "selected_checkpoint_sha256": _sha256(
                selected_checkpoint_sha256, field="selected_checkpoint_sha256"
            ),
            "feature_schema_sha256": _sha256(
                feature_schema_sha256, field="feature_schema_sha256"
            ),
        }
    )
    result["temperature_contract_sha256"] = _canonical_sha256(result)
    return _validate_temperature_contract(result)


def required_backbone_bundle_files(dataset: str) -> tuple[str, ...]:
    """Return the exact loader inventory for one dataset's frozen bundle."""

    from src.data.dataset_registry import normalize_dataset_id
    from src.oracles.gnn_oracle import (
        REQUIRED_CHECKPOINT_FILES,
        TASTE_REQUIRED_CHECKPOINT_FILES,
    )

    normalized = normalize_dataset_id(dataset, allow_historical=False)
    if normalized == "tastemolnet":
        return tuple(REQUIRED_CHECKPOINT_FILES) + tuple(TASTE_REQUIRED_CHECKPOINT_FILES)
    return tuple(REQUIRED_CHECKPOINT_FILES)


register_gnn_backbone(
    GNNBackboneSpec(
        name="gine",
        display_name="GINE",
        edge_feature_mode="native_edge_conditioned_message",
        description="GIN-style sum aggregation with learned bond-conditioned messages.",
        aliases=("gineconv",),
    )
)
register_gnn_backbone(
    GNNBackboneSpec(
        name="gps",
        display_name="GraphGPS",
        edge_feature_mode=(
            "local_gine_native_edge_conditioned_message_plus_global_attention"
        ),
        description=(
            "GraphGPS with local GINE bond-conditioned messages, topology-only "
            "random-walk positional encodings, and global multi-head attention."
        ),
        aliases=("graphgps", "gpsconv"),
    )
)
register_gnn_backbone(
    GNNBackboneSpec(
        name="gatedgcn_plus",
        display_name="GatedGCN+",
        edge_feature_mode=(
            "native_residual_edge_gates_plus_ffn_and_topology_only_rwpe"
        ),
        description=(
            "Pinned GNN+ graph-level GatedGCN recipe with learned bond gates, "
            "node/edge normalization, residual FFN blocks, and topology-only RWPE."
        ),
        aliases=("gatedgcnplus", "gatedgcn+", "gatedgcn"),
    )
)
register_gnn_backbone(
    GNNBackboneSpec(
        name="gin",
        display_name="GIN",
        edge_feature_mode="additive_edge_conditioned_message",
        description="GIN aggregation with the shared learned bond embedding added to messages.",
        aliases=("ginconv",),
    )
)
register_gnn_backbone(
    GNNBackboneSpec(
        name="gcn",
        display_name="GCN",
        edge_feature_mode="normalized_additive_edge_conditioned_message",
        description="Degree-normalized graph convolution retaining shared bond embeddings.",
        aliases=("gcnconv",),
    )
)
register_gnn_backbone(
    GNNBackboneSpec(
        name="gatv2",
        display_name="GATv2",
        edge_feature_mode="native_edge_conditioned_attention",
        description="Dynamic attention over atom pairs and the shared learned bond embedding.",
        aliases=("gat2", "gatv2conv"),
    )
)


__all__ = [
    "GNNBackboneSpec",
    "available_gnn_backbones",
    "build_backbone",
    "fit_backbone_temperature",
    "get_gnn_backbone_spec",
    "iter_gnn_backbone_specs",
    "load_backbone_bundle",
    "normalize_gnn_backbone",
    "required_backbone_bundle_files",
    "register_gnn_backbone",
    "save_backbone_bundle",
    "validate_backbone_feature_schema",
]
