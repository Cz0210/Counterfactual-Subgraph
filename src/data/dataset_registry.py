"""Canonical dataset identities and oracle-backend policy.

The active experiment matrix is intentionally small and explicit.  Historical
dataset identities remain resolvable so old manifests can still be inspected,
but callers can opt into an active-only lookup when constructing a new run.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable


class UnknownDatasetError(ValueError):
    """Raised when a dataset name or alias is not registered."""


class InactiveDatasetError(ValueError):
    """Raised when a historical-only dataset is requested for a new run."""


class OracleBackendNotAllowedError(ValueError):
    """Raised when a dataset is paired with a prohibited oracle backend."""


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Stable metadata required by data, oracle, and evaluation consumers."""

    dataset_id: str
    display_name: str
    aliases: tuple[str, ...]
    task_type: str
    num_classes: int
    label_map_items: tuple[tuple[int, str], ...]
    source_label: int
    counterfactual_mode: str
    allowed_oracle_backends: tuple[str, ...]
    active: bool = True
    historical_only: bool = False

    @property
    def label_map(self) -> dict[int, str]:
        return dict(self.label_map_items)

    @property
    def source_label_name(self) -> str:
        return self.label_map[self.source_label]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "display_name": self.display_name,
            "aliases": list(self.aliases),
            "task_type": self.task_type,
            "num_classes": self.num_classes,
            "label_map": {str(key): value for key, value in self.label_map_items},
            "source_label": self.source_label,
            "source_label_name": self.source_label_name,
            "counterfactual_mode": self.counterfactual_mode,
            "allowed_oracle_backends": list(self.allowed_oracle_backends),
            "active": self.active,
            "historical_only": self.historical_only,
        }


_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        dataset_id="aids",
        display_name="AIDS/HIV",
        aliases=("hiv", "aids_hiv", "aids/hiv"),
        task_type="binary_graph_classification",
        num_classes=2,
        label_map_items=((0, "Inactive"), (1, "Active")),
        source_label=1,
        counterfactual_mode="untargeted_flip",
        allowed_oracle_backends=("rf", "gnn"),
    ),
    DatasetSpec(
        dataset_id="mutagenicity",
        display_name="Mutagenicity",
        aliases=("mut", "mutagenic"),
        task_type="binary_graph_classification",
        num_classes=2,
        label_map_items=((0, "Non-mutagenic"), (1, "Mutagenic")),
        source_label=1,
        counterfactual_mode="untargeted_flip",
        allowed_oracle_backends=("rf", "gnn"),
    ),
    DatasetSpec(
        dataset_id="bace",
        display_name="BACE",
        aliases=("bace1",),
        task_type="binary_graph_classification",
        num_classes=2,
        label_map_items=((0, "Inactive"), (1, "Active")),
        source_label=1,
        counterfactual_mode="untargeted_flip",
        allowed_oracle_backends=("gnn",),
    ),
    DatasetSpec(
        dataset_id="tastemolnet",
        display_name="TasteMolNet",
        aliases=("taste", "bst", "bitter_sweet_tasteless"),
        task_type="multiclass_graph_classification",
        num_classes=3,
        label_map_items=((0, "Bitter"), (1, "Sweet"), (2, "Tasteless")),
        source_label=1,
        counterfactual_mode="untargeted_flip",
        allowed_oracle_backends=("gnn",),
    ),
    DatasetSpec(
        dataset_id="bbbp",
        display_name="BBBP (historical)",
        aliases=("blood_brain_barrier",),
        task_type="binary_graph_classification",
        num_classes=2,
        label_map_items=((0, "Negative"), (1, "Positive")),
        source_label=1,
        counterfactual_mode="untargeted_flip",
        allowed_oracle_backends=("rf", "gnn"),
        active=False,
        historical_only=True,
    ),
)


def _normalize_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return token.strip("_")


def _alias_index(specs: Iterable[DatasetSpec]) -> dict[str, DatasetSpec]:
    result: dict[str, DatasetSpec] = {}
    for spec in specs:
        for raw_name in (spec.dataset_id, spec.display_name, *spec.aliases):
            key = _normalize_token(raw_name)
            existing = result.get(key)
            if existing is not None and existing.dataset_id != spec.dataset_id:
                raise RuntimeError(f"Dataset alias collision for {raw_name!r}.")
            result[key] = spec
    return result


_ALIASES = _alias_index(_SPECS)


def normalize_dataset_id(dataset: str, *, allow_historical: bool = True) -> str:
    """Resolve a canonical dataset id from a case-insensitive alias."""

    key = _normalize_token(dataset)
    spec = _ALIASES.get(key)
    if spec is None:
        registered = ", ".join(sorted(item.dataset_id for item in _SPECS))
        raise UnknownDatasetError(
            f"Unknown dataset {dataset!r}; registered dataset ids: {registered}."
        )
    if spec.historical_only and not allow_historical:
        raise InactiveDatasetError(
            f"Dataset {spec.dataset_id!r} is historical-only and cannot enter "
            "the active experiment matrix."
        )
    return spec.dataset_id


def get_dataset_spec(
    dataset: str,
    *,
    allow_historical: bool = True,
) -> DatasetSpec:
    """Return one immutable dataset specification."""

    dataset_id = normalize_dataset_id(dataset, allow_historical=allow_historical)
    return next(spec for spec in _SPECS if spec.dataset_id == dataset_id)


def active_dataset_specs() -> tuple[DatasetSpec, ...]:
    """Return the ordered four-dataset experiment matrix."""

    return tuple(spec for spec in _SPECS if spec.active)


def active_dataset_ids() -> tuple[str, ...]:
    return tuple(spec.dataset_id for spec in active_dataset_specs())


def assert_oracle_backend_allowed(dataset: str, backend: str) -> None:
    """Fail closed when a dataset is paired with a prohibited oracle.

    BACE and TasteMolNet deliberately expose only the task-specific frozen-GNN
    route.  AIDS and Mutagenicity keep RF compatibility for their already
    frozen experiment lines.
    """

    spec = get_dataset_spec(dataset)
    normalized_backend = _normalize_token(backend)
    if not normalized_backend:
        raise OracleBackendNotAllowedError("Oracle backend must be non-empty.")
    if normalized_backend not in spec.allowed_oracle_backends:
        allowed = ", ".join(spec.allowed_oracle_backends)
        raise OracleBackendNotAllowedError(
            f"Oracle backend {backend!r} is prohibited for dataset "
            f"{spec.dataset_id!r}; allowed backends: {allowed}."
        )


def registry_manifest() -> dict[str, Any]:
    """Return a JSON-serializable active-plus-historical registry manifest."""

    return {
        "schema_version": 1,
        "active_dataset_ids": list(active_dataset_ids()),
        "datasets": [spec.to_dict() for spec in _SPECS],
    }


__all__ = [
    "DatasetSpec",
    "InactiveDatasetError",
    "OracleBackendNotAllowedError",
    "UnknownDatasetError",
    "active_dataset_ids",
    "active_dataset_specs",
    "assert_oracle_backend_allowed",
    "get_dataset_spec",
    "normalize_dataset_id",
    "registry_manifest",
]
