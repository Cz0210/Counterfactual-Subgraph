"""Release-candidate contracts for official-semantics fixed-budget NeuroSED.

These gates validate metadata and the GCF call direction without declaring a
scientific PASS.  Real GEDLIB labels, a trained checkpoint, and an independent
managed verifier remain mandatory external evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from src.data.tastemolnet_neurosed_fixed_budget import (
    PAIR_SAMPLER_MANIFEST_SCHEMA,
    PAIR_SAMPLING_SEED,
    reserve_pair_count,
)
from src.eval.tastemolnet_neurosed_fixed_budget import (
    OFFICIAL_GED_DIRECTION,
    OFFICIAL_SED_EDIT_COSTS,
    PAIR_LABELS_MANIFEST_SCHEMA,
    validation_pair_budget,
)
from src.eval.tastemolnet_neurosed_gate import STRICT_OFFICIAL_PROVENANCE
from src.eval.tastemolnet_neurosed_non_mip import (
    validate_non_mip_selection_manifest,
)
from src.train.tastemolnet_neurosed_official_selector import SELECTOR_TRACE_SCHEMA
from src.utils.tastemolnet_neurosed_gedlib_build import (
    GED_LABEL_BACKEND_VARIANT,
    NON_MIP_METHOD_CONFIGS,
)


OFFICIAL_FIXED_MODEL_CARD_SCHEMA = (
    "tastemolnet_gcf_neurosed_official_fixed_budget_model_card_v2"
)
DISTANCE_DIRECTION_SCHEMA = "tastemolnet_gcf_distance_direction_trace_v1"
READINESS_SCHEMA = "tastemolnet_neurosed_official_fixed_budget_readiness_v1"
GENERATED_QUERY_ROLE = "generated_counterfactual_candidate"
ORIGINAL_TARGET_ROLE = "original_input_graph"
OFFICIAL_GCF_REPOSITORY = "https://github.com/mertkosan/GCFExplainer"
OFFICIAL_GCF_COMMIT = "cc7ca30eb2026c57f20cd6afe2ee621f486fcf2e"
VENDORED_GCF_SOURCE_SHA256 = {
    "neurosed/models.py": (
        "8025f0cdc187625fb9d469a9ec0791694f3e923ee94e3d9084cb74a066397a60"
    ),
    "distance.py": (
        "d81182ccb31ef0fc5aef6a95a7debc6c17e3b495596e4ee3ff1642adf29745c3"
    ),
    "importance.py": (
        "5e364634fcf6fac9c5e16b5d9dc2f53837ab67508421e5076010c1e9cdac33be"
    ),
    "vrrw.py": (
        "89ff1a9dbb9561d33dd4fbc1bffe84e60deeb069948778b39b75dc5c93a59fce"
    ),
    "summary.py": (
        "371ca30b9672bd17b472d261327dc343b989b52150257de8a8ce1c868389af44"
    ),
}
VENDORED_GCF_RETAINED_FILE_SHA256 = {
    "LICENSE": "152d96bfd035aaa192679224694c6b5fd267623ebfaa810a60b775b9dca35b49",
    "README.md": "e3a06e4bff0faba70754fd8750378eff2fb9ef33697b57bfaba8547d7300b37f",
    "data.py": "de92d342ee3a6be9f08dc4b578c4691c9cb457a5d5e30266a9a6e73677564bd1",
    "data/aids/gnn/model_best.pth": (
        "ad066cc678cbbde3a4eb6f91ea3d20a538b7bab0cb7d45c63e99c0ba17197ef5"
    ),
    "data/aids/neurosed/best_model.pt": (
        "887b330c390ba9ecfb545b3a14b7d71ebeb8e72006873cc3d467c2dcad87dd82"
    ),
    "distance.py": "d81182ccb31ef0fc5aef6a95a7debc6c17e3b495596e4ee3ff1642adf29745c3",
    "environment.yml": (
        "0912b96dbd04ad33a178e3b0a2615dc27618734d7c6ac900baa02320cb046ccd"
    ),
    "gcfexplainer_case_study.png": (
        "14fa9bec9063338f9a7f8bce2782339c646554b686f74c3146582ae02d49aa01"
    ),
    "gcfexplainer_coverage_cost.png": (
        "96713f35a44ab368602ef0453d1204b7661fdce7cf18d4cde0c4b9c51aba72c8"
    ),
    "gnn.py": "cfca49b1bb2bfffc5f1d4f80ad1784185ee68408c18f07e9167a027fb7bcdaa1",
    "importance.py": (
        "5e364634fcf6fac9c5e16b5d9dc2f53837ab67508421e5076010c1e9cdac33be"
    ),
    "neurosed/models.py": (
        "8025f0cdc187625fb9d469a9ec0791694f3e923ee94e3d9084cb74a066397a60"
    ),
    "paper.pdf": "a32288d5bf9684f44965f91ebc09e87eb488141a6752c6be50cc40482dc09989",
    "slides.pdf": "d8841a73f6c8ba237794b52a666c798d67820fd17b84a346df8b96755330e82d",
    "summary.py": "371ca30b9672bd17b472d261327dc343b989b52150257de8a8ce1c868389af44",
    "util.py": "6489a02e7a0d6498a5f9e7b1a9a4ebc137e3d26541bd2a605bff9f54b1cf74ce",
    "vrrw.py": "89ff1a9dbb9561d33dd4fbc1bffe84e60deeb069948778b39b75dc5c93a59fce",
}
VENDORED_GCF_RETAINED_DIRECTORIES = (
    ".",
    "data",
    "data/aids",
    "data/aids/gnn",
    "data/aids/neurosed",
    "neurosed",
)
VENDORED_GCF_RETAINED_INVENTORY_SHA256 = (
    "467205d647d8a1be55f129a936ace8be48904eeb2b802e909a8c62cc6088c606"
)


class OfficialFixedBudgetGateError(RuntimeError):
    """A fixed-budget release-candidate contract is incomplete or changed."""


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise OfficialFixedBudgetGateError(f"{label} is not a lowercase SHA256")
    return digest


def _commit(value: Any, *, label: str) -> str:
    commit = str(value or "")
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise OfficialFixedBudgetGateError(f"{label} is not a full Git commit")
    return commit


def _sha256_open_file(file_descriptor: int) -> tuple[str, os.stat_result]:
    before = os.fstat(file_descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise OfficialFixedBudgetGateError("vendored GCF inventory contains non-file")
    digest = hashlib.sha256()
    while True:
        chunk = os.read(file_descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    after = os.fstat(file_descriptor)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise OfficialFixedBudgetGateError("vendored GCF file changed while hashing")
    return digest.hexdigest(), after


def verify_vendored_gcf_retained_inventory(
    vendored_gcf_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Descriptor-reopen the exact retained upstream GCF file inventory."""

    root = Path(vendored_gcf_root)
    if not root.is_absolute() or Path(os.path.normpath(root)) != root:
        raise OfficialFixedBudgetGateError(
            "vendored GCF root must be normalized and absolute"
        )
    current = Path(root.anchor)
    try:
        for component in root.parts[1:]:
            current /= component
            if stat.S_ISLNK(os.lstat(current).st_mode):
                raise OfficialFixedBudgetGateError(
                    "vendored GCF root has a symlink ancestor"
                )
    except OSError as error:
        raise OfficialFixedBudgetGateError(
            "vendored GCF root ancestry cannot be authenticated"
        ) from error
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    try:
        root_fd = os.open(root, os.O_RDONLY | directory | nofollow)
    except OSError as error:
        raise OfficialFixedBudgetGateError(
            "vendored GCF root is unavailable or not a real directory"
        ) from error
    actual: dict[str, str] = {}
    directories: set[str] = set()
    try:
        for dirpath, dirnames, filenames, dir_fd in os.fwalk(
            ".", topdown=True, follow_symlinks=False, dir_fd=root_fd
        ):
            relative_directory = Path(dirpath).as_posix()
            directories.add(relative_directory)
            for name in tuple(dirnames):
                entry = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                if not stat.S_ISDIR(entry.st_mode):
                    raise OfficialFixedBudgetGateError(
                        "vendored GCF inventory contains symlink/non-directory"
                    )
            for name in filenames:
                relative = (Path(dirpath) / name).as_posix()
                if relative.startswith("./"):
                    relative = relative[2:]
                try:
                    file_fd = os.open(name, os.O_RDONLY | nofollow, dir_fd=dir_fd)
                except OSError as error:
                    raise OfficialFixedBudgetGateError(
                        f"vendored GCF file cannot be reopened safely: {relative}"
                    ) from error
                try:
                    actual[relative], _ = _sha256_open_file(file_fd)
                finally:
                    os.close(file_fd)
    finally:
        os.close(root_fd)
    expected_directories = set(VENDORED_GCF_RETAINED_DIRECTORIES)
    if directories != expected_directories:
        raise OfficialFixedBudgetGateError("vendored GCF directory inventory changed")
    if set(actual) != set(VENDORED_GCF_RETAINED_FILE_SHA256):
        raise OfficialFixedBudgetGateError("vendored GCF retained file inventory changed")
    changed = sorted(
        path
        for path, expected in VENDORED_GCF_RETAINED_FILE_SHA256.items()
        if actual.get(path) != expected
    )
    if changed:
        raise OfficialFixedBudgetGateError(
            f"vendored GCF file hash changed: {changed[0]}"
        )
    inventory_sha256 = _stable_sha256(actual)
    if inventory_sha256 != VENDORED_GCF_RETAINED_INVENTORY_SHA256:
        raise OfficialFixedBudgetGateError("vendored GCF inventory digest changed")
    return {
        "root_realpath": os.path.realpath(root),
        "file_count": len(actual),
        "inventory_sha256": inventory_sha256,
    }


def _matrix_shape(value: Any) -> tuple[int, int] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            dimensions = tuple(int(item) for item in shape)
        except (TypeError, ValueError):
            return None
        return dimensions if len(dimensions) == 2 else None
    if type(value) is list:
        if not value or any(type(row) is not list for row in value):
            return None
        widths = {len(row) for row in value}
        if len(widths) != 1:
            return None
        return len(value), widths.pop()
    return None


@dataclass(slots=True)
class GeneratedQueryOriginalTargetBinding:
    """Bind official embedded targets and expose only generated-query calls."""

    model: Any
    original_target_hashes: tuple[str, ...]
    _records: list[dict[str, Any]]
    _call_count: int = 0

    @classmethod
    def create(
        cls,
        model: Any,
        *,
        original_targets: Sequence[Any],
        original_target_hashes: Sequence[str],
    ) -> "GeneratedQueryOriginalTargetBinding":
        targets = list(original_targets)
        hashes = tuple(
            _sha256(value, label="original target graph hash")
            for value in original_target_hashes
        )
        if not targets or len(targets) != len(hashes):
            raise OfficialFixedBudgetGateError(
                "original target graphs/hashes must be non-empty and aligned"
            )
        embed = getattr(model, "embed_targets", None)
        predict = getattr(model, "predict_outer_with_queries", None)
        if not callable(embed) or not callable(predict):
            raise OfficialFixedBudgetGateError(
                "NeuroSED model lacks the official target/query API"
            )
        embed(targets)
        return cls(model=model, original_target_hashes=hashes, _records=[])

    def predict_generated_queries(
        self,
        generated_queries: Sequence[Any],
        *,
        generated_query_hashes: Sequence[str],
        batch_size: int | None = None,
    ) -> Any:
        queries = list(generated_queries)
        hashes = tuple(
            _sha256(value, label="generated query graph hash")
            for value in generated_query_hashes
        )
        if not queries or len(queries) != len(hashes):
            raise OfficialFixedBudgetGateError(
                "generated query graphs/hashes must be non-empty and aligned"
            )
        if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
            raise OfficialFixedBudgetGateError("distance batch size must be positive")
        result = self.model.predict_outer_with_queries(queries, batch_size=batch_size)
        expected_shape = (len(queries), len(self.original_target_hashes))
        if _matrix_shape(result) != expected_shape:
            raise OfficialFixedBudgetGateError(
                "generated-query/original-target distance matrix shape changed"
            )
        for query_hash in hashes:
            for target_hash in self.original_target_hashes:
                self._records.append(
                    {
                        "distance_call_index": self._call_count,
                        "query_graph_hash": query_hash,
                        "target_graph_hash": target_hash,
                        "query_role": GENERATED_QUERY_ROLE,
                        "target_role": ORIGINAL_TARGET_ROLE,
                        "direction": OFFICIAL_GED_DIRECTION,
                    }
                )
        self._call_count += 1
        return result

    def direction_manifest(self) -> dict[str, Any]:
        if self._call_count <= 0 or not self._records:
            raise OfficialFixedBudgetGateError("no official GCF distance call was recorded")
        if any(
            row.get("query_role") != GENERATED_QUERY_ROLE
            or row.get("target_role") != ORIGINAL_TARGET_ROLE
            or row.get("direction") != OFFICIAL_GED_DIRECTION
            for row in self._records
        ):
            raise OfficialFixedBudgetGateError("GCF distance direction was reversed")
        payload = {
            "schema_version": DISTANCE_DIRECTION_SCHEMA,
            "status": "READY_FOR_INDEPENDENT_VERIFICATION",
            "distance_api": "NormGEDModel.predict_outer_with_queries",
            "targets_embedded_before_query_calls": True,
            "query_role": GENERATED_QUERY_ROLE,
            "target_role": ORIGINAL_TARGET_ROLE,
            "direction": "generated_query_to_original_target",
            "reverse_direction_used": False,
            "distance_call_count": self._call_count,
            "distance_pair_record_count": len(self._records),
            "original_target_hashes_sha256": _stable_sha256(
                list(self.original_target_hashes)
            ),
            "records": [dict(row) for row in self._records],
        }
        payload["trace_sha256"] = _stable_sha256(payload)
        return payload


def validate_official_fixed_budget_model_card(
    model_card: Mapping[str, Any],
    *,
    vendored_gcf_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Validate claims and reopen the exact vendored GCF source inventory."""

    card = dict(model_card)
    exact = {
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
        "fixed_budget_extension_documented": True,
        "upstream_greed_independent_pair_role_semantics_unchanged": True,
        "upstream_greed_sampler_byte_for_byte_unchanged": False,
        "exhaustive_pairs": False,
        "cartesian_product_materialized": False,
        "independent_query_target_pairs": True,
        "query_graph_id_differs_from_target_graph_id": True,
        "parent_own_subgraph_shortcut": False,
        "class_label_used_as_supervision": False,
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
        "training_loop_authority": (
            "neuro.train.train_full_batch_interleaved_validation"
        ),
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
        "deterministic_reserve_fraction": 0.10,
        "disk_reservation_pass": True,
        "cpu_contention_gate_pass": True,
        "worker_wrote_pass": False,
        "scientific_release_eligible": True,
        # The model/loss/selector remain upstream-compatible, but replacing
        # F2 with an explicitly selected non-MIP GEDLIB backend is not the
        # complete upstream NeuroSED configuration.
        "full_official_neurosed_semantics_claimed": False,
    }
    if any(
        type(card.get(key)) is not type(value) or card.get(key) != value
        for key, value in exact.items()
    ):
        raise OfficialFixedBudgetGateError(
            "official fixed-budget NeuroSED model-card contract changed"
        )
    train_budget = card.get("train_pair_budget")
    validation_budget = card.get("validation_pair_budget")
    if (
        type(train_budget) is not int
        or train_budget not in (2000, 5000)
        or type(validation_budget) is not int
        or validation_budget != validation_pair_budget(train_budget)
        or card.get("successful_train_pair_count") != train_budget
        or card.get("successful_validation_pair_count") != validation_budget
    ):
        raise OfficialFixedBudgetGateError("fixed train/validation pair budget changed")
    ged_method = card.get("ged_method")
    if (
        ged_method not in NON_MIP_METHOD_CONFIGS
        or card.get("ged_method_args") != NON_MIP_METHOD_CONFIGS.get(ged_method)
        or card.get("selected_ged_backend") != ged_method
        or card.get("selected_ged_backend_config")
        != NON_MIP_METHOD_CONFIGS.get(ged_method)
    ):
        raise OfficialFixedBudgetGateError("non-MIP GEDLIB backend changed")
    if card.get("edit_cost_contract") != OFFICIAL_SED_EDIT_COSTS:
        raise OfficialFixedBudgetGateError("official SED edit-cost contract changed")
    if card.get("strict_official_provenance") != STRICT_OFFICIAL_PROVENANCE:
        raise OfficialFixedBudgetGateError("strict official GREED provenance changed")
    if card.get("vendored_gcf_source_sha256") != VENDORED_GCF_SOURCE_SHA256:
        raise OfficialFixedBudgetGateError("vendored GCF source authority changed")
    if (
        card.get("vendored_gcf_retained_inventory_sha256")
        != VENDORED_GCF_RETAINED_INVENTORY_SHA256
    ):
        raise OfficialFixedBudgetGateError("vendored GCF inventory claim changed")
    if card.get("official_gcf_repository") != OFFICIAL_GCF_REPOSITORY:
        raise OfficialFixedBudgetGateError("official GCF repository changed")
    official_gcf_commit = _commit(
        card.get("official_gcf_commit"), label="official GCF commit"
    )
    if official_gcf_commit != OFFICIAL_GCF_COMMIT:
        raise OfficialFixedBudgetGateError("official GCF commit changed")
    verified_gcf = verify_vendored_gcf_retained_inventory(vendored_gcf_root)
    if (
        verified_gcf["inventory_sha256"]
        != card["vendored_gcf_retained_inventory_sha256"]
    ):
        raise OfficialFixedBudgetGateError("model card does not bind vendored GCF")
    if card.get("official_greed_commit") != STRICT_OFFICIAL_PROVENANCE["greed_commit"]:
        raise OfficialFixedBudgetGateError("official GREED commit changed")
    _commit(card.get("gedlib_commit"), label="GEDLIB commit")
    for field in (
        "pyged_module_sha256",
        "gedlib_build_manifest_sha256",
        "gedlib_config_sha256",
        "feature_schema_sha256",
        "gedlib_benchmark_summary_sha256",
        "pair_budget_plan_sha256",
        "train_pair_labels_manifest_sha256",
        "validation_pair_labels_manifest_sha256",
        "train_pair_sampler_manifest_sha256",
        "validation_pair_sampler_manifest_sha256",
        "selector_trace_sha256",
        "distance_direction_trace_sha256",
        "selected_checkpoint_sha256",
        "non_mip_gedlib_selection_sha256",
        "non_mip_gedlib_selection_manifest_file_sha256",
        "non_mip_selector_verifier_receipt_sha256",
    ):
        _sha256(card.get(field), label=field)
    return card


def _validate_pair_sampler_manifest(
    manifest: Mapping[str, Any],
    *,
    split: str,
    selected_budget: int,
    feature_schema_sha256: str,
) -> dict[str, Any]:
    payload = dict(manifest)
    exact = {
        "schema_version": PAIR_SAMPLER_MANIFEST_SCHEMA,
        "dataset": "tastemolnet",
        "split": split,
        "pair_sampling_seed": PAIR_SAMPLING_SEED,
        "pair_builder": (
            "deterministic_official_style_independent_unstratified_query_target_v2"
        ),
        "independent_query_target_pairs": True,
        "query_graph_id_differs_from_target_graph_id": True,
        "parent_own_subgraph_shortcut": False,
        "cartesian_product_materialized": False,
        "source_target_draws_with_replacement": True,
        "source_target_rng_streams_independent": True,
        "distinct_graph_ids_enforced_by_rejection": True,
        "size_or_class_used_to_select_filter_or_order_pairs": False,
        "size_and_class_diagnostics_computed_after_sampling": True,
        "ged_labels_present": False,
        "class_label_used_as_supervision": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "all_query_ids_subset_of_declared_split": True,
        "all_target_ids_subset_of_declared_split": True,
    }
    if any(
        type(payload.get(key)) is not type(value) or payload.get(key) != value
        for key, value in exact.items()
    ):
        raise OfficialFixedBudgetGateError(f"{split} pair sampler contract changed")
    if payload.get("pair_count") != reserve_pair_count(selected_budget):
        raise OfficialFixedBudgetGateError(
            f"{split} pair sampler does not contain the exact reserve"
        )
    if payload.get("feature_schema_sha256") != feature_schema_sha256:
        raise OfficialFixedBudgetGateError(
            f"{split} pair sampler feature schema changed"
        )
    for field in (
        "source_csv_sha256",
        "feature_schema_sha256",
        "graph_inventory_sha256",
        "pair_ids_sha256",
        "query_graph_ids_sha256",
        "target_graph_ids_sha256",
        "metadata_rows_sha256",
    ):
        _sha256(payload.get(field), label=f"{split} sampler {field}")
    claimed = _sha256(
        payload.get("manifest_sha256"), label=f"{split} sampler manifest"
    )
    if claimed != _stable_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    ):
        raise OfficialFixedBudgetGateError(f"{split} pair sampler hash changed")
    return payload


def verify_official_fixed_budget_readiness(
    *,
    model_card: Mapping[str, Any],
    non_mip_selection_manifest: Mapping[str, Any],
    non_mip_selector_verifier_receipt: Mapping[str, Any],
    train_pair_sampler_manifest: Mapping[str, Any],
    validation_pair_sampler_manifest: Mapping[str, Any],
    train_pair_labels_manifest: Mapping[str, Any],
    validation_pair_labels_manifest: Mapping[str, Any],
    selector_trace: Mapping[str, Any],
    distance_direction_trace: Mapping[str, Any],
    vendored_gcf_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Cross-bind local contracts; return readiness, never a scientific PASS."""

    card = validate_official_fixed_budget_model_card(
        model_card, vendored_gcf_root=vendored_gcf_root
    )
    selection = validate_non_mip_selection_manifest(
        non_mip_selection_manifest, reopen_artifacts=False
    )
    receipt = dict(non_mip_selector_verifier_receipt)
    claimed_receipt_sha256 = _sha256(
        receipt.pop("receipt_sha256", None),
        label="non-MIP selector verifier receipt",
    )
    if claimed_receipt_sha256 != _stable_sha256(receipt):
        raise OfficialFixedBudgetGateError("non-MIP verifier receipt hash changed")
    receipt["receipt_sha256"] = claimed_receipt_sha256
    if (
        receipt.get("schema_version")
        != "tastemolnet_neurosed_non_mip_gedlib_verifier_v1"
        or receipt.get("status") != "PASS"
        or receipt.get("marker") != "[TASTE_NON_MIP_GEDLIB_BACKEND_VERIFIED]"
        or receipt.get("independent_process_reopened_all_candidate_artifacts")
        is not True
        or receipt.get("selection_sha256") != selection["selection_sha256"]
        or receipt.get("selected_ged_backend")
        != selection["selected_ged_backend"]
        or receipt.get("selected_ged_backend_config")
        != selection["backend_config"]
        or receipt.get("GED_LABEL_BACKEND_VARIANT")
        != selection["GED_LABEL_BACKEND_VARIANT"]
        or receipt.get("F2_BLP_USED") is not False
        or receipt.get("GUROBI_USED") is not False
        or receipt.get("selected_neurosed_train_pair_budget")
        != selection["selected_neurosed_train_pair_budget"]
        or receipt.get("selected_neurosed_validation_pair_budget")
        != selection["selected_neurosed_validation_pair_budget"]
    ):
        raise OfficialFixedBudgetGateError("non-MIP verifier receipt changed")
    if (
        card.get("non_mip_gedlib_selection_sha256")
        != selection["selection_sha256"]
        or card.get("non_mip_gedlib_selection_manifest_file_sha256")
        != receipt.get("selection_manifest_sha256")
        or card.get("non_mip_selector_verifier_receipt_sha256")
        != receipt["receipt_sha256"]
        or card.get("selected_ged_backend")
        != selection["selected_ged_backend"]
        or card.get("selected_ged_backend_config") != selection["backend_config"]
        or card.get("ged_method") != selection["selected_ged_backend"]
        or card.get("ged_method_args") != selection["backend_config"]
        or card.get("train_pair_budget")
        != selection["selected_neurosed_train_pair_budget"]
        or card.get("validation_pair_budget")
        != selection["selected_neurosed_validation_pair_budget"]
        or card.get("pyged_module_sha256") != selection["pyged_module_sha256"]
        or card.get("gedlib_commit") != selection["gedlib_commit"]
    ):
        raise OfficialFixedBudgetGateError(
            "model card does not bind non-MIP selection/verifier"
        )
    train_sampler = _validate_pair_sampler_manifest(
        train_pair_sampler_manifest,
        split="train",
        selected_budget=card["train_pair_budget"],
        feature_schema_sha256=card["feature_schema_sha256"],
    )
    validation_sampler = _validate_pair_sampler_manifest(
        validation_pair_sampler_manifest,
        split="validation",
        selected_budget=card["validation_pair_budget"],
        feature_schema_sha256=card["feature_schema_sha256"],
    )
    train_labels = dict(train_pair_labels_manifest)
    validation_labels = dict(validation_pair_labels_manifest)
    selector = dict(selector_trace)
    direction = dict(distance_direction_trace)
    for split, manifest, budget in (
        ("train", train_labels, card["train_pair_budget"]),
        ("validation", validation_labels, card["validation_pair_budget"]),
    ):
        if (
            manifest.get("schema_version") != PAIR_LABELS_MANIFEST_SCHEMA
            or manifest.get("status") != "READY_FOR_INDEPENDENT_VERIFICATION"
            or manifest.get("split") != split
            or manifest.get("requested_pair_count") != budget
            or manifest.get("successful_pair_count") != budget
            or manifest.get("real_pyged_gedlib_labels") is not True
            or manifest.get("timeout_or_error_rows_used_as_labels") is not False
            or manifest.get("selected_in_sampler_order") is not True
            or manifest.get("ged_value_based_selection_used") is not False
            or manifest.get("finite_labels") is not True
            or manifest.get("all_lower_bounds_le_upper_bounds") is not True
            or manifest.get("cache_symmetric") is not False
            or manifest.get("reverse_cache_shared") is not False
            or manifest.get("query_target_order_in_cache_key") is not True
            or manifest.get("large_per_pair_json_debug_dump_used") is not False
            or manifest.get("compact_storage_format")
            not in ("parquet", "arrow_ipc", "numpy_npz")
            or manifest.get("gedlib_commit") != card["gedlib_commit"]
            or manifest.get("ged_method") != card["ged_method"]
            or manifest.get("ged_method_args") != card["ged_method_args"]
            or manifest.get("ged_label_backend_variant")
            != card["ged_label_backend_variant"]
            or manifest.get("GED_LABEL_BACKEND_VARIANT")
            != card["GED_LABEL_BACKEND_VARIANT"]
            or manifest.get("F2_BLP_USED") is not False
            or manifest.get("GUROBI_USED") is not False
            or manifest.get("ged_method_switched_from_official") is not True
            or manifest.get("f2_blp_used") is not False
            or manifest.get("gurobi_used") is not False
            or manifest.get("pyged_module_sha256")
            != card["pyged_module_sha256"]
            or manifest.get("gedlib_build_manifest_sha256")
            != card["gedlib_build_manifest_sha256"]
            or manifest.get("gedlib_config_sha256")
            != card["gedlib_config_sha256"]
            or manifest.get("feature_schema_sha256")
            != card["feature_schema_sha256"]
            or manifest.get("pair_sampler_manifest_sha256")
            != card[f"{split}_pair_sampler_manifest_sha256"]
            or manifest.get("calibration_loaded") is not False
            or manifest.get("test_loaded") is not False
        ):
            raise OfficialFixedBudgetGateError(
                f"{split} official pair-label manifest is not release-ready"
            )
        exact_count = manifest.get("exact_bound_pair_count")
        interval_count = manifest.get("interval_bound_pair_count")
        if (
            type(exact_count) is not int
            or type(interval_count) is not int
            or min(exact_count, interval_count) < 0
            or exact_count + interval_count != budget
        ):
            raise OfficialFixedBudgetGateError(
                f"{split} exact/interval label accounting changed"
            )
        _sha256(manifest.get("compact_labels_sha256"), label=f"{split} labels")
        if manifest.get("manifest_sha256") != _stable_sha256(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        ):
            raise OfficialFixedBudgetGateError(
                f"{split} pair-label manifest hash changed"
            )
    if (
        selector.get("schema_version") != SELECTOR_TRACE_SCHEMA
        or selector.get("status") != "READY_FOR_INDEPENDENT_VERIFICATION"
        or selector.get("selector_contract")
        != "neuro.train.train_full_batch_interleaved_validation"
        or selector.get("validation_before_every_training_batch") is not True
        or selector.get("stopped_before_paired_training_batch") is not True
        or selector.get("epoch_end_validation_used") is not False
        or selector.get("selected_checkpoint_sha256")
        != card["selected_checkpoint_sha256"]
        or selector.get("trace_sha256")
        != _stable_sha256(
            {key: value for key, value in selector.items() if key != "trace_sha256"}
        )
    ):
        raise OfficialFixedBudgetGateError("official selector trace changed")
    if (
        direction.get("schema_version") != DISTANCE_DIRECTION_SCHEMA
        or direction.get("status") != "READY_FOR_INDEPENDENT_VERIFICATION"
        or direction.get("query_role") != GENERATED_QUERY_ROLE
        or direction.get("target_role") != ORIGINAL_TARGET_ROLE
        or direction.get("direction") != "generated_query_to_original_target"
        or direction.get("reverse_direction_used") is not False
        or type(direction.get("distance_call_count")) is not int
        or direction["distance_call_count"] <= 0
        or direction.get("trace_sha256")
        != _stable_sha256(
            {key: value for key, value in direction.items() if key != "trace_sha256"}
        )
    ):
        raise OfficialFixedBudgetGateError("GCF generated-query direction changed")
    bindings = {
        "train_pair_sampler_manifest_sha256": train_sampler["manifest_sha256"],
        "validation_pair_sampler_manifest_sha256": validation_sampler[
            "manifest_sha256"
        ],
        "train_pair_labels_manifest_sha256": train_labels["manifest_sha256"],
        "validation_pair_labels_manifest_sha256": validation_labels[
            "manifest_sha256"
        ],
        "selector_trace_sha256": str(selector.get("trace_sha256") or ""),
        "distance_direction_trace_sha256": str(direction.get("trace_sha256") or ""),
        "vendored_gcf_retained_inventory_sha256": (
            VENDORED_GCF_RETAINED_INVENTORY_SHA256
        ),
        "non_mip_gedlib_selection_sha256": selection["selection_sha256"],
        "non_mip_selector_verifier_receipt_sha256": receipt["receipt_sha256"],
    }
    if any(card.get(field) != digest for field, digest in bindings.items()):
        raise OfficialFixedBudgetGateError("model card does not bind fixed-budget evidence")
    return {
        "schema_version": READINESS_SCHEMA,
        "status": "READY_FOR_MANAGED_INDEPENDENT_VERIFICATION",
        "marker": None,
        "scientific_pass_claimed": False,
        "real_gedlib_execution_required": True,
        "checkpoint_execution_required": True,
        "model_card_contract_valid": True,
        "train_pair_sampler_contract_valid": True,
        "validation_pair_sampler_contract_valid": True,
        "train_pair_labels_contract_valid": True,
        "validation_pair_labels_contract_valid": True,
        "official_selector_contract_valid": True,
        "generated_query_original_target_direction_valid": True,
        "evidence_bindings": bindings,
    }


__all__ = [
    "DISTANCE_DIRECTION_SCHEMA",
    "GENERATED_QUERY_ROLE",
    "GeneratedQueryOriginalTargetBinding",
    "OFFICIAL_GCF_COMMIT",
    "OFFICIAL_GCF_REPOSITORY",
    "OFFICIAL_FIXED_MODEL_CARD_SCHEMA",
    "ORIGINAL_TARGET_ROLE",
    "OfficialFixedBudgetGateError",
    "READINESS_SCHEMA",
    "VENDORED_GCF_SOURCE_SHA256",
    "VENDORED_GCF_RETAINED_FILE_SHA256",
    "VENDORED_GCF_RETAINED_INVENTORY_SHA256",
    "validate_official_fixed_budget_model_card",
    "verify_vendored_gcf_retained_inventory",
    "verify_official_fixed_budget_readiness",
]
