"""Real-A800 producer for the bounded TasteMolNet T12 replay canary.

The producer reuses the held T7 input authority and its exact official
GCF/GINE/NeuroSED path.  It supports three separate process invocations:

``uninterrupted``
    Execute the fixed 16-step canary and publish one terminal observation.
``checkpoint``
    Execute steps 1--8 and publish a durable restart checkpoint only.
``resume``
    In a distinct process, reopen that checkpoint, execute steps 9--16, and
    publish the second terminal observation.

No mode opens calibration/test payloads, prints a T12 cell marker, or releases
the 20k production run.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
from typing import Any, Mapping
import uuid

from src.baselines.gcfexplainer_mutagenicity_adapter import (
    graph_lineage_neighbor_wrapper,
)
from src.baselines.gcfexplainer_mutagenicity_runtime import (
    _official_vrrw_alpha_endpoint_patch,
    _reset_official_vrrw,
    _torch_load_compat,
)
from src.baselines.tastemolnet_gcf_full_resume import (
    CANARY_PREFIX_RECEIPT_SCHEMA,
    GRAPH_IDENTITY_CONTRACT,
    STAGE,
    T12StableGCFBridge,
    TasteGCFFullResumeError,
    build_canary_observation,
    build_replay_scientific_state,
    capture_checkpoint_payload,
    capture_linux_process_identity,
    reopen_checkpoint,
    restore_checkpoint_payload,
    validate_canary_prefix_receipt,
    validate_checkpoint_identity,
    write_canary_observation,
    write_checkpoint,
)
from src.baselines.tastemolnet_gcf_smoke import (
    LABEL_MAP,
    NUM_CLASSES,
    SOURCE_LABEL,
    TasteFrozenGINENativeAdapter,
    _installed_official_importance_args,
    _official_modules,
    _run_official_walk_segment,
    _select_sources,
    _semantic_sha256,
    encode_taste_source_graph,
    load_train_rows,
    taste_record_to_pyg,
)


CANARY_TOTAL_STEPS = 16
CANARY_CHECKPOINT_CURSOR = 8
CANARY_PARENT_COUNT = 8
CANARY_SOURCE_POOL_LIMIT = 64
CANARY_SAMPLE_SIZE = 128
CANARY_CANDIDATE_CAPACITY = 512
CANARY_SEED = 7
CANARY_ALPHA = 1.0
CANARY_TELEPORT = 0.1
RUN_IDENTITY_SCHEMA = "tastemolnet_t12_gpu_replay_run_identity_v1"
PREFIX_RECEIPT_SCHEMA = CANARY_PREFIX_RECEIPT_SCHEMA

THRESHOLD_AUTHORITY_SCHEMA = "tastemolnet_t7_neurosed_threshold_authority_v1"
THRESHOLD_AUTHORITY_MARKER = "[TASTE_T7_NEUROSED_THRESHOLD_AUTHORITY_PASS]"
THRESHOLD_SELECTOR_MARKER = "[TASTE_THRESHOLD_AUTHORITIES_PASS]"
THRESHOLD_INPUT_SCHEMA = "tastemolnet_threshold_selector_inputs_v1"
THRESHOLD_RECEIPT_SCHEMA = "tastemolnet_threshold_authority_selector_receipt_v1"
_THRESHOLD_QUANTILES = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)
_THRESHOLD_AUTHORITY_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "marker",
        "dataset",
        "method_consumer",
        "distance_line",
        "inference_direction",
        "distance_normalization",
        "selection_split",
        "threshold_source_split",
        "threshold_source",
        "objective",
        "quantile_method",
        "dtype",
        "requested_quantiles",
        "raw_quantile_thresholds",
        "theta_star_quantile",
        "neurosed_distance_threshold",
        "finite_strict_flip_distance_count",
        "tie_break",
        "shared_across_t7_training_and_evaluation",
        "threshold_fitted_on_test",
        "selection_used_test",
        "test_used_for_selection",
        "train_payload_loaded",
        "validation_payload_loaded",
        "test_payload_loaded",
        "cf_mode",
        "pair_inventory_sha256",
        "input_authority_sha256",
        "selected_at",
    }
)
_THRESHOLD_INPUT_KEYS = frozenset(
    {
        "schema_version",
        "t3_root",
        "t3_gate_sha256",
        "t3_verification_sha256",
        "t3_checkpoint_sha256",
        "t3_temperature_scaling_sha256",
        "graph_cache_root",
        "graph_cache_manifest_sha256",
        "calibration_cache_sha256",
        "t4_root",
        "t4_verification_sha256",
        "t4_oracle_smoke_sha256",
        "t4_terminal_round",
        "t4_selected_count",
        "t4_valid_deletion_count",
        "t4_strict_flip_count",
        "managed_neurosed_root",
        "neurosed_checkpoint_sha256",
        "neurosed_feature_schema_sha256",
        "official_gcf_root",
        "official_gcf_inventory_sha256",
        "molclr_root",
        "molclr_checkpoint",
        "molclr_checkpoint_sha256",
        "opened_payload_splits",
        "train_payload_loaded",
        "validation_payload_loaded",
        "test_payload_loaded",
        "input_authority_sha256",
    }
)
_THRESHOLD_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "marker",
        "dataset",
        "selection_split",
        "opened_payload_splits",
        "train_payload_loaded",
        "validation_payload_loaded",
        "test_payload_loaded",
        "test_used_for_selection",
        "strict_flip_pair_count",
        "pair_inventory_sha256",
        "neurosed_authority_sha256",
        "wnode_contract_sha256",
        "distance_rows_sha256",
        "input_authority_sha256",
        "wnode_runtime_stats",
        "paper_cell_published",
        "selected_at",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID = re.compile(r"^GPU-[A-Za-z0-9-]+$")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if path.resolve(strict=True) != path or path.is_symlink() or not path.is_file():
        raise TasteGCFFullResumeError(f"{label} is not one physical file")
    data = path.read_bytes()
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteGCFFullResumeError(f"{label} is malformed") from exc
    if type(raw) is not dict:
        raise TasteGCFFullResumeError(f"{label} is not one JSON object")
    return raw, data


def _require_sha256(value: Any, *, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise TasteGCFFullResumeError(f"{field} must be lowercase SHA-256")
    return value


def _absolute(value: str | Path, *, field: str, must_exist: bool = True) -> Path:
    try:
        path = Path(value).expanduser()
    except TypeError as exc:
        raise TasteGCFFullResumeError(f"{field} must be a path") from exc
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise TasteGCFFullResumeError(f"{field} must be normalized and absolute")
    if must_exist:
        if path.resolve(strict=True) != path or path.is_symlink():
            raise TasteGCFFullResumeError(f"{field} is an alias")
    return path


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    data = _json_bytes(value)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def load_threshold_authority(
    path: str | Path,
    *,
    expected_neurosed_checkpoint_sha256: str,
    expected_neurosed_feature_schema_sha256: str,
    expected_t3_checkpoint_id: str,
    expected_t3_temperature_sha256: str,
    expected_t3_gate_sha256: str,
    expected_t3_verification_sha256: str,
    expected_official_inventory_sha256: str,
    expected_managed_neurosed_root: str | Path,
    expected_t3_root: str | Path,
    expected_official_root: str | Path,
) -> tuple[dict[str, Any], str]:
    """Reopen the calibration-only selector output as one external pin.

    The downstream file deliberately does not carry model hashes itself.  The
    selector's sibling input authority and receipt bind it to the exact held
    NeuroSED, T3, official source, calibration pair inventory, and test-isolated
    selection event.  Requiring all three prevents a hand-written scalar from
    masquerading as an external threshold authority.
    """

    authority_path = _absolute(path, field="T12 NeuroSED threshold authority")
    root = authority_path.parent
    if authority_path.name != "t7_neurosed_threshold_authority.json":
        raise TasteGCFFullResumeError("T12 NeuroSED authority filename changed")
    raw, data = _json_object(authority_path, label="T12 NeuroSED threshold authority")
    input_authority, _ = _json_object(
        root / "input_authority.json", label="T12 threshold input authority"
    )
    receipt, _ = _json_object(
        root / "selection_receipt.json", label="T12 threshold selection receipt"
    )
    pass_path = root / "PASS"
    if (
        pass_path.resolve(strict=True) != pass_path
        or pass_path.is_symlink()
        or pass_path.read_bytes() != (THRESHOLD_SELECTOR_MARKER + "\n").encode("ascii")
    ):
        raise TasteGCFFullResumeError("T12 threshold selector PASS is invalid")

    threshold = raw.get("neurosed_distance_threshold")
    raw_quantiles = raw.get("raw_quantile_thresholds")
    if (
        set(raw) != _THRESHOLD_AUTHORITY_KEYS
        or raw.get("schema_version") != THRESHOLD_AUTHORITY_SCHEMA
        or raw.get("status") != "PASS"
        or raw.get("marker") != THRESHOLD_AUTHORITY_MARKER
        or raw.get("dataset") != "tastemolnet"
        or raw.get("method_consumer") != "GCFExplainer"
        or raw.get("distance_line")
        != "official_normged_generated_query_to_original_target_v1"
        or raw.get("inference_direction")
        != "generated_query_to_original_target"
        or raw.get("distance_normalization")
        != "divide_by_sum_graph_element_counts"
        or raw.get("selection_split") != "calibration"
        or raw.get("threshold_source_split") != "calibration"
        or raw.get("threshold_source")
        != "tastemolnet_t4_strict_flip_neurosed_q30_v1"
        or raw.get("objective")
        != (
            "method_independent_empirical_distance_quantiles_over_all_finite_"
            "t4_calibration_strict_flip_residual_to_parent_pairs"
        )
        or raw.get("quantile_method") != "linear"
        or raw.get("dtype") != "float64"
        or raw.get("requested_quantiles") != list(_THRESHOLD_QUANTILES)
        or raw.get("theta_star_quantile") != 0.30
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or float(threshold) < 0.0
        or type(raw_quantiles) is not list
        or len(raw_quantiles) != len(_THRESHOLD_QUANTILES)
        or type(raw.get("finite_strict_flip_distance_count")) is not int
        or raw["finite_strict_flip_distance_count"] <= 0
        or raw.get("tie_break")
        != (
            "numpy_float64_linear_interpolation; equal_adjacent_order_"
            "statistics_retain_the_identical_smaller_threshold"
        )
        or raw.get("shared_across_t7_training_and_evaluation") is not True
        or raw.get("threshold_fitted_on_test") is not False
        or raw.get("selection_used_test") is not False
        or raw.get("test_used_for_selection") is not False
        or raw.get("train_payload_loaded") is not False
        or raw.get("validation_payload_loaded") is not False
        or raw.get("test_payload_loaded") is not False
        or raw.get("cf_mode") != "strict_flip"
        or type(raw.get("selected_at")) is not str
        or not raw["selected_at"]
    ):
        raise TasteGCFFullResumeError(
            "T12 NeuroSED threshold authority semantics changed"
        )
    for index, (row, quantile) in enumerate(
        zip(raw_quantiles, _THRESHOLD_QUANTILES, strict=True)
    ):
        value = row.get("threshold") if type(row) is dict else None
        if (
            type(row) is not dict
            or set(row) != {"quantile", "threshold"}
            or row.get("quantile") != quantile
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or (index > 0 and float(value) < float(raw_quantiles[index - 1]["threshold"]))
        ):
            raise TasteGCFFullResumeError("T12 NeuroSED quantile authority changed")
    if float(threshold) != float(raw_quantiles[3]["threshold"]):
        raise TasteGCFFullResumeError("T12 NeuroSED q30 pin is inconsistent")

    input_digest = _require_sha256(
        raw.get("input_authority_sha256"), field="threshold input authority SHA"
    )
    pair_digest = _require_sha256(
        raw.get("pair_inventory_sha256"), field="threshold pair inventory SHA"
    )
    if (
        set(input_authority) != _THRESHOLD_INPUT_KEYS
        or input_authority.get("schema_version") != THRESHOLD_INPUT_SCHEMA
        or input_authority.get("opened_payload_splits") != ["calibration"]
        or input_authority.get("train_payload_loaded") is not False
        or input_authority.get("validation_payload_loaded") is not False
        or input_authority.get("test_payload_loaded") is not False
        or input_authority.get("input_authority_sha256") != input_digest
        or _sha256_bytes(
            _canonical_bytes(
                {
                    key: value
                    for key, value in input_authority.items()
                    if key != "input_authority_sha256"
                }
            )
        )
        != input_digest
        or input_authority.get("neurosed_checkpoint_sha256")
        != _require_sha256(
            expected_neurosed_checkpoint_sha256,
            field="expected NeuroSED checkpoint SHA",
        )
        or input_authority.get("neurosed_feature_schema_sha256")
        != _require_sha256(
            expected_neurosed_feature_schema_sha256,
            field="expected NeuroSED feature-schema SHA",
        )
        or input_authority.get("t3_checkpoint_sha256")
        != _require_sha256(expected_t3_checkpoint_id, field="expected T3 checkpoint ID")
        or input_authority.get("t3_temperature_scaling_sha256")
        != _require_sha256(
            expected_t3_temperature_sha256, field="expected T3 temperature SHA"
        )
        or input_authority.get("t3_gate_sha256")
        != _require_sha256(expected_t3_gate_sha256, field="expected T3 gate SHA")
        or input_authority.get("t3_verification_sha256")
        != _require_sha256(
            expected_t3_verification_sha256, field="expected T3 verification SHA"
        )
        or input_authority.get("official_gcf_inventory_sha256")
        != _require_sha256(
            expected_official_inventory_sha256,
            field="expected official GCF inventory SHA",
        )
        or _absolute(
            input_authority.get("managed_neurosed_root"),
            field="threshold managed NeuroSED root",
        )
        != _absolute(
            expected_managed_neurosed_root, field="expected managed NeuroSED root"
        )
        or _absolute(input_authority.get("t3_root"), field="threshold T3 root")
        != _absolute(expected_t3_root, field="expected T3 root")
        or _absolute(
            input_authority.get("official_gcf_root"),
            field="threshold official GCF root",
        )
        != _absolute(expected_official_root, field="expected official GCF root")
    ):
        raise TasteGCFFullResumeError("T12 threshold input authority changed")
    for field in (
        "graph_cache_manifest_sha256",
        "calibration_cache_sha256",
        "t4_verification_sha256",
        "t4_oracle_smoke_sha256",
        "molclr_checkpoint_sha256",
    ):
        _require_sha256(input_authority.get(field), field=f"threshold {field}")
    if (
        type(input_authority.get("t4_strict_flip_count")) is not int
        or input_authority["t4_strict_flip_count"]
        != raw["finite_strict_flip_distance_count"]
    ):
        raise TasteGCFFullResumeError("T12 threshold calibration count changed")

    authority_sha = _sha256_bytes(data)
    if (
        set(receipt) != _THRESHOLD_RECEIPT_KEYS
        or receipt.get("schema_version") != THRESHOLD_RECEIPT_SCHEMA
        or receipt.get("status") != "PASS"
        or receipt.get("marker") != THRESHOLD_SELECTOR_MARKER
        or receipt.get("dataset") != "tastemolnet"
        or receipt.get("selection_split") != "calibration"
        or receipt.get("opened_payload_splits") != ["calibration"]
        or receipt.get("train_payload_loaded") is not False
        or receipt.get("validation_payload_loaded") is not False
        or receipt.get("test_payload_loaded") is not False
        or receipt.get("test_used_for_selection") is not False
        or receipt.get("paper_cell_published") is not False
        or receipt.get("strict_flip_pair_count")
        != raw["finite_strict_flip_distance_count"]
        or receipt.get("pair_inventory_sha256") != pair_digest
        or receipt.get("input_authority_sha256") != input_digest
        or receipt.get("neurosed_authority_sha256") != authority_sha
        or type(receipt.get("wnode_runtime_stats")) is not dict
        or type(receipt.get("selected_at")) is not str
        or not receipt["selected_at"]
    ):
        raise TasteGCFFullResumeError("T12 threshold selector receipt changed")
    for field in ("wnode_contract_sha256", "distance_rows_sha256"):
        _require_sha256(receipt.get(field), field=f"threshold receipt {field}")

    checksum_path = root / "sha256sums.txt"
    if checksum_path.resolve(strict=True) != checksum_path or checksum_path.is_symlink():
        raise TasteGCFFullResumeError("T12 threshold checksum inventory is invalid")
    checksums: dict[str, str] = {}
    try:
        for line in checksum_path.read_text(encoding="ascii").splitlines():
            digest, separator, name = line.partition("  ")
            if (
                not separator
                or _SHA256.fullmatch(digest) is None
                or Path(name).name != name
                or name in checksums
            ):
                raise TasteGCFFullResumeError(
                    "T12 threshold checksum inventory is malformed"
                )
            checksums[name] = digest
    except (OSError, UnicodeDecodeError) as exc:
        raise TasteGCFFullResumeError(
            "T12 threshold checksum inventory is unreadable"
        ) from exc
    expected_files = {
        "input_authority.json",
        "calibration_distance_rows.jsonl",
        "selection_receipt.json",
        "t7_neurosed_threshold_authority.json",
        "tastemolnet.json",
    }
    if set(checksums) != expected_files:
        raise TasteGCFFullResumeError("T12 threshold checksum inventory changed")
    for name, digest in checksums.items():
        candidate = root / name
        if (
            candidate.resolve(strict=True) != candidate
            or candidate.is_symlink()
            or _sha256_file(candidate) != digest
        ):
            raise TasteGCFFullResumeError(f"T12 threshold file changed: {name}")
    if (
        checksums["t7_neurosed_threshold_authority.json"] != authority_sha
        or checksums["input_authority.json"] != _sha256_file(root / "input_authority.json")
        or checksums["selection_receipt.json"]
        != _sha256_file(root / "selection_receipt.json")
        or checksums["calibration_distance_rows.jsonl"]
        != receipt["distance_rows_sha256"]
        or checksums["tastemolnet.json"] != receipt["wnode_contract_sha256"]
    ):
        raise TasteGCFFullResumeError("T12 threshold receipt/inventory binding changed")
    return dict(raw), authority_sha


def require_real_a800(*, gpu_uuid: str, torch: Any) -> dict[str, Any]:
    """Bind local cuda:0 to one externally named physical A800 UUID."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if (
        type(gpu_uuid) is not str
        or _GPU_UUID.fullmatch(gpu_uuid) is None
        or (
            re.fullmatch(r"[0-9]+", visible) is None
            and _GPU_UUID.fullmatch(visible) is None
        )
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise TasteGCFFullResumeError(
            "T12 canary requires one numeric/UUID CUDA_VISIBLE_DEVICES selector and CUDA"
        )
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            visible,
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if result.returncode != 0 or len(rows) != 1:
        raise TasteGCFFullResumeError("T12 cannot bind the physical GPU")
    parts = [value.strip() for value in rows[0].split(",")]
    if (
        len(parts) != 4
        or parts[1] != gpu_uuid
        or "A800" not in parts[2].upper()
    ):
        raise TasteGCFFullResumeError("T12 GPU is not the authorized A800 UUID")
    try:
        physical_index = int(parts[0])
        total_memory_mib = int(parts[3])
    except ValueError as exc:
        raise TasteGCFFullResumeError("T12 GPU memory evidence is malformed") from exc
    properties = torch.cuda.get_device_properties(0)
    if "A800" not in str(properties.name).upper():
        raise TasteGCFFullResumeError("T12 torch cuda:0 is not an A800")
    return {
        "schema_version": "tastemolnet_t12_a800_runtime_v1",
        "visible_selector": visible,
        "physical_index": physical_index,
        "gpu_uuid": gpu_uuid,
        "gpu_name": parts[2],
        "nvidia_smi_total_memory_mib": total_memory_mib,
        "torch_device_name": str(properties.name),
        "torch_total_memory_bytes": int(properties.total_memory),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version() or 0),
        "cuda_device_count": 1,
        "cuda_used": True,
    }


def configure_exact_cuda_replay(*, torch: Any) -> dict[str, Any]:
    """Pin and prove every supported Torch/CUDA determinism switch.

    Unsupported PyG/CUDA kernels are allowed to raise at execution time.  That
    is an honest canary failure and must not be weakened to an approximate
    comparison.
    """

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise TasteGCFFullResumeError(
            "T12 exact replay requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    if os.environ.get("PYTHONHASHSEED") != str(CANARY_SEED):
        raise TasteGCFFullResumeError("T12 exact replay requires PYTHONHASHSEED=7")
    required = (
        "use_deterministic_algorithms",
        "are_deterministic_algorithms_enabled",
        "set_deterministic_debug_mode",
        "get_deterministic_debug_mode",
        "set_float32_matmul_precision",
        "get_float32_matmul_precision",
    )
    if any(not hasattr(torch, name) for name in required):
        raise TasteGCFFullResumeError(
            "T12 Torch lacks the required exact-replay controls"
        )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_deterministic_debug_mode("error")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    optional_reductions: dict[str, bool] = {}
    for name in (
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    ):
        if hasattr(torch.backends.cuda.matmul, name):
            setattr(torch.backends.cuda.matmul, name, False)
            optional_reductions[name] = bool(
                getattr(torch.backends.cuda.matmul, name)
            )
    result = {
        "schema_version": "tastemolnet_t12_cuda_determinism_v1",
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "python_hash_seed": os.environ["PYTHONHASHSEED"],
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_debug_mode": int(torch.get_deterministic_debug_mode()),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "optional_reduced_precision_reductions": optional_reductions,
    }
    if (
        result["deterministic_algorithms"] is not True
        or result["deterministic_debug_mode"] != 2
        or result["float32_matmul_precision"] != "highest"
        or result["cudnn_deterministic"] is not True
        or result["cudnn_benchmark"] is not False
        or result["cudnn_allow_tf32"] is not False
        or result["cuda_matmul_allow_tf32"] is not False
        or any(optional_reductions.values())
    ):
        raise TasteGCFFullResumeError("T12 CUDA determinism controls did not stick")
    return result


@dataclass(slots=True)
class _PreparedCanary:
    vrrw: Any
    importance: Any
    input_graphs: list[Any]
    adapter: Any
    bridge: T12StableGCFBridge
    action_counts: Counter[str]
    importance_args: Mapping[str, Any]
    original_neighbor: Any
    parent_evidence: Mapping[str, Any]
    source_cohort_sha256: str
    coverage_runtime: Any


class _BoundedNeuroSEDCoverage:
    """Official coverage math with bounded CUDA-OOM-only batch fallback."""

    _SCHEMA = "tastemolnet_t12_neurosed_retry_state_v1"

    def __init__(self, importance: Any) -> None:
        self.importance = importance
        self.calls: list[dict[str, Any]] = []

    def _is_cuda_oom(self, error: RuntimeError) -> bool:
        torch = self.importance.torch
        oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if isinstance(oom_type, type) and isinstance(error, oom_type):
            return True
        message = str(error).lower()
        return "out of memory" in message and (
            "cuda" in message or "cublas" in message or "cudnn" in message
        )

    def __call__(
        self,
        neurosed_model: Any,
        dataset: Any,
        original_graphs_element_counts: Any,
        threshold: float,
    ) -> Any:
        torch = self.importance.torch
        requested = len(dataset)
        if type(requested) is not int or requested <= 0:
            raise TasteGCFFullResumeError(
                "T12 NeuroSED coverage received an empty dataset"
            )
        graph_element_counts = self.importance.util.graph_element_counts(dataset)
        batch_size = requested
        attempted: list[int] = []
        while True:
            attempted.append(batch_size)
            try:
                distances = neurosed_model.predict_outer_with_queries(
                    dataset, batch_size=batch_size
                ).cpu()
                break
            except RuntimeError as exc:
                if not self._is_cuda_oom(exc) or batch_size <= 1:
                    raise
                next_size = max(1, batch_size // 2)
                if next_size >= batch_size:
                    raise TasteGCFFullResumeError(
                        "T12 NeuroSED OOM fallback did not make progress"
                    ) from exc
                batch_size = next_size
                torch.cuda.empty_cache()
        sums = torch.cartesian_prod(
            graph_element_counts, original_graphs_element_counts
        ).sum(dim=1).view(requested, len(original_graphs_element_counts))
        selected = distances / sums <= threshold
        self.calls.append(
            {
                "call_index": len(self.calls),
                "requested_graph_count": requested,
                "attempted_batch_sizes": attempted,
                "selected_batch_size": batch_size,
                "cuda_oom_retry_count": len(attempted) - 1,
            }
        )
        return selected.float()

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema_version": self._SCHEMA,
            "bounded_cuda_oom_retry": True,
            "calls": [
                {
                    **row,
                    "attempted_batch_sizes": list(row["attempted_batch_sizes"]),
                }
                for row in self.calls
            ],
        }

    def restore_checkpoint_state(self, raw: Mapping[str, Any]) -> None:
        if (
            type(raw) is not dict
            or set(raw) != {
                "schema_version",
                "bounded_cuda_oom_retry",
                "calls",
            }
            or raw.get("schema_version") != self._SCHEMA
            or raw.get("bounded_cuda_oom_retry") is not True
            or type(raw.get("calls")) is not list
        ):
            raise TasteGCFFullResumeError(
                "T12 NeuroSED retry checkpoint schema changed"
            )
        restored: list[dict[str, Any]] = []
        expected_keys = {
            "call_index",
            "requested_graph_count",
            "attempted_batch_sizes",
            "selected_batch_size",
            "cuda_oom_retry_count",
        }
        for index, row in enumerate(raw["calls"]):
            attempts = row.get("attempted_batch_sizes") if type(row) is dict else None
            if (
                type(row) is not dict
                or set(row) != expected_keys
                or row.get("call_index") != index
                or type(row.get("requested_graph_count")) is not int
                or row["requested_graph_count"] <= 0
                or type(attempts) is not list
                or not attempts
                or any(type(value) is not int or value <= 0 for value in attempts)
                or attempts[0] != row["requested_graph_count"]
                or any(
                    current != max(1, previous // 2)
                    for previous, current in zip(attempts, attempts[1:])
                )
                or row.get("selected_batch_size") != attempts[-1]
                or row.get("cuda_oom_retry_count") != len(attempts) - 1
            ):
                raise TasteGCFFullResumeError(
                    "T12 NeuroSED retry checkpoint semantics changed"
                )
            restored.append(
                {**row, "attempted_batch_sizes": list(attempts)}
            )
        self.calls = restored


@contextmanager
def _installed_bounded_neurosed_coverage(
    importance: Any, runtime: _BoundedNeuroSEDCoverage
):
    original = importance.neurosed_threshold_coverage_estimation
    importance.neurosed_threshold_coverage_estimation = runtime
    try:
        yield
    finally:
        importance.neurosed_threshold_coverage_estimation = original


def _prepare_canary(*, sources: Any, device: str) -> _PreparedCanary:
    import numpy as np
    import torch

    loaded = load_train_rows(
        sources.train_bytes,
        source_path=Path(sources.train_contract["path"]),
        expected_num_records=sources.train_contract["num_records"],
        expected_label_counts=sources.train_contract["label_counts"],
    )
    modules = _official_modules(sources.official_root)
    vrrw = modules["vrrw"]
    importance = modules["importance"]
    distance = modules["distance"]
    _reset_official_vrrw(vrrw)
    pool_rows = loaded.sweet_rows[:CANARY_SOURCE_POOL_LIMIT]
    if len(pool_rows) != CANARY_SOURCE_POOL_LIMIT:
        raise TasteGCFFullResumeError("T12 canary Sweet source pool changed")
    pool_records = [encode_taste_source_graph(row, loaded.schema) for row in pool_rows]
    pool_graphs = [
        taste_record_to_pyg(record, origin_index=index)
        for index, record in enumerate(pool_records)
    ]
    pool_adapter = TasteFrozenGINENativeAdapter(
        sources.checkpoint_payloads,
        source_records=pool_records,
        graph_schema=loaded.schema,
        device=device,
    )
    selected_records, parent_evidence = _select_sources(
        adapter=pool_adapter,
        pool_graphs=pool_graphs,
        pool_records=pool_records,
    )
    if len(selected_records) != CANARY_PARENT_COUNT:
        raise TasteGCFFullResumeError("T12 canary parent selection changed")
    del pool_adapter, pool_graphs, pool_records
    torch.cuda.empty_cache()
    input_graphs = [
        taste_record_to_pyg(record, origin_index=index)
        for index, record in enumerate(selected_records)
    ]
    adapter = TasteFrozenGINENativeAdapter(
        sources.checkpoint_payloads,
        source_records=selected_records,
        graph_schema=loaded.schema,
        device=device,
    )
    replay = adapter.score(input_graphs)
    if any(
        not valid or prediction != SOURCE_LABEL
        for valid, prediction in zip(
            replay.valid_fullgraphs, replay.predictions, strict=True
        )
    ):
        raise TasteGCFFullResumeError("T12 selected source replay changed")
    sources.revalidate()
    neurosed = distance.load_neurosed(
        input_graphs,
        neurosed_model_path=f"/proc/self/fd/{sources.neurosed_model.file_fd}",
        device=device,
    )
    sources.revalidate()
    original_counts = importance.util.graph_element_counts(input_graphs)

    random.seed(CANARY_SEED)
    np.random.seed(CANARY_SEED)
    torch.manual_seed(CANARY_SEED)
    torch.cuda.manual_seed_all(CANARY_SEED)
    original_neighbor = vrrw.neighbor_graph_access
    lineage_neighbor = graph_lineage_neighbor_wrapper(original_neighbor)
    action_counts: Counter[str] = Counter()

    def counted_neighbor(graph: Any, action: tuple[Any, ...]) -> Any:
        action_counts[str(action[0])] += 1
        return lineage_neighbor(graph, action)

    vrrw.neighbor_graph_access = counted_neighbor
    vrrw.dataset_name = "tastemolnet"
    vrrw.alpha = CANARY_ALPHA
    vrrw.sample_size = CANARY_SAMPLE_SIZE
    vrrw.is_sample = True
    vrrw.MAX_COUNTERFACTUAL_SIZE = CANARY_CANDIDATE_CAPACITY
    vrrw.input_graphs_covered = torch.zeros(
        CANARY_PARENT_COUNT, dtype=torch.float
    )
    bridge = T12StableGCFBridge(
        adapter=adapter,
        vrrw=vrrw,
        importance=importance,
        neurosed_model=neurosed,
        original_graph_element_counts=original_counts,
        distance_threshold=sources.authority.neurosed_distance_threshold,
        parent_count=CANARY_PARENT_COUNT,
        feature_atomic_numbers=loaded.schema.feature_atomic_numbers,
        coverage_runtime=(coverage_runtime := _BoundedNeuroSEDCoverage(importance)),
    )
    importance_args = {
        "schema_version": "tastemolnet_t12_gcf_neurosed_importance_v1",
        "alpha": CANARY_ALPHA,
        "distance_status": "EVALUATED",
        "distance_threshold": float(sources.authority.neurosed_distance_threshold),
        "selector_status": "NOT_EVALUATED",
        "calibration_loaded": False,
        "test_loaded": False,
    }
    source_cohort = _sha256_bytes(
        _canonical_bytes(
            {
                "schema_version": "tastemolnet_t12_canary_source_cohort_v1",
                "graph_schema_sha256": loaded.evidence["graph_schema_sha256"],
                "ordered_sources": [
                    {
                        "source_graph_hash": str(record["source_graph_hash"]),
                        "molecule_id": str(record["molecule_id"]),
                        "source_row_index": int(record["source_row_index"]),
                    }
                    for record in selected_records
                ],
            }
        )
    )
    return _PreparedCanary(
        vrrw=vrrw,
        importance=importance,
        input_graphs=input_graphs,
        adapter=adapter,
        bridge=bridge,
        action_counts=action_counts,
        importance_args=importance_args,
        original_neighbor=original_neighbor,
        parent_evidence=parent_evidence,
        source_cohort_sha256=source_cohort,
        coverage_runtime=coverage_runtime,
    )


def _runtime_identity(
    *,
    sources: Any,
    threshold_authority_sha256: str,
    gpu: Mapping[str, Any],
    determinism: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    import numpy as np
    import torch

    value = {
        "schema_version": "tastemolnet_t12_runtime_identity_v1",
        "execution_commit": sources.authority.implementation_commit,
        "execution_tree": sources.authority.implementation_tree,
        "official_source_inventory_sha256": (
            sources.authority.official_gcf_inventory_sha256
        ),
        "threshold_authority_sha256": threshold_authority_sha256,
        "python_version": os.sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version() or 0),
        "gpu": dict(gpu),
        "determinism": dict(determinism),
    }
    return value, _sha256_bytes(_canonical_bytes(value))


def _checkpoint_identity(
    *,
    sources: Any,
    attempt_id: str,
    generation_token: str,
    source_cohort_sha256: str,
    threshold_authority_sha256: str,
    runtime_identity_sha256: str,
    gpu_uuid: str,
) -> dict[str, Any]:
    model_config_sha = _sha256_bytes(
        _canonical_bytes(
            {
                name: _sha256_bytes(data)
                for name, data in sorted(sources.checkpoint_payloads.items())
            }
        )
    )
    value = {
        "schema_version": "tastemolnet_t12_checkpoint_identity_v1",
        "stage": STAGE,
        "purpose": "gpu_replay_canary",
        "attempt_id": attempt_id,
        "generation_token": generation_token,
        "total_steps": CANARY_TOTAL_STEPS,
        "checkpoint_cursor": CANARY_CHECKPOINT_CURSOR,
        "source_cohort_sha256": source_cohort_sha256,
        "train_split_sha256": sources.pins.train_split_sha,
        "model_checkpoint_sha256": sources.pins.t3_calibrated_gine_sha,
        "model_config_sha256": model_config_sha,
        "neurosed_checkpoint_sha256": sources.pins.neurosed_model_sha,
        "neurosed_distance_threshold_hex": float(
            sources.authority.neurosed_distance_threshold
        ).hex(),
        "neurosed_threshold_authority_sha256": threshold_authority_sha256,
        "official_source_inventory_sha256": (
            sources.authority.official_gcf_inventory_sha256
        ),
        "execution_commit": sources.authority.implementation_commit,
        "execution_tree": sources.authority.implementation_tree,
        "runtime_identity_sha256": runtime_identity_sha256,
        "gpu_uuid": gpu_uuid,
        "device": "cuda:0",
        "graph_identity_contract": GRAPH_IDENTITY_CONTRACT,
        "seed": CANARY_SEED,
        "alpha_hex": CANARY_ALPHA.hex(),
        "teleport_hex": CANARY_TELEPORT.hex(),
        "sample_size": CANARY_SAMPLE_SIZE,
        "candidate_capacity": CANARY_CANDIDATE_CAPACITY,
        "train_loaded": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "rf_oracle_used": False,
    }
    return validate_checkpoint_identity(value)


def _canary_identity_sha256(
    checkpoint_identity: Mapping[str, Any]
) -> str:
    ignored = {"attempt_id", "generation_token", "checkpoint_cursor"}
    return _sha256_bytes(
        _canonical_bytes(
            {
                "schema_version": "tastemolnet_t12_gpu_replay_identity_v1",
                **{
                    key: value
                    for key, value in checkpoint_identity.items()
                    if key not in ignored
                },
                "checkpoint_cursor": CANARY_CHECKPOINT_CURSOR,
                "terminal_step": CANARY_TOTAL_STEPS,
            }
        )
    )


def _run_identity(
    *,
    run_kind: str,
    checkpoint_identity: Mapping[str, Any],
    canary_identity_sha256: str,
    threshold_authority_path: Path,
    threshold_authority_sha256: str,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": RUN_IDENTITY_SCHEMA,
        "stage": STAGE,
        "run_kind": run_kind,
        "checkpoint_identity": dict(checkpoint_identity),
        "checkpoint_identity_sha256": _sha256_bytes(
            _canonical_bytes(checkpoint_identity)
        ),
        "canary_identity_sha256": canary_identity_sha256,
        "threshold_authority_path": str(threshold_authority_path),
        "threshold_authority_sha256": threshold_authority_sha256,
        "runtime": dict(runtime),
        "runtime_identity_sha256": checkpoint_identity["runtime_identity_sha256"],
        "calibration_loaded": False,
        "test_loaded": False,
        "production_released": False,
    }


def _fresh_root(path: str | Path) -> Path:
    root = _absolute(path, field="T12 canary output root", must_exist=False)
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    if root.resolve(strict=True) != root:
        raise TasteGCFFullResumeError("T12 canary output root is an alias")
    return root


def _reopen_run_identity(root: Path, expected: Mapping[str, Any]) -> None:
    path = root / "run_identity.json"
    if path.resolve(strict=True) != path or path.is_symlink():
        raise TasteGCFFullResumeError("T12 canary run identity is an alias")
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteGCFFullResumeError("T12 canary run identity is unreadable") from exc
    if observed != dict(expected):
        raise TasteGCFFullResumeError("T12 canary run identity changed")


def _validate_terminal(prepared: _PreparedCanary, native: Mapping[str, Any]) -> None:
    if (
        type(native) is not dict
        or len(prepared.vrrw.traversed_hashes) != CANARY_TOTAL_STEPS
        or native.get("traversed_hashes") != prepared.vrrw.traversed_hashes
        or _semantic_sha256(native.get("graph_map"))
        != _semantic_sha256(prepared.vrrw.graph_map)
        or _semantic_sha256(native.get("graph_index_map"))
        != _semantic_sha256(prepared.vrrw.graph_index_map)
        or not prepared.bridge.records
        or not any(record.candidate for record in prepared.bridge.records.values())
        or prepared.bridge.calculate_hash_count
        != prepared.bridge.evaluated_graph_count
        or sum(prepared.action_counts.values()) <= 0
    ):
        raise TasteGCFFullResumeError("T12 bounded official terminal closure failed")


def run_replay_canary_phase(
    *,
    mode: str,
    output_root: str | Path,
    observation_path: str | Path | None,
    checkpoint_manifest: str | Path | None,
    attempt_id: str,
    generation_token: str,
    gpu_uuid: str,
    managed_neurosed_root: str | Path,
    t3_root: str | Path,
    official_root: str | Path,
    threshold_authority_path: str | Path,
) -> dict[str, Any]:
    """Execute exactly one process role of the real bounded A800 canary."""

    if mode not in {"uninterrupted", "checkpoint", "resume"}:
        raise TasteGCFFullResumeError("T12 canary mode is invalid")
    try:
        parsed_attempt = uuid.UUID(attempt_id)
    except (ValueError, AttributeError) as exc:
        raise TasteGCFFullResumeError("T12 canary attempt ID is invalid") from exc
    if parsed_attempt.version != 4 or str(parsed_attempt) != attempt_id:
        raise TasteGCFFullResumeError("T12 canary attempt ID is not UUIDv4")
    _require_sha256(generation_token, field="T12 canary generation token")
    import numpy as np
    import torch

    determinism = configure_exact_cuda_replay(torch=torch)
    gpu = require_real_a800(gpu_uuid=gpu_uuid, torch=torch)
    random.seed(CANARY_SEED)
    np.random.seed(CANARY_SEED)
    torch.manual_seed(CANARY_SEED)
    torch.cuda.manual_seed_all(CANARY_SEED)
    from src.utils.tastemolnet_t7_typed_release_v1 import hold_t7_release_sources

    # The threshold file is opened before source construction so no default can
    # enter hold_t7_release_sources.  Its model hash is checked again after the
    # held source pins are available.
    threshold_path = _absolute(
        threshold_authority_path, field="T12 threshold authority"
    )
    threshold_raw, _ = _json_object(
        threshold_path, label="T12 NeuroSED threshold authority"
    )
    threshold_value = threshold_raw.get("neurosed_distance_threshold")
    if (
        isinstance(threshold_value, bool)
        or not isinstance(threshold_value, (int, float))
        or not math.isfinite(float(threshold_value))
        or float(threshold_value) < 0.0
    ):
        raise TasteGCFFullResumeError("T12 threshold authority has no usable value")
    with hold_t7_release_sources(
        managed_neurosed_root=managed_neurosed_root,
        t3_root=t3_root,
        official_gcf_root=official_root,
        neurosed_distance_threshold=float(threshold_value),
    ) as sources:
        threshold, threshold_sha = load_threshold_authority(
            threshold_path,
            expected_neurosed_checkpoint_sha256=sources.pins.neurosed_model_sha,
            expected_neurosed_feature_schema_sha256=sources.neurosed_evidence[
                "feature_schema_sha256"
            ],
            expected_t3_checkpoint_id=sources.authority.t3_checkpoint_id,
            expected_t3_temperature_sha256=sources.pins.t3_temperature_sha,
            expected_t3_gate_sha256=sources.authority.t3_gate_sha256,
            expected_t3_verification_sha256=(
                sources.authority.t3_verification_sha256
            ),
            expected_official_inventory_sha256=(
                sources.authority.official_gcf_inventory_sha256
            ),
            expected_managed_neurosed_root=(
                sources.authority.managed_neurosed_root
            ),
            expected_t3_root=sources.authority.t3_root,
            expected_official_root=sources.authority.official_gcf_root,
        )
        if float(threshold["neurosed_distance_threshold"]) != float(
            sources.authority.neurosed_distance_threshold
        ):
            raise TasteGCFFullResumeError("T12 held threshold changed")
        runtime, runtime_sha = _runtime_identity(
            sources=sources,
            threshold_authority_sha256=threshold_sha,
            gpu=gpu,
            determinism=determinism,
        )
        prepared = _prepare_canary(sources=sources, device="cuda:0")
        checkpoint_identity = _checkpoint_identity(
            sources=sources,
            attempt_id=attempt_id,
            generation_token=generation_token,
            source_cohort_sha256=prepared.source_cohort_sha256,
            threshold_authority_sha256=threshold_sha,
            runtime_identity_sha256=runtime_sha,
            gpu_uuid=gpu_uuid,
        )
        canary_identity = _canary_identity_sha256(checkpoint_identity)
        run_kind = "uninterrupted" if mode == "uninterrupted" else "resumable"
        run_identity = _run_identity(
            run_kind=run_kind,
            checkpoint_identity=checkpoint_identity,
            canary_identity_sha256=canary_identity,
            threshold_authority_path=threshold_path,
            threshold_authority_sha256=threshold_sha,
            runtime=runtime,
        )
        if mode in {"uninterrupted", "checkpoint"}:
            root = _fresh_root(output_root)
            _write_new(root / "run_identity.json", run_identity)
        else:
            root = _absolute(output_root, field="T12 resumed output root")
            _reopen_run_identity(root, run_identity)
        runtime_name = {
            "uninterrupted": "uninterrupted-runtime",
            "checkpoint": "prefix-runtime",
            "resume": "resumed-runtime",
        }[mode]
        runtime_root = root / runtime_name
        runtime_root.mkdir(mode=0o700, exist_ok=False)
        old_cwd = Path.cwd()
        current_graph: str | None = None
        native_result_path: Path | None = None
        try:
            os.chdir(runtime_root)
            with (
                _installed_bounded_neurosed_coverage(
                    prepared.importance, prepared.coverage_runtime
                ),
                prepared.bridge.installed(),
                _official_vrrw_alpha_endpoint_patch(prepared.vrrw),
                _installed_official_importance_args(
                    prepared.vrrw, prepared.importance_args
                ),
            ):
                if mode == "resume":
                    if checkpoint_manifest is None:
                        raise TasteGCFFullResumeError(
                            "T12 resume requires a checkpoint manifest"
                        )
                    loaded = reopen_checkpoint(
                        checkpoint_manifest,
                        expected_identity=checkpoint_identity,
                        torch=torch,
                    )
                    prefix_raw, _ = _json_object(
                        root / "prefix_receipt.json",
                        label="T12 checkpoint prefix receipt",
                    )
                    prefix_receipt = validate_canary_prefix_receipt(prefix_raw)
                    if (
                        prefix_receipt["checkpoint_manifest"]
                        != str(
                            _absolute(
                                checkpoint_manifest,
                                field="T12 checkpoint manifest",
                            )
                        )
                        or prefix_receipt["canary_identity_sha256"]
                        != canary_identity
                        or prefix_receipt["gpu_uuid"] != gpu_uuid
                        or prefix_receipt["checkpoint_identity_sha256"]
                        != loaded["identity_sha256"]
                        or prefix_receipt["checkpoint_state_sha256"]
                        != loaded["state_sha256"]
                        or prefix_receipt["checkpoint_rng_sha256"]
                        != loaded["rng_sha256"]
                    ):
                        raise TasteGCFFullResumeError(
                            "T12 checkpoint prefix receipt differs on resume"
                        )
                    current_graph = restore_checkpoint_payload(
                        loaded,
                        expected_identity=checkpoint_identity,
                        vrrw=prepared.vrrw,
                        bridge=prepared.bridge,
                        adapter=prepared.adapter,
                        action_counts=prepared.action_counts,
                        np=np,
                        torch=torch,
                    )
                    segment = _run_official_walk_segment(
                        vrrw=prepared.vrrw,
                        input_graphs=prepared.input_graphs,
                        importance_args=prepared.importance_args,
                        teleport_probability=CANARY_TELEPORT,
                        start_step=CANARY_CHECKPOINT_CURSOR + 1,
                        end_step=CANARY_TOTAL_STEPS,
                        resume_graph_hash=current_graph,
                    )
                    if not segment.resume_entry_used_saved_graph:
                        raise TasteGCFFullResumeError(
                            "T12 resume did not consume the saved graph"
                        )
                    current_graph = segment.current_graph_hash
                else:
                    end_step = (
                        CANARY_TOTAL_STEPS
                        if mode == "uninterrupted"
                        else CANARY_CHECKPOINT_CURSOR
                    )
                    segment = _run_official_walk_segment(
                        vrrw=prepared.vrrw,
                        input_graphs=prepared.input_graphs,
                        importance_args=prepared.importance_args,
                        teleport_probability=CANARY_TELEPORT,
                        start_step=1,
                        end_step=end_step,
                    )
                    current_graph = segment.current_graph_hash
                sources.revalidate()
                if mode == "checkpoint":
                    payload = capture_checkpoint_payload(
                        identity=checkpoint_identity,
                        vrrw=prepared.vrrw,
                        bridge=prepared.bridge,
                        adapter=prepared.adapter,
                        action_counts=prepared.action_counts,
                        current_graph_identity=current_graph,
                        np=np,
                        torch=torch,
                    )
                    manifest = write_checkpoint(
                        root / "checkpoints", payload, torch=torch
                    )
                    prefix_receipt = {
                        "schema_version": PREFIX_RECEIPT_SCHEMA,
                        "status": "CHECKPOINT_COMMITTED",
                        "stage": STAGE,
                        "checkpoint_manifest": str(manifest),
                        "checkpoint_manifest_sha256": _sha256_file(manifest),
                        "checkpoint_identity_sha256": payload["identity_sha256"],
                        "checkpoint_state_sha256": payload["state_sha256"],
                        "checkpoint_rng_sha256": payload["rng_sha256"],
                        "checkpoint_cursor": CANARY_CHECKPOINT_CURSOR,
                        "total_steps": CANARY_TOTAL_STEPS,
                        "canary_identity_sha256": canary_identity,
                        "gpu_uuid": gpu_uuid,
                        "process_identity": capture_linux_process_identity(),
                        "calibration_loaded": False,
                        "test_loaded": False,
                        "production_released": False,
                    }
                    validate_canary_prefix_receipt(prefix_receipt)
                    _write_new(root / "prefix_receipt.json", prefix_receipt)
                    return prefix_receipt
                native_result_path = (
                    runtime_root
                    / "results/tastemolnet/runs/counterfactuals.pt"
                )
                if not native_result_path.is_file() or native_result_path.stat().st_size <= 0:
                    raise TasteGCFFullResumeError(
                        "T12 official canary native result is absent"
                    )
                native = _torch_load_compat(native_result_path)
                _validate_terminal(prepared, native)
                scientific = build_replay_scientific_state(
                    vrrw=prepared.vrrw,
                    bridge=prepared.bridge,
                    adapter=prepared.adapter,
                    action_counts=prepared.action_counts,
                    current_graph_identity=current_graph,
                    native_result=native,
                    np=np,
                    torch=torch,
                )
                observation = build_canary_observation(
                    role=(
                        "uninterrupted"
                        if mode == "uninterrupted"
                        else "cross_process_resumed"
                    ),
                    canary_identity_sha256=canary_identity,
                    gpu_uuid=gpu_uuid,
                    process_identity=capture_linux_process_identity(),
                    scientific_state=scientific,
                    native_result_sha256=_sha256_file(native_result_path),
                    checkpoint_reloaded=mode == "resume",
                    generated_to_original_neurosed_assertion=True,
                    checkpoint_process_identity=(
                        prefix_receipt["process_identity"]
                        if mode == "resume"
                        else None
                    ),
                    checkpoint_manifest_sha256=(
                        prefix_receipt["checkpoint_manifest_sha256"]
                        if mode == "resume"
                        else None
                    ),
                    checkpoint_identity_sha256=(
                        prefix_receipt["checkpoint_identity_sha256"]
                        if mode == "resume"
                        else None
                    ),
                    checkpoint_state_sha256=(
                        prefix_receipt["checkpoint_state_sha256"]
                        if mode == "resume"
                        else None
                    ),
                    checkpoint_rng_sha256=(
                        prefix_receipt["checkpoint_rng_sha256"]
                        if mode == "resume"
                        else None
                    ),
                )
                if observation_path is None:
                    raise TasteGCFFullResumeError(
                        "T12 terminal canary mode requires an observation path"
                    )
                published = write_canary_observation(
                    observation_path, observation
                )
                return {
                    "status": "OBSERVATION_COMMITTED",
                    "stage": STAGE,
                    "mode": mode,
                    "observation_path": str(published),
                    "observation_sha256": _sha256_file(published),
                    "native_result_path": str(native_result_path),
                    "native_result_sha256": observation["native_result_sha256"],
                    "canary_identity_sha256": canary_identity,
                    "official_random_walk_steps": CANARY_TOTAL_STEPS,
                    "strict_counterfactual_count": sum(
                        record.candidate
                        for record in prepared.bridge.records.values()
                    ),
                    "parent_evidence": dict(prepared.parent_evidence),
                    "calibration_loaded": False,
                    "test_loaded": False,
                    "production_released": False,
                }
        finally:
            os.chdir(old_cwd)
            prepared.vrrw.neighbor_graph_access = prepared.original_neighbor


__all__ = [
    "CANARY_ALPHA",
    "CANARY_CANDIDATE_CAPACITY",
    "CANARY_CHECKPOINT_CURSOR",
    "CANARY_PARENT_COUNT",
    "CANARY_SAMPLE_SIZE",
    "CANARY_SEED",
    "CANARY_SOURCE_POOL_LIMIT",
    "CANARY_TELEPORT",
    "CANARY_TOTAL_STEPS",
    "PREFIX_RECEIPT_SCHEMA",
    "RUN_IDENTITY_SCHEMA",
    "THRESHOLD_AUTHORITY_SCHEMA",
    "configure_exact_cuda_replay",
    "load_threshold_authority",
    "require_real_a800",
    "run_replay_canary_phase",
]
