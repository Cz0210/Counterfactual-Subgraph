"""Typed artifacts for the real TasteMolNet three-class GNN PPO smoke."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence

from src.rewards.gnn_ppo_reward import TASTE_GNN_PPO_REWARD_SCHEMA
from src.utils.retained_output_directory import (
    HeldPublishedTerminalOutput,
)


TASTE_PPO_STAGE = "T6_OURS_SMOKE"
TASTE_PPO_MARKER = "TASTE_OURS_SMOKE_PASS"
TASTE_PPO_REWARD_MANIFEST_SCHEMA = "tastemolnet_gnn_ppo_reward_manifest_v1"
TASTE_PPO_GATE_SCHEMA = "tastemolnet_ours_ppo_smoke_gate_v1"
TASTE_PPO_OBSERVER_SCHEMA = "tastemolnet_ppo_observer_v1"
TASTE_PPO_CHECKPOINT_IDENTITY_SCHEMA = "tastemolnet_lora_checkpoint_identity_v1"
TASTE_PPO_ADAPTER_PARAMETER_SCHEMA = (
    "tastemolnet_lora_adapter_parameter_identity_v1"
)
TASTE_PPO_VALUE_HEAD_PARAMETER_SCHEMA = (
    "tastemolnet_ppo_value_head_parameter_identity_v1"
)
TASTE_PPO_OUTPUT_EVIDENCE_SCHEMA = "tastemolnet_ours_ppo_output_evidence_v1"
TASTE_PPO_REQUIRED_UPDATE_METRICS = (
    "global_step",
    "rollout_batch_size",
    "reward_mean",
    "reward_min",
    "reward_max",
    "ppo_reward_mean",
    "policy_loss",
    "value_loss",
    "total_loss",
    "approx_kl",
    "parse_ok_rate",
    "valid_rate",
    "direct_substructure_rate",
    "final_substructure_rate",
    "projection_used_rate",
    "oracle_ok_rate",
    "cf_flip_rate",
    "cf_drop_mean",
)

TASTE_REWARD_FIELDS = (
    "schema_version",
    "dataset",
    "num_classes",
    "parent_id",
    "parent_smiles",
    "source_label",
    "raw_fragment",
    "core_fragment",
    "final_fragment",
    "parse_ok",
    "connected",
    "direct_substructure",
    "projection_used",
    "deletion_valid",
    "gnn_scored_deletion",
    "pred_before",
    "pred_after",
    "destination_label",
    "p_before_all_classes",
    "p_after_all_classes",
    "logits_before_all_classes",
    "logits_after_all_classes",
    "source_probability_before",
    "source_probability_after",
    "cf_drop",
    "margin_before",
    "margin_after",
    "margin_drop",
    "cf_flip",
    "reward_valid",
    "reward_substructure",
    "reward_cf",
    "reward_size",
    "reward_projection",
    "reward_kl",
    "reward_total",
    "oracle_backend",
    "classifier_type",
    "rf_oracle_used",
    "calibration_loaded",
    "calibration_dataset_loaded",
    "frozen_temperature_calibration_loaded",
    "test_loaded",
    "oracle_checkpoint_hash",
    "temperature_calibration_hash",
    "feature_schema_hash",
    "policy_initializer_hash",
    "reference_policy_hash",
)


def _is_sha256(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def adapter_parameter_identity(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return one finite, normalized tensor identity for a PEFT LoRA state."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - AutoDL dependency
        raise RuntimeError("Taste PPO adapter identity requires torch") from exc
    tensors: dict[str, dict[str, Any]] = {}
    for raw_name, raw_tensor in state.items():
        if type(raw_name) is not str:
            raise ValueError("Taste PPO adapter contains a non-string tensor key")
        name = raw_name.replace(".lora_A.default.weight", ".lora_A.weight").replace(
            ".lora_B.default.weight", ".lora_B.weight"
        )
        if name in tensors:
            raise ValueError("Taste PPO adapter repeats a normalized tensor key")
        if "lora_" not in name.lower() or not isinstance(raw_tensor, torch.Tensor):
            raise ValueError(f"Taste PPO adapter has an unexpected tensor: {name}")
        tensor = raw_tensor.detach().cpu().contiguous()
        if not bool(torch.is_floating_point(tensor)) or not bool(
            torch.isfinite(tensor.float()).all().item()
        ):
            raise ValueError(f"Taste PPO adapter tensor is non-finite: {name}")
        tensors[name] = {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(value) for value in tensor.shape],
            "sha256": hashlib.sha256(
                tensor.view(torch.uint8).numpy().tobytes(order="C")
            ).hexdigest(),
        }
    if not tensors:
        raise ValueError("Taste PPO adapter contains no LoRA tensors")
    payload = {
        "schema_version": TASTE_PPO_ADAPTER_PARAMETER_SCHEMA,
        "tensor_count": len(tensors),
        "all_tensors_finite": True,
        "tensors": {name: tensors[name] for name in sorted(tensors)},
    }
    return {**payload, "parameter_sha256": _canonical_sha256(payload)}


def value_head_parameter_identity(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return one finite identity for the exact serialized PPO value head."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - AutoDL dependency
        raise RuntimeError("Taste PPO value-head identity requires torch") from exc
    tensors: dict[str, dict[str, Any]] = {}
    for name, raw_tensor in state.items():
        if type(name) is not str or not isinstance(raw_tensor, torch.Tensor):
            raise ValueError("Taste PPO value-head state is malformed")
        tensor = raw_tensor.detach().cpu().contiguous()
        if not bool(torch.is_floating_point(tensor)) or not bool(
            torch.isfinite(tensor.float()).all().item()
        ):
            raise ValueError(f"Taste PPO value-head tensor is non-finite: {name}")
        tensors[name] = {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(value) for value in tensor.shape],
            "sha256": hashlib.sha256(
                tensor.view(torch.uint8).numpy().tobytes(order="C")
            ).hexdigest(),
        }
    if not tensors:
        raise ValueError("Taste PPO value-head state is empty")
    payload = {
        "schema_version": TASTE_PPO_VALUE_HEAD_PARAMETER_SCHEMA,
        "tensor_count": len(tensors),
        "all_tensors_finite": True,
        "tensors": {name: tensors[name] for name in sorted(tensors)},
    }
    return {**payload, "parameter_sha256": _canonical_sha256(payload)}


def adapter_parameter_identity_from_model(model: Any) -> dict[str, Any]:
    """Hash exactly the PEFT adapter tensors currently loaded in memory."""

    try:
        from peft import get_peft_model_state_dict
    except ImportError as exc:  # pragma: no cover - AutoDL dependency
        raise RuntimeError("Taste PPO adapter identity requires PEFT") from exc
    state = get_peft_model_state_dict(model)
    if not isinstance(state, Mapping):
        raise ValueError("Taste PPO PEFT state is not a mapping")
    return adapter_parameter_identity(state)


def validate_taste_adapter_checkpoint_reload(
    checkpoint: str | Path,
    *,
    checkpoint_path_is_retained: bool = False,
    checkpoint_display_path: str | Path | None = None,
    expected_base_model_path: str | Path,
    expected_adapter_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Reload exact held checkpoint bytes without resolving an FD-backed path."""

    requested = Path(checkpoint).expanduser()
    parts = requested.parts
    if checkpoint_path_is_retained and (
        not sys.platform.startswith("linux")
        or len(parts) not in {5, 6}
        or parts[:4] != ("/", "proc", "self", "fd")
        or not parts[4].isdigit()
        or (len(parts) == 6 and parts[5] in {"", ".", ".."})
    ):
        raise ValueError("Taste PPO retained checkpoint path is invalid")
    root_path = requested if checkpoint_path_is_retained else requested.resolve(strict=True)
    display_root = (
        Path(checkpoint_display_path).expanduser()
        if checkpoint_display_path is not None
        else root_path
    )
    if not display_root.is_absolute():
        raise ValueError("Taste PPO checkpoint display path must be absolute")

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )

    def stat_identity(value: os.stat_result, *, directory: bool) -> tuple[int, ...]:
        expected_type = stat.S_ISDIR if directory else stat.S_ISREG
        if not expected_type(value.st_mode) or (not directory and value.st_nlink != 1):
            raise ValueError("Taste PPO checkpoint contains a non-physical node")
        return (
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_mode),
            int(value.st_uid),
            int(value.st_gid),
            int(value.st_nlink),
            int(value.st_size),
            int(value.st_mtime_ns),
            int(value.st_ctime_ns),
        )

    def physical_evidence(identity: tuple[int, ...], *, directory: bool) -> dict[str, int]:
        keys = (
            "device",
            "inode",
            "mode",
            "uid",
            "gid",
            "link_count",
            "size",
            "mtime_ns",
            "ctime_ns",
        )
        evidence = dict(zip(keys, identity, strict=True))
        if directory:
            return {
                key: evidence[key]
                for key in ("device", "inode", "mode", "uid", "gid")
            }
        return evidence

    retained_parent_fd = -1
    retained_parent_identity: tuple[int, ...] | None = None
    retained_child_name: str | None = None
    root_fd = -1
    try:
        if checkpoint_path_is_retained:
            base_fd = os.dup(int(parts[4]))
            try:
                base_identity = stat_identity(os.fstat(base_fd), directory=True)
                if len(parts) == 5:
                    root_fd = base_fd
                    base_fd = -1
                else:
                    retained_parent_fd = base_fd
                    base_fd = -1
                    retained_parent_identity = base_identity
                    retained_child_name = parts[5]
                    named = stat_identity(
                        os.stat(
                            retained_child_name,
                            dir_fd=retained_parent_fd,
                            follow_symlinks=False,
                        ),
                        directory=True,
                    )
                    root_fd = os.open(
                        retained_child_name,
                        directory_flags,
                        dir_fd=retained_parent_fd,
                    )
                    if stat_identity(os.fstat(root_fd), directory=True) != named:
                        raise ValueError(
                            "Taste PPO retained checkpoint child changed while opening"
                        )
            finally:
                if base_fd >= 0:
                    os.close(base_fd)
        else:
            root_fd = os.open(root_path, directory_flags)
    except Exception:
        if root_fd >= 0:
            os.close(root_fd)
        if retained_parent_fd >= 0:
            os.close(retained_parent_fd)
        raise

    root_identity = stat_identity(os.fstat(root_fd), directory=True)
    opened: list[int] = []

    def close_all() -> None:
        for descriptor in reversed(opened):
            os.close(descriptor)
        os.close(root_fd)
        if retained_parent_fd >= 0:
            os.close(retained_parent_fd)

    def held_bytes(name: str) -> tuple[bytes, dict[str, Any], int, tuple[int, ...]]:
        named = stat_identity(
            os.stat(name, dir_fd=root_fd, follow_symlinks=False),
            directory=False,
        )
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        opened.append(descriptor)
        held = stat_identity(os.fstat(descriptor), directory=False)
        if held != named or held[6] <= 0:
            raise ValueError("Taste PPO checkpoint leaf changed while opening")
        data = bytearray()
        offset = 0
        while offset < held[6]:
            chunk = os.pread(descriptor, min(8 * 1024 * 1024, held[6] - offset), offset)
            if not chunk:
                raise ValueError("Taste PPO checkpoint leaf ended early")
            data.extend(chunk)
            offset += len(chunk)
        if os.pread(descriptor, 1, held[6]):
            raise ValueError("Taste PPO checkpoint leaf grew while reading")
        payload = bytes(data)
        identity = {
            "path": str(display_root / name),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
            "physical_identity": physical_evidence(held, directory=False),
        }
        return payload, identity, descriptor, held

    try:
        try:
            os.stat("adapter_model.bin", dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("Taste PPO checkpoint must not contain pickle weights")
        config_bytes, config, config_fd, config_identity = held_bytes(
            "adapter_config.json"
        )
        weights_bytes, weights, weights_fd, weights_identity = held_bytes(
            "adapter_model.safetensors"
        )
        value_head_bytes, value_head, value_head_fd, value_head_identity = held_bytes(
            "decoded_chem_value_head.pt"
        )
        try:
            config_payload = json.loads(config_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Taste PPO adapter config is malformed") from exc
        if type(config_payload) is not dict or not config_payload:
            raise ValueError("Taste PPO adapter config is empty or malformed")
        expected_base = str(Path(expected_base_model_path).expanduser())
        if (
            not Path(expected_base).is_absolute()
            or config_payload.get("base_model_name_or_path") != expected_base
            or "/proc/self/fd/" in expected_base
        ):
            raise ValueError("Taste PPO adapter base-model authority drifted")
        if config_payload != dict(expected_adapter_config):
            raise ValueError("Taste PPO adapter config differs from in-memory PEFT authority")
        from safetensors.torch import load as load_safetensors
    except ImportError as exc:  # pragma: no cover - AutoDL dependency
        close_all()
        raise RuntimeError("Taste PPO checkpoint reload requires safetensors") from exc
    except Exception:
        close_all()
        raise
    try:
        tensor_state = load_safetensors(weights_bytes)
    except Exception as exc:
        close_all()
        raise ValueError("Taste PPO checkpoint safetensors cannot be reloaded") from exc
    try:
        parameter_identity = adapter_parameter_identity(tensor_state)
        try:
            import torch

            value_head_state = torch.load(
                io.BytesIO(value_head_bytes),
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise ValueError("Taste PPO value-head checkpoint cannot be reloaded") from exc
        if not isinstance(value_head_state, Mapping):
            raise ValueError("Taste PPO value-head checkpoint is not a mapping")
        value_parameter_identity = value_head_parameter_identity(value_head_state)
        for name, descriptor, expected in (
            ("adapter_config.json", config_fd, config_identity),
            ("adapter_model.safetensors", weights_fd, weights_identity),
            ("decoded_chem_value_head.pt", value_head_fd, value_head_identity),
        ):
            if (
                stat_identity(os.fstat(descriptor), directory=False) != expected
                or stat_identity(
                    os.stat(name, dir_fd=root_fd, follow_symlinks=False),
                    directory=False,
                )
                != expected
            ):
                raise ValueError("Taste PPO checkpoint leaf identity drifted")
        if stat_identity(os.fstat(root_fd), directory=True) != root_identity:
            raise ValueError("Taste PPO checkpoint directory identity drifted")
        if retained_parent_fd >= 0:
            if (
                stat_identity(os.fstat(retained_parent_fd), directory=True)
                != retained_parent_identity
                or stat_identity(
                    os.stat(
                        retained_child_name,
                        dir_fd=retained_parent_fd,
                        follow_symlinks=False,
                    ),
                    directory=True,
                )
                != root_identity
            ):
                raise ValueError("Taste PPO retained checkpoint child identity drifted")
        tensor_count = parameter_identity["tensor_count"]
        payload = {
            "schema_version": TASTE_PPO_CHECKPOINT_IDENTITY_SCHEMA,
            "adapter_config_name": "adapter_config.json",
            "adapter_config_sha256": config["sha256"],
            "adapter_config_size": config["size"],
            "adapter_weights_name": "adapter_model.safetensors",
            "adapter_weights_sha256": weights["sha256"],
            "adapter_weights_size": weights["size"],
            "value_head_name": "decoded_chem_value_head.pt",
            "value_head_sha256": value_head["sha256"],
            "value_head_parameter_sha256": value_parameter_identity[
                "parameter_sha256"
            ],
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return {
            "schema_version": "tastemolnet_ppo_checkpoint_reload_v1",
            "checkpoint_reload_pass": True,
            "checkpoint": str(display_root),
            "checkpoint_directory_identity": physical_evidence(
                root_identity, directory=True
            ),
            "tensor_count": tensor_count,
            "all_tensors_finite": True,
            "adapter_parameter_identity": parameter_identity,
            "adapter_parameter_sha256": parameter_identity["parameter_sha256"],
            "adapter_config": config,
            "adapter_weights": weights,
            "value_head": value_head,
            "value_head_parameter_identity": value_parameter_identity,
            "value_head_parameter_sha256": value_parameter_identity[
                "parameter_sha256"
            ],
            "policy_checkpoint_hash": hashlib.sha256(encoded).hexdigest(),
            "policy_checkpoint_hash_schema": TASTE_PPO_CHECKPOINT_IDENTITY_SCHEMA,
            "policy_checkpoint_hash_payload": payload,
        }
    finally:
        close_all()


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and not isinstance(value, bool) and math.isfinite(
        float(value)
    )


def _vector(value: Any, *, field: str) -> tuple[float, float, float]:
    if type(value) is not list or len(value) != 3 or not all(
        _finite(item) for item in value
    ):
        raise ValueError(f"Taste PPO {field} must be one finite three-class list")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _probabilities(value: Any, *, field: str) -> tuple[float, float, float]:
    result = _vector(value, field=field)
    if any(item < 0.0 or item > 1.0 for item in result) or not math.isclose(
        sum(result), 1.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError(f"Taste PPO {field} must be one probability distribution")
    return result


def _validate_scored_row(row: Mapping[str, Any], *, index: int) -> int:
    expected = {
        "schema_version": TASTE_GNN_PPO_REWARD_SCHEMA,
        "dataset": "tastemolnet",
        "num_classes": 3,
        "source_label": 1,
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "rf_oracle_used": False,
        "calibration_loaded": False,
        "calibration_dataset_loaded": False,
        "frozen_temperature_calibration_loaded": True,
        "test_loaded": False,
    }
    mismatches = [
        key
        for key, value in expected.items()
        if type(row.get(key)) is not type(value) or row.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"Taste PPO row {index} authority drift: {', '.join(mismatches)}"
        )
    for field in (
        "parse_ok",
        "connected",
        "direct_substructure",
        "projection_used",
        "deletion_valid",
        "gnn_scored_deletion",
        "cf_flip",
    ):
        if type(row.get(field)) is not bool:
            raise ValueError(f"Taste PPO row {index} {field} must be one native bool")
    for field in (
        "oracle_checkpoint_hash",
        "temperature_calibration_hash",
        "feature_schema_hash",
        "policy_initializer_hash",
        "reference_policy_hash",
    ):
        if not _is_sha256(row.get(field)):
            raise ValueError(f"Taste PPO row {index} lacks exact {field}")
    for field in (
        "reward_valid",
        "reward_substructure",
        "reward_cf",
        "reward_size",
        "reward_projection",
        "reward_kl",
        "reward_total",
    ):
        if not _finite(row.get(field)):
            raise ValueError(f"Taste PPO row {index} {field} is not finite")
    before = _probabilities(
        row.get("p_before_all_classes"), field="probabilities before"
    )
    _vector(row.get("logits_before_all_classes"), field="logits before")
    pred_before = row.get("pred_before")
    if (
        type(pred_before) is not int
        or pred_before not in (0, 1, 2)
        or max(range(3), key=before.__getitem__) != pred_before
    ):
        raise ValueError(f"Taste PPO row {index} parent prediction authority drifted")
    if not _finite(row.get("source_probability_before")) or not math.isclose(
        float(row["source_probability_before"]),
        before[1],
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise ValueError(f"Taste PPO row {index} source probability before drifted")
    if row.get("gnn_scored_deletion") is not True:
        exact_absent = {
            "pred_after": None,
            "destination_label": None,
            "p_after_all_classes": None,
            "logits_after_all_classes": None,
            "source_probability_after": None,
            "cf_drop": None,
            "margin_after": None,
            "margin_drop": None,
            "cf_flip": False,
        }
        if any(
            type(row.get(field)) is not type(value) or row.get(field) != value
            for field, value in exact_absent.items()
        ):
            raise ValueError(
                f"Taste PPO row {index} unscored deletion authority drifted"
            )
        return -1
    after = _probabilities(
        row.get("p_after_all_classes"), field="probabilities after"
    )
    _vector(row.get("logits_after_all_classes"), field="logits after")
    destination = row.get("destination_label")
    if (
        type(row.get("pred_after")) is not int
        or type(destination) is not int
        or row.get("pred_after") != destination
        or destination not in (0, 1, 2)
        or max(range(3), key=after.__getitem__) != destination
    ):
        raise ValueError(f"Taste PPO row {index} destination authority drifted")
    if not _finite(row.get("source_probability_after")) or not math.isclose(
        float(row["source_probability_after"]),
        after[1],
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise ValueError(f"Taste PPO row {index} source probability after drifted")
    expected_drop = before[1] - after[1]
    if not _finite(row.get("cf_drop")) or not math.isclose(
        float(row["cf_drop"]), expected_drop, rel_tol=0.0, abs_tol=1e-7
    ):
        raise ValueError(f"Taste PPO row {index} source-probability drop drifted")
    expected_margin_before = before[1] - max(before[0], before[2])
    expected_margin_after = after[1] - max(after[0], after[2])
    expected_margin_drop = expected_margin_before - expected_margin_after
    for field, expected_value in (
        ("margin_before", expected_margin_before),
        ("margin_after", expected_margin_after),
        ("margin_drop", expected_margin_drop),
    ):
        if not _finite(row.get(field)) or not math.isclose(
            float(row[field]), expected_value, rel_tol=0.0, abs_tol=1e-7
        ):
            raise ValueError(f"Taste PPO row {index} {field} drifted")
    expected_flip = pred_before == 1 and destination != 1
    if type(row.get("cf_flip")) is not bool or row.get("cf_flip") is not expected_flip:
        raise ValueError(f"Taste PPO row {index} strict-flip semantics drifted")
    return destination


def build_taste_reward_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    oracle_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Taste PPO reward manifest requires candidate rows")
    missing = sorted(
        {
            field
            for row in rows
            for field in TASTE_REWARD_FIELDS
            if field not in row
        }
    )
    if missing:
        raise ValueError("Taste PPO candidate rows lack: " + ", ".join(missing))
    exact_provenance = {
        "schema_version": "tastemolnet_gnn_ppo_oracle_provenance_v1",
        "dataset": "tastemolnet",
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "rf_oracle_used": False,
        "num_classes": 3,
        "source_label": 1,
        "calibration_loaded": False,
        "calibration_dataset_loaded": False,
        "frozen_temperature_calibration_loaded": True,
        "test_loaded": False,
        "oracle_load_count": 1,
    }
    mismatches = [
        key
        for key, value in exact_provenance.items()
        if type(oracle_provenance.get(key)) is not type(value)
        or oracle_provenance.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Taste PPO oracle provenance drift: " + ", ".join(mismatches)
        )
    for key in (
        "checkpoint_id",
        "temperature_calibration_hash",
        "feature_schema_hash",
        "policy_initializer_hash",
        "reference_policy_hash",
    ):
        if not _is_sha256(oracle_provenance.get(key)):
            raise ValueError(f"Taste PPO oracle provenance lacks {key}")
    if type(oracle_provenance.get("backbone")) is not str or str(
        oracle_provenance["backbone"]
    ).lower() != "gine":
        raise ValueError("Taste PPO oracle provenance is not the frozen GINE")
    if not _finite(oracle_provenance.get("temperature")) or float(
        oracle_provenance["temperature"]
    ) <= 0.0:
        raise ValueError("Taste PPO oracle provenance has invalid temperature")
    expected_row_hashes = {
        "oracle_checkpoint_hash": oracle_provenance["checkpoint_id"],
        "temperature_calibration_hash": oracle_provenance[
            "temperature_calibration_hash"
        ],
        "feature_schema_hash": oracle_provenance["feature_schema_hash"],
        "policy_initializer_hash": oracle_provenance["policy_initializer_hash"],
        "reference_policy_hash": oracle_provenance["reference_policy_hash"],
    }
    destinations: list[int] = []
    for index, row in enumerate(rows):
        destination = _validate_scored_row(row, index=index)
        if any(row.get(field) != expected for field, expected in expected_row_hashes.items()):
            raise ValueError(f"Taste PPO row {index} oracle/policy hash authority drifted")
        if destination >= 0:
            destinations.append(destination)
    reward_values = [row.get("reward_total") for row in rows]
    strict_destinations = [
        int(row["destination_label"])
        for row in rows
        if row.get("gnn_scored_deletion") is True and row.get("cf_flip") is True
    ]
    return {
        "schema_version": TASTE_PPO_REWARD_MANIFEST_SCHEMA,
        "dataset": "tastemolnet",
        "num_classes": 3,
        "source_label": 1,
        "candidate_count": len(rows),
        "required_fields": list(TASTE_REWARD_FIELDS),
        "missing_required_fields": [],
        "all_reward_values_finite": all(_finite(value) for value in reward_values),
        "valid_candidate_count": sum(
            1 for row in rows if row.get("parse_ok") is True and row.get("connected") is True
        ),
        "gnn_scored_deletion_count": len(destinations),
        "strict_flip_count": len(strict_destinations),
        "destination_0_count": strict_destinations.count(0),
        "destination_2_count": strict_destinations.count(2),
        "destination_labels": sorted(set(strict_destinations)),
        "oracle_checkpoint_hash": oracle_provenance["checkpoint_id"],
        "temperature_calibration_hash": oracle_provenance[
            "temperature_calibration_hash"
        ],
        "feature_schema_hash": oracle_provenance["feature_schema_hash"],
        "oracle_backend": "gnn",
        "rf_oracle_used": False,
        "calibration_loaded": False,
        "calibration_dataset_loaded": False,
        "frozen_temperature_calibration_loaded": True,
        "test_loaded": False,
        "candidate_oracle_contract_pass": True,
    }


class TastePPOObserver:
    """Collect real stable-loop callbacks without a BACE schema alias."""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.checkpoints: list[dict[str, Any]] = []
        self.finish: dict[str, Any] | None = None
        self._candidate_rows: list[dict[str, Any]] = []

    def captured_candidate_rows(self) -> list[dict[str, Any]]:
        return [
            json.loads(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
            for row in self._candidate_rows
        ]

    def on_update(
        self,
        *,
        step_index: int,
        batch_ids: Sequence[str],
        reward_logs: Sequence[Mapping[str, Any]],
        metrics: Mapping[str, Any],
    ) -> None:
        if type(step_index) is not int or step_index != len(self.updates) + 1:
            raise ValueError("Taste PPO update steps must be contiguous native integers")
        if (
            type(batch_ids) not in (list, tuple)
            or not batch_ids
            or not all(type(value) is str and value for value in batch_ids)
        ):
            raise ValueError("Taste PPO update batch IDs are malformed")
        if (
            type(reward_logs) not in (list, tuple)
            or len(reward_logs) != len(batch_ids)
            or not all(isinstance(row, Mapping) for row in reward_logs)
        ):
            raise ValueError("Taste PPO update reward rows differ from the rollout batch")
        if not isinstance(metrics, Mapping) or any(
            key not in metrics for key in TASTE_PPO_REQUIRED_UPDATE_METRICS
        ):
            raise ValueError("Taste PPO update metrics are incomplete")
        if (
            type(metrics.get("global_step")) is not int
            or metrics.get("global_step") != step_index
            or type(metrics.get("rollout_batch_size")) is not int
            or metrics.get("rollout_batch_size") != len(batch_ids)
            or any(
                not _finite(metrics.get(key))
                for key in TASTE_PPO_REQUIRED_UPDATE_METRICS
                if key not in {"global_step", "rollout_batch_size"}
            )
        ):
            raise ValueError("Taste PPO update metric values are malformed")
        captured: list[dict[str, Any]] = []
        for raw in reward_logs:
            try:
                normalized = json.loads(
                    json.dumps(
                        dict(raw),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("Taste PPO reward-log row is not native JSON") from exc
            if type(normalized) is not dict or normalized.get("step_index") != step_index:
                raise ValueError("Taste PPO reward-log row lacks its final step identity")
            captured.append(normalized)
        self._candidate_rows.extend(captured)
        self.updates.append(
            {
                "step_index": step_index,
                "batch_ids": list(batch_ids),
                "reward_row_count": len(reward_logs),
                "metrics": dict(metrics),
            }
        )

    def on_checkpoint(
        self,
        *,
        step_index: int,
        checkpoint_dir: Path,
        checkpoint_kind: str,
    ) -> None:
        if (
            type(step_index) is not int
            or step_index <= 0
            or checkpoint_kind != "periodic"
            or not isinstance(checkpoint_dir, Path)
            or not checkpoint_dir.is_absolute()
            or checkpoint_dir.name != f"checkpoint-{step_index}"
        ):
            raise ValueError("Taste PPO checkpoint callback authority is malformed")
        self.checkpoints.append(
            {
                "step_index": step_index,
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint_kind": str(checkpoint_kind),
            }
        )

    def on_finish(self, **kwargs: Any) -> None:
        def native_json(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if is_dataclass(value) and not isinstance(value, type):
                return native_json(asdict(value))
            if isinstance(value, Mapping):
                if not all(type(key) is str for key in value):
                    raise ValueError("Taste PPO finish state has a non-string key")
                return {key: native_json(item) for key, item in value.items()}
            if type(value) in (list, tuple):
                return [native_json(item) for item in value]
            if value is None or type(value) in (bool, int, str):
                return value
            if type(value) is float and math.isfinite(value):
                return value
            raise ValueError("Taste PPO finish state is not native finite JSON")

        normalized = native_json(kwargs)
        if type(normalized) is not dict:
            raise ValueError("Taste PPO finish state is not one JSON object")
        self.finish = normalized

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TASTE_PPO_OBSERVER_SCHEMA,
            "updates": [dict(row) for row in self.updates],
            "checkpoints": [dict(row) for row in self.checkpoints],
            "finish": dict(self.finish) if self.finish is not None else None,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if set(payload) != {"schema_version", "updates", "checkpoints", "finish"}:
            raise ValueError("Taste PPO observer state keys changed")
        if payload.get("schema_version") != TASTE_PPO_OBSERVER_SCHEMA:
            raise ValueError("Taste PPO observer schema changed")
        updates = payload.get("updates")
        checkpoints = payload.get("checkpoints")
        finish = payload.get("finish")
        if type(updates) is not list or not all(type(row) is dict for row in updates):
            raise ValueError("Taste PPO observer updates are malformed")
        if type(checkpoints) is not list or not all(
            type(row) is dict for row in checkpoints
        ):
            raise ValueError("Taste PPO observer checkpoints are malformed")
        steps = [row.get("step_index") for row in updates]
        if any(type(step) is not int for step in steps) or steps != list(
            range(1, len(steps) + 1)
        ):
            raise ValueError("Taste PPO observer updates are not contiguous")
        for index, row in enumerate(updates, start=1):
            batch_ids = row.get("batch_ids")
            metrics = row.get("metrics")
            reward_row_count = row.get("reward_row_count")
            if (
                set(row) != {"step_index", "batch_ids", "reward_row_count", "metrics"}
                or type(batch_ids) is not list
                or not batch_ids
                or not all(type(value) is str and value for value in batch_ids)
                or type(reward_row_count) is not int
                or reward_row_count != len(batch_ids)
                or type(metrics) is not dict
                or any(key not in metrics for key in TASTE_PPO_REQUIRED_UPDATE_METRICS)
                or type(metrics.get("global_step")) is not int
                or metrics.get("global_step") != index
                or type(metrics.get("rollout_batch_size")) is not int
                or metrics.get("rollout_batch_size") != len(batch_ids)
                or any(
                    not _finite(metrics.get(key))
                    for key in TASTE_PPO_REQUIRED_UPDATE_METRICS
                    if key not in {"global_step", "rollout_batch_size"}
                )
            ):
                raise ValueError("Taste PPO observer update state is malformed")
        if finish is not None and type(finish) is not dict:
            raise ValueError("Taste PPO observer finish state is malformed")
        self.updates = [dict(row) for row in updates]
        self.checkpoints = [dict(row) for row in checkpoints]
        self.finish = None if finish is None else dict(finish)


def build_taste_smoke_gate(
    *,
    policy_parameter_hash_before: str,
    policy_parameter_hash_after: str,
    reference_parameter_hash_before: str,
    reference_parameter_hash_after: str,
    policy_t5_identity_before: str,
    reference_t5_identity_before: str,
    reference_t5_identity_after: str,
    expected_t5_reference_policy_hash: str,
    observer: TastePPOObserver,
    checkpoint_reload: Mapping[str, Any],
    periodic_checkpoint_reload: Mapping[str, Any],
    reward_manifest: Mapping[str, Any],
    oracle_provenance: Mapping[str, Any],
    policy_initializer_hash: str,
    t3_gate_sha256: str,
    t4_gate_sha256: str,
    value_head_parameter_sha256: str,
    hard_kl: float = 0.8,
) -> dict[str, Any]:
    for name, digest in {
        "policy_parameter_hash_before": policy_parameter_hash_before,
        "policy_parameter_hash_after": policy_parameter_hash_after,
        "reference_parameter_hash_before": reference_parameter_hash_before,
        "reference_parameter_hash_after": reference_parameter_hash_after,
        "policy_t5_identity_before": policy_t5_identity_before,
        "reference_t5_identity_before": reference_t5_identity_before,
        "reference_t5_identity_after": reference_t5_identity_after,
        "expected_t5_reference_policy_hash": expected_t5_reference_policy_hash,
        "policy_initializer_hash": policy_initializer_hash,
        "t3_gate_sha256": t3_gate_sha256,
        "t4_gate_sha256": t4_gate_sha256,
        "value_head_parameter_sha256": value_head_parameter_sha256,
    }.items():
        if not _is_sha256(digest):
            raise ValueError(f"Taste PPO smoke requires exact {name}")
    updates = list(observer.updates)
    metrics_finite = bool(updates) and all(
        type(update.get("metrics")) is dict
        and all(key in update["metrics"] for key in TASTE_PPO_REQUIRED_UPDATE_METRICS)
        and all(
            _finite(update["metrics"].get(key))
            for key in TASTE_PPO_REQUIRED_UPDATE_METRICS
            if key not in {"global_step", "rollout_batch_size"}
        )
        for update in updates
    )
    kl_safe = bool(updates) and all(
        _finite(update.get("metrics", {}).get("approx_kl"))
        and float(update["metrics"]["approx_kl"]) <= hard_kl
        for update in updates
    )
    strict_flip_count = reward_manifest.get("strict_flip_count")
    destinations = reward_manifest.get("destination_labels")
    failures: list[str] = []
    checks = {
        "minimum_update_count_met": len(updates) >= 5,
        "bounded_update_count_met": len(updates) <= 10,
        "policy_parameters_changed": policy_parameter_hash_before
        != policy_parameter_hash_after,
        "reference_parameters_unchanged": reference_parameter_hash_before
        == reference_parameter_hash_after,
        "policy_initialized_from_exact_t5": policy_t5_identity_before
        == expected_t5_reference_policy_hash,
        "reference_remains_exact_t5": reference_t5_identity_before
        == expected_t5_reference_policy_hash
        and reference_t5_identity_after == expected_t5_reference_policy_hash,
        "checkpoint_saved": observer.finish is not None
        and bool(observer.checkpoints)
        and any(
            type(row.get("step_index")) is int
            and row.get("step_index") == len(updates)
            and row.get("checkpoint_kind") == "periodic"
            for row in observer.checkpoints
        ),
        "checkpoint_reload_pass": checkpoint_reload.get("checkpoint_reload_pass")
        is True,
        "periodic_checkpoint_reload_pass": periodic_checkpoint_reload.get(
            "checkpoint_reload_pass"
        )
        is True,
        "checkpoint_artifact_bound": checkpoint_reload.get(
            "policy_checkpoint_hash_schema"
        )
        == TASTE_PPO_CHECKPOINT_IDENTITY_SCHEMA
        and _is_sha256(checkpoint_reload.get("policy_checkpoint_hash")),
        "periodic_checkpoint_artifact_bound": periodic_checkpoint_reload.get(
            "policy_checkpoint_hash_schema"
        )
        == TASTE_PPO_CHECKPOINT_IDENTITY_SCHEMA
        and _is_sha256(periodic_checkpoint_reload.get("policy_checkpoint_hash")),
        "final_matches_last_periodic_checkpoint": checkpoint_reload.get(
            "policy_checkpoint_hash"
        )
        == periodic_checkpoint_reload.get("policy_checkpoint_hash"),
        "final_checkpoint_matches_trained_policy": checkpoint_reload.get(
            "adapter_parameter_sha256"
        )
        == policy_parameter_hash_after,
        "periodic_checkpoint_matches_trained_policy": periodic_checkpoint_reload.get(
            "adapter_parameter_sha256"
        )
        == policy_parameter_hash_after,
        "final_periodic_adapter_parameters_match": checkpoint_reload.get(
            "adapter_parameter_sha256"
        )
        == periodic_checkpoint_reload.get("adapter_parameter_sha256"),
        "final_value_head_matches_trained_policy": checkpoint_reload.get(
            "value_head_parameter_sha256"
        )
        == value_head_parameter_sha256,
        "periodic_value_head_matches_trained_policy": periodic_checkpoint_reload.get(
            "value_head_parameter_sha256"
        )
        == value_head_parameter_sha256,
        "final_periodic_value_heads_match": checkpoint_reload.get(
            "value_head_parameter_sha256"
        )
        == periodic_checkpoint_reload.get("value_head_parameter_sha256"),
        "all_update_metrics_finite": metrics_finite,
        "kl_within_hard_limit": kl_safe,
        "candidate_pool_nonempty": type(reward_manifest.get("candidate_count"))
        is int
        and reward_manifest["candidate_count"] > 0,
        "all_reward_values_finite": reward_manifest.get(
            "all_reward_values_finite"
        )
        is True,
        "candidate_oracle_contract_pass": reward_manifest.get(
            "candidate_oracle_contract_pass"
        )
        is True,
        "strict_flip_observed": type(strict_flip_count) is int
        and strict_flip_count >= 1,
        "destination_contract_pass": type(destinations) is list
        and bool(destinations)
        and set(destinations).issubset({0, 2}),
        "three_class_oracle": oracle_provenance.get("dataset") == "tastemolnet"
        and type(oracle_provenance.get("num_classes")) is int
        and oracle_provenance.get("num_classes") == 3
        and type(oracle_provenance.get("source_label")) is int
        and oracle_provenance.get("source_label") == 1,
        "same_oracle_manifest": reward_manifest.get("oracle_checkpoint_hash")
        == oracle_provenance.get("checkpoint_id"),
        "same_policy_initializer": policy_initializer_hash
        == oracle_provenance.get("policy_initializer_hash"),
        "same_reference_policy": expected_t5_reference_policy_hash
        == oracle_provenance.get("reference_policy_hash"),
        "rf_guard_pass": oracle_provenance.get("rf_oracle_used") is False,
        "calibration_dataset_not_loaded": oracle_provenance.get(
            "calibration_dataset_loaded"
        )
        is False,
        "test_not_loaded": oracle_provenance.get("test_loaded") is False,
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    return {
        "schema_version": TASTE_PPO_GATE_SCHEMA,
        "stage": TASTE_PPO_STAGE,
        "status": "PASS" if not failures else "FAIL",
        "marker": TASTE_PPO_MARKER,
        "failures": failures,
        "dataset": "tastemolnet",
        "num_classes": 3,
        "source_label": 1,
        "strict_flip": "pred_before == 1 and pred_after != 1",
        "optimizer_step_count": len(updates),
        "policy_parameter_hash_before": policy_parameter_hash_before,
        "policy_parameter_hash_after": policy_parameter_hash_after,
        "reference_parameter_hash_before": reference_parameter_hash_before,
        "reference_parameter_hash_after": reference_parameter_hash_after,
        "policy_t5_identity_before": policy_t5_identity_before,
        "reference_t5_identity_before": reference_t5_identity_before,
        "reference_t5_identity_after": reference_t5_identity_after,
        "expected_t5_reference_policy_hash": expected_t5_reference_policy_hash,
        "policy_initializer_hash": policy_initializer_hash,
        "oracle_checkpoint_hash": oracle_provenance.get("checkpoint_id"),
        "policy_checkpoint_hash": checkpoint_reload.get("policy_checkpoint_hash"),
        "periodic_policy_checkpoint_hash": periodic_checkpoint_reload.get(
            "policy_checkpoint_hash"
        ),
        "policy_adapter_parameter_sha256": policy_parameter_hash_after,
        "final_adapter_parameter_sha256": checkpoint_reload.get(
            "adapter_parameter_sha256"
        ),
        "periodic_adapter_parameter_sha256": periodic_checkpoint_reload.get(
            "adapter_parameter_sha256"
        ),
        "value_head_parameter_sha256": value_head_parameter_sha256,
        "final_value_head_parameter_sha256": checkpoint_reload.get(
            "value_head_parameter_sha256"
        ),
        "periodic_value_head_parameter_sha256": periodic_checkpoint_reload.get(
            "value_head_parameter_sha256"
        ),
        "t3_gate_sha256": t3_gate_sha256,
        "t4_gate_sha256": t4_gate_sha256,
        "strict_flip_count": strict_flip_count,
        "destination_labels": destinations,
        "rf_oracle_used": False,
        "calibration_dataset_loaded": False,
        "test_loaded": False,
        **checks,
    }


def _exact_json_document(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Taste PPO {label} is malformed") from exc
    if type(payload) is not dict:
        raise ValueError(f"Taste PPO {label} is not one JSON object")
    expected = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if data != expected:
        raise ValueError(f"Taste PPO {label} is not canonical JSON")
    return payload


class HeldTastePPOOutput:
    """Descriptor-held, semantically validated terminal Taste T6 output."""

    def __init__(self, root: str | Path) -> None:
        self._authority = HeldPublishedTerminalOutput.open(
            root,
            marker_name="PASS",
            marker_payload=f"[{TASTE_PPO_MARKER}]\n".encode("utf-8"),
        )
        try:
            self._evidence = self._validate()
        except Exception:
            self._authority.close()
            raise

    def _json(self, name: str) -> dict[str, Any]:
        return _exact_json_document(
            self._authority.read_bytes(name),
            label=name,
        )

    def _validate(self) -> dict[str, Any]:
        inventory = self._authority.revalidate()
        gate_bytes = self._authority.read_bytes("gate.json")
        if gate_bytes != self._authority.read_bytes("ppo_gate.json"):
            raise ValueError("Taste PPO duplicate gate documents differ")
        manifest_bytes = self._authority.read_bytes("manifest.json")
        if manifest_bytes != self._authority.read_bytes("ppo_smoke_manifest.json"):
            raise ValueError("Taste PPO duplicate manifest documents differ")
        gate = _exact_json_document(gate_bytes, label="gate.json")
        manifest = _exact_json_document(manifest_bytes, label="manifest.json")
        state = self._json("state.json")
        reward = self._json("reward_manifest.json")
        oracle = self._json("oracle_provenance.json")
        run_manifest = self._json("run_manifest.json")
        observer_state = self._json("observer_state.json")
        if (
            gate.get("schema_version") != TASTE_PPO_GATE_SCHEMA
            or gate.get("stage") != TASTE_PPO_STAGE
            or gate.get("status") != "PASS"
            or gate.get("marker") != TASTE_PPO_MARKER
            or gate.get("failures") != []
        ):
            raise ValueError("Taste PPO terminal gate is not exact PASS")
        for key, value in gate.items():
            if key == "schema_version":
                continue
            if type(manifest.get(key)) is not type(value) or manifest.get(key) != value:
                raise ValueError("Taste PPO manifest differs from its gate")
        if (
            manifest.get("schema_version") != "tastemolnet_ours_ppo_smoke_manifest_v1"
            or manifest.get("output_root") != str(self._authority.path)
            or manifest.get("no_dataset_redistribution") is not True
            or state.get("stage") != TASTE_PPO_STAGE
            or state.get("state") != "PASS"
            or state.get("status") != "PASS"
            or state.get("output_root") != str(self._authority.path)
            or run_manifest.get("schema_version")
            != "tastemolnet_ours_ppo_run_v1"
            or run_manifest.get("stage") != TASTE_PPO_STAGE
            or type(run_manifest.get("num_classes")) is not int
            or run_manifest.get("num_classes") != 3
            or type(run_manifest.get("source_label")) is not int
            or run_manifest.get("source_label") != 1
            or run_manifest.get("rf_oracle_used") is not False
            or run_manifest.get("validation_loaded") is not False
            or run_manifest.get("calibration_loaded") is not False
            or run_manifest.get("test_loaded") is not False
            or run_manifest.get("tokenizer_checkpoint_saved") is not False
            or type(run_manifest.get("model_path")) is not str
        ):
            raise ValueError("Taste PPO terminal manifest/state authority drifted")
        model_path = Path(run_manifest["model_path"])
        expected_adapter_config = run_manifest.get("adapter_config_authority")
        if not model_path.is_absolute() or "/proc/self/fd/" in str(model_path):
            raise ValueError("Taste PPO terminal source-model identity is invalid")
        if type(expected_adapter_config) is not dict:
            raise ValueError("Taste PPO terminal lacks adapter config authority")
        candidate_bytes = self._authority.read_bytes("candidate_pool.jsonl")
        rows: list[dict[str, Any]] = []
        for line in candidate_bytes.splitlines():
            if not line:
                continue
            row = json.loads(line.decode("utf-8"))
            if type(row) is not dict:
                raise ValueError("Taste PPO candidate pool row is malformed")
            rows.append(row)
        expected_candidate_bytes = "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
        ).encode("utf-8")
        if candidate_bytes != expected_candidate_bytes:
            raise ValueError("Taste PPO candidate pool is not exact canonical JSONL")
        if build_taste_reward_manifest(rows, oracle_provenance=oracle) != reward:
            raise ValueError("Taste PPO reward manifest does not match candidates")
        if (
            hashlib.sha256(candidate_bytes).hexdigest()
            != manifest.get("candidate_pool_sha256")
            or hashlib.sha256(self._authority.read_bytes("reward_manifest.json")).hexdigest()
            != manifest.get("reward_manifest_sha256")
            or hashlib.sha256(self._authority.read_bytes("oracle_provenance.json")).hexdigest()
            != manifest.get("oracle_provenance_sha256")
        ):
            raise ValueError("Taste PPO terminal artifact hashes drifted")
        updates = gate.get("optimizer_step_count")
        if type(updates) is not int or not 5 <= updates <= 10:
            raise ValueError("Taste PPO terminal update count is invalid")
        periodic_root = f"checkpoint-{updates}"
        expected_files = {
            "README.md",
            "adapter_config.json",
            "adapter_model.safetensors",
            "decoded_chem_value_head.pt",
            f"{periodic_root}/README.md",
            f"{periodic_root}/adapter_config.json",
            f"{periodic_root}/adapter_model.safetensors",
            f"{periodic_root}/decoded_chem_value_head.pt",
            "candidate_pool.jsonl",
            "policy_provenance.json",
            "downstream_policy_binding.json",
            "parent_selection.json",
            "run_manifest.json",
            "observer_state.json",
            "oracle_provenance.json",
            "reward_manifest.json",
            "gate.json",
            "ppo_gate.json",
            "input_hashes.json",
            "manifest.json",
            "ppo_smoke_manifest.json",
            "state.json",
            f"logs/{TASTE_PPO_STAGE}.log",
        }
        if set(inventory["files"]) != expected_files:
            raise ValueError("Taste PPO terminal file layout changed")
        if set(inventory["directories"]) != {"logs", periodic_root}:
            raise ValueError("Taste PPO terminal directory layout changed")
        retained_checkpoint_path = sys.platform.startswith("linux")
        checkpoint_root = (
            self._authority.stable_path
            if retained_checkpoint_path
            else self._authority.path
        )
        final_reload = validate_taste_adapter_checkpoint_reload(
            checkpoint_root,
            checkpoint_path_is_retained=retained_checkpoint_path,
            checkpoint_display_path=self._authority.path,
            expected_base_model_path=model_path,
            expected_adapter_config=expected_adapter_config,
        )
        periodic_reload = validate_taste_adapter_checkpoint_reload(
            checkpoint_root / f"checkpoint-{updates}",
            checkpoint_path_is_retained=retained_checkpoint_path,
            checkpoint_display_path=self._authority.path / f"checkpoint-{updates}",
            expected_base_model_path=model_path,
            expected_adapter_config=expected_adapter_config,
        )
        if (
            final_reload.get("policy_checkpoint_hash")
            != gate.get("policy_checkpoint_hash")
            or periodic_reload.get("policy_checkpoint_hash")
            != gate.get("periodic_policy_checkpoint_hash")
            or final_reload.get("adapter_parameter_sha256")
            != gate.get("final_adapter_parameter_sha256")
            or periodic_reload.get("adapter_parameter_sha256")
            != gate.get("periodic_adapter_parameter_sha256")
            or final_reload.get("value_head_parameter_sha256")
            != gate.get("final_value_head_parameter_sha256")
            or periodic_reload.get("value_head_parameter_sha256")
            != gate.get("periodic_value_head_parameter_sha256")
        ):
            raise ValueError("Taste PPO terminal checkpoint evidence drifted")
        observer = TastePPOObserver()
        observer.load_state_dict(observer_state)
        finish = observer.finish
        if (
            len(observer.updates) != updates
            or observer.checkpoints
            != [
                {
                    "step_index": updates,
                    "checkpoint_dir": str(
                        self._authority.path / f"checkpoint-{updates}"
                    ),
                    "checkpoint_kind": "periodic",
                }
            ]
            or type(finish) is not dict
            or set(finish)
            != {
                "final_output_dir",
                "candidate_pool_path",
                "candidate_count",
                "global_step",
                "validation_state",
                "early_stop_reason",
            }
            or finish.get("final_output_dir") != str(self._authority.path)
            or finish.get("candidate_pool_path")
            != str(self._authority.path / "candidate_pool.jsonl")
            or type(finish.get("candidate_count")) is not int
            or finish.get("candidate_count") != len(rows)
            or type(finish.get("global_step")) is not int
            or finish.get("global_step") != updates
            or finish.get("validation_state")
            != {
                "best_val_score": None,
                "best_step": None,
                "stale_eval_count": 0,
            }
            or finish.get("early_stop_reason") is not None
        ):
            raise ValueError("Taste PPO observer terminal state drifted")
        if sum(
            int(update["reward_row_count"]) for update in observer.updates
        ) != len(rows):
            raise ValueError("Taste PPO observer/candidate row count drifted")
        for step_index, update in enumerate(observer.updates, start=1):
            if sum(
                1 for row in rows if row.get("step_index") == step_index
            ) != update["reward_row_count"]:
                raise ValueError("Taste PPO observer/candidate step binding drifted")
        reconstructed_gate = build_taste_smoke_gate(
            policy_parameter_hash_before=gate.get(
                "policy_parameter_hash_before"
            ),
            policy_parameter_hash_after=gate.get("policy_parameter_hash_after"),
            reference_parameter_hash_before=gate.get(
                "reference_parameter_hash_before"
            ),
            reference_parameter_hash_after=gate.get(
                "reference_parameter_hash_after"
            ),
            policy_t5_identity_before=gate.get("policy_t5_identity_before"),
            reference_t5_identity_before=gate.get(
                "reference_t5_identity_before"
            ),
            reference_t5_identity_after=gate.get("reference_t5_identity_after"),
            expected_t5_reference_policy_hash=gate.get(
                "expected_t5_reference_policy_hash"
            ),
            observer=observer,
            checkpoint_reload=final_reload,
            periodic_checkpoint_reload=periodic_reload,
            reward_manifest=reward,
            oracle_provenance=oracle,
            policy_initializer_hash=gate.get("policy_initializer_hash"),
            t3_gate_sha256=gate.get("t3_gate_sha256"),
            t4_gate_sha256=gate.get("t4_gate_sha256"),
            value_head_parameter_sha256=gate.get(
                "value_head_parameter_sha256"
            ),
        )
        if reconstructed_gate != gate:
            raise ValueError("Taste PPO terminal gate cannot be independently derived")
        self._authority.tree.reject_byte_sequence(
            b"/proc/self/fd/",
            suffixes=("",),
        )
        self._authority.revalidate()
        return {
            "schema_version": TASTE_PPO_OUTPUT_EVIDENCE_SCHEMA,
            "status": "PASS",
            "stage": TASTE_PPO_STAGE,
            "output_root": str(self._authority.path),
            "output_inventory_sha256": inventory["inventory_sha256"],
            "gate_sha256": hashlib.sha256(gate_bytes).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "policy_checkpoint_hash": final_reload["policy_checkpoint_hash"],
            "periodic_policy_checkpoint_hash": periodic_reload[
                "policy_checkpoint_hash"
            ],
            "policy_adapter_parameter_sha256": gate[
                "policy_adapter_parameter_sha256"
            ],
            "optimizer_step_count": updates,
            "strict_flip_count": gate["strict_flip_count"],
            "destination_labels": gate["destination_labels"],
        }

    def revalidate(self) -> dict[str, Any]:
        observed = self._validate()
        if observed != self._evidence:
            raise ValueError("Taste PPO terminal evidence drifted")
        return dict(observed)

    def close(self) -> None:
        self._authority.close()

    def __enter__(self) -> "HeldTastePPOOutput":
        self.revalidate()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()


def hold_taste_ppo_output(root: str | Path) -> HeldTastePPOOutput:
    return HeldTastePPOOutput(root)


def validate_taste_ppo_output(root: str | Path) -> dict[str, Any]:
    with hold_taste_ppo_output(root) as held:
        return held.revalidate()


__all__ = [
    "TASTE_PPO_ADAPTER_PARAMETER_SCHEMA",
    "TASTE_PPO_GATE_SCHEMA",
    "TASTE_PPO_CHECKPOINT_IDENTITY_SCHEMA",
    "TASTE_PPO_MARKER",
    "TASTE_PPO_OBSERVER_SCHEMA",
    "TASTE_PPO_OUTPUT_EVIDENCE_SCHEMA",
    "TASTE_PPO_REQUIRED_UPDATE_METRICS",
    "TASTE_PPO_REWARD_MANIFEST_SCHEMA",
    "TASTE_PPO_STAGE",
    "TASTE_REWARD_FIELDS",
    "TastePPOObserver",
    "HeldTastePPOOutput",
    "adapter_parameter_identity",
    "adapter_parameter_identity_from_model",
    "build_taste_reward_manifest",
    "build_taste_smoke_gate",
    "hold_taste_ppo_output",
    "validate_taste_ppo_output",
    "validate_taste_adapter_checkpoint_reload",
    "value_head_parameter_identity",
]
