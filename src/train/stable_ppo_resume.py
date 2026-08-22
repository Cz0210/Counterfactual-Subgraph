"""Fail-closed checkpoint/resume artifacts for the shared stable PPO loop."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "stable_decoded_chem_ppo_resume_v1"
STATE_NAME = "stable_ppo_training_state.pt"
MANIFEST_NAME = "stable_ppo_resume_manifest.json"
CANDIDATE_POOL_NAME = "candidate_pool.jsonl"
VALUE_HEAD_NAME = "decoded_chem_value_head.pt"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _physical_file(path: Path, *, label: str) -> Path:
    unresolved = path.expanduser()
    if unresolved.is_symlink() or not unresolved.is_file():
        raise ValueError(f"{label} must be one physical file: {unresolved}")
    resolved = unresolved.resolve(strict=True)
    if resolved.stat().st_size <= 0:
        raise ValueError(f"{label} is empty: {resolved}")
    return resolved


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                dict(payload),
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_torch_save(torch_module: Any, payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        torch_module.save(dict(payload), temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _adapter_files(root: Path) -> tuple[Path, Path]:
    config = _physical_file(root / "adapter_config.json", label="adapter config")
    weights = [
        path
        for path in (
            root / "adapter_model.safetensors",
            root / "adapter_model.bin",
        )
        if path.is_file() and not path.is_symlink()
    ]
    if len(weights) != 1:
        raise ValueError(
            f"Resume checkpoint requires exactly one physical adapter weight file: {root}"
        )
    return config, _physical_file(weights[0], label="adapter weights")


def _file_identity(path: Path) -> dict[str, Any]:
    physical = _physical_file(path, label=path.name)
    return {
        "name": physical.name,
        "sha256": _sha256_file(physical),
        "size": physical.stat().st_size,
    }


def save_stable_ppo_resume_checkpoint(
    *,
    checkpoint_dir: Path,
    torch_module: Any,
    optimizer: Any,
    completed_steps: int,
    current_kl_penalty: float,
    validation_state: Mapping[str, Any],
    last_validation_step: int | None,
    candidate_pool_rows: Sequence[Mapping[str, Any]],
    observer_state: Mapping[str, Any],
    resume_contract: Mapping[str, Any],
) -> Path:
    """Write optimizer/control/RNG state and publish the JSON manifest last."""

    unresolved = checkpoint_dir.expanduser()
    if unresolved.is_symlink():
        raise ValueError("Stable PPO resume checkpoint may not be a symlink")
    root = unresolved.resolve(strict=True)
    if completed_steps <= 0:
        raise ValueError("Stable PPO resume checkpoint root/step is invalid")
    adapter_config, adapter_weights = _adapter_files(root)
    value_head = _physical_file(root / VALUE_HEAD_NAME, label="value head")
    if not candidate_pool_rows:
        raise ValueError("Stable PPO resume checkpoint requires candidate history")
    candidate_path = root / CANDIDATE_POOL_NAME
    _atomic_jsonl(candidate_path, candidate_pool_rows)
    state_path = root / STATE_NAME
    state = {
        "schema_version": SCHEMA_VERSION,
        "completed_steps": int(completed_steps),
        "current_kl_penalty": float(current_kl_penalty),
        "validation_state": dict(validation_state),
        "last_validation_step": last_validation_step,
        "candidate_pool_count": len(candidate_pool_rows),
        "observer_state": dict(observer_state),
        "resume_contract": dict(resume_contract),
        "optimizer_state_dict": optimizer.state_dict(),
        "python_random_state": random.getstate(),
        "torch_cpu_rng_state": torch_module.get_rng_state(),
        "torch_cuda_rng_states": (
            torch_module.cuda.get_rng_state_all()
            if torch_module.cuda.is_available()
            else []
        ),
        "torch_cuda_device_count": (
            torch_module.cuda.device_count()
            if torch_module.cuda.is_available()
            else 0
        ),
    }
    _atomic_torch_save(torch_module, state, state_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "READY",
        "resume_checkpoint_complete": True,
        "checkpoint_dir": str(root),
        "completed_steps": int(completed_steps),
        "candidate_pool_count": len(candidate_pool_rows),
        "resume_contract": dict(resume_contract),
        "resume_contract_sha256": _canonical_hash(resume_contract),
        "artifacts": {
            "adapter_config": _file_identity(adapter_config),
            "adapter_weights": _file_identity(adapter_weights),
            "value_head": _file_identity(value_head),
            "candidate_pool": _file_identity(candidate_path),
            "training_state": _file_identity(state_path),
        },
    }
    manifest_path = root / MANIFEST_NAME
    _atomic_json(manifest_path, manifest)
    return manifest_path


def read_stable_ppo_resume_manifest(checkpoint_dir: Path) -> dict[str, Any]:
    unresolved = checkpoint_dir.expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise ValueError(f"Resume checkpoint must be one physical directory: {unresolved}")
    root = unresolved.resolve(strict=True)
    manifest_path = _physical_file(root / MANIFEST_NAME, label="resume manifest")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Stable PPO resume manifest is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Stable PPO resume manifest must contain one object")
    required = {
        "schema_version": SCHEMA_VERSION,
        "status": "READY",
        "resume_checkpoint_complete": True,
        "checkpoint_dir": str(root),
    }
    mismatches = [
        key for key, expected in required.items() if payload.get(key) != expected
    ]
    contract = payload.get("resume_contract")
    if not isinstance(contract, dict) or payload.get("resume_contract_sha256") != _canonical_hash(
        contract
    ):
        mismatches.append("resume_contract")
    completed_steps = payload.get("completed_steps")
    if (
        isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or completed_steps <= 0
    ):
        mismatches.append("completed_steps")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        mismatches.append("artifacts")
        artifacts = {}
    expected_paths = {
        "adapter_config": root / "adapter_config.json",
        "adapter_weights": (
            root / str((artifacts.get("adapter_weights") or {}).get("name", ""))
        ),
        "value_head": root / VALUE_HEAD_NAME,
        "candidate_pool": root / CANDIDATE_POOL_NAME,
        "training_state": root / STATE_NAME,
    }
    for key, artifact_path in expected_paths.items():
        identity = artifacts.get(key)
        try:
            actual = _file_identity(artifact_path)
        except ValueError:
            mismatches.append(key)
            continue
        if identity != actual:
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            "Stable PPO resume checkpoint failed closed: "
            + ", ".join(sorted(set(mismatches)))
        )
    return payload


@dataclass(frozen=True)
class StablePPOResumeBundle:
    checkpoint_dir: Path
    completed_steps: int
    current_kl_penalty: float
    validation_state: dict[str, Any]
    last_validation_step: int | None
    candidate_pool_rows: list[dict[str, Any]]
    observer_state: dict[str, Any]
    state: dict[str, Any]


def load_stable_ppo_resume_checkpoint(
    *,
    checkpoint_dir: Path,
    torch_module: Any,
    expected_contract: Mapping[str, Any],
    map_location: Any,
) -> StablePPOResumeBundle:
    manifest = read_stable_ppo_resume_manifest(checkpoint_dir)
    if manifest["resume_contract"] != dict(expected_contract):
        raise ValueError("Stable PPO resume contract differs from the current run")
    root = Path(str(manifest["checkpoint_dir"]))
    state_path = root / STATE_NAME
    try:
        state = torch_module.load(
            state_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError as exc:  # pragma: no cover - modern AutoDL torch supports it
        raise RuntimeError("Safe stable PPO resume needs torch.load(weights_only=True)") from exc
    if not isinstance(state, dict):
        raise ValueError("Stable PPO resume state must be one mapping")
    required_state = {
        "schema_version": SCHEMA_VERSION,
        "completed_steps": manifest["completed_steps"],
        "candidate_pool_count": manifest["candidate_pool_count"],
        "resume_contract": manifest["resume_contract"],
    }
    mismatch = [
        key for key, expected in required_state.items() if state.get(key) != expected
    ]
    if mismatch:
        raise ValueError("Stable PPO resume state/manifest drift: " + ", ".join(mismatch))
    rows: list[dict[str, Any]] = []
    with (root / CANDIDATE_POOL_NAME).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Stable PPO resume candidate row is not an object")
                rows.append(row)
    if len(rows) != int(manifest["candidate_pool_count"]):
        raise ValueError("Stable PPO resume candidate count changed")
    validation_state = state.get("validation_state")
    observer_state = state.get("observer_state")
    if (
        not isinstance(validation_state, dict)
        or not isinstance(observer_state, dict)
        or "optimizer_state_dict" not in state
        or "python_random_state" not in state
        or "torch_cpu_rng_state" not in state
    ):
        raise ValueError("Stable PPO resume observer/validation state is malformed")
    return StablePPOResumeBundle(
        checkpoint_dir=root,
        completed_steps=int(state["completed_steps"]),
        current_kl_penalty=float(state["current_kl_penalty"]),
        validation_state=dict(validation_state),
        last_validation_step=(
            int(state["last_validation_step"])
            if state.get("last_validation_step") is not None
            else None
        ),
        candidate_pool_rows=rows,
        observer_state=dict(observer_state),
        state=state,
    )


def restore_stable_ppo_training_state(
    *,
    bundle: StablePPOResumeBundle,
    optimizer: Any,
    value_model: Any,
    torch_module: Any,
) -> None:
    optimizer.load_state_dict(bundle.state["optimizer_state_dict"])
    value_head_path = bundle.checkpoint_dir / VALUE_HEAD_NAME
    value_head_state = torch_module.load(
        value_head_path,
        map_location="cpu",
        weights_only=True,
    )
    if not hasattr(value_model, "v_head"):
        raise ValueError("Stable PPO resume value model has no v_head")
    value_model.v_head.load_state_dict(value_head_state, strict=True)
    random.setstate(bundle.state["python_random_state"])
    torch_module.set_rng_state(bundle.state["torch_cpu_rng_state"].cpu())
    stored_cuda_count = int(bundle.state.get("torch_cuda_device_count", 0))
    if stored_cuda_count:
        if not torch_module.cuda.is_available():
            raise ValueError("CUDA resume state cannot be restored without CUDA")
        current_count = int(torch_module.cuda.device_count())
        if current_count != stored_cuda_count:
            raise ValueError(
                "CUDA device count changed across stable PPO resume: "
                f"stored={stored_cuda_count}, current={current_count}"
            )
        torch_module.cuda.set_rng_state_all(bundle.state["torch_cuda_rng_states"])


def adopt_stable_ppo_checkpoint_prefix(
    *,
    resume_checkpoint: Path,
    output_dir: Path,
    checkpoint_steps: Sequence[int],
) -> list[dict[str, Any]]:
    """Copy completed periodic checkpoints into one fresh resumed output root."""

    resume_manifest = read_stable_ppo_resume_manifest(resume_checkpoint)
    completed = int(resume_manifest["completed_steps"])
    canonical_resume = Path(str(resume_manifest["checkpoint_dir"]))
    if canonical_resume.name != f"checkpoint-{completed}":
        raise ValueError("Resume source is not a canonical periodic checkpoint")
    source_output = canonical_resume.parent
    frozen_contract = resume_manifest["resume_contract"]
    adopted: list[dict[str, Any]] = []
    for step in sorted(set(int(value) for value in checkpoint_steps)):
        if step > completed:
            continue
        source = source_output / f"checkpoint-{step}"
        manifest = read_stable_ppo_resume_manifest(source)
        if int(manifest["completed_steps"]) != step:
            raise ValueError(f"Stable PPO checkpoint-{step} step identity changed")
        if manifest.get("resume_contract") != frozen_contract:
            raise ValueError(
                f"Stable PPO checkpoint-{step} belongs to a different run contract"
            )
        destination = output_dir / source.name
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"Resume prefix destination is not fresh: {destination}")
        shutil.copytree(source, destination, symlinks=False)
        # The copied checkpoint is an immutable byte-for-byte adoption except
        # for its canonical location field.  Re-publish that one path-bound
        # document before validating the destination as an independently
        # resumable checkpoint.
        _atomic_json(
            destination / MANIFEST_NAME,
            {**manifest, "checkpoint_dir": str(destination.resolve(strict=True))},
        )
        copied = read_stable_ppo_resume_manifest(destination)
        adopted.append(
            {
                "step": step,
                "source": str(source),
                "destination": str(destination),
                "source_manifest_sha256": _sha256_file(source / MANIFEST_NAME),
                "copied_manifest_sha256": _sha256_file(destination / MANIFEST_NAME),
                "resume_contract_sha256": copied["resume_contract_sha256"],
            }
        )
    if completed not in {row["step"] for row in adopted}:
        raise ValueError("Resume checkpoint is not in the declared periodic prefix")
    return adopted


def find_latest_stable_ppo_resume_checkpoint(output_root: Path) -> Path | None:
    root = output_root.expanduser()
    if not root.is_dir() or root.is_symlink():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in root.glob("checkpoint-*"):
        try:
            step = int(path.name.removeprefix("checkpoint-"))
            manifest = read_stable_ppo_resume_manifest(path)
            if int(manifest["completed_steps"]) != step:
                continue
            contract = manifest.get("resume_contract") or {}
            if contract.get("stage") != "B7_PPO_FULL":
                continue
            max_steps = contract.get("max_steps")
            if (
                isinstance(max_steps, bool)
                or not isinstance(max_steps, int)
                or step >= max_steps
            ):
                continue
        except (OSError, TypeError, ValueError):
            continue
        candidates.append((step, path.resolve(strict=True)))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


__all__ = [
    "CANDIDATE_POOL_NAME",
    "MANIFEST_NAME",
    "SCHEMA_VERSION",
    "STATE_NAME",
    "StablePPOResumeBundle",
    "adopt_stable_ppo_checkpoint_prefix",
    "find_latest_stable_ppo_resume_checkpoint",
    "load_stable_ppo_resume_checkpoint",
    "read_stable_ppo_resume_manifest",
    "restore_stable_ppo_training_state",
    "save_stable_ppo_resume_checkpoint",
]
