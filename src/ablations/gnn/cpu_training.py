"""BACE CPU timing and training through the existing molecular trainer.

Benchmark epochs are real training epochs in the same resumable attempt.  The
wrapper changes only where execution pauses; the frozen optimizer, sampling,
validation selector and 200-epoch ceiling remain the same on continuation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import tempfile
import time
from typing import Any, Mapping

import yaml


BUNDLE_SCHEMA = "bace_gnn_cpu_bundle_v1"
TRAINED_BACKBONES = ("gin", "gcn", "gatv2", "gatedgcn_plus")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def bundle_file(root: Path, manifest: Mapping[str, Any], relative: str) -> Path:
    """Resolve a declared regular input file without permitting bundle escape."""
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError("Bundle paths must be relative without parent traversal")
    path = root / rel
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Bundle inputs may not contain symlinks")
    path.resolve(strict=True).relative_to(root.resolve(strict=True))
    entry = manifest.get("files", {}).get(relative)
    if not isinstance(entry, Mapping) or not path.is_file():
        raise ValueError(f"Undeclared bundle input: {relative}")
    if path.stat().st_size != entry.get("size") or file_sha256(path) != entry.get("sha256"):
        raise ValueError(f"Bundle input hash/size mismatch: {relative}")
    return path


def load_bundle(root: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / "bundle_manifest.json").read_text())
    from src.ablations.contracts import canonical_json_sha256
    if manifest.get("manifest_sha256") != canonical_json_sha256({
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }):
        raise ValueError("Bundle manifest self-hash mismatch")
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA
        or manifest.get("dataset") != "bace"
        or manifest.get("seed") != 7
        or manifest.get("num_classes") != 2
        or set(manifest.get("splits", {})) != {"train", "validation", "calibration", "test"}
    ):
        raise ValueError("CPU training requires the frozen BACE seed-7 bundle")
    validate_gine_source_inventory(root, manifest)
    return root, manifest


def validate_gine_source_inventory(root: Path, manifest: Mapping[str, Any]) -> None:
    """Retain and verify the source classifier's own checksum inventory."""
    relative_root = str(manifest["gine_reference_root"]).rstrip("/")
    inventory = bundle_file(root, manifest, relative_root + "/sha256sums.txt")
    seen: set[str] = set()
    for line in inventory.read_text().splitlines():
        digest, separator, relative = line.partition("  ")
        if (not separator or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or Path(relative).name != relative or relative in seen):
            raise ValueError("Invalid source GINE SHA inventory")
        seen.add(relative)
        source = bundle_file(root, manifest, relative_root + "/" + relative)
        if file_sha256(source) != digest:
            raise ValueError("Source GINE SHA inventory mismatch")
    required = {"model.pt", "config.yaml", "model_card.json", "feature_schema.json",
                "label_map.json", "split_manifest.json", "training_metrics.json",
                "validation_predictions.csv", "test_evaluation_status.json",
                "temperature_scaling.json", "environment.json", "git_state.json"}
    if not required.issubset(seen):
        raise ValueError("Source GINE SHA inventory omits required classifier files")


def effective_training_config(
    root: Path, manifest: Mapping[str, Any], backbone: str
) -> dict[str, Any]:
    if backbone not in TRAINED_BACKBONES:
        raise ValueError("GINE is adopted from the reference; only four alternatives train")
    source = bundle_file(root, manifest, manifest["training_config_path"])
    architecture = bundle_file(root, manifest, manifest["backbone_configs"][backbone])
    config = copy.deepcopy(yaml.safe_load(source.read_text()))
    architecture_config = yaml.safe_load(architecture.read_text())
    reference = config["gnn"]
    candidate = architecture_config["gnn"]
    if reference.get("backbone") != "gine":
        raise ValueError("Training reference must be the adopted GINE configuration")
    if candidate.get("backbone") != backbone:
        raise ValueError("Requested backbone differs from frozen architecture")
    from src.models.molecular_gnn import MolecularGNNConfig
    from src.models.gnn_backbone_registry import get_gnn_backbone_spec
    defaults = MolecularGNNConfig().to_dict()
    common = ("num_layers", "dropout", "pooling", "readout_layers", "normalization", "residual", "num_classes")
    for field in common:
        if candidate.get(field, defaults[field]) != reference.get(field, defaults[field]):
            raise ValueError(f"Backbone changes matched reference architecture field: {field}")
    extras = {"ffn", "rwpe_walk_length", "rwpe_dim", "rwpe_raw_normalization"} if backbone == "gatedgcn_plus" else set()
    permitted = set(common) | {"backbone", "hidden_dim", "edge_feature_mode"} | extras
    if set(candidate) - permitted:
        raise ValueError("Backbone contains unapproved architecture fields")
    expected_hidden = 160 if backbone == "gatedgcn_plus" else reference.get("hidden_dim", defaults["hidden_dim"])
    if candidate.get("hidden_dim", defaults["hidden_dim"]) != expected_hidden:
        raise ValueError("Backbone changes the authorized hidden dimension")
    edge_mode = get_gnn_backbone_spec(backbone).edge_feature_mode
    if candidate.get("edge_feature_mode", edge_mode) != edge_mode:
        raise ValueError("Backbone edge feature handling differs from the registry")
    config["gnn"].update(copy.deepcopy(candidate))
    if "edge_feature_mode" in config["gnn"]:
        config["gnn"]["edge_feature_mode"] = edge_mode
    training = config["training"]
    if str(training.get("optimizer", "")).lower() != "adamw":
        raise ValueError("The matched molecular trainer supports AdamW only")
    if training.get("scheduler", "constant") not in {None, "none", "constant"}:
        raise ValueError("Nonconstant schedulers are not supported by the frozen trainer")
    # Main-classifier quality thresholds must not censor poor ablation results.
    training["health_gate"] = {"enabled": False}
    config.setdefault("runtime", {}).update({"device": "cpu", "num_workers": 0})
    return config


def _mapping_yaml(values: Mapping[str, Any], indent: int = 0) -> str:
    """Serialize the exact scalar/mapping subset consumed by the real trainer."""
    rows: list[str] = []
    for key, value in sorted(values.items()):
        if not isinstance(key, str) or any(char in key for char in ":\n"):
            raise ValueError("Unsupported frozen config key")
        prefix = " " * indent + key + ":"
        if isinstance(value, Mapping):
            rows.append(prefix + "\n" + _mapping_yaml(value, indent + 2))
        elif value is None or isinstance(value, (str, bool, int, float)):
            rows.append(prefix + " " + json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        else:
            raise ValueError("Frozen trainer config requires mapping/scalar YAML")
    return "".join(rows)


def classify_cpu_admission(training_seconds: float | None, evaluation_seconds: float | None) -> str:
    """Resource decisions use runtime only; unknown evaluation never admits full."""
    if training_seconds is None or not math.isfinite(training_seconds) or training_seconds > 12 * 3600:
        return "GPU_FALLBACK_REQUIRED"
    if evaluation_seconds is not None and math.isfinite(evaluation_seconds) and training_seconds + evaluation_seconds <= 12 * 3600:
        return "CPU_FULL_ELIGIBLE"
    return "CPU_TRAIN_ONLY_ELIGIBLE"


def trainer_arguments(
    *, root: Path, manifest: Mapping[str, Any], backbone: str,
    output_root: Path, effective_config_path: Path, resume: bool,
) -> list[str]:
    args = [
        "--config", str(effective_config_path), "--dataset", "bace",
        "--data-dir", str(root), "--output-dir", str(output_root / "classifier"),
        "--training-state-dir", str(output_root / "training_state"),
        "--profile", "full", "--device", "cpu", "--backbone", backbone,
        "--seed", "7", "--num-workers", "0",
    ]
    for split, relative in manifest["splits"].items():
        path = bundle_file(root, manifest, relative)
        args.extend([f"--{split}-csv", str(path)])
    if resume:
        args.append("--resume-training")
    return args


class _EpochPause(Exception):
    pass


def run_cpu_training(
    *, bundle_root: str | Path, backbone: str, phase: str,
    output_root: str | Path, config_path: str | Path,
    cpu_threads: int = 8, benchmark_epochs: int = 5,
    benchmark_seconds: float = 1200, resume: bool = False,
) -> dict[str, Any]:
    """Run genuine CPU science, pausing only after a committed epoch boundary."""
    if phase not in {"benchmark", "train"}:
        raise ValueError("phase must be benchmark or train")
    if not 1 <= benchmark_epochs <= 5 or not 0 < benchmark_seconds <= 1200:
        raise ValueError("Benchmark must be bounded by five epochs and 1200 seconds")
    if cpu_threads < 1 or cpu_threads > int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_threads)):
        raise ValueError("CPU thread count exceeds the allocated CPUs")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}:
        raise ValueError("CPU ablation must not receive any visible GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(cpu_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    root, manifest = load_bundle(bundle_root)
    config_path = Path(config_path).resolve(strict=True)
    relative_config = str(config_path.relative_to(root))
    bundle_file(root, manifest, relative_config)
    if relative_config != manifest["backbone_configs"].get(backbone):
        raise ValueError("--config must select the bundle's pinned backbone config")
    output = Path(output_root).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Output root must be disjoint from the input bundle")
    if output.exists() and not resume:
        raise FileExistsError("CPU attempt root must be fresh; use --resume for its checkpoint")
    if resume and not output.is_dir():
        raise FileNotFoundError("Resume requires the existing attempt root")
    output.mkdir(parents=True, exist_ok=resume)
    config = effective_training_config(root, manifest, backbone)
    config_file = output / "effective_config.yaml"
    if resume:
        if yaml.safe_load(config_file.read_text()) != config:
            raise ValueError("Frozen effective training config changed across resume")
    else:
        atomic_text(config_file, _mapping_yaml(config))
    from src.utils.env import load_yaml_config
    if load_yaml_config(config_file) != config:
        raise ValueError("Frozen config is not representable by the existing trainer YAML parser")
    schema_path = bundle_file(root, manifest, manifest["feature_schema_path"])
    from src.data.molecular_graph_featurizer import default_molecular_feature_schema
    if json.loads(schema_path.read_text()) != default_molecular_feature_schema().to_dict():
        raise ValueError("The adopted reference feature schema differs from the molecular trainer")

    import torch
    from scripts import train_molecular_gnn as trainer
    torch.set_num_threads(cpu_threads)
    # No scheduler exists in the reference trainer: constant LR has empty state.
    contract = {
        "schema_version": "bace_gnn_cpu_training_v1", "backbone": backbone,
        "bundle_manifest_sha256": file_sha256(root / "bundle_manifest.json"),
        "effective_config_sha256": file_sha256(config_file), "seed": 7,
        "cpu_threads": cpu_threads, "device": "cpu",
        "scheduler": {"kind": "constant", "state": {}},
        "calibration_split_loaded": False, "test_split_loaded": False,
        "temperature_fit_split": "validation", "main_matrix_write": False,
        "performance_threshold_applied": False,
    }
    contract_file = output / "cpu_contract.json"
    if resume:
        if json.loads(contract_file.read_text()) != contract:
            raise ValueError("CPU training execution contract changed across resume")
    else:
        atomic_json(contract_file, contract)
    atomic_json(output / "cpu_progress.json", {
        **contract, "phase": phase, "pid": os.getpid(),
        "status": "CPU_RUNNING", "resuming": resume,
    })
    started = time.monotonic()
    started_cpu = time.process_time()
    last_boundary = started
    samples: list[dict[str, Any]] = []
    validation_samples: list[dict[str, float]] = []
    paused = False
    request = {"stop": False, "saving": False, "external": False}
    stores: list[Any] = []
    parents: list[Any] = []
    original_store = trainer.MolecularGNNResumeStore
    original_parent = trainer.OutputParentAuthority
    original_evaluate = trainer._evaluate
    previous_handlers: dict[int, Any] = {}

    class BoundaryStore(original_store):
        def open(self) -> None:
            super().open()
            stores.append(self)

        def save(self, **values: Any) -> dict[str, Any]:
            nonlocal last_boundary
            if not math.isfinite(float(values["metrics"]["train_loss"])):
                raise ValueError("Nonfinite training loss")
            for tensor in values["model"].state_dict().values():
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise ValueError("Nonfinite model state")
            checkpoint_started = time.monotonic()
            values["metrics"] = {**values["metrics"], "scheduler": {"kind": "constant", "state": {}}}
            request["saving"] = True
            try:
                result = super().save(**values)
            finally:
                request["saving"] = False
            now = time.monotonic()
            row = {
                "epoch": int(values["completed_epoch"]),
                "epoch_wall_seconds": now - last_boundary,
                "checkpoint_seconds": now - checkpoint_started,
                "checkpoint_bytes": result["checkpoint_bytes"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "loss": float(values["metrics"]["train_loss"]),
                "validation_selection": values["metrics"]["selection"],
                "scheduler": {"kind": "constant", "state": {}},
                "validation_forward_seconds": validation_samples[-1]["seconds"] if validation_samples else None,
                "parameters": sum(parameter.numel() for parameter in values["model"].parameters()),
            }
            samples.append(row)
            last_boundary = now
            atomic_json(output / "cpu_progress.json", {
                **contract, "pid": os.getpid(), "completed_epoch": row["epoch"],
                "epoch_samples": samples, "elapsed_seconds": now - started,
                "status": "CHECKPOINT_COMPLETE",
            })
            capped = phase == "benchmark" and (
                len(samples) >= benchmark_epochs or now - started >= benchmark_seconds
            )
            if (capped or request["stop"]) and row["epoch"] < int(config["training"]["max_epochs"]):
                raise _EpochPause()
            return result

    class TrackedParent(original_parent):
        def open(self) -> None:
            super().open()
            parents.append(self)

    def ask_safe_stop(_signal: int, _frame: Any) -> None:
        request["stop"] = True
        request["external"] = True

    def benchmark_deadline(_signal: int, _frame: Any) -> None:
        request["stop"] = True
        if not request["saving"]:
            # An unfinished epoch is discarded; the last committed RNG/state is
            # retained for exact replay. Never interrupt atomic checkpoint save.
            raise _EpochPause()

    def timed_evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        begin = time.monotonic()
        value = original_evaluate(*args, **kwargs)
        validation_samples.append({"seconds": time.monotonic() - begin, "examples": len(value["labels"])})
        return value

    trainer.MolecularGNNResumeStore = BoundaryStore
    trainer.OutputParentAuthority = TrackedParent
    trainer._evaluate = timed_evaluate
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        previous_handlers[signum] = signal.signal(signum, ask_safe_stop)
    previous_alarm = None
    if phase == "benchmark":
        previous_alarm = signal.signal(signal.SIGALRM, benchmark_deadline)
        signal.setitimer(signal.ITIMER_REAL, benchmark_seconds)
    try:
        result = trainer.main(trainer_arguments(
            root=root, manifest=manifest, backbone=backbone, output_root=output,
            effective_config_path=config_file, resume=resume,
        ))
        if result != 0:
            raise RuntimeError(f"Molecular trainer exited with {result}")
    except _EpochPause:
        paused = True
    finally:
        trainer.MolecularGNNResumeStore = original_store
        trainer.OutputParentAuthority = original_parent
        trainer._evaluate = original_evaluate
        if phase == "benchmark":
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm)
        for store in stores:
            store.close()
        for authority in parents:
            authority.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    elapsed = time.monotonic() - started
    checkpoint_path = output / "training_state/latest_checkpoint.json"
    checkpoint_manifest = json.loads(checkpoint_path.read_text()) if checkpoint_path.is_file() else None
    # Exclude cold dataset/model setup if multiple complete epoch samples exist.
    timed = samples[1:] if len(samples) > 1 else samples
    mean_epoch = sum(row["epoch_wall_seconds"] for row in timed) / len(timed) if timed else None
    if mean_epoch is None and (output / "benchmark.json").is_file():
        mean_epoch = json.loads((output / "benchmark.json").read_text()).get("seconds_per_epoch")
    projected_training = None if mean_epoch is None else mean_epoch * int(config["training"]["max_epochs"]) * 1.5
    benchmark = {
        **contract, "phase": phase, "status": "PAUSED_AT_CHECKPOINT" if paused else "TRAINING_PASS",
        "completed_epoch": checkpoint_manifest["completed_epoch"] if checkpoint_manifest else 0,
        "resume_checkpoint": str(output / "training_state" / checkpoint_manifest["checkpoint_file"]) if checkpoint_manifest else None,
        "epoch_samples": samples, "elapsed_seconds": elapsed,
        "rss_peak_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "rss_native_unit": "bytes" if os.uname().sysname == "Darwin" else "KiB",
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if os.uname().sysname == "Darwin" else 1024),
        "cpu_seconds": time.process_time() - started_cpu,
        "cpu_utilization_percent": 100 * (time.process_time() - started_cpu) / max(elapsed, 1e-9),
        "validation_timing_samples": validation_samples,
        "seconds_per_epoch": mean_epoch,
        "full_training_projected_seconds": projected_training,
        "projected_training_hours": None if projected_training is None else projected_training / 3600,
        "full_evaluation_projected_seconds": None,
        "projected_total_ablation_hours": None,
        "calibration_time_estimate_seconds": None,
        "cpu_admission": classify_cpu_admission(projected_training, None),
        "evaluation_eta_state": "NOT_MEASURED",
        "benchmark_budget_seconds": benchmark_seconds,
        "benchmark_stop_boundary": "NEXT_COMMITTED_EPOCH",
        "science_rerun_on_resume": False, "classifier_root": str(output / "classifier") if not paused else None,
        "external_stop_requested": request["external"],
    }
    atomic_json(output / ("benchmark.json" if phase == "benchmark" else "training_terminal.json"), benchmark)
    return benchmark


def run_cpu_auto(**options: Any) -> dict[str, Any]:
    """Benchmark then resume the same attempt if its measured CPU time fits."""
    root = Path(options["output_root"])
    benchmark_path = root / "benchmark.json"
    if options.get("resume") and benchmark_path.is_file():
        benchmark = json.loads(benchmark_path.read_text())
    else:
        benchmark = run_cpu_training(**options, phase="benchmark")
    if benchmark.get("external_stop_requested") is True:
        atomic_json(root / "auto_terminal.json", benchmark)
        return benchmark
    if benchmark["cpu_admission"] == "GPU_FALLBACK_REQUIRED":
        result = {**benchmark, "status": "READY_GNN_GPU_FALLBACK", "gpu_started": False}
        atomic_json(root / "auto_terminal.json", result)
        return result
    if benchmark["status"] == "TRAINING_PASS":
        result = benchmark
    else:
        train_options = {**options, "resume": True}
        result = run_cpu_training(**train_options, phase="train")
    atomic_json(root / "auto_terminal.json", result)
    return result
