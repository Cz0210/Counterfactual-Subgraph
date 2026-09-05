"""Fail-closed isolated loading for the pinned local ChemLLM 2B snapshot.

The parent process freezes the physical snapshot and audits every Python file
before starting a ``python -I -B`` child.  The child repeats those checks,
uses only the local snapshot, creates a fresh dynamic-module cache, and hides
all CUDA devices.  A CPU weight load is the only mode that may emit the
``ACTUAL_LOADED_WEIGHTS`` report used by the scale-ablation gate.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    LLMAblationContractError,
    canonical_json_sha256,
    require_sha256,
)
from .parameter_count import count_actual_loaded_parameters


CHEMLLM_2B_REPOSITORY_ID = "AI4Chem/CHEMLLM-2b-1_5"
CHEMLLM_2B_REVISION = "215c0dbc89417a06bbc3bae43a3ad61e58f0a56e"
CHEMLLM_2B_TOTAL_PARAMETERS = 1_889_110_016
SNAPSHOT_SCHEMA = "chemllm_snapshot_manifest_v1"
ISOLATED_RECEIPT_SCHEMA = "chemllm_2b_isolated_load_receipt_v1"

_BLOCKED_IMPORT_ROOTS = {
    "ftplib",
    "http",
    "paramiko",
    "requests",
    "socket",
    "subprocess",
    "urllib",
}
_BLOCKED_CALLS = {
    "copyfile",
    "shutil.copyfile",
    "__import__",
    "compile",
    "eval",
    "exec",
    "huggingface_hub.hf_hub_download",
    "huggingface_hub.snapshot_download",
    "import_module",
    "importlib.import_module",
    "os.popen",
    "os.makedirs",
    "os.mkdir",
    "os.remove",
    "os.rename",
    "os.replace",
    "os.system",
    "os.unlink",
    "pathlib.Path.rename",
    "pathlib.Path.replace",
    "pathlib.Path.unlink",
    "requests.delete",
    "requests.get",
    "requests.head",
    "requests.patch",
    "requests.post",
    "requests.put",
    "socket.create_connection",
    "socket.socket",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.Popen",
    "subprocess.run",
    "torch.utils.cpp_extension.load",
    "torch.hub.load",
    "urllib.request.urlopen",
}

# Reviewed exact upstream source files.  Their save_vocabulary method is an
# explicit export API, not an import/load/inference path; callers below disable
# that API on the tokenizer.  No filename-only or general write exception.
_AUDITED_UNUSED_TOKENIZER_EXPORTS = {
    ("tokenization_internlm2.py", "444d4c2b0da158e61c34b3c727943f0ad454770c74b307f4d881f03603335eef"),
    ("tokenization_internlm.py", "880e2cebff1d30db2acb485b8fc00299fda7a5efb2c4d8400bd9adf60d1158e0"),
}


def disable_tokenizer_exports(tokenizer: Any) -> None:
    """Inference/admission never saves or rewrites a tokenizer snapshot."""
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise LLMAblationContractError("Tokenizer export is disabled during isolated inference")
    tokenizer.save_vocabulary = forbidden
    tokenizer.save_pretrained = forbidden
_BLOCKED_MUTATING_METHODS = {
    "mkdir",
    "rmdir",
    "touch",
    "unlink",
    "write_bytes",
    "write_text",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _absolute_physical(path_like: str | Path, *, role: str, kind: str) -> Path:
    lexical = Path(path_like).expanduser()
    if not lexical.is_absolute() or lexical.is_symlink():
        raise LLMAblationContractError(f"{role} must be an absolute physical {kind}")
    try:
        resolved = lexical.resolve(strict=True)
    except FileNotFoundError as exc:
        raise LLMAblationContractError(f"{role} does not exist: {lexical}") from exc
    if Path(os.path.abspath(lexical)) != resolved:
        raise LLMAblationContractError(f"{role} may not traverse a symlink")
    predicate = resolved.is_dir if kind == "directory" else resolved.is_file
    if not predicate():
        raise LLMAblationContractError(f"{role} is not a {kind}: {resolved}")
    return resolved


def _safe_child(root: Path, value: str, *, role: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise LLMAblationContractError(f"{role} escapes the snapshot root")
    path = root / relative
    if path.is_symlink():
        raise LLMAblationContractError(f"{role} may not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise LLMAblationContractError(f"{role} is missing: {relative}") from exc
    if root not in resolved.parents or not resolved.is_file():
        raise LLMAblationContractError(f"{role} is not a physical snapshot file")
    return resolved


def _json_object(path: Path, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LLMAblationContractError(f"invalid {role}: {path}") from exc
    if not isinstance(value, dict):
        raise LLMAblationContractError(f"{role} must contain one JSON object")
    return value


@dataclass(frozen=True, slots=True)
class SnapshotFile:
    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.path or self.size < 0:
            raise LLMAblationContractError("invalid snapshot file inventory row")
        object.__setattr__(self, "sha256", require_sha256(self.sha256, field=self.path))


@dataclass(frozen=True, slots=True)
class ChemLLM2BSnapshotPin:
    root: str
    manifest_path: str
    manifest_sha256: str
    repository_id: str
    revision: str
    config_path: str
    weight_files: tuple[SnapshotFile, ...]
    source_files: tuple[SnapshotFile, ...]
    snapshot_inventory_sha256: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["weight_files"] = [asdict(row) for row in self.weight_files]
        payload["source_files"] = [asdict(row) for row in self.source_files]
        return payload


def _inventory_file(root: Path, path: Path) -> SnapshotFile:
    relative = path.relative_to(root).as_posix()
    return SnapshotFile(path=relative, size=path.stat().st_size, sha256=sha256_file(path))


def _weight_paths(root: Path) -> tuple[Path, ...]:
    index_path = root / "model.safetensors.index.json"
    single_path = root / "model.safetensors"
    if index_path.is_file() and not index_path.is_symlink():
        index = _json_object(index_path, role="safetensors index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise LLMAblationContractError("safetensors index lacks weight_map")
        names = sorted({str(value) for value in weight_map.values()})
        return tuple(
            [_safe_child(root, index_path.name, role="safetensors index")]
            + [_safe_child(root, name, role="safetensors shard") for name in names]
        )
    if single_path.is_file() and not single_path.is_symlink():
        return (_safe_child(root, single_path.name, role="safetensors weights"),)
    raise LLMAblationContractError("snapshot lacks physical safetensors weights")


def pin_chemllm_2b_snapshot(
    snapshot_root: str | Path,
    snapshot_manifest: str | Path,
    snapshot_manifest_sha256: str,
) -> ChemLLM2BSnapshotPin:
    """Reopen and byte-bind the exact local 2B revision."""

    root = _absolute_physical(snapshot_root, role="snapshot_root", kind="directory")
    if root.name != CHEMLLM_2B_REVISION:
        raise LLMAblationContractError("snapshot directory is not the pinned revision")
    manifest = _absolute_physical(
        snapshot_manifest, role="snapshot_manifest", kind="file"
    )
    if root not in manifest.parents:
        raise LLMAblationContractError("snapshot manifest must live inside snapshot root")
    expected_manifest_sha = require_sha256(
        snapshot_manifest_sha256, field="snapshot_manifest_sha256"
    )
    actual_manifest_sha = sha256_file(manifest)
    if actual_manifest_sha != expected_manifest_sha:
        raise LLMAblationContractError("snapshot manifest SHA256 changed")
    manifest_payload = _json_object(manifest, role="snapshot manifest")
    expected_manifest = {
        "schema_version": SNAPSHOT_SCHEMA,
        "status": "PASS",
        "repository_id": CHEMLLM_2B_REPOSITORY_ID,
        "revision": CHEMLLM_2B_REVISION,
        "weights_downloaded": True,
    }
    changed = [
        key for key, expected in expected_manifest.items()
        if manifest_payload.get(key) != expected
    ]
    if changed:
        raise LLMAblationContractError(
            "snapshot manifest identity changed: " + ", ".join(changed)
        )
    parameters = manifest_payload.get("parameters")
    if (
        not isinstance(parameters, Mapping)
        or parameters.get("count_source") != "downloaded_safetensors_tensor_headers"
        or parameters.get("total_parameters") != CHEMLLM_2B_TOTAL_PARAMETERS
    ):
        raise LLMAblationContractError("snapshot header parameter evidence changed")

    config_path = _safe_child(root, "config.json", role="model config")
    weights = _weight_paths(root)
    python_files = tuple(sorted(root.rglob("*.py")))
    if not python_files:
        raise LLMAblationContractError("trust_remote_code snapshot has no Python sources")
    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise LLMAblationContractError(f"snapshot contains a symlink: {entry}")
        if not entry.is_dir() and not entry.is_file():
            raise LLMAblationContractError(f"snapshot contains a special file: {entry}")
    weight_rows = tuple(_inventory_file(root, path) for path in weights)
    weight_path_set = set(weights)
    source_paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path not in weight_path_set
    )
    source_rows = tuple(_inventory_file(root, path) for path in source_paths)
    inventory_payload = {
        "repository_id": CHEMLLM_2B_REPOSITORY_ID,
        "revision": CHEMLLM_2B_REVISION,
        "manifest_sha256": actual_manifest_sha,
        "weight_files": [asdict(row) for row in weight_rows],
        "source_files": [asdict(row) for row in source_rows],
    }
    return ChemLLM2BSnapshotPin(
        root=str(root),
        manifest_path=str(manifest),
        manifest_sha256=actual_manifest_sha,
        repository_id=CHEMLLM_2B_REPOSITORY_ID,
        revision=CHEMLLM_2B_REVISION,
        config_path=str(config_path),
        weight_files=weight_rows,
        source_files=source_rows,
        snapshot_inventory_sha256=canonical_json_sha256(inventory_payload),
    )


def _call_name(node: ast.AST) -> str:
    parts: list[str] = []
    cursor: ast.AST | None = node
    while isinstance(cursor, ast.Attribute):
        parts.append(cursor.attr)
        cursor = cursor.value
    if isinstance(cursor, ast.Name):
        parts.append(cursor.id)
    return ".".join(reversed(parts))


def _write_open_call(node: ast.Call) -> bool:
    if _call_name(node.func) not in {"open", "io.open", "Path.open", "pathlib.Path.open"}:
        return False
    mode: ast.AST | None = node.args[1] if len(node.args) > 1 else None
    for keyword in node.keywords:
        if keyword.arg == "mode":
            mode = keyword.value
    if mode is None:
        return False
    if not isinstance(mode, ast.Constant) or not isinstance(mode.value, str):
        return True
    return any(flag in mode.value for flag in "wax+")


def _auto_map_modules(config: Mapping[str, Any]) -> set[str]:
    auto_map = config.get("auto_map")
    if not isinstance(auto_map, Mapping) or "AutoModelForCausalLM" not in auto_map:
        raise LLMAblationContractError("config lacks remote AutoModelForCausalLM mapping")
    modules: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, str):
            reference = value.split("--", 1)[-1]
            module, separator, _ = reference.rpartition(".")
            if separator and module:
                modules.add(module)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                visit(item)

    for value in auto_map.values():
        visit(value)
    return modules


def audit_remote_code(snapshot: ChemLLM2BSnapshotPin) -> dict[str, Any]:
    """Statically reject network, process, dynamic-code, and write side effects."""

    root = Path(snapshot.root)
    config = _json_object(Path(snapshot.config_path), role="model config")
    required_modules = _auto_map_modules(config)
    available_modules = {
        path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
        for path in root.rglob("*.py")
    }
    missing = sorted(required_modules - available_modules)
    if missing:
        raise LLMAblationContractError(
            "auto_map references missing source modules: " + ", ".join(missing)
        )

    rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    unused_export_evidence: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise LLMAblationContractError(f"remote source cannot be parsed: {relative}") from exc
        excluded_lines: set[int] = set()
        if (relative, digest) in _AUDITED_UNUSED_TOKENIZER_EXPORTS:
            for function in ast.walk(tree):
                if isinstance(function, ast.FunctionDef) and function.name == "save_vocabulary":
                    excluded_lines.update(range(function.lineno, function.end_lineno + 1))
                    unused_export_evidence.append({"path": relative, "sha256": digest,
                        "method": "save_vocabulary", "first_line": function.lineno,
                        "last_line": function.end_lineno, "runtime_export_disabled": True})
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_name = alias.name.split(".", 1)[0]
                    if root_name in _BLOCKED_IMPORT_ROOTS:
                        violations.append(
                            {"file": relative, "line": node.lineno, "kind": "IMPORT", "name": alias.name}
                        )
            elif isinstance(node, ast.ImportFrom):
                root_name = str(node.module or "").split(".", 1)[0]
                if root_name in _BLOCKED_IMPORT_ROOTS:
                    violations.append(
                        {"file": relative, "line": node.lineno, "kind": "IMPORT", "name": node.module}
                    )
            elif isinstance(node, ast.Call):
                name = _call_name(node.func)
                leaf = name.rsplit(".", 1)[-1]
                if (
                    name in _BLOCKED_CALLS
                    or leaf in _BLOCKED_MUTATING_METHODS
                    or _write_open_call(node)
                ) and node.lineno not in excluded_lines:
                    violations.append(
                        {"file": relative, "line": node.lineno, "kind": "CALL", "name": name}
                    )
        rows.append({"path": relative, "sha256": digest, "size": path.stat().st_size})
    if violations:
        first = violations[0]
        raise LLMAblationContractError(
            "remote-code static audit failed at "
            f"{first['file']}:{first['line']} ({first['kind']} {first['name']})"
        )
    payload: dict[str, Any] = {
        "schema_version": "chemllm_remote_code_static_audit_v1",
        "status": "PASS",
        "repository_id": snapshot.repository_id,
        "revision": snapshot.revision,
        "required_auto_map_modules": sorted(required_modules),
        "source_files": rows,
        "violation_count": 0,
        "audited_unused_export_methods": unused_export_evidence,
        "policy": {
            "network_imports_blocked": True,
            "subprocess_calls_blocked": True,
            "dynamic_code_calls_blocked": True,
            "write_side_effect_calls_blocked": True,
        },
    }
    payload["code_inventory_sha256"] = canonical_json_sha256(
        {"source_files": rows, "required_auto_map_modules": sorted(required_modules)}
    )
    payload["audit_sha256"] = canonical_json_sha256(payload)
    return payload


def prepare_fresh_output_root(output_root: str | Path) -> Path:
    root = Path(output_root).expanduser()
    if not root.is_absolute() or root.is_symlink():
        raise LLMAblationContractError("output_root must be a fresh absolute path")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(mode=0o750, exist_ok=False)
    return root.resolve(strict=True)


def build_isolated_child_environment(
    source: Mapping[str, str], output_root: str | Path
) -> dict[str, str]:
    """Return a minimal offline environment with fresh model/module caches."""

    root = _absolute_physical(output_root, role="output_root", kind="directory")
    keep = {
        "CONDA_DEFAULT_ENV",
        "CONDA_EXE",
        "CONDA_PREFIX",
        "DYLD_LIBRARY_PATH",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "PATH",
        "SHELL",
        "SSL_CERT_FILE",
    }
    environment = {key: value for key, value in source.items() if key in keep and value}
    paths = {
        "HF_HOME": root / "hf_home",
        "HF_MODULES_CACHE": root / "hf_modules_cache",
        "HUGGINGFACE_HUB_CACHE": root / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": root / "hf_home" / "transformers",
        "TMPDIR": root / "tmp",
        "HOME": root / "home",
        "TORCH_HOME": root / "torch_home",
        "XDG_CACHE_HOME": root / "xdg_cache",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=False)
    environment.update({key: str(value) for key, value in paths.items()})
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    return environment


def build_isolated_child_command(
    *,
    python: str,
    script: str | Path,
    snapshot: ChemLLM2BSnapshotPin,
    output_root: str | Path,
    mode: str,
    tiny_forward: bool,
    code_inventory_sha256: str,
    config: str | Path,
    config_overrides: Iterable[str] = (),
) -> list[str]:
    if mode not in {"metadata", "cpu-load"}:
        raise LLMAblationContractError("isolated load mode must be metadata or cpu-load")
    if tiny_forward and mode != "cpu-load":
        raise LLMAblationContractError("tiny forward requires cpu-load mode")
    command = [
        str(python),
        "-I",
        "-B",
        str(Path(script).resolve(strict=True)),
        "--_isolated-child",
        "--config",
        str(Path(config).resolve(strict=True)),
        "--snapshot-root",
        snapshot.root,
        "--snapshot-manifest",
        snapshot.manifest_path,
        "--snapshot-manifest-sha256",
        snapshot.manifest_sha256,
        "--output-root",
        str(Path(output_root).resolve(strict=True)),
        "--mode",
        mode,
        "--expected-code-inventory-sha256",
        require_sha256(code_inventory_sha256, field="code_inventory_sha256"),
    ]
    for value in config_overrides:
        command.extend(["--set", str(value)])
    if tiny_forward:
        command.append("--tiny-forward")
    return command


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(directory.resolve(strict=True))
    except ValueError:
        return False
    return True


def run_isolated_child_probe(
    snapshot: ChemLLM2BSnapshotPin,
    audit: Mapping[str, Any],
    output_root: str | Path,
    *,
    mode: str,
    tiny_forward: bool,
) -> dict[str, Any]:
    """Import pinned remote code and optionally load all weights on CPU."""

    if (
        sys.flags.isolated != 1
        or sys.dont_write_bytecode is not True
        or sys.flags.no_user_site != 1
    ):
        raise LLMAblationContractError("probe must run under python -I -B")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise LLMAblationContractError("isolated child must hide all CUDA devices")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise LLMAblationContractError("isolated child must disable user site packages")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise LLMAblationContractError("isolated child must remain offline")
    root = _absolute_physical(output_root, role="output_root", kind="directory")
    modules_cache = _absolute_physical(
        os.environ.get("HF_MODULES_CACHE", ""),
        role="HF_MODULES_CACHE",
        kind="directory",
    )
    if root not in modules_cache.parents or any(modules_cache.iterdir()):
        raise LLMAblationContractError("HF_MODULES_CACHE must be fresh inside output_root")

    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except Exception as exc:  # pragma: no cover - exercised on AutoDL
        raise LLMAblationContractError("isolated transformer imports failed") from exc
    if torch.cuda.is_available() or torch.cuda.device_count() != 0:
        raise LLMAblationContractError("CUDA remained visible inside isolated child")

    common = {
        "trust_remote_code": True,
        "local_files_only": True,
        "revision": snapshot.revision,
        "code_revision": snapshot.revision,
    }
    try:
        config = AutoConfig.from_pretrained(snapshot.root, **common)
        tokenizer = AutoTokenizer.from_pretrained(snapshot.root, use_fast=False, **common)
        disable_tokenizer_exports(tokenizer)
        auto_map = getattr(config, "auto_map", None)
        model_reference = auto_map.get("AutoModelForCausalLM") if isinstance(auto_map, Mapping) else None
        if not isinstance(model_reference, str) or not model_reference:
            raise LLMAblationContractError("loaded config lacks AutoModelForCausalLM")
        model_class = get_class_from_dynamic_module(
            model_reference,
            snapshot.root,
            revision=snapshot.revision,
            code_revision=snapshot.revision,
            local_files_only=True,
        )
        class_source = Path(inspect.getfile(model_class)).resolve(strict=True)
    except LLMAblationContractError:
        raise
    except Exception as exc:  # pragma: no cover - exercised on AutoDL
        raise LLMAblationContractError("pinned config/tokenizer/model-class load failed") from exc
    if not _inside(class_source, modules_cache):
        raise LLMAblationContractError("remote model class did not load from fresh HF_MODULES_CACHE")

    parameter_report_path: Path | None = None
    parameter_report_sha: str | None = None
    forward_receipt: dict[str, Any] | None = None
    if mode == "cpu-load":
        try:
            model = AutoModelForCausalLM.from_pretrained(
                snapshot.root,
                config=config,
                trust_remote_code=True,
                local_files_only=True,
                revision=snapshot.revision,
                code_revision=snapshot.revision,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            model.requires_grad_(False)
            model.eval()
        except Exception as exc:  # pragma: no cover - exercised on AutoDL
            raise LLMAblationContractError("pinned CPU weight load failed") from exc
        parameters = list(model.parameters())
        if not parameters or any(parameter.device.type != "cpu" for parameter in parameters):
            raise LLMAblationContractError("loaded model is not wholly resident on CPU")
        if any(bool(getattr(parameter, "is_meta", False)) for parameter in parameters):
            raise LLMAblationContractError("loaded model retained meta tensors")
        report = count_actual_loaded_parameters(model).to_dict()
        if report["total_parameters"] != CHEMLLM_2B_TOTAL_PARAMETERS:
            raise LLMAblationContractError("actual loaded 2B parameter count changed")
        if report["lora_trainable_parameters"] != 0:
            raise LLMAblationContractError("off-the-shelf 2B unexpectedly contains LoRA")
        parameter_report_path = root / "actual_parameter_count_report.json"
        atomic_json(parameter_report_path, report)
        parameter_report_sha = sha256_file(parameter_report_path)
        if tiny_forward:
            try:
                # The pinned model's native chat entrypoint is authoritative;
                # its tokenizer_config carries a conflicting generic INST template.
                prompt = "MOLECULE_SMILES: CCO\nFRAGMENT_SMILES:"
                tokenized = model.build_inputs(tokenizer, prompt, history=[], meta_instruction="")
                inputs = {key: value.to("cpu") for key, value in tokenized.items()}
                with torch.inference_mode():
                    output = model(**inputs, use_cache=False)
                logits = output.logits
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=4, do_sample=False,
                        num_return_sequences=1, use_cache=False,
                        eos_token_id=[tokenizer.eos_token_id,
                            tokenizer.convert_tokens_to_ids("<|im_end|>")],
                        pad_token_id=tokenizer.pad_token_id)
            except Exception as exc:  # pragma: no cover - exercised on AutoDL
                raise LLMAblationContractError("optional tiny forward failed") from exc
            if (logits.ndim != 3 or logits.shape[0] != 1 or not bool(torch.isfinite(logits).all())
                    or generated.ndim != 2 or generated.shape[0] != 1
                    or not 0 < generated.shape[1] - inputs["input_ids"].shape[1] <= 4):
                raise LLMAblationContractError("optional tiny forward produced invalid logits")
            forward_receipt = {
                "status": "PASS",
                "batch": int(logits.shape[0]),
                "sequence_length": int(logits.shape[1]),
                "vocab_size": int(logits.shape[2]),
                "finite": True,
                "native_prompt_api": "model.build_inputs(history=[],meta_instruction='')",
                "tiny_generation_token_count": int(generated.shape[1] - inputs["input_ids"].shape[1]),
                "tiny_generation_max_new_tokens": 4,
                "tiny_generation_only": True,
            }
    elif mode != "metadata":
        raise LLMAblationContractError("isolated load mode must be metadata or cpu-load")

    receipt: dict[str, Any] = {
        "schema_version": ISOLATED_RECEIPT_SCHEMA,
        "status": "PASS",
        "repository_id": snapshot.repository_id,
        "revision": snapshot.revision,
        "snapshot_inventory_sha256": snapshot.snapshot_inventory_sha256,
        "remote_code_audit_sha256": audit.get("audit_sha256"),
        "code_inventory_sha256": audit.get("code_inventory_sha256"),
        "mode": mode,
        "isolated_import_pass": True,
        "trust_remote_code_enabled": True,
        "local_files_only": True,
        "offline_mode": True,
        "python_isolated_flag": True,
        "python_no_bytecode_flag": True,
        "python_no_user_site": True,
        "cuda_visible_devices": "",
        "cuda_available": False,
        "cuda_device_count": 0,
        "hf_modules_cache": str(modules_cache),
        "model_class": f"{model_class.__module__}.{model_class.__name__}",
        "model_class_source": str(class_source),
        "tokenizer_class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}",
        "weights_loaded": mode == "cpu-load",
        "actual_parameter_report": str(parameter_report_path) if parameter_report_path else None,
        "actual_parameter_report_file_sha256": parameter_report_sha,
        "tiny_forward_requested": tiny_forward,
        "tiny_forward": forward_receipt,
        "main_gpu_lock_acquired": False,
        "main_output_root_written": False,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    atomic_json(root / "isolated_load_receipt.json", receipt)
    return receipt


def validate_isolated_load_receipt(path_like: str | Path, *, require_weights: bool) -> dict[str, Any]:
    path = _absolute_physical(path_like, role="isolated_load_receipt", kind="file")
    payload = _json_object(path, role="isolated load receipt")
    claimed = require_sha256(payload.get("receipt_sha256"), field="receipt_sha256")
    body = dict(payload)
    body.pop("receipt_sha256")
    if canonical_json_sha256(body) != claimed:
        raise LLMAblationContractError("isolated load receipt self hash changed")
    expected = {
        "schema_version": ISOLATED_RECEIPT_SCHEMA,
        "status": "PASS",
        "repository_id": CHEMLLM_2B_REPOSITORY_ID,
        "revision": CHEMLLM_2B_REVISION,
        "isolated_import_pass": True,
        "trust_remote_code_enabled": True,
        "local_files_only": True,
        "offline_mode": True,
        "python_isolated_flag": True,
        "python_no_bytecode_flag": True,
        "python_no_user_site": True,
        "cuda_visible_devices": "",
        "cuda_available": False,
        "cuda_device_count": 0,
        "main_gpu_lock_acquired": False,
        "main_output_root_written": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise LLMAblationContractError("isolated load receipt contract changed")
    for field in (
        "snapshot_inventory_sha256",
        "remote_code_audit_sha256",
        "code_inventory_sha256",
    ):
        require_sha256(payload.get(field), field=field)
    cache = _absolute_physical(
        payload.get("hf_modules_cache", ""), role="receipt.HF_MODULES_CACHE", kind="directory"
    )
    if path.parent not in cache.parents:
        raise LLMAblationContractError("receipt HF_MODULES_CACHE escaped output root")
    if require_weights and (
        payload.get("mode") != "cpu-load"
        or payload.get("weights_loaded") is not True
        or not payload.get("actual_parameter_report")
        or not payload.get("actual_parameter_report_file_sha256")
    ):
        raise LLMAblationContractError("CPU-load receipt lacks actual weight evidence")
    if require_weights:
        report_path = _absolute_physical(
            payload["actual_parameter_report"], role="actual_parameter_report", kind="file"
        )
        if report_path.parent != path.parent:
            raise LLMAblationContractError("actual parameter report escaped output root")
        claimed_file_sha = require_sha256(
            payload["actual_parameter_report_file_sha256"],
            field="actual_parameter_report_file_sha256",
        )
        if sha256_file(report_path) != claimed_file_sha:
            raise LLMAblationContractError("actual parameter report file SHA256 changed")
        report = _json_object(report_path, role="actual parameter report")
        report_self_sha = require_sha256(
            report.get("parameter_report_sha256"), field="parameter_report_sha256"
        )
        report_body = dict(report)
        report_body.pop("parameter_report_sha256")
        if canonical_json_sha256(report_body) != report_self_sha:
            raise LLMAblationContractError("actual parameter report self hash changed")
        if (
            report.get("schema_version") != "actual_parameter_count_report_v1"
            or report.get("source") != "ACTUAL_LOADED_WEIGHTS"
            or report.get("total_parameters") != CHEMLLM_2B_TOTAL_PARAMETERS
            or report.get("lora_trainable_parameters") != 0
        ):
            raise LLMAblationContractError("actual 2B parameter report contract changed")
    elif payload.get("weights_loaded") is not False or any(
        payload.get(field) is not None
        for field in (
            "actual_parameter_report",
            "actual_parameter_report_file_sha256",
        )
    ):
        raise LLMAblationContractError("metadata receipt claims weight evidence")
    return payload


__all__ = [
    "CHEMLLM_2B_REPOSITORY_ID",
    "CHEMLLM_2B_REVISION",
    "CHEMLLM_2B_TOTAL_PARAMETERS",
    "ChemLLM2BSnapshotPin",
    "SnapshotFile",
    "atomic_json",
    "audit_remote_code",
    "build_isolated_child_command",
    "build_isolated_child_environment",
    "pin_chemllm_2b_snapshot",
    "prepare_fresh_output_root",
    "run_isolated_child_probe",
    "sha256_file",
    "validate_isolated_load_receipt",
]
