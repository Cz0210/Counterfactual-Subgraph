"""CM-only verified small-package transport; no main authority or registry.

Packaging/importing is record-only and cannot turn a provisional scientific
audit into PASS. Links, traversal, unknown members and overwrites fail closed.
"""
from __future__ import annotations

import ctypes
import errno
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
from typing import Any, Callable
import uuid

from src.baselines.cm_crem_runtime import atomic_json, digest, utc_now

ALLOWED_ROOTS = (
    Path("/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/baselines/cm_crem_global_v1"),
    Path("/Volumes/DireRaven/counterfactual-hpc-offload/cm-crem-global-v1"),
    Path("/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/baselines/cm_crem_global_v1"),
    Path("/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/baselines/cm_crem_global_v2"),
    Path("/Volumes/DireRaven/counterfactual-hpc-offload/cm-crem-global-v2"),
    Path("/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/baselines/cm_crem_global_v2"),
)
SCHEMA = "cm_crem_portable_result_package_v1"
METHOD = "CM-CReM-Global-Budgeted-v1"
TOP_FILES = {
    "spec.json", "successor_spec.json", "asset_successor_spec.json", "resolved_spec.json", "bindings.json", "successor_bindings.json", "runtime_bindings.json", "source_bindings.json",
    "source_receipts.json", "reference_contract.json", "run_manifest.json", "source_manifest.json",
    "environment_manifest.json", "authorization.json", "provenance.json", "protocol.json",
    "attribution.json", "pool_freeze.json", "pool_encodings.json", "selection_freeze.json",
    "test_evaluation.json", "budget_and_timing.json", "candidate_funnel.csv", "candidate_provenance.csv",
    "pool.json", "freeze.json", "calibration_comparison.json",
}
TREE_DIRS = {"pilot", "full", "calibration", "test", "attribution_units", "generation_units", "filter_units",
             "producer_receipts", "audit", "results", "diagnostics", "source_bindings", "provenance", "manifests", "encodings"}
EXCLUDED_COMPONENTS = {"logs", "log", "tmp", "temp", "cache", "caches", "models", "checkpoints",
                       "downloads", "assets", "environment", "environments", ".git", "__pycache__"}
SUFFIXES = {".json", ".jsonl", ".csv", ".npy", ".npz", ".md", ".txt", ".tex", ".png", ".pdf", ".sh"}
CORE_FILES = {"pilot/final_receipt.json", "attribution.json", "pool_freeze.json", "pool_encodings.json",
              "selection_freeze.json", "test_evaluation.json", "audit/final_audit.json",
              "results/export_manifest.json"}
RESULT_FILES = {"prefix_metrics.csv", "figure3_coverage_cost_vs_k.csv", "parent_best_distances.csv",
                "figure4_k10_exact.csv", "figure4_k20_exact.csv", "table2_k10.csv", "table2_k20.csv"}
MAX_MEMBER_BYTES = 2 * 1024**3
MAX_TOTAL_BYTES = 16 * 1024**3
MAX_MEMBERS = 20000


class ReleaseRejected(ValueError):
    """Specific transport/audit blocker; never a scientific zero."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ReleaseRejected(reason)


def _scope(path: str | Path) -> Path:
    path = Path(path)
    _require(path.is_absolute(), "CM path must be absolute")
    _require(".." not in path.parts, "CM path traversal")
    bases = [base for base in ALLOWED_ROOTS if path != base and path.is_relative_to(base)]
    _require(bool(bases), "Path is outside fresh CM baseline subdirectories")
    _require(any(base.is_dir() for base in bases),
             "Approved CM base must already exist on its verified storage domain")
    for part in (path, *path.parents):
        _require(not part.is_symlink(), f"Symlink in CM path: {part}")
    return path


def _name(name: str) -> str:
    _require(isinstance(name, str) and name != "" and "\\" not in name and "\x00" not in name,
             "Invalid portable member name")
    path = PurePosixPath(name)
    _require(not path.is_absolute() and all(p not in {"", ".", ".."} for p in name.split("/")),
             "Unsafe portable member traversal")
    _require(str(path) == name, "Noncanonical portable member")
    return name


def _allowed(name: str) -> bool:
    p = PurePosixPath(_name(name))
    return (not set(p.parts) & EXCLUDED_COMPONENTS and p.suffix in SUFFIXES and
            not any(".tmp" in part or part.startswith(".") for part in p.parts) and
            ((len(p.parts) == 1 and name in TOP_FILES) or p.parts[0] in TREE_DIRS))


def _stat(path: Path) -> tuple[int, ...]:
    s = path.lstat()
    _require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, f"Nonregular or hardlinked source: {path}")
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan(root: Path) -> dict[str, tuple[int, ...]]:
    result = {}
    for top in sorted(root.iterdir()):
        if top.name not in TOP_FILES | TREE_DIRS:
            continue
        _require(not top.is_symlink(), f"Whitelisted source is a symlink: {top}")
        if top.is_file():
            result[top.name] = _stat(top)
            continue
        _require(top.name in TREE_DIRS and top.is_dir(), f"Unexpected source type: {top}")
        for current, directories, files in os.walk(top, followlinks=False):
            for directory in list(directories):
                child = Path(current) / directory
                _require(not child.is_symlink(), f"Source directory symlink: {child}")
                if directory in EXCLUDED_COMPONENTS or directory.startswith("."):
                    directories.remove(directory)
            for filename in files:
                child = Path(current) / filename
                rel = child.relative_to(root).as_posix()
                if _allowed(rel):
                    result[rel] = _stat(child)
    _require(len(result) <= MAX_MEMBERS, "CM release member budget exceeded")
    _require(all(s[2] <= MAX_MEMBER_BYTES for s in result.values()), "CM small package contains oversized file")
    _require(sum(s[2] for s in result.values()) <= MAX_TOTAL_BYTES, "CM small package exceeds total budget")
    return result


def _audit_gate(read_json: Callable[[str], dict], hash_for: Callable[[str], str], names: set[str]) -> dict:
    if "audit/k20_audit.json" in names:
        return _k20_audit_gate(read_json, hash_for, names)
    audit = read_json("audit/final_audit.json")
    _require(audit.get("status") == "BACE_CM_CREM_FINAL_AUDIT_PASS" and
             audit.get("scientific_pass_claimed") is True and audit.get("main_matrix_written") is False,
             "Final scientific audit is not accepted CM PASS")
    science = audit.get("science_hash")
    _require(isinstance(science, str) and re.fullmatch(r"[a-f0-9]{64}", science) is not None,
             "Missing final science identity")
    binding = audit.get("independent_spotcheck", {})
    spot_name = _name(binding.get("path", ""))
    _require(spot_name.startswith("audit/") and spot_name != "audit/final_audit.json" and spot_name in names,
             "Independent spotcheck must be a separate included audit file")
    _require(hash_for(spot_name) == binding.get("sha256"), "Independent spotcheck file hash conflict")
    spot = read_json(spot_name)
    _require(spot.get("status") == "CM_CREM_INDEPENDENT_SPOTCHECK_PASS" and spot.get("independent") is True,
             "No real independent scientific spotcheck receipt")
    _require(spot.get("science_hash") == science and
             type(spot.get("checked_records_count")) is int and spot["checked_records_count"] >= 1,
             "Independent spotcheck has no bound checked records")
    for field, length in (("implementation_commit", 40), ("implementation_sha256", 64)):
        _require(isinstance(spot.get(field), str) and re.fullmatch(r"[a-f0-9]{" + str(length) + "}", spot[field]) is not None,
                 f"Independent spotcheck lacks actual {field}")
    _require(spot.get("source_test_evaluation_sha256") == hash_for("test_evaluation.json") and
             spot.get("source_selection_freeze_sha256") == hash_for("selection_freeze.json"),
             "Independent spotcheck does not bind this test evaluation/freeze")
    required = CORE_FILES | {spot_name} | {"results/source_csv/" + name for name in RESULT_FILES}
    _require(required <= names, "Release missing required scientific records or CSV")
    spec_names = names & {"spec.json", "resolved_spec.json"}
    _require(len(spec_names) == 1, "Release requires exactly one canonical spec.json/resolved_spec.json")
    spec = read_json(next(iter(spec_names)))
    _require(spec.get("science_hash") == science and spec.get("method_id") == METHOD,
             "Spec/science identity conflict")
    _require(bool(names & {"bindings.json", "runtime_bindings.json", "source_bindings.json"}),
             "Release lacks source/runtime bindings")
    export = read_json("results/export_manifest.json")
    _require(export.get("fixture") is False and export.get("dataset") == "bace" and export.get("oracle") == "gine",
             "Synthetic or wrong-oracle export cannot publish")
    for filename, sha in export.get("source_files", {}).items():
        rel = "results/source_csv/" + _name(filename)
        _require(rel in names and hash_for(rel) == sha, "Export CSV source binding conflict")
    _require(set(export.get("source_files", {})) == RESULT_FILES, "Export CSV inventory incomplete")
    return {"science_hash": science, "independent_spotcheck": binding,
            "final_audit_sha256": hash_for("audit/final_audit.json")}


def _k20_audit_gate(read, hash_for, names):
    """Adopt the distinct saved-pool v2 audit, never relabel it as v1 science."""
    from .cm_crem_selection import SelectionFreeze
    required = {"spec.json", "pool.json", "freeze.json", "calibration_comparison.json",
                "calibration/complete.json", "test/complete.json", "test_evaluation.json",
                "audit/k20_audit.json", "results/export_manifest.json"}
    _require(required <= names, "K20 release lacks completed scientific stages")
    spec, audit = read("spec.json"), read("audit/k20_audit.json")
    science = digest({k:v for k,v in spec.items() if k not in
                     {"execution_root", "execution_commit", "output_root", "source_root", "source_spec"}})
    _require(spec.get("experiment_id") == "CM-Global-K20-v2" and
             spec.get("generation_enabled") is False and spec.get("pool_count") == 5474 and
             spec.get("test_used_for_selection") is False, "Wrong K20 scope/config")
    _require(audit.get("status") == "K20_RECORDS_AND_INDEPENDENT_PROTOTYPE_CHECKS_PASS" and
             audit.get("scope_sha256") == science and audit.get("main_matrix_written") is False and
             audit.get("independent_calibration_replay") is True and
             audit.get("original_generation_reused") is True and audit.get("new_generation_calls") == 0 and
             len(audit.get("checked_prototypes", [])) == 2 and
             len(audit.get("independent_new_ot_checks", [])) == 2,
             "K20 real independent audit incomplete")
    freeze_doc = read("freeze.json")
    freeze = SelectionFreeze.from_dict(freeze_doc["freeze"])
    _require(freeze_doc.get("test_loaded_by_this_run") is False and
             freeze.freeze_sha256 == audit.get("selection_freeze_sha") and
             set(audit["checked_prototypes"]) <= set(freeze.selected_candidate_ids), "K20 freeze/audit conflict")
    for name in required - {"spec.json", "results/export_manifest.json"}:
        _require(read(name).get("scope_sha256") == science, "K20 mixed stage scope: " + name)
    _require(read("calibration/complete.json").get("parents") == 66 and
             read("test/complete.json").get("parents") == audit.get("test_count") == 141 and
             read("test/complete.json").get("candidate_count") == len(freeze.selected_candidate_ids) and
             read("test/complete.json").get("test_reads_after_freeze") is True, "K20 cohort/prefix incomplete")
    comparison = read("calibration_comparison.json")
    _require(comparison.get("calibration_only") is True and comparison.get("test_used_for_choice") is False,
             "K20 selection/test boundary missing")
    export = read("results/export_manifest.json")
    _require(export.get("fixture") is False and export.get("dataset") == "bace" and export.get("oracle") == "gine",
             "K20 export oracle/fixture mismatch")
    _require(set(export.get("source_files", {})) == RESULT_FILES, "K20 CSV inventory incomplete")
    for name, sha in export["source_files"].items():
        path = "results/source_csv/" + _name(name)
        _require(path in names and hash_for(path) == sha, "K20 export CSV binding conflict")
    return {"science_hash": science, "final_audit_sha256": hash_for("audit/k20_audit.json"),
            "independent_spotcheck": {"path": "audit/k20_audit.json", "sha256": hash_for("audit/k20_audit.json")},
            "variant": "CM-Global-K20-v2", "source_science_hash": spec["source_science_sha"],
            "scope": "BACE_ORIGINAL_GINE_FULL_TRAIN_POOL_K20_POSTHOC"}


def _atomic_directory(source: Path, destination: Path) -> None:
    """No-replace rename, or receipt-gated exclusive publication on older FS.

    The fallback is NOT an atomic directory rename: mkdir claims an unused name,
    data is made durable first, and acceptance receipts are atomically linked last.
    Readers must require these receipts, never directory existence. No overwrite
    fallback is permitted, including an already existing empty destination.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        result = libc.renamex_np(os.fsencode(source), os.fsencode(destination), 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        result = libc.renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    else:
        raise ReleaseRejected("Atomic no-replace directory finalization unavailable")
    if result != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(str(destination))
        if code in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
            _receipt_directory(source, destination)
            return
        raise OSError(code, os.strerror(code), str(destination))
    fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _receipt_directory(source: Path, destination: Path) -> None:
    markers = [n for n in ("package_receipt.json", "cm_run_publication.json", "cm_import_receipt.json")
               if (source/n).is_file()]
    _require(markers == ["package_receipt.json"] or
             markers == ["cm_run_publication.json", "cm_import_receipt.json"],
             "Receipt-gated publication requires complete known commit markers")
    _require(not source.is_symlink() and source.parent == destination.parent,
             "Receipt publication must remain within one parent directory")
    destination.mkdir(mode=0o700)  # atomic exclusive claim, never exist_ok
    for child in sorted(source.iterdir()):
        if child.name not in markers:
            _require(not child.is_symlink(), "Publication symlink rejected")
            target = destination/child.name
            if target.exists():
                raise FileExistsError(str(target))
            os.rename(child, target)
    _sync_directories(destination)
    for name in markers:
        os.link(source/name, destination/name)  # atomic no-replace acceptance
        (source/name).unlink()
        fd = os.open(destination, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    source.rmdir()  # only the newly created, now-empty staging directory
    fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def finalize_prepared_package(root: str | Path, staging: str | Path) -> dict:
    """Finish the failed directory commit; never rebuild the prepared archive."""
    root, staging = _scope(root), _scope(staging)
    _require(staging.parent == root and re.fullmatch(r"release\.tmp-[0-9a-f]{32}", staging.name),
             "Prepared package staging identity invalid")
    _require({p.name for p in staging.iterdir()} ==
             {"cm_crem_results.tar.gz", "package_manifest.json", "package_receipt.json"},
             "Prepared package is incomplete or has unexpected members")
    outer = json.loads((staging/"package_manifest.json").read_text())
    receipt = json.loads((staging/"package_receipt.json").read_text())
    package = staging/"cm_crem_results.tar.gz"
    _require(_stat(package)[2] == outer["package_bytes"] and _sha(package) == outer["package_sha256"],
             "Prepared package bytes/hash changed")
    gate = _audit_gate(lambda n: json.loads((root/n).read_text()), lambda n: _sha(root/n), set(outer["files"]))
    _require(gate["science_hash"] == receipt["science_hash"] == outer["science_hash"] and
             gate["final_audit_sha256"] == outer["final_audit_sha256"] and
             receipt["package_sha256"] == outer["package_sha256"] and
             receipt["package_path"] == str(root/"release"/package.name),
             "Prepared package does not bind the accepted original run")
    _atomic_directory(staging, root/"release")
    return receipt


def _sync_directories(root: Path) -> None:
    for current, _, _ in os.walk(root, topdown=False):
        fd = os.open(current, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class _HashReader:
    def __init__(self, stream: Any):
        self.stream, self.hash = stream, hashlib.sha256()
        self.count = 0

    def read(self, count: int = -1) -> bytes:
        data = self.stream.read(count)
        self.hash.update(data)
        self.count += len(data)
        return data


def package_run(root: str | Path) -> dict[str, Any]:
    """Package a genuinely accepted finished CM run once into root/release."""
    root = _scope(root)
    _require(root.is_dir(), "CM run does not exist")
    final = root / "release"
    _require(not final.exists(), "Release already exists; adopt its receipt, do not overwrite")
    inventory = _scan(root)
    names = set(inventory)
    read = lambda name: json.loads((root / _name(name)).read_text())
    # Only the few small gate documents are read before streaming the package.
    cached_hashes = {}
    def file_hash(name):
        if name not in cached_hashes:
            cached_hashes[name] = _sha(root / name)
        return cached_hashes[name]
    gate = _audit_gate(read, file_hash, names)
    staging = root / ("release.tmp-" + uuid.uuid4().hex)
    staging.mkdir(mode=0o700)
    package = staging / "cm_crem_results.tar.gz"
    files = {}
    with package.open("xb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
            with tarfile.open(mode="w|", fileobj=compressed, format=tarfile.USTAR_FORMAT) as archive:
                for name in sorted(names):
                    path = root / name
                    _require(_stat(path) == inventory[name], f"Source changed before packaging: {name}")
                    info = tarfile.TarInfo(name)
                    info.size, info.mode = inventory[name][2], 0o644
                    with path.open("rb") as original:
                        hashed = _HashReader(original)
                        archive.addfile(info, hashed)
                    _require(hashed.count == info.size and _stat(path) == inventory[name],
                             f"Source changed while packaging: {name}")
                    files[name] = {"bytes": info.size, "sha256": hashed.hash.hexdigest()}
                internal = {"schema": SCHEMA, "method_id": METHOD, **gate, "files": files,
                            "member_count": len(files), "total_member_bytes": sum(x["bytes"] for x in files.values()),
                            "main_matrix_written": False}
                internal["manifest_sha256"] = digest(internal)
                payload = (json.dumps(internal, sort_keys=True, indent=2) + "\n").encode()
                info = tarfile.TarInfo("package_manifest.json")
                info.size, info.mode = len(payload), 0o644
                archive.addfile(info, io.BytesIO(payload))
        stream.flush()
        os.fsync(stream.fileno())
    _require(_scan(root) == inventory, "Scientific source tree changed during packaging")
    _audit_gate(read, lambda name: files[name]["sha256"], names)
    manifest = {**internal, "package_sha256": _sha(package), "package_bytes": package.stat().st_size,
                "package_filename": package.name, "created_at": utc_now(),
                "internal_manifest_sha256": hashlib.sha256(payload).hexdigest()}
    atomic_json(staging / "package_manifest.json", manifest, immutable=True)
    receipt = {"status": "CM_RESULT_PACKAGE_SEALED", "package_path": str(final / package.name),
               "manifest_path": str(final / "package_manifest.json"), "package_sha256": manifest["package_sha256"],
               "package_bytes": manifest["package_bytes"], "science_hash": gate["science_hash"],
               "main_matrix_written": False}
    atomic_json(staging / "package_receipt.json", receipt, immutable=True)
    _sync_directories(staging)
    _atomic_directory(staging, final)
    return receipt


def finalize_interrupted_import(package, manifest, fresh_destination, staging):
    """Resume a completed extraction after an import-code failure, not science.

    No second extraction/transfer. Because the failed call published no receipt,
    verify all existing members once before the ordinary scientific audit gate.
    """
    package,manifest,destination,staging=map(_scope,(package,manifest,fresh_destination,staging))
    _require(not destination.exists(), 'Import destination must be fresh')
    _require(staging.parent==destination.parent and
             re.fullmatch(re.escape(destination.name)+r'\.import-tmp-[a-f0-9]{32}',staging.name) is not None,
             'Not an original interrupted import staging directory')
    _require(staging.is_dir() and not staging.is_symlink(), 'Unsafe import staging')
    outer=json.loads(manifest.read_text());files=outer.get('files',{})
    _require(outer.get('schema')==SCHEMA and outer.get('method_id')==METHOD and
             outer.get('main_matrix_written') is False,'Invalid CM transport manifest')
    _require(isinstance(files,dict) and 0<len(files)<=MAX_MEMBERS,'Invalid member inventory')
    before=_stat(package)
    _require(before[2]==outer.get('package_bytes') and _sha(package)==outer.get('package_sha256'),
             'Transferred package bytes/SHA conflict')
    expected=set(files)|{'package_manifest.json'};seen=set();total=0
    for current,dirs,names in os.walk(staging,followlinks=False):
        for directory in dirs:
            _require(not (Path(current)/directory).is_symlink(),'Import directory symlink')
        for name in names:
            p=Path(current)/name;relative=p.relative_to(staging).as_posix()
            _require(relative in expected,'Unexpected extracted member')
            stat_before=_stat(p)
            sha=_sha(p)
            _require(_stat(p)==stat_before,'Extracted member changed during verification')
            if relative=='package_manifest.json':
                _require(sha==outer.get('internal_manifest_sha256'),'Internal manifest content conflict')
                internal=json.loads(p.read_text())
            else:
                entry=files[relative]
                _require(_allowed(relative) and type(entry.get('bytes')) is int and
                         0<=entry['bytes']<=MAX_MEMBER_BYTES and stat_before[2]==entry['bytes'] and
                         sha==entry.get('sha256'),'Extracted member identity conflict: '+relative)
                total+=stat_before[2]
            seen.add(relative)
    _require(seen==expected and len(files)==outer.get('member_count') and
             total==outer.get('total_member_bytes') and total<=MAX_TOTAL_BYTES,'Extracted inventory incomplete')
    _require(internal.get('manifest_sha256')==digest({k:v for k,v in internal.items() if k!='manifest_sha256'})
             and all(outer.get(k)==v for k,v in internal.items()),'Inner/outer manifest disagreement')
    gate=_audit_gate(lambda n:json.loads((staging/n).read_text()),lambda n:files[n]['sha256'],set(files))
    _require(gate['science_hash']==outer.get('science_hash') and gate['final_audit_sha256']==outer.get('final_audit_sha256'),
             'Imported scientific gate differs')
    _require(_stat(package)==before,'Package changed during recovery')
    receipt=dict(schema='cm_crem_fresh_import_receipt_v1',status='CM_RESULT_IMPORT_VERIFIED',
        source_package=str(package),package_sha256=outer['package_sha256'],package_bytes=outer['package_bytes'],
        destination=str(destination),science_hash=gate['science_hash'],verified_member_count=len(files),
        verified_member_bytes=total,full_package_hash_count=1,hash_count_scope='THIS_RECOVERY_CALL',
        completed_extraction_reused=True,independent_spotcheck=gate['independent_spotcheck'],created_at=utc_now(),
        main_matrix_written=False,registry_created=False,models_loaded=False,generation_rerun=False,distance_recomputed=False)
    atomic_json(staging/'cm_import_receipt.json',receipt,immutable=True)
    atomic_json(staging/'cm_run_publication.json',{**receipt,'status':'BACE_CM_CREM_RESULT_PUBLISHED',
        'scope':'INDEPENDENT_CM_BASELINE_NOT_ORIGINAL_MAIN_MATRIX','scientific_pass_claimed':True},immutable=True)
    _sync_directories(staging);_atomic_directory(staging,destination)
    return receipt


def verify_import(package: str | Path, manifest: str | Path,
                  fresh_destination: str | Path) -> dict[str, Any]:
    """One complete transport hash plus streamed member verification/extraction."""
    package, manifest, destination = _scope(package), _scope(manifest), _scope(fresh_destination)
    _stat(package)
    _stat(manifest)
    _require(not destination.exists(), "Import destination must be fresh")
    _require(not package.is_relative_to(destination) and not manifest.is_relative_to(destination),
             "Import cannot consume its own destination")
    outer = json.loads(manifest.read_text())
    _require(outer.get("schema") == SCHEMA and outer.get("method_id") == METHOD and
             outer.get("main_matrix_written") is False, "Invalid CM transport manifest")
    files = outer.get("files", {})
    _require(isinstance(files, dict) and 0 < len(files) <= MAX_MEMBERS, "Invalid member inventory")
    for name, entry in files.items():
        _require(_allowed(name), f"Nonwhitelisted portable member: {name}")
        _require(isinstance(entry, dict), "Malformed member identity record")
        _require(type(entry.get("bytes")) is int and 0 <= entry["bytes"] <= MAX_MEMBER_BYTES and
                 isinstance(entry.get("sha256"), str) and re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]) is not None,
                 "Malformed member byte identity")
    total = sum(entry["bytes"] for entry in files.values())
    _require(total <= MAX_TOTAL_BYTES and total == outer.get("total_member_bytes") and len(files) == outer.get("member_count"),
             "Portable total/member count conflict")
    package_before = _stat(package)
    _require(package_before[2] == outer.get("package_bytes") and _sha(package) == outer.get("package_sha256"),
             "Transferred package bytes/SHA conflict")
    destination.parent.mkdir(parents=True, exist_ok=True)
    space = os.statvfs(destination.parent)
    _require(space.f_bavail * space.f_frsize >= total + 1024 * 1024, "Insufficient import space")
    staging = destination.with_name(destination.name + ".import-tmp-" + uuid.uuid4().hex)
    staging.mkdir(mode=0o700)
    seen, internal = set(), None
    with tarfile.open(package, "r|gz") as archive:
        for member in archive:
            name = _name(member.name)
            _require(name not in seen and member.isfile() and not member.issym() and not member.islnk(),
                     "Duplicate/nonregular archive member")
            _require(not member.pax_headers and not getattr(member, "sparse", None), "Extended/sparse member rejected")
            seen.add(name)
            _require(name in files or name == "package_manifest.json", "Unlisted archive member")
            expected_size = files[name]["bytes"] if name in files else 8 * 1024 * 1024
            _require(0 <= member.size <= expected_size and (name == "package_manifest.json" or member.size == expected_size),
                     "Archive member size conflict")
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            count = 0
            source = archive.extractfile(member)
            _require(source is not None, "Unreadable regular archive member")
            with source, target.open("xb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
                    h.update(chunk)
                    count += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            _require(count == member.size, "Short archive member")
            if name == "package_manifest.json":
                _require(h.hexdigest() == outer.get("internal_manifest_sha256"), "Internal manifest content conflict")
                internal = json.loads(target.read_text())
            else:
                _require(h.hexdigest() == files[name]["sha256"], f"Member content SHA conflict: {name}")
    _require(seen == set(files) | {"package_manifest.json"}, "Missing archive members")
    _require(_stat(package) == package_before, "Package changed during import")
    _require(internal is not None and internal.get("manifest_sha256") == digest({k: v for k, v in internal.items() if k != "manifest_sha256"}),
             "Internal manifest identity conflict")
    _require(all(outer.get(k) == v for k, v in internal.items()), "Inner/outer manifest disagreement")
    gate = _audit_gate(lambda name: json.loads((staging / name).read_text()),
                       lambda name: files[name]["sha256"], set(files))
    _require(gate["science_hash"] == outer.get("science_hash") and gate["final_audit_sha256"] == outer.get("final_audit_sha256"),
             "Imported scientific gate differs from package manifest")
    receipt = {"schema": "cm_crem_fresh_import_receipt_v1", "status": "CM_RESULT_IMPORT_VERIFIED",
               "source_package": str(package), "package_sha256": outer["package_sha256"],
               "package_bytes": outer["package_bytes"], "destination": str(destination),
               "science_hash": gate["science_hash"], "verified_member_count": len(files),
               "verified_member_bytes": total, "full_package_hash_count": 1,
               "independent_spotcheck": gate["independent_spotcheck"], "created_at": utc_now(),
               "main_matrix_written": False, "registry_created": False,
               "models_loaded": False, "generation_rerun": False, "distance_recomputed": False}
    atomic_json(staging / "cm_import_receipt.json", receipt, immutable=True)
    atomic_json(staging / "cm_run_publication.json", {**receipt, "status": "BACE_CM_CREM_RESULT_PUBLISHED",
                "scope": "INDEPENDENT_CM_BASELINE_NOT_ORIGINAL_MAIN_MATRIX", "scientific_pass_claimed": True}, immutable=True)
    _sync_directories(staging)
    _atomic_directory(staging, destination)
    return receipt
