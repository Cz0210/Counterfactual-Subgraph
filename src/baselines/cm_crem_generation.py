"""Budgeted, process-isolated invocation of the pinned official CM-CReM path.

No oracle, ranking by predictions, calibration, or test input is used here.
The upstream algorithm bodies are compiled unchanged from an exact source pin;
only their import namespace / database transport is adapted. See the review.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import signal
import sqlite3
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping

UPSTREAM_COMMIT = "b5816b502cde00ee24c652a02cbc54664583f773"
UPSTREAM_SOURCE_SHA256 = "b5e485195ede009c560ee397f458794dc22272c14fb17faf3313fbdc64f39e49"
NATIVE_SYMBOLS = (
    "find_anchors", "atomDeleter", "dfs", "find_connections",
    "create_graph_dictionary", "get_project_path", "TimeoutException",
    "timeout_handler", "crem_fragment_replacement", "flat_index_to_tuple",
)


class GenerationContractError(ValueError):
    """Input/environment/provenance mismatch, never a scientific empty result."""


class InfrastructureFailure(BaseException):
    """Explicit sentinel bypassing upstream's broad ``except Exception``.

    Only OSError/sqlite3.Error are wrapped. It is caught at the isolated worker
    boundary and written as INFRASTRUCTURE_FAILED, not as an empty candidate set.
    """

    def __init__(self, original: BaseException):
        self.original = original
        super().__init__(str(original))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _infrastructure_safe_print(*args: Any, **kwargs: Any) -> None:
    """A log-write EIO must also cross upstream's broad Exception handlers."""
    try:
        print(*args, **kwargs)
    except OSError as error:
        raise InfrastructureFailure(error) from error


def atom_order_sha256(mol: Any) -> str:
    """Ordered atom/bond identity, including stereochemistry but no coordinates."""
    return _digest({
        "atoms": [[a.GetAtomicNum(), a.GetIsotope(), a.GetFormalCharge(),
                   int(a.GetChiralTag()), a.GetNumExplicitHs(), a.GetNoImplicit(),
                   a.GetIsAromatic(), a.GetAtomMapNum()] for a in mol.GetAtoms()],
        "bonds": [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType()),
                   int(b.GetStereo()), list(b.GetStereoAtoms()), int(b.GetBondDir()),
                   b.GetIsAromatic()] for b in mol.GetBonds()],
    })


def official_ring_mask(mol: Any, selected: list[int]) -> list[int]:
    """One union of rings intersecting the initial mask, not transitive closure."""
    n = mol.GetNumAtoms()
    if not selected or len(set(selected)) != len(selected):
        raise GenerationContractError("selected atom indices must be nonempty and unique")
    if any(type(i) is not int or i < 0 or i >= n for i in selected):
        raise GenerationContractError("selected atom index out of range")
    initial = set(selected)
    result = set(initial)
    for ring in mol.GetRingInfo().AtomRings():
        if initial.intersection(ring):
            result.update(ring)
    return sorted(result)


def make_parent_request(parent_id: str, mol: Any, selected_atom_indices: list[int],
                        split: str = "train") -> dict[str, Any]:
    from rdkit import Chem
    if split != "train":
        raise GenerationContractError("generation is train-only")
    if mol is None or mol.GetNumAtoms() < 1 or mol.GetNumAtoms() != mol.GetNumHeavyAtoms():
        raise GenerationContractError("CM input must contain non-hydrogen atoms only")
    Chem.SanitizeMol(mol)
    block = Chem.MolToMolBlock(mol, kekulize=False, forceV3000=True)
    request = {
        "schema": "cm_crem_parent_v2", "parent_id": str(parent_id), "split": split,
        "smiles": Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True),
        "molblock": block, "atom_order_sha256": atom_order_sha256(mol),
        "atom_no_implicit": [a.GetNoImplicit() for a in mol.GetAtoms()],
        "atom_explicit_hs": [a.GetNumExplicitHs() for a in mol.GetAtoms()],
        "bond_directions": [int(b.GetBondDir()) for b in mol.GetBonds()],
        "bond_endpoints": [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()],
        "selected_atom_indices": list(selected_atom_indices),
        "effective_atom_indices": official_ring_mask(mol, list(selected_atom_indices)),
    }
    load_parent_mol(request)
    return request


def load_parent_mol(request: Mapping[str, Any]) -> Any:
    """Restore explicitly recorded representational flags omitted by MolBlock.

    V3000 preserves atom indices but can reverse bonds for wedge serialization,
    and does not preserve every SMILES H / slash flag. Restore the explicitly
    bound original orientation (only after checking the same undirected bond)
    and flags, then require the entire original ordered digest
    AND canonical isomeric identity. No guessed atom reindexing is permitted.
    """
    from rdkit import Chem
    if request.get("schema") not in {"cm_crem_parent_v1", "cm_crem_parent_v2"}:
        raise GenerationContractError("unsupported CM parent transport schema")
    # Restore bracket-H state before sanitization: aromatic [nH] is not safely
    # representable by an aromatic V3000 block alone on every supported RDKit.
    mol = Chem.MolFromMolBlock(request["molblock"], sanitize=False, removeHs=False)
    if mol is None:
        raise GenerationContractError("invalid CM MolBlock")
    if request.get("schema") == "cm_crem_parent_v2":
        mol = _restore_ordered_bonds(mol, request.get("bond_endpoints", []))
    atoms = request.get("atom_no_implicit", [])
    hydrogens = request.get("atom_explicit_hs", [])
    bonds = request.get("bond_directions", [])
    if len(atoms) != mol.GetNumAtoms() or len(hydrogens) != len(atoms) or len(bonds) != mol.GetNumBonds():
        raise GenerationContractError("incomplete atom/bond transport flags")
    for atom, flag, hydrogen_count in zip(mol.GetAtoms(), atoms, hydrogens):
        if type(flag) is not bool or type(hydrogen_count) is not int or not 0 <= hydrogen_count <= 8:
            raise GenerationContractError("invalid noImplicit flag")
        atom.SetNoImplicit(flag)
        atom.SetNumExplicitHs(hydrogen_count)
    for bond, direction in zip(mol.GetBonds(), bonds):
        bond.SetBondDir(Chem.BondDir(int(direction)))
    mol.UpdatePropertyCache(strict=True)
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    original = Chem.MolFromSmiles(request["smiles"])
    if (atom_order_sha256(mol) != request["atom_order_sha256"] or original is None
            or Chem.MolToSmiles(mol, isomericSmiles=True) != Chem.MolToSmiles(original, isomericSmiles=True)):
        raise GenerationContractError("CM atom mapping/ordered graph mismatch")
    return mol


def _restore_ordered_bonds(mol: Any, endpoints: list[list[int]]) -> Any:
    """Undo V3000's wedge-oriented endpoints without changing chemical edges.

    Bond indices and atom indices are not remapped. Direction-sensitive dative
    bonds are never reversed by this adapter. Atom metadata is copied from the
    parsed block, not reconstructed from a guessed canonical atom mapping.
    """
    from rdkit import Chem
    if len(endpoints) != mol.GetNumBonds():
        raise GenerationContractError("incomplete ordered bond transport")
    for bond, pair in zip(mol.GetBonds(), endpoints):
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(x) is not int or not 0 <= x < mol.GetNumAtoms() for x in pair)
                or pair[0] == pair[1]
                or set(pair) != {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}):
            raise GenerationContractError("CM bond transport mapping mismatch")
        actual = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        if pair != actual and str(bond.GetBondType()) not in {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "ZERO"}:
            raise GenerationContractError("unsupported directional bond transport reversal")
    restored = Chem.RWMol()
    for atom in mol.GetAtoms():
        restored.AddAtom(Chem.Atom(atom))
    for bond, (begin, end) in zip(mol.GetBonds(), endpoints):
        restored.AddBond(begin, end, bond.GetBondType())
        target = restored.GetBondWithIdx(bond.GetIdx())
        target.SetIsAromatic(bond.GetIsAromatic())
        target.SetIsConjugated(bond.GetIsConjugated())
    # Stereo neighbors can only be assigned after every chemical edge exists.
    for bond, pair in zip(mol.GetBonds(), endpoints):
        target = restored.GetBondWithIdx(bond.GetIdx())
        stereo_atoms = list(bond.GetStereoAtoms())
        if stereo_atoms:
            if pair != [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]:
                stereo_atoms.reverse()
            target.SetStereoAtoms(*stereo_atoms)
        target.SetStereo(bond.GetStereo())
    for conformer in mol.GetConformers():
        restored.AddConformer(Chem.Conformer(conformer), assignId=True)
    return restored.GetMol()


def parent_seed(science_hash: str, parent_id: str, campaign_seed: int = 7) -> int:
    if len(science_hash) != 64 or any(c not in "0123456789abcdef" for c in science_hash):
        raise GenerationContractError("science_hash must be a resolved lowercase SHA256")
    return int(_digest([science_hash, str(parent_id), campaign_seed])[:16], 16)


def _load_native(source_path: Path, mutate_proxy: Callable[..., Any]) -> tuple[dict, dict]:
    """Compile the real reviewed functions, excluding all unrelated module imports."""
    from functools import reduce
    from itertools import product
    from rdkit import Chem
    raw = source_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != UPSTREAM_SOURCE_SHA256:
        raise GenerationContractError("official linksGenerator.py source pin mismatch")
    tree = ast.parse(raw, filename=str(source_path))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in NATIVE_SYMBOLS]
    if {node.name for node in nodes} != set(NATIVE_SYMBOLS):
        raise GenerationContractError("official CReM dependency closure incomplete")
    selected = ast.Module(body=nodes, type_ignores=[])
    namespace = {"__file__": str(source_path), "os": os, "signal": signal,
                 "random": random, "prod": math.prod, "product": product,
                 "reduce": reduce, "Chem": Chem, "mutate_mol": mutate_proxy,
                 "print": _infrastructure_safe_print}
    exec(compile(selected, str(source_path), "exec"), namespace)
    return namespace, {
        "upstream_commit": UPSTREAM_COMMIT,
        "original_source_sha256": UPSTREAM_SOURCE_SHA256,
        "executed_ast_sha256": hashlib.sha256(ast.dump(selected, include_attributes=False).encode()).hexdigest(),
        "executed_source_sha256": hashlib.sha256((ast.unparse(selected) + "\n").encode()).hexdigest(),
        "executed_source_representation": "ast.unparse of unchanged selected function/class bodies",
        "native_symbols": list(NATIVE_SYMBOLS), "native_function_bodies_changed": False,
        "adaptations": ["CREM_ONLY_IMPORT_NAMESPACE", "EXPLICIT_READONLY_DATABASE",
                        "INFRASTRUCTURE_EXCEPTION_PROPAGATION", "PARENT_WALL_BUDGET",
                        "HASH_TRUNCATION_BEFORE_ORACLE"],
    }


def _readonly_crem(database_path: Path, counters: dict) -> Callable[..., Any]:
    """Replace only crem.crem's sqlite namespace; never monkeypatch shared sqlite3."""
    import crem.crem as native
    original_connect = sqlite3.connect
    database_path = database_path.resolve(strict=True)
    if not database_path.is_file():
        raise GenerationContractError("database is not a regular file")
    def connect(path: str, *args: Any, **kwargs: Any) -> Any:
        if Path(path).resolve(strict=True) != database_path:
            raise GenerationContractError("CReM attempted to open an unbound database")
        if args or kwargs:
            raise GenerationContractError("unexpected CReM SQLite connect arguments")
        connection = original_connect(database_path.as_uri() + "?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        def trace(statement: str) -> None:
            if statement.lstrip().upper().startswith("SELECT"):
                counters["database_select_queries"] += 1
        connection.set_trace_callback(trace)
        return connection
    native.sqlite3 = SimpleNamespace(connect=connect)
    return native.mutate_mol


def _verify_environment() -> dict[str, str]:
    versions = {name: importlib.metadata.version(name) for name in ("crem", "rdkit", "numpy")}
    versions["python"] = ".".join(map(str, sys.version_info[:3]))
    expected = {"crem": "0.2.14", "rdkit": "2023.9.6", "numpy": "1.26.4", "python": "3.11.5"}
    if versions != expected:
        raise GenerationContractError(f"isolated generation environment differs: {versions}")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise GenerationContractError("start generation interpreter with PYTHONHASHSEED=0")
    if sys.flags.ignore_environment or sys.flags.hash_randomization:
        raise GenerationContractError("actual interpreter ignored PYTHONHASHSEED=0; generation must use -s -B, not -I/-E")
    return versions


def retain_raw_outputs(outputs: list[str], seed: int, limit: int = 128) -> tuple[list[dict], dict]:
    """Hash raw strings, before parsing/oracle/distance; exact-string duplicates only."""
    if limit != 128 or any(not isinstance(s, str) for s in outputs):
        raise GenerationContractError("invalid native output or raw budget")
    first: dict[str, int] = {}
    for i, smi in enumerate(outputs):
        first.setdefault(smi, i)
    ordered = sorted(first, key=lambda smi: (_digest([seed, smi]), smi))
    rows = [{"raw_id": _digest([seed, smi]), "smiles": smi,
             "source_return_index": first[smi]} for smi in ordered[:limit]]
    return rows, {"native_return_count": len(outputs), "raw_exact_duplicate_count": len(outputs)-len(first),
                  "raw_unique_count": len(first), "retained_raw_count": len(rows),
                  "raw_truncated_count": max(0, len(first)-limit)}


def run_native_parent(request: Mapping[str, Any], *, source_path: Path, database_path: Path,
                      science_hash: str, mutate_function: Callable[..., Any],
                      counters: dict | None = None) -> dict[str, Any]:
    """One native call. Caller must use isolated worker; injectable CReM is for fixtures."""
    from rdkit import Chem
    import numpy as np
    if request.get("split") != "train" or request.get("schema") not in {"cm_crem_parent_v1", "cm_crem_parent_v2"}:
        raise GenerationContractError("only explicit train parent records may generate")
    mol = load_parent_mol(request)
    selected = list(request["selected_atom_indices"])
    effective = official_ring_mask(mol, selected)
    if effective != list(request["effective_atom_indices"]):
        raise GenerationContractError("effective ring mask changed across environments")
    seed = parent_seed(science_hash, str(request["parent_id"]))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    counters = counters if counters is not None else {"database_select_queries": 0}
    queries: list[dict[str, Any]] = []
    def mutation(molecule: Any, **kwargs: Any) -> Any:
        kwargs["db_name"] = str(database_path)
        if kwargs.get("radius") != 1 or kwargs.get("max_replacements") != 64:
            raise GenerationContractError("native component budget changed")
        row = {"query_index": len(queries), "replace_ids": list(kwargs["replace_ids"]),
               "min_inc": kwargs["min_inc"], "max_inc": kwargs["max_inc"],
               "symmetry_fixes": kwargs["symmetry_fixes"], "max_replacements": 64, "radius": 1,
               "returned_count": 0, "status": "RUNNING"}
        queries.append(row)
        query_started = time.monotonic()
        _infrastructure_safe_print("CM_QUERY " + json.dumps(row, sort_keys=True), flush=True)
        try:
            for item in mutate_function(molecule, **kwargs):
                row["returned_count"] += 1
                yield item
            row["status"] = "COMPLETE"
        except (OSError, sqlite3.Error) as error:
            row.update(status="INFRASTRUCTURE_FAILED", error_type=type(error).__name__, error=str(error))
            raise InfrastructureFailure(error) from error
        except Exception as error:
            row.update(status="UPSTREAM_INNER_TIMEOUT" if type(error).__name__ == "TimeoutException"
                       else "ENGINEERING_FAILED", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            row["elapsed_seconds"] = time.monotonic()-query_started
            _infrastructure_safe_print("CM_QUERY " + json.dumps(row, sort_keys=True), flush=True)
    native, provenance = _load_native(source_path, mutation)
    graph = native["create_graph_dictionary"](mol)
    components = native["find_connections"](set(effective), graph)
    anchors = [native["find_anchors"](mol, component, diffLinker=False, original_indices=True)
               for component in components]
    mask = {"initial": selected, "effective": effective, "components": components,
            "attachment_indices": anchors, "attachment_count": sum(map(len, anchors)),
            "atom_order_sha256": request["atom_order_sha256"]}
    base = {"schema": "cm_crem_generation_parent_v1", "parent_id": str(request["parent_id"]),
            "split": "train", "seed": seed, "science_hash": science_hash, "mask": mask,
            "native": provenance, "query_receipts": queries, "counters": counters,
            "top_level_calls": 0,
            "budget": {"radius": 1, "min_max_inc": 3, "max_replacements_per_component": 64,
                       "component_combinations_cap": 500, "retained_raw_limit": 128,
                       "parent_wall_limit_seconds": 900,
                       "other_mutate_defaults": "pinned crem0.2.14; no override"}}
    if len(effective) == mol.GetNumAtoms():
        return {**base, "status": "NO_REPLACEABLE_CONTEXT", "retained_raw": [],
                "counts": {"native_return_count": 0, "retained_raw_count": 0}}
    base["top_level_calls"] = 1
    try:
        outputs = native["crem_fragment_replacement"](
            mol, set(effective), radius=1, min_max_inc=3, max_replacements=64,
            return_new_atom_indices=False)
    except InfrastructureFailure as error:
        error.generation_context = base
        raise
    if any(row["status"] == "ENGINEERING_FAILED" for row in queries):
        raise GenerationContractError(f"upstream suppressed a CReM exception: {queries[-1]}")
    retained, counts = retain_raw_outputs(outputs, seed)
    return {**base, "status": "GENERATED" if outputs else "NO_NATIVE_REPLACEMENT",
            "retained_raw": retained, "counts": counts}


def _error_record(error: BaseException) -> dict[str, Any]:
    original = error.original if isinstance(error, InfrastructureFailure) else error
    return {**getattr(error, "generation_context", {}),
            "status": "INFRASTRUCTURE_FAILED" if isinstance(error, (InfrastructureFailure, OSError, sqlite3.Error))
            else "ENGINEERING_FAILED", "error_type": type(original).__name__, "error": str(original),
            "errno": getattr(original, "errno", None),
            "sqlite_errorcode": getattr(original, "sqlite_errorcode", None), "retained_raw": []}


def _worker(request: dict, config: dict, connection: Any, log_path: str) -> None:
    """Top-level spawn target; parent timeout cannot be overwritten by native alarm."""
    os.setsid()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = time.monotonic()
    try:
        import contextlib
        with open(log_path, "x", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            environment = _verify_environment()
            counters = {"database_select_queries": 0}
            db = Path(config["database_path"])
            before = db.stat()
            function = _readonly_crem(db, counters)
            result = run_native_parent(request, source_path=Path(config["upstream_root"]) / "source/linksGenerator.py",
                                       database_path=db, science_hash=config["science_hash"],
                                       mutate_function=function, counters=counters)
            after = db.stat()
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise GenerationContractError("static CReM database changed during worker")
            result["environment"] = environment
    except (Exception, InfrastructureFailure) as error:
        result = _error_record(error)
    result.update(parent_id=request.get("parent_id"), elapsed_seconds=time.monotonic()-started,
                  worker_pid=os.getpid(), worker_process_group=os.getpgrp())
    connection.send(result)
    connection.close()


def generate_parent(request: Mapping[str, Any], config: Mapping[str, Any], *, log_path: str | Path) -> dict:
    """One <=900s parent. Only this task's confirmed subprocess group may stop."""
    if config.get("parent_wall_limit_seconds", 900) != 900:
        raise GenerationContractError("parent wall budget must remain 900 seconds")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise GenerationContractError("PYTHONHASHSEED=0 must be set before interpreter startup")
    if sys.flags.ignore_environment:
        raise GenerationContractError("spawn inherits -I/-E; launch generation with -s -B and fixed PYTHONHASHSEED=0")
    ctx = multiprocessing.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_worker, args=(dict(request), dict(config), send, str(log_path)))
    started = time.monotonic()
    process.start()
    send.close()
    result = _supervise_parent(process, receive, request, started, 900)
    result["generation_log"] = str(Path(log_path).absolute())
    if result["status"] == "TIMEOUT_BUDGETED":
        # The log's flushed start/end records remain forensic evidence; do not
        # invent complete counts for the query interrupted by parent timeout.
        result["query_counts_complete"] = False
    return result


def _supervise_parent(process: Any, receive: Any, request: Mapping[str, Any],
                       started: float, limit_seconds: float) -> dict:
    """Private supervision kernel; production caller always supplies 900."""
    try:
        if receive.poll(max(0.0, limit_seconds-(time.monotonic()-started))):
            try:
                result = receive.recv()
            except EOFError:
                result = {"status": "ENGINEERING_FAILED", "error": "isolated worker exited without result"}
            process.join(timeout=5)
            if process.is_alive():
                raise GenerationContractError("worker sent a result but did not release its process")
            return {**result, "worker_exitcode": process.exitcode,
                    "parent_wall_seconds": time.monotonic()-started}
        if process.is_alive():
            try:
                pgid = os.getpgid(process.pid)
                if pgid != process.pid:
                    raise GenerationContractError("timeout worker process group identity did not bind")
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            process.join(timeout=10)
        if process.is_alive():
            raise GenerationContractError("timed-out worker has not exited; no new parent may dispatch")
        return {"schema": "cm_crem_generation_parent_v1", "parent_id": request["parent_id"],
                "status": "TIMEOUT_BUDGETED", "retained_raw": [], "partial_adopted": False,
                "worker_pid": process.pid, "worker_exitcode": process.exitcode,
                "parent_wall_seconds": time.monotonic()-started}
    finally:
        receive.close()
