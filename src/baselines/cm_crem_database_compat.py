"""Read-only compatibility probe for the fixed static CReM fragment database.

Run in a dedicated computation-node asset process; its outer job owns timeout.
This is a public-fixture asset check, not CM generation, a pilot parent, oracle
evaluation or permission to change the experiment's generation budget.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from pathlib import Path
import random
import sqlite3
import stat
import sys
import time
from types import SimpleNamespace
from typing import Any

CREM_SOURCE_SHA256 = "e47fb661e318378e370e12507370f2576c28e7e6ebcf83dd7355a75763568d89"
REQUIRED_COLUMNS = ("env", "freq", "core_num_atoms", "core_smi", "core_sma")
PUBLIC_FIXTURE_SMILES = "CCO"
PUBLIC_FIXTURE_SETTINGS = dict(radius=1, min_inc=0, max_inc=0,
                               max_replacements=4, replace_ids=[0], ncores=1,
                               symmetry_fixes=True)


class DatabaseCompatibilityError(RuntimeError):
    def __init__(self, stage: str, message: str, receipt: dict[str, Any]):
        self.stage, self.receipt = stage, receipt
        super().__init__(message)


def _stat_binding(path: Path) -> dict[str, int]:
    s = path.stat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError("static database is not a regular file")
    return {"device": s.st_dev, "inode": s.st_ino, "bytes": s.st_size,
            "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns}


def _sidecars(path: Path) -> list[str]:
    return [str(Path(str(path) + suffix)) for suffix in ("-wal", "-shm", "-journal")
            if Path(str(path) + suffix).exists()]


def _load_pinned_crem() -> tuple[Any, dict[str, str]]:
    import crem.crem as native
    from rdkit import rdBase
    versions = {name: importlib.metadata.version(name) for name in ("crem", "rdkit", "numpy")}
    versions["python"] = ".".join(map(str, sys.version_info[:3]))
    expected = {"crem": "0.2.14", "rdkit": "2023.9.6", "numpy": "1.26.4", "python": "3.11.5"}
    if versions != expected or rdBase.rdkitVersion != "2023.09.6":
        raise ValueError(f"pinned generation environment mismatch: {versions}, RDKit={rdBase.rdkitVersion}")
    source = Path(native.__file__)
    if hashlib.sha256(source.read_bytes()).hexdigest() != CREM_SOURCE_SHA256:
        raise ValueError("installed crem/crem.py differs from the reviewed0.2.14 wheel")
    parameters = inspect.signature(native.mutate_mol).parameters
    if not {"mol", "db_name", *PUBLIC_FIXTURE_SETTINGS}.issubset(parameters):
        raise ValueError("actual mutate_mol signature is incompatible")
    return native, versions


def verify_static_database_compatibility(database: Path) -> dict[str, Any]:
    """Prove actual radius1 SELECT and CReM replacement using a read-only DB.

    No complete database hash is repeated here: the asset copier must already
    bind the static file to its verified content digest. This function binds
    the same inode/size/mtime/ctime before and after reads, rejects any existing
    journal/WAL, opens immutable read-only SQLite connections and restores the
    CReM-only connection namespace and Python RNG state in all outcomes.
    Raises DatabaseCompatibilityError with stage + serializable receipt.
    """
    started = time.monotonic()
    receipt: dict[str, Any] = {
        "schema": "cm_crem_static_database_compatibility_v1", "status": "VERIFYING",
        "scope": "STATIC_ASSET_PUBLIC_FIXTURE_NOT_CM_OR_PILOT_OR_ORACLE",
        "radius": 1, "full_database_hash_recomputed": False,
        "connection_mode": "mode=ro&immutable=1", "query_only": True,
        "journal_mode_changed": False, "durability_changed": False,
        "fixture_smiles": PUBLIC_FIXTURE_SMILES,
        "fixture_settings": {**PUBLIC_FIXTURE_SETTINGS, "replace_ids": [0]},
        "actual_select_count": 0, "fixture_select_count": 0,
        "actual_connection_count": 0, "fixture_mutate_calls": 0,
    }
    stage = "PATH_AND_HEADER"
    path: Path | None = None
    before: dict[str, int] | None = None
    native = None
    original_namespace = None
    connections: list[sqlite3.Connection] = []
    rng = random.getstate()
    failure: DatabaseCompatibilityError | None = None

    def reject(which: str, message: str) -> None:
        raise DatabaseCompatibilityError(which, message, receipt)

    try:
        path = Path(database).resolve(strict=True)
        before = _stat_binding(path)
        receipt.update(database=str(path), stat_before=before)
        receipt["sidecars_before"] = _sidecars(path)
        if receipt["sidecars_before"]:
            reject("ACTIVE_OR_UNSEALED_SQLITE", "static asset has WAL/journal/shm; refuse immutable open")
        with path.open("rb") as stream:
            if stream.read(16) != b"SQLite format 3\x00":
                reject(stage, "asset is not an SQLite3 database")

        stage = "PINNED_RUNTIME"
        native, versions = _load_pinned_crem()
        receipt.update(versions=versions, crem_source_sha256=CREM_SOURCE_SHA256)
        original_namespace = native.sqlite3
        fixture_phase = False

        def trace(statement: str) -> None:
            if statement.lstrip().upper().startswith("SELECT"):
                receipt["actual_select_count"] += 1
                if fixture_phase:
                    receipt["fixture_select_count"] += 1

        def connect(db_name: str, *args: Any, **kwargs: Any) -> sqlite3.Connection:
            if args or kwargs or Path(db_name).resolve(strict=True) != path:
                reject("UNBOUND_DATABASE_ACCESS", "native CReM attempted an unbound SQLite connection")
            connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
            connections.append(connection)
            receipt["actual_connection_count"] += 1
            connection.execute("PRAGMA query_only=ON")
            if connection.execute("PRAGMA query_only").fetchone() != (1,):
                reject("READ_ONLY_BINDING", "SQLite did not report query_only=1")
            connection.set_trace_callback(trace)
            return connection

        # Only this imported CReM module is adapted; sqlite3.connect is untouched.
        native.sqlite3 = SimpleNamespace(connect=connect)
        connection = connect(str(path))
        stage = "RADIUS1_SCHEMA"
        row = connection.execute("SELECT type FROM sqlite_master WHERE name='radius1'").fetchone()
        if row != ("table",):
            reject(stage, "required radius1 table missing or not a table")
        columns = connection.execute('PRAGMA table_info("radius1")').fetchall()
        receipt["columns"] = [{"name": r[1], "declared_type": r[2]} for r in columns]
        missing = sorted(set(REQUIRED_COLUMNS) - {r[1] for r in columns})
        if missing:
            reject(stage, f"required radius1 columns missing: {missing}")
        stage = "RADIUS1_REAL_SELECT"
        sample = connection.execute("SELECT rowid, env, freq, core_num_atoms, core_smi, core_sma FROM radius1 LIMIT 1").fetchone()
        if sample is None:
            reject(stage, "radius1 is empty; cannot certify query and replacement compatibility")
        if (not isinstance(sample[0], int) or not isinstance(sample[2], int)
                or not isinstance(sample[3], int) or min(sample[2], sample[3]) < 0
                or not all(isinstance(sample[i], str) and sample[i] for i in (1, 4, 5))):
            reject(stage, "actual radius1 row violates the native fragment field contract")
        receipt["first_row_fields_valid"] = True
        receipt["radius1_rowid_supported"] = True

        from rdkit import Chem
        stage = "PUBLIC_FIXTURE_ACTUAL_REPLACEMENT"
        fixture = Chem.MolFromSmiles(PUBLIC_FIXTURE_SMILES)
        fixture_phase = True
        random.seed(7)
        receipt["fixture_mutate_calls"] = 1
        products: list[str] = []
        iterator = native.mutate_mol(fixture, db_name=str(path), **PUBLIC_FIXTURE_SETTINGS)
        try:
            for value in iterator:
                if not isinstance(value, str):
                    reject(stage, "native mutate_mol yielded an unexpected type")
                mol = Chem.MolFromSmiles(value)
                if (mol is None or len(Chem.GetMolFrags(mol)) != 1
                        or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms())):
                    reject(stage, "native public-fixture replacement is not a valid connected complete molecule")
                canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                if canonical == Chem.MolToSmiles(fixture) or canonical in products:
                    reject(stage, "native public-fixture replacement is unchanged or duplicated")
                products.append(canonical)
                if len(products) > PUBLIC_FIXTURE_SETTINGS["max_replacements"]:
                    reject(stage, "native mutation exceeded the asset-fixture replacement bound")
        finally:
            if hasattr(iterator, "close"):
                iterator.close()
        receipt["public_fixture_products"] = products
        receipt["public_fixture_product_count"] = len(products)
        if receipt["fixture_select_count"] < 1:
            reject(stage, "native public fixture performed no actual database SELECT")
        if not products:
            reject(stage, "native query ran but fixed public fixture produced no replacement; compatibility unproven")
    except DatabaseCompatibilityError as error:
        failure = error
    except (OSError, sqlite3.Error) as error:
        receipt["infrastructure_error"] = {"class": type(error).__name__, "errno": getattr(error, "errno", None), "message": str(error)}
        failure = DatabaseCompatibilityError("INFRASTRUCTURE_FAILURE", f"{stage}: {error}", receipt)
    except Exception as error:
        failure = DatabaseCompatibilityError(stage, f"{type(error).__name__}: {error}", receipt)
    finally:
        for connection in connections:
            try:
                connection.close()
            except sqlite3.Error as error:
                failure = DatabaseCompatibilityError("INFRASTRUCTURE_FAILURE", f"connection close: {error}", receipt)
        if native is not None and original_namespace is not None:
            native.sqlite3 = original_namespace
        random.setstate(rng)
        if path is not None and before is not None:
            try:
                receipt["stat_after"] = _stat_binding(path)
                receipt["sidecars_after"] = _sidecars(path)
                if receipt["stat_after"] != before or receipt["sidecars_after"] != receipt.get("sidecars_before", []):
                    failure = DatabaseCompatibilityError("STATIC_SOURCE_CHANGED", "database identity/stat or sidecars changed during read-only check", receipt)
            except (OSError, ValueError) as error:
                failure = DatabaseCompatibilityError("INFRASTRUCTURE_FAILURE", f"post-read binding failed: {error}", receipt)
        receipt["elapsed_seconds"] = time.monotonic() - started
    if failure is not None:
        receipt.update(status="FAILED", failure_stage=failure.stage, error=str(failure))
        raise failure
    receipt.update(status="STATIC_DATABASE_COMPATIBILITY_PASS", unchanged_static_source=True,
                   experiment_generation_performed=False, oracle_calls=0)
    return receipt
