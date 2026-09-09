"""Native CM algorithm + real RDKit fixtures; mocked fragment DB, not science PASS."""
from __future__ import annotations
import errno
import hashlib
import importlib.util
import multiprocessing
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
from types import ModuleType

import pytest
from rdkit import Chem

from src.baselines import cm_crem_generation as cm

ROOT = Path(__file__).resolve().parents[1]
SCIENCE = "a" * 64


@pytest.fixture
def official_source(tmp_path):
    # Vendored review snapshot adds one terminal LF; recover the exact original
    # bytes, whose upstream last line has no LF, and verify the declared pin.
    raw = (ROOT / "patches/cm_crem/upstream_linksGenerator.py.txt").read_bytes().removesuffix(b"\n")
    assert hashlib.sha256(raw).hexdigest() == cm.UPSTREAM_SOURCE_SHA256
    path = tmp_path / "linksGenerator.py"
    path.write_bytes(raw)
    return path


def invoke(source, smi="CCOCC", selected=(0,), mutation=lambda *a, **k: iter(())):
    return cm.run_native_parent(cm.make_parent_request("p", Chem.MolFromSmiles(smi), list(selected)),
                               source_path=source, database_path=Path("/not-opened-fixture.db"),
                               science_hash=SCIENCE, mutate_function=mutation)


@pytest.mark.parametrize("smiles", ["CCO", "CC1CCCCC1", "N[C@@H](C)C(=O)O",
                                    "F/C=C/F", "[13CH3]CO", "C[n+]1ccccc1", "Cc1cc[nH]c1"])
def test_transport_keeps_atom_bond_stereo_and_order(smiles):
    mol = Chem.MolFromSmiles(smiles)
    request = cm.make_parent_request("p", mol, [1])
    assert cm.atom_order_sha256(cm.load_parent_mol(request)) == cm.atom_order_sha256(mol)


def test_wrong_atom_order_rejected():
    mol = Chem.MolFromSmiles("CCON")
    request = cm.make_parent_request("p", mol, [1])
    request["molblock"] = Chem.MolToMolBlock(Chem.RenumberAtoms(mol, [3, 2, 1, 0]),
                                          kekulize=False, forceV3000=True)
    with pytest.raises(cm.GenerationContractError, match="mapping"):
        cm.load_parent_mol(request)


def test_ring_is_one_union_not_transitive():
    mol = Chem.MolFromSmiles("CC1CCC2CCCCC2C1")
    rings = mol.GetRingInfo().AtomRings()
    unique = next(i for i in rings[0] if i not in rings[1])
    expected = set(rings[0])
    assert set(cm.official_ring_mask(mol, [unique])) == expected
    assert expected != set(rings[0]).union(rings[1])


def test_no_context_does_not_query_database(official_source):
    def never(*args, **kwargs):
        raise AssertionError("must not query")
    result = invoke(official_source, "c1ccccc1", (0,), never)
    assert result["status"] == "NO_REPLACEABLE_CONTEXT"
    assert result["top_level_calls"] == 0
    assert result["retained_raw"] == []


def test_single_atom_source_is_no_context_not_input_error(official_source):
    result = invoke(official_source, "C", (0,))
    assert result["status"] == "NO_REPLACEABLE_CONTEXT" and result["top_level_calls"] == 0


def test_native_fallback_sequence_and_fixed_component_budget(official_source):
    calls = []
    def mutation(mol, **kwargs):
        calls.append(kwargs)
        if kwargs["min_inc"] == 3 and not kwargs["symmetry_fixes"]:
            return iter(["NCOCC"])
        return iter(())
    result = invoke(official_source, mutation=mutation)
    assert [(c["min_inc"], c["symmetry_fixes"]) for c in calls] == [
        (i, symmetry) for i in range(4) for symmetry in (True, False)]
    assert all(c["radius"] == 1 and c["max_replacements"] == 64 for c in calls)
    assert all(c["db_name"] == "/not-opened-fixture.db" for c in calls)
    assert result["top_level_calls"] == 1
    assert result["counts"]["retained_raw_count"] == 1


def test_official_components_with_same_anchors_merge(official_source):
    result = invoke(official_source, "CCC", (0, 2), lambda *a, **kw: iter(["NCN"]))
    assert result["query_receipts"][0]["replace_ids"] == [0, 2]
    assert len(result["query_receipts"]) == 1


def test_official_multi_component_attachment(official_source):
    def mutation(mol, **kwargs):
        return iter(["NCOCC"] if kwargs["replace_ids"] == [0] else ["CCOCN"])
    result = invoke(official_source, "CCOCC", (0, 4), mutation)
    assert [row["smiles"] for row in result["retained_raw"]] == ["NCOCN"]
    assert result["mask"]["attachment_count"] == 2


def test_official_multi_component_combinations_preserve_native_500_cap(official_source):
    def mutation(mol, **kwargs):
        return iter((["NCOCC"] if kwargs["replace_ids"] == [0] else ["CCOCN"]) * 64)
    result = invoke(official_source, "CCOCC", (0, 4), mutation)
    assert result["counts"]["native_return_count"] == 500
    assert result["counts"]["retained_raw_count"] == 1
    assert result["counts"]["raw_exact_duplicate_count"] == 499


@pytest.mark.parametrize("error", [OSError(errno.EIO, "disk I/O"), sqlite3.OperationalError("disk I/O error")])
def test_io_failure_is_not_empty_scientific_result(official_source, error):
    def broken(*args, **kwargs):
        raise error
        yield "unreachable"
    with pytest.raises(cm.InfrastructureFailure) as caught:
        invoke(official_source, mutation=broken)
    record = cm._error_record(caught.value)
    assert record["status"] == "INFRASTRUCTURE_FAILED"
    assert record["query_receipts"][0]["status"] == "INFRASTRUCTURE_FAILED"
    assert record["retained_raw"] == []


def test_unexpected_native_error_not_silently_empty(official_source):
    def broken(*args, **kwargs):
        raise TypeError("API drift")
        yield "unreachable"
    with pytest.raises(cm.GenerationContractError, match="suppressed"):
        invoke(official_source, mutation=broken)


def test_even_one_transient_log_eio_cannot_be_swallowed(official_source, monkeypatch):
    import builtins
    actual_print = builtins.print
    attempts = []
    def failing_once(*args, **kwargs):
        attempts.append(args)
        if len(attempts) == 2:
            raise OSError(errno.EIO, "one failed native print")
        return actual_print(*args, **kwargs)
    monkeypatch.setattr(builtins, "print", failing_once)
    with pytest.raises(cm.InfrastructureFailure):
        invoke(official_source)


def test_source_function_body_pin_rejects_edits(official_source):
    official_source.write_text(official_source.read_text().replace("<= 500", "<= 501"))
    with pytest.raises(cm.GenerationContractError, match="source pin"):
        invoke(official_source)


def test_raw_budget_hashes_before_oracle_and_is_order_independent():
    raw = [f"raw-unparsed-{i}" for i in range(170)] + ["raw-unparsed-4"]
    first, counts = cm.retain_raw_outputs(raw, 7)
    reordered, _ = cm.retain_raw_outputs(list(reversed(raw)), 7)
    assert len(first) == 128
    assert [row["raw_id"] for row in first] == [row["raw_id"] for row in reordered]
    assert counts == {"native_return_count": 171, "raw_exact_duplicate_count": 1,
                      "raw_unique_count": 170, "retained_raw_count": 128, "raw_truncated_count": 42}


def test_seed_independent_of_scheduler_order():
    ids = ["p8", "p2", "p9"]
    a = {pid: cm.parent_seed(SCIENCE, pid) for pid in ids}
    b = {pid: cm.parent_seed(SCIENCE, pid) for pid in reversed(ids)}
    assert a == b and len(set(a.values())) == 3


@pytest.mark.parametrize("split", ["validation", "calibration", "test"])
def test_generation_cannot_read_non_train(split):
    with pytest.raises(cm.GenerationContractError, match="train-only"):
        cm.make_parent_request("p", Chem.MolFromSmiles("CCO"), [1], split=split)


def _sleeping_library_worker(send, marker):
    os.setsid()
    # A library resetting its own alarm cannot defeat the separate parent's timer.
    signal.signal(signal.SIGALRM, lambda *args: None)
    signal.alarm(0)
    Path(marker).write_text(str(os.getpid()))
    time.sleep(20)
    send.send({"status": "SHOULD_NOT_BE_ADOPTED"})


def test_external_timeout_isolated_group_no_partial_adoption(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    marker = tmp_path / "started"
    child = ctx.Process(target=_sleeping_library_worker, args=(send, str(marker)))
    child.start()
    send.close()
    # Startup belongs to production900s; fixture first waits for isolation then
    # exercises only the supervision kernel with a tiny private test deadline.
    deadline = time.monotonic() + 10
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(.02)
    assert marker.exists()
    result = cm._supervise_parent(child, receive, {"parent_id": "p"}, time.monotonic(), .1)
    assert result["status"] == "TIMEOUT_BUDGETED"
    assert result["partial_adopted"] is False
    assert result["retained_raw"] == []
    assert not child.is_alive()


def test_production_timeout_cannot_be_expanded_or_reduced(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    with pytest.raises(cm.GenerationContractError, match="900"):
        cm.generate_parent({}, {"parent_wall_limit_seconds": 1000}, log_path=tmp_path / "log")


def test_real_interpreter_hash_seed_not_environment_string_only():
    environment = {**os.environ, "PYTHONHASHSEED": "0"}
    code = "import sys;print(sys.flags.ignore_environment,sys.flags.hash_randomization)"
    safe = subprocess.check_output([sys.executable, "-s", "-B", "-c", code], env=environment, text=True)
    ignored = subprocess.check_output([sys.executable, "-I", "-B", "-c", code], env=environment, text=True)
    assert safe.strip() == "0 0" and ignored.strip() == "1 1"


def test_crem_only_compilation_does_not_load_difflinker(official_source):
    namespace, receipt = cm._load_native(official_source, lambda *args, **kwargs: iter(()))
    assert "delete_Generate_diffLinker" not in namespace
    assert "load_rdkit_molecule" not in namespace
    assert receipt["native_function_bodies_changed"] is False


def test_database_adapter_is_readonly_and_module_scoped(tmp_path, monkeypatch):
    database = tmp_path / "fixture.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE sample (value INTEGER)")
        connection.execute("INSERT INTO sample VALUES(7)")
    package = ModuleType("crem")
    native = ModuleType("crem.crem")
    native.mutate_mol = lambda *args, **kwargs: iter(())
    package.crem = native
    monkeypatch.setitem(sys.modules, "crem", package)
    monkeypatch.setitem(sys.modules, "crem.crem", native)
    counters = {"database_select_queries": 0}
    original_connect = sqlite3.connect
    cm._readonly_crem(database, counters)
    with native.sqlite3.connect(str(database)) as connection:
        assert connection.execute("SELECT value FROM sample").fetchone() == (7,)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("INSERT INTO sample VALUES(8)")
    assert sqlite3.connect is original_connect
    assert counters["database_select_queries"] == 1
    with pytest.raises((cm.GenerationContractError, FileNotFoundError)):
        native.sqlite3.connect(str(tmp_path / "wrong.db"))


def test_real_fixed_crem_on_empty_fixture_not_chembl_science(official_source, tmp_path, monkeypatch):
    wheel = os.environ.get("CM_CREM_TEST_WHEEL")
    if not wheel:
        pytest.skip("optional real CReM wheel integration fixture")
    monkeypatch.syspath_prepend(wheel)
    database = tmp_path / "empty-fixture-not-chembl.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE radius1 (env TEXT, core_smi TEXT, core_sma TEXT, "
                           "freq INTEGER, core_num_atoms INTEGER)")
    counters = {"database_select_queries": 0}
    actual_mutate = cm._readonly_crem(database, counters)
    result = cm.run_native_parent(cm.make_parent_request("p", Chem.MolFromSmiles("CCOCC"), [0]),
                                 source_path=official_source, database_path=database,
                                 science_hash=SCIENCE, mutate_function=actual_mutate, counters=counters)
    assert result["status"] == "NO_NATIVE_REPLACEMENT"
    assert len(result["query_receipts"]) == 8
    assert counters["database_select_queries"] > 0
