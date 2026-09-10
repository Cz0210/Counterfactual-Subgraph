"""Static-asset probe tests; synthetic DBs are never a ChEMBL science claim."""
import errno
import os
from pathlib import Path
import random
import sqlite3
from types import SimpleNamespace

import pytest

from src.baselines import cm_crem_database_compat as compat


@pytest.fixture
def database(tmp_path):
    path = tmp_path / 'fixture.db'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE radius1 (env TEXT, freq INTEGER, core_num_atoms INTEGER, core_smi TEXT, core_sma TEXT)')
        connection.execute('INSERT INTO radius1 VALUES (?, ?, ?, ?, ?)',
                           ('*-C-[*:1]', 5, 1, 'N[*:1]', '[NH2]-[CH2:3]-[*:1]'))
    return path


@pytest.fixture
def native(monkeypatch):
    api = SimpleNamespace(sqlite3=sqlite3)
    def mutate(mol, db_name, **kwargs):
        assert kwargs == compat.PUBLIC_FIXTURE_SETTINGS
        assert api.sqlite3 is not sqlite3
        connection = api.sqlite3.connect(db_name)
        assert connection.execute('PRAGMA query_only').fetchone() == (1,)
        connection.execute('SELECT rowid, core_smi FROM radius1').fetchall()
        yield 'NCO'
    api.mutate_mol = mutate
    monkeypatch.setattr(compat, '_load_pinned_crem', lambda: (api, {'test_only': 'MOCK_NOT_PINNED_RUNTIME'}))
    return api


def test_read_only_select_replacement_and_all_bindings(database, native):
    before = database.read_bytes()
    state = random.getstate()
    original_connect = sqlite3.connect
    result = compat.verify_static_database_compatibility(database)
    assert result['status'] == 'STATIC_DATABASE_COMPATIBILITY_PASS'
    assert result['actual_select_count'] >= 3 and result['fixture_select_count'] >= 1
    assert result['fixture_mutate_calls'] == 1
    assert result['public_fixture_products'] == ['NCO']
    assert result['stat_before'] == result['stat_after']
    assert result['sidecars_before'] == result['sidecars_after'] == []
    assert result['oracle_calls'] == 0 and result['experiment_generation_performed'] is False
    assert database.read_bytes() == before
    assert native.sqlite3 is sqlite3 and sqlite3.connect is original_connect
    assert random.getstate() == state


def test_unsealed_sidecar_rejected_before_immutable_open(database, native):
    Path(str(database) + '-wal').write_bytes(b'not-adoptable')
    with pytest.raises(compat.DatabaseCompatibilityError) as caught:
        compat.verify_static_database_compatibility(database)
    assert caught.value.stage == 'ACTIVE_OR_UNSEALED_SQLITE'
    assert caught.value.receipt['actual_connection_count'] == 0


def test_header_rejected_without_sqlite_or_mutation(tmp_path, native):
    path = tmp_path / 'not.db'; path.write_bytes(b'not SQLite')
    with pytest.raises(compat.DatabaseCompatibilityError) as caught:
        compat.verify_static_database_compatibility(path)
    assert caught.value.stage == 'PATH_AND_HEADER'
    assert caught.value.receipt['fixture_mutate_calls'] == 0


@pytest.mark.parametrize('ddl', ['CREATE TABLE other(x)',
                               'CREATE TABLE radius1(env TEXT, freq INTEGER)'])
def test_missing_actual_table_or_columns_rejected(tmp_path, native, ddl):
    path=tmp_path/'schema.db'
    with sqlite3.connect(path) as connection: connection.execute(ddl)
    with pytest.raises(compat.DatabaseCompatibilityError) as caught:
        compat.verify_static_database_compatibility(path)
    assert caught.value.stage == 'RADIUS1_SCHEMA'


def test_empty_table_is_not_compatible_database(database, native):
    with sqlite3.connect(database) as connection: connection.execute('DELETE FROM radius1')
    with pytest.raises(compat.DatabaseCompatibilityError) as caught:
        compat.verify_static_database_compatibility(database)
    assert caught.value.stage == 'RADIUS1_REAL_SELECT'
    assert caught.value.receipt['fixture_mutate_calls'] == 0


def test_actual_write_attempt_is_denied_and_not_empty_success(database, native):
    def forbidden(mol, db_name, **kwargs):
        connection = native.sqlite3.connect(db_name)
        connection.execute("DELETE FROM radius1")
        yield 'NCO'
    native.mutate_mol = forbidden
    original = database.read_bytes()
    with pytest.raises(compat.DatabaseCompatibilityError) as caught:
        compat.verify_static_database_compatibility(database)
    assert caught.value.stage == 'INFRASTRUCTURE_FAILURE'
    assert 'readonly' in str(caught.value).lower()
    assert database.read_bytes() == original and native.sqlite3 is sqlite3


def test_io_failure_not_reported_as_zero_replacements(database, native):
    def broken(*args, **kwargs):
        raise OSError(errno.EIO, 'test disk error')
        yield 'NCO'
    native.mutate_mol = broken
    state = random.getstate()
    with pytest.raises(compat.DatabaseCompatibilityError) as caught:
        compat.verify_static_database_compatibility(database)
    assert caught.value.stage == 'INFRASTRUCTURE_FAILURE'
    assert caught.value.receipt['infrastructure_error']['errno'] == errno.EIO
    assert native.sqlite3 is sqlite3 and random.getstate() == state


def test_empty_actual_query_cannot_claim_replacement_pass(database, native):
    def empty(mol, db_name, **kwargs):
        native.sqlite3.connect(db_name).execute('SELECT rowid FROM radius1 WHERE 1=0').fetchall()
        yield from ()
    native.mutate_mol = empty
    with pytest.raises(compat.DatabaseCompatibilityError) as caught:
        compat.verify_static_database_compatibility(database)
    assert caught.value.stage == 'PUBLIC_FIXTURE_ACTUAL_REPLACEMENT'
    assert caught.value.receipt['fixture_select_count'] == 1
    assert caught.value.receipt['public_fixture_product_count'] == 0


def test_source_change_detected_despite_successful_query(database, native):
    mutate = native.mutate_mol
    def changed(*args, **kwargs):
        yield from mutate(*args, **kwargs)
        original = database.stat()
        os.utime(database, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000))
    native.mutate_mol = changed
    with pytest.raises(compat.DatabaseCompatibilityError) as caught:
        compat.verify_static_database_compatibility(database)
    assert caught.value.stage == 'STATIC_SOURCE_CHANGED'


def test_unbound_native_database_rejected(database, native):
    def unbound(mol, db_name, **kwargs):
        native.sqlite3.connect(str(database.parent))
        yield 'NCO'
    native.mutate_mol = unbound
    with pytest.raises(compat.DatabaseCompatibilityError) as caught:
        compat.verify_static_database_compatibility(database)
    assert caught.value.stage == 'UNBOUND_DATABASE_ACCESS'


def test_actual_fixed_crem_wheel_replaces_public_fixture_on_synthetic_db(database, monkeypatch):
    wheel = os.environ.get('CM_CREM_TEST_WHEEL')
    if not wheel:
        pytest.skip('fixed wheel path absent; do not claim actual CReM integration')
    monkeypatch.syspath_prepend(wheel)
    import crem.crem as native
    import importlib.metadata
    assert importlib.metadata.version('crem') == '0.2.14'
    monkeypatch.setattr(compat, '_load_pinned_crem', lambda: (native, {'test_only': 'ACTUAL_CREM_SYNTHETIC_DB_LOCAL_RDKIT'}))
    result = compat.verify_static_database_compatibility(database)
    assert result['public_fixture_products'] == ['NCO']
    assert result['fixture_select_count'] >= 2
    assert native.sqlite3 is sqlite3
