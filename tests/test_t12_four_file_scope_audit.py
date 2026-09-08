"""Concrete scope evidence only. These tests do not waive the runtime guard."""
import ast
import dataclasses
import importlib.util
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = '1ad12b560d3ad8533f47e3bc3fd1e6ee315a895a'


def old_source(path):
    return subprocess.check_output(['git', 'show', REFERENCE + ':' + path], cwd=ROOT).decode()


def functions(source):
    return {n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


@pytest.mark.parametrize('file,changed', [
    ('tastemolnet_gcf_full_postprocess.py', {'_validate_generation_pass'}),
    ('tastemolnet_gcf_full_verify.py', {'verify_t12_generation'}),
])
def test_only_declared_formal_gate_functions_changed(file, changed):
    path = 'src/baselines/' + file
    old, new = functions(old_source(path)), functions((ROOT / path).read_text())
    assert old.keys() == new.keys()
    assert {name for name in old if old[name] != new[name]} == changed
    # This does NOT claim equivalence of the changed formal gates, nor production
    # parity. The diagnostic driver must not invoke either of those gates.
    driver = ast.parse((ROOT / 'src/utils/t12_shadow_execution.py').read_text())
    assert not any(isinstance(n, ast.Name) and n.id in changed for n in ast.walk(driver))


def test_existing_selector_and_all_postprocess_evaluation_functions_identical():
    path = 'src/baselines/tastemolnet_gcf_full_postprocess.py'
    old, new = functions(old_source(path)), functions((ROOT / path).read_text())
    for name in old:
        if name != '_validate_generation_pass':
            assert old[name] == new[name], name


def test_original_codec_and_current_default_cache_none_reopen_identically(tmp_path):
    path = 'src/baselines/tastemolnet_gcf_production_state.py'
    legacy = types.ModuleType('t12_legacy_scope_audit')
    legacy.__file__ = str(ROOT / path)
    sys.modules[legacy.__name__] = legacy
    exec(compile(old_source(path), REFERENCE + ':' + path, 'exec'), legacy.__dict__)
    fixture_spec = importlib.util.spec_from_file_location('t12_cache_fixture_scope_audit',
        ROOT / 'tests/baselines/test_t12_future_history_cache.py')
    fixture = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture)
    snapshot, values, _ = fixture.fixture(tmp_path)
    original = legacy.T12CompactHistoryJournal(root=snapshot['history_root'],
        index_root=tmp_path / 'legacy-read-index',
        bounds=legacy.T12ProductionBounds.from_dict(snapshot['bounds']),
        contract_sha256=snapshot['contract_sha256'], attempt_id=snapshot['attempt_id'],
        generation_token=snapshot['generation_token'], resume_snapshot=snapshot, open_writer=False)
    current = fixture.reopen(tmp_path, snapshot, cache=None)
    try:
        assert original.checkpoint_state() == current.checkpoint_state() == snapshot
        assert original.observation_count == current.observation_count == 3
        for identity in values:
            assert dataclasses.asdict(original.lookup_first(identity)) == dataclasses.asdict(
                current.lookup_first(identity))
            assert dataclasses.asdict(original.lookup_first_embedding(identity)) == dataclasses.asdict(
                current.lookup_first_embedding(identity))
    finally:
        original.close()
        current.close()
        sys.modules.pop(legacy.__name__)


def test_history_top_level_codec_helpers_are_identical():
    path = 'src/baselines/tastemolnet_gcf_production_state.py'
    assert functions(old_source(path)) == functions((ROOT / path).read_text())


def test_same_uuid_is_not_misrepresented_as_cross_gpu_resume():
    # This is a currently unresolved execution binding, not a request to relax it.
    from src.baselines.tastemolnet_gcf_full import validate_cross_gpu_resume_identity
    import inspect
    assert 'current_uuid == authority_uuid' in inspect.getsource(validate_cross_gpu_resume_identity)
