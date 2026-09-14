import types
from pathlib import Path
import pytest
from src.baselines.cm_crem_fragment_repair import install, OLD, NEW


def test_patch_is_one_component_selection_not_empty_result_or_ion_strip():
    assert 'MolToSmiles' not in NEW
    assert 'len(components) != 2' in NEW
    assert 'raise ValueError' in NEW
    assert NEW.startswith(OLD)


def test_unknown_source_is_rejected():
    def unknown(): return []
    native=types.SimpleNamespace(__fragment_mol=unknown)
    with pytest.raises(ValueError,match='Unreviewed'): install(native)


def test_component_remapping_is_explicit_and_does_not_strip_original():
    import inspect
    source=inspect.getsource(install)
    assert 'fragsMolAtomMapping=maps' in source
    assert 'mapping[i] for i in ids' in source
    assert 'old in set(protected_ids or [])' in source
    assert 'return original(mol,' in source
    assert 'except' not in source
