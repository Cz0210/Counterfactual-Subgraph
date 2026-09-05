"""A verifier-only retry preserves the completed corrective science."""
import json

import pytest

from src.ablations.gnn import temperature_repair as repair


def test_verifier_retry_does_not_repeat_or_repack_science(tmp_path, monkeypatch):
    root = tmp_path / 'correction'
    (root / 'verified').mkdir(parents=True)
    archive = root / 'verified/bace_gnn_seed7_corrected.tar.gz'
    archive.write_bytes(b'completed-corrective-package')
    original = tmp_path / 'original.tar.gz'
    original.write_bytes(b'original-sealed-package')
    monkeypatch.setattr(repair, 'ORIGINAL_SHA', repair.sha256_file(original))
    monkeypatch.setattr(repair, 'context', lambda p: (root, {
        'original_package': str(original), 'driver_commit': 'old-temperature-driver'}, None, None, None))
    monkeypatch.setenv('SLURM_JOB_ID', '999')
    monkeypatch.setattr('subprocess.check_output', lambda *a, **kw: 'fresh-verifier-driver\n')
    calls = []
    def reopen(path):
        calls.append(path)
        return {'state': 'GNN_CORE_SEED7_CORRECTED_PASS', 'raw_ot_recomputed_count': 0}
    monkeypatch.setattr(repair, 'verify_corrective_package', reopen)
    def forbidden(*a, **kw):
        raise AssertionError('Science must not be repeated')
    for name in ('fit', 'reconcile', 'replay_phase', 'verify_package'):
        monkeypatch.setattr(repair, name, forbidden)
    value = repair.verify_existing_package(root, root / 'verification-attempt-2')
    assert calls == [archive]
    assert archive.read_bytes() == b'completed-corrective-package'
    assert original.read_bytes() == b'original-sealed-package'
    assert value['temperature_driver_commit'] == 'old-temperature-driver'
    assert value['verification_driver_commit'] == 'fresh-verifier-driver'
    assert not any(value[k] for k in ('fit_repeated', 'inference_repeated', 'ot_repeated', 'package_repacked'))
    assert json.loads((root / 'verified/result_package.json').read_text()) == value
    with pytest.raises(ValueError, match='DO_NOT_OVERWRITE'):
        repair.verify_existing_package(root, root / 'verification-attempt-3')

