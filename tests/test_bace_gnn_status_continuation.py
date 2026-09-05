"""Latest publication dependencies must replace obsolete failed job summaries."""
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/hpc/gnn/status_bace_gnn_seed7.py'


def test_publication_status_uses_latest_jobs_and_preserves_history(tmp_path, monkeypatch, capsys):
    base = {'bundle_manifest_sha256': 'bound', 'jobs': {'gin': '1', 'evaluate': '2', 'package': '3'}, 'attempt_roots': {}}
    resume = {'source_bundle_manifest_sha256': 'bound', 'jobs': {'gin': '4', 'evaluate': '5', 'package': '6'}, 'resume_driver_commit': 'resume'}
    publication = {'source_bundle_manifest_sha256': 'bound', 'jobs': {'finalize': '7', 'evaluate': '8', 'package': '9'}, 'publication_driver_commit': 'publish'}
    for name, data in [('submission', base), ('resume_submission', resume), ('publication_submission', publication)]:
        (tmp_path / (name + '.json')).write_text(json.dumps(data))
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout='JobID|State\n7|RUNNING\n')
    monkeypatch.setattr('subprocess.run', fake_run)
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--campaign-root', str(tmp_path)])
    runpy.run_path(str(SCRIPT), run_name='__main__')
    result = json.loads(capsys.readouterr().out)['submission']
    assert result['jobs'] == {'gin': '4', 'evaluate': '8', 'package': '9', 'finalize': '7'}
    assert result['historical_jobs'] == base['jobs']
    assert result['pre_publication_jobs'] == resume['jobs']
    assert calls[0][2] == '4,8,9,7'


def test_publication_status_rejects_wrong_bundle(tmp_path, monkeypatch):
    (tmp_path / 'submission.json').write_text(json.dumps({'bundle_manifest_sha256': 'bound', 'jobs': {}, 'attempt_roots': {}}))
    (tmp_path / 'publication_submission.json').write_text(json.dumps({'source_bundle_manifest_sha256': 'wrong'}))
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--campaign-root', str(tmp_path)])
    with pytest.raises(ValueError, match='publication receipt input binding'):
        runpy.run_path(str(SCRIPT), run_name='__main__')
