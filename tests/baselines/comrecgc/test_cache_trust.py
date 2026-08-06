from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from src.baselines.comrecgc.cache_trust import audit_aids_pyg_cache
from src.baselines.comrecgc.contracts import UPSTREAM_COMMIT


def _upstream(tmp_path: Path, *, mode: int = 0o644) -> Path:
    root = tmp_path / "upstream"
    cache = root / "data/aids/processed"
    cache.mkdir(parents=True)
    os.chmod(cache, 0o755)
    (cache / "data_0.pt").write_bytes(b"trusted-cache")
    os.chmod(cache / "data_0.pt", mode)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "cache"], cwd=root, check=True)
    return root


def test_cache_trust_rejects_group_writable_file(tmp_path: Path, monkeypatch) -> None:
    root = _upstream(tmp_path, mode=0o664)
    monkeypatch.setattr(
        "src.baselines.comrecgc.cache_trust.UPSTREAM_COMMIT",
        subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip(),
    )
    output = tmp_path / "audit.json"
    result = audit_aids_pyg_cache(upstream_root=root, output_path=output)
    assert result["cache_trust_passed"] is False
    assert result["group_writable"] is True
    assert json.loads(output.read_text())["cache_sha256"] == result["cache_sha256"]


def test_cache_trust_freezes_and_rechecks_inventory(tmp_path: Path, monkeypatch) -> None:
    root = _upstream(tmp_path)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    monkeypatch.setattr("src.baselines.comrecgc.cache_trust.UPSTREAM_COMMIT", commit)
    before = audit_aids_pyg_cache(upstream_root=root, output_path=tmp_path / "before.json")
    assert before["cache_trust_passed"] is True
    assert before["environment_has_force_no_weights_only_load"] is False
    after = audit_aids_pyg_cache(
        upstream_root=root,
        output_path=tmp_path / "after.json",
        expected_inventory_sha256=before["cache_sha256"],
    )
    assert after["cache_trust_passed"] is True
    assert after["inventory_sha256_matches_expected"] is True


def test_cache_trust_uses_pinned_upstream_constant() -> None:
    assert len(UPSTREAM_COMMIT) == 40
