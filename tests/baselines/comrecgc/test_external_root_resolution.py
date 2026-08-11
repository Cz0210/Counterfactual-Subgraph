from __future__ import annotations

from pathlib import Path


def test_bace_wrapper_uses_explicit_vendor_root() -> None:
    root = Path(__file__).resolve().parents[3]
    wrapper = (root / "scripts/slurm/comrecgc_bace_project_generate.sh").read_text(
        encoding="utf-8"
    )
    assert "COMRECGC_ROOT=" in wrapper
    assert '--upstream-root "$COMRECGC_ROOT"' in wrapper
    assert "--upstream-root external/COMRECGC" not in wrapper
    assert "verify_comrecgc_checkout.py" in wrapper
