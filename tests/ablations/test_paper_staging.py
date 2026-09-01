from __future__ import annotations

from src.ablations.paper_staging import TEMPLATES, build_paper_staging


def test_paper_staging_contains_only_templates(tmp_path) -> None:
    paths = build_paper_staging(tmp_path)
    assert {path.name for path in paths} == set(TEMPLATES)
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "TODO" in combined
    assert "significantly improves" not in combined
    assert "outperforms" not in combined
    assert "proves external knowledge" not in combined
    assert "no independently matched BACE SFT checkpoint" in combined
    assert "These labels describe the experimental design and" in combined
    assert "are not findings" in combined
