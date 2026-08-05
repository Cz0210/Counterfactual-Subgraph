from __future__ import annotations

from pathlib import Path

from src.baselines.comrecgc.contracts import ordered_ids_sha256, sha256_file
from src.baselines.comrecgc.exporter import build_frozen_candidate_manifest


def test_frozen_manifest_records_file_and_order_without_reranking(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_top20.csv"
    csv_path.write_text("rank,candidate_id,smiles\n1,z,CC\n2,a,CN\n", encoding="utf-8")
    selected = [{"candidate_id": "z"}, {"candidate_id": "a"}]
    payload = build_frozen_candidate_manifest(
        dataset="aids",
        selected=selected,
        csv_path=csv_path,
    )
    assert payload["selected_candidate_ids"] == ["z", "a"]
    assert payload["selected_candidate_order_sha256"] == ordered_ids_sha256(["z", "a"])
    assert payload["file_inventory"]["selected_top20.csv"]["sha256"] == sha256_file(csv_path)
    assert payload["selection_performed_in_eval"] is False
