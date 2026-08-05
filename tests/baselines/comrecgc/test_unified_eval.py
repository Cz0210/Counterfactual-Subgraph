from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.baselines.comrecgc.run_unified_eval import resolve_candidate_input
from src.baselines.comrecgc.contracts import ContractError


def _empty_frozen_csv(path: Path) -> None:
    path.write_text("rank,candidate_id,smiles\n", encoding="utf-8")


def test_smoke_interface_uses_native_order_without_claiming_strict_flip(
    tmp_path: Path,
) -> None:
    candidates = tmp_path / "selected_top20.csv"
    _empty_frozen_csv(candidates)
    audit = tmp_path / "candidate_filter_audit.jsonl"
    rows = [
        {
            "native_rank": 1,
            "candidate_id": "invalid",
            "decode_ok": False,
            "rf_inference_ok": False,
            "canonical_smiles": "",
            "rf_cf_flip": False,
        },
        {
            "native_rank": 2,
            "candidate_id": "z",
            "decode_ok": True,
            "rf_inference_ok": True,
            "canonical_smiles": "CC",
            "source_parent_id": "p2",
            "rf_cf_flip": False,
        },
        {
            "native_rank": 3,
            "candidate_id": "a",
            "decode_ok": True,
            "rf_inference_ok": True,
            "canonical_smiles": "CN",
            "source_parent_id": "p3",
            "rf_cf_flip": True,
        },
    ]
    audit.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = tmp_path / "eval"
    output.mkdir()
    path, resolved, evidence = resolve_candidate_input(
        mode="smoke",
        candidates_csv=candidates,
        candidate_manifest={"candidate_count": 0},
        candidate_filter_audit=audit,
        output_dir=output,
    )
    assert [row["candidate_id"] for row in resolved] == ["z", "a"]
    assert [int(row["native_rank"]) for row in resolved] == [2, 3]
    assert evidence["smoke_interface_only"] is True
    assert evidence["eligible_for_final_results"] is False
    assert evidence["strict_flip_candidate_count"] == 0
    with path.open(encoding="utf-8", newline="") as handle:
        assert "rf_strict_flip" not in (csv.DictReader(handle).fieldnames or [])


def test_full_never_falls_back_to_smoke_interface_candidates(tmp_path: Path) -> None:
    candidates = tmp_path / "selected_top20.csv"
    _empty_frozen_csv(candidates)
    audit = tmp_path / "candidate_filter_audit.jsonl"
    audit.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ContractError, match="exactly 20"):
        resolve_candidate_input(
            mode="full",
            candidates_csv=candidates,
            candidate_manifest={"candidate_count": 0},
            candidate_filter_audit=audit,
            output_dir=tmp_path,
        )


def test_smoke_interface_requires_an_rf_scored_decoded_medoid(tmp_path: Path) -> None:
    candidates = tmp_path / "selected_top20.csv"
    _empty_frozen_csv(candidates)
    audit = tmp_path / "candidate_filter_audit.jsonl"
    audit.write_text(
        json.dumps(
            {
                "native_rank": 1,
                "candidate_id": "x",
                "decode_ok": False,
                "rf_inference_ok": False,
                "canonical_smiles": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="RF-scored medoid"):
        resolve_candidate_input(
            mode="smoke",
            candidates_csv=candidates,
            candidate_manifest={"candidate_count": 0},
            candidate_filter_audit=audit,
            output_dir=tmp_path,
        )
