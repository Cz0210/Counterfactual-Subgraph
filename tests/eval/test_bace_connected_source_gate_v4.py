from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.filter_bace_connected_source_candidates import filter_candidates


def _row(fragment: str, residual: str, *, flip: bool, drop: float) -> dict[str, object]:
    return {
        "molecule_id": "p1",
        "parent_id": "p1",
        "parent_smiles": "CCO",
        "label": 1,
        "candidate_index": 0,
        "source_graph_hash": "abc",
        "final_fragment": fragment,
        "parse_ok": True,
        "valid": True,
        "connected": True,
        "final_substructure": True,
        "direct_substructure": True,
        "parent_without_fragment_smiles": residual,
        "atom_ratio": 1.0 / 3.0,
        "cf_flip": flip,
        "cf_drop": drop,
        "oracle_ok": flip,
    }


def test_source_effect_is_feature_not_gate(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps(_row("O", "CC", flip=False, drop=0.0)) + "\n",
        encoding="utf-8",
    )
    parent_csv = tmp_path / "parents.csv"
    with parent_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("molecule_id", "scaffold"))
        writer.writeheader()
        writer.writerow({"molecule_id": "p1", "scaffold": "CCO"})
    output = tmp_path / "filtered.jsonl"
    audit_path = tmp_path / "audit.json"

    audit = filter_candidates(
        input_jsonl=input_path,
        parent_csv=parent_csv,
        output_jsonl=output,
        audit_json=audit_path,
        generation_round=1,
        generation_regime="A",
        prompt_mode="connected_deletion_v1",
    )

    retained = json.loads(output.read_text(encoding="utf-8"))
    assert retained["cf_flip"] is False
    assert retained["source_residual_connected"] is True
    assert retained["source_scaffold"] == "CCO"
    assert len(retained["candidate_lineage_sha256"]) == 64
    assert audit["source_cf_flip_feature_only"] is True
    assert audit["retained_candidate_count"] == 1
