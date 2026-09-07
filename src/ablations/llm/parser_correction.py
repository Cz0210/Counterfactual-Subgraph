"""Explicit saved-response parser overlay; no generation or oracle execution."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.models.llm_generator import SMILES_EXTRACTION_CONTRACT, clean_generated_smiles
from src.eval.bace_frozen_gnn_contracts import stable_sha256


def reparse_attempts(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Copy validated native train attempts, changing only extracted SMILES."""
    corrected = []
    changes = []
    for row in rows:
        if not isinstance(row.get("raw_text"), str) or row.get("train_only") is not True:
            raise ValueError("PARSER_CORRECTION_REQUIRES_SAVED_NATIVE_TRAIN_RAW_TEXT")
        value = clean_generated_smiles(row["raw_text"])
        if value != row.get("fragment_smiles", ""):
            changes.append({"parent_id": row["parent_id"], "attempt_index": row["attempt_index"],
                "old_fragment_smiles": row.get("fragment_smiles", ""), "corrected_fragment_smiles": value})
        corrected.append({**row, "fragment_smiles": value})
    return corrected, {"schema_version": "bace_saved_raw_parser_correction_v1",
        "parser_contract": SMILES_EXTRACTION_CONTRACT,
        "input_attempts_sha256": stable_sha256(list(rows)),
        "corrected_attempts_sha256": stable_sha256(corrected),
        "attempts": len(rows), "changed_attempts": len(changes), "changes": changes,
        "candidate_generation_rerun": False, "attempt_budget_changed": False,
        "model_loaded": False, "oracle_used_for_extraction": False,
        "calibration_used_for_extraction": False, "test_used_for_extraction": False,
        "syntax_repair_added": False, "selfies_decoded": False}


def reparse_scored_diagnostic(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit saved train scored rows; deliberately cannot be a generated pool."""
    result = []
    seen = set()
    for row in rows:
        if row.get("stage") != "LLM_COMMON_TRAIN_ONLY" or row.get("test_loaded") is not False or row.get("calibration_loaded") is not False:
            raise ValueError("DIAGNOSTIC_REQUIRES_COMPLETED_TRAIN_ONLY_SCORED_ROWS")
        key = (row["parent_id"], int(row["candidate_index"]))
        if key in seen:
            raise ValueError("DUPLICATE_SAVED_ATTEMPT")
        seen.add(key)
        raw = row["raw_output"]
        if not isinstance(raw, str):
            raise ValueError("MISSING_SAVED_RAW_OUTPUT")
        old = row.get("raw_fragment") or ""
        new = clean_generated_smiles(raw)
        result.append({"parent_id": key[0], "attempt_index": key[1], "raw_output": raw,
            "old_fragment": old, "corrected_fragment": new, "changed": old != new})
    return {"schema_version": "bace_parser_correction_diagnostic_v1",
        "parser_contract": SMILES_EXTRACTION_CONTRACT, "attempt_count": len(result),
        "parent_count": len({x[0] for x in seen}),
        "changed_count": sum(r["changed"] for r in result),
        "empty_after_count": sum(not r["corrected_fragment"] for r in result),
        "prose_iodine_removed_count": sum(r["old_fragment"] == "I" and r["corrected_fragment"] != "I" for r in result),
        "rows": result, "model_loaded": False, "oracle_called": False,
        "test_used_for_policy": False, "scientific_evaluation_complete": False}
