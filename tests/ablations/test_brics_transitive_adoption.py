import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ablations.llm.core_execution import CoreLLMVariant, validate_variant_artifact_bindings
from src.ablations.llm.contracts import LLMAblationContractError
from src.eval.bace_frozen_gnn_contracts import sha256_file


def fixture(tmp_path):
    pool = tmp_path / "brics_proposal_pool.jsonl"
    pool.write_text('{}\n')
    vocabulary = {"status": "PASS", "source_split": "train", "calibration_loaded": False,
        "test_loaded": False, "oracle_fields_read": [], "reference_contract": {"sha256": "a" * 64}}
    shortfall = {"status": "PASS", "candidate_duplication_used": False, "oracle_ranking_used": False,
        "shortfall_is_not_backfilled": True, "reference_contract": {"sha256": "a" * 64}}
    vp, sp = tmp_path / "brics_vocab_manifest.json", tmp_path / "brics_proposal_shortfall_receipt.json"
    vp.write_text(json.dumps(vocabulary)); sp.write_text(json.dumps(shortfall))
    identity = lambda p: {"path": str(p), "sha256": sha256_file(p)}
    proposal = {"status": "PASS", "oracle_used": False, "calibration_loaded": False,
        "test_loaded": False, "candidate_pool": identity(pool), "vocabulary_manifest": identity(vp),
        "shortfall_receipt": identity(sp)}
    pp = tmp_path / "brics_proposal_manifest.json"
    pp.write_text(json.dumps(proposal))
    spec = SimpleNamespace(variant=CoreLLMVariant.BRICS_FIXED,
        stages=[SimpleNamespace(adopted_artifacts=[SimpleNamespace(**identity(p)) for p in (pool, vp, sp, pp)])])
    return spec, SimpleNamespace(file_sha256="a" * 64), pp, sp


def test_transitive_brics_adoption_preserves_original_missing_field(tmp_path):
    spec, reference, proposal, _ = fixture(tmp_path)
    before = proposal.read_bytes()
    validate_variant_artifact_bindings(spec, reference)
    assert proposal.read_bytes() == before
    assert "reference_contract" not in json.loads(before)


def test_transitive_brics_rejects_wrong_manifest_hash(tmp_path):
    spec, reference, proposal, _ = fixture(tmp_path)
    value = json.loads(proposal.read_text())
    value["vocabulary_manifest"]["sha256"] = "b" * 64
    proposal.write_text(json.dumps(value))
    with pytest.raises(LLMAblationContractError, match="transitive"):
        validate_variant_artifact_bindings(spec, reference)


def test_transitive_brics_rejects_other_reference_even_if_hash_chain_is_resealed(tmp_path):
    spec, reference, proposal, shortfall = fixture(tmp_path)
    value = json.loads(shortfall.read_text()); value["reference_contract"]["sha256"] = "b" * 64
    shortfall.write_text(json.dumps(value))
    value = json.loads(proposal.read_text()); value["shortfall_receipt"]["sha256"] = sha256_file(shortfall)
    proposal.write_text(json.dumps(value))
    for item in spec.stages[0].adopted_artifacts:
        item.sha256 = sha256_file(Path(item.path))
    with pytest.raises(LLMAblationContractError, match="transitive"):
        validate_variant_artifact_bindings(spec, reference)
