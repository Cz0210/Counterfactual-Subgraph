"""Proposal output schema, templates, and train-BRICS novelty metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Iterable, Mapping

from .brics import BRICSVocabulary
from .contracts import (
    LLMAblationContractError,
    LLMProposerVariant,
    ProposalRequest,
    ProposalResult,
    canonical_json_sha256,
)

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - depends on local environment
    Chem = None


PROPOSAL_SCHEMA_VERSION = "llm_ablation_proposal_record_v2"
PROPOSAL_RECORD_FIELDS = (
    "schema_version",
    "experiment_id",
    "proposal_id",
    "variant",
    "parent_id",
    "parent_smiles",
    "parent_smiles_sha256",
    "source_label",
    "regime",
    "regime_attempt_index",
    "attempt_index",
    "attempt_seed",
    "slot_id",
    "attempt_budget_sha256",
    "max_new_tokens",
    "temperature",
    "top_p",
    "fragment_smiles",
    "canonical_fragment_smiles",
    "raw_text",
    "finish_reason",
    "proposal_shortfall",
    "generator_metadata",
    "train_brics_vocabulary_sha256",
    "novel_to_train_brics",
    "oracle_ranked",
    "gine_evaluated",
    "selector_evaluated",
    "held_out_test_evaluated",
)


def canonicalize_fragment_smiles(smiles: str) -> str | None:
    if Chem is None:
        raise LLMAblationContractError("RDKit is required for proposal canonicalization")
    normalized = str(smiles or "").strip()
    if not normalized:
        return None
    mol = Chem.MolFromSmiles(normalized)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def proposal_output_template() -> dict[str, Any]:
    """Return a non-scientific schema template with no fabricated metrics."""

    payload = {field: None for field in PROPOSAL_RECORD_FIELDS}
    payload.update(
        {
            "schema_version": PROPOSAL_SCHEMA_VERSION,
            "oracle_ranked": False,
            "gine_evaluated": False,
            "selector_evaluated": False,
            "held_out_test_evaluated": False,
        }
    )
    return payload


def build_proposal_record(
    *,
    experiment_id: str,
    request: ProposalRequest,
    result: ProposalResult,
    vocabulary: BRICSVocabulary,
) -> dict[str, Any]:
    """Build one generator-only record; downstream science remains unset."""

    if not str(experiment_id).strip():
        raise LLMAblationContractError("experiment_id must be non-empty")
    metadata = dict(result.metadata)
    proposal_shortfall = result.finish_reason == "proposal_shortfall"
    if (
        "proposal_shortfall" in metadata
        and metadata["proposal_shortfall"] is not proposal_shortfall
    ):
        raise LLMAblationContractError(
            "proposal_shortfall metadata and finish_reason disagree"
        )
    if proposal_shortfall and (result.fragment_smiles or result.raw_text):
        raise LLMAblationContractError("proposal shortfall may not contain candidate text")
    canonical = canonicalize_fragment_smiles(result.fragment_smiles)
    novelty = canonical not in set(vocabulary.fragments) if canonical is not None else None
    identity = {
        "experiment_id": experiment_id,
        "variant": result.variant.value,
        "parent_id": request.parent_id,
        "slot_id": request.slot_id,
        "attempt_budget_sha256": request.attempt_budget_sha256,
        "attempt_index": request.attempt_index,
        "attempt_seed": request.attempt_seed,
    }
    record = {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "proposal_id": canonical_json_sha256(identity),
        "variant": result.variant.value,
        "parent_id": request.parent_id,
        "parent_smiles": request.parent_smiles,
        "parent_smiles_sha256": hashlib.sha256(
            request.parent_smiles.encode("utf-8")
        ).hexdigest(),
        "source_label": request.source_label,
        "regime": request.regime,
        "regime_attempt_index": request.regime_attempt_index,
        "attempt_index": request.attempt_index,
        "attempt_seed": request.attempt_seed,
        "slot_id": request.slot_id,
        "attempt_budget_sha256": request.attempt_budget_sha256,
        "max_new_tokens": request.max_new_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "fragment_smiles": result.fragment_smiles,
        "canonical_fragment_smiles": canonical,
        "raw_text": result.raw_text,
        "finish_reason": result.finish_reason,
        "proposal_shortfall": proposal_shortfall,
        "generator_metadata": metadata,
        "train_brics_vocabulary_sha256": vocabulary.sha256,
        "novel_to_train_brics": novelty,
        "oracle_ranked": False,
        "gine_evaluated": False,
        "selector_evaluated": False,
        "held_out_test_evaluated": False,
    }
    validate_proposal_record(record)
    return record


def validate_proposal_record(record: Mapping[str, Any]) -> None:
    if set(record) != set(PROPOSAL_RECORD_FIELDS):
        missing = sorted(set(PROPOSAL_RECORD_FIELDS) - set(record))
        extra = sorted(set(record) - set(PROPOSAL_RECORD_FIELDS))
        raise LLMAblationContractError(
            f"proposal schema mismatch; missing={missing}, extra={extra}"
        )
    if record["schema_version"] != PROPOSAL_SCHEMA_VERSION:
        raise LLMAblationContractError("unexpected proposal schema version")
    LLMProposerVariant(str(record["variant"]))
    for field in (
        "proposal_id",
        "slot_id",
        "attempt_budget_sha256",
        "parent_smiles_sha256",
        "train_brics_vocabulary_sha256",
    ):
        value = str(record[field] or "")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise LLMAblationContractError(f"{field} must be one lowercase SHA256")
    if not str(record["regime"] or "").strip():
        raise LLMAblationContractError("regime must be non-empty")
    for field in ("experiment_id", "parent_id", "parent_smiles"):
        if not str(record[field] or "").strip():
            raise LLMAblationContractError(f"{field} must be non-empty")
    for field in ("regime_attempt_index", "attempt_index", "attempt_seed"):
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LLMAblationContractError(f"{field} must be a non-negative integer")
    if hashlib.sha256(str(record["parent_smiles"]).encode("utf-8")).hexdigest() != record[
        "parent_smiles_sha256"
    ]:
        raise LLMAblationContractError("parent_smiles_sha256 mismatch")
    expected_slot_id = canonical_json_sha256(
        {
            "parent_id": record["parent_id"],
            "regime": record["regime"],
            "regime_attempt_index": record["regime_attempt_index"],
            "attempt_index": record["attempt_index"],
            "attempt_seed": record["attempt_seed"],
            "max_new_tokens": record["max_new_tokens"],
            "temperature": record["temperature"],
            "top_p": record["top_p"],
        }
    )
    if expected_slot_id != record["slot_id"]:
        raise LLMAblationContractError("proposal slot_id mismatch")
    expected_proposal_id = canonical_json_sha256(
        {
            "experiment_id": record["experiment_id"],
            "variant": record["variant"],
            "parent_id": record["parent_id"],
            "slot_id": record["slot_id"],
            "attempt_budget_sha256": record["attempt_budget_sha256"],
            "attempt_index": record["attempt_index"],
            "attempt_seed": record["attempt_seed"],
        }
    )
    if expected_proposal_id != record["proposal_id"]:
        raise LLMAblationContractError("proposal_id mismatch")
    for field in (
        "oracle_ranked",
        "gine_evaluated",
        "selector_evaluated",
        "held_out_test_evaluated",
    ):
        if record[field] is not False:
            raise LLMAblationContractError(
                f"generator-only proposal must set {field}=false"
            )
    if record["novel_to_train_brics"] not in (True, False, None):
        raise LLMAblationContractError("novel_to_train_brics must be bool or null")
    if record["proposal_shortfall"] not in (True, False):
        raise LLMAblationContractError("proposal_shortfall must be bool")
    if record["proposal_shortfall"]:
        if (
            record["finish_reason"] != "proposal_shortfall"
            or record["fragment_smiles"] != ""
            or record["raw_text"] != ""
            or record["canonical_fragment_smiles"] is not None
            or record["novel_to_train_brics"] is not None
        ):
            raise LLMAblationContractError("proposal shortfall row is internally inconsistent")
    elif record["finish_reason"] == "proposal_shortfall":
        raise LLMAblationContractError("proposal_shortfall finish_reason requires true flag")


@dataclass(frozen=True, slots=True)
class NoveltySummary:
    attempt_rows: int
    total_proposals: int
    proposal_shortfall_count: int
    canonicalizable_proposals: int
    invalid_proposals: int
    in_train_brics_count: int
    novel_count: int
    novelty_rate: float | None
    vocabulary_sha256: str
    schema_version: str = "llm_ablation_novelty_metrics_v2"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["novelty_denominator"] = "canonicalizable_non_shortfall_proposals"
        payload["oracle_used"] = False
        payload["metrics_sha256"] = canonical_json_sha256(payload)
        return payload


def summarize_novelty(
    records: Iterable[Mapping[str, Any]],
    *,
    vocabulary: BRICSVocabulary,
) -> NoveltySummary:
    canonicalizable = invalid = in_vocab = novel = attempts = shortfalls = 0
    vocab = set(vocabulary.fragments)
    for record in records:
        validate_proposal_record(record)
        attempts += 1
        if record["proposal_shortfall"] is True:
            shortfalls += 1
            continue
        canonical = record.get("canonical_fragment_smiles")
        if canonical is None:
            canonical = canonicalize_fragment_smiles(str(record.get("fragment_smiles") or ""))
        if canonical is None:
            invalid += 1
            continue
        canonicalizable += 1
        if canonical in vocab:
            in_vocab += 1
        else:
            novel += 1
    rate = novel / canonicalizable if canonicalizable else None
    return NoveltySummary(
        attempt_rows=attempts,
        total_proposals=attempts - shortfalls,
        proposal_shortfall_count=shortfalls,
        canonicalizable_proposals=canonicalizable,
        invalid_proposals=invalid,
        in_train_brics_count=in_vocab,
        novel_count=novel,
        novelty_rate=rate,
        vocabulary_sha256=vocabulary.sha256,
    )


def run_manifest_template() -> dict[str, Any]:
    """Template for a future science run, deliberately containing no result."""

    return {
        "schema_version": "llm_proposer_ablation_run_manifest_v1",
        "state": "CONFIG_ONLY",
        "science_started": False,
        "model_weights_loaded": False,
        "main_matrix_mutated": False,
        "attempt_budget_sha256": None,
        "brics_vocabulary_sha256": None,
        "generator_assets": None,
        "proposal_schema": proposal_output_template(),
        "common_downstream_plan_sha256": None,
        "candidate_metrics": None,
        "novelty_metrics": None,
        "final_metrics": None,
    }


__all__ = [
    "NoveltySummary",
    "PROPOSAL_RECORD_FIELDS",
    "PROPOSAL_SCHEMA_VERSION",
    "build_proposal_record",
    "canonicalize_fragment_smiles",
    "proposal_output_template",
    "run_manifest_template",
    "summarize_novelty",
    "validate_proposal_record",
]
