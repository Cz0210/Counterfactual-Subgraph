#!/usr/bin/env python3
"""Build the non-executing BACE LLM-proposer ablation plan.

This command validates exact assets, builds the train-only deterministic BRICS
vocabulary, freezes one attempt-matched schedule, and writes schema/downstream
plans.  It never constructs ChemLLM, GINE, a selector, or an evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ablations.llm import (  # noqa: E402
    AVAILABLE,
    BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT,
    ArtifactPin,
    AttemptRegime,
    GeneratorAssets,
    LLMAblationContractError,
    LLMProposerVariant,
    MatchedAttemptBudget,
    ParentInput,
    build_common_downstream_plan,
    build_train_only_brics_vocabulary,
    proposal_output_template,
    run_manifest_template,
    training_molecules_from_mappings,
    validate_attempt_matched_schedules,
)
from src.ablations.llm.contracts import canonical_json_sha256, require_sha256  # noqa: E402


DEFAULT_SPEC = (
    REPO_ROOT
    / "configs"
    / "ablations"
    / "llm"
    / "llm_proposer_ablation_v1.template.json"
)

FROZEN_ATTEMPT_BUDGET = {
    "expected_parent_count": 386,
    "expected_attempts_per_parent": 8,
    "regimes": [
        {
            "name": "base",
            "attempts_per_parent": 4,
            "seed": 7,
            "temperature": 0.3,
            "top_p": 0.9,
            "max_new_tokens": 96,
        },
        {
            "name": "high_temperature",
            "attempts_per_parent": 4,
            "seed": 13,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 96,
        },
    ],
}

FROZEN_MAIN_REFERENCE_SUMMARY = {
    "policy_lineage": "CHEMLLM_BASE_PLUS_FRESH_LORA_PLUS_PPO",
    "matched_independent_sft_checkpoint_available": False,
    "sft_variants_state": BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT,
    "train_parent_count": 386,
    "attempts_per_parent": 8,
    "science_reference_only": True,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LLMAblationContractError(f"expected JSON object: {path}")
    return payload


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise LLMAblationContractError(f"{name} keys mismatch; missing={missing}, extra={extra}")


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LLMAblationContractError(f"{name} must be an object")
    return value


def _reference_identity(value: Any, *, name: str) -> tuple[str, str]:
    payload = _require_mapping(value, name=name)
    path = str(payload.get("path") or "")
    sha256 = require_sha256(payload.get("sha256"), field=f"{name}.sha256")
    if not path:
        raise LLMAblationContractError(f"{name}.path is required")
    return path, sha256


def _assert_pin_matches(
    pin: ArtifactPin,
    *,
    expected_path: Any,
    expected_sha256: Any,
    name: str,
) -> None:
    raw_expected_path = str(expected_path or "")
    if not raw_expected_path:
        raise LLMAblationContractError(f"main reference is missing {name}.path")
    try:
        resolved_expected_path = str(Path(raw_expected_path).resolve(strict=True))
    except FileNotFoundError as exc:
        raise LLMAblationContractError(
            f"main reference {name}.path does not exist: {raw_expected_path}"
        ) from exc
    expected_digest = require_sha256(expected_sha256, field=f"main_reference.{name}.sha256")
    if pin.resolved_path != resolved_expected_path or pin.sha256 != expected_digest:
        raise LLMAblationContractError(
            f"{name} is not bound to the frozen BACE main reference"
        )


def _load_bace_main_reference(
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], ArtifactPin]:
    pin = ArtifactPin.from_mapping(spec, role="bace_main_reference")
    payload = _load_json(Path(pin.resolved_path))
    if payload.get("schema_version") != "bace_ours_main_reference_v1":
        raise LLMAblationContractError("unsupported BACE main reference schema")
    if payload.get("status") != "PASS":
        raise LLMAblationContractError("BACE main reference must be PASS")
    if str(payload.get("dataset") or "").lower() != "bace":
        raise LLMAblationContractError("main reference dataset must be BACE")
    if str(payload.get("method") or "").lower() != "ours":
        raise LLMAblationContractError("main reference method must be Ours")
    if payload.get("source_label") != 1:
        raise LLMAblationContractError("main reference source_label must be 1")
    recorded_contract_sha = require_sha256(
        payload.get("reference_contract_sha256"),
        field="main_reference.reference_contract_sha256",
    )
    contract_body = dict(payload)
    contract_body.pop("reference_contract_sha256", None)
    if canonical_json_sha256(contract_body) != recorded_contract_sha:
        raise LLMAblationContractError("BACE main reference contract SHA mismatch")
    return payload, pin


def _assert_frozen_main_sampling(reference: Mapping[str, Any]) -> Mapping[str, Any]:
    proposal = _require_mapping(reference.get("proposal_contract"), name="proposal_contract")
    if proposal.get("main_proposer_lineage") != "CHEMLLM_BASE_FRESH_LORA_PPO":
        raise LLMAblationContractError("unexpected BACE main proposer lineage")
    if proposal.get("proposal_parent_count") != 386:
        raise LLMAblationContractError("BACE main proposal parent count must be 386")
    if proposal.get("candidate_attempts_per_parent") != 8:
        raise LLMAblationContractError("BACE main attempts per parent must be 8")
    expected_regimes = (
        ("base_sampling", 4, 7, 0.3, 0.9, 96),
        ("high_temperature_sampling", 4, 13, 0.7, 0.9, 96),
    )
    for name, count, seed, temperature, top_p, max_new_tokens in expected_regimes:
        regime = _require_mapping(proposal.get(name), name=f"proposal_contract.{name}")
        actual = (
            regime.get("num_return_sequences"),
            regime.get("seed"),
            regime.get("temperature"),
            regime.get("top_p"),
            regime.get("max_new_tokens"),
        )
        if actual != (count, seed, temperature, top_p, max_new_tokens):
            raise LLMAblationContractError(
                f"BACE main {name} is not the frozen four-sequence contract"
            )
    if reference.get("matched_sft_checkpoint_available") is not False:
        raise LLMAblationContractError(
            "BACE main unexpectedly declares a matched SFT checkpoint"
        )
    return proposal


def _read_train_rows(spec: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Path, str]:
    expected = {
        "path",
        "sha256",
        "molecule_id_column",
        "smiles_column",
        "split_column",
        "label_column",
    }
    _require_exact_keys(spec, expected, name="train_input")
    raw_path = str(spec.get("path") or "")
    path = Path(raw_path)
    if not path.is_absolute():
        raise LLMAblationContractError("train_input.path must be an exact absolute path")
    try:
        path = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise LLMAblationContractError(f"train_input.path does not exist: {raw_path}") from exc
    expected_sha = require_sha256(spec.get("sha256"), field="train_input.sha256")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise LLMAblationContractError("train_input SHA256 mismatch")

    columns = {
        "molecule_id": str(spec["molecule_id_column"]),
        "smiles": str(spec["smiles_column"]),
        "split": str(spec["split_column"]),
        "label": str(spec["label_column"]),
    }
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise LLMAblationContractError("train_input CSV has no header")
        missing = sorted(set(columns.values()) - set(reader.fieldnames))
        if missing:
            raise LLMAblationContractError(f"train_input missing columns: {missing}")
        for row in reader:
            # This projection is the complete field-read allowlist.  Oracle,
            # prediction, reward, and score columns cannot affect the result.
            rows.append({name: row[column] for name, column in columns.items()})
    if not rows:
        raise LLMAblationContractError("train_input CSV is empty")
    return rows, path, actual_sha


def _parse_assets(payload: Mapping[str, Any]) -> dict[str, GeneratorAssets]:
    expected = {variant.value for variant in LLMProposerVariant}
    _require_exact_keys(payload, expected, name="generator_assets")
    assets = {
        variant.value: GeneratorAssets.from_mapping(variant, payload[variant.value])
        for variant in LLMProposerVariant
    }
    return assets


def _bind_generator_assets_to_main(
    assets: Mapping[str, GeneratorAssets],
    *,
    reference: Mapping[str, Any],
    proposal: Mapping[str, Any],
) -> None:
    availability = _require_mapping(
        reference.get("llm_variant_availability"),
        name="llm_variant_availability",
    )
    expected_variants = {variant.value for variant in LLMProposerVariant}
    if set(availability) != expected_variants:
        raise LLMAblationContractError(
            "BACE main reference must declare exactly all four LLM variants"
        )
    expected_status = {
        LLMProposerVariant.CHEMLLM_PRETRAINED.value: AVAILABLE,
        LLMProposerVariant.CHEMLLM_SFT.value: BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT,
        LLMProposerVariant.CHEMLLM_SFT_PPO.value: BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT,
    }
    for variant, status in expected_status.items():
        reference_entry = _require_mapping(
            availability.get(variant),
            name=f"llm_variant_availability.{variant}",
        )
        if reference_entry.get("status") != status:
            raise LLMAblationContractError(
                f"BACE main reference {variant} status must be {status}"
            )
        if assets[variant].availability != status:
            raise LLMAblationContractError(
                f"configured {variant} status does not match BACE main reference"
            )

    brics = assets[LLMProposerVariant.BRICS_FIXED.value]
    if brics.availability != AVAILABLE:
        raise LLMAblationContractError("BRICS_FIXED must be available after vocabulary build")
    brics_reference = _require_mapping(
        availability.get(LLMProposerVariant.BRICS_FIXED.value),
        name="llm_variant_availability.BRICS_FIXED",
    )
    if (
        brics_reference.get("status") not in ("CONFIG_ONLY", AVAILABLE)
        or brics_reference.get("checkpoint") is not None
    ):
        raise LLMAblationContractError("BACE main BRICS availability is inconsistent")

    pretrained = assets[LLMProposerVariant.CHEMLLM_PRETRAINED.value]
    if pretrained.base_model is None or pretrained.tokenizer is None:
        raise LLMAblationContractError("CHEMLLM_PRETRAINED requires base and tokenizer")
    _assert_pin_matches(
        pretrained.base_model,
        expected_path=proposal.get("base_model_path"),
        expected_sha256=proposal.get("base_model_sha"),
        name="CHEMLLM_PRETRAINED.base_model",
    )
    tokenizer_path, tokenizer_sha = _reference_identity(
        proposal.get("tokenizer"),
        name="proposal_contract.tokenizer",
    )
    _assert_pin_matches(
        pretrained.tokenizer,
        expected_path=tokenizer_path,
        expected_sha256=tokenizer_sha,
        name="CHEMLLM_PRETRAINED.tokenizer",
    )
    pretrained_reference = _require_mapping(
        availability.get(LLMProposerVariant.CHEMLLM_PRETRAINED.value),
        name="llm_variant_availability.CHEMLLM_PRETRAINED",
    )
    if (
        pretrained_reference.get("checkpoint") != proposal.get("base_model_path")
        or pretrained_reference.get("checkpoint_sha") != proposal.get("base_model_sha")
    ):
        raise LLMAblationContractError(
            "BACE main PRETRAINED availability and proposal base identity disagree"
        )


def _bind_train_cohort_to_main(
    *,
    rows: list[dict[str, Any]],
    train_path: Path,
    train_sha256: str,
    proposal: Mapping[str, Any],
) -> None:
    if len(rows) != 386:
        raise LLMAblationContractError(f"BACE main requires 386 train parents, got {len(rows)}")
    labels: list[int] = []
    for row in rows:
        raw = row.get("label")
        if isinstance(raw, bool):
            raise LLMAblationContractError("BACE proposal source label must not be bool")
        try:
            labels.append(int(str(raw)))
        except (TypeError, ValueError) as exc:
            raise LLMAblationContractError(f"invalid BACE proposal source label: {raw!r}") from exc
    if set(labels) != {1}:
        raise LLMAblationContractError("all 386 BACE proposal parents must have source_label=1")
    cohort_path, cohort_sha = _reference_identity(
        proposal.get("proposal_parent_cohort"),
        name="proposal_contract.proposal_parent_cohort",
    )
    try:
        resolved_cohort_path = Path(cohort_path).resolve(strict=True)
    except FileNotFoundError as exc:
        raise LLMAblationContractError(
            f"BACE main proposal cohort does not exist: {cohort_path}"
        ) from exc
    if train_path != resolved_cohort_path or train_sha256 != cohort_sha:
        raise LLMAblationContractError(
            "train_input is not the exact frozen BACE main proposal parent cohort"
        )


def _bind_downstream_to_main(
    downstream: Any,
    *,
    reference: Mapping[str, Any],
) -> None:
    split_paths = _require_mapping(
        reference.get("dataset_split_paths"),
        name="dataset_split_paths",
    )
    split_hashes = _require_mapping(
        reference.get("dataset_split_hashes"),
        name="dataset_split_hashes",
    )
    for split in ("train", "validation", "calibration", "test"):
        _assert_pin_matches(
            getattr(downstream, f"{split}_split"),
            expected_path=split_paths.get(split),
            expected_sha256=split_hashes.get(split),
            name=f"{split}_split",
        )
    _assert_pin_matches(
        downstream.gine_checkpoint,
        expected_path=reference.get("gine_checkpoint"),
        expected_sha256=reference.get("gine_checkpoint_sha"),
        name="gine_checkpoint",
    )
    _assert_pin_matches(
        downstream.feature_schema,
        expected_path=reference.get("feature_schema"),
        expected_sha256=reference.get("feature_schema_sha"),
        name="feature_schema",
    )
    temperature_path = Path(str(reference.get("gine_checkpoint_root") or "")) / (
        "temperature_scaling.json"
    )
    _assert_pin_matches(
        downstream.temperature_manifest,
        expected_path=temperature_path,
        expected_sha256=reference.get("temperature_sha"),
        name="temperature_manifest",
    )
    selector = _require_mapping(reference.get("selector_contract"), name="selector_contract")
    selector_path, selector_sha = _reference_identity(
        selector.get("variant_configs"),
        name="selector_contract.variant_configs",
    )
    _assert_pin_matches(
        downstream.selector_config,
        expected_path=selector_path,
        expected_sha256=selector_sha,
        name="selector_config",
    )
    threshold_path, threshold_sha = _reference_identity(
        selector.get("thresholds"),
        name="selector_contract.thresholds",
    )
    _assert_pin_matches(
        downstream.threshold_config,
        expected_path=threshold_path,
        expected_sha256=threshold_sha,
        name="threshold_config",
    )
    _assert_pin_matches(
        downstream.evaluator_config,
        expected_path=(
            Path(str(reference.get("main_final_root") or "")) / "evaluation_manifest.json"
        ),
        expected_sha256=reference.get("evaluation_config_sha"),
        name="evaluator_config",
    )
    _assert_pin_matches(
        downstream.molclr_checkpoint,
        expected_path=reference.get("molclr_root"),
        expected_sha256=reference.get("molclr_sha"),
        name="molclr_checkpoint",
    )


def build_framework_payloads(
    *,
    spec: Mapping[str, Any],
    spec_path: Path,
    runtime_config_path: Path,
) -> dict[str, dict[str, Any]]:
    """Build all small framework artifacts without loading any model."""

    expected_top = {
        "schema_version",
        "framework_build_only",
        "dataset",
        "main_reference",
        "main_proposal_reference",
        "train_input",
        "generator_assets",
        "attempt_budget",
        "common_downstream",
    }
    _require_exact_keys(spec, expected_top, name="ablation spec")
    if spec.get("schema_version") != "llm_proposer_ablation_spec_v1":
        raise LLMAblationContractError("unsupported ablation spec schema")
    if spec.get("framework_build_only") is not True:
        raise LLMAblationContractError("framework_build_only must be JSON true")
    if str(spec.get("dataset") or "").lower() != "bace":
        raise LLMAblationContractError("LLM proposer ablation dataset must be BACE")
    main_reference_summary = spec["main_proposal_reference"]
    if not isinstance(main_reference_summary, Mapping):
        raise LLMAblationContractError("main_proposal_reference must be an object")
    if dict(main_reference_summary) != FROZEN_MAIN_REFERENCE_SUMMARY:
        raise LLMAblationContractError("main_proposal_reference does not match frozen BACE main")
    main_reference, main_reference_pin = _load_bace_main_reference(spec["main_reference"])
    proposal = _assert_frozen_main_sampling(main_reference)

    rows, train_path, train_sha = _read_train_rows(spec["train_input"])
    _bind_train_cohort_to_main(
        rows=rows,
        train_path=train_path,
        train_sha256=train_sha,
        proposal=proposal,
    )
    molecules = training_molecules_from_mappings(rows)
    vocabulary = build_train_only_brics_vocabulary(molecules)
    assets = _parse_assets(spec["generator_assets"])
    _bind_generator_assets_to_main(
        assets,
        reference=main_reference,
        proposal=proposal,
    )

    budget_payload = spec["attempt_budget"]
    if not isinstance(budget_payload, Mapping):
        raise LLMAblationContractError("attempt_budget must be an object")
    _require_exact_keys(
        budget_payload,
        {"expected_parent_count", "expected_attempts_per_parent", "regimes"},
        name="attempt_budget",
    )
    if dict(budget_payload) != FROZEN_ATTEMPT_BUDGET:
        raise LLMAblationContractError(
            "attempt_budget must equal the frozen 386 x (4 base + 4 high-temperature) contract"
        )
    raw_regimes = budget_payload["regimes"]
    if not isinstance(raw_regimes, list):
        raise LLMAblationContractError("attempt_budget.regimes must be a JSON list")
    regimes = tuple(AttemptRegime.from_mapping(regime) for regime in raw_regimes)
    parents = tuple(
        ParentInput(
            parent_id=molecule.molecule_id,
            parent_smiles=molecule.smiles,
            source_label=molecule.label,
        )
        for molecule in molecules
    )
    budget = MatchedAttemptBudget(
        parents=parents,
        regimes=regimes,
        expected_parent_count=budget_payload["expected_parent_count"],
    )
    if budget.attempts_per_parent != budget_payload["expected_attempts_per_parent"]:
        raise LLMAblationContractError(
            "attempt regime total does not match expected_attempts_per_parent"
        )
    validate_attempt_matched_schedules(budget.variant_schedules())
    downstream = build_common_downstream_plan(spec["common_downstream"])
    _bind_downstream_to_main(downstream, reference=main_reference)

    main_reference_binding = {
        "schema_version": "llm_ablation_bace_main_reference_binding_v1",
        "main_reference": main_reference_pin.to_dict(),
        "reference_contract_sha256": main_reference["reference_contract_sha256"],
        "proposal_parent_cohort": dict(proposal["proposal_parent_cohort"]),
        "proposal_parent_count": proposal["proposal_parent_count"],
        "candidate_attempts_per_parent": proposal["candidate_attempts_per_parent"],
        "generator_assets_bound": True,
        "common_downstream_bound": True,
    }
    main_reference_binding["binding_sha256"] = canonical_json_sha256(
        main_reference_binding
    )

    assets_document = {
        "schema_version": "llm_ablation_generator_assets_set_v1",
        "variants": {name: value.to_dict() for name, value in assets.items()},
        "chemlm_loaded": False,
        "adapter_loaded": False,
    }
    assets_document["assets_set_sha256"] = canonical_json_sha256(assets_document)
    output_template = proposal_output_template()
    manifest = run_manifest_template()
    manifest.update(
        {
            "state": "FRAMEWORK_READY_NO_SCIENCE",
            "runtime_config_path": str(runtime_config_path),
            "runtime_config_sha256": sha256_file(runtime_config_path),
            "ablation_spec_path": str(spec_path),
            "ablation_spec_sha256": sha256_file(spec_path),
            "train_input_path": str(train_path),
            "train_input_sha256": train_sha,
            "main_proposal_reference": dict(main_reference_summary),
            "bace_main_reference_binding_sha256": main_reference_binding[
                "binding_sha256"
            ],
            "brics_vocabulary_sha256": vocabulary.sha256,
            "attempt_budget_sha256": budget.to_dict()["budget_sha256"],
            "generator_assets": assets_document,
            "common_downstream_plan_sha256": downstream.to_dict()["plan_sha256"],
            "chemlm_7b_loaded": False,
            "gine_loaded": False,
            "selector_executed": False,
            "evaluator_executed": False,
            "oracle_ranking_used": False,
        }
    )
    manifest["framework_manifest_sha256"] = canonical_json_sha256(manifest)
    return {
        "brics_vocabulary.json": vocabulary.to_dict(),
        "bace_main_reference_binding.json": main_reference_binding,
        "generator_assets.json": assets_document,
        "matched_attempt_budget.json": budget.to_dict(),
        "proposal_output_template.json": output_template,
        "common_downstream_plan.json": downstream.to_dict(),
        "framework_manifest.json": manifest,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def publish_framework_payloads(output_dir: Path, payloads: Mapping[str, Mapping[str, Any]]) -> None:
    output_dir = output_dir.expanduser()
    if not output_dir.is_absolute():
        raise LLMAblationContractError("output-dir must be absolute")
    if output_dir.exists():
        raise LLMAblationContractError(f"output-dir already exists: {output_dir}")
    parent = output_dir.parent.resolve(strict=True)
    staging = parent / f".{output_dir.name}.tmp-{uuid.uuid4()}"
    staging.mkdir(mode=0o700)
    published = False
    try:
        for name, payload in payloads.items():
            _write_json(staging / name, payload)
        dir_fd = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        if output_dir.exists():
            raise LLMAblationContractError(f"output-dir appeared during publication: {output_dir}")
        os.rename(staging, output_dir)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Runtime config provenance path.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Must include inference.fallback_to_heuristic=false.",
    )
    parser.add_argument("--ablation-spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if "inference.fallback_to_heuristic=false" not in set(args.set):
        raise LLMAblationContractError(
            "--set inference.fallback_to_heuristic=false is mandatory"
        )
    runtime_config = Path(args.config).expanduser().resolve(strict=True)
    spec_path = Path(args.ablation_spec).expanduser().resolve(strict=True)
    spec = _load_json(spec_path)
    payloads = build_framework_payloads(
        spec=spec,
        spec_path=spec_path,
        runtime_config_path=runtime_config,
    )
    publish_framework_payloads(Path(args.output_dir), payloads)
    print(json.dumps(payloads["framework_manifest.json"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
