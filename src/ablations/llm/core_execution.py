"""Evidence-bound execution contract for the four core BACE LLM ablations.

The runner in this module is deliberately an orchestrator, not a second
scientific implementation.  It executes the already-reviewed generator,
verifier, selector and evaluator entrypoints from an immutable run spec,
records every produced artifact, and checkpoints only at stage boundaries.
It never invokes a shell and never invents a result when an inner stage fails.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence
from uuid import UUID

from .contracts import LLMAblationContractError, canonical_json_sha256, require_sha256
from .early_launch_gate import EarlyLaunchSnapshot, main_priority_runtime_action
from .runtime_evidence import BACEReferenceEvidence, sha256_file


class CoreLLMVariant(str, Enum):
    """The only four rows in the current main LLM ablation."""

    BRICS_FIXED = "BRICS_FIXED"
    CHEMLLM_7B_OFF_THE_SHELF = "CHEMLLM_7B_OFF_THE_SHELF"
    CHEMLLM_7B_PPO_LORA_MAIN = "CHEMLLM_7B_PPO_LORA_MAIN"
    CHEMLLM_2B_OFF_THE_SHELF = "CHEMLLM_2B_OFF_THE_SHELF"


CORE_VARIANT_ORDER = tuple(item.value for item in CoreLLMVariant)
SFT_AUXILIARY_STATE = "N/A"
SFT_AUXILIARY_REASON = "NO_INDEPENDENT_MATCHED_PROJECT_SFT_CHECKPOINT"
MAIN_ADAPTATION_PATH = "BASE_PLUS_PPO_LORA"

STAGE_ORDER = (
    "candidate_pool",
    "common_verification",
    "selector_freeze",
    "heldout_test",
    "final_audit",
)

REQUIRED_STAGE_OUTPUT = {
    "candidate_pool": "candidate_pool.jsonl",
    "common_verification": "verification_manifest.json",
    "selector_freeze": "selector_manifest.json",
    "heldout_test": "heldout_test_metrics.json",
    "final_audit": "final_audit.json",
}


def _git_sha(value: object, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 40 or any(c not in "0123456789abcdef" for c in normalized):
        raise LLMAblationContractError(f"{field} must be one 40-character Git SHA")
    return normalized


def _uuid4(value: object) -> str:
    normalized = str(value or "").strip()
    try:
        parsed = UUID(normalized)
    except ValueError as exc:
        raise LLMAblationContractError("run_id must be UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != normalized.lower():
        raise LLMAblationContractError("run_id must be canonical UUIDv4")
    return normalized.lower()


def _physical_file(path_like: object, sha256: object, *, role: str) -> Path:
    path = Path(str(path_like or ""))
    if not path.is_absolute() or path.is_symlink():
        raise LLMAblationContractError(f"{role} must be an absolute physical file")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise LLMAblationContractError(f"{role} is not a regular file")
    expected = require_sha256(sha256, field=f"{role}.sha256")
    if sha256_file(resolved) != expected:
        raise LLMAblationContractError(f"{role} SHA256 changed")
    return resolved


def derive_core_reference(reference: BACEReferenceEvidence) -> dict[str, Any]:
    """Project validated v2 evidence onto the truthful core-v1 design.

    The v2 evidence loader has already proven that the adapter was initialized
    from scratch and received 300 PPO updates, with no matched project SFT.
    This function therefore exposes the real main row instead of relabeling it
    as ``SFT+PPO``.
    """

    payload = reference.payload
    if payload.get("main_policy_scientific_name") != (
        "CHEMLLM_7B_OFF_THE_SHELF_PLUS_FRESH_LORA_PPO"
    ):
        raise LLMAblationContractError("main policy is not the audited base-plus-PPO policy")
    ppo = payload.get("ppo")
    if not isinstance(ppo, Mapping) or ppo.get("optimizer_updates") != 300:
        raise LLMAblationContractError("main PPO evidence is incomplete")
    variants = payload.get("stage_variants")
    if not isinstance(variants, Mapping):
        raise LLMAblationContractError("reference stage variants are absent")
    if variants.get("A2_CHEMLLM_7B_PROJECT_SFT", {}).get("status") != (
        "BLOCKED_MISSING_MATCHED_SFT_CHECKPOINT"
    ):
        raise LLMAblationContractError("reference no longer proves matched SFT is absent")
    core: dict[str, Any] = {
        "schema_version": "bace_ours_llm_core_reference_v1",
        "status": "PASS",
        "source_reference": {
            "path": reference.path,
            "file_sha256": reference.file_sha256,
            "self_sha256": reference.self_sha256,
        },
        "dataset": "bace",
        "method": "ours",
        "project_sft_checkpoint_exists": False,
        "main_adaptation_path": MAIN_ADAPTATION_PATH,
        "main_policy_scientific_name": payload["main_policy_scientific_name"],
        "variants": {
            CoreLLMVariant.BRICS_FIXED.value: "READY_AFTER_BRICS_ARTIFACT_BINDING",
            CoreLLMVariant.CHEMLLM_7B_OFF_THE_SHELF.value: (
                "READY_AFTER_7B_RUNTIME_EVIDENCE"
            ),
            CoreLLMVariant.CHEMLLM_7B_PPO_LORA_MAIN.value: (
                "ADOPT_EXISTING_MAIN_RESULT"
            ),
            CoreLLMVariant.CHEMLLM_2B_OFF_THE_SHELF.value: (
                "BLOCKED_UNTIL_2B_ISOLATED_LOAD_AND_PARAMETER_REPORT_PASS"
            ),
        },
        "sft_auxiliary": {
            "enabled": False,
            "state": SFT_AUXILIARY_STATE,
            "reason": SFT_AUXILIARY_REASON,
        },
        "scale_comparison": (
            "CHEMLLM_2B_OFF_THE_SHELF_vs_CHEMLLM_7B_OFF_THE_SHELF"
        ),
    }
    core["core_reference_sha256"] = canonical_json_sha256(core)
    return core


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    path: str
    sha256: str
    role: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, role: str) -> "ArtifactIdentity":
        if not isinstance(payload, Mapping) or set(payload) != {"path", "sha256"}:
            raise LLMAblationContractError(f"{role} must contain only path and sha256")
        path = _physical_file(payload["path"], payload["sha256"], role=role)
        return cls(str(path), str(payload["sha256"]), role)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StagePlan:
    name: str
    action: str
    argv: tuple[str, ...]
    adopted_artifacts: tuple[ArtifactIdentity, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StagePlan":
        name = str(payload.get("name") or "")
        if name not in STAGE_ORDER:
            raise LLMAblationContractError(f"unknown core stage: {name}")
        action = str(payload.get("action") or "")
        if action not in {"EXECUTE", "ADOPT"}:
            raise LLMAblationContractError(f"{name}.action must be EXECUTE or ADOPT")
        raw_argv = payload.get("argv", [])
        if not isinstance(raw_argv, list) or any(
            not isinstance(value, str) or not value for value in raw_argv
        ):
            raise LLMAblationContractError(f"{name}.argv must be a string list")
        raw_adopted = payload.get("adopted_artifacts", [])
        if not isinstance(raw_adopted, list):
            raise LLMAblationContractError(f"{name}.adopted_artifacts must be a list")
        adopted = tuple(
            ArtifactIdentity.from_mapping(item, role=f"{name}.adopted[{index}]")
            for index, item in enumerate(raw_adopted)
        )
        if action == "EXECUTE" and (not raw_argv or adopted):
            raise LLMAblationContractError(f"{name} EXECUTE requires argv and no adoptions")
        if action == "ADOPT" and (raw_argv or not adopted):
            raise LLMAblationContractError(f"{name} ADOPT requires evidence and no argv")
        return cls(name, action, tuple(raw_argv), adopted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "action": self.action,
            "argv": list(self.argv),
            "adopted_artifacts": [item.to_dict() for item in self.adopted_artifacts],
        }


@dataclass(frozen=True, slots=True)
class CoreRunSpec:
    run_id: str
    variant: CoreLLMVariant
    output_root: str
    execution_commit: str
    reference_contract: ArtifactIdentity
    matrix_authority: ArtifactIdentity
    adapter_topology: str
    stages: tuple[StagePlan, ...]
    checkpoint_resume_supported: bool
    run_spec_sha256: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CoreRunSpec":
        if payload.get("schema_version") != "llm_core_variant_run_spec_v1":
            raise LLMAblationContractError("core run-spec schema changed")
        variant = CoreLLMVariant(str(payload.get("variant") or ""))
        run_id = _uuid4(payload.get("run_id"))
        output = Path(str(payload.get("output_root") or ""))
        if not output.is_absolute() or output.is_symlink() or run_id not in output.name:
            raise LLMAblationContractError("fresh output_root must be absolute and include run UUID")
        topology = str(payload.get("adapter_topology") or "")
        expected_topology = {
            CoreLLMVariant.BRICS_FIXED: "NO_MODEL",
            CoreLLMVariant.CHEMLLM_7B_OFF_THE_SHELF: "BASE_ONLY_NO_ADAPTER",
            CoreLLMVariant.CHEMLLM_7B_PPO_LORA_MAIN: (
                "BASE_PLUS_PPO_LORA_ADOPT_MAIN"
            ),
            CoreLLMVariant.CHEMLLM_2B_OFF_THE_SHELF: "BASE_ONLY_NO_ADAPTER",
        }[variant]
        if topology != expected_topology:
            raise LLMAblationContractError(f"{variant.value} adapter topology changed")
        raw_stages = payload.get("stages")
        if not isinstance(raw_stages, list):
            raise LLMAblationContractError("stages must be a list")
        stages = tuple(StagePlan.from_mapping(item) for item in raw_stages)
        if tuple(item.name for item in stages) != STAGE_ORDER:
            raise LLMAblationContractError("core stages must use the fixed scientific order")
        if variant is CoreLLMVariant.CHEMLLM_7B_PPO_LORA_MAIN:
            if any(item.action != "ADOPT" for item in stages):
                raise LLMAblationContractError("main PPO row must be adopted without retraining")
        elif variant is CoreLLMVariant.BRICS_FIXED:
            if stages[0].action != "ADOPT" or any(
                item.action != "EXECUTE" for item in stages[1:]
            ):
                raise LLMAblationContractError(
                    "BRICS must adopt only its train-only pool then execute common downstream"
                )
        elif any(item.action != "EXECUTE" for item in stages):
            raise LLMAblationContractError(
                "off-the-shelf rows must generate and execute every downstream stage"
            )
        if payload.get("checkpoint_resume_supported") is not True:
            raise LLMAblationContractError("checkpoint/resume must be enabled")
        claimed = require_sha256(payload.get("run_spec_sha256"), field="run_spec_sha256")
        body = dict(payload)
        body.pop("run_spec_sha256")
        if canonical_json_sha256(body) != claimed:
            raise LLMAblationContractError("run spec self-hash changed")
        return cls(
            run_id=run_id,
            variant=variant,
            output_root=str(output),
            execution_commit=_git_sha(payload.get("execution_commit"), field="execution_commit"),
            reference_contract=ArtifactIdentity.from_mapping(
                payload.get("reference_contract", {}), role="reference_contract"
            ),
            matrix_authority=ArtifactIdentity.from_mapping(
                payload.get("matrix_authority", {}), role="matrix_authority"
            ),
            adapter_topology=topology,
            stages=stages,
            checkpoint_resume_supported=True,
            run_spec_sha256=claimed,
        )


def load_core_run_spec(path: str | Path, expected_sha256: str) -> CoreRunSpec:
    physical = _physical_file(path, expected_sha256, role="core_run_spec")
    payload = json.loads(physical.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise LLMAblationContractError("core run spec must be one JSON object")
    return CoreRunSpec.from_mapping(payload)


def _reference_identities(value: object) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    if isinstance(value, Mapping):
        path = value.get("path")
        sha = value.get("sha256", value.get("sha"))
        if isinstance(path, str) and isinstance(sha, str):
            try:
                found.add((str(Path(path)), require_sha256(sha, field="reference artifact")))
            except LLMAblationContractError:
                pass
        for child in value.values():
            found.update(_reference_identities(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_reference_identities(child))
    return found


def validate_variant_artifact_bindings(
    spec: CoreRunSpec,
    reference: BACEReferenceEvidence,
) -> None:
    """Bind adoption-only rows to audited science rather than arbitrary files."""

    if spec.variant is CoreLLMVariant.BRICS_FIXED:
        identities = {Path(item.path).name: item for item in spec.stages[0].adopted_artifacts}
        required = {
            "brics_proposal_pool.jsonl",
            "brics_proposal_manifest.json",
            "brics_vocab_manifest.json",
            "brics_proposal_shortfall_receipt.json",
        }
        if not required.issubset(identities):
            raise LLMAblationContractError(
                "BRICS adoption requires pool, proposal/vocabulary manifests and shortfall receipt"
            )
        proposal = json.loads(
            Path(identities["brics_proposal_manifest.json"].path).read_text(encoding="utf-8")
        )
        vocabulary = json.loads(
            Path(identities["brics_vocab_manifest.json"].path).read_text(encoding="utf-8")
        )
        shortfall = json.loads(
            Path(identities["brics_proposal_shortfall_receipt.json"].path).read_text(
                encoding="utf-8"
            )
        )
        if any(
            payload.get("status") != "PASS"
            for payload in (proposal, vocabulary, shortfall)
        ):
            raise LLMAblationContractError("BRICS adopted manifests are not PASS")
        if (
            vocabulary.get("source_split") != "train"
            or vocabulary.get("calibration_loaded") is not False
            or vocabulary.get("test_loaded") is not False
            or vocabulary.get("oracle_fields_read") != []
            or proposal.get("oracle_used") is not False
            or proposal.get("calibration_loaded") is not False
            or proposal.get("test_loaded") is not False
            or shortfall.get("candidate_duplication_used") is not False
            or shortfall.get("oracle_ranking_used") is not False
            or shortfall.get("shortfall_is_not_backfilled") is not True
        ):
            raise LLMAblationContractError("BRICS train-only/no-oracle/shortfall contract changed")
        pool_identity = proposal.get("candidate_pool")
        pool = identities["brics_proposal_pool.jsonl"]
        if not isinstance(pool_identity, Mapping) or (
            str(Path(str(pool_identity.get("path") or "")).resolve()) != pool.path
            or pool_identity.get("sha256") != pool.sha256
        ):
            raise LLMAblationContractError("BRICS proposal manifest does not bind adopted pool")
        reference_identity = proposal.get("reference_contract")
        if not isinstance(reference_identity, Mapping) or (
            reference_identity.get("sha256") != reference.file_sha256
        ):
            raise LLMAblationContractError("BRICS manifest does not bind the BACE reference")
        return

    if spec.variant is not CoreLLMVariant.CHEMLLM_7B_PPO_LORA_MAIN:
        return
    allowed = _reference_identities(reference.payload)
    final_root = Path(str(reference.payload.get("main_final_root") or ""))
    final_sha = reference.payload.get("main_final_audit_sha")
    if final_root.is_absolute() and isinstance(final_sha, str):
        allowed.add((str(final_root / "final_artifact_audit.json"), final_sha))
    keywords = {
        "candidate_pool": ("candidate", "merge"),
        "common_verification": ("verification", "matrix"),
        "selector_freeze": ("selector",),
        "heldout_test": ("evaluation", "test"),
        "final_audit": ("final_artifact_audit",),
    }
    for stage in spec.stages:
        for artifact in stage.adopted_artifacts:
            if (artifact.path, artifact.sha256) not in allowed:
                raise LLMAblationContractError(
                    f"{stage.name} adoption is not an identity in the BACE reference"
                )
        if not any(
            token in Path(artifact.path).name.lower()
            for artifact in stage.adopted_artifacts
            for token in keywords[stage.name]
        ):
            raise LLMAblationContractError(
                f"{stage.name} adoption lacks a stage-specific audited artifact"
            )


def load_authorized_launch_decision(
    path: str | Path,
    expected_sha256: str,
    *,
    spec: CoreRunSpec,
    execution_commit: str,
) -> dict[str, Any]:
    """Reopen a status-produced decision and all mutable evidence it binds."""

    physical = _physical_file(path, expected_sha256, role="core_launch_decision")
    payload = json.loads(physical.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LLMAblationContractError("launch decision must be one object")
    claimed = require_sha256(
        payload.pop("launch_decision_sha256", None), field="launch_decision_sha256"
    )
    if canonical_json_sha256(payload) != claimed:
        raise LLMAblationContractError("launch decision self-hash changed")
    payload["launch_decision_sha256"] = claimed
    if (
        payload.get("schema_version") != "llm_core_launch_decision_v1"
        or payload.get("science_launch_allowed") is not True
        or payload.get("run_spec_sha256") != spec.run_spec_sha256
        or payload.get("variant") != spec.variant.value
        or payload.get("execution_commit") != execution_commit
    ):
        raise LLMAblationContractError("launch decision does not authorize this run")
    for key, identity in (
        ("run_spec", payload.get("run_spec")),
        ("main_snapshot", payload.get("main_snapshot")),
        ("canonical_owner_registry", payload.get("canonical_owner_registry")),
        ("matrix_authority", payload.get("matrix_authority")),
        ("reference_contract", payload.get("reference_contract")),
        ("science_entrypoint", payload.get("science_entrypoint")),
        ("authorization_receipt", payload.get("authorization_receipt")),
    ):
        if not isinstance(identity, Mapping):
            raise LLMAblationContractError(f"launch decision lacks {key} evidence")
        _physical_file(identity.get("path"), identity.get("sha256"), role=key)
    runtime_files = payload.get("runtime_evidence_files")
    if not isinstance(runtime_files, Mapping):
        raise LLMAblationContractError("launch decision lacks runtime evidence inventory")
    for key, identity in runtime_files.items():
        if not isinstance(identity, Mapping):
            raise LLMAblationContractError(f"runtime evidence {key} is malformed")
        _physical_file(
            identity.get("path"), identity.get("sha256"), role=f"runtime_evidence.{key}"
        )
    if payload["matrix_authority"]["sha256"] != spec.matrix_authority.sha256:
        raise LLMAblationContractError("launch decision matrix authority differs from run spec")
    if payload["reference_contract"]["sha256"] != spec.reference_contract.sha256:
        raise LLMAblationContractError("launch decision reference differs from run spec")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_payload(
    spec: CoreRunSpec,
    *,
    state: str,
    completed: Sequence[str],
    receipts: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "llm_core_variant_checkpoint_v1",
        "run_id": spec.run_id,
        "variant": spec.variant.value,
        "run_spec_sha256": spec.run_spec_sha256,
        "execution_commit": spec.execution_commit,
        "state": state,
        "completed_stages": list(completed),
        "next_stage": STAGE_ORDER[len(completed)] if len(completed) < len(STAGE_ORDER) else None,
        "stage_receipts": dict(receipts),
    }
    payload["checkpoint_sha256"] = canonical_json_sha256(payload)
    return payload


def _load_checkpoint(path: Path, spec: CoreRunSpec) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = require_sha256(payload.pop("checkpoint_sha256", None), field="checkpoint_sha256")
    if canonical_json_sha256(payload) != claimed:
        raise LLMAblationContractError("checkpoint self-hash changed")
    payload["checkpoint_sha256"] = claimed
    expected = {
        "schema_version": "llm_core_variant_checkpoint_v1",
        "run_id": spec.run_id,
        "variant": spec.variant.value,
        "run_spec_sha256": spec.run_spec_sha256,
        "execution_commit": spec.execution_commit,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise LLMAblationContractError("checkpoint does not belong to this immutable run")
    completed = payload.get("completed_stages")
    if not isinstance(completed, list) or tuple(completed) != STAGE_ORDER[: len(completed)]:
        raise LLMAblationContractError("checkpoint stage cursor is invalid")
    return payload


def _verify_stage_output(stage: str, root: Path) -> list[dict[str, Any]]:
    expected = root / REQUIRED_STAGE_OUTPUT[stage]
    if expected.is_symlink() or not expected.is_file() or expected.stat().st_size == 0:
        raise LLMAblationContractError(f"{stage} did not produce {expected.name}")
    if stage != "candidate_pool":
        payload = json.loads(expected.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise LLMAblationContractError(f"{expected.name} must contain one object")
        if stage == "selector_freeze" and (
            payload.get("selection_frozen") is not True
            or payload.get("test_loaded") is not False
        ):
            raise LLMAblationContractError("selector must freeze before held-out test")
        if stage == "heldout_test" and payload.get("selection_frozen_before_test") is not True:
            raise LLMAblationContractError("held-out test lacks freeze-order evidence")
        if stage == "final_audit" and payload.get("status") != "PASS":
            raise LLMAblationContractError("final audit is not PASS")
    return [{"path": str(expected), "sha256": sha256_file(expected), "size": expected.stat().st_size}]


def run_core_variant(
    spec: CoreRunSpec,
    *,
    resume: bool,
    live_snapshot_loader: Callable[[], EarlyLaunchSnapshot] | None = None,
) -> dict[str, Any]:
    """Execute or resume one variant; pause only at a committed stage boundary."""

    root = Path(spec.output_root)
    checkpoint_path = root / "checkpoint.json"
    if resume:
        if not root.is_dir() or not checkpoint_path.is_file():
            raise LLMAblationContractError("resume requires an existing committed checkpoint")
        checkpoint = _load_checkpoint(checkpoint_path, spec)
    else:
        if root.exists():
            raise LLMAblationContractError("fresh core run refuses an existing output root")
        root.mkdir(parents=True)
        checkpoint = _checkpoint_payload(spec, state="READY", completed=(), receipts={})
        _atomic_json(checkpoint_path, checkpoint)

    lock_path = root / ".writer.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LLMAblationContractError("another writer owns this run root") from exc
        completed = list(checkpoint["completed_stages"])
        receipts = dict(checkpoint.get("stage_receipts", {}))
        for stage in spec.stages[len(completed) :]:
            if live_snapshot_loader is not None:
                snapshot = live_snapshot_loader()
                action = main_priority_runtime_action(
                    snapshot, ablation_running=True, at_safe_checkpoint=True
                )
                if action == "GRACEFUL_PAUSE_AND_RELEASE_GPU":
                    paused = _checkpoint_payload(
                        spec, state="PAUSED_MAIN_PRIORITY", completed=completed, receipts=receipts
                    )
                    _atomic_json(checkpoint_path, paused)
                    return paused
            if stage.action == "EXECUTE":
                environment = dict(os.environ)
                environment.update(
                    {
                        "LLM_ABLATION_RUN_ID": spec.run_id,
                        "LLM_ABLATION_VARIANT": spec.variant.value,
                        "LLM_ABLATION_OUTPUT_ROOT": str(root),
                        "LLM_ABLATION_STAGE": stage.name,
                    }
                )
                subprocess.run(list(stage.argv), check=True, env=environment)
                artifacts = _verify_stage_output(stage.name, root)
                receipt = {"action": "EXECUTE", "argv_sha256": canonical_json_sha256({"argv": list(stage.argv)}), "artifacts": artifacts}
            else:
                receipt = {
                    "action": "ADOPT",
                    "artifacts": [item.to_dict() for item in stage.adopted_artifacts],
                }
            receipts[stage.name] = receipt
            completed.append(stage.name)
            running = _checkpoint_payload(
                spec,
                state="PASS" if len(completed) == len(STAGE_ORDER) else "RUNNING",
                completed=completed,
                receipts=receipts,
            )
            _atomic_json(checkpoint_path, running)
        terminal = _load_checkpoint(checkpoint_path, spec)
        if terminal.get("state") == "PASS":
            manifest: dict[str, Any] = {
                "schema_version": "llm_core_variant_run_manifest_v1",
                "status": "PASS",
                "run_id": spec.run_id,
                "variant": spec.variant.value,
                "adapter_topology": spec.adapter_topology,
                "main_adaptation_path": MAIN_ADAPTATION_PATH,
                "project_sft_checkpoint_exists": False,
                "sft_auxiliary_state": SFT_AUXILIARY_STATE,
                "sft_auxiliary_reason": SFT_AUXILIARY_REASON,
                "run_spec_sha256": spec.run_spec_sha256,
                "checkpoint_sha256": terminal["checkpoint_sha256"],
                "stage_order": list(STAGE_ORDER),
                "selector_frozen_before_heldout_test": True,
                "science_retrained": (
                    spec.variant is not CoreLLMVariant.CHEMLLM_7B_PPO_LORA_MAIN
                ),
                "main_result_adopted": (
                    spec.variant is CoreLLMVariant.CHEMLLM_7B_PPO_LORA_MAIN
                ),
                "main_result_retraining_permitted": False,
                "stage_receipts": terminal["stage_receipts"],
            }
            manifest["run_manifest_sha256"] = canonical_json_sha256(manifest)
            _atomic_json(root / "run_manifest.json", manifest)
        return terminal


def status_core_run(spec: CoreRunSpec) -> dict[str, Any]:
    root = Path(spec.output_root)
    checkpoint = root / "checkpoint.json"
    if not checkpoint.exists():
        return {
            "state": "NOT_STARTED",
            "variant": spec.variant.value,
            "run_id": spec.run_id,
            "checkpoint_resume_supported": True,
        }
    return _load_checkpoint(checkpoint, spec)


__all__ = [
    "CORE_VARIANT_ORDER",
    "CoreLLMVariant",
    "CoreRunSpec",
    "MAIN_ADAPTATION_PATH",
    "SFT_AUXILIARY_REASON",
    "SFT_AUXILIARY_STATE",
    "STAGE_ORDER",
    "derive_core_reference",
    "load_authorized_launch_decision",
    "load_core_run_spec",
    "run_core_variant",
    "status_core_run",
    "validate_variant_artifact_bindings",
]
