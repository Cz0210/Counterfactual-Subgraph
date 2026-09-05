"""Independent terminal verifier for TasteMolNet T12 20k generation.

This verifier deliberately closes only the train-side official VRRW generation
substrate.  It cannot emit the paper-cell ``[TASTE_GCF_PASS]`` marker: the
externally pinned calibration selector, frozen global set, held-out test and
standardized exports remain separate required stages.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from src.baselines.tastemolnet_gcf_candidate_store import (
    reopen_native_candidate_snapshot,
)
from src.baselines.tastemolnet_gcf_full import (
    PRODUCTION_RECEIPT_SCHEMA,
    PRODUCTION_RUN_SCHEMA,
)
from src.baselines.tastemolnet_gcf_full_resume import (
    PRODUCTION_CHECKPOINT_CURSORS,
    PRODUCTION_TOTAL_STEPS,
    STAGE,
    TasteGCFFullResumeError,
    production_checkpoint_identity,
    production_transition_contract_sha256,
    reopen_checkpoint,
    validate_checkpoint_identity,
)
from src.baselines.tastemolnet_gcf_production_state import (
    T12CompactHistoryJournal,
    T12ProductionBounds,
    TasteT12ProductionStateError,
)
from src.baselines.tastemolnet_gcf_smoke import _semantic_sha256


GENERATION_VERIFY_SCHEMA = "tastemolnet_t12_generation_verification_v1"
GENERATION_PASS_MARKER = "[TASTE_T12_GCF_GENERATION_PASS]"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute(value: str | Path, *, field: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise TasteGCFFullResumeError(f"{field} must be normalized and absolute")
    if path.resolve(strict=True) != path or path.is_symlink():
        raise TasteGCFFullResumeError(f"{field} is an alias")
    return path


def _json(path: Path, *, field: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise TasteGCFFullResumeError(f"{field} is not one physical file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteGCFFullResumeError(f"{field} is unreadable") from exc
    if type(value) is not dict:
        raise TasteGCFFullResumeError(f"{field} is not one JSON object")
    return value


def _publish(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _torch_load(path: Path, *, torch: Any) -> Any:
    info = path.stat()
    if (
        path.resolve(strict=True) != path
        or path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
    ):
        raise TasteGCFFullResumeError("T12 native result is not one physical file")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older supported Torch
        return torch.load(path, map_location="cpu")


def _verify_history(
    *, snapshot: Mapping[str, Any], output_root: Path, cursor: int
) -> dict[str, Any]:
    try:
        bounds = T12ProductionBounds.from_dict(snapshot.get("bounds"))
        journal = T12CompactHistoryJournal(
            root=snapshot.get("history_root"),
            index_root=(output_root / f"history_index_{cursor:08d}").resolve(),
            bounds=bounds,
            contract_sha256=snapshot.get("contract_sha256"),
            attempt_id=snapshot.get("attempt_id"),
            generation_token=snapshot.get("generation_token"),
            resume_snapshot=snapshot,
            open_writer=False,
        )
        try:
            observed = journal.checkpoint_state()
        finally:
            journal.close()
    except (TasteT12ProductionStateError, AttributeError) as exc:
        raise TasteGCFFullResumeError(
            f"T12 {cursor} compact history did not reopen"
        ) from exc
    if observed != dict(snapshot):
        raise TasteGCFFullResumeError(
            f"T12 {cursor} compact history prefix changed"
        )
    return {
        "observation_count": observed["observation_count"],
        "first_seen_graph_count": observed["first_seen_graph_count"],
        "chain_head": observed["chain_head"],
        "segments": [dict(row) for row in observed["segments"]],
    }


def _receipt(
    root: Path, *, cursor: int, expected_attempt: str, expected_token: str
) -> tuple[dict[str, Any], Path]:
    path = root / f"generation_receipt_{cursor:08d}.json"
    value = _json(path, field=f"T12 {cursor} generation receipt")
    if (
        value.get("schema_version") != PRODUCTION_RECEIPT_SCHEMA
        or value.get("status") != "GENERATION_CHECKPOINT_COMMITTED"
        or value.get("stage") != STAGE
        or value.get("attempt_id") != expected_attempt
        or value.get("generation_token") != expected_token
        or value.get("checkpoint_cursor") != cursor
        or value.get("train_loaded") is not True
        or value.get("calibration_loaded") is not False
        or value.get("test_loaded") is not False
        or value.get("rf_oracle_used") is not False
        or value.get("paper_cell_pass") is not False
    ):
        raise TasteGCFFullResumeError(
            f"T12 {cursor} generation receipt semantics changed"
        )
    return value, path


def verify_t12_generation(
    *,
    production_root: str | Path,
    verification_root: str | Path,
    checkpoint_cursors: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Reopen every configured checkpoint, journal, result, and candidate pool."""

    import torch

    root = _absolute(production_root, field="T12 production root")
    output = Path(verification_root).expanduser()
    if not output.is_absolute() or Path(os.path.abspath(output)) != output:
        raise TasteGCFFullResumeError(
            "T12 verification root must be normalized and absolute"
        )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    if output.resolve(strict=True) != output or output.is_symlink():
        raise TasteGCFFullResumeError("T12 verification root is an alias")

    cursors = tuple(
        sorted(PRODUCTION_CHECKPOINT_CURSORS)
        if checkpoint_cursors is None
        else checkpoint_cursors
    )
    if (
        not cursors
        or tuple(sorted(set(cursors))) != cursors
        or any(type(cursor) is not int or cursor <= 0 for cursor in cursors)
        or cursors[-1] != PRODUCTION_TOTAL_STEPS
        or frozenset(cursors) != PRODUCTION_CHECKPOINT_CURSORS
    ):
        raise TasteGCFFullResumeError("T12 verification checkpoint cadence changed")

    run_path = root / "run_identity.json"
    run = _json(run_path, field="T12 production run identity")
    if run.get("schema_version") != PRODUCTION_RUN_SCHEMA:
        raise TasteGCFFullResumeError("T12 production run schema changed")
    identity_template = validate_checkpoint_identity(run.get("identity_template"))
    if (
        identity_template["purpose"] != "production"
        or identity_template["checkpoint_cursor"] != cursors[0]
        or identity_template["total_steps"] != PRODUCTION_TOTAL_STEPS
        or run.get("transition_contract_sha256")
        != production_transition_contract_sha256(identity_template)
        or run.get("train_loaded") is not True
        or run.get("calibration_loaded") is not False
        or run.get("test_loaded") is not False
        or run.get("rf_oracle_used") is not False
    ):
        raise TasteGCFFullResumeError("T12 production identity semantics changed")
    attempt = identity_template["attempt_id"]
    token = identity_template["generation_token"]

    checkpoint_facts: dict[int, dict[str, Any]] = {}
    trace_prefix: list[str] | None = None
    previous_cursor = 0
    for cursor in cursors:
        receipt, receipt_path = _receipt(
            root, cursor=cursor, expected_attempt=attempt, expected_token=token
        )
        manifest = Path(receipt["checkpoint_manifest"])
        expected_manifest = root / "checkpoints" / (
            f"checkpoint-{cursor:08d}.manifest.json"
        )
        if (
            manifest != expected_manifest
            or _sha256_file(manifest) != receipt["checkpoint_manifest_sha256"]
        ):
            raise TasteGCFFullResumeError(
                f"T12 {cursor} checkpoint receipt path/hash changed"
            )
        checkpoint = reopen_checkpoint(
            manifest,
            expected_identity=production_checkpoint_identity(
                identity_template, checkpoint_cursor=cursor
            ),
            torch=torch,
        )
        official = checkpoint["state"]["official"]
        transitions = official["transitions"]
        history = checkpoint["state"]["bridge"].get("history")
        history_fact = _verify_history(
            snapshot=history, output_root=output, cursor=cursor
        )
        traversed = list(official["traversed_hashes"])
        if cursor == cursors[0]:
            trace_prefix = traversed
        elif trace_prefix != traversed[: cursors[0]]:
            raise TasteGCFFullResumeError(
                "T12 later trace does not retain the first committed prefix"
            )
        native_path = Path(receipt["official_native_result"])
        expected_segment = root / (
            f"segment-{previous_cursor + 1:05d}-{cursor:05d}"
        ) / "results/tastemolnet/runs/counterfactuals.pt"
        if (
            native_path != expected_segment
            or _sha256_file(native_path)
            != receipt["official_native_result_sha256"]
        ):
            raise TasteGCFFullResumeError(
                f"T12 {cursor} official native result bytes changed"
            )
        native = _torch_load(native_path, torch=torch)
        expected_native = {
            "graph_map": official["graph_map"],
            "graph_index_map": official["graph_index_map"],
            "counterfactual_candidates": official[
                "counterfactual_candidates"
            ],
            "MAX_COUNTERFACTUAL_SIZE": official["MAX_COUNTERFACTUAL_SIZE"],
            "traversed_hashes": official["traversed_hashes"],
            "input_graphs_covered": official["input_graphs_covered"],
        }
        native_semantic = _semantic_sha256(native)
        if (
            type(native) is not dict
            or set(native) != set(expected_native)
            or native_semantic != _semantic_sha256(expected_native)
            or native_semantic
            != receipt["official_native_result_semantic_sha256"]
        ):
            raise TasteGCFFullResumeError(
                f"T12 {cursor} official native result semantics changed"
            )
        checkpoint_facts[cursor] = {
            "receipt": str(receipt_path),
            "receipt_sha256": _sha256_file(receipt_path),
            "checkpoint_manifest": str(manifest),
            "checkpoint_manifest_sha256": _sha256_file(manifest),
            "identity_sha256": checkpoint["identity_sha256"],
            "state_sha256": checkpoint["state_sha256"],
            "rng_sha256": checkpoint["rng_sha256"],
            "transition_chain_sha256": transitions["chain_sha256"],
            "transition_store_bytes": transitions["committed_store_bytes"],
            "transition_segments": [dict(row) for row in transitions["segments"]],
            "history": history_fact,
            "official_native_result": str(native_path),
            "official_native_result_sha256": _sha256_file(native_path),
            "official_native_result_semantic_sha256": native_semantic,
            "graph_map_sha256": _semantic_sha256(official["graph_map"]),
            "graph_index_map_sha256": _semantic_sha256(
                official["graph_index_map"]
            ),
            "candidate_order_sha256": _semantic_sha256(
                official["counterfactual_candidates"]
            ),
            "candidate_count": len(official["counterfactual_candidates"]),
        }
        del native, expected_native, official, checkpoint
        gc.collect()
        previous_cursor = cursor

    terminal = checkpoint_facts[cursors[-1]]
    for cursor in cursors[:-1]:
        prefix = checkpoint_facts[cursor]
        if (
            terminal["transition_segments"][: len(prefix["transition_segments"])]
            != prefix["transition_segments"]
            or terminal["history"]["segments"][: len(prefix["history"]["segments"])]
            != prefix["history"]["segments"]
        ):
            raise TasteGCFFullResumeError(
                f"T12 terminal journals do not retain their exact {cursor} prefixes"
            )

    terminal_receipt, _unused = _receipt(
        root,
        cursor=cursors[-1],
        expected_attempt=attempt,
        expected_token=token,
    )
    candidate_manifest = Path(terminal_receipt["candidate_manifest"])
    if (
        candidate_manifest.parent != root / "native_candidates"
        or _sha256_file(candidate_manifest)
        != terminal_receipt["candidate_manifest_sha256"]
    ):
        raise TasteGCFFullResumeError("T12 native candidate manifest changed")
    candidates = reopen_native_candidate_snapshot(
        candidate_manifest,
        expected_contract_sha256=production_transition_contract_sha256(
            identity_template
        ),
        expected_attempt_id=attempt,
        expected_generation_token=token,
        torch=torch,
    )
    if (
        _semantic_sha256(candidates["graph_map"])
        != terminal["graph_map_sha256"]
        or _semantic_sha256(candidates["graph_index_map"])
        != terminal["graph_index_map_sha256"]
        or _semantic_sha256(candidates["counterfactual_candidates"])
        != terminal["candidate_order_sha256"]
        or len(candidates["counterfactual_candidates"])
        != terminal["candidate_count"]
    ):
        raise TasteGCFFullResumeError(
            "T12 lossless native candidate snapshot differs from 20k"
        )

    audit = {
        "schema_version": GENERATION_VERIFY_SCHEMA,
        "status": "GENERATION_PASS",
        "passed": True,
        "stage": STAGE,
        "marker": GENERATION_PASS_MARKER,
        "production_root": str(root),
        "run_identity": str(run_path),
        "run_identity_sha256": _sha256_file(run_path),
        "attempt_id": attempt,
        "generation_token": token,
        "checkpoint_cursors": list(cursors),
        "checkpoints": {str(key): value for key, value in checkpoint_facts.items()},
        "candidate_manifest": str(candidate_manifest),
        "candidate_manifest_sha256": _sha256_file(candidate_manifest),
        "candidate_count": len(candidates["counterfactual_candidates"]),
        "external_transition_store_exact_reopen": True,
        "compact_history_exact_reopen": True,
        "trace_first_checkpoint_prefix_retained": True,
        "trace_10k_prefix_retained": 10_000 in cursors,
        "official_native_result_exact": True,
        "lossless_candidate_persistence": True,
        "generated_to_original_neurosed": True,
        "train_loaded": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "test_used_for_selection": False,
        "rf_oracle_used": False,
        "independent_verifier": True,
        "paper_cell_pass": False,
        "remaining_required_stage": (
            "calibration_only_freeze_then_held_out_test_exports_and_"
            "paper_terminal_verifier"
        ),
    }
    audit_path = output / "generation_verification.json"
    _publish(
        audit_path,
        (json.dumps(audit, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )
    _publish(output / "GENERATION_PASS", (GENERATION_PASS_MARKER + "\n").encode())
    return {**audit, "audit_path": str(audit_path)}


__all__ = [
    "GENERATION_PASS_MARKER",
    "GENERATION_VERIFY_SCHEMA",
    "verify_t12_generation",
]
