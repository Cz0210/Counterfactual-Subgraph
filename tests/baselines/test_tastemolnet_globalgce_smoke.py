from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.baselines.globalgce_mutagenicity_adapter import (
    NativeGenerationResult,
)
from src.baselines import tastemolnet_globalgce_smoke as t8
from src.utils.retained_output_directory import (
    FreshOutputDirectory,
    prepare_terminal_output,
)
from src.utils.managed_execution_v2 import (
    create_managed_attempt,
    create_worker_staging,
    load_verified_gate,
)
from src.utils.tastemolnet_t8_managed_v2 import (
    collect_t8_official_startup_evidence,
    seal_t8_worker_evidence,
    t8_managed_config_hash,
    t8_managed_input_hashes,
    verify_and_publish_t8_sealed,
)
from src.utils.terminal_publisher_v2 import open_sealed_worker_artifact


class _FakeScorer:
    checkpoint_id = "a" * 64
    num_classes = 3
    source_label = 1
    temperature = 1.25

    @staticmethod
    def score_smiles(values):
        rows = []
        for value in values:
            if value == "N":
                probabilities = [0.1, 0.1, 0.8]
            elif value == "C":
                probabilities = [0.8, 0.1, 0.1]
            else:
                probabilities = [0.1, 0.8, 0.1]
            rows.append(
                {
                    "predicted_label": max(
                        range(3), key=lambda index: probabilities[index]
                    ),
                    "probabilities": probabilities,
                    "logits": [value * 2.0 - 1.0 for value in probabilities],
                    "checkpoint_id": "a" * 64,
                    "num_classes": 3,
                    "source_label": 1,
                    "backbone": "gine",
                }
            )
        return rows


def _rule_payload(target: int) -> dict:
    # Branches deliberately have different RHS atoms, so both survive merge.
    rhs = [0.0, 1.0, 0.0] if target == 0 else [0.0, 0.0, 1.0]
    return {
        "candidate_id": f"private-{target}",
        "rule": {
            "native_rule_index": 0,
            "lhs_feature": [[0.0, 1.0, 0.0]],
            "lhs_adjacency": [[0.0]],
            "lhs_edge_attr": [],
            "rhs_feature": [rhs],
            "rhs_adjacency": [[0.0]],
            "rhs_edge_attr": [],
            "atom_symbols": ["C", "N"],
            "bond_names": ["no_edge", "single"],
        },
    }


def _rule_payloads(target: int) -> list[dict]:
    rows = []
    for index in range(20):
        row = _rule_payload(target)
        row["rule"]["lhs_feature"] = [[float(index), 1.0, 0.0]]
        rows.append(row)
    return rows


class _FakeGenerator:
    def __init__(self, target_label: int):
        self.target_label = target_label
        self.calls = 0
        self.completion_calls = 0
        self.planned_checkpoint = None

    @staticmethod
    def _file_evidence(path: Path) -> dict:
        observed = path.stat()
        return {
            "device": int(observed.st_dev),
            "inode": int(observed.st_ino),
            "mode": int(observed.st_mode),
            "uid": int(observed.st_uid),
            "gid": int(observed.st_gid),
            "link_count": int(observed.st_nlink),
            "bytes": int(observed.st_size),
            "mtime_ns": int(observed.st_mtime_ns),
            "ctime_ns": int(observed.st_ctime_ns),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    @staticmethod
    def _replace(path: Path, payload: bytes) -> None:
        temporary = path.with_name(f".{path.name}.fake.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)

    def generate(
        self,
        parents,
        *,
        output_dir,
        seed,
        epochs,
        top_k_native,
        learning_rate,
        dropout,
        device,
        resume,
        generation_chunk_size=32,
        generation_num_workers=0,
        memory_log_every_chunks=1,
        gspan_flush_every=256,
        gspan_max_in_memory_candidates=256,
        gspan_exact_top_k_pruning=False,
        gspan_adoption_proof=None,
        start_parent_offset=0,
        on_training_ready=None,
        on_chunk=None,
        rules_only=False,
        expected_resume_checkpoint=None,
        on_resume_checkpoint=None,
        after_epoch_checkpoint=None,
        on_generation_complete=None,
    ):
        self.calls += 1
        root = Path(output_dir)
        checkpoint_root = root / "globalgce_training_checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        identity = ("b" if self.target_label == 0 else "c") * 64
        checkpoint_path = checkpoint_root / "training_checkpoint.pt"
        heartbeat_path = checkpoint_root / "training_heartbeat.json"
        if self.calls == 1:
            checkpoint_path.write_bytes(f"checkpoint-{self.target_label}".encode())
            heartbeat_path.write_text(
                json.dumps(
                    {
                        "schema_version": "globalgce_epoch_checkpoint_v2",
                        "stage": "training",
                        "next_epoch": 1,
                        "resume_identity_sha256": identity,
                    }
                )
                + "\n"
            )
            checkpoint_file = self._file_evidence(checkpoint_path)
            heartbeat_file = self._file_evidence(heartbeat_path)
            self.planned_checkpoint = checkpoint_file
            after_epoch_checkpoint(
                {
                    "checkpoint_schema_version": (
                        "globalgce_epoch_checkpoint_v2"
                    ),
                    "checkpoint_sha256": checkpoint_file["sha256"],
                    "checkpoint_file": checkpoint_file,
                    "heartbeat_file": heartbeat_file,
                    "epoch": 0,
                    "next_epoch": 1,
                    "resume_identity_sha256": identity,
                    "checkpoint_and_heartbeat_durable": True,
                }
            )
            raise AssertionError("planned callback must interrupt")
        assert expected_resume_checkpoint == self.planned_checkpoint
        on_resume_checkpoint(
            {
                "checkpoint_schema_version": "globalgce_epoch_checkpoint_v2",
                "checkpoint_sha256": self.planned_checkpoint["sha256"],
                "checkpoint_file": self.planned_checkpoint,
                "next_epoch": 1,
                "resume_identity_sha256": identity,
                "rng_state_restored": True,
                "model_state_restored": True,
                "optimizer_state_restored": True,
                "scheduler_state_restored": True,
            }
        )
        self._replace(
            checkpoint_path,
            f"terminal-checkpoint-{self.target_label}".encode(),
        )
        self._replace(
            heartbeat_path,
            json.dumps(
                {
                    "schema_version": "globalgce_epoch_checkpoint_v2",
                    "stage": "complete",
                    "next_epoch": epochs + 1,
                    "resume_identity_sha256": identity,
                }
            ).encode()
            + b"\n",
        )
        model_path = root / "globalgce_model.pt"
        rules_path = root / "globalgce_rules.pt"
        model_path.write_bytes(b"model")
        rules_path.write_bytes(b"rules")
        core = {
            "trained_once": True,
            "rule_selection_performed_once": True,
            "num_classes": 3,
            "source_label": 1,
            "target_label": self.target_label,
            "prediction_backend": "frozen_gine_differentiable_bridge",
            "rf_oracle_used": False,
            "training_resume_identity_sha256": identity,
            "training_resume_identity": {
                "schema_version": "globalgce_training_resume_identity_v2",
                "dataset": "TasteMolNet",
                "num_classes": 3,
                "source_label": 1,
                "target_label": self.target_label,
            },
            "globalgce_model_checkpoint_sha256": hashlib.sha256(b"model").hexdigest(),
            "rules_checkpoint_sha256": hashlib.sha256(b"rules").hexdigest(),
        }
        (root / "training_core_summary.json").write_text(json.dumps(core) + "\n")
        api_path = root / t8.OFFICIAL_API_SIGNATURE_FILE
        api_path.write_text(
            json.dumps(
                {
                    "schema_version": t8.OFFICIAL_GLOBALGCE_API_SIGNATURE_SCHEMA,
                    "official_globalgce_commit": t8.OFFICIAL_GLOBALGCE_COMMIT,
                    "signatures": dict(
                        t8.PINNED_OFFICIAL_GLOBALGCE_API_SIGNATURES
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        source = Path(__file__).resolve()
        observed = source.stat()
        source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        required_modules = (
            "models.GTGNN",
            "models.GlobalGCE",
            "models.models_utils",
            "models.fsg",
            "models.gSpan.gSpan",
            "torch",
            "torch_geometric",
            "src.baselines.globalgce_mutagenicity_adapter",
            "src.baselines.globalgce_frozen_gine_bridge",
            "src.oracles.gnn_oracle",
        )
        provenance_path = root / t8.PYTHON_MODULE_PROVENANCE_FILE
        provenance_path.write_text(
            json.dumps(
                {
                    "schema_version": (
                        t8.OFFICIAL_GLOBALGCE_MODULE_PROVENANCE_SCHEMA
                    ),
                    "official_globalgce_commit": t8.OFFICIAL_GLOBALGCE_COMMIT,
                    "isolated_python": True,
                    "no_user_site": True,
                    "entries": [
                        {
                            "module": module,
                            "module_file": str(source),
                            "realpath": str(source),
                            "sha256": source_sha256,
                            "device": int(observed.st_dev),
                            "inode": int(observed.st_ino),
                            "bytes": int(observed.st_size),
                            "package_version": (
                                t8.OFFICIAL_GLOBALGCE_COMMIT
                                if module.startswith("models.")
                                else (
                                    "project-source"
                                    if module.startswith("src.")
                                    else "test-package"
                                )
                            ),
                            "expected_roots": [str(source.parent)],
                        }
                        for module in required_modules
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        catalog_path = root / "native_rule_catalog.jsonl"
        catalog_path.write_text(
            "".join(json.dumps(row) + "\n" for row in _rule_payloads(self.target_label))
        )
        candidate = "C" if self.target_label == 0 else "N"
        record = {
            "raw_smiles": candidate,
            "source_parent_id": parents[0].parent_id,
            "source_parent_smiles": parents[0].smiles,
            "source_split": "train",
            "generator_method": "GlobalGCE",
            "native_conversion_ok": True,
            "native_codec_decoded": True,
        }
        summary = {
            "prediction_backend": "frozen_gine_differentiable_bridge",
            "classifier_family": "gine",
            "oracle_backend": "gnn",
            "rf_oracle_used": False,
            "num_classes": 3,
            "frozen_source_label": 1,
            "frozen_target_label": self.target_label,
            "calibration_loaded": False,
            "test_loaded": False,
            "generation_input_split": "train",
            "training_resume_identity_sha256": identity,
            "valid_native_rule_count": 20,
            "native_rule_catalog_sha256": hashlib.sha256(
                catalog_path.read_bytes()
            ).hexdigest(),
            "official_globalgce_commit": t8.OFFICIAL_GLOBALGCE_COMMIT,
            "official_api_signature_sha256": hashlib.sha256(
                api_path.read_bytes()
            ).hexdigest(),
            "python_module_provenance_sha256": hashlib.sha256(
                provenance_path.read_bytes()
            ).hexdigest(),
            "isolated_python": True,
            "no_user_site": True,
        }
        result = NativeGenerationResult([record], summary)
        self.completion_calls += 1
        on_generation_complete()
        return result


def _train_csv() -> bytes:
    rows = ["molecule_id,smiles,label,split", "b0,O,0,train"]
    rows.extend(f"s{index},{'C' * (index + 2)},1,train" for index in range(16))
    rows.append("t0,F,2,train")
    return ("\n".join(rows) + "\n").encode()


def _config() -> t8.TasteGlobalGCESmokeConfig:
    return t8.TasteGlobalGCESmokeConfig()


def _authority() -> dict:
    return {
        "execution": {
            "commit": "1" * 40,
            "tree": "2" * 40,
            "release_config_sha256": "3" * 64,
            "python_entrypoint_sha256": "4" * 64,
            "autodl_wrapper_sha256": "5" * 64,
            "slurm_wrapper_sha256": "6" * 64,
        },
        "managed_execution": {
            "external_authority_schema": (
                t8.MANAGED_V2_EXTERNAL_AUTHORITY_SCHEMA
            ),
            "protocol": t8.MANAGED_V2_PROTOCOL,
            "protocol_source_commit": t8.MANAGED_V2_SOURCE_COMMIT,
            "task_id": "tastemolnet_t8_globalgce_smoke",
            "run_id": "t8-fixture-run",
            "stage": "T8_GLOBALGCE_SMOKE",
            "authority_record_sha256": "7" * 64,
            "active_generation_sha256": "8" * 64,
            "child_identity_sha256": "9" * 64,
            "process_lineage_sha256": "a" * 64,
            "expected_closure_sha256": "b" * 64,
            "gpu_index": 2,
            "gpu_uuid": "GPU-fixture",
            "gpu_lock_mode": "exclusive",
            "auto_terminate_uncontrolled_children": False,
            "same_child_revalidated_at_terminal": True,
        },
        "predecessors": {
            "t2_gate_sha256": "9" * 64,
            "t2_receipt_sha256": "a" * 64,
            "t2_source_evidence_sha256": "b" * 64,
            "t2_binding_sha256": "c" * 64,
            "t3_gate_sha256": "d" * 64,
            "t3_root_inventory_sha256": "e" * 64,
            "t4_gate_sha256": "f" * 64,
            "t4_root_inventory_sha256": "0" * 64,
            "t3_t4_same_t2_binding": True,
            "t3_t4_same_checkpoint": True,
        },
        "frozen_gine": {
            "checkpoint_id": "a" * 64,
            "checkpoint_inventory_sha256": "1" * 64,
            "checkpoint_stat_inventory_sha256": "2" * 64,
            "checkpoint_sha256s_sha256": "3" * 64,
            "feature_schema_sha256": "4" * 64,
            "temperature_scaling_sha256": "5" * 64,
            "num_classes": 3,
            "source_label": 1,
            "oracle_backend": "gnn",
            "classifier_family": "gine",
            "rf_oracle_used": False,
        },
        "train_split": {
            "sha256": "6" * 64,
            "bytes": len(_train_csv()),
            "row_count": 18,
            "label_counts": {"0": 1, "1": 16, "2": 1},
            "split": "train",
        },
        "official_globalgce": {
            "commit": t8.OFFICIAL_GLOBALGCE_COMMIT,
            "tracked_tree_sha256": "8" * 64,
            "source_inventory_sha256": "9" * 64,
            "clean": True,
        },
        "policy": {
            "base_policy_sha256": "a" * 64,
            "downstream_policy_sha256": "b" * 64,
            "research_compute_allowed": True,
            "aggregate_reporting_allowed": True,
            "data_redistribution_allowed": False,
            "hpc_execution_allowed": False,
            "train_loaded": True,
            "external_validation_loaded": False,
            "calibration_loaded": False,
            "test_loaded": False,
        },
    }


class _HeldFixtureTerminalAuthority:
    def __init__(self, value: dict, *, official_startup: dict | None = None) -> None:
        self.value = json.loads(json.dumps(value))
        self.official_startup = (
            None
            if official_startup is None
            else json.loads(json.dumps(official_startup))
        )

    def revalidate_t8_terminal_authority(self) -> dict:
        return json.loads(json.dumps(self.value))

    def revalidate_t8_official_startup_authority(self) -> dict:
        if self.official_startup is None:
            raise t8.TasteGlobalGCESmokeError(
                "fixture has no independent official startup expectation"
            )
        return json.loads(json.dumps(self.official_startup))


def _run_science(tmp_path: Path):
    state = FreshOutputDirectory.create(tmp_path / "state")
    generators = {target: _FakeGenerator(target) for target in (0, 2)}
    science, tree = t8.run_t8_science(
        train_payload=_train_csv(),
        expected_train_row_count=18,
        expected_train_label_counts={"0": 1, "1": 16, "2": 1},
        scorer=_FakeScorer(),
        generator_factory=lambda target: generators[target],
        state_root=state,
        config=_config(),
    )
    return state, tree, science, generators


def _publish_fixture(root: Path, documents: dict) -> None:
    output = FreshOutputDirectory.create(root)
    for name in t8.OUTPUT_PAYLOAD_FILES:
        output.write_new(name, t8._json_document_bytes(documents[name]))
    prepared = prepare_terminal_output(
        output,
        marker_name="PASS",
        marker_payload=(t8.PASS_MARKER + "\n").encode(),
    )
    os.rename(
        ".PASS.prepared",
        "PASS",
        src_dir_fd=output.descriptor,
        dst_dir_fd=output.descriptor,
    )
    output.committed = True
    prepared.close()


def test_real_branch_contract_runs_both_planned_checkpoint_resumes(tmp_path: Path) -> None:
    state, tree, science, generators = _run_science(tmp_path)
    try:
        assert generators[0].calls == 2
        assert generators[2].calls == 2
        assert generators[0].completion_calls == 1
        assert generators[2].completion_calls == 1
        for target in ("0", "2"):
            assert science["branches"][target]["official_globalgce_commit"] == (
                t8.OFFICIAL_GLOBALGCE_COMMIT
            )
            assert science["branches"][target]["isolated_python"] is True
            assert science["branches"][target]["no_user_site"] is True
        assert science["branches"]["0"]["planned_checkpoint_sha256"] == (
            science["branches"]["0"]["resume_checkpoint_sha256"]
        )
        assert science["branches"]["2"]["rng_state_restored"] is True
        assert science["rule_merge"]["merged_unique_rule_count"] == 40
        assert science["strict_flip_validation"]["destination_distribution"] == {
            "0": 1,
            "2": 1,
        }
        assert science["strict_flip_validation"]["strict_flip_count"] == 2
        serialized = json.dumps(science).lower()
        assert "randomforest" not in serialized
        assert "per_example_predictions" not in serialized
    finally:
        tree.close()
        state.close()


def test_branch_generator_api_is_exact_and_rejects_variadic_kwargs() -> None:
    concrete = inspect.signature(
        t8.OfficialGlobalGCEMutagenicityGenerator.generate
    )
    protocol = inspect.signature(t8.TasteBranchGenerator.generate)
    assert tuple(concrete.parameters) == tuple(protocol.parameters)
    assert all(
        parameter.kind
        not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        for parameter in protocol.parameters.values()
    )
    t8._require_exact_branch_generator_signature(_FakeGenerator(0))

    class VariadicGenerator:
        target_label = 0

        def generate(self, parents, **kwargs):
            raise AssertionError((parents, kwargs))

    with pytest.raises(
        t8.TasteGlobalGCESmokeError,
        match="generator API signature changed",
    ):
        t8._require_exact_branch_generator_signature(VariadicGenerator())


def test_managed_v2_worker_is_raw_only_and_independent_verifier_publishes(
    tmp_path: Path,
) -> None:
    state, tree, science, _generators = _run_science(tmp_path)
    authority_value = _authority()
    startup = collect_t8_official_startup_evidence(
        state_tree=tree,
        science=science,
    )
    authority = _HeldFixtureTerminalAuthority(
        authority_value,
        official_startup=startup,
    )
    managed_root = tmp_path / "managed"
    managed_root.mkdir()
    final = tmp_path / "published" / "t8"
    final.parent.mkdir()
    attempt = create_managed_attempt(
        stage_root=managed_root,
        controller_id=authority_value["managed_execution"]["run_id"],
        task_id=t8.MANAGED_TASK_ID,
        git_commit=authority_value["execution"]["commit"],
        config_hash=t8_managed_config_hash(),
        input_hashes=t8_managed_input_hashes(authority_value),
        boot_id="t8-test-boot",
    )
    staging = create_worker_staging(attempt)
    try:
        sealed = seal_t8_worker_evidence(
            staging,
            science=science,
            state_tree=tree,
            input_authority=authority_value,
            expected_final_path=final,
        )
        assert not (sealed.staging_path / "PASS").exists()
        assert not (sealed.staging_path / "gate.json").exists()
        assert not (sealed.staging_path / "verification.json").exists()
        staging.close()
        tree.close()
        state.close()
        with open_sealed_worker_artifact(
            sealed.seal_path,
            expected_attempt_id=sealed.attempt_id,
            expected_generation_token=sealed.generation_token,
        ) as held:
            publication = verify_and_publish_t8_sealed(
                held,
                final_path=final,
                authority=authority,
            )
        assert publication.final_path == final
        assert load_verified_gate(final)["status"] == "PASS"
        verification = json.loads(
            (final / "verification.json").read_text(encoding="utf-8")
        )["verification"]
        assert verification["schema_version"] == (
            "tastemolnet_t8_independent_verification_v2"
        )
        assert verification["worker_self_signed"] is False
        assert verification["official_globalgce_commit"] == (
            t8.OFFICIAL_GLOBALGCE_COMMIT
        )
        assert verification["target_branches"] == [0, 2]
    finally:
        staging.close()
        attempt.close()
        tree.close()
        state.close()


def test_managed_v2_verifier_rejects_wholly_rehashed_worker_authority(
    tmp_path: Path,
) -> None:
    state, tree, science, _generators = _run_science(tmp_path)
    independently_held = _authority()
    startup = collect_t8_official_startup_evidence(
        state_tree=tree,
        science=science,
    )
    forged = json.loads(json.dumps(independently_held))
    forged["execution"]["commit"] = "a" * 40
    forged["execution"]["tree"] = "b" * 40
    forged["managed_execution"]["run_id"] = "forged-v2-run"
    forged["managed_execution"]["gpu_uuid"] = "GPU-forged-v2"
    forged["managed_execution"]["active_generation_sha256"] = "c" * 64
    forged["predecessors"]["t2_gate_sha256"] = "d" * 64
    managed_root = tmp_path / "managed-forged"
    managed_root.mkdir()
    final = tmp_path / "published-forged" / "t8"
    final.parent.mkdir()
    attempt = create_managed_attempt(
        stage_root=managed_root,
        controller_id=forged["managed_execution"]["run_id"],
        task_id=t8.MANAGED_TASK_ID,
        git_commit=forged["execution"]["commit"],
        config_hash=t8_managed_config_hash(),
        input_hashes=t8_managed_input_hashes(forged),
        boot_id="t8-forged-test-boot",
    )
    staging = create_worker_staging(attempt)
    try:
        sealed = seal_t8_worker_evidence(
            staging,
            science=science,
            state_tree=tree,
            input_authority=forged,
            expected_final_path=final,
        )
        staging.close()
        with open_sealed_worker_artifact(sealed.seal_path) as held:
            with pytest.raises(
                t8.TasteGlobalGCESmokeError,
                match="managed raw evidence changed",
            ):
                verify_and_publish_t8_sealed(
                    held,
                    final_path=final,
                    authority=_HeldFixtureTerminalAuthority(
                        independently_held,
                        official_startup=startup,
                    ),
                )
        assert not final.exists()
    finally:
        staging.close()
        attempt.close()
        tree.close()
        state.close()


def test_retained_branch_rejects_target_directory_replacement(tmp_path: Path) -> None:
    state = FreshOutputDirectory.create(tmp_path / "state")
    branch = t8._HeldBranchDirectory.create(state, target_label=0)
    try:
        os.rename(
            "target-0",
            "displaced-target-0",
            src_dir_fd=state.descriptor,
            dst_dir_fd=state.descriptor,
        )
        os.mkdir("target-0", 0o700, dir_fd=state.descriptor)
        with pytest.raises(
            t8.TasteGlobalGCESmokeError,
            match="retained branch directory changed",
        ):
            branch.revalidate()
    finally:
        branch.close()
        state.close()


def test_retained_planned_checkpoint_rejects_same_byte_leaf_swap(
    tmp_path: Path,
) -> None:
    state = FreshOutputDirectory.create(tmp_path / "state")
    branch = t8._HeldBranchDirectory.create(state, target_label=0)
    holder = None
    try:
        checkpoint_root = branch.path / "globalgce_training_checkpoints"
        checkpoint_root.mkdir()
        checkpoint = checkpoint_root / "training_checkpoint.pt"
        heartbeat = checkpoint_root / "training_heartbeat.json"
        checkpoint.write_bytes(b"epoch-zero")
        heartbeat.write_bytes(b'{"stage":"training"}\n')
        holder = t8._HeldPlannedCheckpoint.capture(
            branch,
            checkpoint_evidence=_FakeGenerator._file_evidence(checkpoint),
            heartbeat_evidence=_FakeGenerator._file_evidence(heartbeat),
        )
        replacement = checkpoint_root / ".replacement"
        replacement.write_bytes(checkpoint.read_bytes())
        os.replace(replacement, checkpoint)
        with pytest.raises(
            t8.TasteGlobalGCESmokeError,
            match="named checkpoint leaf differs before resume",
        ):
            holder.require_named_for_resume()
    finally:
        if holder is not None:
            holder.close()
        branch.close()
        state.close()


def test_terminal_documents_are_exact_aggregate_only_and_held(tmp_path: Path) -> None:
    state, tree, science, _generators = _run_science(tmp_path)
    try:
        documents = t8.build_terminal_documents(
            science=science,
            input_authority=_authority(),
        )
        encoded = json.dumps(documents).lower()
        for forbidden in ("smiles", "molecule_id", "candidate_id", "rf_model.pkl"):
            assert forbidden not in encoded
        output = tmp_path / "terminal"
        _publish_fixture(output, documents)
        with t8.hold_taste_globalgce_smoke_output(
            output,
            authority=_HeldFixtureTerminalAuthority(_authority()),
        ) as held:
            evidence = held.revalidate()
            assert evidence["stage"] == "T8_GLOBALGCE_SMOKE"
            assert evidence["strict_flip_count"] == 2
            assert evidence["destination_distribution"] == {"0": 1, "2": 1}
            assert evidence["task_id"] == "tastemolnet_t8_globalgce_smoke"
            assert evidence["run_id"] == "t8-fixture-run"
            assert evidence["gpu_index"] == 2
            assert evidence["gpu_uuid"] == "GPU-fixture"
            assert evidence["managed_active_generation_sha256"] == "8" * 64
            assert evidence["managed_child_identity_sha256"] == "9" * 64
            assert evidence["frozen_gine_checkpoint_id"] == "a" * 64
            assert evidence["train_split_sha256"] == "6" * 64
            assert evidence["official_source_inventory_sha256"] == "9" * 64
            assert evidence["downstream_policy_sha256"] == "b" * 64
            assert set(path.name for path in output.iterdir()) == t8.TERMINAL_FILES
    finally:
        tree.close()
        state.close()


def test_held_terminal_rejects_same_byte_leaf_replacement(tmp_path: Path) -> None:
    state, tree, science, _generators = _run_science(tmp_path)
    output = tmp_path / "terminal"
    try:
        documents = t8.build_terminal_documents(
            science=science,
            input_authority=_authority(),
        )
        _publish_fixture(output, documents)
        held = t8.hold_taste_globalgce_smoke_output(
            output,
            authority=_HeldFixtureTerminalAuthority(_authority()),
        )
        gate = output / "gate.json"
        replacement = output / "replacement"
        replacement.write_bytes(gate.read_bytes())
        os.replace(replacement, gate)
        with pytest.raises(Exception):
            held.revalidate()
        held.close()
    finally:
        tree.close()
        state.close()


def test_public_consumer_rejects_wholly_rehashed_self_signed_root(
    tmp_path: Path,
) -> None:
    state, tree, science, _generators = _run_science(tmp_path)
    try:
        independently_held = _authority()
        forged = json.loads(json.dumps(independently_held))
        forged["execution"]["commit"] = "a" * 40
        forged["execution"]["tree"] = "b" * 40
        forged["managed_execution"]["run_id"] = "forged-run"
        forged["managed_execution"]["gpu_uuid"] = "GPU-forged"
        forged["managed_execution"]["active_generation_sha256"] = "c" * 64
        forged["managed_execution"]["child_identity_sha256"] = "d" * 64
        forged["predecessors"]["t2_gate_sha256"] = "e" * 64
        forged["frozen_gine"]["checkpoint_id"] = "a" * 64
        forged["official_globalgce"]["commit"] = "f" * 40
        forged["train_split"]["sha256"] = "1" * 64
        forged["policy"]["downstream_policy_sha256"] = "2" * 64
        documents = t8.build_terminal_documents(
            science=science,
            input_authority=forged,
        )
        output = tmp_path / "forged-terminal"
        _publish_fixture(output, documents)
        with pytest.raises(
            t8.TasteGlobalGCESmokeError,
            match="independent held authority",
        ):
            t8.hold_taste_globalgce_smoke_output(
                output,
                authority=_HeldFixtureTerminalAuthority(independently_held),
            )
    finally:
        tree.close()
        state.close()


def test_public_consumer_rejects_raw_mapping_as_expected_authority(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        t8.TasteGlobalGCESmokeError,
        match="held authority",
    ):
        t8.hold_taste_globalgce_smoke_output(
            tmp_path / "absent",
            authority=_authority(),  # type: ignore[arg-type]
        )


def test_public_consumer_rejects_independent_authority_drift_after_open(
    tmp_path: Path,
) -> None:
    state, tree, science, _generators = _run_science(tmp_path)
    held = None
    try:
        authority = _HeldFixtureTerminalAuthority(_authority())
        documents = t8.build_terminal_documents(
            science=science,
            input_authority=authority.value,
        )
        output = tmp_path / "terminal-with-drifting-authority"
        _publish_fixture(output, documents)
        held = t8.hold_taste_globalgce_smoke_output(
            output,
            authority=authority,
        )
        authority.value["managed_execution"]["run_id"] = "drifted-run"
        with pytest.raises(
            t8.TasteGlobalGCESmokeError,
            match="independent held terminal authority changed",
        ):
            held.revalidate()
    finally:
        if held is not None:
            held.close()
        tree.close()
        state.close()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("managed_execution", "gpu_index"), True),
        (("managed_execution", "gpu_uuid"), "GPU-"),
        (("managed_execution", "run_id"), True),
        (("managed_execution", "same_child_revalidated_at_terminal"), 1),
        (("frozen_gine", "rf_oracle_used"), 0),
        (("policy", "test_loaded"), 0),
    ],
)
def test_terminal_input_authority_rejects_hostile_native_types(path, value) -> None:
    authority = _authority()
    authority[path[0]][path[1]] = value
    with pytest.raises(t8.TasteGlobalGCESmokeError):
        t8._validate_terminal_input_authority(authority)


def test_train_parser_rejects_float_label_and_non_train_split() -> None:
    payload = (
        "molecule_id,smiles,label,split\n"
        "a,C,0,train\n"
        "b,N,1.0,train\n"
        "c,O,2,train\n"
    ).encode()
    with pytest.raises(t8.TasteGlobalGCESmokeError, match="canonical 0/1/2"):
        t8.load_taste_train_cohort(
            payload,
            expected_row_count=3,
            expected_label_counts={"0": 1, "1": 1, "2": 1},
        )
    changed = payload.replace(b"1.0,train", b"1,test")
    with pytest.raises(t8.TasteGlobalGCESmokeError, match="split=train"):
        t8.load_taste_train_cohort(
            changed,
            expected_row_count=3,
            expected_label_counts={"0": 1, "1": 1, "2": 1},
        )


def test_smoke_config_rejects_science_surface_drift_before_execution() -> None:
    with pytest.raises(t8.TasteGlobalGCESmokeError, match="frozen smoke"):
        t8.TasteGlobalGCESmokeConfig(source_parent_count=15).validate()


def test_terminal_rejects_per_example_or_binary_claim(tmp_path: Path) -> None:
    state, tree, science, _generators = _run_science(tmp_path)
    try:
        science = json.loads(json.dumps(science))
        science["strict_flip_validation"]["candidate_id"] = "private"
        with pytest.raises(t8.TasteGlobalGCESmokeError):
            t8.build_terminal_documents(
                science=science,
                input_authority=_authority(),
            )
        science.pop("strict_flip_validation")
    finally:
        tree.close()
        state.close()


def test_terminal_rejects_unknown_generic_payload_even_without_forbidden_key(
    tmp_path: Path,
) -> None:
    state, tree, science, _generators = _run_science(tmp_path)
    try:
        science = json.loads(json.dumps(science))
        science["opaque"] = ["CCO", "private-row"]
        with pytest.raises(t8.TasteGlobalGCESmokeError):
            t8.build_terminal_documents(
                science=science,
                input_authority=_authority(),
            )
    finally:
        tree.close()
        state.close()


def test_release_publication_refuses_without_reviewed_final_rename(
    tmp_path: Path,
) -> None:
    state, tree, science, _generators = _run_science(tmp_path)
    output = FreshOutputDirectory.create(tmp_path / "unreleased")
    try:
        documents = t8.build_terminal_documents(
            science=science,
            input_authority=_authority(),
        )
        with pytest.raises(
            t8.TasteGlobalGCESmokeError,
            match="marker-last final-rename",
        ):
            t8.publish_terminal_output(
                output=output,
                documents=documents,
                retained_input_closure=lambda: None,
            )
        assert not (tmp_path / "unreleased" / "PASS").exists()
    finally:
        tree.close()
        state.close()
