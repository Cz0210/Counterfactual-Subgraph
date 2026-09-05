from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.baselines import tastemolnet_globalgce_full as full
from src.baselines.globalgce_mutagenicity_adapter import TrainParent
from src.data.tastemolnet_ppo import TASTEMOLNET_PREPARED_FIELDS
from src.utils.managed_execution_v2 import (
    create_managed_attempt,
    create_worker_staging,
    write_worker_exit,
    write_worker_raw_evidence,
)
from src.utils.tastemolnet_t8_managed_v2 import T8_VERIFICATION_SCHEMA
from src.utils.terminal_publisher_v2 import (
    open_sealed_worker_artifact,
    seal_worker_staging,
    verify_and_publish_sealed_attempt,
)


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _publish_managed_t8(tmp_path: Path, *, oracle_hash: str) -> Path:
    """Publish one real generic managed-v2 terminal with the real T8 receipt."""

    stage_root = tmp_path / "managed-t8-stage"
    stage_root.mkdir()
    attempt = create_managed_attempt(
        stage_root=stage_root,
        controller_id="t13-managed-v2-fixture",
        task_id=full.T8_MANAGED_TASK_ID,
        git_commit="1" * 40,
        config_hash="2" * 64,
        input_hashes={"t8_input": "3" * 64},
        boot_id="t13-managed-v2-fixture-boot",
    )
    staging = create_worker_staging(attempt)
    raw = write_worker_raw_evidence(staging, {"stage": full.T8_STAGE})
    raw.close()
    worker_exit = write_worker_exit(
        staging,
        {"exit_code": 0, "worker_closed_artifact_writers": True},
    )
    worker_exit.close()
    sealed = seal_worker_staging(staging)
    staging.close()
    final = tmp_path / "published" / "t8"
    final.parent.mkdir()
    typed = {
        "schema_version": T8_VERIFICATION_SCHEMA,
        "status": "PASS",
        "stage": full.T8_STAGE,
        "dataset": full.DATASET_ID,
        "method": full.METHOD,
        "task_id": full.T8_MANAGED_TASK_ID,
        "attempt_id": sealed.attempt_id,
        "generation_token": sealed.generation_token,
        "input_authority_sha256": "4" * 64,
        "science_sha256": "5" * 64,
        "official_startup_sha256": "6" * 64,
        "official_globalgce_commit": full.OFFICIAL_GLOBALGCE_COMMIT,
        "oracle_checkpoint_hash": oracle_hash,
        "target_branches": [0, 2],
        "strict_flip_count": 2,
        "destination_distribution": {"0": 1, "2": 1},
        "same_three_class_gine": True,
        "checkpoint_resume_verified": True,
        "official_lhs_to_rhs_verified": True,
        "isolated_imports_verified": True,
        "rf_oracle_used": False,
        "data_redistributed": False,
        "worker_self_signed": False,
        "external_authority_revalidated": True,
    }
    try:
        with open_sealed_worker_artifact(sealed.seal_path) as held:
            verify_and_publish_sealed_attempt(
                held,
                final_path=final,
                verification=typed,
            )
    finally:
        attempt.close()
    return final


def _threshold(path: Path) -> full.ThresholdContract:
    values = [0.1, 0.2, 0.4]
    _json(
        path,
        {
            "schema_version": "four_by_four_frozen_threshold_contract_v1",
            "dataset": "TasteMolNet",
            "thresholds": values,
            "theta_star": 0.2,
            "cost_cap": 0.4,
            "threshold_source": "preregistered calibration protocol",
            "threshold_source_split": "frozen_protocol",
            "threshold_config_hash": full.stable_json_sha256(values),
            "test_used_for_selection": False,
        },
    )
    return full.load_threshold_contract(path)


def _prepared_csv(path: Path, *, split: str, labels: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TASTEMOLNET_PREPARED_FIELDS)
        writer.writeheader()
        for index, label in enumerate(labels):
            row = {field: "" for field in TASTEMOLNET_PREPARED_FIELDS}
            row.update(
                {
                    "molecule_id": f"{split}-{index}",
                    "raw_smiles": "CC",
                    "canonical_smiles": "CC",
                    "model_smiles": "CC",
                    "label": str(label),
                    "label_name": {0: "Bitter", 1: "Sweet", 2: "Tasteless"}[label],
                    "split": split,
                }
            )
            writer.writerow(row)


def _pair(parent: str, candidate: str, *, distance: float | None, destination: int = 0):
    strict = distance is not None
    return {
        "dataset": "TasteMolNet",
        "method": "GlobalGCE",
        "split": "test",
        "parent_id": parent,
        "candidate_id": candidate,
        "applicable": True,
        "pair_strict_flip": strict,
        "wnode_distance": distance,
        "pred_before": 1,
        "pred_after": destination if strict else 1,
        "destination_label": destination if strict else None,
        "cf_drop": 0.5 if strict else 0.0,
        "rf_oracle_used": False,
    }


def test_threshold_contract_is_exact_grid_identity(tmp_path: Path) -> None:
    contract = _threshold(tmp_path / "threshold.json")
    assert contract.values == (0.1, 0.2, 0.4)
    assert contract.config_hash == full.stable_json_sha256([0.1, 0.2, 0.4])

    payload = json.loads((tmp_path / "threshold.json").read_text())
    payload["thresholds"] = [0.1, 0.2, 0.5]
    _json(tmp_path / "threshold.json", payload)
    with pytest.raises(full.TasteGlobalGCEFullError, match="threshold_config_hash"):
        full.load_threshold_contract(tmp_path / "threshold.json")


def test_prepared_split_is_exact_and_keeps_only_sweet(tmp_path: Path) -> None:
    path = tmp_path / "calibration.csv"
    _prepared_csv(path, split="calibration", labels=(0, 1, 2, 1))
    parents = full.load_prepared_split(
        path,
        expected_split="calibration",
        expected_sha256=full.sha256_file(path),
    )
    assert [row.parent_id for row in parents] == ["calibration-1", "calibration-3"]
    assert all(row.label == 1 and row.split == "calibration" for row in parents)

    with pytest.raises(full.TasteGlobalGCEFullError, match="row authority"):
        full.load_prepared_split(
            path,
            expected_split="test",
            expected_sha256=full.sha256_file(path),
        )


def test_test_file_is_not_opened_before_frozen_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = tmp_path / "selection_manifest.json"
    _json(
        selection,
        {
            "schema_version": full.SELECTION_SCHEMA,
            "status": "RUNNING",
            "selection_frozen": False,
            "selector_fitted_on_calibration": True,
            "test_loaded": False,
            "test_used_for_selection": False,
        },
    )
    opened = []
    monkeypatch.setattr(full, "load_prepared_split", lambda *args, **kwargs: opened.append(True))
    authority = SimpleNamespace(test_path=tmp_path / "never-open.csv", declared_test_sha256="a" * 64)
    with pytest.raises(full.TasteGlobalGCEFullError, match="requires frozen"):
        full.authorize_and_load_test_after_freeze(
            authority=authority, selection_manifest_path=selection
        )
    assert opened == []


def test_calibration_selector_is_deterministic_and_test_free() -> None:
    rules = [
        {"candidate_id": "r1"},
        {"candidate_id": "r2"},
        {"candidate_id": "r3"},
    ]
    rows = [
        {"split": "calibration", "parent_id": "p1", "candidate_id": "r1", "pair_strict_flip": True, "wnode_distance": 0.1},
        {"split": "calibration", "parent_id": "p2", "candidate_id": "r1", "pair_strict_flip": False, "wnode_distance": None},
        {"split": "calibration", "parent_id": "p1", "candidate_id": "r2", "pair_strict_flip": False, "wnode_distance": None},
        {"split": "calibration", "parent_id": "p2", "candidate_id": "r2", "pair_strict_flip": True, "wnode_distance": 0.1},
        {"split": "calibration", "parent_id": "p1", "candidate_id": "r3", "pair_strict_flip": True, "wnode_distance": 0.3},
        {"split": "calibration", "parent_id": "p2", "candidate_id": "r3", "pair_strict_flip": True, "wnode_distance": 0.3},
    ]
    # The production route has >=10 rules.  Repeat inert, deterministic rules
    # so this unit fixture exercises the same minimum without faking a lower gate.
    for index in range(4, 11):
        candidate = f"r{index}"
        rules.append({"candidate_id": candidate})
        rows.extend(
            {
                "split": "calibration",
                "parent_id": parent,
                "candidate_id": candidate,
                "pair_strict_flip": False,
                "wnode_distance": None,
            }
            for parent in ("p1", "p2")
        )
    selected, manifest = full.select_rules_on_calibration(rules, rows, theta_star=0.2)
    assert [row["candidate_id"] for row in selected[:2]] == ["r1", "r2"]
    assert manifest["selector_fitted_on_calibration"] is True
    assert manifest["test_loaded"] is False
    assert manifest["test_used_for_selection"] is False


def test_standardized_metrics_plateau_without_copying_rules(tmp_path: Path) -> None:
    threshold = _threshold(tmp_path / "threshold.json")
    ordered = [f"r{index}" for index in range(10)]
    rows = []
    for parent in ("p1", "p2"):
        for index, candidate in enumerate(ordered):
            distance = 0.1 if (parent == "p1" and index == 0) else None
            if parent == "p2" and index == 1:
                distance = 0.2
            rows.append(_pair(parent, candidate, distance=distance, destination=index % 2 * 2))
    metrics = full.compute_standardized_metrics(rows, ordered, threshold)
    assert len(metrics["figure3"]) == 20
    assert metrics["effective_rule_count"] == 10
    assert metrics["figure3"][9]["coverage"] == metrics["figure3"][19]["coverage"]
    assert metrics["prefix"][19]["plateau_after_effective_k"] is True
    assert {row["destination_label"] for row in metrics["destination"]} == {0, 2}


class _FakeBranchGenerator:
    _SYMBOLS = ("C", "N", "O", "F", "Cl", "Br", "S", "P", "B", "I")

    def __init__(self, target_label: int, *, rule_count: int = 5) -> None:
        self.target_label = target_label
        self.rule_count = rule_count
        self.calls = 0

    def _rules(self) -> list[dict]:
        offset = 0 if self.target_label == 0 else 5
        rows = []
        for index in range(self.rule_count):
            symbol = self._SYMBOLS[offset + index]
            rows.append(
                {
                    "candidate_id": f"native-{self.target_label}-{index}",
                    "native_rule_index": index,
                    "lhs_feature": [[0.0, 1.0], [0.0, 1.0]],
                    "lhs_adjacency": [[0.0, 1.0], [1.0, 0.0]],
                    "lhs_edge_attr": [[0.0, 1.0]],
                    "rhs_feature": [[0.0, 1.0], [0.0, 1.0]],
                    "rhs_adjacency": [[0.0, 1.0], [1.0, 0.0]],
                    "rhs_edge_attr": [[0.0, 1.0]],
                    "atom_symbols": [symbol],
                    "bond_names": ["no_edge", "single"],
                }
            )
        return rows

    @staticmethod
    def _training_identity(
        *,
        target_label: int,
        checkpoint_id: str,
        parent_count: int,
        config: full.TasteGlobalGCEFullConfig,
    ) -> tuple[dict, str]:
        inventory = [
            {"name": "model.pt", "bytes": 1, "sha256": checkpoint_id},
            {
                "name": "temperature_scaling.json",
                "bytes": 1,
                "sha256": "b" * 64,
            },
        ]
        oracle = {
            "schema_version": "globalgce_frozen_gine_resume_identity_v1",
            "backend": "frozen_gine",
            "checkpoint_root": "/fixture/frozen-gine",
            "checkpoint_id": checkpoint_id,
            "dataset": full.DATASET,
            "num_classes": 3,
            "source_label": 1,
            "temperature_hex": float(1.0).hex(),
            "temperature_scaling_sha256": "b" * 64,
            "sha256sums_sha256": "c" * 64,
            "inventory": inventory,
            "inventory_sha256": _canonical_sha256({"files": inventory}),
        }
        oracle["identity_sha256"] = _canonical_sha256(oracle)
        official = {
            "schema_version": "globalgce_official_source_resume_identity_v1",
            "root": "/fixture/official-globalgce",
            "files": {"main.py": {"bytes": 1, "sha256": "d" * 64}},
        }
        official["identity_sha256"] = _canonical_sha256(official)
        train_count = max(1, parent_count - 1)
        val_count = parent_count - train_count
        if val_count <= 0:
            raise AssertionError("branch fixture needs at least two parents")
        cohort = {
            "count": parent_count,
            "ordered_sha256": "e" * 64,
            "train_count": train_count,
            "train_ordered_sha256": "f" * 64,
            "val_count": val_count,
            "val_ordered_sha256": "0" * 64,
        }
        identity = {
            "schema_version": "globalgce_training_resume_identity_v2",
            "dataset": full.DATASET,
            "num_classes": 3,
            "source_label": 1,
            "target_label": target_label,
            "oracle_identity": oracle,
            "native_train_cohort": dict(cohort),
            "source_train_cohort": dict(cohort),
            "official_source_identity": official,
            "training_config": {
                "seed": config.seed,
                "epochs": config.epochs,
                "top_k_native": config.top_k_native,
                "learning_rate_hex": config.learning_rate.hex(),
                "dropout_hex": config.dropout.hex(),
                "min_freq": config.min_freq,
                "gspan_flush_every": config.gspan_flush_every,
                "gspan_max_in_memory_candidates": (
                    config.gspan_max_in_memory_candidates
                ),
                "gspan_exact_top_k_pruning": False,
                "gspan_adoption_identity": None,
            },
        }
        normalized, identity_hash = full.normalize_globalgce_training_resume_identity(
            identity
        )
        return normalized, identity_hash

    def generate(self, parents, **kwargs):  # noqa: ANN001, ANN003
        self.calls += 1
        root = Path(kwargs["output_dir"])
        config = full.TasteGlobalGCEFullConfig(
            epochs=kwargs["epochs"],
            top_k_native=kwargs["top_k_native"],
            learning_rate=kwargs["learning_rate"],
            dropout=kwargs["dropout"],
            generation_chunk_size=kwargs["generation_chunk_size"],
            gspan_flush_every=kwargs["gspan_flush_every"],
            gspan_max_in_memory_candidates=kwargs[
                "gspan_max_in_memory_candidates"
            ],
        )
        checkpoint_id = "a" * 64
        identity, identity_hash = self._training_identity(
            target_label=self.target_label,
            checkpoint_id=checkpoint_id,
            parent_count=len(parents),
            config=config,
        )
        full.atomic_jsonl(root / "native_rule_catalog.jsonl", self._rules())
        full.atomic_jsonl(root / "native_rule_rejections.jsonl", [])
        (root / "globalgce_model.pt").write_bytes(b"fixture-model\n")
        (root / "globalgce_rules.pt").write_bytes(b"fixture-rules\n")
        checkpoint = root / "globalgce_training_checkpoints"
        checkpoint.mkdir(parents=True)
        (checkpoint / "training_checkpoint.pt").write_bytes(b"checkpoint\n")
        full.atomic_json(checkpoint / "training_heartbeat.json", {"epoch": 25})
        summary = {
            "dataset_name": full.DATASET,
            "selected_parent_count": len(parents),
            "prediction_backend": "frozen_gine_differentiable_bridge",
            "classifier_family": "gine",
            "oracle_backend": "gnn",
            "rf_oracle_used": False,
            "num_classes": 3,
            "source_label": 1,
            "target_label": self.target_label,
            "frozen_source_label": 1,
            "frozen_target_label": self.target_label,
            "generation_input_split": "train",
            "calibration_loaded": False,
            "test_loaded": False,
            "valid_native_rule_count": self.rule_count,
            "training_resume_identity": identity,
            "training_resume_identity_sha256": identity_hash,
            "gnn_training": {"checkpoint_id": checkpoint_id},
            "gnn_checkpoint_sha256": checkpoint_id,
            "gspan_exact_top_k_pruning": False,
            "trained_once": True,
            "rule_selection_performed_once": True,
            "globalgce_model_checkpoint_sha256": full.sha256_file(
                root / "globalgce_model.pt"
            ),
            "rules_checkpoint_sha256": full.sha256_file(
                root / "globalgce_rules.pt"
            ),
            "trained_model_resumed": False,
        }
        full.atomic_json(root / "training_core_summary.json", summary)
        return SimpleNamespace(training_summary=summary)


def test_native_branch_resume_adopts_completed_manifest(tmp_path: Path) -> None:
    generator = _FakeBranchGenerator(0)
    config = full.TasteGlobalGCEFullConfig(epochs=25)
    parents = [
        TrainParent("a", "CC", 1, "train"),
        TrainParent("b", "CCC", 1, "train"),
    ]
    cohort_hash = full._parent_cohort_sha256(parents)
    first = full.run_native_branch(
        target_label=0,
        generator=generator,
        parents=parents,
        branch_root=tmp_path / "target_0",
        config=config,
        expected_checkpoint_id="a" * 64,
        expected_parent_cohort_sha256=cohort_hash,
    )
    second = full.run_native_branch(
        target_label=0,
        generator=generator,
        parents=parents,
        branch_root=tmp_path / "target_0",
        config=config,
        expected_checkpoint_id="a" * 64,
        expected_parent_cohort_sha256=cohort_hash,
    )
    assert generator.calls == 1
    assert first == second
    assert second["official_epoch_checkpoint_resume_enabled"] is True
    assert second["rules_only_min_valid_native_rules"] == 0

    (tmp_path / "target_0" / "globalgce_rules.pt").write_bytes(b"tampered\n")
    with pytest.raises(full.TasteGlobalGCEFullError, match="artifact bytes"):
        full.run_native_branch(
            target_label=0,
            generator=generator,
            parents=parents,
            branch_root=tmp_path / "target_0",
            config=config,
            expected_checkpoint_id="a" * 64,
            expected_parent_cohort_sha256=cohort_hash,
        )


def test_pair_evaluation_resume_reuses_durable_parent_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    rules = [
        {
            "candidate_id": f"r{index}",
            "rule_content_hash": full.stable_sha256({"rule": index}),
        }
        for index in range(10)
    ]

    def fake_evaluate(*, parent, rules, scorer, provider, split):  # noqa: ANN001
        del scorer, provider
        calls.append(parent.parent_id)
        return [
            {
                "parent_id": parent.parent_id,
                "candidate_id": row["candidate_id"],
                "split": split,
            }
            for row in rules
        ]

    monkeypatch.setattr(full, "evaluate_one_parent", fake_evaluate)
    parents = [
        TrainParent("a", "CC", 1, "calibration"),
        TrainParent("b", "CCC", 1, "calibration"),
    ]
    checkpoint_id = "a" * 64
    identity = full.build_split_evaluation_identity(
        split="calibration",
        parents=parents,
        rules=rules,
        oracle_checkpoint_hash=checkpoint_id,
        molclr_checkpoint_hash="b" * 64,
        threshold_config_hash="c" * 64,
    )
    scorer = SimpleNamespace(checkpoint_id=checkpoint_id)
    checkpoints: list[int] = []
    first, _ = full.evaluate_split_resumable(
        split="calibration",
        parents=parents,
        rules=rules,
        scorer=scorer,
        provider=object(),
        output=tmp_path,
        checkpoint_callback=checkpoints.append,
        evaluation_identity=identity,
    )
    second, _ = full.evaluate_split_resumable(
        split="calibration",
        parents=parents,
        rules=rules,
        scorer=scorer,
        provider=object(),
        output=tmp_path,
        checkpoint_callback=checkpoints.append,
        evaluation_identity=identity,
    )
    assert calls == ["a", "b"]
    assert first == second
    assert checkpoints == [1, 2, 1, 2]

    chunk = tmp_path / "raw/calibration_pair_chunks/00000000.jsonl"
    chunk.write_text(chunk.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(full.TasteGlobalGCEFullError, match="resume chunk"):
        full.evaluate_split_resumable(
            split="calibration",
            parents=parents,
            rules=rules,
            scorer=scorer,
            provider=object(),
            output=tmp_path,
            checkpoint_callback=checkpoints.append,
            evaluation_identity=identity,
        )


def test_sealed_resume_returns_without_reopening_science(tmp_path: Path) -> None:
    output = tmp_path / "sealed"
    output.mkdir()
    config = full.TasteGlobalGCEFullConfig(epochs=25)
    identity = {"schema_version": "test", "config": config.to_dict()}

    class Authority:
        @staticmethod
        def resume_identity(observed):  # noqa: ANN001
            assert observed == config
            return identity

    full.write_checkpoint(output, phase="SEALED", resume_identity=identity)
    full.atomic_json(output / "run_manifest.json", {"status": "SEALED"})
    result = full.run_t13_full(
        authority=Authority(),
        output_dir=output,
        config=config,
        resume=True,
        device="cuda:0",
        wnode_cache_db=tmp_path / "unused.sqlite",
        node_embedding_cache_dir=tmp_path / "unused-cache",
    )
    assert result == {"status": "SEALED"}


def test_slurm_wrapper_runs_science_then_independent_verifier() -> None:
    text = Path("scripts/slurm/run_tastemolnet_globalgce_full.sh").read_text()
    assert "#SBATCH --partition=A800" in text
    assert "#SBATCH --gres=gpu:a800:1" in text
    assert "export PYTHONPATH=$PWD" in text
    assert text.count("python -I -B scripts/run_tastemolnet_globalgce_full.py") == 2
    assert "--verify-only" in text
    assert "--set inference.fallback_to_heuristic=false" in text


def test_native_generator_receives_audited_source_authority(tmp_path, monkeypatch):
    config = full.TasteGlobalGCEFullConfig(epochs=100)
    authority = SimpleNamespace(
        official_root=tmp_path / "official", checkpoint_path=tmp_path / "gine",
        checkpoint_id="a" * 64, train_path=tmp_path / "train.csv",
        resume_identity=lambda _config: {"source": "tiny-fixture"},
    )
    runtime_authority = {"src/gSpan/gspan.py": {"device": 1, "inode": 2, "bytes": 3, "sha256": "b" * 64}}
    audited = []
    def audit(root):
        audited.append(root)
        return {"runtime_source_authority": runtime_authority}
    monkeypatch.setattr(full, "validate_official_globalgce_root", audit)
    monkeypatch.setattr(full, "_checkpoint_payloads", lambda _p: {})
    monkeypatch.setattr(full, "FrozenTasteGINEScorer", lambda *_a, **_k: SimpleNamespace(checkpoint_id=authority.checkpoint_id))
    monkeypatch.setattr(full, "load_full_train_split", lambda _a: [])
    monkeypatch.setattr(full, "select_full_sweet_train_cohort", lambda *_a, **_k: ([], {"selected_cohort_sha256": full.stable_sha256([])}))
    class StopBeforeScience(Exception):
        pass
    def generator(root, **kwargs):
        assert audited == [authority.official_root]
        assert root == authority.official_root
        assert kwargs["official_source_authority"] is runtime_authority
        assert kwargs["require_isolated_imports"] is True
        assert kwargs["source_label"] == 1
        assert kwargs["target_label"] == 0
        assert kwargs["num_classes"] == 3
        raise StopBeforeScience
    monkeypatch.setattr(full, "OfficialGlobalGCEMutagenicityGenerator", generator)
    with pytest.raises(StopBeforeScience):
        full.run_t13_full(authority=authority, output_dir=tmp_path / "fresh", config=config, resume=False, device="cuda:0", wnode_cache_db=tmp_path / "unused.sqlite", node_embedding_cache_dir=tmp_path / "unused-cache")


def test_t8_adoption_accepts_real_managed_v2_nested_verification(
    tmp_path: Path,
) -> None:
    final = _publish_managed_t8(tmp_path, oracle_hash="a" * 64)
    adopted_root, evidence = full.validate_t8_pass(final)
    assert adopted_root == final.resolve()
    assert evidence["typed_verification"]["schema_version"] == (
        T8_VERIFICATION_SCHEMA
    )
    assert evidence["typed_verification"]["oracle_checkpoint_hash"] == "a" * 64
    assert len(evidence["adoption_sha256"]) == 64


def test_terminal_verifier_replays_metrics_and_passes_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "t13"
    raw = output / "raw"
    raw.mkdir(parents=True)
    threshold = _threshold(tmp_path / "threshold.json")
    checkpoint_id = "a" * 64
    molclr_hash = "e" * 64
    t8_root = _publish_managed_t8(tmp_path, oracle_hash=checkpoint_id)
    _t8_root, t8_evidence = full.validate_t8_pass(t8_root)
    config = full.TasteGlobalGCEFullConfig(epochs=25)
    train_parents = [
        TrainParent("train-a", "CC", 1, "train"),
        TrainParent("train-b", "CCC", 1, "train"),
    ]
    train_cohort_hash = full._parent_cohort_sha256(train_parents)
    full.atomic_json(
        raw / "train_cohort_manifest.json",
        {
            "selected_count": len(train_parents),
            "ordered_parent_cohort_sha256": train_cohort_hash,
            "train_only": True,
            "calibration_loaded": False,
            "test_loaded": False,
        },
    )
    branch_roots = {target: raw / f"target_{target}" for target in (0, 2)}
    for target in (0, 2):
        full.run_native_branch(
            target_label=target,
            generator=_FakeBranchGenerator(target, rule_count=5),
            parents=train_parents,
            branch_root=branch_roots[target],
            config=config,
            expected_checkpoint_id=checkpoint_id,
            expected_parent_cohort_sha256=train_cohort_hash,
        )
    merged_rules, merge = full.merge_branch_rules(branch_roots)
    assert merge["target_0_rule_count"] == 5
    assert merge["target_2_rule_count"] == 5
    assert len(merged_rules) == full.MIN_RULES
    full.atomic_jsonl(raw / "merged_rules.jsonl", merged_rules)
    merge.update(
        {
            "dataset": full.DATASET,
            "method": full.METHOD,
            "source_split": "train",
            "calibration_loaded": False,
            "test_loaded": False,
            "merged_rules_sha256": full.sha256_file(raw / "merged_rules.jsonl"),
        }
    )
    full.atomic_json(raw / "merge_manifest.json", merge)

    def fake_evaluate(*, parent, rules, scorer, provider, split):  # noqa: ANN001
        del scorer, provider
        rows = []
        for index, rule in enumerate(rules):
            destination = int((rule.get("target_branches") or [index % 2 * 2])[0])
            row = _pair(
                parent.parent_id,
                str(rule["candidate_id"]),
                distance=0.1,
                destination=destination,
            )
            row["split"] = split
            rows.append(row)
        return rows

    monkeypatch.setattr(full, "evaluate_one_parent", fake_evaluate)
    scorer = SimpleNamespace(checkpoint_id=checkpoint_id)
    calibration_parents = [
        TrainParent("cal-p1", "CC", 1, "calibration"),
        TrainParent("cal-p2", "CCC", 1, "calibration"),
    ]
    calibration_rows, calibration_manifest = full.evaluate_split_resumable(
        split="calibration",
        parents=calibration_parents,
        rules=merged_rules,
        scorer=scorer,
        provider=object(),
        output=output,
        checkpoint_callback=lambda _count: None,
        evaluation_identity=full.build_split_evaluation_identity(
            split="calibration",
            parents=calibration_parents,
            rules=merged_rules,
            oracle_checkpoint_hash=checkpoint_id,
            molclr_checkpoint_hash=molclr_hash,
            threshold_config_hash=threshold.config_hash,
        ),
    )
    selected_rules, selector = full.select_rules_on_calibration(
        merged_rules, calibration_rows, theta_star=threshold.theta_star
    )
    full.atomic_jsonl(raw / "selected_rules.jsonl", selected_rules)
    ordered = [str(row["candidate_id"]) for row in selected_rules]
    selection = {
        "schema_version": full.SELECTION_SCHEMA,
        "dataset": full.DATASET,
        "method": full.METHOD,
        "stage": full.STAGE,
        "status": "FROZEN",
        "selection_frozen": True,
        "selector_fitted_on_calibration": True,
        "test_loaded": False,
        "test_used_for_selection": False,
        "frozen_at": "2026-08-31T00:00:00+00:00",
        **selector,
        **threshold.to_dict(),
        "calibration_manifest": calibration_manifest,
        "selected_rules_sha256": full.sha256_file(raw / "selected_rules.jsonl"),
        "oracle_checkpoint_hash": checkpoint_id,
        "molclr_checkpoint_hash": molclr_hash,
        "rf_oracle_used": False,
    }
    full.atomic_json(raw / "selection_manifest.json", selection)

    test_parents = [
        TrainParent("p1", "CC", 1, "test"),
        TrainParent("p2", "CCC", 1, "test"),
    ]
    test_rows, test_manifest = full.evaluate_split_resumable(
        split="test",
        parents=test_parents,
        rules=selected_rules,
        scorer=scorer,
        provider=object(),
        output=output,
        checkpoint_callback=lambda _count: None,
        evaluation_identity=full.build_split_evaluation_identity(
            split="test",
            parents=test_parents,
            rules=selected_rules,
            oracle_checkpoint_hash=checkpoint_id,
            molclr_checkpoint_hash=molclr_hash,
            threshold_config_hash=threshold.config_hash,
        ),
    )
    test_manifest.update(
        {
            "selection_manifest_sha256": full.sha256_file(
                raw / "selection_manifest.json"
            ),
            "selection_frozen_before_test": True,
            "test_used_for_selection": False,
            "started_at": "2026-08-31T00:01:00+00:00",
        }
    )
    full.atomic_json(
        raw / "test_evaluation_manifest.json",
        test_manifest,
    )
    metrics = full.compute_standardized_metrics(test_rows, ordered, threshold)
    full.atomic_csv(output / "figure3_coverage_vs_k.csv", metrics["figure3"])
    full.atomic_csv(output / "figure4_coverage_vs_threshold.csv", metrics["figure4"])
    full.atomic_csv(output / "prefix_metrics.csv", metrics["prefix"])
    full.atomic_json(output / "prefix_metrics.json", metrics["prefix"])
    full.atomic_csv(output / "parent_best_distances.csv", metrics["parent_best"])
    full.atomic_csv(output / "destination_distribution.csv", metrics["destination"])
    full.atomic_csv(output / "table2_globalgce_k10.csv", metrics["table2"])
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    branch_manifest_hashes = {
        str(target): full.sha256_file(branch_roots[target] / "branch_manifest.json")
        for target in (0, 2)
    }
    merge_manifest_hash = full.sha256_file(raw / "merge_manifest.json")
    common = {
        "dataset": full.DATASET,
        "method": full.METHOD,
        "stage": full.STAGE,
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "oracle_checkpoint": str(checkpoint),
        "oracle_hash": checkpoint_id,
        "oracle_checkpoint_hash": checkpoint_id,
        "dataset_hash": "b" * 64,
        "test_parent_ids_sha256": full.stable_sha256(["p1", "p2"]),
        "test_split_hash": "d" * 64,
        "distance_line": full.DISTANCE_LINE,
        "molclr_checkpoint_hash": molclr_hash,
        "t8_pass_root": str(t8_root),
        "t8_pass_sha256": t8_evidence["adoption_sha256"],
        "t8_oracle_checkpoint_hash": checkpoint_id,
        "cf_mode": full.CF_MODE,
        "threshold_config_hash": threshold.config_hash,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "raw_output_root": str(output),
    }
    full.atomic_json(
        output / "summary.json",
        {
            "schema_version": "test",
            **common,
            "status": "SEALED",
            "frozen": True,
            "raw_output_complete": True,
            "config": config.to_dict(),
            "train_parent_count": len(train_parents),
            "train_parent_cohort_sha256": train_cohort_hash,
            "branch_manifests": branch_manifest_hashes,
            "merge_manifest_sha256": merge_manifest_hash,
        },
    )
    full.atomic_json(
        output / "oracle_manifest.json",
        {"schema_version": "test", **common, "frozen": True},
    )
    full.atomic_json(
        output / "evaluation_manifest.json",
        {"schema_version": "test", **common, "status": "SEALED", "frozen": True},
    )
    inventory = full._immutable_artifact_inventory(output)
    full.atomic_json(
        output / "freeze_manifest.json",
        {
            "schema_version": "test",
            **common,
            "status": "SEALED",
            "frozen": True,
            "files": inventory,
            "inventory_sha256": full.stable_sha256(inventory),
        },
    )
    full.atomic_json(
        output / "run_manifest.json",
        {
            "schema_version": full.RUN_MANIFEST_SCHEMA,
            **common,
            "status": "SEALED",
            "state": "SEALED",
            "run_complete": False,
            "raw_output_complete": True,
            "source_artifacts_complete": True,
            "frozen": True,
            "config": config.to_dict(),
            "train_parent_count": len(train_parents),
            "train_parent_cohort_sha256": train_cohort_hash,
            "branch_manifests": branch_manifest_hashes,
            "merge_manifest_sha256": merge_manifest_hash,
        },
    )
    resume_identity = {"test": True}
    full.write_checkpoint(
        output, phase="SEALED", resume_identity=resume_identity
    )
    (output / "SEALED").write_text("SEALED\n", encoding="utf-8")

    audit = full.verify_t13_output(output)
    assert audit["passed"] is True
    assert audit["registry_status"] in {"FROZEN_PASS", "ADOPTABLE_PASS"}
    assert (output / "PASS").read_text(encoding="utf-8") == "PASS\n"
    assert full.read_json(output / "run_manifest.json")["run_complete"] is True
