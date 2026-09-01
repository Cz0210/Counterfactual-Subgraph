from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import train_tastemolnet_ours_full as train_full
from src.eval import tastemolnet_ours_full as ours
from src.eval.four_by_four_registry import PASS_STATUSES, audit_explicit_candidate


def _threshold(tmp_path: Path) -> tuple[Path, ours.ThresholdContract]:
    values = [0.5, 1.0]
    path = tmp_path / "threshold.json"
    path.write_text(
        json.dumps(
            {
                "dataset": "TasteMolNet",
                "thresholds": values,
                "theta_star": 0.5,
                "cost_cap": 1.0,
                "threshold_source": "frozen_shared_calibration",
                "threshold_source_split": "calibration",
                "threshold_config_hash": ours.stable_sha256(values),
                "test_used_for_selection": False,
                "selection_used_test": False,
                "threshold_fitted_on_test": False,
                "shared_across_methods": True,
                "cf_mode": "strict_flip",
            }
        ),
        encoding="utf-8",
    )
    return path, ours.load_threshold_contract(path)


def _pairs(candidate_ids: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for parent_index, parent in enumerate(("P0", "P1")):
        for index, candidate in enumerate(candidate_ids):
            strict = index == parent_index
            rows.append(
                {
                    "dataset": "TasteMolNet",
                    "method": "Ours",
                    "split": "test",
                    "parent_id": parent,
                    "candidate_id": candidate,
                    "pair_strict_flip": strict,
                    "wnode_distance": 0.25 + parent_index * 0.1 if strict else None,
                    "destination_label": (0, 2)[parent_index] if strict else None,
                    "cf_drop": 0.7 if strict else None,
                    "applicable": True,
                    "rf_oracle_used": False,
                }
            )
    return rows


def test_full_ppo_schedule_is_real_and_restartable() -> None:
    assert train_full.UPDATES == 300
    assert train_full.CHECKPOINT_STEPS == (50, 100, 150, 200, 250, 300)
    source = Path(train_full.__file__).read_text(encoding="utf-8")
    assert "run_stable_decoded_chem_ppo_loop(" in source
    assert "resume_from_checkpoint=resume_checkpoint" in source
    assert "calibration_loaded\": False" in source
    assert "test_loaded\": False" in source


def test_shared_threshold_is_required_and_excludes_test(tmp_path: Path) -> None:
    path, contract = _threshold(tmp_path)
    assert contract.theta_star == 0.5
    clean = json.loads(path.read_text(encoding="utf-8"))
    for field, invalid in (
        ("test_used_for_selection", True),
        ("selection_used_test", True),
        ("threshold_fitted_on_test", True),
        ("shared_across_methods", False),
        ("cf_mode", "best_effort"),
    ):
        path.write_text(json.dumps({**clean, field: invalid}), encoding="utf-8")
        with pytest.raises(ours.TasteOursFullError, match="shared|exclude test"):
            ours.load_threshold_contract(path)


def test_base_high_merge_is_canonical_and_train_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ours, "canonicalize_smiles", lambda value: value or None)
    rows = []
    for index in range(20):
        for mode in ("base", "high_temp"):
            rows.append(
                {
                    "canonical_fragment": f"F{index}",
                    "parent_id": f"P{index % 3}",
                    "stage": mode,
                    "parse_ok": True,
                    "connected": True,
                    "direct_substructure": True,
                    "oracle_ok": True,
                    "cf_flip": index % 2 == 0,
                    "cf_drop": float(index) / 20,
                }
            )
    universe = ours.merge_candidate_modes(rows)
    assert len(universe) == 20
    assert all(row["source_modes"] == ["base", "high_temp"] for row in universe)
    assert len({row["candidate_id"] for row in universe}) == 20


def test_calibration_selector_and_test_metrics_use_frozen_prefix(tmp_path: Path) -> None:
    candidate_ids = [f"C{index}" for index in range(10)]
    candidates = [{"candidate_id": value, "canonical_fragment": value} for value in candidate_ids]
    calibration = []
    for parent in ("A", "B"):
        for index, candidate in enumerate(candidate_ids):
            strict = index < 2
            calibration.append(
                {
                    "split": "calibration",
                    "parent_id": parent,
                    "candidate_id": candidate,
                    "pair_strict_flip": strict,
                    "wnode_distance": 0.2 + 0.01 * index if strict else None,
                }
            )
    selected, trace = ours.select_on_calibration(candidates, calibration, theta_star=0.5)
    assert len(selected) == 10
    assert len(trace) == 10
    _path, threshold = _threshold(tmp_path)
    metrics = ours.standardized_metrics(_pairs([row["candidate_id"] for row in selected]), [row["candidate_id"] for row in selected], threshold)
    assert [row["k"] for row in metrics["figure3"]] == list(range(1, 21))
    assert metrics["table2"][0]["k"] == 10
    assert {row["destination_label"] for row in metrics["destination"]} == {0, 2}


class _FakeScorer:
    checkpoint_id = "a" * 64

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def score_smiles(self, values: list[str]) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for value in values:
            if value.startswith("RESIDUAL::"):
                parent_id = value.split("::", 2)[1]
                probabilities = (
                    [0.05, 0.10, 0.85]
                    if parent_id.endswith("1")
                    else [0.85, 0.10, 0.05]
                )
            else:
                probabilities = [0.05, 0.90, 0.05]
            result.append(
                {
                    "predicted_label": max(
                        range(len(probabilities)), key=probabilities.__getitem__
                    ),
                    "probabilities": probabilities,
                }
            )
        return result


class _FakeDistance:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def distance_for_action(
        self, _parent: str, _residual: str, *, action_context: dict[str, object]
    ) -> dict[str, object]:
        assert (
            action_context["distance_implementation_version"]
            == ours.DISTANCE_IMPLEMENTATION_VERSION
        )
        candidate = str(action_context["candidate_id"])
        return {"ok": True, "distance": 0.1 + int(ours.stable_sha256(candidate)[:4], 16) / 1_000_000}

    def stats_dict(self) -> dict[str, int]:
        return {"fake_distance_calls": 1}

    def close(self) -> None:
        pass


def _fake_outcomes(
    _parent: str, _fragment: str, *, parent_id: str, candidate_id: str
) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            valid=True,
            residual_smiles=f"RESIDUAL::{parent_id}::{candidate_id}",
            match_id=0,
            match_atom_indices=(0,),
        )
    ]


def _write_fake_generation_mode(
    *,
    parents: list[ours.TrainParent],
    authority: ours.OursAuthority,
    scorer: _FakeScorer,
    output: Path,
    config: ours.GenerationConfig,
) -> list[dict[str, object]]:
    root = output / "raw" / "generation" / config.name
    chunks = root / "parent_chunks"
    receipts = root / "parent_chunk_receipts"
    chunks.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    identity = ours.stable_sha256(
        {
            "authority": authority.identity(),
            "config": ours.asdict(config),
            "parents": [(parent.parent_id, parent.smiles) for parent in parents],
        }
    )
    all_rows: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    for position, parent in enumerate(parents):
        rows = ours._score_generation(
            parent=parent,
            raw_outputs=[
                f"{config.name}_fragment_{position}_{index}"
                for index in range(config.num_return_sequences)
            ],
            scorer=scorer,
            config=config,
        )
        chunk = chunks / f"{position:08d}.jsonl"
        receipt_path = receipts / f"{position:08d}.json"
        chunk_identity = {
            "mode": config.name,
            "position": position,
            "parent_id": parent.parent_id,
            "parent_smiles": parent.smiles,
            "parent_split": parent.split,
            "generation_parent_seed": ours._parent_generation_seed(
                config, parent.parent_id
            ),
            "generation_identity": identity,
            "oracle_checkpoint_hash": scorer.checkpoint_id,
            "row_schema_sha256": ours.stable_sha256(
                sorted(ours.GENERATION_ROW_FIELDS)
            ),
        }
        receipt = ours._commit_closed_chunk(
            chunk=chunk,
            receipt_path=receipt_path,
            rows=rows,
            schema_version=ours.GENERATION_CHUNK_RECEIPT_SCHEMA,
            identity=chunk_identity,
            label="test generation",
        )
        all_rows.extend(rows)
        inventory.append(
            {
                "position": position,
                "chunk": str(chunk.relative_to(root)),
                "receipt": str(receipt_path.relative_to(root)),
                "chunk_sha256": receipt["chunk_sha256"],
                "chunk_bytes": receipt["chunk_bytes"],
                "receipt_sha256": ours.sha256_file(receipt_path),
            }
        )
    pool = root / "candidate_pool.jsonl"
    ours.atomic_jsonl(pool, all_rows)
    ours.atomic_json(
        root / "generation_manifest.json",
        {
            "schema_version": ours.GENERATION_MANIFEST_SCHEMA,
            "status": "PASS",
            "mode": config.name,
            "identity": identity,
            "config": ours.asdict(config),
            "parent_count": len(parents),
            "candidate_count": len(all_rows),
            "parent_inventory_sha256": ours.stable_sha256(
                [
                    (parent.parent_id, parent.smiles, parent.split)
                    for parent in parents
                ]
            ),
            "chunk_inventory": inventory,
            "chunk_inventory_sha256": ours.stable_sha256(inventory),
            "candidate_pool_sha256": ours.sha256_file(pool),
            "train_only": True,
            "calibration_loaded": False,
            "test_loaded": False,
            "rf_oracle_used": False,
            "row_schema_sha256": ours.stable_sha256(
                sorted(ours.GENERATION_ROW_FIELDS)
            ),
            "resume_semantic_replay_required": True,
            "semantic_validation": "raw_canonical_deletion_frozen_gine_replay",
        },
    )
    return all_rows


def _sealed_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    threshold_path, threshold = _threshold(tmp_path)
    train = [
        ours.TrainParent(f"TRAIN{index}", f"parent_train_{index}", 1, "train")
        for index in range(3)
    ]
    calibration = [
        ours.TrainParent(f"CAL{index}", f"parent_cal_{index}", 1, "calibration")
        for index in range(2)
    ]
    test = [
        ours.TrainParent(f"TEST{index}", f"parent_test_{index}", 1, "test")
        for index in range(2)
    ]
    splits = {"train": train, "calibration": calibration, "test": test}

    def fake_split(
        _path: Path, *, expected_split: str, expected_sha256: str
    ) -> list[ours.TrainParent]:
        assert len(expected_sha256) == 64
        return list(splits[expected_split])

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ours.GINE_PAYLOAD_FILES:
        (checkpoint / name).write_bytes(b"fixture")
    for name in ("ppo", "policy", "base_model", "molclr"):
        (tmp_path / name).mkdir()
    molclr_checkpoint = tmp_path / "molclr.pt"
    molclr_checkpoint.write_bytes(b"molclr")
    authority = ours.OursAuthority(
        ppo_root=tmp_path / "ppo",
        policy_root=tmp_path / "policy",
        base_model=tmp_path / "base_model",
        checkpoint=checkpoint,
        train_path=tmp_path / "train.csv",
        calibration_path=tmp_path / "calibration.csv",
        test_path=tmp_path / "test.csv",
        molclr_root=tmp_path / "molclr",
        molclr_checkpoint=molclr_checkpoint,
        threshold_path=threshold_path,
        policy_hash="1" * 64,
        checkpoint_id=_FakeScorer.checkpoint_id,
        dataset_hash="b" * 64,
        temperature_calibration_hash="c" * 64,
        feature_schema_hash="d" * 64,
        feature_schema_file_sha256="e" * 64,
        train_sha256="2" * 64,
        calibration_sha256="3" * 64,
        declared_test_sha256="4" * 64,
        split_manifest_sha256="5" * 64,
        molclr_checkpoint_sha256="6" * 64,
        threshold=threshold,
    )
    monkeypatch.setattr(ours, "TasteGINEScorer", _FakeScorer)
    monkeypatch.setattr(ours, "MolCLRNodeWassersteinDistance", _FakeDistance)
    monkeypatch.setattr(ours, "load_prepared_split", fake_split)
    monkeypatch.setattr(ours, "canonicalize_smiles", lambda value: value or None)
    monkeypatch.setattr(ours, "clean_generated_smiles", lambda value: value)
    monkeypatch.setattr(ours, "enumerate_connected_hard_deletions", _fake_outcomes)
    monkeypatch.setattr(ours, "generate_mode_resumable", _write_fake_generation_mode)
    science = tmp_path / "science"
    ours.run_science(
        authority=authority,
        output_dir=science,
        resume=False,
        device="cuda:0",
        wnode_cache_db=tmp_path / "wnode.sqlite",
        node_embedding_cache_dir=tmp_path / "node-cache",
    )
    return threshold_path, science


def _refresh_terminal_hashes(science: Path, *names: str) -> None:
    freeze = ours.read_json(science / "freeze_manifest.json")
    for name in names:
        freeze["files"][name] = {
            "sha256": ours.sha256_file(science / name),
            "bytes": (science / name).stat().st_size,
        }
    freeze["inventory_sha256"] = ours.stable_sha256(freeze["files"])
    ours.atomic_json(science / "freeze_manifest.json", freeze)
    run = ours.read_json(science / "run_manifest.json")
    run["freeze_manifest_sha256"] = ours.sha256_file(
        science / "freeze_manifest.json"
    )
    ours.atomic_json(science / "run_manifest.json", run)


def test_generation_semantic_replay_accepts_bounded_cuda_roundoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ours, "canonicalize_smiles", lambda value: value or None)
    monkeypatch.setattr(ours, "clean_generated_smiles", lambda value: value)
    monkeypatch.setattr(ours, "enumerate_connected_hard_deletions", _fake_outcomes)
    parent = ours.TrainParent("TRAIN0", "parent", 1, "train")
    config = ours.GenerationConfig("base", 0.3, 0.9, 1, 7)
    saved = ours._score_generation(
        parent=parent,
        raw_outputs=["fragment"],
        scorer=_FakeScorer(),
        config=config,
    )

    class PerturbedScorer(_FakeScorer):
        def score_smiles(self, values: list[str]) -> list[dict[str, object]]:
            rows = super().score_smiles(values)
            for value, row in zip(values, rows, strict=True):
                probabilities = list(row["probabilities"])
                # Move the source-class probability in opposite directions for
                # parent and residual rows.  This exercises the bounded replay
                # path for both the probability vectors and the derived
                # cf_drop/reward_total values while preserving normalization.
                drift = -5e-9 if value.startswith("RESIDUAL::") else 5e-9
                probabilities[0] -= drift
                probabilities[1] += drift
                row["probabilities"] = probabilities
            return rows

    ours._require_generation_semantic_replay(
        saved,
        parent=parent,
        scorer=PerturbedScorer(),
        config=config,
    )

    class DriftedScorer(_FakeScorer):
        def score_smiles(self, values: list[str]) -> list[dict[str, object]]:
            rows = super().score_smiles(values)
            for row in rows:
                probabilities = list(row["probabilities"])
                probabilities[0] += 2 * ours.GENERATION_REPLAY_ABS_TOL
                probabilities[2] -= 2 * ours.GENERATION_REPLAY_ABS_TOL
                row["probabilities"] = probabilities
            return rows

    with pytest.raises(ours.TasteOursFullError, match="frozen-GINE replay"):
        ours._require_generation_semantic_replay(
            saved,
            parent=parent,
            scorer=DriftedScorer(),
            config=config,
        )


def test_generation_semantic_replay_accepts_only_proven_symmetric_match_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ours, "canonicalize_smiles", lambda value: value or None)
    monkeypatch.setattr(ours, "clean_generated_smiles", lambda value: value)
    parent = ours.TrainParent("TRAIN0", "parent", 1, "train")
    config = ours.GenerationConfig("base", 0.3, 0.9, 1, 7)

    def symmetric_outcomes(
        _parent: str, _fragment: str, *, parent_id: str, candidate_id: str
    ) -> list[SimpleNamespace]:
        residual = f"RESIDUAL::{parent_id}::{candidate_id}"
        return [
            SimpleNamespace(
                valid=True,
                residual_smiles=residual,
                match_id=11,
                match_atom_indices=(0,),
            ),
            SimpleNamespace(
                valid=True,
                residual_smiles=residual,
                match_id=17,
                match_atom_indices=(1,),
            ),
        ]

    monkeypatch.setattr(
        ours, "enumerate_connected_hard_deletions", symmetric_outcomes
    )
    saved = ours._score_generation(
        parent=parent,
        raw_outputs=["fragment"],
        scorer=_FakeScorer(),
        config=config,
    )
    assert saved[0]["selected_match_index"] == 11
    saved[0]["selected_match_index"] = 17
    ours._require_generation_semantic_replay(
        saved,
        parent=parent,
        scorer=_FakeScorer(),
        config=config,
    )


def test_generation_semantic_replay_rejects_match_index_with_other_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ours, "canonicalize_smiles", lambda value: value or None)
    monkeypatch.setattr(ours, "clean_generated_smiles", lambda value: value)
    parent = ours.TrainParent("TRAIN0", "parent", 1, "train")
    config = ours.GenerationConfig("base", 0.3, 0.9, 1, 7)

    def distinct_outcomes(
        _parent: str, _fragment: str, *, parent_id: str, candidate_id: str
    ) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                valid=True,
                residual_smiles=f"RESIDUAL::{parent_id}::{candidate_id}",
                match_id=11,
                match_atom_indices=(0,),
            ),
            SimpleNamespace(
                valid=True,
                residual_smiles=f"OTHER::{parent_id}::{candidate_id}",
                match_id=17,
                match_atom_indices=(1,),
            ),
        ]

    monkeypatch.setattr(
        ours, "enumerate_connected_hard_deletions", distinct_outcomes
    )
    saved = ours._score_generation(
        parent=parent,
        raw_outputs=["fragment"],
        scorer=_FakeScorer(),
        config=config,
    )
    assert saved[0]["selected_match_index"] == 11
    saved[0]["selected_match_index"] = 17
    with pytest.raises(ours.TasteOursFullError, match="frozen-GINE replay"):
        ours._require_generation_semantic_replay(
            saved,
            parent=parent,
            scorer=_FakeScorer(),
            config=config,
        )


def test_generation_and_pair_semantic_replay_reject_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ours, "canonicalize_smiles", lambda value: value or None)
    monkeypatch.setattr(ours, "clean_generated_smiles", lambda value: value)
    monkeypatch.setattr(ours, "enumerate_connected_hard_deletions", _fake_outcomes)
    scorer = _FakeScorer()
    parent = ours.TrainParent("TRAIN0", "parent", 1, "train")
    config = ours.GenerationConfig("base", 0.3, 0.9, 1, 7)
    generated = ours._score_generation(
        parent=parent, raw_outputs=["fragment"], scorer=scorer, config=config
    )
    generated[0].update(
        {
            "p_after": [0.70, 0.20, 0.10],
            "cf_drop": 0.70,
            "reward_total": 1.70,
        }
    )
    ours._validate_generation_rows(
        generated,
        parent=parent,
        config=config,
        oracle_checkpoint_hash=scorer.checkpoint_id,
    )
    with pytest.raises(ours.TasteOursFullError, match="frozen-GINE replay"):
        ours._require_generation_semantic_replay(
            generated, parent=parent, scorer=scorer, config=config
        )

    pair_parent = ours.TrainParent("TEST0", "parent", 1, "test")
    candidates = [{"candidate_id": "C0", "canonical_fragment": "fragment"}]
    identity = ours.build_pair_evaluation_identity(
        split="test",
        parents=[pair_parent],
        candidates=candidates,
        oracle_checkpoint_hash=scorer.checkpoint_id,
        temperature_calibration_hash="b" * 64,
        feature_schema_hash="c" * 64,
        molclr_checkpoint_hash="d" * 64,
        threshold_config_hash="e" * 64,
        threshold_contract_file_sha256="f" * 64,
    )
    distance = _FakeDistance()
    pairs = ours.evaluate_parent(
        parent=pair_parent,
        candidates=candidates,
        scorer=scorer,
        distance=distance,
        split="test",
        evaluation_identity=identity,
    )
    pairs[0]["wnode_distance"] = 999999.0
    pairs[0]["distance_for_selection"] = 999999.0
    ours._validate_pair_rows(
        pairs,
        parent=pair_parent,
        candidates=candidates,
        split="test",
        evaluation_identity=identity,
    )
    with pytest.raises(ours.TasteOursFullError, match="GINE/WNode replay"):
        ours._require_pair_semantic_replay(
            pairs,
            parent=pair_parent,
            candidates=candidates,
            scorer=scorer,
            distance=distance,
            split="test",
            evaluation_identity=identity,
        )


def test_independent_verifier_replays_and_publishes_fresh_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    threshold_path, science = _sealed_science(tmp_path, monkeypatch)
    final = tmp_path / "final"
    manifest = ours.verify_and_publish(
        science_root=science,
        final_root=final,
        threshold_contract=threshold_path,
    )
    assert manifest["status"] == "PASS"
    assert manifest["base_high_merge_replayed"] is True
    assert (final / "PASS").read_text(encoding="utf-8").strip() == ours.PASS_MARKER
    audit = ours.read_json(final / "final_artifact_audit.json")
    assert audit["audit_passed"] is True
    assert audit["test_pair_chunks_replayed"] is True
    assert ours.sha256_file(final / "freeze_manifest.json") == audit["science_freeze_manifest_sha256"]
    assert ours.read_json(final / "summary.json")["status"] == "PASS"
    registry = audit_explicit_candidate(final, dataset="TasteMolNet", method="Ours")
    assert registry.status in PASS_STATUSES, registry.reason_codes


def test_terminal_verifier_replays_base_high_merge_after_rehashed_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    threshold_path, science = _sealed_science(tmp_path, monkeypatch)
    universe_path = science / "raw" / "candidate_universe.jsonl"
    ours.atomic_jsonl(universe_path, list(reversed(ours.read_jsonl(universe_path))))
    merge_path = science / "raw" / "merge_manifest.json"
    merge = ours.read_json(merge_path)
    merge["candidate_universe_sha256"] = ours.sha256_file(universe_path)
    ours.atomic_json(merge_path, merge)
    _refresh_terminal_hashes(
        science, "raw/candidate_universe.jsonl", "raw/merge_manifest.json"
    )
    with pytest.raises(ours.TasteOursFullError, match="merge cannot be replayed"):
        ours.verify_and_publish(
            science_root=science,
            final_root=tmp_path / "final",
            threshold_contract=threshold_path,
        )


def test_terminal_verifier_rejects_rehashed_incomplete_test_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    threshold_path, science = _sealed_science(tmp_path, monkeypatch)
    manifest_path = science / "raw" / "test_evaluation_manifest.json"
    manifest = ours.read_json(manifest_path)
    manifest["pair_count"] = int(manifest["pair_count"]) + 1
    ours.atomic_json(manifest_path, manifest)
    _refresh_terminal_hashes(science, "raw/test_evaluation_manifest.json")
    with pytest.raises(ours.TasteOursFullError, match="incomplete"):
        ours.verify_and_publish(
            science_root=science,
            final_root=tmp_path / "final",
            threshold_contract=threshold_path,
        )


def test_worker_and_verifier_are_distinct_and_not_disabled() -> None:
    wrapper = Path("scripts/autodl/run_tastemolnet_ours_full.sh").read_text(encoding="utf-8")
    assert "train_tastemolnet_ours_full.py" in wrapper
    assert wrapper.count("run_tastemolnet_ours_full.py") == 2
    assert "--verify-only" in wrapper
    assert "DISABLED" not in wrapper
