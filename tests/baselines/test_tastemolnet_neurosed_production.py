from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data.tastemolnet_neurosed_fixed_budget import (
    FixedBudgetGraph,
    bind_fixed_budget_pair_manifest,
    fixed_budget_pair_manifest,
    sample_fixed_budget_pairs,
)
from src.data.tastemolnet_neurosed_production import (
    FixedBudgetPairInventory,
    load_fixed_budget_pair_inventory,
    read_compact_npz,
    sha256_file,
    write_compact_npz,
)
from src.eval import tastemolnet_neurosed_label_writer as writer
from src.train.tastemolnet_neurosed_fixed_budget import (
    FixedBudgetNeuroSEDTrainConfig,
    _replay_selector_trace,
)
from src.train.tastemolnet_neurosed_official_selector import (
    OfficialBatchInterleavedSelector,
)


def _graph(graph_id: str, channel: int) -> FixedBudgetGraph:
    edges = []
    for index in range(6):
        edges.extend(((index, index + 1), (index + 1, index)))
    return FixedBudgetGraph(
        graph_id=graph_id,
        split="train",
        node_labels=tuple((channel + index) % 3 for index in range(7)),
        directed_edges=tuple(edges),
        scaffold=f"scaffold-{graph_id}",
        class_label=channel,
    )


def _jsonl(rows) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def test_exact_budget_pair_inventory_reopens_without_inventing_reserve(
    tmp_path: Path,
) -> None:
    graphs = [_graph(f"g-{index}", index % 3) for index in range(4)]
    pairs = sample_fixed_budget_pairs(
        graphs,
        split="train",
        pair_count=2,
        seed=7,
        n_hops_query=8,
        traversal_probability_query=1.0,
    )
    pair_text = _jsonl(pair.metadata() for pair in pairs)
    graph_rows = [
        {
            "graph_id": graph.graph_id,
            "split": graph.split,
            "node_labels": list(graph.node_labels),
            "directed_edges": [list(edge) for edge in graph.directed_edges],
            "scaffold": graph.scaffold,
            "class_label_sampling_diagnostic_only": graph.class_label,
            "graph_sha256": graph.graph_sha256,
            "canonical_graph_sha256": graph.canonical_graph_sha256,
        }
        for graph in graphs
    ]
    graph_text = _jsonl(graph_rows)
    (tmp_path / "pairs.jsonl").write_text(pair_text, encoding="utf-8")
    (tmp_path / "graph_inventory.jsonl").write_text(graph_text, encoding="utf-8")
    manifest = fixed_budget_pair_manifest(
        pairs,
        split="train",
        seed=7,
        n_hops_query=8,
        traversal_probability_query=1.0,
        node_limit_query=None,
    )
    manifest.update(
        {
            "source_csv_sha256": "a" * 64,
            "feature_schema_sha256": "b" * 64,
            "graph_inventory_sha256": sha256_file(tmp_path / "graph_inventory.jsonl"),
            "pairs_jsonl_sha256": sha256_file(tmp_path / "pairs.jsonl"),
        }
    )
    manifest = bind_fixed_budget_pair_manifest(manifest)
    (tmp_path / "pair_sampler_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    reopened = load_fixed_budget_pair_inventory(
        tmp_path, split="train", requested_pair_count=2
    )
    assert len(reopened.pairs) == 2
    assert reopened.reserve_available_count == 0


def _compact_row(status: str) -> dict:
    success = status == "SUCCESS"
    return {
        "pair_id": "1" * 64,
        "query_graph_id": "query",
        "target_graph_id": "target",
        "query_hash": "2" * 64,
        "target_hash": "3" * 64,
        "split": "train",
        "lower_bound": 1.0 if success else None,
        "upper_bound": 2.0 if success else None,
        "exact_bound": False,
        "label_contract": "ordered_query_target_lower_upper_interval",
        "backend": "branch",
        "backend_config_hash": "4" * 64,
        "status": status,
        "elapsed_seconds": 0.1,
        "cache_key": "5" * 64,
        "cache_hit": False,
        "error": "" if success else "failure",
    }


@pytest.mark.parametrize("status", ["SUCCESS", "TIMEOUT", "GEDLIB_ERROR"])
def test_compact_npz_round_trip_supports_failure_null_bounds(
    tmp_path: Path, status: str
) -> None:
    path = tmp_path / "labels.npz"
    write_compact_npz(path, [_compact_row(status)])
    assert read_compact_npz(path)[0]["status"] == status
    if status != "SUCCESS":
        assert read_compact_npz(path)[0]["lower_bound"] is None


def _inventory(pair_count: int = 2) -> FixedBudgetPairInventory:
    pairs = tuple(SimpleNamespace(pair_id=f"pair-{index}") for index in range(pair_count))
    return FixedBudgetPairInventory(
        root=Path("/tmp/frozen"),
        split="train",
        requested_pair_count=pair_count,
        manifest={},
        manifest_file_sha256="a" * 64,
        pairs_file_sha256="b" * 64,
        graph_inventory_file_sha256="c" * 64,
        reserve_available_count=0,
        pairs=pairs,
    )


def test_exact_inventory_requires_every_ged_label_to_succeed() -> None:
    observations = [
        {"pair_id": "pair-0", "status": "SUCCESS"},
        {"pair_id": "pair-1", "status": "GEDLIB_ERROR"},
    ]
    selected = writer.select_inventory_labels(_inventory(), observations)
    assert selected["status"] == "BLOCKED_GEDLIB_LABEL_YIELD"
    assert selected["reserve_used"] == 0
    assert selected["successful_pair_count"] == 1


def test_verified_selection_replay_cache_adopts_exact_train_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs = []
    canary_rows = []
    observations = []
    for index in range(100):
        pair_id = f"{index:064x}"
        query_hash = f"{index + 100:064x}"
        target_hash = f"{index + 200:064x}"
        query_graph_id = f"query-{index}"
        target_graph_id = f"target-{index}"
        pairs.append(
            SimpleNamespace(
                pair_id=pair_id,
                metadata={
                    "query_graph_id": query_graph_id,
                    "target_graph_id": target_graph_id,
                },
                query=SimpleNamespace(canonical_graph_sha256=query_hash),
                target=SimpleNamespace(canonical_graph_sha256=target_hash),
            )
        )
        canary_rows.append({"pair_id": pair_id})
        observations.append(
                {
                    "pair_id": pair_id,
                    "query_graph_id": query_graph_id,
                    "target_graph_id": target_graph_id,
                    "status": "SUCCESS",
                "lower_bound": 1.0,
                "upper_bound": 2.0,
                    "exact_bound": False,
                    "error": "",
                "query_canonical_graph_sha256": query_hash,
                "target_canonical_graph_sha256": target_hash,
            }
        )
    canary_path = tmp_path / "pairs.jsonl"
    left_path = tmp_path / "left.jsonl"
    right_path = tmp_path / "right.jsonl"
    canary_path.write_text(_jsonl(canary_rows), encoding="utf-8")
    left_path.write_text(_jsonl(observations), encoding="utf-8")
    right_path.write_text(_jsonl(observations), encoding="utf-8")
    selection = {
        "selected_ged_backend": "branch",
        "pairs_jsonl_path": str(canary_path),
        "candidate_reports": {
            "branch": {
                "replays": [
                    {"outcome_sha256": "a" * 64, "observations_path": str(left_path)},
                    {"outcome_sha256": "a" * 64, "observations_path": str(right_path)},
                ]
            }
        },
    }
    monkeypatch.setattr(
        writer, "validate_non_mip_selection_manifest", lambda *_args, **_kwargs: selection
    )
    monkeypatch.setattr(writer, "load_json", lambda _path: selection)
    inventory = _inventory(100)
    object.__setattr__(inventory, "pairs", tuple(pairs))
    authority = SimpleNamespace(
        method="branch",
        gedlib_config_sha256="d" * 64,
        feature_schema_sha256="e" * 64,
    )
    cache = writer.load_verified_selection_cache(
        tmp_path / "selection.json",
        authority=authority,
        train_inventory=inventory,
    )
    assert len(cache) == 100
    assert all(row["verified_selection_cache"] for row in cache.values())


def test_production_train_config_is_frozen_to_official_notebook() -> None:
    config = FixedBudgetNeuroSEDTrainConfig()
    config.validate()
    assert config.train_batch_size == 200
    assert config.validation_batch_size == 1000
    assert config.cycle_patience == 5
    with pytest.raises(ValueError, match="integer config changed"):
        FixedBudgetNeuroSEDTrainConfig(train_batch_size=128).validate()


def test_independent_selector_replay_rejects_trace_drift() -> None:
    selector = OfficialBatchInterleavedSelector(
        cycle_patience=1, step_size_up=1, step_size_down=1
    )
    for index, metric in enumerate((1.0, 1.0, 1.0, 1.0)):
        decision = selector.observe_validation(metric, training_batch_index=index)
        if decision.checkpoint_candidate:
            selector.bind_checkpoint_candidate(
                validation_event_index=decision.validation_event_index,
                checkpoint_sha256="a" * 64,
            )
        if decision.stop_before_training_batch:
            break
        selector.record_training_update(
            training_batch_index=index,
            optimizer_step_completed=True,
            cyclic_lr_step_completed=True,
            gradient_clip_norm=0.1,
        )
    trace = selector.trace_manifest()
    _replay_selector_trace(trace)
    trace["trace"][1]["validation_metric"] = 0.5
    with pytest.raises(RuntimeError, match="decision trace changed"):
        _replay_selector_trace(trace)


def test_production_trainer_does_not_import_historical_epoch_route() -> None:
    source = Path("src/train/tastemolnet_neurosed_fixed_budget.py").read_text(
        encoding="utf-8"
    )
    assert "from src.train.tastemolnet_neurosed import" not in source
    assert "OfficialBatchInterleavedSelector" in source
    assert '"best_pt_semantics": "official_selector_preupdate_candidate_bytes"' in source
    assert '"best_and_model_bytes_identical": True' in source


def test_slurm_wrappers_match_new_entrypoints() -> None:
    for stem in ("write_fixed_budget_ged_labels", "train_fixed_budget_neurosed"):
        text = Path(f"scripts/slurm/{stem}.sh").read_text(encoding="utf-8")
        assert "#SBATCH --partition=A800" in text
        assert "#SBATCH --gres=gpu:a800:1" in text
        assert "export PYTHONPATH=$PWD" in text
        assert f"scripts/tastemolnet/{stem}.py" in text
        assert "--config configs/hpc.yaml" in text
        assert "--set inference.fallback_to_heuristic=false" in text
    trainer_wrapper = Path(
        "scripts/slurm/train_fixed_budget_neurosed.sh"
    ).read_text(encoding="utf-8")
    assert trainer_wrapper.count(
        "scripts/tastemolnet/train_fixed_budget_neurosed.py"
    ) == 1
    assert "--train-and-verify" in trainer_wrapper


def _load_neurosed_cli_module():
    script = Path("scripts/tastemolnet/train_fixed_budget_neurosed.py").resolve()
    spec = importlib.util.spec_from_file_location("production_neurosed_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_neurosed_path_normalization_accepts_str_path_relative_and_absolute(
    tmp_path: Path,
) -> None:
    module = _load_neurosed_cli_module()
    target = tmp_path / "input.json"
    target.write_text("{}\n", encoding="utf-8")
    assert module.normalize_path(str(target)) == target.resolve()
    assert module.normalize_path(target) == target.resolve()
    assert module.normalize_path(
        Path("input.json"), relative_to=tmp_path
    ) == target.resolve()


def test_neurosed_path_normalization_resolves_symlinks_and_fails_closed(
    tmp_path: Path,
) -> None:
    module = _load_neurosed_cli_module()
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    assert module.normalize_path(link) == target.resolve()
    with pytest.raises(ValueError, match="does not exist"):
        module.normalize_path(tmp_path / "missing.json")
    with pytest.raises(TypeError, match="str or pathlib.Path"):
        module.normalize_path(7)


def test_single_cli_train_and_verify_uses_independent_process_and_pass_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_neurosed_cli_module()
    output_root = tmp_path / "run"
    subprocess_calls: list[list[str]] = []

    for directory in (tmp_path / "labels", tmp_path / "train", tmp_path / "validation", tmp_path / "gcf"):
        directory.mkdir()
    for file_path in (
        tmp_path / "features.json",
        tmp_path / "selection.json",
        tmp_path / "receipt.json",
    ):
        file_path.write_text("{}\n", encoding="utf-8")

    def fake_train(**kwargs):
        Path(kwargs["output_root"]).mkdir()
        return {"state": "READY_FOR_MANAGED_INDEPENDENT_VERIFICATION"}

    def fake_run(command, *, check):
        assert check is True
        subprocess_calls.append(command)
        assert command[-1] == "--verify-existing-root"
        (output_root / "PASS").write_text(
            module.NEUROSED_PASS_MARKER + "\n", encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module, "train_fixed_budget_neurosed", fake_train)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    common = [
        "--config", "configs/hpc.yaml",
        "--set", "inference.fallback_to_heuristic=false",
        "--ged-label-root", str(tmp_path / "labels"),
        "--train-pair-root", str(tmp_path / "train"),
        "--validation-pair-root", str(tmp_path / "validation"),
        "--feature-schema-json", str(tmp_path / "features.json"),
        "--non-mip-selection-manifest", str(tmp_path / "selection.json"),
        "--non-mip-verifier-receipt", str(tmp_path / "receipt.json"),
        "--vendored-gcf-root", str(tmp_path / "gcf"),
        "--output-root", str(output_root),
        "--execution-git-commit", "a" * 40,
        "--execution-git-tree", "b" * 40,
        "--device", "cpu",
        "--train-and-verify",
    ]
    assert module.main(common) == 0
    assert len(subprocess_calls) == 1
    assert (output_root / "PASS").read_text(encoding="utf-8") == (
        module.NEUROSED_PASS_MARKER + "\n"
    )


def test_independent_verifier_is_the_only_neurosed_pass_writer() -> None:
    source = Path("src/train/tastemolnet_neurosed_fixed_budget.py").read_text(
        encoding="utf-8"
    )
    train_body, verify_body = source.split("def verify_fixed_budget_neurosed(", 1)
    assert "NEUROSED_PASS_MARKER" not in train_body.split(
        "def train_fixed_budget_neurosed(", 1
    )[1]
    assert '_write_new(destination / "PASS"' in verify_body
    assert verify_body.index('_write_json(destination / "verification.json"') < (
        verify_body.index('_write_new(destination / "PASS"')
    )
