from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from src.baselines.bace_gnn_baseline_contracts import (
    assert_gine_clean_manifest,
    baseline_spec,
    write_route_preflight,
)
from src.baselines.bace_gnn_baseline_tasks import (
    build_bace_baseline_controller_fragment,
)
from src.baselines.comrecgc.contracts import UPSTREAM_COMMIT
from src.baselines.bace_gine_native_adapter import (
    BACEFrozenGINENativeGraphAdapter,
)
from src.data.molecular_graph_featurizer import default_molecular_feature_schema
from src.eval.bace_frozen_gnn_contracts import atomic_json, atomic_jsonl, file_identity
from src.eval.bace_native_baseline_gnn import (
    BACE_OURS_B12_THRESHOLD_CONFIG_SHA256,
    CALIBRATION_STAGE,
    SELECTION_STAGE,
    TEST_STAGE,
    _authorize_split,
    _fullgraph_pair_rows,
    _load_candidates,
    freeze_native_baseline_final,
    run_native_baseline_selector,
)
from src.eval.mutagenicity_wnode_selector import ThresholdBundle, ThresholdLevel


CHECKPOINT_ID = "a" * 64
MOLCLR_ID = "b" * 64
OURS_THRESHOLD_VALUES = (
    0.0052997113994104825,
    0.006084341066708124,
    0.007764656724591062,
    0.008413518173529859,
    0.009050822647851559,
    0.010458106267325823,
    0.02956508038627219,
)
OURS_THRESHOLD_QUANTILES = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)
OURS_THRESHOLD_WEIGHTS = (4.0, 4.0, 3.0, 3.0, 2.0, 1.0, 1.0)


def _ours_threshold_payload() -> dict[str, object]:
    labels = tuple(f"q{int(quantile * 100):02d}" for quantile in OURS_THRESHOLD_QUANTILES)
    return ThresholdBundle(
        finite_distance_count=14,
        requested_quantiles=OURS_THRESHOLD_QUANTILES,
        requested_weights=OURS_THRESHOLD_WEIGHTS,
        raw_thresholds=OURS_THRESHOLD_VALUES,
        quantile_labels=labels,
        levels=tuple(
            ThresholdLevel(
                threshold_id=label,
                threshold=threshold,
                weight=weight,
                quantiles=(quantile,),
                quantile_labels=(label,),
            )
            for label, threshold, weight, quantile in zip(
                labels,
                OURS_THRESHOLD_VALUES,
                OURS_THRESHOLD_WEIGHTS,
                OURS_THRESHOLD_QUANTILES,
                strict=True,
            )
        ),
        theta_star_quantile=0.30,
        theta_star=OURS_THRESHOLD_VALUES[3],
        cost_cap_quantile=0.90,
        cost_cap=OURS_THRESHOLD_VALUES[-1],
    ).to_dict()


def _write_ours_b12_threshold_source(tmp_path: Path) -> Path:
    b11 = tmp_path / "bace/ours/b11-merged/attempt-0"
    b11.mkdir(parents=True)
    atomic_json(
        b11 / "matrix_manifest.json",
        {
            "schema_version": "bace_frozen_gnn_verification_merge_v1",
            "dataset": "bace",
            "stage": "B11_CROSS_PARENT_VERIFIED",
            "status": "PASS",
            "run_complete": True,
            "calibration_loaded": True,
            "test_loaded": False,
            "oracle_backend": "gnn",
            "classifier_type": "gnn",
            "rf_oracle_used": False,
            "source_label": 1,
            "num_classes": 2,
            "cf_mode": "strict_flip",
            "oracle_checkpoint_hash": CHECKPOINT_ID,
            "molclr_checkpoint_hash": MOLCLR_ID,
            "inputs": {"cohort_name": "calibration"},
        },
    )
    b12 = tmp_path / "bace/ours/b12-selector/attempt-0"
    b12.mkdir(parents=True)
    thresholds = _ours_threshold_payload()
    atomic_json(b12 / "thresholds.json", thresholds)
    atomic_json(
        b12 / "frozen_selection_manifest.json",
        {
            "schema_version": "bace_frozen_gnn_selection_manifest_v1",
            "dataset": "bace",
            "stage": "B12_SELECTOR",
            "status": "FROZEN",
            "selection_frozen": True,
            "selector_fitted_on_calibration": True,
            "calibration_loaded": True,
            "test_loaded": False,
            "test_used": False,
            "oracle_backend": "gnn",
            "classifier_type": "gnn",
            "rf_oracle_used": False,
            "source_label": 1,
            "num_classes": 2,
            "cf_mode": "strict_flip",
            "oracle_checkpoint_hash": CHECKPOINT_ID,
            "molclr_checkpoint_hash": MOLCLR_ID,
            "matrix_manifest_identity": file_identity(b11 / "matrix_manifest.json"),
            "thresholds": thresholds,
        },
    )
    return b12 / "thresholds.json"


def _candidate(method: str, index: int) -> dict[str, object]:
    spec = baseline_spec(method)
    return {
        "candidate_id": f"candidate-{index:02d}",
        "rank": index + 1,
        "native_rank": index + 1,
        "canonical_smiles": "C" * (index + 1),
        "canonical_fragment": "C" * (index + 1),
        "action_kind": spec.action_kind,
        "action_semantics": spec.action_semantics,
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "oracle_checkpoint_hash": CHECKPOINT_ID,
    }


def _fragment(tmp_path: Path, method: str) -> dict[str, object]:
    paths = {
        "python": tmp_path / "env/bin/python",
        "project_root": tmp_path / "project",
        "output_root": tmp_path / "output",
        "gnn_checkpoint": tmp_path / "gine",
        "dataset_dir": tmp_path / "dataset",
        "calibration_split": tmp_path / "bace_calibration.csv",
        "test_split": tmp_path / "bace_test.csv",
        "molclr_root": tmp_path / "molclr",
        "molclr_checkpoint": tmp_path / "molclr.pt",
        "neurosed_checkpoint": tmp_path / "neurosed.pt",
        "thresholds_json": tmp_path / "bace/ours/b12-selector/attempt-0/thresholds.json",
        "official_root": tmp_path / "official",
        "neurosed_manifest": tmp_path / "neurosed.json",
        "globalgce_source_manifest": tmp_path / "source_graph_manifest.jsonl",
        "globalgce_native_train_csv": tmp_path / "train.csv",
    }
    if method != "GCFExplainer":
        paths.pop("thresholds_json")
    return build_bace_baseline_controller_fragment(method=method, **paths)


def test_globalgce_action_and_frozen_gine_bridge_preflight_are_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = baseline_spec("GlobalGCE")
    assert spec.native_route_available is True
    assert spec.action_kind == "lhs_rhs_graph_transformation_rule"
    assert spec.blocker_code is None
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("model.pt", "temperature_scaling.json", "feature_schema.json"):
        (checkpoint / name).write_text(name, encoding="utf-8")
    card = {
        "dataset": "bace",
        "backbone": "gine",
        "oracle_backend": "gnn",
        "rf_oracle_used": False,
        "num_classes": 2,
        "source_label": 1,
        "checkpoint_id": CHECKPOINT_ID,
    }
    monkeypatch.setattr(
        "src.baselines.bace_gnn_baseline_contracts._checkpoint_contract",
        lambda _path: (card, object()),
    )
    monkeypatch.setattr(
        "src.baselines.globalgce_bace_native_rules.run_official_tensor_parity",
        lambda _root: {
            "schema_version": "globalgce_official_tensor_parity_v1",
            "status": "PASS",
            "official_commit": "157e65c2850bc787f229a1ee8c60564906b933f2",
        },
    )
    result = write_route_preflight(
        method="GlobalGCE",
        checkpoint_dir=checkpoint,
        output_dir=tmp_path / "preflight",
        official_root=tmp_path / "official",
    )
    assert result["status"] == "READY"
    assert result["native_action_status"] == "PASS"
    assert result["training_compatibility"][
        "exact_frozen_gine_gradient_to_continuous_rhs"
    ] is True
    assert "official_gtgnn" in result["training_compatibility"][
        "forbidden_fallbacks"
    ]
    assert (tmp_path / "preflight/READY").read_text().strip() == "READY"
    assert not (tmp_path / "preflight/BLOCKED_CODE").exists()
    assert (tmp_path / "preflight/NATIVE_ACTION_READY").is_file()
    assert not (tmp_path / "preflight/PASS").exists()


def test_comrecgc_preflight_validates_pinned_checkout_before_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("model.pt", "temperature_scaling.json", "feature_schema.json"):
        (checkpoint / name).write_text(name, encoding="utf-8")
    card = {
        "dataset": "bace",
        "backbone": "gine",
        "oracle_backend": "gnn",
        "rf_oracle_used": False,
        "num_classes": 2,
        "source_label": 1,
        "checkpoint_id": CHECKPOINT_ID,
    }
    monkeypatch.setattr(
        "src.baselines.bace_gnn_baseline_contracts._checkpoint_contract",
        lambda _path: (card, object()),
    )
    upstream_root = tmp_path / "COMRECGC"
    upstream_root.mkdir()
    observed: list[Path] = []

    def validate_checkout(path: str | Path) -> Path:
        resolved = Path(path).resolve()
        observed.append(resolved)
        return resolved

    monkeypatch.setattr(
        "src.baselines.bace_gnn_baseline_contracts.validate_upstream_checkout",
        validate_checkout,
    )

    result = write_route_preflight(
        method="ComRecGC",
        checkpoint_dir=checkpoint,
        output_dir=tmp_path / "preflight",
        official_root=upstream_root,
    )

    assert observed == [upstream_root.resolve()]
    assert result["status"] == "READY"
    assert result["upstream_checkout_validation"] == {
        "status": "PASS",
        "path": str(upstream_root.resolve()),
        "commit": UPSTREAM_COMMIT,
        "git_safe_directory_scope": "process_exact_path",
    }
    assert (tmp_path / "preflight/READY").is_file()


def test_comrecgc_preflight_requires_explicit_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("model.pt", "temperature_scaling.json", "feature_schema.json"):
        (checkpoint / name).write_text(name, encoding="utf-8")
    card = {
        "dataset": "bace",
        "backbone": "gine",
        "oracle_backend": "gnn",
        "rf_oracle_used": False,
        "num_classes": 2,
        "source_label": 1,
        "checkpoint_id": CHECKPOINT_ID,
    }
    monkeypatch.setattr(
        "src.baselines.bace_gnn_baseline_contracts._checkpoint_contract",
        lambda _path: (card, object()),
    )

    with pytest.raises(ValueError, match="explicit official_root"):
        write_route_preflight(
            method="ComRecGC",
            checkpoint_dir=checkpoint,
            output_dir=tmp_path / "preflight",
        )
    assert not (tmp_path / "preflight").exists()


def test_globalgce_controller_fragment_runs_bridge_then_full_native_route(
    tmp_path: Path,
) -> None:
    fragment = _fragment(tmp_path, "GlobalGCE")
    tasks = {row["task_id"]: row for row in fragment["tasks"]}
    preflight = tasks["bace_globalgce_preflight"]
    assert preflight["task_id"] == "bace_globalgce_preflight"
    assert preflight["resource"] == {"kind": "cpu", "gpus": 0}
    assert preflight["required_markers"] == ["NATIVE_ACTION_READY", "READY"]
    assert "--official-root" in preflight["argv"]
    bridge = tasks["bace_globalgce_bridge_smoke"]
    assert bridge["resource"] == {"kind": "gpu", "gpus": 1}
    assert bridge["dependencies"] == ["bace_globalgce_preflight"]
    generation = tasks["bace_globalgce_train_candidates"]
    assert generation["dependencies"] == ["bace_globalgce_bridge_smoke"]
    assert "--source-manifest" in generation["argv"]
    assert generation["argv"][generation["argv"].index("--min-freq") + 1] == "7"
    assert fragment["static_terminal"] is None


@pytest.mark.parametrize("method", ["GCFExplainer", "ComRecGC"])
def test_ready_controller_fragments_have_exact_dependencies_and_markers(
    tmp_path: Path, method: str
) -> None:
    fragment = _fragment(tmp_path, method)
    tasks = {row["task_id"]: row for row in fragment["tasks"]}
    prefix = "bace_gcfexplainer" if method == "GCFExplainer" else "bace_comrecgc"
    assert fragment["static_terminal"] is None
    assert tasks[f"{prefix}_preflight"]["required_markers"] == ["READY"]
    assert tasks[f"{prefix}_preflight"]["resource"]["kind"] == "cpu"
    assert tasks[f"{prefix}_preflight"]["env"]["PYTHONHASHSEED"] == "0"
    for shard in range(4):
        calibration = tasks[f"{prefix}_calibration_shard_{shard}"]
        assert calibration["resource"] == {"kind": "gpu", "gpus": 1}
        assert calibration["controller_injected_env"] == ["CUDA_VISIBLE_DEVICES"]
        assert calibration["dependencies"] == [f"{prefix}_train_candidates"]
        assert calibration["required_markers"] == ["PASS"]
        test_task = tasks[f"{prefix}_test_shard_{shard}"]
        assert test_task["dependencies"] == [f"{prefix}_selection"]
    final = tasks[f"{prefix}_final_freeze"]
    assert final["required_markers"] == ["PASS", "FINAL_PASS.json"]
    assert final["dependencies"] == [f"{prefix}_selection", f"{prefix}_test_merge"]
    if method == "GCFExplainer":
        selection = tasks["bace_gcfexplainer_selection"]
        threshold_path = str(
            (tmp_path / "bace/ours/b12-selector/attempt-0/thresholds.json").resolve()
        )
        flag = selection["argv"].index("--thresholds-json")
        assert selection["argv"][flag + 1] == threshold_path
        assert threshold_path in selection["inputs"]
    if method == "ComRecGC":
        preflight = tasks["bace_comrecgc_preflight"]
        official = str((tmp_path / "official").resolve())
        official_index = preflight["argv"].index("--official-root")
        assert preflight["argv"][official_index + 1] == official
        assert official in preflight["inputs"]
        generation = tasks["bace_comrecgc_train_generation"]
        assert generation["resume_argv"][-1] == "--resume"
        assert generation["retry_policy"] == (
            "resume_same_root_from_verified_checkpoint"
        )


def test_native_adapter_maps_project_target_to_official_counterfactual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(hidden_dim=2)
            self.classifier = torch.nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.classifier.weight.copy_(
                    torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
                )

        def encode_graph(self, batch: object) -> object:
            return torch.ones((batch.size, 2), dtype=self.anchor.dtype)

    class FakeBatch:
        def __init__(self, size: int) -> None:
            self.size = size

        def to(self, _device: object) -> "FakeBatch":
            return self

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.pt").write_bytes(b"model")
    metadata = {
        "model_card": {
            "dataset": "bace",
            "backbone": "gine",
            "num_classes": 2,
            "source_label": 1,
            "rf_oracle_used": False,
        },
        "checkpoint_id": CHECKPOINT_ID,
        "temperature_scaling": {"temperature": 1.0},
        "feature_schema": default_molecular_feature_schema(),
    }
    monkeypatch.setattr(
        "src.baselines.bace_gine_native_adapter.load_gnn_checkpoint_bundle",
        lambda _root, device: (FakeModel(), metadata),
    )
    monkeypatch.setattr(
        "src.baselines.bace_gine_native_adapter.collate_molecular_graphs",
        lambda rows, edge_feature_dim: FakeBatch(len(rows)),
    )
    adapter = BACEFrozenGINENativeGraphAdapter(
        checkpoint,
        source_records=[{"molecule_id": "p0"}],
        graph_schema=object(),
        device="cpu",
    )
    adapter._decode = MethodType(
        lambda self, graph: (
            ("C", None) if graph.valid else (None, "synthetic_invalid")
        ),
        adapter,
    )
    adapter._portable_graph = MethodType(
        lambda self, smiles, row_index: object(),
        adapter,
    )
    _node, _hidden, log_probs = adapter(
        [SimpleNamespace(valid=True, num_nodes=1), SimpleNamespace(valid=False, num_nodes=1)]
    )
    assert torch.argmax(log_probs, dim=-1).tolist() == [1, 0]
    assert adapter.provenance()["rf_oracle_used"] is False
    assert adapter.provenance()["decode_failure_count"] == 1


def test_clean_manifest_gate_rejects_rf_and_split_leakage() -> None:
    clean = {
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "oracle_checkpoint_hash": CHECKPOINT_ID,
        "calibration_loaded": False,
        "test_loaded": False,
        "available_model_counterfactual_count": 24_140,
        "counterfactuals_path": "/immutable/train/counterfactuals.pt",
        "counterfactuals_sha256": "a" * 64,
    }
    assert_gine_clean_manifest(
        clean, checkpoint_id=CHECKPOINT_ID, require_train_only=True
    )
    with pytest.raises(ValueError, match="forbidden RF"):
        assert_gine_clean_manifest(
            {**clean, "historical_initializer": "rf_model.pkl"},
            checkpoint_id=CHECKPOINT_ID,
            require_train_only=True,
        )
    with pytest.raises(ValueError, match="forbidden RF"):
        assert_gine_clean_manifest(
            {**clean, "teacher_backend": "rf"},
            checkpoint_id=CHECKPOINT_ID,
            require_train_only=True,
        )
    with pytest.raises(ValueError, match="forbidden RF"):
        assert_gine_clean_manifest(
            {**clean, "historical_rf_checkpoint": "/immutable/model.pt"},
            checkpoint_id=CHECKPOINT_ID,
            require_train_only=True,
        )
    with pytest.raises(ValueError, match="test_loaded_not_false"):
        assert_gine_clean_manifest(
            {**clean, "test_loaded": True},
            checkpoint_id=CHECKPOINT_ID,
            require_train_only=True,
        )


def test_test_split_is_unavailable_before_calibration_freeze(tmp_path: Path) -> None:
    test_path = tmp_path / "bace_test.csv"
    test_path.write_text("smiles,label\nCC,1\n", encoding="utf-8")
    selection = tmp_path / "frozen_selection_manifest.json"
    atomic_json(
        selection,
        {
            "stage": SELECTION_STAGE,
            "selection_frozen": False,
            "test_loaded": False,
        },
    )
    with pytest.raises(ValueError, match="incomplete selection freeze"):
        _authorize_split(
            stage=TEST_STAGE,
            split_path=test_path,
            selection_manifest=selection,
        )
    atomic_json(
        selection,
        {
            "stage": SELECTION_STAGE,
            "selection_frozen": True,
            "test_loaded": False,
        },
    )
    assert _authorize_split(
        stage=TEST_STAGE,
        split_path=test_path,
        selection_manifest=selection,
    ) == test_path.resolve()


def test_candidate_loader_preserves_fullgraph_action_and_split_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "generation"
    root.mkdir()
    rows = [_candidate("gcfexplainer", index) for index in range(20)]
    atomic_jsonl(root / "candidate_universe.jsonl", rows)
    atomic_json(
        root / "run_manifest.json",
        {
            "stage": "TRAIN_CANDIDATE_GENERATION",
            "status": "PASS",
            "run_complete": True,
            "method_id": "gcfexplainer",
            "oracle_backend": "gnn",
            "classifier_family": "gine",
            "rf_oracle_used": False,
            "oracle_checkpoint_hash": CHECKPOINT_ID,
            "calibration_loaded": False,
            "test_loaded": False,
        },
    )
    loaded, manifest, _path = _load_candidates(
        method="gcfexplainer",
        stage=CALIBRATION_STAGE,
        predecessor_root=root,
        checkpoint_id=CHECKPOINT_ID,
    )
    assert len(loaded) == 20
    assert manifest["test_loaded"] is False
    assert {row["action_kind"] for row in loaded} == {
        "full_counterfactual_graph"
    }


def test_resource_capped_comrecgc_accepts_ten_rules_without_weakening_gcf(
    tmp_path: Path,
) -> None:
    root = tmp_path / "generation"
    root.mkdir()
    rows = [_candidate("comrecgc", index) for index in range(10)]
    atomic_jsonl(root / "candidate_universe.jsonl", rows)
    manifest = {
        "stage": "TRAIN_CANDIDATE_GENERATION",
        "status": "PASS",
        "run_complete": True,
        "method_id": "comrecgc",
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "oracle_checkpoint_hash": CHECKPOINT_ID,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    atomic_json(root / "run_manifest.json", manifest)
    loaded, _manifest, _path = _load_candidates(
        method="comrecgc",
        stage=CALIBRATION_STAGE,
        predecessor_root=root,
        checkpoint_id=CHECKPOINT_ID,
    )
    assert len(loaded) == 10
    manifest["method_id"] = "gcfexplainer"
    atomic_json(root / "run_manifest.json", manifest)
    atomic_jsonl(
        root / "candidate_universe.jsonl",
        [_candidate("gcfexplainer", index) for index in range(10)],
    )
    with pytest.raises(ValueError, match="at least 20"):
        _load_candidates(
            method="gcfexplainer",
            stage=CALIBRATION_STAGE,
            predecessor_root=root,
            checkpoint_id=CHECKPOINT_ID,
        )


def test_resource_capped_comrecgc_selector_freezes_effective_k_and_plateau(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    atomic_json(
        matrix / "run_manifest.json",
        {
            "stage": CALIBRATION_STAGE,
            "status": "PASS",
            "test_loaded": False,
            "method_id": "comrecgc",
            "oracle_checkpoint_hash": CHECKPOINT_ID,
            "molclr_checkpoint_hash": "b" * 64,
            "candidate_universe_hash": "c" * 64,
        },
    )
    candidates = [_candidate("comrecgc", index) for index in range(10)]
    atomic_jsonl(
        matrix / "pair_matrix.jsonl",
        [
            {"parent_id": "p0", "candidate_id": row["candidate_id"]}
            for row in candidates
        ],
    )

    def fake_selector(*, output_dir: Path, top_k: int, table_k: int, **_kwargs: object) -> None:
        assert top_k == 10
        assert table_k == 10
        output = Path(output_dir)
        (output / "variants/effective").mkdir(parents=True)
        atomic_json(output / "calibration_decision.json", {"selected_variant": "effective"})
        atomic_json(
            output / "variants/effective/selected_top20.json",
            {"candidates": candidates},
        )
        atomic_json(output / "thresholds.json", {"theta_star": 0.5})

    monkeypatch.setattr(
        "src.eval.bace_native_baseline_gnn.run_mutagenicity_wnode_selector",
        fake_selector,
    )
    output = tmp_path / "selection"
    frozen = run_native_baseline_selector(
        method="comrecgc", matrix_output=matrix, output_dir=output
    )
    assert frozen["K"] == 10
    assert frozen["K_MAX"] == 20
    assert frozen["prefixes"]["10"] == frozen["prefixes"]["20"]
    assert len(frozen["prefixes"]["20"]) == 10


def test_gcf_selector_adopts_exact_ours_b12_thresholds_without_refit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix = tmp_path / "gcf-matrix"
    matrix.mkdir()
    candidates = [_candidate("gcfexplainer", index) for index in range(20)]
    atomic_jsonl(
        matrix / "pair_matrix.jsonl",
        [
            {"parent_id": "p0", "candidate_id": row["candidate_id"]}
            for row in candidates
        ],
    )
    atomic_json(
        matrix / "run_manifest.json",
        {
            "stage": CALIBRATION_STAGE,
            "status": "PASS",
            "test_loaded": False,
            "method_id": "gcfexplainer",
            "oracle_checkpoint_hash": CHECKPOINT_ID,
            "molclr_checkpoint_hash": MOLCLR_ID,
            "candidate_universe_hash": "c" * 64,
        },
    )
    thresholds_path = _write_ours_b12_threshold_source(tmp_path)
    observed: dict[str, object] = {}

    def fake_selector(
        *,
        output_dir: Path,
        frozen_thresholds: ThresholdBundle,
        frozen_threshold_provenance: dict[str, object],
        **_kwargs: object,
    ) -> None:
        observed["thresholds"] = frozen_thresholds.to_dict()
        observed["provenance"] = frozen_threshold_provenance
        output = Path(output_dir)
        (output / "variants/effective").mkdir(parents=True)
        atomic_json(
            output / "calibration_decision.json",
            {"selected_variant": "effective"},
        )
        atomic_json(
            output / "variants/effective/selected_top20.json",
            {"candidates": candidates},
        )
        atomic_json(output / "thresholds.json", frozen_thresholds.to_dict())

    monkeypatch.setattr(
        "src.eval.bace_native_baseline_gnn.run_mutagenicity_wnode_selector",
        fake_selector,
    )
    frozen = run_native_baseline_selector(
        method="GCFExplainer",
        matrix_output=matrix,
        output_dir=tmp_path / "gcf-selection",
        thresholds_json=thresholds_path,
    )
    assert observed["thresholds"] == _ours_threshold_payload()
    provenance = observed["provenance"]
    assert isinstance(provenance, dict)
    assert provenance["source_method"] == "Ours"
    assert provenance["test_used"] is False
    assert frozen["thresholds"] == _ours_threshold_payload()
    assert (
        frozen["threshold_config_hash"]
        == BACE_OURS_B12_THRESHOLD_CONFIG_SHA256
    )
    assert frozen["ordered_rule_ids"] == [
        row["candidate_id"] for row in candidates
    ]


def test_gcf_selector_requires_explicit_ours_b12_thresholds(tmp_path: Path) -> None:
    matrix = tmp_path / "gcf-matrix"
    matrix.mkdir()
    atomic_json(
        matrix / "run_manifest.json",
        {
            "stage": CALIBRATION_STAGE,
            "status": "PASS",
            "test_loaded": False,
            "method_id": "gcfexplainer",
            "oracle_checkpoint_hash": CHECKPOINT_ID,
            "molclr_checkpoint_hash": MOLCLR_ID,
            "candidate_universe_hash": "c" * 64,
        },
    )
    atomic_jsonl(
        matrix / "pair_matrix.jsonl",
        [
            {"parent_id": "p0", "candidate_id": f"candidate-{index:02d}"}
            for index in range(20)
        ],
    )
    with pytest.raises(ValueError, match="explicit --thresholds-json"):
        run_native_baseline_selector(
            method="GCFExplainer",
            matrix_output=matrix,
            output_dir=tmp_path / "selection",
        )


def test_fullgraph_wnode_uses_pair_distance_without_fake_match_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PairOnlyProvider:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def distance(self, left: str, right: str) -> dict[str, object]:
            self.calls.append((left, right))
            return {"ok": True, "distance": 0.25, "error": None}

        def distance_for_action(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("full-graph candidates have no match action context")

    class Oracle:
        @staticmethod
        def predict_records(_graphs: object, *, batch_size: int) -> list[dict[str, object]]:
            assert batch_size == 8
            return [{"predicted_label": 0, "probabilities": [0.9, 0.1]}]

    monkeypatch.setattr(
        "src.eval.bace_native_baseline_gnn._graph",
        lambda *_args, **_kwargs: object(),
    )
    provider = PairOnlyProvider()
    rows = _fullgraph_pair_rows(
        parents=[SimpleNamespace(parent_id="p0", smiles="CC")],
        before_rows=[{"predicted_label": 1, "probabilities": [0.1, 0.9]}],
        candidates=[
            {
                "candidate_id": "c0",
                "canonical_smiles": "CO",
                "rank": 1,
                "native_rank": 1,
            }
        ],
        featurizer=object(),
        oracle=Oracle(),
        provider=provider,
        card={"checkpoint_id": CHECKPOINT_ID},
        spec=baseline_spec("gcfexplainer"),
        method_id="gcfexplainer",
        oracle_batch_size=8,
    )

    assert provider.calls == [("CC", "CO")]
    assert rows[0]["pair_strict_flip"] is True
    assert rows[0]["wnode_distance"] == 0.25


def test_final_freeze_uses_only_frozen_order_and_test_matrix(tmp_path: Path) -> None:
    selection = tmp_path / "selection"
    test = tmp_path / "test_merge"
    selection.mkdir()
    test.mkdir()
    ids = [f"candidate-{index:02d}" for index in range(20)]
    frozen = {
        "stage": SELECTION_STAGE,
        "status": "FROZEN",
        "method_id": "comrecgc",
        "selection_frozen": True,
        "test_loaded": False,
        "oracle_checkpoint_hash": CHECKPOINT_ID,
        "molclr_checkpoint_hash": "b" * 64,
        "ordered_rule_ids": ids,
        "thresholds": {"theta_star": 0.5},
    }
    atomic_json(selection / "frozen_selection_manifest.json", frozen)
    test_manifest = {
        "stage": TEST_STAGE,
        "status": "PASS",
        "method_id": "comrecgc",
        "selection_frozen_before_test": True,
        "test_loaded": True,
        "oracle_checkpoint_hash": CHECKPOINT_ID,
        "molclr_checkpoint_hash": "b" * 64,
    }
    atomic_json(test / "run_manifest.json", test_manifest)
    pair_rows = []
    for parent in ("p0", "p1"):
        for index, candidate in enumerate(ids):
            pair_rows.append(
                {
                    "parent_id": parent,
                    "candidate_id": candidate,
                    "pair_strict_flip": index % 2 == 0,
                    "wnode_distance": 0.25 if index % 2 == 0 else None,
                }
            )
    atomic_jsonl(test / "pair_matrix.jsonl", pair_rows)
    result = freeze_native_baseline_final(
        method="comrecgc",
        selection_output=selection,
        test_output=test,
        output_dir=tmp_path / "final",
    )
    assert result["status"] == "PASS"
    assert result["action_kind"] == "native_common_recourse_fullgraph"
    assert result["rf_oracle_used"] is False
    assert (tmp_path / "final/PASS").is_file()
    metrics = json.loads((tmp_path / "final/final_metrics.json").read_text())
    assert metrics["prefix_metrics"][-1]["CCRCov"] == 1.0


def test_resource_capped_comrecgc_final_metrics_plateau_after_ten_rules(
    tmp_path: Path,
) -> None:
    selection = tmp_path / "selection"
    test = tmp_path / "test_merge"
    selection.mkdir()
    test.mkdir()
    ids = [f"candidate-{index:02d}" for index in range(10)]
    atomic_json(
        selection / "frozen_selection_manifest.json",
        {
            "stage": SELECTION_STAGE,
            "status": "FROZEN",
            "method_id": "comrecgc",
            "selection_frozen": True,
            "test_loaded": False,
            "oracle_checkpoint_hash": CHECKPOINT_ID,
            "molclr_checkpoint_hash": "b" * 64,
            "ordered_rule_ids": ids,
            "effective_rule_count": 10,
            "K_MAX": 20,
            "thresholds": {"theta_star": 0.5},
        },
    )
    atomic_json(
        test / "run_manifest.json",
        {
            "stage": TEST_STAGE,
            "status": "PASS",
            "method_id": "comrecgc",
            "selection_frozen_before_test": True,
            "test_loaded": True,
            "oracle_checkpoint_hash": CHECKPOINT_ID,
            "molclr_checkpoint_hash": "b" * 64,
        },
    )
    atomic_jsonl(
        test / "pair_matrix.jsonl",
        [
            {
                "parent_id": parent,
                "candidate_id": candidate,
                "pair_strict_flip": index % 2 == 0,
                "wnode_distance": 0.25 if index % 2 == 0 else None,
            }
            for parent in ("p0", "p1")
            for index, candidate in enumerate(ids)
        ],
    )
    result = freeze_native_baseline_final(
        method="comrecgc",
        selection_output=selection,
        test_output=test,
        output_dir=tmp_path / "final",
    )
    assert result["effective_rule_count"] == 10
    metrics = json.loads((tmp_path / "final/final_metrics.json").read_text())
    assert len(metrics["prefix_metrics"]) == 20
    assert metrics["prefix_metrics"][9] == {
        **metrics["prefix_metrics"][19],
        "K": 10,
    }
