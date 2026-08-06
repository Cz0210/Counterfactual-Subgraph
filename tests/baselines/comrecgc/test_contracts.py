from __future__ import annotations

import json
import pickle
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path

import pytest

from src.baselines.comrecgc.audit import validate_final_manifest, validate_monotonic
from src.baselines.comrecgc.contracts import (
    ADAPTATION_MODE,
    CF_MODE,
    DISTANCE_LINE,
    GenerationParameters,
    RecourseParameters,
    ContractError,
    ordered_ids_sha256,
    write_json,
)
from src.baselines.comrecgc.project_dataset import project_label_to_internal
from src.baselines.comrecgc.preregistration import validate_chemistry_trace_evidence
from src.baselines.comrecgc.runtime import (
    _EndpointSafeGraphMap,
    _materialize_dataset_indices,
    patched_official_runtime,
    validate_counterfactual_payload,
)
from src.baselines.comrecgc import upstream


def test_generation_profiles_are_frozen() -> None:
    smoke = GenerationParameters.for_mode("smoke")
    assert smoke.steps == 100
    assert smoke.sample_size == 128
    smoke.validate("smoke")
    GenerationParameters.for_mode("full").validate("full")
    invalid = GenerationParameters.for_mode("smoke")
    with pytest.raises(ContractError):
        invalid.validate("full")


def test_common_recourse_profiles_are_frozen() -> None:
    assert RecourseParameters.for_mode("smoke").recourse_size == 5
    assert RecourseParameters.for_mode("full").cf_size == 100_000
    RecourseParameters.for_mode("full").validate("full")


def test_order_hash_is_order_sensitive() -> None:
    assert ordered_ids_sha256(["a", "b"]) != ordered_ids_sha256(["b", "a"])


def test_atomic_json_write(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    write_json(path, {"value": 3})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 3}
    assert not list(tmp_path.glob("*.tmp"))


def test_mutagenicity_chemistry_requires_true_trace_parity(tmp_path: Path) -> None:
    evidence = tmp_path / "trace_summary.json"
    write_json(
        evidence,
        {
            "trace_only": True,
            "rng_calls_added": 0,
            "candidate_count": 2,
            "candidate_lineage_resolved_count": 2,
        },
    )
    write_json(tmp_path / "_TRACE_COMPLETE.json", {"trace_complete": True})

    with pytest.raises(ValueError, match="trace parity"):
        validate_chemistry_trace_evidence(evidence, dataset="mutagenicity")


def test_aids_chemistry_accepts_complete_streamed_trace_without_claiming_parity(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "trace_summary.json"
    write_json(
        evidence,
        {
            "trace_only": True,
            "rng_calls_added": 0,
            "candidate_count": 2,
            "candidate_lineage_resolved_count": 2,
        },
    )
    write_json(tmp_path / "_TRACE_COMPLETE.json", {"trace_complete": True})

    result = validate_chemistry_trace_evidence(evidence, dataset="aids")

    assert result["trace_integrity_passed"] is True
    assert result["trace_parity_required"] is False
    assert result["trace_parity_passed"] is False


def test_monotonicity_gate() -> None:
    validate_monotonic([0.0, 0.2, 0.2, 1.0], field="coverage")
    with pytest.raises(ContractError):
        validate_monotonic([0.0, 0.3, 0.2], field="coverage")


def test_final_semantic_gate() -> None:
    validate_final_manifest(
        {
            "method": "COMRECGC",
            "cf_mode": CF_MODE,
            "distance_line": DISTANCE_LINE,
            "adaptation_mode": ADAPTATION_MODE,
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
            "calibration_loaded": False,
            "test_used_for_selection": False,
            "threshold_fitted_on_test": False,
        }
    )


def test_project_label_mapping_is_explicit() -> None:
    assert project_label_to_internal(1) == 0
    assert project_label_to_internal(0) == 1
    with pytest.raises(ContractError):
        project_label_to_internal(2)


def test_upstream_payload_contract() -> None:
    graph_map, candidates = validate_counterfactual_payload(
        {"graph_map": {"hash": [object()]}, "counterfactual_candidates": [{"graph_hash": "hash"}]}
    )
    assert list(graph_map) == ["hash"]
    assert candidates[0]["graph_hash"] == "hash"
    with pytest.raises(RuntimeError):
        validate_counterfactual_payload({"graph_map": {}, "counterfactual_candidates": []})


def test_native_source_rows_are_eagerly_materialized() -> None:
    class LazyRows:
        def __init__(self) -> None:
            self.open = True

        def __getitem__(self, index: int) -> str:
            if not self.open:
                raise FileNotFoundError("relative processed path unavailable")
            return f"graph-{index}"

    rows = LazyRows()
    materialized = _materialize_dataset_indices(rows, [3, 1])
    rows.open = False

    assert materialized == ["graph-3", "graph-1"]


def test_native_runtime_freezes_feature_dimension_before_cwd_switch() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "src/baselines/comrecgc/runtime.py"
    ).read_text(encoding="utf-8")
    feature_line = source.index("num_features = int(graphs.num_features)")
    cwd_line = source.index("os.chdir(runtime_root)", feature_line)
    dataset_line = source.index("GraphListDataset(sources, num_features)", cwd_line)
    assert feature_line < cwd_line < dataset_line


def test_native_aids_gnn_does_not_reopen_the_trusted_cache(
    tmp_path: Path, monkeypatch
) -> None:
    import types

    import src.baselines.comrecgc.runtime as runtime

    checkpoint = tmp_path / "data/aids/gnn/model_best.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    observed: dict[str, object] = {}

    class Model:
        def __init__(self, **kwargs) -> None:
            observed["kwargs"] = kwargs

        def to(self, device: str):
            observed["device"] = device
            return self

        def load_state_dict(self, state_dict) -> None:
            observed["state_dict"] = state_dict

        def eval(self):
            observed["eval"] = True
            return self

    fake_torch = types.SimpleNamespace(load=lambda *args, **kwargs: {"weight": 1})
    monkeypatch.setattr(runtime, "_torch_stack", lambda: (fake_torch, object()))
    module = types.SimpleNamespace(GNN=Model)

    result = runtime._load_native_aids_gnn_from_trusted_features(
        gnn_module=module,
        upstream_root=tmp_path,
        num_features=9,
        device="cuda:0",
    )

    assert isinstance(result, Model)
    assert observed["kwargs"] == {
        "num_features": 9,
        "num_classes": 2,
        "num_layers": 3,
        "dim": 20,
        "dropout": 0.0,
    }
    assert observed["device"] == "cuda:0"
    assert observed["state_dict"] == {"weight": 1}
    assert observed["eval"] is True


def test_native_trusted_payload_is_resolved_before_upstream_chdir() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "src/baselines/comrecgc/runtime.py"
    ).read_text(encoding="utf-8")
    resolve_line = source.index("trusted_payload_path.resolve(strict=True)")
    cwd_line = source.index("os.chdir(Path(upstream_root)", resolve_line)
    load_line = source.index("load_aids_tensor_payload(\n                        trusted_payload_path", cwd_line)
    assert resolve_line < cwd_line < load_line


def test_endpoint_safe_graph_map_preserves_normal_deletion_and_serializes_plain_dict() -> None:
    module = type("Module", (), {})()
    module.counterfactual_candidates = [{"graph_hash": "tail"}]
    module.graph_index_map = {}
    graph_map = _EndpointSafeGraphMap(module, {"tail": [1], "keep": [2]})

    del graph_map["tail"]

    assert graph_map == {"keep": [2]}
    assert graph_map.missing_unmaterialized_eviction_count == 0
    restored = pickle.loads(pickle.dumps(graph_map))
    assert type(restored) is dict
    assert restored == {"keep": [2]}


def test_endpoint_safe_graph_map_only_allows_unmaterialized_tail_eviction() -> None:
    module = type("Module", (), {})()
    module.counterfactual_candidates = [{"graph_hash": "unmaterialized"}]
    module.graph_index_map = {}
    graph_map = _EndpointSafeGraphMap(module, {"keep": [2]})

    del graph_map["unmaterialized"]

    assert graph_map == {"keep": [2]}
    assert graph_map.missing_unmaterialized_eviction_count == 1
    with pytest.raises(KeyError):
        del graph_map["different"]
    module.graph_index_map["unmaterialized"] = 0
    with pytest.raises(KeyError):
        del graph_map["unmaterialized"]


@pytest.mark.parametrize("raise_inside", [False, True])
def test_endpoint_safe_runtime_restores_plain_map_and_official_functions(
    monkeypatch, raise_inside: bool
) -> None:
    import src.baselines.comrecgc.runtime as runtime

    original_call = object()
    original_neighbor = lambda graph, action: graph
    original_move = object()
    module = SimpleNamespace(
        call=original_call,
        neighbor_graph_access=original_neighbor,
        move_to_next_graph=original_move,
        graph_map={"keep": [1]},
        graph_index_map={},
        counterfactual_candidates=[{"graph_hash": "unmaterialized"}],
    )
    patched_call = object()
    monkeypatch.setattr(runtime, "_safe_call_factory", lambda **_kwargs: patched_call)
    audit: dict[str, object] = {}

    def exercise() -> None:
        with patched_official_runtime(
            module,
            model=object(),
            embedding_model=object(),
            gnn_device="cpu",
            embedding_device="cpu",
            batch_size=1,
            compatibility_audit=audit,
        ):
            assert module.call is patched_call
            del module.graph_map["unmaterialized"]
            if raise_inside:
                raise RuntimeError("expected")

    if raise_inside:
        with pytest.raises(RuntimeError, match="expected"):
            exercise()
    else:
        exercise()

    assert type(module.graph_map) is dict
    assert module.graph_map == {"keep": [1]}
    assert module.call is original_call
    assert module.neighbor_graph_access is original_neighbor
    assert module.move_to_next_graph is original_move
    assert audit == {
        "patch": "candidate_map_unmaterialized_eviction_none_safe_v1",
        "missing_unmaterialized_eviction_count": 1,
        "rng_calls_added": 0,
        "candidate_order_changed": False,
    }


def test_upstream_import_does_not_write_bytecode(tmp_path: Path, monkeypatch) -> None:
    observed: list[bool] = []
    original = sys.dont_write_bytecode
    monkeypatch.setattr(upstream, "validate_upstream_checkout", lambda path: tmp_path)

    def fake_import(name: str) -> ModuleType:
        observed.append(sys.dont_write_bytecode)
        return ModuleType(name)

    monkeypatch.setattr(upstream.importlib, "import_module", fake_import)
    with upstream.imported_upstream(tmp_path) as modules:
        assert set(modules) == set(upstream.UPSTREAM_MODULES)
        assert sys.dont_write_bytecode is True

    assert observed == [True] * len(upstream.UPSTREAM_MODULES)
    assert sys.dont_write_bytecode is original


def test_upstream_import_restores_bytecode_flag_after_error(
    tmp_path: Path, monkeypatch
) -> None:
    original = sys.dont_write_bytecode
    monkeypatch.setattr(upstream, "validate_upstream_checkout", lambda path: tmp_path)
    monkeypatch.setattr(
        upstream.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(RuntimeError("import failed")),
    )

    with pytest.raises(RuntimeError, match="import failed"):
        with upstream.imported_upstream(tmp_path):
            pass

    assert sys.dont_write_bytecode is original
