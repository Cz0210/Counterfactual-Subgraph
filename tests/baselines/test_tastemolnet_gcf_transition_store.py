from __future__ import annotations

from pathlib import Path
import uuid

import numpy as np
import pytest

from src.baselines import tastemolnet_gcf_full_resume as resume
from src.baselines import tastemolnet_gcf_full as full
from src.baselines import tastemolnet_gcf_full_verify as full_verify
from src.baselines import tastemolnet_gcf_candidate_store as candidates
from src.baselines import tastemolnet_gcf_production_state as production
from src.baselines import tastemolnet_gcf_transition_store as transition


GRAPH_A = "a" * 64
GRAPH_B = "b" * 64
GRAPH_C = "c" * 64
REPOSITORY = Path(__file__).resolve().parents[2]


def _value(torch, targets):
    actions = [("NOTHING", None, None)] + [
        ("NLC", index, index + 1) for index in range(len(targets) - 1)
    ]
    importance = np.asarray(
        [[0.75 + index / 100.0, 0.5] for index in range(len(targets))],
        dtype=np.float64,
    )
    coverage = torch.tensor(
        [[1, 0, index % 2, 1] for index in range(len(targets))],
        dtype=torch.float32,
    ).to_sparse()
    return targets, actions, importance, coverage


def _store(tmp_path: Path, *, snapshot=None, open_writer=True):
    return transition.T12ExternalTransitionStore(
        root=(tmp_path / "transitions").resolve(),
        parent_count=4,
        sample_size=10_000,
        candidate_capacity=100_000,
        contract_sha256="d" * 64,
        attempt_id="attempt-" + str(uuid.UUID(int=1)),
        generation_token="e" * 64,
        expanded_capacity=1,
        max_store_bytes=1024**3,
        resume_snapshot=snapshot,
        open_writer=open_writer,
    )


def _assert_value_equal(torch, observed, expected):
    assert observed[0] == expected[0]
    assert observed[1] == expected[1]
    assert np.array_equal(np.asarray(observed[2]), np.asarray(expected[2]))
    assert observed[3].dtype == expected[3].dtype
    assert torch.equal(observed[3].to_dense(), expected[3].to_dense())


def test_external_transition_store_reopens_exact_prefix_and_bounds_lru(tmp_path):
    torch = pytest.importorskip("torch")
    first = _store(tmp_path)
    value_a = _value(torch, [GRAPH_A, GRAPH_B, GRAPH_C])
    value_b = _value(torch, [GRAPH_B, GRAPH_A])
    first[GRAPH_A] = value_a
    first[GRAPH_B] = value_b
    assert len(first) == 2
    assert first.audit()["expanded_entry_count"] == 1
    _assert_value_equal(torch, first[GRAPH_A], value_a)
    assert first.audit()["expanded_entry_count"] == 1
    snapshot_10k = first.export_checkpoint_state()
    assert snapshot_10k["checkpoint_export_contains_coverage_payload"] is False
    assert snapshot_10k["external_journal_is_authority"] is True
    assert snapshot_10k["committed_store_bytes"] < 128 * 1024
    first.close()

    second = _store(tmp_path, snapshot=snapshot_10k)
    assert second.export_checkpoint_state() == snapshot_10k
    _assert_value_equal(torch, second[GRAPH_A], value_a)
    del second[GRAPH_B]
    second[GRAPH_C] = _value(torch, [GRAPH_C, GRAPH_A])
    snapshot_20k = second.export_checkpoint_state()
    assert len(snapshot_20k["segments"]) == 2
    assert snapshot_20k["active_sources"] == [GRAPH_A, GRAPH_C]
    second.close()

    verifier = _store(
        tmp_path, snapshot=snapshot_20k, open_writer=False
    )
    assert verifier.export_checkpoint_state() == snapshot_20k
    assert list(verifier) == [GRAPH_A, GRAPH_C]
    verifier.close()
    assert (
        transition.T12ExternalTransitionStore.verify_checkpoint_state(
            snapshot_20k
        )
        == snapshot_20k
    )


def test_external_transition_store_reopens_an_exact_empty_prefix(tmp_path):
    first = _store(tmp_path)
    snapshot = first.export_checkpoint_state()
    assert snapshot["segments"] == []
    assert snapshot["active_sources"] == []
    first.close()
    reopened = _store(tmp_path, snapshot=snapshot, open_writer=False)
    assert reopened.export_checkpoint_state() == snapshot
    reopened.close()


def test_generation_verifier_independently_reopens_compact_history(tmp_path):
    attempt = str(uuid.uuid4())
    bounds = production.T12ProductionBounds.pinned(parent_count=2)
    history = production.T12CompactHistoryJournal(
        root=(tmp_path / "history").resolve(),
        index_root=(tmp_path / "writer-index").resolve(),
        bounds=bounds,
        contract_sha256="7" * 64,
        attempt_id=attempt,
        generation_token="8" * 64,
    )
    history.append_observation(
        graph_identity_sha256=GRAPH_A,
        probabilities=(0.1, 0.2, 0.7),
        prediction=2,
        candidate=True,
        valid_fullgraph=True,
        coverage_vector=(1, 0),
        embedding_sha256="9" * 64,
        failure_reason="",
        lineage_sha256="6" * 64,
        neurosed_query_sha256="5" * 64,
    )
    snapshot = history.checkpoint_state()
    history.close()
    verification = tmp_path / "verification"
    verification.mkdir()
    fact = full_verify._verify_history(
        snapshot=snapshot, output_root=verification.resolve(), cursor=10_000
    )
    assert fact["observation_count"] == 1
    assert fact["first_seen_graph_count"] == 1
    assert fact["chain_head"] == snapshot["chain_head"]


def test_external_transition_checkpoint_rejects_committed_payload_tamper(tmp_path):
    torch = pytest.importorskip("torch")
    writer = _store(tmp_path)
    writer[GRAPH_A] = _value(torch, [GRAPH_A, GRAPH_B])
    snapshot = writer.export_checkpoint_state()
    writer.close()
    segment = tmp_path / "transitions" / snapshot["segments"][0]["segment_file"]
    payload = bytearray(segment.read_bytes())
    payload[-33] ^= 1
    segment.write_bytes(payload)
    with pytest.raises(
        transition.TasteT12TransitionStoreError, match="hash chain"
    ):
        transition.T12ExternalTransitionStore.verify_checkpoint_state(snapshot)


def test_external_transition_store_rejects_nonbinary_coverage(tmp_path):
    torch = pytest.importorskip("torch")
    writer = _store(tmp_path)
    value = list(_value(torch, [GRAPH_A]))
    value[3] = torch.tensor([[0.5, 0.0, 1.0, 0.0]], dtype=torch.float32)
    with pytest.raises(
        transition.TasteT12TransitionStoreError, match="exactly binary"
    ):
        writer[GRAPH_A] = tuple(value)
    writer.close()


def test_production_resource_gate_accepts_only_external_exact_store(tmp_path):
    pytest.importorskip("torch")
    bounds = production.T12ProductionBounds.pinned(parent_count=4)
    raw = resume.production_transition_bound_report(bounds=bounds)
    assert raw["production_launch_ready"] is False
    store = _store(tmp_path)
    installed = resume.production_transition_bound_report(
        bounds=bounds, transition_store=store
    )
    assert installed["production_launch_ready"] is True
    assert installed["external_transition_store_installed"] is True
    assert installed["external_transition_store_audit"][
        "scientific_parameters_changed"
    ] is False
    store.close()


def test_native_candidate_snapshot_is_lossless_and_independently_reopenable(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    coverage_a = torch.tensor([1.0, 0.0, 1.0, 0.0]).to_sparse()
    coverage_b = torch.tensor([0.0, 1.0, 0.0, 1.0]).to_sparse()
    vrrw = type("VRRW", (), {})()
    vrrw.graph_map = {
        GRAPH_A: {
            "x": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "edge_index": torch.empty((2, 0), dtype=torch.int64),
        },
        GRAPH_B: {
            "x": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            "edge_index": torch.empty((2, 0), dtype=torch.int64),
        },
    }
    vrrw.graph_index_map = {GRAPH_A: 0, GRAPH_B: 1}
    vrrw.counterfactual_candidates = [
        {
            "frequency": 7,
            "graph_hash": GRAPH_A,
            "importance_parts": np.asarray([0.8, 0.5], dtype=np.float64),
            "input_graphs_covering_list": coverage_a,
        },
        {
            "frequency": 3,
            "graph_hash": GRAPH_B,
            "importance_parts": np.asarray([0.7, 0.25], dtype=np.float64),
            "input_graphs_covering_list": coverage_b,
        },
    ]
    root = (tmp_path / "native-candidates").resolve()
    manifest = candidates.write_native_candidate_snapshot(
        root,
        vrrw=vrrw,
        checkpoint_cursor=20_000,
        contract_sha256="1" * 64,
        attempt_id="attempt-1",
        generation_token="2" * 64,
        torch=torch,
    )
    reopened = candidates.reopen_native_candidate_snapshot(
        manifest,
        expected_contract_sha256="1" * 64,
        expected_attempt_id="attempt-1",
        expected_generation_token="2" * 64,
        torch=torch,
    )
    assert list(reopened["graph_map"]) == [GRAPH_A, GRAPH_B]
    assert [row["frequency"] for row in reopened["counterfactual_candidates"]] == [
        7,
        3,
    ]
    assert torch.equal(
        reopened["counterfactual_candidates"][0][
            "input_graphs_covering_list"
        ].to_dense(),
        coverage_a.to_dense(),
    )


def test_native_candidate_snapshot_rejects_order_index_drift(tmp_path):
    torch = pytest.importorskip("torch")
    vrrw = type("VRRW", (), {})()
    vrrw.graph_map = {GRAPH_A: {"x": torch.ones((1, 1))}}
    vrrw.graph_index_map = {GRAPH_A: 1}
    vrrw.counterfactual_candidates = [
        {
            "frequency": 2,
            "graph_hash": GRAPH_A,
            "importance_parts": [0.8, 0.5],
            "input_graphs_covering_list": torch.ones(4).to_sparse(),
        }
    ]
    with pytest.raises(
        candidates.TasteT12CandidateStoreError, match="order/index"
    ):
        candidates.write_native_candidate_snapshot(
            (tmp_path / "native-candidates").resolve(),
            vrrw=vrrw,
            checkpoint_cursor=20_000,
            contract_sha256="1" * 64,
            attempt_id="attempt-1",
            generation_token="2" * 64,
            torch=torch,
        )


def test_official_native_result_is_exactly_bound_to_live_terminal_state(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    runtime = tmp_path.resolve() / "runtime"
    path = runtime / "results/tastemolnet/runs/counterfactuals.pt"
    path.parent.mkdir(parents=True)
    vrrw = type("VRRW", (), {})()
    vrrw.graph_map = {GRAPH_A: {"x": torch.ones((1, 1))}}
    vrrw.graph_index_map = {GRAPH_A: 0}
    vrrw.counterfactual_candidates = [
        {
            "frequency": 1,
            "graph_hash": GRAPH_A,
            "importance_parts": np.asarray([0.8, 0.5]),
            "input_graphs_covering_list": torch.tensor([1.0, 0.0]).to_sparse(),
        }
    ]
    vrrw.MAX_COUNTERFACTUAL_SIZE = 100_000
    vrrw.traversed_hashes = [GRAPH_A]
    vrrw.input_graphs_covered = torch.tensor([1.0, 0.0])
    payload = {
        name: getattr(vrrw, name)
        for name in (
            "graph_map",
            "graph_index_map",
            "counterfactual_candidates",
            "MAX_COUNTERFACTUAL_SIZE",
            "traversed_hashes",
            "input_graphs_covered",
        )
    }
    torch.save(payload, path)
    observed, raw_sha, semantic_sha = full._load_and_validate_native_result(
        runtime_root=runtime, vrrw=vrrw, torch=torch
    )
    assert observed == path
    assert len(raw_sha) == len(semantic_sha) == 64
    vrrw.traversed_hashes.append(GRAPH_B)
    with pytest.raises(
        resume.TasteGCFFullResumeError, match="terminal live state"
    ):
        full._load_and_validate_native_result(
            runtime_root=runtime, vrrw=vrrw, torch=torch
        )


def test_t12_production_wrappers_pin_fresh_resume_and_generation_only_verify():
    run_slurm = (
        REPOSITORY / "scripts/slurm/run_tastemolnet_gcf_full.sh"
    ).read_text(encoding="utf-8")
    verify_slurm = (
        REPOSITORY / "scripts/slurm/verify_tastemolnet_gcf_full_generation.sh"
    ).read_text(encoding="utf-8")
    for wrapper in (run_slurm, verify_slurm):
        for required in (
            "#SBATCH --partition=A800",
            "#SBATCH --gres=gpu:a800:1",
            "#SBATCH --output=logs/%j.out",
            "#SBATCH --error=logs/%j.err",
            "source ~/.bashrc",
            "conda activate smiles_pip118",
            "cd /share/home/u20526/czx/counterfactual-subgraph",
            "export PYTHONPATH=$PWD",
            "--config configs/hpc.yaml",
            "--set inference.fallback_to_heuristic=false",
        ):
            assert required in wrapper
    sidecar = (
        REPOSITORY / "scripts/autodl/run_tastemolnet_t12_after_t11_v1.sh"
    ).read_text(encoding="utf-8")
    assert "T12_WAIT_PID_START_TICKS" in sidecar
    assert "--mode fresh" in sidecar
    assert "--mode resume" in sidecar
    assert "checkpoint-00010000.manifest.json" in sidecar
    assert "checkpoint-00020000.manifest.json" in sidecar
    assert "verify_tastemolnet_gcf_full_generation.py" in sidecar
    assert "GENERATION_PASS" in sidecar
    assert "[TASTE_GCF_PASS]" not in sidecar


def test_t12_full_imports_lineage_wrapper_from_its_owner_module():
    source = (
        REPOSITORY / "src/baselines/tastemolnet_gcf_full.py"
    ).read_text(encoding="utf-8")
    assert (
        "from src.baselines.gcfexplainer_mutagenicity_adapter import (\n"
        "            graph_lineage_neighbor_wrapper,\n"
        "        )"
    ) in source
    assert (
        "from src.baselines.tastemolnet_gcf_smoke import "
        "graph_lineage_neighbor_wrapper"
    ) not in source
