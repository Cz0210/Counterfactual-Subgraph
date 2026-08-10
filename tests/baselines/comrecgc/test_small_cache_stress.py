from __future__ import annotations

from src.baselines.comrecgc.stress_gate import run_transition_eviction_stress_gate


def test_small_cache_stress_preserves_every_live_graph(tmp_path) -> None:
    result = run_transition_eviction_stress_gate(
        output_root=tmp_path / "stress", steps=256, cache_max_entries=32
    )
    assert result["stress_gate_passed"] is True
    assert result["cache_bound_respected"] is True
    assert result["unresolved_lookups"] == 0
    assert result["result_parity_with_unbounded_reference"] is True
