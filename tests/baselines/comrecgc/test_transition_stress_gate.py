from __future__ import annotations

from src.baselines.comrecgc.stress_gate import run_transition_eviction_stress_gate


def test_scaled_transition_eviction_gate_exercises_real_rehydration(tmp_path) -> None:
    result = run_transition_eviction_stress_gate(
        output_root=tmp_path, steps=512, cache_max_entries=16
    )
    assert result["stress_gate_passed"] is True
    assert result["active_eviction_prevented"] > 0
    assert result["eviction_committed"] > 0
    assert result["deferred_flushed"] > 0
    assert result["unresolved_lookups"] == 0
    assert result["result_parity_with_unbounded_reference"] is True
    assert result["cache_bound_respected"] is True
