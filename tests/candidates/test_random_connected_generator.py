from __future__ import annotations

from src.candidates.base_generator import CandidateRequest
from src.candidates.random_connected_subgraph_generator import (
    RandomConnectedSubgraphGenerator,
)


def _request(**overrides: object) -> CandidateRequest:
    values: dict[str, object] = {
        "parent_id": "p1",
        "parent_smiles": "CCOC(=O)NCCc1ccccc1",
        "parent_split": "train",
        "label": 1,
        "candidates_per_parent": 4,
        "size_targets": (2, 3, 4, 5),
        "seed": 17,
        "max_attempts": 200,
    }
    values.update(overrides)
    return CandidateRequest(**values)  # type: ignore[arg-type]


def test_random_connected_is_deterministic_connected_and_schema_compatible() -> None:
    generator = RandomConnectedSubgraphGenerator()
    first = generator.generate(_request())
    second = generator.generate(_request())
    assert first == second
    assert first.generated_count == 4
    assert [row["generation_rank"] for row in first.rows] == [1, 2, 3, 4]
    assert all(row["valid"] and row["connected"] for row in first.rows)
    assert all(row["direct_substructure"] and row["final_substructure"] for row in first.rows)
    assert all(row["candidate_source"] == "random_connected_size_matched" for row in first.rows)


def test_random_connected_records_shortfall() -> None:
    batch = RandomConnectedSubgraphGenerator().generate(
        _request(candidates_per_parent=2, size_targets=(99,), max_attempts=3)
    )
    assert batch.generated_count == 0
    assert batch.shortfall_count == 2
    assert batch.shortfall_reason_counts["target_size_unavailable"] == 3
    assert batch.shortfall_reason_counts["max_attempts_exhausted"] == 2


def test_random_candidate_rejects_test_parent() -> None:
    try:
        RandomConnectedSubgraphGenerator().generate(_request(parent_split="test"))
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("test candidate generation was accepted")
