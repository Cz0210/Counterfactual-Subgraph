from __future__ import annotations

from src.candidates.base_generator import CandidateRequest
from src.candidates.random_brics_generator import RandomBRICSGenerator


def _request(smiles: str = "CCOC(=O)NCCc1ccccc1") -> CandidateRequest:
    return CandidateRequest(
        parent_id="p1",
        parent_smiles=smiles,
        parent_split="val",
        label=1,
        candidates_per_parent=3,
        size_targets=(2, 4, 6),
        seed=23,
        max_attempts=100,
    )


def test_random_brics_is_deterministic_and_deduplicated() -> None:
    generator = RandomBRICSGenerator()
    first = generator.generate(_request())
    second = generator.generate(_request())
    assert first == second
    fragments = [row["final_fragment"] for row in first.rows]
    assert len(fragments) == len(set(fragments))
    assert all(row["candidate_source"] == "random_brics_size_matched" for row in first.rows)
    assert first.generated_count + first.shortfall_count == first.requested_count


def test_random_brics_no_bond_shortfall_is_explicit() -> None:
    result = RandomBRICSGenerator().generate(_request("CCCC"))
    assert result.generated_count == 0
    assert result.shortfall_count == 3
    assert result.shortfall_reason_counts == {"no_brics_bonds": 3}
