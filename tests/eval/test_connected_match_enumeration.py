from __future__ import annotations

from src.chem.hard_deletion import enumerate_connected_hard_deletions


def test_exact_unique_match_sets_preserve_atom_mapping() -> None:
    outcomes = enumerate_connected_hard_deletions("CCC", "C")
    assert [outcome.match_atom_indices for outcome in outcomes] == [(0,), (1,), (2,)]
    assert len({outcome.match_atom_indices for outcome in outcomes}) == 3
    assert outcomes[0].valid is True
    assert outcomes[1].valid is False
    assert outcomes[1].residual_num_components == 2
    assert outcomes[2].valid is True


def test_exact_bond_query_does_not_match_incompatible_bond() -> None:
    assert enumerate_connected_hard_deletions("CC", "C=C") == []
