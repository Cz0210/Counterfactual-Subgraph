from __future__ import annotations

from rdkit import Chem

from src.chem.hard_deletion import apply_hard_deletion_match


def test_internal_linker_deletion_is_rejected_as_disconnected() -> None:
    parent = Chem.MolFromSmiles("CCOCC")
    outcome = apply_hard_deletion_match(parent, (2,))
    assert outcome.sanitize_ok is True
    assert outcome.residual_num_components == 2
    assert outcome.residual_connected is False
    assert outcome.contains_dot is True
    assert outcome.valid is False
    assert outcome.invalid_reason == "disconnected_residual"


def test_pendant_deletion_keeps_one_sanitized_component() -> None:
    parent = Chem.MolFromSmiles("CCCO")
    outcome = apply_hard_deletion_match(parent, (3,))
    assert outcome.valid is True
    assert outcome.residual_smiles == "CCC"
    assert outcome.sanitize_ok is True
    assert outcome.residual_num_components == 1
    assert outcome.residual_connected is True
    assert outcome.contains_dot is False


def test_empty_residual_is_rejected() -> None:
    parent = Chem.MolFromSmiles("C")
    outcome = apply_hard_deletion_match(parent, (0,))
    assert outcome.valid is False
    assert outcome.invalid_reason == "empty_residual_after_deletion"
