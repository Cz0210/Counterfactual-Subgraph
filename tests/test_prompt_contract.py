from src.data.prompts import build_counterfactual_prompt
from src.data.schemas import MoleculeRecord


def test_counterfactual_prompt_uses_fragment_only_contract() -> None:
    record = MoleculeRecord(record_id=1, smiles="CCO", label=1)

    prompt = build_counterfactual_prompt(record)

    assert "deletion is most likely to flip the molecule label" in prompt
    assert "Output SMILES only, no extra text." in prompt
    assert prompt.endswith("FRAGMENT_SMILES:")
