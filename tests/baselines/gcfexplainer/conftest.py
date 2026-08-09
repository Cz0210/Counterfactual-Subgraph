from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.baselines.gcfexplainer_mutagenicity_adapter import GeneratedDecodeResult


class FakeTeacher:
    available = True

    def __init__(self, target_smiles: set[str]) -> None:
        self.target_smiles = set(target_smiles)

    def score_smiles(self, smiles: str, label: int | None = None, **_kwargs):
        prediction = 0 if smiles in self.target_smiles else 1
        probability0 = 0.9 if prediction == 0 else 0.1
        probability1 = 1.0 - probability0
        probability = probability0 if label == 0 else probability1
        return {
            "teacher_result_ok": True,
            "teacher_label": prediction,
            "teacher_prob": probability,
        }


def fake_graph(index: int) -> SimpleNamespace:
    return SimpleNamespace(candidate_test_index=index, gcf_origin_index=[0])


def decode_result(index: int, *, valid: bool = True, smiles: str | None = None):
    if not valid:
        return GeneratedDecodeResult(
            decode_ok=False,
            canonical_smiles="",
            raw_smiles="",
            failure_reason="generated_valence_sanitize_failed",
            projected_new_edge_count=0,
            retained_edge_count=0,
            removed_source_edge_count=0,
            inherited_atom_state_count=0,
            reset_atom_state_count=0,
            source_parent_id="BACE_SOURCE",
        )
    canonical = smiles or f"C{index}"
    return GeneratedDecodeResult(
        decode_ok=True,
        canonical_smiles=canonical,
        raw_smiles=canonical,
        failure_reason="",
        projected_new_edge_count=1,
        retained_edge_count=2,
        removed_source_edge_count=0,
        inherited_atom_state_count=1,
        reset_atom_state_count=1,
        source_parent_id="BACE_SOURCE",
    )


def ranked_candidates(count: int):
    return [
        (
            {
                "candidate_id": f"NATIVE_{index:04d}",
                "native_rank": index + 1,
                "graph_hash": str(index),
                "frequency": count - index,
                "covered_parent_count_at_rank": max(0, 5 - index),
            },
            fake_graph(index),
        )
        for index in range(count)
    ]


@pytest.fixture
def source_records():
    return [{"molecule_id": "BACE_SOURCE"}]
