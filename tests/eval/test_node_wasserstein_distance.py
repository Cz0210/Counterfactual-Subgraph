from __future__ import annotations

import itertools
import math

import numpy as np

from scripts.evaluate_ccrcov_with_molclr_node_wasserstein import ResumeCheckpoint
from src.eval.distance_cache import canonical_pair_key
from src.eval.node_wasserstein_distance import (
    compute_node_wasserstein_distance,
    node_wasserstein_pair_key,
    uniform_node_mass,
)


def _small_uniform_emd2(a: np.ndarray, b: np.ndarray, cost: np.ndarray) -> float:
    assert math.isclose(float(a.sum()), 1.0)
    assert math.isclose(float(b.sum()), 1.0)
    if len(a) == len(b):
        return min(
            sum(float(cost[index, target]) for index, target in enumerate(permutation)) / len(a)
            for permutation in itertools.permutations(range(len(b)))
        )
    return float(np.min(cost))


def test_identical_embeddings_are_zero_and_uniform_mass_sums_to_one() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    distance, metadata = compute_node_wasserstein_distance(
        embeddings, embeddings, emd2_fn=_small_uniform_emd2
    )
    assert distance <= 1e-12
    assert math.isclose(float(uniform_node_mass(7).sum()), 1.0)
    assert metadata["mass_sum_a"] == metadata["mass_sum_b"] == 1.0


def test_symmetry() -> None:
    left = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    right = np.asarray([[0.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    forward, _ = compute_node_wasserstein_distance(left, right, emd2_fn=_small_uniform_emd2)
    reverse, _ = compute_node_wasserstein_distance(right, left, emd2_fn=_small_uniform_emd2)
    assert math.isclose(forward, reverse, rel_tol=1e-12, abs_tol=1e-12)


def test_size_penalty_formula_and_zero_beta() -> None:
    left = np.ones((2, 3), dtype=np.float32)
    right = np.ones((4, 3), dtype=np.float32)
    zero, zero_meta = compute_node_wasserstein_distance(
        left, right, size_penalty_beta=0.0, emd2_fn=lambda _a, _b, _m: 0.0
    )
    penalized, meta = compute_node_wasserstein_distance(
        left, right, size_penalty_beta=0.6, emd2_fn=lambda _a, _b, _m: 0.0
    )
    assert zero == zero_meta["base_ot_distance"] == 0.0
    assert math.isclose(meta["size_penalty"], 0.6 * 2 / 4)
    assert math.isclose(penalized, 0.3)


def test_wnode_pair_keys_are_symmetric_independent_and_beta_sensitive() -> None:
    kwargs = dict(
        checkpoint_identity="checkpoint",
        feature_cost="cosine",
        node_mass="uniform",
        size_penalty_beta=0.0,
    )
    forward = node_wasserstein_pair_key(canonical_smiles_a="CCO", canonical_smiles_b="CCN", **kwargs)
    reverse = node_wasserstein_pair_key(canonical_smiles_a="CCN", canonical_smiles_b="CCO", **kwargs)
    beta = node_wasserstein_pair_key(
        canonical_smiles_a="CCO", canonical_smiles_b="CCN", **{**kwargs, "size_penalty_beta": 0.1}
    )
    fgw = canonical_pair_key(
        distance_type="molclr_node_fgw", version="molclr_node_fgw_v1",
        canonical_smiles_a="CCO", canonical_smiles_b="CCN", molclr_ckpt="checkpoint",
        fgw_lambda=0.5, structure_mode="shortest_path_unweighted",
        feature_cost="cosine", atom_penalty=0.0,
    )
    assert forward == reverse
    assert forward != beta
    assert forward != fgw


def test_resume_checkpoint_preserves_completed_pairs_without_duplicate_rows(tmp_path) -> None:
    checkpoint = ResumeCheckpoint(tmp_path, "fullgraph", "fingerprint", True)
    rows = [{"parent_id": "p1", "candidate_id": "c1", "distance": 0.2}]
    checkpoint.save(rows, {("p1", "c1")})
    loaded, completed = checkpoint.load()
    assert len(loaded) == 1
    assert completed == {("p1", "c1")}
    remaining = [pair for pair in (("p1", "c1"), ("p1", "c2")) if pair not in completed]
    assert remaining == [("p1", "c2")]
