"""Opt-in BACE molecular adapter; no change to the historical/Taste decoder.

The pinned decoder emits sigmoid node weights, sigmoid adjacency probabilities,
and affine bond logits (including NONE). This *new* adapter combines the latter
two into one categorical state before either training or deployment. The native
LHS matching and boundary attachment operation remain local graph replacement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from rdkit import Chem

from src.baselines.globalgce_bace_native_rules import (
    GlobalGCENativeRule, NativeParentTensors, _apply_atom_attributes, _edge_position,
)
from src.baselines.globalgce_frozen_gine_bridge import (
    FrozenGINEDifferentiableBridge, _HardGraph, _straight_through,
)
from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer

SCHEMA = "bace_globalgce_chemaligned_joint_states_v2"
CONTRACT = {
    "schema_version": SCHEMA,
    "method": "GlobalGCE-ChemAligned",
    "adjacency_input": "sigmoid_probability_symmetric_zero_diagonal",
    "edge_input": "pinned_official_affine_logits_lower_triangle_row_major",
    "node_input": "sigmoid_nonnegative_weights_argmax_padding0",
    "none_index": 0,
    "conditional_bond_distribution": "softmax_last_axis",
    "joint_none": "1-a+a*q_NONE",
    "joint_bond": "a*q_bond",
    "hard_state": "joint_argmax_lowest_index_tie_NONE_first",
    "adjacency_output": "derived_from_non_NONE_state_and_active_nodes",
    "node_mapping": "inverse_LHS_mapping_by_tensor_index_then_appended_padding",
    "attachment": "preserve_all_parent_edges_outside_replacement_square",
    "validation_scope": "complete_parent_replacement_not_isolated_RHS",
    "chemical_validation": "connected_nonempty_RDKit_sanitize_no_repair",
    "atom_attributes": "inherit_unchanged_element_parent_attributes_else_neutral_defaults",
    "gradient": "straight_through_estimator_not_exact_discrete_derivative",
    "invalid_product_oracle": "not_called_no_fallback_features",
    "test_used_to_choose_contract": False,
    "benchmark_test_previously_seen": True,
    "old_results_modified": False,
}


def joint_states(adjacency: torch.Tensor, edge_logits: torch.Tensor) -> torch.Tensor:
    n = adjacency.shape[-1]
    if adjacency.ndim != 2 or adjacency.shape != (n, n):
        raise ValueError("adjacency shape")
    if edge_logits.ndim != 2 or edge_logits.shape[0] != n*(n-1)//2:
        raise ValueError("lower-triangle edge shape")
    if edge_logits.shape[-1] < 2 or not torch.isfinite(edge_logits).all():
        raise ValueError("bond logits domain")
    if not torch.isfinite(adjacency).all() or (adjacency < 0).any() or (adjacency > 1).any():
        raise ValueError("adjacency must be a probability, not logits")
    if not torch.allclose(adjacency, adjacency.T, atol=1e-6, rtol=0):
        raise ValueError("asymmetric adjacency")
    if torch.any(adjacency.diagonal() != 0):
        raise ValueError("self loop")
    rows, cols = torch.tril_indices(n, n, offset=-1, device=adjacency.device)
    a = .5*(adjacency[rows, cols]+adjacency[cols, rows])
    q = edge_logits.softmax(-1)
    return torch.cat(((1-a+a*q[:, 0]).unsqueeze(-1), a[:, None]*q[:, 1:]), -1)


def hard_state_tensors(features: torch.Tensor, states: torch.Tensor):
    if features.ndim != 2 or not torch.isfinite(features).all() or (features < 0).any():
        raise ValueError("node weights domain")
    if torch.any(features.sum(-1) <= 0):
        raise ValueError("empty node distribution")
    n = len(features)
    labels = features.argmax(-1)
    pair_labels = states.argmax(-1)
    rows, cols = torch.tril_indices(n, n, offset=-1, device=features.device)
    pair_labels = torch.where((labels[rows] > 0) & (labels[cols] > 0), pair_labels, 0)
    edges = F.one_hot(pair_labels, states.shape[-1]).to(features.dtype)
    adjacency = features.new_zeros((n, n))
    adjacency[rows, cols] = (pair_labels > 0).to(features.dtype)
    adjacency = adjacency + adjacency.T
    return F.one_hot(labels, features.shape[-1]).to(features.dtype), adjacency, edges


def corrected_rule(rule: GlobalGCENativeRule, raw_feature, raw_adj, raw_edge) -> GlobalGCENativeRule:
    f, a, e = hard_state_tensors(raw_feature, joint_states(raw_adj, raw_edge))
    from dataclasses import replace
    result = replace(rule, rhs_feature=f.detach().cpu(), rhs_adjacency=a.detach().cpu(),
                     rhs_edge_attr=e.detach().cpu())
    result.validate()  # LHS syntax; whole products are validated in materialize.
    return result


def apply_states(parent: NativeParentTensors, rule: GlobalGCENativeRule,
                 mapping: Mapping[int, int], feature, states):
    """Differentiable assignment; immutable parent, no RNG or detached RHS."""
    inverse = {int(v): int(k) for k, v in mapping.items()}
    if len(inverse) != len(mapping) or set(inverse) != set(rule.lhs_nodes):
        raise ValueError("LHS mapping is not bijective")
    pn = len(parent.feature)
    if any(k < 0 or k >= pn for k in mapping):
        raise ValueError("mapping outside parent")
    next_node = pn
    mask = []
    for index in range(rule.maximum_nodes):
        if index in inverse:
            mask.append(inverse[index])
        else:
            mask.append(next_node)
            next_node += 1
    total = next_node
    f = feature.new_zeros((total, feature.shape[-1])); f[:, 0] = 1
    f[:pn] = parent.feature.to(feature.device)
    p = states.new_zeros((total*(total-1)//2, states.shape[-1])); p[:, 0] = 1
    p[:len(parent.edge_attr)] = parent.edge_attr.to(states.device)
    f[mask] = feature
    for right in range(rule.maximum_nodes):
        for left in range(right):
            p[_edge_position(mask[left], mask[right])] = states[_edge_position(left, right)]
    return f, p, tuple(mask)


@dataclass
class Materialized:
    canonical_smiles: str
    molecule: Any
    feature: Any
    adjacency: Any
    edge_attr: Any
    active_indices: tuple[int, ...]
    mask_order: tuple[int, ...]
    boundary_count: int
    attributes_inherited: int
    attributes_reset: int


def materialize(parent, rule, mapping, feature, states) -> Materialized:
    f, p, mask = apply_states(parent, rule, mapping, feature, states)
    hf, ha, he = hard_state_tensors(f, p)
    labels = hf.argmax(-1)
    active = tuple(i for i in range(len(labels)) if int(labels[i]) > 0)
    if not active:
        raise ValueError("EMPTY_COMPLETE_PRODUCT")
    index_map = {old: new for new, old in enumerate(active)}
    attr = {int(row["native_node_index"]): row for row in parent.atom_attributes}
    mol = Chem.RWMol(); inherited = reset = 0
    for old in active:
        label = int(labels[old])
        if label > len(rule.atom_symbols):
            raise ValueError("UNKNOWN_ATOM")
        atom = Chem.Atom(rule.atom_symbols[label-1])
        if old in attr and int(attr[old]["atomic_num"]) == atom.GetAtomicNum():
            _apply_atom_attributes(atom, attr[old]); inherited += 1
        else:
            reset += 1
        mol.AddAtom(atom)
    bonds = {"single": Chem.BondType.SINGLE, "double": Chem.BondType.DOUBLE,
             "triple": Chem.BondType.TRIPLE, "aromatic": Chem.BondType.AROMATIC}
    for right_pos, right in enumerate(active):
        for left in active[:right_pos]:
            label = int(he[_edge_position(left, right)].argmax())
            if label:
                mol.AddBond(index_map[left], index_map[right], bonds[rule.bond_names[label]])
    product = mol.GetMol()
    if len(Chem.GetMolFrags(product)) != 1:
        raise ValueError("DISCONNECTED_COMPLETE_PRODUCT")
    try:
        Chem.SanitizeMol(product)
    except Exception as exc:
        raise ValueError("SANITIZATION_FAILED_COMPLETE_PRODUCT") from exc
    canonical = Chem.MolToSmiles(product, canonical=True, isomericSmiles=True)
    boundary = 0
    for inside in set(mask) & set(range(len(parent.feature))):
        for outside in range(len(parent.feature)):
            if outside in mask:
                continue
            slot = _edge_position(inside, outside)
            # Atom removal legitimately removes incident edges; an external
            # attachment cannot be called preserved if its endpoint vanished.
            before = int(parent.edge_attr[slot].argmax())
            after = int(he[slot].argmax())
            if before != after:
                raise ValueError("BOUNDARY_ATTACHMENT_REMOVED")
            boundary += int(before > 0)
    return Materialized(canonical, product, f, ha, p, active, mask, boundary, inherited, reset)


class ChemAlignedBridge(FrozenGINEDifferentiableBridge):
    """Same frozen GINE, hard graph derived from the complete chemical product."""

    def score_materialized(self, product: Materialized):
        # Reuse existing GINE/ST embedding execution, but supply the *validated*
        # full graph and source attributes. No old chemical fallback is called.
        ft = MolecularGraphFeaturizer(self.feature_schema)
        edge_rows = {}
        for bond in product.molecule.GetBonds():
            edge_rows[tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))] = ft._encode_bond(bond)
        graph = _HardGraph(
            x=torch.tensor([ft._encode_atom(a) for a in product.molecule.GetAtoms()], dtype=torch.long, device=self.device),
            active_native_indices=product.active_indices,
            native_to_active={old: new for new, old in enumerate(product.active_indices)},
            edge_features=edge_rows, hard_edges=frozenset(edge_rows), sanitized=True, failure_reason=None,
        )
        # The base relaxation computes adjacency * bond_presence. Set its
        # adjacency factor to one because presence is already in joint states;
        # passing log(p) means softmax recovers p (no double scaling).
        n = len(product.feature)
        a = torch.ones((n, n), device=self.device, dtype=product.feature.dtype)-torch.eye(n, device=self.device)
        logits, audit = self._one_graph(product.feature, a, product.edge_attr.clamp_min(1e-30).log(),
                                        hard_graph_override=graph)
        return {"logits": logits, "y_pred": F.log_softmax(logits/self.temperature, -1), "audit": audit}
