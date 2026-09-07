"""A+ opt-in, GIN-aligned attachment-preserving GlobalGCE materialization.

Historical/Taste decoders remain unchanged. Only this new version keeps the
parent RDKit graph, constrains real boundary anchors, and uses the same joint
NONE/bond categorical state in training, validation and final deployment.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Mapping
import torch
from torch.nn import functional as F
from rdkit import Chem

from src.baselines.globalgce_bace_native_rules import (
    GlobalGCENativeRule, NativeParentTensors, build_parent_native_tensors,
    _edge_position,
)
from src.baselines.globalgce_frozen_gine_bridge import (
    FrozenGINEDifferentiableBridge, _normalize_distribution, _schema_index,
    _straight_through,
)
from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer

SCHEMA = "bace_globalgce_gin_aplus_attachment_v1"
CONTRACT = {
    "schema": SCHEMA, "method_variant": "GlobalGCE-ChemAligned-GIN-Aplus",
    "source_driver": "cd051072a3274ba0189f614a081706559c7515b6",
    "adjacency": "sigmoid_probability_symmetric_zero_diagonal",
    "edge_logits": "official_affine_lower_triangle_NONE_first",
    "joint_NONE": "1-a+a*softmax(edge)[0]",
    "joint_bond": "a*softmax(edge)[bond]",
    "hard_decode": "joint_argmax_NONE_lowest_tie",
    "node_mapping": "inverse_exact_LHS_tensor_slot_to_parent_index",
    "boundary_anchors": "original_atom_identity_fixed_when_incident_to_outside",
    "outside_graph": "copy_original_RDKit_atoms_bonds_stereo_only_edit_LHS_square",
    "new_nodes": "original_decoder_padding_slots_no_extra_attachment_search",
    "chemical_gate": "nonempty_connected_complete_product_RDKit_sanitize",
    "invalid_graph_oracle": "not_called",
    "gradient": "straight_through_estimator_not_exact_discrete_derivative",
    "training_oracle": "frozen_corrected_GIN_seed7",
    "calibration_or_test_used_for_repair": False,
    "benchmark_test_previously_seen": True,
}

@dataclass(frozen=True)
class AlignedParent:
    canonical_smiles: str
    feature: Any
    adjacency: Any
    edge_attr: Any
    atom_attributes: tuple
    source_molecule: Any

def build_parent(smiles, *, atom_symbols, bond_names):
    original = build_parent_native_tensors(smiles, atom_symbols=atom_symbols, bond_names=bond_names)
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("INVALID_PARENT")
    Chem.Kekulize(molecule, clearAromaticFlags=True)
    # Native tensor ordering is the ORIGINAL supplied SMILES ordering, not a
    # reparse of canonical_smiles, which may permute boundary/stereo endpoints.
    return AlignedParent(original.canonical_smiles, original.feature, original.adjacency,
        original.edge_attr, original.atom_attributes, molecule)

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


def apply_states(parent, rule, mapping: Mapping[int, int], feature, states):
    """Keep the original square complement, with explicit fixed anchor atoms."""
    inverse = {int(v): int(k) for k, v in mapping.items()}
    if len(inverse) != len(mapping) or set(inverse) != set(rule.lhs_nodes):
        raise ValueError("LHS_MAPPING_NOT_BIJECTIVE")
    pn = len(parent.feature)
    if any(k < 0 or k >= pn for k in mapping):
        raise ValueError("LHS_MAPPING_OUTSIDE_PARENT")
    if feature.shape != rule.lhs_feature.shape or states.shape != rule.lhs_edge_attr.shape:
        raise ValueError("RHS_SHAPE_CHANGED")
    anchors = {int(inside) for inside in mapping for outside in range(pn)
        if outside not in mapping and int(parent.edge_attr[_edge_position(inside, outside)].argmax()) > 0}
    rhs = feature.clone()
    for inside in anchors:
        rhs[int(mapping[inside])] = parent.feature[inside].to(feature.device)
    next_node, mask = pn, []
    for index in range(rule.maximum_nodes):
        if index in inverse:
            mask.append(inverse[index])
        else:
            mask.append(next_node); next_node += 1
    f = feature.new_zeros((next_node, feature.shape[-1])); f[:, 0] = 1
    f[:pn] = parent.feature.to(feature.device); f[mask] = rhs
    p = states.new_zeros((next_node*(next_node-1)//2, states.shape[-1])); p[:, 0] = 1
    p[:len(parent.edge_attr)] = parent.edge_attr.to(states.device)
    for right in range(rule.maximum_nodes):
        for left in range(right):
            p[_edge_position(mask[left], mask[right])] = states[_edge_position(left, right)]
    return f, p, tuple(mask), tuple(sorted(anchors))


@dataclass
class Materialized:
    canonical_smiles: str
    molecule: Any
    feature: Any
    edge_attr: Any
    active_indices: tuple[int, ...]
    mask_order: tuple[int, ...]
    boundary_count: int
    anchors: tuple[int, ...]


def materialize(parent, rule, mapping, feature, states):
    if not isinstance(parent, AlignedParent):
        raise ValueError("ORIGINAL_PARENT_MOLECULE_REQUIRED_NO_CANONICAL_REINDEX")
    f, p, mask, anchors = apply_states(parent, rule, mapping, feature, states)
    hf, _, he = hard_state_tensors(f, p)
    labels = hf.argmax(-1).detach().cpu().tolist()
    active = tuple(i for i, label in enumerate(labels) if label > 0)
    if not active:
        raise ValueError("EMPTY_COMPLETE_PRODUCT")
    pn = len(parent.feature)
    mol = Chem.RWMol(Chem.Mol(parent.source_molecule))
    bonds = {"single": Chem.BondType.SINGLE, "double": Chem.BondType.DOUBLE,
        "triple": Chem.BondType.TRIPLE, "aromatic": Chem.BondType.AROMATIC}
    for index, label in enumerate(labels):
        if label > len(rule.atom_symbols):
            raise ValueError("UNKNOWN_ATOM")
        atom = Chem.Atom(rule.atom_symbols[label-1]) if label else Chem.Atom(0)
        if index >= pn:
            mol.AddAtom(atom)
        elif label and atom.GetAtomicNum() != mol.GetAtomWithIdx(index).GetAtomicNum():
            # Same-element original atoms retain charge/isotope/chiral metadata.
            # Changed elements use neutral new-atom semantics, never a search.
            mol.ReplaceAtom(index, atom)
    for right in range(len(labels)):
        for left in range(right):
            label = int(he[_edge_position(left, right)].argmax())
            existing = mol.GetBondBetweenAtoms(left, right)
            wanted = bonds[rule.bond_names[label]] if label else None
            if existing is not None and existing.GetBondType() != wanted:
                mol.RemoveBond(left, right); existing = None
            if wanted is not None and existing is None:
                mol.AddBond(left, right, wanted)
    # Unchanged outside atoms/bonds are the original RDKit objects; identity
    # never destroys and reconstructs E/Z direction or tetrahedral ordering.
    for index in reversed(range(len(labels))):
        if not labels[index]:
            mol.RemoveAtom(index)
    product = mol.GetMol()
    if len(Chem.GetMolFrags(product)) != 1:
        raise ValueError("DISCONNECTED_COMPLETE_PRODUCT")
    try:
        Chem.SanitizeMol(product)
        Chem.AssignStereochemistry(product, cleanIt=True, force=True)
    except Exception as exc:
        raise ValueError("SANITIZATION_FAILED_COMPLETE_PRODUCT") from exc
    boundary = 0
    for inside in anchors:
        for outside in range(pn):
            if outside in mapping:
                continue
            slot = _edge_position(inside, outside)
            before, after = int(parent.edge_attr[slot].argmax()), int(he[slot].argmax())
            if before != after:
                raise ValueError("BOUNDARY_ATTACHMENT_CHANGED")
            boundary += int(before > 0)
    return Materialized(Chem.MolToSmiles(product, canonical=True, isomericSmiles=True),
        product, f, p, active, mask, boundary, anchors)


class GINAlignedBridge(FrozenGINEDifferentiableBridge):
    """Exact hard frozen GIN forward; opt-in ST input gradients only."""

    def __init__(self, model, *, feature_schema, atom_symbols, bond_names,
                 checkpoint_id, temperature, device="cpu", expected_num_classes=2):
        if model.config.backbone != "gin" or model.config.num_classes != expected_num_classes:
            raise ValueError("APLUS_REQUIRES_FROZEN_GIN_NOT_GINE")
        if model.config.pooling != "mean" or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("FROZEN_GIN_POOLING_OR_TEMPERATURE")
        if not atom_symbols or bond_names[0] != "no_edge":
            raise ValueError("NATIVE_CODEC_REQUIRED")
        self.model = model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False); parameter.grad = None
        self.feature_schema, self.atom_symbols = feature_schema, tuple(atom_symbols)
        self.bond_names, self.checkpoint_id = tuple(bond_names), checkpoint_id
        self.temperature, self.num_classes = temperature, expected_num_classes
        self.device, self.last_audit = torch.device(device), {}
        self._atomic_field = _schema_index(feature_schema, kind="node", name="atomic_num")
        self._bond_field = _schema_index(feature_schema, kind="edge", name="bond_type")

    def score_materialized(self, product):
        ft = MolecularGraphFeaturizer(self.feature_schema)
        x = torch.tensor([ft._encode_atom(a) for a in product.molecule.GetAtoms()],
            dtype=torch.long, device=self.device)
        active = product.active_indices
        selected = product.feature[list(active)].to(self.device)
        distribution = self._mapped_node_distribution(selected)
        hidden = None
        for field, embedding in enumerate(self.model.node_encoder.embeddings):
            value = embedding(x[:, field])
            if field == self._atomic_field:
                value = _straight_through(value, distribution @ embedding.weight)
            hidden = value if hidden is None else hidden+value
        presence = _normalize_distribution(selected, name="node_presence")[:, 1:].sum(-1)
        gate_node = _straight_through(torch.ones_like(presence), presence)
        hidden = hidden*gate_node[:, None]
        edges = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): ft._encode_bond(b)
            for b in product.molecule.GetBonds()}
        default = tuple(f.encode({"bond_type":"SINGLE", "stereo":"STEREONONE",
            "is_conjugated":0, "is_in_ring":0}[f.name]) for f in ft.schema.edge_fields)
        src, dst, hard_rows, soft_rows, hard_gates = [], [], [], [], []
        for i, ni in enumerate(active):
            for j, nj in enumerate(active):
                if i == j: continue
                key = tuple(sorted((i,j)))
                src.append(i); dst.append(j); hard_rows.append(edges.get(key, default))
                soft_rows.append(product.edge_attr[_edge_position(ni,nj)])
                hard_gates.append(float(key in edges))
        if src:
            edge_index = torch.tensor([src,dst], dtype=torch.long, device=self.device)
            hard = torch.tensor(hard_rows, dtype=torch.long, device=self.device)
            # State probabilities are already joint. log+softmax is identity;
            # no second adjacency multiplication or double temperature scaling.
            probabilities = torch.stack(soft_rows).to(self.device)
            mapped, presence_edge = self._mapped_edge_distribution(probabilities.clamp_min(1e-30).log())
            edge_hidden = None
            for field, embedding in enumerate(self.model.edge_encoder.embeddings):
                value = embedding(hard[:,field])
                if field == self._bond_field:
                    value = _straight_through(value, mapped@embedding.weight)
                edge_hidden = value if edge_hidden is None else edge_hidden+value
            edge_gate = _straight_through(hidden.new_tensor(hard_gates), presence_edge)
        else:
            edge_index = torch.empty((2,0), dtype=torch.long, device=self.device)
            edge_hidden, edge_gate = hidden.new_empty((0,hidden.shape[-1])), hidden.new_empty((0,))
        for layer, norm in zip(self.model.layers,self.model.normalizations,strict=True):
            if layer.backbone != "gin": raise ValueError("GIN_LAYER_CHANGED")
            messages = (hidden[edge_index[0]]+edge_hidden)*edge_gate[:,None]
            summed = layer._aggregate_sum(messages,edge_index[1],len(hidden))
            updated = layer.update_mlp((1+layer.eps)*hidden+summed)
            if self.model.config.residual and updated.shape == hidden.shape: updated = updated+hidden
            hidden = self.model.dropout(torch.relu(norm(updated)))
        pooled = (hidden*gate_node[:,None]).sum(0,keepdim=True)/gate_node.sum().clamp_min(1)
        logits = self.model.classifier(pooled)
        return {"logits":logits, "y_pred":F.log_softmax(logits/self.temperature,-1),
            "audit":{"classifier":"gin", "sanitized":True, "joint_states_once":True}}

