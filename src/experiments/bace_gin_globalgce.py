"""Scheme-A GlobalGCE: original 80 outputs, no fitting or new generation.

This adapter extracts only the previously audited joint hard decoding and
attachment-aware complete-product materializer from cd051072.  It neither
imports that campaign's training bridge nor loads its repaired weights.
Classification belongs to the common frozen-GIN evaluator, after chemistry.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.eval.bace_frozen_gnn_contracts import atomic_json, stable_sha256, utc_now

ORIGINAL_MODEL_SHA = "13eb16900da00e656c90b56945fde32def3492f7ee8aaf9e8bb257f6d2a126a2"
ORIGINAL_RULES_SHA = "5df95a277f64792ef4f660ebef14f96f92a36aadb9c9f84401817aabc3fed2a5"
MATERIALIZER_SOURCE_COMMIT = "cd051072a3274ba0189f614a081706559c7515b6"
JOINT_CONTRACT = {
    "adjacency_input": "sigmoid_probability_symmetric_zero_diagonal",
    "edge_input": "pinned_official_affine_logits_lower_triangle_row_major",
    "joint_none": "1-a+a*q_NONE", "joint_bond": "a*q_bond",
    "hard_state": "joint_argmax_lowest_index_tie_NONE_first",
    "attachment": "preserve_all_parent_edges_outside_replacement_square",
    "validation_scope": "complete_parent_replacement_not_isolated_RHS",
    "chemical_validation": "connected_nonempty_RDKit_sanitize_no_repair",
}


def joint_states(adjacency, edge_logits):
    import torch
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


def hard_state_tensors(features, states):
    import torch
    from torch.nn.functional import one_hot
    if features.ndim != 2 or not torch.isfinite(features).all() or (features < 0).any():
        raise ValueError("node weights domain")
    if torch.any(features.sum(-1) <= 0):
        raise ValueError("empty node distribution")
    n = len(features)
    labels, pair_labels = features.argmax(-1), states.argmax(-1)
    rows, cols = torch.tril_indices(n, n, offset=-1, device=features.device)
    pair_labels = torch.where((labels[rows] > 0) & (labels[cols] > 0), pair_labels, 0)
    edges = one_hot(pair_labels, states.shape[-1]).to(features.dtype)
    adjacency = features.new_zeros((n, n))
    adjacency[rows, cols] = (pair_labels > 0).to(features.dtype)
    return one_hot(labels, features.shape[-1]).to(features.dtype), adjacency+adjacency.T, edges


def apply_states(parent, rule, mapping: Mapping[int, int], feature, states):
    from src.baselines.globalgce_bace_native_rules import _edge_position
    inverse = {int(v): int(k) for k, v in mapping.items()}
    if len(inverse) != len(mapping) or set(inverse) != set(rule.lhs_nodes):
        raise ValueError("LHS mapping is not bijective")
    pn = len(parent.feature)
    if any(k < 0 or k >= pn for k in mapping):
        raise ValueError("mapping outside parent")
    next_node, mask = pn, []
    for index in range(rule.maximum_nodes):
        if index in inverse:
            mask.append(inverse[index])
        else:
            mask.append(next_node)
            next_node += 1
    f = feature.new_zeros((next_node, feature.shape[-1])); f[:, 0] = 1
    f[:pn] = parent.feature.to(feature.device)
    p = states.new_zeros((next_node*(next_node-1)//2, states.shape[-1])); p[:, 0] = 1
    p[:len(parent.edge_attr)] = parent.edge_attr.to(states.device)
    f[mask] = feature
    for right in range(rule.maximum_nodes):
        for left in range(right):
            p[_edge_position(mask[left], mask[right])] = states[_edge_position(left, right)]
    return f, p, tuple(mask)


def materialize(parent, rule, mapping, feature, states):
    """Whole-parent chemistry; a disconnected local RHS may still be legal."""
    from rdkit import Chem
    from src.baselines.globalgce_bace_native_rules import _edge_position, _apply_atom_attributes
    f, p, mask = apply_states(parent, rule, mapping, feature, states)
    hf, _, he = hard_state_tensors(f, p)
    labels = hf.argmax(-1)
    active = tuple(i for i in range(len(labels)) if int(labels[i]) > 0)
    if not active:
        raise ValueError("EMPTY_COMPLETE_PRODUCT")
    index_map = {old: new for new, old in enumerate(active)}
    attr = {int(row['native_node_index']): row for row in parent.atom_attributes}
    mol = Chem.RWMol(); inherited = reset = 0
    for old in active:
        label = int(labels[old])
        if label > len(rule.atom_symbols):
            raise ValueError("UNKNOWN_ATOM")
        atom = Chem.Atom(rule.atom_symbols[label-1])
        if old in attr and int(attr[old]['atomic_num']) == atom.GetAtomicNum():
            _apply_atom_attributes(atom, attr[old]); inherited += 1
        else:
            reset += 1
        mol.AddAtom(atom)
    bonds = {'single': Chem.BondType.SINGLE, 'double': Chem.BondType.DOUBLE,
             'triple': Chem.BondType.TRIPLE, 'aromatic': Chem.BondType.AROMATIC}
    for pos, right in enumerate(active):
        for left in active[:pos]:
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
    boundary = 0
    for inside in set(mask) & set(range(len(parent.feature))):
        for outside in range(len(parent.feature)):
            if outside in mask:
                continue
            slot = _edge_position(inside, outside)
            before, after = int(parent.edge_attr[slot].argmax()), int(he[slot].argmax())
            if before != after:
                raise ValueError("BOUNDARY_ATTACHMENT_REMOVED")
            boundary += int(before > 0)
    return {'canonical_smiles': Chem.MolToSmiles(product, canonical=True, isomericSmiles=True),
            'boundary_attachment_count': boundary, 'source_attributes_inherited': inherited,
            'source_attributes_reset': reset, 'mask_order': list(mask)}


@dataclass(frozen=True)
class OriginalGlobalGCERule:
    candidate_id: str
    native_index: int
    template: Any
    raw_features: Any
    states: Any


def load_original_pool(manifest: Mapping[str, Any]) -> tuple[OriginalGlobalGCERule, ...]:
    """Load the original saved outputs, never a generator/repair checkpoint."""
    import torch
    from src.baselines.globalgce_bace_native_rules import GlobalGCENativeRule
    source = manifest['original_rules_checkpoint']
    if source['sha256'] != ORIGINAL_RULES_SHA or manifest['original_model_checkpoint']['sha256'] != ORIGINAL_MODEL_SHA:
        raise ValueError('not the fixed original source')
    if manifest['rule_count'] != 80 or manifest.get('new_repair_weights_used') is not False:
        raise ValueError('fixed pool scope')
    # These are trusted project tensor outputs with an existing hash receipt,
    # not arbitrary remote-code model checkpoints. No model is instantiated.
    saved = torch.load(source['path'], map_location='cpu', weights_only=False)
    rows = manifest['rules']
    if len(rows) != 80 or any(len(saved[key]) != 80 for key in ('feat', 'adj', 'edge_attr',
            'features_reconst', 'adj_reconst', 'edge_attrs_reconst')):
        raise ValueError('original80 tensor inventory mismatch')
    atoms, bonds = tuple(manifest['atom_symbols']), tuple(manifest['bond_names'])
    result = []
    for row in rows:
        i = row['native_rule_index']
        rule = GlobalGCENativeRule(row['candidate_id'], i, saved['feat'][i], saved['adj'][i],
            saved['edge_attr'][i], saved['feat'][i], saved['adj'][i], saved['edge_attr'][i], atoms, bonds)
        rule.validate()
        result.append(OriginalGlobalGCERule(row['candidate_id'], i, rule,
            saved['features_reconst'][i].detach().clone(),
            joint_states(saved['adj_reconst'][i], saved['edge_attrs_reconst'][i]).detach()))
    return tuple(result)


def apply_original_rule(parent_smiles: str, rule: OriginalGlobalGCERule) -> list[dict[str, Any]]:
    from src.baselines.globalgce_bace_native_rules import build_parent_native_tensors, enumerate_labeled_rule_matches
    parent = build_parent_native_tensors(parent_smiles, atom_symbols=rule.template.atom_symbols,
                                        bond_names=rule.template.bond_names)
    records = []
    for index, mapping in enumerate(enumerate_labeled_rule_matches(parent, rule.template)):
        record = {'candidate_id': rule.candidate_id, 'native_rule_index': rule.native_index,
                  'match_index': index, 'mapping': list(mapping.items()), 'valid': False,
                  'action_kind': 'native_lhs_rhs_replacement', 'failure_reason': None}
        try:
            record.update(materialize(parent, rule.template, mapping, rule.raw_features, rule.states),
                          valid=True, connected=True, sanitized=True, boundary_attachments_preserved=True)
        except ValueError as exc:
            record['failure_reason'] = str(exc)
        records.append(record)
    return records


def adoption_manifest(summary, recovery, binding, terminal, catalog, *, references):
    """Reuse completed chemistry evidence, not the old GINE flip/metrics."""
    if summary.get('test_loaded') is not False or recovery.get('test_loaded') is not False:
        raise ValueError('original generation test boundary')
    if binding.get('model_retrained') is not False or binding.get('mining_reused') is not True:
        raise ValueError('rematerialization changed model/mining')
    if terminal.get('state') != 'REMATERIALIZATION_COMPLETE' or terminal.get('execution_valid') is not True:
        raise ValueError('incomplete rematerialization is not a zero result')
    if terminal.get('test_loaded') is not False or terminal.get('weights_changed') is not False:
        raise ValueError('rematerialization scientific scope')
    model, rules = recovery['source_model_checkpoint'], recovery['source_rules_checkpoint']
    if model['sha256'] != ORIGINAL_MODEL_SHA or rules['sha256'] != ORIGINAL_RULES_SHA:
        raise ValueError('repair checkpoint cannot replace original outputs')
    if binding['source_model'] != model['path'] or binding['source_rules'] != rules['path']:
        raise ValueError('rematerialization source path drift')
    if summary['rules_checkpoint'] != rules['path'] or summary['globalgce_model_checkpoint'] != model['path']:
        raise ValueError('training summary source drift')
    if binding['rule_count'] != 80 or len(catalog) != 80:
        raise ValueError('must preserve original80')
    indexes = [int(row['rule']['native_rule_index']) for row in catalog]
    ids = [row['candidate_id'] for row in catalog]
    if sorted(indexes) != list(range(80)) or len(set(ids)) != 80:
        raise ValueError('candidate inventory missing/duplicate')
    if any(row.get('source_split') != 'train' for row in catalog):
        raise ValueError('candidate is not a frozen train proposal')
    if len(binding['train_ids']) != 360 or len(binding['validation_ids']) != 98:
        raise ValueError('original train/validation scope changed')
    funnels = {}
    for split in ('train', 'validation'):
        index, result = terminal[split+'_index'], terminal[split]
        parent_ids = binding['train_ids' if split == 'train' else 'validation_ids']
        if len(set(parent_ids)) != len(parent_ids) or index['parents'] != len(parent_ids):
            raise ValueError('materialization parent binding')
        if index['pairs'] != len(parent_ids)*80 or result['rules'] != 80:
            raise ValueError('materialization pair coverage')
        if index['matches'] != result['counts']['mappings'] or result.get('test_loaded') is not False:
            raise ValueError('materialization mapping coverage/test boundary')
        if result.get('calibration_loaded') is not False:
            raise ValueError('materialization used calibration')
        counts = result['counts']
        failures = {k: v for k, v in counts.items() if k.isupper()}
        if sum(failures.values()) + counts.get('valid', 0) != counts['mappings']:
            raise ValueError('materialization rejection funnel does not close')
        funnels[split] = {'parent_count': len(parent_ids), 'parent_ids_sha256': stable_sha256(parent_ids),
            'pair_count': index['pairs'], 'matched_pairs': index['matched_pairs'],
            'mapping_count': counts['mappings'], 'valid_complete_products': counts.get('valid', 0),
            'first_failure_counts': failures, 'old_oracle_flip_counts_adopted': False}
    if set(binding['train_ids']) & set(binding['validation_ids']):
        raise ValueError('train/validation overlap')
    all_invalid = all(v['valid_complete_products'] == 0 and v['mapping_count'] > 0 for v in funnels.values())
    vocab = summary['codec_metadata']
    out = {'schema_version': 'bace_gin_original_globalgce_pool_v1', 'method': 'GlobalGCE',
        'method_variant': 'OriginalPoolJointHardMaterializer', 'rule_count': 80,
        'state': 'BLOCKED_MATERIALIZATION' if all_invalid else 'READY_FOR_FROZEN_GIN_CALIBRATION',
        'original_model_checkpoint': model, 'original_rules_checkpoint': rules,
        'original_catalog': recovery['candidate_universe'],
        'mining_identity': summary['gspan_exact_top_k_proof']['selected_identity_sha256'],
        'atom_symbols': [vocab['node_label_mapping'][str(i)] for i in range(1, len(vocab['node_label_mapping']))],
        'bond_names': [vocab['edge_label_mapping'][str(i)] for i in range(len(vocab['edge_label_mapping']))],
        'rules': [{'candidate_id': row['candidate_id'], 'native_rule_index': i,
                   'original_rule_content_hash': row['rule_content_hash']} for row, i in zip(catalog, indexes)],
        'materializer_source_commit': MATERIALIZER_SOURCE_COMMIT, 'joint_contract': JOINT_CONTRACT,
        'new_repair_weights_used': False, 'generation_rerun': False, 'generator_refitted': False,
        'new_oracle_inference_performed': False, 'test_read': False, 'evaluation_rows_fabricated': False,
        'zero_coverage_claimed': False, 'scientific_metrics': None, 'train_validation_funnels': funnels,
        'evidence_scope': 'original360_train_and98_validation_labels1_all_LHS_mappings_independent_of_oracle_prediction',
        'evidence_references': references, 'created_at': utc_now()}
    out['manifest_sha256'] = stable_sha256(out)
    return out


def seal_manifest(rematerialization_root: str, output_root: str):
    root, out = Path(rematerialization_root), Path(output_root)
    if not root.is_absolute() or not out.is_absolute():
        raise ValueError('absolute roots required')
    references = {}
    def read(name, path):
        raw = path.read_bytes()
        references[name] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        return json.loads(raw)
    binding = read('input_binding', root/'input_binding.json')
    terminal = read('rematerialization_terminal', root/'terminal.json')
    contract = read('rematerialization_contract', root/'repair_contract.json')
    if any(contract['joint_contract'].get(k) != v for k, v in JOINT_CONTRACT.items()):
        raise ValueError('materialization contract drift')
    summary_path = Path(binding['training_summary'])
    summary = read('training_summary', summary_path)
    recovery = read('recovery_receipt', summary_path.parent/'recovery_receipt.json')
    raw = Path(recovery['candidate_universe']['path']).read_bytes()
    if len(raw) != recovery['candidate_universe']['size'] or hashlib.sha256(raw).hexdigest() != recovery['candidate_universe']['sha256']:
        raise ValueError('small original catalog hash binding')
    catalog = [json.loads(line) for line in raw.splitlines() if line]
    manifest = adoption_manifest(summary, recovery, binding, terminal, catalog, references=references)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'original_pool_manifest.json', manifest)
    return manifest
