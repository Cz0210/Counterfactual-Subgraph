"""BACE-only train/validation rematerialization and one bounded repair campaign."""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from rdkit import RDLogger

from src.baselines.bace_globalgce_chemaligned import (
    CONTRACT, SCHEMA, ChemAlignedBridge, corrected_rule, joint_states, materialize,
)
from src.baselines.globalgce_bace_native_rules import (
    GlobalGCENativeRule, build_parent_native_tensors, enumerate_labeled_rule_matches,
    validate_official_globalgce_root,
)
from src.baselines.globalgce_mutagenicity_adapter import _import_official_modules
from src.eval.bace_frozen_gnn_contracts import atomic_json, stable_sha256, utc_now


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    rows = list(csv.DictReader(Path(path).open()))
    out = []
    for row in rows:
        pid = next((row[x] for x in ("sample_id", "molecule_id", "id", "compound_id") if row.get(x)), None)
        smiles = next((row[x] for x in ("model_smiles", "canonical_smiles", "smiles", "SMILES") if row.get(x)), None)
        label = next((row[x] for x in ("label", "target", "Label") if x in row), None)
        if pid is None or smiles is None or label is None:
            raise ValueError(f"train/validation CSV schema missing: {list(row)}")
        out.append({"id": pid, "smiles": smiles, "label": int(label)})
    if len({r['id'] for r in out}) != len(out):
        raise ValueError("duplicate split IDs")
    return out


def validate_config(config):
    if config['dataset'] != 'BACE' or config['seed'] != 7 or config['max_epochs'] != 100:
        raise ValueError("repair dataset/seed/budget contract")
    if config['source_label'] != 1 or config['target_label'] != 0:
        raise ValueError("BACE class mapping")
    if config['fresh_campaign_max'] != 1 or config['test_access'] is not False:
        raise ValueError("campaign/test scope")
    for name in ('source_manifest', 'training_summary', 'train_csv', 'validation_csv', 'gnn_checkpoint', 'official_root'):
        if not Path(config[name]).is_absolute():
            raise ValueError(f"absolute input required: {name}")
    if Path(config['train_csv']).name != 'train.csv' or Path(config['validation_csv']).name != 'val.csv':
        raise ValueError("fitter only accepts the bound train.csv and val.csv")
    if config['joint_contract'] != CONTRACT:
        raise ValueError("unfrozen molecular adapter contract")


def setup(config, device):
    validate_config(config)
    summary = read_json(config['training_summary'])
    manifest = read_json(config['source_manifest'])
    if manifest.get('test_loaded') is not False or summary.get('test_loaded') is not False:
        raise ValueError("source generation train-only evidence missing")
    if manifest['oracle_checkpoint'] != config['gnn_checkpoint']:
        raise ValueError("frozen GINE changed")
    vocab = summary['codec_metadata']
    atoms = tuple(vocab['node_label_mapping'][str(i)] for i in range(1, len(vocab['node_label_mapping'])))
    bonds = tuple(vocab['edge_label_mapping'][str(i)] for i in range(len(vocab['edge_label_mapping'])))
    rules = torch.load(summary['rules_checkpoint'], map_location='cpu', weights_only=False)
    templates = []
    for i in range(len(rules['feat'])):
        templates.append(GlobalGCENativeRule(
            f'chemaligned-rule-{i:04d}', i, rules['feat'][i].detach().cpu(),
            rules['adj'][i].detach().cpu(), rules['edge_attr'][i].detach().cpu(),
            rules['feat'][i].detach().cpu(), rules['adj'][i].detach().cpu(),
            rules['edge_attr'][i].detach().cpu(), atoms, bonds))
        templates[-1].validate()
    train_all, validation = read_rows(config['train_csv']), read_rows(config['validation_csv'])
    if set(r['id'] for r in train_all) & set(r['id'] for r in validation):
        raise ValueError("train/validation overlap")
    by_id = {r['id']: r for r in train_all}
    ids = manifest['source_parent_ids']
    train = [by_id[i] for i in ids]
    if len(train) != 360 or any(r['label'] != 1 for r in train):
        raise ValueError("original source train cohort drift")
    validation = [r for r in validation if r['label'] == 1]
    bridge = ChemAlignedBridge.from_checkpoint(config['gnn_checkpoint'], atom_symbols=atoms,
                                                bond_names=bonds, device=device)
    return summary, rules, templates, train, validation, bridge


def predict_smiles(bridge, smiles):
    f = bridge.feature_schema
    from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer
    graph = MolecularGraphFeaturizer(f).featurize(smiles)
    with torch.no_grad():
        logits = bridge.model(x=torch.tensor(graph.node_features, dtype=torch.long, device=bridge.device),
                              edge_index=torch.tensor(graph.edge_index, dtype=torch.long, device=bridge.device),
                              edge_attr=torch.tensor(graph.edge_features, dtype=torch.long, device=bridge.device))
        probs = (logits/bridge.temperature).softmax(-1)[0].cpu().tolist()
    return {'predicted_label': int(np.argmax(probs)), 'probabilities': probs}


def build_index(rows, templates, bridge, output, role):
    result = []
    counts = collections.Counter()
    with (output/f'{role}_index.jsonl').open('x') as stream:
        for parent_pos, row in enumerate(rows):
            parent = build_parent_native_tensors(row['smiles'], atom_symbols=templates[0].atom_symbols,
                                                bond_names=templates[0].bond_names)
            pred = predict_smiles(bridge, row['smiles'])
            counts['parents'] += 1
            counts['source_eligible'] += int(pred['predicted_label'] == 1)
            for rule_pos, template in enumerate(templates):
                maps = enumerate_labeled_rule_matches(parent, template)
                counts['pairs'] += 1; counts['matched_pairs'] += bool(maps)
                for mapping in maps:
                    item = {'parent_position': parent_pos, 'parent_id': row['id'], 'rule_index': rule_pos,
                            'mapping': list(mapping.items()), 'before': pred}
                    stream.write(json.dumps(item, separators=(',', ':'))+'\n')
                    result.append((row, parent, template, mapping, pred))
            atomic_json(output/'heartbeat.json', {'time': utc_now(), 'pid': os.getpid(), 'stage': 'BUILD_'+role.upper()+'_INDEX',
                                                'parents': parent_pos+1, 'matches': len(result)})
    counts['matches'] = len(result)
    return result, dict(counts)


def rematerialize(index, rules, bridge, output, role):
    counts = collections.Counter(); witnesses = []; valid_ids = set(); flip_ids = set()
    unique = set(); start = time.monotonic()
    states = [joint_states(a, e) for a, e in zip(rules['adj_reconst'], rules['edge_attrs_reconst'])]
    with (output/f'{role}_applications.jsonl').open('x') as stream:
        for i, (row, parent, template, mapping, before) in enumerate(index):
            rid = template.native_rule_index
            record = {'parent_id': row['id'], 'rule_index': rid, 'mapping': list(mapping.items()), 'valid': False,
                      'before': before, 'split': role, 'strict_flip': False}
            counts['mappings'] += 1
            try:
                product = materialize(parent, template, mapping, rules['features_reconst'][rid], states[rid])
                prediction = predict_smiles(bridge, product.canonical_smiles)
                flip = before['predicted_label'] == 1 and prediction['predicted_label'] == 0
                record.update(valid=True, after=prediction, strict_flip=flip, canonical_smiles=product.canonical_smiles,
                              boundary_attachment_count=product.boundary_count,
                              inherited_atom_attributes=product.attributes_inherited)
                counts['valid'] += 1; counts['strict_flip'] += int(flip)
                valid_ids.add(row['id']); unique.add(product.canonical_smiles)
                if flip:
                    flip_ids.add(row['id'])
                    if len(witnesses) < 20: witnesses.append(record)
            except ValueError as exc:
                counts[str(exc)] += 1; record['first_failure'] = str(exc)
            stream.write(json.dumps(record, separators=(',', ':'))+'\n')
            if (i+1) % 100 == 0:
                stream.flush()
                atomic_json(output/'heartbeat.json', {'time': utc_now(), 'pid': os.getpid(), 'stage': role.upper()+'_REMATERIALIZATION',
                                                    'completed': i+1, 'total': len(index), 'counts': dict(counts)})
    counts.update(parents_with_valid=len(valid_ids), parents_with_flip=len(flip_ids), unique_products=len(unique))
    result = {'counts': dict(counts), 'elapsed_seconds': time.monotonic()-start, 'witnesses': witnesses,
              'rules': len(states), 'role': role, 'test_loaded': False, 'calibration_loaded': False}
    atomic_json(output/f'{role}_rematerialization.json', result)
    return result


def run_rematerialization(config_path, output_root):
    config = read_json(config_path); output = Path(output_root); output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'repair_contract.json', config)
    atomic_json(output/'owner.json', {'pid': os.getpid(), 'created_at': utc_now(), 'config_sha': stable_sha256(config),
                                    'cpu_only': True, 'test_access': False})
    summary, rules, templates, train, val, bridge = setup(config, 'cpu')
    atomic_json(output/'input_binding.json', {'training_summary': config['training_summary'],
               'source_model': summary['globalgce_model_checkpoint'], 'source_rules': summary['rules_checkpoint'],
               'train_ids': [r['id'] for r in train], 'validation_ids': [r['id'] for r in val],
               'rule_count': len(templates), 'mining_reused': True, 'model_retrained': False,
               'original_source_checkpoint_sha': read_json(Path(config['training_summary']).parent/'recovery_receipt.json')['source_model_checkpoint']['sha256']})
    train_index, train_counts = build_index(train, templates, bridge, output, 'train')
    train_result = rematerialize(train_index, rules, bridge, output, 'train')
    del train_index
    val_index, val_counts = build_index(val, templates, bridge, output, 'validation')
    val_result = rematerialize(val_index, rules, bridge, output, 'validation')
    result = {'state': 'REMATERIALIZATION_COMPLETE', 'train_index': train_counts, 'validation_index': val_counts,
              'train': train_result, 'validation': val_result, 'repair_training_required': train_result['counts'].get('strict_flip', 0) == 0,
              'execution_valid': True, 'performance_target_met': train_result['counts'].get('strict_flip', 0) > 0,
              'test_loaded': False, 'weights_changed': False, 'completed_at': utc_now()}
    atomic_json(output/'terminal.json', result)
    return result
