"""BACE Reach-v2 reuse of *raw graph-pair costs*, not classifier decisions.

The old action cache stays untouched. A one-time, split-scoped migration reads
only manifest-bound completed parent records. It retains the original action
provenance and explicitly proves that the unchanged MolCLR/OT implementation
consumes canonical graph contents, not a backbone, pool, temperature or flip.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from src.eval.bace_frozen_gnn_contracts import atomic_json, read_json, stable_sha256

SCHEMA = 'bace_reach_v2_raw_graph_cost_adoption_v1'
KERNELS = (
    'src/eval/node_wasserstein_distance.py',
    'src/eval/molclr_node_embeddings.py',
    'src/embeddings/molclr_gnn_embedding.py',
    'src/data/molecular_graph_featurizer.py',
)


def raw_contract_from_bundle(manifest):
    return dict(wnode=manifest['wnode_config'],
        feature_schema=manifest['files'][manifest['feature_schema_path']],
        molclr_checkpoint=manifest['files'][manifest['molclr_checkpoint_path']],
        molclr_source={r: i for r, i in manifest['files'].items()
                       if r.startswith(manifest['molclr_source_root'] + '/')})


def kernel_identity_proof(repo: Path, source_commit: str):
    if not re.fullmatch(r'[a-f0-9]{40}', source_commit):
        raise ValueError('RAW_COST_SOURCE_COMMIT_NOT_PINNED')
    proof = {}
    for relative in KERNELS:
        current = (repo / relative).read_bytes()
        old = subprocess.check_output(['git', 'show', source_commit + ':' + relative], cwd=repo)
        if current != old:
            raise ValueError('RAW_COST_IMPLEMENTATION_DRIFT:' + relative)
        proof[relative] = hashlib.sha256(current).hexdigest()
    return proof


def _bound_json(path: Path, expected: str):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('RAW_COST_SOURCE_BINDING_CONFLICT:' + str(path))
    return json.loads(data)


def graph_key(parent, residual, contract_sha):
    from src.eval.molclr_node_embeddings import canonicalize_smiles
    p, r = canonicalize_smiles(parent), canonicalize_smiles(residual)
    if p is None or r is None or '.' in r:
        raise ValueError('RAW_COST_GRAPH_CONTENT_INVALID')
    # Direction is deliberately retained; no floating-point symmetry assumption.
    return stable_sha256(dict(parent=p, residual=r, raw_contract_sha256=contract_sha)), p, r


def validate_ours_final_freeze(freeze, evidence_root: Path):
    """Validate the actual three-control Ours freeze, not a pretend GNN freeze.

    These small sealed receipts were copied after calibration selection closed.
    No source test records or test metrics are consulted by this validator.
    """
    from src.eval.bace_reach_v2 import unseal
    def valid_seal(value):
        return value.get('self_sha256') == stable_sha256(
            {k: v for k, v in value.items() if k != 'self_sha256'})
    if (not valid_seal(freeze)
            or freeze.get('state') != 'REACH_V2_FINAL_CONFIGURATION_FROZEN'
            or freeze.get('selected_control') != 'new_pool_reach_first'
            or freeze.get('selected_using_test') is not False
            or freeze.get('test_opened') is not False
            or freeze.get('test_campaigns_max') != 1
            or freeze.get('main_matrix_write') is not False
            or freeze.get('claim_new_untouched_test') is not False):
        raise ValueError('OURS_ACTUAL_FINAL_FREEZE_REQUIRED')
    docs = {}
    for name, filename in (
            ('search_contract', 'search_contract.json'),
            ('candidate_freeze', 'candidate_freeze.json'),
            ('selector_freeze', 'selector_freeze.json'),
            ('train_gate', 'train_reach_gate.json')):
        docs[name] = unseal(evidence_root / filename)
        if docs[name]['self_sha256'] != freeze[name + '_sha256']:
            raise ValueError('OURS_FINAL_FREEZE_DEPENDENCY_CHANGED:' + name)
    selector, gate, pool = (docs[k] for k in ('selector_freeze', 'train_gate', 'candidate_freeze'))
    if (selector.get('test_opened') is not False
            or gate.get('state') != 'NO_ADDITIONAL_PPO_REQUIRED_BY_TRAIN_GATE'
            or selector.get('candidate_freeze_sha256') != pool['self_sha256']
            or gate.get('candidate_freeze_sha256') != pool['self_sha256']
            or pool.get('search_contract_sha256') != docs['search_contract']['self_sha256']):
        raise ValueError('OURS_FINAL_FREEZE_STAGE_ORDER_CONFLICT')
    expected = dict(old_pool_old_selector=selector['controls']['old_pool_old_selector'],
        new_pool_old_selector=selector['controls']['new_pool_old_selector'],
        new_pool_reach_first=selector['reach_first']['ordered_rule_ids'])
    if freeze.get('controls') != expected or any(
            len(ids) != 20 or len(set(ids)) != 20 for ids in expected.values()):
        raise ValueError('OURS_FINAL_THREE_CONTROL_BINDING_CONFLICT')


def build_index(source_spec, *, split, output: Path, repo: Path,
                test_freeze_path=None, test_freeze_sha=None, validate_test_freeze=None):
    """No model/OT inference. Read source parent members once, then seal a map.

    A test migration is permitted only after the caller's actual new global
    freeze validator reads its hash-bound receipt; a boolean is not accepted.
    """
    if split not in ('calibration', 'test'):
        raise ValueError('RAW_COST_SPLIT_UNSUPPORTED')
    freeze_sha = None
    if split == 'test':
        if not callable(validate_test_freeze) or not test_freeze_path or not test_freeze_sha:
            raise ValueError('RAW_COST_TEST_BEFORE_NEW_GLOBAL_FREEZE')
        freeze = _bound_json(Path(test_freeze_path), test_freeze_sha)
        validate_test_freeze(freeze)
        freeze_sha = test_freeze_sha
    binding = stable_sha256(dict(source=source_spec, split=split, new_test_freeze_sha256=freeze_sha))
    if output.exists():
        saved = read_json(output)
        if saved.get('binding_sha256') != binding or saved.get('self_sha256') != stable_sha256(
                {k: v for k, v in saved.items() if k != 'self_sha256'}):
            raise ValueError('SEALED_RAW_COST_INDEX_CONFLICT')
        return saved
    source = Path(source_spec['correction_root']).resolve(strict=True)
    destination = output.resolve()
    if destination == source or source in destination.parents:
        raise ValueError('RAW_COST_INDEX_CANNOT_MODIFY_SEALED_SOURCE')
    names = ('acceptance', 'original_inventory', 'repair_contract', 'raw_distance_contract')
    docs = {}
    for name in names:
        item = source_spec[name]
        rel = Path(item['relative_path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('RAW_COST_UNSAFE_SOURCE_MEMBER')
        path = (source / rel).resolve(strict=True)
        path.relative_to(source)
        docs[name] = _bound_json(path, item['sha256'])
    acceptance, inventory, repair, contract = (docs[n] for n in names)
    if (acceptance.get('state') != 'GNN_CORE_SEED7_CORRECTED_PASS'
            or acceptance.get('cache_provenance_gaps') != []
            or acceptance.get('raw_ot_recomputed_count') != 0
            or acceptance.get('main_matrix_write') is not False
            or inventory.get('state') != 'PASS'
            or repair.get('original_package_sha256') != acceptance.get('original_package_sha256')
            or repair.get('original_exact_driver_commit') != inventory.get('publication_driver_commit')):
        raise ValueError('RAW_COST_COMPLETED_SOURCE_ACCEPTANCE_CONFLICT')
    proof = kernel_identity_proof(repo, repair['original_exact_driver_commit'])
    if contract['wnode'].get('solver') != 'exact_emd2':
        raise ValueError('RAW_COST_NOT_EXACT_FROZEN_CONTRACT')
    contract_sha = stable_sha256(contract)
    evaluation = Path(repair['source_spec']['evaluation_root']).resolve(strict=True)
    files = inventory['files']
    pattern = re.compile(r'evaluation/(gine|gin|gcn|gatv2|gatedgcn_plus)/' + split + r'/parents/[a-f0-9]{64}\.json')
    members = [rel for rel in files if pattern.fullmatch(rel)]
    expected_units = repair['required_counts'][split + '_parent_units']
    if len(members) != expected_units:
        raise ValueError('RAW_COST_SOURCE_PARTITION_INCOMPLETE')
    values, finite_count = {}, 0
    for rel in sorted(members):
        path = (evaluation / rel.removeprefix('evaluation/')).resolve(strict=True)
        path.relative_to(evaluation)
        saved = _bound_json(path, files[rel]['sha256'])
        science = saved['science']
        if saved.get('science_sha256') != stable_sha256(science):
            raise ValueError('RAW_COST_SOURCE_PARENT_SCIENCE_CONFLICT')
        for row in science['match_rows']:
            if row.get('distance_ok') is not True:
                continue
            value = row.get('wnode_distance')
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or value < 0
                    or row.get('delete_valid') is not True or row.get('sanitize_ok') is not True
                    or row.get('residual_connected') is not True):
                raise ValueError('RAW_COST_INVALID_FINITE_SOURCE_RECORD')
            key, parent, residual = graph_key(row['parent_smiles'], row['residual_smiles'], contract_sha)
            provenance = dict(source_parent_member=rel, source_parent_sha256=files[rel]['sha256'],
                source_match_sha256=stable_sha256(row), original_action_context={k: row[k] for k in
                    ('parent_id', 'candidate_id', 'match_index', 'match_atom_indices',
                     'oracle_checkpoint_hash', 'action_semantics_version')})
            if key in values and values[key]['distance'] != value:
                raise ValueError('RAW_GRAPH_COST_SOURCE_CONFLICT:' + key)
            if key not in values:
                values[key] = dict(parent=parent, residual=residual, distance=value, source_records=[])
            values[key]['source_records'].append(provenance)
            finite_count += 1
    result = dict(schema=SCHEMA, state='RAW_COST_ADOPTION_INDEX_SEALED_NOT_SCIENCE_PASS',
        binding_sha256=binding, source_spec=source_spec, split=split,
        source_parent_units=len(members), source_finite_match_records=finite_count,
        raw_contract=contract, raw_contract_sha256=contract_sha, kernel_identity=proof,
        new_test_freeze_sha256=freeze_sha, graph_costs=values,
        raw_cost_count=len(values), old_cache_keys_modified=False,
        source_flip_masks_reused=False, source_selected_match_minima_reused=False,
        model_inference_performed=False, ot_recomputed=0)
    result['self_sha256'] = stable_sha256(result)
    atomic_json(output, result)
    return result


class VerifiedRawGraphDistance:
    """The wrapper returns raw costs; the caller recomputes its own flips/minima."""
    def __init__(self, delegate, *, index, current_raw_contract, repo: Path):
        if (index.get('schema') != SCHEMA or index.get('self_sha256') != stable_sha256(
                {k: v for k, v in index.items() if k != 'self_sha256'})
                or index['raw_contract'] != current_raw_contract):
            raise ValueError('RAW_COST_NEW_CONTRACT_OR_INDEX_CONFLICT')
        for rel, digest in index['kernel_identity'].items():
            if hashlib.sha256((repo / rel).read_bytes()).hexdigest() != digest:
                raise ValueError('RAW_COST_CONSUMER_IMPLEMENTATION_DRIFT:' + rel)
        if set(index['kernel_identity']) != set(KERNELS):
            raise ValueError('RAW_COST_INDEPENDENCE_PROOF_INCOMPLETE')
        frozen = current_raw_contract['wnode']
        for key in ('feature_cost', 'node_mass', 'size_penalty_beta', 'distance_namespace'):
            if getattr(delegate.config, key) != frozen[key]:
                raise ValueError('RAW_COST_DELEGATE_NUMERICAL_CONTRACT_DRIFT:' + key)
        self.delegate, self.index = delegate, index
        self.used, self.fresh, self.local = [], 0, {}

    def distance_for_action(self, parent, residual, *, action_context):
        required = ('candidate_id', 'match_atom_indices', 'teacher_sha256',
                    'action_semantics_version', 'match_selection_policy', 'distance_implementation_version')
        if any(k not in action_context for k in required):
            raise ValueError('RAW_COST_CURRENT_ACTION_CONTEXT_INCOMPLETE')
        key, _, _ = graph_key(parent, residual, self.index['raw_contract_sha256'])
        record = self.index['graph_costs'].get(key)
        if record is not None:
            self.used.append(dict(raw_graph_key=key, source_index_sha256=self.index['self_sha256'],
                current_action_context=action_context, source_records=record['source_records']))
            return dict(ok=True, distance=record['distance'], cache_hit=True, error=None,
                        metadata={'reuse': 'EXPLICIT_GRAPH_CONTENT_RAW_OT_ADOPTION'})
        if key in self.local and self.local[key].get('ok'):
            return dict(self.local[key], cache_hit=True)
        if key not in self.local:
            # The unchanged delegate has an explicit graph-pair API and key.
            # This is a new raw-layer request, not deletion of an old key field.
            self.local[key] = self.delegate.distance(parent, residual)
            self.fresh += not self.local[key].get('cache_hit', False)
        return self.local[key]

    def stats_dict(self):
        result = self.delegate.stats_dict()
        result.update(raw_graph_costs_adopted=len(self.used), new_raw_graph_requests=self.fresh,
            raw_graph_source_index_sha256=self.index['self_sha256'], backbone_flip_masks_adopted=False)
        return result

    def close(self):
        self.delegate.close()
