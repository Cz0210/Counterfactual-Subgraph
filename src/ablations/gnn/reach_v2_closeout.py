"""Held-out closeout for the existing Reach-v2 GNN parent-chunk evaluator.

No training, fitting, inference, OT or scheduling. Adopt only the unchanged
classifier report; explanation rows are reconstructed from this new pool's own
parent/match checkpoints and the ten pre-test global selectors.
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tarfile

from src.ablations.gnn.reach_v2_adapter import BACKBONES, SCOPE_NAME, require_global_freeze, split_chunk_size
from src.eval.bace_frozen_gnn_contracts import atomic_csv, atomic_json, read_json, sha256_file, stable_sha256

AUDIT_NAME = 'gnn_reach_v2_final_audit.json'
ACCEPTANCE_NAME = 'gnn_reach_v2_independent_acceptance.json'


def _bound(path, expected):
    path = Path(path)
    if not expected or path.is_symlink() or sha256_file(path) != expected:
        raise ValueError('V2_SMALL_ARTIFACT_BINDING_CONFLICT:' + str(path))
    return read_json(path)


def seal_test_dependencies(spec, spec_sha, pool_sha):
    """Freeze/index hashes are outputs, not guessed fields in the original spec."""
    root = Path(spec['output_root'])
    freeze_path = Path(spec['global_selector_freeze'])
    if freeze_path != root / 'CALIBRATION_FREEZE.json':
        raise ValueError('V2_TEST_FREEZE_MUST_BELONG_TO_CURRENT_ROOT')
    freeze = read_json(freeze_path)
    require_global_freeze(freeze, pool_sha)
    if freeze.get('spec_sha256') != spec_sha:
        raise ValueError('V2_TEST_FREEZE_SPEC_CONFLICT')
    index_path = Path(spec['raw_cost_indexes']['test']['path'])
    index = read_json(index_path)
    if (index.get('split') != 'test' or index.get('new_test_freeze_sha256') != sha256_file(freeze_path)
            or index.get('self_sha256') != stable_sha256({k:v for k,v in index.items() if k != 'self_sha256'})):
        raise ValueError('V2_TEST_INDEX_FREEZE_BINDING_CONFLICT')
    receipt = dict(scope=SCOPE_NAME, spec_sha256=spec_sha, pool_sha256=pool_sha,
        global_freeze_sha256=sha256_file(freeze_path), test_index_path=str(index_path),
        test_index_sha256=sha256_file(index_path), test_inference_started=False,
        main_matrix_write=False)
    receipt['self_sha256'] = stable_sha256(receipt)
    path = root / 'post_freeze_dependencies.json'
    if path.exists() and read_json(path) != receipt:
        raise ValueError('V2_POST_FREEZE_DEPENDENCIES_IMMUTABLE')
    if not path.exists(): atomic_json(path, receipt)
    return receipt


def require_test_dependencies(spec, spec_sha, pool_sha):
    root = Path(spec['output_root'])
    value = read_json(root / 'post_freeze_dependencies.json')
    if (value.get('scope') != SCOPE_NAME or value.get('spec_sha256') != spec_sha
            or value.get('pool_sha256') != pool_sha or value.get('main_matrix_write') is not False
            or value.get('self_sha256') != stable_sha256({k:v for k,v in value.items() if k != 'self_sha256'})
            or value.get('test_index_path') != spec['raw_cost_indexes']['test']['path']):
        raise ValueError('V2_POST_FREEZE_DEPENDENCIES_CONFLICT')
    frozen = _bound(spec['global_selector_freeze'], value['global_freeze_sha256'])
    require_global_freeze(frozen, pool_sha)
    if frozen.get('spec_sha256') != spec_sha:
        raise ValueError('V2_TEST_FREEZE_SPEC_CONFLICT')
    _bound(value['test_index_path'], value['test_index_sha256'])
    return value, frozen


def verify_own_match_minima(science, *, candidate_ids, model_sha):
    """Replay scalar flip/minimum aggregation; never infer a missing match."""
    pairs, matches = science['pair_rows'], science['match_rows']
    if len(pairs) != len(candidate_ids) or {p['candidate_id'] for p in pairs} != set(candidate_ids):
        raise ValueError('V2_PARENT_CANDIDATE_UNIVERSE_INCOMPLETE')
    grouped, seen = {c: [] for c in candidate_ids}, set()
    parent_ids = {p['parent_id'] for p in pairs}
    if len(parent_ids) != 1:
        raise ValueError('V2_ONE_SCIENTIFIC_PARENT_REQUIRED')
    for row in matches:
        key = row['candidate_id'], row['match_index']
        if (key in seen or key[0] not in grouped or row['parent_id'] not in parent_ids
                or row.get('oracle_checkpoint_hash') != model_sha):
            raise ValueError('V2_MATCH_IDENTITY_OR_BACKBONE_CONFLICT')
        seen.add(key)
        valid = bool(row['delete_valid'] and row.get('residual_smiles'))
        strict = valid and row['pred_before'] == 1 and row['pred_after'] == 0
        if bool(row['cf_flip']) != strict or bool(row['teacher_strict_flip']) != strict:
            raise ValueError('V2_OWN_STRICT_FLIP_CONFLICT')
        if valid:
            for side in ('before', 'after'):
                probs = row['p_' + side]
                if (len(probs) != 2 or any(not math.isfinite(float(p)) or not 0 <= float(p) <= 1 for p in probs)
                        or int(max(range(2), key=lambda i: probs[i])) != row['pred_' + side]):
                    raise ValueError('V2_OWN_PREDICTION_PROBABILITY_CONFLICT')
            if float(row['cf_drop']) != float(row['p_before'][1] - row['p_after'][1]):
                raise ValueError('V2_OWN_CFDROP_CONFLICT')
        if strict and (not row['distance_ok'] or not math.isfinite(float(row['wnode_distance']))
                       or float(row['wnode_distance']) < 0):
            raise ValueError('V2_OWN_STRICT_FLIP_RAW_DISTANCE_GAP')
        if row['distance_ok'] and not strict:
            raise ValueError('V2_NONFLIP_DISTANCE_USED_AS_RECOURSE')
        grouped[key[0]].append(row)
    for pair in pairs:
        rows = grouped[pair['candidate_id']]
        if (pair.get('oracle_checkpoint_hash') != model_sha or pair['num_matches'] != len(rows)
                or pair['num_valid_residuals'] != sum(bool(r['delete_valid']) for r in rows)
                or bool(pair['applicable']) != bool(rows)
                or pair['num_strict_flip_matches'] != sum(bool(r['cf_flip']) for r in rows)):
            raise ValueError('V2_PAIR_MATCH_FUNNEL_CONFLICT')
        eligible = sorted((r for r in rows if r['teacher_strict_flip'] and r['distance_ok']),
            key=lambda r: (float(r['wnode_distance']), -float(r['cf_drop']), tuple(r['match_atom_indices'])))
        best = eligible[0] if eligible else None
        if bool(pair['pair_strict_flip']) != bool(best):
            raise ValueError('V2_PAIR_OWN_FLIP_AGGREGATION_CONFLICT')
        for field, match_field, missing in (
            ('best_match_index', 'match_index', None), ('best_match_atom_indices', 'match_atom_indices', []),
            ('wnode_distance', 'wnode_distance', None), ('cf_drop', 'cf_drop', None),
            ('residual_smiles', 'residual_smiles', None), ('pred_after', 'pred_after', None)):
            if pair.get(field) != (best[match_field] if best else missing):
                raise ValueError('V2_PAIR_OWN_MATCH_MINIMUM_CONFLICT:' + field)
    return len(matches)


def collect_test_chunks(spec, *, spec_sha, pool_sha, candidates):
    from src.ablations.gnn.cpu_evaluation import matrix_from_pairs
    from src.ablations.gnn.sharded_evaluation import stable_partition
    root = Path(spec['output_root'])
    dependencies, frozen = require_test_dependencies(spec, spec_sha, pool_sha)
    collected, files = {}, {}
    for name in BACKBONES:
        selected_ids = set().union(*(set(frozen['selectors'][f'{name}/{m}']['ordered_rule_ids'])
                                    for m in ('native', 'common')))
        selected = [c for c in candidates if c['candidate_id'] in selected_ids]
        if {c['candidate_id'] for c in selected} != selected_ids:
            raise ValueError('V2_SELECTED_CANDIDATE_MISSING')
        native, seen, all_rows, match_count = None, set(), [], 0
        for index in range(spec['slots']['test']):
            directory = root / name / 'test' / f'{index:04d}'
            terminal_path = directory / 'terminal.json'
            terminal = read_json(terminal_path)
            expected = dict(state='PARENT_CHUNK_COMPLETE_NOT_CORE_PASS', scope=SCOPE_NAME,
                spec_sha256=spec_sha, pool_sha256=pool_sha, backbone=name, split='test', index=index,
                global_selector_called=False, main_matrix_write=False,
                model_files=spec['model_files'][name],
                global_freeze_sha256=dependencies['global_freeze_sha256'])
            if type(terminal.get('index')) is not int or any(terminal.get(k) != v for k,v in expected.items()):
                raise ValueError('V2_TEST_SHARD_TERMINAL_CONFLICT')
            declared = terminal['native_cohort_ids']
            if declared != sorted(set(declared)) or native is not None and native != declared:
                raise ValueError('V2_TEST_NATIVE_COHORT_CONFLICT')
            native = declared
            expected_ids = stable_partition(native, chunk_size=split_chunk_size(spec,'test'), slots=spec['slots']['test'])[index]
            ids = terminal['parent_ids']
            if ids != expected_ids or seen.intersection(ids):
                raise ValueError('V2_TEST_PARENT_PARTITION_CONFLICT')
            parent_ids, chunk_rows = set(), []
            files[str(terminal_path.relative_to(root))] = sha256_file(terminal_path)
            for path in sorted((directory / 'parents').glob('*.json')):
                saved = read_json(path)
                science = saved['science']
                if (saved.get('scope') != SCOPE_NAME or saved.get('backbone') != name
                        or saved.get('spec_sha256') != spec_sha or saved.get('pool_sha256') != pool_sha
                        or saved.get('science_sha256') != stable_sha256(science)):
                    raise ValueError('V2_TEST_PARENT_CONTENT_CONFLICT')
                match_count += verify_own_match_minima(science, candidate_ids=selected_ids,
                                                       model_sha=spec['model_files'][name]['model.pt'])
                actual = {r['parent_id'] for r in science['pair_rows']}
                if parent_ids.intersection(actual) or not actual <= set(ids):
                    raise ValueError('V2_TEST_PARENT_DUPLICATE_OR_OUTSIDE_CHUNK')
                parent_ids.update(actual)
                chunk_rows.extend(science['pair_rows'])
                files[str(path.relative_to(root))] = sha256_file(path)
            if parent_ids != set(ids) or len(chunk_rows) != terminal['pair_count']:
                raise ValueError('V2_TEST_PARENT_CHECKPOINT_INCOMPLETE')
            matrix_from_pairs(ids, selected, chunk_rows, root=root, split='test')
            all_rows.extend(chunk_rows)
            seen.update(ids)
        if seen != set(native):
            raise ValueError('V2_TEST_PARTITION_OMITS_PARENTS')
        collected[name] = dict(parent_ids=native, pairs=all_rows, candidates=selected, match_count=match_count)
    common = sorted(set.intersection(*(set(collected[n]['parent_ids']) for n in BACKBONES)))
    calibration_ids = set().union(*(set(frozen['selectors'][r].get('calibration_parent_ids', []))
                                    for r in frozen['selectors']))
    # New freeze records contain explicit parent IDs; reject a pre-closeout
    # lookalike rather than silently skip split-intersection validation.
    if not calibration_ids:
        raise ValueError('V2_FROZEN_CALIBRATION_PARENT_IDS_REQUIRED')
    if frozen.get('own_match_minimum_replayed') is not True:
        raise ValueError('V2_CALIBRATION_OWN_MATCH_MINIMUM_AUDIT_REQUIRED')
    if calibration_ids.intersection(set().union(*(set(collected[n]['parent_ids']) for n in BACKBONES))):
        raise ValueError('V2_CALIBRATION_TEST_PARENT_OVERLAP')
    return collected, common, frozen, files


def adopt_classifier_metrics(spec, *, output, frozen):
    """Read held-out classifier report only after new selectors are frozen."""
    require_global_freeze(frozen, sha256_file(spec['candidate_universe']))
    adoption = spec['classifier_adoption']
    root = Path(adoption['root']).resolve(strict=True)
    def member(item):
        rel = Path(item['relative_path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('V2_UNSAFE_CLASSIFIER_ADOPTION_MEMBER')
        path = (root / rel).resolve(strict=True)
        path.relative_to(root)
        if sha256_file(path) != item['sha256']:
            raise ValueError('V2_CLASSIFIER_ADOPTION_MEMBER_CHANGED')
        return path
    acceptance_path = member(adoption['acceptance'])
    acceptance = read_json(acceptance_path)
    if acceptance.get('state') != 'GNN_CORE_SEED7_CORRECTED_PASS' or acceptance.get('main_matrix_write') is not False:
        raise ValueError('V2_CLASSIFIER_ADOPTION_REQUIRES_CORRECTED_ACCEPTANCE')
    proof_path = member(adoption['independent_science_replay'])
    proof = read_json(proof_path)
    if (proof.get('state') != 'PASS' or acceptance.get('independent_science_replay_sha256')
            != adoption['independent_science_replay']['sha256']):
        raise ValueError('V2_CLASSIFIER_ADOPTION_INDEPENDENT_AUDIT_NOT_BOUND')
    producer_path = member(adoption['final_audit'])
    producer = read_json(producer_path)
    if proof.get('source_final_audit_sha256') != adoption['final_audit']['sha256']:
        raise ValueError('V2_CLASSIFIER_ADOPTION_PRODUCER_NOT_BOUND')
    bundle_path=Path(spec['bundle_root'])/'bundle_manifest.json'
    bundle=_bound(bundle_path,proof['cohort_contract']['bundle_manifest_sha256'])
    feature_sha=bundle['files'][bundle['feature_schema_path']]['sha256']
    # The already accepted independent replay binds exact weights/temperatures
    # and the full unchanged input bundle; never rehash or unpack its big archive.
    for name in BACKBONES:
        for leaf, field in (('model.pt','model_sha256'),('temperature_scaling.json','temperature_sha256')):
            if proof['models'][name][field] != spec['model_files'][name][leaf]:
                raise ValueError('V2_CLASSIFIER_ADOPTION_MODEL_OR_TEMPERATURE_CHANGED')
        if spec['model_files'][name]['feature_schema.json'] != feature_sha:
            raise ValueError('V2_CLASSIFIER_ADOPTION_FEATURE_SCHEMA_CHANGED')
    table_path = member(adoption['classifier_table'])
    if producer['files'].get('gnn_seed7_classifier_table.csv') != adoption['classifier_table']['sha256']:
        raise ValueError('V2_CLASSIFIER_METRICS_NOT_IN_ACCEPTED_INVENTORY')
    with table_path.open(newline='') as handle: rows = list(csv.DictReader(handle))
    if len(rows) != 5 or {r['backbone'] for r in rows} != set(BACKBONES):
        raise ValueError('V2_CLASSIFIER_TABLE_NOT_EXACT_FIVE_MODELS')
    for row in rows:
        if 'trainable_parameter_count' in row:
            row['frozen_inference_trainable_parameter_count']=row.pop('trainable_parameter_count')
    receipt = dict(scope=SCOPE_NAME, state='UNCHANGED_CLASSIFIER_METRICS_ADOPTED',
        source=adoption, model_files=spec['model_files'], classifier_retrained=False,
        temperature_refitted=False, prior_test_evaluated=True,
        explanation_metrics_adopted=False, main_matrix_write=False)
    atomic_json(output / 'classifier_adoption_receipt.json', receipt)
    atomic_csv(output / 'gnn_seed7_classifier_table.csv', rows)
    return rows


def finish_test(spec, *, spec_sha, pool_sha, candidates):
    from src.ablations.gnn.cpu_evaluation import matrix_from_pairs, explanation_metrics, _latex
    from src.eval.mutagenicity_wnode_selector import threshold_bundle_from_dict
    root = Path(spec['output_root'])
    if (root / AUDIT_NAME).exists():
        return verify_closeout(root)
    complete, common, frozen, source_files = collect_test_chunks(spec,
        spec_sha=spec_sha, pool_sha=pool_sha, candidates=candidates)
    source_files.update(frozen['calibration_files'])
    thresholds_payload = _bound(spec['thresholds'], spec['thresholds_sha256'])
    thresholds = threshold_bundle_from_dict(thresholds_payload)
    classifier = adopt_classifier_metrics(spec, output=root, frozen=frozen)
    native_rows, common_rows, stability, summaries = [], [], [], {}
    outputs = ['classifier_adoption_receipt.json','gnn_seed7_classifier_table.csv']
    atomic_json(root/'thresholds.json',thresholds_payload)
    atomic_json(root/'candidate_universe.json',candidates)
    outputs.extend(['thresholds.json','candidate_universe.json'])
    for name in BACKBONES:
        for mode in ('native','common'):
            role = f'{name}/{mode}'
            selector = frozen['selectors'][role]
            order = selector['ordered_rule_ids']
            ids = complete[name]['parent_ids'] if mode == 'native' else common
            chosen = [c for c in candidates if c['candidate_id'] in set(order)]
            pairs = [p for p in complete[name]['pairs'] if p['parent_id'] in set(ids) and p['candidate_id'] in set(order)]
            matrix = matrix_from_pairs(ids, chosen, pairs, root=root, split='test')
            result = explanation_metrics(matrix, [matrix.candidate_index[c] for c in order], thresholds)
            result.update(scope=SCOPE_NAME, backbone=name, cohort_mode=mode,
                auxiliary_metrics_k=20, selector_sha256=selector['self_sha256'])
            summaries[role] = result
            prefix = root / name / mode
            atomic_json(prefix / 'explanation_metrics.json', result)
            atomic_json(prefix / 'cohort_manifest.json', dict(test_parent_ids=ids,
                calibration_parent_ids=selector['calibration_parent_ids'],
                global_freeze_sha256=sha256_file(root/'CALIBRATION_FREEZE.json'),
                definition='true_source_and_correctly_predicted_source', frozen_model=spec['model_files'][name]))
            outputs.extend([f'{role}/explanation_metrics.json', f'{role}/cohort_manifest.json'])
            if result['prefix_rows']:
                for key, leaf in (('prefix_rows','prefix_metrics.csv'),('parent_rows','parent_prefix_metrics.csv'),
                                  ('threshold_rows','threshold_metrics.csv')):
                    atomic_csv(prefix/leaf, result[key]); outputs.append(f'{role}/{leaf}')
            for k in (10,20):
                record = result['prefix_rows'][k-1] if result['prefix_rows'] else {}
                row = dict(backbone=name, cohort=mode, K=k, cohort_size=len(ids),
                    state=result['state'], coverage=record.get('ccrcov_theta_star'),
                    strict_flip_count=record.get('strict_flip_parent_count'),
                    conditional_median_WNode=record.get('conditional_median_cost'),
                    fixed_capped_mean_cost=record.get('fixed_capped_mean_cost'))
                (native_rows if mode=='native' else common_rows).append(row)
    for name in BACKBONES:
        for mode in ('native','common'):
            a=set(frozen['selectors'][f'gine/{mode}']['ordered_rule_ids'])
            b=set(frozen['selectors'][f'{name}/{mode}']['ordered_rule_ids'])
            pa=set(summaries[f'gine/{mode}']['covered_parent_ids']); pb=set(summaries[f'{name}/{mode}']['covered_parent_ids'])
            stability.append(dict(backbone=name, cohort=mode, K=20,
                exact_rule_jaccard=len(a&b)/len(a|b),
                covered_parent_jaccard=len(pa&pb)/len(pa|pb) if pa|pb else None))
    for leaf, rows in (('gnn_seed7_explanation_native.csv',native_rows),
                       ('gnn_seed7_explanation_common.csv',common_rows),('gnn_seed7_rule_stability.csv',stability)):
        atomic_csv(root/leaf,rows); outputs.append(leaf)
    _latex(root/'gnn_seed7_table.tex',native_rows+common_rows,
           ('backbone','cohort','K','cohort_size','coverage','conditional_median_WNode','fixed_capped_mean_cost'))
    outputs.append('gnn_seed7_table.tex')
    run=dict(scope=SCOPE_NAME, spec_sha256=spec_sha, pool_sha256=pool_sha, execution_commit=spec['execution_commit'],
        candidate_count=len(candidates), model_files=spec['model_files'], main_matrix_write=False,
        training_rerun=False,temperature_refit=False,old_66_explanation_adopted=False,
        prior_test_evaluated=True,repair_selected_using_test=False,
        global_freeze_sha256=sha256_file(root/'CALIBRATION_FREEZE.json'),
        classifier_table_rows=len(classifier), native_parent_counts={n:len(complete[n]['parent_ids']) for n in BACKBONES},
        common_test_parent_count=len(common), own_match_minimum_replayed=True)
    atomic_json(root/'run_manifest.json',run); outputs.append('run_manifest.json')
    for rel in ('CALIBRATION_FREEZE.json','post_freeze_dependencies.json','version_scope.json'):
        outputs.append(rel)
    source_files.update({rel:sha256_file(root/rel) for rel in outputs})
    audit=dict(scope=SCOPE_NAME,state='GNN_REACH_V2_COMPLETE_PENDING_INDEPENDENT_AUDIT',spec_sha256=spec_sha,pool_sha256=pool_sha,
        backbones=list(BACKBONES),seed=7,main_matrix_write=False,
        all_ten_new_pool_explanations_complete=True, own_flip_and_match_minimum_replayed=True,
        completed_parent_units=sum(len(complete[n]['parent_ids']) for n in BACKBONES),
        common_test_parent_count=len(common), old_66_explanation_adopted=False,
        auxiliary_metrics_explicit_k=True, files=source_files)
    atomic_json(root/AUDIT_NAME,audit)
    return verify_closeout(root)


def verify_closeout(root):
    """Independent compact-artifact reopening; no original model/package scan."""
    root=Path(root)
    audit=read_json(root/AUDIT_NAME)
    if (audit.get('scope') != SCOPE_NAME or audit.get('state') != 'GNN_REACH_V2_COMPLETE_PENDING_INDEPENDENT_AUDIT'
            or audit.get('main_matrix_write') is not False or audit.get('old_66_explanation_adopted') is not False
            or audit.get('backbones') != list(BACKBONES)):
        raise ValueError('V2_NEW_CORE_AUDIT_REQUIRED')
    for rel,digest in audit['files'].items():
        if Path(rel).is_absolute() or '..' in Path(rel).parts or (root/rel).is_symlink() or sha256_file(root/rel)!=digest:
            raise ValueError('V2_FINAL_ARTIFACT_CHANGED:'+rel)
    frozen=read_json(root/'CALIBRATION_FREEZE.json'); require_global_freeze(frozen,audit['pool_sha256'])
    run=read_json(root/'run_manifest.json')
    if run['spec_sha256']!=audit['spec_sha256'] or run['pool_sha256']!=audit['pool_sha256']:
        raise ValueError('V2_FINAL_RUN_BINDING_CONFLICT')
    thresholds=read_json(root/'thresholds.json')
    theta,cap=float(thresholds['theta_star']),float(thresholds['cost_cap'])
    for name in BACKBONES:
        pairs={}
        for rel in audit['files']:
            if rel.startswith(name+'/test/') and '/parents/' in rel:
                data=read_json(root/rel)['science']
                for pair in data['pair_rows']:
                    key=pair['parent_id'],pair['candidate_id']
                    if key in pairs:raise ValueError('V2_INDEPENDENT_DUPLICATE_PAIR')
                    pairs[key]=pair
        for mode in ('native','common'):
            row=read_json(root/name/mode/'explanation_metrics.json')
            cohort=read_json(root/name/mode/'cohort_manifest.json')
            if (row['selector_sha256']!=frozen['selectors'][f'{name}/{mode}']['self_sha256']
                    or row['cohort_size']!=len(cohort['test_parent_ids'])
                    or row['state'] not in ('PASS','VALID_EMPTY_COHORT')
                    or len(row['prefix_rows']) != (20 if row['cohort_size'] else 0)):
                raise ValueError('V2_FINAL_EXPLANATION_INCOMPLETE')
            ids=cohort['test_parent_ids'];order=frozen['selectors'][f'{name}/{mode}']['ordered_rule_ids']
            if not ids:
                if row['state']!='VALID_EMPTY_COHORT' or row['CCRCov@20'] is not None:
                    raise ValueError('V2_EMPTY_TEST_COHORT_MUST_BE_NA')
                continue
            # Independent numeric replay from this backbone's committed pairs,
            # not from producer parent-prefix CSV or a copied GINE matrix.
            import numpy as np
            best=np.full(len(ids),np.inf);curve=[]
            for k,cid in enumerate(order,1):
                distances=[]
                for pid in ids:
                    pair=pairs[(pid,cid)]
                    distances.append(float(pair['wnode_distance']) if pair['pair_strict_flip'] else math.inf)
                best=np.minimum(best,distances);finite=best[np.isfinite(best)]
                metrics=row['prefix_rows'][k-1]
                expected=dict(k=k,ccrcov_theta_star=float(np.mean(best<=theta)),
                    strict_flip_parent_count=int(np.count_nonzero(np.isfinite(best))),
                    conditional_median_cost=float(np.median(finite)) if finite.size else None,
                    fixed_capped_mean_cost=float(np.mean(np.minimum(best,cap))))
                if any(metrics.get(key)!=value for key,value in expected.items()):
                    raise ValueError('V2_INDEPENDENT_PREFIX_METRIC_CONFLICT:'+name+'/'+mode+'/'+str(k))
                curve.append(expected['ccrcov_theta_star'])
            if (row['CCRCov@10']!=curve[9] or row['CCRCov@20']!=curve[19]
                    or row['AUC_over_K_1_20']!=float(sum((a+b)/2 for a,b in zip(curve[:-1],curve[1:])))):
                raise ValueError('V2_INDEPENDENT_K_SCOPE_OR_AUC_CONFLICT')
    acceptance={**audit,'state':'GNN_REACH_V2_CORE_PASS',
        'source_final_audit_sha256':sha256_file(root/AUDIT_NAME),
        'independent_own_prefix_metrics_replayed':True,'model_inference_rerun':False,'ot_recomputed':False}
    path=root/ACCEPTANCE_NAME
    if path.exists() and read_json(path)!=acceptance:
        raise ValueError('V2_INDEPENDENT_ACCEPTANCE_CANNOT_BE_OVERWRITTEN')
    if not path.exists():atomic_json(path,acceptance)
    return acceptance


def package_closeout(spec):
    """Small new results plus hash-bound classifier adoption, never weights."""
    root=Path(spec['output_root']); output=Path(spec['package_root'])
    if not output.is_absolute() or output==root or root in output.parents or output in root.parents:
        raise ValueError('V2_FRESH_DISJOINT_PACKAGE_ROOT_REQUIRED')
    with (root/'closeout.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        audit=verify_closeout(root)
        output.mkdir(parents=True,exist_ok=False)
        paths={rel:root/rel for rel in audit['files']}; paths[AUDIT_NAME]=root/AUDIT_NAME
        paths[ACCEPTANCE_NAME]=root/ACCEPTANCE_NAME
        inventory={rel:dict(sha256=sha256_file(path),bytes=path.stat().st_size) for rel,path in paths.items()}
        manifest=dict(scope=SCOPE_NAME,state='GNN_REACH_V2_PACKAGE_COMPLETE',files=inventory,
            audit_sha256=sha256_file(root/AUDIT_NAME),classifier_weights_included=False,
            independent_acceptance_sha256=sha256_file(root/ACCEPTANCE_NAME),
            original_classifier_package_bound_by_adoption_receipt=True,main_matrix_write=False)
        atomic_json(output/'package_manifest.json',manifest)
        paths['package_manifest.json']=output/'package_manifest.json'
        partial=output/'bace_gnn_reach_v2.tar.gz.partial'
        with tarfile.open(partial,'w:gz') as archive:
            for rel,path in sorted(paths.items()): archive.add(path,arcname=rel,recursive=False)
        with tarfile.open(partial,'r:gz') as archive:
            seen=set()
            for member in archive:
                if member.name in seen or member.name not in paths or not member.isfile():
                    raise ValueError('V2_PACKAGE_UNEXPECTED_OR_DUPLICATE_MEMBER')
                seen.add(member.name); digest=hashlib.sha256()
                stream=archive.extractfile(member)
                for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
                if digest.hexdigest()!=sha256_file(paths[member.name]):raise ValueError('V2_PACKAGE_ROUNDTRIP_MISMATCH')
            if seen!=set(paths):raise ValueError('V2_PACKAGE_MISSING_MEMBER')
        final=output/'bace_gnn_reach_v2.tar.gz';os.replace(partial,final)
        receipt=dict(state='GNN_REACH_V2_PACKAGE_COMPLETE',path=str(final),bytes=final.stat().st_size,
            sha256=sha256_file(final),manifest_sha256=sha256_file(output/'package_manifest.json'),main_matrix_write=False)
        atomic_json(output/'result_package.json',receipt)
        return receipt
