"""Taste/Mut CM post-filter adapter over the accepted oracle/WNode/K20 kernels.

No generation, attribution, scheduler or matrix authority. Matrices are bounded
parent blocks, committed as NPZ + receipt; NaN is never a scientific outcome.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

import numpy as np

from .cm_crem_runtime import atomic_json, checked_root, digest, file_sha, read_json, require_compute_node, utc_now
from .cm_crem_dataset_pilot import oracle_for, TERMINALS
from .cm_crem_dataset_full import pilot_scope, partition_remaining
from .cm_crem_selection import SelectionFreeze, evaluate_frozen_test
from .cm_crem_k20 import freeze_k20

BASE = '/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/baselines/cm_crem_global_v2'
STATUS = {0: 'UNCOMPUTED', 1: 'OK', 2: 'BEFORE_NOT_SOURCE', 3: 'NO_STRICT_FLIP', 4: 'ERROR'}


def commit_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k],v,equal_nan=True) for k,v in arrays.items()):
                raise ValueError('Uncommitted NPZ differs; preserve and diagnose')
        return
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temp.open('xb') as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.link(temp, path); temp.unlink()


def decode_status(values, statuses):
    if np.any(~np.isin(statuses, [1, 2, 3])):
        raise ValueError('UNCOMPUTED/ERROR/unknown logical slot is not semantic infinity')
    if np.any(~np.isfinite(values[statuses == 1])) or np.any(values[statuses == 1] < 0):
        raise ValueError('Invalid finite distance')
    if np.any(~np.isposinf(values[statuses != 1])):
        raise ValueError('Semantic invalid slot must be explicitly infinite')
    return np.asarray([[STATUS[int(s)] for s in row] for row in statuses]).reshape(values.shape)


def check_cohort(rows, binding, split):
    out = []
    for i, row in enumerate(rows):
        if binding.get('label_filter') is not None and int(row[binding['label_field']]) != binding['label_filter']:
            continue
        if binding.get('split_field') and row[binding['split_field']] != split:
            raise ValueError('Wrong source split')
        pid = str(row[binding['id_field']]) if binding.get('id_field') else str(i)
        out.append({'parent_id': pid, 'smiles': row[binding['smiles_field']], 'split': split})
    if binding.get('sort_ids'): out.sort(key=lambda r: r['parent_id'])
    ids = [r['parent_id'] for r in out]
    if len(ids) != binding['count'] or len(set(ids)) != len(ids):
        raise ValueError('Base cohort count/unique IDs differs')
    if binding.get('inventory_sha256') and digest([(r['parent_id'], r['smiles'], split) for r in out]) != binding['inventory_sha256']:
        raise ValueError('Original ordered parent/SMILES/split identity differs')
    return out


class Postfilter:
    def __init__(self, spec):
        self.path = Path(spec); self.s = read_json(spec)
        self.root = checked_root(self.s['output_root'], BASE)
        self.sha = digest({k: v for k, v in self.s.items() if k not in {'execution_commit', 'output_root'}})
        self.p = read_json(self.s['pilot_spec'])
        if file_sha(self.s['pilot_spec']) != self.s['pilot_spec_sha256']:
            raise ValueError('Pilot source changed')
        if self.p['dataset'] not in {'tastemolnet', 'mutagenicity'}:
            raise ValueError('Dataset adapter scope')
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parents[2], text=True).strip()
        if actual != self.s['execution_commit']: raise ValueError('Immutable execution commit differs')
        self.fixture = self.s.get('fixture', False)
        if self.s['deadline_utc'] != '2026-09-16T16:27:44Z': raise ValueError('Deadline changed')
        if self.s['batch_size'] != 32 or not 1 <= self.s['parent_block_size'] <= 8:
            raise ValueError('Frozen batch32 / bounded parent block required')
        self.oracle = self.feature = self.wnode = None

    def put(self, name, value):
        atomic_json(self.root/name, {'contract_sha256': self.sha, **value}, immutable=True)

    def get(self, name):
        value = read_json(self.root/name)
        if value.get('contract_sha256') != self.sha: raise ValueError('Mixed evaluation contract: ' + name)
        return value

    def admission(self):
        require_compute_node()
        if datetime.now(timezone.utc) >= datetime.fromisoformat(self.s['deadline_utc'].replace('Z', '+00:00')):
            raise RuntimeError('Original deadline reached at committed block boundary')
        self.root.mkdir(parents=True, exist_ok=True)
        v = os.statvfs(self.root)
        if v.f_bavail*v.f_frsize < self.s['reserve_bytes']: raise RuntimeError('HPC free byte reserve reached')

    def load_models(self):
        if self.oracle is None:
            from .cm_crem_oracle import FrozenCMWNode
            self.oracle, self.feature = oracle_for(self.p)
            self.wnode = FrozenCMWNode.from_resolved(self.p['resolved_wnode'])

    def predict(self, rows, split):
        self.load_models()
        if self.p['oracle_backend'] == 'gine': return self.oracle.predict_rows(rows, split=split)
        from .cm_crem_oracle import graph_identity
        probabilities = self.oracle.predict_proba([r['smiles'] for r in rows])
        return [{**r, 'probabilities': p.tolist(), 'predicted_label': int(p.argmax()),
                 'oracle_weight_sha256': self.p['oracle_sha256'], 'temperature': None,
                 'full_graph_id': graph_identity(r['smiles'], self.feature, require_connected=False)['candidate_id']}
                for r, p in zip(rows, probabilities, strict=True)]

    def pool(self):
        if self.fixture:
            original = read_json(Path(self.p['output_root'])/'pilot_pool.json')
            if original['scope_sha256'] != pilot_scope(self.p) or original['test_read']:
                raise ValueError('Train-only fixture pilot pool binding differs')
            candidates = sorted(original['candidates'],key=lambda c:c['candidate_id'])[:8]
            return {'candidates':candidates,'candidate_ids':[c['candidate_id'] for c in candidates],
                    'pool_sha256':digest(candidates)}
        source = Path(self.s['full_root']); pool = read_json(source/'pool_freeze.json')
        if pool['status'] != 'TRAIN_POOL_FROZEN' or pool['calibration_read'] or pool['test_read']:
            raise ValueError('No train-only frozen pool')
        if digest(pool['candidates']) != pool['pool_sha256'] or pool['candidate_ids'] != [r['candidate_id'] for r in pool['candidates']]:
            raise ValueError('Pool content or order changed')
        if len(pool['candidates']) > 6000 or len(set(pool['candidate_ids'])) != len(pool['candidate_ids']):
            raise ValueError('Pool budget / identity invalid')
        return pool

    def seal(self):
        self.admission(); source = Path(self.s['full_root']); pool = self.pool()
        if self.fixture:
            receipt = read_json(Path(self.p['output_root'])/'pilot_receipt.json')
            if receipt['status'] != 'PILOT32_GENERATION_FILTER_WNODE_COMPLETE' or receipt['scope_sha256'] != pilot_scope(self.p):
                raise ValueError('Completed pilot required for fixture')
            self.put('pool_binding.json', {'pool_sha256':digest(pool['candidates']), 'candidate_ids':pool['candidate_ids'],
                     'fixture':True,'test_read':False,'pilot_receipt_sha256':digest(receipt)})
            return
        fixture = read_json(Path(self.s['fixture_root'])/'audit/final_audit.json')
        fixture_spec = read_json(Path(self.s['fixture_root'])/'spec.json')
        if (fixture['status'] != 'TRAIN_FIXTURE_AUDIT_PASS' or fixture_spec['pilot_spec_sha256'] != self.s['pilot_spec_sha256']
                or fixture_spec['execution_commit'] != self.s['execution_commit']):
            raise ValueError('Matching execution train-only fixture not passed')
        full = read_json(source/'full_roster.json'); pilot = read_json(Path(self.p['output_root'])/'pilot_manifest.json')['parents']
        roster = full['parents']; seen = set()
        for shard in range(self.s['generation_shards']):
            expected = partition_remaining(roster, pilot, shard, self.s['generation_shards'])
            receipt = read_json(source/f'generation-shard-{shard}.json')
            if receipt['parent_ids'] != [r['parent_id'] for r in expected] or receipt['full_contract_sha256'] != full['full_contract_sha256']:
                raise ValueError('Generation shard union incomplete or overlapping')
            seen.update(receipt['parent_ids'])
        if seen & {p['parent_id'] for p in pilot} or len(seen)+32 != len(roster):
            raise ValueError('Pilot/full roster union not closed')
        # Reuse closed generation/filter receipts; never invoke generation again.
        unique = {}; raw_count = 0; statuses = {}
        for p in roster:
            old = p['parent_id'] in {x['parent_id'] for x in pilot}
            root = Path(self.p['output_root']) if old else source
            key = digest(p['parent_id'])[:20]+'.json'
            g, f = read_json(root/'generated'/key), read_json(root/'filter'/key)
            binding = ('scope_sha256', pilot_scope(self.p)) if old else ('full_contract_sha256', full['full_contract_sha256'])
            if g.get(binding[0]) != binding[1] or f.get(binding[0]) != binding[1]: raise ValueError('Mixed generation/filter contract')
            if g['parent_id'] != p['parent_id'] or f['parent_id'] != p['parent_id'] or g['status'] not in TERMINALS:
                raise ValueError('Incomplete generation/filter parent')
            raw_count += len(g['retained_raw']); statuses[g['status']] = statuses.get(g['status'], 0)+1
            raw_ids = {r['raw_id'] for r in g['retained_raw']}
            for c in f['accepted']:
                if c['predicted_label'] not in self.p['allowed_destinations']: raise ValueError('Non-target accepted prototype')
                if any(o['parent_id'] != p['parent_id'] or o['raw_id'] not in raw_ids for o in c['origins']):
                    raise ValueError('Candidate provenance not in saved train products')
                cid = c['candidate_id']
                if cid not in unique: unique[cid] = dict(c, origins=list(c['origins']))
                else: unique[cid]['origins'].extend(c['origins'])
        expected_ids = sorted(unique)[:6000]
        original = read_json(source/'pool_freeze.json')
        if original['candidate_ids'] != expected_ids or digest([unique[c] for c in expected_ids]) != original['pool_sha256']:
            raise ValueError('Global pool did not use complete roster / stable dedup')
        self.put('pool_binding.json', {'pool_sha256': digest(pool['candidates']), 'source_pool_sha256': original['pool_sha256'],
                 'candidate_ids': pool['candidate_ids'], 'train_parent_count': len(roster), 'pilot_adopted': 32,
                 'raw_count': raw_count, 'generation_statuses': statuses, 'fixture': self.fixture,
                 'full_contract_sha256': full['full_contract_sha256'], 'test_read': False})

    def parents(self, split):
        if split == 'test': self.freeze()
        if self.fixture:
            rows = read_json(Path(self.p['output_root'])/'pilot_manifest.json')['parents'][:4]
            return [{**p, 'split': 'train'} for p in rows]
        b = self.s['parents'][split]
        if file_sha(b['path']) != b['sha256']: raise ValueError('Original split file changed')
        with Path(b['path']).open() as f: rows = list(csv.DictReader(f))
        return check_cohort(rows, b, split)

    def freeze(self):
        raw = self.get('selection_freeze.json')
        return SelectionFreeze.from_dict({k: v for k, v in raw.items()})

    def cache(self, role, rows, split):
        """One immutable NPZ per original batch32; no per-pair cache files."""
        self.load_models(); all_predictions = []; encoded = {}
        for start in range(0, len(rows), 32):
            batch = rows[start:start+32]; name = f'encoding/{role}-{start:06d}'
            if (self.root/(name+'.json')).exists(): meta = self.get(name+'.json')
            else:
                self.admission(); predictions = self.predict(batch, split)
                enc = self.wnode.encode_rows(batch, featurizer=self.feature)
                arrays = {}; records = []
                for i, e in enumerate(enc):
                    arrays[f'H{i}'] = np.asarray(e['H'], dtype=np.float32)
                    records.append({k: v for k, v in e.items() if k != 'H'})
                commit_npz(self.root/(name+'.npz'), **arrays)
                self.put(name+'.json', {'input_sha256': digest(batch), 'predictions': predictions, 'records': records,
                         'npz_sha256': file_sha(self.root/(name+'.npz')), 'producer_pid': os.getpid(), 'batch_size': len(batch)})
                meta = self.get(name+'.json')
            if meta['input_sha256'] != digest(batch) or meta['npz_sha256'] != file_sha(self.root/(name+'.npz')):
                raise ValueError('Committed batch cache changed')
            with np.load(self.root/(name+'.npz'), allow_pickle=False) as z:
                for i, record in enumerate(meta['records']):
                    e = {**record, 'H': z[f'H{i}'].copy()}; encoded[e['candidate_id']] = e
            all_predictions.extend(meta['predictions'])
        return all_predictions, encoded

    def prepare(self, split):
        self.get('pool_binding.json'); pool = self.pool(); rows = self.parents(split)
        # Always preserve the complete original pool batches, even for selected-test audit.
        cp, ce = self.cache('pool', pool['candidates'], 'frozen_train_prototype')
        pp, pe = self.cache(split, rows, 'train_fixture' if self.fixture else split)
        if any(c['predicted_label'] not in self.p['allowed_destinations'] for c in cp):
            raise ValueError('Frozen prototype no longer predicts allowed destination')
        if split == 'test':
            chosen = self.freeze().selected_candidate_ids; by = {c['candidate_id']: c for c in cp}; cp = [by[c] for c in chosen]
        self.put(split+'_prepared.json', {'parent_ids': [r['parent_id'] for r in pp],
                 'candidate_ids': [c['candidate_id'] for c in cp], 'fixture': self.fixture,
                 'input_sha256': digest(rows), 'source_mask': [r['predicted_label'] == self.p['source_label'] for r in pp]})
        return pp, pe, cp, ce

    def blocks(self, split, shard, shards):
        from src.eval.node_wasserstein_distance import compute_node_wasserstein_distance
        self.admission(); pp, pe, cp, ce = self.prepare(split)
        width = self.s['parent_block_size']; complete = []
        for block, start in enumerate(range(0, len(pp), width)):
            if block % shards != shard: continue
            name = f'{split}/block-{block:04d}'; parents = pp[start:start+width]
            if (self.root/(name+'.json')).exists(): complete.append(block); continue
            self.admission(); started = time.monotonic()
            values = np.full((len(parents), len(cp)), np.nan, dtype=np.float64)
            states = np.zeros(values.shape, dtype=np.uint8); computed = reused = 0; cache = {}
            for i, p in enumerate(parents):
                for j, c in enumerate(cp):
                    if p['predicted_label'] != self.p['source_label']: values[i,j] = np.inf; states[i,j] = 2; continue
                    if c['predicted_label'] not in self.p['allowed_destinations']: values[i,j] = np.inf; states[i,j] = 3; continue
                    left, right = pe[p['full_graph_id']], ce[c['full_graph_id']]
                    for key in ('producer','molclr_checkpoint_sha256','numerical_contract_sha256','node_extraction_version'):
                        if left[key] != right[key]: raise ValueError('Raw distance encoding producer conflict: '+key)
                    key = tuple(sorted((left['encoding_sha256'], right['encoding_sha256'])))
                    if key not in cache:
                        v, _ = compute_node_wasserstein_distance(left['H'], right['H'], feature_cost='cosine', node_mass='uniform', size_penalty_beta=0.0)
                        if not np.isfinite(v) or v < 0: raise ValueError('Exact WNode failure')
                        cache[key] = v; computed += 1
                    else: reused += 1
                    values[i,j] = cache[key]; states[i,j] = 1
            decode_status(values, states)
            commit_npz(self.root/(name+'.npz'), values=values, states=states)
            self.put(name+'.json', {'parent_ids': [r['parent_id'] for r in parents], 'candidate_ids': [r['candidate_id'] for r in cp],
                     'npz_sha256': file_sha(self.root/(name+'.npz')), 'logical_pair_count': int(values.size),
                     'unique_distance_compute_count': computed, 'cache_reuse_count': reused,
                     'seconds': time.monotonic()-started, 'producer_pid': os.getpid(), 'job_id': os.environ['SLURM_JOB_ID']})
            complete.append(block)
            atomic_json(self.root/f'progress-{split}-{shard}.json', {'completed_blocks': complete, 'stage': split, 'updated_at': utc_now()})
        self.put(f'{split}/shard-{shard}.json', {'blocks': complete, 'shards': shards})

    def matrix(self, split):
        prepared = self.get(split+'_prepared.json'); n, m = len(prepared['parent_ids']), len(prepared['candidate_ids'])
        values = np.full((n,m), np.nan); states = np.zeros((n,m), dtype=np.uint8); width = self.s['parent_block_size']
        for block, start in enumerate(range(0,n,width)):
            name = f'{split}/block-{block:04d}'; meta = self.get(name+'.json')
            if meta['parent_ids'] != prepared['parent_ids'][start:start+width] or meta['candidate_ids'] != prepared['candidate_ids']:
                raise ValueError('Block coverage/order mismatch')
            if meta['npz_sha256'] != file_sha(self.root/(name+'.npz')): raise ValueError('Block content changed')
            with np.load(self.root/(name+'.npz'), allow_pickle=False) as z:
                values[start:start+width] = z['values']; states[start:start+width] = z['states']
        return values, decode_status(values, states), prepared['parent_ids'], prepared['candidate_ids'], np.array(prepared['source_mask'],dtype=bool)

    def select(self):
        d,s,p,c,m = self.matrix('calibration'); e = self.s['evaluation']; pool = self.get('pool_binding.json')
        freeze, report = freeze_k20(d,s,p,c,m,e['theta'],e['cap'],e['grid'],self.sha,pool['pool_sha256'])
        self.put('selection_freeze.json', freeze.to_dict()); self.put('selection_report.json', report)

    def result(self):
        d,s,p,c,m = self.matrix('test')
        return evaluate_frozen_test(self.freeze(),d,pair_status=s,parent_ids=p,candidate_ids=c,source_mask=m,contract_sha256=self.sha)

    def audit(self):
        self.admission(); self.seal()
        d,s,p,c,m = self.matrix('calibration'); e = self.s['evaluation']; saved = self.freeze()
        replay,_ = freeze_k20(d,s,p,c,m,e['theta'],e['cap'],e['grid'],self.sha,self.get('pool_binding.json')['pool_sha256'])
        if replay != saved: raise ValueError('Independent global selection replay differs')
        # Independently rebuild the original batch32/last-batch, not singleton.
        checked = 0
        for role, rows, split in [('pool',self.pool()['candidates'],'frozen_train_prototype'),
                                  ('calibration',self.parents('calibration'),'train_fixture' if self.fixture else 'calibration'),
                                  ('test',self.parents('test'),'train_fixture' if self.fixture else 'test')]:
            for start in range(0,len(rows),32):
                meta = self.get(f'encoding/{role}-{start:06d}.json')
                if meta['producer_pid'] == os.getpid(): raise ValueError('Audit must run in a separate process')
                actual = self.predict(rows[start:start+32],split)
                if actual != meta['predictions']: raise ValueError('Independent same-batch original oracle differs: '+role+':'+str(start))
                checked += len(actual)
        # Actual exact WNode replay, fixed first valid64 pairs, no test-driven sampling.
        from .cm_crem_oracle import raw_distance_record
        ot_checked = 0
        for split in ['calibration','test']:
            pp,pe,cp,ce = self.prepare(split); matrix,*_ = self.matrix(split)
            for i,parent in enumerate(pp):
                for j,candidate in enumerate(cp):
                    if not np.isfinite(matrix[i,j]) or ot_checked >= 64: continue
                    left,right = pe[parent['full_graph_id']],ce[candidate['full_graph_id']]
                    # Reuse original audited raw-record kernel; restore list serialization for its digest.
                    left = {**left,'H':left['H'].tolist()}; right = {**right,'H':right['H'].tolist()}
                    actual = raw_distance_record(left,right,numerical_contract=self.p['resolved_wnode'])
                    if actual['distance'] != matrix[i,j]: raise ValueError('Independent raw WNode replay differs')
                    ot_checked += 1
        result = self.result(); metrics = result.prefix_metrics()
        if any(metrics[i]['coverage'] < metrics[i-1]['coverage'] or metrics[i]['cost'] > metrics[i-1]['cost'] for i in range(1,20)):
            raise ValueError('Nested prefix monotonicity failed')
        self.put('test_evaluation.json',result.to_dict())
        self.put('audit/final_audit.json', {'status': 'TRAIN_FIXTURE_AUDIT_PASS' if self.fixture else 'CM_DATASET_POSTFILTER_AUDIT_PASS',
                 'fixture': self.fixture, 'scientific_pass_claimed': not self.fixture,
                 'selection_freeze_sha':saved.freeze_sha256,'oracle_rows_replayed':checked,'exact_pairs_replayed':ot_checked,
                 'test_count':len(result.parent_ids),'auditor_pid':os.getpid(),'job_id':os.environ['SLURM_JOB_ID'],
                 'test_result_sha256':digest(result.to_dict()),'main_authority_written':False})

    def export(self):
        from .cm_crem_export import export_results
        audit = self.get('audit/final_audit.json')
        if audit['status'] != ('TRAIN_FIXTURE_AUDIT_PASS' if self.fixture else 'CM_DATASET_POSTFILTER_AUDIT_PASS'):
            raise ValueError('Completed independent audit required')
        export_results(self.result(),self.root,dataset=self.p['dataset'],oracle=self.p['oracle_backend'],
                       fixture=self.fixture,dataset_audit=audit)

    def package(self):
        self.get('audit/final_audit.json'); self.get('pool_binding.json')
        archive = self.root/'result_package.tar.gz'
        names = ['spec.json','pool_binding.json','selection_freeze.json','selection_report.json','test_evaluation.json',
                 'audit/final_audit.json','results']
        if archive.exists(): raise FileExistsError('Package already exists; adopt receipt, do not repeat')
        with tarfile.open(archive.with_suffix('.part'),'w:gz') as t:
            for name in names: t.add(self.root/name,arcname=name)
        os.rename(archive.with_suffix('.part'),archive)
        self.put('result_package.json',{'status':'PACKAGED','path':str(archive),'bytes':archive.stat().st_size,
                 'sha256':file_sha(archive),'import_state':'PENDING_SCOPED_TRANSFER','fixture':self.fixture})


def main(argv=None):
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--spec',required=True)
    p.add_argument('--action',required=True,choices=['seal','prepare','blocks','select','audit','export','package','status'])
    p.add_argument('--split',choices=['calibration','test'],default='calibration');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args(argv);x=Postfilter(a.spec)
    if a.action=='status':
        print(json.dumps({'root':str(x.root),'progress':[read_json(f) for f in x.root.glob('progress-*.json')],
                          'final_audit':read_json(x.root/'audit/final_audit.json') if (x.root/'audit/final_audit.json').exists() else None}));return
    x.admission()
    if a.action=='prepare':x.prepare(a.split)
    elif a.action=='blocks':x.blocks(a.split,a.shard,a.shards)
    else:getattr(x,a.action)()
