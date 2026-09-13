"""CM full train continuation: preserve pilot RNG namespace and completed units.

Dataset-specific stages reuse the original oracle/native worker. No scheduler,
test selection, training, or authority lives in this module.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .cm_crem_dataset_pilot import TERMINALS, validate, oracle_for
from .cm_crem_runtime import atomic_json, read_json, digest, file_sha, require_compute_node, utc_now, checked_root


def pilot_scope(spec):
    return digest({k: v for k, v in spec.items()
                   if k not in {'execution_commit', 'execution_root', 'output_root'}})


def partition_remaining(roster, pilot, shard, shards):
    ids = [p['parent_id'] for p in roster]
    chosen = [p['parent_id'] for p in pilot]
    if len(ids) != len(set(ids)) or len(chosen) != 32 or len(set(chosen)) != 32:
        raise ValueError('Full roster / exact pilot32 identities invalid')
    by_id = {p['parent_id']: p for p in roster}
    if any(by_id.get(p['parent_id']) != p for p in pilot):
        raise ValueError('Pilot is not an exact unchanged subset of full train roster')
    if not 0 <= shard < shards <= 4:
        raise ValueError('Invalid bounded shard assignment')
    remaining = [p for p in roster if p['parent_id'] not in set(chosen)]
    return remaining[shard::shards]


class DatasetFull:
    def __init__(self, path):
        self.path = Path(path).resolve(); self.spec = read_json(path)
        self.pilot_path = Path(self.spec['pilot_spec'])
        if file_sha(self.pilot_path) != self.spec['pilot_spec_sha256']:
            raise ValueError('Pilot spec identity changed')
        self.p = read_json(self.pilot_path); validate(self.p)
        self.pilot = Path(self.p['output_root']); self.seed_scope = pilot_scope(self.p)
        self.root = checked_root(self.spec['output_root'], '/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/baselines/cm_crem_global_v2')
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parents[2], text=True).strip()
        if actual != self.spec['execution_commit']:
            raise ValueError('Immutable execution commit differs')
        receipt = self.old('pilot_receipt.json')
        if receipt['status'] != 'PILOT32_GENERATION_FILTER_WNODE_COMPLETE' or receipt['parent_count'] != 32:
            raise ValueError('Completed scientific pilot32 is required')
        self.roster = self.old('train_source_manifest.json')['parents']
        self.pilot_parents = self.old('pilot_manifest.json')['parents']
        partition_remaining(self.roster, self.pilot_parents, 0, self.spec['shards'])
        self.pilot_ids = {p['parent_id'] for p in self.pilot_parents}
        self.contract = digest({'pilot_scope': self.seed_scope, 'roster': self.roster,
                                'full_library_cap': 6000, 'authorization': 'CODEX_MAIN20_CLOSEOUT_20260913'})

    def old(self, name):
        value = read_json(self.pilot/name)
        if value.get('scope_sha256') != self.seed_scope:
            raise ValueError('Pilot artifact belongs to another scope: ' + name)
        return value

    def put(self, name, value):
        atomic_json(self.root/name, {'full_contract_sha256': self.contract, **value}, immutable=True)

    def get(self, name):
        value = read_json(self.root/name)
        if value.get('full_contract_sha256') != self.contract:
            raise ValueError('Full continuation mixed contracts: ' + name)
        return value

    def deadline(self):
        from datetime import datetime, timezone
        if datetime.now(timezone.utc) >= datetime.fromisoformat(self.p['deadline_utc'].replace('Z', '+00:00')):
            raise RuntimeError('Original campaign deadline reached; stop before next parent')

    def plan(self):
        records = self.old('attribution.json')['records']
        attrs = {a['parent_id']: a for a in records}
        from .cm_crem_generation import parent_seed
        adopted = []
        for parent in self.pilot_parents:
            a = attrs[parent['parent_id']]
            g = self.old('generated/' + digest(parent['parent_id'])[:20] + '.json')
            if g['status'] not in TERMINALS or g['parent_id'] != parent['parent_id']:
                raise ValueError('Incomplete pilot generation')
            if g.get('science_hash', self.seed_scope) != self.seed_scope:
                raise ValueError('Pilot generation RNG namespace differs')
            if g.get('seed', parent_seed(self.seed_scope, parent['parent_id'])) != parent_seed(self.seed_scope, parent['parent_id']):
                raise ValueError('Pilot parent RNG differs')
            if a['generation_request']['parent_id'] != parent['parent_id']:
                raise ValueError('Pilot attribution identity differs')
            adopted.append({'parent_id': parent['parent_id'], 'generation': str(self.pilot/'generated'/(digest(parent['parent_id'])[:20]+'.json')),
                            'request_sha256': digest(a['generation_request']), 'generation_sha256': digest(g)})
        result = {'status': 'TRAIN_ROSTER_FROZEN', 'parents': self.roster, 'pilot_adopted': adopted,
                  'remaining_count': len(self.roster)-32, 'pilot_count': 32,
                  'generation_science_hash': self.seed_scope,
                  'rng_scope': 'EXACT_PILOT_PER_PARENT_NAMESPACE_NOT_GLOBAL_TRAJECTORY',
                  'test_read': False, 'deadline_utc': self.p['deadline_utc']}
        self.put('full_roster.json', result)
        return {k: v for k, v in result.items() if k not in {'parents', 'pilot_adopted'}}

    def assigned(self, shard):
        self.get('full_roster.json')
        return partition_remaining(self.roster, self.pilot_parents, shard, self.spec['shards'])

    def attribute(self, shard):
        oracle, _ = oracle_for(self.p)
        rows = self.assigned(shard)
        for parent in rows:
            name = 'attribution_units/' + digest(parent['parent_id'])[:20] + '.json'
            if (self.root/name).exists():
                self.get(name); continue
            self.deadline()
            if self.p['oracle_backend'] == 'rf':
                from .cm_crem_rf_attribution import attribute
                a = attribute(oracle, parent)
            else:
                a = oracle.attribute_train_parent(parent)
            if a.get('before_label', a.get('before', {}).get('predicted_label', 1)) != 1:
                raise ValueError('Frozen full train-source predicate changed')
            self.put(name, {'record': a})
        self.put(f'attribution-shard-{shard}.json', {'parent_ids': [p['parent_id'] for p in rows], 'new_count': len(rows)})

    def generate(self, shard):
        from .cm_crem_assets import prepare_job_scratch, stage_static_database, validate_database_source
        from .cm_crem_generation import generate_parent
        e = self.p['generation_execution']; receipt = read_json(e['database_receipt'])
        if Path(sys.executable).resolve() != Path(e['generator_python']).resolve():
            raise ValueError('Wrong isolated generator environment')
        self.get(f'attribution-shard-{shard}.json')
        source = validate_database_source(read_json(self.p['original_cm_spec']), receipt)
        scratch = prepare_job_scratch(self.root, required_bytes=receipt['uncompressed_bytes'], reserve_bytes=2*1024**3)
        self.put('scratch-'+os.environ['SLURM_JOB_ID']+f'-{shard}.json', scratch)
        if scratch['status'] != 'JOB_SCRATCH_READY': raise RuntimeError(scratch)
        staged = stage_static_database(e['database_path'], e['database_receipt'], reserve_bytes=2*1024**3,
                    expected_url=source['actual_source_url'], scratch_receipt=scratch)
        if staged['status'] != 'LOCAL_DATABASE_READY': raise RuntimeError(staged)
        self.put('database-'+os.environ['SLURM_JOB_ID']+f'-{shard}.json', staged)
        rows = self.assigned(shard)
        for i, parent in enumerate(rows):
            name = 'generated/' + digest(parent['parent_id'])[:20] + '.json'
            if (self.root/name).exists(): g = self.get(name)
            else:
                self.deadline()
                a = self.get('attribution_units/'+digest(parent['parent_id'])[:20]+'.json')['record']
                log = self.root/'logs'/('native-'+digest(parent['parent_id'])[:20]+'.log')
                log.parent.mkdir(exist_ok=True)
                # An unfinished log is preserved under its original attempt; never overwritten.
                if log.exists(): log = log.with_name(log.name+'.job-'+os.environ['SLURM_JOB_ID'])
                g = generate_parent(a['generation_request'], {'database_path': staged['database_path'],
                      'upstream_root': e['upstream_root'], 'science_hash': self.seed_scope,
                      'parent_wall_limit_seconds': 900}, log_path=log)
                self.put(name, g)
            if g['status'] not in TERMINALS: raise RuntimeError('Native parent failed: '+parent['parent_id']+' '+str(g.get('error')))
            atomic_json(self.root/f'progress-{shard}.json', {'stage': 'FULL_REMAINING_GENERATION', 'completed': i+1,
                        'assigned': len(rows), 'pilot_adopted_separately': 32, 'updated_at': utc_now(), 'job_id': os.environ['SLURM_JOB_ID']})
        self.put(f'generation-shard-{shard}.json', {'status': 'GENERATION_SHARD_COMPLETE', 'parent_ids': [p['parent_id'] for p in rows]})

    def filter(self):
        from .cm_crem_oracle import graph_identity
        for shard in range(self.spec['shards']): self.get(f'generation-shard-{shard}.json')
        oracle, feature = oracle_for(self.p); unique = {}; raw_count = 0
        for parent in self.roster:
            key = digest(parent['parent_id'])[:20]
            if parent['parent_id'] in self.pilot_ids:
                g = self.old('generated/'+key+'.json'); f = self.old('filter/'+key+'.json')
            else:
                g = self.get('generated/'+key+'.json')
                if g['status'] not in TERMINALS: raise ValueError('Failed generation is not scientific zero')
                name = 'filter/'+key+'.json'
                if (self.root/name).exists(): f = self.get(name)
                else:
                    self.deadline()
                    if self.p['oracle_backend'] == 'gine': f = oracle.filter_generated(parent, g)
                    else:
                        accepted, rejected = [], []
                        for r in g['retained_raw']:
                            try: identity = graph_identity(r['smiles'], feature)
                            except (ValueError, RuntimeError) as err:
                                rejected.append({'raw_id': r['raw_id'], 'reason': str(err)}); continue
                            probs = oracle.predict_proba([r['smiles']])[0]; label = int(probs.argmax())
                            c = {**identity, 'smiles': identity['canonical_smiles'], 'probabilities': probs.tolist(),
                                 'predicted_label': label, 'origins': [{'parent_id': parent['parent_id'], 'raw_id': r['raw_id']}],
                                 'oracle_weight_sha256': self.p['oracle_sha256']}
                            if label in self.p['allowed_destinations']: accepted.append(c)
                            else: rejected.append({'raw_id': r['raw_id'], 'reason': 'NOT_DESTINATION'})
                        f = {'parent_id': parent['parent_id'], 'accepted': accepted, 'rejected': rejected, 'raw_count': len(g['retained_raw'])}
                    self.put(name, f)
            raw_count += len(g['retained_raw'])
            for c in f['accepted']:
                if c['predicted_label'] not in self.p['allowed_destinations']: raise ValueError('Non-target prototype')
                cid = c['candidate_id']
                if cid not in unique: unique[cid] = dict(c, origins=list(c['origins']))
                else: unique[cid]['origins'].extend(c['origins'])
        ids = sorted(unique)[:6000]
        self.put('pool_freeze.json', {'status': 'TRAIN_POOL_FROZEN', 'candidates': [unique[c] for c in ids],
                 'candidate_ids': ids, 'pool_sha256': digest([unique[c] for c in ids]), 'raw_count': raw_count,
                 'eligible_unique_before_cap': len(unique), 'cap': 6000, 'test_read': False,
                 'calibration_read': False, 'source_parent_count': len(self.roster)})
        return {'stage': 'TRAIN_POOL_FROZEN', 'raw_count': raw_count, 'eligible_unique': len(unique), 'retained': len(ids)}

    def run_shard(self, shard):
        self.attribute(shard)
        env = dict(os.environ, PYTHONHASHSEED='0', PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='')
        subprocess.run([self.p['generation_execution']['generator_python'], '-s', '-B',
             str(Path(__file__).parents[2]/'scripts/run_cm_crem_dataset_full.py'), '--config', 'configs/hpc.yaml',
             '--spec', str(self.path), '--action', 'generate', '--shard', str(shard)], env=env, check=True)


def main(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', required=True); p.add_argument('--spec', required=True)
    p.add_argument('--action', required=True, choices=['plan', 'run-shard', 'attribute', 'generate', 'filter', 'status'])
    p.add_argument('--shard', type=int, default=0); a = p.parse_args(argv); x = DatasetFull(a.spec)
    if a.action == 'status':
        print(json.dumps({'root': str(x.root), 'progress': [read_json(q) for q in x.root.glob('progress-*.json')],
              'pool_frozen': (x.root/'pool_freeze.json').exists()})); return
    if a.action != 'plan': require_compute_node(); x.deadline()
    if a.action in {'run-shard', 'attribute', 'generate'}: result = getattr(x, a.action.replace('-', '_'))(a.shard)
    else: result = getattr(x, a.action)()
    print(json.dumps(result))
