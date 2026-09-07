"""Reuse only hash-bound completed match-level distances, split by split.

No SQLite is opened. Predictions and flip decisions are always recomputed by
the common evaluator. Test records are inaccessible before its fresh freeze.
"""
from __future__ import annotations
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from src.eval.bace_frozen_gnn_contracts import stable_sha256


class SavedMatchDistanceReuse:
    def __init__(self, distance: Any, *, source_root: str | Path, audit_sha256: str,
                 current_contract: Mapping[str, Any], output_root: str | Path,
                 current_binding: str):
        self.distance=distance
        self.root=Path(source_root).resolve(strict=True)
        self.output=Path(output_root).resolve()
        if self.root==self.output or self.root in self.output.parents or self.output in self.root.parents:
            raise ValueError('RAW_DISTANCE_SOURCE_MUST_BE_DISJOINT')
        self.lock=(self.root/'writer.lock').open('r')
        self.binding=current_binding
        self.audit_sha256=audit_sha256
        try:
            fcntl.flock(self.lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
            self._bind_source(current_contract)
        except BaseException:
            self.lock.close()
            raise

    def _bind_source(self, current_contract):
        self.audit=self._read('final_audit.json',self.audit_sha256)
        if self.audit.get('state')!='PASS' or self.audit.get('main_matrix_write') is not False:
            raise ValueError('RAW_DISTANCE_SOURCE_NOT_COMPLETED_LLM_AUDIT')
        old=self._member('run_manifest.json')
        payload={k:v for k,v in old.items() if k!='self_sha256'}
        if old.get('self_sha256')!=stable_sha256(payload) or old['binding_sha256']!=self.audit['binding_sha256']:
            raise ValueError('RAW_DISTANCE_SOURCE_MANIFEST_BINDING')
        for key in ('task_spec_sha256','reference_sha256','bundle_sha256','variant','selector_input_sha256','cohort_definition'):
            if old.get(key)!=current_contract.get(key):
                raise ValueError('RAW_DISTANCE_CONTRACT_MISMATCH:'+key)
        for key in ('pool_sha256','receipt_sha256'):
            if old['pool'].get(key)!=current_contract['pool'].get(key):
                raise ValueError('RAW_DISTANCE_ORIGINAL_POOL_MISMATCH:'+key)
        for name in ('cpu_evaluation.py','bace_frozen_gnn_pool.py','bace_frozen_gnn_verification.py'):
            if old['scientific_sources'].get(name)!=current_contract['scientific_sources'].get(name):
                raise ValueError('RAW_DISTANCE_KERNEL_MISMATCH:'+name)
        commit=old['execution_commit']
        if not re.fullmatch(r'[0-9a-f]{40}',commit):
            raise ValueError('RAW_DISTANCE_SOURCE_COMMIT_UNPINNED')
        repo=Path(__file__).resolve().parents[3]
        self.kernel_proof={}
        for relative in ('src/eval/node_wasserstein_distance.py','src/eval/molclr_node_embeddings.py',
                         'src/data/molecular_graph_featurizer.py','src/chem/hard_deletion.py'):
            prior=subprocess.check_output(['git','show',commit+':'+relative],cwd=repo)
            current=(repo/relative).read_bytes()
            if prior!=current:
                raise ValueError('RAW_DISTANCE_SOURCE_IMPLEMENTATION_DRIFT:'+relative)
            self.kernel_proof[relative]=hashlib.sha256(current).hexdigest()
        universe=self._member('candidate_universe.jsonl',jsonl=True)
        self.old_candidates={r['candidate_id']:r['canonical_fragment'] for r in universe}
        self.values={}
        self.loaded_splits=[]
        self.reused=0
        self.new_requests=0
        self.bound_records=0

    def _read(self, relative: str, digest: str, *, jsonl: bool=False):
        rel=Path(relative)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('RAW_DISTANCE_UNSAFE_MEMBER')
        path=self.root/rel
        if path.is_symlink() or path.resolve().parent!=path.parent.resolve():
            raise ValueError('RAW_DISTANCE_SYMLINK_MEMBER')
        path.resolve(strict=True).relative_to(self.root)
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError('RAW_DISTANCE_MEMBER_HASH:'+relative)
        return [json.loads(x) for x in raw.splitlines() if x.strip()] if jsonl else json.loads(raw)

    def _member(self, relative: str, *, jsonl: bool=False):
        digest=self.audit['files'].get(relative)
        if not digest:
            raise ValueError('RAW_DISTANCE_MEMBER_UNBOUND:'+relative)
        return self._read(relative,digest,jsonl=jsonl)

    @staticmethod
    def key(parent_smiles: str,residual_smiles: str,context: Mapping[str,Any]):
        keys=('parent_id','candidate_id','match_index','match_atom_indices','teacher_sha256',
              'oracle_checkpoint_id','action_semantics_version','match_selection_policy','distance_implementation_version')
        if any(key not in context for key in keys):
            raise ValueError('RAW_DISTANCE_ACTION_CONTEXT_INCOMPLETE')
        return stable_sha256({'parent_smiles':parent_smiles,'residual_smiles':residual_smiles,
                              'context':{key:context[key] for key in keys}})

    def prepare_split(self,split: str,candidates):
        if split not in ('calibration','test'):
            raise ValueError('RAW_DISTANCE_SPLIT_UNSUPPORTED')
        if split=='test':
            # Read the fresh selector itself, not a caller-supplied boolean.
            frozen=json.loads((self.output/'selector_manifest.json').read_text())
            if (frozen.get('self_sha256')!=stable_sha256({k:v for k,v in frozen.items() if k!='self_sha256'})
                    or frozen.get('binding_sha256')!=self.binding or frozen.get('test_loaded') is not False
                    or frozen.get('selection_frozen') is not True):
                raise ValueError('RAW_DISTANCE_TEST_BEFORE_FRESH_FREEZE')
        from src.chem.hard_deletion import CONNECTED_MATCH_SELECTION_POLICY
        from src.eval.bace_frozen_gnn_verification import DISTANCE_IMPLEMENTATION_VERSION
        current={r['candidate_id']:r['canonical_fragment'] for r in candidates}
        self.values={}
        prefix='parent_checkpoints/'+split+'/'
        members=[rel for rel in self.audit['files'] if rel.startswith(prefix)
                 and rel.endswith('.json') and Path(rel).name!='progress.json']
        for relative in sorted(members):
            checkpoint=self._member(relative)
            science=checkpoint.get('science')
            if not isinstance(science,dict) or checkpoint.get('science_sha256')!=stable_sha256(science):
                raise ValueError('RAW_DISTANCE_CHECKPOINT_SCIENCE_HASH:'+relative)
            for row in science['match_rows']:
                identity=row['candidate_id']
                if identity not in current:
                    continue
                if current[identity]!=self.old_candidates.get(identity) or row['canonical_fragment']!=current[identity]:
                    raise ValueError('RAW_DISTANCE_CANONICAL_RULE_CONFLICT')
                value=row.get('wnode_distance')
                if not row.get('distance_ok') or not row.get('cf_flip'):
                    continue
                if (row.get('delete_valid') is not True or row.get('sanitize_ok') is not True
                        or row.get('residual_connected') is not True or row.get('rf_oracle_used') is not False
                        or row.get('oracle_backend')!='gnn' or not isinstance(value,(float,int))
                        or not math.isfinite(value) or value<0):
                    raise ValueError('RAW_DISTANCE_INVALID_FINITE_MATCH')
                context={key:row[key] for key in ('parent_id','candidate_id','match_index','match_atom_indices','action_semantics_version')}
                context.update(teacher_sha256=row['oracle_checkpoint_hash'],oracle_checkpoint_id=row['oracle_checkpoint_hash'],
                    match_selection_policy=CONNECTED_MATCH_SELECTION_POLICY,distance_implementation_version=DISTANCE_IMPLEMENTATION_VERSION)
                key=self.key(row['parent_smiles'],row['residual_smiles'],context)
                if key in self.values and self.values[key]!=value:
                    raise ValueError('RAW_DISTANCE_DUPLICATE_CONFLICT')
                self.values[key]=value
                self.bound_records+=1
        self.loaded_splits.append(split)

    def distance_for_action(self,parent_smiles,residual_smiles,*,action_context):
        key=self.key(parent_smiles,residual_smiles,action_context)
        if key in self.values:
            self.reused+=1
            return {'distance':self.values[key],'ok':True,'cache_hit':True,'error':None,
                    'metadata':{'reuse':'HASH_BOUND_COMPLETED_MATCH_RECORD','source_audit_sha256':self.audit_sha256}}
        self.new_requests+=1
        return self.distance.distance_for_action(parent_smiles,residual_smiles,action_context=action_context)

    def stats_dict(self):
        result=self.distance.stats_dict()
        result['pair_distance_cache_hits']+=self.reused
        result.update(saved_match_distance_reused=self.reused,saved_match_distance_bound_records=self.bound_records,
                      saved_match_distance_new_requests=self.new_requests,saved_match_distance_loaded_splits=list(self.loaded_splits),
                      saved_match_distance_source_audit_sha256=self.audit_sha256)
        result['saved_match_distance_kernel_proof']=self.kernel_proof
        return result

    def close(self):
        try:self.distance.close()
        finally:self.lock.close()
