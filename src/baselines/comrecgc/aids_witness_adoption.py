"""Independent finite-witness verification and exact all-zero partition adoption."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

from . import external_memory_dbscan as engine
from .aids_global_witness import source_binding
from .rf_aligned_pool import atomic_json, digest, file_sha

MODE = "aids_global_radius_witness_one_component_v1"


def adoption_memory_plan(pairs):
    components={'new_labels_and_core_pages':pairs['rows']*9,
                'static_ledger_and_runtime':1024**3,'buffers_owner_atomic':192*1024**2}
    if sum(components.values())>2*1024**3: raise ValueError('Partition stage exceeds 2GiB')
    return {'phase':'AIDS_WITNESS_PARTITION_ADOPTION','components_bytes':components,
            'charged_peak_bound_bytes':2*1024**3,'max_new_files':32,'gpu_requested':False,
            'full_vectors_scan_required':False,'summary_memory_admitted':False}


def verify_edges(vectors, binding, state):
    """Recompute only finite saved witness graph edges under the original kernel."""
    anchors, failures, seeds = binding['anchor_ids'], set(binding['failure_ids']), binding['seed_ids']
    if binding.get('seed_failure_ledger_complete') is not True:
        raise ValueError('Complete original seed/failure closure is required')
    rows=set(anchors)
    for records in state['witnesses'].values(): rows.update(r['row_id'] for r in records)
    rows.update(r['row_id'] for r in state['outside_failure_neighbor'].values())
    rows=sorted(rows)
    if any(not 0 <= i < len(vectors) for i in rows): raise ValueError('Witness row outside original universe')
    model, version=engine._fit_anchor_neighbors(np.asarray(vectors[anchors]),eps=binding['eps'])
    if version != binding['sklearn_version']: raise ValueError('Original kernel version differs')
    ds, ids=model.radius_neighbors(np.asarray(vectors[rows]),return_distance=True)
    edges={r:{anchors[int(a)]:float(d) for a,d in zip(neigh,dist)} for r,neigh,dist in zip(rows,ids,ds)}
    for a in anchors:
        records=state['witnesses'][str(a)]
        if len(records)<binding['min_samples'] or len({r['row_id'] for r in records}) != len(records):
            raise ValueError(f'Anchor {a} has insufficient distinct actual neighbors')
        for record in records:
            d=edges[record['row_id']].get(a)
            if d is None or d!=record['distance'] or d.hex()!=record['distance_hex']:
                raise ValueError(f'Exact witness distance differs: {a}/{record["row_id"]}')
        actual=sorted(b for b in edges[a] if b!=a)
        if actual != sorted(state['failure_edges'][str(a)]):
            raise ValueError(f'Saved anchor graph changed for {a}')
        outside=state['outside_failure_neighbor'].get(str(a))
        if outside and (outside['row_id'] in failures or outside['row_id']==a
                        or edges[outside['row_id']].get(a)!=outside['distance']):
            raise ValueError('False attachment to already-proven nonfailure component')
    seed_component={seeds[0]}
    for _ in seeds:
        seed_component.update(b for a in list(seed_component) for b in edges[a] if b in seeds)
    if not set(seeds)<=seed_component: raise ValueError('Original seeds not one exact component')
    connected=set(seeds)|{int(a) for a in state['outside_failure_neighbor']}
    for _ in anchors:
        connected.update(a for a in anchors if any(b in connected for b in edges[a]))
    if not failures<=connected: raise ValueError('Core anchors alone do not prove whole-universe connectivity')
    return {'status':'PASS','actual_witness_rows_replayed':len(rows),
        'all_points_core_proven':True,'single_epsilon_component_proven':True,
        'labels_are_exact_sklearn_order':True,'label_value':0,'core_mask_value':True,
        'approximation_used':False,'full_pair_store_regenerated':False,
        'proof':'all nonfailures attach to connected seeds; every failure has >=min_samples actual neighbors and actual path to seed component',
        'seed_failure_ledger_complete':True,'verified_anchor_count':len(anchors)}


def verify_source(config, recourse, evidence):
    from .rf_aligned_cluster_phase import sealed_pairs
    pairs=sealed_pairs(config,recourse)
    expected=source_binding(config,recourse,pairs)
    binding=json.loads((evidence/'witness_input_binding.json').read_text())
    if binding!=expected: raise ValueError('Witness input binding no longer closes')
    state=json.loads((evidence/'witness_checkpoint.json').read_text())
    sha=state.pop('checkpoint_sha256',None)
    if sha!=digest(state): raise ValueError('Witness checkpoint changed')
    key=digest({'binding':binding,'anchors':binding['anchor_ids'],'failures':binding['failure_ids'],
        'seeds':binding['seed_ids'],'eps':binding['eps'],'min_samples':binding['min_samples'],
        'kernel':'sklearn NearestNeighbors brute euclidean float32'})
    if state['binding_sha']!=key: raise ValueError('Witness semantic input digest mismatch')
    vectors=np.load(binding['vectors_path'],mmap_mode='r',allow_pickle=False)
    proof=verify_edges(vectors,binding,state)
    proof.update(witness_root=str(evidence),witness_checkpoint_sha256=sha,
        witness_binding_sha256=file_sha(evidence/'witness_input_binding.json'),
        pairs_manifest_sha256=pairs['manifest_sha256'],old_shortcut_state='INCONCLUSIVE',
        old_failed_dbscan_root=str(recourse/'dbscan'))
    return proof,pairs


def adopt(config, *, recourse_root, evidence_root):
    from .rf_aligned_cluster_phase import sealed_pairs, require_start_admission, PhaseObserver
    evidence=Path(config['witness_evidence_root'])
    pairs=sealed_pairs(config,recourse_root)
    plan=adoption_memory_plan(pairs)
    require_start_admission(config,plan,evidence_root)
    proof,pairs=verify_source(config,recourse_root,evidence)
    root=Path(config['witness_partition_root'])
    if root==recourse_root/'dbscan': raise ValueError('Original failed DBSCAN root is immutable')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'run_manifest.json').exists():
        return validate_partition(root/'run_manifest.json',expected_sha=file_sha(root/'run_manifest.json'))
    paths={}
    with PhaseObserver(config,plan,evidence_root) as observer:
        for name,dtype,value in [('labels',np.intp,0),('core_mask',np.bool_,True)]:
            path=root/(name+'.npy'); temp=root/(name+'.partial.npy')
            values=np.lib.format.open_memmap(temp,mode='w+',dtype=dtype,shape=(pairs['rows'],))
            for start in range(0,pairs['rows'],1048576):
                values[start:start+1048576]=value
            values.flush();del values
            temp.replace(path)
            paths[name+'_path']=str(path);paths[name+'_sha256']=file_sha(path)
            observer.boundary()
    identity=json.loads((recourse_root/'dbscan/checkpoint.json').read_text())['identity']
    proof.update(scientific_identity_sha256=engine._stable_hash(identity),num_samples=pairs['rows'],**paths)
    atomic_json(root/'global_witness_proof.json',proof)
    manifest={'schema_version':engine.SCHEMA_VERSION,'run_complete':True,'scientific_identity':identity,
        'scientific_identity_sha256':engine._stable_hash(identity),'num_samples':pairs['rows'],'num_features':64,
        'cluster_count':1,'core_count':pairs['rows'],'noise_count':0,'neighbor_counts_available':False,
        'clustering_path':MODE,'approximation_used':False,'shortcut_proof_path':str(root/'global_witness_proof.json'),
        'shortcut_proof_sha256':file_sha(root/'global_witness_proof.json'),'pairs_stat_identity':pairs['arrays']['pairs'],
        'witness_run_config':str(evidence_root.parent/'phase_run_manifest.json'),
        'independent_verification_performed':True,'labels_are_exact_sklearn_order':True,**paths}
    atomic_json(root/'run_manifest.json',manifest)
    terminal={'state':'EXACT_GLOBAL_WITNESS_PARTITION_COMPLETE','manifest':str(root/'run_manifest.json'),
        'manifest_sha256':file_sha(root/'run_manifest.json'),'cluster_count':1,'rows':pairs['rows'],
        'pair_rows_recomputed':0,'new_labels_have_full_certificate':True}
    atomic_json(evidence_root/'terminal.json',terminal)
    return terminal


def validate_partition(path, *, expected_sha):
    path=Path(path)
    if file_sha(path)!=expected_sha: raise ValueError('New partition manifest hash mismatch')
    m=json.loads(path.read_text())
    if m['clustering_path']!=MODE or not m['run_complete'] or m['cluster_count']!=1 or m['noise_count']!=0:
        raise ValueError('Wrong global witness partition schema')
    pp=Path(m['shortcut_proof_path'])
    if pp.parent!=path.parent or file_sha(pp)!=m['shortcut_proof_sha256']:
        raise ValueError('Independent witness proof missing')
    proof=json.loads(pp.read_text())
    config=json.loads(Path(m['witness_run_config']).read_text())
    verified,pairs=verify_source(config,Path(proof['old_failed_dbscan_root']).parent,Path(proof['witness_root']))
    if any(proof.get(k)!=v for k,v in verified.items()): raise ValueError('Independent proof semantic mismatch')
    for name,value,dtype in [('labels',0,np.intp),('core_mask',True,np.bool_)]:
        p=Path(m[name+'_path'])
        if p.parent!=path.parent or file_sha(p)!=m[name+'_sha256']: raise ValueError('Partition array hash mismatch')
        array=np.load(p,mmap_mode='r',allow_pickle=False)
        if array.shape!=(pairs['rows'],) or array.dtype!=dtype: raise ValueError('Partition array schema mismatch')
        for start in range(0,len(array),1048576):
            if not np.all(array[start:start+1048576]==value): raise ValueError('Noncanonical all-core partition')
    return m
