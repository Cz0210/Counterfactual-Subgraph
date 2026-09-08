"""Narrow LLM-GIN source conflict repair, never an alternative OT metric.

Historical source graphs identify a pair, but do not bind actual embedding
inputs or execution. Conflicting records are retained as provenance-incomplete
historical namespaces. Only their pair closure receives a new measurement from
the existing CPU migration producer. Uncontested sealed costs remain adopted.
"""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

from src.experiments import bace_eval_migration as m


def overlay_path(spec, root):
    root=Path(root).resolve()
    root.relative_to(Path(spec['output_root']).resolve().parent)
    if root==Path(spec['output_root']).resolve() or spec['family']!='llm_gin':
        raise ValueError('FRESH_LLM_RECONCILIATION_ROOT_REQUIRED')
    return root


def scan(spec, manifest):
    """One source-index pass per split, no model inference, no active DB reads."""
    from src.ablations.gnn.reach_raw_distance_reuse import graph_key
    config=SimpleNamespace(**manifest['wnode_config'])
    result={}
    for split in ('calibration','test'):
        conflicts={}
        wrapper=m.load_raw(spec,split,SimpleNamespace(config=config),manifest,conflicts=conflicts)
        usage=[]
        # Existing calibration units alone determine whether a repair propagates.
        for role in spec['roles']:
            for path in sorted((Path(spec['output_root'])/role/split).glob('parent-*.json')):
                record=m.reopen(path)
                for row in record['match_rows']:
                    if not row.get('distance_ok'):continue
                    key,_,_=graph_key(row['parent_smiles'],row['residual_smiles'],wrapper.index['raw_contract_sha256'])
                    if key in conflicts:
                        usage.append(dict(pair_key=key,role=role,split=split,
                            parent_id=record['parent_id'],unit_path=str(path),
                            candidate_id=row['candidate_id'],match_atom_indices=row['match_atom_indices']))
        result[split]=dict(index=wrapper.index,conflicts=conflicts,existing_unit_usage=usage)
    return result


def classify(scan_result):
    cal=scan_result['calibration']
    if cal['conflicts']:
        # Never keep an old selector solely because delta is small. The broader
        # calibration repair needs an explicit affected reduction implementation.
        raise ValueError('CALIBRATION_CONFLICT_REDUCTION_REQUIRED:'+str(len(cal['conflicts'])))
    return dict(classification='HISTORICAL_NUMERICAL_PRODUCER_INPUTS_INCOMPLETE',
        pair_counts={s:len(v['conflicts']) for s,v in scan_result.items()},
        calibration_values_changed=False,selector_replay_required=False,
        historical_numerical_equality_claimed=False)


def array_proof(value):
    import numpy as np
    a=np.ascontiguousarray(value)
    return dict(dtype=str(a.dtype),shape=list(a.shape),strides=list(a.strides),
        sha256=hashlib.sha256(a.tobytes()).hexdigest())


def graph_proof(smiles):
    from rdkit import Chem
    from src.eval.molclr_node_embeddings import smiles_to_molclr_data
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:raise ValueError('INVALID_CONFLICT_GRAPH')
    graph=dict(atoms=[dict(index=a.GetIdx(),element=a.GetAtomicNum(),isotope=a.GetIsotope(),
        charge=a.GetFormalCharge(),aromatic=a.GetIsAromatic(),chiral=int(a.GetChiralTag()),
        explicit_h=a.GetNumExplicitHs(),implicit_h=a.GetNumImplicitHs(),radical=a.GetNumRadicalElectrons())
        for a in mol.GetAtoms()],bonds=[dict(begin=b.GetBeginAtomIdx(),end=b.GetEndAtomIdx(),
        type=str(b.GetBondType()),aromatic=b.GetIsAromatic(),stereo=int(b.GetStereo()),
        stereo_atoms=list(b.GetStereoAtoms()),direction=int(b.GetBondDir())) for b in mol.GetBonds()])
    data=smiles_to_molclr_data(smiles)
    tensors={name:array_proof(getattr(data,name).cpu().numpy()) for name in ('x','edge_index','edge_attr')}
    return dict(canonical_smiles=smiles,attributed_graph=graph,
        attributed_graph_sha256=m.stable_sha256(graph),actual_input_tensors=tensors)


def producer_contract(manifest):
    import numpy as np
    import torch
    import ot
    from rdkit import rdBase
    from src.ablations.gnn.reach_raw_distance_reuse import KERNELS,raw_contract_from_bundle
    repo=Path(__file__).resolve().parents[2]
    return dict(raw_contract=raw_contract_from_bundle(manifest),device='cpu',
        embedding_batch_graphs=1,embedding_dtype='float32',
        cost_dtype='float64',mass_dtype='float64',solver='exact_emd2',
        torch=str(torch.__version__),numpy=np.__version__,pot=ot.__version__,rdkit=rdBase.rdkitVersion,
        threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        kernels={p:m.sha256_file(repo/p) for p in KERNELS},
        contract_selection='EXISTING_FROZEN_MIGRATION_CPU_PRODUCER_NOT_SELECTED_USING_RESULTS')


def measure(delegate, entry, producer, path):
    import numpy as np
    from src.eval.node_wasserstein_distance import (compute_node_wasserstein_distance,
        cosine_node_cost_matrix,uniform_node_mass)
    identity=dict(parent=graph_proof(entry['parent']),residual=graph_proof(entry['residual']),
        direction='parent_to_residual')
    binding=m.stable_sha256(dict(pair=identity,numerical_contract=producer))
    if path.exists():
        record=m.reopen(path)
        if record['binding']!=binding:raise ValueError('CANONICAL_PRODUCER_RESUME_DRIFT')
        return record
    left=delegate.embedder.get(entry['parent']);right=delegate.embedder.get(entry['residual'])
    cfg=delegate.config
    value,metadata=compute_node_wasserstein_distance(left.H,right.H,
        feature_cost=cfg.feature_cost,node_mass=cfg.node_mass,size_penalty_beta=cfg.size_penalty_beta)
    if not np.isfinite(value) or value<0:raise ValueError('CANONICAL_RAW_DISTANCE_INVALID')
    # Actual complete numeric input is preserved once for this affected pair.
    sidecar=path.with_suffix('.npz')
    with sidecar.open('xb') as f:
        np.savez_compressed(f,left_H=left.H,right_H=right.H,left_atoms=left.atom_numbers,
            right_atoms=right.atom_numbers,cost=cosine_node_cost_matrix(left.H,right.H),
            left_mass=uniform_node_mass(len(left.H)),right_mass=uniform_node_mass(len(right.H)))
    return m.seal(path,dict(binding=binding,pair_identity=identity,numerical_contract=producer,
        numerical_contract_sha256=m.stable_sha256(producer),distance=value,metadata=metadata,
        raw_input_sidecar=str(sidecar),raw_input_sha256=m.sha256_file(sidecar),
        left_embedding=array_proof(left.H),right_embedding=array_proof(right.H),
        historical_observations=entry['observations'],
        old_producer_inputs_available=False,new_measurement_proves_old_producer_correct=False))


def reconcile(spec, root):
    """Run inside the existing job's exclusive writer lease on a CPU node."""
    import os
    import torch
    from src.experiments.bace_gin_ours import original_bundle
    from src.ablations.gnn.cpu_evaluation import _distance
    from src.ablations.llm.compact_node_cache import install_compact_node_cache
    root=overlay_path(spec,root);root.mkdir(parents=True,exist_ok=True)
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('RECOMPUTE_COMPUTE_NODE_ONLY')
    frozen=m.require_freeze(spec)
    binding=dict(spec_sha256=m.stable_sha256(spec),freeze_sha256=frozen['self_sha256'])
    final=root/'reconciliation.json'
    if final.exists():
        receipt=m.reopen(final)
        if receipt['binding']!=binding:raise ValueError('RECONCILIATION_BINDING_CHANGED')
        return m.bound(receipt['test_index'])
    bundle,manifest=original_bundle(spec)
    calibration_units=sum(len(m.role_records(spec,role,'calibration')) for role in spec['roles'])
    if calibration_units!=264:raise ValueError('CALIBRATION_PARTITION_INCOMPLETE')
    result=scan(spec,manifest)
    summary=classify(result)
    m.seal(root/'scan.json',dict(binding=binding,**summary,
        conflicts={s:v['conflicts'] for s,v in result.items()},
        existing_unit_usage={s:v['existing_unit_usage'] for s,v in result.items()}))
    torch.set_num_threads(spec['cpu_threads'])
    producer=producer_contract(manifest)
    m.seal(root/'producer.json',producer) # Before any new distance is observed.
    index=copy.deepcopy(result['test']['index']);measured={}
    delegate=_distance(bundle,manifest,root/'canonical_raw_cache')
    install_compact_node_cache(delegate)
    thresholds=m.bound(spec['thresholds'])
    try:
        for key,entry in sorted(result['test']['conflicts'].items()):
            record=measure(delegate,entry,producer,root/(key+'.json'))
            measured[key]=dict(path=str(root/(key+'.json')),sha256=m.sha256_file(root/(key+'.json')))
            prior=index['graph_costs'][key]
            canonical_key=m.stable_sha256(dict(pair_identity=record['pair_identity'],numerical_contract=producer))
            index['graph_costs'][key]=dict(prior,distance=record['distance'],canonical_pair_key=canonical_key,
                numerical_contract_sha256=record['numerical_contract_sha256'],
                source_records=prior['source_records']+[dict(canonical_reconciliation=measured[key],
                    pair_identity_sha256=m.stable_sha256(record['pair_identity']),
                    numerical_contract_sha256=record['numerical_contract_sha256'])])
    finally:delegate.close()
    index['canonical_reconciliation']=dict(binding=binding,numerical_contract_sha256=m.stable_sha256(producer),pairs=measured)
    index['self_sha256']=m.stable_sha256({k:v for k,v in index.items() if k!='self_sha256'})
    m.atomic_json(root/'test_index.json',index)
    m.seal(final,dict(binding=binding,state='RAW_SOURCE_CONFLICT_RECONCILED',**summary,
        affected_pairs_recomputed=len(measured),calibration_units_reused=calibration_units,freeze_changed=False,
        historical_values_preserved=True,original_sources_modified=False,main_matrix_write=False,
        test_index=dict(path=str(root/'test_index.json'),sha256=m.sha256_file(root/'test_index.json')),
        numerical_producer_sha256=m.stable_sha256(producer),threshold_contract_sha256=m.stable_sha256(thresholds)))
    return index
