"""Sealed own-backbone raw-cost extensions; no inference, OT or scheduler."""
from __future__ import annotations
import copy
import math
from pathlib import Path

from src.ablations.gnn.reach_v2_adapter import BACKBONES, SCOPE_NAME, split_chunk_size
from src.ablations.gnn.reach_v2_closeout import verify_own_match_minima
from src.ablations.gnn.reach_raw_distance_reuse import SCHEMA, graph_key
from src.eval.bace_frozen_gnn_contracts import atomic_json,read_json,sha256_file,stable_sha256


def resolve_calibration_index(spec, spec_sha, pool_sha, backbone, extension=None):
    """Only a complete preceding family may add raw costs for the next one."""
    base=spec['raw_cost_indexes']['calibration']
    if extension is None:
        if sha256_file(base['path'])!=base['sha256']:raise ValueError('RAW_BASE_INDEX_CHANGED')
        return read_json(base['path']),base['sha256']
    path=Path(extension).resolve(strict=True)
    path.relative_to(Path(spec['output_root']).resolve()/'raw_extensions')
    row=read_json(path)
    expected=list(BACKBONES[:BACKBONES.index(backbone)])
    if (row.get('state')!='OWN_BACKBONE_CALIBRATION_RAW_EXTENSION_SEALED'
            or row.get('spec_sha256')!=spec_sha or row.get('pool_sha256')!=pool_sha
            or row.get('completed_backbones')!=expected
            or row.get('base_index_file_sha256')!=base['sha256']
            or row.get('self_sha256')!=stable_sha256({k:v for k,v in row.items() if k!='self_sha256'})):
        raise ValueError('RAW_EXTENSION_COMPLETED_FAMILY_BINDING_CONFLICT')
    index=row['index']
    if index.get('schema')!=SCHEMA or index.get('split')!='calibration':raise ValueError('RAW_EXTENSION_SPLIT_CONFLICT')
    return index,sha256_file(path)


def extend_calibration_costs(spec, *, spec_sha, pool_sha, candidates, backbone, extension=None):
    """Use all own calibrated rows only after that entire family completed."""
    index,source_sha=resolve_calibration_index(spec,spec_sha,pool_sha,backbone,extension)
    if index.get('self_sha256')!=stable_sha256({k:v for k,v in index.items() if k!='self_sha256'}):
        raise ValueError('RAW_EXTENSION_SOURCE_INDEX_SELF_HASH')
    root=Path(spec['output_root']);target=root/'raw_extensions'/f'{backbone}.json'
    if target.exists():
        existing=read_json(target)
        if (existing.get('spec_sha256')!=spec_sha or existing.get('input_index_sha256')!=source_sha
                or existing.get('self_sha256')!=stable_sha256({k:v for k,v in existing.items() if k!='self_sha256'})):
            raise ValueError('IMMUTABLE_RAW_EXTENSION_CONFLICT')
        return existing
    values=copy.deepcopy(index['graph_costs']);native=None;seen=set();source_files={};finite=0
    size=split_chunk_size(spec,'calibration');ids=[c['candidate_id'] for c in candidates]
    for chunk in range(spec['slots']['calibration']):
        directory=root/backbone/'calibration'/f'{chunk:04d}';terminal=read_json(directory/'terminal.json')
        expected=dict(state='PARENT_CHUNK_COMPLETE_NOT_CORE_PASS',scope=SCOPE_NAME,spec_sha256=spec_sha,
            pool_sha256=pool_sha,backbone=backbone,split='calibration',index=chunk,
            model_files=spec['model_files'][backbone],global_selector_called=False,main_matrix_write=False)
        if any(terminal.get(k)!=v for k,v in expected.items()) or type(terminal['index']) is not int:
            raise ValueError('RAW_EXTENSION_INCOMPLETE_FAMILY_TERMINAL')
        declared=terminal['native_cohort_ids']
        if declared!=sorted(set(declared)) or native is not None and declared!=native:
            raise ValueError('RAW_EXTENSION_COHORT_CONFLICT')
        native=declared;expected_ids=native[chunk*size:(chunk+1)*size]
        if terminal['parent_ids']!=expected_ids:raise ValueError('RAW_EXTENSION_PARTITION_CONFLICT')
        chunk_seen=set();pair_count=0
        source_files[str((directory/'terminal.json').relative_to(root))]=sha256_file(directory/'terminal.json')
        for path in sorted((directory/'parents').glob('*.json')):
            saved=read_json(path);science=saved['science']
            if (saved.get('spec_sha256')!=spec_sha or saved.get('pool_sha256')!=pool_sha
                    or saved.get('backbone')!=backbone or saved.get('science_sha256')!=stable_sha256(science)):
                raise ValueError('RAW_EXTENSION_PARENT_BINDING_CONFLICT')
            verify_own_match_minima(science,candidate_ids=ids,model_sha=spec['model_files'][backbone]['model.pt'])
            parent_ids={r['parent_id'] for r in science['pair_rows']}
            if len(parent_ids)!=1 or chunk_seen.intersection(parent_ids) or seen.intersection(parent_ids):
                raise ValueError('RAW_EXTENSION_DUPLICATE_PARENT')
            chunk_seen.update(parent_ids);pair_count+=len(science['pair_rows'])
            digest=sha256_file(path);source_files[str(path.relative_to(root))]=digest
            for row in science['match_rows']:
                if row.get('distance_ok') is not True:continue
                d=row['wnode_distance']
                if (isinstance(d,bool) or not isinstance(d,(int,float)) or not math.isfinite(d) or d<0
                        or any(row.get(k) is not True for k in ('delete_valid','sanitize_ok','residual_connected'))):
                    raise ValueError('RAW_EXTENSION_INVALID_FINITE_RECORD')
                key,parent,residual=graph_key(row['parent_smiles'],row['residual_smiles'],index['raw_contract_sha256'])
                if key in values and values[key]['distance']!=d:raise ValueError('RAW_EXTENSION_DIFFERENT_GRAPH_COST')
                record=values.setdefault(key,dict(parent=parent,residual=residual,distance=d,source_records=[]))
                record['source_records'].append(dict(source_parent_member=str(path),source_parent_sha256=digest,
                    source_match_sha256=stable_sha256(row),original_action_context={k:row[k] for k in
                    ('parent_id','candidate_id','match_index','match_atom_indices','oracle_checkpoint_hash','action_semantics_version')}))
                finite+=1
        if chunk_seen!=set(expected_ids) or pair_count!=terminal['pair_count']:
            raise ValueError('RAW_EXTENSION_MISSING_PARENT')
        seen.update(chunk_seen)
    if seen!=set(native):raise ValueError('RAW_EXTENSION_FAMILY_OMITS_PARENTS')
    derived=copy.deepcopy(index);derived.pop('self_sha256')
    derived.update(graph_costs=values,raw_cost_count=len(values),
        source_parent_units=index['source_parent_units']+len(seen),
        source_finite_match_records=index['source_finite_match_records']+finite,
        source_spec=dict(previous_index_sha256=source_sha,completed_backbone=backbone,
            current_execution_spec_sha256=spec_sha,current_pool_sha256=pool_sha,completed_parent_files=source_files))
    derived['binding_sha256']=stable_sha256(derived['source_spec']);derived['self_sha256']=stable_sha256(derived)
    result=dict(state='OWN_BACKBONE_CALIBRATION_RAW_EXTENSION_SEALED',spec_sha256=spec_sha,pool_sha256=pool_sha,
        completed_backbones=list(BACKBONES[:BACKBONES.index(backbone)+1]),
        base_index_file_sha256=spec['raw_cost_indexes']['calibration']['sha256'],input_index_sha256=source_sha,
        completed_parent_count=len(seen),source_files=source_files,index=derived,
        original_index_unchanged=True,model_inference_performed=False,ot_recomputed=0,
        source_flip_masks_reused=False,source_match_minima_reused=False,main_matrix_write=False)
    result['self_sha256']=stable_sha256(result);atomic_json(target,result)
    return result
