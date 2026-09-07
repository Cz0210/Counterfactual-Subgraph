"""One-time A+ input/oracle rebinding, reusing original mining and match IDs."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import torch
from rdkit import Chem
from src.baselines.bace_globalgce_aplus import (
    CONTRACT, GINAlignedBridge, build_parent, materialize,
)
from src.baselines.globalgce_bace_native_rules import GlobalGCENativeRule, enumerate_labeled_rule_matches
from src.eval.bace_frozen_gnn_contracts import atomic_json, stable_sha256, sha256_file, utc_now

def read_json(path):
    return json.loads(Path(path).read_text())

def read_rows(path):
    out=[]
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            pid=next((row[k] for k in ('sample_id','parent_id','molecule_id','id','compound_id') if row.get(k)),None)
            smiles=next((row[k] for k in ('model_smiles','canonical_smiles','smiles','SMILES') if row.get(k)),None)
            label=next((row[k] for k in ('label','target','Label') if k in row),None)
            if pid is None or smiles is None or label is None: raise ValueError('REAL_SPLIT_COLUMNS_REQUIRED')
            out.append({'id':pid,'smiles':smiles,'label':int(label)})
    if len({r['id'] for r in out})!=len(out): raise ValueError('DUPLICATE_SPLIT_IDS')
    return out

def setup(config, device):
    if config.get('joint_contract')!=CONTRACT or config.get('source_label')!=1 or config.get('target_label')!=0:
        raise ValueError('APLUS_CONTRACT_NOT_SEALED')
    if config.get('seed')!=7 or config.get('max_epochs')!=100 or config.get('test_access') is not False:
        raise ValueError('APLUS_SCOPE_CHANGED')
    for key in ('gnn_checkpoint','training_summary','source_manifest','train_csv','validation_csv','warmstart_checkpoint'):
        if not Path(config[key]).is_absolute(): raise ValueError('ABSOLUTE_INPUT_REQUIRED:'+key)
    if Path(config['train_csv']).name!='train.csv' or Path(config['validation_csv']).name!='val.csv':
        raise ValueError('TRAIN_VALIDATION_ONLY')
    summary, manifest=read_json(config['training_summary']),read_json(config['source_manifest'])
    if summary.get('test_loaded') is not False or manifest.get('test_loaded') is not False:
        raise ValueError('ORIGINAL_TRAIN_ONLY_SOURCE_REQUIRED')
    if manifest['oracle_checkpoint']!=config['warmstart_source_oracle']:
        raise ValueError('ORIGINAL_GINE_GENERATION_IDENTITY_CHANGED')
    codec=summary['codec_metadata']
    atoms=tuple(codec['node_label_mapping'][str(i)] for i in range(1,len(codec['node_label_mapping'])))
    bonds=tuple(codec['edge_label_mapping'][str(i)] for i in range(len(codec['edge_label_mapping'])))
    rules=torch.load(summary['rules_checkpoint'],map_location='cpu',weights_only=False)
    templates=[GlobalGCENativeRule(f'aplus-rule-{i:04d}',i,rules['feat'][i],rules['adj'][i],rules['edge_attr'][i],
        rules['feat'][i],rules['adj'][i],rules['edge_attr'][i],atoms,bonds) for i in range(len(rules['feat']))]
    if len(templates)!=80: raise ValueError('ORIGINAL_MINED_LHS80_REQUIRED')
    for r in templates: r.validate()
    train_all,val_all=read_rows(config['train_csv']),read_rows(config['validation_csv'])
    if {r['id'] for r in train_all}&{r['id'] for r in val_all}: raise ValueError('SPLIT_OVERLAP')
    # Preserve true-label source definition, but not the old GINE prediction
    # prefilter: GIN eligibility is computed below, never copied from old rows.
    train=[r for r in train_all if r['label']==1]; val=[r for r in val_all if r['label']==1]
    if len(train)!=386 or len(val)!=98: raise ValueError('FROZEN_TRUE_LABEL_COHORT_CHANGED')
    bridge=GINAlignedBridge.from_checkpoint(config['gnn_checkpoint'],atom_symbols=atoms,bond_names=bonds,device=device)
    if bridge.checkpoint_id!=config['gin_model_sha256'] or bridge.temperature!=config['gin_temperature']:
        raise ValueError('FROZEN_CORRECTED_GIN_CHANGED')
    return summary,rules,templates,train,val,bridge

def predict_smiles(bridge,smiles):
    from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer
    graph=MolecularGraphFeaturizer(bridge.feature_schema).featurize(smiles)
    with torch.no_grad():
        logits=bridge.model(x=torch.tensor(graph.node_features,dtype=torch.long,device=bridge.device),
            edge_index=torch.tensor(graph.edge_index,dtype=torch.long,device=bridge.device),
            edge_attr=torch.tensor(graph.edge_features,dtype=torch.long,device=bridge.device))
        prob=(logits/bridge.temperature).softmax(-1)[0]
    return {'predicted_label':int(logits.argmax(-1)), 'probabilities':prob.cpu().tolist(),
        'raw_logits':logits[0].cpu().tolist(),'oracle_backbone':'gin'}

def positive_control(row, templates, bridge):
    """Fixed first train parent/first nonaromatic double-bond reduction fixture.

    No oracle-dependent choice or candidate adoption. This is a known chemically
    legal interface control, not a generated recourse or additional search.
    """
    p=build_parent(row['smiles'],atom_symbols=templates[0].atom_symbols,bond_names=templates[0].bond_names)
    source=Chem.MolFromSmiles(row['smiles'])
    eligible=[b for b in source.GetBonds() if b.GetBondType()==Chem.BondType.DOUBLE and not b.GetIsAromatic()]
    if not eligible: raise ValueError('FIXED_POSITIVE_CONTROL_NO_DOUBLE_BOND')
    bond=eligible[0]; left,right=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
    from src.baselines.globalgce_bace_native_rules import _edge_position
    states=p.edge_attr.clone(); states[_edge_position(left,right)]=0
    states[_edge_position(left,right),templates[0].bond_names.index('single')]=1
    rule=GlobalGCENativeRule('positive-control-not-candidate',0,p.feature,p.adjacency,p.edge_attr,
        p.feature,p.adjacency,p.edge_attr,templates[0].atom_symbols,templates[0].bond_names)
    product=materialize(p,rule,{i:i for i in range(len(p.feature))},p.feature,states)
    actual=bridge.score_materialized(product)
    expected=predict_smiles(bridge,product.canonical_smiles)
    if not torch.allclose(actual['logits'].detach().cpu()[0],torch.tensor(expected['raw_logits']),atol=2e-6,rtol=0):
        raise ValueError('POSITIVE_CONTROL_HARD_GIN_PARITY_FAILED')
    return {'state':'PASS','fixture_kind':'fixed_train_bond_reduction_interface_control_not_candidate',
        'parent_id':row['id'],'parent_smiles':row['smiles'],'bond_indices':[left,right],
        'complete_product':product.canonical_smiles,'actual':expected,
        'candidate_adopted':False,'target_flip_claimed':False}

def prepare(config_path,output_root):
    config=read_json(config_path); output=Path(output_root)
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'repair_contract.json',config)
    summary,rules,templates,train,val,bridge=setup(config,'cpu')
    old_config=read_json(config['warmstart_source_config'])
    checkpoint=torch.load(config['warmstart_checkpoint'],map_location='cpu',weights_only=False)
    if checkpoint['epoch_completed']!=35 or checkpoint['config_sha256']!=stable_sha256(old_config):
        raise ValueError('REAL_EPOCH35_SOURCE_BINDING_FAILED')
    if old_config['gnn_checkpoint']!=config['warmstart_source_oracle'] or config['warmstart_source_oracle']==config['gnn_checkpoint']:
        raise ValueError('WARMSTART_MUST_DISCLOSE_GINE_TO_GIN')
    digest=sha256_file(config['warmstart_checkpoint'])
    if digest!=config['warmstart_checkpoint_sha256']: raise ValueError('PAUSED_CHECKPOINT_CHANGED')
    bindings={}; fixtures=[]
    for role,parents in (('train',train),('validation',val)):
        byid={r['id']:r for r in parents}
        predictions={r['id']:predict_smiles(bridge,r['smiles']) for r in parents}
        old_path=Path(config['old_match_root'])/(role+'_index.jsonl')
        old_ids=set(); counts={'reused_mappings':0,'new_parent_mappings':0}; old_digest=hashlib.sha256()
        temporary=output/(role+'_index.jsonl.partial')
        with temporary.open('w') as dest, old_path.open('rb') as source:
            for raw in source:
                old_digest.update(raw); item=json.loads(raw); pid=item['parent_id']
                if pid not in byid: raise ValueError('OLD_MAPPING_OUTSIDE_TRUE_SOURCE_SPLIT')
                rid=int(item['rule_index'])
                if not 0<=rid<80: raise ValueError('OLD_MAPPING_RULE_OUT_OF_RANGE')
                old_ids.add(pid); item['before']=predictions[pid]
                item['old_oracle_prediction_adopted']=False
                dest.write(json.dumps(item,separators=(',',':'))+'\n'); counts['reused_mappings']+=1
            # Only previously GINE-excluded true-label train parents are new.
            # Original 80×360/98 rematerialization is never repeated.
            for row in parents:
                if row['id'] in old_ids: continue
                p=build_parent(row['smiles'],atom_symbols=templates[0].atom_symbols,bond_names=templates[0].bond_names)
                for rid,template in enumerate(templates):
                    for mapping in enumerate_labeled_rule_matches(p,template):
                        item={'parent_id':row['id'],'rule_index':rid,'mapping':list(mapping.items()),
                            'before':predictions[row['id']], 'old_oracle_prediction_adopted':False}
                        dest.write(json.dumps(item,separators=(',',':'))+'\n');counts['new_parent_mappings']+=1
            dest.flush();os.fsync(dest.fileno())
        target=output/(role+'_index.jsonl');os.replace(temporary,target)
        bindings[role]={'true_label_source_parents':len(parents),
            'gin_source_parents':sum(p['predicted_label']==1 for p in predictions.values()),
            'old_mapping_sha256':old_digest.hexdigest(),'index_sha256':sha256_file(target),
            'new_parent_count':len(set(byid)-old_ids),**counts}
        atomic_json(output/(role+'_parent_predictions.json'),predictions)
        # Fixed first source parent/matching LHS is an engineering identity
        # witness, not a performance-dependent choice or generated recourse.
        if role=='train':
            with target.open() as stream:
                for line in stream:
                    item=json.loads(line); row=byid[item['parent_id']]; template=templates[item['rule_index']]
                    p=build_parent(row['smiles'],atom_symbols=template.atom_symbols,bond_names=template.bond_names)
                    product=materialize(p,template,dict(item['mapping']),template.lhs_feature,template.lhs_edge_attr)
                    expected=predictions[row['id']];actual=predict_smiles(bridge,product.canonical_smiles)
                    if product.canonical_smiles!=p.canonical_smiles or not torch.allclose(
                        torch.tensor(actual['raw_logits']),torch.tensor(expected['raw_logits']),atol=2e-6,rtol=0):
                        raise ValueError('REAL_TRAIN_IDENTITY_ORACLE_FAILED')
                    fixtures.append({'fixture_kind':'real_train_LHS_identity_not_recourse','parent_id':row['id'],
                        'parent_smiles':row['smiles'],'rule_index':item['rule_index'],'mapping':item['mapping'],
                        'complete_product':product.canonical_smiles,'boundary_count':product.boundary_count,
                        'expected':expected,'actual':actual,'state':'PASS','target_flip_claimed':False})
                    if len(fixtures)==4: break
        atomic_json(output/'heartbeat.json',{'stage':'INPUT_BINDING','role':role,'pid':os.getpid(),'time':utc_now()})
    control=positive_control(train[0],templates,bridge)
    result={'state':'INPUTS_BOUND','config_sha256':stable_sha256(config),'bindings':bindings,
        'warmstart_mode':'WEIGHTS_ONLY_GINE_EPOCH35_TO_GIN_NEW_CAMPAIGN',
        'warmstart_checkpoint_sha256':digest,'old_epoch35_ledger_modified':False,
        'new_optimizer_reset':True,'new_epoch_start':0,'identity_fixtures':fixtures,'positive_control':control,
        'test_loaded':False,'calibration_loaded':False,'mining_rerun':False,'generator_inference':False,
        'created_at':utc_now()}
    atomic_json(output/'terminal.json',result)
    return result
