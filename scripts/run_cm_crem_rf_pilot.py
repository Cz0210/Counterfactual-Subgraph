#!/usr/bin/env python3
"""Bounded16-train RF adapter, real native generation and original RF checks."""
import argparse,csv,io,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.baselines.cm_crem_runtime import read_json,atomic_json,digest,require_compute_node,file_sha

def run(spec_path,action):
    require_compute_node();s=read_json(spec_path);root=Path(s['output_root']);root.mkdir(exist_ok=True)
    if not root.is_relative_to('/share/home/u20526/czx'):raise ValueError('HPC scope')
    if s['deadline_utc']!='2026-09-16T16:27:44Z':raise ValueError('Original deadline changed')
    if datetime.now(timezone.utc)>=datetime.fromisoformat(s['deadline_utc'].replace('Z','+00:00')):
        raise RuntimeError('Original absolute deadline reached')
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if actual!=s['execution_commit']:raise ValueError('Immutable RF adapter code differs')
    sha=digest({k:v for k,v in s.items() if k not in ('execution_commit','execution_root','output_root')})
    def put(name,value):return atomic_json(root/name,{'scope_sha':sha,**value},immutable=True)
    if action=='run':
        run(spec_path,'attribute')
        original=read_json(s['original_cm_spec'])
        env=dict(os.environ,PYTHONHASHSEED='0',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='')
        subprocess.run([original['execution']['generator_python'],'-s','-B',str(__file__),
            '--config','configs/hpc.yaml','--spec',str(spec_path),'--action','generate'],env=env,check=True)
        return run(spec_path,'verify')
    if action=='attribute':
        if (root/'attribution.json').exists():return
        from src.oracles.oracle_factory import LegacyRFOracle
        from src.baselines.cm_crem_rf_attribution import attribute
        if file_sha(s['rf_path'])!=s['rf_sha']:raise ValueError('RF actual weights differ')
        oracle=LegacyRFOracle(s['rf_path'],num_classes=2,source_label=1)
        text=Path(s['source_csv']).read_text();lines=list(csv.DictReader(io.StringIO(text)))
        if set(lines[0])!={'smiles','label'}:raise ValueError('Original train CSV schema needs mapping')
        parents=[{'parent_id':'AIDS_CM_train_'+digest([i,r['smiles']])[:16],'smiles':r['smiles'],'split':'train','input_row':i}
                 for i,r in enumerate(lines) if int(r['label'])==1]
        parents.sort(key=lambda p:p['parent_id'])
        records=[]
        for p in parents:
            if int(oracle.predict_proba([p['smiles']])[0].argmax())!=1:continue
            a=attribute(oracle,p);b=attribute(oracle,p)
            if a!=b:raise ValueError('RF attribution repeat differs')
            records.append(a)
            if len(records)==16:break
        if len(records)!=16:raise ValueError('Need16 actual train-source parents')
        put('attribution.json',{'records':records,'state':'RF_16_ATTRIBUTION_AND_REPEAT_PASS_NOT_COMPLETE_PILOT',
            'source_csv_sha':file_sha(s['source_csv']),'input_rows':len(lines),'test_read':False,
            'source_scope':s['source_scope'],'full32parent_pilot_completed':False})
    elif action=='generate':
        original=read_json(s['original_cm_spec']);e=original['execution']
        if Path(sys.executable).resolve()!=Path(e['generator_python']).resolve():raise ValueError('Wrong CReM isolated environment')
        from src.baselines.cm_crem_assets import prepare_job_scratch,stage_static_database,validate_database_source
        from src.baselines.cm_crem_generation import generate_parent
        receipt=read_json(e['database_receipt']);source=validate_database_source(original,receipt)
        scratch=prepare_job_scratch(root,required_bytes=receipt['uncompressed_bytes'],reserve_bytes=2*1024**3)
        put('scratch-'+os.environ['SLURM_JOB_ID']+'.json',scratch)
        if scratch['status']!='JOB_SCRATCH_READY':raise RuntimeError(scratch)
        staged=stage_static_database(e['database_path'],e['database_receipt'],reserve_bytes=2*1024**3,
            expected_url=source['actual_source_url'],scratch_receipt=scratch)
        put('database-'+os.environ['SLURM_JOB_ID']+'.json',staged)
        if staged['status']!='LOCAL_DATABASE_READY':raise RuntimeError(staged)
        for a in read_json(root/'attribution.json')['records']:
            name='generated/'+a['parent_id']+'.json'
            if (root/name).exists():continue
            (root/'logs').mkdir(exist_ok=True)
            value=generate_parent(a['generation_request'],{'database_path':staged['database_path'],
                'upstream_root':e['upstream_root'],'science_hash':sha,'parent_wall_limit_seconds':900},
                log_path=root/'logs'/('native-'+a['parent_id']+'.log'))
            put(name,value)
            if value['status'] not in {'GENERATED','GENERATION_COMPLETE','NO_NATIVE_REPLACEMENT','NO_REPLACEABLE_CONTEXT','TIMEOUT_BUDGETED'}:
                raise RuntimeError(value)
    elif action=='verify':
        from rdkit import Chem
        from src.oracles.oracle_factory import LegacyRFOracle
        oracle=LegacyRFOracle(s['rf_path'],num_classes=2,source_label=1);rows=[]
        for a in read_json(root/'attribution.json')['records']:
            g=read_json(root/'generated'/ (a['parent_id']+'.json'));valid=[];invalid=[]
            for r in g['retained_raw']:
                mol=Chem.MolFromSmiles(r['smiles'])
                if mol is None or len(Chem.GetMolFrags(mol))!=1:invalid.append(r['raw_id']);continue
                pred=oracle.predict_proba([r['smiles']])[0]
                valid.append({**r,'probabilities':pred.tolist(),'predicted_label':int(pred.argmax()),'strict_flip':int(pred.argmax())==0})
            rows.append({'parent_id':a['parent_id'],'generation_status':g['status'],'elapsed_seconds':g.get('elapsed_seconds',g.get('parent_wall_seconds')),
                'raw_count':len(g['retained_raw']),'valid':valid,'invalid_raw_ids':invalid})
        put('rf_adapter_receipt.json',{'state':'RF_FEATURE_OCCLUSION_16_REAL_GENERATION_CHECKED',
            'parents':16,'rows':rows,'test_read':False,'full_pilot_and_distance_timing':False,
            'next_required_stage':'32-parent dataset-specific pilot including >=64 original-WNode pairs',
            'method':'CM-CReM / RF-FeatureOcclusion','scope':s['source_scope'],'new_RF_training':False})

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',default='configs/hpc.yaml');p.add_argument('--spec',required=True,type=Path)
    p.add_argument('--action',choices=['run','attribute','generate','verify'],required=True);a=p.parse_args();run(a.spec,a.action)
