#!/usr/bin/env python3
"""Bounded native controls for existing AIDS single-cut repair, no new pilot."""
import argparse,json,random,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.baselines.cm_crem_runtime import atomic_json,read_json,digest,require_compute_node

def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--source-root',required=True);p.add_argument('--diagnosis-root',required=True);p.add_argument('--output-root',required=True);a=p.parse_args();require_compute_node()
 root=Path(a.output_root)
 if root.exists():raise ValueError('FRESH_CONTROL_ROOT_REQUIRED')
 if not root.is_relative_to('/share/home/u20526/czx'):raise ValueError('HPC_SCOPE')
 root.mkdir(parents=True,mode=0o700);source=Path(a.source_root);spec=read_json(source/'spec.json');old=read_json(spec['original_cm_spec']);e=old['execution']
 from datetime import datetime,timezone
 if datetime.now(timezone.utc)>=datetime.fromisoformat(spec['deadline_utc'].replace('Z','+00:00')):raise RuntimeError('Original deadline')
 if Path(sys.executable).resolve()!=Path(e['generator_python']).resolve():raise ValueError('Original generation environment required')
 from src.baselines.cm_crem_assets import prepare_job_scratch,stage_static_database,validate_database_source
 receipt=read_json(e['database_receipt']);origin=validate_database_source(old,receipt)
 scratch=prepare_job_scratch(root,required_bytes=receipt['uncompressed_bytes'],reserve_bytes=2*1024**3)
 if scratch['status']!='JOB_SCRATCH_READY':raise RuntimeError(scratch)
 staged=stage_static_database(e['database_path'],e['database_receipt'],reserve_bytes=2*1024**3,expected_url=origin['actual_source_url'],scratch_receipt=scratch)
 if staged['status']!='LOCAL_DATABASE_READY':raise RuntimeError(staged)
 atomic_json(root/'scratch.json',staged,immutable=True)
 import crem.crem as native
 from rdkit import Chem
 from src.baselines.cm_crem_fragment_repair import install
 original=native.__fragment_mol
 args=dict(radius=1,min_inc=0,max_inc=0,max_replacements=4,replace_ids=[0],symmetry_fixes=True,ncores=1)
 start=time.monotonic();random.seed(7)
 positive_before=list(native.mutate_mol(Chem.MolFromSmiles('CCO'),staged['database_path'],**args))
 attrs=read_json(source/'attribution.json');completed=[]
 for row in attrs['records']:
  path=source/'generated'/(row['parent_id']+'.json')
  if path.exists() and read_json(path).get('status') in {'GENERATED','GENERATION_COMPLETE','NO_NATIVE_REPLACEMENT','NO_REPLACEABLE_CONTEXT','TIMEOUT_BUDGETED'}:completed.append(row)
 completed=sorted(completed,key=lambda r:r['parent_id'])[:2]
 if len(completed)!=2:raise ValueError('Two actual saved completed inputs required')
 comparisons=[]
 for row in completed:
  req=row['generation_request'];smiles=req.get('parent_smiles',req.get('smiles'))
  if not smiles:raise ValueError('Actual request SMILES field unavailable')
  mol=Chem.MolFromSmiles(smiles)
  if mol is None:raise ValueError('Saved input parse failure')
  before=list(original(mol,radius=1,symmetry_fixes=True))
  comparisons.append(dict(parent_id=row['parent_id'],input_sha256=digest(req),smiles=smiles,before=before))
 patch=install(native);random.seed(7)
 positive_after=list(native.mutate_mol(Chem.MolFromSmiles('CCO'),staged['database_path'],**args))
 if not positive_before or positive_before!=positive_after:raise ValueError('AUTHOR_POSITIVE_REPLACEMENT_CHANGED_OR_EMPTY')
 for row in comparisons:
  after=list(native.__fragment_mol(Chem.MolFromSmiles(row.pop('smiles')),radius=1,symmetry_fixes=True))
  if row['before']!=after:raise ValueError('SAVED_COMPLETED_NATIVE_FRAGMENTATION_DIFF:'+row['parent_id'])
  row['fragment_count']=len(after);row['fragmentation_sha256']=digest(after);row.pop('before');row['exact']=True
 d=read_json(Path(a.diagnosis_root)/'diagnosis.json');r=d['result']
 if r['status']!='NO_NATIVE_REPLACEMENT' or r.get('retained_raw'):raise ValueError('Prior corrected empty terminal not legitimate empty')
 atomic_json(root/'controls.json',dict(status='NATIVE_POSITIVE_AND_SAVED_INPUT_CONTROLS_PASS',repair=patch,
  positive_fixture='CCO',positive_settings=args,positive_native=positive_before,positive_repaired=positive_after,
  completed_inputs= comparisons,completed_science_regenerated=False,
  corrected_parent=dict(parent_id=d['parent_id'],classification='VERIFIED_NO_REPLACEMENT',source_diagnosis=str(Path(a.diagnosis_root)/'diagnosis.json'),source_request_sha256=d['source_request_sha256'],prior_original_error='KeyError empty attachment at zero database queries; preserved prior traceback'),
  elapsed_seconds=time.monotonic()-start,full32_pilot_complete=False,scientific_pass_claimed=False),immutable=True)
 print('NATIVE_POSITIVE_AND_SAVED_INPUT_CONTROLS_PASS')
if __name__=='__main__':main()
