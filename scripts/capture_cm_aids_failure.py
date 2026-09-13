#!/usr/bin/env python3
"""Replay only one saved failed CM-AIDS request, retaining full native traceback."""
import argparse, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines.cm_crem_runtime import read_json, atomic_json, digest, require_compute_node, checked_root

def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('--config', required=True)
    p.add_argument('--source-root', required=True); p.add_argument('--output-root', required=True)
    p.add_argument('--parent-id', required=True); a=p.parse_args(); require_compute_node()
    root=checked_root(a.output_root, '/share/home/u20526/czx'); root.mkdir(mode=0o700, parents=True, exist_ok=True)
    source=Path(a.source_root); spec=read_json(source/'spec.json')
    from datetime import datetime, timezone
    if datetime.now(timezone.utc)>=datetime.fromisoformat(spec['deadline_utc'].replace('Z','+00:00')): raise RuntimeError('Deadline reached')
    attrs=read_json(source/'attribution.json'); records=[r for r in attrs['records'] if r['parent_id']==a.parent_id]
    if len(records)!=1: raise ValueError('Failed request not uniquely found')
    original=read_json(source/'generated'/(a.parent_id+'.json'))
    if original['status']!='ENGINEERING_FAILED': raise ValueError('Only failed units may be replayed')
    expected=digest({k:v for k,v in spec.items() if k not in ('execution_commit','execution_root','output_root')})
    if attrs['scope_sha']!=expected: raise ValueError('Attribution binding changed')
    from src.baselines.cm_crem_assets import prepare_job_scratch, stage_static_database, validate_database_source
    from src.baselines.cm_crem_generation import generate_parent
    old=read_json(spec['original_cm_spec']); e=old['execution']; receipt=read_json(e['database_receipt'])
    if Path(sys.executable).resolve()!=Path(e['generator_python']).resolve(): raise ValueError('Wrong generator environment')
    dbsource=validate_database_source(old,receipt)
    scratch=prepare_job_scratch(root,required_bytes=receipt['uncompressed_bytes'],reserve_bytes=2*1024**3)
    if scratch['status']!='JOB_SCRATCH_READY': raise RuntimeError(scratch)
    staged=stage_static_database(e['database_path'],e['database_receipt'],reserve_bytes=2*1024**3,
        expected_url=dbsource['actual_source_url'],scratch_receipt=scratch)
    if staged['status']!='LOCAL_DATABASE_READY': raise RuntimeError(staged)
    atomic_json(root/'scratch.json',staged,immutable=True)
    result=generate_parent(records[0]['generation_request'],dict(database_path=staged['database_path'],
        upstream_root=e['upstream_root'],science_hash=expected,parent_wall_limit_seconds=900), log_path=root/'native.log')
    atomic_json(root/'diagnosis.json',dict(result=result,source_root=str(source),parent_id=a.parent_id,
        source_request_sha256=digest(records[0]['generation_request']),new_parent_count=1,
        successful_generation_units_replayed=0,scientific_pass_claimed=False),immutable=True)
    print(result['status'])
if __name__=='__main__': main()
