#!/usr/bin/env python3
"""Record-only accepted CM Taste/Mut portable package using existing transport."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.baselines.cm_crem_release import package_run
from src.baselines.cm_crem_runtime import read_json,atomic_json,file_sha
from src.baselines.cm_crem_descriptive import validate_scope,EXPERIMENT
p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--root',required=True)
a=p.parse_args();root=Path(a.root);spec=read_json(root/'spec.json')
if spec.get('study_scope',{}).get('experiment_id')==EXPERIMENT:
    roster_path=Path(spec['full_root'])/'full_roster.json';roster=read_json(roster_path)
    provenance=dict(generation_ids=[r['parent_id'] for r in roster['parents']],
        roster_path=str(roster_path),roster_sha256=file_sha(roster_path),
        full_contract_sha256=roster['full_contract_sha256'],
        reconstructed_metadata_only=True,scientific_records_changed=False)
    validate_scope(spec['study_scope'],provenance,read_json(root/'calibration_prepared.json')['parent_ids'],read_json(root/'test_prepared.json')['parent_ids'])
    atomic_json(root/'scope_provenance.json',provenance,immutable=True)
print(json.dumps(package_run(a.root)))
