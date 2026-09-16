#!/usr/bin/env python3
"""Finite Mac receiver for this one R3 stage, no science dispatch/restart."""
import argparse,json,os,shlex,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.eval.ours_taste_focus_matrix import dump_json,read_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--remote-root',required=True);a=p.parse_args()
    assert a.config.resolve()==(ROOT/'configs/hpc.yaml').resolve()
    assert a.remote_root.startswith('/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/experiments/ours_taste_k20_theta010_v3/')
    end=datetime.fromisoformat(read_json(a.root/'resolved_contract.json')['absolute_deadline'])
    receipt=a.root/'R3_receiver_owner.json';assert not receipt.exists()
    dump_json(receipt,{'pid':os.getpid(),'command':sys.argv,'deadline':end.isoformat(),'science_dispatch':False})
    remote=a.remote_root+'/R3-search'
    while datetime.now(timezone.utc)<end:
        command=f'if test -f {shlex.quote(remote+"/failure.json")}; then cat {shlex.quote(remote+"/failure.json")}; elif test -f {shlex.quote(remote+"/terminal.json")}; then cat {shlex.quote(remote+"/terminal.json")}; fi'
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','autodl-a800',command],text=True,capture_output=True,timeout=40)
        if r.returncode==0 and r.stdout.strip():
            state=json.loads(r.stdout)
            if state['state']=='FAILED':raise RuntimeError('R3_FAILED_PRESERVED: '+r.stdout)
            target=a.root/'R3-search';target.mkdir(exist_ok=True)
            names=['terminal.json','budget_ledger.json','pool_freeze.json','candidate_pool_P2.json','train_search_freeze.json']
            if state['state']=='R3_COMPLETE_AWAITING_SAVED_RAW_AUDIT':
                names+=['train_change.json','calibration_bounds.json','selection_freeze.json','calibration_prefix.csv','test_prefix.csv','test_selected_union.npz']
                if state['new_test_rules']:names+=['test_new_pairs.jsonl.gz']
            for name in names:subprocess.run(['scp','-q',f'autodl-a800:{remote}/{name}',str(target/name)],check=True,timeout=120)
            if state['state']=='R3_NO_NEW_UNIQUE_RULES_VALID_RESULT':
                dump_json(a.root/'R3_receiver_terminal.json',{'state':'NO_NEW_UNIQUE_P2_EQUALS_P1','source':state});return
            subprocess.run([sys.executable,'-I','-B',str(ROOT/'scripts/run_ours_taste_theta010.py'),'--config',str(a.config),'--action','export-r3','--root',str(a.root)],check=True)
            release=a.root/'release-R3'
            subprocess.run([sys.executable,'-I','-B',str(ROOT/'scripts/replot_ours_taste_theta010.py'),'--source-dir',str(release/'source_csv'),'--out-dir',str(release/'figures'),'--table-dir',str(release/'tables')],check=True)
            # New independent release, never a main-matrix append.
            subprocess.run(['scp','-q','-r',str(release),f'autodl-a800:{a.remote_root}/'],check=True,timeout=240)
            dump_json(a.root/'R3_receiver_terminal.json',{'state':'RAW_AUDITED_EXPORTED_SCOPED_TRANSFER_COMPLETE','release':str(release),'completed_at':datetime.now(timezone.utc).isoformat()});return
        time.sleep(60)
    dump_json(a.root/'R3_receiver_terminal.json',{'state':'ORIGINAL_DEADLINE_REACHED_NO_RESTART','science_outputs_preserved':True})

if __name__=='__main__':main()
