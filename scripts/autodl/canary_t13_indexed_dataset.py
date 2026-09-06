#!/usr/bin/env python3
"""AutoDL-only train-input T13 indexed data + bounded real training canary."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def run_diagnostic_child(command):
    """Keep the owning canary alive until its exact child releases resources."""
    child=None
    stopped=[]
    def stop(signum,frame):
        stopped.append(signum)
        if child is not None and child.poll() is None:
            child.terminate()
    previous=signal.signal(signal.SIGTERM,stop)
    try:
        child=subprocess.Popen(command)
        if stopped and child.poll() is None:
            child.terminate()
        returncode=child.wait()
        if stopped:
            raise SystemExit(128+stopped[0])
        return subprocess.CompletedProcess(command,returncode)
    finally:
        signal.signal(signal.SIGTERM,previous)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--set',action='append',default=[])
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--official-root',type=Path)
    parser.add_argument('--train-csv',type=Path)
    parser.add_argument('--gnn-checkpoint',type=Path)
    parser.add_argument('--gspan-adoption-proof',type=Path)
    parser.add_argument('--targets',default='0,2')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--diagnostic-profile',choices=('native','deterministic'),default='native')
    parser.add_argument('--verify-checkpoint-only',type=Path)
    args=parser.parse_args(argv)
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise ValueError('T13_CANARY_REQUIRES_PYTHON_I_B')
    if args.set!=['inference.fallback_to_heuristic=false'] or not args.config.is_file():
        raise ValueError('T13_CANARY_CONFIG_CONTRACT')
    if args.diagnostic_profile=='deterministic':
        # Fresh process, before torch/CUDA imports. Native mode keeps its environment.
        if 'torch' in sys.modules:
            raise ValueError('T13_DETERMINISTIC_PROFILE_REQUIRES_FRESH_PROCESS')
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    from src.baselines.t13_indexed_canary import checkpoint_reopen,T13IndexedCanaryComplete,T13CanaryParityError
    from src.eval.bace_frozen_gnn_contracts import atomic_json,sha256_file
    if args.verify_checkpoint_only is not None:
        if args.output_root.exists():raise ValueError('T13_FRESH_RELOAD_RECEIPT_REQUIRED')
        args.output_root.mkdir(parents=True)
        atomic_json(args.output_root/'verification.json',checkpoint_reopen(args.verify_checkpoint_only))
        return 0
    if args.device!='cuda:0' or args.targets!='0,2' or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('T13_CANARY_GPU1_OWNER_AND_TARGETS_REQUIRED')
    if any(getattr(args,name) is None for name in ('official_root','train_csv','gnn_checkpoint','gspan_adoption_proof')):
        raise ValueError('T13_CANARY_REAL_INPUTS_REQUIRED')
    if args.output_root.exists():raise ValueError('T13_CANARY_FRESH_ROOT_REQUIRED')
    args.output_root.mkdir(parents=True)
    from src.baselines.tastemolnet_globalgce_full import (_checkpoint_payloads,load_full_train_split,
        select_full_sweet_train_cohort,FrozenTasteGINEScorer)
    from src.baselines.globalgce_bace_native_rules import validate_official_globalgce_root
    from src.baselines.globalgce_mutagenicity_adapter import OfficialGlobalGCEMutagenicityGenerator
    official=validate_official_globalgce_root(args.official_root)
    payloads=_checkpoint_payloads(args.gnn_checkpoint)
    split=json.loads(payloads['split_manifest.json'])
    if sha256_file(args.train_csv)!=split['files']['train']['sha256']:
        raise ValueError('T13_CANARY_TRAIN_HASH_DRIFT')
    train=load_full_train_split(SimpleNamespace(train_path=args.train_csv,
        train_count=split['train_manifest']['num_records'],train_label_counts=split['train_manifest']['label_counts']))
    scorer=FrozenTasteGINEScorer(payloads,device=args.device,batch_size=256)
    selected,cohort=select_full_sweet_train_cohort(train,scorer=scorer,batch_size=256)
    atomic_json(args.output_root/'train_cohort_manifest.json',cohort)
    del scorer
    stop=threading.Event()
    samples=[]
    def monitor():
        while not stop.is_set():
            row={'elapsed_seconds':time.monotonic()-started}
            try:
                for line in Path('/proc/self/status').read_text().splitlines():
                    if line.startswith(('VmRSS:','VmHWM:')):row[line.split(':')[0]+'_bytes']=int(line.split()[1])*1024
                cgroup=Path('/sys/fs/cgroup/memory')
                for name in ('memory.usage_in_bytes','memory.limit_in_bytes','memory.failcnt'):
                    row[name]=int((cgroup/name).read_text())
            except (OSError,ValueError):pass
            samples.append(row)
            atomic_json(args.output_root/'memory_progress.json',{'samples':samples[-120:],'total_samples':len(samples)})
            stop.wait(5)
    started=time.monotonic();thread=threading.Thread(target=monitor,daemon=True);thread.start()
    reports={};failure=None
    try:
        for target in (0,2):
            branch=args.output_root/f'target_{target}'
            generator=OfficialGlobalGCEMutagenicityGenerator(args.official_root,native_train_csv=args.train_csv,
                dataset_name='TasteMolNet',min_freq=2,frozen_gine_checkpoint=args.gnn_checkpoint,
                source_label=1,target_label=target,num_classes=3,
                official_source_authority=official['runtime_source_authority'],require_isolated_imports=True,
                rules_only_min_valid_native_rules=0)
            generator.t13_indexed_options={'storage':'t13_indexed_augmentation_v1','canary_output':branch/'training_canary',
                                          'diagnostic_profile':args.diagnostic_profile}
            try:
                generator.generate(selected,output_dir=branch,seed=7,epochs=100,top_k_native=20,
                    learning_rate=0.1,dropout=0.5,device=args.device,resume=False,
                    gspan_adoption_proof=args.gspan_adoption_proof,rules_only=True)
            except T13IndexedCanaryComplete as complete:
                reports[str(target)]=complete.report
            except T13CanaryParityError as error:
                failure=dict(target=target,**error.report)
                break
            else:raise ValueError('T13_CANARY_DID_NOT_STOP_BEFORE_FORMAL_TRAINING')
            checkpoint=branch/'training_canary/training_checkpoint.pt'
            fresh=branch/'independent_reload'
            subprocess.run([sys.executable,'-I','-B',str(Path(__file__).resolve()),'--config',str(args.config),
                '--set','inference.fallback_to_heuristic=false','--output-root',str(fresh),
                '--verify-checkpoint-only',str(checkpoint)],check=True)
            replay=json.loads((fresh/'verification.json').read_text())
            expected=json.loads((branch/'training_canary/checkpoint_reload.json').read_text())
            if replay!=expected:raise ValueError('T13_INDEPENDENT_RELOAD_DRIFT')
    finally:
        stop.set();thread.join(timeout=10)
        atomic_json(args.output_root/'memory_samples.json',{'samples':samples})
    if failure is not None:
        failure_report=dict(state='T13_COMPONENT_DIAGNOSTIC_FAILED',diagnostic_profile=args.diagnostic_profile,
            failure=failure,targets_completed=list(reports),targets_order=[0,2],seed=7,configured_epochs=100,
            train_only=True,test_loaded=False,calibration_loaded=False,mining_recomputed=False,
            full_successor_started=False,formal_launch_allowed=False,tolerance_used=False)
        atomic_json(args.output_root/'canary.json',failure_report)
        # Only a demonstrated native eager self-repeat failure triggers this fresh
        # matched diagnosis. It never consumes the owner's one formal-start claim.
        if (args.diagnostic_profile=='native' and failure.get('eager_repetitions_completed')==3
                and failure.get('eager_self_repeatable') is False):
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            child=args.output_root/'matched-deterministic'
            command=[sys.executable,'-I','-B',str(Path(__file__).resolve()),'--config',str(args.config),
                '--set','inference.fallback_to_heuristic=false','--output-root',str(child),
                '--official-root',str(args.official_root),'--train-csv',str(args.train_csv),
                '--gnn-checkpoint',str(args.gnn_checkpoint),'--gspan-adoption-proof',str(args.gspan_adoption_proof),
                '--device',args.device,'--targets','0,2','--diagnostic-profile','deterministic']
            completed=run_diagnostic_child(command)
            diagnostic_path=child/f"target_{failure['target']}"/'training_canary/component_diagnostics.json'
            matched=None
            if diagnostic_path.is_file() and (child/'train_cohort_manifest.json').is_file():
                other=json.loads(diagnostic_path.read_text())
                left=failure.get('initial_state_bindings',[])
                right=other.get('initial_state_bindings',[])
                matched=dict(
                    initial_model_optimizer_scheduler_rng_index_exact=bool(left and right and
                        left[0]['state_sha256']==right[0]['state_sha256']),
                    source_cohort_exact=sha256_file(args.output_root/'train_cohort_manifest.json')==
                        sha256_file(child/'train_cohort_manifest.json'),
                    deterministic_diagnostics_sha256=sha256_file(diagnostic_path))
            failure_report['matched_deterministic_diagnostic']={
                'root':str(child),'returncode':completed.returncode,
                'canary_json':str(child/'canary.json'),
                'matched_start_evidence':matched,
                'native_failure_preserved':True,'formal_launch_allowed':False}
            atomic_json(args.output_root/'canary.json',failure_report)
        print(json.dumps(failure_report,sort_keys=True))
        return 1
    report=dict(state=('T13_INDEXED_DATA_AND_SHORT_TRAINING_CANARY_PASS' if args.diagnostic_profile=='native'
                       else 'T13_MATCHED_DETERMINISTIC_DIAGNOSTIC_PASS'),targets=reports,
        seed=7,configured_epochs=100,targets_order=[0,2],train_only=True,calibration_loaded=False,test_loaded=False,
        diagnostic_profile=args.diagnostic_profile,formal_launch_allowed=args.diagnostic_profile=='native',
        deterministic_diagnostic_not_formal_approval=args.diagnostic_profile=='deterministic',
        full_successor_started=False,mining_recomputed=False,independent_reload_pass=True,
        index_contract_pass=all(r['dataset_identity']['sample_count']>0 and r['dataset_identity']['index_sha256'] for r in reports.values()),
        mask_rng_batch_parity=all(r['dataset_identity']['all_masks_reconstructed_exactly']
            and r['dataset_identity']['materialization_rng_unchanged'] and r['eager_lazy_batch_exact'] for r in reports.values()),
        training_step_parity=all(r['forward_loss_exact'] and r['model_optimizer_scheduler_rng_exact'] for r in reports.values()),
        reload_parity=all(r['checkpoint_reload_exact'] for r in reports.values()),
        full_trajectory_parity_claimed=False,
        memory_admission_state='REQUIRES_OWNER_RESOURCE_GATE_REVIEW',memory_samples_sha256=sha256_file(args.output_root/'memory_samples.json'))
    atomic_json(args.output_root/'canary.json',report)
    print(json.dumps(report,sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
