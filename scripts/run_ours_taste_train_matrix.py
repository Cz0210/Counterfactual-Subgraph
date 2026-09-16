#!/usr/bin/env python3
"""Taste-only train development matrix, using original GINE/deletion/WNode.

This one finite stage owns the existing UUID lock in the science process itself.
It neither dispatches other tasks nor creates another controller or lock scheme.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.eval.ours_taste_focus_matrix import dump_json,read_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--spec',type=Path,required=True)
    a=p.parse_args()
    assert a.config.resolve()==(ROOT/'configs/hpc.yaml').resolve()
    spec=read_json(a.spec);out=Path(spec['output_root']);compact=Path(spec['compact_root'])
    assert spec['campaign'] in ('OURS_TASTEMOLNET_REACH_V1','OURS_TASTE_K20_THETA010_V3') and spec['split']=='train'
    assert os.environ['CUDA_VISIBLE_DEVICES']==spec['gpu_uuid']
    assert len(spec['gpu_uuid'])>20 and spec['matrix_parent_limit']==256
    assert datetime.now(timezone.utc)<datetime.fromisoformat(spec['search_stop_at'])
    assert not out.exists(), 'FRESH_STAGE_ROOT_REQUIRED'
    out.mkdir(parents=True)
    from src.utils.autodl_runtime import GPUFileLock
    from src.ablations.llm.existing_gpu_owner import memory_headroom
    contract=read_json(compact/'contract.json')
    stage_sha=hashlib.sha256(a.spec.read_bytes()).hexdigest()
    cgroup=Path(spec['cgroup_memory_root'])
    before_memory=memory_headroom(Path('/proc'),cgroup)
    assert before_memory>=spec['minimum_headroom_bytes'], 'OWN_STAGE_RAM_ADMISSION'
    fs=os.statvfs(out)
    assert fs.f_favail>=spec['minimum_available_slots'], 'OWN_STAGE_SLOT_ADMISSION'
    assert fs.f_bavail*fs.f_frsize>=spec['minimum_free_bytes'], 'OWN_STAGE_BYTES_ADMISSION'
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    assert spec['gpu_uuid'] not in rows, 'GPU_HAS_EXISTING_SCIENCE'
    from src.eval.ours_taste_development import validate_gpu_reservation
    observed_uuid=subprocess.check_output(['nvidia-smi',f'--id={spec["gpu_index"]}',
                   '--query-gpu=uuid','--format=csv,noheader,nounits'],text=True)
    validate_gpu_reservation(read_json(Path(spec['registry'])),spec['gpu_index'],spec['gpu_uuid'],observed_uuid)
    pause=[False]
    signal.signal(signal.SIGTERM,lambda *_:pause.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:pause.__setitem__(0,True))
    with GPUFileLock(Path(spec['lock_root']),gpu_index=spec['gpu_index'],gpu_uuid=spec['gpu_uuid'],
                     owner={'run_id':spec['campaign'],'spec_sha256':stage_sha,'stage':spec['stage'],'root':str(out)}) as lease:
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
        assert spec['gpu_uuid'] not in rows
        import numpy as np
        import torch
        from src.eval.ours_taste_development import stratified_development,compact_embedder_class
        from src.eval.tastemolnet_ours_full import TasteGINEScorer,GINE_PAYLOAD_FILES,TrainParent,evaluate_parent
        from src.eval.node_wasserstein_distance import MolCLRNodeWassersteinConfig,MolCLRNodeWassersteinDistance
        torch.set_num_threads(8)
        torch.manual_seed(7)
        parents=stratified_development(read_json(compact/'train_saved_predictions.json'))
        pool=read_json(compact/'candidate_pool.json')
        dump_json(out/'development_cohort.json',{'state':'FROZEN_TRAIN_ONLY','parents':parents,
                   'strategy':'seed7_scaffold_margin_quartile_round_robin','test_read':False})
        scorer=TasteGINEScorer({n:(Path(contract['checkpoint'])/n).read_bytes() for n in GINE_PAYLOAD_FILES},device='cuda:0',batch_size=256)
        assert scorer.checkpoint_id==contract['checkpoint_id']
        if spec.get('seed_node_database'):
            import sqlite3
            # Completed prior Ours train stage only; no active DB/WAL is copied.
            prior=Path(spec['seed_node_database'])
            assert read_json(prior.parent/'terminal.json')['state']=='TRAIN_P0_MATRIX_COMPLETE'
            with sqlite3.connect('file:'+str(prior)+'?mode=ro',uri=True) as source_db:
                with sqlite3.connect(str(out/'nodes.sqlite')) as target_db:source_db.backup(target_db)
        embedder=compact_embedder_class()(compact_db=out/'nodes.sqlite',molclr_root=spec['molclr_root'],
                    molclr_ckpt=spec['molclr_checkpoint'],node_emb_cache_dir=spec['existing_node_cache'],device='cuda:0')
        class StrictDistance(MolCLRNodeWassersteinDistance):
            def distance_for_action(self,*args,**kwargs):
                result=super().distance_for_action(*args,**kwargs)
                if not result['ok']: raise RuntimeError('DISTANCE_ERROR_NOT_NEGATIVE: '+str(result))
                return result
        provider=StrictDistance(MolCLRNodeWassersteinConfig(molclr_root=spec['molclr_root'],molclr_ckpt=spec['molclr_checkpoint'],
                    cache_db=out/'distances.sqlite',node_emb_cache_dir=spec['existing_node_cache'],device='cuda:0',
                    distance_namespace='tastemolnet_ours_full_wnode_v1'),embedder=embedder)
        identity=read_json(compact/'calibration_adoption.json')['input_identity']
        if spec['stage'] in ('SEARCH_CALIBRATE_TEST','THETA010_SELECTED_TEST','THETA010_BOUNDED_SEARCH'):
            try:
                if spec['stage']=='THETA010_SELECTED_TEST':
                    from src.eval.ours_taste_theta010 import evaluate_selected as run
                elif spec['stage']=='THETA010_BOUNDED_SEARCH':
                    from src.eval.ours_taste_theta010_search import run
                else:
                    from src.eval.ours_taste_search_chain import run
                dump_json(out/'owner.json',{'pid':os.getpid(),'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
                   'stage':spec['stage'],'gpu_uuid':spec['gpu_uuid'],'existing_lock_path':str(lease.path),'spec_sha256':stage_sha,'cwd':str(ROOT)})
                run(spec,out,compact,contract,scorer,provider,identity,pause)
            except Exception as exc:
                dump_json(out/'failure.json',{'state':'FAILED','type':type(exc).__name__,'error':str(exc),'old_results_preserved':True})
                raise
            finally:
                provider.close();embedder.close()
            return 0
        d=np.full((len(parents),len(pool)),np.nan)
        predictions=np.full(len(parents),-1,dtype=np.int8)
        started=time.monotonic();funnel=Counter();completed=0
        stat=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
        owner={'pid':os.getpid(),'start_ticks':int(stat[19]),'cmdline':sys.argv,'cwd':str(ROOT),
               'gpu_uuid':spec['gpu_uuid'],'visible_device':'cuda:0','existing_lock_path':str(lease.path),
               'lock_fd':lease._handle.fileno(),'lock_inode':os.fstat(lease._handle.fileno()).st_ino,'spec_sha256':stage_sha}
        dump_json(out/'owner.json',owner)
        try:
            for i,parent in enumerate(parents):
                if pause[0] or datetime.now(timezone.utc)>=datetime.fromisoformat(spec['search_stop_at']):break
                assert memory_headroom(Path('/proc'),cgroup)>=spec['runtime_safety_bytes'], 'RAM_SAFETY_BOUNDARY'
                assert os.statvfs(out).f_favail>=8192, 'SLOT_SAFETY_BOUNDARY'
                result=evaluate_parent(parent=TrainParent(parent['parent_id'],parent['smiles'],1,'train'),
                         candidates=pool,scorer=scorer,distance=provider,split='train',evaluation_identity=identity)
                predictions[i]=result[0]['pred_before']
                assert predictions[i]==parent['pred_before'], 'SAVED_TRAIN_SOURCE_CHANGED'
                for j,r in enumerate(result):
                    assert r['candidate_id']==pool[j]['candidate_id'] and r['split']=='train'
                    d[i,j]=float(r['wnode_distance']) if r['pair_strict_flip'] else np.inf
                    funnel['pairs']+=1;funnel['applicable']+=bool(r['applicable'])
                    funnel['valid_residual']+=r['num_valid_residuals']>0;funnel['finite_flip']+=bool(r['pair_strict_flip'])
                # One compact append-only segment, not one file per pair.
                with (out/'train_pairs.jsonl.gz').open('ab') as raw:
                    with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz:
                        for r in result:gz.write((json.dumps(r,sort_keys=True)+'\n').encode())
                    raw.flush();os.fsync(raw.fileno())
                embedder.commit();completed=i+1
                with (out/'train.npz.tmp').open('wb') as f:
                    np.savez_compressed(f,distances=d,predictions=predictions,parents=np.asarray([r['parent_id'] for r in parents]),
                                        candidates=np.asarray([r['candidate_id'] for r in pool]));f.flush();os.fsync(f.fileno())
                (out/'train.npz.tmp').replace(out/'train.npz')
                dump_json(out/'progress.json',{**owner,'state':'TRAIN_MATRIX_RUNNING','completed_parents':completed,
                        'total_parents':len(parents),'pair_count':funnel['pairs'],'funnel':dict(funnel),
                        'elapsed_seconds':time.monotonic()-started,'rss_peak_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        'cuda_peak_bytes':torch.cuda.max_memory_allocated(),'observed_at':datetime.now(timezone.utc).isoformat()})
                if completed==4:
                    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=spec['phase_increment_bound_bytes'], 'MEASURED_RAM_EXCEEDS_PHASE_BOUND'
                    assert torch.cuda.max_memory_allocated()<=spec['gpu_peak_bound_bytes'], 'MEASURED_GPU_EXCEEDS_PHASE_BOUND'
                    dump_json(out/'real_four_parent_resource_receipt.json',read_json(out/'progress.json'))
            state='TRAIN_P0_MATRIX_COMPLETE' if completed==len(parents) else 'PAUSED_AT_PARENT_BOUNDARY'
            dump_json(out/'terminal.json',{'state':state,'completed_parents':completed,'funnel':dict(funnel),
                'new_distance_queries':provider.stats_dict(),'test_read':False,'expanded_pool':False,
                'next_stage':'TRAIN_BOUNDS_THEN_AUTHORIZED_BOUNDED_EXPANSION'})
        except Exception as exc:
            dump_json(out/'failure.json',{'state':'FAILED','type':type(exc).__name__,'error':str(exc),
                      'completed_parents':completed,'failed_parent_index':completed,'uncomputed_preserved_as_nan':True})
            raise
        finally:
            provider.close();embedder.close()
    return 0

if __name__=='__main__': raise SystemExit(main())
