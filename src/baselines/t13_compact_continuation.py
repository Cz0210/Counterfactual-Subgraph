"""Same-run T13 epoch29 continuation through the existing official trainer.

No new optimizer loop, fresh-start ledger, or selector. Diagnostic updates are
never imported. Target0 restores the untouched prepared epoch29; target2 is its
original, sequential first branch. Evaluation requires separate stage admission.
"""
import json
from pathlib import Path


def validate_recovery(plan,checkpoint):
    probe=json.loads(Path(plan['gpu_probe_receipt']).read_text())
    if (probe.get('state')!='REAL_GPU_TWO_TRAIN_ONE_VALIDATION_RELOAD_PASS'
            or probe.get('formal_updates')!=0 or checkpoint.get('next_epoch')!=30
            or plan.get('formal_quota_used')!='1/1' or plan.get('target_order')!=[0,2]):
        raise ValueError('COMPLETE_PROBE_AND_ORIGINAL_EPOCH29_REQUIRED')
    if 'model_state' not in checkpoint or 'optimizer_state' not in checkpoint:
        raise ValueError('ORIGINAL_TRAINING_STATE_MISSING')


def continue_branches(*,plan,out,checkpoint,selected,cohort,official,adapter,sample,torch):
    from src.eval.bace_frozen_gnn_contracts import atomic_json
    from src.baselines.globalgce_resumable import _atomic_torch_save
    from src.baselines.t13_bounded_payload import install_committed_expansion
    from src.baselines.tastemolnet_globalgce_full import (
        run_native_branch,TasteGlobalGCEFullConfig,_parent_cohort_sha256)
    validate_recovery(plan,checkpoint)
    raw=out/'raw';raw.mkdir(exist_ok=True)
    atomic_json(raw/'train_cohort_manifest.json',cohort)
    config=TasteGlobalGCEFullConfig(seed=7,epochs=100)
    original=adapter.train_globalgce_resumable
    current={'target':0}
    def train(**kw):
        target=current['target'];sample('target_'+str(target)+'_trainer_entry')
        ck=Path(kw['checkpoint_dir'])/'training_checkpoint.pt'
        if target==0:
            install_committed_expansion(kw['model'].fsg,descriptor=plan['committed_compact_payload'],
                expected_identity=checkpoint['augmented_dataset_identity'])
            if ck.exists():raise ValueError('FRESH_RECOVERY_ROOT_ALREADY_HAS_CHECKPOINT')
            # Copy only the already CPU-prepared ORIGINAL state, never probe state.
            _atomic_torch_save(torch,checkpoint,ck)
            kw['resume']=True
        previous=kw.get('after_epoch_checkpoint')
        def committed(receipt):
            if previous:previous(receipt)
            atomic_json(out/'formal_progress.json',dict(state='TRAINING_CHECKPOINT_COMMITTED',
                target=target,epoch=receipt['epoch'],next_epoch=receipt['next_epoch'],
                new_optimizer_steps=receipt['epoch']-29 if target==0 else receipt['epoch']+1,
                checkpoint=receipt['checkpoint_file'],formal_quota='1/1',
                source_attempt_id=plan['original_formal_attempt_id'],diagnostic_updates_adopted=0))
            # A failed resource check stops only after the original durable boundary.
            sample('target_'+str(target)+'_epoch_'+str(receipt['epoch'])+'_committed')
            from datetime import datetime,timezone
            if datetime.now(timezone.utc)>=datetime.fromisoformat(plan['deadline_utc'].replace('Z','+00:00')):
                raise RuntimeError('ORIGINAL_DEADLINE_AT_DURABLE_EPOCH_BOUNDARY')
        kw['after_epoch_checkpoint']=committed
        atomic_json(out/'formal_progress.json',dict(state='RESTORING_EPOCH29' if target==0 else 'ORIGINAL_TARGET2_FIRST_TRAINING',
            target=target,new_optimizer_steps=0,formal_quota='1/1',validation_epoch30_required=target==0,
            source_attempt_id=plan['original_formal_attempt_id'],diagnostic_updates_adopted=0))
        return original(**kw)
    adapter.train_globalgce_resumable=train
    results={}
    try:
        for target in [0,2]:
            current['target']=target
            generator=adapter.OfficialGlobalGCEMutagenicityGenerator(Path(plan['official_root']),
                native_train_csv=Path(plan['train_csv']),dataset_name='TasteMolNet',min_freq=2,
                frozen_gine_checkpoint=Path(plan['gnn_checkpoint']),source_label=1,target_label=target,num_classes=3,
                official_source_authority=official['runtime_source_authority'],require_isolated_imports=True,
                rules_only_min_valid_native_rules=0)
            generator.t13_indexed_options=dict(storage='t13_indexed_augmentation_v1')
            # Target2 has no historical compact checkpoint; use the original indexed
            # constructor at its normal RNG point, not target0's target-specific masks.
            results[str(target)]=run_native_branch(target_label=target,generator=generator,parents=selected,
                branch_root=raw/('target_'+str(target)),config=config,
                expected_checkpoint_id=plan['gnn_checkpoint_id'],expected_parent_cohort_sha256=_parent_cohort_sha256(selected),
                gspan_adoption_proof=plan['gspan_adoption_proof'])
    finally:adapter.train_globalgce_resumable=original
    atomic_json(out/'training_terminal.json',dict(state='TWO_NATIVE_BRANCHES_COMPLETE_NOT_CELL_PASS',
        targets=list(results),formal_quota='1/1',source_attempt_id=plan['original_formal_attempt_id'],
        next_stage='EXISTING_TASTE_FULL_MERGE_CALIBRATION_TEST_UNDER_ITS_OWN_STORAGE_ADMISSION',
        final_evaluation_admission_required=True))
    return 0
