#!/usr/bin/env python3
"""Execute reviewed T12 science with separate, fully checked wrapper identity."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.main_ready_task_specs import load_spec,file_sha256
from src.utils.t12_same_gpu_resume_v9 import validate_same_gpu_resume_identity


def reviewed_view(spec):
    c=spec['science_contract'];b=c['v7_wrapper_binding']
    template=Path(b['template']);base_path=Path(b['source_base_binding'])
    if file_sha256(template)!=b['template_sha256'] or file_sha256(base_path)!=b['source_base_binding_sha256']:
        raise ValueError('T12_EXISTING_REVIEWED_BINDING_CHANGED')
    original=json.loads(template.read_text());base=json.loads(base_path.read_text())
    receipt=json.loads(Path(c['scientific_source_equivalence_receipt']).read_text())
    if c['scientific_source_equivalence_file_sha256']!=original['science_contract']['scientific_source_equivalence_file_sha256']:
        raise ValueError('T12_APPROVED_RECEIPT_CHANGED')
    audited={r['path']:r['current_sha256'] for r in receipt['audited_differences']}
    if base['reviewed_files']!=audited or len(audited)!=4:
        raise ValueError('T12_NOT_EXACT_APPROVED_FOUR_FILE_COMBINATION')
    for relative,sha in audited.items():
        if file_sha256(Path(base['base_root'])/relative)!=sha:
            raise ValueError('T12_REVIEWED_SCIENTIFIC_BYTES_CHANGED:'+relative)
    view=copy.deepcopy(spec)
    view['repo_root']=base['base_root']
    view['execution_commit']=receipt['current_commit']
    view['science_contract']['current_execution_tree']=receipt['current_tree']
    return view,original['entrypoint']


def main():
    args=sys.argv[1:]
    spec=load_spec(Path(args[args.index('--task-spec')+1]))
    view,entry=reviewed_view(spec)
    # load_spec above is an owner utility. Route subsequent scientific imports
    # to the physically verified base, not the newer integration checkout.
    import src,src.utils
    src.__path__[:]=[str(Path(view['repo_root'])/'src')]
    src.utils.__path__[:]=[str(Path(view['repo_root'])/'src/utils')]
    # The unchanged original bootstrap imports its pinned scientific base and
    # four reviewed physical files. This only corrects which tree is audited.
    loader=importlib.util.spec_from_file_location('t12_reviewed_existing_cli',entry)
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    import scripts.autodl.run_t12_accelerated_from250_v1 as native
    original_validate=native._validate_source_equivalence
    def validate(current):
        checked,_=reviewed_view(current)
        return original_validate(checked)  # Full existing clean-tree/inventory/receipt checks.
    native._validate_source_equivalence=validate
    import src.baselines.tastemolnet_gcf_full as generation
    original_transport=generation.validate_cross_gpu_resume_identity
    def transport(**identities):
        current=identities['current'];authority=identities['authority']
        if current['identity_template']['gpu_uuid']==authority['identity_template']['gpu_uuid']:
            # Persist the actual runtime comparison before accepting anything.
            from src.eval.bace_frozen_gnn_contracts import atomic_json
            atomic_json(Path(spec['output_root'])/'same_gpu_identity_inputs_v9.json',identities)
            return validate_same_gpu_resume_identity(**identities)
        return original_transport(**identities)
    generation.validate_cross_gpu_resume_identity=transport
    try:
        return module.main(args)
    finally:
        native._validate_source_equivalence=original_validate
        generation.validate_cross_gpu_resume_identity=original_transport

if __name__=='__main__':raise SystemExit(main())
