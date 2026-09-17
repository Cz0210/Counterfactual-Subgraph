"""Real, bounded adapter/observer regression; never advances the random walk.

The reviewed scientific checkout is imported unchanged. Evidence is produced
first and independently verified in a separate process. No fixture result is
substituted for a real model output, and failures consume the inference ledger.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def charge(path, kind, count):
    """Charge BEFORE the actual call, including failures; no automatic reset."""
    path = Path(path)
    with path.with_suffix('.lock').open('a+b') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = read(path) if path.exists() else {
            'authorization': 'T12_V8_REAL_ADAPTER_64_64',
            'gine': 0, 'neurosed_pairs': 0, 'transitions': 0}
        if kind not in ('gine', 'neurosed_pairs') or count < 1:
            raise ValueError('INVALID_INFERENCE_CHARGE')
        if ledger[kind] + count > 64:
            raise ValueError('T12_V8_INFERENCE_BUDGET_EXHAUSTED:' + kind)
        ledger[kind] += count
        write(path, ledger)


def bootstrap(template):
    spec = read(template)
    base = read(Path(template).parent / 'reviewed_source_base.json')
    receipt = read(spec['science_contract']['scientific_source_equivalence_receipt'])
    approved = {r['path']: r['current_sha256'] for r in receipt['audited_differences']}
    if len(approved) != 4 or approved != base['reviewed_files']:
        raise ValueError('EXACT_REVIEWED_FOUR_FILES_REQUIRED')
    for relative, digest in approved.items():
        if sha(Path(base['base_root']) / relative) != digest:
            raise ValueError('REVIEWED_SOURCE_CHANGED:' + relative)
    import src, src.utils
    src.__path__[:] = [str(Path(base['base_root']) / 'src')]
    src.utils.__path__[:] = [str(Path(base['base_root']) / 'src/utils')]
    loader = importlib.util.spec_from_file_location('t12_original_bootstrap', spec['entrypoint'])
    original = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(original)
    return spec, base


def private_rng(objects, np, torch, tensor_value):
    """Inventory actual private generators, not an invented placeholder RNG."""
    found = {}; seen = set()
    def visit(value, key, depth):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, np.random.Generator):
            found[key] = tensor_value(value.bit_generator.state)
        elif isinstance(value, np.random.RandomState):
            found[key] = tensor_value(value.get_state())
        elif isinstance(value, random.Random):
            found[key] = tensor_value(value.getstate())
        elif isinstance(value, torch.Generator):
            found[key] = tensor_value(value.get_state())
        elif depth and hasattr(value, '__dict__'):
            for name, child in vars(value).items():
                if not name.startswith('__'):
                    visit(child, key + '.' + name, depth - 1)
    for key, value in objects.items():
        visit(value, key, 2)
    return found


def produce(template, root, budget):
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    spec, base = bootstrap(template)
    c = spec['science_contract']
    lease = Path(spec['gpu_request']['lease_path'])
    if lease.is_symlink() or not lease.is_file():
        raise ValueError('ORIGINAL_GPU_LEASE_REQUIRED')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['gpu_request']['uuid']:
        raise ValueError('EXACT_T12_GPU_UUID_REQUIRED')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('CUBLAS_ENV_REQUIRED_BEFORE_CUDA')
    from src.utils.autodl_runtime import query_gpu_inventory
    from src.ablations.llm.existing_gpu_owner import memory_headroom
    with lease.open('a+b') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu = next(g for g in query_gpu_inventory() if g.uuid == spec['gpu_request']['uuid'])
        slots = os.statvfs(root); local = os.statvfs('/root/autodl-tmp')
        headroom = memory_headroom(Path('/proc'), Path('/sys/fs/cgroup/memory'))
        # Existing 64-GiB T12 admission + T13 increment32 + safety64 retained.
        admission = dict(pid=os.getpid(), gpu_uuid=gpu.uuid, lock_fd=held.fileno(),
            headroom=headroom, required_headroom=160 << 30,
            slots=slots.f_favail, required_slots=8960,
            nvme_available=local.f_bavail * local.f_frsize,
            gpu_processes=[p.pid for p in gpu.processes], measured=time.time())
        write(root / 'admission.json', admission)
        if (gpu.processes or headroom < 160 << 30 or slots.f_favail < 8960
                or local.f_bavail * local.f_frsize < 2 << 30):
            raise ValueError('T12_PROBE_RESOURCE_ADMISSION_FAILED')
        import numpy as np
        import torch
        from src.utils.t12_shadow_recovery import tensor_value, rng_snapshot
        from src.utils.t12_raw_evidence import BoundSelectedStepObserver, RawEvidenceResolver
        from src.utils.tastemolnet_t7_typed_release_v1 import hold_t7_release_sources
        from src.baselines import tastemolnet_gcf_replay_canary as native
        from src.baselines import tastemolnet_gcf_smoke as smoke
        from src.baselines.tastemolnet_gcf_full_resume import T12StableGCFBridge
        import src.baselines.tastemolnet_gcf_full_resume as bridge_module
        backend = native.configure_exact_cuda_replay(torch=torch)
        threshold = read(c['threshold_authority'])['neurosed_distance_threshold']
        source_files = [Path(base['base_root']) / p for p in base['reviewed_files']]
        source_files += [Path(m.__file__) for m in (native, smoke, bridge_module)]
        import src.utils.t12_raw_evidence as raw_module
        import src.utils.t12_shadow_recovery as observer_module
        source_files += [Path(raw_module.__file__), Path(observer_module.__file__)]
        contract = dict(template=str(template), template_sha=sha(template),
            sources={str(p): sha(p) for p in source_files}, input_hashes=spec['input_hashes'],
            raw_contract_sha=c['shadow_binding']['raw_contract_sha256'], backend=backend,
            device='cuda:0', gpu_uuid=gpu.uuid, torch=torch.__version__,
            groups=['normal', 'repeat', 'duplicate_cache', 'lineage_rejection'],
            rejection_scope='existing native missing_source_index branch; unchanged train graph, omitted lineage only',
            transition_count=0, test_loaded=False, budget_path=str(budget))
        write(root / 'contract.json', contract)
        with hold_t7_release_sources(managed_neurosed_root=c['managed_neurosed_root'],
                t3_root=c['t3_root'], official_gcf_root=c['official_root'],
                neurosed_distance_threshold=float(threshold)) as sources:
            loaded = smoke.load_train_rows(sources.train_bytes,
                source_path=Path(sources.train_contract['path']),
                expected_num_records=sources.train_contract['num_records'],
                expected_label_counts=sources.train_contract['label_counts'])
            records = [smoke.encode_taste_source_graph(r, loaded.schema) for r in loaded.sweet_rows[:2]]
            write(root / 'inputs.json', dict(records=records, schema=loaded.schema.to_dict()))
            for observed in (False, True):
                random.seed(7); np.random.seed(7); torch.manual_seed(7); torch.cuda.manual_seed_all(7)
                modules = smoke._official_modules(sources.official_root)
                walk, importance, distance = (modules[x] for x in ('vrrw', 'importance', 'distance'))
                native._reset_official_vrrw(walk)
                graphs = [smoke.taste_record_to_pyg(r, origin_index=i) for i, r in enumerate(records)]
                adapter = smoke.TasteFrozenGINENativeAdapter(sources.checkpoint_payloads,
                    source_records=records, graph_schema=loaded.schema, device='cuda:0')
                neurosed = distance.load_neurosed(graphs,
                    neurosed_model_path=f'/proc/self/fd/{sources.neurosed_model.file_fd}', device='cuda:0')
                counts = importance.util.graph_element_counts(graphs)
                walk.input_graphs_covered = torch.zeros(len(graphs), dtype=torch.float)
                coverage = native._BoundedNeuroSEDCoverage(importance)
                bridge = T12StableGCFBridge(adapter=adapter, vrrw=walk, importance=importance,
                    neurosed_model=neurosed, original_graph_element_counts=counts,
                    distance_threshold=float(threshold), parent_count=len(graphs),
                    feature_atomic_numbers=loaded.schema.feature_atomic_numbers, coverage_runtime=coverage)
                resolver = RawEvidenceResolver(contract['raw_contract_sha'])
                observer = BoundSelectedStepObserver(None, np=np, torch=torch, resolver=resolver)
                objects = dict(walk=walk, adapter=adapter, neurosed=neurosed, bridge=bridge)
                neural_modules = {'gine':adapter.model}
                def find_models(value,prefix,depth):
                    if isinstance(value,torch.nn.Module):
                        neural_modules[prefix]=value
                    elif depth and hasattr(value,'__dict__'):
                        for key,child in vars(value).items():
                            if not key.startswith('__'):find_models(child,prefix+'.'+key,depth-1)
                find_models(neurosed,'neurosed',2)
                if not any(k.startswith('neurosed') for k in neural_modules):
                    raise ValueError('NEUROSED_REAL_MODEL_BUFFER_INVENTORY_MISSING')
                def parameter_digest(model):
                    digest=hashlib.sha256()
                    for name,value in model.named_parameters():
                        value=value.detach().cpu().contiguous()
                        digest.update(name.encode());digest.update(str(value.dtype).encode())
                        digest.update(str(tuple(value.shape)).encode());digest.update(value.numpy().tobytes())
                    return digest.hexdigest()
                def state():
                    return dict(rng=rng_snapshot(np, torch), private_rng=private_rng(objects,np,torch,tensor_value),
                        gine_buffers=tensor_value(dict(adapter.model.named_buffers())),
                        all_model_buffers={k:tensor_value(dict(m.named_buffers())) for k,m in neural_modules.items()},
                        parameter_sha={k:parameter_digest(m) for k,m in neural_modules.items()},
                        model_modes={k:m.training for k,m in neural_modules.items()},
                        candidate_registry=tensor_value(walk.counterfactual_candidates),
                        graph_index=tensor_value(walk.graph_index_map),
                        covered=tensor_value(walk.input_graphs_covered),
                        first_seen=tensor_value({k:dataclasses.asdict(v) for k,v in bridge.records.items()}),
                        lineage=tensor_value(bridge.lineage_occurrences),
                        scorer_report=adapter.scorer.report(), coverage_calls=copy.deepcopy(coverage.calls))
                raw_logits=[]; raw_distances=[]; batches=[]; graph_tensors=[]
                collate_original=adapter.scorer.collate_fn
                def captured_collate(rows):
                    result=collate_original(rows)
                    graph_tensors.append({k:tensor_value(getattr(result,k,None))
                        for k in ('x','edge_index','edge_attr','batch','ptr')})
                    return result
                adapter.scorer.collate_fn=captured_collate
                score_original = adapter.scorer.score
                def counted_score(rows, **kw):
                    charge(budget, 'gine', len(rows))
                    batches.append([r.graph_sha256 for r in rows])
                    return score_original(rows, **kw)
                adapter.scorer.score = counted_score
                predict_original = neurosed.predict_outer_with_queries
                def counted_predict(rows, **kw):
                    charge(budget, 'neurosed_pairs', len(rows)*len(graphs))
                    result = predict_original(rows, **kw)
                    raw_distances.append(tensor_value(result))
                    return result
                neurosed.predict_outer_with_queries = counted_predict
                hook = adapter.model.classifier.register_forward_hook(
                    lambda m,i,o: raw_logits.append(tensor_value(o)))
                before = state(); calls=[]
                manager = observer.installed() if observed else contextlib.nullcontext()
                try:
                    with manager:
                        for name, indexes in [('normal',[0,1]), ('repeat',[0,1]), ('duplicate_cache',[0,0])]:
                            result = bridge.call([graphs[i] for i in indexes], {})
                            ids = [bridge.calculate_hash(e) for e in result[1]]
                            calls.append(dict(group=name, input_indexes=indexes, graph_ids=ids,
                                output=tensor_value(result), state=state()))
                        rejected = graphs[0].clone()
                        del rejected.gcf_origin_index
                        value = adapter.score([rejected])
                        calls.append(dict(group='lineage_rejection',
                            output=tensor_value(dataclasses.asdict(value)), state=state()))
                    after = state()  # Observe actual post-state; do not reset RNG.
                finally:
                    hook.remove()
                payload = dict(before=before, calls=calls, after=after,
                    batch_graph_sha=batches, raw_logits=raw_logits, raw_neurosed=raw_distances,
                    collated_graph_tensors=graph_tensors,
                    observer_rows=list(resolver.rows.values()), observer_events=observer.query_events,
                    pending_transition=observer.pending,
                    resolved={k:resolver.resolve(k) for k in resolver.rows})
                write(root / ('on.json' if observed else 'off.json'), payload)
                sources.revalidate()
                del bridge, adapter, neurosed, observer, objects
        write(root/'producer.json', dict(status='EVIDENCE_WRITTEN_NOT_VERIFIED',
            files={p.name:sha(p) for p in [root/'contract.json',root/'inputs.json',root/'off.json',root/'on.json']},
            budget=read(budget), transitions=0))


def verify(root, *, publish=True):
    """No model imports: reopen both actual arms and independently check evidence."""
    root=Path(root); producer=read(root/'producer.json')
    for name,digest in producer['files'].items():
        if sha(root/name)!=digest:
            raise ValueError('REGRESSION_EVIDENCE_CHANGED:'+name)
    contract=read(root/'contract.json'); left=read(root/'off.json');right=read(root/'on.json')
    failures=[]
    for field in ('before','calls','after','batch_graph_sha','raw_logits','raw_neurosed','collated_graph_tensors'):
        if left[field]!=right[field]:failures.append('OBSERVER_CHANGED:'+field)
    if not left['raw_logits'] or not left['raw_neurosed']:
        failures.append('NO_REAL_MODEL_ARRAYS')
    if left['pending_transition'] is not None or right['pending_transition'] is not None:
        failures.append('UNAUTHORIZED_WALK_TRANSITION')
    if len(right['resolved'])!=2 or any(x['status']!='PASS' for x in right['resolved'].values()):
        failures.append('GRAPH_RAW_BINDING_INCOMPLETE')
    for arm in (left,right):
        calls=arm['calls']
        if [r['group'] for r in calls]!=contract['groups']:
            failures.append('SCENARIO_ORDER_CHANGED')
        if calls[-1]['output']['valid_fullgraphs'] != [False]:
            failures.append('REJECTION_NOT_EXECUTED')
        if calls[-1]['output']['failure_reasons'] != ['missing_source_index']:
            failures.append('REJECTION_REASON_CHANGED')
        if len(arm['raw_neurosed'])!=1:
            failures.append('EXPECTED_REAL_CACHE_HIT_NOT_OBSERVED')
        if arm['before']['rng']!=arm['after']['rng']:
            failures.append('FROZEN_EVAL_CHANGED_RNG')
        if arm['before']['gine_buffers']!=arm['after']['gine_buffers']:
            failures.append('FROZEN_GINE_BUFFERS_CHANGED')
        for key in ('all_model_buffers','parameter_sha','model_modes','private_rng'):
            if arm['before'][key]!=arm['after'][key]:failures.append('FROZEN_STATE_CHANGED:'+key)
    if any(producer['budget'][k]>64 for k in ('gine','neurosed_pairs')):
        failures.append('INFERENCE_BUDGET_EXCEEDED')
    for path,digest in contract['sources'].items():
        if sha(path)!=digest:failures.append('SOURCE_CHANGED:'+path)
    receipt=dict(schema='t12_real_adapter_observer_regression_v9',
        status='FAIL' if failures else 'PASS', failures=failures,
        observer_changes_science=bool(failures),raw_binding_tested=not failures,
        evidence_root=str(root),evidence=producer['files'],producer_sha256=sha(root/'producer.json'),
        contract_sha256=sha(root/'contract.json'),model_input_hashes=contract['input_hashes'],
        observer_source_sha256={p:d for p,d in contract['sources'].items() if 't12_raw_evidence' in p or 't12_shadow_recovery' in p},
        scope='REAL_ADAPTER_OBSERVER_ONLY_NOT_ALGORITHM_PARITY',
        transition_count=0, full_restore_or_parity_proven=False, budget=producer['budget'])
    if publish:write(root/'real-adapter-regression.json',receipt)
    if failures:raise ValueError('REAL_OBSERVER_REGRESSION_FAILED:'+','.join(failures))
    return receipt
