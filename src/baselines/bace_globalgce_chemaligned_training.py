"""Single seed7 repair-finetune of the saved official BACE generator.

No mining, teacher training, test, calibration, or matrix authority is imported.
Training consumes the complete recorded train match universe; a predeclared
seed7 permutation supplies the official upper budget of5x500 applications per
optimizer update. Small differentiable backward pieces are execution only;
the optimizer is stepped once per epoch, never once per micro-piece.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.baselines.bace_globalgce_chemaligned import joint_states, materialize
from src.baselines.bace_globalgce_chemaligned_runner import setup, read_json, predict_smiles
from src.baselines.globalgce_bace_native_rules import build_parent_native_tensors, validate_official_globalgce_root
from src.baselines.globalgce_mutagenicity_adapter import _import_official_modules
from src.eval.bace_frozen_gnn_contracts import atomic_json, stable_sha256, utc_now

TRAINING_CONTRACT = {
    'schema_version': 'bace_chemaligned_repair_finetune_objective_v2',
    'initialization': 'original_generator_checkpoint_repair_finetune',
    'seed': 7, 'epochs': 100, 'optimizer_updates': 100,
    'optimizer': 'Adam', 'learning_rate': 0.1, 'weight_decay': 1e-5,
    'scheduler': {'class': 'StepLR', 'step_size': 10, 'gamma': .9},
    'max_logical_batches': 5, 'logical_batch_size': 500,
    'sampling': 'all_train_match_universe_seed7_torch_randperm_prefix2500_each_epoch',
    'node_similarity': 'mean_l2_normalized_node_weights_vs_lhs_onehot',
    'edge_similarity': 'mean_l2_joint_state_probabilities_vs_lhs_onehot',
    'adjacency_similarity': 'binary_cross_entropy_joint_nonNONE_vs_lhs_adj',
    'kl': 'original_pinned_Gaussian_formula',
    'weights': {'node': 10., 'edge': 10., 'adjacency': 100., 'kl': 100000., 'counterfactual': 10.},
    'repair_phase': 'post_warmup_full_objective_every_epoch_old_checkpoint_after100epochs',
    'invalid_complete_product': 'constant_negative_log_float32_epsilon_no_oracle_no_fallback_gradient',
    'validation_frequency_epochs': 5,
    'validation_selection': 'max_strict_flip_parents_then_valid_parents_then_min_mean_source_target_NLL_then_earliest',
    'validation_scope': 'all_bound_validation_source_matches_no_subsampling',
    'checkpoint': 'first_optimizer_update_then_every_epoch_atomic_latest_and_best',
    'calibration_used': False, 'test_used': False,
    'gradient_estimator': 'straight_through_for_valid_full_graphs_not_exact_discrete_gradient',
    'not_claimed': 'original_eager_trajectory_or_original_objective_numerical_equivalence',
}


def snapshot_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])


def atomic_torch(path, payload):
    temporary = path.with_suffix(path.suffix+'.partial')
    with temporary.open('wb') as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def assert_semantic_equal(left, right, path='checkpoint'):
    """Compare recovery values, not object IDs or serialization bytes."""
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left.detach().cpu(), right.detach().cpu()):
            raise ValueError(f'{path}: tensor recovery mismatch')
    elif isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or left.dtype != right.dtype or not np.array_equal(left, right):
            raise ValueError(f'{path}: numpy recovery mismatch')
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys(): raise ValueError(f'{path}: keys differ')
        for key in left: assert_semantic_equal(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right): raise ValueError(f'{path}: sequence differs')
        for i, (a, b) in enumerate(zip(left, right)): assert_semantic_equal(a, b, f'{path}[{i}]')
    elif left != right:
        raise ValueError(f'{path}: scalar recovery mismatch')


def load_index(path, parents, templates):
    by_id = {r['id']: r for r in parents}; cache = {}; index = []
    for text in Path(path).read_text().splitlines():
        row = json.loads(text); pid = row['parent_id']
        if pid not in by_id: raise ValueError('index parent outside frozen split')
        if pid not in cache:
            cache[pid] = build_parent_native_tensors(by_id[pid]['smiles'],
                atom_symbols=templates[0].atom_symbols, bond_names=templates[0].bond_names)
        index.append((by_id[pid], cache[pid], templates[row['rule_index']], dict(row['mapping']), row['before']))
    return index


def load_generator(config, summary, frozen_rules, bridge, device):
    authority = validate_official_globalgce_root(config['official_root'])
    modules = _import_official_modules(Path(config['official_root'])/'src',
                                      expected_source_authority=authority['runtime_source_authority'])
    width, nodes = frozen_rules['feat'].shape[-1], frozen_rules['feat'].shape[-2]
    model = modules['GlobalGCE'](width, 64, 32, frozen_rules['edge_attr'].shape[-1], .5,
        3, nodes, len(frozen_rules['feat']), False, summary['frequent_subgraphs_path'], device, bridge).to(device)
    model.fsg.fs_max_nodes = nodes
    model.create_decoders()
    original = torch.load(summary['globalgce_model_checkpoint'], map_location=device, weights_only=False)
    model.load_state_dict(original, strict=True)
    return model, authority


def reconstruction(rules):
    node = rules['features_reconst']
    node = node/node.sum(-1, keepdim=True).clamp_min(1e-12)
    states = torch.stack([joint_states(a, e) for a, e in zip(rules['adj_reconst'], rules['edge_attrs_reconst'])])
    # Both edge and adjacency regularizers now consume the same joint state,
    # rather than supervising incompatible independent hard outputs.
    dx = torch.linalg.vector_norm((node-rules['feat']).flatten(1), dim=1).mean()
    de = torch.linalg.vector_norm((states-rules['edge_attr']).flatten(1), dim=1).mean()
    n = node.shape[1]; rows, cols = torch.tril_indices(n, n, offset=-1, device=node.device)
    da = F.binary_cross_entropy((1-states[..., 0]).clamp(1e-7, 1-1e-7), rules['adj'][:, rows, cols])
    mu, logvar = rules['z_mu'], rules['z_logvar']
    kl = (.5*((1-logvar)+(logvar.exp()+mu.pow(2))/torch.tensor(1., device=node.device).exp()-1)).mean()
    return 10*dx+10*de+100*da+100000*kl, states, {'node': float(dx.detach()), 'edge': float(de.detach()),
                                               'adjacency': float(da.detach()), 'kl': float(kl.detach())}


def train_update(model, optimizer, fss, bridge, index, permutation):
    model.train(); bridge.eval(); optimizer.zero_grad(set_to_none=True)
    rules = model.get_rules(fss)
    regularizer, states, components = reconstruction(rules)
    regularizer.backward(retain_graph=True)
    counters = collections.Counter(); nll_sum = 0.; eligible = set(); flipped = set(); products = set(); witnesses = []
    for position in permutation:
        row, parent, template, mapping, before = index[int(position)]
        counters['lhs_matches'] += 1
        if before['predicted_label'] != 1:
            counters['source_ineligible'] += 1
            continue
        eligible.add(row['id']); rid = template.native_rule_index
        try:
            product = materialize(parent, template, mapping, rules['features_reconst'][rid], states[rid])
        except ValueError as exc:
            counters[str(exc)] += 1
            nll_sum += -float(np.log(np.finfo(np.float32).eps))
            continue
        counters['valid_complete_product'] += 1
        result = bridge.score_materialized(product)
        loss = -result['y_pred'][0, 0]
        if not torch.isfinite(loss): raise ValueError('nonfinite frozen GINE training NLL')
        # One optimizer update after the complete logical budget. This small
        # backward frees per-application GINE buffers while retaining the one
        # common sampled generator decoder graph; no microbatch update drift.
        (10*loss/len(permutation)).backward(retain_graph=True)
        nll_sum += float(loss.detach())
        flip = int(result['logits'].argmax(-1)) == 0
        counters['strict_flip'] += int(flip); products.add(product.canonical_smiles)
        if flip:
            flipped.add(row['id'])
            if len(witnesses) < 20:
                witnesses.append({'parent_id': row['id'], 'rule_index': rid, 'mapping': list(mapping.items()),
                    'parent_smiles': row['smiles'], 'canonical_smiles': product.canonical_smiles, 'before': before,
                    'after_probabilities': result['y_pred'].detach().exp()[0].cpu().tolist(), 'split': 'train'})
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('nonfinite generator gradient')
    if any(p.grad is not None for p in bridge.model.parameters()):
        raise ValueError('frozen GINE received parameter gradients')
    optimizer.step()
    return {'counts': dict(counters), 'loss_components': components,
            'mean_target_nll': nll_sum/max(1, len(permutation)), 'unique_products': len(products),
            'source_eligible_parents': len(eligible), 'flipped_train_parents': len(flipped), 'witnesses': witnesses}


def validation_score(model, fss, bridge, index):
    counts = collections.Counter(); valid = set(); flip = set(); nll = 0.; witnesses = []
    model.eval(); bridge.eval()
    with torch.no_grad():
        rules = model.get_rules(fss)
        states = [joint_states(a, e) for a, e in zip(rules['adj_reconst'], rules['edge_attrs_reconst'])]
        for row, parent, template, mapping, before in index:
            counts['lhs_matches'] += 1
            if before['predicted_label'] != 1: continue
            rid = template.native_rule_index
            try:
                product = materialize(parent, template, mapping, rules['features_reconst'][rid], states[rid])
            except ValueError as exc:
                counts[str(exc)] += 1; nll += -float(np.log(np.finfo(np.float32).eps)); continue
            result = bridge.score_materialized(product)
            valid.add(row['id']); counts['valid_complete_product'] += 1
            nll += float(-result['y_pred'][0, 0])
            if int(result['logits'].argmax(-1)) == 0:
                flip.add(row['id']); counts['strict_flip'] += 1
                if len(witnesses) < 20:
                    witnesses.append({'parent_id': row['id'], 'rule_index': rid, 'mapping': list(mapping.items()),
                                      'canonical_smiles': product.canonical_smiles, 'before': before,
                                      'after_probabilities': result['y_pred'].exp()[0].cpu().tolist(), 'split': 'validation'})
    score = (len(flip), len(valid), -nll/max(1, len(index)))
    return score, {'counts': dict(counts), 'parents_with_flip': len(flip), 'parents_with_valid': len(valid),
                   'mean_target_nll': nll/max(1, len(index)), 'witnesses': witnesses}, rules


def confirm_best_train_feasibility(model, fss, bridge, index):
    """Train-only existence test of the validation-selected frozen weights.

    Stopping at the first true recourse is an explicitly labelled feasibility
    witness, not a full train coverage estimate or a test-based stopping rule.
    """
    model.eval(); counts = collections.Counter()
    with torch.no_grad():
        rules = model.get_rules(fss)
        states = [joint_states(a,e) for a,e in zip(rules['adj_reconst'],rules['edge_attrs_reconst'])]
        for row,parent,template,mapping,before in index:
            counts['examined_train_matches']+=1; rid=template.native_rule_index
            try: product=materialize(parent,template,mapping,rules['features_reconst'][rid],states[rid])
            except ValueError as exc: counts[str(exc)]+=1; continue
            counts['valid_complete_products']+=1
            result=bridge.score_materialized(product)
            if before['predicted_label']==1 and int(result['logits'].argmax(-1))==0:
                return {'strict_flip_witness_found':True,'search_complete':False,'counts':dict(counts),
                    'total_train_matches':len(index),'parent_id':row['id'],'parent_smiles':row['smiles'],
                    'canonical_smiles':product.canonical_smiles,'mapping':list(mapping.items()),'native_rule_index':rid,
                    'before':before,'after_probabilities':result['y_pred'].exp()[0].cpu().tolist(),
                    'calibration_loaded':False,'test_loaded':False}
    return {'strict_flip_witness_found':False,'search_complete':True,'counts':dict(counts),
        'total_train_matches':len(index),'calibration_loaded':False,'test_loaded':False}


def real_oracle_identity_canary(model, fss, bridge, item):
    """Known legal train identity fixture through real generator and GINE.

    Forward identity is an explicitly synthetic engineering fixture, never a
    generated witness or training objective. Backward uses the actual decoder
    graph, proving the valid-product path even when old decoded graphs fail.
    """
    rng=snapshot_rng(); training=model.training; initial=copy.deepcopy(model.state_dict())
    model.train(); model.zero_grad(set_to_none=True)
    row,parent,template,mapping,before=item; rid=template.native_rule_index
    rules=model.get_rules(fss); raw=rules['features_reconst'][rid]
    states=joint_states(rules['adj_reconst'][rid],rules['edge_attrs_reconst'][rid])
    features=template.lhs_feature.to(raw.device)+(raw-raw.detach())
    edges=template.lhs_edge_attr.to(states.device)+(states-states.detach())
    product=materialize(parent,template,mapping,features,edges)
    if product.canonical_smiles != parent.canonical_smiles:
        raise ValueError('real train identity materialization changed the parent')
    actual=bridge.score_materialized(product); expected=predict_smiles(bridge,parent.canonical_smiles)
    probabilities=actual['y_pred'].exp()[0]
    if not torch.allclose(probabilities.detach().cpu(),torch.tensor(expected['probabilities']),atol=2e-6,rtol=0):
        raise ValueError('real GINE identity-forward parity failed')
    (-actual['y_pred'][0,0]).backward()
    finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    norm=sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    if not finite or norm<=0 or any(p.grad is not None for p in bridge.model.parameters()):
        raise ValueError('real generator/GINE identity gradient failed')
    result={'state':'PASS','parent_id':row['id'],'canonical_smiles':product.canonical_smiles,
        'probabilities':probabilities.detach().cpu().tolist(),'generator_gradient_l1':norm,
        'frozen_GINE_gradient':False,'fixture_kind':'synthetic_train_identity_not_generated_recourse',
        'target_flip_claimed':False,'calibration_loaded':False,'test_loaded':False}
    model.zero_grad(set_to_none=True); model.load_state_dict(initial,strict=True); model.train(training); restore_rng(rng)
    return result


def require_gpu_canary(config, config_sha):
    root=Path(config['gpu_canary_root'])
    terminal=read_json(root/'terminal.json'); execution=read_json(root/'execution_receipt.json')
    reload_receipt=read_json(root/'reload_receipt.json'); identity=read_json(root/'identity_oracle_canary.json')
    if (terminal.get('state')!='CANARY_COMPLETE' or terminal.get('optimizer_updates')!=2
            or execution.get('config_sha256')!=config_sha or execution.get('device')!='cuda:0'
            or reload_receipt.get('state')!='PASS' or reload_receipt.get('fresh_generator_loaded') is not True
            or reload_receipt.get('engineering_next_optimizer_update')!=3 or identity.get('state')!='PASS'
            or identity.get('frozen_GINE_gradient') is not False or identity.get('target_flip_claimed') is not False):
        raise ValueError('formal requires the bound real GPU/update/reload/identity canary')


def run_training(config_path, rematerialization_root, output_root, device, *, resume=False, canary=False,
                 boundary_check=lambda: None):
    boundary_check()
    config = read_json(config_path)
    if config.get('training_contract') != TRAINING_CONTRACT:
        raise ValueError('actual training objective not sealed')
    config_sha = stable_sha256(config); evidence = Path(rematerialization_root)
    if not canary and not resume:
        require_gpu_canary(config,config_sha)
    terminal = read_json(evidence/'terminal.json')
    if terminal.get('state') != 'REMATERIALIZATION_COMPLETE' or not terminal['repair_training_required']:
        raise ValueError('train feasibility must require the one repair campaign')
    if read_json(evidence/'repair_contract.json')['joint_contract'] != config['joint_contract']:
        raise ValueError('rematerialization/materializer contract drift')
    output = Path(output_root)
    run_kind = 'ENGINEERING_CANARY' if canary else 'FORMAL_REPAIR_FINETUNE'
    if resume:
        checkpoint = torch.load(output/'latest.pt', map_location='cpu', weights_only=False)
        if checkpoint['config_sha256'] != config_sha: raise ValueError('checkpoint contract changed')
        if checkpoint.get('run_kind') != run_kind or checkpoint.get('output_root') != str(output.resolve()):
            raise ValueError('canary promotion or cross-root checkpoint resume is forbidden')
        if not canary:
            ledger = read_json(config['formal_campaign_ledger'])
            if ledger.get('fresh_campaigns_used') != 1 or ledger.get('output_root') != str(output) or ledger.get('config_sha256') != config_sha:
                raise ValueError('formal resume requires the original one-shot ledger')
    else:
        output.mkdir(parents=True, exist_ok=False); checkpoint = None
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(7)
    summary, original_rules, templates, train, val, bridge = setup(config, device)
    model, authority = load_generator(config, summary, original_rules, bridge, device)
    fss = {k: original_rules[k].detach().to(device) for k in ('feat', 'adj', 'edge_attr')}
    train_index = load_index(evidence/'train_index.jsonl', train, templates)
    val_index = load_index(evidence/'validation_index.jsonl', val, templates)
    train_index = [row for row in train_index if row[4]['predicted_label'] == 1]
    if not train_index or not val_index: raise ValueError('empty bound train or validation match universe')
    optimizer = torch.optim.Adam(model.parameters(), lr=.1, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.9)
    schedule = torch.Generator(device='cpu').manual_seed(7)
    epoch_start = 0; best = None
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True); optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']); restore_rng(checkpoint['rng'])
        schedule.set_state(checkpoint['schedule_rng']); epoch_start = checkpoint['epoch_completed']
        best = checkpoint['best_validation_score']
    elif not canary:
        ledger = Path(config['formal_campaign_ledger'])
        with ledger.open('x') as stream:
            json.dump({'fresh_campaigns_used': 1, 'max_fresh_campaigns': 1, 'seed': 7,
                       'output_root': str(output), 'config_sha256': config_sha, 'started_at': utc_now()}, stream)
            stream.flush(); os.fsync(stream.fileno())
    atomic_json(output/'execution_receipt.json', {'config_sha256': config_sha, 'pid': os.getpid(),
                'stage': 'GPU_TRAINING_CANARY' if canary else 'GENERATOR_REPAIR_FINETUNE',
                'official_source_commit': authority['official_commit'], 'classifier_unchanged': True,
                'test_loaded': False, 'calibration_loaded': False, 'mining_rerun': False,
                'source_generator': summary['globalgce_model_checkpoint'], 'train_matches': len(train_index),
                'validation_matches': len(val_index), 'device': device, 'torch_version':torch.__version__,
                'torch_threads':torch.get_num_threads(), 'cudnn_tf32':torch.backends.cudnn.allow_tf32,
                'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
                'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()})
    if canary and not resume:
        atomic_json(output/'identity_oracle_canary.json',real_oracle_identity_canary(model,fss,bridge,train_index[0]))
    max_epochs = 2 if canary else 100
    for epoch in range(epoch_start, max_epochs):
        # Existing owner may pause only before a new complete optimizer update.
        # The prior update's full state and sampler cursor are already durable.
        boundary_check()
        started = time.monotonic()
        indices = torch.randperm(len(train_index), generator=schedule)[:min(2500, len(train_index))]
        if canary: indices = indices[:8]  # Explicit bounded engineering test, not full science.
        result = train_update(model, optimizer, fss, bridge, train_index, indices)
        scheduler.step()
        record = {'epoch': epoch+1, 'optimizer_updates': epoch+1, 'time': utc_now(),
                  'elapsed_seconds': time.monotonic()-started, **result}
        if (epoch+1) % 5 == 0 or epoch+1 == max_epochs:
            score, validation, rules = validation_score(model, fss, bridge, val_index[:8] if canary else val_index)
            record['validation'] = validation
            if best is None or tuple(score) > tuple(best):
                best = tuple(score)
                atomic_torch(output/'best_generator.pt', model.state_dict())
                atomic_torch(output/'best_rules.pt', {k: v.detach().cpu() for k, v in rules.items()})
                atomic_json(output/'best_validation.json', {'epoch': epoch+1, 'score': list(score), **validation})
        payload = {'epoch_completed': epoch+1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(), 'rng': snapshot_rng(), 'schedule_rng': schedule.get_state(),
                   'config_sha256': config_sha, 'best_validation_score': best,
                   'run_kind':run_kind,'output_root':str(output.resolve()),
                   'train_index_binding': str(evidence/'train_index.jsonl'), 'joint_contract': config['joint_contract']}
        atomic_torch(output/'latest.pt', payload)
        # Full semantic payload: optimizer moments, counters, sampler and RNG
        # are as important as the classifier-independent generator weights.
        # RNG and sampler byte tensors must remain CPU even for CUDA weights.
        # load_state_dict transfers parameter/optimizer values appropriately.
        reopened = torch.load(output/'latest.pt', map_location='cpu', weights_only=False)
        assert_semantic_equal(payload, reopened)
        with (output/'training.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n'); stream.flush(); os.fsync(stream.fileno())
        atomic_json(output/'heartbeat.json', {**record, 'pid': os.getpid(), 'stage': 'TRAINING', 'checkpoint': str(output/'latest.pt')})
    if canary:
        boundary_check()
        # A fresh independent generator and optimizer actually consume the
        # saved state and take a finite next update. This is engineering work,
        # not another full campaign and not a claim of old-trajectory parity.
        reloaded, _ = load_generator(config, summary, original_rules, bridge, device)
        reloaded.load_state_dict(reopened['model'], strict=True)
        reload_optimizer = torch.optim.Adam(reloaded.parameters(), lr=.1, weight_decay=1e-5)
        reload_optimizer.load_state_dict(reopened['optimizer'])
        reload_scheduler = torch.optim.lr_scheduler.StepLR(reload_optimizer, step_size=10, gamma=.9)
        reload_scheduler.load_state_dict(reopened['scheduler'])
        assert_semantic_equal(reopened['model'], reloaded.state_dict())
        assert_semantic_equal(reopened['optimizer'], reload_optimizer.state_dict())
        assert_semantic_equal(reopened['scheduler'], reload_scheduler.state_dict())
        restore_rng(reopened['rng']); schedule.set_state(reopened['schedule_rng'])
        next_indices = torch.randperm(len(train_index), generator=schedule)[:min(8, len(train_index))]
        next_record = train_update(reloaded, reload_optimizer, fss, bridge, train_index, next_indices)
        reload_scheduler.step()
        atomic_json(output/'reload_receipt.json', {'state': 'PASS', 'fresh_generator_loaded': True,
            'optimizer_scheduler_rng_sampler_restored': True, 'saved_optimizer_updates': max_epochs,
            'engineering_next_optimizer_update': max_epochs+1, 'next_update': next_record,
            'full_campaign_quota_consumed': False, 'calibration_loaded': False, 'test_loaded': False})
    state = 'CANARY_COMPLETE' if canary else 'REPAIR_TRAINING_COMPLETE'
    feasible = None
    if not canary:
        model.load_state_dict(torch.load(output/'best_generator.pt',map_location=device,weights_only=False),strict=True)
        feasible=confirm_best_train_feasibility(model,fss,bridge,train_index)
        atomic_json(output/'best_train_feasibility.json',feasible)
    result = {'state': state, 'optimizer_updates': max_epochs, 'best_validation_score': best,
              'test_loaded': False, 'calibration_loaded': False, 'mining_rerun': False,
              'performance_target_met': bool(feasible and feasible['strict_flip_witness_found']),
              'research_state': 'RESEARCH_TARGET_UNMET' if feasible and not feasible['strict_flip_witness_found'] else state,
              'completed_at': utc_now()}
    atomic_json(output/'terminal.json', result)
    if feasible and feasible['strict_flip_witness_found']:
        from src.baselines.bace_globalgce_chemaligned_export import export_pool
        export_pool(config_path, evidence, output/'train_candidates', output)
    return result
