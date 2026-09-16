"""One authorized V2-budget search round, using train only and absolute theta.

Saved policy seeds plus structural beam expansion; no new LM/PPO calls. Both
old oracle reuse and new graph queries are counted, never as batch counts.
"""
import hashlib,json,math,time
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from rdkit import Chem
from .ours_taste_focus_matrix import read_json,dump_json
from .ours_taste_theta010 import digest,raw_valid_distances,metrics
from .ours_taste_focus_selector import ceilings,Selector,write_csv
from .ours_taste_search import seed_states,fragment_for_set,connected_sets
from .ours_taste_search_chain import append_segment,evaluate_delta,check_resource
from .tastemolnet_ours_full import stable_sha256,load_prepared_split,select_on_calibration,CONNECTED_ACTION_SEMANTICS,CONNECTED_MATCH_SELECTION_POLICY,DISTANCE_IMPLEMENTATION_VERSION
from src.chem.hard_deletion import apply_hard_deletion_match


def choose_strata(cohort,best):
    groups={name:defaultdict(list) for name in ('far','none','covered')}
    for i,(parent,v) in enumerate(zip(cohort,best)):
        category='none' if not np.isfinite(v) else 'far' if v>.1 else 'covered'
        groups[category][parent['scaffold']].append(i)
    def stable(i):return hashlib.sha256(('7:'+cohort[i]['parent_id']).encode()).hexdigest()
    queues={}
    for cat,scaffolds in groups.items():
        for v in scaffolds.values():v.sort(key=stable)
        q=[]
        while scaffolds:
            for s in sorted(list(scaffolds)):
                q.append(scaffolds[s].pop(0))
                if not scaffolds[s]:del scaffolds[s]
        queues[cat]=q
    chosen=[];counts={}
    for cat,n in [('far',64),('none',48),('covered',16)]:
        part=queues[cat][:n];chosen.extend(part);queues[cat]=queues[cat][n:];counts[cat]=len(part)
    for cat in ('none','far','covered'):
        part=queues[cat][:128-len(chosen)];chosen.extend(part);counts[cat]+=len(part)
    assert len(chosen)==len(set(chosen))<=128
    return chosen,counts


def search(parent,pool,scorer,provider,known):
    mol=Chem.MolFromSmiles(parent['smiles']);assert parent['pred_before']==1
    origins=seed_states(mol,parent,pool);allowed={1,2,3,4,6}
    pending=sorted((s for s in origins if len(s) in allowed),key=lambda s:(len(s),s))
    # A bounded connected seed expansion bridges size5 to size6 without ever
    # querying a size5 deletion (not in this search's authorized size subset).
    visited=set();beam=[];cache=dict(known);events=[];candidates={};q=w=hits=0
    while pending and q<64 and len(visited)<4096:
        atoms=pending.pop(0)
        if atoms in visited:continue
        visited.add(atoms);frag=fragment_for_set(mol,atoms)
        if frag is None:continue
        outcome=apply_hard_deletion_match(mol,atoms,parent_id=parent['parent_id'],match_id=0)
        if not outcome.valid:continue
        residual=outcome.residual_smiles;reused=residual in cache
        if reused:after=cache[residual];hits+=1
        else:after=scorer.score_smiles([residual])[0];cache[residual]=after;q+=1
        logits=after['logits'];margin=float(logits[1])-max(float(logits[0]),float(logits[2]));assert math.isfinite(margin)
        flip=int(after['predicted_label']) in (0,2);value=None
        cid='TASTE_RULE_'+stable_sha256({'fragment':frag})[:24].upper()
        if flip and w<32:
            measured=provider.distance_for_action(parent['smiles'],residual,action_context={
              'parent_id':parent['parent_id'],'candidate_id':cid,'match_index':0,'match_atom_indices':list(atoms),
              'teacher_sha256':scorer.checkpoint_id,'oracle_checkpoint_id':scorer.checkpoint_id,
              'action_semantics_version':CONNECTED_ACTION_SEMANTICS,'match_selection_policy':CONNECTED_MATCH_SELECTION_POLICY,
              'distance_implementation_version':DISTANCE_IMPLEMENTATION_VERSION})
            w+=1;assert measured['ok'];value=float(measured['distance']);assert math.isfinite(value) and value>=0
        covered=value is not None and value<=.1
        events.append({'parent_id':parent['parent_id'],'deletion_atoms':list(atoms),'canonical_fragment':frag,
          'residual_smiles':residual,'oracle':after,'oracle_query':not reused,'oracle_reused':reused,
          'strict_flip':flip,'raw_wnode':value,'distance_state':'MEASURED' if value is not None else 'NOT_REQUESTED_BUDGET_OR_NONFLIP',
          'covered_theta010':covered,'search_target_theta':.1,'original_parent_used':True,'one_connected_deletion':True})
        candidates.setdefault(cid,{'candidate_id':cid,'canonical_fragment':frag,'source_parent_ids':[parent['parent_id']],
          'source_modes':['SAVED_LM_SEEDED_THETA010_STRUCTURAL_SEARCH'],'train_strict_flip_witnesses':int(flip),
          'train_theta010_witnesses':int(covered),'train_deletion_size':len(atoms),'provenance':'V2_BUDGET_THETA010_SEARCH_NOT_NEW_LM'})
        key=(-int(covered),-int(flip),value if value is not None else float('inf'),margin,len(atoms),atoms)
        beam.append((key,atoms));beam.sort();beam=beam[:8]
        if not pending:
            nxt=set()
            for _,state in beam:
                for s in connected_sets(mol,[state]):
                    if len(s)==5:
                        nxt.update(t for t in connected_sets(mol,[s]) if len(t)==6 and t not in visited)
                    elif len(s) in allowed and s not in visited:nxt.add(s)
            pending=sorted(nxt,key=lambda s:(len(s),s))
    return list(candidates.values()),events,{'new_residual_oracle':q,'search_wnode':w,'oracle_cache_hits':hits,'visited_states':len(visited),'new_lm_outputs':0,'exhaustive':False}


def run(spec,out,compact,contract,scorer,provider,identity,pause):
    root=Path(spec['protocol_root']);c=read_json(root/'resolved_contract.json');prior=Path(spec['prior_remote_root'])
    decision=read_json(root/'R3_branch_decision.json')
    assert decision['state']=='ELIGIBLE_PREDECLARED_AFTER_R2_DELIVERY' and decision['decision_before_R2_test']
    assert read_json(root/'final_audit.json')['theta_actual']==.1,'DELIVER_R2_FIRST'
    assert c['new_search_target_theta']==.1
    pool=read_json(prior/'search-round1/candidate_pool_P1.json');ids0=[r['candidate_id'] for r in pool]
    assert digest(pool)==c['pool_p1_hash']
    cohort=read_json(prior/'train-development-attempt2/development_cohort.json')['parents']
    tr0=np.load(prior/'train-development-attempt2/train.npz');tr1=np.load(prior/'search-round1/train_new_matrix.npz')
    train=np.hstack([tr0['distances'],tr1['distances']]);assert not np.isnan(train).any()
    selected,counts=choose_strata(cohort,train.min(1))
    dump_json(out/'train_search_freeze.json',{'parents':[cohort[i]['parent_id'] for i in selected],'strata':counts,
        'theta':.1,'seed':7,'pool_hash':c['pool_p1_hash'],'sizes':[1,2,3,4,6],'beam':8,'test_read':False,
        'new_lm_outputs':0,'reason':'Reuse saved policy seeds; authorized structural discovery before any optional fresh LM calls'})
    import gzip
    known=defaultdict(dict)
    with gzip.open(prior/'search-round1/search_events.jsonl.gz','rt') as stream:
        for line in stream:
            r=json.loads(line);known[r['parent_id']][r['residual_smiles']]=r['oracle']
    ledger={'new_residual_oracle':0,'search_wnode':0,'oracle_cache_hits':0,'new_lm_outputs':0,'completed_parents':0,'historical_v1_queries':1679}
    additions={};stop=datetime.fromisoformat(c['search_cutoff'])
    for i in selected:
        if pause[0] or datetime.now(timezone.utc)>=stop:break
        check_resource(spec);new,events,used=search(cohort[i],pool,scorer,provider,known[cohort[i]['parent_id']])
        for k in ('new_residual_oracle','search_wnode','oracle_cache_hits'):ledger[k]+=used[k]
        ledger['completed_parents']+=1
        assert ledger['new_residual_oracle']<=8192 and ledger['search_wnode']<=4096
        for r in new:
            if r['candidate_id'] in ids0:continue
            target=additions.setdefault(r['candidate_id'],r)
            if target is not r:
                target['source_parent_ids']=sorted(set(target['source_parent_ids']+r['source_parent_ids']))
                target['train_theta010_witnesses']+=r['train_theta010_witnesses'];target['train_strict_flip_witnesses']+=r['train_strict_flip_witnesses']
        append_segment(out/'search_events.jsonl.gz',events);dump_json(out/'budget_ledger.json',ledger)
        dump_json(out/'progress.json',{'state':'R3_TRAIN_THETA010_SEARCH',**ledger,'observed_at':datetime.now(timezone.utc).isoformat()})
    if pause[0]:raise RuntimeError('PAUSED_AT_TRAIN_PARENT_BOUNDARY')
    delta=sorted(additions.values(),key=lambda r:(-r['train_theta010_witnesses'],-r['train_strict_flip_witnesses'],-len(r['source_parent_ids']),r['train_deletion_size'],r['candidate_id']))[:512]
    p2=pool+delta;dump_json(out/'candidate_pool_P2.json',p2)
    dump_json(out/'pool_freeze.json',{'old_p1_count':len(pool),'new_unique_seen':len(additions),'new_unique_retained':len(delta),'p2_count':len(p2),'pool_hash':digest(p2),'test_used':False,'calibration_used_to_cut_pool':False})
    if not delta:
        dump_json(out/'terminal.json',{'state':'R3_NO_NEW_UNIQUE_RULES_VALID_RESULT','P2_equals_P1':True,'budget':ledger,'completed_at':datetime.now(timezone.utc).isoformat()});return
    end=datetime.fromisoformat(c['absolute_deadline'])
    td,_=evaluate_delta(cohort,delta,'train',scorer,provider,identity,out,pause,end,spec)
    dump_json(out/'train_change.json',{'P1_cov010':int((train<=.1).any(1).sum()),'P2_cov010':int((np.hstack([train,td])<=.1).any(1).sum()),'P1_reach':int(np.isfinite(train).any(1).sum()),'P2_reach':int(np.isfinite(np.hstack([train,td])).any(1).sum()),'base':len(cohort)})
    cp=load_prepared_split(Path(contract['calibration_path']),expected_split='calibration',expected_sha256=contract['calibration_sha256'])
    cd,pred=evaluate_delta([{'parent_id':p.parent_id,'smiles':p.smiles} for p in cp],delta,'calibration',scorer,provider,identity,out,pause,end,spec)
    oldcal=np.load(prior/'search-round1/calibration_P1.npz');assert np.array_equal(pred,oldcal['predictions'])
    cal=np.hstack([oldcal['distances'],cd]);ids=[r['candidate_id'] for r in p2]
    np.savez_compressed(out/'calibration_P2.npz',distances=cal,parents=oldcal['parents'],predictions=pred,candidates=np.array(ids))
    bounds=ceilings(cal,pred==1,.1);dump_json(out/'calibration_bounds.json',bounds)
    records=[{'candidate_id':cid,'parent_id':pid,'split':'calibration','pair_strict_flip':bool(np.isfinite(cal[i,j])),'wnode_distance':float(cal[i,j])} for i,pid in enumerate(oldcal['parents']) for j,cid in enumerate(ids)]
    initial,_=select_on_calibration(p2,records,theta_star=.1);del records
    s=Selector(cal,ids,.1,c['cost_cap'],c['auxiliary_theta_grid'],[Chem.MolFromSmiles(r['canonical_fragment']).GetNumHeavyAtoms() for r in p2])
    seq,trace=s.optimize([ids.index(r['candidate_id']) for r in initial],[b['selected'] for b in bounds.values()],**c['selector_budget'])
    r2=read_json(root/'R2_selection_freeze.json')['candidate_ids']
    recommend='R3' if s.key(seq)<s.key([ids.index(i) for i in r2]) else 'R2'
    order=[ids[i] for i in seq]
    dump_json(out/'selection_freeze.json',{'candidate_ids':order,'pool_hash':digest(p2),'order_sha':digest(order),'theta':.1,'cost_cap':c['cost_cap'],'test_used':False,'recommended_on_calibration':recommend,'trace':trace,'created_at':datetime.now(timezone.utc).isoformat()})
    write_csv(out/'calibration_prefix.csv',metrics(cal,pred,seq,.1,c['cost_cap']))
    saved=np.load(root/'R2-test-attempt2/test_selected_union.npz');missing=[i for i in order if i not in saved['candidates']]
    parents=load_prepared_split(Path(contract['test_path']),expected_split='test',expected_sha256=contract['declared_test_sha256'])
    assert [p.parent_id for p in parents]==saved['parents'].tolist()==c['test_ids']
    if missing:
        dd,pp=evaluate_delta([{'parent_id':p.parent_id,'smiles':p.smiles} for p in parents],[p2[ids.index(i)] for i in missing],'test',scorer,provider,identity,out,pause,end,spec)
        assert np.array_equal(pp,saved['predictions'])
    union=sorted(set(saved['candidates'].tolist()+order));data=np.column_stack([saved['distances'][:,saved['candidates'].tolist().index(i)] if i in saved['candidates'] else dd[:,missing.index(i)] for i in union])
    np.savez_compressed(out/'test_selected_union.npz',distances=data,parents=saved['parents'],predictions=saved['predictions'],candidates=np.array(union))
    write_csv(out/'test_prefix.csv',[{'variant':'R3','theta':theta,**r} for theta in (c['theta_old'],.1) for r in metrics(data,saved['predictions'],[union.index(i) for i in order],theta,c['cost_cap'])])
    dump_json(out/'terminal.json',{'state':'R3_COMPLETE_AWAITING_SAVED_RAW_AUDIT','new_unique':len(delta),'new_test_rules':missing,'new_test_pairs':len(missing)*len(parents),'recommended_on_calibration':recommend,'budget':ledger,'completed_at':datetime.now(timezone.utc).isoformat()})
