"""Bounded LM-seeded connected-set search on train molecules only."""
from collections import Counter
import hashlib
import json
import math
from rdkit import Chem
from rdkit.Chem import BRICS
from src.chem.hard_deletion import enumerate_connected_hard_deletions,apply_hard_deletion_match
from src.eval.molclr_node_embeddings import canonicalize_smiles
from src.eval.tastemolnet_ours_full import stable_sha256


def fragment_for_set(mol,atoms):
    if not atoms or len(atoms)==mol.GetNumAtoms(): return None
    try:
        text=Chem.MolFragmentToSmiles(mol,atomsToUse=sorted(atoms),canonical=True,isomericSmiles=True)
        query=Chem.MolFromSmiles(text)
        if query is None or len(Chem.GetMolFrags(query))!=1 or query.GetNumAtoms()!=len(atoms):return None
        return Chem.MolToSmiles(query,canonical=True,isomericSmiles=True)
    except ValueError:return None


def connected_sets(mol,seeds):
    """Grow/shrink one atom, always applying the resulting set to original mol."""
    neighbors={i:{n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()} for i in range(mol.GetNumAtoms())}
    result=set()
    for seed in seeds:
        border=set().union(*(neighbors[i] for i in seed))-set(seed)
        for atom in border:result.add(tuple(sorted(set(seed)|{atom})))
        for atom in seed:
            smaller=set(seed)-{atom}
            if not smaller:continue
            reached={min(smaller)};stack=list(reached)
            while stack:
                for nxt in neighbors[stack.pop()]&smaller-reached: reached.add(nxt);stack.append(nxt)
            if reached==smaller:result.add(tuple(sorted(smaller)))
    return result


def seed_states(mol,parent,pool):
    found={}
    # Only saved LM rules originating at this train parent are initial LM seeds.
    for row in pool:
        if parent['parent_id'] not in row.get('source_parent_ids',[]):continue
        for outcome in enumerate_connected_hard_deletions(parent['smiles'],row['canonical_fragment']):
            found[outcome.match_atom_indices]='LLM_SEEDED_SEARCH'
    broken={tuple(sorted(bond)) for bond,_ in BRICS.FindBRICSBonds(mol)}
    remaining=set(range(mol.GetNumAtoms()))
    while remaining:
        comp={min(remaining)};stack=list(comp)
        while stack:
            i=stack.pop()
            for atom in mol.GetAtomWithIdx(i).GetNeighbors():
                j=atom.GetIdx()
                if tuple(sorted((i,j))) not in broken and j not in comp:comp.add(j);stack.append(j)
        remaining-=comp
        if len(comp)<mol.GetNumAtoms():found.setdefault(tuple(sorted(comp)),'STRUCTURE_SEARCH')
    for atom in mol.GetAtoms():
        if atom.GetDegree()==1:found.setdefault((atom.GetIdx(),),'STRUCTURE_SEARCH')
    return found


def search_parent(parent,pool,scorer,query_budget=64,beam_width=8):
    assert query_budget<=64 and beam_width<=8 and parent['pred_before']==1
    mol=Chem.MolFromSmiles(parent['smiles']);assert mol is not None
    origins=seed_states(mol,parent,pool)
    pending=sorted(origins,key=lambda x:(len(x),x));visited=set();beam=[];queries=0
    residual_cache={};events=[];candidates={};reasons=Counter()
    # A finite proposal ceiling also bounds chemistry-only invalid states.
    while pending and queries<query_budget and len(visited)<4096:
        current=pending.pop(0)
        if current in visited:continue
        visited.add(current)
        frag=fragment_for_set(mol,current)
        if frag is None:reasons['invalid_connected_fragment']+=1;continue
        outcome=apply_hard_deletion_match(mol,current,parent_id=parent['parent_id'],match_id=0)
        if not outcome.valid:reasons[outcome.invalid_reason]+=1;continue
        residual=outcome.residual_smiles
        cached=residual in residual_cache
        if not cached:
            if queries>=query_budget:break
            after=scorer.score_smiles([residual])[0];queries+=1;residual_cache[residual]=after
        else:after=residual_cache[residual]
        logits=after.get('logits')
        if logits is None:raise ValueError('SEARCH_REQUIRES_REAL_LOGITS')
        margin=float(logits[1])-max(float(logits[0]),float(logits[2]))
        assert math.isfinite(margin)
        flip=after['predicted_label'] in (0,2)
        origin=origins.get(current,'LLM_SEEDED_SEARCH')
        event={'parent_id':parent['parent_id'],'deletion_atoms':list(current),'canonical_fragment':frag,
               'residual_smiles':residual,'source':origin,'oracle':after,'strict_flip':flip,
               'oracle_query':not cached,'one_connected_deletion':True,'original_parent_used':True}
        events.append(event)
        # Structurally valid non-flips are permitted by this explicit pool-construction adaptation.
        cid='TASTE_RULE_'+stable_sha256({'fragment':frag})[:24].upper()
        entry=candidates.setdefault(cid,{'candidate_id':cid,'canonical_fragment':frag,'source_parent_ids':[],
                  'source_modes':[],'train_strict_flip_witnesses':0,'provenance':'SEARCH_ASSISTED_POOL_CONSTRUCTION',
                  'train_deletion_size':len(current)})
        if parent['parent_id'] not in entry['source_parent_ids']:entry['source_parent_ids'].append(parent['parent_id'])
        if origin not in entry['source_modes']:entry['source_modes'].append(origin)
        entry['train_strict_flip_witnesses']+=int(flip)
        beam.append((margin,len(current),current,origin));beam.sort();beam=beam[:beam_width]
        if not pending:
            next_states=set()
            for beam_row in beam:
                for state in connected_sets(mol,[beam_row[2]]):
                    if state not in visited:
                        origins.setdefault(state,beam_row[3]);next_states.add(state)
            pending.extend(next_states)
            pending.sort(key=lambda x:(len(x),x))
    return list(candidates.values()),events,{'oracle_queries':queries,'visited_states':len(visited),
           'invalid_funnel':dict(reasons),'exhaustive':False,'lm_new_outputs':0,'shrink_queries':0}
