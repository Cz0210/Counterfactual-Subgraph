#!/usr/bin/env python3
"""Reduce located original Mut frozen-order pairs; never infer paper compatibility."""
import argparse,csv,json,math,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.baselines.cm_crem_runtime import atomic_json,file_sha,require_compute_node


def truth(value):
    if str(value).lower() not in {'true','false','1','0'}: raise ValueError('Missing boolean evidence')
    return str(value).lower() in {'true','1'}


def reduce(rows,ids,theta,cap):
    parents=list(dict.fromkeys(r['parent_id'] for r in rows))
    pairs={}
    for r in rows:
        key=(r['parent_id'],r['candidate_id'])
        if key in pairs:raise ValueError('Duplicate parent/candidate row')
        if r['candidate_id'] not in ids:raise ValueError('Pair outside frozen sequence')
        d=float(r['distance'])
        if not math.isfinite(d) or d<0:raise ValueError('Missing/nonfinite raw distance')
        flip=int(r['pred_before'])==1 and int(r['pred_after'])==0
        if flip!=truth(r['teacher_strict_flip']):raise ValueError('Strict flip evidence conflict')
        if r['solver']!='exact_emd2' or r['feature_cost']!='cosine' or r['node_mass']!='uniform':raise ValueError('Distance contract changed')
        pairs[key]=d if flip and truth(r['delete_valid']) else math.inf
    if len(pairs)!=len(parents)*len(ids):raise ValueError('Incomplete Cartesian input')
    best={p:math.inf for p in parents};prefix=[];detail=[]
    for k in range(1,21):
        if k<=len(ids):
            for p in parents:best[p]=min(best[p],pairs[p,ids[k-1]])
        finite=[v for v in best.values() if math.isfinite(v)]
        covered=sum(v<=theta for v in best.values())
        prefix.append(dict(k=k,k_effective=min(k,len(ids)),denominator=len(parents),covered_count=covered,
             coverage=covered/len(parents),strict_recourse_count=len(finite),
             conditional_median=statistics.median(finite) if finite else 'N/A',
             fixed_capped_mean=statistics.mean(min(v,cap) for v in best.values()),theta_star=theta,cost_cap=cap))
        detail.extend(dict(k=k,parent_id=p,best_distance=v,strict_recourse_available=math.isfinite(v)) for p,v in best.items())
    return prefix,detail


def write_csv(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--out-dir',required=True);a=p.parse_args();require_compute_node()
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False)
    base=Path('/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/mutagenicity/final')
    threshold=base/'ours_wnode_a2_test_v1/thresholds.json';t=json.load(threshold.open())
    if t['test_used'] or t['threshold_source']!='frozen_calibration_selector':raise ValueError('Threshold not calibration frozen')
    sources={str(threshold):file_sha(threshold)};summary=[]
    for method,name,orderfile in [('GCFExplainer','gcfexplainer_native5000_top20_wnode_test_v1','selected_sequence.jsonl'),
                                 ('GlobalGCE','globalgce_wnode_frequency_top20_test_v1','selected_top20.csv')]:
        r=base/name;order=r/orderfile;raw=r/'test_pair_details.csv'
        chosen=[json.loads(s) for s in order.read_text().splitlines()] if order.suffix=='.jsonl' else list(csv.DictReader(order.open()))
        if [int(x['rank']) for x in chosen]!=list(range(1,21)):raise ValueError('Original rank sequence incomplete')
        ids=[x['candidate_id'] for x in chosen]
        if len(set(ids))!=20:raise ValueError('Duplicate candidate sequence')
        prefix,detail=reduce(list(csv.DictReader(raw.open())),ids,t['theta_star'],t['cost_cap'])
        folder=out/method;folder.mkdir();write_csv(folder/'prefix_metrics.csv',prefix);write_csv(folder/'parent_best_distances.csv',detail)
        best=[d['best_distance'] for d in detail if d['k']==20];grid=sorted({0.,t['theta_star'],t['cost_cap'],*[x for x in best if math.isfinite(x)]})
        ecdf=[dict(k=20,threshold=x,coverage=sum(d<=x for d in best)/len(best),covered_count=sum(d<=x for d in best),denominator=len(best)) for x in grid]
        write_csv(folder/'figure4_k20_exact.csv',ecdf);write_csv(folder/'table2_k20.csv',[prefix[-1]])
        sources.update({str(order):file_sha(order),str(raw):file_sha(raw)})
        summary.append(dict(dataset='Mutagenicity',method=method,**prefix[-1]))
    write_csv(out/'recovered_k20.csv',summary)
    atomic_json(out/'audit.json',dict(status='ORIGINAL_ORDER_RAW_REDUCTION_COMPLETE',sources=sources,
        original_orders_found=2,orders_reconstructed=0,new_oracle_or_distance_evaluation=False,
        independent_model_execution_review=False,paper_family_binding='PENDING_CURRENT_AUTHORITY_THRESHOLD_PLACEHOLDER_RECONCILIATION',
        main_authority_written=False),immutable=True)
    print(json.dumps(summary))
if __name__=='__main__':main()
