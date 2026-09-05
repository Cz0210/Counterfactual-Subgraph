#!/usr/bin/env python3
"""Tiny CPU-only real-bundle forward/load/parameter audit; no training."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def run(bundle_root, output):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    import yaml
    import rdkit
    from src.ablations.gnn.hpc_bundle import verify_bundle, atomic_json
    from src.ablations.gnn.cpu_training import effective_training_config
    from src.ablations.gnn.temperature_preflight import validate_explicit_validation_temperature_config
    from src.data.molecular_graph_featurizer import MolecularFeatureSchema, MolecularGraphFeaturizer
    from src.models.gnn_backbone_registry import build_backbone, get_gnn_backbone_spec
    from src.models.gatedgcn_plus_backbone import GATEDGCN_PLUS_ALLOWED_HIDDEN_DIMS, GATEDGCN_PLUS_OFFICIAL_COMMIT
    from src.models.graphgps_backbone import compute_topology_only_random_walk_pe
    from src.oracles.gnn_oracle import load_gnn_checkpoint_bundle
    torch.set_num_threads(1)
    root=Path(bundle_root)
    manifest=verify_bundle(root)
    temperature_preflight = {
        name: validate_explicit_validation_temperature_config(
            effective_training_config(root, manifest, name))
        for name in ("gin", "gcn", "gatv2", "gatedgcn_plus")
    }
    schema=MolecularFeatureSchema.from_dict(json.loads((root/manifest["feature_schema_path"]).read_text()))
    with (root/manifest["splits"]["train"]).open() as stream:
        first=next(csv.DictReader(stream))
    graph=MolecularGraphFeaturizer(schema).featurize(first["smiles"])
    x=torch.tensor(graph.node_features,dtype=torch.long)
    edges=torch.tensor(graph.edge_index,dtype=torch.long)
    attr=torch.tensor(graph.edge_features,dtype=torch.long)
    batch=torch.zeros(len(x),dtype=torch.long)
    pe=compute_topology_only_random_walk_pe(edges,num_nodes=len(x),walk_length=16)
    inputs=SimpleNamespace(x=x,edge_index=edges,edge_attr=attr,batch=batch,random_walk_pe=pe)
    gine,_=load_gnn_checkpoint_bundle(root/manifest["gine_reference_root"],device="cpu")
    reference_count=sum(p.numel() for p in gine.parameters())
    models={"gine":gine}
    for name in ("gin","gcn","gatv2","gatedgcn_plus"):
        config=yaml.safe_load((root/manifest["backbone_configs"][name]).read_text())
        models[name]=build_backbone(name,config,feature_schema=schema,num_classes=2)
    rows={}
    for name,model in models.items():
        model.eval()
        with torch.no_grad():
            logits=model(inputs)
        if tuple(logits.shape)!=(1,2) or not torch.isfinite(logits).all():
            raise ValueError(f"invalid CPU forward: {name}")
        rows[name]={"forward":"PASS","parameter_count":sum(p.numel() for p in model.parameters()),
                    "edge_feature_mode":get_gnn_backbone_spec(name).edge_feature_mode}
    widths=[]
    base=yaml.safe_load((root/manifest["backbone_configs"]["gatedgcn_plus"]).read_text())
    for width in GATEDGCN_PLUS_ALLOWED_HIDDEN_DIMS:
        base["gnn"]["hidden_dim"]=width
        model=build_backbone("gatedgcn_plus",base,feature_schema=schema,num_classes=2)
        count=sum(p.numel() for p in model.parameters())
        widths.append({"hidden_dim":width,"parameter_count":count,"difference_ratio":abs(count-reference_count)/reference_count})
    selected=min(widths,key=lambda item:(item["difference_ratio"],item["hidden_dim"]))
    frozen=yaml.safe_load((root/manifest["backbone_configs"]["gatedgcn_plus"]).read_text())["gnn"]["hidden_dim"]
    if frozen != selected["hidden_dim"]:
        raise ValueError("frozen parameter match differs from actual closest candidate")
    result={"status":"PASS","cpu_only":True,"torch":torch.__version__,"rdkit":rdkit.__version__,
            "python":platform.python_version(),"backbones":rows,"parameter_match":selected,"parameter_candidates":widths,
            "gatedgcn_plus_source_commit":GATEDGCN_PLUS_OFFICIAL_COMMIT,"train_rows_used":1,
            "test_used_for_selection":False,"main_matrix_write_allowed":False,
            "new_ablation_temperature_preflight":temperature_preflight,
            "gine_temperature_refit":False,
            "bundle_manifest_sha256":manifest["manifest_sha256"]}
    atomic_json(output,result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/hpc.yaml")
    p.add_argument("--bundle-root",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    print(json.dumps(run(a.bundle_root,a.output),indent=2))
