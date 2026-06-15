# MolCLR GNN Embedding Selector Workflow

This workflow adds a graph-neural embedding redundancy option for the existing
class-level selector without changing selector defaults, training code, reward
code, or existing selected-subgraph artifacts.

## Purpose

The selector already supports:

```bash
--sim-metric embedding
--embedding-field final_fragment_embedding
```

MolCLR-GNN support writes a separate field:

```text
final_fragment_gnn_embedding
```

The selector then uses cosine similarity over this field as the redundancy
penalty. Coverage gain, counterfactual score, size penalty, filters, and CAMC
evaluation are unchanged.

## External MolCLR Runtime

MolCLR code and checkpoints are not committed to this repository. On HPC, place
the MolCLR checkout and pretrained checkpoint on local storage and pass them at
runtime:

```bash
MOLCLR_ROOT=/path/to/MolCLR \
MOLCLR_CKPT=/path/to/molclr/pretrained/checkpoint.pth \
sbatch scripts/slurm/add_molclr_gnn_embeddings_label1.sh
```

The code does not download MolCLR, PyG, checkpoints, or other model assets at
runtime. Missing `torch_geometric`, RDKit, MolCLR model classes, or checkpoint
files produce explicit errors.

## Ours Label-1 Flow

Generate MolCLR-GNN fragment embeddings:

```bash
MOLCLR_ROOT=/path/to/MolCLR \
MOLCLR_CKPT=/path/to/molclr/pretrained/checkpoint.pth \
sbatch scripts/slurm/add_molclr_gnn_embeddings_label1.sh
```

Run the selector with MolCLR-GNN embedding redundancy:

```bash
sbatch scripts/slurm/select_ours_molclr_gnn_embedding_label1.sh
```

Default input:

```text
outputs/hpc/full_candidate_pools/stable300_label1_merged_base_temp07/candidate_pool.jsonl
```

Default embedded output:

```text
outputs/hpc/full_candidate_pools/stable300_label1_merged_base_temp07/candidate_pool_with_molclr_gnn_embeddings.jsonl
```

Default selected top20:

```text
outputs/hpc/selectors/molclr_gnn_ours_embedding_label1/beta_20p0_gamma_5p0
```

## GT-Fullgraph Seed-13 Flow

The GT baseline first needs a selector-readable candidate pool. If it is
missing, run the CAMC motif-pool conversion workflow before generating MolCLR
embeddings.

```bash
sbatch scripts/slurm/convert_camc_gt_fullgraph_motif_pools_label1_clean.sh
```

Then generate MolCLR-GNN embeddings for seed 13 / `label1_1594411`:

```bash
MOLCLR_ROOT=/path/to/MolCLR \
MOLCLR_CKPT=/path/to/molclr/pretrained/checkpoint.pth \
sbatch scripts/slurm/add_molclr_gnn_embeddings_gt_label1_seed13.sh
```

Run the relaxed GT selector:

```bash
sbatch scripts/slurm/select_gt_molclr_gnn_embedding_label1_seed13.sh
```

The relaxed GT path does not require true teacher-rescored `cf_flip` and uses
`--min-cf-drop -999`, matching the existing GT proxy selector workflow.

## Legacy CAMC Re-evaluation

After both selectors finish, evaluate selected top20 sets with the legacy HIV
quick CAMC evaluator:

```bash
sbatch scripts/slurm/eval_hiv_quick_molclr_gnn_selectors_label1_legacy_camc.sh
```

Outputs:

```text
outputs/hpc/comparison/hiv_quick/legacy_camc_eval_label1/molclr_gnn_ours_beta20_gamma5_seed13
outputs/hpc/comparison/hiv_quick/legacy_camc_eval_label1/molclr_gnn_gt_beta20_gamma5_seed13
```

## Dry Run

For a small dependency and checkpoint smoke test:

```bash
MOLCLR_ROOT=/path/to/MolCLR \
MOLCLR_CKPT=/path/to/molclr/pretrained/checkpoint.pth \
MAX_ROWS=20 \
sbatch scripts/slurm/add_molclr_gnn_embeddings_label1.sh
```

This writes only the first 20 candidate rows to the configured output JSONL.
