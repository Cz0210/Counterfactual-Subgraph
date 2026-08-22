# AutoDL user-approved frozen v4 adoption

This path adopts exactly six legacy paper cells without rerunning science:

- AIDS: Ours, GCFExplainer, GlobalGCE;
- Mutagenicity: Ours, GCFExplainer, GlobalGCE.

It does **not** adopt CLEAR and it does **not** create ComRecGC evidence.

## Trust boundary

The tracked policy is
`configs/autodl/user_approved_frozen_v4_adoption_v1.json`. It pins the exact
source-directory basename and SHA-256 of these five files:

- `_RUN_COMPLETE.json`;
- `combined_manifest.json`;
- `figure3_gcf_style_aids_mut_data.csv`;
- `figure4_gcf_style_aids_mut_data.csv`;
- `table2_gcf_style_aids_mut.csv`.

No PNG, PDF, Markdown, or audit-text file is opened. Each adopted file receives
one content read and one SHA computation; those bytes are reused for parsing,
validation, byte-for-byte copying, row projection, and every manifest. On
AutoDL, procfs must show no writable descriptor to any adopted source file, and
the file stat identities must remain unchanged across the scan.

The source bundle does not embed full raw method output or the dataset, held-out
split, RF-checkpoint, and MolCLR-checkpoint hashes required by the generic
registry. Those fields remain empty with explicit
`NOT_EMBEDDED_IN_FROZEN_V4_SOURCE` status. The six cells are therefore
`ADOPTABLE_PASS` under `USER_APPROVED_FROZEN_V4`; they are not relabeled as
ordinary provenance-complete `FROZEN_PASS` results.

## AutoDL command

Run from an immutable execution worktree at the reviewed commit:

```bash
TS="$(date -u +%Y%m%dT%H%M%SZ)"
export AUTODL_PYTHON=/root/miniconda3/envs/smiles_pip118/bin/python
export AUTODL_RUNTIME_ROOT=/autodl-fs/data/counterfactual-subgraph-runtime
export FROZEN_V4_SOURCE_ROOT=/autodl-fs/data/incoming/counterfactual-subgraph-autodl-step0-20260820-141726/payload/project/outputs/hpc/eval/paper/aids_mutagenicity_wnode_gcf_style_matched_aids_v4
export FROZEN_V4_OUTPUT_ROOT="$AUTODL_RUNTIME_ROOT/outputs/autodl/paper_matrix/four_methods_four_datasets_v1/adoptions/user_approved_frozen_v4_${TS}"
scripts/autodl/adopt_user_approved_frozen_v4.sh
```

The output root must be absent and below the persistent runtime root. The
command is CPU-only and does not reserve a GPU.

## Output contract

The fresh aggregate root contains:

```text
adoption_manifest.json
adopted_source_inventory.json
explicit_cells.json
registry_exception_schema.json
supersession_manifest.json
source_bundle/                 # exact five approved files only
cells/
  aids/{ours,gcfexplainer,globalgce}/standardized/
  mutagenicity/{ours,gcfexplainer,globalgce}/standardized/
_RUN_COMPLETE.json
PASS
```

Every standardized root contains Figure 3, Figure 4, Table 2, prefix, N/A
parent/destination, summary, oracle/evaluation/run, artifact/freeze/audit, and
registry-exception files. Figure/Table numeric strings come directly from the
frozen source. Fields absent from v4 use `N/A` plus a reason; they are never
filled with zero or inferred values.

The 601-point Figure 4 grid is required to be identical across the three
adopted methods **within each dataset**. AIDS and Mutagenicity retain their own
frozen grids; no cross-dataset grid equality is imposed.

Use the six mappings in `explicit_cells.json` as repeatable
`--explicit-cell=DATASET/METHOD=ROOT` arguments to
`scripts/autodl/audit_four_methods_four_datasets.py`.

## Duplicate work policy

`supersession_manifest.json` authorizes a **new** controller manifest to omit
not-started duplicate repair/evaluation for these six cells. It does not edit
an old controller and does not stop any running task. Running ComRecGC work is
outside this adoption scope and must drain normally.

The Slurm wrapper is static CLI parity and exits with code 78 before the command.
Do not submit it; this campaign is AutoDL-only.
