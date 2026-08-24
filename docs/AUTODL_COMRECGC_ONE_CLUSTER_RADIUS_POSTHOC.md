# ComRecGC One-Cluster Radius Boundary Post-Hoc Gate

## Purpose

The c766 exact route computes official one-cluster coverage with Torch, but its
lineage trace historically widened each float32 NumPy distance to Python
float64 before applying strict `distance < delta`. For `delta=0.02`, the
float32 representation is slightly smaller than the Python float. A point
equal to float32 delta can therefore be retained by the widened comparison
even though Torch correctly excludes it as an exact boundary point.

New downstream runs cast the Python radius to the NumPy distance dtype before
strict comparison. No tolerance is introduced.

## Post-hoc audit

`audit_one_cluster_radius_boundary.py` accepts only a SHA-pinned terminal
one-cluster manifest and writes to a fresh root. It validates the terminal
manifest and all existing summary artifacts, hashes the source vector array,
and reopens the authority-bound physical or implicit pair rows.

Using the terminal fixed block size and saved Torch/NumPy centroids, it streams
the full vector source twice without running DBSCAN:

- historical widened NumPy mask;
- corrected radius-cast-to-distance-dtype NumPy mask;
- official Torch mask;
- raw mask SHA256 and pairwise difference counts;
- retained parent/candidate sets and exact-boundary counts;
- historical and corrected retained centroids, first medoids, and selected
  one-cluster traces.

The historical mask must reproduce the terminal retained-mask bytes, and the
Torch replay must reproduce the terminal official coverage fields. Source
stat identities must remain unchanged throughout.

If `old_vs_dtype_cast_diff_count=0`, the audit writes `PASS` last and the live
terminal is adoptable without rerunning DBSCAN. If the count is nonzero, it
writes `BLOCKED`, produces a hash-bound `corrected_downstream_trace.json`, and
requires a fresh downstream-only replay from the existing exact DBSCAN
manifest. Pair generation, theta filtering, and DBSCAN must not be rerun.

## AutoDL-only command

```bash
python scripts/baselines/comrecgc/audit_one_cluster_radius_boundary.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --terminal-one-cluster-manifest /absolute/one_cluster_summary/run_manifest.json \
  --expected-terminal-manifest-sha256 <sha256> \
  --output-dir /absolute/fresh/radius_boundary_posthoc
```

The paired Slurm script is static CLI synchronization only and always exits;
this audit is AutoDL CPU-only and must not run on HPC.
