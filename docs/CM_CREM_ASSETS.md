# CM-CReM static assets and node-local scratch

## 2026-09-10 authorized source overlay

The initially requested source remains
`https://www.qsar4u.com/files/cremdb/chembl22_sa2.db.gz` in the original scientific
identity. Its historical HTTP 403 evidence is retained. The user has separately
authorized the author's Zenodo DOI `10.5281/zenodo.16909329`, by Pavel Polishchuk,
with DataCite database-specific `CC-BY-4.0` rights. This is a source-provenance
overlay, not proof of byte equality with an unavailable older download, nor a
reason to rerun the unchanged 32-parent attribution / 64-pair pilot.

Pinned compressed asset: `chembl22_sa2.db.gz`, 350212897 bytes;
MD5 `91ce6b3d61270e927910162eeb63db43`;
SHA256 `6fe7f9534ae705fc508fa9be1d0c6a1baac988d5c1e67c8314e6524f4545c8fb`.
These identify the authorized gzip; actual decompression, source-copy proof and
CReM query compatibility are separate runtime evidence. Test fixtures are not
an official database and cannot supply production compatibility.

`validate_database_source(spec, receipt, require_compatibility=True, require_content=True)` returns
`actual_source_url`, `expected_url`, source mode, overlay digest and compatibility
verification state. It does not mutate either input or open/hash the DB. Invalid
bindings raise a specific `ValueError`. Its authorized overlay fields are:

```text
asset_source_overlay:
  record_doi, record_url, download_url, filename
  published_md5, compressed_bytes, compressed_sha256
  historical_url, old_file_bytes_compared=false
  license:
    identifier=CC-BY-4.0
    rights_uri=https://creativecommons.org/licenses/by/4.0/legalcode
    scope=database
    metadata_path, metadata_sha256
```

The metadata file is the small actual DataCite JSON response (or attributes),
copied alongside task assets. The helper verifies its content SHA, DOI, author
and rights. The download URL must use the exact approved Zenodo record/file,
optionally with `?download=1`; arbitrary mirror URLs are not accepted.
The exact DataCite `/legalcode` rights URI and the base CC-BY-4.0 URI are both
accepted without modifying the actual metadata.

The static copy receipt requires `status=VERIFIED_STATIC_COPY`, `url` equal to
the actual source, compressed MD5/SHA/bytes, and uncompressed SHA/bytes. Its
`compatibility` field embeds the real output of
`verify_static_database_compatibility`, with
`compatibility_database_sha256` equal to the verified uncompressed SHA. Radius1
schema, actual SELECTs, native bounded public-fixture replacement, pinned runtime
and unchanged read-only static state must all be evidenced. A bare PASS is
insufficient. `require_compatibility=False` is only an earlier source-binding
operation; its returned `compatibility_validated=false` is not generation-ready.
Before decompression, `require_content=False, require_compatibility=False` accepts
an actual `VERIFIED_COMPRESSED_COPY` receipt and the authorized compressed pins
without inventing an uncompressed SHA. It returns `content_validated=false` and
`uncompressed_sha256=null`; full staging/compatibility gates remain required.

## Job-local staging API

The existing `_job_scratch()` contract is unchanged: a supplied site directory
must itself be existing, owned, exclusive, job-bound and truly node-local.

Where the site leaves `SLURM_TMPDIR` empty, the previously audited T8 convention
allows project-private `mktemp` below `${TMPDIR:-/tmp}`. The explicit helper is:

```python
prepared = prepare_job_scratch(
    run_root, required_bytes=uncompressed_bytes, reserve_bytes=1024**3)
if prepared["status"] != "JOB_SCRATCH_READY":
    raise RuntimeError(prepared)
source = validate_database_source(spec, database_receipt)
staged = stage_static_database(
    source_path, receipt_path, reserve_bytes=1024**3,
    expected_url=source["actual_source_url"], scratch_receipt=prepared)
```

`run_root` must be an existing owned fresh-run child under the HPC CM baseline
root. The helper requires a numeric `SLURM_JOB_ID` plus matching actual
`SLURMD_NODENAME` and hostname. It does not create scratch on a login node merely
because a number is available. Selection uses actual `SLURM_TMPDIR`, then
actual `TMPDIR`, then the explicitly authorized `/tmp` convention. A nonempty
but unsafe earlier choice blocks rather than silently falling back.

Before creating a directory it checks the real Linux mount, available capacity,
file slots and parent permissions. Only real local disk filesystem types are
accepted; shared `/share`/`/ssdfs`, project persistent, AutoDL NVMe, tmpfs and
overlay are rejected. A trusted root-owned sticky `/tmp` is a permitted parent,
but the new `cm-crem-job-JOBID-*` child is exclusively owned mode0700.

The helper preserves raw environment values, never manufactures or overwrites
`SLURM_TMPDIR`/`TMPDIR`, and writes a small immutable job/host/path/mount receipt
under `run_root/scratch_receipts`. The same job reuses that verified directory.
`stage_static_database` accepts that dict or its absolute `receipt_path`, verifies
the on-disk binding against actual job, host, raw environment, inode, ownership
and mount, and performs its existing bounded static copy. Without the optional
receipt it still uses the strict legacy path. No new controller, lock registry,
local database fallback, deletion or shared-environment change is introduced.

The first staging call verifies the copied bytes and target once, preserving the
source. Subsequent same-job calls use immutable receipt/stat identities, not a
full DB hash per parent. Partial stages remain blocked and preserved. EIO is
infrastructure failure, never an empty candidate or zero result.

Focused regression:

```bash
/Users/cz0210/miniconda3/envs/smiles_local/bin/python -m pytest tests/test_cm_crem_assets.py -q
```
