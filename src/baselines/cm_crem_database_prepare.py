"""One compute-node admission of the user-approved author database archive."""
from __future__ import annotations
import gzip
import hashlib
import os
from pathlib import Path
from src.baselines.cm_crem_runtime import atomic_json, checked_root, read_json, utc_now

HPC_SCOPE = Path('/share/home/u20526/czx')

class DigestReader:
    def __init__(self, stream):
        self.stream, self.md5, self.sha, self.count = stream, hashlib.md5(), hashlib.sha256(), 0
    def read(self, n=-1):
        data = self.stream.read(n)
        self.md5.update(data); self.sha.update(data); self.count += len(data)
        return data

def unpack_verified(archive: Path, output: Path, expected: dict, max_bytes: int) -> dict:
    """Single compressed read verifies MD5/SHA and gzip CRC while unpacking."""
    before = archive.stat()
    unpacked, sha = 0, hashlib.sha256()
    with archive.open('rb') as source, output.open('xb') as target:
        reader = DigestReader(source)
        with gzip.GzipFile(fileobj=reader, mode='rb') as compressed:
            while True:
                block = compressed.read(1024 * 1024)
                if not block:
                    break
                if unpacked == 0 and not block.startswith(b'SQLite format 3\0'):
                    raise ValueError('SQLITE_HEADER_INVALID')
                unpacked += len(block)
                if unpacked > max_bytes:
                    raise ValueError('UNPACKED_CAPACITY_BOUND_EXCEEDED')
                target.write(block); sha.update(block)
        if (reader.count != expected['compressed_bytes'] or
                reader.md5.hexdigest() != expected['published_md5'] or
                reader.sha.hexdigest() != expected['compressed_sha256']):
            raise ValueError('AUTHOR_MD5_OR_TRANSFER_SHA_OR_SIZE_CONFLICT')
        if unpacked < 16:
            raise ValueError('EMPTY_DATABASE')
        target.flush(); os.fsync(target.fileno())
    after = archive.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('SOURCE_ARCHIVE_CHANGED')
    output.chmod(0o444)
    return dict(compressed_bytes=reader.count, compressed_md5=reader.md5.hexdigest(),
                compressed_sha256=reader.sha.hexdigest(), uncompressed_bytes=unpacked,
                uncompressed_sha256=sha.hexdigest(), gzip_integrity='PASS')

def prepare(spec_path: Path, archive: Path, output_root: Path, run_root: Path,
            max_database_bytes: int = 8 * 1024**3) -> dict:
    from src.baselines.cm_crem_assets import prepare_job_scratch, validate_database_source
    from src.baselines.cm_crem_database_compat import verify_static_database_compatibility
    spec = read_json(spec_path)
    root = checked_root(output_root, HPC_SCOPE)
    archive = checked_root(archive, HPC_SCOPE)
    if not root.is_dir() or archive.parent != root:
        raise ValueError('EXACT_ASSET_ROOT_REQUIRED')
    if (root / 'database_receipt.json').exists():
        raise ValueError('ALREADY_VERIFIED_DO_NOT_REPEAT_PREPARATION')
    source = spec['asset_source_overlay']
    # License/source admission precedes any decompression or SQLite operation.
    provisional = dict(status='VERIFIED_COMPRESSED_COPY', url=source['download_url'],
                       compressed_md5=source['published_md5'],
                       compressed_sha256=source['compressed_sha256'],
                       compressed_bytes=source['compressed_bytes'], source_overlay=source)
    validate_database_source(spec, provisional, require_compatibility=False, require_content=False)
    reserve = 2 * 1024**3
    fs = os.statvfs(root)
    if fs.f_bavail * fs.f_frsize < max_database_bytes + reserve:
        raise ValueError('PERSISTENT_DATABASE_AND_RESERVE_CAPACITY_NOT_MET')
    scratch = prepare_job_scratch(run_root, required_bytes=max_database_bytes + archive.stat().st_size,
                                  reserve_bytes=reserve)
    if scratch['status'] != 'JOB_SCRATCH_READY':
        raise RuntimeError(scratch)
    atomic_json(root/'prepare_scratch.json', scratch, immutable=True)
    local = Path(scratch['root'])/'chembl22_sa2.db'
    integrity = unpack_verified(archive, local, source, max_database_bytes)
    atomic_json(root/'integrity.json', integrity, immutable=True)
    compatibility = verify_static_database_compatibility(local)
    atomic_json(root/'compatibility.json', compatibility, immutable=True)
    # New immutable persistent copy, checked as it is transferred from scratch.
    destination, tmp = root/'chembl22_sa2.db', root/'chembl22_sa2.db.part'
    if destination.exists():
        raise ValueError('DESTINATION_ALREADY_EXISTS')
    sha, count = hashlib.sha256(), 0
    with local.open('rb') as src, tmp.open('xb') as dst:
        for block in iter(lambda: src.read(1024*1024), b''):
            dst.write(block); sha.update(block); count += len(block)
        dst.flush(); os.fsync(dst.fileno())
    if sha.hexdigest() != integrity['uncompressed_sha256'] or count != integrity['uncompressed_bytes']:
        raise ValueError('PERSISTENT_COPY_CONFLICT')
    tmp.chmod(0o444)
    os.link(tmp, destination); tmp.unlink()
    final_archive = root/'chembl22_sa2.db.gz'
    if archive != final_archive:
        os.link(archive, final_archive); archive.unlink()
    result = {**provisional, **integrity, 'status':'VERIFIED_STATIC_COPY', 'schema':'cm_crem_author_archive_adoption_v1',
              'created_at':utc_now(), 'database_path':str(destination), 'local_database':str(local),
              'job_id':os.environ['SLURM_JOB_ID'], 'scratch':scratch,
              'download_state':'PUBLISHED_MD5_MATCH', 'license_state':'CC_BY_4_0_DATABASE_PERMISSION',
              'compatibility':compatibility, 'compatibility_database_sha256':integrity['uncompressed_sha256'],
              'source_description':'Author Zenodo archive ChEMBL22 SA2',
              'old_qsar_file_bytes_compared':False, 'old_failure_preserved':True,
              'database_read_only':True, 'original_archive_preserved':True}
    validate_database_source(spec, result)
    atomic_json(root/'database_receipt.json', result, immutable=True)
    return result
