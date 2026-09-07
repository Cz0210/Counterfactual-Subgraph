"""One-pass adoption of sealed native *whole-graph* WNode costs for BACE GIN.

Only finite raw costs are transferred. Old predictions, strict-flip masks,
selection infinities and the old selector are never used as new science.
The native cache namespace is retained as provenance, not silently rewritten.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from src.eval.bace_frozen_gnn_contracts import (
    atomic_json, load_bace_parents, read_json, stable_sha256,
)

BINDING_SCHEMA = "bace_native_fullgraph_raw_source_v1"
# Existing VerifiedRawGraphDistance consumer schema, not a new cache platform.
SCHEMA = "bace_reach_v2_raw_graph_cost_adoption_v1"
NATIVE_NAMESPACE = "bace_native_fullgraph_frozen_gine_wnode_v1"
OPERATIONS = {
    "gcfexplainer": ("full_counterfactual_graph", "official_vrrw_neurosed_greedy_fullgraph_v1"),
    "comrecgc": ("native_common_recourse_fullgraph", "official_comrecgc_lineage_unique_transition_medoid_v1"),
}
NUMERIC = ("distance_line", "distance_type", "feature_cost", "node_mass", "size_penalty_beta", "solver")
NATIVE_SOURCE = "src/eval/bace_native_baseline_gnn.py"


def stream_sha256(value):
    """Exactly stable_sha256's JSON bytes, without a full serialized copy."""
    digest = hashlib.sha256()
    for token in json.JSONEncoder(sort_keys=True, separators=(',', ':'), ensure_ascii=True).iterencode(value):
        digest.update(token.encode('utf-8'))
    return digest.hexdigest()


def _stream_atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            for token in json.JSONEncoder(sort_keys=True, separators=(',', ':'), ensure_ascii=True).iterencode(value):
                handle.write(token)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        # Preserve an interrupted fresh output for diagnosis, never old input.
        raise


def index_memory_bound(finite_rows, pair_rows, candidate_bytes):
    """Conservative next-stage bound for this concrete compact Python layout.

    Every finite source row is charged as a distinct graph key/map slot/value,
    even duplicates. Shared literal strings are charged per row too. There is
    no full JSON string after stream_sha256/_stream_atomic_json. Sort workspace
    and allocator slack are charged per row; a separate 1GiB fixed reserve
    covers imports, chemistry memoization, candidates and process overhead.
    """
    provenance = {'source_member': 'original_pair_matrix', 'line': 2**40}
    value = {'distance': 1., 'source_records': []}
    value['source_records'].append(provenance)  # Match the real append allocation.
    # A conservative map slot and sort tuple allowance, including pointer arrays.
    per_row = (sys.getsizeof('0' * 64) + sys.getsizeof(value) + sys.getsizeof(1.)
        + sys.getsizeof(value['source_records']) + sys.getsizeof(provenance)
        + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in provenance.items())
        + sum(sys.getsizeof(k) for k in value) + 128)
    accounted = finite_rows * per_row + pair_rows + 4 * candidate_bytes
    bound = (accounted * 5 + 3) // 4 + 1024**3
    return dict(layout='raw_graph_key -> distance + one compact source-line provenance',
        per_finite_row_conservative_bytes=per_row, finite_rows=finite_rows,
        pair_seen_bitmap_bytes=pair_rows, candidate_input_copy_bound_bytes=4*candidate_bytes,
        allocator_safety_factor=1.25, fixed_import_chemistry_process_reserve_bytes=1024**3,
        serialized_full_copy_count=0, estimated_peak_rss_bound_bytes=bound)


def _bound(path, sha256):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("NATIVE_RAW_SMALL_INPUT_CHANGED:" + str(path))
    return json.loads(data)


def _stat(path):
    value = Path(path).stat()
    return {k: getattr(value, k) for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}


def _reject_writer(path):
    """Bounded /proc FD identity check; source also must be sealed and unchanged."""
    if not Path('/proc').is_dir():
        return
    target = Path(path).resolve(strict=True)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    if fd.resolve(strict=True) != target:
                        continue
                    info = (proc / 'fdinfo' / fd.name).read_text()
                    flag = next(int(line.split()[1], 8) for line in info.splitlines() if line.startswith('flags:'))
                    if flag & os.O_ACCMODE in (os.O_WRONLY, os.O_RDWR):
                        raise ValueError('NATIVE_RAW_SOURCE_HAS_ACTIVE_WRITER:' + proc.name)
                except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
                    continue
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


def _native_call_proof(repo, commit):
    """Pin original function semantics separately from the current driver commit."""
    source = subprocess.check_output(['git', 'show', commit + ':' + NATIVE_SOURCE], cwd=repo).decode()
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_fullgraph_pair_rows')
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
             and n.func.value.id == 'provider']
    if len(calls) != 1 or calls[0].func.attr != 'distance' or len(calls[0].args) != 2 or calls[0].keywords:
        raise ValueError('NATIVE_RAW_ORIGINAL_PAIR_CALL_CHANGED')
    current = ast.parse((repo / NATIVE_SOURCE).read_text())
    now = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == function.name)
    if ast.dump(function) != ast.dump(now):
        raise ValueError('NATIVE_RAW_ORIGINAL_FULLGRAPH_FUNCTION_DRIFT')
    # Namespace occurs in pair/action cache keys and metadata, never the numeric
    # compute_node_wasserstein_distance call. Verify the actual call expression.
    kernel = ast.parse((repo / 'src/eval/node_wasserstein_distance.py').read_text())
    numerical = [n for n in ast.walk(kernel) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'compute_node_wasserstein_distance']
    if len(numerical) != 1 or any(k.arg not in ('feature_cost', 'node_mass', 'size_penalty_beta', 'emd2_fn')
                                  for k in numerical[0].keywords):
        raise ValueError('NATIVE_RAW_NAMESPACE_INDEPENDENCE_NOT_PROVEN')
    return dict(source_code_evidence_commit=commit, source_file=NATIVE_SOURCE,
        fullgraph_function_ast_sha256=hashlib.sha256(ast.dump(function).encode()).hexdigest(),
        source_distance_namespace=NATIVE_NAMESPACE,
        namespace_role='CACHE_KEY_AND_METADATA_ONLY_NOT_NUMERICAL_ARGUMENT',
        native_operation_is_deletion=False, source_execution_commit_claimed=False)


@lru_cache(maxsize=50000)
def _canonical(text):
    from src.eval.molclr_node_embeddings import canonicalize_smiles
    value = canonicalize_smiles(text)
    if value is None or '.' in value:
        raise ValueError('NATIVE_RAW_INVALID_OR_DISCONNECTED_GRAPH')
    return value


def _source_docs(binding, split, repo):
    # Load this small existing utility directly: importing the GNN package's
    # broad public __init__ is unnecessary for a no-inference migration worker.
    location = repo / 'src/ablations/gnn/reach_raw_distance_reuse.py'
    module_spec = importlib.util.spec_from_file_location('_bace_native_raw_kernel', location)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    kernel_identity_proof = module.kernel_identity_proof
    if binding.get('schema') != BINDING_SCHEMA or binding.get('method_id') not in OPERATIONS:
        raise ValueError('NATIVE_RAW_SOURCE_SCHEMA_REQUIRED')
    item = binding['splits'][split]
    doc = _bound(**item['merge_manifest'])
    kind, semantics = OPERATIONS[binding['method_id']]
    expected = dict(schema_version='bace_native_baseline_verification_merge_v1', status='PASS',
        run_complete=True, dataset='bace', method_id=binding['method_id'], action_kind=kind,
        action_semantics=semantics, source_label=1,
        stage='BASELINE_CALIBRATION_VERIFY' if split == 'calibration' else 'BASELINE_TEST_EVAL',
        test_loaded=split == 'test', selection_frozen_before_test=split == 'test')
    if any(doc.get(k) != v for k, v in expected.items()):
        raise ValueError('NATIVE_RAW_COMPLETED_SOURCE_CONTRACT_CHANGED')
    for key in ('pair_count', 'parent_count', 'selected_candidate_count'):
        if doc[key] != item[key]:
            raise ValueError('NATIVE_RAW_SOURCE_COUNT_CHANGED:' + key)
    contract = binding['raw_contract']
    if (contract['wnode']['solver'] != 'exact_emd2'
            or doc['molclr_checkpoint_hash'] != contract['molclr_checkpoint']['sha256']
            or doc['pair_matrix_hash'] != item['pair_file']['sha256']
            or doc['candidate_universe_hash'] != item['candidate_file']['sha256']):
        raise ValueError('NATIVE_RAW_ENCODER_OR_PAIR_BINDING_CHANGED')
    for rel, identity in contract['molclr_source'].items():
        suffix = rel.removeprefix(binding['bundle_molclr_source_root'] + '/')
        path = Path(binding['original_molclr_source_root']) / suffix
        if hashlib.sha256(path.read_bytes()).hexdigest() != identity['sha256']:
            raise ValueError('NATIVE_RAW_ENCODER_SOURCE_DRIFT:' + suffix)
    proof = kernel_identity_proof(repo, binding['source_kernel_commit'])
    call = _native_call_proof(repo, binding['original_fullgraph_call_commit'])
    shard_docs = [_bound(**{k: s[k] for k in ('path', 'sha256')}) for s in doc['inputs']['shard_manifests']]
    parent_ids, split_identity = [], None
    for shard in shard_docs:
        stats = shard['distance_provider_stats']
        if (shard.get('status') != 'PASS' or shard.get('run_complete') is not True
                or shard.get('test_loaded') != (split == 'test')
                or shard['action_kind'] != kind or shard['action_semantics'] != semantics
                or shard['feature_schema_sha256'] != contract['feature_schema']['sha256']
                or shard['molclr_checkpoint_hash'] != contract['molclr_checkpoint']['sha256']
                or stats['distance_namespace'] != NATIVE_NAMESPACE
                or any(stats[k] != contract['wnode'][k] for k in NUMERIC)):
            raise ValueError('NATIVE_RAW_SHARD_NUMERIC_OR_GRAPH_CONTRACT_CHANGED')
        parent_ids.extend(shard['parent_ids'])
        if split_identity is not None and shard['split_identity'] != split_identity:
            raise ValueError('NATIVE_RAW_SOURCE_SPLIT_CHANGED')
        split_identity = shard['split_identity']
    if len(parent_ids) != item['parent_count'] or len(set(parent_ids)) != len(parent_ids):
        raise ValueError('NATIVE_RAW_SOURCE_PARENT_PARTITION_INCOMPLETE')
    split_path = Path(split_identity['path'])
    if hashlib.sha256(split_path.read_bytes()).hexdigest() != split_identity['sha256']:
        raise ValueError('NATIVE_RAW_ORIGINAL_SPLIT_BYTES_CHANGED')
    parents = {p.parent_id: _canonical(p.smiles) for p in load_bace_parents(split_path, source_label=1)}
    if set(parents) != set(parent_ids):
        raise ValueError('NATIVE_RAW_SOURCE_PARENT_ID_CHANGED')
    candidate_path = Path(item['candidate_file']['path'])
    candidate_data = candidate_path.read_bytes()
    if hashlib.sha256(candidate_data).hexdigest() != item['candidate_file']['sha256']:
        raise ValueError('NATIVE_RAW_CANDIDATE_UNIVERSE_CHANGED')
    candidates = {}
    for line in candidate_data.splitlines():
        row = json.loads(line)
        if (row['candidate_id'] in candidates or row['action_kind'] != kind
                or row['action_semantics'] != semantics):
            raise ValueError('NATIVE_RAW_CANDIDATE_OPERATION_CHANGED')
        candidates[row['candidate_id']] = _canonical(row['canonical_smiles'])
    if len(candidates) != item['selected_candidate_count']:
        raise ValueError('NATIVE_RAW_CANDIDATE_COUNT_CHANGED')
    return doc, parents, candidates, proof, call


def build_native_index(binding, *, split, output, repo, test_freeze_path=None,
                       test_freeze_sha=None, validate_test_freeze=None):
    """Single read of an immutable old matrix. Test is gated before *any* source read."""
    if split not in ('calibration', 'test'):
        raise ValueError('NATIVE_RAW_SPLIT_UNSUPPORTED')
    freeze_sha = None
    if split == 'test':
        if not callable(validate_test_freeze) or not test_freeze_path or not test_freeze_sha:
            raise ValueError('NATIVE_RAW_TEST_BEFORE_NEW_SELECTOR_FREEZE')
        freeze = _bound(test_freeze_path, test_freeze_sha)
        validate_test_freeze(freeze)
        freeze_sha = test_freeze_sha
    output, repo = Path(output), Path(repo)
    binding_sha = stable_sha256(dict(source=binding, split=split, new_test_freeze_sha256=freeze_sha))
    if output.exists():
        old = read_json(output)
        if old.get('binding_sha256') != binding_sha or old.get('self_sha256') != stream_sha256(
                {k: v for k, v in old.items() if k != 'self_sha256'}):
            raise ValueError('NATIVE_RAW_SEALED_INDEX_CONFLICT')
        return old
    item = binding['splits'][split]
    source = Path(item['pair_file']['path']).resolve(strict=True)
    if source.parent == output.resolve().parent or source.parent in output.resolve().parents:
        raise ValueError('NATIVE_RAW_CANNOT_WRITE_SEALED_SOURCE')
    source_stat = _stat(source)
    if source_stat['st_size'] != item['pair_file']['size']:
        raise ValueError('NATIVE_RAW_SEALED_SOURCE_SIZE_CHANGED')
    _reject_writer(source)
    doc, parents, candidates, proof, call = _source_docs(binding, split, repo)
    parent_order, candidate_order = {p: i for i, p in enumerate(parents)}, {c: i for i, c in enumerate(candidates)}
    seen = bytearray(len(parents) * len(candidates))
    contract = binding['raw_contract']
    contract_sha = stable_sha256(contract)
    values, missing, count, finite = {}, 0, 0, 0
    digest = hashlib.sha256()
    started = time.monotonic()
    with source.open('rb') as handle:
        for lineno, line in enumerate(handle, 1):
            digest.update(line)
            if not line.strip():
                raise ValueError('NATIVE_RAW_UNEXPECTED_BLANK_ROW')
            row = json.loads(line)
            pid, cid = row.get('parent_id'), row.get('candidate_id')
            if pid not in parents or cid not in candidates:
                raise ValueError('NATIVE_RAW_ROW_OUTSIDE_BOUND_UNIVERSE')
            index = parent_order[pid] * len(candidates) + candidate_order[cid]
            if seen[index]:
                raise ValueError('NATIVE_RAW_DUPLICATE_PARENT_CANDIDATE')
            seen[index] = 1
            if (_canonical(row['parent_smiles']) != parents[pid]
                    or _canonical(row['canonical_smiles']) != candidates[cid]
                    or row.get('action_kind') != doc['action_kind']
                    or row.get('action_semantics') != doc['action_semantics']
                    or row.get('applicable') is not True
                    or row.get('method_id') != binding['method_id']
                    or row.get('oracle_checkpoint_hash') != doc['oracle_checkpoint_hash']):
                raise ValueError('NATIVE_RAW_ROW_GRAPH_OR_OPERATION_CHANGED:' + str(lineno))
            if any(row.get(k) is not None for k in ('match_index', 'match_atom_indices', 'residual_smiles')):
                raise ValueError('NATIVE_RAW_NOT_UNIQUE_FULLGRAPH_COST')
            count += 1
            value = row.get('wnode_distance')
            if value is None:
                missing += 1
                continue
            if (type(value) not in (float, int) or not math.isfinite(value) or value < 0
                    or row.get('failure_reason') not in (None, '')):
                raise ValueError('NATIVE_RAW_NONFINITE_OR_FAILED_DISTANCE:' + str(lineno))
            key = stable_sha256(dict(parent=parents[pid], residual=candidates[cid], raw_contract_sha256=contract_sha))
            provenance = dict(source_member='original_pair_matrix', line=lineno)
            if key in values and values[key]['distance'] != value:
                raise ValueError('NATIVE_RAW_DIRECTIONAL_GRAPH_COST_CONFLICT')
            if key not in values:
                values[key] = dict(distance=value, source_records=[])
            values[key]['source_records'].append(provenance)
            finite += 1
            if count % 50000 == 0:
                atomic_json(output.parent / (output.stem + '.progress.json'), dict(state='RUNNING',
                    split=split, processed_rows=count, finite_raw_rows=finite, no_raw_distance_rows=missing,
                    pid=os.getpid(), elapsed_seconds=time.monotonic()-started))
    if digest.hexdigest() != item['pair_file']['sha256'] or _stat(source) != source_stat:
        raise ValueError('NATIVE_RAW_SOURCE_CHANGED_DURING_SINGLE_STREAM')
    if count != item['pair_count'] or not all(seen) or finite != item['finite_count']:
        raise ValueError('NATIVE_RAW_SOURCE_PARTITION_OR_FINITE_COUNT_INCOMPLETE')
    result = dict(schema=SCHEMA, state='RAW_COST_ADOPTION_INDEX_SEALED_NOT_SCIENCE_PASS',
        binding_sha256=binding_sha, source_spec=binding, split=split,
        source_parent_units=len(parents), source_pair_rows=count, source_finite_match_records=finite,
        source_missing_raw_distance_rows=missing, raw_contract=contract, raw_contract_sha256=contract_sha,
        kernel_identity=proof, native_call_proof=call, new_test_freeze_sha256=freeze_sha,
        source_members={'original_pair_matrix': dict(item['pair_file'], source_stat=source_stat,
            merge_manifest=item['merge_manifest'])}, graph_costs=values, raw_cost_count=len(values),
        old_cache_keys_modified=False, source_flip_masks_reused=False,
        source_selected_match_minima_reused=False, source_selection_caps_reused=False,
        model_inference_performed=False, ot_recomputed=0,
        source_file_reads=1, elapsed_seconds=time.monotonic()-started)
    result['index_memory_bound'] = index_memory_bound(finite, count, Path(item['candidate_file']['path']).stat().st_size)
    result['self_sha256'] = stream_sha256(result)
    _stream_atomic_json(output, result)
    atomic_json(output.parent / (output.stem + '.progress.json'), dict(state='COMPLETED',
        split=split, processed_rows=count, finite_raw_rows=finite, no_raw_distance_rows=missing,
        raw_cost_count=len(values), index=str(output), self_sha256=result['self_sha256']))
    return result
