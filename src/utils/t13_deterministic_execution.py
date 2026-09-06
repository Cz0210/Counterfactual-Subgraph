"""Narrow adoption of the existing T13 deterministic diagnostic, not a new runner.

Large immutable checkpoints are bound by their existing reload receipts. Only
small JSON/code bindings are rechecked; this never runs a canary or mines data.
"""
from __future__ import annotations

from datetime import datetime, timezone
import copy
import json
import os
from pathlib import Path
import subprocess
import uuid

from src.utils.t8_hpc_t13_successor_v1 import (
    atomic_json_no_replace, canonical_sha256, require_self_hash,
)
from src.utils.t13_lazy_recovery_guard import file_sha

SCHEMA = "t13_matched_deterministic_execution_contract_v1"
BACKEND = dict(torch_version="2.7.1+cu118", cuda_version="11.8",
    cublas_workspace_config=":4096:8", deterministic_algorithms=True,
    deterministic_warn_only=False, cudnn_deterministic=True,
    cudnn_benchmark=False, matmul_allow_tf32=False, cudnn_allow_tf32=True)
# These determine the model, data, RNG, optimizer and checkpoint semantics.
# The CLI/owner may change, but these must match the passed scientific checkout.
SCIENCE_FILES = (
    "src/baselines/t13_indexed_augmentation.py",
    "src/baselines/t13_indexed_canary.py",
    "src/baselines/t13_component_diagnostics.py",
    "src/baselines/globalgce_mutagenicity_adapter.py",
    "src/baselines/globalgce_resumable.py",
    "src/baselines/tastemolnet_globalgce_full.py",
    "src/baselines/tastemolnet_globalgce_smoke.py",
    "src/baselines/globalgce_bace_native_rules.py",
    "src/oracles/gnn_oracle.py", "configs/hpc.yaml",
)


def read(path):
    return json.loads(Path(path).read_text())


def bound_json(path, expected=None):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("T13_EVIDENCE_FILE_REQUIRED:" + str(path))
    sha = file_sha(path)
    if expected is not None and sha != expected:
        raise ValueError("T13_EVIDENCE_HASH_CHANGED:" + str(path))
    return read(path), dict(path=str(path), sha256=sha, bytes=path.stat().st_size)


def inspect_evidence(root):
    root = Path(root)
    canary, main_ref = bound_json(root / "canary.json")
    if (canary.get("state") != "T13_MATCHED_DETERMINISTIC_DIAGNOSTIC_PASS"
            or canary.get("targets_order") != [0, 2] or canary.get("seed") != 7
            or canary.get("configured_epochs") != 100
            or canary.get("diagnostic_profile") != "deterministic"):
        raise ValueError("T13_MATCHED_DETERMINISTIC_EVIDENCE_REQUIRED")
    for name in ("test_loaded", "calibration_loaded", "mining_recomputed",
                 "full_successor_started", "full_trajectory_parity_claimed"):
        if canary.get(name) is not False:
            raise ValueError("T13_EVIDENCE_SCOPE_CHANGED:" + name)
    for name in ("train_only", "independent_reload_pass", "index_contract_pass",
                 "mask_rng_batch_parity", "training_step_parity", "reload_parity"):
        if canary.get(name) is not True:
            raise ValueError("T13_EVIDENCE_GATE_MISSING:" + name)
    refs = {"canary": main_ref}
    targets = {}
    for target in (0, 2):
        report = canary["targets"][str(target)]
        branch = root / f"target_{target}"
        diag, ref = bound_json(branch / "training_canary/component_diagnostics.json",
                              report["component_diagnostics_sha256"])
        refs[f"target_{target}_diagnostics"] = ref
        if (report.get("numeric_contract") != BACKEND or diag.get("numeric_contract") != BACKEND
                or report.get("optimizer_updates_per_arm") != 2
                or report.get("eager_repetitions") != 3
                or diag.get("state") != "T13_COMPONENT_DIAGNOSTIC_PASS"
                or diag.get("eager_repetitions_completed") != 3
                or diag.get("eager_self_repeatable") is not True
                or diag.get("tolerance_used") is not False
                or diag.get("warn_only_accepted") is not False
                or diag.get("first_difference") is not None
                or diag.get("execution_error") is not None):
            raise ValueError("T13_COMPONENT_OR_BACKEND_GATE_MISSING")
        initials = diag.get("initial_state_bindings", [])
        if ([r["arm"] for r in initials] != ["eager_0", "eager_1", "eager_2", "lazy"]
                or len({r["state_sha256"] for r in initials}) != 1):
            raise ValueError("T13_INITIAL_STATE_BINDING_CHANGED")
        comparisons = diag.get("comparisons", [])
        if ([(r["kind"], r["epoch"]) for r in comparisons] != [
                ("EAGER_SELF_CONTROL", 0), ("EAGER_SELF_CONTROL", 0),
                ("EAGER_LAZY", 0), ("EAGER_LAZY", 1)]):
            raise ValueError("T13_COMPONENT_COVERAGE_INCOMPLETE")
        expected_components = {"before", "batches", "rules", "losses", "gradients",
                               "rng_after_rules", "rng_after_backward", "after_update"}
        for row in comparisons:
            if row.get("exact") is not True or set(row["components"]) != expected_components:
                raise ValueError("T13_COMPONENT_PARITY_INCOMPLETE")
            for component in row["components"].values():
                if component.get("exact") is not True or component.get("tolerance_used") is not False:
                    raise ValueError("T13_COMPONENT_PARITY_FAILED")
        for name in ("eager_lazy_batch_exact", "forward_loss_exact",
                     "model_optimizer_scheduler_rng_exact", "checkpoint_reload_exact"):
            if report.get(name) is not True:
                raise ValueError("T13_TARGET_GATE_MISSING:" + name)
        index = report["dataset_identity"]
        if (not index["index_sha256"] or not index["masks_sha256"] or index["sample_count"] <= 0
                or index["materialization_rng_unchanged"] is not True
                or index["all_masks_reconstructed_exactly"] is not True
                or index["sampler"]["num_workers"] != 0):
            raise ValueError("T13_INDEX_OR_RNG_GATE_MISSING")
        local, local_ref = bound_json(branch / "training_canary/checkpoint_reload.json")
        independent, independent_ref = bound_json(branch / "independent_reload/verification.json")
        if local != independent or local.get("state") != "PASS" or local.get("checkpoint_sha256") != report["checkpoint_sha256"]:
            raise ValueError("T13_INDEPENDENT_RELOAD_BINDING_CHANGED")
        refs[f"target_{target}_reload"] = local_ref
        refs[f"target_{target}_independent_reload"] = independent_ref
        targets[str(target)] = dict(index_identity={k:v for k,v in index.items() if k != "graph_idxs"},
            initial_state_bindings=initials, checkpoint_sha256=report["checkpoint_sha256"],
            component_evidence_sha256=report["component_evidence_sha256"],
            optimizer_updates_per_arm=2, eager_repetitions=3,
            diagnostic_checkpoint_promoted=False, numeric_contract=report["numeric_contract"])
    native, native_ref = bound_json(root.parent / "canary.json")
    if (native.get("state") != "T13_COMPONENT_DIAGNOSTIC_FAILED"
            or native.get("failure", {}).get("eager_self_repeatable") is not False):
        raise ValueError("T13_NATIVE_FAILURE_EVIDENCE_REQUIRED")
    refs["native_failure"] = native_ref
    memory, memory_ref = bound_json(root / "memory_samples.json", canary["memory_samples_sha256"])
    rows = memory["samples"]
    for target in (0, 2):
        boundary, ref = bound_json(root / f"target_{target}/training_canary/memory_boundaries.json")
        refs[f"target_{target}_memory_boundaries"] = ref
        rows = rows + boundary["samples"]
    if not rows:
        raise ValueError("T13_MEMORY_EVIDENCE_MISSING")
    refs["memory_samples"] = memory_ref
    failcnts = [int(r["memory.failcnt"]) for r in rows if "memory.failcnt" in r]
    peak = max(int(r.get("VmHWM_bytes", r.get("VmRSS_bytes", 0))) for r in rows)
    headroom = min(int(r["memory.limit_in_bytes"])-int(r["memory.usage_in_bytes"])
                   for r in rows if "memory.limit_in_bytes" in r and "memory.usage_in_bytes" in r)
    if not failcnts or min(failcnts) != max(failcnts) or peak <= 0:
        raise ValueError("T13_MEMORY_EVENT_EVIDENCE_FAILED")
    return dict(evidence=refs, targets=targets, backend=BACKEND,
                process_peak_bytes=peak, min_headroom_bytes=headroom,
                canary_failcnt_increment=0)


def validate_contract(path, expected_sha, spec):
    contract, _ = bound_json(path, expected_sha)
    require_self_hash(contract, "self_sha256", "T13 deterministic contract")
    if (contract.get("schema_version") != SCHEMA
            or contract.get("entry_mode") != "MATCHED_DETERMINISTIC_CONTRACT"
            or contract.get("authorized_by") != "user_project_owner"
            or contract.get("formal_task_spec_sha256") != spec["task_spec_sha256"]
            or contract.get("backend") != BACKEND
            or contract.get("formal_full_started") is not False
            or contract.get("full_100_epoch_trajectory_proven") is not False
            or contract.get("diagnostic_checkpoint_promoted") is not False):
        raise ValueError("T13_DETERMINISTIC_CONTRACT_BINDING_FAILED")
    auth, _ = bound_json(contract['new_authorization_receipt'],contract['new_authorization_sha256'])
    require_self_hash(auth,'self_sha256','T13 deterministic adoption authorization')
    if (auth.get('authorized_by')!='user_project_owner' or auth.get('max_formal_starts')!=1
            or auth.get('adopt_matched_deterministic_execution_contract') is not True
            or auth.get('single_formal_start_after_contract_and_resource_pass') is not True
            or auth.get('original_authorization_path')!=contract['original_authorization_path']
            or auth.get('formal_quota_ledger')!=contract['formal_quota_ledger']):
        raise ValueError('T13_DETERMINISTIC_NARROW_AUTHORIZATION_MISSING')
    if set(contract['unchanged_science_files']) != set(SCIENCE_FILES) or set(contract['targets'])!={'0','2'}:
        raise ValueError('T13_INCOMPLETE_SCIENCE_BINDING_INVENTORY')
    for row in contract["evidence"].values():
        bound_json(row["path"], row["sha256"])
    # No repeated large checkpoint/model hashing: immutable science bindings are
    # checked against the original source inventory and existing input receipts.
    for name, sha in contract["unchanged_science_files"].items():
        if file_sha(Path(spec["repo_root"]) / name) != sha:
            raise ValueError("T13_SCIENTIFIC_SOURCE_CHANGED:" + name)
    return contract


def seal_formal_spec(*, source_spec_root, evidence_root, original_authorization,
                     repo_root, fresh_root, fresh_science_root, fresh_cache_root):
    """Seal this one existing owner interface. Does not acquire a lease/start."""
    from src.utils.t8_hpc_t13_successor_v1 import validate_spec_set, write_t13_release
    source_spec_root, repo_root, fresh_root = map(Path, (source_spec_root, repo_root, fresh_root))
    auth = Path(original_authorization)
    original = read(auth)
    require_self_hash(original, 'self_sha256', 'original T13 authorization')
    if (original.get('max_full_starts') != 1 or (auth.parent/'full_start.json').exists()):
        raise ValueError('T13_ORIGINAL_ONE_FORMAL_START_NOT_AVAILABLE')
    old = validate_spec_set(source_spec_root, check_files=True)
    source_repo = Path(old['t13']['repo_root'])
    commit = subprocess.check_output(['git','-C',str(repo_root),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(repo_root),'status','--porcelain'],text=True).strip():
        raise ValueError('T13_CLEAN_FORMAL_EXECUTION_WORKTREE_REQUIRED')
    code = {}
    for name in SCIENCE_FILES:
        before = file_sha(source_repo/name)
        if file_sha(repo_root/name) != before:
            raise ValueError('T13_CHANGED_SCIENCE_REQUIRES_AFFECTED_CANARY:'+name)
        code[name] = before
    evidence = inspect_evidence(evidence_root)
    old_owner = Path(evidence_root).parents[1]
    heartbeat = read(old_owner/'heartbeat.json')
    if read(old_owner/'terminal.json').get('state') != 'FAILED':
        raise ValueError('T13_PREVIOUS_CANARY_OWNER_NOT_TERMINAL')
    for pid in (heartbeat['owner_pid'], heartbeat.get('science_pid',0)):
        if pid and Path('/proc',str(pid)).exists():
            raise ValueError('T13_PREVIOUS_OWNER_IDENTITY_REVIEW_REQUIRED')
    for path in (fresh_root,Path(fresh_science_root),Path(fresh_cache_root)):
        if not path.is_absolute() or path.exists() or path.is_symlink():
            raise ValueError('T13_FRESH_ABSOLUTE_OUTPUT_REQUIRED:'+str(path))
    fresh_root.mkdir(parents=True)
    spec_root=fresh_root/'spec-bundle'; spec_root.mkdir()
    attempt=str(uuid.uuid4())
    specs={role:copy.deepcopy(old[role]) for role in ('import','t13','publisher')}
    for spec in specs.values():
        spec['execution_commit']=commit
        if 'repo_root' in spec: spec['repo_root']=str(repo_root)
    t13=specs['t13']; t13.update(attempt_id=attempt,task_id='t13-from-hpc-'+attempt,
        output_root=str(fresh_science_root),owner_entrypoint=str(repo_root/'scripts/autodl/run_t13_from_hpc_owner_v1.py'))
    cache=Path(fresh_cache_root)
    t13['input_paths']['wnode_cache_db']=str(cache/'wnode.sqlite')
    t13['input_paths']['node_embedding_cache_dir']=str(cache/'node-embeddings')
    replacements={str(source_spec_root):str(spec_root),str(source_repo):str(repo_root),
        old['t13']['output_root']:str(fresh_science_root),
        old['t13']['input_paths']['wnode_cache_db']:t13['input_paths']['wnode_cache_db'],
        old['t13']['input_paths']['node_embedding_cache_dir']:t13['input_paths']['node_embedding_cache_dir']}
    t13['command']=[replacements.get(x,x.replace(str(source_repo)+'/',str(repo_root)+'/')) for x in t13['command']]
    specs['publisher']['expected_terminal_root']=str(fresh_science_root)
    for role,spec in specs.items():
        spec.pop('task_spec_sha256',None); spec['task_spec_sha256']=canonical_sha256(spec)
        atomic_json_no_replace(spec_root/old['manifest']['specs'][role],spec)
    manifest=copy.deepcopy(old['manifest']); manifest.update(spec_root=str(spec_root),execution_commit=commit,
        created_at=datetime.now(timezone.utc).isoformat(),
        task_spec_sha256s={r:s['task_spec_sha256'] for r,s in specs.items()})
    manifest.pop('spec_set_sha256',None); manifest['spec_set_sha256']=canonical_sha256(manifest)
    atomic_json_no_replace(spec_root/'spec_set_manifest.json',manifest)
    validate_spec_set(spec_root,check_files=True)
    write_t13_release(spec_root=spec_root,import_root=t13['required_import_root'],output=fresh_root/'release.json')
    authorization=dict(schema_version='t13_deterministic_adoption_user_authorization_v1',
        authorized_by='user_project_owner',created_at=datetime.now(timezone.utc).isoformat(),
        adopt_matched_deterministic_execution_contract=True,repair_existing_owner_formal_interface=True,
        single_formal_start_after_contract_and_resource_pass=True,max_formal_starts=1,
        rerun_passed_canary=False,diagnostic_checkpoint_promotion=False,
        original_authorization_path=str(auth),original_authorization_sha256=file_sha(auth),
        formal_quota_ledger=str(auth.parent/'full_start.json'),
        main_matrix_write=False,other_tasks_may_be_signaled=False)
    authorization['self_sha256']=canonical_sha256(authorization)
    atomic_json_no_replace(fresh_root/'authorization_receipt.json',authorization)
    contract=dict(schema_version=SCHEMA,entry_mode='MATCHED_DETERMINISTIC_CONTRACT',
        authorized_by='user_project_owner',created_at=datetime.now(timezone.utc).isoformat(),
        original_authorization_path=str(auth),original_authorization_sha256=file_sha(auth),
        formal_quota_ledger=str(auth.parent/'full_start.json'),
        new_authorization_receipt=str(fresh_root/'authorization_receipt.json'),
        new_authorization_sha256=file_sha(fresh_root/'authorization_receipt.json'),
        formal_task_spec_path=str(spec_root/'t13_from_hpc_task_spec.json'),
        formal_task_spec_sha256=t13['task_spec_sha256'],
        scientific_evidence_commit=old['t13']['execution_commit'],formal_driver_commit=commit,
        unchanged_science_files=code,source_spec_root=str(source_spec_root),
        input_paths=t13['input_paths'],science_contract=t13['science_contract'],
        native_control_state='FAILED_NONDETERMINISTIC_CONTROL',
        matched_deterministic_canary_state='PASS',evidence_scope='SHORT_TRAINING_AND_RELOAD',
        full_100_epoch_trajectory_proven=False,formal_full_started=False,
        diagnostic_checkpoint_promoted=False,already_passed_canary_rerun=False,
        inherited_thread_policy='UNCHANGED_LAUNCHER_ENVIRONMENT_AND_PINNED_TORCH_DEFAULTS',
        historical_thread_count_recorded=False,
        thread_policy_note='Original diagnostic did not record numeric thread counts; no new thread override is introduced. Actual formal counts are recorded, not falsely claimed as historical measurements.',
        **evidence)
    contract['self_sha256']=canonical_sha256(contract)
    contract_path=fresh_root/'deterministic_execution_contract.json'
    atomic_json_no_replace(contract_path,contract)
    validate_contract(contract_path,file_sha(contract_path),t13)
    dispatch=dict(state='SEALED_WAITING_FORMAL_OWNER',spec_root=str(spec_root),
        release=str(fresh_root/'release.json'),owner_root=str(fresh_root/'owner'),
        science_root=str(fresh_science_root),execution_commit=commit,
        original_authorization=str(auth),original_authorization_sha256=file_sha(auth),
        deterministic_execution_contract=str(contract_path),deterministic_execution_sha256=file_sha(contract_path),
        formal_start_consumed=False,canary_rerun=False)
    atomic_json_no_replace(fresh_root/'dispatch.json',dispatch)
    return dispatch


def apply_backend(torch, expected):
    """Called in the actual new worker, before creating a CUDA context."""
    if expected != BACKEND or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("T13_BACKEND_ENVIRONMENT_NOT_ESTABLISHED_BEFORE_CHILD")
    if torch.cuda.is_initialized():
        raise ValueError("T13_BACKEND_MUST_PRECEDE_CUDA_INITIALIZATION")
    if torch.__version__ != expected["torch_version"] or torch.version.cuda != expected["cuda_version"]:
        raise ValueError("T13_BACKEND_VERSION_CHANGED")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    observed = dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    if observed != expected:
        raise ValueError("T13_ACTUAL_WORKER_BACKEND_READBACK_FAILED")
    return observed


def activate_from_environment(repo_root):
    path = os.environ.get("T13_DETERMINISTIC_EXECUTION_CONTRACT")
    if not path:
        return None  # Existing/native entry is unchanged.
    contract = read(path)
    spec = read(contract["formal_task_spec_path"])
    contract = validate_contract(path, os.environ["T13_DETERMINISTIC_EXECUTION_SHA256"], spec)
    if str(Path(repo_root).resolve()) != spec["repo_root"]:
        raise ValueError("T13_ACTUAL_WORKER_CHECKOUT_CHANGED")
    import torch
    observed = apply_backend(torch, contract["backend"])
    receipt = dict(state="T13_RUNTIME_BACKEND_PASS", pid=os.getpid(), parent_pid=os.getppid(),
        cwd=str(Path.cwd()), execution_commit=spec["execution_commit"],
        scientific_evidence_commit=contract["scientific_evidence_commit"],
        contract_sha256=contract["self_sha256"], observed_backend=observed,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_initialized_at_readback=torch.cuda.is_initialized(),
        torch_num_threads=torch.get_num_threads(), torch_num_interop_threads=torch.get_num_interop_threads(),
        thread_environment={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")},
        created_at=datetime.now(timezone.utc).isoformat(), science_completed=False)
    if receipt["cuda_visible_devices"] != spec["gpu_uuid"]:
        raise ValueError("T13_ACTUAL_WORKER_GPU_UUID_CHANGED")
    atomic_json_no_replace(Path(os.environ["T13_RUNTIME_BACKEND_RECEIPT"]), receipt)
    return receipt
