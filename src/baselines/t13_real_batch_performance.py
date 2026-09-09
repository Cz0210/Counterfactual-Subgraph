"""Bounded, future-only performance comparison on actual indexed Taste batches.

This is not a training successor and cannot promote its diagnostic checkpoint.
The caller supplies the existing owner's lease and a fresh output directory.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import types

GIB = 1024 ** 3
REFERENCE_COMMIT = "c0eb892dd13ef05a5891c4acf1c5f4fef3966f67"
SCOPE = "T13_REAL_TRAIN_TWO_BATCH_VALIDATION_ONE_BATCH_PERFORMANCE"


class PerformanceComplete(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__(report["state"])


def validate_plan(plan):
    exact = dict(scope=SCOPE, train_batches=2, validation_batches=1,
                 max_wall_seconds=1800, batch_size=500, num_workers=0,
                 seed=7, epochs=100, source_label=1, synthetic=False,
                 formal_start=False, active_handover=False, remining=False,
                 test_loaded=False, calibration_loaded=False,
                 tmpfs_reserved_bytes=0, process_peak_budget_bytes=64 * GIB)
    for key, value in exact.items():
        if type(plan.get(key)) is not type(value) or plan[key] != value:
            raise ValueError("T13_PERFORMANCE_PLAN_CONTRACT:" + key)
    if plan.get("target_label") not in (0, 2):
        raise ValueError("T13_PERFORMANCE_TARGET")
    for key in ("output_root", "source_checkpoint", "source_index_manifest",
                "source_cohort_manifest", "train_csv", "gnn_checkpoint",
                "official_root", "gspan_adoption_proof", "reference_source"):
        if not Path(plan[key]).is_absolute():
            raise ValueError("T13_PERFORMANCE_ABSOLUTE_PATH:" + key)
    output = Path(plan["output_root"]).resolve()
    if not output.is_relative_to(Path("/autodl-fs/data/counterfactual-subgraph-runtime")):
        raise ValueError("T13_PERFORMANCE_PERSISTENT_OUTPUT_REQUIRED")
    for key in ("source_checkpoint", "source_index_manifest", "source_cohort_manifest"):
        if Path(plan[key]).resolve().is_relative_to(output):
            raise ValueError("T13_PERFORMANCE_OUTPUT_OVERLAPS_SOURCE")
    return plan


def snapshot_checkpoint(source, destination):
    """One held inode snapshot; an atomic publisher may advance the source path."""
    digest = hashlib.sha256()
    with Path(source).open("rb") as incoming, Path(destination).open("xb") as outgoing:
        before = os.fstat(incoming.fileno())
        while block := incoming.read(1024 * 1024):
            digest.update(block)
            outgoing.write(block)
        after = os.fstat(incoming.fileno())
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("T13_CHECKPOINT_HELD_INODE_CHANGED")
    return dict(source=str(source), snapshot=str(destination), bytes=before.st_size,
                sha256=digest.hexdigest(), source_inode=before.st_ino)


def reference_module(path, expected_sha):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha:
        raise ValueError("T13_IMMUTABLE_REFERENCE_BRIDGE_CHANGED")
    name = "_t13_real_batch_reference_bridge"
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


class ObservedOracle:
    """Read-only recording of actual oracle outputs without new oracle calls."""
    def __init__(self, wrapped, copier):
        self.wrapped, self.copier = wrapped, copier
        self.records = []

    def __getattr__(self, key):
        return getattr(self.wrapped, key)

    def __call__(self, *args, **kwargs):
        result = self.wrapped(*args, **kwargs)
        self.records.append(self.copier({key: result[key] for key in
                            ("y_pred", "logits", "bridge_audit") if key in result}))
        return result


def require_identity(checkpoint, identity, index_manifest, resume_identity):
    if checkpoint.get("augmented_dataset_identity") != identity or identity != index_manifest:
        raise ValueError("T13_REAL_INDEX_OR_MASK_IDENTITY_CHANGED")
    if checkpoint.get("resume_identity") != resume_identity:
        raise ValueError("T13_REAL_MODEL_INPUT_IDENTITY_CHANGED")
    if checkpoint.get("sampler_state") != dict(identity["sampler"], next_epoch=checkpoint["next_epoch"]):
        raise ValueError("T13_REAL_SAMPLER_CURSOR_CHANGED")
    if identity["sampler"]["batch_size"] != 500 or identity["sampler"]["num_workers"] != 0:
        raise ValueError("T13_REAL_BATCH_SEMANTICS_CHANGED")


def make_interceptor(plan, checkpoint, reference, output, sample):
    """Replace only the fresh diagnostic adapter's training call, never live code."""
    import torch
    from src.baselines import globalgce_frozen_gine_bridge as optimized
    from src.baselines.globalgce_resumable import (
        _get_fs_expanded_data_from_adoption, _atomic_torch_save, _restore_numpy_rng_state)
    from src.baselines.t13_component_diagnostics import cpu_copy, exact_difference
    from src.baselines.t13_indexed_canary import rng_state, restore_rng, state_digest
    from src.eval.bace_frozen_gnn_contracts import atomic_json

    def intercept(**kwargs):
        model = kwargs["model"]
        if kwargs["gspan_adoption_proof"] is None:
            raise ValueError("T13_REAL_CANARY_MINING_FORBIDDEN")
        descriptor = plan.get('committed_compact_payload')
        if not descriptor:
            raise ValueError('COMPACT_MASK_INDEX_PAYLOAD_MISSING_NOT_RESEEDABLE')
        from src.baselines.t13_bounded_payload import install_committed_expansion
        install_committed_expansion(model.fsg, descriptor=descriptor,
                                   expected_identity=checkpoint['augmented_dataset_identity'])
        sample("before_committed_compact_index_load")
        (fss, train, validation, _unused_test_loader), adoption = _get_fs_expanded_data_from_adoption(
            model=model, train_loader=kwargs["train_loader"], proof_path=kwargs["gspan_adoption_proof"])
        identity = train.dataset.dataset.identity
        require_identity(checkpoint, identity, json.loads(Path(plan["source_index_manifest"]).read_text()),
                         kwargs["resume_identity"])
        if train.batch_size != 500 or validation.batch_size != 500 or train.num_workers or validation.num_workers:
            raise ValueError("T13_REAL_LOADER_CONTRACT_CHANGED")
        sample("compact_index_verified")
        initial = cpu_copy(checkpoint)
        initial_rng = dict(python=initial["python_rng_state"], numpy=None,
                           torch=initial["torch_rng_state"].cpu(),
                           cuda=[x.cpu() for x in initial["cuda_rng_state"]])
        _restore_numpy_rng_state(kwargs["numpy_module"], initial["numpy_rng_state"])
        initial_rng["numpy"] = kwargs["numpy_module"].random.get_state()
        restore_rng(initial_rng)
        train_iter = iter(train)
        batches = [next(train_iter), next(train_iter)]
        validation_batch = next(iter(validation))
        batch_bindings = dict(train=[dict(indices=cpu_copy(b["index"]).tolist(), sha256=state_digest(b)) for b in batches],
                              validation=dict(indices=cpu_copy(validation_batch["index"]).tolist(),
                                              sha256=state_digest(validation_batch)))
        # Materializing the same selected real batches is outside timed model work.
        # All arms restore the identical post-loader RNG snapshot afterwards.
        arm_rng = cpu_copy(rng_state())
        atomic_json(output / "actual_batch_binding.json", dict(batch_bindings, dataset_identity=identity,
                    adopted_patterns=adoption, unique_train_batches=2, unique_validation_batches=1,
                    checkpoint_epoch=int(initial["next_epoch"]) - 1, diagnostic_updates_not_full_epochs=True))

        def optimizer_for(target, saved=None):
            optimizer = torch.optim.Adam(target.parameters(), lr=kwargs["learning_rate"], weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
            optimizer.load_state_dict((saved or initial)["optimizer_state"])
            scheduler.load_state_dict((saved or initial)["scheduler_state"])
            return optimizer, scheduler

        def snapshot(target, optimizer, scheduler):
            return cpu_copy(dict(model_state=target.state_dict(), optimizer_state=optimizer.state_dict(),
                                 scheduler_state=scheduler.state_dict(), rng=rng_state()))

        records, timings = {}, {}
        original_oracle = model.gt_gnn
        original_bridge_class = original_oracle.bridge.__class__
        oracle_weights = state_digest(original_oracle.bridge.model.state_dict())
        if any(p.requires_grad for p in original_oracle.parameters()):
            raise ValueError("T13_REAL_ORACLE_MUST_REMAIN_FROZEN")
        observed_oracle = ObservedOracle(original_oracle, cpu_copy)
        model.gt_gnn = observed_oracle
        try:
            for arm in ("reference", "optimized", "optimized_reload"):
                model.gt_gnn.bridge.__class__ = (reference.FrozenGINEDifferentiableBridge
                    if arm == "reference" else optimized.FrozenGINEDifferentiableBridge)
                model.load_state_dict(initial["model_state"])
                model.zero_grad(set_to_none=True)
                optimizer, scheduler = optimizer_for(model)
                restore_rng(arm_rng)
                current = model
                rows, times = [], []
                for step, source_batch in enumerate(batches):
                    # Native recourse may mutate its collated input. Every arm
                    # receives an independent real-batch copy with identical layout.
                    batch = copy.deepcopy(source_batch)
                    current.train(); current.gt_gnn.eval()
                    observed_oracle.records.clear()
                    torch.cuda.synchronize(); before = time.perf_counter()
                    rules = current.get_rules(fss)
                    losses = current.run_one_batch(rules, batch)
                    chosen = losses[3] if int(initial["next_epoch"]) + step < 35 else losses[0]
                    if not all(torch.isfinite(v).all() for v in losses):
                        raise ValueError("T13_REAL_CANARY_NONFINITE_LOSS")
                    chosen.backward()
                    gradients = {name: cpu_copy(p.grad) for name, p in current.named_parameters()}
                    optimizer.step(); optimizer.zero_grad(); scheduler.step()
                    torch.cuda.synchronize(); times.append(time.perf_counter() - before)
                    rows.append(dict(rules=cpu_copy(rules), losses=cpu_copy(losses), gradients=gradients,
                                     oracle_outputs=cpu_copy(observed_oracle.records),
                                     after=snapshot(current, optimizer, scheduler)))
                    sample(arm + ":update" + str(step + 1))
                    if arm == "optimized_reload" and step == 0:
                        path = output / "diagnostic_checkpoint.pt"
                        _atomic_torch_save(torch, rows[-1]["after"], path)
                        saved = torch.load(path, map_location="cpu", weights_only=False)
                        if not exact_difference(rows[-1]["after"], saved)["exact"]:
                            raise ValueError("T13_REAL_SAVED_CONTAINER_CHANGED")
                        # A fresh model and new optimizer objects, not a shallow state_dict alias.
                        # Large read-only source data and the frozen oracle are not
                        # duplicated: only trainable generator state gets new objects.
                        current = copy.deepcopy(model, memo={id(model.fsg): model.fsg,
                                                            id(model.gt_gnn): model.gt_gnn})
                        current.load_state_dict(saved["model_state"])
                        optimizer, scheduler = optimizer_for(current, saved)
                        restore_rng(saved["rng"])
                        atomic_json(output / "reload_binding.json", dict(
                            state_sha256=state_digest(saved), fresh_model_object=True,
                            fresh_optimizer_object=True, checkpoint_promotable=False))
                current.eval(); current.gt_gnn.eval()
                observed_oracle.records.clear()
                torch.cuda.synchronize(); before = time.perf_counter()
                with torch.no_grad():
                    # Same pinned official function and all original validation fields.
                    evaluated = kwargs["test_globalgce"]([copy.deepcopy(validation_batch)], current,
                                                        kwargs["pred_model"], rules)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - before)
                rows.append(dict(validation=cpu_copy(evaluated), rng=cpu_copy(rng_state()),
                                 oracle_outputs=cpu_copy(observed_oracle.records)))
                records[arm] = rows
                timings[arm] = times
                sample(arm + ":validation_complete")
                _atomic_torch_save(torch, records, output / "comparison_records.pt")
                if current is not model:
                    del current
        finally:
            original_oracle.bridge.__class__ = original_bridge_class
            model.gt_gnn = original_oracle
        if state_digest(original_oracle.bridge.model.state_dict()) != oracle_weights:
            raise ValueError("T13_REAL_FROZEN_ORACLE_WEIGHTS_CHANGED")
        forward = exact_difference(records["reference"], records["optimized"])
        reload = exact_difference(records["optimized"], records["optimized_reload"])
        report = dict(state="T13_REAL_BATCH_COMPONENTS_PASS_PENDING_INDEPENDENT_RELOAD" if forward["exact"] and reload["exact"]
                      else "T13_REAL_BATCH_PERFORMANCE_FAILED", scope=SCOPE,
                      reference_vs_optimized=forward, fresh_model_reload=reload,
                      timings_seconds=timings, unique_train_batches=2, unique_validation_batches=1,
                      train_execution_count=6, validation_execution_count=3,
                      synthetic=False, full_trajectory_proven=False, formal_start=False,
                      diagnostic_checkpoint_promotable=False, active_handover_ready=False,
                      remaining_handover_gate="NO_DEPLOYED_SAFE_PAUSE_INTERFACE_IN_ACTIVE_WORKER",
                      independent_process_reload_state="NOT_RUN")
        atomic_json(output / "performance.json", report)
        raise PerformanceComplete(report)
    return intercept
