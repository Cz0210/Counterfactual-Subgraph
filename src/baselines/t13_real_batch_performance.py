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
SCOPE = "T13_FIVE_BATCH_ACCUMULATION_FULL_DUE_VALIDATION_PERFORMANCE_V2"
DIAGNOSTIC_ARMS = ("reference", "optimized", "optimized_reload")
UPDATES_PER_ARM = 2
MAX_DIAGNOSTIC_UPDATES = 8


class PerformanceComplete(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__(report["state"])


def validate_plan(plan):
    exact = dict(scope=SCOPE, train_batches=5, validation_batches="ALL_WHEN_DUE",
                 updates_per_arm=UPDATES_PER_ARM, diagnostic_arm_count=len(DIAGNOSTIC_ARMS),
                 diagnostic_optimizer_updates=6, max_diagnostic_optimizer_updates=MAX_DIAGNOSTIC_UPDATES,
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
        self.compact = False
        self.record_count = 0
        self.record_digest = hashlib.sha256()

    def begin_phase(self, *, compact=False):
        self.records.clear()
        self.compact = compact
        self.record_count = 0
        self.record_digest = hashlib.sha256()

    def phase_records(self):
        if not self.compact:
            return self.copier(self.records)
        return dict(scope="ALL_ORDERED_ORACLE_OUTPUTS", count=self.record_count,
                    ordered_state_sha256=self.record_digest.hexdigest(), raw_records_retained=False)

    def __getattr__(self, key):
        return getattr(self.wrapped, key)

    def __call__(self, *args, **kwargs):
        result = self.wrapped(*args, **kwargs)
        record = self.copier({key: result[key] for key in
                             ("y_pred", "logits", "bridge_audit") if key in result})
        if self.compact:
            from src.baselines.t13_indexed_canary import state_digest
            self.record_digest.update(state_digest(record).encode("ascii"))
            self.record_count += 1
        else:
            self.records.append(record)
        return result


def run_formal_diagnostic_update(*, model, fss, train_loader, validation_loader,
                                pred_model, epoch, optimizer, scheduler, test_globalgce,
                                best_loss, best_state_seen, observed_oracle=None,
                                sample=lambda phase: None, optimizer_step_observer=lambda: None):
    """One pinned formal update, with observation but no alternate mathematics.

    Keep get_rules before loader iteration, five batches sharing that same graph,
    summed losses, exactly one backward (no retain_graph), and the native
    optimizer/zero_grad/scheduler order. Due validation consumes the complete
    original loader with the pre-update rules, as the source implementation does.
    """
    import torch
    from src.baselines.t13_component_diagnostics import cpu_copy
    from src.baselines.t13_indexed_canary import state_digest

    model.train()
    model.gt_gnn.eval()
    loss = loss_kl = loss_sim = loss_cfe = 0.0
    if observed_oracle is not None:
        observed_oracle.begin_phase(compact=True)
    rules = model.get_rules(fss)
    bindings = []
    # Deliberately retain the native enumerate/break placement. Its iterator
    # fetches batch6 before breaking, even though only five enter the objective.
    for batch_index, source_batch in enumerate(train_loader):
        if batch_index >= 5:
            break
        batch = copy.deepcopy(source_batch)
        bindings.append(dict(indices=cpu_copy(batch["index"]).tolist(), sha256=state_digest(batch)))
        values = model.run_one_batch(rules, batch)
        if not all(bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("T13_REAL_CANARY_NONFINITE_LOSS")
        loss += values[0]
        loss_kl += values[1]
        loss_sim += values[2]
        loss_cfe += values[3]
        sample("train_batch" + str(batch_index + 1))
    if len(bindings) != 5:
        raise ValueError("T13_FORMAL_FIVE_FULL_TRAIN_BATCHES_REQUIRED")
    sample("before_backward")
    (loss_cfe if epoch < 35 else loss).backward()
    gradients = {name: cpu_copy(parameter.grad) for name, parameter in model.named_parameters()}
    sample("after_backward")
    optimizer.step()
    optimizer_step_observer()
    optimizer.zero_grad()
    scheduler.step()
    sample("after_optimizer_zero_grad_scheduler")
    training_oracle = observed_oracle.phase_records() if observed_oracle is not None else None
    evaluated = None
    validation_binding = None
    validation_oracle = None
    if epoch % 5 == 0:
        if observed_oracle is not None:
            observed_oracle.begin_phase(compact=True)
        seen_batches = seen_examples = 0
        order_digest = hashlib.sha256()

        def full_validation():
            nonlocal seen_batches, seen_examples
            for source_batch in validation_loader:
                batch = copy.deepcopy(source_batch)
                ids = cpu_copy(batch["index"]).tolist()
                seen_batches += 1
                seen_examples += len(ids)
                order_digest.update(json.dumps(ids, separators=(",", ":")).encode())
                yield batch

        sample("before_full_due_validation")
        with torch.no_grad():
            model.eval()
            evaluated = test_globalgce(full_validation(), model, pred_model, rules)
            val_loss = float(evaluated["loss"].detach().cpu())
            if val_loss < best_loss:
                best_loss = val_loss
                best_state_seen = True
        if seen_examples != len(validation_loader.dataset) or seen_batches != len(validation_loader):
            raise ValueError("T13_FULL_VALIDATION_WAS_NOT_FULLY_CONSUMED")
        validation_binding = dict(batch_count=seen_batches, example_count=seen_examples,
                                  ordered_parent_indices_sha256=order_digest.hexdigest(), complete=True)
        validation_oracle = observed_oracle.phase_records() if observed_oracle is not None else None
        sample("after_full_due_validation")
    return dict(epoch=epoch, rules=cpu_copy(rules), losses=cpu_copy((loss, loss_kl, loss_sim, loss_cfe)),
                gradients=gradients, train_batch_bindings=bindings, training_oracle=training_oracle,
                validation=cpu_copy(evaluated), validation_binding=validation_binding,
                validation_oracle=validation_oracle, best_loss=best_loss, best_state_seen=best_state_seen)


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
        if not any(epoch % 5 == 0 for epoch in range(int(initial["next_epoch"]),
                                                   int(initial["next_epoch"]) + UPDATES_PER_ARM)):
            raise ValueError("T13_TWO_UPDATE_DIAGNOSTIC_HAS_NO_DUE_FULL_VALIDATION")
        # No iterator or batch is created before this snapshot. The source calls
        # get_rules before iter(train), and DataLoader iterator creation uses RNG.
        arm_rng = cpu_copy(rng_state())
        atomic_json(output / "actual_batch_binding.json", dict(dataset_identity=identity,
                    adopted_patterns=adoption, train_batches_per_update=5,
                    validation_policy="FULL_ORIGINAL_LOADER_WHEN_EPOCH_MOD5_EQ0",
                    materialization="BOUNDED_PER_BATCH_INSIDE_ORIGINAL_ITERATION_ORDER",
                    checkpoint_epoch=int(initial["next_epoch"]) - 1, diagnostic_updates_not_full_epochs=True))

        def optimizer_for(target, saved=None):
            optimizer = torch.optim.Adam(target.parameters(), lr=kwargs["learning_rate"], weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
            optimizer.load_state_dict((saved or initial)["optimizer_state"])
            scheduler.load_state_dict((saved or initial)["scheduler_state"])
            return optimizer, scheduler

        def snapshot(target, optimizer, scheduler, *, epoch, best_loss, best_state_seen):
            return cpu_copy(dict(model_state=target.state_dict(), optimizer_state=optimizer.state_dict(),
                                 scheduler_state=scheduler.state_dict(), rng=rng_state(),
                                 next_epoch=epoch + 1, best_loss=best_loss, best_state_seen=best_state_seen,
                                 augmented_dataset_identity=identity,
                                 sampler_state=dict(identity["sampler"], next_epoch=epoch + 1)))

        records, timings = {}, {}
        completed_updates = 0
        original_oracle = model.gt_gnn
        original_bridge_class = original_oracle.bridge.__class__
        oracle_weights = state_digest(original_oracle.bridge.model.state_dict())
        if any(p.requires_grad for p in original_oracle.parameters()):
            raise ValueError("T13_REAL_ORACLE_MUST_REMAIN_FROZEN")
        observed_oracle = ObservedOracle(original_oracle, cpu_copy)
        model.gt_gnn = observed_oracle
        try:
            for arm in DIAGNOSTIC_ARMS:
                model.gt_gnn.bridge.__class__ = (reference.FrozenGINEDifferentiableBridge
                    if arm == "reference" else optimized.FrozenGINEDifferentiableBridge)
                model.load_state_dict(initial["model_state"])
                model.zero_grad(set_to_none=True)
                optimizer, scheduler = optimizer_for(model)
                restore_rng(arm_rng)
                current = model
                rows, times = [], []
                best_loss, best_state_seen = float(initial["best_loss"]), bool(initial["best_state_seen"])
                for step in range(UPDATES_PER_ARM):
                    epoch = int(initial["next_epoch"]) + step
                    if completed_updates >= MAX_DIAGNOSTIC_UPDATES:
                        raise ValueError("T13_DIAGNOSTIC_OPTIMIZER_BUDGET_EXHAUSTED")

                    def on_optimizer_step():
                        nonlocal completed_updates
                        completed_updates += 1
                        atomic_json(output / "diagnostic_update_ledger.json", dict(
                            completed_optimizer_updates=completed_updates, maximum_authorized=MAX_DIAGNOSTIC_UPDATES,
                            planned_optimizer_updates=len(DIAGNOSTIC_ARMS) * UPDATES_PER_ARM,
                            last_arm=arm, last_epoch=epoch, validation_may_still_be_pending=True,
                            formal_quota_consumed=0, scientific_resume_validated=False))

                    def observe_phase(phase):
                        torch.cuda.synchronize()
                        sample(arm + ":epoch" + str(epoch) + ":" + phase)

                    torch.cuda.synchronize(); before = time.perf_counter()
                    row = run_formal_diagnostic_update(model=current, fss=fss, train_loader=train,
                        validation_loader=validation, pred_model=kwargs["pred_model"], epoch=epoch,
                        optimizer=optimizer, scheduler=scheduler, test_globalgce=kwargs["test_globalgce"],
                        best_loss=best_loss, best_state_seen=best_state_seen,
                        observed_oracle=observed_oracle, sample=observe_phase,
                        optimizer_step_observer=on_optimizer_step)
                    best_loss, best_state_seen = row["best_loss"], row["best_state_seen"]
                    torch.cuda.synchronize(); times.append(time.perf_counter() - before)
                    row["after"] = snapshot(current, optimizer, scheduler, epoch=epoch,
                        best_loss=best_loss, best_state_seen=best_state_seen)
                    rows.append(row)
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
                        best_loss, best_state_seen = saved["best_loss"], saved["best_state_seen"]
                        atomic_json(output / "reload_binding.json", dict(
                            state_sha256=state_digest(saved), fresh_model_object=True,
                            fresh_optimizer_object=True, checkpoint_promotable=False))
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
                      timings_seconds=timings, train_batches_per_update=5,
                      train_execution_count=completed_updates * 5,
                      diagnostic_optimizer_updates=completed_updates,
                      maximum_authorized_optimizer_updates=MAX_DIAGNOSTIC_UPDATES,
                      full_validation_execution_count=sum(row["validation"] is not None
                          for rows in records.values() for row in rows),
                      validation_policy="FULL_ORIGINAL_LOADER_WHEN_EPOCH_MOD5_EQ0",
                      synthetic=False, full_trajectory_proven=False, formal_start=False,
                      diagnostic_checkpoint_promotable=False, active_handover_ready=False,
                      scientific_resume_validated=False,
                      remaining_handover_gate="SAME_RUN_RESTORATION_REQUIRES_STORAGE_PAYLOAD_MEMORY_AND_OWNER_BINDING",
                      independent_process_reload_state="NOT_RUN")
        atomic_json(output / "performance.json", report)
        raise PerformanceComplete(report)
    return intercept
