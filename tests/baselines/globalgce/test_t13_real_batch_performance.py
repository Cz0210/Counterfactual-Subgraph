import copy
import hashlib
from pathlib import Path

import pytest

from src.baselines.t13_real_batch_performance import (
    GIB, SCOPE, ObservedOracle, require_identity, snapshot_checkpoint, validate_plan,
)


def plan():
    result = dict(scope=SCOPE, train_batches=2, validation_batches=1, max_wall_seconds=1800,
                  batch_size=500, num_workers=0, seed=7, epochs=100, source_label=1,
                  target_label=0, synthetic=False, formal_start=False, active_handover=False,
                  remining=False, test_loaded=False, calibration_loaded=False,
                  tmpfs_reserved_bytes=0, process_peak_budget_bytes=64 * GIB,
                  output_root="/autodl-fs/data/counterfactual-subgraph-runtime/outputs/performance/fresh")
    for key in ("source_checkpoint", "source_index_manifest", "source_cohort_manifest", "train_csv",
                "gnn_checkpoint", "official_root", "gspan_adoption_proof", "reference_source"):
        result[key] = "/autodl-fs/data/counterfactual-subgraph-runtime/protected/" + key
    return result


@pytest.mark.parametrize("key,value", [
    ("train_batches", 3), ("validation_batches", 2), ("synthetic", True),
    ("batch_size", 1), ("num_workers", 1), ("max_wall_seconds", 1801),
    ("formal_start", True), ("active_handover", True), ("remining", True),
    ("test_loaded", True), ("calibration_loaded", True), ("target_label", 1),
    ("tmpfs_reserved_bytes", GIB), ("epochs", 99), ("seed", 8),
])
def test_real_batch_scope_cannot_expand_or_be_synthetic(key, value):
    value_plan = plan()
    value_plan[key] = value
    with pytest.raises(ValueError):
        validate_plan(value_plan)


def test_real_plan_is_not_formal_or_handover_authorization():
    assert validate_plan(plan())["active_handover"] is False


@pytest.mark.parametrize("output", ["/dev/shm/t13", "/root/autodl-tmp/t13", "/tmp/t13"])
def test_only_persistent_output(output):
    value = plan()
    value["output_root"] = output
    with pytest.raises(ValueError, match="PERSISTENT"):
        validate_plan(value)


def test_source_cannot_be_inside_output():
    value = plan()
    value["source_checkpoint"] = value["output_root"] + "/checkpoint.pt"
    with pytest.raises(ValueError, match="OVERLAPS"):
        validate_plan(value)


def test_snapshot_preserves_source_and_cannot_overwrite(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"immutable checkpoint")
    destination = tmp_path / "copy"
    receipt = snapshot_checkpoint(source, destination)
    assert source.read_bytes() == destination.read_bytes() == b"immutable checkpoint"
    assert receipt["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        snapshot_checkpoint(source, destination)


def identity_binding():
    identity = dict(index_sha256="index", masks_sha256="mask", sample_count=1273625,
                    sampler=dict(batch_size=500, num_workers=0))
    resume = dict(target_label=0, dataset_name="TasteMolNet")
    checkpoint = dict(augmented_dataset_identity=copy.deepcopy(identity), next_epoch=23,
                      sampler_state=dict(identity["sampler"], next_epoch=23), resume_identity=resume)
    return checkpoint, identity, copy.deepcopy(identity), resume


def test_actual_index_masks_and_checkpoint_binding():
    require_identity(*identity_binding())


@pytest.mark.parametrize("field", ["index_sha256", "masks_sha256", "sample_count"])
def test_changed_compact_index_is_not_accepted(field):
    args = identity_binding()
    args[1][field] = "changed"
    with pytest.raises(ValueError, match="INDEX_OR_MASK"):
        require_identity(*args)


def test_sampler_boundary_must_match_checkpoint():
    args = identity_binding()
    args[0]["sampler_state"]["next_epoch"] = 22
    with pytest.raises(ValueError, match="SAMPLER_CURSOR"):
        require_identity(*args)


def test_old_official_active_worker_has_no_perf_pause_callback():
    source = (Path(__file__).parents[3] / "src/baselines/tastemolnet_globalgce_full.py").read_text()
    assert "after_epoch_checkpoint=None" in source


def test_entrypoint_cannot_run_without_inherited_owner_fd():
    source = (Path(__file__).parents[3] / "scripts/benchmarks/benchmark_t13_real_batch.py").read_text()
    assert 'args.held_gpu_fd is None or args.owner_evidence is None' in source
    assert "flock(competitor" in source
    assert 'age <= 120' in source
    assert 'gpu_index") != 2' in source
    assert "SIGKILL" not in source
    assert "full_start.json" not in source


def test_no_new_platform_no_import_of_actual_science_on_inspect():
    source = (Path(__file__).parents[3] / "src/baselines/t13_real_batch_performance.py").read_text()
    assert "GPUFileLock(" not in source
    assert "matrix" not in source
    assert "NO_DEPLOYED_SAFE_PAUSE_INTERFACE_IN_ACTIVE_WORKER" in source


def test_observation_does_not_invoke_extra_oracle_or_alias_outputs():
    class Oracle:
        calls = 0
        def __call__(self, value):
            self.calls += 1
            return {"logits": [value], "y_pred": [value], "bridge_audit": {"valid": True}}
    original = Oracle()
    observed = ObservedOracle(original, copy.deepcopy)
    result = observed(4)
    result["logits"][0] = 100
    assert original.calls == 1
    assert observed.records[0]["logits"] == [4]


def test_each_arm_clones_mutable_native_batch():
    source = (Path(__file__).parents[3] / "src/baselines/t13_real_batch_performance.py").read_text()
    assert "batch = copy.deepcopy(source_batch)" in source
    assert "[copy.deepcopy(validation_batch)]" in source
