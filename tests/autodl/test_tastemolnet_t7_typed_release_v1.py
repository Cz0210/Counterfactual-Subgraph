from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.baselines.tastemolnet_gcf_smoke import PASS_MARKER
from src.eval.tastemolnet_neurosed_gate import STRICT_OFFICIAL_PROVENANCE
from src.eval.tastemolnet_neurosed_official_fixed_budget import OFFICIAL_GCF_COMMIT
from src.utils.managed_execution_v2 import create_managed_attempt, create_worker_staging
from src.utils.tastemolnet_t7_typed_release_v1 import (
    PASS_MARKER as TYPED_PASS_MARKER,
    TasteGCFReleasePinsV1,
    TasteGCFSourceAuthorityV1,
    build_t7_release_candidate,
    validate_t7_release_root,
    verify_and_publish_t7_release,
)
from src.utils.tastemolnet_t7_typed_runtime_v1 import (
    open_t7_sealed,
    seal_t7_worker_evidence,
    t7_managed_config_hash,
    t7_managed_input_hashes,
    t7_release_binding,
    validate_native_t7_science,
    verify_and_publish_t7_sealed,
)
from tests.baselines.test_tastemolnet_gcf_smoke import _science_summary, _trace_row


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _pins() -> TasteGCFReleasePinsV1:
    return TasteGCFReleasePinsV1(
        dataset="tastemolnet",
        source_label=1,
        num_classes=3,
        official_gcf_commit=OFFICIAL_GCF_COMMIT,
        neurosed_commit=STRICT_OFFICIAL_PROVENANCE["greed_commit"],
        neurosed_model_sha="1" * 64,
        neurosed_config_sha="2" * 64,
        neurosed_pair_manifest_sha="3" * 64,
        t3_calibrated_gine_sha="4" * 64,
        t3_temperature_sha="5" * 64,
        dataset_sha="6" * 64,
        train_split_sha="7" * 64,
        validation_split_sha="8" * 64,
        calibration_split_sha="9" * 64,
        test_split_sha="a" * 64,
        inference_direction="generated_to_original",
        neurosed_calibration_loaded=False,
        neurosed_test_loaded=False,
    )


def _authority(tmp_path: Path) -> TasteGCFSourceAuthorityV1:
    return TasteGCFSourceAuthorityV1(
        managed_neurosed_root=str(tmp_path / "neurosed"),
        managed_neurosed_pass_sha256="b" * 64,
        managed_neurosed_gate_sha256="c" * 64,
        managed_neurosed_verification_sha256="d" * 64,
        managed_neurosed_inventory_sha256="e" * 64,
        neurosed_model_path=str(tmp_path / "neurosed/artifacts/best.pt"),
        neurosed_config_path=str(tmp_path / "neurosed/artifacts/config.yaml"),
        neurosed_pair_manifest_path=str(
            tmp_path / "neurosed/artifacts/pair_manifest.json"
        ),
        neurosed_feature_schema_path=str(
            tmp_path / "neurosed/artifacts/feature_schema.json"
        ),
        neurosed_sha256s_path=str(
            tmp_path / "neurosed/artifacts/sha256sums.txt"
        ),
        t3_root=str(tmp_path / "t3"),
        t3_gate_sha256="f" * 64,
        t3_verification_sha256="0" * 64,
        t3_root_inventory_sha256="1" * 64,
        t3_checkpoint_dir=str(tmp_path / "t3/artifacts/checkpoint"),
        t3_checkpoint_id="2" * 64,
        t3_split_manifest_sha256="3" * 64,
        t3_feature_schema_sha256="4" * 64,
        split_paths={
            role: str(tmp_path / f"{role}.csv")
            for role in ("train", "validation", "calibration", "test")
        },
        official_gcf_root=str(tmp_path / "official"),
        official_gcf_inventory_sha256="5" * 64,
        neurosed_distance_threshold=0.25,
        implementation_commit="a" * 40,
        implementation_tree="b" * 40,
        no_neurosed_retraining=True,
        split_payloads_deserialized=False,
        test_payload_deserialized=False,
    )


class _FakeSources:
    def __init__(self, pins: TasteGCFReleasePinsV1, authority: TasteGCFSourceAuthorityV1):
        self.pins = pins
        self.authority = authority
        self.revalidations = 0

    def revalidate(self) -> None:
        self.revalidations += 1

    def __enter__(self) -> "_FakeSources":
        self.revalidate()
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_typed_pins_are_exact_latest_protocol_fields() -> None:
    mapping = _pins().mapping()
    assert set(mapping) == {
        "schema_version", "dataset", "source_label", "num_classes",
        "official_gcf_commit", "neurosed_commit", "neurosed_model_sha",
        "neurosed_config_sha", "neurosed_pair_manifest_sha",
        "t3_calibrated_gine_sha", "t3_temperature_sha", "dataset_sha",
        "train_split_sha", "validation_split_sha", "calibration_split_sha",
        "test_split_sha", "inference_direction",
        "neurosed_calibration_loaded", "neurosed_test_loaded",
    }
    assert "neurosed_distance_threshold" not in mapping
    assert mapping["inference_direction"] == "generated_to_original"
    assert mapping["neurosed_calibration_loaded"] is False
    assert mapping["neurosed_test_loaded"] is False


def test_candidate_writer_cannot_publish_and_independent_verifier_reopens_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pins = _pins()
    authority = _authority(tmp_path)
    calls: list[_FakeSources] = []

    def hold(**_kwargs: object) -> _FakeSources:
        held = _FakeSources(pins, authority)
        calls.append(held)
        return held

    monkeypatch.setattr(
        "src.utils.tastemolnet_t7_typed_release_v1.hold_t7_release_sources",
        hold,
    )
    candidate = tmp_path / "candidate"
    release = tmp_path / "release"
    built = build_t7_release_candidate(
        managed_neurosed_root=tmp_path / "unused-neurosed",
        t3_root=tmp_path / "unused-t3",
        official_gcf_root=tmp_path / "unused-official",
        neurosed_distance_threshold=0.25,
        output_root=candidate,
    )
    assert built["status"] == "READY_FOR_INDEPENDENT_VERIFICATION"
    assert not (candidate / "PASS").exists()
    result = verify_and_publish_t7_release(
        candidate_root=candidate,
        output_root=release,
    )
    assert result["marker"] == TYPED_PASS_MARKER
    assert len(calls) == 2
    assert all(item.revalidations >= 2 for item in calls)
    validated = validate_t7_release_root(release, reopen_sources=False)
    assert validated["status"] == "PASS"
    assert (release / "PASS").read_text() == TYPED_PASS_MARKER + "\n"


def _fake_release(tmp_path: Path) -> SimpleNamespace:
    release_root = tmp_path / "typed-release"
    release_root.mkdir()
    pins = _pins()
    authority = _authority(tmp_path)
    predecessor = _predecessor(pins, authority)
    evidence = {
        "release_id": "123e4567-e89b-42d3-a456-426614174000",
        "release_pins_sha256": pins.sha256,
        "source_authority_sha256": authority.sha256,
        "verification_sha256": "6" * 64,
        "gate_sha256": "7" * 64,
        "pass_sha256": "8" * 64,
    }
    state = {"calls": 0}

    def revalidate() -> dict[str, object]:
        state["calls"] += 1
        return dict(evidence)

    return SimpleNamespace(
        release_root=release_root,
        pins=pins,
        authority=authority,
        sources=SimpleNamespace(neurosed_evidence=predecessor),
        revalidate=revalidate,
        state=state,
    )


def _predecessor(
    pins: TasteGCFReleasePinsV1,
    authority: TasteGCFSourceAuthorityV1,
) -> dict[str, object]:
    return {
        "schema_version": "tastemolnet_gcf_neurosed_managed_final_v1",
        "status": "PASS",
        "marker": "MANAGED_EXECUTION_V2_PASS",
        "final_root": "/private/neurosed",
        "attempt_id": "123e4567-e89b-42d3-a456-426614174001",
        "generation_token": "123e4567-e89b-42d3-a456-426614174002",
        "pass_path": "/private/neurosed/PASS",
        "pass_sha256": authority.managed_neurosed_pass_sha256,
        "gate_path": "/private/neurosed/gate.json",
        "gate_sha256": authority.managed_neurosed_gate_sha256,
        "verification_path": "/private/neurosed/verification.json",
        "verification_sha256": authority.managed_neurosed_verification_sha256,
        "source_inventory_sha256": "9" * 64,
        "published_inventory_sha256": "a" * 64,
        "checkpoint_path": "/private/neurosed/artifacts/best.pt",
        "checkpoint_sha256": pins.neurosed_model_sha,
        "feature_schema_path": "/private/neurosed/artifacts/feature_schema.json",
        "feature_schema_sha256": "b" * 64,
        "sha256s_path": "/private/neurosed/artifacts/sha256sums.txt",
        "sha256s_sha256": "c" * 64,
        "t7_consumer": {
            "role": "GCF_AUXILIARY_DISTANCE_MODEL",
            "classifier": False,
            "calibration_loaded": False,
            "test_loaded": False,
            "checkpoint_sha256": pins.neurosed_model_sha,
        },
    }


def _science(
    binding: dict[str, object], predecessor: dict[str, object]
) -> dict[str, object]:
    summary = _science_summary()
    summary["adapter"]["checkpoint_id"] = binding["t3_checkpoint_id_sha256"]
    summary["neurosed_distance_threshold"] = binding[
        "neurosed_distance_threshold"
    ]
    summary["neurosed_predecessor"] = copy.deepcopy(predecessor)
    return {"trace": [_trace_row([0.6, 0.1, 0.3])], "summary": summary}


def test_worker_seals_and_only_independent_verifier_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTO_TERMINATE_UNCONTROLLED_CHILDREN", "0")
    release = _fake_release(tmp_path)
    binding = t7_release_binding(release)
    science = _science(binding, release.sources.neurosed_evidence)
    stage = tmp_path / "stage"
    destination_parent = tmp_path / "final"
    stage.mkdir()
    destination_parent.mkdir()
    destination = destination_parent / "t7"
    run_id = "taste-t7-unit"
    with create_managed_attempt(
        stage_root=stage,
        controller_id=run_id,
        task_id="T7_GCF_SMOKE_TYPED_RELEASE_V1",
        git_commit=binding["execution_commit"],
        config_hash=t7_managed_config_hash(binding),
        input_hashes=t7_managed_input_hashes(binding),
        boot_id="test-boot",
    ) as attempt, create_worker_staging(attempt) as staging:
        sealed = seal_t7_worker_evidence(
            staging,
            science=science,
            release_binding=binding,
            expected_final_path=destination,
            run_id=run_id,
            gpu_uuid="GPU-unit-0",
        )
        assert not (staging.path / "PASS").exists()
        assert not (staging.path / "gate.json").exists()

    with open_t7_sealed(
        sealed.staging_path,
        expected_attempt_id=sealed.attempt_id,
        expected_generation_token=sealed.generation_token,
    ) as held:
        publication, verification = verify_and_publish_t7_sealed(
            held,
            final_path=destination,
            release=release,
            run_id=run_id,
            gpu_uuid="GPU-unit-0",
        )
    assert publication.final_path == destination
    assert verification["domain_marker"] == PASS_MARKER
    assert verification["neurosed_retrained"] is False
    assert verification["independent_verifier"] is True
    assert release.state["calls"] >= 4
    assert (destination / "PASS").read_text() == "[MANAGED_EXECUTION_V2_PASS]\n"


def test_science_rejects_split_leakage_and_t3_drift(tmp_path: Path) -> None:
    release = _fake_release(tmp_path)
    binding = t7_release_binding(release)
    leaked = _science(binding, release.sources.neurosed_evidence)
    leaked["summary"]["test_payload_loaded"] = True
    with pytest.raises(Exception, match="science semantics"):
        validate_native_t7_science(leaked, expected_binding=binding)
    drifted = _science(binding, release.sources.neurosed_evidence)
    drifted["summary"]["adapter"]["checkpoint_id"] = "f" * 64
    with pytest.raises(Exception, match="typed release inputs"):
        validate_native_t7_science(drifted, expected_binding=binding)


def test_fresh_wrapper_validates_release_before_gpu0_and_holds_one_lock() -> None:
    wrapper = (
        PROJECT_ROOT / "scripts/autodl/run_tastemolnet_gcf_smoke_v2.sh"
    ).read_text()
    assert wrapper.index("validate \\") < wrapper.index("gpu_inventory.py")
    assert 'row["index"] == 0 and row["stable_idle"]' in wrapper
    assert "--gpu-index 0" in wrapper
    assert "gpu_lock.py" in wrapper
    assert "tastemolnet_t7_managed_runner_v3.py" in wrapper
    assert "train_fixed_budget_neurosed" not in wrapper
    assert "TASTE_T7_GCF_WRAPPER_RELEASED=0" not in wrapper


def test_runner_uses_distinct_worker_and_verifier_processes() -> None:
    runner = (
        PROJECT_ROOT / "scripts/autodl/tastemolnet_t7_managed_runner_v3.py"
    ).read_text()
    assert '"--mode", "worker"' in runner
    assert '"--mode", "verifier"' in runner
    assert runner.count("subprocess.run(") == 2
    assert "hold_verified_t7_release" in runner
    assert "train_fixed_budget_neurosed" not in runner
