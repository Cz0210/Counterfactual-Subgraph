from pathlib import Path

import pytest

from src.utils import autodl_mut_independent_adoption as m


def test_device_only_remount_is_not_a_scientific_change():
    old = dict(size=30, inode=5, device=126, mtime_ns=1, ctime_ns=2, mode=33188)
    assert m.same_immutable_stat(old, {**old, "device": 76})
    for key in ("size", "inode", "mtime_ns", "ctime_ns", "mode"):
        assert not m.same_immutable_stat(old, {**old, key: old[key] + 1})


def test_authorization_is_nominated_and_does_not_claim_parity(tmp_path):
    source = tmp_path / m.SOURCE_NAME
    source.mkdir()
    auth = dict(schema_version=m.AUTH_SCHEMA, status="APPROVED", historical_source_root=str(source),
                allow_independent_trace_on_adoption=True, trace_on_off_parity_required=False,
                test_used_for_adoption_decision=False)
    m._validate_authorization(auth, source)
    for key, bad in (("historical_source_root", "/different"),
                     ("allow_independent_trace_on_adoption", False),
                     ("trace_on_off_parity_required", True),
                     ("test_used_for_adoption_decision", True)):
        with pytest.raises(m.AdoptionError):
            m._validate_authorization({**auth, key: bad}, source)


def test_large_objects_are_never_json_loaded(tmp_path):
    p = tmp_path / "large.json"
    with p.open("wb") as f:
        f.truncate(m.SMALL_FILE_LIMIT + 1)
    with pytest.raises(m.AdoptionError, match="EVIDENCE_INSUFFICIENT"):
        m.obj(p)


def test_old_v2_does_not_skip_existing_500_step_gate(tmp_path):
    from scripts.autodl.run_mut_comrecgc_parity_standardization import _validate_historical_adoption
    p = tmp_path / "old.json"
    m.write_json(p, {"schema_version": "mut_comrecgc_historical50k_adoption_v2", "status": "PASS"})
    with pytest.raises((ValueError, FileNotFoundError)):
        _validate_historical_adoption(p, source_root=tmp_path)


def test_v3_dispatches_its_own_validator_not_fake_v2(tmp_path, monkeypatch):
    from scripts.autodl.run_mut_comrecgc_parity_standardization import _validate_historical_adoption
    p = tmp_path / "new.json"
    m.write_json(p, {"schema_version": m.SCHEMA})
    seen = []
    def validate(path, *, source_root):
        seen.append((path, source_root))
        return {"trace_parity_passed": False}
    monkeypatch.setattr(m, "validate_receipt", validate)
    assert _validate_historical_adoption(p, source_root=tmp_path) == {"trace_parity_passed": False}
    assert seen == [(p, tmp_path)]


def test_chemistry_v3_discloses_no_parity_no_resume_claim(tmp_path, monkeypatch):
    from src.baselines.comrecgc.preregistration import validate_chemistry_trace_evidence
    p = tmp_path / "adoption.json"
    m.write_json(p, {"schema_version": m.SCHEMA, "source_generation_root": str(tmp_path)})
    monkeypatch.setattr(m, "validate_receipt", lambda *a, **k: {
        "candidate_count": 100235, "source_lineage_path": "/trace", "source_lineage_sha256": "a" * 64,
        "source_payload_path": "/payload", "source_payload_sha256": m.SOURCE_PAYLOAD_SHA256})
    result = validate_chemistry_trace_evidence(p, dataset="mutagenicity")
    assert result["trace_integrity_passed"] is True
    assert result["trace_parity_required"] is False
    assert result["trace_parity_passed"] is False
    assert result["500_step_semantic_equivalence_passed"] is False
    with pytest.raises(ValueError):
        validate_chemistry_trace_evidence(p, dataset="bace")


def test_science_disagreement_never_becomes_metadata_gap():
    with pytest.raises(m.AdoptionError) as caught:
        m._fields({"candidate_graph_hashes_sha256": "wrong"},
                  {"candidate_graph_hashes_sha256": m.UNIVERSE}, "pair")
    assert caught.value.category == "SCIENCE_INVALID"


def test_native_selected_rules_use_actual_list_schema(tmp_path):
    p = tmp_path / "selected_common_recourses.json"
    rows = [{"rank": k} for k in range(100)]
    m.write_json(p, rows)
    assert m.selected_rows(p) == rows
    m.write_json(p, rows[:-1])
    with pytest.raises(m.AdoptionError, match="SCIENCE_INVALID"):
        m.selected_rows(p)


def test_cli_has_no_generation_or_signal_interface():
    source = (Path(__file__).resolve().parents[2] / "scripts/autodl/adopt_mut_historical_independent.py").read_text()
    assert 'choices=("audit", "seal")' in source
    for forbidden in ("os.kill(", "subprocess.Popen(", "torch.load(", "sqlite3.connect("):
        assert forbidden not in source


def test_no_full_parity_or_algorithm_reload_claims_in_new_adoption():
    source = Path(m.__file__).read_text()
    assert '"500_step_semantic_equivalence_passed": False' in source
    assert '"algorithm_checkpoint_reload_claimed": False' in source
    assert '"candidate_universe_reconstructed_this_attempt": False' in source
    assert '"source_generation_oracle": "frozen_project_GNN_native_importance"' in source
