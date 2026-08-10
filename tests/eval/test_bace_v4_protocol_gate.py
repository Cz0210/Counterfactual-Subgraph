from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_bace_v4_protocol_gate import audit_gate


def test_protocol_gate_rejects_test_selected_ours(tmp_path: Path) -> None:
    ours=tmp_path/"ours.json"; ours.write_text(json.dumps({"selection_frozen":True,"test_used":True,"action_semantics_version":"connected_sanitized_residual_v1","all_selected_have_connected_valid_calibration_action":True,"ranks":list(range(1,21)),"selected_candidate_ids":[f"o{i}" for i in range(20)]}))
    gcf=tmp_path/"gcf"; gcf.mkdir(); (gcf/"run_manifest.json").write_text(json.dumps({"test_loaded":False,"candidate_attrition":{"native_order_preserved":True,"scan_all":True,"scan_exhausted":True}})); (gcf/"candidate_universe.jsonl").write_text("".join(json.dumps({"native_rank":i+1,"connected":True})+"\n" for i in range(20)))
    threshold=tmp_path/"threshold.json"; threshold.write_text(json.dumps({"shared_across_methods":True,"threshold_fitted_on_test":False,"method_specific_threshold":False}))
    files=[]
    for name in ("teacher","molclr","cal","test"):
        p=tmp_path/name; p.write_text(name); files.append(p)
    try:
        audit_gate(ours_selection=ours,gcf_audit_root=gcf,thresholds_json=threshold,teacher_path=files[0],molclr_checkpoint=files[1],calibration_csv=files[2],test_csv=files[3],output_dir=tmp_path/"out",git_commit="abc")
    except RuntimeError as exc:
        assert "failed closed" in str(exc)
    else:
        raise AssertionError("test-selected Ours should fail the protocol gate")
