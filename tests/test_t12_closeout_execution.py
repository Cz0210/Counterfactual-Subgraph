import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.utils.main_ready_task_specs import stable_sha256
from src.utils.t12_raw_evidence import RawEvidenceResolver
from src.utils.t12_shadow_recovery import JointLedger, build_shadow_plan, read_ledger
from src.utils.t12_shadow_execution import run_live_tail


def row():
    return dict(graph_identity="g", model_graph={"feature_schema": "unchanged", "nodes": [6]},
                canonical_query_sha="q", canonical_probabilities=[0.1, 0.8, 0.1],
                observed_probabilities=[0.1, 0.8, 0.1],
                valid_fullgraph=True,
                raw_classifier={"dtype": "float32", "shape": [3], "values": [1., 2., 1.]},
                raw_neurosed={"dtype": "float32", "shape": [2], "values": [0., 1.]},
                raw_normalizer={"dtype": "float32", "shape": [2], "values": [2., 3.]},
                threshold=0.1)


def test_raw_binding_persists_actual_values_and_no_alias(tmp_path):
    resolver = RawEvidenceResolver("a" * 64)
    source = row()
    resolver.remember(source)
    source["raw_classifier"]["values"][0] = 900
    assert resolver.resolve("g")["evidence"]["raw_classifier"]["values"][0] == 1
    path = tmp_path / "raw.gz"
    resolver.save(path)
    restored = RawEvidenceResolver("a" * 64)
    restored.load(path)
    assert restored.resolve("g") == resolver.resolve("g")


def test_probability_not_raw_and_missing_cache_not_faked():
    resolver = RawEvidenceResolver("a" * 64)
    value = row()
    value["raw_classifier"] = None
    resolver.remember(value)
    assert resolver.resolve("g")["missing"] == ["RAW_CLASSIFIER_LOGITS"]
    assert resolver.resolve("old-250")["status"] == "CACHE_RAW_EVIDENCE_MISSING"
    assert resolver.resolve("g", canonical_probabilities=[1, 0, 0])["status"].endswith("MISMATCH")


def test_raw_content_and_contract_corruption_rejected(tmp_path):
    resolver = RawEvidenceResolver("a" * 64)
    resolver.remember(row())
    resolver.rows["g"]["raw_neurosed"]["values"][0] = 7
    with pytest.raises(ValueError, match="BINDING_CHANGED"):
        resolver.resolve("g")
    with pytest.raises(ValueError, match="BOUND_REACHED"):
        RawEvidenceResolver("a" * 64, max_bytes=10).remember(row())


def test_continuous_tail_keeps_same_objects_and_seals_before_tail(tmp_path):
    plan = build_shadow_plan(run_id="fixture", reference_root="/reference", output_root="/output",
        source_bindings={"fixture": True}, existing_continuous_ledgers={}, activation_plan="/formal")
    ledger = JointLedger(tmp_path, start=251, end=500, binding_sha=plan["plan_sha256"])
    for step in range(251, 501):
        ledger.append({"step": step, "fixture": True})
    cp500 = tmp_path / "500.json"
    cp500.write_text(json.dumps({"checkpoint_cursor": 500}))
    cp510 = tmp_path / "510.json"
    observer = SimpleNamespace(ledger=ledger)
    vrrw = SimpleNamespace(traversed_hashes=list(range(500)))
    original_objects = [vrrw, object(), object()]
    events = []
    original_cwd = Path.cwd()

    def walk(**kwargs):
        assert Path.cwd() == tmp_path / "continuous-native-runtime"
        assert cp500.exists() and (tmp_path / "steps-00251-00500.jsonl.joint.json").exists()
        assert kwargs["vrrw"] is vrrw
        assert kwargs["start_step"] == 501 and kwargs["end_step"] == 510
        assert kwargs["resume_graph_hash"] == "current500"
        for step in range(501, 511):
            observer.ledger.append({"step": step, "fixture": True})
            vrrw.traversed_hashes.append(step)
        events.append("walk")
        return SimpleNamespace(current_graph_hash="current510")

    def commit(**kwargs):
        assert kwargs["vrrw"] is original_objects[0]
        assert kwargs["bridge"] is original_objects[1]
        assert kwargs["adapter"] is original_objects[2]
        assert kwargs["completed_steps"] == 510
        cp510.write_text(json.dumps({"checkpoint_cursor": 510}))
        events.append("commit")
        return cp510

    live = dict(checkpoint_cursor=500, checkpoint_manifest=cp500,
        vrrw=vrrw, bridge=original_objects[1], adapter=original_objects[2],
        action_counts={}, current_graph_identity="current500", input_graphs=[],
        importance_args={}, orchestrator=SimpleNamespace(commit=commit),
        sources=SimpleNamespace(revalidate=lambda: events.append("revalidate")), np=None, torch=None)
    result = run_live_tail(plan=plan, arm="reference", observer=observer,
        resolver=RawEvidenceResolver("a" * 64), ledger_root=tmp_path, live=live, walk=walk)
    assert events == ["walk", "revalidate", "commit"]
    assert Path.cwd() == original_cwd
    assert result["new_transitions"] == 10
    assert result["status"] == "DIAGNOSTIC_TAIL_COMMITTED_NOT_PARITY"
    assert len(read_ledger(tmp_path / "steps-00501-00510.jsonl.gz",
        binding_sha=plan["plan_sha256"], start=501, end=510)) == 10


def test_tail_without_conditional_budget_rejected(tmp_path):
    plan = build_shadow_plan(run_id="fixture", reference_root="/reference", output_root="/output",
        source_bindings={"fixture": True}, existing_continuous_ledgers={"reference": {"complete": True}},
        activation_plan="/formal")
    with pytest.raises(ValueError, match="UNBUDGETED"):
        run_live_tail(plan=plan, arm="reference", observer=None, resolver=None,
                      ledger_root=tmp_path, live={}, walk=None)


def test_default_full_kernel_ast_is_unchanged_except_explicit_hook():
    import ast
    import hashlib
    import subprocess
    from src.utils.tastemolnet_t12_accelerated_from250 import AUDITED_DIAGNOSTIC_TAIL_SOURCE_SHA256
    root = Path(__file__).resolve().parents[1]
    path = "src/baselines/tastemolnet_gcf_full.py"
    old = ast.parse(subprocess.check_output(["git", "show", "f0dec58c87bb4c45f4bcdec9f344341b71d51861:" + path], cwd=root))
    current_bytes = (root / path).read_bytes()
    assert hashlib.sha256(current_bytes).hexdigest() == AUDITED_DIAGNOSTIC_TAIL_SOURCE_SHA256
    current = ast.parse(current_bytes)

    class StripExplicitHook(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module == "typing":
                node.names = [n for n in node.names if n.name != "Callable"]
            return node

        def visit_FunctionDef(self, node):
            if node.name == "run_t12_generation_segment":
                index = next(i for i, arg in enumerate(node.args.kwonlyargs)
                             if arg.arg == "diagnostic_after_checkpoint")
                assert isinstance(node.args.kw_defaults[index], ast.Constant)
                assert node.args.kw_defaults[index].value is None
                del node.args.kwonlyargs[index]
                del node.args.kw_defaults[index]
            return self.generic_visit(node)

        def visit_If(self, node):
            if any(isinstance(n, ast.Name) and n.id == "diagnostic_after_checkpoint"
                   for n in ast.walk(node.test)):
                return None
            return self.generic_visit(node)

    stripped = StripExplicitHook().visit(current)
    assert ast.dump(stripped, include_attributes=False) == ast.dump(old, include_attributes=False)


def test_owner_call_side_really_inherits_exclusive_fd(tmp_path, monkeypatch):
    import fcntl
    import os
    import sys
    import src.utils.t12_shadow_recovery as module
    import src.utils.final16_owner_registry_v1 as registry
    lease_path = tmp_path / "existing-owner.lease"
    child_path = tmp_path / "child.py"
    output = tmp_path / "proof.json"
    child_path.write_text('''import os,fcntl,json,sys
from pathlib import Path
fd=int(os.environ['T12_OWNER_HELD_GPU_FD'])
fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
path=Path(sys.argv[sys.argv.index('--owner-binding')+1])
binding=json.loads(path.read_text())
assert os.getppid()==binding['owner_pid']
assert os.environ['CUDA_VISIBLE_DEVICES']==binding['gpu_uuid']
assert os.fstat(fd).st_ino==os.stat(binding['lease_path']).st_ino
with open(binding['lease_path'],'rb') as contender:
 try: fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError: pass
 else: raise AssertionError('lease was not held')
Path(sys.argv[sys.argv.index('--output')+1]).write_text('FD_INHERITED_AND_EXCLUSIVE')
''')
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps(dict(owner_pid=os.getpid(), owner_start_ticks=7,
        lease_path=str(lease_path), gpu_uuid="GPU-test-uuid")))
    parity = tmp_path / "parity.json"
    parity.write_text("{}")
    # This test isolates FD transport. Separate unchanged tests validate the
    # full scientific-parity gate; a transport fixture cannot certify science.
    monkeypatch.setattr(module, "validate_full_parity", lambda _: None)
    monkeypatch.setattr(registry, "process_start_ticks", lambda *_: 7)
    with lease_path.open("a+b") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = module.dispatch_inherited_activation(binding_path=binding, plan_path=tmp_path / "plan.json",
            parity_path=parity, output=output, python=sys.executable, entrypoint=child_path,
            config=tmp_path / "config.yaml", held_lease=lease)
        assert code == 0 and output.read_text() == "FD_INHERITED_AND_EXCLUSIVE"
