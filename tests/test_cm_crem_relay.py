"""Bounded local/transport interface checks; no scheduler or remote writes."""
import importlib.util
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("cm_relay_fixture", REPO/"scripts/run_cm_crem_relay.py")
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)


def test_fixed_stage_dag_and_no_duplicate_science_stages():
    assert len(relay.STAGES) == len(set(relay.STAGES))
    assert relay.STAGES.index("pilot-closeout") < relay.STAGES.index("attribution")
    assert relay.STAGES.index("select") < relay.STAGES.index("test") < relay.STAGES.index("audit")
    assert relay.STAGES[-2:] == ["export", "package"]


@pytest.mark.parametrize("root", ["/tmp/not_cm", "/share/home/u20526/czx/foo;echo_bad"])
def test_remote_scope_rejected_before_network(tmp_path, root):
    process = subprocess.run([sys.executable, str(REPO/"scripts/run_cm_crem_relay.py"),
        "--hpc-run-root", root, "--hpc-execution-root", "/share/home/u20526/czx/worktrees/cm-test",
        "--local-root", str(tmp_path), "--start-time-utc", "2026-09-09T16:27:44Z", "--once"],
        capture_output=True, text=True)
    assert process.returncode != 0
    assert "ValueError" in process.stderr


def test_transport_compatible_with_mac_rsync_269_and_scoped_lifetime():
    text = (REPO/"scripts/run_cm_crem_relay.py").read_text()
    assert '"--partial"' in text
    assert "--delete" not in text and "--append-verify" not in text and "--info" not in text
    assert "168*3600" in text and "time.sleep(300)" in text
    assert 'failures >= 2' in text
    assert "verify_import(local_package, local_manifest, imported)" in text
    assert "cm_import_receipt.json" in text


def test_record_only_import_help_and_no_model_load():
    process = subprocess.run([sys.executable, "-I", "-B", str(REPO/"scripts/import_cm_crem.py"), "--help"],
                             capture_output=True, text=True)
    assert process.returncode == 0
    assert all(x in process.stdout for x in ("--package", "--manifest", "--destination"))


@pytest.mark.parametrize("field,value", [("science_hash", "other"), ("package_sha256", "other"),
                                        ("package_bytes", 100), ("status", "PENDING")])
def test_reuse_requires_exact_package_not_just_same_science(field, value):
    manifest = {"science_hash": "s", "package_sha256": "p", "package_bytes": 101}
    receipt = {**manifest, "status": "CM_RESULT_IMPORT_VERIFIED"}
    relay.require_import_identity(receipt, manifest)
    receipt[field] = value
    with pytest.raises(ValueError, match="exact CM package"):
        relay.require_import_identity(receipt, manifest)


def test_external_mount_must_exist_and_remote_root_is_package_bound():
    text = (REPO/"scripts/run_cm_crem_relay.py").read_text()
    assert 'os.path.ismount("/Volumes/DireRaven")' in text
    assert 'Existing transfer root is not bound to this package' in text


@pytest.fixture
def terminal_campaign(tmp_path, monkeypatch):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 10, 1, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(relay, 'datetime', FixedDatetime)
    def absent(pid, signal):
        assert pid == 43210 and signal == 0
        raise ProcessLookupError()
    monkeypatch.setattr(relay.os, 'kill', absent)
    monkeypatch.setattr(relay, 'ssh_read', lambda *a, **k: pytest.fail('argv T0 must not query or submit anything'))
    root='/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/baselines/cm_crem_global_v1/campaign'
    spec_path=root+'/successor.json'
    args=SimpleNamespace(start_time_utc='2026-09-09T16:27:44Z', hpc_run_root=root,
                         hpc_execution_root='/share/home/u20526/czx/worktrees/new',
                         hpc_alias='tongji-hpc', resume_after_terminal=True)
    old={'pid':43210, 'hpc_run_root':root, 'max_lifetime_hours':168,
         'execution_root':'/share/home/u20526/czx/worktrees/old', 'spec':root+'/spec.json'}
    state={'pid':43210,'hpc_run_root':root,'status':'BLOCKED_ASSET_OVER_6H'}
    launch={'argv':['python','relay.py','--hpc-run-root',root,'--local-root',str(tmp_path),
                    '--start-time-utc',args.start_time_utc], 'max_hours':168}
    for name,data in [('relay_identity.json',old),('state.json',state),('launch_intent.json',launch),
                      ('launch_receipt.json', {'pid':43210,'launched':True})]:
        (tmp_path/name).write_text(json.dumps(data, indent=3)+'\n')
    (tmp_path/'relay.lock').touch()
    return tmp_path,args,spec_path


def _claim(campaign):
    local,args,spec_path=campaign
    with (local/'relay.lock').open('a') as lock:
        return relay.claim_relay_identity(args,local,spec_path,lock)


def test_resume_preserves_exact_terminal_evidence_and_original_t0(terminal_campaign):
    local,args,spec_path=terminal_campaign
    originals={name:(local/name).read_bytes() for name in
               ('relay_identity.json','state.json','launch_intent.json','launch_receipt.json')}
    t0=_claim(terminal_campaign)
    assert t0 == datetime(2026,9,9,16,27,44,tzinfo=timezone.utc)
    current=json.loads((local/'relay_identity.json').read_text())
    archive=Path(current['prior_terminal_archive'])
    assert current['start_time_utc']=='2026-09-09T16:27:44Z'
    assert current['planning_deadline_utc']=='2026-09-16T16:27:44Z'
    assert current['pid']==os.getpid() and current['spec']==spec_path
    for name,body in originals.items():
        assert (archive/name).read_bytes()==body
        assert (archive/name).stat().st_mode & 0o222 == 0
    assert (local/'launch_intent.json').read_bytes()==originals['launch_intent.json']
    receipt=json.loads((archive/'preservation_receipt.json').read_text())
    assert not receipt['science_resubmitted'] and not receipt['submission_receipts_modified']
    assert json.loads((local/'state.json').read_text())['status']=='RESUMING_AFTER_ASSET_TERMINAL'


@pytest.mark.parametrize('which,value', [('state.json','BLOCKED_FAILED_STAGE'),
                                       ('state.json','WAITING_SLURM')])
def test_only_asset_terminal_can_resume(terminal_campaign,which,value):
    local,_,_=terminal_campaign
    state=json.loads((local/which).read_text());state['status']=value
    (local/which).write_text(json.dumps(state))
    with pytest.raises(ValueError,match='Only BLOCKED_ASSET_OVER_6H'):_claim(terminal_campaign)
    assert not (local/'relay_attempts').exists()


@pytest.mark.parametrize('outcome',[None,PermissionError()])
def test_live_or_inaccessible_old_pid_never_taken_over(terminal_campaign,monkeypatch,outcome):
    def live(*args):
        if outcome is not None:raise outcome
    monkeypatch.setattr(relay.os,'kill',live)
    with pytest.raises(ValueError,match='still exists'):_claim(terminal_campaign)
    assert not (terminal_campaign[0]/'relay_attempts').exists()


def test_resume_rejects_new_start_time(terminal_campaign):
    terminal_campaign[1].start_time_utc='2026-09-10T00:59:00Z'
    with pytest.raises(ValueError,match='cannot reset'):_claim(terminal_campaign)


def test_resume_rejects_expired_original_horizon(terminal_campaign,monkeypatch):
    class ExpiredDatetime(datetime):
        @classmethod
        def now(cls,tz=None):return cls(2026,9,17,tzinfo=timezone.utc)
    monkeypatch.setattr(relay,'datetime',ExpiredDatetime)
    with pytest.raises(ValueError,match='horizon is expired'):_claim(terminal_campaign)


def test_legacy_start_can_only_come_from_bound_original_spec(terminal_campaign,monkeypatch):
    local,args,_=terminal_campaign
    launch=json.loads((local/'launch_intent.json').read_text())
    launch['argv']=launch['argv'][:-2]
    (local/'launch_intent.json').write_text(json.dumps(launch))
    calls=[]
    def read(alias,argv):
        calls.append(argv)
        return json.dumps({'spec':args.hpc_run_root+'/spec.json',
                           'start_time_utc':'2026-09-09T16:27:44Z','planning_horizon_hours':168})
    monkeypatch.setattr(relay,'ssh_read',read)
    _claim(terminal_campaign)
    assert len(calls)==1 and calls[0][-1].endswith('/spec.json')
    assert not any('submit_cm_crem_stage' in item for item in calls[0])


def test_same_campaign_flock_remains_exclusive_after_claim(terminal_campaign):
    local,args,spec_path=terminal_campaign
    with (local/'relay.lock').open('a') as lock:
        relay.claim_relay_identity(args,local,spec_path,lock)
        code='import fcntl,sys; f=open(sys.argv[1],"a"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)'
        contender=subprocess.run([sys.executable,'-c',code,str(local/'relay.lock')],capture_output=True,text=True)
        assert contender.returncode!=0 and 'BlockingIOError' in contender.stderr


def test_wrong_lock_inode_rejected(terminal_campaign):
    local,args,spec_path=terminal_campaign
    with (local/'not-the-campaign-lock').open('a') as lock:
        with pytest.raises(ValueError,match='original campaign lock'):
            relay.claim_relay_identity(args,local,spec_path,lock)


def test_existing_identity_needs_explicit_resume_flag(terminal_campaign):
    terminal_campaign[1].resume_after_terminal=False
    with pytest.raises(ValueError,match='explicit terminal-resume'):_claim(terminal_campaign)


def test_fresh_identity_records_original_horizon(tmp_path):
    (tmp_path/'relay.lock').touch()
    args=SimpleNamespace(start_time_utc='2026-09-09T16:27:44Z', hpc_run_root='/hpc/campaign',
                         hpc_execution_root='/hpc/code', resume_after_terminal=False)
    with (tmp_path/'relay.lock').open('a') as lock:
        relay.claim_relay_identity(args,tmp_path,'/hpc/campaign/spec.json',lock)
    identity=json.loads((tmp_path/'relay_identity.json').read_text())
    assert identity['planning_deadline_utc']=='2026-09-16T16:27:44Z'


def test_archive_symlink_cannot_escape_campaign(terminal_campaign,tmp_path):
    local,_,_=terminal_campaign
    elsewhere=tmp_path/'elsewhere';elsewhere.mkdir()
    (local/'relay_attempts').symlink_to(elsewhere,target_is_directory=True)
    with pytest.raises(ValueError,match='local regular directory'):_claim(terminal_campaign)
    assert list(elsewhere.iterdir())==[]


def test_lock_symlink_is_not_a_new_lease(terminal_campaign):
    local,args,spec_path=terminal_campaign
    (local/'relay.lock').unlink()
    (local/'another-lock').touch()
    (local/'relay.lock').symlink_to(local/'another-lock')
    with (local/'relay.lock').open('a') as lock:
        with pytest.raises(ValueError,match='symlink'):
            relay.claim_relay_identity(args,local,spec_path,lock)
