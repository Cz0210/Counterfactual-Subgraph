import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from src.utils.t12_shadow_recovery import build_shadow_plan, require_natural_510
from src.utils.t12_shadow_execution import require_restore500_admission


class Restore500Admission(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.plan = build_shadow_plan(run_id='same-plan', reference_root=str(self.root),
            output_root=str(self.root/'output'), source_bindings={'active_reader':{'pid':9,'start_ticks':4}},
            existing_continuous_ledgers={}, activation_plan=str(self.root/'formal'))
        self.stage = next(s for s in self.plan['stages'] if s['stage_id']=='reference_reload_501_510')
        self.contract = {'source_science_attempt_id':'original', 'generation_token':'original-token'}
        cp = self.root/'checkpoints/checkpoint-00000500.manifest.json'
        cp.parent.mkdir()
        self.manifest = {'checkpoint_cursor':500, 'status':'COMMITTED', 'attempt_id':'original',
            'generation_token':'original-token', 'payload_file':'checkpoint-00000500.pt',
            'state_sha256':'state', 'rng_sha256':'rng'}
        self.joint = {'checkpoint':str(cp.parent/self.manifest['payload_file']),
            'state':'EXTERNAL_JOURNAL_JOINT500_PASS_PENDING_REAL_ADAPTER_RESTORE',
            'preserved':24, 'original_databases_opened':False, 'source_paths_only_relocated':True,
            'state_sha256':'state', 'rng_sha256':'rng'}
        self.binding = {'fork_source_checkpoint':str(cp), 'fork_source_root':str(self.root),
            'restore500_admission':{'schema':'t12_external_joint500_load_admission_v1',
                'plan_sha256':self.plan['plan_sha256'], 'stage_id':self.stage['stage_id'],
                'checkpoint_promotable':False, 'checkpoint_manifest':str(cp),
                'joint_receipt':str(self.root/'joint.json')}}
        self.seal()

    def seal(self):
        a = self.binding['restore500_admission']
        for k, v in [('joint_receipt',self.joint),('checkpoint_manifest',self.manifest)]:
            raw=json.dumps(v).encode();Path(a[k]).write_bytes(raw)
            a[k+'_sha256']=hashlib.sha256(raw).hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def call(self, **kw):
        return require_restore500_admission(plan=self.plan, stage=self.stage, binding=self.binding,
            contract=self.contract, process_alive=kw.get('alive',lambda *_:False))

    def test_no_circular_510_but_no_parity_claim(self):
        before=self.plan['plan_sha256'];r=self.call()
        self.assertEqual(r['new_scientific_steps'],0)
        self.assertFalse(r['checkpoint_promotable'])
        self.assertEqual(before,self.plan['plan_sha256'])
        with self.assertRaises(FileNotFoundError):
            require_natural_510(self.plan,process_alive=lambda *_:False)

    def test_incomplete_and_wrong_rng_rejected(self):
        self.joint['preserved']=23;self.seal()
        with self.assertRaisesRegex(ValueError,'INCOMPLETE'):self.call()
        self.joint['preserved']=24;self.joint['rng_sha256']='other';self.seal()
        with self.assertRaisesRegex(ValueError,'INCOMPLETE'):self.call()

    def test_live_predecessor_rejected(self):
        with self.assertRaisesRegex(ValueError,'LIVE_PREDECESSOR'):self.call(alive=lambda *_:True)

    def test_changed_receipt_rejected(self):
        Path(self.binding['restore500_admission']['joint_receipt']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'RECEIPT_CHANGED'):self.call()

    def test_only_existing_ten_step_reload(self):
        self.stage=self.plan['stages'][0]
        with self.assertRaisesRegex(ValueError,'SCOPE_MISMATCH'):self.call()
