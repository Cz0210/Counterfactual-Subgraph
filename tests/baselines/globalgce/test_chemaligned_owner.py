import json
from pathlib import Path
import tempfile
import unittest

from src.ablations.gnn.early_policy import gpu_allowed
from src.baselines.bace_globalgce_chemaligned_owner import predecessor_ready, training_command
from src.eval.bace_frozen_gnn_contracts import sha256_file, stable_sha256


class OwnerTests(unittest.TestCase):
    def test_global_scope_requires_bound_contract_and_exclusive_gpu0(self):
        row=dict(owners_healthy=True,registry_healthy=True,memory_safe=True,storage_safe=True,
            checkpoint_resume_pass=True,main_ready_waiting_gpu=False,gpu_main_reservation=False,
            active_early_ablation_gpus=0,gpu_idle_seconds=0,gpu_index=0,
            actual_gpu_observation={'process_count':0},chemaligned_contract_verified=True)
        self.assertTrue(gpu_allowed(row,family='globalgce_chemaligned')['allowed'])
        for change in ({'gpu_index':1},{'gpu_main_reservation':True},
                       {'chemaligned_contract_verified':False},
                       {'actual_gpu_observation':{'process_count':1}},
                       {'main_ready_waiting_gpu':True}):
            self.assertFalse(gpu_allowed({**row,**change},family='globalgce_chemaligned')['allowed'])

    def test_bound_predecessor_not_merely_exit0(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); contract=root/'contract.json'; frozen=root/'freeze.json'
            contract.write_text(json.dumps({'self_sha256':'train-contract'}))
            descriptor={'search_contract':str(contract),'search_contract_sha256':sha256_file(contract),
                        'candidate_freeze':str(frozen)}
            self.assertFalse(predecessor_ready(descriptor))
            value={'state':'TRAIN_ONLY_POOL_FROZEN','proposal_source':'OURS_MAIN_PPO_66',
                   'search_contract_sha256':'train-contract','test_opened':False}
            frozen.write_text(json.dumps({**value,'self_sha256':stable_sha256(value)}))
            self.assertTrue(predecessor_ready(descriptor))
            value['test_opened']=True
            frozen.write_text(json.dumps({**value,'self_sha256':stable_sha256(value)}))
            with self.assertRaises(ValueError):predecessor_ready(descriptor)

    def test_canary_and_formal_commands_use_distinct_roots(self):
        spec={'runtime_config':'/runtime/config.yaml','training_contract':'/repair/contract.json'}
        config={'gpu_canary_root':'/repair/canary','formal_output_root':'/repair/formal',
                'rematerialization_root':'/repair/remat'}
        canary=training_command(spec,config,'canary',resume=False)
        formal=training_command(spec,config,'formal',resume=True)
        self.assertEqual(canary[canary.index('--action')+1],'train-canary')
        self.assertEqual(formal[formal.index('--action')+1],'train')
        self.assertEqual(formal[formal.index('--output-root')+1],'/repair/formal')
        self.assertIn('--resume',formal);self.assertNotIn('--resume',canary)
        self.assertEqual(canary[canary.index('--device')+1],'cuda:0')

    def test_missing_canary_cannot_consume_formal_quota(self):
        from src.baselines.bace_globalgce_chemaligned_training import require_gpu_canary
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                require_gpu_canary({'gpu_canary_root':td},'config')


if __name__=='__main__':unittest.main()
