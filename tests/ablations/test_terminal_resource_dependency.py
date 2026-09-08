"""No Torch/models: terminal evidence and real flock exclusion only."""
import copy
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.eval.bace_frozen_gnn_contracts import atomic_json, sha256_file
from src.utils.terminal_resource_dependency import verify_terminal_dependency
from src.baselines.bace_globalgce_aplus_owner import training_command


class TerminalResourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.proc=self.root/'proc'
        boot=self.proc/'sys/kernel/random/boot_id'; boot.parent.mkdir(parents=True); boot.write_text('boot')
        self.lock=self.root/'gpu.lock'; self.lock.write_text('old metadata')
        self.terminal=self.root/'terminal.json'; atomic_json(self.terminal,{'status':'FAILED'})
        info=self.lock.stat()
        self.receipt={'schema':'t14_terminal_resource_release_v1','task_id':'t14',
            'science_dependency_for_global_aids':False,'future_stage':'WAITING_PARITY',
            'scientific_status':'FAILED','physical_lease_released':True,'boot_id':'boot',
            'terminal_path':str(self.terminal),'terminal_sha256':sha256_file(self.terminal),
            'retired_processes':[{'pid':999999,'start_ticks':42}],
            'lease_path':str(self.lock),'lease_device_inode':[info.st_dev,info.st_ino]}
        self.registry={'tasks':[{'task_id':'t14','owner_state':'BLOCKED',
            'stage':'FAILED_PARITY_WAITING_COMPONENT_EVIDENCE','owner_pid':None,'owner_start_ticks':None}],
            'gpu_leases':[{'task_id':'t14','state':'RELEASED','lease_path':str(self.lock)}]}

    def verify(self):
        path=self.root/'release.json'; atomic_json(path,self.receipt)
        return verify_terminal_dependency({'path':str(path),'sha256':sha256_file(path)},self.registry,self.proc)

    def test_failed_science_not_healthy_predecessor(self):
        old=self.terminal.read_bytes(); self.assertEqual(self.verify(),'t14')
        self.assertEqual(old,self.terminal.read_bytes())

    def test_new_stage_owner_rejected(self):
        self.registry['tasks'][0]['owner_pid']=123
        with self.assertRaisesRegex(ValueError,'NEW_OWNER'): self.verify()

    def test_future_reservation_rejected(self):
        self.registry['gpu_leases'][0]['state']='PREDEPLOYED'
        with self.assertRaisesRegex(ValueError,'NOT_RELEASED'): self.verify()

    def test_old_owner_still_alive_rejected(self):
        with patch('src.utils.terminal_resource_dependency.process_start_ticks',return_value=42):
            with self.assertRaisesRegex(ValueError,'STILL_LIVE'): self.verify()

    def test_lock_competing_process_blocks(self):
        command=[sys.executable,'-c',
            'import fcntl,sys; f=open(sys.argv[1],"r+"); fcntl.flock(f,fcntl.LOCK_EX); print("locked",flush=True); sys.stdin.read()',str(self.lock)]
        child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(),'locked')
            with self.assertRaises(BlockingIOError): self.verify()
        finally:
            child.communicate('')
        self.assertEqual(self.verify(),'t14')

    def test_lock_replacement_rejected(self):
        self.lock.rename(self.root/'preserved.lock'); self.lock.write_text('new')
        with self.assertRaisesRegex(ValueError,'IDENTITY_CHANGED'): self.verify()

    def test_terminal_changed_rejected(self):
        atomic_json(self.terminal,{'status':'PASS'})
        with self.assertRaisesRegex(ValueError,'TERMINAL_CHANGED'): self.verify()

    def test_scientific_dependency_not_bypassed(self):
        self.receipt['science_dependency_for_global_aids']=True
        with self.assertRaisesRegex(ValueError,'SCOPE_INVALID'): self.verify()

    def test_original_science_entrypoint_resumes_same_root(self):
        entry=self.root/'science.py'; entry.write_text('# original science\n')
        spec={'science_entrypoint':str(entry),'science_entrypoint_sha256':sha256_file(entry),
              'runtime_config':'/old/config','training_contract':'/old/contract'}
        cfg={'formal_output_root':'/same/formal','rematerialization_root':'/same/inputs'}
        cmd=training_command(spec,cfg,'formal',resume=True)
        self.assertIn(str(entry),cmd); self.assertIn('--resume',cmd)
        self.assertEqual(cmd[cmd.index('--output-root')+1],'/same/formal')
        entry.write_text('# modified')
        with self.assertRaisesRegex(ValueError,'BINDING_FAILED'): training_command(spec,cfg,'formal',resume=True)


if __name__=='__main__': unittest.main()
