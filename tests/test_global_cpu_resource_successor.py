import unittest
from src.utils.global_cpu_resource_successor import (serial_chain_peak, assert_export_only,
    prepare_cpu_spec, assert_dynamic_config_only)


class ResourceSuccessorTests(unittest.TestCase):
    def test_serial_export_cpu_max_retains_waiter(self):
        self.assertEqual(serial_chain_peak(96, 128, 128), 136)
        self.assertEqual(20000+2*(8+32+4096+136), 28544)

    def test_unknown_peak_refused(self):
        with self.assertRaises(ValueError): serial_chain_peak(96, None, 128)

    def test_exact_serial_stage_required(self):
        assert_export_only({'cpu_successors': [{'opens_test': False, 'command': ['run', '--action', 'export']}]})
        for stages in ([], [{'opens_test': True, 'command': ['run','--action','export']}],
                       [{'opens_test': False, 'command': ['run','--action','evaluate']}], [{}, {}]):
            with self.assertRaises(ValueError): assert_export_only({'cpu_successors': stages})

    def test_resource_only_no_pool_or_test_change(self):
        old = {'main_matrix_write': False, 'cpu_resource_config': {'path': '/old'},
               'pool': {'path': '/same'}, 'test': {'never_read': True}, 'cpu_handoff_root': '/same/owner'}
        new = prepare_cpu_spec(old, resource_descriptor={'path': '/new'})
        self.assertEqual(old['cpu_resource_config']['path'], '/old')
        self.assertEqual(new['pool'], old['pool'])
        self.assertEqual(new['test'], old['test'])
        self.assertEqual(new['cpu_handoff_root'], old['cpu_handoff_root'])

    def test_no_main_registry_adoption(self):
        with self.assertRaises(ValueError): prepare_cpu_spec({'main_matrix_write': True}, resource_descriptor={})

    def test_aids_memory_bytes_unchanged(self):
        old = {'stage_file_policy': {'path': '/old'}, 'reserve': 384, 'bytes': 100}
        new = {**old, 'stage_file_policy': {'path': '/new'}}
        assert_dynamic_config_only(old, new)
        with self.assertRaises(ValueError): assert_dynamic_config_only(old, {**new, 'reserve': 0})

    def test_no_cpu_signal_or_spawn_interface(self):
        from pathlib import Path
        code = (Path(__file__).resolve().parents[1]/'scripts/autodl/prepare_global_cpu_resource_successor.py').read_text()
        self.assertNotIn('os.kill(', code)
        self.assertNotIn('subprocess.', code)
        self.assertNotIn('atomic_write_owner_registry(', code)
        self.assertIn('CPU_SPEC_SEALED_NOT_ACTIVATED', code)


if __name__ == '__main__': unittest.main()
