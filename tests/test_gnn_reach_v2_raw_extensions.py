import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.ablations.gnn import reach_v2_raw_extensions as ext
from src.ablations.gnn.reach_v2_adapter import BACKBONES,SCOPE_NAME
from src.eval.bace_frozen_gnn_contracts import atomic_json,read_json,sha256_file,stable_sha256

fixture_module=importlib.util.spec_from_file_location('closeout_fixture',Path(__file__).with_name('test_gnn_reach_v2_closeout.py'))
fixture=importlib.util.module_from_spec(fixture_module);fixture_module.loader.exec_module(fixture)

def prepare(base):
    root=base/'result';root.mkdir();ids=[f'r{i}' for i in range(20)]
    index=dict(schema=ext.SCHEMA,split='calibration',graph_costs={},source_parent_units=0,
        source_finite_match_records=0,raw_contract_sha256='contract',kernel_identity={},source_spec={})
    index['self_sha256']=stable_sha256(index);atomic_json(base/'index.json',index)
    spec=dict(output_root=str(root),raw_cost_indexes={'calibration':{'path':str(base/'index.json'),
        'sha256':sha256_file(base/'index.json')}},slots={'calibration':1},chunk_size=1,
        model_files={n:{'model.pt':n+'-weight'} for n in BACKBONES})
    directory=root/'gine/calibration/0000';data=fixture.science('cal-parent','gine-weight',ids)
    for i,row in enumerate(data['match_rows']):
        row.update(parent_smiles='CCO',residual_smiles=('CC','CO')[i],sanitize_ok=True,
            residual_connected=True,action_semantics_version='connected_sanitized_residual_v1')
    data['pair_rows'][0]['residual_smiles']='CO'
    atomic_json(directory/'parents/p.json',dict(scope=SCOPE_NAME,backbone='gine',spec_sha256='spec',
        pool_sha256='pool',science=data,science_sha256=stable_sha256(data)))
    atomic_json(directory/'terminal.json',dict(state='PARENT_CHUNK_COMPLETE_NOT_CORE_PASS',scope=SCOPE_NAME,
        spec_sha256='spec',pool_sha256='pool',backbone='gine',split='calibration',index=0,
        model_files=spec['model_files']['gine'],global_selector_called=False,main_matrix_write=False,
        native_cohort_ids=['cal-parent'],parent_ids=['cal-parent'],pair_count=20))
    return spec,[{'candidate_id':i} for i in ids]

def fake_key(parent,residual,contract):return residual,parent,residual

class ExtensionTests(unittest.TestCase):
    def test_complete_family_exports_raw_costs_without_masks(self):
        with tempfile.TemporaryDirectory() as d,patch.object(ext,'graph_key',fake_key):
            spec,candidates=prepare(Path(d));before=sha256_file(spec['raw_cost_indexes']['calibration']['path'])
            result=ext.extend_calibration_costs(spec,spec_sha='spec',pool_sha='pool',candidates=candidates,backbone='gine')
            self.assertEqual(result['index']['raw_cost_count'],2);self.assertEqual(result['ot_recomputed'],0)
            self.assertFalse(result['source_flip_masks_reused']);self.assertEqual(before,sha256_file(spec['raw_cost_indexes']['calibration']['path']))
            path=Path(spec['output_root'])/'raw_extensions/gine.json'
            resolved,_=ext.resolve_calibration_index(spec,'spec','pool','gin',path)
            self.assertEqual(resolved['graph_costs']['CO']['distance'],.1)
            with self.assertRaisesRegex(ValueError,'BINDING'):ext.resolve_calibration_index(spec,'spec','pool','gcn',path)

    def test_missing_parent_or_chunk_not_exported(self):
        with tempfile.TemporaryDirectory() as d:
            spec,candidates=prepare(Path(d));spec['slots']['calibration']=2
            with patch.object(ext,'graph_key',fake_key),self.assertRaises(FileNotFoundError):
                ext.extend_calibration_costs(spec,spec_sha='spec',pool_sha='pool',candidates=candidates,backbone='gine')
            self.assertFalse((Path(spec['output_root'])/'raw_extensions/gine.json').exists())

    def test_wrong_minimum_does_not_enter_raw_adoption(self):
        with tempfile.TemporaryDirectory() as d:
            spec,candidates=prepare(Path(d));path=Path(spec['output_root'])/'gine/calibration/0000/parents/p.json'
            row=read_json(path);row['science']['pair_rows'][0]['best_match_index']=0
            row['science_sha256']=stable_sha256(row['science']);atomic_json(path,row)
            with self.assertRaisesRegex(ValueError,'MINIMUM'):
                ext.extend_calibration_costs(spec,spec_sha='spec',pool_sha='pool',candidates=candidates,backbone='gine')

    def test_conflicting_existing_graph_cost_rejected(self):
        with tempfile.TemporaryDirectory() as d,patch.object(ext,'graph_key',fake_key):
            spec,candidates=prepare(Path(d));path=Path(spec['raw_cost_indexes']['calibration']['path'])
            row=read_json(path);row['graph_costs']['CC']=dict(parent='CCO',residual='CC',distance=.7,source_records=[])
            row['self_sha256']=stable_sha256({k:v for k,v in row.items() if k!='self_sha256'});atomic_json(path,row)
            spec['raw_cost_indexes']['calibration']['sha256']=sha256_file(path)
            with self.assertRaisesRegex(ValueError,'DIFFERENT_GRAPH_COST'):
                ext.extend_calibration_costs(spec,spec_sha='spec',pool_sha='pool',candidates=candidates,backbone='gine')

if __name__=='__main__':unittest.main()
