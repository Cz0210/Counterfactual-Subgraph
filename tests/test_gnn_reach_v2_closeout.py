import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.ablations.gnn import reach_v2_closeout as close
from src.ablations.gnn.reach_v2_adapter import BACKBONES, SCOPE_NAME
from src.eval.bace_frozen_gnn_contracts import atomic_csv, atomic_json, read_json, sha256_file, stable_sha256


def science(pid='test-common', model='gine-weight', ids=None):
    ids=ids or [f'r{i}' for i in range(20)]
    pairs,matches=[],[]
    for rid in ids:
        values=[]
        if rid=='r0':
            for index,distance in enumerate((.3,.1)):
                row=dict(parent_id=pid,candidate_id=rid,match_index=index,match_atom_indices=[index],
                    oracle_checkpoint_hash=model,delete_valid=True,residual_smiles='CC',
                    pred_before=1,pred_after=0,p_before=[.2,.8],p_after=[.9,.1],
                    cf_flip=True,teacher_strict_flip=True,cf_drop=.8-.1,
                    distance_ok=True,wnode_distance=distance)
                values.append(row); matches.append(row)
        best=values[-1] if values else None
        pairs.append(dict(parent_id=pid,candidate_id=rid,oracle_checkpoint_hash=model,
            applicable=bool(values),num_matches=len(values),num_valid_residuals=len(values),
            num_strict_flip_matches=len(values),pair_strict_flip=bool(values),
            best_match_index=best['match_index'] if best else None,
            best_match_atom_indices=best['match_atom_indices'] if best else [],
            residual_smiles=best['residual_smiles'] if best else None,
            pred_after=best['pred_after'] if best else None,
            cf_drop=best['cf_drop'] if best else None,
            wnode_distance=best['wnode_distance'] if best else None))
    return dict(pair_rows=pairs,match_rows=matches)


def fixture(base, *, empty_common=False):
    from src.eval.mutagenicity_wnode_selector import derive_thresholds
    root=base/'new';root.mkdir()
    candidates=[dict(candidate_id=f'r{i}',canonical_fragment='C') for i in range(67)]
    pool=base/'pool.jsonl'
    pool.write_text(''.join(__import__('json').dumps(r)+'\n' for r in candidates))
    pool_sha=sha256_file(pool);spec_sha='spec'; selectors={}
    for n in BACKBONES:
        for m in ('native','common'):
            order=[f'r{i}' for i in range(20)]
            if n=='gin':order=list(reversed(order))
            row=dict(backbone=n,cohort_mode=m,pool_sha256=pool_sha,test_loaded=False,
                global_selector_after_complete_merge=True,ordered_rule_ids=order,
                prefixes={str(k):order[:k] for k in range(1,21)},calibration_parent_ids=['cal-common','cal-'+n])
            row['self_sha256']=stable_sha256(row);selectors[n+'/'+m]=row
    frozen=dict(scope=SCOPE_NAME,spec_sha256=spec_sha,pool_sha256=pool_sha,test_loaded=False,
        selectors=selectors,own_match_minimum_replayed=True,calibration_files={})
    atomic_json(root/'CALIBRATION_FREEZE.json',frozen)
    atomic_json(root/'version_scope.json',dict(scope=SCOPE_NAME,spec_sha256=spec_sha,pool_sha256=pool_sha,main_matrix_write=False))
    test_index=base/'test_index.json'
    index=dict(split='test',new_test_freeze_sha256=sha256_file(root/'CALIBRATION_FREEZE.json'))
    index['self_sha256']=stable_sha256(index);atomic_json(test_index,index)
    thresholds=derive_thresholds(np.array([.1,.2,.3])).to_dict()
    atomic_json(base/'thresholds.json',thresholds)
    model_files={n:{'model.pt':n+'-weight','temperature_scaling.json':n+'-temp','feature_schema.json':'features'} for n in BACKBONES}
    spec=dict(output_root=str(root),candidate_universe=str(pool),global_selector_freeze=str(root/'CALIBRATION_FREEZE.json'),
        raw_cost_indexes={'test':{'path':str(test_index)}},slots={'test':2},chunk_size=1,
        model_files=model_files,thresholds=str(base/'thresholds.json'),thresholds_sha256=sha256_file(base/'thresholds.json'),
        execution_commit='a'*40,package_root=str(base/'package'))
    close.seal_test_dependencies(spec,spec_sha,pool_sha)
    for n in BACKBONES:
        native=sorted(['test-'+n+'-a','test-'+n+'-b'] if empty_common else ['test-common','test-'+n])
        for index,pid in enumerate(native):
            directory=root/n/'test'/f'{index:04d}'
            data=science(pid,model_files[n]['model.pt'])
            atomic_json(directory/'parents'/f'{pid}.json',dict(scope=SCOPE_NAME,backbone=n,
                spec_sha256=spec_sha,pool_sha256=pool_sha,science=data,science_sha256=stable_sha256(data)))
            atomic_json(directory/'terminal.json',dict(state='PARENT_CHUNK_COMPLETE_NOT_CORE_PASS',scope=SCOPE_NAME,
                spec_sha256=spec_sha,pool_sha256=pool_sha,backbone=n,split='test',index=index,
                parent_ids=[pid],native_cohort_ids=native,pair_count=20,global_selector_called=False,
                main_matrix_write=False,model_files=model_files[n],global_freeze_sha256=sha256_file(root/'CALIBRATION_FREEZE.json')))
    old=base/'old';old.mkdir();bundle=base/'bundle';bundle.mkdir()
    atomic_json(bundle/'bundle_manifest.json',dict(feature_schema_path='schema',files={'schema':{'sha256':'features'}}))
    atomic_csv(old/'table.csv',[dict(backbone=n,parameter_count=100,num_examples=238,NLL=.2) for n in BACKBONES])
    atomic_json(old/'final.json',dict(files={'gnn_seed7_classifier_table.csv':sha256_file(old/'table.csv')}))
    proof=dict(state='PASS',source_final_audit_sha256=sha256_file(old/'final.json'),
        cohort_contract={'bundle_manifest_sha256':sha256_file(bundle/'bundle_manifest.json')},
        models={n:dict(model_sha256=n+'-weight',temperature_sha256=n+'-temp') for n in BACKBONES})
    atomic_json(old/'proof.json',proof)
    atomic_json(old/'acceptance.json',dict(state='GNN_CORE_SEED7_CORRECTED_PASS',main_matrix_write=False,
        independent_science_replay_sha256=sha256_file(old/'proof.json')))
    spec['bundle_root']=str(bundle)
    spec['classifier_adoption']=dict(root=str(old),**{key:dict(relative_path=leaf,sha256=sha256_file(old/leaf))
        for key,leaf in [('acceptance','acceptance.json'),('independent_science_replay','proof.json'),
                         ('final_audit','final.json'),('classifier_table','table.csv')]})
    return spec,spec_sha,pool_sha,candidates


class CloseoutTests(unittest.TestCase):
    def test_full_pool_probe_and_selected_test_chunks_keep_separate_sizes(self):
        from src.ablations.gnn.reach_v2_adapter import split_chunk_size
        spec=dict(chunk_size=1,chunk_size_by_split=dict(calibration=1,test=16))
        self.assertEqual(split_chunk_size(spec,'calibration'),1)
        self.assertEqual(split_chunk_size(spec,'test'),16)
        spec['chunk_size_by_split']['test']=33
        with self.assertRaises(ValueError):split_chunk_size(spec,'test')
    def test_own_match_minimum_and_funnel(self):
        data=science()
        self.assertEqual(close.verify_own_match_minima(data,candidate_ids=[f'r{i}' for i in range(20)],model_sha='gine-weight'),2)
        data['pair_rows'][0]['best_match_index']=0
        with self.assertRaisesRegex(ValueError,'MINIMUM'):close.verify_own_match_minima(data,candidate_ids=[f'r{i}' for i in range(20)],model_sha='gine-weight')

    def test_other_backbone_flip_cannot_be_adopted(self):
        with self.assertRaisesRegex(ValueError,'BACKBONE'):close.verify_own_match_minima(science(),candidate_ids=[f'r{i}' for i in range(20)],model_sha='gin-weight')

    def test_raw_distance_gap_is_not_zero_recourse(self):
        data=science();data['match_rows'][0]['distance_ok']=False
        with self.assertRaisesRegex(ValueError,'DISTANCE_GAP'):close.verify_own_match_minima(data,candidate_ids=[f'r{i}' for i in range(20)],model_sha='gine-weight')

    def test_nonflip_mask_cannot_be_disguised(self):
        data=science();data['match_rows'][0]['pred_after']=1
        with self.assertRaisesRegex(ValueError,'STRICT_FLIP'):close.verify_own_match_minima(data,candidate_ids=[f'r{i}' for i in range(20)],model_sha='gine-weight')

    def test_test_dependencies_require_all_ten_real_freezes(self):
        with tempfile.TemporaryDirectory() as d:
            spec,sha,pool,_=fixture(Path(d));root=Path(spec['output_root'])
            frozen=read_json(root/'CALIBRATION_FREEZE.json');frozen['selectors'].pop('gin/common')
            atomic_json(root/'CALIBRATION_FREEZE.json',frozen)
            with self.assertRaises(ValueError):close.seal_test_dependencies(spec,sha,pool)

    def test_original_spec_need_not_change_after_freeze(self):
        with tempfile.TemporaryDirectory() as d:
            spec,sha,pool,_=fixture(Path(d));original=stable_sha256(spec)
            self.assertEqual(close.seal_test_dependencies(spec,sha,pool)['spec_sha256'],sha)
            self.assertEqual(stable_sha256(spec),original)
            self.assertNotIn('sha256',spec['raw_cost_indexes']['test'])

    def test_test_chunks_complete_disjoint(self):
        with tempfile.TemporaryDirectory() as d:
            spec,sha,pool,candidates=fixture(Path(d))
            complete,common,_,_=close.collect_test_chunks(spec,spec_sha=sha,pool_sha=pool,candidates=candidates)
            self.assertEqual(common,['test-common']);self.assertEqual(sum(len(x['parent_ids']) for x in complete.values()),10)

    def test_missing_chunk_not_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            spec,sha,pool,candidates=fixture(Path(d));spec['slots']['test']=3
            with self.assertRaises(FileNotFoundError):close.collect_test_chunks(spec,spec_sha=sha,pool_sha=pool,candidates=candidates)

    def test_wrong_or_repeated_parent_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            spec,sha,pool,candidates=fixture(Path(d));path=Path(spec['output_root'])/'gine/test/0001/terminal.json'
            row=read_json(path);row['parent_ids']=['test-common'];atomic_json(path,row)
            with self.assertRaisesRegex(ValueError,'PARTITION'):close.collect_test_chunks(spec,spec_sha=sha,pool_sha=pool,candidates=candidates)

    def test_raw_index_dictionary_is_not_chunk_index(self):
        path=Path(__file__).parents[1]/'scripts/hpc/gnn/run_bace_gnn_reach_v2_chunk.py'
        module_spec=importlib.util.spec_from_file_location('reach_chunk_test',path)
        module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
        self.assertEqual(module.chunk_terminal(index=3,state='complete')['index'],3)
        for bad in ({'split':'test'},True,-1):
            with self.assertRaisesRegex(ValueError,'INDEX'):module.chunk_terminal(index=bad)

    def test_only_matching_corrected_classifier_report_adopted(self):
        with tempfile.TemporaryDirectory() as d:
            spec,sha,pool,_=fixture(Path(d));root=Path(spec['output_root'])
            rows=close.adopt_classifier_metrics(spec,output=root,frozen=read_json(root/'CALIBRATION_FREEZE.json'))
            self.assertEqual(len(rows),5)
            self.assertFalse(read_json(root/'classifier_adoption_receipt.json')['explanation_metrics_adopted'])
            spec['model_files']['gin']['temperature_scaling.json']='wrong'
            with self.assertRaisesRegex(ValueError,'TEMPERATURE'):close.adopt_classifier_metrics(spec,output=root,frozen=read_json(root/'CALIBRATION_FREEZE.json'))

    def test_full_numeric_closeout_and_package_without_model_or_ot(self):
        with tempfile.TemporaryDirectory() as d:
            spec,sha,pool,candidates=fixture(Path(d));root=Path(spec['output_root'])
            # Chemistry is an explicit synthetic fixture; numerical prefix code
            # is the real evaluator. No production chemistry path is replaced.
            with patch('src.eval.mutagenicity_wnode_selector.build_candidate_chemistry',
                       side_effect=lambda rows,**kw:SimpleNamespace(structural_similarity=np.eye(len(rows)))):
                result=close.finish_test(spec,spec_sha=sha,pool_sha=pool,candidates=candidates)
            self.assertEqual(result['state'],'GNN_REACH_V2_CORE_PASS')
            self.assertEqual(result['completed_parent_units'],10)
            self.assertFalse(result['old_66_explanation_adopted'])
            native=read_json(root/'gine/native/explanation_metrics.json')
            gin=read_json(root/'gin/native/explanation_metrics.json')
            self.assertEqual(native['CCRCov@10'],1.)
            self.assertEqual(gin['CCRCov@10'],0.)
            self.assertEqual(gin['CCRCov@20'],1.)
            packaged=close.package_closeout(spec)
            self.assertTrue(Path(packaged['path']).is_file())
            self.assertFalse(read_json(Path(spec['package_root'])/'package_manifest.json')['classifier_weights_included'])

    def test_empty_common_is_na_not_zero(self):
        with tempfile.TemporaryDirectory() as d:
            spec,sha,pool,candidates=fixture(Path(d),empty_common=True);root=Path(spec['output_root'])
            with patch('src.eval.mutagenicity_wnode_selector.build_candidate_chemistry',
                       side_effect=lambda rows,**kw:SimpleNamespace(structural_similarity=np.eye(len(rows)))):
                close.finish_test(spec,spec_sha=sha,pool_sha=pool,candidates=candidates)
            result=read_json(root/'gin/common/explanation_metrics.json')
            self.assertEqual(result['state'],'VALID_EMPTY_COHORT');self.assertIsNone(result['CCRCov@20'])

    def test_independent_replay_rejects_rehashed_wrong_k10(self):
        with tempfile.TemporaryDirectory() as d:
            spec,sha,pool,candidates=fixture(Path(d));root=Path(spec['output_root'])
            with patch('src.eval.mutagenicity_wnode_selector.build_candidate_chemistry',
                       side_effect=lambda rows,**kw:SimpleNamespace(structural_similarity=np.eye(len(rows)))):
                close.finish_test(spec,spec_sha=sha,pool_sha=pool,candidates=candidates)
            rel='gin/native/explanation_metrics.json';row=read_json(root/rel)
            row['prefix_rows'][9]['ccrcov_theta_star']=1.;atomic_json(root/rel,row)
            audit=read_json(root/close.AUDIT_NAME);audit['files'][rel]=sha256_file(root/rel)
            atomic_json(root/close.AUDIT_NAME,audit)
            with self.assertRaisesRegex(ValueError,'INDEPENDENT_PREFIX'):
                close.verify_closeout(root)

    def test_old_audit_cannot_be_renamed_v2(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);atomic_json(root/close.AUDIT_NAME,dict(state='PASS',candidate_count=66))
            with self.assertRaisesRegex(ValueError,'NEW_CORE'):close.verify_closeout(root)


if __name__=='__main__':unittest.main()
