import importlib.util
from pathlib import Path
import pytest

SPEC=importlib.util.spec_from_file_location('gapview',Path(__file__).parents[1]/'scripts/render_main20_gap_view.py')
module=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(module)

def test_mut_conditional_cost_keeps_above_theta_values():
    rows=[dict(k=k,coverage=1/3) for k in range(1,21)]
    parents=[dict(k=k,parent_id=str(i),best_distance=value,strict_recourse_available=flag)
             for k in range(1,21) for i,value,flag in [(0,'.01','true'),(1,'.09','true'),(2,'inf','false')]]
    prefix,curve=module.adapt(rows,'Mutagenicity','CM-CReM',.02,.1,parents,'sealed-source')
    assert prefix[-1]['cost']==pytest.approx(.05)
    assert prefix[-1]['fixed_capped_mean_cost']==pytest.approx(.2/3)
    point=next(r for r in curve if r['k']==20 and r['distance']==.02)
    assert point['coverage']==prefix[-1]['coverage']

def test_raw_coverage_conflict_is_rejected():
    with pytest.raises(ValueError,match='saved coverage/raw conflict'):
        module.adapt([dict(k=1,coverage=1)],'BACE','CM-CReM',.01,.1,
            [dict(k=1,parent_id='a',best_distance='.02',strict_recourse_available='true')],'source')
