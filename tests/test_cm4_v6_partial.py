import importlib.util
from pathlib import Path
import numpy as np
import pytest
s=importlib.util.spec_from_file_location('v6_replot',Path(__file__).resolve().parents[1]/'scripts/replot_cm4_v6_partial.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_prefix_not_full_k_or_capped():
    r=m.reduce_columns([[.12,.03],[.04,.02],[np.inf,np.inf]],[1,1,0],['first','second'],.034)
    assert r[0]['covered']==1 and r[1]['covered']==2
    assert r[0]['true_prefix_sha']!=r[1]['true_prefix_sha']
    assert r[1]['new_covered']==1 and r[1]['changed_best_distance']==2
    assert r[19]['effective_k']==2

def test_real_saturation_still_improves_cost():
    r=m.reduce_columns([[.08,.02],[.09,.03]],[1,1],['a','b'],.034)
    assert r[0]['covered']==r[-1]['covered']==2
    assert r[0]['capped_mean']>r[1]['capped_mean']

def test_unknown_and_non_source_rejected():
    for d,mask in [([[np.nan]],[1]),([[.01]],[0])]:
        with pytest.raises(ValueError):m.reduce_columns(d,mask,['a'],.034)
