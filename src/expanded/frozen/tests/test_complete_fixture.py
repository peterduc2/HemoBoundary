"""End-to-end 45-source/36-test fixture, not actual HemoBoundary data.
Original external preprocessing and native author APIs are independently audited.
"""
from hb_data import DataStore
from hb_algorithms import EstimateCache
from hb_benchmark import benchmark_group,output_method_names
from fixtures import create_project

def test_complete_saved_schema_default_models(tmp_path):
    root=create_project(tmp_path/'Project',seeds=(1,),external=False)
    pids=('VP001','VP002','VP003','VP004')
    store=DataStore(root,tmp_path/'out',(1,),('SHIN',),pids);store.preflight()
    training={p:store.shin(1,p,'train') for p in pids if p!='VP001'}
    target=store.shin(1,'VP001','test');hp,hg,hs=store.saved(target,1)
    cache=EstimateCache(tmp_path/'out',{'test_only':True})
    families=('CANONICAL_TD_DD_OLS','CANONICAL_TD_DD_ARIRLS','FIR_SMOOTH')
    rows=benchmark_group(training,[(target,hp,hg,hs)],1,'FULL_FIXTURE','VP001',cache,tmp_path/'out','fixture_protocol',families=families)
    assert set(r['method'] for r in rows)==output_method_names(families)
    assert all(r.get('onset_n_eligible',24)==24 for r in rows)
