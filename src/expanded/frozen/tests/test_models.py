import numpy as np
import pytest
from pathlib import Path
import original_data as od
import hb_models as hm
from hb_data import Bank,make_trials,digest_json
from hb_algorithms import EstimateCache
from hb_models import (group_folds,stack_source,project_boundaries,fit_family,fit_control,
                       threshold_from_source,fit_aide,select_heads)
from hb_benchmark import benchmark_group,output_method_names,check_completed
from hb_evaluation import BOUNDARIES


def banks():
    result={}
    for i,pid in enumerate(('VP001','VP002','VP003','VP004')):
        profile=od.NoiseProfile(2,np.ones(2),np.ones(2)*.4,.6,.5)
        H,R,vals=make_trials(profile,('MODEL_FIXTURE',pid),12,6)
        result[pid]=Bank('SHIN',pid,1,'train',H,R,vals['state_feat'],{k:vals[k] for k in ('present','onset','peak','offset','censored')},digest_json(pid),{'fixture':True})
    return result

@pytest.fixture(scope='module')
def source():return banks()

def test_disjoint_group_folds(source):
    pids,t,g=stack_source(source)
    for tr,va in group_folds(g,42):assert not set(g[tr])&set(g[va])
    with pytest.raises(ValueError):stack_source(source,'VP001')

def test_geometry_no_artificial_offset():
    p=project_boundaries([[2,4,24],[1,3,2]])
    assert np.isnan(p[0,2]) and p[1,2]>p[1,1]

def test_detection_threshold_uses_supplied_sources_only():
    score=np.array([2,3,-2,-3,3,4,-3,-4.]);y=np.array([1,1,0,0,1,1,0,0]);g=np.repeat(['A','B'],4)
    th,ba=threshold_from_source(score,y,g);assert ba==1 and -2<th<=2

def test_ridge_readout_target_labels_do_not_enter(source,tmp_path):
    cache=EstimateCache(tmp_path,{'fixture':True});src={p:b for p,b in source.items() if p!='VP001'}
    model=fit_family('FIR_SMOOTH',src,cache,'VP001',42)
    before=model.predict(source['VP001'],cache)
    changed={k:v.copy() for k,v in source['VP001'].truth.items()};changed['onset'][:]=987
    original=source['VP001'].truth;source['VP001'].truth=changed
    after=model.predict(source['VP001'],cache);source['VP001'].truth=original
    for key in before:np.testing.assert_array_equal(before[key]['boundaries'],after[key]['boundaries'])
    assert model.audit['source_participants']==['VP002','VP003','VP004']
    assert all(r['source_group_cv_loss']>=0 for r in model.audit['readout_candidates'])

def test_control_and_own_detector(source,tmp_path,monkeypatch):
    monkeypatch.setattr(hm,'TREE_COUNT',8)
    src={p:b for p,b in source.items() if p!='VP001'}
    for method in ('RIDGE9_CONTROL','EXTRATREES9_CONTROL'):
        model=fit_control(method,src,'VP001',7);pred=model.predict(source['VP001'])[method]
        assert pred['boundaries'].shape==(18,3) and pred['decision'].shape==(18,)

def test_estimator_checkpoint_integrity(source,tmp_path):
    cache=EstimateCache(tmp_path,{'fixture':True});a,s,m=cache.get(source['VP001'],'FIR_SMOOTH','lambda1')
    cache2=EstimateCache(tmp_path,{'fixture':True});b,t,m2=cache2.get(source['VP001'],'FIR_SMOOTH','lambda1')
    np.testing.assert_array_equal(a,b);assert m['key']==m2['key']

def test_pipeline_group_and_resume(source,tmp_path,monkeypatch):
    monkeypatch.setattr(hm,'TREE_COUNT',8)
    cache=EstimateCache(tmp_path,{'fixture':True});src={p:b for p,b in source.items() if p!='VP001'};bank=source['VP001']
    ref=np.column_stack([bank.truth[k] for k in ('onset','peak','offset')]);gate=bank.truth['present']
    target=[(bank,ref,gate,'fixture_hb_sha')]
    rows=benchmark_group(src,target,1,'TEST_LOSO_VP001','VP001',cache,tmp_path,'fixture_protocol',families=('FIR_SMOOTH',))
    assert set(r['method'] for r in rows)==output_method_names(('FIR_SMOOTH',))
    for r in rows:
        if r['method'].startswith('MAHEUX'):assert r['supported']=='onset'
    repeat=benchmark_group(src,target,1,'TEST_LOSO_VP001','VP001',cache,tmp_path,'fixture_protocol',families=('FIR_SMOOTH',))
    assert len(repeat)==len(rows)
    cp=tmp_path/'checkpoints/SHIN/seed_01/VP001'
    import json
    m=json.loads((cp/'CHECKPOINT.json').read_text());(cp/'metrics.csv').write_text('bad')
    assert check_completed(cp,m['key'],output_method_names(('FIR_SMOOTH',))) is None
