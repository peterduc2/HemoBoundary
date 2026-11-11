from pathlib import Path
import numpy as np
import pytest
import shutil
from hb_data import DataStore,cache_path,save_npz,assert_parity,check_prediction_rows,read_rows,pred_path
from fixtures import create_project

@pytest.fixture(scope='module')
def original_fixture(tmp_path_factory):
    return create_project(tmp_path_factory.mktemp('original')/'Project')

@pytest.fixture
def ds(tmp_path,original_fixture):
    shutil.copytree(original_fixture.parent,tmp_path/'copied')
    root=tmp_path/'copied'/'Project';out=tmp_path/'output';pids=('VP001','VP002','VP003','VP004')
    store=DataStore(root,out,(1,2),('SHIN','BALINT','KHAN'),pids);store.preflight();return store

@pytest.mark.parametrize('seed',(1,2))
@pytest.mark.parametrize('split',('train','test'))
def test_replay_and_serialization(ds,seed,split):
    bank=ds.shin(seed,'VP001',split);assert bank.H.shape==(45 if split=='train' else 36,3,51)
    assert bank.audit['historical_feature_parity']
    again=ds.shin(seed,'VP001',split);np.testing.assert_array_equal(again.H,bank.H)
    assert np.std(bank.H[:,:,:10])>0 # never zero padding

def test_different_seeds(ds):
    a=ds.shin(1,'VP001','test');b=ds.shin(2,'VP001','test');assert not np.array_equal(a.H,b.H)

def test_original_bank_mismatch_aborts(ds):
    p=cache_path(ds.root,1,'VP001')
    with np.load(p) as z:d={k:np.array(z[k]) for k in z.files}
    d['tr_state_feat'][0,0,0]+=1;save_npz(p,**d)
    with pytest.raises(ValueError,match='replay mismatch'):ds.shin(1,'VP001','train')

@pytest.mark.parametrize('dataset,pid',[('BALINT','C01'),('KHAN','S02')])
def test_external_fixed_bank(ds,dataset,pid):
    bank=ds.external(dataset,pid);assert bank.H.shape==(36,3,51)
    assert bank.audit['kind'].startswith('ORIGINAL_EXTERNAL')
    assert bank.audit['historical_feature_parity'] is False
    for seed in (1,2):pred,gate,sha=ds.saved(bank,seed);assert pred.shape==(36,3)

def test_labels_not_predictors(ds):
    b=ds.shin(1,'VP001','test');assert set(b.predictors())=={'H','R','features'}
    assert not {'onset','states','censored','clean'}&set(b.predictors())

def test_prediction_identity(ds):
    b=ds.shin(1,'VP001','test');rows=read_rows(pred_path(ds.root,1,'VP001'));rows[0]['truth_peak']='888'
    with pytest.raises(ValueError):check_prediction_rows(rows,'VP001',b.truth,1,'SHIN')

def test_replay_cache_corruption_detected(ds):
    b=ds.shin(1,'VP001','train');p=ds.replay_cache/'SHIN/seed_01/VP001_train.npz'
    with np.load(p) as z:d={k:np.array(z[k]) for k in z.files}
    d['H'][0,0,0]+=500;save_npz(p,**d)
    corrected=ds.shin(1,'VP001','train');np.testing.assert_array_equal(corrected.H,b.H)
