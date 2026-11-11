import numpy as np
import pytest
from hb_evaluation import (timing_metrics,detection_metrics,holm,common_valid_metrics,
                            paired_inference,summarize,HB)

def truth():
    return {'present':np.array([1,1,0]),'onset':np.array([1.,2.,np.nan]),'peak':np.array([4.,6.,np.nan]),
            'offset':np.array([12.,25.,np.nan]),'censored':np.array([0,1,0])}

def test_censoring_and_nonmissing_error_not_capped():
    tr=truth();p=np.array([[1.,4.,19.],[2.,6.,np.nan],[np.nan]*3]);m=timing_metrics(tr,p,[1,1,0])
    assert m['offset_n_eligible']==1 and m['offset_loss_missing4_sec']==7.
    assert m['onset_mae_valid_sec']==0

def test_failure_penalty_and_coverage_are_separate():
    tr=truth();p=np.array([[1.,4.,np.nan],[2.,6.,np.nan],[np.nan]*3]);m=timing_metrics(tr,p,[1,1,0])
    assert m['offset_loss_missing4_sec']==4 and m['offset_coverage']==0 and np.isnan(m['offset_mae_valid_sec'])

def test_shared_gate_does_not_claim_independent_detector():
    tr=truth();p=np.array([[1.,4.,12.],[2.,6.,np.nan],[np.nan]*3]);m=timing_metrics(tr,p,[1,0,0])
    assert m['onset_n_valid']==1 and m['onset_loss_missing4_sec']==2

def test_common_valid_subset():
    tr=truth();a=np.array([[1,4,np.nan],[2,6,np.nan],[np.nan]*3]);b=np.array([[1,4,12],[2,6,np.nan],[np.nan]*3])
    m=common_valid_metrics(tr,a,b,[1,1,0]);assert m['onset_common_n']==2 and m['offset_common_n']==0

def test_detection_failure_not_free_negative():
    m=detection_metrics([1,0],[1,np.nan]);assert m['balanced_accuracy']==.5 and m['specificity']==0

def test_holm_monotonic_reference():
    np.testing.assert_allclose(holm([.03,.001,.02]),[.04,.003,.04])
    with pytest.raises(ValueError):holm([.01,np.nan])

def test_paired_signflip_direction_and_floor():
    x=np.ones(26);s=paired_inference(x,8,reps=2000)
    assert s['mean_gain']==1 and s['raw_p']==1/2001 and s['n_positive']==26

def test_global_complete_family_and_participant_unit(tmp_path):
    expected=[];rows=[]
    for ds,n in [('SHIN',26),('BALINT',26),('KHAN',25)]:
        for seed in range(1,11):
            for p in range(n):
                pid=f'P{p:02d}';expected.append((ds,seed,pid))
                for meth,err in [(HB,.2),('METHOD_ADAPTED',1.)]:
                    rows.append({'dataset':ds,'seed':seed,'participant':pid,'method':meth,'mode':'shared_gate_timing','supported':'onset',
                                 'onset_loss_missing4_sec':err+.01*seed+.001*p})
    _,comp=summarize(rows,tmp_path,expected,range(1,11),pvalues=True)
    assert {r['n_participants'] for r in comp}=={25,26}
    assert all(r['family_size']==3 for r in comp)
    assert all(abs(r['mean_gain']-.8)<1e-12 for r in comp)
    with pytest.raises(ValueError):summarize(rows[:-1],tmp_path,expected,range(1,11),pvalues=False)
