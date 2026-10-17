import numpy as np
import pytest
from scipy import signal,linalg
import original_data as od
from hb_data import make_trials
from hb_algorithms import (aide_scores,aide_design,ar_fit,robust_beta,ar_irls_beta,my_filter,
                           fir_curves,native_bounds,curve_features,spm_kernel,canonical_design)

@pytest.fixture
def signals():
    p=od.NoiseProfile(3,np.ones(3),np.ones(3)*.4,.6,.5)
    return make_trials(p,('test_algorithms',),12,6)[:2]

def test_aide_ols_equations(signals):
    H,R=signals;score,events=aide_scores(H,R);D,timing=aide_design()
    alpha=H.std(2)/R.std(2);y=.5*(H-alpha[:,:,None]*R);y=y[:,:,::2]
    values=[]
    for reg in D:
        x=np.column_stack([np.ones(len(reg)),reg]);ts=[]
        for row in y[0]:
            beta=np.linalg.lstsq(x,row,rcond=None)[0];res=row-x@beta
            se=np.sqrt(np.sum(res*res)/(len(reg)-2)*np.linalg.inv(x.T@x)[1,1]);ts.append(beta[1]/se)
        values.append(np.median(ts))
    np.testing.assert_allclose(score[0],max(values),rtol=1e-9)
    np.testing.assert_array_equal(events[0],timing[np.argmax(values)])

def test_filter_initial_condition():
    a=np.array([1.,-.4]);x=np.arange(12.).reshape(6,2)+4
    expected=signal.lfilter(a,[1],x-x[0],axis=0)+a.sum()*x[0]
    np.testing.assert_allclose(my_filter(a,x),expected)

def test_ar_prefix_bic_matches_independent_fits():
    rng=np.random.default_rng(8);y=signal.lfilter([1],[1,-.6],rng.normal(size=51));n=len(y);L=[]
    for v in (y,y[::-1]):
        z=np.zeros((n,8))
        for j in range(1,9):z[j:,j-1]=v[:-j]
        L.append(z)
    X=np.column_stack([np.ones(2*n),np.vstack(L)]);yy=np.r_[y,y[::-1]];candidates=[]
    for k in range(1,10):
        b=np.linalg.lstsq(X[:,:k],yy,rcond=None)[0];rss=np.mean((yy-X[:,:k]@b)**2)
        bic=2*n*(np.log(2*np.pi*rss)+1)+k*np.log(2*n);candidates.append((bic,b))
    expected=min(candidates,key=lambda z:z[0])[1]
    np.testing.assert_allclose(ar_fit(y,8),expected,atol=1e-10,rtol=1e-8)

def test_robust_coefficients_recover_line_with_outlier():
    x=np.linspace(-1,1,100);X=np.c_[np.ones(100),x];y=2+3*x;y[5]+=30
    b,w=robust_beta(X,y);np.testing.assert_allclose(b,[2,3],atol=1e-6)
    assert w[5]<.01

def test_arirls_finite_on_noisy_response():
    rng=np.random.default_rng(9);_,X,_=canonical_design('impulse')[4];beta=np.array([1,.2,.1,.05,-.1])
    y=X@beta+signal.lfilter([1],[1,-.5],rng.normal(0,.01,51));b,p,w=ar_irls_beta(X,y)
    assert b.shape==(5,) and np.isfinite(b).all() and 0<=p<=8

def test_dispersion_is_not_second_time_derivative():
    _,h=spm_kernel();_,dd=spm_kernel('dispersion');temporal2=np.gradient(np.gradient(h))
    assert not np.allclose(dd,temporal2,atol=1e-7)

def test_fir_multi_channel_shapes(signals):
    H,R=signals;curves,scores,audit=fir_curves(H,R,1.)
    assert curves.shape==(len(H),41) and scores.shape==(len(H),) and np.isfinite(curves).all()

def test_linear_threshold_interpolation():
    t=np.arange(0,20.01,.5);y=np.maximum(0,1-abs(t-6)/4)
    b=native_bounds(y,.2);np.testing.assert_allclose(b,[2.8,6,9.2],atol=1e-10)

def test_missing_offset_keeps_onset_peak():
    t=np.arange(0,20.01,.5);y=np.where(t<=5,t/5,1-.01*(t-5))
    b=native_bounds(y,.1);assert np.isfinite(b[:2]).all() and np.isnan(b[2])

def test_curve_features_finite_even_failure():
    x=np.full((2,41),np.nan);x[1]=np.sin(np.arange(41)*.1)
    feat=curve_features(x);assert feat.shape[0]==2 and np.isfinite(feat).all()
