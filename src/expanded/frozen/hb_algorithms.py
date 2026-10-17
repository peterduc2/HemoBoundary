"""Literature-based estimators and explicit trial/boundary adapters.
No name here claims reproduction of the full experimental protocol of a paper.
The published GLM/search/HRF kernels and the NEW boundary readouts are separate.
"""
from __future__ import annotations
import contextlib, hashlib, importlib.metadata, inspect, io, json
from functools import lru_cache
from collections import OrderedDict
from pathlib import Path
import numpy as np
from scipy import linalg, signal, stats
from hb_data import save_npz, write_json, digest_json, digest_file

FULL_T=np.arange(-5.,20.01,.5)
T=FULL_T[FULL_T>=0]
POST=FULL_T>=0
FRACTIONS=(.02,.04,.06,.08,.10,.12,.14,.16,.18,.20,.25,.30,.40,.50)
CURVE_FAMILIES=('CANONICAL_TD_DD_OLS','CANONICAL_TD_DD_ARIRLS','FIR_SMOOTH','HRFUNC_AUTHOR')
VARIANTS={'CANONICAL_TD_DD_OLS':('impulse','block2','block5'),
          'CANONICAL_TD_DD_ARIRLS':('impulse','block2','block5'),
          'FIR_SMOOTH':('lambda0.1','lambda1','lambda10'),
          'HRFUNC_AUTHOR':('lambda0.001',)}
KERNEL_VERSION='HB_V8_MULTICHANNEL_KERNELS_1'
REGISTRY={
 'PINTI_AIDE_SEARCH':{'paper':'Pinti et al. (2017)','doi':'10.1016/j.neuroimage.2017.05.001','outputs':'response detection only',
    'implementation':'Python single-event GLM-search core; multichannel aggregation and source threshold are adaptations. NOT full iterative AIDE/peakfinder reproduction.', 'matlab_parity':False},
 'CANONICAL_TD_DD_OLS':{'paper':'Santosa et al. (2019)','doi':'10.1117/1.NPh.6.2.025009','outputs':'fitted hemodynamic waveform',
    'implementation':'SPM canonical + temporal + dispersion basis, OLS; shift/width search and trial boundary readout are adaptations. Not the complete AnalyzIR pipeline.', 'matlab_parity':False},
 'CANONICAL_TD_DD_ARIRLS':{'paper':'Barker et al. (2013); Santosa et al. (2019)','doi':'10.1117/1.NPh.6.2.025009','outputs':'fitted hemodynamic waveform',
    'implementation':'Coefficient-only Python implementation of ar_fit/stepwise/myFilter/Tukey IRLS. Covariance/t-statistics from AnalyzIR are NOT reproduced. Initial OLS shift selection and trial readout are adaptations.', 'matlab_parity':False},
 'FIR_SMOOTH':{'paper':'Santosa et al. (2019), FIR family','doi':'10.1117/1.NPh.6.2.025009','outputs':'event-related FIR response',
    'implementation':'Explicit event-lag design with second-difference Tikhonov penalty; smoothing and boundary readout are adaptations, not Santosa author code.', 'matlab_parity':False},
 'HRFUNC_AUTHOR':{'paper':'Schaedig et al. (2026)','doi':'10.1117/1.NPh.13.S1.S17801','outputs':'event-related HRF',
    'implementation':'Original hrfunc==1.3.1 montage.estimate_hrf call on replayed full multichannel trial; no synthetic padding. edge_expansion=0, duration=20.5; trial readout is an adaptation.',
    'python_author_package':True,'single_event_degeneracy':'One impulse gives X.T X=I. Scalar ridge changes amplitude only; this cannot be sold as a de-noising/continuous-recording HRfunc reproduction.'},
 'MAHEUX_PCT20':{'paper':'Maheux et al. (2017)','doi':'10.1111/psyp.12648','outputs':'onset latency only',
    'implementation':'Single-trial percentage-of-peak measure; no jackknife/inference procedure from the original paper. Native latency diagnostic, not exact complete method.'},
 'MAHEUX_PCT50':{'paper':'Maheux et al. (2017)','doi':'10.1111/psyp.12648','outputs':'onset latency only',
    'implementation':'As above, 50 percent; no recovery output invented.'},
 'RIDGE9_CONTROL':{'paper':None,'outputs':'supervised onset/peak/offset and detection','implementation':'Task-matched control using identical nine temporal input features; not labelled SOTA.'},
 'EXTRATREES9_CONTROL':{'paper':None,'outputs':'supervised onset/peak/offset and detection','implementation':'Task-matched Extra Trees control using identical nine input features; not labelled SOTA.'}
}


for _name,_entry in REGISTRY.items():
    _entry['reporting_role']=('related_method_diagnostic' if _name=='HRFUNC_AUTHOR' else 'same_information_control' if _name.endswith('CONTROL') else 'literature_method_family_with_documented_adaptations')


def spm_kernel(kind='canonical',dt=.5,oversampling=20,length=40.0):
    """SPM/Nilearn gamma-difference convention, including parameter derivatives.
    Dispersion derivative is NOT the second temporal derivative.
    """
    step=dt/oversampling
    tt=np.linspace(0,length,int(round(length/step)))
    def base(onset=0.,dispersion=1.):
        u=tt-onset
        h=stats.gamma.pdf(u,6/dispersion,loc=step,scale=dispersion)-.167*stats.gamma.pdf(u,16.,loc=step,scale=1.)
        return h/h.sum()
    h=base()
    if kind=='temporal':h=(h-base(onset=.1))/.1
    if kind=='dispersion':h=(h-base(dispersion=1.01))/.01
    return tt,h

@lru_cache(maxsize=None)
def canonical_design(variant):
    width={'impulse':0.,'block2':2.,'block5':5.}[variant]
    shifts=np.arange(-4.,6.01,1.)
    models=[]
    for lag in shifts:
        cols=[]
        for kind in ('canonical','temporal','dispersion'):
            high,h=spm_kernel(kind)
            if width>0:
                box=(high<width).astype(float)
                h=np.convolve(h,box)[:len(h)]
            col=np.interp(FULL_T-lag,high,h,left=0,right=0)
            cols.append(col)
        B=np.column_stack(cols)
        # Keep the same span but orthogonalize to improve numerical conditioning.
        for j in range(1,3):B[:,j]-=B[:,:j]@np.linalg.lstsq(B[:,:j],B[:,j],rcond=None)[0]
        norm=np.linalg.norm(B,axis=0)
        if np.any(norm<1e-10):continue
        B=B/norm
        nuisance=np.column_stack([np.ones(51),(FULL_T-FULL_T.mean())/np.ptp(FULL_T)])
        X=np.column_stack([B,nuisance])
        models.append((float(lag),X,np.linalg.pinv(X)))
    return models


def my_filter(ar,X):
    X=np.asarray(X,float);initial=X[0].copy()
    return signal.lfilter(ar,[1.],X-initial,axis=0)+ar.sum()*initial


def ar_fit(y,pmax=8):
    """Forward/backward, zero-padded lag design, prefix-model BIC.
    Matches the coefficient calculation described by public AnalyzIR ar_fit,
    lagmatrix and stepwise for finite, full-rank inputs. Reference output parity
    across MATLAB/Octave has NOT been asserted.
    """
    y=np.asarray(y,float);n=len(y);pmax=max(1,min(int(pmax),n-1))
    def lagged(a):
        L=np.zeros((n,pmax))
        for j in range(1,pmax+1):L[j:,j-1]=a[:-j]
        return L
    X=np.column_stack([np.ones(2*n),np.vstack([lagged(y),lagged(y[::-1])])]);yy=np.r_[y,y[::-1]]
    Q,R=np.linalg.qr(X,mode='reduced');invR=np.linalg.pinv(R)
    candidates=[]
    for k in range(1,X.shape[1]+1):
        beta=invR[:k,:k]@(Q[:,:k].T@yy)
        resid=yy-X[:,:k]@beta
        rss=max(float(np.mean(resid**2)),np.finfo(float).tiny)
        bic=(2*n)*(np.log(2*np.pi*rss)+1)+k*np.log(2*n)
        candidates.append((bic,k,beta))
    _,k,beta=min(candidates,key=lambda z:(z[0],z[1]));return beta


def robust_beta(X,y,tune=4.685,limit=50):
    """Tukey IRLS coefficients with leverage-adjusted MAD and rank correction.
    No inferential covariance or native t-statistic is claimed.
    """
    X=np.asarray(X,float);y=np.asarray(y,float);n,p=X.shape
    Q,R,perm=linalg.qr(X,mode='economic',pivoting=True)
    tol=abs(R[0,0])*max(n,p)*np.finfo(float).eps
    rank=int(np.sum(np.abs(np.diag(R))>tol))
    if rank==0:raise ValueError('Zero-rank robust design')
    active=perm[:rank];Xa=X[:,active]
    b=np.zeros(p);b[active]=linalg.solve_triangular(R[:rank,:rank],Q[:,:rank].T@y)
    lev=np.minimum(.9999,np.sum(Q[:,:rank]**2,axis=1));adj=1/np.sqrt(1-lev)
    tiny=1e-6*np.std(y,ddof=1);tiny=tiny if tiny>0 else 1.
    eps=np.sqrt(np.finfo(float).eps);weights=np.ones(n);wxrank=rank
    for _ in range(limit):
        r=(y-X@b)*adj
        scale=np.median(np.sort(np.abs(r))[max(1,wxrank)-1:])/.6745
        u=r/(max(scale,tiny)*tune);weights=np.where(np.abs(u)<1,(1-u*u)**2,0.)
        sw=np.sqrt(weights);old=b.copy()
        new,_,wxrank,_=np.linalg.lstsq(Xa*sw[:,None],y*sw,rcond=None)
        b[active]=new
        if not np.any(np.abs(b-old)>eps*np.maximum(np.abs(b),np.abs(old))):break
    return b,weights


def ar_irls_beta(X,y,pmax=8,tune=4.685):
    b=np.linalg.lstsq(X,y,rcond=None)[0];prev=np.full_like(b,1e6);order=0;weights=np.ones(len(y))
    for _ in range(10):
        denominator=np.linalg.norm(prev)
        if denominator and np.linalg.norm(b-prev)/denominator<=1e-2:break
        prev=b.copy();a=ar_fit(y-X@b,pmax=pmax);f=np.r_[1.,-a[1:]];order=len(a)-1
        b,weights=robust_beta(my_filter(f,X),my_filter(f,y),tune=tune)
    if not np.isfinite(b).all():raise ArithmeticError('AR-IRLS produced nonfinite coefficients')
    return b,order,weights


def _signals(H,R):
    # All channels made available. Sign-inverted HbR is a fixed orientation,
    # not a choice made by examining target reference boundaries.
    return np.concatenate([H,-R],axis=1)


def canonical_curves(H,R,variant,robust=False):
    data=_signals(H,R);n,c,nt=data.shape;curves=np.empty((n,41));scores=np.empty(n);fail=[];orders=[]
    models=canonical_design(variant)
    for i,Y in enumerate(data):
        Y=Y.T
        fits=[]
        for lag,X,Pi in models:
            b=Pi@Y;fit=X@b
            fits.append((float(np.mean((fit-Y)**2)),lag,X,b))
        _,lag,X,b=min(fits,key=lambda x:x[0])
        if robust:
            try:
                bb=[]
                for channel in range(c):
                    bet,order,_=ar_irls_beta(X,Y[:,channel]);bb.append(bet);orders.append(order)
                b=np.column_stack(bb)
            except (ArithmeticError,ValueError,np.linalg.LinAlgError) as exc:
                curves[i]=np.nan;scores[i]=np.nan;fail.append({'trial':i,'error':str(exc)});continue
        response=X[:,:3]@b[:3]
        curve=np.median(response,axis=1)
        curves[i]=curve[POST]
        residual=Y-X@b
        scale=np.sqrt(np.mean(residual**2))+1e-10
        scores[i]=max(0.,float(np.max(curve[POST])))/scale
    return curves,scores,{'failures':fail,'channel_ar_order_max':max(orders) if orders else None,'spatial_aggregation':'median of fitted HbO and sign-inverted HbR channel responses'}


def fir_curves(H,R,lam):
    """Explicit single-event FIR with identifiable nuisance and smoothness penalty.
    Baseline rows are observations from the original simulation, not zero padding.
    """
    E=np.zeros((51,41));E[POST]=np.eye(41)
    nuisance=np.column_stack([np.ones(51),(FULL_T-FULL_T.mean())/np.ptp(FULL_T)])
    X=np.column_stack([E,nuisance]);D=np.diff(np.eye(41),n=2,axis=0)
    P=np.zeros((43,43));P[:41,:41]=D.T@D
    A=X.T@X+float(lam)*P
    Pi=np.linalg.solve(A,X.T)
    Y=_signals(H,R)
    coef=np.einsum('pt,nct->ncp',Pi,Y)
    reconstructed=np.einsum('tp,ncp->nct',X,coef)
    response=coef[:,:,:41]
    curve=np.median(response,axis=1)
    resid=Y-reconstructed
    score=np.maximum(0,np.max(curve,axis=1))/(np.sqrt(np.mean(resid**2,axis=(1,2)))+1e-10)
    return curve,score,{'failures':[],'basis':'41 event-lag impulses with second-difference ridge'}

@lru_cache(maxsize=1)
def aide_design():
    tt=np.arange(-5.,21.,1.);hrft=np.arange(0.,41.)
    h=stats.gamma.pdf(hrft,6.)-stats.gamma.pdf(hrft,10.)/6.;h/=np.max(h)
    design=[];labels=[]
    for onset in range(0,20):
        for duration in range(1,21-onset):
            event=((tt>=onset)&(tt<onset+duration)).astype(float)
            reg=np.convolve(event,h)[:len(tt)];reg-=reg.mean();norm=np.linalg.norm(reg)
            if norm<=1e-12:continue
            design.append(reg/norm);labels.append((onset,duration))
    return np.asarray(design),np.asarray(labels)


def aide_scores(H,R):
    # CBSI algebra is applied separately to each paired channel; the original
    # paper's full event segmentation/overlap algorithm is NOT claimed here.
    alpha=np.std(H,axis=2)/np.maximum(np.std(R,axis=2),1e-12)
    corrected=.5*(H-alpha[:,:,None]*R)
    y=corrected[:,:,::2];y-=y.mean(axis=2,keepdims=True)
    design,timing=aide_design();coef=np.einsum('nct,kt->nck',y,design)
    rss=np.maximum(np.sum(y*y,axis=2)[:,:,None]-coef**2,1e-12)
    tstat=coef/np.sqrt(rss/(y.shape[2]-2))
    aggregate=np.median(tstat,axis=1);idx=np.argmax(aggregate,axis=1)
    return aggregate[np.arange(len(H)),idx],timing[idx]


def hrfunc_channel_estimates(H,R,lam=.001):
    """Unmodified AUTHOR PACKAGE on each complete generated epoch, no padding.
    Native output is standardized. Baseline origin is restored analytically from
    the actual pre-event samples before NEW temporal-boundary extraction.
    """
    import hrfunc, mne
    standardize_name=hrfunc.montage.estimate_hrf.__globals__.get('standardize_name')
    if not callable(standardize_name):raise RuntimeError('Cannot resolve the installed author channel-name normalizer')
    installed=importlib.metadata.version('hrfunc')
    if installed!='1.3.1':raise RuntimeError(f'Expected hrfunc 1.3.1, found {installed}; do not reuse this cache under another kernel')
    c=H.shape[0];data=np.concatenate([H,R],axis=0)
    if data.shape[1]!=51:raise ValueError('Need original 51-point epoch; padding is prohibited')
    sds=np.std(data,axis=1);means=data.mean(axis=1)
    if np.any(sds<=1e-12):raise ValueError('Constant channel is unsupported by native HRfunc normalization')
    names=[f'S{k+1}_D{k+1} hbo' for k in range(c)]+[f'S{k+1}_D{k+1} hbr' for k in range(c)]
    info=mne.create_info(names,2.,ch_types=['hbo']*c+['hbr']*c)
    # Coordinates are distinct channel identifiers only, NOT anatomy. No atlas,
    # localization or cross-channel spatial interpolation is used. Distinct IDs
    # also prevent older HRtree versions from aliasing HbO/HbR nodes.
    for k in range(2*c):info['chs'][k]['loc'][:3]=(.01+float(k)*.02,0.,0.)
    raw=mne.io.RawArray(data,info,verbose='ERROR');events=[0]*51;events[10]=1
    with contextlib.redirect_stdout(io.StringIO()):
        montage=hrfunc.montage(raw)
        montage.estimate_hrf(raw,events,duration=20.5,lmbda=float(lam),edge_expansion=0.,preprocess=False,timeout=30,source_id='original_simulated_trial')
    estimates=[]
    # standardize_name in 1.3.1 removes spaces; match uniquely, never use best-fit.
    def normalized(s):return str(standardize_name(str(s))).lower()
    for name in names:
        matches=[node for k,node in montage.channels.items() if normalized(k)==normalized(name)]
        if len(matches)!=1:raise RuntimeError(f'HRfunc channel mapping not unique for {name}: {list(montage.channels)}')
        est=getattr(matches[0],'estimates',None)
        if est is None or len(est)!=1:raise RuntimeError(f'HRfunc failed to return one estimate for {name}')
        a=np.asarray(est[0],float)
        if a.shape!=(41,) or not np.isfinite(a).all():raise RuntimeError(f'HRfunc returned {a.shape}, expected 41 supported points including 20s')
        estimates.append(a)
    est=np.stack(estimates)
    # The one-impulse normal matrix is identity. This algebraic audit detects
    # interface/time-axis changes rather than silently interpreting wrong arrays.
    expected=((data-means[:,None])/sds[:,None])[:,POST]/(1+lam)
    delta=float(np.max(np.abs(est-expected)))
    if not np.allclose(est,expected,rtol=2e-6,atol=2e-6):
        raise RuntimeError(f'HRfunc 1.3.1 algebra/time-axis conformance failed (max difference {delta}); results not accepted')
    baseline_std=(np.median(data[:,:10],axis=1)-means)/sds
    restored=(est-baseline_std[:,None]/(1+lam))*sds[:,None]
    curve=np.median(np.concatenate([restored[:c],-restored[c:]],axis=0),axis=0)
    return curve,{'native_algebra_max_error':delta,'single_impulse_XTX_is_identity':True,'zero_padding':False,'spatial_channels':c}


def hrfunc_curves(H,R):
    curves=np.full((len(H),41),np.nan);scores=np.full(len(H),np.nan);fail=[];errors=[]
    for i in range(len(H)):
        try:
            curve,audit=hrfunc_channel_estimates(H[i],R[i]);curves[i]=curve;errors.append(audit['native_algebra_max_error'])
            pre=.5*(np.median(H[i,:,:10],axis=0)-np.median(R[i,:,:10],axis=0))
            scores[i]=max(0,float(curve.max()))/(np.std(pre)+1e-10)
        except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:
            # API or algebra errors are not poor estimator accuracy. Abort.
            if isinstance(exc,RuntimeError):raise
            fail.append({'trial':i,'error':str(exc)})
    return curves,scores,{'failures':fail,'max_author_algebra_error':max(errors) if errors else None,'single_event_degeneracy':True}


def native_bounds(curve,onfrac,offfrac=None):
    """Observed-support threshold adapter with linear crossings and local peak interpolation.
    A missing offset does NOT erase otherwise valid onset/peak estimates.
    No extrapolated endpoint is assigned to unobserved recovery.
    """
    offfrac=onfrac if offfrac is None else offfrac
    y=np.asarray(curve,float)
    out=np.full(3,np.nan)
    if y.shape!=(41,) or not np.isfinite(y).all():return out
    k=int(np.argmax(y));amp=float(y[k])
    if amp<=1e-10 or k==0:return out
    # Endpoint peaks are unresolved, but onset may still be observed.
    left=np.flatnonzero(y[:k+1]>=onfrac*amp)
    if len(left):
        j=int(left[0]);out[0]=T[j]
        if j>0 and y[j]!=y[j-1]:out[0]=T[j-1]+(onfrac*amp-y[j-1])/(y[j]-y[j-1])*.5
    if k==40:return out
    peak=T[k]
    den=y[k-1]-2*y[k]+y[k+1]
    if den<-1e-12:peak+=float(np.clip(.25*(y[k-1]-y[k+1])/den,-.25,.25))
    out[1]=peak
    right=np.flatnonzero(y[k+1:]<=offfrac*amp)
    if len(right):
        j=int(k+1+right[0]);offset=T[j]
        if y[j]!=y[j-1]:offset=T[j-1]+(offfrac*amp-y[j-1])/(y[j]-y[j-1])*.5
        if offset>peak:out[2]=offset
    if np.isfinite(out[0]) and not out[0]<peak:out[0]=np.nan
    return out


def curve_features(curves):
    """Identical readout features for all fitted-waveform comparators.
    Only estimates are used; no clean waveform, true states, or target labels.
    """
    rows=[]
    for curve in curves:
        good=bool(np.isfinite(curve).all());x=np.nan_to_num(curve,nan=0.,posinf=0.,neginf=0.)
        amp=max(np.max(np.abs(x)),1e-8);z=x/amp
        landmarks=np.array([native_bounds(x,f) for f in (.02,.05,.1,.2,.5,.8,.95)])
        miss=~np.isfinite(landmarks);land=np.nan_to_num(landmarks,nan=0.)/20.
        rows.append(np.r_[z,np.gradient(z,.5),land.ravel(),miss.ravel().astype(float),np.log1p(amp),float(good),np.trapezoid(np.maximum(z,0),T)/20.])
    return np.asarray(rows,float)


class EstimateCache:
    def __init__(self,out,environment):
        self.out=Path(out);self.environment=environment
        self.code_sha=digest_file(Path(__file__))
        self.memory=OrderedDict()
    def get(self,bank,family,variant):
        key=digest_json({'version':KERNEL_VERSION,'code':self.code_sha,'bank':bank.signature,'family':family,'variant':variant,'environment':self.environment})
        if key in self.memory:
            self.memory.move_to_end(key);return self.memory[key]
        fp=self.out/'estimator_outputs'/bank.dataset/(f'seed_{bank.seed:02d}')/bank.participant/f'{bank.split}_{family}_{variant}.npz'
        meta=fp.with_suffix('.json')
        if fp.is_file() and meta.is_file():
            m=json.loads(meta.read_text())
            if m.get('key')==key and m.get('file_sha256')==digest_file(fp):
                with np.load(fp,allow_pickle=False) as z:result=(np.array(z['curves']),np.array(z['score']),m)
                self.memory[key]=result
                if len(self.memory)>512:self.memory.popitem(last=False)
                return result
        if family.startswith('CANONICAL_TD_DD'):
            curves,score,audit=canonical_curves(bank.H,bank.R,variant,robust=family.endswith('ARIRLS'))
        elif family=='FIR_SMOOTH':curves,score,audit=fir_curves(bank.H,bank.R,float(variant.removeprefix('lambda')))
        elif family=='HRFUNC_AUTHOR':curves,score,audit=hrfunc_curves(bank.H,bank.R)
        elif family=='PINTI_AIDE_SEARCH':
            score,events=aide_scores(bank.H,bank.R);curves=np.full((bank.n,41),np.nan)
            audit={'event_timing_is_not_hemodynamic_boundary':True,'candidate_event_timing':events.tolist(),'failures':[]}
        else:raise KeyError(family)
        if curves.shape!=(bank.n,41) or score.shape!=(bank.n,):raise ValueError('Estimator returned wrong array shape')
        audit.update({'key':key,'family':family,'variant':variant,'bank_signature':bank.signature})
        save_npz(fp,curves=curves,score=score);audit['file_sha256']=digest_file(fp);write_json(meta,audit)
        self.memory[key]=(curves,score,audit)
        if len(self.memory)>512:self.memory.popitem(last=False)
        return curves,score,audit
