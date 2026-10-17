#!/usr/bin/env python3
"""Cohen-Adad et al. (2007) shift-GLM, independently reconstructed for V8.

Original reference:
  Cohen-Adad et al. (2007), Medical Image Analysis 11, 616-629,
  doi:10.1016/j.media.2007.06.002, Eqs. (5)-(17), sections 2.2.2,
  2.2.3, 2.4, 3.2, 3.3, and Eqs. (18)-(23).

This code implements the Bayesian/least-squares drift-projected GLM,
protocol-protecting cosine drift selection, and a relative temporal
shift scan. A second canonical + first temporal derivative regression
is an internal comparator based on the paper's HRF derivative discussion.

CRITICAL LIMITATIONS: The published work studied continuous optical
recordings with long rest periods and repeated stimuli, and shifted
raw data relative to protocol regressors. Here we *scan shifted regressors*
within separate 51-sample V8 trials, use 8 candidate DCT drifts rather than
100 from 2000-sample recordings, select top-3 channels, and use added
source-calibrated onset/offset thresholds. These are V8 adaptations, NOT
an exact author-software reproduction or a published 4-boundary model.

Original frozen HemoBoundary V1/V2 remain READ ONLY. No p-values inferred.
"""


from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import traceback
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path('/content/gdrive/MyDrive/HonorProject/shin2018_word_generation_state_preparation_v1')
V8_V1 = ROOT / 'HB_V8_MULTICHANNEL_PUBLISHED_COMPARISON'
V8_V2 = ROOT / 'HB_V8_CPU_ONLY_SEEDS_06_TO_10'
FROZEN = Path('/content/HEMOBOUNDARY_V8_PACKAGE_a953339593ea')
FROZEN_ZIP = Path('/content/COHENADAD2007_FROZEN_V8_ARCHIVE_a953339593ea.zip')
OUT = ROOT / 'HB_COHENADAD2007_SHIFT_GLM_PAPER_GUIDED_770_V1'
FROZEN_ARCHIVE_HASH = 'a953339593ea228ef36aa618e5f2b0075c9153830b91c6daf50eef0e7e354d5d'
V8_PROTOCOL_HASH = '6190f376593bb9befc92427bf2efbf70410194215e3dd60baacf325fd3864d55'
SEEDS = tuple(range(1,11))
DATASETS = ('SHIN','BALINT','KHAN')
COUNTS = {'SHIN':26,'BALINT':26,'KHAN':25}
MAIN = 'COHENADAD2007_SHIFT_GLM_V8_ADAPTER'
ABLATION = 'COHENADAD2007_DERIVATIVE_GLM_V8_ADAPTER'
OUR_METHODS = (MAIN,ABLATION)
COMPARATORS = ('HEMOBOUNDARY_STRICT_SAVED','EXTRATREES9_CONTROL',
               'RIDGE9_CONTROL','CANONICAL_TD_DD_OLS__NATIVE_THRESH')
SCIENTIFIC_VERSION = 'COHENADAD2007_PAPER_GUIDED_V8_EPOCH_TWO_VARIANTS_20261015_V1'
N = 51
CUE = 10
TAPS = 41
DT = 0.5
FS = 2.0
DCT_CANDIDATES = 8            # V8 epoch adaptation, not the paper's 100 DCT terms
DCT_CORR_THRESHOLD = 0.15   # authors' reported parameter for DCT selection
SHIFT_GRID = tuple(np.round(np.arange(-3.,7.00001,.5),8).tolist())
TOP_K = 3                   # NEW multi-channel aggregation, not in the 2007 method
ONSET_OPTIONS = (0.1,0.2,0.3)
OFFSET_OPTIONS = (0.1,0.2,0.3)
THRESH_QUANTILES = tuple(np.linspace(.02,.98,33).tolist())
EXPECTED_V8_METHODS = 14


def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):
    d=hashlib.sha256()
    with Path(p).open('rb') as f:
        for blk in iter(lambda:f.read(1024*1024),b''):d.update(blk)
    return d.hexdigest()
def stable_json(o):return json.dumps(o,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha_obj(o):return sha_bytes(stable_json(o).encode())
def json_safe(o):
    """Record undefined denominators as JSON null, never nonstandard NaN/Infinity."""
    if isinstance(o,(np.floating,float)):
        f=float(o)
        return f if np.isfinite(f) else None
    if isinstance(o,(np.integer,int)) and not isinstance(o,bool):return int(o)
    if isinstance(o,np.ndarray):return json_safe(o.tolist())
    if isinstance(o,dict):return {str(k):json_safe(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)):return [json_safe(x) for x in o]
    if isinstance(o,(bool,str)) or o is None:return o
    return str(o)

def save_json(p,o):
    Path(p).write_text(json.dumps(json_safe(o),indent=2,sort_keys=True,allow_nan=False),encoding='utf-8')

def safe_float(v):
    try: return float(v)
    except (TypeError,ValueError): return float('nan')


def mount_drive():
    if not (ROOT.is_dir()):
        from google.colab import drive
        drive.mount('/content/gdrive')
    if not ROOT.is_dir():raise FileNotFoundError('Expected exact HemoBoundary project: '+str(ROOT))


def restore_original():
    """No V8 scientific code is patched, no benchmark job is launched."""
    if not FROZEN_ZIP.is_file():
        raise FileNotFoundError('Notebook must first place its embedded exact V8 zip at '+str(FROZEN_ZIP))
    if sha_file(FROZEN_ZIP)!=FROZEN_ARCHIVE_HASH:
        raise RuntimeError('Exact original V8 archive hash FAILED. No unpacking permitted')
    with zipfile.ZipFile(FROZEN_ZIP) as z:
        members=json.loads(z.read('PACKAGE_FILES_SHA256.json'))
        if FROZEN.exists():
            if not FROZEN.is_dir():raise RuntimeError('Frozen target path is not a directory')
        else:
            FROZEN.mkdir(parents=True,exist_ok=False)
            for item in z.infolist():
                name=item.filename
                if item.is_dir():continue
                pp=Path(name)
                if pp.is_absolute() or '..' in pp.parts or '\\' in name:
                    raise RuntimeError('Unsafe frozen ZIP member')
                dest=FROZEN/pp
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(z.read(name))
        for name,expected in members.items():
            f=FROZEN/name
            if not f.is_file() or sha_file(f)!=expected:
                raise RuntimeError('Frozen original V8 scientific file hash failed: '+name)
        extra={f.name for f in FROZEN.glob('*.py')}-{x for x in members if x.endswith('.py')}
        if extra:raise RuntimeError('Unexpected Python files in frozen V8 package: '+repr(extra))
    for p in (V8_V1,V8_V2):
        if not (p/'PROTOCOL.json').is_file():raise FileNotFoundError('Missing untouched original V8 protocol: '+str(p))
    p1=json.loads((V8_V1/'PROTOCOL.json').read_text())
    p2=json.loads((V8_V2/'PROTOCOL.json').read_text())
    if p1!=p2 or sha_obj(p1)!=V8_PROTOCOL_HASH:raise RuntimeError('Original V1/V2 protocol differs')
    for filename,expected in p1['code'].items():
        f=FROZEN/filename
        if not f.is_file() or sha_file(f)!=expected:
            raise RuntimeError('Frozen V8 scientific file differs from original protocol '+filename)
    if str(FROZEN) not in sys.path:sys.path.insert(0,str(FROZEN))
    import hb_benchmark as hb
    from hb_algorithms import CURVE_FAMILIES
    methods=hb.output_method_names(CURVE_FAMILIES)
    if len(methods)!=EXPECTED_V8_METHODS or not set(COMPARATORS).issubset(set(methods)):
        raise RuntimeError('Original method registry is not the frozen 14-comparator registry')
    print('PASS frozen V8 archive',FROZEN_ARCHIVE_HASH[:16], '| scientific code & V1/V2 protocol',flush=True)
    return p1


# The canonical double-gamma corresponds to the standard SPM form used in
# Cohen-Adad et al., Sec. 3.3 and Eq. 19-20. Parameters are explicitly
# predeclared; HbO is evaluated (HbR can be studied separately later).
from scipy.special import gammaln


def gamma_pdf(t, shape, scale=1.):
    """Causal gamma density with exact zeros at t<=0 (shape>1)."""
    z=np.asarray(t,dtype=float)
    o=np.zeros_like(z)
    valid=z>0
    if np.any(valid):
        w=z[valid]/scale
        o[valid]=np.exp((shape-1)*np.log(w)-w-gammaln(shape))/scale
    return o


def canonical_hrf(t):
    """SPM double-gamma with peak ~5 s, normalized on a fixed time support."""
    t=np.asarray(t,dtype=float)
    v=gamma_pdf(t,6.,1.)-gamma_pdf(t,16.,1.)/6.
    # Fixed global normalization, independent of temporal shifts.
    return v / CANONICAL_SCALE


CANONICAL_SCALE = float(np.max(
    gamma_pdf(np.arange(0.,31.,.01),6.,1.)
    -gamma_pdf(np.arange(0.,31.,.01),16.,1.)/6.
))


class CohenAdadEstimator:
    """Precomputed FWL drift projection + shift/derivative GLMs.

    Variant 1: the Cohen-Adad beta-extremum shift scan (task-epoch
    implementation, fixed nuisance basis).  For each HbO channel, scan
    beta for a shifted canonical model. Choose the largest *positive* beta,
    compute its positive t statistic, and pool the top 3 channels by t.

    Variant 0: the paper's alternative canonical+first temporal-derivative
    GLM, Eqs 18,22,23 (reduced 2-column version, no scale derivative).
    """
    def __init__(self):
        self.sample_times=(np.arange(N)-CUE)*DT
        self.rel_times=np.arange(TAPS)*DT
        self.shifts=np.asarray(SHIFT_GRID,float)
        self.template0=canonical_hrf(self.sample_times)
        self.derivative=np.gradient(self.template0,DT)
        self.B_shift,self.removed_shift=self.make_drift(self.template0)
        self.B_deriv,self.removed_deriv=self.make_drift(
            np.stack((self.template0,self.derivative)))
        self.J_shift=self.make_projection(self.B_shift)
        self.J_deriv=self.make_projection(self.B_deriv)
        self.shift_models=np.stack([
            canonical_hrf(self.sample_times-delta) for delta in self.shifts
        ],axis=0)
        self.shift_proj=self.shift_models@self.J_shift
        self.shift_norm2=np.einsum('ij,ij->i',self.shift_proj,self.shift_proj)
        if np.min(self.shift_norm2)<1e-8:
            raise RuntimeError('Unidentifiable shift regressor after nuisance projection')
        self.df_shift=N-self.B_shift.shape[1]-1
        if self.df_shift<=10:raise RuntimeError('Too many nuisance regressors for 51-sample epoch')
        self.Z=np.column_stack((self.template0,self.derivative))
        self.Zproj=self.J_deriv@self.Z
        gram=self.Zproj.T@self.Zproj
        if np.linalg.cond(gram)>1e10:raise RuntimeError('Unidentifiable derivative GLM')
        self.Ginv=np.linalg.inv(gram)
        self.deriv_df=N-self.B_deriv.shape[1]-2
        if self.deriv_df<=10:raise RuntimeError('Too many derivative GLM parameters')

    @staticmethod
    def make_projection(B):
        if B.ndim!=2 or not np.isfinite(B).all():raise ValueError('Bad drift matrix')
        Q,R=np.linalg.qr(B,mode='reduced')
        if np.min(np.abs(np.diag(R)))<1e-8:raise RuntimeError('Rank-deficient drift subspace')
        J=np.eye(N)-Q@Q.T
        if not np.allclose(J,J.T,atol=1e-11) or not np.allclose(J@J,J,atol=1e-10):
            raise AssertionError('Drift projection not symmetric/idempotent')
        return J

    @staticmethod
    def make_drift(protect):
        """Eq. 6/7 + Sec 2.4: DCT components correlated with HRF are removed.

        The intercept is retained, and 8 DCT candidates are compared against
        the unshifted HRF. The nuisance basis is held fixed during scanning.
        This fixed basis is an explicit short-epoch adaptation.
        """
        template=np.atleast_2d(protect).astype(float)
        cols=[np.ones(N,float)]
        removed=[]
        n=np.arange(N,dtype=float)
        protected=template-template.mean(axis=1,keepdims=True)
        pnorm=np.linalg.norm(protected,axis=1)
        for k in range(1,DCT_CANDIDATES+1):
            c=np.cos(np.pi*k*(n+.5)/N)
            c-=np.mean(c)
            den=np.linalg.norm(c)
            corr=np.max(np.abs(protected@c)/(np.maximum(pnorm,1e-12)*den))
            if corr>DCT_CORR_THRESHOLD:
                removed.append(k)
            else:
                cols.append(c/den)
        B=np.column_stack(cols)
        return B,tuple(removed)

    def fit_shift(self,X):
        """All-channel projected fits, choosing max-positive-beta shift per channel."""
        Y=X@self.J_shift
        yy=np.sum(Y*Y,axis=1)
        cross=Y@self.shift_proj.T   # channels x shifts
        beta=cross/self.shift_norm2[None,:]
        # Maximal positive beta is the paper's core relative shift criterion.
        index=np.argmax(beta,axis=1)
        rr=np.arange(X.shape[0])
        amplitude=beta[rr,index]
        nom=cross[rr,index]
        sse=np.maximum(yy-amplitude*nom, np.maximum(yy*1e-12,1e-28))
        t=amplitude*np.sqrt(self.df_shift*self.shift_norm2[index]/sse)
        waves=self.shift_models[index,CUE:]
        peaks=np.argmax(waves,axis=1)
        ok=np.isfinite(amplitude)&(amplitude>0)&np.isfinite(t)&(t>0)&(peaks>0)
        return t,waves,ok

    def fit_derivative(self,X):
        """Canonical+temporal derivative GLM in the projected drift subspace."""
        Y=X@self.J_deriv
        c=Y@self.Zproj
        coef=c@self.Ginv
        y2=np.sum(Y*Y,axis=1)
        explained=np.einsum('ij,ij->i',coef,c)
        remaining=np.maximum(y2-explained,np.maximum(y2*1e-12,1e-28))
        f=np.maximum(explained,0.)/2.*self.deriv_df/remaining
        statistic=np.sqrt(np.maximum(f,0.))
        wave=coef@self.Z.T
        waves=wave[:,CUE:]
        peaks=np.argmax(waves,axis=1)
        maxima=np.max(waves,axis=1)
        ok=(np.isfinite(statistic)&(statistic>0)&np.isfinite(maxima)&
            (maxima>0)&np.isfinite(coef[:,0])&(coef[:,0]>0)&(peaks>0))
        return statistic,waves,ok

    def estimate_epoch(self,H,variant=1):
        X=np.asarray(H,dtype=float)
        if X.ndim!=2 or X.shape[1]!=N:raise ValueError('Expected HbO channel x 51 bank')
        valid=np.isfinite(X).all(axis=1) & (np.std(np.where(np.isfinite(X),X,0.),axis=1)>1e-15)
        X=X[valid]
        if X.shape[0]<1:return float('-inf'),np.full(TAPS,np.nan),0
        if variant==1:t,profiles,ok=self.fit_shift(X)
        elif variant==0:t,profiles,ok=self.fit_derivative(X)
        else:raise ValueError('variant must be 1=shift or 0=derivative')
        candidates=np.flatnonzero(ok)
        if len(candidates)==0:return float('-inf'),np.full(TAPS,np.nan),int(X.shape[0])
        selected=sorted(candidates,key=lambda i:(-float(t[i]),int(i)))[:TOP_K]
        waves=profiles[selected]
        scale=np.max(waves,axis=1)
        if not np.isfinite(scale).all() or np.any(scale<=0):
            return float('-inf'),np.full(TAPS,np.nan),int(X.shape[0])
        norm=waves/scale[:,None]
        template=np.median(norm,axis=0)
        score=float(np.log1p(np.median(t[selected])))
        if not (np.isfinite(template).all() and np.isfinite(score)):
            return float('-inf'),np.full(TAPS,np.nan),int(X.shape[0])
        return score,template,int(X.shape[0])


def extract_boundaries(curve,onset_fraction,offset_fraction):
    h=np.asarray(curve,float)
    if h.shape!=(TAPS,) or not np.isfinite(h).all():return np.full(3,np.nan)
    peak_idx=int(np.argmax(h)); mx=float(h[peak_idx])
    if peak_idx<1 or mx<=1e-12:return np.full(3,np.nan)
    tpeak=peak_idx*DT
    # Last rising threshold crossing before peak (separates early noise peaks).
    level=onset_fraction*mx
    crossing=[]
    for j in range(peak_idx):
        if h[j]<=level and h[j+1]>level:
            crossing.append(j+(level-h[j])/(h[j+1]-h[j]))
    if not crossing:
        return np.array([np.nan,tpeak,np.nan])
    # crossing[-1] is in sample-index units, multiply by DT.
    onset=float(crossing[-1]*DT)
    offset=float('nan'); level=offset_fraction*mx
    for j in range(peak_idx,TAPS-1):
        if h[j]>=level and h[j+1]<level:
            v=(level-h[j])/(h[j+1]-h[j])
            offset=float((j+v)*DT)
            break
    if not np.isfinite(onset) or onset>=tpeak: onset=float('nan')
    if np.isfinite(offset) and offset<=tpeak:offset=float('nan')
    return np.array([onset,tpeak,offset],float)


def features_for_bank(bank,engine,variant):
    scores=np.empty(bank.n,float)
    waves=np.empty((bank.n,TAPS),float)
    nch=np.empty(bank.n,int)
    for i in range(bank.n):
        scores[i],waves[i],nch[i]=engine.estimate_epoch(bank.H[i],variant=variant)
    return scores,waves,nch

def thresholds_from_source(all_source_features,pids):
    # Fixed inverse (Eq20) is unsupervised; source labels calibrate only detector.
    scores=np.concatenate([all_source_features[p]['score'] for p in pids])
    finite=scores[np.isfinite(scores)]
    if len(finite)<10:raise RuntimeError('Insufficient source detection scores')
    candidates=np.unique(np.quantile(finite,THRESH_QUANTILES))
    candidates=np.unique(np.r_[float(finite.min())-1e-9,candidates,float(finite.max())+1e-9])
    best=(-float('inf'),None)
    for thr in candidates:
        per=[]
        for pid in pids:
            rec=all_source_features[pid]; y=rec['truth']['present']
            d=(rec['score']>=thr).astype(float)
            pos=(y==1);neg=(y==0)
            if not pos.any() or not neg.any():raise RuntimeError('Missing source positive/negative classes')
            per.append(.5*(float(np.mean(d[pos]))+float(np.mean(1.-d[neg]))))
        ba=float(np.mean(per))
        # For equal BA, prefer conservative HIGHER threshold.
        val=(ba,float(thr))
        if val>best:best=val
    return best[1],best[0]


def select_boundary_fractions(source,pids):
    from hb_evaluation import timing_metrics,BOUNDARIES
    options=[]
    for onset in ONSET_OPTIONS:
        for offset in OFFSET_OPTIONS:
            per=[]
            for pid in pids:
                rec=source[pid]
                cache=rec.setdefault('_candidate_boundaries',{})
                optkey=(onset,offset)
                if optkey not in cache:
                    cache[optkey]=np.array([extract_boundaries(x,onset,offset) for x in rec['wave']])
                pred=cache[optkey]
                m=timing_metrics(rec['truth'],pred,np.ones(len(pred)))
                four=[safe_float(m[k+'_loss_missing4_sec']) for k in BOUNDARIES]
                if not np.isfinite(four).all():raise RuntimeError('Source timing objective nonfinite')
                per.append(float(np.mean(four)))
            options.append((float(np.mean(per)),float(onset),float(offset)))
    return sorted(options)[0]


def run_tests():
    engine=CohenAdadEstimator()
    assert N==51 and CUE==10 and DT==.5 and TOP_K==3
    assert abs(engine.template0[CUE])<1e-15
    assert 4.5 <= engine.rel_times[np.argmax(engine.template0[CUE:])] <= 5.5
    assert not engine.removed_shift or all(1<=k<=DCT_CANDIDATES for k in engine.removed_shift)
    for proj in (engine.J_shift,engine.J_deriv):
        assert np.max(np.abs(proj@proj-proj))<1e-10
    # Verify Eq13 (FWL) against an independently evaluated full GLM least squares.
    rng=np.random.default_rng(4107)
    yy=rng.normal(size=N)
    j=10
    tpl=engine.shift_models[j]
    beta_fwl=(tpl@engine.J_shift@yy)/(tpl@engine.J_shift@tpl)
    beta_full=np.linalg.lstsq(np.column_stack([tpl,engine.B_shift]),yy,rcond=None)[0][0]
    assert np.isclose(beta_fwl,beta_full,rtol=1e-9,atol=1e-9)
    # Controlled known stimulus-relative shift, true peak ~6.5s after cue.
    true_shift=1.5
    signal=canonical_hrf(engine.sample_times-true_shift)
    X=np.stack([(1.+.08*i)*signal + .02*rng.normal(size=N) for i in range(12)])
    for variant in (1,0):
        score,wave,n=engine.estimate_epoch(X,variant)
        assert n==12 and np.isfinite(score) and np.isfinite(wave).all()
        predicted_peak=np.argmax(wave)*DT
        assert abs(predicted_peak-(5.+true_shift))<=2.0,(variant,predicted_peak)
        b=extract_boundaries(wave,.2,.1)
        assert len(b)==3 and np.isfinite(b[1])
    assert np.isnan(extract_boundaries(np.full(TAPS,np.nan),.2,.1)).all()
    assert np.isnan(extract_boundaries(np.zeros(TAPS),.2,.1)).all()
    rising=np.linspace(0,1,TAPS)
    assert np.isnan(extract_boundaries(rising,.2,.1)[2])
    print('PASS: canonical SPM gamma, Cohen-Adad Eq13 FWL, protocol-orthogonal DCT, '+
          'shift and derivative numerical tests, event timing, censoring',flush=True)
    return engine

def protocol_object(code_sha):
    return {'scientific_version':SCIENTIFIC_VERSION,'code_sha256':code_sha,
        'v8_protocol_sha256':V8_PROTOCOL_HASH,'v8_archive_sha256':FROZEN_ARCHIVE_HASH,
        'models':list(OUR_METHODS),'seeds':list(SEEDS),'counts':COUNTS,
        'paper':'Cohen-Adad et al (2007), Medical Image Analysis 11, 616-629',
        'implementation_class':'independent paper-guided V8 epoch adaptation, NOT author code',
        'canonical_hrf':'SPM-style double-gamma shapes 6,16 undershoot 1/6; normalized fixed maximum',
        'shift_grid_s':list(SHIFT_GRID),'sampling_hz':FS,'event_sample':CUE,
        'dct_candidate_components':DCT_CANDIDATES,'dct_protection_abs_correlation':DCT_CORR_THRESHOLD,
        'dct_basis':'intercept + DCT cosines, protect against unshifted HRF (and derivative in alt model)',
        'shift_criterion':'largest positive GLM coefficient across shifts, then t-ranked top-k HbO channels',
        'derivative_criterion':'canonical + first temporal derivative GLM, positive canonical beta and F-ranked top-k',
        'top_k_channels':TOP_K,'onset_options':ONSET_OPTIONS,'offset_options':OFFSET_OPTIONS,
        'source_training':'ALL available 25 Shin (LOSO) / 26 Shin (external), 45 trials each, NO heldout calibration',
        'source_threshold_calibration':'subject-balanced BA maximization on train (source only)',
        'source_fraction_calibration':'subject-balanced four-boundary 4s loss on train (source only)',
        'new_adapter_parts':['51-sample task epochs instead of continuous rest/task recordings',
            'shifted regressors in fixed-length epochs instead of shifting/truncating recordings',
            '8 DCT candidates instead of 100 from 2000-point recordings',
            'HbO-only channel-level GLM, top-k channel aggregation',
            'source-calibrated fractional boundaries and independent detection'],
        'bspc_claim':'Cohen-Adad-inspired shift GLM adapted to V8 trial timing, NOT published native four-boundary SOTA',
        'statistical_unit':'participant means after all source seeds; no inferential p-values claimed',
        'numpy_version':np.__version__,'original_v8_untouched':True}

def verify_checkpoint(cp,key,config_sha):
    marker=cp/'CHECKPOINT.json'
    if not marker.exists():
        if cp.exists():raise RuntimeError('Partial unverified checkpoint exists: '+str(cp))
        return None
    m=json.loads(marker.read_text())
    if m.get('complete') is not True or m.get('key')!=key or m.get('config_sha256')!=config_sha:
        raise RuntimeError('Incompatible checkpoint, do not overwrite: '+str(cp))
    for f,h in m['files'].items():
        if sha_file(cp/f)!=h:raise RuntimeError('Corrupted previously completed checkpoint '+str(cp/f))
    with np.load(cp/'PREDICTIONS.npz',allow_pickle=False) as z:
        if z['pred0'].shape!=(36,3) or z['pred1'].shape!=(36,3):raise RuntimeError('Checkpoint prediction shape invalid')
    return m


def save_checkpoint(cp,key,config_sha,bank,hb_sha,data,newmetrics,baseline):
    cp.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pending_',dir=cp.parent) as tmp:
        tmp=Path(tmp)
        np.savez_compressed(tmp/'PREDICTIONS.npz',
            pred0=data[MAIN]['pred'],pred1=data[ABLATION]['pred'],
            score0=data[MAIN]['score'],score1=data[ABLATION]['score'],
            gate0=data[MAIN]['gate'],gate1=data[ABLATION]['gate'])
        save_json(tmp/'SOURCE_CALIBRATION.json',{k:data[k]['audit'] for k in OUR_METHODS})
        save_json(tmp/'CASE_METRICS.json',{'dataset':bank.dataset,'participant':bank.participant,
                  'seed':int(data[MAIN]['seed']),'metrics':newmetrics,'comparison':baseline})
        audit={'bank_signature':bank.signature,'saved_hb_prediction_sha256':hb_sha,
               'case_identity':(bank.dataset,int(data[MAIN]['seed']),bank.participant),
               'no_target_truth_used_for_calibration':True,
               'external_ground_truth_only_in_scoring':True,
               'source_bank_signatures':{k:v for k,v in data[MAIN]['source_signatures'].items()},
               'method_class':'paper_guided_not_author_code',
               'method_label':'Cohen-Adad 2007 shift/derivative GLM V8 epoch adapter'}
        save_json(tmp/'INPUT_AUDIT.json',audit)
        files={f.name:sha_file(f) for f in tmp.iterdir() if f.is_file()}
        save_json(tmp/'CHECKPOINT.json',{'complete':True,'key':key,'config_sha256':config_sha,
                                       'files':files})
        if cp.exists():raise RuntimeError('Refusing overwrite '+str(cp))
        os.rename(tmp,cp)


def write_cached_npz(cp,key,arrays):
    """Single-file atomic source/target cache, with in-file signature."""
    cp.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='.npz_pending_',suffix='.npz',
                                     dir=cp.parent,delete=False) as temp:
        temp_path=Path(temp.name)
        np.savez_compressed(temp,key=np.asarray(key),**arrays)
    try:os.replace(temp_path,cp)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    save_json(cp.with_suffix('.meta.json'),
              {'key':key,'file_sha256':sha_file(cp)})


def audit_cached_npz(cp,key):
    """A crash between NPZ and sidecar can be recovered by verifying embedded key.

    This never overwrites a corrupt or scientifically incompatible cache.
    """
    meta=cp.with_suffix('.meta.json')
    if not cp.exists() and not meta.exists():return False
    if not cp.is_file():raise RuntimeError('Incomplete cache, missing npz '+str(cp))
    try:
        with np.load(cp,allow_pickle=False) as z:
            if str(z['key'])!=key:raise RuntimeError('Cache internal signature mismatch '+str(cp))
            for k in ('score','wave'):
                if k not in z:raise RuntimeError('Cache missing scientific array '+str(cp)+' '+k)
    except (ValueError,KeyError,OSError,zipfile.BadZipFile) as exc:
        raise RuntimeError('Corrupt cache, refusing reuse '+str(cp)) from exc
    if not meta.is_file():
        # Recover verified complete file after interrupted metadata write.
        save_json(meta,{'key':key,'file_sha256':sha_file(cp)})
    else:
        m=json.loads(meta.read_text())
        if m.get('key')!=key or m.get('file_sha256')!=sha_file(cp):
            raise RuntimeError('Wrong/damaged cache: '+str(cp))
    return True

def source_cache_path(seed,pid,variant):
    return OUT/'source_features'/f'seed_{seed:02d}'/f'{pid}_variant{variant}.npz'


def source_features(store,seed,pid,variant,engine,cfg):
    cp=source_cache_path(seed,pid,variant)
    bank=store.shin(seed,pid,'train')
    key=sha_obj({'config':cfg,'seed':seed,'participant':pid,
                 'signature':bank.signature,'variant':variant})
    if audit_cached_npz(cp,key):
        with np.load(cp,allow_pickle=False) as z:
            score=z['score'].astype(float);wave=z['wave'].astype(float);valid=z['valid'].astype(int)
    else:
        score,wave,valid=features_for_bank(bank,engine,variant)
        write_cached_npz(cp,key,{'score':score,'wave':wave,'valid':valid})
        print(f' SOURCE seed {seed:02d} {pid}: variant{variant} '+
              f'fits={bank.n}, curves={np.isfinite(wave).all(axis=1).sum()}',flush=True)
    if score.shape!=(bank.n,) or wave.shape!=(bank.n,TAPS):
        raise RuntimeError('Unexpected source feature cache shape')
    return {'score':score,'wave':wave,'valid':valid,'truth':bank.truth,'signature':bank.signature}


def target_features(bank,seed,variant,engine,cfg):
    if bank.dataset=='SHIN':folder=OUT/'heldout_features'/'SHIN'/f'seed_{seed:02d}'
    else:folder=OUT/'heldout_features'/bank.dataset
    cp=folder/f'{bank.participant}_variant{variant}.npz'
    key=sha_obj({'cfg':cfg,'signature':bank.signature,'variant':variant})
    if audit_cached_npz(cp,key):
        with np.load(cp,allow_pickle=False) as z:
            return z['score'].astype(float),z['wave'].astype(float)
    score,wave,valid=features_for_bank(bank,engine,variant)
    write_cached_npz(cp,key,{'score':score,'wave':wave,'valid':valid})
    return score,wave


def calibrate_source(seed,target_pid,source,variant,cfg):
    allowed=tuple(sorted(set(source)-({target_pid} if target_pid else set())))
    if len(allowed)!=(25 if target_pid else 26):
        raise RuntimeError('Target leakage/source participant missing')
    calname=(target_pid or 'ALL_EXTERNAL')+f'_variant{variant}.json'
    cp=OUT/'source_calibration'/f'seed_{seed:02d}'/calname
    sigs={p:source[p]['signature'] for p in allowed}
    key=sha_obj({'cfg':cfg,'seed':seed,'sources':sigs,'variant':variant})
    if cp.exists():
        v=json.loads(cp.read_text())
        if v.get('key')!=key:raise RuntimeError('Changed source calibration; cannot reuse '+str(cp))
        return v
    partial={p:source[p] for p in allowed}
    thr,ba=thresholds_from_source(partial,allowed)
    loss,on,off=select_boundary_fractions(partial,allowed)
    result={'key':key,'seed':seed,'excluded_target':target_pid,'source_pids':list(allowed),
        'source_signatures':sigs,'score_threshold':float(thr),'onset_fraction':float(on),
        'offset_fraction':float(off),'source_train_balanced_accuracy':float(ba),
        'source_train_timing_4s_loss':float(loss),'source_calibration_only':True,
        'heldout_labels_used_for_calibration':False}
    cp.parent.mkdir(parents=True,exist_ok=True);save_json(cp,result)
    print(f' CALIB seed {seed:02d} {target_pid or "EXTERNAL"} variant{variant}: '+
          f'BA={ba:.4f} loss={loss:.4f} threshold={thr:.4g} '+
          f'on={on:.2f} off={off:.2f}',flush=True)
    return result


def read_original_v8_checkpoint(seed,bank,hb_sha):
    import hb_benchmark as hb
    from hb_algorithms import CURVE_FAMILIES
    from hb_data import read_rows
    bdir=(V8_V1 if seed<=5 else V8_V2)
    cp=bdir/'checkpoints'/bank.dataset/f'seed_{seed:02d}'/bank.participant
    marker=cp/'CHECKPOINT.json'
    if not marker.is_file():raise FileNotFoundError('Original frozen V8 checkpoint not found '+str(marker))
    m=json.loads(marker.read_text())
    if m.get('complete')!=True or m.get('protocol_sha256')!=V8_PROTOCOL_HASH:
        raise RuntimeError('Unverified original V8 checkpoint '+str(cp))
    checked=hb.check_completed(cp,m['key'],hb.output_method_names(CURVE_FAMILIES))
    if checked is None:raise RuntimeError('Original V8 checkpoint failed original hash/audit validation '+str(cp))
    audit=json.loads((cp/'INPUT_AUDIT.json').read_text())
    if audit.get('bank_signature')!=bank.signature or audit.get('original_hb_sha256')!=hb_sha:
        raise RuntimeError('Original V8 checkpoint mismatches bank or saved HemoBoundary output')
    subset=[r for r in checked if r.get('method') in COMPARATORS]
    expected={('shared_gate_timing',x) for x in COMPARATORS}
    got={(r.get('mode'),r.get('method')) for r in subset}
    if not expected.issubset(got):raise RuntimeError('Missing frozen comparator timing rows')
    return subset


def evaluate_case(bank,seed,source_by_variant,engine,cfg):
    from hb_evaluation import timing_metrics,detection_metrics,common_valid_metrics
    hbpred,hbgate,hbsha=store_global.saved(bank,seed)
    base_metrics=read_original_v8_checkpoint(seed,bank,hbsha)
    all_data={};all_metric=[]
    for variant,method in ((1,MAIN),(0,ABLATION)):
        source=source_by_variant[variant]
        cal=calibrate_source(seed,bank.participant if bank.dataset=='SHIN' else None,
                             source,variant,cfg)
        score,wave=target_features(bank,seed,variant,engine,cfg)
        pred=np.array([extract_boundaries(h,cal['onset_fraction'],
                                          cal['offset_fraction']) for h in wave])
        gate=(score>=cal['score_threshold']).astype(float)
        identity={'dataset':bank.dataset,'participant':bank.participant,'seed':seed,
                  'method':method,'supported':'onset,peak,offset,duration'}
        dm=detection_metrics(bank.truth['present'],gate)
        sm=timing_metrics(bank.truth,pred,hbgate)
        sm.update(common_valid_metrics(bank.truth,pred,hbpred,hbgate))
        om=timing_metrics(bank.truth,pred,gate)
        for mode,metrics in (('independent_detection',dm),('shared_gate_timing',sm),('own_gate_timing',om)):
            all_metric.append({**identity,'mode':mode,**metrics})
        all_data[method]={'score':score,'wave':wave,'pred':pred,'gate':gate,
            'audit':cal,'seed':seed,'source_signatures':cal['source_signatures']}
    return all_data,all_metric,base_metrics,hbsha


def checkpoint_rows(cp):
    d=json.loads((cp/'CASE_METRICS.json').read_text())
    return d['metrics']+d['comparison']


def compile_summary(code_sha):
    rows=[];identities=[]
    cfg=sha_obj(protocol_object(code_sha))
    for seed in SEEDS:
        for ds in DATASETS:
            folder=OUT/'checkpoints'/ds/f'seed_{seed:02d}'
            for cp in sorted(folder.glob('*/CHECKPOINT.json')) if folder.exists() else []:
                loc=cp.parent;dataset=ds;pid=loc.name
                marker=json.loads(cp.read_text())
                key=marker['key']
                verify_checkpoint(loc,key,cfg)
                contents=json.loads((loc/'CASE_METRICS.json').read_text())
                if (contents['dataset'],contents['seed'],contents['participant'])!=(ds,seed,pid):
                    raise RuntimeError('Case identity mismatch')
                identities.append((ds,seed,pid))
                rows.extend(contents['metrics']+contents['comparison'])
    required={(ds,seed,f'VP{i:03d}') for seed in SEEDS for ds in ('SHIN',) for i in range(1,27)}
    required|={(ds,seed,f'C{i:02d}') for seed in SEEDS for ds in ('BALINT',) for i in range(1,27)}
    required|={(ds,seed,f'S{i:02d}') for seed in SEEDS for ds in ('KHAN',) for i in range(1,26)}
    if len(identities)!=len(set(identities)):
        raise RuntimeError('Duplicate checkpoint identity')
    fully=set(identities)==required
    # Every participant first average across available seeds. Do not treat seeds as biological subjects.
    subject=defaultdict(list)
    metrics=('balanced_accuracy','onset_loss_missing4_sec','peak_loss_missing4_sec',
             'offset_loss_missing4_sec','duration_loss_missing4_sec',
             'onset_mae_valid_sec','peak_mae_valid_sec','offset_mae_valid_sec',
             'duration_mae_valid_sec','onset_coverage','peak_coverage','offset_coverage',
             'duration_coverage')
    for r in rows:
        k=(r['dataset'],r['participant'],r['method'],r['mode'])
        subject[k].append(r)
    subjects=[]
    for (ds,pid,method,mode),rr in sorted(subject.items()):
        item={'dataset':ds,'participant':pid,'method':method,'mode':mode,'n_seeds':len(rr)}
        for m in metrics:
            vals=np.array([safe_float(x.get(m)) for x in rr],float)
            item[m]=float(np.mean(vals[np.isfinite(vals)])) if np.isfinite(vals).any() else None
        subjects.append(item)
    summ=defaultdict(list)
    for r in subjects:summ[(r['dataset'],r['method'],r['mode'])].append(r)
    descriptive=[]
    for (ds,method,mode),rr in sorted(summ.items()):
        item={'dataset':ds,'method':method,'mode':mode,'n_subjects':len(rr)}
        for m in metrics:
            vals=np.array([safe_float(x[m]) for x in rr],float)
            item[m]=float(np.mean(vals[np.isfinite(vals)])) if np.isfinite(vals).any() else None
        descriptive.append(item)
    out={'complete':fully,'n_verified_cases':len(identities),'expected_cases':770,
         'max_seen_seed':max((k[1] for k in identities),default=0),
         'study':'CohenAdad2007 paper-guided shift/derivative GLMs adapted to frozen V8 synthetic trial task; NOT published SOTA four-boundary method',
         'statistical_unit':'participant averaged over available independent source seeds',
         'p_values_computed':False,'original_v8_untouched':True,
         'results':descriptive}
    save_json(OUT/'PROGRESS_SUMMARY.json',out)
    if fully:save_json(OUT/'FULL_RUN_SUMMARY.json',out)
    if subjects:csv_write(OUT/'PARTICIPANT_SEED_AVERAGES.csv',subjects)
    if descriptive:csv_write(OUT/'DESCRIPTIVE_DATASET_METHOD_METRICS.csv',descriptive)
    return out


def csv_write(p,records):
    keys=list(dict.fromkeys(k for r in records for k in r))
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(records)


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('selftest','pilot','full','summary'),default='full')
    parser.add_argument('--max-new-cases',type=int,default=0,help='0 means unlimited')
    args=parser.parse_args(argv)
    if args.max_new_cases<0:raise ValueError('Max new cases must be >=0')
    engine=run_tests()
    if args.mode=='selftest':return
    mount_drive();restore_original()
    from hb_data import DataStore,PIDS
    import hb_benchmark
    code_sha=sha_file(Path(__file__))
    config=protocol_object(code_sha)
    cfg=sha_obj(config)
    OUT.mkdir(exist_ok=True,parents=True)
    lock=(OUT/'.writer.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as e:raise RuntimeError('Another Cohen-Adad runner is writing to this output') from e
    protocol_path=OUT/'PROTOCOL.json'
    if protocol_path.is_file():
        if sha_obj(json.loads(protocol_path.read_text()))!=cfg:
            raise RuntimeError('Output scientific configuration does not match; do not overwrite')
    else:
        existing=[x.name for x in OUT.iterdir() if x.name!='.writer.lock']
        if existing:raise RuntimeError('Unverified preexisting output without protocol: '+repr(existing))
        save_json(protocol_path,config)
    if args.mode=='summary':
        r=compile_summary(code_sha)
        print('SUMMARY verified cases',r['n_verified_cases'],'/770 complete=',r['complete'],flush=True)
        return
    global store_global
    new=0
    targets=SEEDS if args.mode=='full' else (1,)
    for seed in targets:
        print('\n'+'='*90+'\nCOHEN-ADAD 2007 SOURCE SEED '+str(seed)+'/10'+'\n'+'='*90,flush=True)
        store_global=DataStore(ROOT,OUT,(seed,),DATASETS,PIDS)
        store_global.preflight()
        source={variant:{pid:source_features(store_global,seed,pid,variant,engine,cfg)
                         for pid in PIDS} for variant in (1,0)}
        identities=[('SHIN',pid) for pid in PIDS]
        for ds in ('BALINT','KHAN'):
            ids=store_global.external_ids(ds)
            if len(ids)!=COUNTS[ds]:raise RuntimeError('External participant count changed: '+ds)
            identities.extend((ds,pid) for pid in ids)
        if len(identities)!=77:raise RuntimeError('Expected 77 evaluation participants per seed')
        if args.mode=='pilot':identities=[('SHIN','VP001'),('BALINT','C01'),('KHAN','S01')]
        for ds,pid in identities:
            bank=(store_global.shin(seed,pid,'test') if ds=='SHIN' else store_global.external(ds,pid))
            _,_,hb_sha=store_global.saved(bank,seed)
            source_signatures={variant:{p:source[variant][p]['signature']
                   for p in sorted(source[variant]) if not(ds=='SHIN' and p==pid)}
                   for variant in (1,0)}
            key=sha_obj({'cfg':cfg,'dataset':ds,'seed':seed,'pid':pid,
                          'bank_signature':bank.signature,'hb_sha':hb_sha,
                          'source_signatures':source_signatures})
            cp=OUT/'checkpoints'/ds/f'seed_{seed:02d}'/pid
            if verify_checkpoint(cp,key,cfg) is not None:
                print(f'REUSED SHA256-verified seed={seed:02d} {ds} {pid}',flush=True)
                continue
            data,metric,comparators,sha=evaluate_case(bank,seed,source,engine,cfg)
            save_checkpoint(cp,key,cfg,bank,sha,data,metric,comparators)
            new+=1
            displays=[]
            for method in OUR_METHODS:
                timing=next(x for x in metric if x['method']==method and x['mode']=='shared_gate_timing')
                det=next(x for x in metric if x['method']==method and x['mode']=='independent_detection')
                vals=[safe_float(timing[f'{k}_loss_missing4_sec'])
                      for k in ('onset','peak','offset','duration')]
                displays.append((method.split('_V8_')[0],round(det['balanced_accuracy'],4),
                                 np.round(vals,3).tolist()))
            print(f'COMPLETE new={new:04d} seed={seed:02d} {ds:6} {pid:5} | {displays}',flush=True)
            if args.max_new_cases and new>=args.max_new_cases:
                p=compile_summary(code_sha)
                print('SESSION LIMIT',new,'new; verified',p['n_verified_cases'],'/770',flush=True)
                return
        p=compile_summary(code_sha)
        print(f'SEED {seed:02d} COMPLETE | SHA256-verified cases {p["n_verified_cases"]}/770',flush=True)
    p=compile_summary(code_sha)
    print('FULL FINISH' if p['complete'] else 'PILOT FINISH',p['n_verified_cases'],'/770',flush=True)
    print('RESULTS:',OUT,flush=True)
    if p['complete']:
        print('Report as adapted GLM timing methods, NOT original published SOTA four-boundary outputs.',flush=True)



if __name__=='__main__':main()
