#!/usr/bin/env python3
"""Lina et al. (2010) wavelet HRF inverse, *paper-guided* V8 trial adaptation.

Equation-matched mathematical core: Eqs 15-21, 26-27 and 30 of
'Wavelet-based estimation of the hemodynamic responses in diffuse optical imaging'
Medical Image Analysis 14 (2010), 606-616, doi:10.1016/j.media.2010.04.006.

IMPORTANT: not authors' MATLAB code. Original paper uses longer continuous
stimulus recordings; our V8 task is 51-sample isolated 2-Hz epochs and
4 trial-level boundaries. Padding, single-cue design, per-epoch PCA,
channel pooling, detector, crossing thresholds and LOSO scoring are NEW
explicit adaptations. Do not call it an exact original SOTA reproduction.

The model's frozen numerical core is never tuned against heldout references.
Outputs are confined to a dedicated NEW output directory, with source-aware
cache signatures, checkpoint hashes and read-only original V8 comparisons.
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
FROZEN_ZIP = Path('/content/LINA2010_FROZEN_V8_ARCHIVE_a953339593ea.zip')
OUT = ROOT / 'HB_LINA2010_WAVELET_HRF_PAPER_GUIDED_770_V1'
FROZEN_ARCHIVE_HASH = 'a953339593ea228ef36aa618e5f2b0075c9153830b91c6daf50eef0e7e354d5d'
V8_PROTOCOL_HASH = '6190f376593bb9befc92427bf2efbf70410194215e3dd60baacf325fd3864d55'
SEEDS = tuple(range(1,11))
DATASETS = ('SHIN','BALINT','KHAN')
COUNTS = {'SHIN':26,'BALINT':26,'KHAN':25}
MAIN = 'LINA2010_WAVELET_PCA1_V8_ADAPTER'
ABLATION = 'LINA2010_WAVELET_NOPCA_V8_ADAPTER'
OUR_METHODS = (MAIN,ABLATION)
COMPARATORS = ('HEMOBOUNDARY_STRICT_SAVED','EXTRATREES9_CONTROL',
               'HRFUNC_AUTHOR__NATIVE_THRESH','HRFUNC_AUTHOR__SOURCE_RIDGE_READOUT')
SCIENTIFIC_VERSION = 'LINA2010_PAPER_GUIDED_V8_EPOCH_TWO_VARIANTS_20261015_V1'
N = 51
CUE = 10
TAPS = 41
DT = 0.5
FS = 2.0
WAVELET_LEVEL = 3
PAD_LENGTH = 64
C_EXPONENT = 0.8  # paper's post-PCA HbO/HbR estimate; fixed for both ablation variants
ALPHA = 0.5       # Lina et al. Eq. 21
TOP_K = 3         # NEW fixed channel aggregation, NOT specified in paper
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


# Exact filter taps for Daubechies' orthogonal wavelet with 3 vanishing moments.
# Analysis lowpass is reversed relative to the scaling coefficients; the analysis
# highpass is its quadrature mirror. A matrix-orthogonality test validates them.
DB3_LO = np.array([.035226291882100656,-.08544127388224149,
                   -.13501102001039084,.45987750211849154,
                   .8068915093110928,.3326705529500826],float)
DB3_HI = DB3_LO[::-1]*np.power(-1.0,np.arange(6))


def db3_analysis(m):
    if m%2: raise ValueError('Dyadic periodic DB3 analysis requires an even length')
    low=np.zeros((m//2,m)); high=np.zeros_like(low)
    for k in range(m//2):
        for r in range(6):
            loc=(2*k+r)%m
            low[k,loc]+=DB3_LO[r]
            high[k,loc]+=DB3_HI[r]
    return low,high


def wavelet_detail_operator():
    # Linear *boundary-operator*, not an additional observed or predicted HRF.
    # Reflect known observations 51->64 and never evaluate an HRF beyond t=20.
    # This is an explicit deviation from the paper's long-recording DWT.
    P=np.zeros((PAD_LENGTH,N))
    for i in range(PAD_LENGTH):
        k=i if i<N else 2*(N-1)-i
        if not 0<=k<N:raise AssertionError('Reflection index invalid')
        P[i,k]=1.
    approx=np.eye(PAD_LENGTH)
    details=[]
    for j in range(1,WAVELET_LEVEL+1):
        lo,hi=db3_analysis(approx.shape[0]); details.append((j,hi@approx))
        approx=lo@approx
    full=np.vstack([approx,*[mat for _,mat in details[::-1]]])
    err=np.max(np.abs(full@full.T-np.eye(PAD_LENGTH)))
    if err>1e-8:raise AssertionError(f'DB3 wavelet transform is not orthogonal: {err}')
    # exclude coarsest approximation, as prescribed after Eq.17 in Lina 2010.
    W=np.concatenate([mat@P for _,mat in details],axis=0)
    scales=np.concatenate([np.full(mat.shape[0],j,dtype=int) for j,mat in details])
    assert W.shape==(56,N) and scales.shape==(56,)
    return W,scales


def r_smoothness():
    # Lina Eq19, including end diagonal=5, interior diagonal=6.
    K=TAPS;R=np.zeros((K,K),float)
    np.fill_diagonal(R,6.);R[0,0]=R[-1,-1]=5.
    for i in range(K-1):R[i,i+1]=R[i+1,i]=-4.
    for i in range(K-2):R[i,i+2]=R[i+2,i]=1.
    # Paper Eq.19 has explicit 1/(dT^4); preserve this scaling.
    R /= DT**4
    eig=np.linalg.eigvalsh(R)
    if eig.min()< -1e-8:raise AssertionError('Smoothness matrix is not PSD')
    return R


class WaveletEstimator:
    def __init__(self):
        self.W,scales=wavelet_detail_operator()
        self.scales=scales
        self.weight=2.0**(-C_EXPONENT*scales.astype(float)) # Eq15-16 inverse noise covariance
        X=np.zeros((N,TAPS),float)
        for k in range(TAPS):X[CUE+k,k]=1.0 # one observed task impulse at 0s
        self.X=X
        self.D=self.W@X
        self.A=(self.D.T*self.weight)@self.D
        self.R=r_smoothness()
        # Eq21 lambda = alpha * tr(Dt Rc^-1 D) / (DT^4 tr(Rh^-1)).
        self.lmbda=ALPHA*np.trace(self.A)/(DT**4*np.trace(self.R))
        self.G=self.A+self.lmbda*self.R
        eig=np.linalg.eigvalsh(self.G)
        if eig[0]<=0:raise RuntimeError('Wavelet regularized normal matrix not positive definite')
        self.H=np.linalg.solve(self.G,(self.D.T*self.weight)@self.W) # [41,51]
        self.df=max(1,self.W.shape[0]-np.linalg.matrix_rank(self.D))
        self.cue_index=CUE
        assert self.H.shape==(TAPS,N)
    def estimate_epoch(self,H,PCA=True):
        X=np.asarray(H,float)
        if X.ndim!=2 or X.shape[1]!=N:raise ValueError('Expected channel×51 HbO waveform')
        ok=np.isfinite(X).all(axis=1)&(np.std(np.where(np.isfinite(X),X,0.),axis=1)>1e-15)
        X=X[ok,:]
        if X.shape[0] < (2 if PCA else 1):
            return float('-inf'),np.full(TAPS,np.nan),int(ok.sum())
        if PCA:
            # Paper Eq26/27: one largest spatial wavelet-domain PC removed.
            Z=X@self.W.T
            # Correlation across valid optodes in wavelet coefficient space.
            Z=Z-Z.mean(axis=1,keepdims=True)
            norm=np.linalg.norm(Z,axis=1)
            good=norm>1e-18
            if np.count_nonzero(good)<2:
                return float('-inf'),np.full(TAPS,np.nan),int(ok.sum())
            # Handle extremely-flat channels deterministically without imputation.
            X=X[good];Z=Z[good];norm=norm[good]
            corr=(Z/norm[:,None])@(Z/norm[:,None]).T
            vals,vecs=np.linalg.eigh(corr)
            u=vecs[:,-1]
            X=X-np.outer(u,u@X)
        Y=X@self.W.T
        h=X@self.H.T
        residual=Y-h@self.D.T
        sigma2=np.sum((residual**2)*self.weight,axis=1)/self.df
        floor=np.maximum(np.mean((Y**2)*self.weight,axis=1)*1e-9,1e-30)
        sigma2=np.maximum(sigma2,floor)
        dev=np.einsum('bi,ij,bj->b',h,self.G,h,optimize=True)/sigma2
        maxpos=np.max(h,axis=1)
        ok=np.isfinite(dev)&(dev>=0)&np.isfinite(maxpos)&(maxpos>1e-15)
        ok &= (np.argmax(h,axis=1)>0)
        if not np.any(ok):return float('-inf'),np.full(TAPS,np.nan),int(X.shape[0])
        candidates=np.flatnonzero(ok)
        ranked=sorted(candidates,key=lambda i:(-dev[i],i))[:TOP_K]
        # Pool only positive HbO curves; this and top-k selection are a V8 adapter.
        waves=h[ranked]/maxpos[ranked,None]
        template=np.median(waves,axis=0)
        score=float(np.log1p(np.median(dev[ranked])))
        if not np.isfinite(template).all() or not np.isfinite(score):
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


def features_for_bank(bank,engine,pca):
    scores=np.empty(bank.n,float);waves=np.empty((bank.n,TAPS),float)
    nch=np.empty(bank.n,int)
    for i in range(bank.n):
        scores[i],waves[i],nch[i]=engine.estimate_epoch(bank.H[i],PCA=pca)
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
    e=WaveletEstimator()
    assert np.isclose(DT,.5) and N==51 and PAD_LENGTH==64
    assert abs(e.lmbda)>1e-12 and len(np.unique(e.scales))==3
    assert np.max(np.abs(e.G-e.G.T))<1e-8
    assert (np.linalg.eigvalsh(e.G)>0).all()
    # Independent normal-equation verification of linear inverse.
    rng=np.random.default_rng(1049)
    z=rng.normal(size=N)
    target=e.H@z
    residual=e.W@z-e.D@target
    stationarity=(e.D.T@(residual*e.weight)-e.lmbda*e.R@target)
    assert np.linalg.norm(stationarity)<1e-7
    t=np.arange(TAPS)*DT
    h=(t**3)*np.exp(-t/1.5)
    h=h/max(h)
    test=np.zeros((9,N)); test[:,CUE:]=h
    test+=rng.normal(0,.008,(9,N))
    for pca in (False,True):
        score,wave,n=e.estimate_epoch(test,PCA=pca)
        assert n==9 and np.isfinite(score) and np.isfinite(wave).all()
    _,curve,_=e.estimate_epoch(test,False)
    assert 0<float(np.nanmax(curve)) and len(extract_boundaries(curve,.2,.1))==3
    assert np.isnan(extract_boundaries(np.full(TAPS,np.nan),.2,.1)).all()
    assert np.isnan(extract_boundaries(np.zeros(TAPS),.2,.1)).all()
    # Must return missing *offset* when falling edge is not observed.
    rising=np.linspace(0,1,TAPS)
    assert np.isnan(extract_boundaries(rising,.2,.1)[2])
    print('PASS: orthogonal DB3, DWT detail/scales, Eq19-21 matrix, PCA1 and no-PCA numerical tests',flush=True)
    return e


def protocol_object(code_sha):
    return {'scientific_version':SCIENTIFIC_VERSION,'code_sha256':code_sha,
            'v8_protocol_sha256':V8_PROTOCOL_HASH,'v8_archive_sha256':FROZEN_ARCHIVE_HASH,
            'models':list(OUR_METHODS),'seeds':list(SEEDS),'counts':COUNTS,
            'paper':'Lina et al (2010) Medical Image Analysis 14 606-616',
            'implementation_class':'independent paper-guided V8 epoch adaptation, NOT author MATLAB code',
            'wavelet':'db3, 3-level orthogonal dyadic (64), reflected 51->64 DWT boundary operator; coarsest omitted',
            'sampling_hz':FS,'event_sample':CUE,'estimation_duration_s':20.,
            'alpha':ALPHA,'one_over_f_exponent':C_EXPONENT,'pca_remove_components':[1,0],
            'top_k_channels':TOP_K,'onset_options':ONSET_OPTIONS,'offset_options':OFFSET_OPTIONS,
            'source_training':'ALL available 25 Shin (LOSO) / 26 Shin (external), 45 trials each, NO heldout calibration',
            'source_threshold_calibration':'subject-balanced BA maximization (source only)',
            'source_fraction_calibration':'subject-balanced four-boundary 4s loss (source only)',
            'new_adapter_parts':['reflect boundary operation','single impulse per isolated epoch','within-epoch PCA',
                'top-k HbO channel pooling','fraction-based boundary extraction','source-only detector calibration'],
            'bspc_claim':'Wavelet-domain HRF method adapted to trial-level boundary task, NOT a published native four-boundary SOTA.',
            'statistical_unit':'participant means after 10 seeds; no inferential p-values claimed',
            'numpy_version':np.__version__,
            'hb_checkpoints_unchanged':True}


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
               'wavelet_label':'Lina 2010 Eq20 V8 51-sample adaptation'}
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

def source_cache_path(seed,pid,pca):
    return OUT/'source_features'/f'seed_{seed:02d}'/f'{pid}_pca{pca}.npz'


def source_features(store,seed,pid,pca,engine,cfg):
    cp=source_cache_path(seed,pid,pca)
    bank=store.shin(seed,pid,'train')
    key=sha_obj({'config':cfg,'seed':seed,'participant':pid,'signature':bank.signature,'pca':pca})
    meta=cp.with_suffix('.meta.json')
    if audit_cached_npz(cp,key):
        with np.load(cp,allow_pickle=False) as z:
            score=z['score'].astype(float);wave=z['wave'].astype(float);valid=z['valid'].astype(int)
    else:
        score,wave,valid=features_for_bank(bank,engine,pca=bool(pca))
        write_cached_npz(cp,key,{'score':score,'wave':wave,'valid':valid})
        print(f' SOURCE seed {seed:02d} {pid}: PCA{pca} wavelet fits={bank.n}, valid curves={np.isfinite(wave).all(axis=1).sum()}',flush=True)
    if score.shape!=(bank.n,) or wave.shape!=(bank.n,TAPS):
        raise RuntimeError('Unexpected source feature cache shape')
    return {'score':score,'wave':wave,'valid':valid,'truth':bank.truth,'signature':bank.signature}


def target_features(bank,seed,pca,engine,cfg):
    # External waveforms are shared across all source seeds; cache by *bank* signature.
    if bank.dataset=='SHIN':folder=OUT/'heldout_features'/'SHIN'/f'seed_{seed:02d}'
    else:folder=OUT/'heldout_features'/bank.dataset
    cp=folder/f'{bank.participant}_pca{pca}.npz'
    key=sha_obj({'cfg':cfg,'signature':bank.signature,'pca':pca})
    marker=cp.with_suffix('.meta.json')
    if audit_cached_npz(cp,key):
        with np.load(cp,allow_pickle=False) as z:return z['score'].astype(float),z['wave'].astype(float)
    score,wave,valid=features_for_bank(bank,engine,pca=bool(pca))
    write_cached_npz(cp,key,{'score':score,'wave':wave,'valid':valid})
    return score,wave


def calibrate_source(seed,target_pid,source,pca,cfg):
    allowed=tuple(sorted(set(source)-({target_pid} if target_pid else set())))
    if len(allowed)!=(25 if target_pid else 26):raise RuntimeError('Target leakage/source participant missing')
    calname=(target_pid or 'ALL_EXTERNAL')+f'_pca{pca}.json'
    cp=OUT/'source_calibration'/f'seed_{seed:02d}'/calname
    sigs={p:source[p]['signature'] for p in allowed}
    key=sha_obj({'cfg':cfg,'seed':seed,'sources':sigs,'pca':pca})
    if cp.exists():
        v=json.loads(cp.read_text())
        if v.get('key')!=key:raise RuntimeError('Changed source calibration, cannot reuse '+str(cp))
        return v
    partial={p:source[p] for p in allowed}
    thr,ba=thresholds_from_source(partial,allowed)
    loss,on,off=select_boundary_fractions(partial,allowed)
    out={'key':key,'seed':seed,'excluded_target':target_pid,'source_pids':list(allowed),
         'source_signatures':sigs,'score_threshold':float(thr),'onset_fraction':on,
         'offset_fraction':off,'source_training_subject_balanced_BA':ba,
         'source_training_timing_4s_loss':loss,'source_calibration_only':True,
         'no_target_truth_used':True}
    cp.parent.mkdir(parents=True,exist_ok=True);save_json(cp,out)
    print(f' CALIB seed {seed:02d} {target_pid or "EXTERNAL"} PCA{pca}: BA={ba:.4f} loss={loss:.4f} score>={thr:.4g} on={on:.2f} off={off:.2f}',flush=True)
    return out


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


def evaluate_case(bank,seed,source_by_pca,engine,cfg):
    from hb_evaluation import timing_metrics, detection_metrics,common_valid_metrics
    from hb_data import digest_file
    hbpred,hbgate,hbsha=store_global.saved(bank,seed)
    base_metrics=read_original_v8_checkpoint(seed,bank,hbsha)
    all_data={};all_metric=[]
    for pca,method in ((1,MAIN),(0,ABLATION)):
        source=source_by_pca[pca]
        cal=calibrate_source(seed,bank.participant if bank.dataset=='SHIN' else None,source,pca,cfg)
        score,wave=target_features(bank,seed,pca,engine,cfg)
        pred=np.array([extract_boundaries(h,cal['onset_fraction'],cal['offset_fraction']) for h in wave])
        gate=(score>=cal['score_threshold']).astype(float)
        # Missing curve = 0 decision for independent detection; no false free-negative failures.
        identity={'dataset':bank.dataset,'participant':bank.participant,'seed':seed,
                  'method':method,'supported':'onset,peak,offset,duration'}
        dm=detection_metrics(bank.truth['present'],gate)
        sm=timing_metrics(bank.truth,pred,hbgate)
        sm.update(common_valid_metrics(bank.truth,pred,hbpred,hbgate))
        om=timing_metrics(bank.truth,pred,gate)
        for mode,metrics in (('independent_detection',dm),('shared_gate_timing',sm),('own_gate_timing',om)):
            all_metric.append({**identity,'mode':mode,**metrics})
        all_data[method]={'score':score,'wave':wave,'pred':pred,'gate':gate,'audit':cal,'seed':seed,
            'source_signatures':cal['source_signatures']}
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
         'study':'Lina2010 paper-guided wavelet estimator adapted to frozen V8 synthetic trial task; NOT published SOTA four-boundary method',
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
    parser.add_argument('--max-new-cases',type=int,default=0,help='0 means no cap')
    args=parser.parse_args(argv)
    engine=run_tests()
    if args.mode=='selftest':return
    mount_drive();restore_original()
    from hb_data import DataStore,PIDS
    import hb_benchmark
    script=Path(__file__)
    code_sha=sha_file(script)
    config=protocol_object(code_sha)
    cfg=sha_obj(config)
    OUT.mkdir(exist_ok=True,parents=True)
    lock=(OUT/'.writer.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as e:raise RuntimeError('Another wavelet runner writes to this output; do not double-start') from e
    prot=OUT/'PROTOCOL.json'
    if prot.is_file():
        if sha_obj(json.loads(prot.read_text()))!=cfg:raise RuntimeError('Out belongs to incompatible numerical code/config')
    elif any(OUT.iterdir()):
        # lock is a legitimate singleton initial-file here.
        extra=[x.name for x in OUT.iterdir() if x.name not in ('.writer.lock',)]
        if extra:raise RuntimeError('Unverified pre-existing wavelet output directory: '+repr(extra))
        save_json(prot,config)
    else:save_json(prot,config)
    if args.mode=='summary':
        result=compile_summary(code_sha);print('Summary verified cases',result['n_verified_cases']);return
    global store_global
    done=0
    targets=SEEDS if args.mode=='full' else (1,)
    for seed in targets:
        store_global=DataStore(ROOT,OUT,(seed,),DATASETS,PIDS)
        store_global.preflight()
        # Always cache source train outputs separately, strict LOSO excludes target in calibration.
        source={pca:{p:source_features(store_global,seed,p,pca,engine,cfg) for p in PIDS}
                for pca in (1,0)}
        identities=[]
        for pid in PIDS:identities.append(('SHIN',pid))
        for ds in ('BALINT','KHAN'):
            ids=store_global.external_ids(ds)
            if len(ids)!=COUNTS[ds]:raise RuntimeError('Wrong external participant count '+ds)
            identities.extend((ds,pid) for pid in ids)
        if len(identities)!=77:raise AssertionError('Seed must have 77 cases')
        if args.mode=='pilot':identities=[('SHIN','VP001'),('BALINT','C01'),('KHAN','S01')]
        for ds,pid in identities:
            bank=(store_global.shin(seed,pid,'test') if ds=='SHIN' else store_global.external(ds,pid))
            hb_pred,hb_gate,hb_sha=store_global.saved(bank,seed)
            cal_source={p:source[p] for p in (1,0)}
            source_signatures={k:{p:source[k][p]['signature'] for p in sorted(source[k]) if not(ds=='SHIN' and p==pid)}
                               for k in (1,0)}
            key=sha_obj({'cfg':cfg,'dataset':ds,'seed':seed,'pid':pid,'bank_signature':bank.signature,
                         'hb_sha':hb_sha,'source_signatures':source_signatures})
            cp=OUT/'checkpoints'/ds/f'seed_{seed:02d}'/pid
            marker=verify_checkpoint(cp,key,cfg)
            if marker is not None:
                print(f'REUSED validated checkpoint seed={seed:02d} {ds} {pid}',flush=True)
                continue
            data,met,baseline,hb_sh= evaluate_case(bank,seed,cal_source,engine,cfg)
            save_checkpoint(cp,key,cfg,bank,hb_sh,data,met,baseline)
            done+=1
            summaries=[]
            for m in OUR_METHODS:
                row=next(x for x in met if x['method']==m and x['mode']=='shared_gate_timing')
                det=next(x for x in met if x['method']==m and x['mode']=='independent_detection')
                summ=[float(row[f'{b}_loss_missing4_sec']) for b in ('onset','peak','offset','duration')]
                summaries.append((m,float(det['balanced_accuracy']),np.round(summ,3).tolist()))
            print(f'COMPLETE new={done:04d} seed={seed:02d} {ds:6} {pid:5} | '+str(summaries),flush=True)
            if args.max_new_cases and done>=args.max_new_cases:
                res=compile_summary(code_sha)
                print('SESSION LIMIT HIT:',done,'new; verified overall',res['n_verified_cases'],'/770. Rerun to resume.',flush=True)
                return
        res=compile_summary(code_sha)
        print(f'SEED {seed:02d} FINISHED; checkpoint verified cases {res["n_verified_cases"]}/770',flush=True)
    res=compile_summary(code_sha)
    print('FULL FINISH' if res['complete'] else 'PILOT FINISH',res['n_verified_cases'],'/770')
    print('RESULTS',OUT,flush=True)
    if res['complete']:
        print('No automatic inferential p-values; publication claims require boundary and signal assumptions review.')

if __name__=='__main__':main()
