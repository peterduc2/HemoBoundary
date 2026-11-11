"""Strict input discovery and deterministic replay of the ORIGINAL evaluation cases.
A replay is admitted only after reference-label and available feature parity checks.
It is not a new synthetic dataset and is never saved over an original bank.
"""
from __future__ import annotations
import csv, gzip, hashlib, io, json, os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import original_data as od

PIDS=tuple(f'VP{i:03d}' for i in range(1,27))
SEEDS=tuple(range(1,11))
D95='95_N4_FUNCTIONAL_ABC_LBR_FULLY_NESTED_10SEEDS_V1'
D98='98_B0_STRICT_HIERARCHICAL_SEED01_FULLY_NESTED_V1'
D99='99_B0_STRICT_HIERARCHICAL_10SEEDS_V1'
D102='102_OPTIONB_TABLE3_10SEED_BALINT_KHAN_ZERO_SHOT_V1'
RELROOT='MyDrive/HonorProject/shin2018_word_generation_state_preparation_v1'
REPLAY_VERSION='HB_V8_REPLAY_1'
LABEL_KEYS=('present','onset','peak','offset','censored')


def digest_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def digest_json(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def write_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False),encoding='utf-8');os.replace(tmp,p)

def save_npz(p,**arrays):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f,**arrays)
    os.replace(tmp,p)

def read_rows(p):
    p=Path(p);op=gzip.open if p.name.endswith('.gz') else open
    with op(p,'rt',newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def write_rows(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    tmp=p.with_name(p.name+'.tmp')
    op=gzip.open if p.name.endswith('.gz') else open
    with op(tmp,'wt',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    os.replace(tmp,p)

def number(value):
    if value is None or str(value).strip().lower() in ('','nan','none','na'):return float('nan')
    return float(value)

def locate_root(explicit=None):
    candidates=[]
    if explicit:candidates.append(Path(explicit).expanduser())
    if os.environ.get('HB_SHIN_ROOT'):candidates.append(Path(os.environ['HB_SHIN_ROOT']))
    candidates += [Path(m)/RELROOT for m in ('/content/gdrive','/content/drive')]
    matches=[r for r in candidates if (r/D95/'seed_01/fresh_cache/VP001.npz').is_file()]
    if not matches:
        raise FileNotFoundError('Original strict banks were not found. Checked: '+', '.join(map(str,candidates)))
    return matches[0]

def cache_path(root,seed,pid):
    return Path(root)/(D95 if seed==1 else D99)/f'seed_{seed:02d}/fresh_cache/{pid}.npz'

def pred_path(root,seed,pid):
    base=Path(root)/D98 if seed==1 else Path(root)/D99/f'seed_{seed:02d}'
    return base/'checkpoints'/f'{pid}_ROWS.csv.gz'

def external_h5(root,name):
    rel={'BALINT':'BALINT_V55_EXTERNAL_PREP/processed/balint_epochs_10hz.h5',
         'KHAN':'KHAN2026_FINGER_HEMOBOUNDARY_EXTERNAL_PREP/processed/khan2026_finger_epochs_10hz.h5'}
    return Path(root).parent/rel[name]

@dataclass
class Bank:
    dataset:str
    participant:str
    seed:int
    split:str
    H:np.ndarray             # trials x channels x 51 ORIGINAL simulated points (-5..20)
    R:np.ndarray
    features:np.ndarray      # trials x 41 x 9, verified against historical cache where stored
    truth:dict
    signature:str
    audit:dict
    @property
    def n(self):return len(self.H)
    def predictors(self):return {'H':self.H,'R':self.R,'features':self.features}


def make_trials(profile,namespace,npos,nneg):
    rng=np.random.default_rng(od.stable_seed(*namespace))
    trials=[od.synthetic_trial(profile,rng,True) for _ in range(npos)]
    trials.extend(od.synthetic_trial(profile,rng,False) for _ in range(nneg));rng.shuffle(trials)
    H=np.stack([od.baseline_z(t['hbo']) for t in trials])
    R=np.stack([od.baseline_z(t['hbr']) for t in trials])
    ev=[od.evidence(t['hbo'],t['hbr']) for t in trials]
    vals={k:np.asarray([t[k] for t in trials]) for k in LABEL_KEYS}
    vals['states']=np.stack([t['states'] for t in trials])
    vals['clean']=np.stack([t['clean_shape'] for t in trials]).astype(np.float32)
    vals['obs']=np.stack([od.whole_vector(e) for e in ev])
    vals['state_feat']=np.stack([od.state_features(e) for e in ev])
    vals['ch_anti']=0.5*(H[:,:,od.POST_MASK]-R[:,:,od.POST_MASK])
    return H,R,vals


def assert_parity(replayed,historical,context,keys):
    maxerr={}
    for k in keys:
        a=np.asarray(replayed[k]);b=np.asarray(historical[k])
        if a.shape!=b.shape or not np.allclose(a,b,rtol=2e-6,atol=2e-6,equal_nan=True):
            raise ValueError(f'{context}: deterministic replay mismatch for {k}. STOP; no substitute test set is allowed.')
        finite=np.isfinite(a)&np.isfinite(b)
        maxerr[k]=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.0
    return maxerr


class DataStore:
    def __init__(self,root,out,seeds,datasets=('SHIN','BALINT','KHAN'),pids=PIDS):
        self.root=Path(root);self.out=Path(out);self.seeds=tuple(seeds);self.datasets=tuple(datasets);self.pids=tuple(pids)
        self.profiles={};self.ext_stores={};self.ext_rows={};self.ext_qc={};self._hashes={}
        self.preps={p:self.root/'02_preprocessed'/f'{p}_WG_ALIGNED_PREPROCESSED.npz' for p in pids}
        self.replay_cache=self.out/'replayed_original_signals'
        self.original_manifest=digest_file(Path(__file__).parent/'provenance/DATA_SOURCE_MANIFEST.json')
        self.common_c=None
    def sha(self,p):
        p=Path(p);stat=p.stat();key=(str(p),stat.st_size,stat.st_mtime_ns)
        if key not in self._hashes:self._hashes[key]=digest_file(p)
        return self._hashes[key]
    def preflight(self):
        required=list(self.preps.values())
        for seed in self.seeds:
            required += [cache_path(self.root,seed,p) for p in self.pids]
            if 'SHIN' in self.datasets:required += [pred_path(self.root,seed,p) for p in self.pids]
        for ds in self.datasets:
            if ds!='SHIN':
                required += [external_h5(self.root,ds),self.root/D102/'audit/EXTERNAL_FIXED_BANK_QC.csv']
                required += [self.root/D102/'checkpoints'/f'seed_{s:02d}_PER_TRIAL.csv.gz' for s in self.seeds]
        absent=[str(p) for p in required if not p.is_file()]
        if absent:
            raise FileNotFoundError('Required ORIGINAL inputs are absent. No 2-channel or zero-padding fallback will be used.\n'+'\n'.join(absent[:25]))
        channels=[]
        for p in self.preps.values():
            with np.load(p,allow_pickle=False) as z:
                if 'hbo_causal_lp02_epochs' not in z:raise ValueError(f'{p}: unexpected preprocessing format')
                channels.append(z['hbo_causal_lp02_epochs'].shape[1])
        self.common_c=min(channels)
        print(f'INPUT PREFLIGHT PASS: {len(self.seeds)} seeds, {len(self.pids)} source participants, {self.datasets}; common source channels={self.common_c}',flush=True)
    def shin(self,seed,pid,split):
        assert split in ('train','test')
        cp=cache_path(self.root,seed,pid);pp=self.preps[pid];prefix='tr' if split=='train' else 'te'
        sign=digest_json({'version':REPLAY_VERSION,'adapter_code':digest_file(Path(__file__)),'generator_code':digest_file(Path(od.__file__)),'code':self.original_manifest,'prep':self.sha(pp),'bank':self.sha(cp),'seed':seed,'split':split,'pid':pid,'common_c':self.common_c})
        dest=self.replay_cache/'SHIN'/f'seed_{seed:02d}'/f'{pid}_{split}.npz'
        existing=self._read_replay(dest,sign)
        if existing is not None:return self._bank(existing,'SHIN',pid,seed,split,sign)
        if pid not in self.profiles:
            profile=od.estimate_noise_profile(pp)
            c=self.common_c
            if profile.n_channels>c:profile=od.NoiseProfile(c,profile.hbo_scale[:c],profile.hbr_scale[:c],profile.phi_hbo,profile.phi_hbr)
            self.profiles[pid]=profile
        ns=('RAW_D_SCREEN',pp.stem,split)
        if seed!=1:ns=('MULTISEED_REPEAT',seed,*ns)
        H,R,vals=make_trials(self.profiles[pid],ns,30 if split=='train' else 24,15 if split=='train' else 12)
        with np.load(cp,allow_pickle=False) as z:
            hist={k:np.asarray(z[f'{prefix}_{k}']) for k in (*LABEL_KEYS,'states','obs','state_feat')}
            for k in ('clean','ch_anti'):
                if f'{prefix}_{k}' in z.files:hist[k]=np.asarray(z[f'{prefix}_{k}'])
        delta=assert_parity(vals,hist,f'SHIN {seed}/{pid}/{split}',tuple(hist))
        audit={'kind':'DETERMINISTIC_REPLAY_FEATURE_AND_LABEL_PARITY','source_bank_sha256':self.sha(cp),'prep_sha256':self.sha(pp),'maximum_errors':delta,
               'preperiod':'original simulated baseline, not measured real baseline and not zeros','postperiod':'original simulated observations; no appended samples',
               'synthesis_filtering':'AR-noise and generated response; no low-pass applied AFTER synthesis','historical_feature_parity':True}
        payload={'H':H,'R':R,'features':vals['state_feat'],**{k:vals[k] for k in LABEL_KEYS},'audit':np.array(json.dumps(audit)),'signature':np.array(sign)}
        save_npz(dest,**payload);write_json(dest.with_suffix('.sha256.json'),{'file_sha256':digest_file(dest),'signature':sign});return self._bank(payload,'SHIN',pid,seed,split,sign)
    def _read_replay(self,p,signature):
        if not p.exists():return None
        try:
            m=json.loads(p.with_suffix('.sha256.json').read_text())
            if m.get('file_sha256')!=digest_file(p) or m.get('signature')!=signature:return None
            with np.load(p,allow_pickle=False) as z:
                if str(z['signature'])!=signature:return None
                return {k:np.array(z[k]) for k in z.files}
        except (ValueError,OSError,KeyError):return None
    def _bank(self,d,dataset,pid,seed,split,sign):
        H=np.asarray(d['H'],float);R=np.asarray(d['R'],float);f=np.asarray(d['features'],float)
        if H.shape!=R.shape or H.ndim!=3 or H.shape[2]!=51 or f.shape!=(len(H),41,9):raise ValueError('Replay dimensions invalid')
        if not np.isfinite(H).all() or not np.isfinite(R).all() or not np.isfinite(f).all():raise ValueError('Non-finite model observations')
        truth={k:np.asarray(d[k]) for k in LABEL_KEYS}
        if not set(np.unique(truth['present'])).issubset({0,1}) or not set(np.unique(truth['censored'])).issubset({0,1}):raise ValueError('Non-binary labels')
        return Bank(dataset,pid,seed,split,H,R,f,truth,sign,json.loads(str(d['audit'])))
    def external_rows(self,ds,seed):
        k=(ds,seed)
        if k not in self.ext_rows:
            p=self.root/D102/'checkpoints'/f'seed_{seed:02d}_PER_TRIAL.csv.gz'
            rows=[r for r in read_rows(p) if r.get('dataset')==ds and r.get('model')=='OPTIONB_HEMOBOUNDARY']
            if not rows:raise ValueError(f'{p}: missing strict {ds} OPTIONB_HEMOBOUNDARY rows; older V95 results are not accepted')
            self.ext_rows[k]=rows
        return self.ext_rows[k]
    def external_ids(self,ds):
        return tuple(sorted({r['participant'] for r in self.external_rows(ds,self.seeds[0])}))
    def external(self,ds,pid):
        hp=external_h5(self.root,ds);qcpath=self.root/D102/'audit/EXTERNAL_FIXED_BANK_QC.csv'
        sign=digest_json({'version':REPLAY_VERSION,'adapter_code':digest_file(Path(__file__)),'generator_code':digest_file(Path(od.__file__)),'code':self.original_manifest,'h5':self.sha(hp),'qc':self.sha(qcpath),'dataset':ds,'pid':pid,'fixed_seed_namespace':1})
        dest=self.replay_cache/ds/f'{pid}_test.npz'
        existing=self._read_replay(dest,sign)
        if existing is not None:return self._bank(existing,ds,pid,0,'test',sign)
        if ds not in self.ext_stores:
            self.ext_stores[ds]=od.load_external_hdf5(hp)
            ids=tuple(sorted(set(self.ext_stores[ds]['participant'].astype(str))))
            if ids!=self.external_ids(ds):raise ValueError(f'{ds}: HDF5 participant list differs from strict saved external predictions')
            self.ext_qc[ds]={r['participant']:r for r in read_rows(qcpath) if r.get('dataset')==ds}
        profile,valid,total=od.noise_profile_from_external(self.ext_stores[ds],pid)
        qc=self.ext_qc[ds].get(pid)
        if qc is None:raise ValueError(f'{ds}/{pid}: missing original external profile QC')
        current={'usable_channels':valid,'channels_before_validity_filter':total,'phi_hbo':profile.phi_hbo,'phi_hbr':profile.phi_hbr,
                 'median_hbo_noise_scale':float(np.median(profile.hbo_scale)),'median_hbr_noise_scale':float(np.median(profile.hbr_scale))}
        for k,v in current.items():
            if k not in qc or not np.isclose(number(qc[k]),v,atol=1e-8,rtol=2e-6):raise ValueError(f'{ds}/{pid}: original profile QC mismatch {k}')
        H,R,vals=make_trials(profile,('RAW_D_SCREEN_EXTERNAL',ds,pid,'test'),24,12)
        for seed in self.seeds:check_prediction_rows(self.external_rows(ds,seed),pid,vals,seed,ds)
        audit={'kind':'ORIGINAL_EXTERNAL_GENERATOR_PROFILE_QC_AND_ALL_SEED_LABEL_PARITY','h5_sha256':self.sha(hp),'qc_sha256':self.sha(qcpath),'profile':current,
               'historical_feature_parity':False,'reason':'original external script did not save a raw-signal bank/hash; replay is verified against original generator, source HDF5 profile QC and all saved truth rows',
               'preperiod':'original generated noisy baseline; no zero padding','fixed_bank_across_source_seeds':True}
        payload={'H':H,'R':R,'features':vals['state_feat'],**{k:vals[k] for k in LABEL_KEYS},'audit':np.array(json.dumps(audit)),'signature':np.array(sign)}
        save_npz(dest,**payload);write_json(dest.with_suffix('.sha256.json'),{'file_sha256':digest_file(dest),'signature':sign});return self._bank(payload,ds,pid,0,'test',sign)
    def saved(self,bank,seed):
        if bank.dataset=='SHIN':
            p=pred_path(self.root,seed,bank.participant)
            rows=[r for r in read_rows(p) if r.get('row',r.get('stage'))=='BASE_PLUS_A_PLUS_B_PLUS_C_PLUS_LBR']
        else:
            p=self.root/D102/'checkpoints'/f'seed_{seed:02d}_PER_TRIAL.csv.gz';rows=self.external_rows(bank.dataset,seed)
        q=check_prediction_rows(rows,bank.participant,bank.truth,seed,bank.dataset)
        pred=np.array([[number(r[f'pred_{k}']) for k in ('onset','peak','offset')] for r in q])
        gate=np.array([number(r['pred_response']) for r in q],int)
        if not set(gate).issubset({0,1}):raise ValueError('Saved response gate is not binary')
        return pred,gate,self.sha(p)


def check_prediction_rows(rows,pid,truth,seed,dataset):
    q=[r for r in rows if r.get('participant')==pid]
    q.sort(key=lambda r:int(r['synthetic_index']))
    if len(q)!=36 or [int(r['synthetic_index']) for r in q]!=list(range(36)):
        raise ValueError(f'{dataset}/{seed}/{pid}: need each synthetic_index 0..35 exactly once')
    for r in q:
        if 'repeat_seed' in r and int(float(r['repeat_seed']))!=seed:raise ValueError('Saved repeat seed differs')
    hist={k:np.array([number(r['truth_'+k]) for r in q]) for k in LABEL_KEYS}
    assert_parity(truth,hist,f'{dataset}/{seed}/{pid}: saved label audit',LABEL_KEYS)
    return q
