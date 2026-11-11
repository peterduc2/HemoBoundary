"""Trial-level recorded-background adaptation of published fNIRS methods.

This evaluation requires source-calibration and reference files from the
preceding experiments. It does not reproduce the full published pipelines.
"""
import os, sys, re, gzip, json, hashlib, shutil, subprocess, traceback, zipfile, io
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats

os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'

try:
    import mne, h5py
except ImportError:
    raise ImportError('Install requirements.txt before running recorded adapters')

ROOT=Path(os.environ.get('HEMOBOUNDARY_PROJECT_ROOT', '/content/gdrive/MyDrive/HonorProject'))
SHIN=ROOT/'shin2018_word_generation_state_preparation_v1'
REST_SOURCE=ROOT/'HEMOBOUNDARY_ARRAY_REST_REPAIR_20261026'
ZIP_SOURCE=ROOT/'HEMOBOUNDARY_ARRAY_REST_REPAIR_20261026.zip'
OUT=Path(os.environ.get('HEMOBOUNDARY_RECORDED_ADAPTER_OUTPUT', str(ROOT/'HemoBoundary_Recorded_Adapters')))/'hrfunc'
OUT.mkdir(parents=True,exist_ok=True)
REPO=Path(__file__).resolve().parents[1]
COMMIT='07bf96334b58e3f1e8be82beb45d25e101ce5cd3'
MORPH_SHA='0e3bfaa393fdcc9a7f648db3079e239824dcd7c968d63a5bac3513a45cc4aaeb'
TAG='HRFUNC'


def sha(path):
    hh=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            hh.update(chunk)
    return hh.hexdigest()

def write_json(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf8')

def fail(reason):
    write_json('STATUS.json',{'status':'BLOCKED','reason':str(reason),'method':TAG})
    print('BLOCKED:',reason)
    raise RuntimeError(str(reason))

def bootstrap():
    if not SHIN.is_dir():fail('Missing original Shin root: '+str(SHIN))
    if not (REPO/'src'/'morphology.py').is_file():
        fail('Verified HemoBoundary source is missing from the repository')
    if sha(REPO/'src'/'morphology.py')!=MORPH_SHA:fail('Pinned HemoBoundary morphology SHA mismatch')
    import importlib.util
    spec=importlib.util.spec_from_file_location('hb_pinned_morph',REPO/'src'/'morphology.py')
    morph=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=morph
    spec.loader.exec_module(morph)
    morph.SHIN_ROOT=SHIN
    morph.N4_10SEED_DIR=SHIN/'95_N4_FUNCTIONAL_ABC_LBR_FULLY_NESTED_10SEEDS_V1'
    morph.HIERARCHICAL_SEED1_DIR=SHIN/'98_B0_STRICT_HIERARCHICAL_SEED01_FULLY_NESTED_V1'
    morph.HIERARCHICAL_10SEED_DIR=SHIN/'99_B0_STRICT_HIERARCHICAL_10SEEDS_V1'
    morph.protected_self_tests()
    write_json('PROVENANCE.json',{'source_commit':COMMIT,'morphology_sha256':MORPH_SHA,
        'python':sys.version,'numpy':np.__version__,'mne':mne.__version__,'method':TAG,
        'evaluation':'14 recorded resting subjects, 224 fixed injected/control conditions, ONE frozen source configuration',
        'external_label_tuning':False,'original_rest_normalization':True})
    return morph


def get_saved():
    """Read the archived reference labels for this evaluation without substitution."""
    primary=REST_SOURCE/'REST_RECORDED_BACKGROUND_TRIALS.csv.gz'
    metadata=REST_SOURCE/'REST_FILE_ELIGIBILITY.json'
    if primary.is_file() and metadata.is_file():
        df=pd.read_csv(primary)
        eligibility=json.loads(metadata.read_text())
    elif ZIP_SOURCE.is_file():
        with zipfile.ZipFile(ZIP_SOURCE) as z:
            df=pd.read_csv(io.BytesIO(gzip.decompress(z.read('REST_RECORDED_BACKGROUND_TRIALS.csv.gz'))))
            eligibility=json.loads(z.read('REST_FILE_ELIGIBILITY.json'))
    else:fail('Fourth-dataset original results not found at EXACT folder or ZIP path')
    hb=df[df.method=='HEMOBOUNDARY_FULL'].copy().sort_values(['participant','trial_index'])
    if len(hb)!=224 or hb.participant.nunique()!=14 or not (hb.groupby('participant').size()==16).all():
        fail('Original 14x16 HemoBoundary results are incomplete/duplicated')
    if set(hb.seed)!={1} or set(hb.family)!={'RECORDED_REST_INJECTED'}:fail('Unexpected source seed or family')
    if set(hb.trial_index)!=set(range(16)):fail('Unexpected trial indices')
    hb.to_csv(OUT/'FROZEN_HEMOBOUNDARY_REFERENCE.csv.gz',index=False,compression='gzip')
    eligible=eligibility.get('eligible',[])
    if len(eligible)!=14:fail('Eligibility record does not enumerate exactly 14 original recordings')
    return hb.reset_index(drop=True),eligible


def unaugmented_pairs(path):
    with h5py.File(path,'r') as z:
        if 'nirs' not in z:raise ValueError('No nirs entry in SNIRF')
        pairs=set()
        for key in z['nirs']:
            if not re.fullmatch(r'data\d+',key):continue
            levels=defaultdict(set)
            for mname in z['nirs'][key]:
                if not re.fullmatch(r'measurementList\d+',mname):continue
                m=z['nirs'][key][mname]
                if all(k in m for k in ('sourceIndex','detectorIndex','dataTypeLabel')):
                    levels[(int(m['sourceIndex'][()]),int(m['detectorIndex'][()]))].add(int(m['dataTypeLabel'][()]))
            pairs|={f'S{a}_D{b}' for (a,b),values in levels.items() if values=={0}}
    if not pairs:raise ValueError('No independently verified unaugmented SNIRF channel pairs')
    return pairs


def read_recorded(path):
    valid_pairs=unaugmented_pairs(path)
    raw=mne.io.read_raw_snirf(str(path),preload=True,verbose='ERROR')
    raw.pick(picks='fnirs_cw_amplitude')
    raw=mne.preprocessing.nirs.optical_density(raw)
    raw=mne.preprocessing.nirs.beer_lambert_law(raw,ppf=6.0)
    raw.filter(0.01,0.2,verbose='ERROR')
    raw.resample(2.,verbose='ERROR')
    channels=[q for q in sorted(valid_pairs) if q+' hbo' in raw.ch_names and q+' hbr' in raw.ch_names]
    if len(channels)<4:raise ValueError('Not enough source-verified HbO/HbR channels')
    H=raw.get_data(picks=[q+' hbo' for q in channels]);R=raw.get_data(picks=[q+' hbr' for q in channels])
    good=np.isfinite(H).all(1)&np.isfinite(R).all(1)
    good &= (np.std(H,axis=1)>1e-12)&(np.std(R,axis=1)>1e-12)
    if sum(good)<4:raise ValueError('Not enough finite/nonconstant channels')
    return H[good],R[good],float(raw.info['sfreq'])


def recreate_subject(m,pid,eligible,hb):
    k=int(pid.split('_')[-1])-1
    record=eligible[k]
    path=Path(record['path'])
    if not path.is_file():fail(f'Missing source SNIRF: {path}')
    if not path.parent.name.lower().startswith('subj'):fail('Unexpected participant folder')
    kept=hb[hb.participant==pid].sort_values('trial_index').reset_index(drop=True)
    if not (kept.external_original_subject_id==record['subject_id']).all():fail('Subject mapping mismatch')
    actual_sha=sha(path)
    if not (kept.recording_sha256==actual_sha).all():fail('SNIRF checksum changed: '+pid)
    H,R,fs=read_recorded(path)
    if abs(fs-2.)>1e-8:fail('Unexpected resampled rest frequency')
    if not (kept.background_channels==H.shape[0]).all():fail('Changed channel selection: '+pid)
    window_size=len(m.FULL_T)
    step=int(round(30*fs))
    windows=[]
    for start in range(0,H.shape[1]-window_size+1,step):
        if len(windows)>=4:break
        hh,rr=H[:,start:start+window_size],R[:,start:start+window_size]
        if np.isfinite(hh).all() and np.isfinite(rr).all():windows.append((hh,rr))
    if len(windows)!=4:fail('Cannot reconstruct exactly four background windows for '+pid)
    raw_H=[];raw_R=[];truth=[]
    for j,(hh,rr) in enumerate(windows):
        raw_H.append(hh);raw_R.append(rr)
        truth.append((0,np.nan,np.nan,np.nan,0,np.nan))
        rng=np.random.default_rng(73000+100*k+j)
        shape,on,peak,off,_=m.original_half_cosine_shape(rng)
        scale=np.maximum(np.std(hh[:,m.FULL_T<0],axis=1),1e-12)
        for amplitude in (.5,1.,2.):
            support=rng.uniform(.6,1.,size=len(scale))
            h=hh.copy();r=rr.copy()
            h[:,m.POST_MASK]+=(amplitude*scale*support)[:,None]*shape[None,:]
            r[:,m.POST_MASK]-=(.5*amplitude*scale*support)[:,None]*shape[None,:]
            raw_H.append(h);raw_R.append(r)
            truth.append((1,on,peak,off,int(off>19.75),amplitude))
    t=np.asarray(truth,float)
    for i,(col,j) in enumerate([('truth_present',0),('truth_onset',1),('truth_peak',2),
         ('truth_offset',3),('truth_censored',4),('amplitude_in_baseline_sd',5)]):
        a=kept[col].to_numpy(float)
        if not np.allclose(a,t[:,j],atol=1e-7,rtol=1e-7,equal_nan=True):
            fail(f'Generated truth differs from original saved record: {pid} {col}')
    return kept,raw_H,raw_R


def participant_metrics(m,source,variant_name,score,raw_bounds,gate):
    truth={key:source['truth_'+key].to_numpy() for key in ('present','onset','peak','offset','censored')}
    b=np.asarray(raw_bounds,float)
    result=[]
    for mode,decisions in [('native',np.asarray(gate,int)),
                           ('shared_hb',source.native_response.to_numpy(int)),
                           ('ungated_positive',np.ones(len(source),int))]:
        predictions=pd.DataFrame({'pred_response':decisions})
        for j,key in enumerate(('onset','peak','offset')):
            x=b[:,j].copy();x[decisions==0]=np.nan
            predictions['pred_'+key]=x
        metrics=m.participant_metrics(truth,predictions)
        if mode=='native':
            # Nonfinite method scores are treated as missing predictions.
            tested=~np.isnan(np.asarray(score,float))
            positives=np.asarray(truth['present'],int)==1
            sensitivity=float(np.mean(tested[positives] & (decisions[positives]==1)))
            specificity=float(np.mean(tested[~positives] & (decisions[~positives]==0)))
            metrics['response_balanced_accuracy']=.5*(sensitivity+specificity)
            metrics['detection_valid_fraction']=float(tested.mean())
            metrics['sensitivity']=sensitivity
            metrics['specificity']=specificity
        else:
            metrics['response_balanced_accuracy']=np.nan
        result.append({'participant':source.participant.iloc[0],'method':variant_name,'gate':mode,**metrics})
    return result


def participant_original_hb(m,source):
    bounds=source[['raw_onset','raw_peak','raw_offset']].to_numpy(float)
    return participant_metrics(m,source,'HEMOBOUNDARY_FULL',
         source.response_probability.to_numpy(float),bounds,source.native_response.to_numpy(int))


def summarize(run_rows,metrics):
    rr=pd.DataFrame(run_rows); pm=pd.DataFrame(metrics)
    if set(rr.participant)!=set(f'REST_{i:03d}' for i in range(1,15)):
        fail('Incomplete participant trial prediction coverage')
    rr.to_csv(OUT/'PER_TRIAL_PREDICTIONS.csv.gz',index=False,compression='gzip')
    pm.to_csv(OUT/'PARTICIPANT_METRICS.csv',index=False)
    ms=pm.groupby(['method','gate'],dropna=False)[['onset_mae_sec','peak_mae_sec','offset_mae_sec',
                  'duration_mae_sec','response_balanced_accuracy']].mean().reset_index()
    ms.to_csv(OUT/'METHOD_SUMMARY.csv',index=False)
    print('\nPARTICIPANT-LEVEL MEANS:\n',ms.to_string(index=False))
    comps=[]
    losses=('onset_mae_sec','peak_mae_sec','offset_mae_sec','duration_mae_sec')
    # Correct the pooled method family after computing participant-paired tests.
    for (method,gate),pm_c in pm[pm.method!='HEMOBOUNDARY_FULL'].groupby(['method','gate']):
        pm_b=pm[(pm.method=='HEMOBOUNDARY_FULL')&(pm.gate==gate)]
        merged=pm_c.merge(pm_b,on='participant',validate='one_to_one',suffixes=('_x','_hb'))
        if len(merged)!=14:fail('Incomplete matched statistical comparison')
        for metric in losses:
            v=(merged[metric+'_x']-merged[metric+'_hb']).to_numpy(float)
            if not np.isfinite(v).all():fail('Nonfinite paired metrics')
            signs=np.where((np.arange(1<<len(v),dtype=np.uint32)[:,None]>>np.arange(len(v)))&1,1.,-1.)
            p=float(np.mean(np.abs((signs*v).mean(axis=1))>=abs(v.mean())-1e-12))
            comps.append({'method':method,'gate':gate,'metric':metric,'n_participants':14,
                          'mean_advantage_hb_sec':float(v.mean()),'two_sided_exact_signflip_p':p,
                          'note':'Exploratory unadjusted p; combined Holm family requires all 3 method scripts'})
    pd.DataFrame(comps).to_csv(OUT/'PAIRED_EXACT_SIGNFLIP_UNADJUSTED.csv',index=False)
    print('Statistical unit: 14 independent participants, NOT 224 test conditions.')
    return ms


def results_zip():
    target=OUT.with_suffix('.zip')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in sorted(OUT.rglob('*')):
            if file.is_file():z.write(file,file.relative_to(OUT))
    print('\nOUTPUT FOLDER:',OUT)
    print('SINGLE ZIP:',target,'MB:',round(target.stat().st_size/1e6,2))



def execute(m,hb,eligible):
    try:
        import hrfunc
    except ImportError:
        raise RuntimeError('Install hrfunc==1.3.1 explicitly using docs/RUN_ORDER.md')
    import importlib.metadata as metadata
    if metadata.version('hrfunc')!='1.3.1':fail('Original hrfunc author package must be version 1.3.1')
    root=REPO/'src'/'expanded'/'frozen'
    sys.path.insert(0,str(root))
    from hb_data import DataStore,PIDS,digest_file
    import hb_algorithms as alg, hb_models as models, hb_benchmark as benchmark
    from hb_evaluation import stable_seed
    # Require numerical compatibility with the installed author API.
    benchmark.native_package_tests()
    store=DataStore(SHIN,OUT,seeds=(1,),datasets=('SHIN',),pids=PIDS)
    store.preflight()
    all_source={p:store.shin(1,p,'train') for p in PIDS if p!='VP001'}
    if len(all_source)!=25:fail('Expected 25 non-target Shin source training banks')
    original_cache=SHIN/'HB_V8_MULTICHANNEL_PUBLISHED_COMPARISON'
    class SourceCache:
        def __init__(self):self.memory={}
        def get(self,bank,family,variant):
            if family!='HRFUNC_AUTHOR' or variant!='lambda0.001':raise KeyError((family,variant))
            ident=(bank.dataset,bank.participant,bank.split,bank.signature)
            if ident in self.memory:return self.memory[ident]
            if bank.dataset=='SHIN':
                fp=original_cache/'estimator_outputs'/'SHIN'/'seed_01'/bank.participant/'train_HRFUNC_AUTHOR_lambda0.001.npz'
                auditfp=fp.with_suffix('.json')
                if fp.is_file() and auditfp.is_file():
                    audit=json.loads(auditfp.read_text())
                    if audit.get('file_sha256')==digest_file(fp) and audit.get('bank_signature')==bank.signature:
                        with np.load(fp,allow_pickle=False) as z:
                            pair=(np.asarray(z['curves']),np.asarray(z['score']),audit)
                        self.memory[ident]=pair
                        print('Reuse verified original HRfunc source cache:',bank.participant,flush=True)
                        return pair
            # Participant checkpoints support resuming interrupted runs.
            cp=OUT/'source_hrfunc_curves'/bank.dataset/bank.participant
            cp.mkdir(parents=True,exist_ok=True)
            target=cp/f'{bank.split}_HRFUNC_AUTHOR_lambda0.001.npz'
            audit_file=cp/f'{bank.split}_audit.json'
            if target.is_file() and audit_file.is_file():
                audit=json.loads(audit_file.read_text())
                if audit.get('sha256')!=sha(target) or audit.get('bank_signature')!=bank.signature:
                    fail('Stale/modified HRfunc partial source cache: '+str(target))
                with np.load(target,allow_pickle=False) as z:
                    pair=(np.asarray(z['curves']),np.asarray(z['score']),audit)
                if pair[0].shape!=(bank.n,41) or pair[1].shape!=(bank.n,):fail('Source cache shape mismatch')
                self.memory[ident]=pair
                print('Reuse own saved HRfunc kernel:',bank.participant,bank.split,flush=True)
                return pair
            print('Computing author HRfunc kernel:',bank.participant,bank.split,flush=True)
            curves,scores,audit=alg.hrfunc_curves(bank.H,bank.R)
            if curves.shape!=(bank.n,41) or scores.shape!=(bank.n,):fail('HRfunc invalid kernel shape')
            np.savez_compressed(target,curves=curves,score=scores)
            audit_file.write_text(json.dumps({**audit,'sha256':sha(target),'bank_signature':bank.signature},indent=2,default=str))
            pair=(curves,scores,audit);self.memory[ident]=pair
            return pair
    cache=SourceCache()
    fit_seed=stable_seed(benchmark.VERSION,1,'LOSO_VP001','same_group_folds_all_methods')
    fitted=models.fit_family('HRFUNC_AUTHOR',all_source,cache,excluded='VP001',seed=fit_seed)
    # Check selected settings against available source-calibration records.
    selection=original_cache/'selection_audits'/'seed_01'/'LOSO_VP001.json'
    if selection.is_file():
        prev=json.loads(selection.read_text())
        matches=[a for a in prev.get('models',[]) if a.get('family')=='HRFUNC_AUTHOR']
        if len(matches)!=1:fail('Original HRfunc selection audit missing or ambiguous')
        old=matches[0]
        for key in ('native_selection','readout_selection','detector_selection'):
            if json.dumps(old.get(key),sort_keys=True)!=json.dumps(fitted.audit.get(key),sort_keys=True):
                fail('HRfunc source-selected settings differ from original LOSO source audit: '+key)
        print('HRfunc original selection parity verified')
    else:
        print('Reference HRfunc source selection unavailable; the current selection is not an exact reproduction')
    write_json('SOURCE_ONLY_FITTED_HRFUNC.json',fitted.audit)
    pred_rows=[];metric_rows=[]
    variants=('HRFUNC_AUTHOR__NATIVE_THRESH','HRFUNC_AUTHOR__SOURCE_RIDGE_READOUT')
    for pid in sorted(hb.participant.unique()):
        print('HRFUNC',pid,flush=True)
        source,Hlist,Rlist=recreate_subject(m,pid,eligible,hb)
        metric_rows.extend(participant_original_hb(m,source))
        # A REST Bank satisfies the author's frozen FittedFamily.predict interface.
        from types import SimpleNamespace
        H=np.stack(Hlist);R=np.stack(Rlist)
        bank=SimpleNamespace(dataset='REST',participant=pid,split='test',seed=1,
            signature=sha(Path(eligible[int(pid[-3:])-1]['path']))+':rest_injection_v1',
            H=H,R=R,n=len(H),features=np.zeros((len(H),41,9)))
        outputs=fitted.predict(bank,cache)
        if set(outputs)!=set(variants):fail('HRfunc did not return two required variants')
        for name in variants:
            pp=outputs[name]
            b=np.asarray(pp['boundaries'],float);d=np.asarray(pp['decision'],float)
            score=np.asarray(pp['score'],float)
            if len(b)!=16 or b.shape!=(16,3) or d.shape!=(16,):fail('Wrong HRfunc prediction shapes')
            # Failures must stay missing, never be rewarded as correct negative predictions.
            gate=np.where(np.isfinite(d),d,0).astype(int)
            metric_rows.extend(participant_metrics(m,source,name,score,b,gate))
            for j in range(16):
                pred_rows.append({'participant':pid,'trial_index':j,'method':name,'score':score[j],
                      'decision':gate[j],'onset':b[j,0],'peak':b[j,1],'offset':b[j,2],
                      'truth_present':int(source.truth_present.iloc[j]),
                      'amplitude_in_baseline_sd':source.amplitude_in_baseline_sd.iloc[j],
                      'recording_sha256':source.recording_sha256.iloc[j]})
        # A completed participant CSV checkpoint makes interrupted native-kernel runs visible.
        pd.DataFrame([x for x in pred_rows if x['participant']==pid]).to_csv(OUT/f'{pid}_HRFUNC_CHECKPOINT.csv.gz',index=False,compression='gzip')
    summarize(pred_rows,metric_rows)
    # Apply Holm correction after all three method families are complete.
    # No external evaluation labels are used for calibration.
    others={
        'COHEN':OUT.parent/'cohen',
        'LINA':OUT.parent/'lina',
        'HRFUNC':OUT,
    }
    if all((path/'STATUS.json').exists() and
           json.loads((path/'STATUS.json').read_text()).get('status')=='COMPUTATION_COMPLETE'
           for path in others.values() if path!=OUT):
        frames=[];comparisons=[]
        for key,path in others.items():
            tab=pd.read_csv(path/'PARTICIPANT_METRICS.csv')
            tab=tab[tab.method!='HEMOBOUNDARY_FULL'] if key!='HRFUNC' else tab
            frames.append(tab)
            comparisons.append(pd.read_csv(path/'PAIRED_EXACT_SIGNFLIP_UNADJUSTED.csv'))
        merged=pd.concat(frames,ignore_index=True)
        c=pd.concat(comparisons,ignore_index=True)
        if c.groupby('gate').size().to_dict()!={'native':24,'shared_hb':24,'ungated_positive':24}:
            fail('Expected 6 published-method variants x 4 timing measures x 3 evaluation gates')
        for gate,ix in c.groupby('gate').groups.items():
            values=c.loc[ix,'two_sided_exact_signflip_p'].to_numpy(float)
            order=np.argsort(values,kind='mergesort');adjusted=np.empty_like(values);prev=0.
            for rank,j in enumerate(order):
                prev=max(prev,(len(values)-rank)*values[j]);adjusted[j]=min(1.,prev)
            c.loc[ix,'holm_p_within_6_method_4_outcome_gate_family']=adjusted
        merged.to_csv(OUT/'THREE_METHODS_ALL_PARTICIPANT_METRICS.csv',index=False)
        c.to_csv(OUT/'THREE_METHODS_FINAL_HOLM_CORRECTED.csv',index=False)
        print('\nTHREE-METHOD COMBINED HOLM RESULTS (NATIVE GATE):')
        print(c[c.gate=='native'][['method','metric','mean_advantage_hb_sec','two_sided_exact_signflip_p',
          'holm_p_within_6_method_4_outcome_gate_family']].to_string(index=False))
    else:
        print('THREE-METHOD HOLM MERGE PENDING: run Cohen-Adad and Lina notebooks, then rerun HRfunc.')



if __name__ == "__main__":
    try:
        print('BEGIN:',TAG,flush=True)
        morph=bootstrap()
        saved,subjects=get_saved()
        execute(morph,saved,subjects)
        write_json('STATUS.json',{'status':'COMPUTATION_COMPLETE','method':TAG,
          'n_independent_participants':14,'test_conditions':224,
          'adaptations_not_exact_published_paper_reproductions':True,
          'full_multiple_testing_family_holm':'PENDING_UNTIL_THREE_METHOD_BENCHMARKS_MERGED'})
    except Exception as err:
        print('ERROR:',type(err).__name__,str(err))
        traceback.print_exc()
        write_json('STATUS.json',{'status':'BLOCKED_OR_FAILED','method':TAG,
            'error':repr(err),'traceback':traceback.format_exc()})
        raise
    finally:
        results_zip()
