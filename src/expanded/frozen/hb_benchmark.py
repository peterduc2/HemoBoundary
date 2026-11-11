#!/usr/bin/env python3
"""Single orchestrator for the frozen HemoBoundary published-method comparison.

Run through the supplied isolated Colab launcher, or in a dedicated environment.
Default: SHIN LOSO plus BALINT/KHAN external transfer, 10 source seeds.
No HemoBoundary model is trained, no original file is modified, no missing input
is replaced with a different synthetic bank. Labels are used for replay auditing,
source-only calibration and final evaluation, not target-driven estimation.
"""
from __future__ import annotations
import argparse, contextlib, importlib.metadata, json, os, platform, sys, time, traceback
from pathlib import Path
from collections import defaultdict
import numpy as np
from hb_data import (PIDS,SEEDS,Bank,DataStore,locate_root,read_rows,write_rows,write_json,
                     digest_file,digest_json,number)
from hb_algorithms import (REGISTRY,CURVE_FAMILIES,VARIANTS,FRACTIONS,EstimateCache,
                           native_bounds,spm_kernel,hrfunc_channel_estimates,POST)
from hb_models import (fit_aide,fit_family,fit_control,CONTROL_METHODS,READOUT_ALPHAS,
                       TREE_LEAVES,TREE_COUNT,INNER_FOLDS)
from hb_evaluation import (HB,BOUNDARIES,detection_metrics,timing_metrics,common_valid_metrics,
                           stable_seed,summarize)

VERSION='HB_V8_MULTICHANNEL_REPLAY_PUBLISHED_METHOD_BENCHMARK_20261014'
DATASETS=('SHIN','BALINT','KHAN')
DEFAULT_OUT='HB_V8_MULTICHANNEL_PUBLISHED_COMPARISON'


def environment():
    packages={}
    for p in ('numpy','scipy','pandas','scikit-learn','h5py','mne','nilearn','hrfunc'):
        try:packages[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:packages[p]=None
    installed={str(d.metadata.get('Name',d.name)).lower():d.version for d in importlib.metadata.distributions()}
    return {'python':platform.python_version(),'packages':packages,'all_installed_distributions':dict(sorted(installed.items())),
            'thread_policy':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}}

def native_package_tests():
    """Execute, rather than simulate, the installed authors' Python API.
    This test is called in Colab; passing local tests alone is insufficient.
    """
    from nilearn.glm.first_level import spm_hrf,spm_time_derivative,spm_dispersion_derivative
    actual={}
    for kind,fun in (('canonical',spm_hrf),('temporal',spm_time_derivative),('dispersion',spm_dispersion_derivative)):
        _,ours=spm_kernel(kind)
        expected=np.asarray(fun(.5,oversampling=20,time_length=40.),float)
        if ours.shape!=expected.shape or not np.allclose(ours,expected,atol=1e-10,rtol=1e-8):
            raise RuntimeError(f'{kind}: SPM basis differs from installed Nilearn; stop before benchmarking')
        actual[kind+'_max_error']=float(np.max(np.abs(ours-expected)))
    rng=np.random.default_rng(492);t=np.arange(-5,20.01,.5)
    response=np.exp(-.5*((t-6.)/2.5)**2)*(t>=0)
    H=rng.normal(0,.02,(3,51))+np.array([1.,.8,1.2])[:,None]*response
    R=rng.normal(0,.02,(3,51))-.45*np.array([1.,.8,1.2])[:,None]*response
    curve,audit=hrfunc_channel_estimates(H,R)
    if curve.shape!=(41,):raise RuntimeError('Native HRfunc conformance failed')
    actual.update(audit);actual['hrfunc_version']=importlib.metadata.version('hrfunc')
    actual['test_input']='independent software fixture, not experimental data'
    actual['matlab_equivalence_verified']=False
    print('NATIVE PYTHON TESTS PASS: Nilearn SPM basis/derivatives and original HRfunc unpadded multichannel API',flush=True)
    return actual

def code_fingerprint():
    base=Path(__file__).parent
    return {p.name:digest_file(p) for p in sorted(base.glob('*.py'))}

def protocol(config,env):
    return {'version':VERSION,'code':code_fingerprint(),'environment':env,
      'datasets':list(config['datasets']),'source_seeds':list(config['seeds']),'source_participants':list(PIDS),
      'families':list(config['families']),'controls':list(CONTROL_METHODS),'aide_search':True,'percent_latency':True,
      'fractions':list(FRACTIONS),'variants':{f:list(VARIANTS[f]) for f in config['families']},
      'readout_alphas':list(READOUT_ALPHAS),'tree_leaves':list(TREE_LEAVES),'tree_count':TREE_COUNT,'source_group_folds':INNER_FOLDS,
      'primary_loss':'fixed 4-second penalty for missing predictions, finite errors NOT clipped',
      'additional_metrics':'conditional MAE, coverage, paired common-valid MAE, 2 and 8 second missing-loss sensitivity',
      'primary_timing_gate':'frozen HemoBoundary response decision; not independent end-to-end comparison',
      'independent_detection':'each new pipeline has its own SOURCE-calibrated response score or source-OOF logistic detector',
      'input_replay':'same original 2Hz multichannel generated trials with their original simulated noisy baseline',
      'no_zero_padding':True,'no_hb_retraining':True,'no_external_labels_in_selection':True,
      'native_and_learned_readouts_both_reported':True,'post_development_benchmark_not_preregistered':True,
      'synthetic_reference_primary_peak':'start of plateau from original generator; not necessarily unique observed maximum',
      'external_provenance_limitation':'original V102 did not save a raw feature bank; generator/profile QC and all saved labels are checked, not historical raw byte equality',
      'matlab_numerical_equivalence_established':False,
      'claims_not_supported':['general superiority over original continuous-recording HRfunc',
        'full AIDE iteration/peakfinder reproduction','full AnalyzIR covariance/inference reproduction',
        'exact biological trial timing accuracy','new tasks or untested unseen morphology robustness']}

def output_method_names(families):
    return {HB,'PINTI_AIDE_SEARCH','MAHEUX_PCT20','MAHEUX_PCT50',*CONTROL_METHODS,
            *(f+'__'+a for f in families for a in ('NATIVE_THRESH','SOURCE_RIDGE_READOUT'))}

def numeric_rows(path):
    out=[]
    nonnumeric={'dataset','participant','method','mode','supported'}
    for r in read_rows(path):
        out.append({k:(v if k in nonnumeric else int(number(v)) if k=='seed' else number(v)) for k,v in r.items()})
    return out

def record_predictions(bank,seed,predictions,hb_pred,hb_gate):
    base={'dataset':bank.dataset,'seed':int(seed),'participant':bank.participant};metrics=[];trials=[]
    all_predictions={HB:{'boundaries':hb_pred,'decision':hb_gate.astype(float),'score':np.full(bank.n,np.nan),'supported':BOUNDARIES},**predictions}
    for method,p in all_predictions.items():
        bounds=p['boundaries'];decision=p['decision'];supported=p['supported'];score=p['score']
        identity={**base,'method':method,'supported':','.join(supported)}
        if decision is not None:
            dm=detection_metrics(bank.truth['present'],decision)
            metrics.append({**identity,'mode':'independent_detection',**dm})
        if bounds is not None and supported:
            for mode,gate in (('shared_gate_timing',hb_gate),('own_gate_timing',decision)):
                if gate is None:continue
                tm=timing_metrics(bank.truth,bounds,gate,supported)
                if mode=='shared_gate_timing':tm.update(common_valid_metrics(bank.truth,bounds,hb_pred,hb_gate,supported))
                metrics.append({**identity,'mode':mode,**tm})
        for i in range(bank.n):
            b=np.full(3,np.nan) if bounds is None else bounds[i]
            trials.append({**base,'method':method,'synthetic_index':i,
                **{'truth_'+k:float(bank.truth[k][i]) for k in ('present','onset','peak','offset','censored')},
                'pred_response':float(decision[i]) if decision is not None else np.nan,
                'saved_hb_response_gate':int(hb_gate[i]),'score':float(score[i]) if score is not None else np.nan,
                'pred_onset':float(b[0]),'pred_peak':float(b[1]),'pred_offset':float(b[2]),
                'supported':','.join(supported),'bank_signature':bank.signature})
    return metrics,trials

def check_completed(path,key,expected_methods):
    stamp=path/'CHECKPOINT.json'
    if not stamp.is_file():return None
    try:
        m=json.loads(stamp.read_text())
        if m.get('key')!=key:return None
        for filename,sha in m['files'].items():
            if not (path/filename).is_file() or digest_file(path/filename)!=sha:return None
        ia=json.loads((path/'INPUT_AUDIT.json').read_text())
        audit_path=path.parents[3]/ia['selected_model_audit']
        if not audit_path.is_file() or digest_file(audit_path)!=ia['selected_model_audit_sha256']:return None
        rows=numeric_rows(path/'metrics.csv')
        if set(r['method'] for r in rows)!=expected_methods:return None
        return rows
    except (OSError,ValueError,KeyError):return None

def benchmark_group(source,targets,seed,group_name,excluded,cache,out,protocol_sha,source_pids=None,families=CURVE_FAMILIES):
    source_pids=tuple(sorted(source)) if source_pids is None else tuple(source_pids)
    if excluded in source:raise AssertionError('Target leaked into source fitter')
    expected_methods=output_method_names(families);needed=[];reused=[]
    source_key={p:source[p].signature for p in source_pids}
    for bank,hb_pred,hb_gate,hb_sha in targets:
        cp=out/'checkpoints'/bank.dataset/f'seed_{seed:02d}'/bank.participant
        key=digest_json({'protocol':protocol_sha,'sources':source_key,'bank':bank.signature,'hb_predictions':hb_sha,'seed':seed})
        previous=check_completed(cp,key,expected_methods)
        if previous is not None:
            print(f'{bank.dataset} seed {seed:02d} {bank.participant}: reused content-verified V8 checkpoint',flush=True);reused.extend(previous)
        else:needed.append((bank,hb_pred,hb_gate,hb_sha,cp,key))
    if not needed:return reused
    print(f'Seed {seed:02d} {group_name}: fitting on {len(source)} Shin source participants; {len(needed)} evaluation banks',flush=True)
    fit_seed=stable_seed(VERSION,seed,group_name,'same_group_folds_all_methods')
    fitted=[fit_aide(source,cache,excluded)]
    for family in families:
        print(f'  source selection: {family}',flush=True)
        fitted.append(fit_family(family,source,cache,excluded,fit_seed))
    for control in CONTROL_METHODS:
        print(f'  source selection: {control}',flush=True)
        fitted.append(fit_control(control,source,excluded,fit_seed))
    group_audit=out/'selection_audits'/f'seed_{seed:02d}'/(group_name+'.json')
    write_json(group_audit,{'seed':seed,'group':group_name,'excluded_target':excluded,'source_participants':list(source_pids),'models':[f.audit for f in fitted]})
    rows=list(reused)
    for bank,hb_pred,hb_gate,hb_sha,cp,key in needed:
        predictions={}
        for f in fitted:predictions.update(f.predict(bank,cache))
        # Published percentage-of-peak latency measure only; no invented recovery
        # output and no claim to have run the paper's jackknife group procedure.
        raw=.5*(np.median(bank.H[:,:,POST],axis=1)-np.median(bank.R[:,:,POST],axis=1))
        for method,fraction in (('MAHEUX_PCT20',.2),('MAHEUX_PCT50',.5)):
            bound=np.stack([native_bounds(c,fraction) for c in raw]);bound[:,1:]=np.nan
            predictions[method]={'boundaries':bound,'decision':None,'score':None,'supported':('onset',),'wrapper_detection':False}
        metrics,trials=record_predictions(bank,seed,predictions,hb_pred,hb_gate)
        if set(r['method'] for r in metrics)!=expected_methods:raise AssertionError('Missing method output; not a complete benchmark')
        write_rows(cp/'metrics.csv',metrics);write_rows(cp/'trials.csv.gz',trials)
        write_json(cp/'INPUT_AUDIT.json',{'bank_signature':bank.signature,'bank_audit':bank.audit,'original_hb_sha256':hb_sha,'source_signatures':source_key,
                                        'selected_model_audit':str(group_audit.relative_to(out)),'selected_model_audit_sha256':digest_file(group_audit)})
        files={name:digest_file(cp/name) for name in ('metrics.csv','trials.csv.gz','INPUT_AUDIT.json')}
        write_json(cp/'CHECKPOINT.json',{'key':key,'protocol_sha256':protocol_sha,'files':files,'complete':True})
        rows.extend(metrics)
        print(f'{bank.dataset} seed {seed:02d} {bank.participant}: all {len(expected_methods)} declared methods/adapters completed',flush=True)
    return rows

def run_benchmark(root,out,mode='full',datasets=DATASETS,families=CURVE_FAMILIES,native_checks=True):
    seeds=SEEDS if mode=='full' else (1,)
    source_pids=PIDS
    target_pids=PIDS if mode!='pilot' else ('VP001',)
    root=Path(root);out=Path(out)
    protected=[root,root/'02_preprocessed',*[root/n for n in ('95_N4_FUNCTIONAL_ABC_LBR_FULLY_NESTED_10SEEDS_V1','98_B0_STRICT_HIERARCHICAL_SEED01_FULLY_NESTED_V1','99_B0_STRICT_HIERARCHICAL_10SEEDS_V1','102_OPTIONB_TABLE3_10SEED_BALINT_KHAN_ZERO_SHOT_V1')]]
    for p in protected:
        if out.resolve()==p.resolve() or (p!=root and p.resolve() in out.resolve().parents):raise ValueError('Output cannot overwrite an original experiment directory')
    out.mkdir(parents=True,exist_ok=True)
    # Shared notebook runs are not safe to write simultaneously into one output.
    import fcntl
    lock=(out/'.run.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another V8 process is writing this output; use that run or another output directory')
    env=environment();config={'datasets':datasets,'seeds':SEEDS,'families':families}
    # Mode changes pilot->full do not change algorithms or candidate settings.
    prot=protocol(config,env);psha=digest_json(prot)
    pp=out/'PROTOCOL.json'
    if pp.is_file() and digest_json(json.loads(pp.read_text()))!=psha:
        raise RuntimeError('The output belongs to different code/settings/packages. Use a NEW --output directory; stale experimental results are not reused.')
    write_json(pp,prot);write_json(out/'METHOD_REGISTRY.json',REGISTRY)
    store=DataStore(root,out,seeds,datasets,source_pids);store.preflight()
    if native_checks:write_json(out/'NATIVE_PACKAGE_TESTS.json',native_package_tests())
    else:write_json(out/'NATIVE_PACKAGE_TESTS.json',{'status':'not_run_TEST_FIXTURE_ONLY','paper_use_allowed':False})
    if mode=='preflight':
        first=store.shin(1,'VP001','train');store.shin(1,'VP001','test')
        for ds in datasets:
            if ds!='SHIN':store.external(ds,store.external_ids(ds)[0])
        print('PRE-FLIGHT and first-bank original-data parity checks complete. No benchmark results generated.');return []
    cache=EstimateCache(out,env);metrics=[];expected=[]
    for seed in seeds:
        training={p:store.shin(seed,p,'train') for p in source_pids}
        print(f'Original seed {seed:02d} source banks replayed with label/feature parity. HemoBoundary remains read-only.',flush=True)
        if 'SHIN' in datasets:
            for pid in target_pids:
                bank=store.shin(seed,pid,'test');hb_pred,hb_gate,hb_sha=store.saved(bank,seed)
                source={p:b for p,b in training.items() if p!=pid}
                metrics.extend(benchmark_group(source,[(bank,hb_pred,hb_gate,hb_sha)],seed,'LOSO_'+pid,pid,cache,out,psha,families=families))
                expected.append(('SHIN',seed,pid))
        external_targets=[]
        for ds in datasets:
            if ds=='SHIN':continue
            ids=store.external_ids(ds)
            if len(ids)!={'BALINT':26,'KHAN':25}[ds]:raise ValueError(f'{ds}: unexpected participant count. Do not reuse recovery-only exclusions for timing validation.')
            if mode=='pilot':ids=ids[:1]
            for pid in ids:
                bank=store.external(ds,pid);hp,hg,hs=store.saved(bank,seed)
                external_targets.append((bank,hp,hg,hs));expected.append((ds,seed,pid))
        if external_targets:
            metrics.extend(benchmark_group(training,external_targets,seed,'ALL_SHIN_TO_EXTERNAL',None,cache,out,psha,families=families))
        write_rows(out/'PARTICIPANT_METRICS_PROGRESS.csv',metrics)
        print(f'Seed {seed:02d} done. {len(expected)} dataset/seed/participant cases complete.',flush=True)
        cache.memory.clear()
    full=mode=='full' and set(seeds)==set(SEEDS) and all(sum(e[0]==ds for e in expected)==10*{'SHIN':26,'BALINT':26,'KHAN':25}[ds] for ds in datasets)
    summary,comparisons=summarize(metrics,out,expected,seeds,pvalues=full)
    audit={'status':'COMPUTATION_COMPLETE' if full else 'PILOT_OR_SEED1_COMPLETE','full_requested_coverage':full,
           'evaluated_datasets':list(datasets),'n_dataset_seed_participant_cases':len(expected),'n_source_seeds':len(seeds),
           'hb_retrained':False,'no_zero_padding':True,'sources_exclude_outer_target':True,'external_supervision':False,
           'native_tests':native_checks,'matlab_parity':False,'experiment_not_preregistered':True,
           'paper_claim':'Only the explicitly named implementations and adapters on the declared semi-synthetic banks, NOT generic SOTA superiority.',
           'external_raw_hash_provenance':'Original historical raw hash unavailable; replay source/QC/truth audits supplied',
           'registry':'METHOD_REGISTRY.json','protocol_sha256':psha,
           'statistical_unit':'participant after source-seed averaging; inferential dependence from overlapping training sets remains a limitation',
           'next_required_review':'Inspect error/coverage/failure/censoring, native package tests, and method/input/target compatibility before manuscript use.'}
    write_json(out/'AUDIT.json',audit)
    print('\nDONE:',out,'\nNo automated PASS constitutes scientific certification. See AUDIT.json and METHOD_REGISTRY.json.')
    return metrics

def recover_summary(out):
    """Recovery is read-only on checkpoints, uses csv/numpy, no pandas reshape."""
    out=Path(out);prot=json.loads((out/'PROTOCOL.json').read_text());psha=digest_json(prot)
    expected_methods=output_method_names(prot['families']);rows=[];expected=[]
    for ds in prot['datasets']:
        for seed in SEEDS:
            parent=out/'checkpoints'/ds/f'seed_{seed:02d}'
            count=0
            for folder in sorted(parent.iterdir()) if parent.is_dir() else []:
                if not folder.is_dir():continue
                stamp=folder/'CHECKPOINT.json'
                if not stamp.is_file():continue
                m=json.loads(stamp.read_text())
                if m['protocol_sha256']!=psha:raise ValueError('Mixed protocols in finalization')
                q=check_completed(folder,m['key'],expected_methods)
                if q is None:raise ValueError(f'Damaged checkpoint {folder}')
                if set((r['dataset'],int(r['seed']),r['participant']) for r in q)!={(ds,seed,folder.name)}:raise ValueError('Checkpoint identity mismatch')
                rows.extend(q);expected.append((ds,seed,folder.name));count+=1
            if count!={'SHIN':26,'BALINT':26,'KHAN':25}[ds]:raise ValueError(f'{ds} seed {seed}: found {count}, not full declared coverage')
    return summarize(rows,out,expected,SEEDS,pvalues=True)

def cli():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('full','seed1','pilot','preflight','summarize'),default='full')
    parser.add_argument('--datasets',default='SHIN,BALINT,KHAN')
    parser.add_argument('--root');parser.add_argument('--output')
    args=parser.parse_args()
    ds=tuple(args.datasets.upper().split(','))
    if not ds or len(set(ds))!=len(ds) or not set(ds).issubset(DATASETS):parser.error('datasets must be a unique comma-separated subset of SHIN,BALINT,KHAN')
    args.datasets=ds;return args

def main():
    args=cli()
    root=locate_root(args.root);out=Path(args.output) if args.output else root/DEFAULT_OUT
    if args.mode=='summarize':recover_summary(out)
    else:run_benchmark(root,out,args.mode,args.datasets)

if __name__=='__main__':
    main()
