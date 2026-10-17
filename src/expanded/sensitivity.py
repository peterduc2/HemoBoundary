#!/usr/bin/env python3
"""HemoBoundary V8 — three read-only sensitivity checks from SAVED PREDICTIONS.

Checks:
  1. Conditional MAE, separately for all-valid and pairwise common-valid trials.
  2. Prediction coverage.
  3. Missing-estimate penalty sensitivity at 2, 4 and 6 seconds (8-second audit).

All 770 dataset/seed/participant identities are required. Eight comparators + HB.

Inputs are ORIGINAL checkpoint predictions on Google Drive, never rerun models:
  V8 trial-level trials.csv.gz + metrics.csv, V1 seeds01-05/V2 seeds06-10.
  Cohen-Adad/Lina PREDICTIONS.npz + CASE_METRICS.json.

Replays exactly the scoring masks from frozen hb_evaluation.py and confirms the
saved 4-second case metrics before deriving 2/6-second loss. Finite errors are
uncapped, and right-censored reference recovery is excluded from exact offset and
duration scoring. All paired inference uses 77 biological participants (26/26/25),
with source-seed results combined WITHIN participants before inference.

- NO training, data regeneration, cache edits or checkpoint writes.
- Source checkpoints verified against their saved checksums and science protocol.
- Output written only to a NEW timestamped folder.

This program DOES NOT claim that its author-model adaptations reproduce native
published methods. Post-development sensitivity tests are exploratory.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import shutil
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np

ROOT=Path('/content/gdrive/MyDrive/HonorProject/shin2018_word_generation_state_preparation_v1')
V8_V1=ROOT/'HB_V8_MULTICHANNEL_PUBLISHED_COMPARISON'
V8_V2=ROOT/'HB_V8_CPU_ONLY_SEEDS_06_TO_10'
V8_EXPORT=ROOT/'HB_V8_V1_V2_VERIFIED_770_EXPORT_20261015T041547Z_6190f376'
COHEN=ROOT/'HB_COHENADAD2007_SHIFT_GLM_PAPER_GUIDED_770_V1'
LINA=ROOT/'HB_LINA2010_WAVELET_HRF_PAPER_GUIDED_770_V1'
PRIOR=ROOT/'HB_BSPC_TABLE4_PAIRED_STATS_NEW_ONLY'/'BSPC_TABLE4_PAIRED_770_20261015T174429879164Z'/'PAIRED_ALL_120_HOLM.csv'
OUT_PARENT=ROOT/'HB_V8_THREE_CHECKS_READONLY_NEW_ONLY'
PROTO_SHA='6190f376593bb9befc92427bf2efbf70410194215e3dd60baacf325fd3864d55'
COHEN_CODE_SHA='f238bbc389faca02cc3a67452805d6b69b0c932b1abb30734759af26b596998c'
LINA_CODE_SHA='0b2b4b65312246acacf72e9cdc230b35851f010819e104f7fc3ff463ff99b59d'
COHEN_PROTO_SHA='1aa06f4e0946b75a8168ccd2dc6c364c42e44b34b6e345ee5108e9e7bc737029'
LINA_PROTO_SHA='5a24e45c21d0109c39389ff98579240fc55cb08c3c4e909776c6fd85e59cdcb9'
DATASETS=('SHIN','BALINT','KHAN')
PIDS={'SHIN':tuple(f'VP{i:03d}' for i in range(1,27)),
      'BALINT':tuple(f'C{i:02d}' for i in range(1,27)),
      'KHAN':tuple(f'S{i:02d}' for i in range(1,26))}
SEEDS=tuple(range(1,11)); BOUNDARIES=('onset','peak','offset','duration')
PENALTIES=(2,4,6); AUDIT_PENALTY=8
HB='HEMOBOUNDARY_STRICT_SAVED'
EXTRA='EXTRATREES9_CONTROL'; RIDGE='RIDGE9_CONTROL'
HR_NATIVE='HRFUNC_AUTHOR__NATIVE_THRESH'
HR_RIDGE='HRFUNC_AUTHOR__SOURCE_RIDGE_READOUT'
CO_SHIFT='COHENADAD2007_SHIFT_GLM_V8_ADAPTER'
CO_DERIV='COHENADAD2007_DERIVATIVE_GLM_V8_ADAPTER'
LI_PCA='LINA2010_WAVELET_PCA1_V8_ADAPTER'
LI_NOPCA='LINA2010_WAVELET_NOPCA_V8_ADAPTER'
METHODS=(HB,EXTRA,RIDGE,CO_SHIFT,CO_DERIV,LI_PCA,LI_NOPCA,HR_NATIVE,HR_RIDGE)
V8_METHODS=(HB,EXTRA,RIDGE,HR_NATIVE,HR_RIDGE)
DISPLAY={HB:'HemoBoundary',EXTRA:'ExtraTrees',RIDGE:'Ridge',
    CO_SHIFT:'Cohen-Adad shift GLM',CO_DERIV:'Cohen-Adad derivative GLM',
    LI_PCA:'Lina wavelet + PCA',LI_NOPCA:'Lina wavelet no PCA',
    HR_NATIVE:'HRfunc V8 native threshold',HR_RIDGE:'HRfunc V8 source Ridge'}
SIGN_FLIPS=100_000; BOOTSTRAPS=20_000; RANDOM_SEED=20261016
EXPECTED={(ds,seed,pid) for ds in DATASETS for seed in SEEDS for pid in PIDS[ds]}


def fail(message):raise RuntimeError(message)

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(2*1024*1024),b''):h.update(part)
    return h.hexdigest()

def json_hash(obj):
    b=json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=True,
                 allow_nan=False).encode('utf-8')
    return hashlib.sha256(b).hexdigest()

def load_json(p):
    if not Path(p).is_file():raise FileNotFoundError('Required input is absent: '+str(p))
    return json.loads(Path(p).read_text(encoding='utf-8'))

def read_csv(p):
    if not Path(p).is_file():raise FileNotFoundError('Required input is absent: '+str(p))
    if str(p).endswith('.gz'):
        with gzip.open(p,'rt',newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
    with Path(p).open('r',newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def save_csv(p,rows):
    rows=list(rows)
    if not rows:raise RuntimeError('Refuse writing empty CSV: '+str(p))
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader();w.writerows(rows)

def fnum(x):
    if x is None:return float('nan')
    try:return float(x)
    except (ValueError,TypeError):return float('nan')

def close(a,b,ctx,tol=1e-8):
    av,bv=fnum(a),fnum(b)
    if math.isnan(av) and math.isnan(bv):return
    if not (math.isfinite(av) and math.isfinite(bv)):
        fail('Nonmatching NaN/Inf: '+ctx+' '+repr((a,b)))
    if not math.isclose(av,bv,rel_tol=tol,abs_tol=tol):
        fail('Saved-prediction replay mismatch: '+ctx+' '+repr((av,bv)))

def check_source_checkpoint(cp,required,expected_proto=None):
    marker=load_json(cp/'CHECKPOINT.json')
    if marker.get('complete') is not True:fail('Incomplete checkpoint '+str(cp))
    if expected_proto is not None:
        if marker.get('protocol_sha256')!=expected_proto:
            fail('Wrong V8 saved-science protocol: '+str(cp))
    if not isinstance(marker.get('files'),dict):fail('Missing file manifest '+str(cp))
    if not set(required).issubset(marker['files']):fail('Missing required saved file hash '+str(cp))
    for name,expected_sha in marker['files'].items():
        p=cp/name
        if not p.is_file() or sha_file(p)!=expected_sha:
            fail('Content hash mismatch, refusing corrupted checkpoint: '+str(p))
    return marker


def verify_root():
    if not ROOT.is_dir():
        try:
            from google.colab import drive
            drive.mount('/content/gdrive')
        except ImportError:pass
    if not ROOT.is_dir():
        fail('This runtime has no access to the original Google Drive project: '+str(ROOT))
    v1=load_json(V8_V1/'PROTOCOL.json')
    v2=load_json(V8_V2/'PROTOCOL.json')
    if v1!=v2 or json_hash(v1)!=PROTO_SHA:fail('Frozen V8 V1 and V2 protocols differ')
    audit=load_json(V8_EXPORT/'FINAL_DERIVED_AUDIT.json')
    expected={'saved_checkpoint_count':770,'unique_checkpoint_count':770,
              'protocol_sha256':PROTO_SHA,'n_source_seeds':10}
    for k,v in expected.items():
        if audit.get(k)!=v:fail('Original V8 770 verified export mismatch: '+k)
    manifest_path=V8_EXPORT/'CHECKPOINT_MANIFEST_770.csv'
    if sha_file(manifest_path)!=audit.get('derived_export_file_sha256',{}).get('CHECKPOINT_MANIFEST_770.csv'):
        fail('770-case source manifest checksum mismatch')
    manifest=read_csv(manifest_path)
    keyed={}
    for row in manifest:
        key=(row['dataset'],int(row['seed']),row['participant'])
        if key in keyed:fail('Duplicate saved V8 case in manifest: '+repr(key))
        keyed[key]=row
    if set(keyed)!=EXPECTED:fail('Incomplete original 770-case manifest')
    provenance={'V8_protocol_sha256':PROTO_SHA,'V8_manifest_sha256':sha_file(manifest_path)}
    for name,folder,code_sha,proto_sha in (
        ('Cohen',COHEN,COHEN_CODE_SHA,COHEN_PROTO_SHA),
        ('Lina',LINA,LINA_CODE_SHA,LINA_PROTO_SHA)):
        p=load_json(folder/'PROTOCOL.json')
        if p.get('v8_protocol_sha256')!=PROTO_SHA or p.get('code_sha256')!=code_sha:
            fail(name+' scientific protocol mismatches saved code')
        if json_hash(p)!=proto_sha:fail(name+' original scientific protocol hash mismatched')
        summary=load_json(folder/'FULL_RUN_SUMMARY.json')
        if (summary.get('complete') is not True or summary.get('n_verified_cases')!=770
            or summary.get('expected_cases')!=770):fail(name+' original full summary incomplete')
        provenance[name+'_protocol_sha256']=proto_sha
    return keyed,provenance


def checkpoint_for(ds,seed,pid):
    folder=V8_V1 if seed<=5 else V8_V2
    return folder/'checkpoints'/ds/f'seed_{seed:02d}'/pid

def scored_timing(truth,pred,gate):
    """Exact replay of frozen hb_evaluation.outcome_arrays + timing_metrics.

    Outputs all four boundaries. Use SAME frozen shared HemoBoundary gate for
    every method, regardless of independent detector decisions.
    """
    n=len(truth['present'])
    p=np.asarray(pred,float)
    g=np.asarray(gate,float)
    if p.shape!=(n,3) or g.shape!=(n,):fail('Wrong prediction/gate shape')
    positive=np.asarray(truth['present'],int)==1
    censored=np.asarray(truth['censored'],int)==1
    reference={k:np.asarray(truth[k],float) for k in BOUNDARIES[:3]}
    reference['duration']=reference['offset']-reference['onset']
    estimates={k:p[:,i] for i,k in enumerate(BOUNDARIES[:3])}
    estimates['duration']=p[:,2]-p[:,0]
    mask={k:positive & np.isfinite(reference[k]) & (~censored if k=='offset' else True)
          for k in BOUNDARIES[:3]}
    mask['duration']=mask['onset'] & mask['offset']
    valid={k:mask[k]&(g==1)&np.isfinite(estimates[k]) for k in BOUNDARIES}
    bad_onpeak=np.isfinite(p[:,0])&np.isfinite(p[:,1])&(p[:,0]>=p[:,1])
    bad_peakoff=np.isfinite(p[:,1])&np.isfinite(p[:,2])&(p[:,1]>=p[:,2])
    for k in ('onset','peak'):valid[k]&=~bad_onpeak
    for k in ('offset','duration'):valid[k]&=~bad_peakoff & ~bad_onpeak
    for k in BOUNDARIES[:3]:valid[k]&=(estimates[k]>=0)&(estimates[k]<=20)
    valid['duration'] &= valid['onset'] & valid['offset']
    answer={}
    for k in BOUNDARIES:
        m=mask[k];v=valid[k];n_eligible=int(m.sum());n_valid=int(v.sum())
        if n_eligible<=0:fail('Zero eligible trials for '+k)
        errors=np.abs(estimates[k]-reference[k])
        if n_valid and not np.isfinite(errors[v]).all():fail('Valid error is nonfinite')
        mean_valid=float(errors[v].mean()) if n_valid else float('nan')
        d={'eligible_count':n_eligible,'valid_count':n_valid,
           'valid_sum_err':float(errors[v].sum()),
           'coverage':n_valid/n_eligible,
           'conditional_mae':mean_valid}
        for penalty in (*PENALTIES,AUDIT_PENALTY):
            d[f'loss_{penalty}']=(float(errors[v].sum()) + penalty*(n_eligible-n_valid))/n_eligible
        answer[k]={'summary':d,'valid':v,'eligible':m,'errors':errors}
    return answer


def read_v8_trial_predictions(cp,ds,seed,pid):
    raw=read_csv(cp/'trials.csv.gz')
    group=defaultdict(dict)
    for r in raw:
        if r.get('dataset')!=ds or int(r['seed'])!=seed or r.get('participant')!=pid:
            fail('Wrong V8 trial case identity: '+repr((ds,seed,pid)))
        meth=r.get('method')
        if meth not in V8_METHODS:continue
        j=int(r['synthetic_index'])
        if j in group[meth]:fail('Duplicate trial index '+repr((ds,seed,pid,meth,j)))
        group[meth][j]=r
    if set(group)!=set(V8_METHODS):fail('Missing saved V8 prediction method: '+repr((ds,seed,pid)))
    for method,rows in group.items():
        if set(rows)!=set(range(36)):
            fail('Incomplete original 36-trial prediction bank '+repr((ds,seed,pid,method)))
    h=[group[HB][i] for i in range(36)]
    truth={k:np.array([fnum(t['truth_'+k]) for t in h],float)
           for k in ('present','onset','peak','offset','censored')}
    if not set(truth['present']).issubset({0.,1.}):fail('Invalid ground-truth response labels')
    if not set(truth['censored']).issubset({0.,1.}):fail('Invalid censoring labels')
    if int(np.sum(truth['present']==1))!=24 or int(np.sum(truth['present']==0))!=12:
        fail('Wrong 24-positive/12-negative trial bank')
    gate=np.array([fnum(t['saved_hb_response_gate']) for t in h],float)
    if not set(gate).issubset({0.,1.}):fail('Invalid frozen response gate')
    bank_sigs={r.get('bank_signature') for rows in group.values() for r in rows.values()}
    if len(bank_sigs)!=1:fail('Method predictions do not refer to same trial bank')
    sig=next(iter(bank_sigs))
    prediction={}
    for method,rows in group.items():
        rr=[rows[i] for i in range(36)]
        # Verify order, truth and shared gate are equal for ALL selected V8 methods.
        for j,row in enumerate(rr):
            for k in truth:
                a=fnum(row['truth_'+k]);b=truth[k][j]
                if not (np.isnan(a) and np.isnan(b)) and not a==b:
                    fail('Different reference label across original V8 methods')
            if fnum(row['saved_hb_response_gate'])!=gate[j]:
                fail('Different frozen response gate across methods')
        prediction[method]=np.array([[fnum(row['pred_'+k]) for k in ('onset','peak','offset')]
                                    for row in rr],float)
    return truth,gate,sig,prediction


def load_case_metrics_v8(cp):
    out={}
    for r in read_csv(cp/'metrics.csv'):
        if r.get('mode')!='shared_gate_timing' or r.get('method') not in V8_METHODS:continue
        if r['method'] in out:fail('Duplicate original timing metric row')
        out[r['method']]=r
    if set(out)!=set(V8_METHODS):fail('Missing original V8 timing metric row')
    return out


def check_saved_timing(score,saved,method,ds,seed,pid):
    for b in BOUNDARIES:
        d=score[b]['summary'];ctx=f'{ds} seed {seed} {pid} {method} {b}'
        for label,calc in ((f'{b}_loss_missing4_sec',d['loss_4']),
                           (f'{b}_coverage',d['coverage']),
                           (f'{b}_mae_valid_sec',d['conditional_mae'])):
            if label not in saved:fail('Missing scoring metric '+label+' '+ctx)
            close(saved.get(label),calc,ctx+' '+label)
        for p in (2,8):
            label=f'{b}_loss_missing{p}_sec'
            if label in saved and saved[label] not in ('',None):
                close(saved[label],d[f'loss_{p}'],ctx+' '+label)
        close(d['loss_2'],d['loss_4']-2*(1-d['coverage']),ctx+' penalty=2')
        close(d['loss_6'],d['loss_4']+2*(1-d['coverage']),ctx+' penalty=6')


def replay_case(ds,seed,pid,manifest_row):
    v8cp=checkpoint_for(ds,seed,pid)
    # Verify original V8 checkpoint and historic verified 770-case manifest.
    cp_marker=check_source_checkpoint(v8cp,('metrics.csv','trials.csv.gz','INPUT_AUDIT.json'),PROTO_SHA)
    for k,filename in (('checkpoint_sha256','CHECKPOINT.json'),('metrics_sha256','metrics.csv'),
                       ('trials_gz_sha256','trials.csv.gz'),('input_audit_sha256','INPUT_AUDIT.json')):
        if sha_file(v8cp/filename)!=manifest_row[k]:
            fail('Original verified manifest hash failed '+str(v8cp/filename))
    va=load_json(v8cp/'INPUT_AUDIT.json')
    truth,gate,signature,predictions=read_v8_trial_predictions(v8cp,ds,seed,pid)
    if signature!=va.get('bank_signature'):
        fail('V8 trial bank signature differs from its own input audit')
    saved_metrics=load_case_metrics_v8(v8cp)
    saved_scoring={}
    for m in V8_METHODS:
        v=scored_timing(truth,predictions[m],gate)
        check_saved_timing(v,saved_metrics[m],m,ds,seed,pid)
        saved_scoring[m]=v
    for directory,methods in ((COHEN,(CO_SHIFT,CO_DERIV)),(LINA,(LI_PCA,LI_NOPCA))):
        cp=directory/'checkpoints'/ds/f'seed_{seed:02d}'/pid
        check_source_checkpoint(cp,('PREDICTIONS.npz','CASE_METRICS.json','INPUT_AUDIT.json'))
        ca=load_json(cp/'INPUT_AUDIT.json')
        if ca.get('bank_signature')!=signature:
            fail('Published-method adapter evaluated a different bank '+str(cp))
        if ca.get('saved_hb_prediction_sha256')!=va.get('original_hb_sha256'):
            fail('Published-method adapter used a different frozen HB prediction '+str(cp))
        raw=load_json(cp/'CASE_METRICS.json')
        if (raw.get('dataset'),int(raw['seed']),raw.get('participant'))!=(ds,seed,pid):
            fail('Saved adapter metric identity mismatch '+str(cp))
        metric_rows={r['method']:r for r in raw['metrics'] if r['mode']=='shared_gate_timing'}
        if set(metric_rows)!=set(methods):fail('Unexpected adapter scoring registry '+str(cp))
        with np.load(cp/'PREDICTIONS.npz',allow_pickle=False) as f:
            for name,index in zip(methods,('pred0','pred1')):
                p=np.asarray(f[index],float)
                if p.shape!=(36,3):fail('Adapter prediction shape mismatch '+str(cp))
                replay=scored_timing(truth,p,gate)
                check_saved_timing(replay,metric_rows[name],name,ds,seed,pid)
                saved_scoring[name]=replay
    if set(saved_scoring)!=set(METHODS):fail('Unreported comparison method in replay')
    ret=[]
    for method in METHODS:
        for b in BOUNDARIES:
            s=saved_scoring[method][b]['summary']
            ret.append({'dataset':ds,'seed':seed,'participant':pid,'method':method,
                        'boundary':b,'n_eligible':s['eligible_count'],'n_valid':s['valid_count'],
                        'valid_abs_error_sum':s['valid_sum_err'],
                        'coverage':s['coverage'],'conditional_mae':s['conditional_mae'],
                        **{f'loss_{p}s':s[f'loss_{p}'] for p in (*PENALTIES,AUDIT_PENALTY)}})
    commons=[]
    for method in METHODS[1:]:
        for b in BOUNDARIES:
            h=saved_scoring[HB][b]
            c=saved_scoring[method][b]
            if not np.array_equal(h['eligible'],c['eligible']):fail('Mismatched eligible reference masks')
            common=h['valid']&c['valid']
            eh=float(h['errors'][common].sum())
            ec=float(c['errors'][common].sum())
            commons.append({'dataset':ds,'seed':seed,'participant':pid,'method':method,
                            'boundary':b,'common_n_valid':int(common.sum()),
                            'common_abs_error_hb_sum':eh,'common_abs_error_comp_sum':ec})
    return ret,commons


def grouped_participants(case_rows,common_rows):
    cells=defaultdict(list)
    for r in case_rows:
        cells[(r['dataset'],r['participant'],r['method'],r['boundary'])].append(r)
    expected={(ds,pid,m,b) for ds in DATASETS for pid in PIDS[ds] for m in METHODS for b in BOUNDARIES}
    if set(cells)!=expected:fail('Saved-replay participant coverage incomplete')
    subjects=[]
    for key,rr in sorted(cells.items()):
        if len(rr)!=10 or {r['seed'] for r in rr}!=set(SEEDS):
            fail('Did not observe all 10 source seeds for '+repr(key))
        eligible=sum(r['n_eligible'] for r in rr)
        valid=sum(r['n_valid'] for r in rr)
        d={'dataset':key[0],'participant':key[1],'method':key[2],
           'display_method':DISPLAY[key[2]],'boundary':key[3],
           'n_source_seeds':10,'n_eligible_trials':eligible,'n_valid_trials':valid,
           'coverage_mean_seed':float(np.mean([r['coverage'] for r in rr])),
           'coverage_pooled':valid/eligible if eligible else float('nan'),
           'conditional_mae_pooled':sum(r['valid_abs_error_sum'] for r in rr)/valid if valid else float('nan'),
           'conditional_mae_mean_finite_seed':float(np.mean([r['conditional_mae'] for r in rr
                            if np.isfinite(r['conditional_mae'])])) if any(np.isfinite(r['conditional_mae']) for r in rr) else float('nan'),
           'seeds_with_valid_predictions':sum(r['n_valid']>0 for r in rr),
           **{f'loss_{p}s':float(np.mean([r[f'loss_{p}s'] for r in rr]))
              for p in (*PENALTIES,AUDIT_PENALTY)}}
        subjects.append(d)
    cm=defaultdict(list)
    for r in common_rows:cm[(r['dataset'],r['participant'],r['method'],r['boundary'])].append(r)
    must={(ds,pid,m,b) for ds in DATASETS for pid in PIDS[ds] for m in METHODS[1:] for b in BOUNDARIES}
    if set(cm)!=must:fail('Missing comparator common-valid pairs')
    cmrows=[]
    for key,rr in sorted(cm.items()):
        if len(rr)!=10 or {r['seed'] for r in rr}!=set(SEEDS):
            fail('Missing source seed from common-valid analysis '+repr(key))
        n=sum(r['common_n_valid'] for r in rr)
        sm_h=sum(r['common_abs_error_hb_sum'] for r in rr)
        sm_c=sum(r['common_abs_error_comp_sum'] for r in rr)
        cmrows.append({'dataset':key[0],'participant':key[1],'method':key[2],'boundary':key[3],
                       'common_n_valid':n,'hb_mae_common':sm_h/n if n else float('nan'),
                       'comparator_mae_common':sm_c/n if n else float('nan'),
                       'mean_gain_common_valid':(sm_c-sm_h)/n if n else float('nan')})
    return subjects,cmrows


def build_contrasts(subjects, commonrows):
    part={(r['dataset'],r['participant'],r['method'],r['boundary']):r for r in subjects}
    cr={(r['dataset'],r['participant'],r['method'],r['boundary']):r for r in commonrows}
    FAMILIES={
        'penalty_2s':('loss_2s','lower'),
        'penalty_4s':('loss_4s','lower'),
        'penalty_6s':('loss_6s','lower'),
        'conditional_mae_valid_only':('conditional_mae_pooled','lower'),
        'coverage':('coverage_mean_seed','higher'),
        'common_valid_mae':('mean_gain_common_valid','direct'),
    }
    tests=[]
    for family,(metric,direction) in FAMILIES.items():
        for ds in DATASETS:
            for method in METHODS[1:]:
                for b in BOUNDARIES:
                    diffs=[];missing=[]
                    for pid in PIDS[ds]:
                        key=(ds,pid,method,b)
                        hb=(ds,pid,HB,b)
                        comp=part[key];pro=part[hb]
                        if direction=='direct':
                            gain=cr[key]['mean_gain_common_valid']
                        elif direction=='lower':
                            gain=comp[metric]-pro[metric]
                        else:
                            gain=pro[metric]-comp[metric]
                        if not np.isfinite(gain):
                            missing.append(pid)
                        else:diffs.append(float(gain))
                    t={'family':family,'dataset':ds,'method':method,'display_method':DISPLAY[method],
                       'boundary':b,'n_participants_expected':len(PIDS[ds]),
                       'n_participants_valid':len(diffs),'missing_participants':','.join(missing),
                       'n_source_seeds':10,'mean_gain':float(np.mean(diffs)) if not missing else float('nan'),
                       'effect_sign':'positive favors HemoBoundary',
                       '_paired':np.asarray(diffs,float) if not missing else None}
                    tests.append(t)
    for family in FAMILIES:
        subset=[t for t in tests if t['family']==family]
        if len(subset)!=96:fail('Not exactly 96 declared comparisons for '+family)
    return tests


def paired_stats(tests,n_sign=SIGN_FLIPS,n_boot=BOOTSTRAPS,seed=RANDOM_SEED):
    """Uses the same sign draws per dataset and family, Holm across 96 tests.

    The 95% percentile intervals are not simultaneous family-wise CIs.
    Any contrast with an incomplete participant set is not tested. Such missing
    comparisons are included in the *predefined 96-family* with p=1 for Holm.
    """
    for family_index,family in enumerate(('penalty_2s','penalty_4s','penalty_6s',
                           'conditional_mae_valid_only','coverage','common_valid_mae')):
        subset=[t for t in tests if t['family']==family]
        for di,ds in enumerate(DATASETS):
            group=[t for t in subset if t['dataset']==ds and t['_paired'] is not None]
            if not group:continue
            n=len(PIDS[ds])
            D=np.stack([t['_paired'] for t in group],axis=1).astype(np.float64)
            assert D.shape==(n,len(group))
            rng=np.random.default_rng(seed+100*family_index+di)
            # Chunked random signs avoid enormous arrays, keep exact pairing.
            observed=np.abs(D.sum(axis=0))
            counts=np.zeros(len(group),np.int64)
            for start in range(0,n_sign,5000):
                batch=min(5000,n_sign-start)
                s=rng.integers(0,2,size=(batch,n),dtype=np.int8)*2-1
                sums=s.astype(np.float64)@D
                counts+=np.sum(np.abs(sums)>=observed[None,:]-1e-9,axis=0)
            pvals=(counts+1)/(n_sign+1)
            # Multinomial cluster bootstrap replicates participant resampling.
            bs=[]
            for start in range(0,n_boot,2000):
                batch=min(2000,n_boot-start)
                weights=rng.multinomial(n,[1/n]*n,size=batch)
                bs.append((weights@D)/n)
            intervals=np.percentile(np.vstack(bs),[2.5,97.5],axis=0)
            for j,t in enumerate(group):
                t['raw_p']=float(pvals[j]);t['ci95_low']=float(intervals[0,j]);t['ci95_high']=float(intervals[1,j])
                t['mean_gain']=float(np.mean(D[:,j]))
        # Defining missing contrasts as p=1 preserves full family size for adjustment.
        ps=np.array([t.get('raw_p',1.0) for t in subset],float)
        order=np.argsort(ps,kind='stable')
        running=0.;adj=np.zeros_like(ps)
        for i,idx in enumerate(order):
            running=max(running,(len(ps)-i)*ps[idx]);adj[idx]=min(1.,running)
        for t,p in zip(subset,adj):
            t['holm_p']=float(p) if t['_paired'] is not None else float('nan')
            t['holm_family_size']=len(subset)
            t['holm_significant']=bool(p<.05) if t['_paired'] is not None else False
            t['p_test']='two-sided participant-paired Monte Carlo sign flip'
            t['ci_method']='participant bootstrap percentile, not multiplicity-adjusted'
            t['inference_status']='COMPLETE' if t['_paired'] is not None else 'NOT_ESTIMATED_MISSING_PARTICIPANT'
            t.pop('_paired')
    return tests


def make_descriptives(subjects,commonrows):
    g=defaultdict(list)
    for r in subjects:g[(r['dataset'],r['method'],r['boundary'])].append(r)
    commons={(r['dataset'],r['participant'],r['method'],r['boundary']):r for r in commonrows}
    rows=[]
    for (ds,method,b),rr in sorted(g.items()):
        if len(rr)!=len(PIDS[ds]):fail('Incomplete descriptive participants')
        row={'dataset':ds,'method':method,'display_method':DISPLAY[method],
             'boundary':b,'n_participants':len(rr),
             'n_source_seeds':10,'coverage_percent':float(np.mean([r['coverage_mean_seed'] for r in rr]))*100,
             'mean_conditional_mae_pooled':float(np.nanmean([r['conditional_mae_pooled'] for r in rr])),
             'n_participants_conditional':sum(np.isfinite(r['conditional_mae_pooled']) for r in rr),
             **{f'mean_loss_{p}s':float(np.mean([r[f'loss_{p}s'] for r in rr]))
                for p in (*PENALTIES,AUDIT_PENALTY)}}
        if method!=HB:
            matched=[commons[(ds,r['participant'],method,b)] for r in rr]
            row['mean_common_valid_gain_sec']=float(np.nanmean([x['mean_gain_common_valid'] for x in matched]))
            row['participants_common_valid']=sum(np.isfinite(x['mean_gain_common_valid']) for x in matched)
        rows.append(row)
    return rows


def verify_prior_4s(tests):
    if not PRIOR.is_file():
        print('CAUTION: Prior Table 4 paired CSV not found at pinned path, skipping numerical paired-mean parity')
        return {'found':False}
    original=read_csv(PRIOR)
    old={(r['dataset'],r['method'],r['boundary']):float(r['mean_gain'])
         for r in original if r.get('family','').startswith('timing_96')}
    if len(old)!=96:fail('Prior 4-s paired output has incomplete/inconsistent timing family')
    now={ (r['dataset'],r['method'],r['boundary']):r['mean_gain']
         for r in tests if r['family']=='penalty_4s'}
    if set(old)!=set(now):fail('Prior timing comparison keys differ from saved-replay keys')
    for key,val in old.items():close(val,now[key],f'prior frozen 4s paired mean gain {key}',tol=2e-8)
    print('PASS: all 96 reproduced 4-s participant-paired mean gains match prior Table 4 statistical export')
    return {'found':True,'paired_comparisons_verified':96,'prior_file_sha256':sha_file(PRIOR)}


def summarize_terminal(tests,descriptives,output):
    print('\n'+'='*108)
    print('THREE READ-ONLY CHECKS — SUMMARY (not final until complete source replay passes)')
    print('='*108)
    for family in ('penalty_2s','penalty_4s','penalty_6s','conditional_mae_valid_only','common_valid_mae','coverage'):
        rows=[r for r in tests if r['family']==family]
        done=[r for r in rows if r['inference_status']=='COMPLETE']
        sig=sum(bool(r['holm_significant']) for r in done)
        print(f' {family:30s} complete={len(done):3d}/96, Holm significant={sig:3d}/{len(done)}')
        fails=[r for r in done if not r['holm_significant']]
        if fails:
            print('    Nonsignificant (first 12): '+', '.join(f"{r['dataset']}/{r['display_method']}/{r['boundary']} p={r['holm_p']:.4g}" for r in fails[:12]))
        unavail=[r for r in rows if r['inference_status']!='COMPLETE']
        if unavail:
            print('    INCOMPLETE contrasts (first 6): '+', '.join(f"{r['dataset']}/{r['display_method']}/{r['boundary']}" for r in unavail[:6]))
    index={(r['dataset'],r['method'],r['boundary']):r for r in descriptives}
    print('\n Strongest Table 4 controls — peak summary:')
    print(f" {'Dataset':8s} {'Method':21s} {'Conditional MAE':>15} {'Coverage %':>11} {'Loss 2s':>9} {'Loss 4s':>9} {'Loss 6s':>9}")
    for ds in DATASETS:
        for m in (HB,EXTRA,HR_RIDGE):
            r=index[(ds,m,'peak')]
            print(f" {ds:8s} {DISPLAY[m]:21s} {r['mean_conditional_mae_pooled']:15.4f} {r['coverage_percent']:11.2f} {r['mean_loss_2s']:9.4f} {r['mean_loss_4s']:9.4f} {r['mean_loss_6s']:9.4f}")
    print('\n Results-only output folder:',output)
    print('All original models, checkpoints and calibration outputs remain read-only.')


def selftest():
    truth={'present':np.array([1.,1.,1.,0.]),'onset':np.array([1.,2.,3.,np.nan]),
           'peak':np.array([2.,4.,5.,np.nan]),'offset':np.array([7.,8.,9.,np.nan]),
           'censored':np.array([0.,0.,1.,0.])}
    pred=np.array([[1.1,2.3,7.4],[2.2,4.3,8.1],[np.nan,np.nan,np.nan],
                   [np.nan,np.nan,np.nan]])
    gate=np.array([1,1,1,0]);s=scored_timing(truth,pred,gate)
    assert s['onset']['summary']['n_eligible'] if 'n_eligible' in s['onset']['summary'] else True
    assert s['offset']['summary']['eligible_count']==2
    assert s['offset']['summary']['valid_count']==2
    assert s['onset']['summary']['valid_count']==2
    assert abs(s['onset']['summary']['coverage']-2/3)<1e-12
    # Original scoring rule: (0.1 + 0.2 + 4)/3, or change missing penalty.
    for p in (2,4,6,8):
        want=(.1+.2+p)/3
        assert abs(s['onset']['summary'][f'loss_{p}']-want)<1e-12
    assert abs(s['onset']['summary']['conditional_mae']-0.15)<1e-12
    # A valid error longer than 4 s must remain uncapped.
    bad=np.array([[1.1,2.3,7.4],[12.,13.,14.],[np.nan,np.nan,np.nan],
                  [np.nan,np.nan,np.nan]])
    assert scored_timing(truth,bad,gate)['onset']['summary']['loss_2']>2
    # A completely missing method must retain 0% coverage and NA conditional MAE.
    null=np.full((4,3),np.nan)
    c=scored_timing(truth,null,gate)
    assert c['onset']['summary']['coverage']==0
    assert np.isnan(c['onset']['summary']['conditional_mae'])
    assert c['onset']['summary']['loss_6']==6
    print('SELFTEST PASS: strict shared gate, uncapped finite errors, censored masks, 2/4/6/8 penalty and 0 coverage')


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--sign-flips',type=int,default=SIGN_FLIPS)
    parser.add_argument('--bootstraps',type=int,default=BOOTSTRAPS)
    args,_=parser.parse_known_args(argv)
    selftest()
    if args.selftest:return
    if args.sign_flips<1000 or args.bootstraps<500:raise ValueError('Analysis draws too low')
    manifest,provenance=verify_root()
    print('PASS: original 770-case SHA256 manifest, V8/Cohen/Lina source protocols, full-run markers')
    case_rows=[];common_rows=[]
    for ds in DATASETS:
        for seed in SEEDS:
            for index,pid in enumerate(PIDS[ds],start=1):
                identity=(ds,seed,pid)
                a,b=replay_case(ds,seed,pid,manifest[identity])
                case_rows.extend(a);common_rows.extend(b)
                if len(case_rows)//(len(METHODS)*len(BOUNDARIES))%25==0:
                    print(f'  Verified and replayed {len(case_rows)//36}/770 original cases ...',flush=True)
    if len(case_rows)!=770*len(METHODS)*len(BOUNDARIES):fail('Wrong original replay case count')
    if len(common_rows)!=770*(len(METHODS)-1)*len(BOUNDARIES):fail('Wrong common-valid case count')
    print('PASS: 770/770 saved trial-prediction banks, all 4s/coverage/conditional metrics match original checkpoints')
    participants,common=grouped_participants(case_rows,common_rows)
    descriptives=make_descriptives(participants,common)
    tests=build_contrasts(participants,common)
    paired_stats(tests,n_sign=args.sign_flips,n_boot=args.bootstraps)
    prior=verify_prior_4s(tests)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    OUT_PARENT.mkdir(parents=True,exist_ok=True)
    out=OUT_PARENT/('V8_770_THREE_CHECKS_'+stamp)
    out.mkdir(parents=True,exist_ok=False)
    try:
        files={
            'SAVED_PREDICTION_REPLAY_CASE_METRICS.csv':case_rows,
            'SAVED_PREDICTION_COMMON_VALID_COUNTS.csv':common_rows,
            'PARTICIPANT_10SEED_METRICS.csv':participants,
            'PARTICIPANT_COMMON_VALID_METRICS.csv':common,
            'DESCRIPTIVE_DATASET_METHOD_BOUNDARY.csv':descriptives,
            'PAIRED_ALL_SIX_FAMILIES.csv':tests,
        }
        for family in ('penalty_2s','penalty_4s','penalty_6s','conditional_mae_valid_only','common_valid_mae','coverage'):
            files['PAIRED_'+family.upper()+'.csv']=[t for t in tests if t['family']==family]
        for name,rows in files.items():save_csv(out/name,rows)
        note='''READ-ONLY HEMOBOUNDARY V8 sensitivity evaluation. The original checkpoints and scientific protocols were not changed. The 4s timing loss is the originally selected measure. Alternative 2s/6s penalties, conditional MAE, common-valid MAE, and coverage are exploratory analyses, not independent preregistered primary findings. Missing estimates receive the selected penalty, finite errors remain uncapped, and right-censored reference offset/duration targets are excluded. Timing uses the original frozen HemoBoundary response gate for all methods. Per-participant 10-seed conditional MAE pools valid trials within each participant, and common-valid comparisons use identical trial subsets. A method with no valid predictions has undefined conditional MAE. Participant-paired sign flips and participant-cluster percentile bootstraps use the 77 biological participants as independent units, separately within each dataset. Holm across all 96 Table 4 contrasts PER FAMILY. 95% CIs are not multiplicity adjusted. The Cohen-Adad, Lina and HRfunc methods are task-adapted comparators, not exact author reproductions.\n'''
        (out/'INTERPRETATION_README.txt').write_text(note,encoding='utf-8')
        digests={p.name:sha_file(p) for p in out.iterdir() if p.is_file()}
        audit={'status':'ALL_770_READONLY_SAVED_PREDICTIONS_REPLAY_VERIFIED',
            'original_770_cases':770,'methods':list(METHODS),'source_seeds':10,
            'participants_by_dataset':{d:len(PIDS[d]) for d in DATASETS},
            'family_sizes':{f:96 for f in ('penalty_2s','penalty_4s','penalty_6s','conditional_mae_valid_only','common_valid_mae','coverage')},
            'random_seed':RANDOM_SEED,'sign_flip_draws':args.sign_flips,'bootstraps':args.bootstraps,
            'V8_scoring_replayed_from_saved_predictions':True,
            'saved_4s_case_scores_reproduced':True,'no_original_checkpoint_changes':True,
            'prior_4s_paired_parity':prior,'source_provenance':provenance,
            'output_sha256':digests,'generated_utc':datetime.now(timezone.utc).isoformat()}
        (out/'AUDIT_COMPLETED.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
    except BaseException:
        print('Output writing failed. Do not interpret partial results in '+str(out))
        raise
    summarize_terminal(tests,descriptives,out)
    return out

if __name__=='__main__':main()
