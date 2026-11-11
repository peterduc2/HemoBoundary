"""Evaluation only: reference labels enter here, never a waveform estimator.

The historical 4-second missing-prediction loss is retained under an explicit
name. It is NOT described as ordinary MAE. Unpenalized MAE, coverage and paired
common-valid MAE are reported separately. No numerical offset is manufactured
for a censored reference.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from hb_data import write_rows, write_json, number, digest_json

BOUNDARIES=('onset','peak','offset','duration')
PENALTIES=(2.,4.,8.)
HB='HEMOBOUNDARY_STRICT_SAVED'

def mean_finite(x):
    a=np.asarray(x,float);a=a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float('nan')

def detection_metrics(truth,decision):
    y=np.asarray(truth,int);d=np.asarray(decision,float)
    if d.shape!=y.shape or not set(y).issubset({0,1}):raise ValueError('Detection shape/label error')
    finite=np.isfinite(d)
    if not set(d[finite]).issubset({0.,1.}):raise ValueError('Predicted decisions must be 0 or 1')
    # An execution failure is incorrect, not a free negative prediction.
    sensitivity=np.mean(finite[y==1] & (d[y==1]==1)) if np.any(y==1) else np.nan
    specificity=np.mean(finite[y==0] & (d[y==0]==0)) if np.any(y==0) else np.nan
    return {'balanced_accuracy':float((sensitivity+specificity)/2),
            'sensitivity':float(sensitivity),'specificity':float(specificity),
            'detection_valid_fraction':float(finite.mean()),'n_positive':int(np.sum(y==1)),
            'n_negative':int(np.sum(y==0))}

def eligible_masks(truth):
    pos=np.asarray(truth['present'],int)==1;cens=np.asarray(truth['censored'],int)==1
    masks={k:pos & np.isfinite(truth[k]) & (~cens if k=='offset' else True)
           for k in BOUNDARIES[:3]}
    masks['duration']=masks['onset'] & masks['offset']
    return masks

def outcome_arrays(truth,pred,gate):
    p=np.asarray(pred,float);g=np.asarray(gate,float)
    if p.shape!=(len(truth['present']),3) or g.shape!=(len(p),):raise ValueError('Timing output shape mismatch')
    ref={k:np.asarray(truth[k],float) for k in BOUNDARIES[:3]}
    ref['duration']=ref['offset']-ref['onset']
    estimates={k:p[:,j] for j,k in enumerate(BOUNDARIES[:3])}
    estimates['duration']=p[:,2]-p[:,0]
    masks=eligible_masks(truth);valid={}
    for k in BOUNDARIES:
        valid[k]=masks[k] & (g==1) & np.isfinite(estimates[k])
    # Timing heads cannot gain credit for contradicting the declared ordering.
    bad_onpeak=np.isfinite(p[:,0])&np.isfinite(p[:,1])&(p[:,0]>=p[:,1])
    bad_peakoff=np.isfinite(p[:,1])&np.isfinite(p[:,2])&(p[:,1]>=p[:,2])
    for k in ('onset','peak'):valid[k]&=~bad_onpeak
    for k in ('offset','duration'):valid[k]&=~bad_peakoff & ~bad_onpeak
    for k in BOUNDARIES[:3]:valid[k]&=(estimates[k]>=0)&(estimates[k]<=20)
    valid['duration']&=valid['onset']&valid['offset']
    return ref,estimates,masks,valid

def timing_metrics(truth,pred,gate,supported=BOUNDARIES):
    ref,est,masks,valid=outcome_arrays(truth,pred,gate);out={}
    if 'offset' in supported:
        cens=(np.asarray(truth['present'])==1)&(np.asarray(truth['censored'])==1)
        out['n_censored_reference']=int(cens.sum())
        out['finite_offset_on_censored_reference_fraction']=float(np.mean(np.isfinite(np.asarray(pred)[cens,2]) & (np.asarray(gate)[cens]==1))) if cens.any() else np.nan
    for k in supported:
        m=masks[k];v=valid[k];n=int(m.sum());nvalid=int(v.sum())
        errors=np.abs(est[k]-ref[k])
        out[f'{k}_n_eligible']=n;out[f'{k}_n_valid']=nvalid
        out[f'{k}_coverage']=nvalid/n if n else np.nan
        out[f'{k}_mae_valid_sec']=float(errors[v].mean()) if nvalid else np.nan
        for penalty in PENALTIES:
            # Finite large errors are not clipped to the missing penalty.
            loss=np.full(len(m),penalty,float);loss[v]=errors[v]
            out[f'{k}_loss_missing{int(penalty)}_sec']=float(loss[m].mean()) if n else np.nan
    return out

def selection_loss(truth,pred,groups):
    # Participant-balanced, fixed equal weight over four historical 4s losses.
    groups=np.asarray(groups);values=[]
    for group in sorted(set(groups)):
        ix=groups==group;t={k:np.asarray(v)[ix] for k,v in truth.items()}
        m=timing_metrics(t,np.asarray(pred)[ix],np.ones(int(ix.sum())))
        loss=np.mean([m[f'{k}_loss_missing4_sec'] for k in BOUNDARIES])
        if not np.isfinite(loss):raise ValueError('Source objective has missing eligible boundary targets')
        values.append(float(loss))
    return float(np.mean(values))

def common_valid_metrics(truth,candidate,hb,gate,supported=BOUNDARIES):
    ref,est,m,vc=outcome_arrays(truth,candidate,gate)
    _,eh,_,vh=outcome_arrays(truth,hb,gate);out={}
    for k in supported:
        common=vc[k]&vh[k];n=int(common.sum());eligible=int(m[k].sum())
        out[f'{k}_common_n']=n
        out[f'{k}_common_coverage']=n/eligible if eligible else np.nan
        out[f'{k}_common_candidate_mae_sec']=float(np.mean(np.abs(est[k][common]-ref[k][common]))) if n else np.nan
        out[f'{k}_common_hb_mae_sec']=float(np.mean(np.abs(eh[k][common]-ref[k][common]))) if n else np.nan
        out[f'{k}_common_gain_sec']=(out[f'{k}_common_candidate_mae_sec']-out[f'{k}_common_hb_mae_sec']) if n else np.nan
    return out

def stable_seed(*parts):return int(digest_json(list(parts))[:8],16)

def paired_inference(gains,seed,reps=20000):
    a=np.asarray(gains,float)
    if not np.isfinite(a).all():raise ValueError('Incomplete paired sample; do not silently drop participants')
    if len(a)<3:return {'mean_gain':float(a.mean()) if len(a) else np.nan,'ci95_low':np.nan,'ci95_high':np.nan,'raw_p':np.nan}
    rng=np.random.default_rng(seed);obs=float(a.mean());extreme=0;boot=[]
    for start in range(0,reps,1000):
        size=min(1000,reps-start)
        idx=rng.integers(0,len(a),size=(size,len(a)));boot.extend(a[idx].mean(axis=1))
        signs=rng.choice(np.array([-1.,1.]),size=(size,len(a)))
        extreme+=int(np.sum(np.abs((signs*a).mean(axis=1))>=abs(obs)-1e-12))
    return {'mean_gain':obs,'ci95_low':float(np.quantile(boot,.025)),
            'ci95_high':float(np.quantile(boot,.975)),'raw_p':(extreme+1)/(reps+1),
            'n_positive':int(np.sum(a>1e-12)),'n_negative':int(np.sum(a< -1e-12)),
            'n_zero':int(np.sum(np.abs(a)<=1e-12)),'monte_carlo_draws':int(reps)}

def holm(p):
    p=np.asarray(p,float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Holm requires the complete finite declared p-value family')
    order=np.argsort(p,kind='stable');out=np.empty_like(p);prev=0.
    for rank,idx in enumerate(order):prev=max(prev,(len(p)-rank)*p[idx]);out[idx]=min(1.,prev)
    return out

def summarize(metrics,out,expected,seeds,pvalues=True):
    """All methods must have the same requested dataset/seed/participant coverage.
    Scope is explicit: previously reported R1-R7 p-values are NOT reused/merged.
    """
    out=Path(out);seen=set();rows=[]
    for r in metrics:
        k=(r['dataset'],int(r['seed']),r['participant'],r['method'],r['mode'])
        if k in seen:raise ValueError(f'Duplicate metric row {k}')
        seen.add(k);rows.append(r)
    methods_by_mode=defaultdict(set)
    for r in rows:methods_by_mode[r['mode']].add(r['method'])
    for mode,methods in methods_by_mode.items():
        for method in methods:
            actual={(r['dataset'],int(r['seed']),r['participant']) for r in rows if r['method']==method and r['mode']==mode}
            if actual!=set(expected):raise ValueError(f'Incomplete or mixed coverage for {method}/{mode}')
    write_rows(out/'PARTICIPANT_METRICS.csv',rows)
    grouped=defaultdict(list)
    identity={'dataset','seed','participant','method','mode','supported'}
    for r in rows:grouped[(r['dataset'],r['method'],r['mode'])].append(r)
    summaries=[];subject_means=[]
    for (ds,method,mode),rr in sorted(grouped.items()):
        keys=[k for k in rr[0] if k not in identity and isinstance(rr[0][k],(int,float,np.integer,np.floating))]
        pids=sorted({r['participant'] for r in rr})
        for pid in pids:
            pp=[r for r in rr if r['participant']==pid]
            subject_means.append({'dataset':ds,'method':method,'mode':mode,'participant':pid,'n_seeds':len(pp),
                                  **{k:mean_finite([r.get(k,np.nan) for r in pp]) for k in keys}})
        for k in keys:
            seedmeans=[mean_finite([r.get(k,np.nan) for r in rr if int(r['seed'])==s]) for s in seeds]
            personmeans=[mean_finite([r.get(k,np.nan) for r in rr if r['participant']==p]) for p in pids]
            summaries.append({'dataset':ds,'method':method,'mode':mode,'metric':k,
                'mean':mean_finite(seedmeans),'sd_across_seed_means':float(np.std(seedmeans,ddof=1)) if len(seedmeans)>1 else np.nan,
                'sd_across_participant_means':float(np.std(personmeans,ddof=1)) if len(personmeans)>1 else np.nan,
                'n_participants':len(pids),'n_seeds':len(seeds),'n_seed_participant_pairs':len(rr)})
    write_rows(out/'METHOD_SUMMARY.csv',summaries);write_rows(out/'PARTICIPANT_SEED_AVERAGED.csv',subject_means)
    # Only shared-gate timing and independent detection define the two primary
    # families. Native and learned readouts are BOTH included, never cherry-picked.
    comparisons=[];person={(r['dataset'],r['method'],r['mode'],r['participant']):r for r in subject_means}
    datasets=sorted({e[0] for e in expected})
    for ds in datasets:
        pids=sorted({p for d,s,p in expected if d==ds})
        for mode in ('shared_gate_timing','independent_detection'):
            for method in sorted(methods_by_mode.get(mode,())):
                if method==HB:continue
                keys=('balanced_accuracy',) if mode=='independent_detection' else tuple(f'{b}_loss_missing4_sec' for b in BOUNDARIES)
                for key in keys:
                    a=[person[(ds,method,mode,p)].get(key,np.nan) for p in pids]
                    b=[person[(ds,HB,mode,p)].get(key,np.nan) for p in pids]
                    if all(not np.isfinite(x) for x in a):continue # Unsupported target, declared before inference.
                    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError(f'Missing primary metric {ds}/{method}/{key}')
                    gains=np.array(b)-a if mode=='independent_detection' else np.array(a)-b
                    st=paired_inference(gains,stable_seed('V8',ds,method,key)) if pvalues else {'mean_gain':float(gains.mean()),'ci95_low':np.nan,'ci95_high':np.nan,'raw_p':np.nan}
                    comparisons.append({'dataset':ds,'method':method,'metric':key,'family':'detection_all_datasets' if mode=='independent_detection' else 'timing_all_methods_all_datasets',
                         'n_participants':len(pids),'n_seeds':len(seeds),'positive_gain_favors':'HemoBoundary',**st})
    for family in sorted({r['family'] for r in comparisons}):
        rr=[r for r in comparisons if r['family']==family]
        corrected=holm([r['raw_p'] for r in rr]) if pvalues else [np.nan]*len(rr)
        for r,p in zip(rr,corrected):r['holm_p']=float(p);r['family_size']=len(rr)
    write_rows(out/'PAIRED_COMPARISONS.csv',comparisons)
    print('\nCOMPLETE SUMMARY (shared gate, historical missing=4 s loss; not conditional MAE)')
    for row in summaries:
        if row['mode']=='shared_gate_timing' and row['metric'].endswith('_loss_missing4_sec'):
            print(f"{row['dataset']:7s} {row['method']:36s} {row['metric']:29s} {row['mean']:.5f}")
    print('\nPrimary timing and detection families are corrected separately across ALL requested datasets.')
    print('Conditional MAE, coverage, 2/8 s sensitivity and common-valid comparisons are in PARTICIPANT_METRICS.csv.')
    print('These are post-development comparisons; p-values do not certify faithful reproduction or fairness.')
    return summaries,comparisons
