"""Training and source-only selection, independent of external/held-out labels.

Waveform reconstruction, optional supervised boundary readout and response
classification are separate objects. The readout is a NEW benchmark adaptation,
not something attributed to Pinti/Santosa/Schaedig. All source participants have
equal total influence on model-selection objectives.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from hb_algorithms import FRACTIONS, VARIANTS, native_bounds, curve_features
from hb_evaluation import selection_loss, timing_metrics, BOUNDARIES, detection_metrics, stable_seed

READOUT_ALPHAS=(1.,10.,100.)
TREE_LEAVES=(1,4)
TREE_COUNT=128
INNER_FOLDS=3
CONTROL_METHODS=('RIDGE9_CONTROL','EXTRATREES9_CONTROL')

def group_folds(groups,seed):
    groups=np.asarray(groups);unique=np.array(sorted(set(groups)),dtype=object)
    if len(unique)<3:raise ValueError('At least three source participants are needed for grouped selection')
    rng=np.random.default_rng(seed);rng.shuffle(unique)
    folds=[]
    for valids in np.array_split(unique,INNER_FOLDS):
        vi=np.isin(groups,valids);tr=np.flatnonzero(~vi);va=np.flatnonzero(vi)
        if set(groups[tr])&set(groups[va]):raise AssertionError('Group separation failed')
        folds.append((tr,va))
    return folds

def subset_truth(truth,ix):return {k:np.asarray(v)[ix] for k,v in truth.items()}

def stack_source(source,excluded=None):
    pids=tuple(sorted(source))
    if any(p!=source[p].participant or source[p].dataset!='SHIN' or source[p].split!='train' for p in pids):
        raise ValueError('Source fit requires correctly identified SHIN training banks only')
    if excluded is not None and excluded in source:raise ValueError('Target participant was passed to a calibration fitter')
    truth={k:np.concatenate([source[p].truth[k] for p in pids]) for k in ('present','onset','peak','offset','censored')}
    groups=np.concatenate([np.full(source[p].n,p,object) for p in pids])
    return pids,truth,groups

def source_weights(groups,mask):
    g=np.asarray(groups)[mask];weights=np.empty(len(g),float)
    for pid in set(g):weights[g==pid]=1./np.sum(g==pid)
    weights*=len(weights)/weights.sum()
    return weights

def project_boundaries(pred):
    """Common geometry adapter. Never clip an unobserved offset to 20 s.
    These are model estimates, not declarations that real recovery was observed.
    """
    p=np.asarray(pred,float).copy()
    finite=np.isfinite(p[:,0]);p[finite,0]=np.clip(p[finite,0],0.,19.)
    finite=np.isfinite(p[:,1]);p[finite,1]=np.clip(p[finite,1],.5,19.5)
    both=np.isfinite(p[:,:2]).all(axis=1);p[both,1]=np.maximum(p[both,1],p[both,0]+.5)
    valid=np.isfinite(p[:,2])&np.isfinite(p[:,1]);p[valid,2]=np.maximum(p[valid,2],p[valid,1]+.5)
    p[np.isfinite(p[:,2]) & ((p[:,2]>20.)|(p[:,2]<0.)),2]=np.nan
    return p

@dataclass
class Heads:
    scaler:object
    onset_peak:object
    offset:object
    kind:str
    def predict(self,X):
        xx=self.scaler.transform(X)
        p=np.column_stack([self.onset_peak.predict(xx),self.offset.predict(xx)])
        return project_boundaries(p)

def fit_heads(X,truth,groups,kind,param,seed):
    X=np.asarray(X,float);groups=np.asarray(groups)
    if X.ndim!=2 or not np.isfinite(X).all():raise ValueError('Invalid predictor matrix')
    scaler=StandardScaler().fit(X);Z=scaler.transform(X)
    pos=np.asarray(truth['present'])==1
    onpk=pos & np.isfinite(truth['onset']) & np.isfinite(truth['peak'])
    off=pos & (np.asarray(truth['censored'])==0) & np.isfinite(truth['offset'])
    if onpk.sum()<10 or off.sum()<10:raise ValueError('Too few observable positive source targets')
    if kind=='ridge':
        m1=Ridge(alpha=float(param));m2=Ridge(alpha=float(param))
    elif kind=='trees':
        kw={'n_estimators':TREE_COUNT,'min_samples_leaf':int(param),'max_features':1.,'n_jobs':1,'random_state':int(seed)}
        m1=ExtraTreesRegressor(**kw);m2=ExtraTreesRegressor(**kw)
    else:raise KeyError(kind)
    m1.fit(Z[onpk],np.column_stack([truth['onset'],truth['peak']])[onpk],sample_weight=source_weights(groups,onpk))
    m2.fit(Z[off],np.asarray(truth['offset'])[off],sample_weight=source_weights(groups,off))
    return Heads(scaler,m1,m2,kind)

def select_heads(features_by_variant,truth,groups,kind,seed,valid_by_variant=None):
    folds=group_folds(groups,seed);params=READOUT_ALPHAS if kind=='ridge' else TREE_LEAVES
    records=[]
    for variant,X in features_by_variant.items():
        valid=np.ones(len(X),bool) if valid_by_variant is None else np.asarray(valid_by_variant[variant],bool)
        for parameter in params:
            predicted=np.full((len(X),3),np.nan)
            for fi,(tr,va) in enumerate(folds):
                fit_ix=tr[valid[tr]]
                model=fit_heads(X[fit_ix],subset_truth(truth,fit_ix),groups[fit_ix],kind,parameter,stable_seed(seed,fi,'head'))
                predicted[va]=model.predict(X[va])
                predicted[va[~valid[va]]]=np.nan
            loss=selection_loss(truth,predicted,groups)
            records.append({'variant':variant,'parameter':float(parameter),'source_group_cv_loss':loss})
    best=min(records,key=lambda r:(r['source_group_cv_loss'],r['variant'],-r['parameter']))
    valid=np.ones(len(groups),bool) if valid_by_variant is None else np.asarray(valid_by_variant[best['variant']],bool)
    model=fit_heads(features_by_variant[best['variant']][valid],subset_truth(truth,valid),groups[valid],kind,best['parameter'],seed)
    return model,best,records

def threshold_from_source(scores,labels,groups):
    s=np.asarray(scores,float);y=np.asarray(labels,int);groups=np.asarray(groups)
    vals=s[np.isfinite(s)]
    if not len(vals):raise ValueError('All source detection scores failed; incompatible method is not a valid weak baseline')
    # Fixed number of source quantiles and two endpoint decisions; deterministic tie rule.
    unique=np.unique(np.quantile(vals,np.linspace(0,1,101)))
    thresholds=np.r_[np.nextafter(vals.min(),-np.inf),unique,np.nextafter(vals.max(),np.inf)]
    result=[]
    for threshold in thresholds:
        d=np.where(np.isfinite(s),(s>=threshold).astype(float),np.nan)
        accuracies=[detection_metrics(y[groups==pid],d[groups==pid])['balanced_accuracy'] for pid in sorted(set(groups))]
        result.append((float(np.mean(accuracies)),float(threshold)))
    best=max(result,key=lambda z:(z[0],z[1]))
    return best[1],best[0]

@dataclass
class Detector:
    scaler:object
    model:object
    threshold:float
    def predict(self,X):
        prob=self.model.predict_proba(self.scaler.transform(X))[:,1]
        return (prob>=self.threshold).astype(float),prob

def fit_control_detector(X,y,groups,seed):
    folds=group_folds(groups,seed);prob=np.empty(len(X))
    for tr,va in folds:
        scaler=StandardScaler().fit(X[tr]);m=LogisticRegression(C=1.,class_weight='balanced',solver='lbfgs',max_iter=2000)
        m.fit(scaler.transform(X[tr]),y[tr],sample_weight=source_weights(groups[tr],np.ones(len(tr),bool)))
        prob[va]=m.predict_proba(scaler.transform(X[va]))[:,1]
    threshold,ba=threshold_from_source(prob,y,groups)
    scaler=StandardScaler().fit(X);m=LogisticRegression(C=1.,class_weight='balanced',solver='lbfgs',max_iter=2000)
    m.fit(scaler.transform(X),y,sample_weight=source_weights(groups,np.ones(len(groups),bool)))
    return Detector(scaler,m,threshold),{'threshold':threshold,'source_group_oof_balanced_accuracy':ba,'detector':'fixed-C=1 class-balanced logistic regression on nine-feature grid'}

def native_selection(curves_by_variant,truth,groups):
    """Per-boundary fractions from source data; no hidden test target consulted.
    Onset/offset fractions selected separately, then variant by fixed joint loss.
    """
    records=[]
    for variant,curves in curves_by_variant.items():
        allbounds={f:np.stack([native_bounds(c,f) for c in curves]) for f in FRACTIONS}
        losses=[]
        for f,bounds in allbounds.items():
            vals={k:[] for k in ('onset','offset')}
            for pid in sorted(set(groups)):
                ix=groups==pid;metrics=timing_metrics(subset_truth(truth,ix),bounds[ix],np.ones(int(ix.sum())))
                for k in vals:vals[k].append(metrics[f'{k}_loss_missing4_sec'])
            losses.append({'fraction':float(f),'onset_loss':float(np.mean(vals['onset'])),'offset_loss':float(np.mean(vals['offset']))})
        on=min(losses,key=lambda r:(r['onset_loss'],r['fraction']))['fraction']
        off=min(losses,key=lambda r:(r['offset_loss'],r['fraction']))['fraction']
        pred=allbounds[on].copy();pred[:,2]=allbounds[off][:,2]
        loss=selection_loss(truth,pred,groups)
        records.append({'variant':variant,'onset_fraction':on,'offset_fraction':off,'source_loss':loss,'fraction_scores':losses})
    best=min(records,key=lambda r:(r['source_loss'],r['variant']))
    return best,records

@dataclass
class FittedFamily:
    name:str
    native:dict
    head:Heads
    selected:dict
    detector_thresholds:dict
    audit:dict
    def predict(self,bank,cache):
        output={}
        for adapter,config in (('NATIVE_THRESH',self.native),('SOURCE_RIDGE_READOUT',self.selected)):
            variant=config['variant'];curves,score,_=cache.get(bank,self.name,variant)
            if adapter=='NATIVE_THRESH':bounds=np.stack([native_bounds(c,config['onset_fraction'],config['offset_fraction']) for c in curves])
            else:
                bounds=self.head.predict(curve_features(curves))
                bounds[~np.isfinite(curves).all(axis=1)]=np.nan # a failed kernel is not replaced by a constant predictor.
            threshold=self.detector_thresholds[adapter]
            decision=np.where(np.isfinite(score),(score>=threshold).astype(float),np.nan)
            output[self.name+'__'+adapter]={'boundaries':bounds,'decision':decision,'score':score,'supported':BOUNDARIES,'wrapper_detection':True}
        return output

def fit_family(name,source,cache,excluded,seed):
    pids,truth,groups=stack_source(source,excluded)
    curves_by_variant={};score_by_variant={};features_by_variant={}
    for variant in VARIANTS[name]:
        chunks=[cache.get(source[p],name,variant) for p in pids]
        curves=np.concatenate([x[0] for x in chunks]);scores=np.concatenate([x[1] for x in chunks])
        if np.isfinite(curves).all(axis=1).mean()<.9:raise ValueError(f'{name}/{variant}: less than 90% source compatibility; stop rather than report invalid SOTA performance')
        curves_by_variant[variant]=curves;score_by_variant[variant]=scores;features_by_variant[variant]=curve_features(curves)
    native,native_audit=native_selection(curves_by_variant,truth,groups)
    # Numerical estimator failures remain failures during cross-validation: learn
    # from finite curves only, but score all held-out source responses (penalty).
    model,best,head_audit=select_heads(features_by_variant,truth,groups,'ridge',seed,{v:np.isfinite(c).all(axis=1) for v,c in curves_by_variant.items()})
    thresholds={};detection_audit={}
    for adapter,variant in (('NATIVE_THRESH',native['variant']),('SOURCE_RIDGE_READOUT',best['variant'])):
        th,ba=threshold_from_source(score_by_variant[variant],truth['present'],groups)
        thresholds[adapter]=th;detection_audit[adapter]={'threshold':th,'source_ba':ba}
    audit={'family':name,'source_participants':list(pids),'excluded_target':excluded,'source_bank_signatures':{p:source[p].signature for p in pids},
           'native_selection':native,'native_candidates':native_audit,'readout_selection':best,'readout_candidates':head_audit,'detector_selection':detection_audit,
           'readout_group_cv_folds':INNER_FOLDS,'readout_is_new_adaptation':True}
    return FittedFamily(name,native,model,best,thresholds,audit)

@dataclass
class FittedControl:
    name:str
    head:Heads
    detector:Detector
    audit:dict
    def predict(self,bank,cache=None):
        X=bank.features.reshape(bank.n,-1);bounds=self.head.predict(X);decision,prob=self.detector.predict(X)
        return {self.name:{'boundaries':bounds,'decision':decision,'score':prob,'supported':BOUNDARIES,'wrapper_detection':True}}

def fit_control(name,source,excluded,seed):
    pids,truth,groups=stack_source(source,excluded)
    X=np.concatenate([source[p].features.reshape(source[p].n,-1) for p in pids]);kind='ridge' if name.startswith('RIDGE') else 'trees'
    model,best,records=select_heads({'nine_feature_grid':X},truth,groups,kind,seed)
    detector,da=fit_control_detector(X,truth['present'].astype(int),groups,seed)
    audit={'family':name,'source_participants':list(pids),'excluded_target':excluded,'source_bank_signatures':{p:source[p].signature for p in pids},
           'selection':best,'candidates':records,'detector_selection':da,'not_labelled_sota':True}
    return FittedControl(name,model,detector,audit)

@dataclass
class FittedAIDE:
    threshold:float
    audit:dict
    def predict(self,bank,cache):
        _,score,_=cache.get(bank,'PINTI_AIDE_SEARCH','single_event')
        decision=np.where(np.isfinite(score),(score>=self.threshold).astype(float),np.nan)
        return {'PINTI_AIDE_SEARCH':{'boundaries':None,'decision':decision,'score':score,'supported':(),'wrapper_detection':True}}

def fit_aide(source,cache,excluded):
    pids,truth,groups=stack_source(source,excluded)
    scores=np.concatenate([cache.get(source[p],'PINTI_AIDE_SEARCH','single_event')[1] for p in pids])
    th,ba=threshold_from_source(scores,truth['present'],groups)
    return FittedAIDE(th,{'family':'PINTI_AIDE_SEARCH','source_participants':list(pids),'excluded_target':excluded,'threshold':th,'source_ba':ba,
                         'source_bank_signatures':{p:source[p].signature for p in pids},'full_aide_matlab_parity':False})
