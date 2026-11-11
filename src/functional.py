"""Functional necessity of HemoBoundary components."""
from __future__ import annotations
import gc
import hashlib
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.linear_model import LogisticRegression, Ridge
warnings.filterwarnings('ignore')
SEEDS = tuple(range(1, 11))
JOB_DIR = 'all_seeds_01_10'
DRIVE_ROOT = Path(os.environ.get('HEMOBOUNDARY_MYDRIVE', '/content/gdrive/MyDrive'))
SHIN_ROOT = Path(os.environ.get('HEMOBOUNDARY_SHIN_ROOT', str(DRIVE_ROOT / 'HonorProject' / 'HemoBoundary' / 'SHIN')))
OUT_ROOT = SHIN_ROOT / '110_FUNCTIONAL_NECESSITY_ABLATION_10SEED_V2_STANDALONE'
OUT_DIR = OUT_ROOT / JOB_DIR
ALL_PIDS = tuple((f'VP{i:03d}' for i in range(1, 27)))
N4_10SEED_DIR = SHIN_ROOT / '95_N4_FUNCTIONAL_ABC_LBR_FULLY_NESTED_10SEEDS_V1'
HIERARCHICAL_SEED1_DIR = SHIN_ROOT / '98_B0_STRICT_HIERARCHICAL_SEED01_FULLY_NESTED_V1'
HIERARCHICAL_10SEED_DIR = SHIN_ROOT / '99_B0_STRICT_HIERARCHICAL_10SEEDS_V1'
GRID_DT = 0.5
GRID_END = 20.0
GRID = np.arange(0.0, GRID_END + GRID_DT / 2.0, GRID_DT)
MIN_GAP = 0.5
MISS_PENALTY_SEC = 4.0
CHANGE_WINDOWS = (3, 5, 9)
PREDICTOR_KEYS = ('obs', 'state_feat')
LABEL_KEYS = ('present', 'onset', 'peak', 'offset', 'censored', 'states')
BASE_N4_RANDOM_SEED = 20260912
ACTIVE_RUN_SEED = 1
LOCAL_OOF_FOLDS_COHERENT = 3
ROW_BASE = 'BASE_CANONICAL_HRF_GLM_DERIVATIVES'
ROW_A = 'BASE_PLUS_A_TEMPORAL_BOUNDARY'
ROW_AB = 'BASE_PLUS_A_PLUS_B_RECOVERY'
ROW_ABC = 'BASE_PLUS_A_PLUS_B_PLUS_C_RESPONSE'
ROW_FULL = 'BASE_PLUS_A_PLUS_B_PLUS_C_PLUS_LBR'

def mount_drive():
    if DRIVE_ROOT.is_dir():
        return
    from google.colab import drive
    drive.mount(str(DRIVE_ROOT.parent), force_remount=False)
def atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    compression = 'gzip' if str(path).endswith('.gz') else None
    df.to_csv(tmp, index=False, compression=compression)
    os.replace(tmp, path)
def atomic_json(obj: Mapping[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)
def base_stable_seed(*parts: Any) -> int:
    raw = '|'.join((str(x) for x in parts)).encode('utf-8')
    return int(hashlib.sha256(raw).hexdigest()[:8], 16) % (2 ** 31 - 1)
def stable_seed(*parts: Any) -> int:
    if int(ACTIVE_RUN_SEED) == 1:
        return base_stable_seed(*parts)
    return base_stable_seed('MULTISEED_REPEAT', int(ACTIVE_RUN_SEED), *parts)
def configure_seed(seed: int):
    global ACTIVE_RUN_SEED
    ACTIVE_RUN_SEED = int(seed)
    np.random.seed(base_stable_seed('numpy_legacy', int(seed)) % (2 ** 32 - 1))
    n4.RANDOM_SEED = int(BASE_N4_RANDOM_SEED) + int(seed) - 1
def canonical_cfg(cfg: Mapping[str, Any]) -> str:
    return json.dumps({k: cfg[k] for k in sorted(cfg)}, sort_keys=True, separators=(',', ':'))
def finite_model_array(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x)
    if not np.issubdtype(a.dtype, np.number):
        return a
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0, copy=True)
def predictor_view(data: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: data[k] for k in PREDICTOR_KEYS}
def require_predictors_only(x: Mapping[str, np.ndarray]) -> None:
    if set(x) != set(PREDICTOR_KEYS):
        raise RuntimeError(f'Prediction payload must contain ONLY {PREDICTOR_KEYS}; got {sorted(x)}')
def grouped_folds(pids: Sequence[str], seed: int, n: int=3) -> List[List[str]]:
    p = np.array(sorted(set(map(str, pids))), dtype=object)
    if len(p) < 3:
        raise RuntimeError('At least three source participants are required.')
    np.random.default_rng(seed).shuffle(p)
    return [q.tolist() for q in np.array_split(p, min(n, len(p))) if len(q)]
def subset(data: Mapping[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v)[mask] for k, v in data.items()}
def cat(parts: Sequence[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
def training_rows(caches: Mapping[str, Any], pids: Sequence[str]) -> Dict[str, np.ndarray]:
    return cat([caches[p]['train'] for p in sorted(pids)])
class FitAudit:

    def __init__(self, allowed: Sequence[str], forbidden: Sequence[str], context: str):
        self.allowed = set(allowed)
        self.forbidden = set(forbidden)
        self.context = context
        self.rows = []

    def record(self, data: Mapping[str, np.ndarray], stage: str, held: Sequence[str]=()):
        actual = set(map(str, data['pid']))
        blocked = self.forbidden | set(held)
        if not actual or not actual.issubset(self.allowed) or actual & blocked:
            raise RuntimeError(f'{self.context}/{stage}: invalid fit membership {actual & blocked}')
        self.rows.append(dict(
            context=self.context,
            stage=stage,
            fit_participants='|'.join(sorted(actual)),
            forbidden='|'.join(sorted(blocked)),
            n_rows=int(len(data['pid']))
        ))
def smooth_series(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, float)
    if int(window) <= 1:
        return x
    w = min(int(window), len(x) if len(x) % 2 == 1 else len(x) - 1)
    if w < 3:
        return x
    if w % 2 == 0:
        w -= 1
    return signal.savgol_filter(x, w, 2 if w >= 5 else 1, mode='interp')
def change_branch_features(data: Mapping[str, np.ndarray], frac: float) -> np.ndarray:
    F = finite_model_array(data['state_feat']).astype(float, copy=False)
    anti = F[:, :, 2]
    rows: List[List[float]] = []
    for a in anti:
        scale_rows = []
        for w in CHANGE_WINDOWS:
            z = smooth_series(a, int(w))
            dz = np.gradient(z, GRID_DT)
            ddz = np.gradient(dz, GRID_DT)
            early = np.flatnonzero(GRID <= 8.0)
            oi = int(early[np.argmax(dz[early])]) if len(early) else int(np.argmax(dz))
            pi = int(oi + np.argmax(z[oi:]))
            mx = float(z[pi])
            aft = np.arange(pi + 1, len(z))
            offi = len(z) - 1
            if len(aft) and mx > 0:
                cand = aft[z[aft] <= float(frac) * mx]
                if len(cand):
                    offi = int(cand[0])
            feat = [
                float(GRID[oi]),
                float(GRID[pi]),
                float(GRID[offi]),
                float(np.max(dz)),
                float(np.min(dz)),
                mx,
                float(np.max(ddz)),
                float(np.min(ddz)),
                float(np.mean(np.abs(dz)))
            ]
            scale_rows.append(feat)
        A = np.asarray(scale_rows, float)
        flat = A.reshape(-1).tolist()
        for col in (0, 1, 2):
            xx = A[:, col]
            flat.extend([float(np.median(xx)), float(np.quantile(xx, 0.75) - np.quantile(xx, 0.25)), float(np.max(xx) - np.min(xx))])
        flat.extend([float(np.mean(A[:, 3])), float(np.mean(A[:, 4])), float(np.mean(A[:, 5])), float(np.std(A[:, 3])), float(np.std(A[:, 4]))])
        rows.append(flat)
    return finite_model_array(np.asarray(rows, float))
def enforce_geometry(on: np.ndarray, pk: np.ndarray, off: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    on = np.asarray(on, float).copy()
    pk = np.asarray(pk, float).copy()
    off = np.asarray(off, float).copy()
    on = np.clip(on, 0.0, GRID_END - 2 * MIN_GAP)
    pk = np.maximum(pk, on + MIN_GAP)
    pk = np.clip(pk, MIN_GAP, GRID_END - MIN_GAP)
    off = np.maximum(off, pk + MIN_GAP)
    off = np.clip(off, 2 * MIN_GAP, GRID_END)
    return (on, pk, off)
class StableScaler:

    def fit(self, X, sample_weight=None):
        X = np.asarray(X, float)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise RuntimeError('Scaler needs finite 2D training features.')
        w = np.ones(len(X), float) if sample_weight is None else np.asarray(sample_weight, float)
        if len(w) != len(X) or not np.isfinite(w).all() or (w < 0).any() or (w.sum() <= 0):
            raise RuntimeError('Invalid source sample weights.')
        self.mean_ = np.average(X, axis=0, weights=w)
        self.var_ = np.average((X - self.mean_) ** 2, axis=0, weights=w)
        self.var_ = np.maximum(self.var_, 0.0)
        eps = np.finfo(float).eps
        nn = float(w.sum())
        constant = self.var_ <= nn * eps * self.var_ + (nn * self.mean_ * eps) ** 2
        self.scale_ = np.sqrt(self.var_)
        self.scale_[constant | (self.scale_ == 0)] = 1.0
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or (not np.isfinite(X).all()):
            raise RuntimeError('Scaler feature schema/finite check failed.')
        return (X - self.mean_) / self.scale_
def fixed_order_path(p, stay_bias=0.0):
    logp = np.log(np.clip(p, 1e-09, 1.0))
    nn, tt, ss = logp.shape
    dp = np.full((nn, ss), -np.inf)
    dp[:, 0] = logp[:, 0, 0]
    back = np.zeros((nn, tt, ss), np.int8)
    for k in range(1, tt):
        nxt = np.full_like(dp, -np.inf)
        for state in range(ss):
            vs = dp[:, state] + stay_bias
            va = dp[:, state - 1] if state > 0 else np.full(nn, -np.inf)
            keep = vs >= va
            nxt[:, state] = np.where(keep, vs, va) + logp[:, k, state]
            back[:, k, state] = np.where(keep, state, state - 1)
        dp = nxt
    states = np.full(nn, ss - 1, dtype=int)
    path = np.zeros((nn, tt), np.int8)
    ii = np.arange(nn)
    for k in range(tt - 1, -1, -1):
        path[:, k] = states
        if k > 0:
            states = back[ii, k, states]
    return path
def hard_state_features(p, r):
    nn, tt, ss = p.shape
    path = fixed_order_path(p)
    occ = p.mean(axis=1)
    mx = p.max(axis=1)
    centres = (p * GRID[None, :, None]).sum(axis=1) / np.maximum(p.sum(axis=1), 1e-12)
    summaries = np.stack((occ, mx, centres), axis=2).reshape(nn, 15)
    dwell = np.column_stack([(path == state).mean(axis=1) for state in range(5)])
    oi = np.argmax(path == 1, axis=1)
    pi = np.argmax(np.where(path == 2, p[:, :, 2], -np.inf), axis=1)
    fi = np.argmax(path == 4, axis=1)
    ii = np.arange(nn)
    times = np.column_stack((GRID[oi], GRID[pi], GRID[fi]))
    conf = np.column_stack((p[ii, oi, 1], p[ii, pi, 2], p[ii, fi, 4]))
    ent = -(p * np.log(np.clip(p, 1e-09, 1))).sum(axis=2)
    return np.column_stack((
        summaries,
        dwell,
        times,
        conf,
        ent.mean(axis=1),
        ent.max(axis=1),
        r,
        (p[:, :, 1] + p[:, :, 2]).mean(axis=1),
        (p[:, :, 3] + p[:, :, 4]).mean(axis=1)
    ))
def local_profiles(x, probabilities, anchors, response_prob, radius):
    require_predictors_only(x)
    F = finite_model_array(x['state_feat'])
    P = np.asarray(probabilities, float)
    nn = len(F)
    rel = np.linspace(-float(radius), float(radius), 9)
    out = []
    for b in range(3):
        rows = []
        for i in range(nn):
            centre = float(anchors[i, b])
            t = centre + rel
            f = np.column_stack([np.interp(t, GRID, F[i, :, c]) for c in range(F.shape[2])])
            p = np.column_stack([np.interp(t, GRID, P[i, :, c]) for c in range(P.shape[2])])
            seen = ((t >= 0) & (t <= GRID_END)).astype(float)
            context = np.r_[anchors[i], response_prob[i], anchors[i, 1] - anchors[i, 0], anchors[i, 2] - anchors[i, 1]]
            rows.append(np.r_[f.ravel(), p.ravel(), seen, context])
        out.append(np.asarray(rows, float))
    return out
def make_prediction_threshold(_n4, raw, prob, threshold):
    raw = np.asarray(raw, float)
    prob = np.asarray(prob, float)
    on, pk, off = enforce_geometry(raw[:, 0], raw[:, 1], raw[:, 2])
    miss = prob < float(threshold)
    on = on.astype(float)
    pk = pk.astype(float)
    off = off.astype(float)
    on[miss] = np.nan
    pk[miss] = np.nan
    off[miss] = np.nan
    return pd.DataFrame({'pred_response': (~miss).astype(int), 'pred_onset': on, 'pred_peak': pk, 'pred_offset': off})
n4 = SimpleNamespace(
    GRID=GRID,
    RANDOM_SEED=BASE_N4_RANDOM_SEED,
    finite_model_array=finite_model_array,
    predictor_view=predictor_view,
    require_predictors_only=require_predictors_only,
    grouped_folds=grouped_folds,
    subset=subset,
    training_rows=training_rows,
    FitAudit=FitAudit,
    StableScaler=StableScaler,
    hard_state_features=hard_state_features,
    change_branch_features=change_branch_features,
    enforce_geometry=enforce_geometry,
    local_profiles=local_profiles
)

@dataclass
class CoherentStateModel:
    scaler: Any
    clf: Any
def fit_coherent_state(n4, data, state_C, audit, stage_prefix):
    audit.record(data, stage_prefix + '_state_encoder')
    F = n4.finite_model_array(data['state_feat']).astype(float, copy=False)
    X = F.reshape(-1, F.shape[-1])
    y = np.asarray(data['states'], int).reshape(-1)
    w = np.asarray(data['weight'], float)
    fw = np.repeat(w, F.shape[1])
    fw = fw / np.mean(fw)
    if len(X) > 60000:
        idx = np.random.default_rng(stable_seed(
            'COHERENT_STATE_SUBSAMPLE',
            stage_prefix,
            *sorted(set(map(str, data['pid']))),
            float(state_C)
        )).choice(len(X), 60000, replace=False)
        X = X[idx]
        y = y[idx]
        fw = fw[idx]
    scaler = n4.StableScaler().fit(X, sample_weight=fw)
    clf = LogisticRegression(C=float(state_C), class_weight='balanced', max_iter=800, random_state=n4.RANDOM_SEED)
    clf.fit(scaler.transform(X), y, sample_weight=fw)
    return CoherentStateModel(scaler=scaler, clf=clf)
def coherent_state_probabilities(n4, model, x):
    n4.require_predictors_only(x)
    F = n4.finite_model_array(x['state_feat']).astype(float, copy=False)
    nn, tt, dd = F.shape
    p0 = model.clf.predict_proba(model.scaler.transform(F.reshape(-1, dd)))
    P = np.full((nn * tt, 5), 1e-08, float)
    for j, cls in enumerate(model.clf.classes_):
        P[:, int(cls)] = p0[:, j]
    P /= np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    return P.reshape(nn, tt, 5)
def coherent_state_features(n4, P):
    zero_response = np.zeros(len(P), float)
    full = n4.hard_state_features(P, zero_response)
    if full.shape[1] != 31:
        raise RuntimeError(f'Unexpected hard_state_features width {full.shape[1]} (expected 31).')
    return np.column_stack([full[:, :28], full[:, 29:]])
def coherent_ab_features(n4, state_model, x, change_frac):
    P = coherent_state_probabilities(n4, state_model, x)
    SX = coherent_state_features(n4, P)
    CX = n4.change_branch_features(x, float(change_frac))
    return (n4.finite_model_array(np.column_stack([SX, CX])), P)
@dataclass
class CoherentTimingHeads:
    scaler: Any
    regs: Dict[str, Any]
    names: Tuple[str, ...]

    def raw(self, X):
        Z = self.scaler.transform(X)
        return {name: self.regs[name].predict(Z) for name in self.names}
def fit_coherent_timing_heads(n4, X, data, alpha, names, audit, stage_prefix):
    audit.record(data, stage_prefix + '_timing_heads')
    names = tuple(names)
    w = np.asarray(data['weight'], float)
    scaler = n4.StableScaler().fit(X, sample_weight=w)
    Z = scaler.transform(X)
    present = np.asarray(data['present'], int) == 1
    censored = np.asarray(data['censored'], int) == 1
    regs = {}
    for name in names:
        y = np.asarray(data[name], float)
        mask = present & np.isfinite(y)
        if name == 'offset':
            mask &= ~censored
        if int(np.sum(mask)) < 15:
            raise RuntimeError(f'{stage_prefix}: too few rows for {name}.')
        rw = w[mask] / np.mean(w[mask])
        regs[name] = Ridge(alpha=float(alpha)).fit(Z[mask], y[mask], sample_weight=rw)
    return CoherentTimingHeads(scaler=scaler, regs=regs, names=names)
@dataclass
class CoherentABModel:
    n4: Any
    state: CoherentStateModel
    heads: CoherentTimingHeads
    change_frac: float

    def raw(self, x):
        X, P = coherent_ab_features(self.n4, self.state, x, self.change_frac)
        pred = self.heads.raw(X)
        on, pk, off = self.n4.enforce_geometry(np.asarray(pred['onset'], float), np.asarray(pred['peak'], float), np.asarray(pred['offset'], float))
        return (np.column_stack([on, pk, off]), P, X)
def fit_coherent_AB(n4, data, cfg, audit, stage_prefix='AB'):
    state = fit_coherent_state(n4, data, float(cfg['state_C']), audit, stage_prefix)
    X, _ = coherent_ab_features(n4, state, n4.predictor_view(data), float(cfg['change_frac']))
    heads = fit_coherent_timing_heads(n4, X, data, cfg['timing_alpha'], ('onset', 'peak', 'offset'), audit, stage_prefix)
    return CoherentABModel(n4=n4, state=state, heads=heads, change_frac=float(cfg['change_frac']))
@dataclass
class CoherentABCModel:
    n4: Any
    ab: CoherentABModel
    response_scaler: Any
    response_clf: Any
    response_threshold: float

    def raw(self, x):
        raw, P, X = self.ab.raw(x)
        prob = self.response_clf.predict_proba(self.response_scaler.transform(X))[:, 1]
        return (raw, np.asarray(prob, float), P)

    def predict(self, x):
        raw, prob, _ = self.raw(x)
        return make_prediction_threshold(self.n4, raw, prob, self.response_threshold)
def fit_coherent_ABC(n4, data, cfg, audit, stage_prefix='ABC'):
    ab = fit_coherent_AB(n4, data, cfg, audit, stage_prefix=stage_prefix + '_AB')
    X, _ = coherent_ab_features(n4, ab.state, n4.predictor_view(data), ab.change_frac)
    audit.record(data, stage_prefix + '_response_detector')
    w = np.asarray(data['weight'], float)
    y = np.asarray(data['present'], int)
    response_scaler = n4.StableScaler().fit(X, sample_weight=w)
    response_clf = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=n4.RANDOM_SEED)
    response_clf.fit(response_scaler.transform(X), y, sample_weight=w / np.mean(w))
    return CoherentABCModel(n4=n4, ab=ab, response_scaler=response_scaler, response_clf=response_clf, response_threshold=float(cfg['response_threshold']))
def coherent_oof_anchors(n4, data, abc_cfg, outer_forbidden, audit_parent, context):
    pids = sorted(set(map(str, data['pid'])))
    nn = len(data['pid'])
    raw = np.zeros((nn, 3), float)
    rp = np.zeros(nn, float)
    P = np.zeros((nn, len(n4.GRID), 5), float)
    covered = np.zeros(nn, int)
    folds = n4.grouped_folds(pids, stable_seed('COHERENT_LBR_OOF', context, canonical_cfg(abc_cfg), *pids), LOCAL_OOF_FOLDS_COHERENT)
    for held in folds:
        mask = np.isin(data['pid'], held)
        tr = n4.subset(data, ~mask)
        va = n4.subset(data, mask)
        tr_pids = sorted(set(map(str, tr['pid'])))
        audit = n4.FitAudit(tr_pids, set(outer_forbidden) | set(held), context + '/LBR_OOF')
        model = fit_coherent_ABC(n4, tr, abc_cfg, audit, stage_prefix='LBR_OOF_ABC')
        r, p, pp = model.raw(n4.predictor_view(va))
        raw[mask] = r
        rp[mask] = p
        P[mask] = pp
        covered[mask] += 1
        audit_parent.rows.extend(audit.rows)
    if not np.all(covered == 1):
        raise RuntimeError('Coherent LBR OOF anchor coverage failed.')
    return (raw, rp, P)
@dataclass
class CoherentFullModel:
    n4: Any
    abc: CoherentABCModel
    local_cfg: Dict[str, float]
    scalers: List[Any]
    regs: List[Any]

    def refined_raw(self, x, coarse_raw=None, rp=None, P=None):
        if coarse_raw is None or rp is None or P is None:
            coarse_raw, rp, P = self.abc.raw(x)
        coarse_raw = np.asarray(coarse_raw, float).copy()
        on, pk, off = self.n4.enforce_geometry(coarse_raw[:, 0], coarse_raw[:, 1], coarse_raw[:, 2])
        coarse_raw = np.column_stack([on, pk, off])
        profiles = self.n4.local_profiles(x, P, coarse_raw, rp, float(self.local_cfg['radius']))
        refined = np.asarray(coarse_raw, float).copy()
        for b in range(3):
            delta = self.regs[b].predict(self.scalers[b].transform(profiles[b]))
            radius = float(self.local_cfg['radius'])
            refined[:, b] += float(self.local_cfg['strength']) * np.clip(delta, -radius, radius)
        on, pk, off = self.n4.enforce_geometry(refined[:, 0], refined[:, 1], refined[:, 2])
        return np.column_stack([on, pk, off])

    def predict(self, x):
        raw, rp, P = self.abc.raw(x)
        refined = self.refined_raw(x, raw, rp, P)
        return make_prediction_threshold(self.n4, refined, rp, self.abc.response_threshold)
def fit_coherent_full(n4, data, abc_cfg, local_cfg, audit, outer_forbidden, context):
    abc = fit_coherent_ABC(n4, data, abc_cfg, audit, stage_prefix='FULL_ABC')
    raw_oof, rp_oof, P_oof = coherent_oof_anchors(n4, data, abc_cfg, outer_forbidden, audit, context)
    profiles = n4.local_profiles(n4.predictor_view(data), P_oof, raw_oof, rp_oof, float(local_cfg['radius']))
    audit.record(data, 'FULL_LBR_residual_heads_on_source_OOF_anchors')
    present = np.asarray(data['present'], int) == 1
    censored = np.asarray(data['censored'], int) == 1
    scalers = []
    regs = []
    for b, name in enumerate(('onset', 'peak', 'offset')):
        y = np.asarray(data[name], float)
        mask = present & np.isfinite(y)
        if name == 'offset':
            mask &= ~censored
        scaler = n4.StableScaler().fit(profiles[b][mask])
        residual = y[mask] - raw_oof[mask, b]
        reg = Ridge(alpha=float(local_cfg['local_alpha'])).fit(scaler.transform(profiles[b][mask]), residual)
        scalers.append(scaler)
        regs.append(reg)
    return CoherentFullModel(n4=n4, abc=abc, local_cfg=dict(local_cfg), scalers=scalers, regs=regs)
def seed_cache_dir(seed: int) -> Path:
    if int(seed) == 1:
        return N4_10SEED_DIR / 'seed_01' / 'fresh_cache'
    return HIERARCHICAL_10SEED_DIR / f'seed_{int(seed):02d}' / 'fresh_cache'
def seed_selected_table(seed: int) -> Path:
    if int(seed) == 1:
        return HIERARCHICAL_SEED1_DIR / 'tables' / 'Table3_B0_STRICT_HIERARCHICAL_SELECTED_CONFIGS_SEED01.csv'
    return HIERARCHICAL_10SEED_DIR / f'seed_{int(seed):02d}' / 'tables' / 'Table3_STRICT_SELECTED_CONFIGS.csv'
def load_seed_caches(seed: int):
    cache_dir = seed_cache_dir(seed)
    if not cache_dir.is_dir():
        raise FileNotFoundError(f'Expected completed seed cache does not exist:\n{cache_dir}')
    caches = {}
    for pid in ALL_PIDS:
        path = cache_dir / f'{pid}.npz'
        if not path.is_file():
            raise FileNotFoundError(f'Missing completed cache: {path}')
        with np.load(path, allow_pickle=False) as z:
            item = {}
            for prefix, split in (('tr', 'train'), ('te', 'test')):
                fields = (*PREDICTOR_KEYS, *LABEL_KEYS)
                missing = [k for k in fields if f'{prefix}_{k}' not in z.files]
                if missing:
                    raise RuntimeError(f'{path.name}/{split} missing {missing}')
                d = {k: np.array(z[f'{prefix}_{k}'], copy=True) for k in fields}
                nn = len(d['present'])
                if d['state_feat'].shape != (nn, len(GRID), 9):
                    raise RuntimeError(f"{path.name}/{split}: state feature shape mismatch {d['state_feat'].shape}")
                for key in PREDICTOR_KEYS:
                    d[key] = finite_model_array(d[key])
                d['pid'] = np.full(nn, pid, dtype='U5')
                d['weight'] = np.ones(nn, float)
                item[split] = d
            caches[pid] = item
    return (caches, cache_dir)
def normalize_pid(x) -> str:
    s = str(x).strip().upper()
    if s.startswith('VP'):
        digits = ''.join((c for c in s if c.isdigit()))
        if digits:
            return f'VP{int(digits):03d}'
    digits = ''.join((c for c in s if c.isdigit()))
    return f'VP{int(digits):03d}' if digits else s
def parse_json_cfg(x):
    if isinstance(x, dict):
        return dict(x)
    s = str(x).strip()
    if not s:
        return None
    return {k: float(v) for k, v in json.loads(s).items()}
def load_full_config(seed: int, pid: str):
    table = seed_selected_table(seed)
    if not table.is_file():
        raise FileNotFoundError(f'Required model-configuration table does not exist:\n{table}')
    q = pd.read_csv(table)
    pid_col = 'participant' if 'participant' in q.columns else 'outer_target' if 'outer_target' in q.columns else None
    if pid_col is None:
        raise RuntimeError(f'Selected-config table has no participant column: {list(q.columns)}')
    if 'row' not in q.columns:
        raise RuntimeError(f'Selected-config table has no row column: {list(q.columns)}')
    pids = q[pid_col].map(normalize_pid)
    full = q[(pids == pid) & (q['row'].astype(str) == ROW_FULL)].copy()
    if len(full) != 1:
        raise RuntimeError(f'Expected exactly one full configuration for seed={seed}, participant={pid}; found {len(full)}.')
    row = full.iloc[0]
    abc_cfg = parse_json_cfg(row['global_cfg'])
    local_cfg = parse_json_cfg(row['local_cfg'])
    expected_global = {'state_C', 'change_frac', 'timing_alpha', 'response_threshold'}
    expected_local = {'radius', 'local_alpha', 'strength'}
    if abc_cfg is None or set(abc_cfg) != expected_global:
        raise RuntimeError(f'Unexpected ABC config for seed={seed}, {pid}: {abc_cfg}')
    if local_cfg is None or set(local_cfg) != expected_local:
        raise RuntimeError(f'Unexpected LBR config for seed={seed}, {pid}: {local_cfg}')
    return (abc_cfg, local_cfg, table)
def source_neutral_values(train: Mapping[str, np.ndarray]):
    present = np.asarray(train['present'], int) == 1
    censored = np.asarray(train['censored'], int) == 1
    onset = np.asarray(train['onset'], float)
    peak = np.asarray(train['peak'], float)
    offset = np.asarray(train['offset'], float)
    m_on = present & np.isfinite(onset)
    m_pk = present & np.isfinite(peak)
    m_off = present & ~censored & np.isfinite(offset)
    return {'onset': float(np.median(onset[m_on])), 'peak': float(np.median(peak[m_pk])), 'offset': float(np.median(offset[m_off]))}
def make_all_pass_prediction(refined_raw, response_prob):
    refined_raw = np.asarray(refined_raw, float)
    on, pk, off = enforce_geometry(refined_raw[:, 0], refined_raw[:, 1], refined_raw[:, 2])
    return pd.DataFrame({'pred_response': np.ones(len(refined_raw), dtype=int), 'pred_onset': on, 'pred_peak': pk, 'pred_offset': off})
def functional_predictions(model: CoherentFullModel, x, train):
    coarse, rp, P = model.abc.raw(x)
    full_refined = model.refined_raw(x, coarse, rp, P)
    full = make_prediction_threshold(n4, full_refined, rp, model.abc.response_threshold)
    no_lbr = make_prediction_threshold(n4, coarse, rp, model.abc.response_threshold)
    no_c = make_all_pass_prediction(full_refined, rp)
    neutral = source_neutral_values(train)
    raw_a = np.asarray(coarse, float).copy()
    raw_a[:, 0] = neutral['onset']
    raw_a[:, 1] = neutral['peak']
    a_refined = model.refined_raw(x, raw_a, rp, P)
    no_a = make_prediction_threshold(n4, a_refined, rp, model.abc.response_threshold)
    raw_b = np.asarray(coarse, float).copy()
    raw_b[:, 2] = neutral['offset']
    b_refined = model.refined_raw(x, raw_b, rp, P)
    no_b = make_prediction_threshold(n4, b_refined, rp, model.abc.response_threshold)
    return (
        {'FULL': full, 'A_OUTPUT_NEUTRALIZED': no_a, 'B_OUTPUT_NEUTRALIZED': no_b, 'C_GATE_NEUTRALIZED': no_c, 'LBR_NEUTRALIZED': no_lbr},
        neutral,
        rp,
        coarse,
        full_refined
    )
def balanced_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, int)
    p = np.asarray(p, int)
    pos = y == 1
    neg = y == 0
    sens = np.mean(p[pos] == 1) if np.any(pos) else np.nan
    spec = np.mean(p[neg] == 0) if np.any(neg) else np.nan
    return float(np.nanmean([sens, spec]))
def participant_metrics(truth, pred):
    present = np.asarray(truth['present'], int) == 1
    censored = np.asarray(truth['censored'], int) == 1
    decision = pred['pred_response'].to_numpy(int)
    ton = np.asarray(truth['onset'], float)
    tpk = np.asarray(truth['peak'], float)
    toff = np.asarray(truth['offset'], float)
    pon = pred['pred_onset'].to_numpy(float)
    ppk = pred['pred_peak'].to_numpy(float)
    poff = pred['pred_offset'].to_numpy(float)
    out = {'response_balanced_accuracy': balanced_accuracy(np.asarray(truth['present'], int), decision)}
    conditional = {}
    for name, tt, pp, exact in ((
        'onset',
        ton,
        pon,
        present & np.isfinite(ton)
    ), ('peak', tpk, ppk, present & np.isfinite(tpk)), ('offset', toff, poff, present & ~censored & np.isfinite(toff))):
        m = exact & (decision == 1) & np.isfinite(pp)
        conditional[f'conditional_{name}_mae_sec'] = float(np.mean(np.abs(pp[m] - tt[m]))) if np.any(m) else np.nan
        conditional[f'conditional_{name}_localized_n'] = int(np.sum(m))
    dur_true = toff - ton
    dur_pred = poff - pon
    exact_dur = present & ~censored & np.isfinite(dur_true)
    m = exact_dur & (decision == 1) & np.isfinite(dur_pred)
    conditional['conditional_duration_mae_sec'] = float(np.mean(np.abs(dur_pred[m] - dur_true[m]))) if np.any(m) else np.nan
    conditional['conditional_duration_localized_n'] = int(np.sum(m))
    out.update(conditional)
    e2e_values = {}
    for name, tt, pp, exact in ((
        'onset',
        ton,
        pon,
        present & np.isfinite(ton)
    ), ('peak', tpk, ppk, present & np.isfinite(tpk)), ('offset', toff, poff, present & ~censored & np.isfinite(toff))):
        idx = np.flatnonzero(exact)
        err = np.full(len(idx), MISS_PENALTY_SEC, float)
        if len(idx):
            good = (decision[idx] == 1) & np.isfinite(pp[idx])
            err[good] = np.abs(pp[idx][good] - tt[idx][good])
        value = float(np.mean(err)) if len(err) else np.nan
        out[f'e2e_{name}_mae_sec'] = value
        e2e_values[name] = value
    idx = np.flatnonzero(exact_dur)
    err = np.full(len(idx), MISS_PENALTY_SEC, float)
    if len(idx):
        good = (decision[idx] == 1) & np.isfinite(dur_pred[idx])
        err[good] = np.abs(dur_pred[idx][good] - dur_true[idx][good])
    value = float(np.mean(err)) if len(err) else np.nan
    out['e2e_duration_mae_sec'] = value
    e2e_values['duration'] = value
    vals = np.asarray([e2e_values['onset'], e2e_values['peak'], e2e_values['offset'], e2e_values['duration']], float)
    scales = np.asarray([1.0, 1.0, 1.5, 1.5], float)
    out['e2e_selection_loss'] = float(np.nanmean(vals / scales))
    out['predicted_response_rate'] = float(np.mean(decision == 1))
    out['n_positive'] = int(np.sum(present))
    out['n_negative'] = int(np.sum(~present))
    return out
def per_trial_frame(seed, pid, variant, truth, pred, response_prob, neutral, abc_cfg, local_cfg):
    return pd.DataFrame({
        'repeat_seed': int(seed),
        'participant': pid,
        'variant': variant,
        'trial_index': np.arange(len(pred), dtype=int),
        'truth_present': np.asarray(truth['present'], int),
        'truth_onset': np.asarray(truth['onset'], float),
        'truth_peak': np.asarray(truth['peak'], float),
        'truth_offset': np.asarray(truth['offset'], float),
        'truth_censored': np.asarray(truth['censored'], int),
        'pred_response': pred['pred_response'].to_numpy(int),
        'response_probability_from_C': np.asarray(response_prob, float),
        'pred_onset': pred['pred_onset'].to_numpy(float),
        'pred_peak': pred['pred_peak'].to_numpy(float),
        'pred_offset': pred['pred_offset'].to_numpy(float),
        'source_neutral_onset': float(neutral['onset']),
        'source_neutral_peak': float(neutral['peak']),
        'source_neutral_offset': float(neutral['offset']),
        'selected_abc_cfg': canonical_cfg(abc_cfg),
        'selected_lbr_cfg': canonical_cfg(local_cfg)
    })
def protected_self_tests():
    assert len(ALL_PIDS) == 26
    assert len(GRID) == 41
    assert set(SEEDS).issubset(set(range(1, 11)))
    assert len(set(SEEDS)) == len(SEEDS)
    dummy_raw = np.asarray([[1.0, 4.0, 10.0], [2.0, 5.0, 12.0]])
    dummy_prob = np.asarray([0.9, 0.1])
    pred = make_prediction_threshold(n4, dummy_raw, dummy_prob, 0.5)
    assert pred['pred_response'].tolist() == [1, 0]
    assert np.isfinite(pred.loc[0, 'pred_onset'])
    assert np.isnan(pred.loc[1, 'pred_onset'])
    print('Standalone functional necessity protected self-tests: PASS')
def main():
    mount_drive()
    protected_self_tests()
    if not SHIN_ROOT.is_dir():
        raise FileNotFoundError(f'Expected SHIN project folder does not exist:\n{SHIN_ROOT}')
    for sub in ('checkpoints', 'results', 'audit'):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    print('=' * 180)
    print('HEMOBOUNDARY — FUNCTIONAL NECESSITY')
    print('=' * 180)
    print('Assigned seeds                 :', list(SEEDS))
    print('Participants per seed          : 26')
    print('Nested hyperparameter search   : NO')
    print('External core .py required     : NO')
    print('Source-code Drive search       : NO')
    print('Seed-specific completed banks  : YES')
    print('Seed/participant selected cfg   : YES')
    print('Output                         :', OUT_DIR)
    print()
    all_metric_parts = []
    all_trial_parts = []
    all_audit_parts = []
    for seed in SEEDS:
        print('\n' + '=' * 180)
        print(f'SEED {seed:02d}')
        print('=' * 180)
        configure_seed(seed)
        caches, cache_dir = load_seed_caches(seed)
        print('Cache:', cache_dir)
        print('Config table:', seed_selected_table(seed))
        for index, pid in enumerate(ALL_PIDS, 1):
            cp_metrics = OUT_DIR / 'checkpoints' / f'seed_{seed:02d}_{pid}_METRICS.csv'
            cp_trials = OUT_DIR / 'checkpoints' / f'seed_{seed:02d}_{pid}_TRIALS.csv.gz'
            cp_audit = OUT_DIR / 'checkpoints' / f'seed_{seed:02d}_{pid}_AUDIT.csv'
            if cp_metrics.is_file() and cp_trials.is_file() and cp_audit.is_file():
                print(f'[seed {seed:02d}] [{index:02d}/26] {pid}: checkpoint')
                all_metric_parts.append(pd.read_csv(cp_metrics))
                all_trial_parts.append(pd.read_csv(cp_trials))
                all_audit_parts.append(pd.read_csv(cp_audit))
                continue
            abc_cfg, local_cfg, cfg_table = load_full_config(seed, pid)
            source_pids = [p for p in ALL_PIDS if p != pid]
            train = training_rows(caches, source_pids)
            truth = caches[pid]['test']
            x = predictor_view(truth)
            audit = FitAudit(source_pids, [pid], f'FUNCTIONAL_NECESSITY/seed={seed:02d}/outer={pid}')
            print(f'[seed {seed:02d}] [{index:02d}/26] {pid}: fitting full model', flush=True)
            model = fit_coherent_full(n4, train, abc_cfg, local_cfg, audit, outer_forbidden={pid}, context=f'FUNCTIONAL_NECESSITY/seed={seed:02d}/outer={pid}')
            variants, neutral, rp, coarse, full_refined = functional_predictions(model, x, train)
            native_full = model.predict(x)
            for col in ('pred_onset', 'pred_peak', 'pred_offset'):
                a = native_full[col].to_numpy(float)
                b = variants['FULL'][col].to_numpy(float)
                if not np.allclose(a, b, atol=1e-10, rtol=0.0, equal_nan=True):
                    raise RuntimeError(f'FULL parity failed for {pid}/{col}.')
            if not np.array_equal(native_full['pred_response'].to_numpy(int), variants['FULL']['pred_response'].to_numpy(int)):
                raise RuntimeError(f'FULL response parity failed for {pid}.')
            metric_rows = []
            trial_rows = []
            for variant, pred in variants.items():
                mm = participant_metrics(truth, pred)
                metric_rows.append({
                    'repeat_seed': int(seed),
                    'participant': pid,
                    'variant': variant,
                    **mm,
                    'source_neutral_onset': float(neutral['onset']),
                    'source_neutral_peak': float(neutral['peak']),
                    'source_neutral_offset': float(neutral['offset']),
                    'selected_abc_cfg': canonical_cfg(abc_cfg),
                    'selected_lbr_cfg': canonical_cfg(local_cfg)
                })
                trial_rows.append(per_trial_frame(seed, pid, variant, truth, pred, rp, neutral, abc_cfg, local_cfg))
            metric_df = pd.DataFrame(metric_rows)
            trial_df = pd.concat(trial_rows, ignore_index=True)
            audit_df = pd.DataFrame(audit.rows)
            audit_df.insert(0, 'repeat_seed', int(seed))
            audit_df.insert(1, 'outer_participant', pid)
            audit_df['selected_abc_cfg'] = canonical_cfg(abc_cfg)
            audit_df['selected_lbr_cfg'] = canonical_cfg(local_cfg)
            audit_df['selected_config_source'] = str(cfg_table)
            audit_df['seed_cache_source'] = str(cache_dir)
            atomic_csv(metric_df, cp_metrics)
            atomic_csv(trial_df, cp_trials)
            atomic_csv(audit_df, cp_audit)
            all_metric_parts.append(metric_df)
            all_trial_parts.append(trial_df)
            all_audit_parts.append(audit_df)
            del (model, train, truth, x, variants, trial_rows, trial_df)
            gc.collect()
        del caches
        gc.collect()
    metrics = pd.concat(all_metric_parts, ignore_index=True)
    trials = pd.concat(all_trial_parts, ignore_index=True)
    audits = pd.concat(all_audit_parts, ignore_index=True)
    atomic_csv(metrics, OUT_DIR / 'results' / 'PARTICIPANT_METRICS.csv')
    atomic_csv(trials, OUT_DIR / 'results' / 'PER_TRIAL_PREDICTIONS.csv.gz')
    atomic_csv(audits, OUT_DIR / 'audit' / 'FIT_MEMBERSHIP_AUDIT.csv.gz')
    summary = metrics.groupby('variant', as_index=False).agg(
        rows=('participant', 'size'),
        participants=('participant', 'nunique'),
        seeds=('repeat_seed', 'nunique'),
        response_BA=('response_balanced_accuracy', 'mean'),
        onset_MAE=('e2e_onset_mae_sec', 'mean'),
        peak_MAE=('e2e_peak_mae_sec', 'mean'),
        offset_MAE=('e2e_offset_mae_sec', 'mean'),
        duration_MAE=('e2e_duration_mae_sec', 'mean'),
        selection_loss=('e2e_selection_loss', 'mean')
    ).sort_values('variant').reset_index(drop=True)
    atomic_csv(summary, OUT_DIR / 'results' / 'SHARD_SUMMARY.csv')
    atomic_json({
        'analysis': 'Frozen functional necessity / output-intervention ablation',
        'job_dir': JOB_DIR,
        'seeds': list(SEEDS),
        'participants_per_seed': 26,
        'variants': ['FULL', 'A_OUTPUT_NEUTRALIZED', 'B_OUTPUT_NEUTRALIZED', 'C_GATE_NEUTRALIZED', 'LBR_NEUTRALIZED'],
        'nested_hyperparameter_search': False,
        'external_core_file_required': False,
        'source_code_search': False,
        'seed_specific_existing_caches': True,
        'outer_target_used_for_fitting': False,
        'neutral_values_source': "25 outer-source participants' TRAIN banks only",
        'interpretation': 'functional output intervention; not a fully factorial architecture independence experiment'
    }, OUT_DIR / 'audit' / 'RUN_MANIFEST.json')
    print('\n' + '=' * 180)
    print('JOB COMPLETE')
    print('=' * 180)
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print()
    print('Saved:', OUT_DIR)
if __name__ == '__main__':
    main()
