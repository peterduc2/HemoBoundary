"""Generalization to alternative response morphologies."""
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
OUT_ROOT = SHIN_ROOT / '111_OOD_HRF_SHAPE_STRESS_10SEED_V1_STANDALONE'
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
FULL_T = np.arange(-5.0, GRID_END + GRID_DT / 2.0, GRID_DT)
POST_MASK = FULL_T >= 0
N_TEST_POS = 24
N_TEST_NEG = 12
GENERATED_FAMILIES = ('MATCHED_HALF_COSINE_NEW', 'OOD_SKEWED_BETA', 'OOD_BIPHASIC_UNDERSHOOT', 'OOD_BIEXPONENTIAL_TAIL', 'OOD_SHOULDER_DOUBLE_PEAK')
EVAL_FAMILIES = ('PAPER_MATCHED_CACHE', *GENERATED_FAMILIES)
METHODS = ('FULL', 'ABC_PARENT')

@dataclass
class NoiseProfile:
    n_channels: int
    hbo_scale: np.ndarray
    hbr_scale: np.ndarray
    phi_hbo: float
    phi_hbr: float
def preprocessed_path(pid: str) -> Path:
    return SHIN_ROOT / '02_preprocessed' / f'{pid}_WG_ALIGNED_PREPROCESSED.npz'
def safe_lag1(a: np.ndarray) -> float:
    vals = []
    for row in np.asarray(a, float):
        x = row[:-1]
        y = row[1:]
        if len(x) >= 3 and np.std(x) > 1e-10 and (np.std(y) > 1e-10):
            vals.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.clip(np.nanmedian(vals) if vals else 0.5, 0.0, 0.95))
def estimate_noise_profile(path: Path) -> NoiseProfile:
    if not path.is_file():
        raise FileNotFoundError(f'Missing Shin preprocessed file: {path}')
    with np.load(path, allow_pickle=False) as z:
        required = ('nirs_times', 'hbo_causal_lp02_epochs', 'hbr_causal_lp02_epochs')
        missing = [k for k in required if k not in z.files]
        if missing:
            raise RuntimeError(f'{path.name}: missing {missing}')
        t = np.asarray(z['nirs_times'], float)
        H = np.asarray(z['hbo_causal_lp02_epochs'], float)
        R = np.asarray(z['hbr_causal_lp02_epochs'], float)
    bm = (t >= -5.0) & (t < 0.0)
    if int(np.sum(bm)) < 4:
        raise RuntimeError(f'{path.name}: insufficient pre-cue samples.')
    C = min(H.shape[1], R.shape[1])
    H = H[:, :C, :][:, :, bm]
    R = R[:, :C, :][:, :, bm]
    H = H - np.nanmedian(H, axis=2, keepdims=True)
    R = R - np.nanmedian(R, axis=2, keepdims=True)
    hs = np.nanmedian(np.nanstd(H, axis=2), axis=0)
    rs = np.nanmedian(np.nanstd(R, axis=2), axis=0)
    h_fallback = np.nanmedian(hs[np.isfinite(hs) & (hs > 0)]) if np.any(np.isfinite(hs) & (hs > 0)) else 1.0
    r_fallback = np.nanmedian(rs[np.isfinite(rs) & (rs > 0)]) if np.any(np.isfinite(rs) & (rs > 0)) else 1.0
    hs = np.where(np.isfinite(hs) & (hs > 1e-08), hs, h_fallback)
    rs = np.where(np.isfinite(rs) & (rs > 1e-08), rs, r_fallback)
    sample_H = H[:min(24, len(H))].reshape(-1, H.shape[-1])
    sample_R = R[:min(24, len(R))].reshape(-1, R.shape[-1])
    return NoiseProfile(
        n_channels=int(C),
        hbo_scale=np.asarray(hs, float),
        hbr_scale=np.asarray(rs, float),
        phi_hbo=safe_lag1(sample_H),
        phi_hbr=safe_lag1(sample_R)
    )
def ar_noise(scale: np.ndarray, phi: float, rng: np.random.Generator, n_time: int) -> np.ndarray:
    C = len(scale)
    x = np.zeros((C, n_time), float)
    common = rng.normal(size=n_time)
    for k in range(n_time):
        eps = 0.75 * rng.normal(size=C) + 0.25 * common[k]
        if k == 0:
            x[:, k] = eps
        else:
            x[:, k] = float(phi) * x[:, k - 1] + math.sqrt(max(1.0 - float(phi) ** 2, 1e-06)) * eps
    return x * np.asarray(scale, float)[:, None]
def base_noise_trial(profile: NoiseProfile, rng: np.random.Generator):
    H = ar_noise(profile.hbo_scale, profile.phi_hbo, rng, len(FULL_T))
    R = ar_noise(profile.hbr_scale, profile.phi_hbr, rng, len(FULL_T))
    systemic = rng.uniform(0.0, 0.8) * float(np.nanmedian(profile.hbo_scale)) * np.sin(2.0 * np.pi * rng.uniform(0.06, 0.12) * FULL_T + rng.uniform(0.0, 2.0 * np.pi))
    H = H + systemic[None, :]
    R = R + rng.uniform(0.5, 1.0) * systemic[None, :]
    return (H, R)
def baseline_z(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, float)
    bm = FULL_T < 0
    med = np.nanmedian(arr[:, bm], axis=1, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(arr[:, bm] - med), axis=1, keepdims=True)
    sd = np.nanstd(arr[:, bm], axis=1, keepdims=True)
    scale = np.where(np.isfinite(mad) & (mad > 1e-08), mad, np.where(sd > 1e-08, sd, 1.0))
    return (arr - med) / scale
def evidence(hbo: np.ndarray, hbr: np.ndarray) -> Dict[str, np.ndarray]:
    H = baseline_z(hbo)[:, POST_MASK]
    R = baseline_z(hbr)[:, POST_MASK]
    ghbo = np.nanmedian(H, axis=0)
    ghbr = np.nanmedian(R, axis=0)
    anti = 0.5 * (ghbo - ghbr)
    dhbo = np.gradient(ghbo, GRID_DT)
    dhbr = np.gradient(ghbr, GRID_DT)
    danti = np.gradient(anti, GRID_DT)
    dH = np.gradient(H, GRID_DT, axis=1)
    dR = np.gradient(R, GRID_DT, axis=1)
    cons_hbo = np.mean(np.sign(dH) == np.sign(dhbo)[None, :], axis=0)
    cons_hbr = np.mean(np.sign(dR) == np.sign(dhbr)[None, :], axis=0)
    iqr_hbo = np.nanquantile(H, 0.75, axis=0) - np.nanquantile(H, 0.25, axis=0)
    iqr_hbr = np.nanquantile(R, 0.75, axis=0) - np.nanquantile(R, 0.25, axis=0)
    return {
        'H': H,
        'R': R,
        'ghbo': ghbo,
        'ghbr': ghbr,
        'anti': anti,
        'dhbo': dhbo,
        'dhbr': dhbr,
        'danti': danti,
        'cons_hbo': cons_hbo,
        'cons_hbr': cons_hbr,
        'iqr_hbo': iqr_hbo,
        'iqr_hbr': iqr_hbr
    }
def whole_vector(ev: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([ev['ghbo'], ev['ghbr'], ev['anti'], ev['danti']]).astype(np.float32)
def state_features(ev: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([
        ev['ghbo'],
        ev['ghbr'],
        ev['anti'],
        ev['dhbo'],
        ev['dhbr'],
        ev['danti'],
        ev['cons_hbo'],
        ev['cons_hbr'],
        0.5 * (ev['iqr_hbo'] + ev['iqr_hbr'])
    ]).astype(np.float32)
def beta_bump(u: np.ndarray, mode_fraction: float, concentration: float) -> np.ndarray:
    u = np.asarray(u, float)
    p = float(np.clip(mode_fraction, 0.03, 0.97))
    k = float(max(concentration, 2.5))
    a = 1.0 + k * p
    b = 1.0 + k * (1.0 - p)
    out = np.zeros_like(u, dtype=float)
    mask = (u > 0.0) & (u < 1.0)
    if np.any(mask):
        uu = u[mask]
        log_y = (a - 1.0) * np.log(uu) + (b - 1.0) * np.log1p(-uu)
        log_norm = (a - 1.0) * math.log(p) + (b - 1.0) * math.log(1.0 - p)
        out[mask] = np.exp(log_y - log_norm)
    return out
def original_half_cosine_shape(rng: np.random.Generator):
    onset = float(rng.uniform(0.5, 4.5))
    rise = float(rng.uniform(1.0, 3.5))
    active = float(rng.uniform(0.5, 3.0))
    if rng.random() < 0.18:
        decay = float(rng.uniform(7.0, 12.0))
    else:
        decay = float(rng.uniform(2.0, 8.0))
    peak = onset + rise
    active_end = peak + active
    offset = active_end + decay
    y = np.zeros(len(GRID), float)
    for i, tt in enumerate(GRID):
        if tt < onset:
            y[i] = 0.0
        elif tt < peak:
            u = (tt - onset) / max(rise, 1e-08)
            y[i] = 0.5 - 0.5 * np.cos(np.pi * u)
        elif tt < active_end:
            y[i] = 1.0
        elif tt < offset:
            u = (tt - active_end) / max(decay, 1e-08)
            y[i] = 0.5 + 0.5 * np.cos(np.pi * u)
        else:
            y[i] = 0.0
    return (y, onset, peak, offset, {'rise_sec': rise, 'active_sec': active, 'recovery_sec': decay})
def common_ood_timing(rng: np.random.Generator):
    onset = float(rng.uniform(0.5, 4.0))
    peak_delay = float(rng.uniform(1.2, 3.8))
    long_recovery = bool(rng.random() < 0.18)
    if long_recovery:
        recovery = float(rng.uniform(11.0, 18.0))
    else:
        recovery = float(rng.uniform(3.5, 9.0))
    peak_target = onset + peak_delay
    offset = peak_target + recovery
    return {'onset': onset, 'peak_delay': peak_delay, 'peak_target': peak_target, 'recovery': recovery, 'offset': offset, 'long_recovery': long_recovery}
def skewed_beta_shape(timing, rng: np.random.Generator):
    onset = float(timing['onset'])
    offset = float(timing['offset'])
    duration = offset - onset
    peak_target = float(timing['peak_target'])
    p = (peak_target - onset) / max(duration, 1e-08)
    concentration = float(rng.uniform(4.5, 10.0))
    u = (GRID - onset) / max(duration, 1e-08)
    y = beta_bump(u, p, concentration)
    return (y, onset, peak_target, offset, {'mode_fraction': p, 'concentration': concentration})
def biphasic_undershoot_shape(timing, rng: np.random.Generator):
    onset = float(timing['onset'])
    offset = float(timing['offset'])
    duration = offset - onset
    peak_target = float(timing['peak_target'])
    p = (peak_target - onset) / max(duration, 1e-08)
    concentration = float(rng.uniform(5.0, 10.0))
    u = (GRID - onset) / max(duration, 1e-08)
    positive = beta_bump(u, p, concentration)
    undershoot_duration = float(rng.uniform(2.0, 5.0))
    undershoot_fraction = float(rng.uniform(0.15, 0.35))
    uu = (GRID - offset) / undershoot_duration
    undershoot = beta_bump(uu, float(rng.uniform(0.4, 0.6)), float(rng.uniform(4.0, 8.0)))
    y = positive - undershoot_fraction * undershoot
    return (
        y,
        onset,
        peak_target,
        offset,
        {'main_mode_fraction': p, 'main_concentration': concentration, 'undershoot_duration_sec': undershoot_duration, 'undershoot_fraction': undershoot_fraction}
    )
def biexponential_tail_shape(timing, rng: np.random.Generator):
    onset = float(timing['onset'])
    offset = float(timing['offset'])
    duration = offset - onset
    target_peak_delay = float(timing['peak_delay'])
    tau_r = float(rng.uniform(0.45, 1.15))
    tau_d = float(tau_r * (math.exp(target_peak_delay / tau_r) - 1.0))
    tau_d = float(np.clip(tau_d, 2.0, 80.0))
    taper_start_frac = float(rng.uniform(0.62, 0.78))

    def evaluate(tt):
        tt = np.asarray(tt, float)
        u = tt - onset
        y = np.zeros_like(u, dtype=float)
        m = (u >= 0.0) & (tt < offset)
        if np.any(m):
            um = u[m]
            raw = (1.0 - np.exp(-um / tau_r)) * np.exp(-um / tau_d)
            taper_start = taper_start_frac * duration
            taper = np.ones_like(um)
            tm = um > taper_start
            if np.any(tm):
                v = (um[tm] - taper_start) / max(duration - taper_start, 1e-08)
                taper[tm] = 0.5 * (1.0 + np.cos(np.pi * np.clip(v, 0.0, 1.0)))
            y[m] = raw * taper
        return y
    dense_t = np.linspace(onset, min(offset, onset + max(duration, 0.001)), 5000)
    dense_y = evaluate(dense_t)
    mx = float(np.max(dense_y))
    if not np.isfinite(mx) or mx <= 1e-12:
        raise RuntimeError('Biexponential stress waveform degenerated.')
    peak = float(dense_t[int(np.argmax(dense_y))])
    y = evaluate(GRID) / mx
    return (y, onset, peak, offset, {'tau_r_sec': tau_r, 'tau_d_sec': tau_d, 'taper_start_fraction': taper_start_frac, 'target_peak_delay_sec': target_peak_delay})
def shoulder_double_peak_shape(timing, rng: np.random.Generator):
    onset = float(timing['onset'])
    offset = float(timing['offset'])
    duration = offset - onset
    peak_target = float(timing['peak_target'])
    p1 = (peak_target - onset) / max(duration, 1e-08)
    p2 = float(np.clip(p1 + rng.uniform(0.16, 0.32), p1 + 0.08, 0.88))
    c1 = float(rng.uniform(7.0, 12.0))
    c2 = float(rng.uniform(8.0, 14.0))
    shoulder_amp = float(rng.uniform(0.2, 0.45))

    def evaluate(tt):
        u = (np.asarray(tt, float) - onset) / max(duration, 1e-08)
        return beta_bump(u, p1, c1) + shoulder_amp * beta_bump(u, p2, c2)
    dense_t = np.linspace(onset, offset, 5000)
    dense_y = evaluate(dense_t)
    mx = float(np.max(dense_y))
    if not np.isfinite(mx) or mx <= 1e-12:
        raise RuntimeError('Shoulder stress waveform degenerated.')
    peak = float(dense_t[int(np.argmax(dense_y))])
    y = evaluate(GRID) / mx
    y[(GRID <= onset) | (GRID >= offset)] = 0.0
    return (
        y,
        onset,
        peak,
        offset,
        {'main_mode_fraction': p1, 'shoulder_mode_fraction': p2, 'main_concentration': c1, 'shoulder_concentration': c2, 'shoulder_amplitude': shoulder_amp}
    )
def make_family_shape(family: str, timing, rng: np.random.Generator):
    if family == 'OOD_SKEWED_BETA':
        return skewed_beta_shape(timing, rng)
    if family == 'OOD_BIPHASIC_UNDERSHOOT':
        return biphasic_undershoot_shape(timing, rng)
    if family == 'OOD_BIEXPONENTIAL_TAIL':
        return biexponential_tail_shape(timing, rng)
    if family == 'OOD_SHOULDER_DOUBLE_PEAK':
        return shoulder_double_peak_shape(timing, rng)
    raise ValueError(family)
def add_signal_to_noise(H0, R0, shape, amplitude, support, hbr_ratio, hbr_channel_scale):
    H = np.asarray(H0, float).copy()
    R = np.asarray(R0, float).copy()
    post_shape = np.zeros(len(FULL_T), float)
    post_shape[POST_MASK] = np.asarray(shape, float)
    H += (amplitude * support)[:, None] * post_shape[None, :]
    R -= (amplitude * float(hbr_ratio) * hbr_channel_scale)[:, None] * post_shape[None, :]
    return (H, R)
def trial_to_predictors(H, R, present, onset, peak, offset):
    ev = evidence(H, R)
    return {
        'present': int(present),
        'onset': float(onset) if present else np.nan,
        'peak': float(peak) if present else np.nan,
        'offset': float(offset) if present else np.nan,
        'censored': int(bool(present) and float(offset) > GRID_END - 0.25),
        'obs': whole_vector(ev),
        'state_feat': state_features(ev)
    }
def generate_stress_banks(profile: NoiseProfile, seed: int, pid: str):
    family_trials = {family: [] for family in GENERATED_FAMILIES}
    parameter_rows = []
    for trial_index in range(N_TEST_POS):
        noise_rng = np.random.default_rng(stable_seed('OOD_SHAPE_STRESS', seed, pid, 'POSITIVE_NOISE', trial_index))
        H0, R0 = base_noise_trial(profile, noise_rng)
        amp_rng = np.random.default_rng(stable_seed('OOD_SHAPE_STRESS', seed, pid, 'POSITIVE_AMPLITUDE', trial_index))
        support = amp_rng.uniform(0.55, 1.0, size=profile.n_channels)
        amplitude = amp_rng.uniform(1.5, 4.5) * profile.hbo_scale
        hbr_ratio = float(amp_rng.uniform(0.35, 0.7))
        hbr_channel_scale = amp_rng.uniform(0.7, 1.1, size=profile.n_channels)
        matched_rng = np.random.default_rng(stable_seed('OOD_SHAPE_STRESS', seed, pid, 'MATCHED_HALF_COSINE_NEW', trial_index))
        shape, onset, peak, offset, params = original_half_cosine_shape(matched_rng)
        H, R = add_signal_to_noise(H0, R0, shape, amplitude, support, hbr_ratio, hbr_channel_scale)
        family_trials['MATCHED_HALF_COSINE_NEW'].append(trial_to_predictors(H, R, True, onset, peak, offset))
        parameter_rows.append({
            'repeat_seed': int(seed),
            'participant': pid,
            'positive_trial_index': int(trial_index),
            'family': 'MATCHED_HALF_COSINE_NEW',
            'truth_onset': float(onset),
            'truth_peak': float(peak),
            'truth_offset': float(offset),
            'truth_censored': int(offset > GRID_END - 0.25),
            **params
        })
        common_rng = np.random.default_rng(stable_seed('OOD_SHAPE_STRESS', seed, pid, 'OOD_COMMON_TIMING', trial_index))
        timing = common_ood_timing(common_rng)
        for family in GENERATED_FAMILIES[1:]:
            shape_rng = np.random.default_rng(stable_seed('OOD_SHAPE_STRESS', seed, pid, family, trial_index))
            shape, onset, peak, offset, params = make_family_shape(family, timing, shape_rng)
            H, R = add_signal_to_noise(H0, R0, shape, amplitude, support, hbr_ratio, hbr_channel_scale)
            family_trials[family].append(trial_to_predictors(H, R, True, onset, peak, offset))
            parameter_rows.append({
                'repeat_seed': int(seed),
                'participant': pid,
                'positive_trial_index': int(trial_index),
                'family': family,
                'truth_onset': float(onset),
                'truth_peak': float(peak),
                'truth_offset': float(offset),
                'truth_censored': int(offset > GRID_END - 0.25),
                'common_peak_target': float(timing['peak_target']),
                'common_long_recovery': bool(timing['long_recovery']),
                **params
            })
    for negative_index in range(N_TEST_NEG):
        noise_rng = np.random.default_rng(stable_seed('OOD_SHAPE_STRESS', seed, pid, 'NEGATIVE_NOISE', negative_index))
        H0, R0 = base_noise_trial(profile, noise_rng)
        for family in GENERATED_FAMILIES:
            family_trials[family].append(trial_to_predictors(H0, R0, False, np.nan, np.nan, np.nan))
    perm = np.random.default_rng(stable_seed('OOD_SHAPE_STRESS', seed, pid, 'COMMON_PERMUTATION')).permutation(N_TEST_POS + N_TEST_NEG)
    banks = {}
    for family, rows in family_trials.items():
        if len(rows) != N_TEST_POS + N_TEST_NEG:
            raise RuntimeError(f'{family}: incorrect stress-bank row count.')
        rows = [rows[int(i)] for i in perm]
        banks[family] = {
            'present': np.asarray([r['present'] for r in rows], int),
            'onset': np.asarray([r['onset'] for r in rows], float),
            'peak': np.asarray([r['peak'] for r in rows], float),
            'offset': np.asarray([r['offset'] for r in rows], float),
            'censored': np.asarray([r['censored'] for r in rows], int),
            'obs': finite_model_array(np.asarray([r['obs'] for r in rows], np.float32)),
            'state_feat': finite_model_array(np.asarray([r['state_feat'] for r in rows], np.float32))
        }
    return (banks, pd.DataFrame(parameter_rows))
def balanced_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, int)
    p = np.asarray(p, int)
    pos = y == 1
    neg = y == 0
    sens = float(np.mean(p[pos] == 1)) if np.any(pos) else np.nan
    spec = float(np.mean(p[neg] == 0)) if np.any(neg) else np.nan
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
    for name, tt, pp, exact in ((
        'onset',
        ton,
        pon,
        present & np.isfinite(ton)
    ), ('peak', tpk, ppk, present & np.isfinite(tpk)), ('offset', toff, poff, present & ~censored & np.isfinite(toff))):
        localized = exact & (decision == 1) & np.isfinite(pp)
        out[f'conditional_{name}_mae_sec'] = float(np.mean(np.abs(pp[localized] - tt[localized]))) if np.any(localized) else np.nan
        out[f'conditional_{name}_localized_rate'] = float(np.mean((decision[exact] == 1) & np.isfinite(pp[exact]))) if np.any(exact) else np.nan
        idx = np.flatnonzero(exact)
        err = np.full(len(idx), MISS_PENALTY_SEC, float)
        if len(idx):
            good = (decision[idx] == 1) & np.isfinite(pp[idx])
            err[good] = np.abs(pp[idx][good] - tt[idx][good])
        out[f'{name}_mae_sec'] = float(np.mean(err)) if len(err) else np.nan
    true_duration = toff - ton
    pred_duration = poff - pon
    exact_duration = present & ~censored & np.isfinite(true_duration)
    localized = exact_duration & (decision == 1) & np.isfinite(pred_duration)
    out['conditional_duration_mae_sec'] = float(np.mean(np.abs(pred_duration[localized] - true_duration[localized]))) if np.any(localized) else np.nan
    out['conditional_duration_localized_rate'] = float(np.mean((decision[exact_duration] == 1) & np.isfinite(pred_duration[exact_duration]))) if np.any(exact_duration) else np.nan
    idx = np.flatnonzero(exact_duration)
    err = np.full(len(idx), MISS_PENALTY_SEC, float)
    if len(idx):
        good = (decision[idx] == 1) & np.isfinite(pred_duration[idx])
        err[good] = np.abs(pred_duration[idx][good] - true_duration[idx][good])
    out['duration_mae_sec'] = float(np.mean(err)) if len(err) else np.nan
    vals = np.asarray([out['onset_mae_sec'], out['peak_mae_sec'], out['offset_mae_sec'], out['duration_mae_sec']], float)
    scales = np.asarray([1.0, 1.0, 1.5, 1.5], float)
    out['selection_loss'] = float(np.nanmean(vals / scales))
    out['predicted_response_rate'] = float(np.mean(decision == 1))
    out['truth_censored_fraction_positive'] = float(np.mean(censored[present])) if np.any(present) else np.nan
    out['n_positive'] = int(np.sum(present))
    out['n_negative'] = int(np.sum(~present))
    out['n_uncensored_positive'] = int(np.sum(present & ~censored))
    return out
def predict_methods(model: CoherentFullModel, truth):
    x = predictor_view(truth)
    raw, rp, P = model.abc.raw(x)
    refined = model.refined_raw(x, raw, rp, P)
    full = make_prediction_threshold(n4, refined, rp, model.abc.response_threshold)
    abc = make_prediction_threshold(n4, raw, rp, model.abc.response_threshold)
    return ({'FULL': full, 'ABC_PARENT': abc}, np.asarray(rp, float))
def per_trial_frame(seed, pid, family, method, truth, pred, response_prob, abc_cfg, local_cfg):
    return pd.DataFrame({
        'repeat_seed': int(seed),
        'participant': pid,
        'family': family,
        'method': method,
        'trial_index': np.arange(len(pred), dtype=int),
        'truth_present': np.asarray(truth['present'], int),
        'truth_onset': np.asarray(truth['onset'], float),
        'truth_peak': np.asarray(truth['peak'], float),
        'truth_offset': np.asarray(truth['offset'], float),
        'truth_censored': np.asarray(truth['censored'], int),
        'pred_response': pred['pred_response'].to_numpy(int),
        'response_probability': np.asarray(response_prob, float),
        'pred_onset': pred['pred_onset'].to_numpy(float),
        'pred_peak': pred['pred_peak'].to_numpy(float),
        'pred_offset': pred['pred_offset'].to_numpy(float),
        'selected_abc_cfg': canonical_cfg(abc_cfg),
        'selected_lbr_cfg': canonical_cfg(local_cfg)
    })
def protected_self_tests():
    assert len(ALL_PIDS) == 26
    assert len(GRID) == 41
    assert set(SEEDS).issubset(set(range(1, 11)))
    assert len(set(SEEDS)) == len(SEEDS)
    assert len(GENERATED_FAMILIES) == 5
    timing = {'onset': 1.0, 'peak_delay': 2.0, 'peak_target': 3.0, 'recovery': 6.0, 'offset': 9.0, 'long_recovery': False}
    for family in GENERATED_FAMILIES[1:]:
        rng = np.random.default_rng(12345)
        y, onset, peak, offset, _ = make_family_shape(family, timing, rng)
        assert y.shape == GRID.shape
        assert np.isfinite(y).all()
        assert onset < peak < offset
        assert float(np.max(y)) > 0.2
    y, onset, peak, offset, _ = original_half_cosine_shape(np.random.default_rng(7))
    assert onset < peak < offset
    assert np.isfinite(y).all()
    truth = {
        'present': np.asarray([1, 1, 0, 0]),
        'onset': np.asarray([1.0, 2.0, np.nan, np.nan]),
        'peak': np.asarray([3.0, 4.0, np.nan, np.nan]),
        'offset': np.asarray([8.0, 9.0, np.nan, np.nan]),
        'censored': np.asarray([0, 0, 0, 0])
    }
    pred = pd.DataFrame({
        'pred_response': [1, 0, 0, 0],
        'pred_onset': [1.1, np.nan, np.nan, np.nan],
        'pred_peak': [3.2, np.nan, np.nan, np.nan],
        'pred_offset': [8.3, np.nan, np.nan, np.nan]
    })
    mm = participant_metrics(truth, pred)
    assert np.isclose(mm['response_balanced_accuracy'], 0.75)
    assert mm['onset_mae_sec'] > mm['conditional_onset_mae_sec']
    print('Standalone OOD HRF-shape stress protected self-tests: PASS')
def _run_morphology_seeds(all_audit_parts, all_metric_parts, all_param_parts, all_trial_parts):
    for seed in SEEDS:
        configure_seed(seed)
        print('\n' + '=' * 180)
        print(f'SEED {seed:02d}')
        print('=' * 180)
        caches, cache_dir = load_seed_caches(seed)
        print('Original source/test cache    :', cache_dir)
        print('Selected configurations:', seed_selected_table(seed))
        for index, pid in enumerate(ALL_PIDS, 1):
            cp_metrics = OUT_DIR / 'checkpoints' / f'seed_{seed:02d}_{pid}_METRICS.csv'
            cp_trials = OUT_DIR / 'checkpoints' / f'seed_{seed:02d}_{pid}_TRIALS.csv.gz'
            cp_params = OUT_DIR / 'checkpoints' / f'seed_{seed:02d}_{pid}_SHAPE_PARAMETERS.csv.gz'
            cp_audit = OUT_DIR / 'checkpoints' / f'seed_{seed:02d}_{pid}_AUDIT.csv.gz'
            if cp_metrics.is_file() and cp_trials.is_file() and cp_params.is_file() and cp_audit.is_file():
                print(f'[seed {seed:02d}] [{index:02d}/26] {pid}: checkpoint')
                all_metric_parts.append(pd.read_csv(cp_metrics))
                all_trial_parts.append(pd.read_csv(cp_trials))
                all_param_parts.append(pd.read_csv(cp_params))
                all_audit_parts.append(pd.read_csv(cp_audit))
                continue
            abc_cfg, local_cfg, cfg_table = load_full_config(seed, pid)
            source_pids = [p for p in ALL_PIDS if p != pid]
            train = training_rows(caches, source_pids)
            audit = FitAudit(source_pids, [pid], f'OOD_HRF_STRESS/seed={seed:02d}/outer={pid}')
            print(f'[seed {seed:02d}] [{index:02d}/26] {pid}: evaluate response families', flush=True)
            model = fit_coherent_full(n4, train, abc_cfg, local_cfg, audit, outer_forbidden={pid}, context=f'OOD_HRF_STRESS/seed={seed:02d}/outer={pid}')
            profile = estimate_noise_profile(preprocessed_path(pid))
            generated_banks, parameter_df = generate_stress_banks(profile, seed, pid)
            family_truth = {'PAPER_MATCHED_CACHE': caches[pid]['test'], **generated_banks}
            metric_rows = []
            trial_rows = []
            for family in EVAL_FAMILIES:
                truth = family_truth[family]
                predictions, response_prob = predict_methods(model, truth)
                if family == 'PAPER_MATCHED_CACHE':
                    native_full = model.predict(predictor_view(truth))
                    for col in ('pred_response', 'pred_onset', 'pred_peak', 'pred_offset'):
                        aa = native_full[col].to_numpy()
                        bb = predictions['FULL'][col].to_numpy()
                        if col == 'pred_response':
                            ok = np.array_equal(aa, bb)
                        else:
                            ok = np.allclose(aa, bb, atol=1e-10, rtol=0.0, equal_nan=True)
                        if not ok:
                            raise RuntimeError(f'{pid}: FULL parity failed for {col}.')
                for method in METHODS:
                    pred = predictions[method]
                    mm = participant_metrics(truth, pred)
                    metric_rows.append({
                        'repeat_seed': int(seed),
                        'participant': pid,
                        'family': family,
                        'generator_class': 'MATCHED_REFERENCE' if family in {'PAPER_MATCHED_CACHE', 'MATCHED_HALF_COSINE_NEW'} else 'OUT_OF_GENERATOR',
                        'method': method,
                        **mm,
                        'selected_abc_cfg': canonical_cfg(abc_cfg),
                        'selected_lbr_cfg': canonical_cfg(local_cfg),
                        'target_noise_phi_hbo': float(profile.phi_hbo),
                        'target_noise_phi_hbr': float(profile.phi_hbr),
                        'target_noise_channels': int(profile.n_channels)
                    })
                    trial_rows.append(per_trial_frame(seed, pid, family, method, truth, pred, response_prob, abc_cfg, local_cfg))
            metric_df = pd.DataFrame(metric_rows)
            trial_df = pd.concat(trial_rows, ignore_index=True)
            parameter_df['selected_abc_cfg'] = canonical_cfg(abc_cfg)
            parameter_df['selected_lbr_cfg'] = canonical_cfg(local_cfg)
            audit_df = pd.DataFrame(audit.rows)
            audit_df.insert(0, 'repeat_seed', int(seed))
            audit_df.insert(1, 'outer_participant', pid)
            audit_df['selected_config_source'] = str(cfg_table)
            audit_df['original_cache_source'] = str(cache_dir)
            audit_df['stress_target_noise_source'] = str(preprocessed_path(pid))
            audit_df['stress_families_used_for_fitting'] = False
            audit_df['target_participant_used_for_fitting'] = False
            atomic_csv(metric_df, cp_metrics)
            atomic_csv(trial_df, cp_trials)
            atomic_csv(parameter_df, cp_params)
            atomic_csv(audit_df, cp_audit)
            all_metric_parts.append(metric_df)
            all_trial_parts.append(trial_df)
            all_param_parts.append(parameter_df)
            all_audit_parts.append(audit_df)
            del (model, train, generated_banks, family_truth, metric_rows, trial_rows, metric_df, trial_df, parameter_df, audit_df)
            gc.collect()
        del caches
        gc.collect()
def main():
    mount_drive()
    protected_self_tests()
    if not SHIN_ROOT.is_dir():
        raise FileNotFoundError(f'Expected SHIN project folder does not exist:\n{SHIN_ROOT}')
    for sub in ('checkpoints', 'results', 'audit'):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    print('=' * 180)
    print('HEMOBOUNDARY — STANDALONE OUT-OF-GENERATOR HRF SHAPE STRESS TEST')
    print('=' * 180)
    print('Assigned seeds                 :', list(SEEDS))
    print('Participants per seed          : 26')
    print('Training banks                 : original source TRAIN only')
    print('Nested hyperparameter search   : NO')
    print('Stress-family tuning           : NO')
    print('Target adaptation              : NO')
    print('External core .py required     : NO')
    print('Source-code Drive search       : NO')
    print('Generated families             :', list(GENERATED_FAMILIES))
    print('Methods                        :', list(METHODS))
    print('Output                         :', OUT_DIR)
    print()
    all_metric_parts = []
    all_trial_parts = []
    all_param_parts = []
    all_audit_parts = []
    _run_morphology_seeds(all_audit_parts, all_metric_parts, all_param_parts, all_trial_parts)
    metrics = pd.concat(all_metric_parts, ignore_index=True)
    trials = pd.concat(all_trial_parts, ignore_index=True)
    params = pd.concat(all_param_parts, ignore_index=True)
    audits = pd.concat(all_audit_parts, ignore_index=True)
    expected_metric_rows = len(SEEDS) * 26 * len(EVAL_FAMILIES) * len(METHODS)
    if len(metrics) != expected_metric_rows:
        raise RuntimeError(f'Expected {expected_metric_rows} metric rows; found {len(metrics)}.')
    atomic_csv(metrics, OUT_DIR / 'results' / 'PARTICIPANT_METRICS.csv')
    atomic_csv(trials, OUT_DIR / 'results' / 'PER_TRIAL_PREDICTIONS.csv.gz')
    atomic_csv(params, OUT_DIR / 'audit' / 'STRESS_SHAPE_PARAMETERS.csv.gz')
    atomic_csv(audits, OUT_DIR / 'audit' / 'FIT_MEMBERSHIP_AUDIT.csv.gz')
    summary = metrics.groupby(['family', 'generator_class', 'method'], as_index=False).agg(
        rows=('participant', 'size'),
        participants=('participant', 'nunique'),
        seeds=('repeat_seed', 'nunique'),
        response_BA=('response_balanced_accuracy', 'mean'),
        onset_MAE=('onset_mae_sec', 'mean'),
        peak_MAE=('peak_mae_sec', 'mean'),
        offset_MAE=('offset_mae_sec', 'mean'),
        duration_MAE=('duration_mae_sec', 'mean'),
        selection_loss=('selection_loss', 'mean'),
        truth_censored_fraction=('truth_censored_fraction_positive', 'mean')
    ).sort_values(['method', 'family']).reset_index(drop=True)
    atomic_csv(summary, OUT_DIR / 'results' / 'SHARD_SUMMARY.csv')
    atomic_json({
        'analysis': 'Frozen out-of-generator HRF shape stress test',
        'job_dir': JOB_DIR,
        'seeds': list(SEEDS),
        'participants_per_seed': 26,
        'training_bank_family': 'Source-participant training banks only',
        'evaluation_families': list(EVAL_FAMILIES),
        'methods': list(METHODS),
        'nested_hyperparameter_search': False,
        'stress_family_tuning': False,
        'target_adaptation': False,
        'external_core_file_required': False,
        'source_code_search': False,
        'target_noise_profile_role': 'Noise realism for held-out stress bank only; never enters source fitting',
        'stress_positive_noise_matched_across_generated_families': True,
        'stress_negative_trials_identical_across_generated_families': True,
        'statistical_unit_for_final_merge': 'participant after averaging seeds within participant'
    }, OUT_DIR / 'audit' / 'RUN_MANIFEST.json')
    print('\n' + '=' * 180)
    print('JOB COMPLETE')
    print('=' * 180)
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print()
    print('Saved:', OUT_DIR)
if __name__ == '__main__':
    main()
