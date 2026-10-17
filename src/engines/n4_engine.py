"""State and temporal feature models."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time
import re
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import scipy
from scipy import signal
from scipy.special import logsumexp
from scipy.optimize import minimize
import sklearn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
VERSION = 'MOD13_FOUR_MECHANISM_TESTS_V1_20260912'
OUT_DIR = '89_MOD13_four_mechanism_tests_v1'
ALL_PIDS = tuple((f'VP{i:03d}' for i in range(1, 27)))
GRID_DT = 0.5
GRID_END = 20.0
GRID = np.arange(0.0, GRID_END + GRID_DT / 2, GRID_DT)
CENSOR_BOUND = 19.75
MIN_GAP = 0.5
MISS_PENALTY_SEC = 4.0
CHANGE_WINDOWS = (3, 5, 9)
STATE_C = 0.01
CHANGE_FRAC = 0.3
INNER_FOLDS = 3
ZOOM_OOF_FOLDS = 3
RANDOM_SEED = 20260912
METRICS = ('response_balanced_accuracy', 'onset_mae_sec', 'peak_mae_sec', 'offset_mae_sec', 'duration_mae_sec', 'selection_loss')
TIMING = METRICS[1:5]
LABEL_KEYS = ('present', 'onset', 'peak', 'offset', 'censored', 'states')
PREDICTOR_KEYS = ('obs', 'state_feat')
EXPECTED_SPLIT_ROWS = {'train': 45, 'test': 36}
BASELINE_NAME = 'MOD13_HSCR_REFINED180_SAVED'
NAMES = {'N1': 'Censored duration-state smoothing', 'N2': 'Joint timing loss', 'N3': 'Clock-shift equivariance', 'N4': 'Local boundary zoom'}
GRIDS = {
    'N1': [dict(duration_mix=m, temperature=t, alpha=a) for m, t, a in itertools.product((0.5, 1.0), (1.0, 1.5), (10.0, 30.0))],
    'N2': [dict(alpha=a, duration_weight=d, censor_weight=c) for a, d, c in itertools.product((0.1, 1.0, 10.0), (0.5, 1.0), (0.0, 0.5))],
    'N3': [dict(shift_seconds=s, augmentation_weight=w, alpha=a) for s, w, a in itertools.product((0.5, 1.0), (0.5, 1.0), (10.0, 30.0))],
    'N4': [dict(radius=r, alpha=a, strength=s) for r, a, s in itertools.product((1.0, 2.0), (10.0, 100.0), (0.5, 1.0))]
}

def stable_seed(*parts: Any) -> int:
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16) % (2 ** 31 - 1)
def finite_model_array(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
def predictor_view(data: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: data[k] for k in PREDICTOR_KEYS}
def require_predictors_only(x: Mapping[str, np.ndarray]) -> None:
    if set(x) != set(PREDICTOR_KEYS):
        raise RuntimeError(f'Prediction payload must contain ONLY {PREDICTOR_KEYS}; got {sorted(x)}')
def grouped_folds(pids: Sequence[str], seed: int, n: int=3) -> List[List[str]]:
    p = np.array(sorted(set(pids)), dtype=object)
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
        self.rows.append(dict(context=self.context, stage=stage, fit_participants=sorted(actual), forbidden=sorted(blocked), n_rows=int(len(data['pid']))))
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
    """Rich D12 evidence rather than final D12 predictions."""
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
            x = A[:, col]
            flat.extend([float(np.median(x)), float(np.quantile(x, 0.75) - np.quantile(x, 0.25)), float(np.max(x) - np.min(x))])
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
def balanced_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, int)
    p = np.asarray(p, int)
    pos, neg = (y == 1, y == 0)
    sens = np.mean(p[pos] == 1) if np.any(pos) else np.nan
    spec = np.mean(p[neg] == 0) if np.any(neg) else np.nan
    return float(np.nanmean([sens, spec]))
def trial_metric_summary(truth: Mapping[str, np.ndarray], pred: pd.DataFrame) -> Dict[str, float]:
    present = np.asarray(truth['present'], int) == 1
    cens = np.asarray(truth['censored'], int) == 1
    decision = pred['pred_response'].to_numpy(int)
    out: Dict[str, float] = {'response_balanced_accuracy': balanced_accuracy(np.asarray(truth['present'], int), decision)}
    metrics = {}
    for b, tcol, pcol in [('onset', 'onset', 'pred_onset'), ('peak', 'peak', 'pred_peak'), ('offset', 'offset', 'pred_offset')]:
        t = np.asarray(truth[tcol], float)
        p = pred[pcol].to_numpy(float)
        valid_truth = present & np.isfinite(t)
        if b == 'offset':
            valid_truth &= ~cens
        err = np.full(np.sum(valid_truth), MISS_PENALTY_SEC, float)
        idx = np.flatnonzero(valid_truth)
        finite = np.isfinite(p[idx]) & (decision[idx] == 1)
        err[finite] = np.abs(p[idx][finite] - t[idx][finite])
        metrics[f'{b}_mae_sec'] = float(np.mean(err)) if len(err) else np.nan
        metrics[f'{b}_localized_rate'] = float(np.mean(finite)) if len(err) else np.nan
    ton = np.asarray(truth['onset'], float)
    toff = np.asarray(truth['offset'], float)
    pon = pred['pred_onset'].to_numpy(float)
    poff = pred['pred_offset'].to_numpy(float)
    valid = present & ~cens & np.isfinite(ton) & np.isfinite(toff)
    idx = np.flatnonzero(valid)
    err = np.full(len(idx), MISS_PENALTY_SEC, float)
    finite = np.isfinite(pon[idx]) & np.isfinite(poff[idx]) & (decision[idx] == 1)
    err[finite] = np.abs(poff[idx][finite] - pon[idx][finite] - (toff[idx][finite] - ton[idx][finite]))
    metrics['duration_mae_sec'] = float(np.mean(err)) if len(err) else np.nan
    out.update(metrics)
    scales = np.array([1.0, 1.0, 1.5, 1.5])
    vals = np.array([out['onset_mae_sec'], out['peak_mae_sec'], out['offset_mae_sec'], out['duration_mae_sec']])
    out['selection_loss'] = float(np.nanmean(vals / scales))
    return out
class StableScaler:
    """Source-only, two-pass weighted standardization with safe constant columns.
    This avoids negative roundoff variances from weighted online variance updates.
    """

    def fit(self, X, sample_weight=None):
        X = np.asarray(X, float)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise RuntimeError('Scaler needs finite 2D training features.')
        w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, float)
        if len(w) != len(X) or not np.isfinite(w).all() or (w < 0).any() or (w.sum() <= 0):
            raise RuntimeError('Invalid source sample weights.')
        self.mean_ = np.average(X, axis=0, weights=w)
        self.var_ = np.average((X - self.mean_) ** 2, axis=0, weights=w)
        self.var_ = np.maximum(self.var_, 0.0)
        eps = np.finfo(float).eps
        n = float(w.sum())
        constant = self.var_ <= n * eps * self.var_ + (n * self.mean_ * eps) ** 2
        self.scale_ = np.sqrt(self.var_)
        self.scale_[constant | (self.scale_ == 0)] = 1.0
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or (not np.isfinite(X).all()):
            raise RuntimeError('Scaler feature schema/finite check failed.')
        return (X - self.mean_) / self.scale_
@dataclass
class StateModel:
    scaler: Any
    clf: Any
    response_scaler: Any
    response_clf: Any
    fit_pids: Tuple[str, ...]
def fit_state(data, audit, stage='state_encoder'):
    audit.record(data, stage)
    F = finite_model_array(data['state_feat'])
    obs = finite_model_array(data['obs'])
    X = F.reshape(-1, F.shape[-1])
    y = np.asarray(data['states'], int).reshape(-1)
    weights = np.asarray(data['weight'], float)
    fw = np.repeat(weights, F.shape[1])
    fw = fw / np.mean(fw)
    if len(X) > 60000:
        ix = np.random.default_rng(stable_seed('state_subsample', *sorted(set(data['pid'])))).choice(len(X), 60000, replace=False)
        X, y, fw = (X[ix], y[ix], fw[ix])
    sc = StableScaler().fit(X, sample_weight=fw)
    clf = LogisticRegression(C=STATE_C, class_weight='balanced', max_iter=800, random_state=RANDOM_SEED)
    clf.fit(sc.transform(X), y, sample_weight=fw)
    rs = StableScaler().fit(obs, sample_weight=weights)
    rc = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED)
    rc.fit(rs.transform(obs), np.asarray(data['present'], int), sample_weight=weights / weights.mean())
    return StateModel(sc, clf, rs, rc, tuple(sorted(set(data['pid']))))
def state_probabilities(model, x):
    require_predictors_only(x)
    F = finite_model_array(x['state_feat'])
    n, t, d = F.shape
    p0 = model.clf.predict_proba(model.scaler.transform(F.reshape(-1, d)))
    p = np.full((n * t, 5), 1e-08)
    for j, k in enumerate(model.clf.classes_):
        p[:, int(k)] = p0[:, j]
    p /= p.sum(axis=1, keepdims=True)
    r = model.response_clf.predict_proba(model.response_scaler.transform(finite_model_array(x['obs'])))[:, 1]
    return (p.reshape(n, t, 5), r)
def fixed_order_path(p, stay_bias=0.0):
    """Original complete five-state Viterbi, vectorized. Uniform stay bias is invariant.
    Used only as the inherited representation in N2/N3/N4 and by parity tests.
    N1 replaces it with actual age-dependent hazards and a free terminal state.
    """
    logp = np.log(np.clip(p, 1e-09, 1.0))
    n, t, s = logp.shape
    dp = np.full((n, s), -np.inf)
    dp[:, 0] = logp[:, 0, 0]
    back = np.zeros((n, t, s), np.int8)
    for k in range(1, t):
        nxt = np.full_like(dp, -np.inf)
        for state in range(s):
            vs = dp[:, state] + stay_bias
            va = dp[:, state - 1] if state > 0 else np.full(n, -np.inf)
            keep = vs >= va
            nxt[:, state] = np.where(keep, vs, va) + logp[:, k, state]
            back[:, k, state] = np.where(keep, state, state - 1)
        dp = nxt
    states = np.full(n, s - 1, dtype=int)
    path = np.zeros((n, t), np.int8)
    ii = np.arange(n)
    for k in range(t - 1, -1, -1):
        path[:, k] = states
        if k > 0:
            states = back[ii, k, states]
    return path
def hard_state_features(p, r):
    n, t, s = p.shape
    path = fixed_order_path(p)
    occ = p.mean(axis=1)
    mx = p.max(axis=1)
    centres = (p * GRID[None, :, None]).sum(axis=1) / np.maximum(p.sum(axis=1), 1e-12)
    summaries = np.stack((occ, mx, centres), axis=2).reshape(n, 15)
    dwell = np.column_stack([(path == state).mean(axis=1) for state in range(5)])
    oi = np.argmax(path == 1, axis=1)
    pi = np.argmax(np.where(path == 2, p[:, :, 2], -np.inf), axis=1)
    fi = np.argmax(path == 4, axis=1)
    ii = np.arange(n)
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
def fit_hazards(data, duration_mix, audit):
    audit.record(data, 'duration_hazards')
    S = 5
    T = len(GRID)
    exposed = np.zeros((S, T))
    ended = np.zeros((S, T))
    positive = np.asarray(data['present'], int) == 1
    for seq, w in zip(np.asarray(data['states'], int)[positive], np.asarray(data['weight'])[positive]):
        for s in range(4):
            ii = np.flatnonzero(seq == s)
            if not len(ii):
                continue
            if np.any(np.diff(ii) != 1):
                raise RuntimeError('State label sequence is not contiguous.')
            d = len(ii)
            exposed[s, :d] += w
            if ii[-1] < T - 1 and seq[ii[-1] + 1] == s + 1:
                ended[s, d - 1] += w
    hazard = np.zeros((5, T), float)
    for s in range(4):
        pooled = (ended[s].sum() + 0.5) / (exposed[s].sum() + 5.0)
        local = (ended[s] + 5.0 * pooled) / (exposed[s] + 5.0)
        hazard[s] = np.clip(duration_mix * local + (1 - duration_mix) * pooled, 0.0001, 0.95)
    hazard[4] = 0.0
    return hazard
def duration_smooth(p, hazards, temperature=1.0, batch_size=48):
    """Forward/backward inference in a (state, elapsed-age) expanded chain.
    The last observed state is unrestricted: unfinished recovery is allowed.
    Returns state posteriors and entry-transition probabilities, not hard labels.
    """
    n, T, S = p.shape
    A = T
    lh = np.full_like(hazards, -np.inf)
    positive = hazards > 0
    lh[positive] = np.log(hazards[positive])
    ls = np.log1p(-hazards)
    allpost = []
    allentry = []
    for begin in range(0, n, batch_size):
        pp = p[begin:begin + batch_size]
        N = len(pp)
        e = np.log(np.clip(pp, 1e-09, 1)) / float(temperature)
        f = np.full((N, T, S, A), -np.inf)
        f[:, 0, 0, 0] = e[:, 0, 0]
        f[:, 0] -= logsumexp(f[:, 0], axis=(1, 2))[:, None, None]
        for t in range(1, T):
            z = np.full((N, S, A), -np.inf)
            z[:, :, 1:] = f[:, t - 1, :, :-1] + ls[None, :, :-1]
            z[:, 1:, 0] = logsumexp(f[:, t - 1, :-1, :] + lh[None, :-1, :], axis=2)
            z += e[:, t, :, None]
            z -= logsumexp(z, axis=(1, 2))[:, None, None]
            f[:, t] = z
        post = np.zeros((N, T, S))
        entries = np.zeros((N, T, S - 1))
        beta = np.zeros((N, S, A))
        for t in range(T - 1, -1, -1):
            z = f[:, t] + beta
            z -= logsumexp(z, axis=(1, 2))[:, None, None]
            post[:, t] = np.exp(z).sum(axis=2)
            if t == 0:
                break
            bprev = np.full_like(beta, -np.inf)
            bprev[:, :, :-1] = ls[None, :, :-1] + e[:, t, :, None] + beta[:, :, 1:]
            adv = lh[None, :-1, :] + e[:, t, 1:, None] + beta[:, 1:, 0, None]
            bprev[:, :-1, :] = np.logaddexp(bprev[:, :-1, :], adv)
            denom = logsumexp(f[:, t - 1] + bprev, axis=(1, 2))
            num = logsumexp(f[:, t - 1, :-1, :] + adv, axis=2)
            entries[:, t] = np.exp(num - denom[:, None])
            norm = logsumexp(bprev, axis=(1, 2))
            beta = bprev - norm[:, None, None]
        allpost.append(post)
        allentry.append(entries)
    return (np.concatenate(allpost), np.concatenate(allentry))
def soft_state_features(post, entries, r):
    occ = post.mean(axis=1)
    mx = post.max(axis=1)
    centres = (post * GRID[None, :, None]).sum(axis=1) / np.maximum(post.sum(axis=1), 1e-12)
    summaries = np.stack((occ, mx, centres), axis=2).reshape(len(post), 15)
    timings = []
    confs = []
    masses = []
    widths = []
    for j in (0, 1, 3):
        v = entries[:, :, j]
        mass = v.sum(axis=1)
        t = (v * GRID[None, :]).sum(axis=1) / np.maximum(mass, 1e-12)
        t = np.where(mass > 1e-08, t, GRID_END)
        sd = np.sqrt(np.maximum((v * (GRID[None, :] - t[:, None]) ** 2).sum(axis=1) / np.maximum(mass, 1e-12), 0))
        timings.append(t)
        confs.append(v.max(axis=1))
        masses.append(mass)
        widths.append(sd)
    entropy = -(post * np.log(np.clip(post, 1e-12, 1))).sum(axis=2)
    return np.column_stack((
        summaries,
        occ,
        *timings,
        *confs,
        entropy.mean(axis=1),
        entropy.max(axis=1),
        r,
        (post[:, :, 1] + post[:, :, 2]).mean(axis=1),
        (post[:, :, 3] + post[:, :, 4]).mean(axis=1),
        *masses,
        *widths,
        post[:, -1, 4]
    ))
@dataclass
class Representation:
    state: StateModel
    mode: str = 'base'
    hazards: Any = None
    temperature: float = 1.0

    def transform(self, x):
        require_predictors_only(x)
        p, r = state_probabilities(self.state, x)
        if self.mode == 'duration':
            pp, en = duration_smooth(p, self.hazards, self.temperature)
            sx = soft_state_features(pp, en, r)
        else:
            sx = hard_state_features(p, r)
        cx = change_branch_features(x, CHANGE_FRAC)
        X = np.column_stack((sx, cx))
        if not np.isfinite(X).all():
            raise RuntimeError('Non-finite derived representation.')
        return (X, p)
@dataclass
class Heads:
    scaler: Any
    response: Any
    regressors: Any
    coefficients: Any = None

    def raw(self, X):
        Z = self.scaler.transform(X)
        prob = self.response.predict_proba(Z)[:, 1]
        if self.coefficients is None:
            t = np.column_stack([r.predict(Z) for r in self.regressors])
        else:
            t = np.column_stack((Z, np.ones(len(Z)))) @ self.coefficients
        if not np.isfinite(t).all():
            raise RuntimeError('Non-finite timing-head prediction.')
        return (t, prob)
def direct_heads(X, data, alpha, audit):
    audit.record(data, 'boundary_and_response_heads')
    w = np.asarray(data['weight'], float)
    w = w / w.mean()
    sc = StableScaler().fit(X, sample_weight=w)
    Z = sc.transform(X)
    resp = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED)
    resp.fit(Z, np.asarray(data['present'], int), sample_weight=w)
    regs = []
    for b in ('onset', 'peak', 'offset'):
        truth = np.asarray(data[b], float)
        mask = (np.asarray(data['present']) == 1) & np.isfinite(truth)
        if b == 'offset':
            mask &= np.asarray(data['censored']) == 0
        if mask.sum() < 15:
            raise RuntimeError(f'Too few observed positive labels for {b}.')
        rw = w[mask] / w[mask].mean()
        regs.append(Ridge(alpha=float(alpha)).fit(Z[mask], truth[mask], sample_weight=rw))
    return Heads(sc, resp, regs)
def make_prediction(raw, prob):
    raw = np.asarray(raw, float)
    prob = np.asarray(prob, float)
    on, pk, off = enforce_geometry(raw[:, 0], raw[:, 1], raw[:, 2])
    miss = prob < 0.5
    on[miss] = np.nan
    pk[miss] = np.nan
    off[miss] = np.nan
    return pd.DataFrame(dict(pred_response=(~miss).astype(int), pred_onset=on, pred_peak=pk, pred_offset=off))
def joint_loss_and_grad(flat, Z, truth, present, censored, weight, alpha, duration_weight, censor_weight):
    d = Z.shape[1]
    W = flat.reshape(d, 3)
    pr = Z @ W
    G = np.zeros_like(pr)
    loss = 0.0
    eps = 0.02

    def term(err, mask, importance):
        nonlocal loss
        ww = weight[mask]
        ww = ww / max(ww.sum(), 1e-12)
        v = np.sqrt(err * err + eps * eps)
        loss += importance * float(np.sum(ww * (v - eps)))
        return importance * ww * err / v
    for j in range(3):
        m = present & np.isfinite(truth[:, j])
        if j == 2:
            m &= ~censored
        if m.any():
            G[m, j] += term(pr[m, j] - truth[m, j], m, 1.0 if j < 2 else 1 / 1.5)
    md = present & ~censored & np.isfinite(truth[:, 0]) & np.isfinite(truth[:, 2])
    if md.any():
        err = pr[md, 2] - pr[md, 0] - (truth[md, 2] - truth[md, 0])
        grad = term(err, md, float(duration_weight) / 1.5)
        G[md, 2] += grad
        G[md, 0] -= grad
    mc = present & censored
    if mc.any() and censor_weight > 0:
        err = np.maximum(CENSOR_BOUND - pr[mc, 2], 0.0)
        G[mc, 2] -= term(err, mc, float(censor_weight) / 1.5) * (err > 0)
    for a, b in ((0, 1), (1, 2)):
        m = present
        err = np.maximum(MIN_GAP + pr[m, a] - pr[m, b], 0.0)
        grad = term(err, m, 0.1) * (err > 0)
        G[m, a] += grad
        G[m, b] -= grad
    n = max(int(present.sum()), 1)
    loss += float(alpha) / (2 * n) * float(np.sum(W[:-1] ** 2))
    grad = Z.T @ G
    grad[:-1] += float(alpha) / n * W[:-1]
    return (loss, grad.reshape(-1))
def joint_heads(X, data, cfg, audit):
    heads = direct_heads(X, data, 10.0, audit)
    audit.record(data, 'joint_loss_optimization')
    Z = heads.scaler.transform(X)
    Z = np.column_stack((Z, np.ones(len(Z))))
    truth = np.column_stack([np.asarray(data[b], float) for b in ('onset', 'peak', 'offset')])
    pos = np.asarray(data['present'], int) == 1
    cen = np.asarray(data['censored'], int) == 1
    W = np.column_stack([np.r_[r.coef_, r.intercept_] for r in heads.regressors])
    opt = minimize(joint_loss_and_grad, W.ravel(), args=(
        Z,
        truth,
        pos,
        cen,
        np.asarray(data['weight']),
        cfg['alpha'],
        cfg['duration_weight'],
        cfg['censor_weight']
    ), method='L-BFGS-B', jac=True, options=dict(maxiter=3000, ftol=1e-10, gtol=1e-06, maxls=40, maxcor=30))
    if not np.isfinite(opt.fun) or not np.isfinite(opt.x).all():
        raise RuntimeError('Joint timing loss diverged.')
    if not opt.success and np.linalg.norm(opt.jac, np.inf) > 0.005:
        raise RuntimeError(f'Joint timing optimization did not converge: {opt.message}; grad={np.linalg.norm(opt.jac, np.inf):.4g}')
    heads.coefficients = opt.x.reshape(Z.shape[1], 3)
    return (heads, dict(optimizer_success=bool(opt.success), iterations=int(opt.nit), objective=float(opt.fun), gradient_inf=float(np.linalg.norm(opt.jac, np.inf))))
def shift_training(data, seconds, augmentation_weight):
    """Create shifted source examples, never target/test examples.
    Edge values are extended; left states pad as baseline, right states stay in the
    last observed state. Shifts that move onset out of view are not used.
    """
    pieces = [{k: np.array(v, copy=True) for k, v in data.items()}]
    for shift in (-float(seconds), float(seconds)):
        pos = np.asarray(data['present']) == 1
        mask = ~pos | (data['onset'] + shift >= 0.25) & (data['peak'] + shift <= GRID_END - 0.5)
        q = subset(data, mask)
        q = {k: np.array(v, copy=True) for k, v in q.items()}
        m = np.asarray(q['present']) == 1
        for b in ('onset', 'peak', 'offset'):
            q[b][m] += shift
        q['censored'] = ((q['censored'] == 1) | (q['offset'] > CENSOR_BOUND) & m).astype(np.int8)
        query = GRID - shift
        obs = q['obs'].reshape(len(q['obs']), 4, len(GRID))
        for i in range(len(obs)):
            for c in range(3):
                obs[i, c] = np.interp(query, GRID, obs[i, c])
            obs[i, 3] = np.gradient(obs[i, 2], GRID_DT)
        q['obs'] = obs.reshape(len(obs), -1)
        F = q['state_feat']
        for i in range(len(F)):
            for c in range(F.shape[2]):
                F[i, :, c] = np.interp(query, GRID, F[i, :, c])
            F[i, :, 3] = np.gradient(F[i, :, 0], GRID_DT)
            F[i, :, 4] = np.gradient(F[i, :, 1], GRID_DT)
            F[i, :, 5] = np.gradient(F[i, :, 2], GRID_DT)
        j = np.clip(np.rint(query / GRID_DT).astype(int), 0, len(GRID) - 1)
        q['states'] = q['states'][:, j].copy()
        q['states'][:, query < 0] = 0
        q['weight'] = q['weight'] * float(augmentation_weight) / 2
        pieces.append(q)
    return cat(pieces)
def local_profiles(x, probabilities, anchors, response_prob, radius):
    require_predictors_only(x)
    F = finite_model_array(x['state_feat'])
    P = np.asarray(probabilities, float)
    n = len(F)
    rel = np.linspace(-float(radius), float(radius), 9)
    out = []
    for b in range(3):
        rows = []
        for i in range(n):
            centre = float(anchors[i, b])
            t = centre + rel
            f = np.column_stack([np.interp(t, GRID, F[i, :, c]) for c in range(F.shape[2])])
            p = np.column_stack([np.interp(t, GRID, P[i, :, c]) for c in range(P.shape[2])])
            seen = ((t >= 0) & (t <= GRID_END)).astype(float)
            context = np.r_[anchors[i], response_prob[i], anchors[i, 1] - anchors[i, 0], anchors[i, 2] - anchors[i, 1]]
            rows.append(np.r_[f.ravel(), p.ravel(), seen, context])
        out.append(np.asarray(rows, float))
    return out
@dataclass
class CandidateFit:
    representation: Representation
    heads: Heads
    kind: str
    cfg: Dict[str, float]
    zoom_scalers: Any = None
    zoom_models: Any = None

    def predict(self, x):
        require_predictors_only(x)
        X, P = self.representation.transform(x)
        raw, prob = self.heads.raw(X)
        if self.kind == 'N4':
            f = local_profiles(x, P, raw, prob, self.cfg['radius'])
            for b in range(3):
                delta = self.zoom_models[b].predict(self.zoom_scalers[b].transform(f[b]))
                raw[:, b] += self.cfg['strength'] * np.clip(delta, -self.cfg['radius'], self.cfg['radius'])
        return make_prediction(raw, prob)
class Scope:
    """Caches new learned components only within one source-fit scope.
    The same scope is shared across candidate settings, not across disjoint data.
    """

    def __init__(self, data, audit):
        self.data = data
        self.audit = audit
        self._base_state = None
        self._base_bundle = None
        self._duration = {}
        self._aug = {}
        self._zoom = None
        self._local = {}

    def base(self):
        if self._base_bundle is None:
            self._base_state = fit_state(self.data, self.audit)
            rep = Representation(self._base_state)
            X, _ = rep.transform(predictor_view(self.data))
            self._base_bundle = (rep, X)
        return self._base_bundle

    def duration(self, mix, temp):
        k = (mix, temp)
        if k not in self._duration:
            rep0, _ = self.base()
            h = fit_hazards(self.data, mix, self.audit)
            rep = Representation(rep0.state, 'duration', h, temp)
            X, _ = rep.transform(predictor_view(self.data))
            self._duration[k] = (rep, X)
        return self._duration[k]

    def augment(self, shift, mass):
        k = (shift, mass)
        if k not in self._aug:
            q = shift_training(self.data, shift, mass)
            self.audit.record(q, 'shift_augmentation')
            rep = Representation(fit_state(q, self.audit, 'augmented_state_encoder'))
            X, _ = rep.transform(predictor_view(q))
            self._aug[k] = (q, rep, X)
        return self._aug[k]

    def zoom_anchors(self):
        if self._zoom is not None:
            return self._zoom
        pids = sorted(set(self.data['pid']))
        n = len(self.data['pid'])
        raw = np.zeros((n, 3))
        rp = np.zeros(n)
        P = np.zeros((n, len(GRID), 5))
        covered = np.zeros(n, int)
        for held in grouped_folds(pids, stable_seed('zoom_oof', *pids), ZOOM_OOF_FOLDS):
            m = np.isin(self.data['pid'], held)
            tr = subset(self.data, ~m)
            va = subset(self.data, m)
            self.audit.record(tr, 'zoom_coarse_fit', held)
            subaudit = FitAudit(sorted(set(tr['pid'])), self.audit.forbidden | set(held), self.audit.context + '/zoom_oof')
            rep = Representation(fit_state(tr, subaudit))
            Xt, _ = rep.transform(predictor_view(tr))
            heads = direct_heads(Xt, tr, 10.0, subaudit)
            Xv, Pv = rep.transform(predictor_view(va))
            r, p = heads.raw(Xv)
            raw[m] = r
            rp[m] = p
            P[m] = Pv
            covered[m] += 1
            self.audit.rows.extend(subaudit.rows)
        if not np.all(covered == 1):
            raise RuntimeError('OOF anchor coverage failed.')
        self._zoom = (raw, rp, P)
        return self._zoom

    def fit(self, kind, cfg):
        notes = {}
        if kind == 'N1':
            rep, X = self.duration(cfg['duration_mix'], cfg['temperature'])
            h = direct_heads(X, self.data, cfg['alpha'], self.audit)
        elif kind == 'N2':
            rep, X = self.base()
            h, notes = joint_heads(X, self.data, cfg, self.audit)
        elif kind == 'N3':
            q, rep, X = self.augment(cfg['shift_seconds'], cfg['augmentation_weight'])
            h = direct_heads(X, q, cfg['alpha'], self.audit)
            notes['augmented_training_rows'] = len(q['pid'])
        elif kind == 'N4':
            rep, X = self.base()
            h = direct_heads(X, self.data, 10.0, self.audit)
            raw, rp, P = self.zoom_anchors()
            if cfg['radius'] not in self._local:
                self._local[cfg['radius']] = local_profiles(predictor_view(self.data), P, raw, rp, cfg['radius'])
            F = self._local[cfg['radius']]
            scs = []
            regs = []
            self.audit.record(self.data, 'local_residual_heads_on_OOF_anchors')
            for b, name in enumerate(('onset', 'peak', 'offset')):
                y = np.asarray(self.data[name])
                mask = (self.data['present'] == 1) & np.isfinite(y)
                if name == 'offset':
                    mask &= self.data['censored'] == 0
                sc = StableScaler().fit(F[b][mask])
                residual = y[mask] - raw[mask, b]
                reg = Ridge(alpha=cfg['alpha']).fit(sc.transform(F[b][mask]), residual)
                scs.append(sc)
                regs.append(reg)
            return (CandidateFit(rep, h, kind, cfg, scs, regs), notes)
        else:
            raise ValueError(kind)
        return (CandidateFit(rep, h, kind, cfg), notes)
@dataclass
class NoiseProfile:
    n_channels: int
    hbo_scale: np.ndarray
    hbr_scale: np.ndarray
    phi_hbo: float
    phi_hbr: float
def safe_lag1(a: np.ndarray) -> float:
    vals = []
    for row in a:
        x, y = (row[:-1], row[1:])
        if len(x) >= 3 and np.std(x) > 1e-10 and (np.std(y) > 1e-10):
            vals.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.clip(np.nanmedian(vals) if vals else 0.5, 0.0, 0.95))
def ar_noise(scale: np.ndarray, phi: float, rng: np.random.Generator, n_time: int) -> np.ndarray:
    C = len(scale)
    x = np.zeros((C, n_time), float)
    common = rng.normal(size=n_time)
    for k in range(n_time):
        eps = 0.75 * rng.normal(size=C) + 0.25 * common[k]
        if k == 0:
            x[:, k] = eps
        else:
            x[:, k] = phi * x[:, k - 1] + math.sqrt(max(1.0 - phi ** 2, 1e-06)) * eps
    return x * scale[:, None]
def piecewise_shape(onset: float, rise: float, active: float, decay: float) -> Tuple[np.ndarray, float, float, np.ndarray]:
    peak = onset + rise
    active_end = peak + active
    offset = active_end + decay
    y = np.zeros(len(GRID), float)
    states = np.zeros(len(GRID), int)
    for i, tt in enumerate(GRID):
        if tt < onset:
            y[i] = 0.0
            states[i] = 0
        elif tt < peak:
            u = (tt - onset) / max(rise, 1e-08)
            y[i] = 0.5 - 0.5 * np.cos(np.pi * u)
            states[i] = 1
        elif tt < active_end:
            y[i] = 1.0
            states[i] = 2
        elif tt < offset:
            u = (tt - active_end) / max(decay, 1e-08)
            y[i] = 0.5 + 0.5 * np.cos(np.pi * u)
            states[i] = 3
        else:
            y[i] = 0.0
            states[i] = 4
    return (y, peak, offset, states)
def synthetic_trial(profile: NoiseProfile, rng: np.random.Generator, positive: bool) -> Dict[str, Any]:
    H = ar_noise(profile.hbo_scale, profile.phi_hbo, rng, len(FULL_T))
    R = ar_noise(profile.hbr_scale, profile.phi_hbr, rng, len(FULL_T))
    systemic = rng.uniform(0.0, 0.8) * np.nanmedian(profile.hbo_scale) * np.sin(2 * np.pi * rng.uniform(0.06, 0.12) * FULL_T + rng.uniform(0, 2 * np.pi))
    H += systemic[None, :]
    R += rng.uniform(0.5, 1.0) * systemic[None, :]
    if not positive:
        return dict(
            hbo=H,
            hbr=R,
            present=0,
            onset=np.nan,
            peak=np.nan,
            offset=np.nan,
            censored=0,
            clean_shape=np.zeros(len(GRID), float),
            states=np.zeros(len(GRID), int)
        )
    onset = float(rng.uniform(0.5, 4.5))
    rise = float(rng.uniform(1.0, 3.5))
    active = float(rng.uniform(0.5, 3.0))
    if rng.random() < 0.18:
        decay = float(rng.uniform(7.0, 12.0))
    else:
        decay = float(rng.uniform(2.0, 8.0))
    shape, peak, offset, states = piecewise_shape(onset, rise, active, decay)
    post_shape = np.zeros(len(FULL_T), float)
    post_shape[POST_MASK] = shape
    support = rng.uniform(0.55, 1.0, size=profile.n_channels)
    amp = rng.uniform(1.5, 4.5) * profile.hbo_scale
    H += (amp * support)[:, None] * post_shape[None, :]
    R -= (amp * rng.uniform(0.35, 0.7) * rng.uniform(0.7, 1.1, size=profile.n_channels))[:, None] * post_shape[None, :]
    return dict(hbo=H, hbr=R, present=1, onset=onset, peak=peak, offset=offset, censored=int(offset > GRID_END - 0.25), clean_shape=shape, states=states)
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
    return dict(H=H, R=R, ghbo=ghbo, ghbr=ghbr, anti=anti, dhbo=dhbo, dhbr=dhbr, danti=danti, cons_hbo=cons_hbo, cons_hbr=cons_hbr, iqr_hbo=iqr_hbo, iqr_hbr=iqr_hbr)
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
import h5py
EXTERNAL_VERSION = 'MOD13_N4_FROZEN_BALINT_KHAN_ZERO_SHOT_V1_20260913'
EXTERNAL_OUT_DIR = '91_MOD13_N4_FROZEN_BALINT_KHAN_ZERO_SHOT_v1'
N4_LOCKED = dict(radius=1.0, alpha=100.0, strength=0.5)
GLOBAL_HSCR_ALPHA = 10.0
BASELINE = (-5.0, 0.0)
FULL_T = np.arange(BASELINE[0], GRID_END + GRID_DT / 2.0, GRID_DT)
POST_MASK = FULL_T >= 0
EXTERNAL_TEST_POS = 24
EXTERNAL_TEST_NEG = 12
BALINT_DEFAULT = Path('/content/gdrive/MyDrive/HonorProject/HemoBoundary/Balint/processed/balint_epochs_10hz.h5')
KHAN_DEFAULT = Path('/content/gdrive/MyDrive/HonorProject/KHAN2026_FINGER_HEMOBOUNDARY_EXTERNAL_PREP/processed/khan2026_finger_epochs_10hz.h5')
BOOT_REPS_EXTERNAL = 20000
SIGNFLIP_REPS_EXTERNAL = 20000

def ext_file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
def ext_try_mount():
    if Path('/content/gdrive/MyDrive').exists() or Path('/content/drive/MyDrive').exists():
        return
    try:
        from google.colab import drive
        drive.mount('/content/gdrive', force_remount=False)
    except Exception:
        pass
def ext_find_shin_root(explicit=None):
    ext_try_mount()
    choices = []
    if explicit:
        choices.append(Path(explicit))
    choices.extend([Path('/content/gdrive/MyDrive/HonorProject/shin2018_word_generation_state_preparation_v1'), Path('/content/drive/MyDrive/HonorProject/shin2018_word_generation_state_preparation_v1')])
    for p in choices:
        if p.is_dir():
            return p
    raise FileNotFoundError('Could not locate SHIN project root. Use --shin-root if needed.')
def ext_locate_source_cache(root, explicit=None):
    choices = []
    if explicit:
        choices.append(Path(explicit))
    choices.extend([root / '82_D5_D13_merged_fully_nested_high_risk_v1' / 'synthetic_cache', root / '88_MOD13_HSCR_refined_grid180_v3' / 'synthetic_cache'])
    for p in choices:
        if p.is_dir() and all(((p / f'{pid}.npz').is_file() for pid in ALL_PIDS)):
            return p
    raise FileNotFoundError('Could not locate the completed 26-participant SHIN RAW_D_SCREEN cache.')
def ext_load_source_caches(cache_dir):
    caches = {}
    for pid in ALL_PIDS:
        path = Path(cache_dir) / f'{pid}.npz'
        with np.load(path, allow_pickle=False) as z:
            item = {}
            for prefix, split in (('tr', 'train'), ('te', 'test')):
                fields = (*PREDICTOR_KEYS, *LABEL_KEYS)
                missing = [k for k in fields if f'{prefix}_{k}' not in z.files]
                if missing:
                    raise RuntimeError(f'{path.name}/{split} missing {missing}')
                d = {k: np.array(z[f'{prefix}_{k}'], copy=True) for k in fields}
                n = len(d['present'])
                if d['state_feat'].shape != (n, len(GRID), 9):
                    raise RuntimeError(f'{path.name}/{split}: state feature shape mismatch.')
                if d['obs'].shape != (n, 4 * len(GRID)):
                    raise RuntimeError(f'{path.name}/{split}: obs shape mismatch.')
                for k in PREDICTOR_KEYS:
                    d[k] = finite_model_array(d[k])
                d['pid'] = np.full(n, pid, dtype='U5')
                d['weight'] = np.ones(n, float)
                item[split] = d
            caches[pid] = item
    return caches
def ext_atomic_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    df.to_csv(tmp, index=False, compression='gzip' if str(path).endswith('.gz') else None)
    os.replace(tmp, path)
def ext_atomic_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)
def read_frozen_n4_config(root):
    table = root / '89_MOD13_four_mechanism_tests_v1' / 'reports' / 'FULL' / 'Table3_Selected_Settings.csv'
    audit = dict(table=str(table), expected_lock=N4_LOCKED, table_found=table.is_file())
    if not table.is_file():
        audit.update(status='LOCKED_CONSTANT_FALLBACK', reason='Full saved SHIN N4 selection table not found; using the predeclared lock from the completed 26-participant result.')
        return (dict(N4_LOCKED), audit)
    q = pd.read_csv(table)
    if 'candidate' not in q.columns:
        raise RuntimeError('Saved SHIN selection table has no candidate column.')
    q = q[q['candidate'].astype(str) == 'N4'].copy()
    required = {'radius', 'alpha', 'strength', 'participant'}
    if not required.issubset(q.columns):
        raise RuntimeError(f'Saved SHIN N4 selection table missing {sorted(required - set(q.columns))}')
    if q['participant'].astype(str).nunique() != 26:
        raise RuntimeError('Expected complete 26-participant N4 development selections before external freeze.')
    counts = q.groupby(['radius', 'alpha', 'strength'], as_index=False).size().sort_values([
        'size',
        'radius',
        'alpha',
        'strength'
    ], ascending=[False, True, True, True]).reset_index(drop=True)
    top = counts.iloc[0]
    modal = dict(radius=float(top['radius']), alpha=float(top['alpha']), strength=float(top['strength']))
    audit.update(status='SHIN_MODAL_CONFIG_LOCK', modal_config=modal, modal_count=int(top['size']), all_config_counts=counts.to_dict(orient='records'))
    for k in N4_LOCKED:
        if not np.isclose(float(modal[k]), float(N4_LOCKED[k])):
            raise RuntimeError(f'Saved complete SHIN N4 mode no longer matches the predeclared external lock: saved={modal}, expected={N4_LOCKED}. Do not silently change the external model.')
    return (modal, audit)
def fit_frozen_shin_n4(caches, cfg):
    """
    Final source-only fit:
      - all 26 SHIN participant TRAIN banks
      - N4 OOF anchors remain participant-cross-fitted within those source rows
      - no Bálint/Khan data enters any fit
    """
    train = training_rows(caches, ALL_PIDS)
    audit = FitAudit(ALL_PIDS, [], 'FROZEN_SHIN_EXTERNAL_MODEL')
    scope = Scope(train, audit)
    model, notes = scope.fit('N4', cfg)
    return (model, audit.rows, notes)
def predict_global_hscr_from_n4_model(model, predictors):
    require_predictors_only(predictors)
    X, _ = model.representation.transform(predictors)
    raw, prob = model.heads.raw(X)
    return make_prediction(raw, prob)
def _decode_str_array(a):
    a = np.asarray(a)
    out = []
    for x in a.reshape(-1):
        if isinstance(x, bytes):
            out.append(x.decode('utf-8', errors='replace'))
        else:
            out.append(str(x))
    return np.asarray(out, dtype=str)
def _h5_dataset_inventory(path):
    rows = []
    with h5py.File(path, 'r') as h5:

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                rows.append(dict(path='/' + name, shape=tuple(obj.shape), dtype=str(obj.dtype), ndim=int(obj.ndim)))
        h5.visititems(visit)
    return pd.DataFrame(rows)
def _all_datasets(h5):
    out = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out['/' + name] = obj
    h5.visititems(visit)
    return out
def _score_name(name, tokens):
    low = name.lower()
    return sum((5 if low.endswith('/' + t) or low.split('/')[-1] == t else 1 for t in tokens if t in low))
def _find_time_dataset(datasets):
    candidates = []
    for name, ds in datasets.items():
        if ds.ndim != 1 or not np.issubdtype(ds.dtype, np.number):
            continue
        if len(ds) < 20 or len(ds) > 5000:
            continue
        score = _score_name(name, ('time', 'times', 'epoch_time', 't'))
        if score <= 0:
            continue
        a = np.asarray(ds[:], float).reshape(-1)
        if not np.isfinite(a).all() or not np.all(np.diff(a) > 0):
            continue
        if np.nanmin(a) > -0.1 or np.nanmax(a) < 10.0:
            continue
        candidates.append((score, name, a))
    if not candidates:
        raise RuntimeError('Could not identify a monotonic HDF5 epoch time vector spanning pre/post stimulus.')
    candidates.sort(key=lambda x: (-x[0], len(x[2]), x[1]))
    return (candidates[0][1], candidates[0][2])
def _find_participant_dataset(datasets):
    candidates = []
    allowed_leafs = {'participant', 'participants', 'participant_id', 'participant_ids', 'subject', 'subjects', 'subject_id', 'subject_ids', 'sub', 'sub_id'}
    for name, ds in datasets.items():
        if ds.ndim != 1:
            continue
        leaf = name.rstrip('/').split('/')[-1].lower()
        if leaf not in allowed_leafs:
            continue
        try:
            vals = _decode_str_array(ds[:])
        except Exception:
            continue
        unique = sorted(set(vals.tolist()))
        if not 2 <= len(unique) <= 200:
            continue
        looks_like = sum((bool(re.fullmatch('(?i)(?:sub[-_]?)?[A-Z]*\\d{1,4}', v.strip())) for v in unique))
        if looks_like == 0 and leaf in {'sub', 'sub_id'}:
            continue
        score = 10
        if 'participant' in leaf:
            score += 3
        if leaf.endswith('_id') or leaf.endswith('_ids'):
            score += 1
        candidates.append((score, name, vals))
    if not candidates:
        return (None, None)
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return (candidates[0][1], candidates[0][2])
def _normalize_trial_channel_time(a, n_trials, n_time, label):
    x = np.asarray(a, float)
    if x.ndim != 3:
        raise RuntimeError(f'{label}: expected 3D trial/channel/time data, got {x.shape}')
    t_axes = [i for i, s in enumerate(x.shape) if s == n_time]
    if len(t_axes) != 1:
        raise RuntimeError(f'{label}: could not uniquely identify time axis length {n_time} in shape {x.shape}')
    x = np.moveaxis(x, t_axes[0], -1)
    first_two = x.shape[:2]
    tr_axes = [i for i, s in enumerate(first_two) if s == n_trials]
    if len(tr_axes) != 1:
        raise RuntimeError(f'{label}: could not uniquely identify trial axis length {n_trials} in shape {x.shape}')
    if tr_axes[0] == 1:
        x = np.swapaxes(x, 0, 1)
    if x.shape[0] != n_trials or x.shape[-1] != n_time:
        raise RuntimeError(f'{label}: normalization failed, shape={x.shape}')
    return x
def _find_separate_chromophore_pair(datasets, n_trials, n_time):
    hbo = []
    hbr = []
    for name, ds in datasets.items():
        if ds.ndim != 3 or not np.issubdtype(ds.dtype, np.number):
            continue
        low = name.lower()
        score_o = _score_name(name, ('hbo', 'oxy', 'oxygenated'))
        score_r = _score_name(name, ('hbr', 'deoxy', 'deoxygenated'))
        if 'deoxy' in low or 'deoxygenated' in low:
            score_o = 0
        if score_o > 0:
            hbo.append((score_o, name, ds))
        if score_r > 0:
            hbr.append((score_r, name, ds))
    pairs = []
    for so, no, dso in hbo:
        for sr, nr, dsr in hbr:
            try:
                O = _normalize_trial_channel_time(dso[:], n_trials, n_time, no)
                R = _normalize_trial_channel_time(dsr[:], n_trials, n_time, nr)
            except RuntimeError:
                continue
            if O.shape == R.shape:
                pairs.append((so + sr, no, nr, O, R))
    if not pairs:
        return None
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    _, no, nr, O, R = pairs[0]
    return (no, nr, O, R)
def _find_channel_names(datasets):
    candidates = []
    for name, ds in datasets.items():
        if ds.ndim != 1:
            continue
        score = _score_name(name, ('channel_names', 'channels', 'ch_names', 'channel'))
        if score <= 0:
            continue
        try:
            vals = _decode_str_array(ds[:])
        except Exception:
            continue
        if len(vals) >= 2:
            candidates.append((score, name, vals))
    if not candidates:
        return (None, None)
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return (candidates[0][1], candidates[0][2])
def _normalize_combined_epochs(a, n_trials, n_time, n_channels, label):
    x = np.asarray(a, float)
    if x.ndim != 3:
        raise RuntimeError(f'{label}: combined epoch dataset must be 3D.')
    axes = list(range(3))
    ta = [i for i, s in enumerate(x.shape) if s == n_time]
    ca = [i for i, s in enumerate(x.shape) if s == n_channels]
    ra = [i for i, s in enumerate(x.shape) if s == n_trials]
    if len(ta) != 1 or len(ca) != 1 or len(ra) != 1:
        raise RuntimeError(f'{label}: could not uniquely map trial/channel/time axes in shape {x.shape}')
    x = np.transpose(x, (ra[0], ca[0], ta[0]))
    return x
def _split_combined_by_channel_names(datasets, n_trials, n_time):
    cn_path, names = _find_channel_names(datasets)
    if names is None:
        return None
    idx_o = []
    idx_r = []
    keys_o = []
    keys_r = []
    for i, name in enumerate(names):
        low = name.lower()
        is_r = 'hbr' in low or 'deoxy' in low or 'deoxygenated' in low
        is_o = 'hbo' in low or (('oxy' in low or 'oxygenated' in low) and (not is_r))
        key = re.sub('(?i)(hbo|hbr|deoxy(?:genated)?|oxy(?:genated)?)', '', low)
        key = re.sub('[^a-z0-9]+', '', key)
        if is_o:
            idx_o.append(i)
            keys_o.append(key)
        elif is_r:
            idx_r.append(i)
            keys_r.append(key)
    if not idx_o or not idx_r:
        return None
    map_o = {k: i for k, i in zip(keys_o, idx_o) if k}
    map_r = {k: i for k, i in zip(keys_r, idx_r) if k}
    common = sorted(set(map_o) & set(map_r))
    if common:
        io = np.asarray([map_o[k] for k in common], int)
        ir = np.asarray([map_r[k] for k in common], int)
    else:
        c = min(len(idx_o), len(idx_r))
        io = np.asarray(idx_o[:c], int)
        ir = np.asarray(idx_r[:c], int)
    candidates = []
    for path, ds in datasets.items():
        if ds.ndim != 3 or not np.issubdtype(ds.dtype, np.number):
            continue
        score = _score_name(path, ('epoch', 'epochs', 'data', 'hb', 'nirs'))
        if score <= 0:
            continue
        try:
            X = _normalize_combined_epochs(ds[:], n_trials=n_trials, n_time=n_time, n_channels=len(names), label=path)
        except RuntimeError:
            continue
        O = X[:, io, :]
        R = X[:, ir, :]
        candidates.append((score, path, cn_path, O, R))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    _, data_path, _, O, R = candidates[0]
    return (data_path + '::HbO/HbR channel split', cn_path, O, R)
def _load_flat_hdf5(path):
    with h5py.File(path, 'r') as h5:
        datasets = _all_datasets(h5)
        time_path, times = _find_time_dataset(datasets)
        participant_path, participants = _find_participant_dataset(datasets)
        if participants is None:
            return None
        n_trials = len(participants)
        n_time = len(times)
        pair = _find_separate_chromophore_pair(datasets, n_trials=n_trials, n_time=n_time)
        if pair is None:
            pair = _split_combined_by_channel_names(datasets, n_trials=n_trials, n_time=n_time)
        if pair is None:
            return None
        hbo_path, hbr_path, H, R = pair
    return dict(
        times=times,
        participant=participants,
        hbo=H,
        hbr=R,
        schema='FLAT_HDF5',
        audit=dict(time_dataset=time_path, participant_dataset=participant_path, hbo_source=hbo_path, hbr_source=hbr_path)
    )
def _participant_group_candidates(h5):
    candidates = []

    def visit(name, obj):
        if not isinstance(obj, h5py.Group):
            return
        leaf = name.rstrip('/').split('/')[-1]
        if re.fullmatch('(?i)(?:sub[-_])?[CS]?\\d{1,3}', leaf):
            candidates.append(name)
    h5.visititems(visit)
    return sorted(set(candidates))
def _load_grouped_hdf5(path):
    participant_blocks = []
    audit_rows = []
    with h5py.File(path, 'r') as h5:
        global_ds = _all_datasets(h5)
        try:
            global_time_path, global_times = _find_time_dataset(global_ds)
        except RuntimeError:
            global_time_path, global_times = (None, None)
        groups = _participant_group_candidates(h5)
        for gp in groups:
            grp = h5[gp]
            datasets = {}

            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets['/' + gp + '/' + name] = obj
            grp.visititems(visit)
            if not datasets:
                continue
            if global_times is not None:
                times = global_times
                time_path = global_time_path
            else:
                try:
                    time_path, times = _find_time_dataset(datasets)
                except RuntimeError:
                    continue
            hbo_ds = []
            hbr_ds = []
            for name, ds in datasets.items():
                if ds.ndim != 3 or not np.issubdtype(ds.dtype, np.number):
                    continue
                low = name.lower()
                so = _score_name(name, ('hbo', 'oxy', 'oxygenated'))
                sr = _score_name(name, ('hbr', 'deoxy', 'deoxygenated'))
                if 'deoxy' in low or 'deoxygenated' in low:
                    so = 0
                if so > 0:
                    hbo_ds.append((so, name, ds))
                if sr > 0:
                    hbr_ds.append((sr, name, ds))
            best = None
            for so, no, dso in hbo_ds:
                for sr, nr, dsr in hbr_ds:
                    if tuple(dso.shape) != tuple(dsr.shape):
                        continue
                    shape = dso.shape
                    t_axes = [i for i, s in enumerate(shape) if s == len(times)]
                    if len(t_axes) != 1:
                        continue
                    ta = t_axes[0]
                    other = [i for i in range(3) if i != ta]
                    trial_lengths = []
                    for n2, ds2 in datasets.items():
                        if ds2.ndim == 1 and _score_name(n2, ('trial', 'trials', 'label', 'event')) > 0:
                            trial_lengths.append(len(ds2))
                    trial_axis = None
                    for length in trial_lengths:
                        matching = [ax for ax in other if shape[ax] == length]
                        if len(matching) == 1:
                            trial_axis = matching[0]
                            break
                    if trial_axis is None:
                        channel_like = [ax for ax in other if shape[ax] <= 128]
                        if len(channel_like) == 1:
                            trial_axis = [ax for ax in other if ax != channel_like[0]][0]
                        else:
                            trial_axis = other[0]
                    channel_axis = [ax for ax in other if ax != trial_axis][0]
                    O = np.transpose(np.asarray(dso[:], float), (trial_axis, channel_axis, ta))
                    R = np.transpose(np.asarray(dsr[:], float), (trial_axis, channel_axis, ta))
                    score = so + sr
                    if best is None or score > best[0]:
                        best = (score, no, nr, O, R)
            if best is None:
                continue
            _, no, nr, O, R = best
            pid = gp.rstrip('/').split('/')[-1]
            participant_blocks.append((pid, times, O, R))
            audit_rows.append(dict(
                participant=pid,
                group='/' + gp,
                hbo_source=no,
                hbr_source=nr,
                time_source=time_path,
                n_trials=int(len(O)),
                n_channels=int(O.shape[1])
            ))
    if not participant_blocks:
        return None
    ref_t = participant_blocks[0][1]
    for pid, t, _, _ in participant_blocks:
        if len(t) != len(ref_t) or not np.allclose(t, ref_t, atol=1e-08, rtol=0):
            raise RuntimeError(f'Grouped HDF5 participant {pid} has a different time vector.')
    participants = []
    H = []
    R = []
    min_c = min((min(x[2].shape[1], x[3].shape[1]) for x in participant_blocks))
    for pid, _, h, r in participant_blocks:
        participants.extend([pid] * len(h))
        H.append(h[:, :min_c, :])
        R.append(r[:, :min_c, :])
    return dict(
        times=np.asarray(ref_t, float),
        participant=np.asarray(participants, dtype=str),
        hbo=np.concatenate(H, axis=0),
        hbr=np.concatenate(R, axis=0),
        schema='GROUPED_HDF5',
        audit=dict(group_rows=audit_rows)
    )
def load_external_hdf5(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    flat = _load_flat_hdf5(path)
    if flat is not None:
        return flat
    grouped = _load_grouped_hdf5(path)
    if grouped is not None:
        return grouped
    inventory = _h5_dataset_inventory(path)
    raise RuntimeError('Could not resolve HbO/HbR/time/participant arrays in external HDF5.\nThe script wrote no results and will not guess a schema.\nHDF5 inventory:\n' + inventory.to_string(index=False))
def _participant_arrays(store, pid):
    mask = store['participant'].astype(str) == str(pid)
    if not np.any(mask):
        raise RuntimeError(f'External participant {pid} has no epochs.')
    H = np.asarray(store['hbo'][mask], float)
    R = np.asarray(store['hbr'][mask], float)
    t = np.asarray(store['times'], float)
    C = min(H.shape[1], R.shape[1])
    H = H[:, :C, :]
    R = R[:, :C, :]
    bm = (t >= BASELINE[0]) & (t < BASELINE[1])
    if bm.sum() < 4:
        raise RuntimeError(f'{pid}: external store has insufficient -5..0 s baseline.')
    finite_fraction = np.mean(np.isfinite(H[:, :, bm]) & np.isfinite(R[:, :, bm]), axis=(0, 2))
    hsd = np.nanmedian(np.nanstd(H[:, :, bm], axis=2), axis=0)
    rsd = np.nanmedian(np.nanstd(R[:, :, bm], axis=2), axis=0)
    valid = (finite_fraction >= 0.8) & np.isfinite(hsd) & np.isfinite(rsd) & (hsd > 1e-12) & (rsd > 1e-12)
    if valid.sum() < 2:
        valid = finite_fraction >= 0.8
    if valid.sum() < 2:
        raise RuntimeError(f'{pid}: fewer than two usable HbO/HbR channels.')
    return (t, H[:, valid, :], R[:, valid, :], valid)
def noise_profile_from_external(store, pid):
    t, H, R, valid = _participant_arrays(store, pid)
    bm = (t >= BASELINE[0]) & (t < BASELINE[1])
    Hb = H[:, :, bm]
    Rb = R[:, :, bm]
    Hb = Hb - np.nanmedian(Hb, axis=2, keepdims=True)
    Rb = Rb - np.nanmedian(Rb, axis=2, keepdims=True)
    hs = np.nanmedian(np.nanstd(Hb, axis=2), axis=0)
    rs = np.nanmedian(np.nanstd(Rb, axis=2), axis=0)
    hs_positive = hs[np.isfinite(hs) & (hs > 0)]
    rs_positive = rs[np.isfinite(rs) & (rs > 0)]
    h_fallback = float(np.nanmedian(hs_positive)) if len(hs_positive) else 1.0
    r_fallback = float(np.nanmedian(rs_positive)) if len(rs_positive) else 1.0
    hs = np.where(np.isfinite(hs) & (hs > 1e-12), hs, h_fallback)
    rs = np.where(np.isfinite(rs) & (rs > 1e-12), rs, r_fallback)
    sample_H = Hb[:min(24, len(Hb))].reshape(-1, Hb.shape[-1])
    sample_R = Rb[:min(24, len(Rb))].reshape(-1, Rb.shape[-1])
    profile = NoiseProfile(n_channels=int(len(hs)), hbo_scale=hs, hbr_scale=rs, phi_hbo=safe_lag1(sample_H), phi_hbr=safe_lag1(sample_R))
    return (profile, int(valid.sum()), int(len(valid)))
def external_synthetic_test(profile, dataset_name, pid):
    rng = np.random.default_rng(stable_seed('RAW_D_SCREEN_EXTERNAL', dataset_name, pid, 'test'))
    trials = [synthetic_trial(profile, rng, True) for _ in range(EXTERNAL_TEST_POS)]
    trials.extend((synthetic_trial(profile, rng, False) for _ in range(EXTERNAL_TEST_NEG)))
    rng.shuffle(trials)
    out = dict(present=[], onset=[], peak=[], offset=[], censored=[], states=[], obs=[], state_feat=[])
    for tr in trials:
        ev = evidence(tr['hbo'], tr['hbr'])
        out['present'].append(tr['present'])
        out['onset'].append(tr['onset'])
        out['peak'].append(tr['peak'])
        out['offset'].append(tr['offset'])
        out['censored'].append(tr['censored'])
        out['states'].append(tr['states'])
        out['obs'].append(whole_vector(ev))
        out['state_feat'].append(state_features(ev))
    q = {
        'present': np.asarray(out['present'], int),
        'onset': np.asarray(out['onset'], float),
        'peak': np.asarray(out['peak'], float),
        'offset': np.asarray(out['offset'], float),
        'censored': np.asarray(out['censored'], int),
        'states': np.asarray(out['states'], int),
        'obs': finite_model_array(np.asarray(out['obs'], float)),
        'state_feat': finite_model_array(np.asarray(out['state_feat'], float))
    }
    return q
def _interp_epoch_channels(arr, source_t, target_t):
    arr = np.asarray(arr, float)
    out = np.full((arr.shape[0], len(target_t)), np.nan, float)
    for c in range(arr.shape[0]):
        x = arr[c]
        m = np.isfinite(x) & np.isfinite(source_t)
        if m.sum() < 2:
            continue
        out[c] = np.interp(target_t, source_t[m], x[m], left=x[m][0], right=x[m][-1])
    keep = np.mean(np.isfinite(out), axis=1) >= 0.95
    return out[keep]
def real_external_predictors(store, pid):
    t, H, R, _ = _participant_arrays(store, pid)
    obs = []
    state = []
    kept = []
    for i in range(len(H)):
        h = _interp_epoch_channels(H[i], t, FULL_T)
        r = _interp_epoch_channels(R[i], t, FULL_T)
        C = min(len(h), len(r))
        if C < 2:
            continue
        ev = evidence(h[:C], r[:C])
        obs.append(whole_vector(ev))
        state.append(state_features(ev))
        kept.append(i)
    if not obs:
        return (None, np.asarray([], int))
    return (dict(obs=finite_model_array(np.asarray(obs, float)), state_feat=finite_model_array(np.asarray(state, float))), np.asarray(kept, int))
def _participant_metrics(dataset, pid, truth, pred, method):
    m = trial_metric_summary(truth, pred)
    return dict(dataset=dataset, participant=pid, method=method, **{k: v for k, v in m.items() if not k.endswith('_localized_rate')})
def _bootstrap_ci(x, seed):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(x))
    if len(x) == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    values = np.empty(BOOT_REPS_EXTERNAL, float)
    done = 0
    while done < BOOT_REPS_EXTERNAL:
        n = min(2000, BOOT_REPS_EXTERNAL - done)
        ix = rng.integers(0, len(x), size=(n, len(x)))
        values[done:done + n] = x[ix].mean(axis=1)
        done += n
    lo, hi = np.quantile(values, [0.025, 0.975])
    return (mean, float(lo), float(hi))
def _signflip_p(x, seed):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan
    obs = abs(float(np.mean(x)))
    if obs < 1e-15:
        return 1.0
    rng = np.random.default_rng(seed)
    hits = 0
    done = 0
    while done < SIGNFLIP_REPS_EXTERNAL:
        n = min(2000, SIGNFLIP_REPS_EXTERNAL - done)
        signs = rng.choice([-1.0, 1.0], size=(n, len(x)))
        stat = np.abs(np.mean(signs * x[None, :], axis=1))
        hits += int(np.sum(stat >= obs - 1e-15))
        done += n
    return float((hits + 1) / (SIGNFLIP_REPS_EXTERNAL + 1))
def _holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(order) - rank) * p[idx])
        out[idx] = min(1.0, running)
    return out
def paired_external_stats(pm, dataset):
    q = pm[pm['dataset'] == dataset].copy()
    base = q[q['method'] == 'FROZEN_HSCR'].set_index('participant')
    n4 = q[q['method'] == 'FROZEN_HSCR_PLUS_LOCAL_REFINEMENT'].set_index('participant')
    common = sorted(set(base.index) & set(n4.index))
    rows = []
    metrics = ['response_balanced_accuracy', 'onset_mae_sec', 'peak_mae_sec', 'offset_mae_sec', 'duration_mae_sec']
    for metric in metrics:
        a = pd.to_numeric(base.loc[common, metric], errors='coerce').to_numpy(float)
        b = pd.to_numeric(n4.loc[common, metric], errors='coerce').to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if metric == 'response_balanced_accuracy':
            gain = b[ok] - a[ok]
        else:
            gain = a[ok] - b[ok]
        mean, lo, hi = _bootstrap_ci(gain, stable_seed('EXT_N4_BOOT', dataset, metric))
        p = _signflip_p(gain, stable_seed('EXT_N4_SIGN', dataset, metric))
        rows.append(dict(
            dataset=dataset,
            reference='FROZEN_HSCR',
            candidate='FROZEN_HSCR_PLUS_LOCAL_REFINEMENT',
            metric=metric,
            n_participants=int(len(gain)),
            reference_mean=float(np.mean(a[ok])) if np.any(ok) else np.nan,
            candidate_mean=float(np.mean(b[ok])) if np.any(ok) else np.nan,
            mean_gain_positive_candidate_better=mean,
            ci95_low=lo,
            ci95_high=hi,
            participants_better=int(np.sum(gain > 1e-12)),
            participants_equal=int(np.sum(np.isclose(gain, 0.0))),
            participants_worse=int(np.sum(gain < -1e-12)),
            signflip_p=p
        ))
    out = pd.DataFrame(rows)
    out['holm_p_4_timing'] = np.nan
    ix = out.index[out['metric'].isin(TIMING)].tolist()
    out.loc[ix, 'holm_p_4_timing'] = _holm(out.loc[ix, 'signflip_p'].to_numpy(float))
    return out
def performance_external(pm):
    return pm.groupby(['dataset', 'method'], as_index=False).agg(
        n_participants=('participant', 'nunique'),
        response_balanced_accuracy=('response_balanced_accuracy', 'mean'),
        onset_mae_sec=('onset_mae_sec', 'mean'),
        peak_mae_sec=('peak_mae_sec', 'mean'),
        offset_mae_sec=('offset_mae_sec', 'mean'),
        duration_mae_sec=('duration_mae_sec', 'mean'),
        selection_loss=('selection_loss', 'mean')
    ).sort_values(['dataset', 'selection_loss']).reset_index(drop=True)
def evaluate_external_dataset(dataset_name, h5_path, model, output_dir, include_real_descriptive=True):
    store = load_external_hdf5(h5_path)
    pids = sorted(set(store['participant'].astype(str).tolist()))
    if len(pids) < 2:
        raise RuntimeError(f'{dataset_name}: expected multiple external participants.')
    per_trial = []
    pm = []
    real_rows = []
    qc = []
    for j, pid in enumerate(pids, 1):
        profile, valid_c, total_c = noise_profile_from_external(store, pid)
        test = external_synthetic_test(profile, dataset_name, pid)
        predictors = predictor_view(test)
        pred_base = predict_global_hscr_from_n4_model(model, predictors)
        pred_n4 = model.predict(predictors)
        for method, pred in (('FROZEN_HSCR', pred_base), ('FROZEN_HSCR_PLUS_LOCAL_REFINEMENT', pred_n4)):
            row = pd.DataFrame(dict(
                dataset=dataset_name,
                participant=pid,
                method=method,
                synthetic_index=np.arange(len(pred)),
                truth_present=test['present'],
                truth_onset=test['onset'],
                truth_peak=test['peak'],
                truth_offset=test['offset'],
                truth_censored=test['censored'],
                pred_response=pred['pred_response'].to_numpy(),
                pred_onset=pred['pred_onset'].to_numpy(),
                pred_peak=pred['pred_peak'].to_numpy(),
                pred_offset=pred['pred_offset'].to_numpy()
            ))
            per_trial.append(row)
            pm.append(_participant_metrics(dataset_name, pid, test, pred, method))
        if include_real_descriptive:
            real_x, kept = real_external_predictors(store, pid)
            if real_x is not None:
                rb = predict_global_hscr_from_n4_model(model, real_x)
                rn = model.predict(real_x)
                for method, pred in (('FROZEN_HSCR', rb), ('FROZEN_HSCR_PLUS_LOCAL_REFINEMENT', rn)):
                    rr = pd.DataFrame(dict(
                        dataset=dataset_name,
                        participant=pid,
                        real_trial_index=kept,
                        method=method,
                        pred_response=pred['pred_response'].to_numpy(),
                        pred_onset=pred['pred_onset'].to_numpy(),
                        pred_peak=pred['pred_peak'].to_numpy(),
                        pred_offset=pred['pred_offset'].to_numpy(),
                        timing_truth_available=False
                    ))
                    real_rows.append(rr)
        qc.append(dict(
            dataset=dataset_name,
            participant=pid,
            h5_schema=store['schema'],
            n_real_epochs=int(np.sum(store['participant'].astype(str) == pid)),
            usable_channels=int(valid_c),
            channels_before_validity_filter=int(total_c),
            phi_hbo=float(profile.phi_hbo),
            phi_hbr=float(profile.phi_hbr),
            median_hbo_noise_scale=float(np.median(profile.hbo_scale)),
            median_hbr_noise_scale=float(np.median(profile.hbr_scale))
        ))
        print(f'[{dataset_name} {j:02d}/{len(pids):02d}] {pid}: {valid_c}/{total_c} channels | known-boundary zero-shot complete', flush=True)
    per_trial = pd.concat(per_trial, ignore_index=True)
    pm = pd.DataFrame(pm)
    qc = pd.DataFrame(qc)
    real = pd.concat(real_rows, ignore_index=True) if real_rows else pd.DataFrame()
    stats = paired_external_stats(pm, dataset_name)
    ds_out = Path(output_dir) / dataset_name
    ds_out.mkdir(parents=True, exist_ok=True)
    ext_atomic_csv(per_trial, ds_out / 'ZERO_SHOT_KNOWN_BOUNDARY_PER_TRIAL.csv.gz')
    ext_atomic_csv(pm, ds_out / 'ZERO_SHOT_PARTICIPANT_METRICS.csv')
    ext_atomic_csv(stats, ds_out / 'N4_VS_HSCR_PAIRED_STATS.csv')
    ext_atomic_csv(qc, ds_out / 'EXTERNAL_INPUT_QC.csv')
    if len(real):
        ext_atomic_csv(real, ds_out / 'REAL_TRIAL_DESCRIPTIVE_PREDICTIONS.csv.gz')
    ext_atomic_json(dict(
        dataset=dataset_name,
        hdf5=str(h5_path),
        hdf5_schema=store['schema'],
        hdf5_audit=store['audit'],
        target_participants=pids,
        target_rows_used_for_fitting=False,
        target_labels_used_for_fitting=False,
        target_timing_used_for_model_selection=False,
        known_boundary_protocol='24 positive + 12 negative trials per participant, same RAW_D_SCREEN response/noise generator family, noise profile estimated from external pre-cue physiology',
        real_timing_truth_available=False,
        real_trial_predictions_role='DESCRIPTIVE_ONLY'
    ), ds_out / 'EXTERNAL_PROTOCOL_AUDIT.json')
    return (per_trial, pm, stats, qc, real)
def _fixture_source_data(n_pids=6):
    caches = {}
    profile = NoiseProfile(n_channels=5, hbo_scale=np.linspace(0.8, 1.2, 5), hbr_scale=np.linspace(0.7, 1.1, 5), phi_hbo=0.55, phi_hbr=0.5)
    for k in range(n_pids):
        pid = f'VP{k + 1:03d}'
        rng = np.random.default_rng(stable_seed('EXT_SELFTEST_SOURCE', pid))
        trials = [synthetic_trial(profile, rng, True) for _ in range(14)]
        trials += [synthetic_trial(profile, rng, False) for _ in range(6)]
        rng.shuffle(trials)
        d = dict(present=[], onset=[], peak=[], offset=[], censored=[], states=[], obs=[], state_feat=[])
        for tr in trials:
            ev = evidence(tr['hbo'], tr['hbr'])
            for name in ('present', 'onset', 'peak', 'offset', 'censored', 'states'):
                d[name].append(tr[name])
            d['obs'].append(whole_vector(ev))
            d['state_feat'].append(state_features(ev))
        q = {
            'present': np.asarray(d['present'], int),
            'onset': np.asarray(d['onset'], float),
            'peak': np.asarray(d['peak'], float),
            'offset': np.asarray(d['offset'], float),
            'censored': np.asarray(d['censored'], int),
            'states': np.asarray(d['states'], int),
            'obs': finite_model_array(np.asarray(d['obs'], float)),
            'state_feat': finite_model_array(np.asarray(d['state_feat'], float)),
            'pid': np.full(len(trials), pid, dtype='U5'),
            'weight': np.ones(len(trials), float)
        }
        caches[pid] = {'train': q}
    return caches
def _write_h5_fixture(path, grouped=False):
    rng = np.random.default_rng(1234)
    t = np.arange(-5.0, 20.0, 0.1)
    pids = ['C01', 'C02']
    with h5py.File(path, 'w') as h5:
        h5.create_dataset('times', data=t)
        if not grouped:
            part = np.asarray([pids[0]] * 5 + [pids[1]] * 5, dtype=h5py.string_dtype('utf-8'))
            H = rng.normal(size=(10, 4, len(t)))
            R = rng.normal(size=(10, 4, len(t)))
            h5.create_dataset('participant', data=part)
            h5.create_dataset('hbo_epochs', data=H)
            h5.create_dataset('hbr_epochs', data=R)
        else:
            root = h5.create_group('participants')
            for pid in pids:
                g = root.create_group(pid)
                g.create_dataset('hbo', data=rng.normal(size=(5, 4, len(t))))
                g.create_dataset('hbr', data=rng.normal(size=(5, 4, len(t))))
                g.create_dataset('trial_index', data=np.arange(5))
def external_self_tests():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        flat = Path(td) / 'flat.h5'
        grouped = Path(td) / 'grouped.h5'
        _write_h5_fixture(flat, grouped=False)
        _write_h5_fixture(grouped, grouped=True)
        a = load_external_hdf5(flat)
        b = load_external_hdf5(grouped)
        assert a['hbo'].shape == (10, 4, 250)
        assert b['hbo'].shape == (10, 4, 250)
        assert len(set(a['participant'])) == 2
        assert len(set(b['participant'])) == 2
    source = _fixture_source_data()
    train = cat([source[p]['train'] for p in sorted(source)])
    audit = FitAudit(sorted(source), [], 'EXT_SELFTEST')
    scope = Scope(train, audit)
    model, _ = scope.fit('N4', dict(N4_LOCKED))
    profile = NoiseProfile(n_channels=5, hbo_scale=np.ones(5), hbr_scale=np.ones(5), phi_hbo=0.5, phi_hbr=0.5)
    ext = external_synthetic_test(profile, 'FIXTURE', 'C01')
    base = predict_global_hscr_from_n4_model(model, predictor_view(ext))
    zoom = model.predict(predictor_view(ext))
    assert len(base) == EXTERNAL_TEST_POS + EXTERNAL_TEST_NEG
    assert len(zoom) == len(base)
    assert set(base.columns) == {'pred_response', 'pred_onset', 'pred_peak', 'pred_offset'}
    print('External model checks: PASS')
def parse_external_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--shin-root', default=None)
    p.add_argument('--source-cache', default=None)
    p.add_argument('--balint-h5', default=None)
    p.add_argument('--khan-h5', default=None)
    p.add_argument('--output', default=None)
    p.add_argument('--datasets', default='BALINT,KHAN', help='BALINT, KHAN, or BALINT,KHAN')
    p.add_argument('--skip-real-descriptive', action='store_true', help='Skip descriptive predictions on real external task epochs.')
    p.add_argument('--self-test-only', action='store_true')
    args, unknown = p.parse_known_args(argv)
    if unknown:
        print('Ignoring notebook/runtime arguments:', unknown)
    return args
def main_external(argv=None):
    args = parse_external_args(argv)
    external_self_tests()
    if args.self_test_only:
        return
    root = ext_find_shin_root(args.shin_root)
    cache_dir = ext_locate_source_cache(root, args.source_cache)
    caches = ext_load_source_caches(cache_dir)
    cfg, freeze_audit = read_frozen_n4_config(root)
    print('=' * 170)
    print('HEMOBOUNDARY — BOUNDARY ESTIMATION')
    print('BÁLINT + KHAN ZERO-SHOT EXTERNAL VALIDATION')
    print('=' * 170)
    print('SHIN source cache             :', cache_dir)
    print('Source participants           :', len(caches))
    print('Model configuration:', cfg)
    print('Global regularization:', GLOBAL_HSCR_ALPHA)
    print('External fitting              : NONE')
    print('External tuning               : NONE')
    print('External labels in fitting    : NONE')
    print('Known-boundary external truth : semi-synthetic on target-derived baseline physiology')
    print('Real external timing truth    : unavailable; real predictions descriptive only')
    model, fit_audit, notes = fit_frozen_shin_n4(caches, cfg)
    out = Path(args.output) if args.output else root / EXTERNAL_OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    ext_atomic_json(dict(
        version=EXTERNAL_VERSION,
        frozen_n4_config=cfg,
        frozen_global_hscr_alpha=GLOBAL_HSCR_ALPHA,
        freeze_audit=freeze_audit,
        shin_fit_audit=fit_audit,
        source_fit_notes=notes,
        source_cache=str(cache_dir),
        source_cache_hashes={pid: ext_file_sha(Path(cache_dir) / f'{pid}.npz') for pid in ALL_PIDS},
        external_fitting=False,
        external_tuning=False,
        external_adaptation=False,
        external_labels_used_for_fitting=False
    ), out / 'FROZEN_MODEL_AUDIT.json')
    requested = [x.strip().upper() for x in args.datasets.split(',') if x.strip()]
    allowed = {'BALINT', 'KHAN'}
    if not requested or not set(requested).issubset(allowed):
        raise ValueError('--datasets must contain BALINT and/or KHAN.')
    all_pm = []
    all_stats = []
    all_qc = []
    for name in requested:
        if name == 'BALINT':
            h5 = Path(args.balint_h5) if args.balint_h5 else BALINT_DEFAULT
            if not h5.is_file():
                alt = Path(str(h5).replace('/content/gdrive/', '/content/drive/'))
                if alt.is_file():
                    h5 = alt
        else:
            h5 = Path(args.khan_h5) if args.khan_h5 else KHAN_DEFAULT
            if not h5.is_file():
                alt = Path(str(h5).replace('/content/gdrive/', '/content/drive/'))
                if alt.is_file():
                    h5 = alt
        print('\n' + '=' * 170)
        print(f'{name} — EXTERNAL EVALUATION')
        print('=' * 170)
        print('HDF5:', h5)
        _, pm, stats, qc, _ = evaluate_external_dataset(name, h5, model, out, include_real_descriptive=not args.skip_real_descriptive)
        all_pm.append(pm)
        all_stats.append(stats)
        all_qc.append(qc)
    all_pm = pd.concat(all_pm, ignore_index=True)
    all_stats = pd.concat(all_stats, ignore_index=True)
    all_qc = pd.concat(all_qc, ignore_index=True)
    perf = performance_external(all_pm)
    ext_atomic_csv(all_pm, out / 'ALL_EXTERNAL_PARTICIPANT_METRICS.csv')
    ext_atomic_csv(all_stats, out / 'ALL_EXTERNAL_N4_VS_HSCR_STATS.csv')
    ext_atomic_csv(all_qc, out / 'ALL_EXTERNAL_INPUT_QC.csv')
    ext_atomic_csv(perf, out / 'TABLE_EXTERNAL_PERFORMANCE.csv')
    print('\n' + '=' * 170)
    print('TABLE — EXTERNAL PERFORMANCE')
    print('=' * 170)
    print(perf.to_string(index=False))
    print('\n' + '=' * 170)
    print('TABLE — LOCAL REFINEMENT VS GLOBAL HSCR')
    print('=' * 170)
    print(all_stats.to_string(index=False))
    print('\nFINAL AUDIT')
    print('  SHIN fitted                  : YES — source only')
    print('  Bálint fitted/tuned          : NO')
    print('  Khan fitted/tuned            : NO')
    print('  external target adaptation   : NO')
    print('  N4 config changed externally : NO')
    print('  real timing accuracy claim   : NO')
    print('  results                      :', out)
if __name__ == '__main__':
    main_external()
