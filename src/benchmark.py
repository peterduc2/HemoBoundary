"""Timing benchmarks evaluated against HemoBoundary predictions."""

from __future__ import annotations

import argparse

import hashlib

import json

import os

import shutil

from dataclasses import dataclass, field

from pathlib import Path

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np

import pandas as pd

from scipy import signal

from scipy.interpolate import UnivariateSpline

from scipy.stats import gamma

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

ALL_SEEDS = tuple(range(1, 11))

ALL_PIDS = tuple(f"VP{i:03d}" for i in range(1, 27))

GRID_DT = 0.5
GRID_END = 20.0

GRID = np.arange(0.0, GRID_END + GRID_DT / 2.0, GRID_DT)

MISS_PENALTY_SEC = 4.0

MIN_GAP = 0.5

BOOT_REPS = 20_000

SIGNFLIP_REPS = 20_000

R1 = "R1_THRESHOLD20"

R2 = "R2_THRESHOLD50"

R3 = "R3_FRACTIONAL_AREA"

R4 = "R4_FLEXIBLE_SPLINE"
R5 = "R5_CHANGE_POINT"

R6 = "R6_ORDERED_STATE"

R7 = "R7_CANONICAL_HRF_SHIFTED_GLM"

BENCHMARKS = (R1, R2, R3, R4, R5, R6, R7)

STAGE_ABC = "BASE_PLUS_A_PLUS_B_PLUS_C_RESPONSE"

STAGE_FULL = "BASE_PLUS_A_PLUS_B_PLUS_C_PLUS_LBR"

DISPLAY = {
    R1: "20% peak-threshold baseline",
    R2: "50% peak-threshold baseline",
    R3: "Fractional-area timing",
    R4: "Flexible spline",
    R5: "Change-point timing",
    R6: "Source-trained ordered-state comparator",
    R7: "Canonical-HRF shifted GLM",
    STAGE_ABC: "Base + A + B + C",
    STAGE_FULL: "HemoBoundary",
}
FAMILY = {
    R1: "Threshold heuristic",
    R2: "Threshold heuristic",
    R3: "Fractional-area heuristic",
    R4: "Spline / morphology",
    R5: "Change-point",
    R6: "Sequential / state",
    R7: "GLM / HRF",
    STAGE_ABC: "HemoBoundary ablation",
    STAGE_FULL: "HemoBoundary",
}

TIMING_METRICS = (
    "onset_mae_sec", "peak_mae_sec", "offset_mae_sec", "duration_mae_sec"
)

ALL_METRICS = ("response_balanced_accuracy", *TIMING_METRICS, "selection_loss")

COMPARISON_VERSION = "HEMOBOUNDARY_OPTIONB_TABLE2_10SEED_BENCHMARK_V1_20260916"

HIERARCHICAL_SEED1_DIRNAME = "98_B0_STRICT_HIERARCHICAL_SEED01_FULLY_NESTED_V1"

HIERARCHICAL_MULTI_DIRNAME = "99_B0_STRICT_HIERARCHICAL_10SEEDS_V1"
HIERARCHICAL_SEED1_BANK_PARENT = "95_N4_FUNCTIONAL_ABC_LBR_FULLY_NESTED_10SEEDS_V1"

COMPARISON_OUTPUT_DIRNAME = "101_OPTIONB_TABLE2_10SEED_BENCHMARK_V1"
def stable_seed(*parts: Any) -> int:
    raw = '|'.join((str(x) for x in parts)).encode('utf-8')
    return int(hashlib.sha256(raw).hexdigest()[:8], 16) % (2 ** 31 - 1)

def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    compression = 'gzip' if str(path).endswith('.gz') else None
    df.to_csv(tmp, index=False, compression=compression)
    os.replace(tmp, path)

def atomic_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)

def try_mount_drive() -> None:
    if Path('/content/drive/MyDrive').exists() or Path('/content/gdrive/MyDrive').exists():
        return
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
    except Exception:
        pass
def parse_seed_list(text: str) -> List[int]:
    if not text.strip():
        return list(ALL_SEEDS)
    vals = []
    for token in text.split(','):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value not in ALL_SEEDS:
            raise ValueError(f'Seed must be 1..10, got {value}')
        vals.append(value)
    vals = sorted(set(vals))
    if not vals:
        raise ValueError('No valid seeds requested.')
    return vals

def has_complete_cache(folder: Path) -> bool:
    return folder.is_dir() and all(((folder / f'{pid}.npz').is_file() for pid in ALL_PIDS))
def _split_from_npz(z, prefix: str) -> Dict[str, np.ndarray]:
    keys = ('present', 'onset', 'peak', 'offset', 'censored', 'states', 'obs', 'state_feat')
    missing = [f'{prefix}_{k}' for k in keys if f'{prefix}_{k}' not in z.files]
    if missing:
        raise RuntimeError('Cache missing required arrays: ' + ', '.join(missing))
    out = {k: np.asarray(z[f'{prefix}_{k}']) for k in keys}
    out['obs'] = np.nan_to_num(np.asarray(out['obs'], float), nan=0.0, posinf=0.0, neginf=0.0)
    out['state_feat'] = np.nan_to_num(np.asarray(out['state_feat'], float), nan=0.0, posinf=0.0, neginf=0.0)
    return out

def load_cache(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as z:
        return {'train': _split_from_npz(z, 'tr'), 'test': _split_from_npz(z, 'te')}

def load_all_seed_caches(cache_dir: Path) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    return {pid: load_cache(cache_dir / f'{pid}.npz') for pid in ALL_PIDS}

def concat_training(caches: Mapping[str, Dict[str, Dict[str, np.ndarray]]], pids: Sequence[str]) -> Dict[str, np.ndarray]:
    pids = list(pids)
    keys = tuple(caches[pids[0]]['train'].keys())
    return {k: np.concatenate([caches[pid]['train'][k] for pid in pids], axis=0) for k in keys}
def frozen_full_gate_rows(per_trial: pd.DataFrame, pid: str) -> pd.DataFrame:
    q = per_trial[(per_trial['participant'].astype(str) == str(pid)) & (per_trial['stage'].astype(str) == STAGE_FULL)].copy()
    q = q.sort_values('synthetic_index').reset_index(drop=True)
    if len(q) != 36:
        raise RuntimeError(f'{pid}: expected 36 saved HemoBoundary FULL test rows, found {len(q)}')
    if sorted(q['synthetic_index'].astype(int).tolist()) != list(range(36)):
        raise RuntimeError(f'{pid}: saved HemoBoundary synthetic_index is not exactly 0..35')
    return q
def truth_parity_check(cached: Mapping[str, np.ndarray], saved: pd.DataFrame, seed: int, pid: str) -> Dict[str, Any]:
    fields = (
        ("present", "truth_present"),
        ("onset", "truth_onset"),
        ("peak", "truth_peak"),
        ("offset", "truth_offset"),
        ("censored", "truth_censored"),
    )
    max_abs = 0.0
    exact_ok = True
    for cache_key, saved_key in fields:
        a = np.asarray(cached[cache_key])
        b = pd.to_numeric(saved[saved_key], errors='coerce').to_numpy()
        if len(a) != len(b):
            raise RuntimeError(f'Seed {seed:02d} {pid}: truth row-count mismatch.')
        if cache_key in {'present', 'censored'}:
            ok = np.array_equal(a.astype(int), b.astype(int))
            exact_ok &= bool(ok)
            if not ok:
                raise RuntimeError(f'Seed {seed:02d} {pid}: {cache_key} truth mismatch between cache and saved HemoBoundary result.')
        else:
            same_nan = np.isnan(a) == np.isnan(b)
            finite = np.isfinite(a) & np.isfinite(b)
            diff = np.abs(a[finite].astype(float) - b[finite].astype(float))
            local_max = float(np.max(diff)) if len(diff) else 0.0
            max_abs = max(max_abs, local_max)
            ok = bool(np.all(same_nan) and local_max <= 1e-10)
            exact_ok &= ok
            if not ok:
                raise RuntimeError(f'Seed {seed:02d} {pid}: {cache_key} truth mismatch (max abs diff {local_max}).')
    return {
        'repeat_seed': int(seed),
        'participant': pid,
        'truth_parity_pass': bool(exact_ok),
        'max_abs_timing_truth_difference': float(max_abs),
        'n_test_rows': int(len(saved))
    }
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

def smooth_anti(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    if len(x) >= 5:
        return signal.savgol_filter(x, window_length=5, polyorder=2, mode='interp')
    return x.copy()

def anti_from_data(data: Mapping[str, np.ndarray]) -> np.ndarray:
    F = np.asarray(data['state_feat'], float)
    if F.ndim != 3 or F.shape[1] != len(GRID) or F.shape[2] < 3:
        raise RuntimeError(f'Expected state_feat [trial,{len(GRID)},>=3], got {F.shape}')
    return np.asarray(F[:, :, 2], float)

def first_crossing(t: np.ndarray, y: np.ndarray, level: float, end_idx: int) -> float:
    q = np.flatnonzero(y[:end_idx + 1] >= level)
    return float(t[q[0]]) if len(q) else float(t[0])
def recovery_crossing(t: np.ndarray, y: np.ndarray, level: float, peak_idx: int) -> float:
    if peak_idx + 1 >= len(y):
        return float(t[-1])
    q = np.flatnonzero(y[peak_idx + 1:] <= level)
    if len(q):
        return float(t[peak_idx + 1 + q[0]])
    return float(t[-1])
def predict_threshold(data: Mapping[str, np.ndarray], frac: float) -> np.ndarray:
    A = anti_from_data(data)
    rows = []
    for raw in A:
        y = smooth_anti(raw)
        peak_idx = int(np.argmax(y))
        peak_value = float(y[peak_idx])
        floor = float(np.min(y[:max(1, peak_idx + 1)]))
        amplitude = max(peak_value - floor, 1e-08)
        level = floor + float(frac) * amplitude
        onset = first_crossing(GRID, y, level, peak_idx)
        peak = float(GRID[peak_idx])
        offset = recovery_crossing(GRID, y, level, peak_idx)
        rows.append((onset, peak, offset))
    out = np.asarray(rows, float)
    on, pk, off = enforce_geometry(out[:, 0], out[:, 1], out[:, 2])
    return np.column_stack([on, pk, off])
def predict_fractional_area(data: Mapping[str, np.ndarray]) -> np.ndarray:
    A = anti_from_data(data)
    rows = []
    for raw in A:
        y = smooth_anti(raw)
        baseline = float(np.median(y[:3]))
        z = np.clip(y - baseline, 0.0, None)
        if np.sum(z) <= 1e-12:
            z = np.abs(y - baseline)
        if np.sum(z) <= 1e-12:
            rows.append((0.0, GRID_DT, 2 * GRID_DT))
            continue
        c = np.cumsum(z)
        c = c / c[-1]

        def qtime(q):
            idx = int(np.searchsorted(c, q, side='left'))
            idx = min(max(idx, 0), len(GRID) - 1)
            return float(GRID[idx])
        rows.append((qtime(0.1), qtime(0.5), qtime(0.9)))
    out = np.asarray(rows, float)
    on, pk, off = enforce_geometry(out[:, 0], out[:, 1], out[:, 2])
    return np.column_stack([on, pk, off])
def predict_flexible_spline(data: Mapping[str, np.ndarray]) -> np.ndarray:
    A = anti_from_data(data)
    dense_t = np.arange(0.0, GRID_END + 0.025, 0.05)
    rows = []
    for raw in A:
        y = np.asarray(raw, float)
        rough = float(np.median(np.abs(np.diff(y, n=2)))) if len(y) >= 3 else 0.0
        s = max(1e-06, len(y) * max(rough, 0.05) ** 2)
        try:
            spline = UnivariateSpline(GRID, y, k=3, s=s)
            yd = np.asarray(spline(dense_t), float)
        except Exception:
            yd = np.interp(dense_t, GRID, smooth_anti(y))
        peak_idx = int(np.argmax(yd))
        peak_value = float(yd[peak_idx])
        floor = float(np.min(yd[:max(1, peak_idx + 1)]))
        amplitude = max(peak_value - floor, 1e-08)
        level = floor + 0.2 * amplitude
        onset = first_crossing(dense_t, yd, level, peak_idx)
        peak = float(dense_t[peak_idx])
        offset = recovery_crossing(dense_t, yd, level, peak_idx)
        rows.append((onset, peak, offset))
    out = np.asarray(rows, float)
    on, pk, off = enforce_geometry(out[:, 0], out[:, 1], out[:, 2])
    return np.column_stack([on, pk, off])
def segment_sse(x: np.ndarray) -> float:
    if len(x) == 0:
        return np.inf
    m = float(np.mean(x))
    return float(np.sum((x - m) ** 2))
def predict_change_point(data: Mapping[str, np.ndarray]) -> np.ndarray:
    A = anti_from_data(data)
    rows = []
    for raw in A:
        y = smooth_anti(raw)
        n = len(y)
        best_score = np.inf
        best_i = 1
        best_j = n - 2
        for i in range(2, n - 4):
            for j in range(i + 2, n - 2):
                score = segment_sse(y[:i]) + segment_sse(y[i:j]) + segment_sse(y[j:])
                middle_gain = float(np.mean(y[i:j]) - np.mean(y[:i]))
                score -= 1e-06 * middle_gain
                if score < best_score:
                    best_score = score
                    best_i = i
                    best_j = j
        onset = float(GRID[best_i])
        mid = y[best_i:best_j + 1]
        if len(mid):
            peak_idx = int(best_i + np.argmax(mid))
        else:
            peak_idx = int(np.argmax(y))
        peak = float(GRID[peak_idx])
        offset = float(GRID[best_j])
        rows.append((onset, peak, offset))
    out = np.asarray(rows, float)
    on, pk, off = enforce_geometry(out[:, 0], out[:, 1], out[:, 2])
    return np.column_stack([on, pk, off])
@dataclass
class OrderedStateModel:
    scaler: StandardScaler
    clf: LogisticRegression

def fit_ordered_state_model(train: Mapping[str, np.ndarray], seed: int) -> OrderedStateModel:
    F = np.asarray(train['state_feat'], float)
    states = np.asarray(train['states'], int)
    X = F.reshape(-1, F.shape[-1])
    y = states.reshape(-1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if len(X) > 60000:
        rng = np.random.default_rng(stable_seed('R6_SOURCE_FRAME_SUBSAMPLE', seed, len(X)))
        idx = rng.choice(len(X), 60000, replace=False)
        X = X[idx]
        y = y[idx]
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.01, class_weight='balanced', max_iter=800, solver='lbfgs', random_state=20260915 + int(seed))
    clf.fit(scaler.transform(X), y)
    return OrderedStateModel(scaler=scaler, clf=clf)
def ordered_state_probabilities(model: OrderedStateModel, data: Mapping[str, np.ndarray]) -> np.ndarray:
    F = np.asarray(data['state_feat'], float)
    n, t, d = F.shape
    X = np.nan_to_num(F.reshape(-1, d), nan=0.0, posinf=0.0, neginf=0.0)
    p0 = model.clf.predict_proba(model.scaler.transform(X))
    P = np.full((len(X), 5), 1e-08, float)
    for j, cls in enumerate(model.clf.classes_):
        P[:, int(cls)] = p0[:, j]
    P /= np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    return P.reshape(n, t, 5)
def ordered_path(P: np.ndarray) -> np.ndarray:
    n, t, s = P.shape
    logp = np.log(np.clip(P, 1e-09, 1.0))
    dp = np.full((n, s), -np.inf)
    dp[:, 0] = logp[:, 0, 0]
    back = np.zeros((n, t, s), np.int8)
    for k in range(1, t):
        nxt = np.full_like(dp, -np.inf)
        for state in range(s):
            stay = dp[:, state]
            advance = dp[:, state - 1] if state > 0 else np.full(n, -np.inf)
            keep = stay >= advance
            nxt[:, state] = np.where(keep, stay, advance) + logp[:, k, state]
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
def predict_ordered_state(model: OrderedStateModel, data: Mapping[str, np.ndarray]) -> np.ndarray:
    P = ordered_state_probabilities(model, data)
    path = ordered_path(P)
    rows = []
    for i in range(len(path)):
        seq = path[i]
        s1 = np.flatnonzero(seq == 1)
        s2 = np.flatnonzero(seq == 2)
        s4 = np.flatnonzero(seq == 4)
        onset_idx = int(s1[0]) if len(s1) else int(np.argmax(P[i, :, 1]))
        if len(s2):
            peak_idx = int(s2[np.argmax(P[i, s2, 2])])
        else:
            peak_idx = int(np.argmax(P[i, :, 2]))
        offset_idx = int(s4[0]) if len(s4) else len(GRID) - 1
        rows.append((float(GRID[onset_idx]), float(GRID[peak_idx]), float(GRID[offset_idx])))
    out = np.asarray(rows, float)
    on, pk, off = enforce_geometry(out[:, 0], out[:, 1], out[:, 2])
    return np.column_stack([on, pk, off])
def canonical_hrf(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float)
    x = np.clip(t, 0.0, None)
    h = gamma.pdf(x, 6.0) - gamma.pdf(x, 16.0) / 6.0
    h[t < 0] = 0.0
    mx = float(np.max(h))
    if mx > 0:
        h = h / mx
    return h

def canonical_landmarks() -> Tuple[float, float, float]:
    dense = np.arange(0.0, 30.0 + 0.005, 0.01)
    h = canonical_hrf(dense)
    pk = int(np.argmax(h))
    level = 0.2 * float(h[pk])
    pre = np.flatnonzero(h[:pk + 1] >= level)
    post = np.flatnonzero(h[pk + 1:] <= level)
    onset = float(dense[pre[0]]) if len(pre) else 0.0
    peak = float(dense[pk])
    offset = float(dense[pk + 1 + post[0]]) if len(post) else float(dense[-1])
    return (onset, peak, offset)

CANONICAL_LANDMARKS = canonical_landmarks()
def predict_canonical_hrf_shifted_glm(data: Mapping[str, np.ndarray]) -> np.ndarray:
    A = anti_from_data(data)
    shift_grid = np.arange(0.0, 6.0 + GRID_DT / 2.0, GRID_DT)
    trend = (GRID - np.mean(GRID)) / max(np.std(GRID), 1e-12)
    rows = []
    base_on, base_pk, base_off = CANONICAL_LANDMARKS
    for raw in A:
        y = np.asarray(raw, float)
        best = None
        for shift in shift_grid:
            h = canonical_hrf(GRID - float(shift))
            X = np.column_stack([h, np.ones(len(GRID)), trend])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            fit = X @ beta
            sse = float(np.sum((y - fit) ** 2))
            amp = float(beta[0])
            if amp < 0:
                sse += float(np.sum((amp * h) ** 2))
            candidate = (sse, float(shift), amp)
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        shift = best[1]
        rows.append((shift + base_on, shift + base_pk, shift + base_off))
    out = np.asarray(rows, float)
    on, pk, off = enforce_geometry(out[:, 0], out[:, 1], out[:, 2])
    return np.column_stack([on, pk, off])
def benchmark_prediction_frame(raw_times: np.ndarray, frozen_response: np.ndarray) -> pd.DataFrame:
    raw_times = np.asarray(raw_times, float)
    response = np.asarray(frozen_response, int)
    if raw_times.shape != (len(response), 3):
        raise RuntimeError(f'Timing shape mismatch: {raw_times.shape} vs n={len(response)}')
    on, pk, off = enforce_geometry(raw_times[:, 0], raw_times[:, 1], raw_times[:, 2])
    miss = response != 1
    on = on.astype(float)
    pk = pk.astype(float)
    off = off.astype(float)
    on[miss] = np.nan
    pk[miss] = np.nan
    off[miss] = np.nan
    return pd.DataFrame({'pred_response': response.astype(int), 'pred_onset': on, 'pred_peak': pk, 'pred_offset': off})

def balanced_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, int)
    p = np.asarray(p, int)
    pos = y == 1
    neg = y == 0
    sens = np.mean(p[pos] == 1) if np.any(pos) else np.nan
    spec = np.mean(p[neg] == 0) if np.any(neg) else np.nan
    return float(np.nanmean([sens, spec]))
def trial_metric_summary(truth: Mapping[str, np.ndarray], pred: pd.DataFrame) -> Dict[str, float]:
    present = np.asarray(truth['present'], int) == 1
    cens = np.asarray(truth['censored'], int) == 1
    decision = pred['pred_response'].to_numpy(int)
    out: Dict[str, float] = {'response_balanced_accuracy': balanced_accuracy(np.asarray(truth['present'], int), decision)}
    for name in ('onset', 'peak', 'offset'):
        t = np.asarray(truth[name], float)
        p = pred[f'pred_{name}'].to_numpy(float)
        valid_truth = present & np.isfinite(t)
        if name == 'offset':
            valid_truth &= ~cens
        idx = np.flatnonzero(valid_truth)
        err = np.full(len(idx), MISS_PENALTY_SEC, float)
        finite = np.isfinite(p[idx]) & (decision[idx] == 1)
        err[finite] = np.abs(p[idx][finite] - t[idx][finite])
        out[f'{name}_mae_sec'] = float(np.mean(err)) if len(err) else np.nan
    ton = np.asarray(truth['onset'], float)
    toff = np.asarray(truth['offset'], float)
    pon = pred['pred_onset'].to_numpy(float)
    poff = pred['pred_offset'].to_numpy(float)
    valid = present & ~cens & np.isfinite(ton) & np.isfinite(toff)
    idx = np.flatnonzero(valid)
    err = np.full(len(idx), MISS_PENALTY_SEC, float)
    finite = np.isfinite(pon[idx]) & np.isfinite(poff[idx]) & (decision[idx] == 1)
    err[finite] = np.abs(poff[idx][finite] - pon[idx][finite] - (toff[idx][finite] - ton[idx][finite]))
    out['duration_mae_sec'] = float(np.mean(err)) if len(err) else np.nan
    vals = np.asarray([out['onset_mae_sec'], out['peak_mae_sec'], out['offset_mae_sec'], out['duration_mae_sec']], float)
    scales = np.asarray([1.0, 1.0, 1.5, 1.5], float)
    out['selection_loss'] = float(np.nanmean(vals / scales))
    return out
def reference_predictions(seed: int, pid: str, caches: Mapping, truth: Mapping) -> Dict[str, np.ndarray]:
    predictions = {
        R1: predict_threshold(truth, 0.20),
        R2: predict_threshold(truth, 0.50),
        R3: predict_fractional_area(truth),
        R4: predict_flexible_spline(truth),
        R5: predict_change_point(truth),
        R7: predict_canonical_hrf_shifted_glm(truth),
    }
    source = concat_training(caches, [p for p in ALL_PIDS if p != pid])
    state_model = fit_ordered_state_model(source, stable_seed("R6", seed, pid))
    predictions[R6] = predict_ordered_state(state_model, truth)
    return predictions
def trial_prediction_rows(seed: int, pid: str, method: str, truth: Mapping, pred: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "repeat_seed": int(seed),
        "participant": pid,
        "synthetic_index": np.arange(len(pred), dtype=int),
        "method": method,
        "display_name": DISPLAY[method],
        "family": FAMILY[method],
        "truth_present": np.asarray(truth["present"], int),
        "truth_onset": np.asarray(truth["onset"], float),
        "truth_peak": np.asarray(truth["peak"], float),
        "truth_offset": np.asarray(truth["offset"], float),
        "truth_censored": np.asarray(truth["censored"], int),
        "pred_response": pred["pred_response"].to_numpy(int),
        "pred_onset": pred["pred_onset"].to_numpy(float),
        "pred_peak": pred["pred_peak"].to_numpy(float),
        "pred_offset": pred["pred_offset"].to_numpy(float),
        "response_gate_source": STAGE_FULL,
    })

def benchmark_metric_row(seed: int, pid: str, method: str, truth: Mapping, pred: pd.DataFrame) -> dict:
    return {
        "repeat_seed": int(seed),
        "participant": pid,
        "method": method,
        "display_name": DISPLAY[method],
        "family": FAMILY[method],
        **trial_metric_summary(truth, pred),
    }
def run_benchmarks_for_participant(seed: int, pid: str, caches: Mapping,
                                   saved_full_rows: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    truth = caches[pid]["test"]
    response = pd.to_numeric(saved_full_rows["pred_response"], errors="raise").to_numpy(int)
    if not np.isin(response, [0, 1]).all():
        raise RuntimeError(f"Seed {seed:02d} {pid}: HemoBoundary response decisions are not binary.")
    predictions = reference_predictions(seed, pid, caches, truth)
    trial_parts, metric_rows = [], []
    for method in BENCHMARKS:
        pred = benchmark_prediction_frame(predictions[method], response)
        metric_rows.append(benchmark_metric_row(seed, pid, method, truth, pred))
        trial_parts.append(trial_prediction_rows(seed, pid, method, truth, pred))
    return pd.concat(trial_parts, ignore_index=True), pd.DataFrame(metric_rows)
def bootstrap_ci(x: Sequence[float], seed: int) -> Tuple[float, float, float]:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(a))
    if len(a) == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    values = np.empty(BOOT_REPS, float)
    done = 0
    while done < BOOT_REPS:
        n = min(2000, BOOT_REPS - done)
        ix = rng.integers(0, len(a), size=(n, len(a)))
        values[done:done + n] = a[ix].mean(axis=1)
        done += n
    lo, hi = np.quantile(values, [0.025, 0.975])
    return (mean, float(lo), float(hi))
def signflip_p(x: Sequence[float], seed: int) -> float:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return np.nan
    obs = abs(float(np.mean(a)))
    if obs < 1e-15:
        return 1.0
    rng = np.random.default_rng(seed)
    hits = 0
    done = 0
    while done < SIGNFLIP_REPS:
        n = min(2000, SIGNFLIP_REPS - done)
        signs = rng.choice([-1.0, 1.0], size=(n, len(a)))
        stat = np.abs(np.mean(signs * a[None, :], axis=1))
        hits += int(np.sum(stat >= obs - 1e-15))
        done += n
    return float((hits + 1) / (SIGNFLIP_REPS + 1))
def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan, float)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return out
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        out[idx] = min(1.0, running)
    return out

def paired_timing_gains(benchmark_pm: pd.DataFrame, existing_full: pd.DataFrame,
                        method: str, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    a = benchmark_pm.loc[benchmark_pm["method"] == method,
                         ["repeat_seed", "participant", metric]].rename(columns={metric: "benchmark"})
    b = existing_full[["repeat_seed", "participant", metric]].rename(columns={metric: "hemoboundary"})
    paired = a.merge(b, on=["repeat_seed", "participant"], how="inner", validate="one_to_one")
    if len(paired) != len(ALL_SEEDS) * len(ALL_PIDS):
        raise RuntimeError(f"{method}/{metric}: expected 260 paired seed-participant rows, found {len(paired)}.")
    paired["gain"] = paired["benchmark"].to_numpy(float) - paired["hemoboundary"].to_numpy(float)
    by_participant = paired.groupby("participant", sort=True)["gain"].mean().reindex(ALL_PIDS)
    if by_participant.isna().any():
        raise RuntimeError(f"{method}/{metric}: participant-average gain incomplete.")
    by_seed = paired.groupby("repeat_seed", sort=True)["gain"].mean()
    return by_participant.to_numpy(float), by_seed.to_numpy(float)
def timing_stat_row(method: str, metric: str, gains: np.ndarray, seed_means: np.ndarray) -> dict:
    mean, lo, hi = bootstrap_ci(gains, stable_seed("TABLE2_BOOT", method, metric))
    p = signflip_p(gains, stable_seed("TABLE2_SIGN", method, metric))
    return {
        "candidate": STAGE_FULL,
        "candidate_display": DISPLAY[STAGE_FULL],
        "reference": method,
        "reference_display": DISPLAY[method],
        "metric": metric,
        "n_seeds": len(ALL_SEEDS),
        "n_participants": len(ALL_PIDS),
        "mean_gain_sec_positive_HemoBoundary_better": mean,
        "ci95_low": lo,
        "ci95_high": hi,
        "participants_mean_gain_better": int(np.sum(gains > 1e-12)),
        "participants_mean_gain_equal": int(np.sum(np.isclose(gains, 0.0))),
        "participants_mean_gain_worse": int(np.sum(gains < -1e-12)),
        "positive_mean_gain_seeds": int(np.sum(seed_means > 1e-12)),
        "zero_mean_gain_seeds": int(np.sum(np.isclose(seed_means, 0.0))),
        "negative_mean_gain_seeds": int(np.sum(seed_means < -1e-12)),
        "signflip_p": p,
    }
def participant_averaged_stats(benchmark_pm: pd.DataFrame, existing_pm: pd.DataFrame) -> pd.DataFrame:
    full = existing_pm[existing_pm["method"] == STAGE_FULL].copy()
    rows = []
    for method in BENCHMARKS:
        for metric in TIMING_METRICS:
            gains, seed_means = paired_timing_gains(benchmark_pm, full, method, metric)
            rows.append(timing_stat_row(method, metric, gains, seed_means))
    out = pd.DataFrame(rows)
    out["holm_p_within_benchmark_4_timing"] = np.nan
    for method in BENCHMARKS:
        ix = out.index[out["reference"] == method].tolist()
        out.loc[ix, "holm_p_within_benchmark_4_timing"] = holm_adjust(
            out.loc[ix, "signflip_p"].to_numpy(float))
    out["holm_p_global_28_timing_tests"] = holm_adjust(out["signflip_p"].to_numpy(float))
    return out
def seed_method_performance(all_pm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = [*BENCHMARKS, STAGE_ABC, STAGE_FULL]
    for seed in ALL_SEEDS:
        qseed = all_pm[all_pm['repeat_seed'] == seed]
        for method in order:
            q = qseed[qseed['method'] == method]
            if q['participant'].nunique() != 26:
                raise RuntimeError(f'Seed {seed:02d}/{method}: expected 26 participant rows.')
            row = {
                "repeat_seed": int(seed),
                "method": method,
                "display_name": DISPLAY[method],
                "family": FAMILY[method],
                "n_participants": 26,
            }
            for metric in ALL_METRICS:
                x = pd.to_numeric(q[metric], errors='coerce').to_numpy(float)
                row[metric] = float(np.nanmean(x))
            rows.append(row)
    return pd.DataFrame(rows)
def across_seed_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    order = [*BENCHMARKS, STAGE_ABC, STAGE_FULL]
    rows = []
    for method in order:
        q = per_seed[per_seed['method'] == method]
        if q['repeat_seed'].nunique() != 10:
            raise RuntimeError(f'{method}: expected 10 seed-level performance rows.')
        row = {'method': method, 'display_name': DISPLAY[method], 'family': FAMILY[method], 'n_seeds': 10, 'n_participants_per_seed': 26}
        for metric in ALL_METRICS:
            x = pd.to_numeric(q[metric], errors='coerce').to_numpy(float)
            row[f'{metric}_mean'] = float(np.nanmean(x))
            row[f'{metric}_sd'] = float(np.nanstd(x, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)

def paper_ready_table(summary: pd.DataFrame) -> pd.DataFrame:
    q = summary.copy()
    out = q[['method', 'display_name', 'family', 'n_seeds', 'n_participants_per_seed']].copy()
    for metric in ('response_balanced_accuracy', 'onset_mae_sec', 'peak_mae_sec', 'offset_mae_sec', 'duration_mae_sec'):
        mean = q[f'{metric}_mean'].to_numpy(float)
        sd = q[f'{metric}_sd'].to_numpy(float)
        decimals = 4 if metric == 'response_balanced_accuracy' else 4
        out[metric] = [f'{m:.{decimals}f} ± {s:.{decimals}f}' for m, s in zip(mean, sd)]
    out["response_gate_note"] = [
        "Frozen HemoBoundary response gate" if method in BENCHMARKS
        else "Own saved model response gate"
        for method in q["method"]
    ]
    return out
def selftest_waveforms() -> tuple[dict, np.random.Generator]:
    rng = np.random.default_rng(123)
    n, t = 12, len(GRID)
    curves = []
    for i in range(n):
        onset, peak, offset = 1.0 + 0.05*i, 5.0 + 0.03*i, 12.0 + 0.04*i
        y = np.zeros(t)
        for k, tt in enumerate(GRID):
            if tt < onset:
                y[k] = 0
            elif tt < peak:
                u = (tt - onset) / max(peak - onset, 1e-6)
                y[k] = np.sin(0.5 * np.pi * u) ** 2
            elif tt < offset:
                u = (tt - peak) / max(offset - peak, 1e-6)
                y[k] = np.cos(0.5 * np.pi * u) ** 2
            else:
                y[k] = 0
        y += rng.normal(scale=0.02, size=t)
        curves.append(y)
    state_feat = np.zeros((n, t, 9), float)
    state_feat[:, :, 2] = np.asarray(curves)
    return {"state_feat": state_feat}, rng
def selftest_references(data: Mapping) -> None:
    n = len(data["state_feat"])
    predictions = [
        (R1, predict_threshold(data, 0.2)),
        (R2, predict_threshold(data, 0.5)),
        (R3, predict_fractional_area(data)),
        (R4, predict_flexible_spline(data)),
        (R5, predict_change_point(data)),
        (R7, predict_canonical_hrf_shifted_glm(data)),
    ]
    for method, pred in predictions:
        assert pred.shape == (n, 3), method
        assert np.isfinite(pred).all(), method
        assert np.all(pred[:, 0] < pred[:, 1]), method
        assert np.all(pred[:, 1] < pred[:, 2]), method
def selftest_ordered_state(rng: np.random.Generator) -> None:
    n, t = 60, len(GRID)
    features = np.zeros((n, t, 9), float)
    states = np.zeros((n, t), int)
    for i in range(n):
        b1, b2, b3, b4 = 4+i%3, 11+i%2, 20+i%3, 31+i%2
        seq = np.zeros(t, int)
        seq[b1:b2], seq[b2:b3], seq[b3:b4], seq[b4:] = 1, 2, 3, 4
        states[i] = seq
        for s in range(5):
            features[i, seq == s, s % 9] = 2.0
        features[i] += rng.normal(scale=0.2, size=(t, 9))
    model = fit_ordered_state_model({"state_feat": features, "states": states}, seed=123)
    pred = predict_ordered_state(model, {"state_feat": features[:5].copy()})
    assert pred.shape == (5, 3)
    assert np.isfinite(pred).all()
    assert np.all(pred[:, 0] < pred[:, 1])
    assert np.all(pred[:, 1] < pred[:, 2])
def selftest_metrics() -> None:
    truth = {
        "present": np.asarray([1, 1, 0, 0]),
        "onset": np.asarray([1.0, 2.0, np.nan, np.nan]),
        "peak": np.asarray([4.0, 5.0, np.nan, np.nan]),
        "offset": np.asarray([10.0, 11.0, np.nan, np.nan]),
        "censored": np.asarray([0, 0, 0, 0]),
    }
    pred_frame = pd.DataFrame({
        "pred_response": [1, 1, 0, 0],
        "pred_onset": [1.1, 2.1, np.nan, np.nan],
        "pred_peak": [4.1, 5.1, np.nan, np.nan],
        "pred_offset": [10.1, 11.1, np.nan, np.nan],
    })
    metrics = trial_metric_summary(truth, pred_frame)
    assert metrics["response_balanced_accuracy"] == 1.0
    assert np.isfinite(metrics["selection_loss"])

def self_tests() -> None:
    data, rng = selftest_waveforms()
    selftest_references(data)
    selftest_ordered_state(rng)
    selftest_metrics()
    print("Benchmark checks: PASS")
def discover_comparison_shin_root(explicit: Optional[str]=None) -> Path:
    try_mount_drive()
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([
        Path("/content/drive/MyDrive/HonorProject/shin2018_word_generation_state_preparation_v1"),
        Path("/content/gdrive/MyDrive/HonorProject/shin2018_word_generation_state_preparation_v1"),
    ])
    for p in candidates:
        if p.is_dir():
            return p
    raise FileNotFoundError('Could not locate SHIN project root. Use --shin-root PATH.')

def comparison_seed_result_dir(shin_root: Path, seed: int) -> Path:
    seed = int(seed)
    if seed == 1:
        return shin_root / HIERARCHICAL_SEED1_DIRNAME
    return shin_root / HIERARCHICAL_MULTI_DIRNAME / f'seed_{seed:02d}'
def comparison_seed_cache_dir(shin_root: Path, seed: int) -> Path:
    """
    Seed 1 initial evaluation deliberately reused the already-created seed-1 bank
    from the original 10-seed N4 experiment. Seeds 2-10 created their own
    fresh seed-specific banks under later evaluation.
    """
    seed = int(seed)
    if seed == 1:
        p = shin_root / HIERARCHICAL_SEED1_BANK_PARENT / 'seed_01' / 'fresh_cache'
    else:
        p = shin_root / HIERARCHICAL_MULTI_DIRNAME / f'seed_{seed:02d}' / 'fresh_cache'
    if not has_complete_cache(p):
        raise FileNotFoundError(f'Seed {seed:02d}: complete 26-participant cache not found: {p}')
    return p

def comparison_checkpoint_bundle(shin_root: Path, seed: int, pid: str) -> Dict[str, Path]:
    ck = comparison_seed_result_dir(shin_root, seed) / 'checkpoints'
    return {
        "rows": ck / f"{pid}_ROWS.csv.gz",
        "selected": ck / f"{pid}_SELECTED.json",
        "inner": ck / f"{pid}_INNER.csv.gz",
        "audit": ck / f"{pid}_AUDIT.csv.gz",
    }
def comparison_seed_completion(shin_root: Path, seed: int) -> None:
    missing = []
    for pid in ALL_PIDS:
        bundle = comparison_checkpoint_bundle(shin_root, seed, pid)
        bad = [key for key, path in bundle.items() if not path.is_file()]
        if bad:
            missing.append((pid, bad))
    if missing:
        details = "\n".join(f"  {pid}: missing {bad}" for pid, bad in missing)
        raise RuntimeError(f"Seed {seed:02d}: benchmark input checkpoints are incomplete:\n{details}")
def comparison_read_seed_per_trial(shin_root: Path, seed: int) -> pd.DataFrame:
    """
    Combine saved participant checkpoints.

    This is read-only and uses no fitting.
    """
    comparison_seed_completion(shin_root, seed)
    parts = []
    for pid in ALL_PIDS:
        path = comparison_checkpoint_bundle(shin_root, seed, pid)['rows']
        q = pd.read_csv(path)
        q['participant'] = q['participant'].astype(str)
        if set(q['participant'].unique()) != {pid}:
            raise RuntimeError(f'Seed {seed:02d} {pid}: participant checkpoint ID mismatch.')
        if 'stage' not in q.columns:
            if 'row' not in q.columns:
                raise RuntimeError(f'Seed {seed:02d} {pid}: checkpoint has neither row nor stage.')
            q['stage'] = q['row'].astype(str)
        parts.append(q)
    out = pd.concat(parts, ignore_index=True)
    expected = {(pid, stage) for pid in ALL_PIDS for stage in (STAGE_ABC, STAGE_FULL)}
    actual = set(zip(out['participant'].astype(str), out['stage'].astype(str)))
    if not expected.issubset(actual):
        raise RuntimeError(f'Seed {seed:02d}: A+B+C and full-model rows are incomplete.')
    return out
def comparison_existing_participant_metrics(saved_per_trial: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Recompute A+B+C and HemoBoundary metrics from the saved
    predictions using the SAME metric implementation used by the benchmark.
    """
    rows = []
    for pid in ALL_PIDS:
        for stage in (STAGE_ABC, STAGE_FULL):
            q = saved_per_trial[
                (saved_per_trial["participant"].astype(str) == pid)
                & (saved_per_trial["stage"].astype(str) == stage)
            ].sort_values("synthetic_index").reset_index(drop=True)
            if len(q) != 36:
                raise RuntimeError(f'Seed {seed:02d} {pid}/{stage}: expected 36 rows, got {len(q)}.')
            truth = {
                'present': q['truth_present'].to_numpy(int),
                'onset': q['truth_onset'].to_numpy(float),
                'peak': q['truth_peak'].to_numpy(float),
                'offset': q['truth_offset'].to_numpy(float),
                'censored': q['truth_censored'].to_numpy(int)
            }
            pred = pd.DataFrame({
                'pred_response': q['pred_response'].to_numpy(int),
                'pred_onset': q['pred_onset'].to_numpy(float),
                'pred_peak': q['pred_peak'].to_numpy(float),
                'pred_offset': q['pred_offset'].to_numpy(float)
            })
            m = trial_metric_summary(truth, pred)
            rows.append({
                "repeat_seed": int(seed),
                "participant": pid,
                "method": stage,
                "display_name": DISPLAY[stage],
                "family": FAMILY[stage],
                **m,
            })
    out = pd.DataFrame(rows)
    if len(out) != 26 * 2:
        raise RuntimeError(f'Seed {seed:02d}: expected 52 model metric rows.')
    return out
def comparison_timing_only_paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    The seven references are TIMING comparators sharing HemoBoundary's frozen
    response gate. BA therefore should not be displayed as independent
    benchmark performance.
    """
    rows = []
    order = [*BENCHMARKS, STAGE_ABC, STAGE_FULL]
    for method in order:
        q = summary[summary['method'] == method]
        if len(q) != 1:
            raise RuntimeError(f'{method}: summary row missing.')
        q = q.iloc[0]
        row = {
            'method': method,
            'display_name': q['display_name'],
            'family': q['family'],
            'n_seeds': int(q['n_seeds']),
            'n_participants_per_seed': int(q['n_participants_per_seed'])
        }
        for metric in TIMING_METRICS:
            row[metric] = f"{float(q[f'{metric}_mean']):.4f} ± {float(q[f'{metric}_sd']):.4f}"
        row['response_gate_note'] = 'Frozen final HemoBoundary response gate' if method in BENCHMARKS else 'Own saved model response gate'
        rows.append(row)
    return pd.DataFrame(rows)
def comparison_parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--shin-root', default=None)
    p.add_argument('--seeds', default='', help='Comma-separated subset of 1..10. Empty = all seeds.')
    p.add_argument('--force', action='store_true', help='Rerun benchmark checkpoints for requested seeds only.')
    p.add_argument('--self-test-only', action='store_true')
    args, unknown = p.parse_known_args()
    if unknown:
        print('Ignoring notebook/runtime arguments:', unknown)
    return args

@dataclass
class BenchmarkRun:
    shin_root: Path
    out: Path
    seeds: List[int]
    force: bool
    trials: List[pd.DataFrame] = field(default_factory=list)
    metrics: List[pd.DataFrame] = field(default_factory=list)
    parity: List[dict] = field(default_factory=list)
def prepare_benchmark_run(args) -> BenchmarkRun:
    shin_root = discover_comparison_shin_root(args.shin_root)
    seeds = parse_seed_list(args.seeds)
    out = shin_root / COMPARISON_OUTPUT_DIRNAME
    for name in ("results", "tables", "audit", "checkpoints"):
        (out / name).mkdir(parents=True, exist_ok=True)
    if args.force:
        for seed in seeds:
            folder = out / "checkpoints" / f"seed_{seed:02d}"
            if folder.exists():
                shutil.rmtree(folder)
    return BenchmarkRun(shin_root, out, seeds, args.force)
def benchmark_participant(run: BenchmarkRun, seed: int, pid: str, ordinal: int,
                          caches: Mapping, saved: pd.DataFrame) -> None:
    ckdir = run.out / "checkpoints" / f"seed_{seed:02d}"
    ckdir.mkdir(parents=True, exist_ok=True)
    checkpoint, metrics_path = ckdir / f"{pid}.csv.gz", ckdir / f"{pid}_metrics.csv"
    gate = frozen_full_gate_rows(saved, pid)
    run.parity.append(truth_parity_check(caches[pid]["test"], gate, seed, pid))
    if checkpoint.is_file() and metrics_path.is_file() and not run.force:
        print(f"  [{ordinal:02d}/26] {pid}: RESUME", flush=True)
        trial, metrics = pd.read_csv(checkpoint), pd.read_csv(metrics_path)
    else:
        print(f"  [{ordinal:02d}/26] {pid}: R1-R7", flush=True)
        trial, metrics = run_benchmarks_for_participant(seed, pid, caches, gate)
        atomic_csv(trial, checkpoint)
        atomic_csv(metrics, metrics_path)
    run.trials.append(trial)
    run.metrics.append(metrics)

def benchmark_seed(run: BenchmarkRun, seed: int) -> None:
    comparison_seed_completion(run.shin_root, seed)
    cache_dir = comparison_seed_cache_dir(run.shin_root, seed)
    caches = load_all_seed_caches(cache_dir)
    saved = comparison_read_seed_per_trial(run.shin_root, seed)
    comparison_existing_participant_metrics(saved, seed)
    print(f"SEED {seed:02d} | Cache: {cache_dir}")
    for ordinal, pid in enumerate(ALL_PIDS, 1):
        benchmark_participant(run, seed, pid, ordinal, caches, saved)
def collect_saved_checkpoints(run: BenchmarkRun) -> None:
    done = {
        (int(q["repeat_seed"].iloc[0]), str(q["participant"].iloc[0]))
        for q in run.metrics if len(q)
    }
    for seed in ALL_SEEDS:
        folder = run.out / "checkpoints" / f"seed_{seed:02d}"
        for pid in ALL_PIDS:
            if (seed, pid) in done:
                continue
            checkpoint, metrics_path = folder / f"{pid}.csv.gz", folder / f"{pid}_metrics.csv"
            if checkpoint.is_file() and metrics_path.is_file():
                run.trials.append(pd.read_csv(checkpoint))
                run.metrics.append(pd.read_csv(metrics_path))
                done.add((seed, pid))

def load_all_saved_hierarchical_metrics(shin_root: Path) -> pd.DataFrame:
    parts = []
    for seed in ALL_SEEDS:
        try:
            comparison_seed_completion(shin_root, seed)
            saved = comparison_read_seed_per_trial(shin_root, seed)
            parts.append(comparison_existing_participant_metrics(saved, seed))
        except Exception:
            continue
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
def collect_current_results(run: BenchmarkRun):
    benchmark_pm = pd.concat(run.metrics, ignore_index=True) if run.metrics else pd.DataFrame()
    benchmark_trial = pd.concat(run.trials, ignore_index=True) if run.trials else pd.DataFrame()
    return benchmark_pm, benchmark_trial, load_all_saved_hierarchical_metrics(run.shin_root)

def save_available_results(run: BenchmarkRun, benchmark_pm: pd.DataFrame,
                           benchmark_trial: pd.DataFrame) -> None:
    if len(benchmark_trial):
        atomic_csv(benchmark_trial, run.out / "results" / "TABLE2_OPTIONB_BENCHMARK_PER_TRIAL_ALL_SEEDS.csv.gz")
    if len(benchmark_pm):
        atomic_csv(benchmark_pm, run.out / "results" / "TABLE2_OPTIONB_BENCHMARK_PARTICIPANT_METRICS_ALL_SEEDS.csv.gz")
    if run.parity:
        parity = pd.DataFrame(run.parity).drop_duplicates(["repeat_seed", "participant"], keep="last")
        atomic_csv(parity, run.out / "audit" / "CACHE_PARITY_AUDIT.csv")

def check_run_completion(benchmark_pm: pd.DataFrame, existing_pm: pd.DataFrame) -> Tuple[bool, int, bool]:
    done = set(zip(benchmark_pm["repeat_seed"].astype(int), benchmark_pm["participant"].astype(str))) if len(benchmark_pm) else set()
    benchmark_ready = len(done) == 260 and all((s, p) in done for s in ALL_SEEDS for p in ALL_PIDS)
    existing_ready = len(existing_pm) == 10 * 26 * 2
    return benchmark_ready and existing_ready, len(done), existing_ready
def save_partial_manifest(run: BenchmarkRun, completed: int, existing_ready: bool) -> None:
    print(f"Partial benchmark saved: {completed}/260. A+B+C and full-model outputs complete: {existing_ready}")
    atomic_json({
        "version": COMPARISON_VERSION,
        "complete": False,
        "completed_benchmark_seed_participant_pairs": completed,
        "expected_pairs": 260,
        "ABC_rerun": False,
        "HemoBoundary_rerun": False,
        "reference_response_gate": "saved hierarchical HemoBoundary HemoBoundary response decision",
    }, run.out / "audit" / "TABLE2_OPTIONB_RUN_MANIFEST.json")

def calculate_final_tables(benchmark_pm: pd.DataFrame, existing_pm: pd.DataFrame) -> dict:
    all_pm = pd.concat([benchmark_pm, existing_pm], ignore_index=True, sort=False)
    per_seed = seed_method_performance(all_pm)
    summary = across_seed_summary(per_seed)
    return {
        "all": all_pm,
        "per_seed": per_seed,
        "summary": summary,
        "paper_full": paper_ready_table(summary),
        "paper_timing": comparison_timing_only_paper_table(summary),
        "stats": participant_averaged_stats(benchmark_pm, existing_pm),
    }
def save_final_tables(run: BenchmarkRun, tables: Mapping[str, pd.DataFrame]) -> None:
    files = {
        "all": ("results", "TABLE2_OPTIONB_ALL_METHOD_PARTICIPANT_METRICS_ALL_SEEDS.csv.gz"),
        "per_seed": ("results", "TABLE2_OPTIONB_PER_SEED_METHOD_PERFORMANCE.csv"),
        "summary": ("tables", "Table2_OPTIONB_10SEED_BENCHMARK_MEAN_SD.csv"),
        "paper_full": ("tables", "Table2_OPTIONB_FULL_AUDIT_WITH_SHARED_BA.csv"),
        "paper_timing": ("tables", "Table2_OPTIONB_PAPER_READY_TIMING_ONLY.csv"),
        "stats": ("tables", "Table2_OPTIONB_HEMOBOUNDARY_VS_BENCHMARK_STATS.csv"),
    }
    for key, (folder, filename) in files.items():
        atomic_csv(tables[key], run.out / folder / filename)
def save_final_manifest(run: BenchmarkRun) -> None:
    atomic_json({
        "version": COMPARISON_VERSION,
        "complete": True,
        "seeds": list(ALL_SEEDS),
        "participants": list(ALL_PIDS),
        "n_seed_participant_pairs": 260,
        "benchmarks": list(BENCHMARKS),
        "current_architecture": "hierarchical HemoBoundary Base -> +A -> +B -> +C -> +LBR",
        "ABC_rerun": False,
        "HemoBoundary_rerun": False,
        "strict_saved_predictions_reused": True,
        "reference_timing_recomputed": True,
        "synthetic_bank_regenerated": False,
        "shared_reference_response_gate": "saved final hierarchical HemoBoundary HemoBoundary response decision",
        "reference_BA_not_for_independent_ranking": True,
        "R6_source_trained": True,
        "R6_target_participant_excluded_from_fit": True,
        "target_timing_truth_used_for_benchmark_fit": False,
        "primary_inference_unit": "participant",
        "260_rows_treated_as_independent": False,
        "primary_gain_definition": "reference_MAE_minus_HemoBoundary_MAE",
        "holm_within_reference_family_size": 4,
        "holm_global_timing_family_size": 28,
    }, run.out / "audit" / "TABLE2_OPTIONB_RUN_MANIFEST.json")
def print_final_tables(run: BenchmarkRun, tables: Mapping[str, pd.DataFrame]) -> None:
    for title, key in (
        ("PAPER-READY TIMING COMPARISON", "paper_timing"),
        ("NUMERIC MEAN AND SD", "summary"),
        ("PARTICIPANT-AVERAGED PAIRED STATISTICS", "stats"),
    ):
        print(f"\n{title}")
        kwargs = {"float_format": lambda x: f"{x:.6f}"} if key != "paper_timing" else {}
        print(tables[key].to_string(index=False, **kwargs))
    print(f"Complete: 10 seeds, 26 participants per seed. Output: {run.out}")
def main_comparison_table2() -> None:
    args = comparison_parse_args()
    self_tests()
    if args.self_test_only:
        print("Benchmark integration checks: PASS")
        return
    run = prepare_benchmark_run(args)
    for seed in run.seeds:
        benchmark_seed(run, seed)
    collect_saved_checkpoints(run)
    benchmark_pm, benchmark_trial, existing_pm = collect_current_results(run)
    save_available_results(run, benchmark_pm, benchmark_trial)
    complete, count, existing_ready = check_run_completion(benchmark_pm, existing_pm)
    if not complete:
        save_partial_manifest(run, count, existing_ready)
        return
    tables = calculate_final_tables(benchmark_pm, existing_pm)
    save_final_tables(run, tables)
    save_final_manifest(run)
    print_final_tables(run, tables)
if __name__ == "__main__":
    main_comparison_table2()

