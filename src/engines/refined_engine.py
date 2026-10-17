from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
RANDOM_SEED = 20260911
GRID_DT = 0.5
GRID_END = 20.0
BASELINE = (-5.0, 0.0)
FULL_T = np.arange(BASELINE[0], GRID_END + GRID_DT / 2, GRID_DT)
POST_MASK = FULL_T >= 0
GRID = FULL_T[POST_MASK]
N_TRAIN_POS = 30
N_TRAIN_NEG = 15
N_TEST_POS = 24
N_TEST_NEG = 12
INNER_FOLDS = 3
MISS_PENALTY_SEC = 4.0
MIN_GAP = 0.5
BOOT_REPS = 10000
SIGNFLIP_REPS = 10000

def stable_seed(*parts: Any) -> int:
    raw = '|'.join((str(x) for x in parts)).encode('utf-8')
    return int(hashlib.sha256(raw).hexdigest()[:8], 16) % (2 ** 31 - 1)
def try_mount_drive() -> None:
    if Path('/content/gdrive/MyDrive').exists() or Path('/content/drive/MyDrive').exists():
        return
    try:
        from google.colab import drive
    except Exception:
        return
    drive.mount('/content/gdrive', force_remount=False)
def discover_root(explicit: Optional[str]=None) -> Path:
    try_mount_drive()
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [Path('/content/gdrive/MyDrive/HonorProject/shin2018_word_generation_state_preparation_v1'), Path('/content/drive/MyDrive/HonorProject/shin2018_word_generation_state_preparation_v1')]
    for p in candidates:
        if (p / '02_preprocessed').is_dir():
            return p
    raise FileNotFoundError('Could not locate SHIN root containing 02_preprocessed.')
def participant_files(root: Path) -> Dict[str, Path]:
    files = sorted((root / '02_preprocessed').glob('VP*_WG_ALIGNED_PREPROCESSED.npz'))
    if not files:
        raise FileNotFoundError('No VP*_WG_ALIGNED_PREPROCESSED.npz files found.')
    return {p.name.split('_WG_')[0]: p for p in files}
def load_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    required = ['nirs_times', 'hbo_causal_lp02_epochs', 'hbr_causal_lp02_epochs']
    missing = [k for k in required if k not in d]
    if missing:
        raise RuntimeError(f'{path.name}: missing {missing}')
    return d
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
def estimate_noise_profile(path: Path) -> NoiseProfile:
    d = load_npz(path)
    t = np.asarray(d['nirs_times'], float)
    H = np.asarray(d['hbo_causal_lp02_epochs'], float)
    R = np.asarray(d['hbr_causal_lp02_epochs'], float)
    bm = (t >= BASELINE[0]) & (t < BASELINE[1])
    if bm.sum() < 4:
        raise RuntimeError(f'{path.name}: insufficient pre-cue samples.')
    C = min(H.shape[1], R.shape[1])
    H = H[:, :C, :][:, :, bm]
    R = R[:, :C, :][:, :, bm]
    H = H - np.nanmedian(H, axis=2, keepdims=True)
    R = R - np.nanmedian(R, axis=2, keepdims=True)
    hs = np.nanmedian(np.nanstd(H, axis=2), axis=0)
    rs = np.nanmedian(np.nanstd(R, axis=2), axis=0)
    hs = np.where(np.isfinite(hs) & (hs > 1e-08), hs, np.nanmedian(hs[hs > 0]) if np.any(hs > 0) else 1.0)
    rs = np.where(np.isfinite(rs) & (rs > 1e-08), rs, np.nanmedian(rs[rs > 0]) if np.any(rs > 0) else 1.0)
    sample_H = H[:min(24, len(H))].reshape(-1, H.shape[-1])
    sample_R = R[:min(24, len(R))].reshape(-1, R.shape[-1])
    return NoiseProfile(C, hs, rs, safe_lag1(sample_H), safe_lag1(sample_R))
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
def channel_anti(ev: Mapping[str, np.ndarray], common_c: int) -> np.ndarray:
    H = ev['H'][:common_c]
    R = ev['R'][:common_c]
    return (0.5 * (H - R)).astype(np.float32)
def build_cache_for_participant(path: Path, cache_path: Path, mode: str, common_c: int=0) -> None:
    if cache_path.is_file():
        return
    profile = estimate_noise_profile(path)
    if common_c > 0 and profile.n_channels > common_c:
        profile = NoiseProfile(common_c, profile.hbo_scale[:common_c], profile.hbr_scale[:common_c], profile.phi_hbo, profile.phi_hbr)

    def make(split: str, npos: int, nneg: int) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(stable_seed('RAW_D_SCREEN', path.stem, split))
        trials = [synthetic_trial(profile, rng, True) for _ in range(npos)]
        trials += [synthetic_trial(profile, rng, False) for _ in range(nneg)]
        rng.shuffle(trials)
        out: Dict[str, List[Any]] = dict(present=[], onset=[], peak=[], offset=[], censored=[], clean=[], states=[], obs=[], state_feat=[], ch_anti=[])
        for tr in trials:
            ev = evidence(tr['hbo'], tr['hbr'])
            out['present'].append(tr['present'])
            out['onset'].append(tr['onset'])
            out['peak'].append(tr['peak'])
            out['offset'].append(tr['offset'])
            out['censored'].append(tr['censored'])
            out['clean'].append(tr['clean_shape'])
            out['states'].append(tr['states'])
            out['obs'].append(whole_vector(ev))
            out['state_feat'].append(state_features(ev))
            if mode == 'B':
                out['ch_anti'].append(channel_anti(ev, common_c))
        result = {
            'present': np.asarray(out['present'], np.int8),
            'onset': np.asarray(out['onset'], float),
            'peak': np.asarray(out['peak'], float),
            'offset': np.asarray(out['offset'], float),
            'censored': np.asarray(out['censored'], np.int8),
            'clean': np.asarray(out['clean'], np.float32),
            'states': np.asarray(out['states'], np.int8),
            'obs': np.asarray(out['obs'], np.float32),
            'state_feat': np.asarray(out['state_feat'], np.float32)
        }
        if mode == 'B':
            result['ch_anti'] = np.asarray(out['ch_anti'], np.float32)
        return result
    train = make('train', N_TRAIN_POS, N_TRAIN_NEG)
    test = make('test', N_TEST_POS, N_TEST_NEG)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for prefix, d in [('tr', train), ('te', test)]:
        for k, v in d.items():
            arrays[f'{prefix}_{k}'] = v
    np.savez_compressed(cache_path, **arrays)
FEATURE_ARRAY_KEYS = {'obs', 'state_feat', 'ch_anti', 'clean'}

def finite_model_array(x: np.ndarray) -> np.ndarray:
    """Deterministic, label-free sanitization for model inputs only."""
    a = np.asarray(x)
    if not np.issubdtype(a.dtype, np.number):
        return a
    if np.all(np.isfinite(a)):
        return a
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0, copy=True)
def assert_finite_model_inputs(data: Mapping[str, np.ndarray], context: str) -> None:
    bad = {}
    for key in FEATURE_ARRAY_KEYS:
        if key not in data:
            continue
        a = np.asarray(data[key])
        if np.issubdtype(a.dtype, np.number):
            nbad = int(np.size(a) - np.sum(np.isfinite(a)))
            if nbad:
                bad[key] = nbad
    if bad:
        raise RuntimeError(f'{context}: non-finite model inputs remained after sanitization: {bad}')
def load_cache(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as z:
        out = {'train': {}, 'test': {}}
        for k in z.files:
            if k.startswith('tr_'):
                split = 'train'
                key = k[3:]
            elif k.startswith('te_'):
                split = 'test'
                key = k[3:]
            else:
                continue
            arr = z[k]
            if key in FEATURE_ARRAY_KEYS:
                arr = finite_model_array(arr)
            out[split][key] = arr
    assert_finite_model_inputs(out['train'], f'{path.name}:train')
    assert_finite_model_inputs(out['test'], f'{path.name}:test')
    return out
def concat_split(caches: Mapping[str, Dict[str, Dict[str, np.ndarray]]], pids: Sequence[str], split: str) -> Dict[str, np.ndarray]:
    keys = list(caches[pids[0]][split].keys())
    out = {}
    for k in keys:
        arr = np.concatenate([caches[p][split][k] for p in pids], axis=0)
        if k in FEATURE_ARRAY_KEYS:
            arr = finite_model_array(arr)
        out[k] = arr
    assert_finite_model_inputs(out, f'concat_split:{split}')
    return out
def participant_folds(pids: Sequence[str], n_splits: int, seed: int) -> List[List[str]]:
    p = np.asarray(sorted(set(map(str, pids))), object)
    rng = np.random.default_rng(seed)
    rng.shuffle(p)
    return [list(x) for x in np.array_split(p, min(n_splits, len(p))) if len(x)]
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
def one_se_choose(rows: List[Dict[str, Any]], param_cols: Sequence[str], preference_fn) -> Tuple[Dict[str, Any], pd.DataFrame]:
    d = pd.DataFrame(rows)
    summary = []
    for keys, q in d.groupby(list(param_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        x = pd.to_numeric(q['loss'], errors='coerce').to_numpy(float)
        x = x[np.isfinite(x)]
        row = {c: v for c, v in zip(param_cols, keys)}
        row['mean_loss'] = float(np.mean(x)) if len(x) else np.inf
        row['se_loss'] = float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else np.inf
        row['n_val_participants'] = len(x)
        summary.append(row)
    s = pd.DataFrame(summary)
    valid = s[np.isfinite(s['mean_loss'])].copy()
    best_idx = valid['mean_loss'].idxmin()
    best = valid.loc[best_idx]
    cutoff = float(best['mean_loss'] + best['se_loss'])
    admissible = valid[valid['mean_loss'] <= cutoff + 1e-12]
    chosen_idx = sorted(admissible.index.tolist(), key=lambda i: preference_fn(admissible.loc[i].to_dict()))[0]
    chosen = admissible.loc[chosen_idx]
    s['raw_best'] = 0
    s.loc[best_idx, 'raw_best'] = 1
    s['one_se_cutoff'] = cutoff
    s['one_se_admissible'] = (s['mean_loss'] <= cutoff + 1e-12).astype(int)
    s['selected'] = 0
    s.loc[chosen_idx, 'selected'] = 1
    return ({c: chosen[c] for c in param_cols}, s)
def bootstrap_ci(x: np.ndarray, seed: int) -> Tuple[float, float, float]:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = np.empty(BOOT_REPS)
    for i in range(BOOT_REPS):
        vals[i] = np.mean(rng.choice(a, len(a), replace=True))
    return (float(np.mean(a)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))
def signflip_p(x: np.ndarray, seed: int) -> float:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return np.nan
    obs = abs(float(np.mean(a)))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(SIGNFLIP_REPS):
        hits += abs(float(np.mean(a * rng.choice([-1.0, 1.0], size=len(a))))) >= obs - 1e-15
    return float((hits + 1) / (SIGNFLIP_REPS + 1))
def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
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
def prepare_all_caches(file_map: Mapping[str, Path], cache_dir: Path, mode: str) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    common_c = min((load_npz(p)['hbo_causal_lp02_epochs'].shape[1] for p in file_map.values()))
    caches = {}
    for pid, path in file_map.items():
        cp = cache_dir / f'{pid}.npz'
        build_cache_for_participant(path, cp, mode=mode, common_c=common_c)
        caches[pid] = load_cache(cp)
    return caches
def predictions_to_rows(pid: str, candidate: str, truth: Mapping[str, np.ndarray], pred: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    n = len(pred)
    out = pd.DataFrame({
        'participant': [pid] * n,
        'candidate': [candidate] * n,
        'synthetic_index': np.arange(n),
        'truth_present': truth['present'],
        'truth_onset': truth['onset'],
        'truth_peak': truth['peak'],
        'truth_offset': truth['offset'],
        'truth_censored': truth['censored'],
        'pred_response': pred['pred_response'].to_numpy(int),
        'pred_onset': pred['pred_onset'].to_numpy(float),
        'pred_peak': pred['pred_peak'].to_numpy(float),
        'pred_offset': pred['pred_offset'].to_numpy(float)
    })
    for k, v in params.items():
        out[f'selected_{k}'] = v
    return out
def aggregate_participant_metrics(pred_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pid, cand), q in pred_rows.groupby(['participant', 'candidate'], sort=True):
        truth = {
            'present': q['truth_present'].to_numpy(int),
            'onset': q['truth_onset'].to_numpy(float),
            'peak': q['truth_peak'].to_numpy(float),
            'offset': q['truth_offset'].to_numpy(float),
            'censored': q['truth_censored'].to_numpy(int)
        }
        pred = q[['pred_response', 'pred_onset', 'pred_peak', 'pred_offset']].reset_index(drop=True)
        r = trial_metric_summary(truth, pred)
        r.update({'participant': pid, 'candidate': cand})
        rows.append(r)
    return pd.DataFrame(rows)
def performance_table(pm: pd.DataFrame) -> pd.DataFrame:
    metrics = ['response_balanced_accuracy', 'onset_mae_sec', 'peak_mae_sec', 'offset_mae_sec', 'duration_mae_sec', 'selection_loss']
    rows = []
    for cand, q in pm.groupby('candidate'):
        row = {'candidate': cand, 'n_participants': q['participant'].nunique()}
        for m in metrics:
            row[m] = float(np.nanmean(q[m]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values('selection_loss').reset_index(drop=True)
def paired_candidate_stats(pm: pd.DataFrame) -> pd.DataFrame:
    cands = sorted(pm['candidate'].unique())
    if len(cands) != 2:
        return pd.DataFrame()
    a = pm[pm.candidate == cands[0]].set_index('participant')
    b = pm[pm.candidate == cands[1]].set_index('participant')
    common = sorted(set(a.index) & set(b.index))
    rows = []
    for metric in ['onset_mae_sec', 'peak_mae_sec', 'offset_mae_sec', 'duration_mae_sec', 'selection_loss']:
        x = a.loc[common, metric].to_numpy(float) - b.loc[common, metric].to_numpy(float)
        mean, lo, hi = bootstrap_ci(x, stable_seed('pair', metric))
        p = signflip_p(x, stable_seed('pairp', metric))
        rows.append({
            'candidate_A': cands[0],
            'candidate_B': cands[1],
            'metric': metric,
            'mean_A_minus_B_positive_means_B_better': mean,
            'ci95_low': lo,
            'ci95_high': hi,
            'signflip_p': p,
            'B_better_participants': int(np.sum(x > 0)),
            'A_better_participants': int(np.sum(x < 0))
        })
    out = pd.DataFrame(rows)
    out['holm_p'] = holm_adjust(out['signflip_p'])
    return out
VERSION = 'HEMOBOUNDARY_MOD13_HSCR_REFINED_GRID180_v3_2026_09_12'
OUTPUT_DIRNAME = '88_MOD13_HSCR_refined_grid180_v3'
CANDIDATE = 'MOD13_HSCR'
CANDIDATE_LABEL = 'Hybrid State–Change Refinement'
STATE_C_GRID = (0.01, 0.03, 0.1, 0.3)
STAY_BIAS_GRID = (0.0, 0.1, 0.25)
CHANGE_FRAC_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
HEAD_ALPHA_GRID = (3.0, 10.0, 30.0)
INTERACTION_GRID = (0,)
STRUCTURED_GRID = (0,)
CHANGE_WINDOWS = (3, 5, 9)
EXISTING_CACHE_DIRNAME = '82_D5_D13_merged_fully_nested_high_risk_v1/synthetic_cache'
@dataclass
class StateBranch:
    scaler: StandardScaler
    clf: LogisticRegression
    response_scaler: StandardScaler
    response_clf: LogisticRegression
    C: float
def fit_state_branch(data: Mapping[str, np.ndarray], C: float) -> StateBranch:
    F = finite_model_array(data['state_feat']).astype(float, copy=False)
    states = np.asarray(data['states'], int)
    present = np.asarray(data['present'], int)
    Xs = F.reshape(-1, F.shape[-1])
    ys = states.reshape(-1)
    rng = np.random.default_rng(stable_seed('HSCR_STATE_SUBSAMPLE', len(Xs), C))
    if len(Xs) > 60000:
        idx = rng.choice(len(Xs), 60000, replace=False)
        Xs = Xs[idx]
        ys = ys[idx]
    scaler = StandardScaler().fit(Xs)
    Xsz = finite_model_array(scaler.transform(Xs))
    clf = LogisticRegression(C=float(C), class_weight='balanced', max_iter=800, solver='lbfgs', random_state=RANDOM_SEED).fit(Xsz, ys)
    obs = finite_model_array(data['obs']).astype(float, copy=False)
    rsc = StandardScaler().fit(obs)
    obsz = finite_model_array(rsc.transform(obs))
    rclf = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED).fit(obsz, present)
    return StateBranch(scaler=scaler, clf=clf, response_scaler=rsc, response_clf=rclf, C=float(C))
def ordered_state_path(logp: np.ndarray, stay_bias: float) -> Optional[np.ndarray]:
    T, S = logp.shape
    if S != 5:
        return None
    dp = np.full((T, S), -np.inf)
    back = np.full((T, S), -1, int)
    dp[0, 0] = logp[0, 0]
    for t in range(1, T):
        for s in range(S):
            vstay = dp[t - 1, s] + float(stay_bias) if np.isfinite(dp[t - 1, s]) else -np.inf
            vadv = dp[t - 1, s - 1] if s > 0 and np.isfinite(dp[t - 1, s - 1]) else -np.inf
            if vstay >= vadv:
                prev, val = (s, vstay)
            else:
                prev, val = (s - 1, vadv)
            if np.isfinite(val):
                dp[t, s] = val + logp[t, s]
                back[t, s] = prev
    if not np.isfinite(dp[-1, S - 1]):
        return None
    path = np.empty(T, int)
    s = S - 1
    for t in range(T - 1, -1, -1):
        path[t] = s
        if t > 0:
            s = int(back[t, s])
            if s < 0:
                return None
    return path
def full_state_probabilities(model: StateBranch, trial_F: np.ndarray) -> np.ndarray:
    F = finite_model_array(trial_F).astype(float, copy=False)
    P0 = model.clf.predict_proba(finite_model_array(model.scaler.transform(F)))
    P = np.full((len(F), 5), 1e-08, float)
    for j, cls in enumerate(model.clf.classes_):
        P[:, int(cls)] = P0[:, j]
    P /= np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    return P
def latent_state_features(model: StateBranch, data: Mapping[str, np.ndarray], stay_bias: float) -> np.ndarray:
    """Feature-level latent physiology, NOT final D2 timestamps."""
    F = finite_model_array(data['state_feat']).astype(float, copy=False)
    obs = finite_model_array(data['obs']).astype(float, copy=False)
    response_prob = model.response_clf.predict_proba(finite_model_array(model.response_scaler.transform(obs)))[:, 1]
    rows: List[List[float]] = []
    for i in range(len(F)):
        P = full_state_probabilities(model, F[i])
        entropy_t = -np.sum(P * np.log(np.clip(P, 1e-09, 1.0)), axis=1)
        feat: List[float] = []
        for s in range(5):
            ps = P[:, s]
            den = float(np.sum(ps))
            centre = float(np.sum(GRID * ps) / den) if den > 1e-09 else 0.0
            feat.extend([float(np.mean(ps)), float(np.max(ps)), centre])
        path = ordered_state_path(np.log(np.clip(P, 1e-09, 1.0)), stay_bias=float(stay_bias))
        if path is None:
            dwell = np.zeros(5, float)
            t_on = 0.0
            t_peak = float(GRID[int(np.argmax(P[:, 2]))])
            t_off = float(GRID[-1])
            c_on = 0.0
            c_peak = float(np.max(P[:, 2]))
            c_off = 0.0
        else:
            dwell = np.array([np.mean(path == s) for s in range(5)], float)
            idx1 = np.flatnonzero(path == 1)
            idx2 = np.flatnonzero(path == 2)
            idx4 = np.flatnonzero(path == 4)
            on_idx = int(idx1[0]) if len(idx1) else 0
            peak_idx = int(idx2[np.argmax(P[idx2, 2])]) if len(idx2) else int(np.argmax(P[:, 2]))
            off_idx = int(idx4[0]) if len(idx4) else len(GRID) - 1
            t_on = float(GRID[on_idx])
            t_peak = float(GRID[peak_idx])
            t_off = float(GRID[off_idx])
            c_on = float(P[on_idx, 1])
            c_peak = float(P[peak_idx, 2])
            c_off = float(P[off_idx, 4])
        feat.extend(dwell.tolist())
        feat.extend([
            t_on,
            t_peak,
            t_off,
            c_on,
            c_peak,
            c_off,
            float(np.mean(entropy_t)),
            float(np.max(entropy_t)),
            float(response_prob[i]),
            float(np.mean(P[:, 1] + P[:, 2])),
            float(np.mean(P[:, 3] + P[:, 4]))
        ])
        rows.append(feat)
    return finite_model_array(np.asarray(rows, float))
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
def interaction_features(state_X: np.ndarray, change_X: np.ndarray) -> np.ndarray:
    """Physiologically interpretable cross-branch interactions."""
    S = finite_model_array(state_X).astype(float, copy=False)
    C = finite_model_array(change_X).astype(float, copy=False)
    state_times = S[:, 20:23]
    state_conf = S[:, 23:26]
    cp_times = C[:, [27, 30, 33]]
    diff = cp_times - state_times
    absdiff = np.abs(diff)
    mean_times = 0.5 * (cp_times + state_times)
    cp_strength = np.column_stack([np.mean(C[:, [3, 12, 21]], axis=1), np.mean(np.abs(C[:, [4, 13, 22]]), axis=1), np.mean(C[:, [5, 14, 23]], axis=1)])
    conf_x_strength = state_conf * cp_strength
    dwell = S[:, 15:20]
    cp_iqr = C[:, [28, 31, 34]]
    X = np.column_stack([diff, absdiff, mean_times, conf_x_strength, dwell, cp_iqr, np.mean(absdiff, axis=1), np.max(absdiff, axis=1)])
    return finite_model_array(X)
@dataclass
class HSCRModel:
    state_branch: StateBranch
    stay_bias: float
    change_frac: float
    state_scaler: StandardScaler
    change_scaler: StandardScaler
    interaction_scaler: Optional[StandardScaler]
    combined_scaler: StandardScaler
    response_clf: LogisticRegression
    heads: Dict[str, Ridge]
    alpha: float
    use_interactions: bool
    structured: bool
def _combined_feature_fit(state_X: np.ndarray, change_X: np.ndarray, use_interactions: bool) -> Tuple[
    StandardScaler,
    StandardScaler,
    Optional[StandardScaler],
    StandardScaler,
    np.ndarray
]:
    ss = StandardScaler().fit(state_X)
    cs = StandardScaler().fit(change_X)
    Sz = finite_model_array(ss.transform(state_X))
    Cz = finite_model_array(cs.transform(change_X))
    blocks = [Sz, Cz]
    inter_scaler = None
    if use_interactions:
        I = interaction_features(state_X, change_X)
        inter_scaler = StandardScaler().fit(I)
        Iz = finite_model_array(inter_scaler.transform(I))
        blocks.append(Iz)
    Z0 = finite_model_array(np.column_stack(blocks))
    comb = StandardScaler().fit(Z0)
    Z = finite_model_array(comb.transform(Z0))
    return (ss, cs, inter_scaler, comb, Z)
def _combined_feature_apply(model: HSCRModel, state_X: np.ndarray, change_X: np.ndarray) -> np.ndarray:
    Sz = finite_model_array(model.state_scaler.transform(state_X))
    Cz = finite_model_array(model.change_scaler.transform(change_X))
    blocks = [Sz, Cz]
    if model.use_interactions:
        if model.interaction_scaler is None:
            raise RuntimeError('HSCR interaction scaler missing.')
        I = interaction_features(state_X, change_X)
        Iz = finite_model_array(model.interaction_scaler.transform(I))
        blocks.append(Iz)
    Z0 = finite_model_array(np.column_stack(blocks))
    return finite_model_array(model.combined_scaler.transform(Z0))
def fit_hscr(
    data: Mapping[str, np.ndarray],
    state_C: float,
    stay_bias: float,
    change_frac: float,
    alpha: float,
    use_interactions: bool,
    structured: bool
) -> HSCRModel:
    branch = fit_state_branch(data, C=float(state_C))
    SX = latent_state_features(branch, data, stay_bias=float(stay_bias))
    CX = change_branch_features(data, frac=float(change_frac))
    ss, cs, iscaler, comb, Z = _combined_feature_fit(SX, CX, bool(use_interactions))
    y = np.asarray(data['present'], int)
    response_clf = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED).fit(Z, y)
    onset = np.asarray(data['onset'], float)
    peak = np.asarray(data['peak'], float)
    offset = np.asarray(data['offset'], float)
    cens = np.asarray(data['censored'], int)
    pos = y == 1
    heads: Dict[str, Ridge] = {}
    if bool(structured):
        m_on = pos & np.isfinite(onset) & np.isfinite(peak)
        if np.sum(m_on) < 30:
            raise RuntimeError('HSCR structured onset/rise head has too few rows.')
        heads['onset'] = Ridge(alpha=float(alpha)).fit(Z[m_on], onset[m_on])
        rise = np.log(np.clip(peak[m_on] - onset[m_on], 0.25, None))
        heads['rise_log'] = Ridge(alpha=float(alpha)).fit(Z[m_on], rise)
        m_rec = pos & (cens == 0) & np.isfinite(peak) & np.isfinite(offset)
        if np.sum(m_rec) < 30:
            raise RuntimeError('HSCR recovery head has too few rows.')
        recovery = np.log(np.clip(offset[m_rec] - peak[m_rec], 0.5, None))
        heads['recovery_log'] = Ridge(alpha=float(alpha)).fit(Z[m_rec], recovery)
    else:
        for b, t in (('onset', onset), ('peak', peak), ('offset', offset)):
            m = pos & np.isfinite(t)
            if b == 'offset':
                m &= cens == 0
            if np.sum(m) < 30:
                raise RuntimeError(f'HSCR direct {b} head has too few rows.')
            heads[b] = Ridge(alpha=float(alpha)).fit(Z[m], t[m])
    return HSCRModel(
        state_branch=branch,
        stay_bias=float(stay_bias),
        change_frac=float(change_frac),
        state_scaler=ss,
        change_scaler=cs,
        interaction_scaler=iscaler,
        combined_scaler=comb,
        response_clf=response_clf,
        heads=heads,
        alpha=float(alpha),
        use_interactions=bool(use_interactions),
        structured=bool(structured)
    )
def predict_hscr(model: HSCRModel, data: Mapping[str, np.ndarray]) -> pd.DataFrame:
    SX = latent_state_features(model.state_branch, data, stay_bias=model.stay_bias)
    CX = change_branch_features(data, frac=model.change_frac)
    Z = _combined_feature_apply(model, SX, CX)
    prob = model.response_clf.predict_proba(Z)[:, 1]
    if model.structured:
        on = model.heads['onset'].predict(Z)
        rise = np.exp(np.clip(model.heads['rise_log'].predict(Z), np.log(0.25), np.log(8.0)))
        rec = np.exp(np.clip(model.heads['recovery_log'].predict(Z), np.log(0.5), np.log(15.0)))
        pk = on + rise
        off = pk + rec
    else:
        on = model.heads['onset'].predict(Z)
        pk = model.heads['peak'].predict(Z)
        off = model.heads['offset'].predict(Z)
    on, pk, off = enforce_geometry(on, pk, off)
    miss = prob < 0.5
    on[miss] = np.nan
    pk[miss] = np.nan
    off[miss] = np.nan
    return pd.DataFrame({'pred_response': (prob >= 0.5).astype(int), 'pred_onset': on, 'pred_peak': pk, 'pred_offset': off})
HSCR_GRID = tuple(((
    float(state_C),
    float(stay_bias),
    float(frac),
    float(alpha),
    int(interactions),
    int(structured)
) for state_C in STATE_C_GRID for stay_bias in STAY_BIAS_GRID for frac in CHANGE_FRAC_GRID for alpha in HEAD_ALPHA_GRID for interactions in INTERACTION_GRID for structured in STRUCTURED_GRID))
assert len(HSCR_GRID) == 180
def tune_hscr(caches: Mapping[str, Dict[str, Dict[str, np.ndarray]]], source_pids: Sequence[str], context: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    folds = participant_folds(source_pids, INNER_FOLDS, stable_seed(VERSION, context, 'inner_folds'))
    for fold, val_pids in enumerate(folds):
        tr_pids = [p for p in source_pids if p not in val_pids]
        train = concat_split(caches, tr_pids, 'train')
        state_models: Dict[float, StateBranch] = {}
        state_train_features: Dict[Tuple[float, float], np.ndarray] = {}
        state_val_features: Dict[Tuple[str, float, float], np.ndarray] = {}
        change_train_features: Dict[float, np.ndarray] = {}
        change_val_features: Dict[Tuple[str, float], np.ndarray] = {}
        for C in STATE_C_GRID:
            branch = fit_state_branch(train, C=float(C))
            state_models[float(C)] = branch
            for sb in STAY_BIAS_GRID:
                state_train_features[float(C), float(sb)] = latent_state_features(branch, train, stay_bias=float(sb))
                for pid in val_pids:
                    state_val_features[pid, float(C), float(sb)] = latent_state_features(branch, caches[pid]['test'], stay_bias=float(sb))
        for frac in CHANGE_FRAC_GRID:
            change_train_features[float(frac)] = change_branch_features(train, frac=float(frac))
            for pid in val_pids:
                change_val_features[pid, float(frac)] = change_branch_features(caches[pid]['test'], frac=float(frac))
        for state_C, stay_bias, frac, alpha, interactions, structured in HSCR_GRID:
            branch = state_models[float(state_C)]
            SX = state_train_features[float(state_C), float(stay_bias)]
            CX = change_train_features[float(frac)]
            ss, cs, iscaler, comb, Z = _combined_feature_fit(SX, CX, bool(interactions))
            y = np.asarray(train['present'], int)
            response = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED).fit(Z, y)
            onset = np.asarray(train['onset'], float)
            peak = np.asarray(train['peak'], float)
            offset = np.asarray(train['offset'], float)
            cens = np.asarray(train['censored'], int)
            pos = y == 1
            heads: Dict[str, Ridge] = {}
            if bool(structured):
                m = pos & np.isfinite(onset) & np.isfinite(peak)
                heads['onset'] = Ridge(alpha=float(alpha)).fit(Z[m], onset[m])
                heads['rise_log'] = Ridge(alpha=float(alpha)).fit(Z[m], np.log(np.clip(peak[m] - onset[m], 0.25, None)))
                mr = pos & (cens == 0) & np.isfinite(peak) & np.isfinite(offset)
                heads['recovery_log'] = Ridge(alpha=float(alpha)).fit(Z[mr], np.log(np.clip(offset[mr] - peak[mr], 0.5, None)))
            else:
                for b, t in (('onset', onset), ('peak', peak), ('offset', offset)):
                    m = pos & np.isfinite(t)
                    if b == 'offset':
                        m &= cens == 0
                    heads[b] = Ridge(alpha=float(alpha)).fit(Z[m], t[m])
            shell = HSCRModel(
                state_branch=branch,
                stay_bias=float(stay_bias),
                change_frac=float(frac),
                state_scaler=ss,
                change_scaler=cs,
                interaction_scaler=iscaler,
                combined_scaler=comb,
                response_clf=response,
                heads=heads,
                alpha=float(alpha),
                use_interactions=bool(interactions),
                structured=bool(structured)
            )
            for pid in val_pids:
                truth = caches[pid]['test']
                SXv = state_val_features[pid, float(state_C), float(stay_bias)]
                CXv = change_val_features[pid, float(frac)]
                Zval = _combined_feature_apply(shell, SXv, CXv)
                prob = shell.response_clf.predict_proba(Zval)[:, 1]
                if shell.structured:
                    on = shell.heads['onset'].predict(Zval)
                    rise = np.exp(np.clip(shell.heads['rise_log'].predict(Zval), np.log(0.25), np.log(8.0)))
                    rec = np.exp(np.clip(shell.heads['recovery_log'].predict(Zval), np.log(0.5), np.log(15.0)))
                    pk = on + rise
                    off = pk + rec
                else:
                    on = shell.heads['onset'].predict(Zval)
                    pk = shell.heads['peak'].predict(Zval)
                    off = shell.heads['offset'].predict(Zval)
                on, pk, off = enforce_geometry(on, pk, off)
                miss = prob < 0.5
                on[miss] = np.nan
                pk[miss] = np.nan
                off[miss] = np.nan
                pred = pd.DataFrame({'pred_response': (prob >= 0.5).astype(int), 'pred_onset': on, 'pred_peak': pk, 'pred_offset': off})
                loss = trial_metric_summary(truth, pred)['selection_loss']
                rows.append({
                    'fold': fold,
                    'participant': pid,
                    'state_C': float(state_C),
                    'stay_bias': float(stay_bias),
                    'change_frac': float(frac),
                    'alpha': float(alpha),
                    'interactions': int(interactions),
                    'structured': int(structured),
                    'loss': float(loss)
                })
    return one_se_choose(rows, [
        'state_C',
        'stay_bias',
        'change_frac',
        'alpha',
        'interactions',
        'structured'
    ], lambda r: (int(r['interactions']), int(r['structured']), -float(r['alpha']), float(r['state_C']), float(r['stay_bias']), float(r['change_frac'])))
def load_existing_raw_comparison(root: Path) -> pd.DataFrame:
    rows = []
    p2 = root / '75a_raw_D1_D2_strict_screen_v1' / 'results' / 'D1_D2_PARTICIPANT_METRICS.csv'
    if p2.is_file():
        q = pd.read_csv(p2)
        q = q[q['candidate'].astype(str) == 'D2_LATENT_STATE'].copy()
        if len(q):
            rows.append(q)
    p12 = root / '82_D5_D13_merged_fully_nested_high_risk_v1' / 'results' / 'D5_D13_PARTICIPANT_METRICS.csv'
    if p12.is_file():
        q = pd.read_csv(p12)
        q = q[q['candidate'].astype(str) == 'D12_MULTISCALE_CHANGE'].copy()
        if len(q):
            rows.append(q)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
def get_caches(root: Path, fmap: Mapping[str, Path], shard_tag: str) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], Path, str]:
    existing = root / EXISTING_CACHE_DIRNAME
    required = [existing / f'{pid}.npz' for pid in sorted(fmap)]
    if all((p.is_file() for p in required)):
        caches = {pid: load_cache(existing / f'{pid}.npz') for pid in sorted(fmap)}
        return (caches, existing, 'REUSED_EXISTING_D5_D13_RAW_SCREEN_CACHE')
    fallback = root / OUTPUT_DIRNAME / f'synthetic_cache_{shard_tag}'
    caches = prepare_all_caches(fmap, fallback, mode='B')
    return (caches, fallback, 'BUILT_SHARD_LOCAL_CACHE')
def self_tests_hscr() -> None:
    assert len(HSCR_GRID) == 180
    prof = NoiseProfile(8, np.ones(8) * 0.1, np.ones(8) * 0.1, 0.5, 0.5)
    rng = np.random.default_rng(123)
    trials = [synthetic_trial(prof, rng, bool(i % 2)) for i in range(120)]
    temp = []
    for tr in trials:
        ev = evidence(tr['hbo'], tr['hbr'])
        temp.append((whole_vector(ev), state_features(ev), tr))
    data = {
        'obs': finite_model_array(np.asarray([x[0] for x in temp])),
        'state_feat': finite_model_array(np.asarray([x[1] for x in temp])),
        'states': np.asarray([x[2]['states'] for x in temp]),
        'present': np.asarray([x[2]['present'] for x in temp]),
        'onset': np.asarray([x[2]['onset'] for x in temp], float),
        'peak': np.asarray([x[2]['peak'] for x in temp], float),
        'offset': np.asarray([x[2]['offset'] for x in temp], float),
        'censored': np.asarray([x[2]['censored'] for x in temp], int)
    }
    model = fit_hscr(data, state_C=0.1, stay_bias=0.25, change_frac=0.2, alpha=10.0, use_interactions=True, structured=True)
    pred = predict_hscr(model, data)
    assert len(pred) == len(trials)
    assert set(pred.columns) == {'pred_response', 'pred_onset', 'pred_peak', 'pred_offset'}
    sf = latent_state_features(model.state_branch, data, model.stay_bias)
    cf = change_branch_features(data, model.change_frac)
    it = interaction_features(sf, cf)
    assert sf.shape[0] == len(trials)
    assert cf.shape[0] == len(trials)
    assert it.shape[0] == len(trials)
    assert np.all(np.isfinite(sf))
    assert np.all(np.isfinite(cf))
    assert np.all(np.isfinite(it))
    print('Boundary model checks: PASS')
def parse_args_hscr(argv: Optional[Sequence[str]]=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--shin-root', default=None)
    p.add_argument('--output', default=None)
    p.add_argument('--force', action='store_true')
    p.add_argument('--self-test-only', action='store_true')
    args, unknown = p.parse_known_args(argv)
    if unknown:
        print('Ignoring notebook/runtime arguments:', unknown)
    return args
def run_hscr_shard(shard_tag: str, shard_targets: Sequence[str], argv: Optional[Sequence[str]]=None) -> None:
    args = parse_args_hscr(argv)
    self_tests_hscr()
    if args.self_test_only:
        return
    root = discover_root(args.shin_root)
    fmap = participant_files(root)
    all_pids = sorted(fmap)
    if len(all_pids) != 26:
        raise RuntimeError(f'Expected 26 SHIN participants, found {len(all_pids)}.')
    targets = [pid for pid in shard_targets if pid in set(all_pids)]
    missing = sorted(set(shard_targets) - set(targets))
    if missing:
        raise RuntimeError('Shard targets missing from SHIN cohort: ' + ', '.join(missing))
    out = Path(args.output) if args.output else root / OUTPUT_DIRNAME
    ckdir = out / 'checkpoints'
    for d in (ckdir, out / 'results', out / 'tables', out / 'audit'):
        d.mkdir(parents=True, exist_ok=True)
    caches, cache_dir, cache_status = get_caches(root, fmap, shard_tag)
    print('=' * 176)
    print(f'HEMOBOUNDARY — BOUNDARY MODEL EVALUATION {shard_tag}')
    print('=' * 176)
    print('Candidate                      :', CANDIDATE_LABEL)
    print('Outer targets                  :', targets)
    print('Full SHIN source pool          :', len(all_pids))
    print('Source participants / fold     :', len(all_pids) - 1)
    print('Raw-screen protocol            : SAME deterministic RAW_D_SCREEN')
    print('Existing raw cache             :', cache_status)
    print('Cache directory                :', cache_dir)
    print('D2 old result rerun            : NO')
    print('D12 old result rerun           : NO')
    print('Feature branches: fitted within each training fold')
    print('State branch tuning            :', {'C': STATE_C_GRID, 'stay_bias': STAY_BIAS_GRID})
    print('Change branch tuning           :', {'windows': CHANGE_WINDOWS, 'frac': CHANGE_FRAC_GRID})
    print('Head tuning                    :', {'alpha': HEAD_ALPHA_GRID, 'interactions': INTERACTION_GRID, 'structured': STRUCTURED_GRID})
    print('Total nested configurations    :', len(HSCR_GRID))
    print('Held-out target labels         : NEVER used')
    print('Bálint/Khan                    : NOT touched')
    pred_parts = []
    tuning_parts = []
    for oi, target in enumerate(targets, 1):
        cp = ckdir / f'{target}_MOD13_HSCR_REFINED180.csv.gz'
        tp = ckdir / f'{target}_MOD13_HSCR_REFINED180_TUNING.csv.gz'
        if cp.is_file() and tp.is_file() and (not args.force):
            print(f'[{oi:02d}/{len(targets):02d}] {target}: RESUME')
            pred_parts.append(pd.read_csv(cp))
            tuning_parts.append(pd.read_csv(tp))
            continue
        source_pids = [p for p in all_pids if p != target]
        print(f'[{oi:02d}/{len(targets):02d}] {target}')
        params, tuning = tune_hscr(caches, source_pids, context=f'outer={target}')
        print(f"    selected C={float(params['state_C']):.1f}, stay={float(params['stay_bias']):.2f}, frac={float(params['change_frac']):.2f}, alpha={float(params['alpha']):.1f}, interactions={int(params['interactions'])}, structured={int(params['structured'])}")
        train = concat_split(caches, source_pids, 'train')
        truth = caches[target]['test']
        model = fit_hscr(
            train,
            state_C=float(params['state_C']),
            stay_bias=float(params['stay_bias']),
            change_frac=float(params['change_frac']),
            alpha=float(params['alpha']),
            use_interactions=bool(int(params['interactions'])),
            structured=bool(int(params['structured']))
        )
        pred = predict_hscr(model, truth)
        rows = predictions_to_rows(target, CANDIDATE, truth, pred, params)
        tuning = tuning.copy()
        tuning['outer_participant'] = target
        rows.to_csv(cp, index=False, compression='gzip')
        tuning.to_csv(tp, index=False, compression='gzip')
        pred_parts.append(rows)
        tuning_parts.append(tuning)
    shard_pred = pd.concat(pred_parts, ignore_index=True)
    shard_tuning = pd.concat(tuning_parts, ignore_index=True, sort=False)
    shard_pm = aggregate_participant_metrics(shard_pred)
    shard_perf = performance_table(shard_pm)
    shard_pred.to_csv(out / 'results' / f'SHARD_{shard_tag}_MOD13_HSCR_REFINED180_PER_TRIAL.csv.gz', index=False, compression='gzip')
    shard_pm.to_csv(out / 'results' / f'SHARD_{shard_tag}_MOD13_HSCR_REFINED180_PARTICIPANT_METRICS.csv', index=False)
    shard_tuning.to_csv(out / 'audit' / f'SHARD_{shard_tag}_MOD13_HSCR_REFINED180_NESTED_SELECTION.csv.gz', index=False, compression='gzip')
    shard_perf.to_csv(out / 'tables' / f'SHARD_{shard_tag}_Table1_MOD13_HSCR_REFINED180_PERFORMANCE.csv', index=False)
    print('\nSHARD PERFORMANCE')
    print(shard_perf.to_string(index=False))
    all_cp = [ckdir / f'{pid}_MOD13_HSCR_REFINED180.csv.gz' for pid in all_pids]
    all_tp = [ckdir / f'{pid}_MOD13_HSCR_REFINED180_TUNING.csv.gz' for pid in all_pids]
    full_ready = all((p.is_file() for p in all_cp)) and all((p.is_file() for p in all_tp))
    if full_ready:
        full_pred = pd.concat([pd.read_csv(p) for p in all_cp], ignore_index=True)
        full_pm = aggregate_participant_metrics(full_pred)
        full_perf = performance_table(full_pm)
        prior = load_existing_raw_comparison(root)
        if len(prior):
            combined_pm = pd.concat([prior, full_pm], ignore_index=True, sort=False)
            combined_perf = performance_table(combined_pm)
        else:
            combined_perf = full_perf.copy()
        full_pred.to_csv(out / 'results' / 'FULL_MOD13_HSCR_REFINED180_PER_TRIAL.csv.gz', index=False, compression='gzip')
        full_pm.to_csv(out / 'results' / 'FULL_MOD13_HSCR_REFINED180_PARTICIPANT_METRICS.csv', index=False)
        full_perf.to_csv(out / 'tables' / 'FULL_Table1_MOD13_HSCR_REFINED180_PERFORMANCE.csv', index=False)
        combined_perf.to_csv(out / 'tables' / 'FULL_Table2_MOD13_HSCR_REFINED180_WITH_SAVED_D2_D12.csv', index=False)
        print('\n' + '=' * 176)
        print('PARTICIPANT EVALUATION COMPLETE')
        print('=' * 176)
        print(full_perf.to_string(index=False))
        print('\nCOMBINED READ-ONLY RANKING WITH SAVED D2 / D12')
        print(combined_perf.to_string(index=False))
    else:
        missing_cp = [p.stem for p in all_cp if not p.is_file()]
        print('\nFULL 26-PARTICIPANT STATUS: WAITING FOR OTHER SHARD')
        print('Missing outer checkpoints:', missing_cp)
    manifest = {
        'version': VERSION,
        'candidate': CANDIDATE,
        'candidate_label': CANDIDATE_LABEL,
        'shard_tag': shard_tag,
        'shard_targets': targets,
        'full_source_participants': all_pids,
        'full_source_pool_per_outer_fold': len(all_pids) - 1,
        'raw_screen_protocol': 'RAW_D_SCREEN',
        'existing_cache_status': cache_status,
        'old_D2_rerun': False,
        'old_D12_rerun': False,
        'new_state_branch_rebuilt_outer_fold': True,
        'new_change_branch_rebuilt_outer_fold': True,
        'new_interaction_features_rebuilt_outer_fold': True,
        'new_boundary_heads_rebuilt_outer_fold': True,
        'inner_participant_selection': True,
        'held_out_target_labels_used': False,
        'n_nested_configs': len(HSCR_GRID),
        'external_datasets_touched': False,
        'full_26_ready_at_end': bool(full_ready)
    }
    (out / 'audit' / f'SHARD_{shard_tag}_RUN_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('\nFINAL AUDIT')
    print('  shard                      :', shard_tag)
    print('  targets                    :', targets)
    print('  source participants/fold   :', len(all_pids) - 1)
    print('  D2/D12 old result rerun    : NO')
    print('Feature branches: fitted within training folds')
    print('  target labels in tuning    : NO')
    print('  full 26 result ready       :', 'YES' if full_ready else 'NO')
    print('  results                    :', out)
SHARD_TAG = 'PART1_VP001_VP013'
SHARD_TARGETS = tuple((f'VP{i:03d}' for i in range(1, 14)))
if __name__ == '__main__':
    run_hscr_shard(SHARD_TAG, SHARD_TARGETS)
