"""Evaluation on external fNIRS datasets."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
REFINED_ENGINE_SOURCE = (Path(__file__).resolve().parent / 'engines' / 'refined_engine.py').read_text(encoding='utf-8')
N4_ENGINE_SOURCE = (Path(__file__).resolve().parent / 'engines' / 'n4_engine.py').read_text(encoding='utf-8')
VERSION = 'HEMOBOUNDARY_N4_FUNCTIONAL_ABC_LBR_FULLY_NESTED_10SEEDS_V1_20260913'
OUT_DIRNAME = '95_N4_FUNCTIONAL_ABC_LBR_FULLY_NESTED_10SEEDS_V1'
ALL_PIDS = tuple((f'VP{i:03d}' for i in range(1, 27)))
ALL_REPEAT_SEEDS = tuple(range(1, 11))
RUN_PART_NAME = 'PART1_SEEDS_02_04'
RUN_SEEDS = (2, 3, 4)
ACTIVE_RUN_SEED = 1
SHARD_NAME = 'ALL26'
SHARD_TARGETS = ALL_PIDS
STAGE_A = 'A_TEMPORAL_BOUNDARY_ESTIMATION'
STAGE_AB = 'AB_PLUS_RECOVERY_ESTIMATION'
STAGE_ABC = 'ABC_PLUS_RESPONSE_DETECTION'
STAGE_FULL = 'HEMOBOUNDARY_ABC_PLUS_LBR'
STAGE_ORDER = (STAGE_A, STAGE_AB, STAGE_ABC, STAGE_FULL)
STAGE_LABELS = {
    STAGE_A: 'A — Temporal Boundary Estimation',
    STAGE_AB: 'A+B — + Recovery Estimation',
    STAGE_ABC: 'A+B+C — + Response Detection',
    STAGE_FULL: 'HemoBoundary — A+B+C+Local Boundary Refinement'
}
INNER_FOLDS = 3
LOCAL_OOF_FOLDS = 3
BOOT_REPS = 20000
SIGNFLIP_REPS = 20000
BASE_GLOBAL = dict(state_C=0.01, change_frac=0.3, global_alpha=10.0, response_threshold=0.5)
GLOBAL_TIMING_GRID = [
    dict(BASE_GLOBAL),
    dict(BASE_GLOBAL, state_C=0.003),
    dict(BASE_GLOBAL, state_C=0.03),
    dict(BASE_GLOBAL, change_frac=0.2),
    dict(BASE_GLOBAL, change_frac=0.4),
    dict(BASE_GLOBAL, global_alpha=3.0),
    dict(BASE_GLOBAL, global_alpha=30.0)
]
GLOBAL_RESPONSE_GRID = GLOBAL_TIMING_GRID + [dict(BASE_GLOBAL, response_threshold=0.4), dict(BASE_GLOBAL, response_threshold=0.6)]
BASE_LBR = dict(radius=1.0, local_alpha=100.0, strength=0.5)
LBR_GRID = [
    dict(BASE_LBR),
    dict(BASE_LBR, radius=0.5),
    dict(BASE_LBR, radius=1.5),
    dict(BASE_LBR, local_alpha=30.0),
    dict(BASE_LBR, local_alpha=300.0),
    dict(BASE_LBR, strength=0.25),
    dict(BASE_LBR, strength=0.75)
]

def base_stable_seed(*parts: Any) -> int:
    """Seed hash independent of the active robustness repeat."""
    raw = '|'.join((str(x) for x in parts)).encode('utf-8')
    return int(hashlib.sha256(raw).hexdigest()[:8], 16) % (2 ** 31 - 1)
def stable_seed(*parts: Any) -> int:
    """Seed 1 matches the uploaded script; seeds 2..10 add a repeat namespace."""
    if int(ACTIVE_RUN_SEED) == 1:
        return base_stable_seed(*parts)
    return base_stable_seed('MULTISEED_REPEAT', int(ACTIVE_RUN_SEED), *parts)
def canonical_cfg(cfg: Mapping[str, Any]) -> str:
    return json.dumps({k: cfg[k] for k in sorted(cfg)}, sort_keys=True, separators=(',', ':'))
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{RUN_PART_NAME}.{os.getpid()}')
    compression = 'gzip' if str(path).endswith('.gz') else None
    df.to_csv(tmp, index=False, compression=compression)
    os.replace(tmp, path)
def atomic_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{RUN_PART_NAME}.{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)
def try_mount_drive() -> None:
    if Path('/content/gdrive/MyDrive').exists() or Path('/content/drive/MyDrive').exists():
        return
    try:
        from google.colab import drive
        drive.mount('/content/gdrive', force_remount=False)
    except Exception:
        pass
def find_root(explicit: Optional[str]=None) -> Path:
    try_mount_drive()
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [Path('/content/gdrive/MyDrive/HonorProject/shin2018_word_generation_state_preparation_v1'), Path('/content/drive/MyDrive/HonorProject/shin2018_word_generation_state_preparation_v1')]
    for p in candidates:
        if (p / '02_preprocessed').is_dir():
            return p
    raise FileNotFoundError('Could not find SHIN root containing 02_preprocessed. Use --shin-root.')
def import_embedded(source: str, name: str, temp_dir: Path):
    p = temp_dir / f'{name}.py'
    p.write_text(source, encoding='utf-8')
    compile(source, str(p), 'exec')
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod
def add_pid_weight(cache):
    out = {}
    for pid, item in cache.items():
        out[pid] = {}
        for split, d0 in item.items():
            d = {k: np.asarray(v) for k, v in d0.items()}
            n = len(d['present'])
            d['pid'] = np.full(n, pid, dtype='U5')
            d['weight'] = np.ones(n, dtype=float)
            out[pid][split] = d
    return out
def cat(n4, parts):
    return n4.cat(parts)
def training_rows(n4, caches, pids):
    return n4.cat([caches[p]['train'] for p in sorted(pids)])
def validation_rows(n4, caches, pids):
    return n4.cat([caches[p]['test'] for p in sorted(pids)])
def bootstrap_ci(x, seed):
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(a))
    if len(a) == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    vals = np.empty(BOOT_REPS, float)
    done = 0
    while done < BOOT_REPS:
        n = min(2000, BOOT_REPS - done)
        ix = rng.integers(0, len(a), size=(n, len(a)))
        vals[done:done + n] = np.mean(a[ix], axis=1)
        done += n
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return (mean, float(lo), float(hi))
def signflip_p(x, seed):
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
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
def holm_adjust(pvals):
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan, float)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return out
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        out[idx] = min(1.0, running)
    return out
@dataclass
class ConfigRepresentation:
    n4: Any
    state: Any
    change_frac: float

    def transform(self, x):
        self.n4.require_predictors_only(x)
        p, r = self.n4.state_probabilities(self.state, x)
        sx = self.n4.hard_state_features(p, r)
        cx = self.n4.change_branch_features(x, float(self.change_frac))
        X = np.column_stack([sx, cx])
        X = self.n4.finite_model_array(X)
        if not np.isfinite(X).all():
            raise RuntimeError('Non-finite configurable representation.')
        return (X, p)
@dataclass
class ConfigGlobalModel:
    n4: Any
    representation: ConfigRepresentation
    heads: Any
    response_threshold: float
    cfg: Dict[str, float]

    def raw(self, x):
        X, P = self.representation.transform(x)
        raw, prob = self.heads.raw(X)
        on, pk, off = self.n4.enforce_geometry(raw[:, 0], raw[:, 1], raw[:, 2])
        return (np.column_stack([on, pk, off]), np.asarray(prob, float), P)

    def predict(self, x):
        raw, prob, _P = self.raw(x)
        return make_prediction_threshold(self.n4, raw, prob, self.response_threshold)
def fit_state_configurable(n4, data, state_C, audit, stage='state_encoder'):
    audit.record(data, stage)
    F = n4.finite_model_array(data['state_feat'])
    obs = n4.finite_model_array(data['obs'])
    X = F.reshape(-1, F.shape[-1])
    y = np.asarray(data['states'], int).reshape(-1)
    weights = np.asarray(data['weight'], float)
    frame_w = np.repeat(weights, F.shape[1])
    frame_w = frame_w / np.mean(frame_w)
    if len(X) > 60000:
        idx = np.random.default_rng(stable_seed('nested_state_subsample', *sorted(set(map(str, data['pid']))), state_C)).choice(len(X), 60000, replace=False)
        X = X[idx]
        y = y[idx]
        frame_w = frame_w[idx]
    scaler = n4.StableScaler().fit(X, sample_weight=frame_w)
    clf = LogisticRegression(C=float(state_C), class_weight='balanced', max_iter=800, random_state=n4.RANDOM_SEED)
    clf.fit(scaler.transform(X), y, sample_weight=frame_w)
    response_scaler = n4.StableScaler().fit(obs, sample_weight=weights)
    response_clf = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=n4.RANDOM_SEED)
    response_clf.fit(response_scaler.transform(obs), np.asarray(data['present'], int), sample_weight=weights / np.mean(weights))
    return n4.StateModel(scaler, clf, response_scaler, response_clf, tuple(sorted(set(map(str, data['pid'])))))
def fit_global_model(n4, data, cfg, audit, stage_prefix='global'):
    state = fit_state_configurable(n4, data, state_C=float(cfg['state_C']), audit=audit, stage=stage_prefix + '_state_encoder')
    rep = ConfigRepresentation(n4=n4, state=state, change_frac=float(cfg['change_frac']))
    X, _P = rep.transform(n4.predictor_view(data))
    heads = n4.direct_heads(X, data, float(cfg['global_alpha']), audit)
    return ConfigGlobalModel(n4=n4, representation=rep, heads=heads, response_threshold=float(cfg['response_threshold']), cfg=dict(cfg))
def make_prediction_threshold(n4, raw, prob, threshold):
    raw = np.asarray(raw, float)
    prob = np.asarray(prob, float)
    on, pk, off = n4.enforce_geometry(raw[:, 0], raw[:, 1], raw[:, 2])
    miss = prob < float(threshold)
    on = on.astype(float)
    pk = pk.astype(float)
    off = off.astype(float)
    on[miss] = np.nan
    pk[miss] = np.nan
    off[miss] = np.nan
    return pd.DataFrame({'pred_response': (~miss).astype(int), 'pred_onset': on, 'pred_peak': pk, 'pred_offset': off})
def configurable_oof_anchors(n4, data, global_cfg, outer_forbidden, audit_parent, context):
    pids = sorted(set(map(str, data['pid'])))
    n = len(data['pid'])
    raw = np.zeros((n, 3), float)
    response_prob = np.zeros(n, float)
    probabilities = np.zeros((n, len(n4.GRID), 5), float)
    covered = np.zeros(n, int)
    folds = n4.grouped_folds(pids, stable_seed('nested_local_oof', context, canonical_cfg(global_cfg), *pids), LOCAL_OOF_FOLDS)
    for held in folds:
        mask = np.isin(data['pid'], held)
        tr = n4.subset(data, ~mask)
        va = n4.subset(data, mask)
        tr_pids = sorted(set(map(str, tr['pid'])))
        subaudit = n4.FitAudit(tr_pids, set(outer_forbidden) | set(held), context + '/local_oof')
        submodel = fit_global_model(n4, tr, global_cfg, subaudit, stage_prefix='local_oof')
        r, rp, P = submodel.raw(n4.predictor_view(va))
        raw[mask] = r
        response_prob[mask] = rp
        probabilities[mask] = P
        covered[mask] += 1
        audit_parent.rows.extend(subaudit.rows)
    if not np.all(covered == 1):
        raise RuntimeError('Configurable local OOF anchor coverage failed.')
    return (raw, response_prob, probabilities)
@dataclass
class ConfigFullModel:
    n4: Any
    global_model: ConfigGlobalModel
    local_cfg: Dict[str, float]
    local_scalers: Any
    local_models: Any

    def predict(self, x):
        raw, response_prob, P = self.global_model.raw(x)
        profiles = self.n4.local_profiles(x, P, raw, response_prob, float(self.local_cfg['radius']))
        refined = np.array(raw, copy=True)
        for b in range(3):
            delta = self.local_models[b].predict(self.local_scalers[b].transform(profiles[b]))
            refined[:, b] += float(self.local_cfg['strength']) * np.clip(delta, -float(self.local_cfg['radius']), float(self.local_cfg['radius']))
        return make_prediction_threshold(self.n4, refined, response_prob, self.global_model.response_threshold)
def fit_full_model(n4, data, global_cfg, local_cfg, audit, outer_forbidden, context):
    global_model = fit_global_model(n4, data, global_cfg, audit, stage_prefix='full_global')
    raw_oof, rp_oof, P_oof = configurable_oof_anchors(n4, data, global_cfg, outer_forbidden=outer_forbidden, audit_parent=audit, context=context)
    profiles = n4.local_profiles(n4.predictor_view(data), P_oof, raw_oof, rp_oof, float(local_cfg['radius']))
    scalers = []
    models = []
    audit.record(data, 'nested_local_residual_heads_on_OOF_anchors')
    for b, name in enumerate(('onset', 'peak', 'offset')):
        y = np.asarray(data[name], float)
        mask = (np.asarray(data['present'], int) == 1) & np.isfinite(y)
        if name == 'offset':
            mask &= np.asarray(data['censored'], int) == 0
        scaler = n4.StableScaler().fit(profiles[b][mask])
        residual = y[mask] - raw_oof[mask, b]
        reg = Ridge(alpha=float(local_cfg['local_alpha'])).fit(scaler.transform(profiles[b][mask]), residual)
        scalers.append(scaler)
        models.append(reg)
    return ConfigFullModel(n4=n4, global_model=global_model, local_cfg=dict(local_cfg), local_scalers=scalers, local_models=models)
def conditional_errors(truth, raw):
    present = np.asarray(truth['present'], int) == 1
    censored = np.asarray(truth['censored'], int) == 1
    raw = np.asarray(raw, float)
    out = {}
    for j, name in enumerate(('onset', 'peak', 'offset')):
        y = np.asarray(truth[name], float)
        mask = present & np.isfinite(y) & np.isfinite(raw[:, j])
        if name == 'offset':
            mask &= ~censored
        out[name] = float(np.mean(np.abs(raw[mask, j] - y[mask]))) if np.any(mask) else np.nan
    ton = np.asarray(truth['onset'], float)
    toff = np.asarray(truth['offset'], float)
    mask = present & ~censored & np.isfinite(ton) & np.isfinite(toff) & np.isfinite(raw[:, 0]) & np.isfinite(raw[:, 2])
    out['duration'] = float(np.mean(np.abs(raw[mask, 2] - raw[mask, 0] - (toff[mask] - ton[mask])))) if np.any(mask) else np.nan
    return out
def score_A(truth, raw):
    e = conditional_errors(truth, raw)
    return float(np.nanmean([e['onset'], e['peak']]))
def score_AB(truth, raw):
    e = conditional_errors(truth, raw)
    return float(np.nanmean([e['onset'], e['peak'], e['offset'] / 1.5, e['duration'] / 1.5]))
def score_ABC(n4, truth, pred):
    return float(n4.trial_metric_summary(truth, pred)['selection_loss'])
def participant_inner_score(n4, stage, validation_by_pid, model):
    vals = []
    for pid, truth in validation_by_pid.items():
        x = n4.predictor_view(truth)
        if stage == STAGE_A:
            raw, _prob, _P = model.raw(x)
            loss = score_A(truth, raw)
        elif stage == STAGE_AB:
            raw, _prob, _P = model.raw(x)
            loss = score_AB(truth, raw)
        elif stage == STAGE_ABC:
            pred = model.predict(x)
            loss = score_ABC(n4, truth, pred)
        elif stage == STAGE_FULL:
            pred = model.predict(x)
            loss = score_ABC(n4, truth, pred)
        else:
            raise ValueError(stage)
        vals.append(float(loss))
    return float(np.mean(vals))
def inner_fold_map(n4, outer_source_pids, outer_target):
    folds = n4.grouped_folds(outer_source_pids, stable_seed('functional_nested_inner', outer_target, *outer_source_pids), INNER_FOLDS)
    flat = [p for fold in folds for p in fold]
    if sorted(flat) != sorted(outer_source_pids):
        raise RuntimeError(f'{outer_target}: inner participant coverage failed.')
    return folds
def select_nested_configs(n4, caches, outer_target, outer_source_pids):
    """One shared inner-CV pass selects independent configurations for all 4 rows."""
    folds = inner_fold_map(n4, outer_source_pids, outer_target)
    scores_A = {canonical_cfg(c): [] for c in GLOBAL_TIMING_GRID}
    scores_AB = {canonical_cfg(c): [] for c in GLOBAL_TIMING_GRID}
    scores_ABC = {canonical_cfg(c): [] for c in GLOBAL_RESPONSE_GRID}
    full_pairs = [(g, l) for g in GLOBAL_RESPONSE_GRID for l in LBR_GRID]
    scores_FULL = {(canonical_cfg(g), canonical_cfg(l)): [] for g, l in full_pairs}
    score_rows = []
    audit_rows = []
    for inner_fold_idx, held_pids in enumerate(folds, 1):
        inner_train_pids = [p for p in outer_source_pids if p not in held_pids]
        inner_train = training_rows(n4, caches, inner_train_pids)
        validation_by_pid = {pid: caches[pid]['test'] for pid in held_pids}
        forbidden = set(held_pids) | {outer_target}
        global_models = {}
        for global_idx, global_cfg in enumerate(GLOBAL_RESPONSE_GRID):
            gkey = canonical_cfg(global_cfg)
            is_timing_candidate = any((canonical_cfg(x) == gkey for x in GLOBAL_TIMING_GRID))
            audit = n4.FitAudit(inner_train_pids, forbidden, f'outer={outer_target}/inner={inner_fold_idx}/global={global_idx}')
            model = fit_global_model(n4, inner_train, global_cfg, audit, stage_prefix='nested_global')
            global_models[gkey] = model
            audit_rows.extend(audit.rows)
            if is_timing_candidate:
                a_loss = participant_inner_score(n4, STAGE_A, validation_by_pid, model)
                ab_loss = participant_inner_score(n4, STAGE_AB, validation_by_pid, model)
                scores_A[gkey].append(a_loss)
                scores_AB[gkey].append(ab_loss)
                score_rows.extend([{
                    'outer_target': outer_target,
                    'inner_fold': inner_fold_idx,
                    'row': STAGE_A,
                    'global_cfg': gkey,
                    'local_cfg': '',
                    'loss': a_loss,
                    'inner_validation_pids': '|'.join(held_pids)
                }, {'outer_target': outer_target, 'inner_fold': inner_fold_idx, 'row': STAGE_AB, 'global_cfg': gkey, 'local_cfg': '', 'loss': ab_loss, 'inner_validation_pids': '|'.join(held_pids)}])
            abc_loss = participant_inner_score(n4, STAGE_ABC, validation_by_pid, model)
            scores_ABC[gkey].append(abc_loss)
            score_rows.append({
                'outer_target': outer_target,
                'inner_fold': inner_fold_idx,
                'row': STAGE_ABC,
                'global_cfg': gkey,
                'local_cfg': '',
                'loss': abc_loss,
                'inner_validation_pids': '|'.join(held_pids)
            })
        for global_idx, global_cfg in enumerate(GLOBAL_RESPONSE_GRID):
            gkey = canonical_cfg(global_cfg)
            for local_idx, local_cfg in enumerate(LBR_GRID):
                lkey = canonical_cfg(local_cfg)
                audit = n4.FitAudit(inner_train_pids, forbidden, f'outer={outer_target}/inner={inner_fold_idx}/full_g={global_idx}/full_l={local_idx}')
                full_model = fit_full_model(
                    n4,
                    inner_train,
                    global_cfg,
                    local_cfg,
                    audit,
                    outer_forbidden=forbidden,
                    context=f'outer={outer_target}/inner={inner_fold_idx}/full_g={global_idx}/full_l={local_idx}'
                )
                audit_rows.extend(audit.rows)
                loss = participant_inner_score(n4, STAGE_FULL, validation_by_pid, full_model)
                scores_FULL[gkey, lkey].append(loss)
                score_rows.append({
                    'outer_target': outer_target,
                    'inner_fold': inner_fold_idx,
                    'row': STAGE_FULL,
                    'global_cfg': gkey,
                    'local_cfg': lkey,
                    'loss': loss,
                    'inner_validation_pids': '|'.join(held_pids)
                })

    def choose_global(score_dict, grid):
        ranked = []
        for idx, cfg in enumerate(grid):
            key = canonical_cfg(cfg)
            vals = score_dict[key]
            if len(vals) != len(folds):
                raise RuntimeError('Incomplete nested candidate coverage.')
            ranked.append((float(np.mean(vals)), idx, cfg))
        ranked.sort(key=lambda x: (x[0], x[1]))
        return (dict(ranked[0][2]), float(ranked[0][0]))
    selected_A, score_A_best = choose_global(scores_A, GLOBAL_TIMING_GRID)
    selected_AB, score_AB_best = choose_global(scores_AB, GLOBAL_TIMING_GRID)
    selected_ABC, score_ABC_best = choose_global(scores_ABC, GLOBAL_RESPONSE_GRID)
    full_ranked = []
    for pair_idx, (global_cfg, local_cfg) in enumerate(full_pairs):
        key = (canonical_cfg(global_cfg), canonical_cfg(local_cfg))
        vals = scores_FULL[key]
        if len(vals) != len(folds):
            raise RuntimeError('Incomplete full nested candidate coverage.')
        full_ranked.append((float(np.mean(vals)), pair_idx, global_cfg, local_cfg))
    full_ranked.sort(key=lambda x: (x[0], x[1]))
    best = full_ranked[0]
    selected_FULL_global = dict(best[2])
    selected_FULL_local = dict(best[3])
    score_FULL_best = float(best[0])
    selected = {
        STAGE_A: {'global': selected_A, 'local': None, 'inner_loss': score_A_best},
        STAGE_AB: {'global': selected_AB, 'local': None, 'inner_loss': score_AB_best},
        STAGE_ABC: {'global': selected_ABC, 'local': None, 'inner_loss': score_ABC_best},
        STAGE_FULL: {'global': selected_FULL_global, 'local': selected_FULL_local, 'inner_loss': score_FULL_best}
    }
    return (selected, pd.DataFrame(score_rows), pd.DataFrame(audit_rows))
def stage_outer_rows(n4, target, truth, selected, caches):
    source_pids = [p for p in ALL_PIDS if p != target]
    train = training_rows(n4, caches, source_pids)
    x = n4.predictor_view(truth)
    all_rows = []
    audit_rows = []
    cfg = selected[STAGE_A]['global']
    audit = n4.FitAudit(source_pids, [target], f'outer={target}/final_A')
    model = fit_global_model(n4, train, cfg, audit, stage_prefix='outer_A')
    raw, prob, _P = model.raw(x)
    all_rows.append(make_stage_frame(
        target,
        STAGE_A,
        truth,
        raw,
        prob,
        pred=None,
        selected_global=cfg,
        selected_local=None,
        inner_loss=selected[STAGE_A]['inner_loss']
    ))
    audit_rows.extend(audit.rows)
    cfg = selected[STAGE_AB]['global']
    audit = n4.FitAudit(source_pids, [target], f'outer={target}/final_AB')
    model = fit_global_model(n4, train, cfg, audit, stage_prefix='outer_AB')
    raw, prob, _P = model.raw(x)
    all_rows.append(make_stage_frame(
        target,
        STAGE_AB,
        truth,
        raw,
        prob,
        pred=None,
        selected_global=cfg,
        selected_local=None,
        inner_loss=selected[STAGE_AB]['inner_loss']
    ))
    audit_rows.extend(audit.rows)
    cfg = selected[STAGE_ABC]['global']
    audit = n4.FitAudit(source_pids, [target], f'outer={target}/final_ABC')
    model = fit_global_model(n4, train, cfg, audit, stage_prefix='outer_ABC')
    raw, prob, _P = model.raw(x)
    pred = model.predict(x)
    all_rows.append(make_stage_frame(
        target,
        STAGE_ABC,
        truth,
        raw,
        prob,
        pred=pred,
        selected_global=cfg,
        selected_local=None,
        inner_loss=selected[STAGE_ABC]['inner_loss']
    ))
    audit_rows.extend(audit.rows)
    global_cfg = selected[STAGE_FULL]['global']
    local_cfg = selected[STAGE_FULL]['local']
    audit = n4.FitAudit(source_pids, [target], f'outer={target}/final_FULL')
    model = fit_full_model(n4, train, global_cfg, local_cfg, audit, outer_forbidden={target}, context=f'outer={target}/final_FULL')
    raw, prob, _P = model.global_model.raw(x)
    pred = model.predict(x)
    all_rows.append(make_stage_frame(
        target,
        STAGE_FULL,
        truth,
        raw,
        prob,
        pred=pred,
        selected_global=global_cfg,
        selected_local=local_cfg,
        inner_loss=selected[STAGE_FULL]['inner_loss']
    ))
    audit_rows.extend(audit.rows)
    return (pd.concat(all_rows, ignore_index=True), pd.DataFrame(audit_rows))
def make_stage_frame(target, stage, truth, raw, response_prob, pred, selected_global, selected_local, inner_loss):
    n = len(truth['present'])
    raw = np.asarray(raw, float)
    if stage == STAGE_A:
        pred_response = np.full(n, np.nan)
        pred_onset = raw[:, 0]
        pred_peak = raw[:, 1]
        pred_offset = np.full(n, np.nan)
    elif stage == STAGE_AB:
        pred_response = np.full(n, np.nan)
        pred_onset = raw[:, 0]
        pred_peak = raw[:, 1]
        pred_offset = raw[:, 2]
    else:
        pred_response = pred['pred_response'].to_numpy(float)
        pred_onset = pred['pred_onset'].to_numpy(float)
        pred_peak = pred['pred_peak'].to_numpy(float)
        pred_offset = pred['pred_offset'].to_numpy(float)
    return pd.DataFrame({
        'repeat_seed': int(ACTIVE_RUN_SEED),
        'participant': target,
        'stage': stage,
        'stage_label': STAGE_LABELS[stage],
        'synthetic_index': np.arange(n, dtype=int),
        'truth_present': np.asarray(truth['present'], int),
        'truth_onset': np.asarray(truth['onset'], float),
        'truth_peak': np.asarray(truth['peak'], float),
        'truth_offset': np.asarray(truth['offset'], float),
        'truth_censored': np.asarray(truth['censored'], int),
        'pred_response': pred_response,
        'response_probability': np.asarray(response_prob, float),
        'pred_onset': pred_onset,
        'pred_peak': pred_peak,
        'pred_offset': pred_offset,
        'selected_global_cfg': canonical_cfg(selected_global),
        'selected_local_cfg': canonical_cfg(selected_local) if selected_local is not None else '',
        'inner_selected_loss': float(inner_loss)
    })
def balanced_accuracy(y, pred):
    y = np.asarray(y, int)
    pred = np.asarray(pred, int)
    pos = y == 1
    neg = y == 0
    sens = np.mean(pred[pos] == 1) if np.any(pos) else np.nan
    spec = np.mean(pred[neg] == 0) if np.any(neg) else np.nan
    return float(np.nanmean([sens, spec]))
def conditional_metric_row(q):
    present = q['truth_present'].to_numpy(int) == 1
    censored = q['truth_censored'].to_numpy(int) == 1
    out = {}
    response = pd.to_numeric(q['pred_response'], errors='coerce').to_numpy(float)
    if np.isfinite(response).all():
        out['response_balanced_accuracy'] = balanced_accuracy(q['truth_present'].to_numpy(int), response.astype(int))
    else:
        out['response_balanced_accuracy'] = np.nan
    for name in ('onset', 'peak', 'offset'):
        truth = q[f'truth_{name}'].to_numpy(float)
        pred = q[f'pred_{name}'].to_numpy(float)
        mask = present & np.isfinite(truth) & np.isfinite(pred)
        if name == 'offset':
            mask &= ~censored
        out[f'{name}_conditional_mae_sec'] = float(np.mean(np.abs(pred[mask] - truth[mask]))) if np.any(mask) else np.nan
    ton = q['truth_onset'].to_numpy(float)
    toff = q['truth_offset'].to_numpy(float)
    pon = q['pred_onset'].to_numpy(float)
    poff = q['pred_offset'].to_numpy(float)
    mask = present & ~censored & np.isfinite(ton) & np.isfinite(toff) & np.isfinite(pon) & np.isfinite(poff)
    out['duration_conditional_mae_sec'] = float(np.mean(np.abs(poff[mask] - pon[mask] - (toff[mask] - ton[mask])))) if np.any(mask) else np.nan
    return out
def participant_conditional_metrics(per_trial):
    rows = []
    for (pid, stage), q in per_trial.groupby(['participant', 'stage'], sort=True):
        r = conditional_metric_row(q)
        r.update(participant=pid, stage=stage, stage_label=STAGE_LABELS[stage])
        rows.append(r)
    return pd.DataFrame(rows)
def aggregate_conditional(pm):
    rows = []
    metrics = ('response_balanced_accuracy', 'onset_conditional_mae_sec', 'peak_conditional_mae_sec', 'offset_conditional_mae_sec', 'duration_conditional_mae_sec')
    for stage in STAGE_ORDER:
        q = pm[pm['stage'] == stage]
        row = {
            'stage': stage,
            'stage_label': STAGE_LABELS[stage],
            'n_participants': int(q['participant'].nunique()),
            'A': 1,
            'B': int(stage in (STAGE_AB, STAGE_ABC, STAGE_FULL)),
            'C': int(stage in (STAGE_ABC, STAGE_FULL)),
            'LBR': int(stage == STAGE_FULL)
        }
        for metric in metrics:
            x = pd.to_numeric(q[metric], errors='coerce').to_numpy(float)
            x = x[np.isfinite(x)]
            row[metric] = float(np.mean(x)) if len(x) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
def participant_e2e_metrics(n4, per_trial):
    rows = []
    for stage in (STAGE_ABC, STAGE_FULL):
        for pid, q in per_trial[per_trial['stage'] == stage].groupby('participant', sort=True):
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
            r = n4.trial_metric_summary(truth, pred)
            r.update(participant=pid, stage=stage, stage_label=STAGE_LABELS[stage])
            rows.append(r)
    return pd.DataFrame(rows)
def aggregate_e2e(pm):
    metrics = ('response_balanced_accuracy', 'onset_mae_sec', 'peak_mae_sec', 'offset_mae_sec', 'duration_mae_sec', 'selection_loss')
    rows = []
    for stage in (STAGE_ABC, STAGE_FULL):
        q = pm[pm['stage'] == stage]
        row = {'stage': stage, 'stage_label': STAGE_LABELS[stage], 'n_participants': int(q['participant'].nunique())}
        for metric in metrics:
            row[metric] = float(np.nanmean(pd.to_numeric(q[metric], errors='coerce')))
        rows.append(row)
    return pd.DataFrame(rows)
def lbr_paired_stats(pm):
    a = pm[pm['stage'] == STAGE_ABC].set_index('participant')
    b = pm[pm['stage'] == STAGE_FULL].set_index('participant')
    common = sorted(set(a.index) & set(b.index))
    rows = []
    for metric in ('onset_mae_sec', 'peak_mae_sec', 'offset_mae_sec', 'duration_mae_sec'):
        av = a.loc[common, metric].to_numpy(float)
        bv = b.loc[common, metric].to_numpy(float)
        gain = av - bv
        mean, lo, hi = bootstrap_ci(gain, stable_seed('nested_LBR_boot', metric))
        rows.append({
            'reference': STAGE_ABC,
            'candidate': STAGE_FULL,
            'metric': metric,
            'n_participants': len(common),
            'reference_mean': float(np.mean(av)),
            'candidate_mean': float(np.mean(bv)),
            'mean_gain_sec_positive_LBR_better': mean,
            'ci95_low': lo,
            'ci95_high': hi,
            'participants_LBR_better': int(np.sum(gain > 1e-12)),
            'participants_equal': int(np.sum(np.isclose(gain, 0.0))),
            'participants_LBR_worse': int(np.sum(gain < -1e-12)),
            'signflip_p': signflip_p(gain, stable_seed('nested_LBR_sign', metric))
        })
    out = pd.DataFrame(rows)
    out['holm_p_4_timing'] = holm_adjust(out['signflip_p'].to_numpy(float))
    return out
def selected_configs_frame(selected_by_outer):
    rows = []
    for outer_target, selected in selected_by_outer.items():
        for stage in STAGE_ORDER:
            item = selected[stage]
            g = item['global']
            l = item['local']
            row = {
                'outer_target': outer_target,
                'stage': stage,
                'stage_label': STAGE_LABELS[stage],
                'inner_selected_loss': float(item['inner_loss']),
                'state_C': g['state_C'],
                'change_frac': g['change_frac'],
                'global_alpha': g['global_alpha'],
                'response_threshold': g['response_threshold'],
                'radius': l['radius'] if l is not None else np.nan,
                'local_alpha': l['local_alpha'] if l is not None else np.nan,
                'strength': l['strength'] if l is not None else np.nan
            }
            rows.append(row)
    return pd.DataFrame(rows)
def selection_frequency(selected):
    keys = ['stage', 'state_C', 'change_frac', 'global_alpha', 'response_threshold', 'radius', 'local_alpha', 'strength']
    q = selected.groupby(keys, dropna=False).size().reset_index(name='outer_folds_selected')
    q['selection_fraction'] = q['outer_folds_selected'] / selected['outer_target'].nunique()
    return q.sort_values(['stage', 'outer_folds_selected'], ascending=[True, False])
def protected_self_tests(ref, n4):
    ref.self_tests_hscr()
    n4.external_self_tests()
    assert len(GLOBAL_TIMING_GRID) == 7
    assert len(GLOBAL_RESPONSE_GRID) == 9
    assert len(LBR_GRID) == 7
    assert len(GLOBAL_RESPONSE_GRID) * len(LBR_GRID) == 63
    assert BASE_GLOBAL == {'state_C': 0.01, 'change_frac': 0.3, 'global_alpha': 10.0, 'response_threshold': 0.5}
    assert BASE_LBR == {'radius': 1.0, 'local_alpha': 100.0, 'strength': 0.5}
    vals = [(1.0, i) for i in range(3)]
    vals.sort()
    assert vals[0][1] == 0
    print('HemoBoundary component checks: PASS')
def configure_repeat_seed(ref, n4, seed: int) -> None:
    """Configure one complete robustness repeat."""
    global ACTIVE_RUN_SEED
    ACTIVE_RUN_SEED = int(seed)
    if not hasattr(ref, '_MULTISEED_BASE_STABLE_SEED'):
        ref._MULTISEED_BASE_STABLE_SEED = ref.stable_seed
        ref._MULTISEED_BASE_RANDOM_SEED = int(ref.RANDOM_SEED)
    if not hasattr(n4, '_MULTISEED_BASE_STABLE_SEED'):
        n4._MULTISEED_BASE_STABLE_SEED = n4.stable_seed
        n4._MULTISEED_BASE_RANDOM_SEED = int(n4.RANDOM_SEED)
    ref_base = ref._MULTISEED_BASE_STABLE_SEED
    n4_base = n4._MULTISEED_BASE_STABLE_SEED
    if int(seed) == 1:
        ref.stable_seed = ref_base
        n4.stable_seed = n4_base
        ref.RANDOM_SEED = int(ref._MULTISEED_BASE_RANDOM_SEED)
        n4.RANDOM_SEED = int(n4._MULTISEED_BASE_RANDOM_SEED)
    else:

        def ref_seed(*parts, _seed=int(seed), _base=ref_base):
            return _base('MULTISEED_REPEAT', _seed, *parts)

        def n4_seed(*parts, _seed=int(seed), _base=n4_base):
            return _base('MULTISEED_REPEAT', _seed, *parts)
        ref.stable_seed = ref_seed
        n4.stable_seed = n4_seed
        ref.RANDOM_SEED = int(ref._MULTISEED_BASE_RANDOM_SEED) + int(seed) - 1
        n4.RANDOM_SEED = int(n4._MULTISEED_BASE_RANDOM_SEED) + int(seed) - 1
    np.random.seed(base_stable_seed('numpy_legacy', int(seed)) % (2 ** 32 - 1))
def seed_output_dir(parent: Path, seed: int) -> Path:
    return parent / f'seed_{int(seed):02d}'
def seed_complete_path(parent: Path, seed: int) -> Path:
    return seed_output_dir(parent, seed) / 'audit' / 'SEED_COMPLETE.json'
def read_seed_table(parent: Path, seed: int, filename: str) -> pd.DataFrame:
    path = seed_output_dir(parent, seed) / 'tables' / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    q = pd.read_csv(path)
    if 'repeat_seed' not in q.columns:
        q.insert(0, 'repeat_seed', int(seed))
    return q
def aggregate_across_seed_means(per_seed_table: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows = []
    for stage, q in per_seed_table.groupby('stage', sort=False):
        row = {'stage': stage, 'stage_label': q['stage_label'].iloc[0] if 'stage_label' in q else stage, 'n_seeds': int(q['repeat_seed'].nunique())}
        for metric in metrics:
            x = pd.to_numeric(q[metric], errors='coerce').to_numpy(float)
            x = x[np.isfinite(x)]
            row[f'{metric}_mean_across_seeds'] = float(np.mean(x)) if len(x) else np.nan
            row[f'{metric}_sd_across_seeds'] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0 if len(x) == 1 else np.nan
            row[f'{metric}_min_seed'] = float(np.min(x)) if len(x) else np.nan
            row[f'{metric}_max_seed'] = float(np.max(x)) if len(x) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
def multiseed_lbr_stats(parent: Path, seeds: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Primary 10-seed inference keeps participant as the statistical unit."""
    pm_parts = []
    for seed in seeds:
        q = read_seed_table(parent, seed, 'Table7_PARTICIPANT_END_TO_END_METRICS.csv')
        pm_parts.append(q)
    pm = pd.concat(pm_parts, ignore_index=True)
    metrics = ('onset_mae_sec', 'peak_mae_sec', 'offset_mae_sec', 'duration_mae_sec')
    per_seed_rows = []
    stats_rows = []
    for metric in metrics:
        p = pm.pivot_table(index=['repeat_seed', 'participant'], columns='stage', values=metric, aggfunc='first')
        if STAGE_ABC not in p.columns or STAGE_FULL not in p.columns:
            raise RuntimeError(f'Missing ABC/Full rows for multiseed metric {metric}.')
        gain = (p[STAGE_ABC] - p[STAGE_FULL]).rename('gain')
        gain = gain[np.isfinite(gain)]
        for (seed, pid), value in gain.items():
            per_seed_rows.append({'repeat_seed': int(seed), 'participant': str(pid), 'metric': metric, 'gain_sec_positive_LBR_better': float(value)})
        participant_mean = gain.groupby(level='participant').mean()
        seed_mean = gain.groupby(level='repeat_seed').mean()
        x = participant_mean.to_numpy(float)
        mean, lo, hi = bootstrap_ci(x, base_stable_seed('10seed_LBR_participant_boot', metric))
        pval = signflip_p(x, base_stable_seed('10seed_LBR_participant_sign', metric))
        stats_rows.append({
            'reference': STAGE_ABC,
            'candidate': STAGE_FULL,
            'metric': metric,
            'n_seeds': int(len(set(seeds))),
            'n_participants': int(len(participant_mean)),
            'mean_gain_sec_positive_LBR_better': mean,
            'ci95_low': lo,
            'ci95_high': hi,
            'participants_mean_gain_better': int(np.sum(x > 1e-12)),
            'participants_mean_gain_equal': int(np.sum(np.isclose(x, 0.0))),
            'participants_mean_gain_worse': int(np.sum(x < -1e-12)),
            'positive_mean_gain_seeds': int(np.sum(seed_mean.to_numpy(float) > 1e-12)),
            'zero_mean_gain_seeds': int(np.sum(np.isclose(seed_mean.to_numpy(float), 0.0))),
            'negative_mean_gain_seeds': int(np.sum(seed_mean.to_numpy(float) < -1e-12)),
            'seed_mean_gain_sd': float(np.std(seed_mean.to_numpy(float), ddof=1)) if len(seed_mean) > 1 else 0.0,
            'signflip_p_participant_mean_gain': pval
        })
    stats = pd.DataFrame(stats_rows)
    stats['holm_p_4_timing'] = holm_adjust(stats['signflip_p_participant_mean_gain'].to_numpy(float))
    return (pd.DataFrame(per_seed_rows), stats)
def write_available_seed_summary(parent: Path, seeds: Sequence[int]) -> None:
    ready = [s for s in seeds if seed_complete_path(parent, s).is_file()]
    if not ready:
        return
    out = parent / 'multiseed_summary'
    out.mkdir(parents=True, exist_ok=True)
    table2 = pd.concat([read_seed_table(parent, seed, 'Table2_FULLY_NESTED_END_TO_END_ABC_VS_HEMOBOUNDARY.csv') for seed in ready], ignore_index=True)
    atomic_csv(table2, out / f'{RUN_PART_NAME}_AVAILABLE_E2E_PER_SEED.csv')
    print(f'{RUN_PART_NAME}: complete assigned seeds available for summary: {ready}')
def merge_all_10_seed_results(parent: Path) -> bool:
    ready = [seed for seed in ALL_REPEAT_SEEDS if seed_complete_path(parent, seed).is_file()]
    missing = [seed for seed in ALL_REPEAT_SEEDS if seed not in ready]
    print('\n10-SEED MERGE STATUS')
    print('  ready  :', ready)
    print('  missing:', missing)
    if missing:
        return False
    out = parent / 'multiseed_summary'
    out.mkdir(parents=True, exist_ok=True)
    t1 = pd.concat([read_seed_table(parent, seed, 'Table1_FULLY_NESTED_FUNCTIONAL_ABLATION.csv') for seed in ALL_REPEAT_SEEDS], ignore_index=True)
    t2 = pd.concat([read_seed_table(parent, seed, 'Table2_FULLY_NESTED_END_TO_END_ABC_VS_HEMOBOUNDARY.csv') for seed in ALL_REPEAT_SEEDS], ignore_index=True)
    t3 = pd.concat([read_seed_table(parent, seed, 'Table3_FULLY_NESTED_LBR_PAIRED_STATS.csv') for seed in ALL_REPEAT_SEEDS], ignore_index=True)
    t5 = pd.concat([read_seed_table(parent, seed, 'Table5_OUTER_SELECTED_CONFIGS.csv') for seed in ALL_REPEAT_SEEDS], ignore_index=True)
    t1_metrics = (
        'response_balanced_accuracy',
        'onset_conditional_mae_sec',
        'peak_conditional_mae_sec',
        'offset_conditional_mae_sec',
        'duration_conditional_mae_sec'
    )
    t2_metrics = ('response_balanced_accuracy', 'onset_mae_sec', 'peak_mae_sec', 'offset_mae_sec', 'duration_mae_sec', 'selection_loss')
    t1_summary = aggregate_across_seed_means(t1, t1_metrics)
    t2_summary = aggregate_across_seed_means(t2, t2_metrics)
    gain_rows, robust_stats = multiseed_lbr_stats(parent, ALL_REPEAT_SEEDS)
    selection_keys = ['stage', 'state_C', 'change_frac', 'global_alpha', 'response_threshold', 'radius', 'local_alpha', 'strength']
    selection_multi = t5.groupby(selection_keys, dropna=False).size().reset_index(name='seed_outer_folds_selected')
    selection_multi['selection_fraction_of_260_seed_outer_folds'] = selection_multi['seed_outer_folds_selected'] / float(len(ALL_REPEAT_SEEDS) * len(ALL_PIDS))
    selection_multi = selection_multi.sort_values(['stage', 'seed_outer_folds_selected'], ascending=[True, False])
    atomic_csv(t1, out / 'MULTISEED_Table1_PER_SEED.csv')
    atomic_csv(t1_summary, out / 'MULTISEED_Table1_MEAN_SD_MIN_MAX.csv')
    atomic_csv(t2, out / 'MULTISEED_Table2_PER_SEED.csv')
    atomic_csv(t2_summary, out / 'MULTISEED_Table2_MEAN_SD_MIN_MAX.csv')
    atomic_csv(t3, out / 'MULTISEED_Table3_LBR_STATS_PER_SEED.csv')
    atomic_csv(gain_rows, out / 'MULTISEED_LBR_PARTICIPANT_SEED_GAINS.csv')
    atomic_csv(robust_stats, out / 'MULTISEED_Table3_PARTICIPANT_AVERAGED_LBR_STATS.csv')
    atomic_csv(t5, out / 'MULTISEED_SELECTED_CONFIGS_ALL_SEEDS.csv')
    atomic_csv(selection_multi, out / 'MULTISEED_SELECTION_FREQUENCY.csv')
    atomic_json({
        'version': VERSION,
        'seeds': list(ALL_REPEAT_SEEDS),
        'n_seeds': len(ALL_REPEAT_SEEDS),
        'n_participants_per_seed': len(ALL_PIDS),
        'total_seed_outer_folds': len(ALL_REPEAT_SEEDS) * len(ALL_PIDS),
        'seed1_matches_uploaded_seed_protocol': True,
        'seeds_2_to_10_repeat_namespace': True,
        'repeat_changes': ['fresh RAW_D_SCREEN synthetic trial draw', 'inner participant-grouped folds', 'LBR participant-OOF folds', 'estimator random_state'],
        'primary_multiseed_inference_unit': 'participant',
        'primary_multiseed_test': 'within participant: average ABC-minus-Full timing gain across 10 seeds; then bootstrap CI and sign-flip test across 26 participant mean gains',
        'pseudo_replication_of_260_participant_seed_rows': False
    }, out / 'MULTISEED_RUN_MANIFEST.json')
    print('\n' + '=' * 180)
    print('10-SEED TABLE 2 — MEAN / SD ACROSS SEEDS')
    print('=' * 180)
    print(t2_summary.to_string(index=False))
    print('\n' + '=' * 180)
    print('10-SEED LBR ROBUSTNESS — PARTICIPANT-AVERAGED GAINS')
    print('=' * 180)
    print(robust_stats.to_string(index=False))
    print('\nSaved 10-seed summary:', out)
    return True
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--shin-root', default=None)
    p.add_argument(
        '--output',
        default=None,
        help='Parent output folder. Each repeat writes to seed_01 ... seed_10. Default: <SHIN root>/95_N4_FUNCTIONAL_ABC_LBR_FULLY_NESTED_10SEEDS_V1'
    )
    p.add_argument('--seed', type=int, default=0, help='Run only one seed assigned to this part; 0 runs all assigned seeds.')
    p.add_argument('--max-participants', type=int, default=0, help='Smoke test only. A partial seed is never marked complete.')
    p.add_argument('--force', action='store_true', help="Delete and rerun ONLY the selected seed folders, never other parts' seeds.")
    p.add_argument('--merge-only', action='store_true', help='Do no fitting; merge completed seed folders if all 10 are present.')
    p.add_argument('--self-test-only', action='store_true')
    args, unknown = p.parse_known_args()
    if unknown:
        print('Ignoring notebook/runtime arguments:', unknown)
    return args
def run_one_seed(ref, n4, root, fmap, parent_out, seed, args):
    configure_repeat_seed(ref, n4, int(seed))
    seed_out = seed_output_dir(parent_out, seed)
    complete_marker = seed_complete_path(parent_out, seed)
    if complete_marker.is_file() and (not args.force) and (args.max_participants <= 0):
        print(f'\nSEED {seed:02d}: COMPLETE marker found — skipping. Use --force to rerun this seed.')
        return
    if args.force and seed_out.exists():
        print(f'SEED {seed:02d}: --force removes ONLY {seed_out}')
        shutil.rmtree(seed_out)
    for sub in ('fresh_cache', 'checkpoints', 'results', 'tables', 'audit'):
        (seed_out / sub).mkdir(parents=True, exist_ok=True)
    print('\n' + '=' * 180)
    print(f'HEMOBOUNDARY — FULLY NESTED A+B+C+LBR — ROBUSTNESS SEED {seed:02d}')
    print('=' * 180)
    print('Runner part                  :', RUN_PART_NAME)
    print('Assigned seeds               :', list(RUN_SEEDS))
    print('Current seed                 :', int(seed))
    print('Outer targets                : ALL 26 participants')
    print('Outer evaluation             : participant LOSO')
    print('Inner selection              : 3-fold participant-grouped CV')
    print('Inner train rows             : source TRAIN banks only')
    print('Inner validation rows        : inner-held participant TEST banks')
    print('LBR residual fitting         : source-participant OOF coarse anchors')
    print('Global timing grid           :', len(GLOBAL_TIMING_GRID), 'configs')
    print('Global response grid         :', len(GLOBAL_RESPONSE_GRID), 'configs')
    print('Full global x LBR grid       :', len(GLOBAL_RESPONSE_GRID) * len(LBR_GRID), 'configs')
    print('Earlier result tables     : NOT read')
    print('Seed 1 original parity       : YES')
    print('Fresh bank varies seed>1     : YES')
    print('Inner/LBR folds vary seed>1  : YES')
    print('Estimator random state varies: YES')
    print('Output                       :', seed_out)
    print()
    caches0 = ref.prepare_all_caches(fmap, seed_out / 'fresh_cache', mode='B')
    caches = add_pid_weight(caches0)
    cache_audit = pd.DataFrame([{
        'repeat_seed': int(seed),
        'participant': pid,
        'path': str(seed_out / 'fresh_cache' / f'{pid}.npz'),
        'sha256': file_sha256(seed_out / 'fresh_cache' / f'{pid}.npz'),
        'train_rows': len(caches[pid]['train']['present']),
        'test_rows': len(caches[pid]['test']['present'])
    } for pid in ALL_PIDS])
    atomic_csv(cache_audit, seed_out / 'audit' / 'FRESH_CACHE_AUDIT.csv')
    targets = list(ALL_PIDS)
    if args.max_participants > 0:
        targets = targets[:args.max_participants]
    per_trial_parts = []
    inner_score_parts = []
    fit_audit_parts = []
    selected_by_outer = {}
    for oi, target in enumerate(targets, 1):
        cp_rows = seed_out / 'checkpoints' / f'{target}_NESTED_ROWS.csv.gz'
        cp_sel = seed_out / 'checkpoints' / f'{target}_SELECTED.json'
        cp_scores = seed_out / 'checkpoints' / f'{target}_INNER_SCORES.csv.gz'
        cp_audit = seed_out / 'checkpoints' / f'{target}_FIT_AUDIT.csv.gz'
        if cp_rows.is_file() and cp_sel.is_file() and cp_scores.is_file() and cp_audit.is_file() and (not args.force):
            print(f'[seed {seed:02d} {oi:02d}/{len(targets):02d}] {target}: checkpoint')
            per_trial_parts.append(pd.read_csv(cp_rows))
            inner_score_parts.append(pd.read_csv(cp_scores))
            fit_audit_parts.append(pd.read_csv(cp_audit))
            selected_by_outer[target] = json.loads(cp_sel.read_text(encoding='utf-8'))
            continue
        print(f'[seed {seed:02d} {oi:02d}/{len(targets):02d}] {target}: nested selection', flush=True)
        source_pids = [p for p in ALL_PIDS if p != target]
        selected, inner_scores, inner_audit = select_nested_configs(n4, caches, target, source_pids)
        print('    A    :', selected[STAGE_A])
        print('    A+B  :', selected[STAGE_AB])
        print('    A+B+C:', selected[STAGE_ABC])
        print('    Full :', selected[STAGE_FULL])
        rows, outer_audit = stage_outer_rows(n4, target, caches[target]['test'], selected, caches)
        inner_scores = inner_scores.copy()
        inner_scores.insert(0, 'repeat_seed', int(seed))
        combined_audit = pd.concat([inner_audit, outer_audit], ignore_index=True)
        combined_audit.insert(0, 'repeat_seed', int(seed))
        atomic_csv(rows, cp_rows)
        atomic_csv(inner_scores, cp_scores)
        atomic_csv(combined_audit, cp_audit)
        atomic_json(selected, cp_sel)
        per_trial_parts.append(rows)
        inner_score_parts.append(inner_scores)
        fit_audit_parts.append(combined_audit)
        selected_by_outer[target] = selected
    per_trial = pd.concat(per_trial_parts, ignore_index=True)
    if 'repeat_seed' not in per_trial.columns:
        per_trial.insert(0, 'repeat_seed', int(seed))
    inner_scores = pd.concat(inner_score_parts, ignore_index=True)
    if 'repeat_seed' not in inner_scores.columns:
        inner_scores.insert(0, 'repeat_seed', int(seed))
    fit_audit = pd.concat(fit_audit_parts, ignore_index=True)
    if 'repeat_seed' not in fit_audit.columns:
        fit_audit.insert(0, 'repeat_seed', int(seed))
    conditional_pm = participant_conditional_metrics(per_trial)
    functional_table = aggregate_conditional(conditional_pm)
    e2e_pm = participant_e2e_metrics(n4, per_trial)
    e2e_table = aggregate_e2e(e2e_pm)
    paired = lbr_paired_stats(e2e_pm)
    selected_frame = selected_configs_frame(selected_by_outer)
    selection_freq = selection_frequency(selected_frame)
    for df in (conditional_pm, functional_table, e2e_pm, e2e_table, paired, selected_frame, selection_freq):
        if 'repeat_seed' not in df.columns:
            df.insert(0, 'repeat_seed', int(seed))
    atomic_csv(per_trial, seed_out / 'results' / 'FULLY_NESTED_ABC_LBR_PER_TRIAL.csv.gz')
    atomic_csv(conditional_pm, seed_out / 'tables' / 'Table6_PARTICIPANT_CONDITIONAL_METRICS.csv')
    atomic_csv(e2e_pm, seed_out / 'tables' / 'Table7_PARTICIPANT_END_TO_END_METRICS.csv')
    atomic_csv(functional_table, seed_out / 'tables' / 'Table1_FULLY_NESTED_FUNCTIONAL_ABLATION.csv')
    atomic_csv(e2e_table, seed_out / 'tables' / 'Table2_FULLY_NESTED_END_TO_END_ABC_VS_HEMOBOUNDARY.csv')
    atomic_csv(paired, seed_out / 'tables' / 'Table3_FULLY_NESTED_LBR_PAIRED_STATS.csv')
    atomic_csv(selection_freq, seed_out / 'tables' / 'Table4_NESTED_SELECTION_FREQUENCY.csv')
    atomic_csv(selected_frame, seed_out / 'tables' / 'Table5_OUTER_SELECTED_CONFIGS.csv')
    atomic_csv(inner_scores, seed_out / 'audit' / 'INNER_CANDIDATE_SCORES.csv.gz')
    atomic_csv(fit_audit, seed_out / 'audit' / 'FIT_MEMBERSHIP_AUDIT.csv.gz')
    manifest = {
        'version': VERSION,
        'runner_part': RUN_PART_NAME,
        'runner_assigned_seeds': list(RUN_SEEDS),
        'repeat_seed': int(seed),
        'seed1_original_parity': bool(int(seed) == 1),
        'repeat_seed_changes_raw_screen_draw': bool(int(seed) != 1),
        'repeat_seed_changes_inner_folds': bool(int(seed) != 1),
        'repeat_seed_changes_local_oof_folds': bool(int(seed) != 1),
        'repeat_seed_changes_estimator_random_state': bool(int(seed) != 1),
        'outer_evaluation': 'participant LOSO',
        'inner_selection': '3-fold participant-grouped CV',
        'inner_training_bank': 'TRAIN',
        'inner_validation_bank': 'TEST',
        'all_rows_nested': True,
        'outer_target_used_in_tuning': False,
        'A_grid_size': len(GLOBAL_TIMING_GRID),
        'AB_grid_size': len(GLOBAL_TIMING_GRID),
        'ABC_grid_size': len(GLOBAL_RESPONSE_GRID),
        'FULL_grid_size': len(GLOBAL_RESPONSE_GRID) * len(LBR_GRID),
        'global_grid': GLOBAL_RESPONSE_GRID,
        'lbr_grid': LBR_GRID,
        'local_residual_OOF_folds': LOCAL_OOF_FOLDS,
        'historical_result_tables_read': False,
        'fresh_RAW_D_SCREEN_bank': True,
        'component_structure': 'A + B + C + LBR = HemoBoundary',
        'participants': targets,
        'complete_26_participant_run': bool(targets == list(ALL_PIDS))
    }
    atomic_json(manifest, seed_out / 'audit' / 'RUN_MANIFEST.json')
    if targets == list(ALL_PIDS):
        atomic_json({'repeat_seed': int(seed), 'n_participants': 26, 'complete': True, 'version': VERSION}, complete_marker)
    print('\n' + '=' * 180)
    print(f'SEED {seed:02d} — TABLE 1 FUNCTIONAL A+B+C+LBR')
    print('=' * 180)
    print(functional_table.to_string(index=False))
    print('\n' + '=' * 180)
    print(f'SEED {seed:02d} — TABLE 2 END-TO-END ABC VS HEMOBOUNDARY')
    print('=' * 180)
    print(e2e_table.to_string(index=False))
    print('\n' + '=' * 180)
    print(f'SEED {seed:02d} — TABLE 3 LBR PAIRED STATISTICS')
    print('=' * 180)
    print(paired.to_string(index=False))
    print('\nSaved seed result:', seed_out)
    del caches, caches0, per_trial_parts, inner_score_parts, fit_audit_parts
    gc.collect()
def main():
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix=f'hemoboundary_10seed_{RUN_PART_NAME}_') as td:
        td = Path(td)
        ref = import_embedded(REFINED_ENGINE_SOURCE, f'_nested_refined_engine_{RUN_PART_NAME}', td)
        n4 = import_embedded(N4_ENGINE_SOURCE, f'_nested_n4_engine_{RUN_PART_NAME}', td)
        configure_repeat_seed(ref, n4, 1)
        protected_self_tests(ref, n4)
        if args.self_test_only:
            return
        root = find_root(args.shin_root)
        parent_out = Path(args.output) if args.output else root / OUT_DIRNAME
        parent_out.mkdir(parents=True, exist_ok=True)
        if args.merge_only:
            merge_all_10_seed_results(parent_out)
            return
        if args.seed:
            if int(args.seed) not in set(RUN_SEEDS):
                raise ValueError(f'Seed {args.seed} is not assigned to {RUN_PART_NAME}. This part owns {list(RUN_SEEDS)}.')
            seeds = [int(args.seed)]
        else:
            seeds = list(RUN_SEEDS)
        fmap = ref.participant_files(root)
        if sorted(fmap) != list(ALL_PIDS):
            raise RuntimeError('Expected exact 26-participant SHIN cohort.')
        print('=' * 180)
        print('HEMOBOUNDARY — N4 FUNCTIONAL A+B+C+LBR — 10-SEED ROBUSTNESS')
        print('=' * 180)
        print('Runner part                  :', RUN_PART_NAME)
        print('Seeds in this code           :', seeds)
        print('All planned seeds            :', list(ALL_REPEAT_SEEDS))
        print('Participants per seed        : 26')
        print('Shared parent output         :', parent_out)
        print('Seed folders                 : seed_01 ... seed_10')
        print('Parallel-safe                : YES — disjoint seed folders')
        print('--force scope                : selected seeds only')
        print()
        for seed in seeds:
            run_one_seed(ref, n4, root, fmap, parent_out, int(seed), args)
        write_available_seed_summary(parent_out, seeds)
        merge_all_10_seed_results(parent_out)
from scipy.stats import gamma as scipy_gamma
COHERENT_VERSION = 'HEMOBOUNDARY_B0_COHERENT_INCREMENTAL_SEED01_FULLY_NESTED_V1_20260915'
COHERENT_OUT_DIRNAME = '97_B0_COHERENT_INCREMENTAL_SEED01_FULLY_NESTED_V1'
COHERENT_SEED = 1
SHARD_NAME = 'VP001_VP013'
SHARD_TARGETS = ('VP001', 'VP002', 'VP003', 'VP004', 'VP005', 'VP006', 'VP007', 'VP008', 'VP009', 'VP010', 'VP011', 'VP012', 'VP013')
ROW_BASE = 'BASE_CANONICAL_HRF_GLM_DERIVATIVES'
ROW_A = 'BASE_PLUS_A_TEMPORAL_BOUNDARY'
ROW_AB = 'BASE_PLUS_A_PLUS_B_RECOVERY'
ROW_ABC = 'BASE_PLUS_A_PLUS_B_PLUS_C_RESPONSE'
ROW_FULL = 'BASE_PLUS_A_PLUS_B_PLUS_C_PLUS_LBR'
ROW_ORDER = (ROW_BASE, ROW_A, ROW_AB, ROW_ABC, ROW_FULL)
ROW_LABELS = {
    ROW_BASE: 'Base — Canonical HRF GLM + temporal + dispersion derivatives',
    ROW_A: 'Base + A',
    ROW_AB: 'Base + A + B',
    ROW_ABC: 'Base + A + B + C',
    ROW_FULL: 'Base + A + B + C + LBR — HemoBoundary'
}
ROW_MEANING = {
    ROW_BASE: 'Conventional complete-output base estimator',
    ROW_A: 'Add Temporal Boundary Estimation',
    ROW_AB: 'Add Recovery Estimation',
    ROW_ABC: 'Add Response Detection',
    ROW_FULL: 'Final proposed HemoBoundary with Local Boundary Refinement'
}
BASE_RESPONSE_C = 0.5
BASE_RESPONSE_THRESHOLD = 0.5
BASE_BOUNDARY_FRACTION = 0.2
BASE_GLM_RIDGE = 1e-06
A_BASE_CFG = dict(state_C=0.01, timing_alpha=10.0)
A_GRID = [dict(A_BASE_CFG), dict(A_BASE_CFG, state_C=0.003), dict(A_BASE_CFG, state_C=0.03), dict(A_BASE_CFG, timing_alpha=3.0), dict(A_BASE_CFG, timing_alpha=30.0)]
AB_BASE_CFG = dict(state_C=0.01, change_frac=0.3, timing_alpha=10.0)
AB_GRID = [
    dict(AB_BASE_CFG),
    dict(AB_BASE_CFG, state_C=0.003),
    dict(AB_BASE_CFG, state_C=0.03),
    dict(AB_BASE_CFG, change_frac=0.2),
    dict(AB_BASE_CFG, change_frac=0.4),
    dict(AB_BASE_CFG, timing_alpha=3.0),
    dict(AB_BASE_CFG, timing_alpha=30.0)
]
ABC_BASE_CFG = dict(state_C=0.01, change_frac=0.3, timing_alpha=10.0, response_threshold=0.5)
ABC_GRID = [
    dict(ABC_BASE_CFG),
    dict(ABC_BASE_CFG, state_C=0.003),
    dict(ABC_BASE_CFG, state_C=0.03),
    dict(ABC_BASE_CFG, change_frac=0.2),
    dict(ABC_BASE_CFG, change_frac=0.4),
    dict(ABC_BASE_CFG, timing_alpha=3.0),
    dict(ABC_BASE_CFG, timing_alpha=30.0),
    dict(ABC_BASE_CFG, response_threshold=0.4),
    dict(ABC_BASE_CFG, response_threshold=0.6)
]
LBR_BASE_CFG = dict(radius=1.0, local_alpha=100.0, strength=0.5)
LBR_GRID_COHERENT = [
    dict(LBR_BASE_CFG),
    dict(LBR_BASE_CFG, radius=0.5),
    dict(LBR_BASE_CFG, radius=1.5),
    dict(LBR_BASE_CFG, local_alpha=30.0),
    dict(LBR_BASE_CFG, local_alpha=300.0),
    dict(LBR_BASE_CFG, strength=0.25),
    dict(LBR_BASE_CFG, strength=0.75)
]
LOCAL_OOF_FOLDS_COHERENT = 3
def coherent_canonical_hrf(t: np.ndarray, scale: float=1.0) -> np.ndarray:
    t = np.asarray(t, float)
    x = np.clip(t / float(scale), 0.0, None)
    h = scipy_gamma.pdf(x, 6.0) - scipy_gamma.pdf(x, 16.0) / 6.0
    h[t < 0] = 0.0
    mx = float(np.max(np.abs(h)))
    if mx > 0:
        h = h / mx
    return h
def coherent_glm_basis(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, float)
    h = coherent_canonical_hrf(grid, 1.0)
    dt = float(np.median(np.diff(grid)))
    temporal = np.gradient(h, dt)
    eps = 0.08
    wide = coherent_canonical_hrf(grid, 1.0 + eps)
    narrow = coherent_canonical_hrf(grid, 1.0 - eps)
    dispersion = (wide - narrow) / (2.0 * eps)
    B = np.column_stack([h, temporal, dispersion]).astype(float)
    for j in range(B.shape[1]):
        nrm = float(np.sqrt(np.sum(B[:, j] ** 2)))
        if nrm > 1e-12:
            B[:, j] /= nrm
    return B
def coherent_anti(n4, x: Mapping[str, np.ndarray]) -> np.ndarray:
    n4.require_predictors_only(x)
    F = n4.finite_model_array(x['state_feat']).astype(float, copy=False)
    if F.ndim != 3 or F.shape[1] != len(n4.GRID) or F.shape[2] < 3:
        raise RuntimeError(f'Unexpected state_feat shape {F.shape}.')
    return np.asarray(F[:, :, 2], float)
def coherent_glm_features_times(n4, x: Mapping[str, np.ndarray]):
    Y = coherent_anti(n4, x)
    grid = np.asarray(n4.GRID, float)
    B = coherent_glm_basis(grid)
    trend = (grid - np.mean(grid)) / max(float(np.std(grid)), 1e-12)
    X = np.column_stack([B, np.ones(len(grid)), trend])
    penalty = np.diag([BASE_GLM_RIDGE, BASE_GLM_RIDGE, BASE_GLM_RIDGE, 0.0, 0.0])
    projection = np.linalg.pinv(X.T @ X + penalty) @ X.T
    feats, times = ([], [])
    for y in Y:
        beta = projection @ np.asarray(y, float)
        task = B @ beta[:3]
        fitted = X @ beta
        resid = np.asarray(y, float) - fitted
        sse = float(np.sum(resid ** 2))
        centred = np.asarray(y, float) - float(np.mean(y))
        sst = float(np.sum(centred ** 2))
        r2 = 1.0 - sse / max(sst, 1e-12)
        peak_idx = int(np.argmax(task))
        peak_val = float(task[peak_idx])
        floor = float(np.min(task[:peak_idx + 1]))
        amp = max(peak_val - floor, 1e-08)
        level = floor + BASE_BOUNDARY_FRACTION * amp
        pre = np.flatnonzero(task[:peak_idx + 1] >= level)
        on_idx = int(pre[0]) if len(pre) else 0
        post = np.flatnonzero(task[peak_idx + 1:] <= level)
        off_idx = int(peak_idx + 1 + post[0]) if len(post) else len(grid) - 1
        on, pk, off = n4.enforce_geometry(np.asarray([grid[on_idx]], float), np.asarray([grid[peak_idx]], float), np.asarray([grid[off_idx]], float))
        positive = np.clip(task, 0.0, None)
        feats.append([
            float(beta[0]),
            float(beta[1]),
            float(beta[2]),
            float(r2),
            float(peak_val),
            float(np.sum(positive)),
            float(np.mean(np.abs(resid))),
            float(np.std(resid)),
            float(np.max(y)),
            float(np.min(y))
        ])
        times.append([float(on[0]), float(pk[0]), float(off[0])])
    return (n4.finite_model_array(np.asarray(feats, float)), np.asarray(times, float))
@dataclass
class CoherentBaseModel:
    n4: Any
    scaler: Any
    response_clf: Any

    def raw(self, x):
        feat, raw = coherent_glm_features_times(self.n4, x)
        prob = self.response_clf.predict_proba(self.scaler.transform(feat))[:, 1]
        return (raw, np.asarray(prob, float))

    def predict(self, x):
        raw, prob = self.raw(x)
        return make_prediction_threshold(self.n4, raw, prob, BASE_RESPONSE_THRESHOLD)
def fit_coherent_base(n4, data, audit, stage_prefix='BASE'):
    audit.record(data, stage_prefix + '_response_classifier')
    feat, _ = coherent_glm_features_times(n4, n4.predictor_view(data))
    y = np.asarray(data['present'], int)
    w = np.asarray(data['weight'], float)
    scaler = n4.StableScaler().fit(feat, sample_weight=w)
    clf = LogisticRegression(C=BASE_RESPONSE_C, class_weight='balanced', max_iter=1000, random_state=n4.RANDOM_SEED)
    clf.fit(scaler.transform(feat), y, sample_weight=w / np.mean(w))
    return CoherentBaseModel(n4=n4, scaler=scaler, response_clf=clf)
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
        X, y, fw = (X[idx], y[idx], fw[idx])
    scaler = n4.StableScaler().fit(X, sample_weight=fw)
    clf = LogisticRegression(C=float(state_C), class_weight='balanced', max_iter=800, random_state=n4.RANDOM_SEED)
    clf.fit(scaler.transform(X), y, sample_weight=fw)
    return CoherentStateModel(scaler=scaler, clf=clf)
def coherent_state_probabilities(n4, model, x):
    n4.require_predictors_only(x)
    F = n4.finite_model_array(x['state_feat']).astype(float, copy=False)
    n, t, d = F.shape
    p0 = model.clf.predict_proba(model.scaler.transform(F.reshape(-1, d)))
    P = np.full((n * t, 5), 1e-08, float)
    for j, cls in enumerate(model.clf.classes_):
        P[:, int(cls)] = p0[:, j]
    P /= np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    return P.reshape(n, t, 5)
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
class CoherentAModel:
    n4: Any
    state: CoherentStateModel
    heads: CoherentTimingHeads

    def raw(self, x):
        P = coherent_state_probabilities(self.n4, self.state, x)
        X = coherent_state_features(self.n4, P)
        pred = self.heads.raw(X)
        return (np.asarray(pred['onset'], float), np.asarray(pred['peak'], float), P)
def fit_coherent_A(n4, data, cfg, audit, stage_prefix='A'):
    state = fit_coherent_state(n4, data, float(cfg['state_C']), audit, stage_prefix)
    P = coherent_state_probabilities(n4, state, n4.predictor_view(data))
    X = coherent_state_features(n4, P)
    heads = fit_coherent_timing_heads(n4, X, data, cfg['timing_alpha'], ('onset', 'peak'), audit, stage_prefix)
    return CoherentAModel(n4=n4, state=state, heads=heads)
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
    n = len(data['pid'])
    raw = np.zeros((n, 3), float)
    rp = np.zeros(n, float)
    P = np.zeros((n, len(n4.GRID), 5), float)
    covered = np.zeros(n, int)
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

    def predict(self, x):
        raw, rp, P = self.abc.raw(x)
        profiles = self.n4.local_profiles(x, P, raw, rp, float(self.local_cfg['radius']))
        refined = np.asarray(raw, float).copy()
        for b in range(3):
            delta = self.regs[b].predict(self.scalers[b].transform(profiles[b]))
            radius = float(self.local_cfg['radius'])
            refined[:, b] += float(self.local_cfg['strength']) * np.clip(delta, -radius, radius)
        return make_prediction_threshold(self.n4, refined, rp, self.abc.response_threshold)
def fit_coherent_full(n4, data, abc_cfg, local_cfg, audit, outer_forbidden, context):
    abc = fit_coherent_ABC(n4, data, abc_cfg, audit, stage_prefix='FULL_ABC')
    raw_oof, rp_oof, P_oof = coherent_oof_anchors(n4, data, abc_cfg, outer_forbidden, audit, context)
    profiles = n4.local_profiles(n4.predictor_view(data), P_oof, raw_oof, rp_oof, float(local_cfg['radius']))
    audit.record(data, 'FULL_LBR_residual_heads_on_source_OOF_anchors')
    present = np.asarray(data['present'], int) == 1
    censored = np.asarray(data['censored'], int) == 1
    scalers, regs = ([], [])
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
def predict_row_A(n4, base, a_model, x):
    base_raw, base_prob = base.raw(x)
    on, pk, _P = a_model.raw(x)
    raw = np.column_stack([on, pk, base_raw[:, 2]])
    on2, pk2, off2 = n4.enforce_geometry(raw[:, 0], raw[:, 1], raw[:, 2])
    raw = np.column_stack([on2, pk2, off2])
    return make_prediction_threshold(n4, raw, base_prob, BASE_RESPONSE_THRESHOLD)
def predict_row_AB(n4, base, ab_model, x):
    raw, _P, _X = ab_model.raw(x)
    _base_raw, base_prob = base.raw(x)
    return make_prediction_threshold(n4, raw, base_prob, BASE_RESPONSE_THRESHOLD)
def coherent_inner_folds(n4, source_pids, target):
    folds = n4.grouped_folds(source_pids, stable_seed('COHERENT_B0_INNER', COHERENT_SEED, target, *source_pids), INNER_FOLDS)
    flat = sorted((p for fold in folds for p in fold))
    if flat != sorted(source_pids):
        raise RuntimeError(f'{target}: inner participant coverage failed.')
    return folds
def mean_e2e_loss(n4, validation_by_pid, predictor):
    vals = []
    for pid, truth in validation_by_pid.items():
        pred = predictor(n4.predictor_view(truth))
        vals.append(float(n4.trial_metric_summary(truth, pred)['selection_loss']))
    return float(np.mean(vals))
def coherent_choose(score_dict, grid, nfolds):
    ranked = []
    for idx, cfg in enumerate(grid):
        key = canonical_cfg(cfg)
        vals = score_dict[key]
        if len(vals) != nfolds:
            raise RuntimeError('Incomplete inner candidate coverage.')
        ranked.append((float(np.mean(vals)), idx, dict(cfg)))
    ranked.sort(key=lambda x: (x[0], x[1]))
    return (ranked[0][2], ranked[0][0])
def select_coherent_configs(n4, caches, outer_target, outer_source_pids):
    folds = coherent_inner_folds(n4, outer_source_pids, outer_target)
    scores_A = {canonical_cfg(c): [] for c in A_GRID}
    scores_AB = {canonical_cfg(c): [] for c in AB_GRID}
    scores_ABC = {canonical_cfg(c): [] for c in ABC_GRID}
    full_pairs = [(g, l) for g in ABC_GRID for l in LBR_GRID_COHERENT]
    scores_FULL = {(canonical_cfg(g), canonical_cfg(l)): [] for g, l in full_pairs}
    score_rows = []
    audit_rows = []
    for fi, held_pids in enumerate(folds, 1):
        tr_pids = [p for p in outer_source_pids if p not in held_pids]
        train = training_rows(n4, caches, tr_pids)
        validation_by_pid = {pid: caches[pid]['test'] for pid in held_pids}
        forbidden = set(held_pids) | {outer_target}
        base_audit = n4.FitAudit(tr_pids, forbidden, f'coherent/outer={outer_target}/inner={fi}/BASE')
        base = fit_coherent_base(n4, train, base_audit, stage_prefix='INNER_BASE')
        audit_rows.extend(base_audit.rows)
        for ci, cfg in enumerate(A_GRID):
            audit = n4.FitAudit(tr_pids, forbidden, f'coherent/outer={outer_target}/inner={fi}/A={ci}')
            model = fit_coherent_A(n4, train, cfg, audit, stage_prefix='INNER_A')
            audit_rows.extend(audit.rows)
            loss = mean_e2e_loss(n4, validation_by_pid, lambda x, b=base, m=model: predict_row_A(n4, b, m, x))
            scores_A[canonical_cfg(cfg)].append(loss)
            score_rows.append({
                'outer_target': outer_target,
                'inner_fold': fi,
                'row': ROW_A,
                'global_cfg': canonical_cfg(cfg),
                'local_cfg': '',
                'loss': loss,
                'inner_validation_pids': '|'.join(held_pids)
            })
        for ci, cfg in enumerate(AB_GRID):
            audit = n4.FitAudit(tr_pids, forbidden, f'coherent/outer={outer_target}/inner={fi}/AB={ci}')
            model = fit_coherent_AB(n4, train, cfg, audit, stage_prefix='INNER_AB')
            audit_rows.extend(audit.rows)
            loss = mean_e2e_loss(n4, validation_by_pid, lambda x, b=base, m=model: predict_row_AB(n4, b, m, x))
            scores_AB[canonical_cfg(cfg)].append(loss)
            score_rows.append({
                'outer_target': outer_target,
                'inner_fold': fi,
                'row': ROW_AB,
                'global_cfg': canonical_cfg(cfg),
                'local_cfg': '',
                'loss': loss,
                'inner_validation_pids': '|'.join(held_pids)
            })
        for ci, cfg in enumerate(ABC_GRID):
            audit = n4.FitAudit(tr_pids, forbidden, f'coherent/outer={outer_target}/inner={fi}/ABC={ci}')
            model = fit_coherent_ABC(n4, train, cfg, audit, stage_prefix='INNER_ABC')
            audit_rows.extend(audit.rows)
            loss = mean_e2e_loss(n4, validation_by_pid, lambda x, m=model: m.predict(x))
            scores_ABC[canonical_cfg(cfg)].append(loss)
            score_rows.append({
                'outer_target': outer_target,
                'inner_fold': fi,
                'row': ROW_ABC,
                'global_cfg': canonical_cfg(cfg),
                'local_cfg': '',
                'loss': loss,
                'inner_validation_pids': '|'.join(held_pids)
            })
        for gi, gcfg in enumerate(ABC_GRID):
            for li, lcfg in enumerate(LBR_GRID_COHERENT):
                audit = n4.FitAudit(tr_pids, forbidden, f'coherent/outer={outer_target}/inner={fi}/FULL_g={gi}_l={li}')
                model = fit_coherent_full(
                    n4,
                    train,
                    gcfg,
                    lcfg,
                    audit,
                    outer_forbidden=forbidden,
                    context=f'coherent/outer={outer_target}/inner={fi}/FULL_g={gi}_l={li}'
                )
                audit_rows.extend(audit.rows)
                loss = mean_e2e_loss(n4, validation_by_pid, lambda x, m=model: m.predict(x))
                key = (canonical_cfg(gcfg), canonical_cfg(lcfg))
                scores_FULL[key].append(loss)
                score_rows.append({
                    'outer_target': outer_target,
                    'inner_fold': fi,
                    'row': ROW_FULL,
                    'global_cfg': canonical_cfg(gcfg),
                    'local_cfg': canonical_cfg(lcfg),
                    'loss': loss,
                    'inner_validation_pids': '|'.join(held_pids)
                })
    selected_A, loss_A = coherent_choose(scores_A, A_GRID, len(folds))
    selected_AB, loss_AB = coherent_choose(scores_AB, AB_GRID, len(folds))
    selected_ABC, loss_ABC = coherent_choose(scores_ABC, ABC_GRID, len(folds))
    full_ranked = []
    for idx, (gcfg, lcfg) in enumerate(full_pairs):
        key = (canonical_cfg(gcfg), canonical_cfg(lcfg))
        vals = scores_FULL[key]
        if len(vals) != len(folds):
            raise RuntimeError('Incomplete full inner candidate coverage.')
        full_ranked.append((float(np.mean(vals)), idx, dict(gcfg), dict(lcfg)))
    full_ranked.sort(key=lambda x: (x[0], x[1]))
    best = full_ranked[0]
    selected = {
        ROW_A: {'global': selected_A, 'local': None, 'inner_loss': float(loss_A)},
        ROW_AB: {'global': selected_AB, 'local': None, 'inner_loss': float(loss_AB)},
        ROW_ABC: {'global': selected_ABC, 'local': None, 'inner_loss': float(loss_ABC)},
        ROW_FULL: {'global': best[2], 'local': best[3], 'inner_loss': float(best[0])}
    }
    return (selected, pd.DataFrame(score_rows), pd.DataFrame(audit_rows))
def coherent_frame(target, row, truth, pred, selected_global='', selected_local='', inner_loss=np.nan):
    return pd.DataFrame({
        'participant': target,
        'row': row,
        'configuration': ROW_LABELS[row],
        'meaning': ROW_MEANING[row],
        'synthetic_index': np.arange(len(pred), dtype=int),
        'truth_present': np.asarray(truth['present'], int),
        'truth_onset': np.asarray(truth['onset'], float),
        'truth_peak': np.asarray(truth['peak'], float),
        'truth_offset': np.asarray(truth['offset'], float),
        'truth_censored': np.asarray(truth['censored'], int),
        'pred_response': pred['pred_response'].to_numpy(int),
        'pred_onset': pred['pred_onset'].to_numpy(float),
        'pred_peak': pred['pred_peak'].to_numpy(float),
        'pred_offset': pred['pred_offset'].to_numpy(float),
        'selected_global_cfg': selected_global,
        'selected_local_cfg': selected_local,
        'inner_selected_loss': float(inner_loss) if np.isfinite(inner_loss) else np.nan
    })
def outer_all_rows(n4, caches, target, selected):
    source_pids = [p for p in ALL_PIDS if p != target]
    train = training_rows(n4, caches, source_pids)
    truth = caches[target]['test']
    x = n4.predictor_view(truth)
    rows = []
    audits = []
    audit = n4.FitAudit(source_pids, [target], f'coherent/outer={target}/BASE')
    base = fit_coherent_base(n4, train, audit, 'OUTER_BASE')
    pred_base = base.predict(x)
    rows.append(coherent_frame(target, ROW_BASE, truth, pred_base, selected_global='FIXED_CONVENTIONAL_BASE'))
    audits.extend(audit.rows)
    acfg = selected[ROW_A]['global']
    audit = n4.FitAudit(source_pids, [target], f'coherent/outer={target}/A')
    a_model = fit_coherent_A(n4, train, acfg, audit, 'OUTER_A')
    pred_a = predict_row_A(n4, base, a_model, x)
    rows.append(coherent_frame(target, ROW_A, truth, pred_a, selected_global=canonical_cfg(acfg), inner_loss=selected[ROW_A]['inner_loss']))
    audits.extend(audit.rows)
    abcfg = selected[ROW_AB]['global']
    audit = n4.FitAudit(source_pids, [target], f'coherent/outer={target}/AB')
    ab_model = fit_coherent_AB(n4, train, abcfg, audit, 'OUTER_AB')
    pred_ab = predict_row_AB(n4, base, ab_model, x)
    rows.append(coherent_frame(target, ROW_AB, truth, pred_ab, selected_global=canonical_cfg(abcfg), inner_loss=selected[ROW_AB]['inner_loss']))
    audits.extend(audit.rows)
    abccfg = selected[ROW_ABC]['global']
    audit = n4.FitAudit(source_pids, [target], f'coherent/outer={target}/ABC')
    abc_model = fit_coherent_ABC(n4, train, abccfg, audit, 'OUTER_ABC')
    pred_abc = abc_model.predict(x)
    rows.append(coherent_frame(target, ROW_ABC, truth, pred_abc, selected_global=canonical_cfg(abccfg), inner_loss=selected[ROW_ABC]['inner_loss']))
    audits.extend(audit.rows)
    fg = selected[ROW_FULL]['global']
    fl = selected[ROW_FULL]['local']
    audit = n4.FitAudit(source_pids, [target], f'coherent/outer={target}/FULL')
    full_model = fit_coherent_full(n4, train, fg, fl, audit, outer_forbidden={target}, context=f'coherent/outer={target}/FULL')
    pred_full = full_model.predict(x)
    rows.append(coherent_frame(
        target,
        ROW_FULL,
        truth,
        pred_full,
        selected_global=canonical_cfg(fg),
        selected_local=canonical_cfg(fl),
        inner_loss=selected[ROW_FULL]['inner_loss']
    ))
    audits.extend(audit.rows)
    return (pd.concat(rows, ignore_index=True), pd.DataFrame(audits))
def coherent_participant_metrics(n4, per_trial):
    rows = []
    for (pid, row), q in per_trial.groupby(['participant', 'row'], sort=False):
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
        m = n4.trial_metric_summary(truth, pred)
        rows.append({'participant': str(pid), 'row': str(row), 'configuration': ROW_LABELS[str(row)], 'meaning': ROW_MEANING[str(row)], **m})
    out = pd.DataFrame(rows)
    out['row'] = pd.Categorical(out['row'], categories=list(ROW_ORDER), ordered=True)
    return out.sort_values(['row', 'participant']).reset_index(drop=True)
def coherent_summary(pm):
    rows = []
    for row in ROW_ORDER:
        q = pm[pm['row'].astype(str) == row]
        if not len(q):
            continue
        r = {'row': row, 'configuration': ROW_LABELS[row], 'meaning': ROW_MEANING[row], 'n_participants': int(q['participant'].nunique())}
        for metric in ('response_balanced_accuracy', 'onset_mae_sec', 'peak_mae_sec', 'offset_mae_sec', 'duration_mae_sec', 'selection_loss'):
            r[metric] = float(np.nanmean(pd.to_numeric(q[metric], errors='coerce')))
        rows.append(r)
    return pd.DataFrame(rows)
def coherent_incremental_stats(pm):
    transitions = [(ROW_BASE, ROW_A, '+A'), (ROW_A, ROW_AB, '+B'), (ROW_AB, ROW_ABC, '+C'), (ROW_ABC, ROW_FULL, '+LBR')]
    rows = []
    for reference, candidate, label in transitions:
        a = pm[pm['row'].astype(str) == reference].set_index('participant')
        b = pm[pm['row'].astype(str) == candidate].set_index('participant')
        common = sorted(set(a.index) & set(b.index))
        for metric in ('response_balanced_accuracy', 'onset_mae_sec', 'peak_mae_sec', 'offset_mae_sec', 'duration_mae_sec', 'selection_loss'):
            av = a.loc[common, metric].to_numpy(float)
            bv = b.loc[common, metric].to_numpy(float)
            if metric == 'response_balanced_accuracy':
                gain = bv - av
            else:
                gain = av - bv
            mean, lo, hi = bootstrap_ci(gain, stable_seed('COHERENT_INCREMENTAL_BOOT', reference, candidate, metric))
            p = signflip_p(gain, stable_seed('COHERENT_INCREMENTAL_SIGN', reference, candidate, metric))
            rows.append({
                'transition': label,
                'reference': reference,
                'candidate': candidate,
                'metric': metric,
                'n_participants': len(common),
                'mean_gain_positive_added_component_better': mean,
                'ci95_low': lo,
                'ci95_high': hi,
                'participants_better': int(np.sum(gain > 1e-12)),
                'participants_equal': int(np.sum(np.isclose(gain, 0.0))),
                'participants_worse': int(np.sum(gain < -1e-12)),
                'signflip_p': p
            })
    out = pd.DataFrame(rows)
    out['holm_p_4_timing_within_transition'] = np.nan
    timing = {'onset_mae_sec', 'peak_mae_sec', 'offset_mae_sec', 'duration_mae_sec'}
    for transition in out['transition'].drop_duplicates():
        ix = out.index[(out['transition'] == transition) & out['metric'].isin(timing)].tolist()
        out.loc[ix, 'holm_p_4_timing_within_transition'] = holm_adjust(out.loc[ix, 'signflip_p'].to_numpy(float))
    return out
def selected_frame_from_dicts(selected_by_outer):
    rows = []
    for target, selected in selected_by_outer.items():
        rows.append({'outer_target': target, 'row': ROW_BASE, 'global_cfg': 'FIXED_CONVENTIONAL_BASE', 'local_cfg': '', 'inner_selected_loss': np.nan})
        for row in (ROW_A, ROW_AB, ROW_ABC, ROW_FULL):
            item = selected[row]
            rows.append({
                'outer_target': target,
                'row': row,
                'global_cfg': canonical_cfg(item['global']),
                'local_cfg': canonical_cfg(item['local']) if item['local'] is not None else '',
                'inner_selected_loss': float(item['inner_loss'])
            })
    return pd.DataFrame(rows)
def try_full_merge(n4, out):
    ckdir = out / 'checkpoints'
    required = [(ckdir / f'{pid}_ROWS.csv.gz', ckdir / f'{pid}_SELECTED.json', ckdir / f'{pid}_INNER.csv.gz', ckdir / f'{pid}_AUDIT.csv.gz') for pid in ALL_PIDS]
    if not all((all((p.is_file() for p in bundle)) for bundle in required)):
        missing = [pid for pid, bundle in zip(ALL_PIDS, required) if not all((p.is_file() for p in bundle))]
        print('\nFULL MERGE: waiting for other shard.')
        print('Missing participants:', missing)
        return False
    lock = out / 'FULL_MERGE.lock'
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        print('\nFULL MERGE: another shard is already merging.')
        return False
    try:
        all_rows = pd.concat([pd.read_csv(ckdir / f'{pid}_ROWS.csv.gz') for pid in ALL_PIDS], ignore_index=True)
        all_inner = pd.concat([pd.read_csv(ckdir / f'{pid}_INNER.csv.gz') for pid in ALL_PIDS], ignore_index=True)
        all_audit = pd.concat([pd.read_csv(ckdir / f'{pid}_AUDIT.csv.gz') for pid in ALL_PIDS], ignore_index=True)
        selected_by_outer = {pid: json.loads((ckdir / f'{pid}_SELECTED.json').read_text(encoding='utf-8')) for pid in ALL_PIDS}
        pm = coherent_participant_metrics(n4, all_rows)
        summary = coherent_summary(pm)
        stats = coherent_incremental_stats(pm)
        selected = selected_frame_from_dicts(selected_by_outer)
        atomic_csv(all_rows, out / 'results' / 'FULL_B0_COHERENT_PER_TRIAL_SEED01.csv.gz')
        atomic_csv(pm, out / 'results' / 'FULL_B0_COHERENT_PARTICIPANT_METRICS_SEED01.csv')
        atomic_csv(summary, out / 'tables' / 'Table1_B0_COHERENT_ABLATION_SEED01.csv')
        atomic_csv(stats, out / 'tables' / 'Table2_B0_COHERENT_INCREMENTAL_STATS_SEED01.csv')
        atomic_csv(selected, out / 'tables' / 'Table3_B0_COHERENT_SELECTED_CONFIGS_SEED01.csv')
        atomic_csv(all_inner, out / 'audit' / 'FULL_INNER_CANDIDATE_SCORES_SEED01.csv.gz')
        atomic_csv(all_audit, out / 'audit' / 'FULL_FIT_MEMBERSHIP_AUDIT_SEED01.csv.gz')
        atomic_json({
            'version': COHERENT_VERSION,
            'repeat_seed': 1,
            'complete': True,
            'participants': list(ALL_PIDS),
            'n_participants': 26,
            'rows': list(ROW_ORDER),
            'all_five_rows_rerun': True,
            'old_current_ABC_reused': False,
            'old_current_HemoBoundary_reused': False,
            'seed_bank_reused': True,
            'outer_target_used_in_tuning': False,
            'inner_selection': '3-fold participant-grouped CV',
            'inner_training_bank': 'TRAIN',
            'inner_validation_bank': 'TEST',
            'LBR_source_participant_OOF_anchors': True,
            'A_grid': A_GRID,
            'AB_grid': AB_GRID,
            'ABC_grid': ABC_GRID,
            'LBR_grid': LBR_GRID_COHERENT
        }, out / 'audit' / 'FULL_RUN_MANIFEST.json')
        print('\n' + '=' * 180)
        print('PARTICIPANT-LEVEL COMPONENT EVALUATION')
        print('=' * 180)
        print(summary.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
        print('\n' + '=' * 180)
        print('FULL INCREMENTAL PAIRED STATISTICS')
        print('=' * 180)
        print(stats.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
        print('\nFULL MERGE: COMPLETE')
        return True
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
def coherent_self_tests(ref, n4):
    protected_self_tests(ref, n4)
    assert len(A_GRID) == 5
    assert len(AB_GRID) == 7
    assert len(ABC_GRID) == 9
    assert len(LBR_GRID_COHERENT) == 7
    assert len(ABC_GRID) * len(LBR_GRID_COHERENT) == 63
    n = 8
    F = np.zeros((n, len(n4.GRID), 9), float)
    h = coherent_canonical_hrf(n4.GRID)
    rng = np.random.default_rng(123)
    for i in range(n):
        F[i, :, 2] = (1.0 + 0.05 * i) * h + rng.normal(scale=0.02, size=len(n4.GRID))
    x = {'obs': np.zeros((n, 4 * len(n4.GRID)), float), 'state_feat': F}
    feat, times = coherent_glm_features_times(n4, x)
    assert feat.shape[0] == n
    assert times.shape == (n, 3)
    assert np.all(np.isfinite(feat))
    assert np.all(np.isfinite(times))
    assert np.all(times[:, 0] < times[:, 1])
    assert np.all(times[:, 1] < times[:, 2])
    print('Component ablation checks: PASS')
def parse_coherent_args():
    p = argparse.ArgumentParser()
    p.add_argument('--shin-root', default=None)
    p.add_argument('--experiment-root', default=None)
    p.add_argument('--output', default=None)
    p.add_argument('--max-participants', type=int, default=0, help='Smoke-test the first N targets owned by this shard.')
    p.add_argument('--force', action='store_true', help="Rerun only this shard's participant checkpoint bundles. The other shard is never deleted.")
    p.add_argument('--self-test-only', action='store_true')
    args, unknown = p.parse_known_args()
    if unknown:
        print('Ignoring notebook/runtime arguments:', unknown)
    return args
def main_coherent():
    args = parse_coherent_args()
    with tempfile.TemporaryDirectory(prefix='hemoboundary_coherent_b0_') as td:
        td = Path(td)
        ref = import_embedded(REFINED_ENGINE_SOURCE, '_coherent_b0_refined_engine', td)
        n4 = import_embedded(N4_ENGINE_SOURCE, '_coherent_b0_n4_engine', td)
        configure_repeat_seed(ref, n4, COHERENT_SEED)
        coherent_self_tests(ref, n4)
        if args.self_test_only:
            return
        shin_root = find_root(args.shin_root)
        experiment_root = Path(args.experiment_root) if args.experiment_root else shin_root / OUT_DIRNAME
        seed_dir = experiment_root / 'seed_01'
        cache_dir = seed_dir / 'fresh_cache'
        if not (cache_dir.is_dir() and all(((cache_dir / f'{pid}.npz').is_file() for pid in ALL_PIDS))):
            raise FileNotFoundError(f'The coherent pilot reuses the completed seed-1 bank only. Complete cache missing: {cache_dir}')
        out = Path(args.output) if args.output else shin_root / COHERENT_OUT_DIRNAME
        for sub in ('checkpoints', 'results', 'tables', 'audit'):
            (out / sub).mkdir(parents=True, exist_ok=True)
        caches0 = {pid: ref.load_cache(cache_dir / f'{pid}.npz') for pid in ALL_PIDS}
        caches = add_pid_weight(caches0)
        targets = list(SHARD_TARGETS)
        if args.max_participants > 0:
            targets = targets[:args.max_participants]
        if args.force:
            for pid in targets:
                for suffix in ('_ROWS.csv.gz', '_SELECTED.json', '_INNER.csv.gz', '_AUDIT.csv.gz'):
                    p = out / 'checkpoints' / f'{pid}{suffix}'
                    if p.is_file():
                        p.unlink()
        print('=' * 180)
        print('HEMOBOUNDARY — COMPONENT EVALUATION')
        print('=' * 180)
        print('Shard                       :', SHARD_NAME)
        print('Outer targets               :', targets)
        print('Outer evaluation            : participant LOSO')
        print('Source participants / target: 25')
        print('Inner selection             : 3-fold participant-grouped CV')
        print('Inner training              : source TRAIN banks')
        print('Inner validation            : inner-held participant TEST banks')
        print('Seed-1 bank                 : REUSED; no regeneration')
        print('All five model rows         : RERUN coherently')
        print('Old A+B+C reused            : NO')
        print('Old HemoBoundary reused     : NO')
        print('LBR residual anchors        : source-participant OOF')
        print('Outer target in tuning      : NEVER')
        print('Output                      :', out)
        print()
        shard_rows = []
        shard_inner = []
        shard_audit = []
        selected_by_outer = {}
        for oi, target in enumerate(targets, 1):
            ck = out / 'checkpoints'
            cp_rows = ck / f'{target}_ROWS.csv.gz'
            cp_sel = ck / f'{target}_SELECTED.json'
            cp_inner = ck / f'{target}_INNER.csv.gz'
            cp_audit = ck / f'{target}_AUDIT.csv.gz'
            if cp_rows.is_file() and cp_sel.is_file() and cp_inner.is_file() and cp_audit.is_file() and (not args.force):
                print(f'[{oi:02d}/{len(targets):02d}] {target}: RESUME', flush=True)
                shard_rows.append(pd.read_csv(cp_rows))
                shard_inner.append(pd.read_csv(cp_inner))
                shard_audit.append(pd.read_csv(cp_audit))
                selected_by_outer[target] = json.loads(cp_sel.read_text(encoding='utf-8'))
                continue
            print(f'[{oi:02d}/{len(targets):02d}] {target}: nested five-row ablation', flush=True)
            source_pids = [p for p in ALL_PIDS if p != target]
            selected, inner_scores, inner_audit = select_coherent_configs(n4, caches, target, source_pids)
            print('    +A   :', selected[ROW_A])
            print('    +B   :', selected[ROW_AB])
            print('    +C   :', selected[ROW_ABC])
            print('    +LBR :', selected[ROW_FULL])
            rows, outer_audit = outer_all_rows(n4, caches, target, selected)
            audit = pd.concat([inner_audit, outer_audit], ignore_index=True)
            atomic_csv(rows, cp_rows)
            atomic_csv(inner_scores, cp_inner)
            atomic_csv(audit, cp_audit)
            atomic_json(selected, cp_sel)
            shard_rows.append(rows)
            shard_inner.append(inner_scores)
            shard_audit.append(audit)
            selected_by_outer[target] = selected
        per_trial = pd.concat(shard_rows, ignore_index=True)
        pm = coherent_participant_metrics(n4, per_trial)
        summary = coherent_summary(pm)
        stats = coherent_incremental_stats(pm)
        selected_frame = selected_frame_from_dicts(selected_by_outer)
        atomic_csv(per_trial, out / 'results' / f'SHARD_{SHARD_NAME}_PER_TRIAL.csv.gz')
        atomic_csv(pm, out / 'results' / f'SHARD_{SHARD_NAME}_PARTICIPANT_METRICS.csv')
        atomic_csv(summary, out / 'tables' / f'SHARD_{SHARD_NAME}_Table1_ABLATION.csv')
        atomic_csv(stats, out / 'tables' / f'SHARD_{SHARD_NAME}_Table2_INCREMENTAL_STATS.csv')
        atomic_csv(selected_frame, out / 'tables' / f'SHARD_{SHARD_NAME}_Table3_SELECTED_CONFIGS.csv')
        atomic_csv(pd.concat(shard_inner, ignore_index=True), out / 'audit' / f'SHARD_{SHARD_NAME}_INNER_SCORES.csv.gz')
        atomic_csv(pd.concat(shard_audit, ignore_index=True), out / 'audit' / f'SHARD_{SHARD_NAME}_FIT_AUDIT.csv.gz')
        atomic_json({
            'version': COHERENT_VERSION,
            'shard': SHARD_NAME,
            'shard_targets': list(SHARD_TARGETS),
            'executed_targets': targets,
            'repeat_seed': 1,
            'all_five_rows_rerun': True,
            'old_ABC_reused': False,
            'old_HemoBoundary_reused': False,
            'seed_bank_reused': True,
            'outer_target_used_in_tuning': False,
            'LBR_source_participant_OOF_anchors': True
        }, out / 'audit' / f'SHARD_{SHARD_NAME}_MANIFEST.json')
        print('\n' + '=' * 180)
        print(f'SHARD {SHARD_NAME} — ABLATION SUMMARY')
        print('=' * 180)
        print(summary.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
        print('\nAttempting full 26-participant merge...')
        try_full_merge(n4, out)
HIERARCHICAL_VERSION = 'HEMOBOUNDARY_B0_STRICT_HIERARCHICAL_SEED01_FULLY_NESTED_V1_20260916'
HIERARCHICAL_OUT_DIRNAME = '98_B0_STRICT_HIERARCHICAL_SEED01_FULLY_NESTED_V1'
HIERARCHICAL_A_GRID = [
    dict(state_C=0.01, timing_alpha=10.0),
    dict(state_C=0.003, timing_alpha=10.0),
    dict(state_C=0.03, timing_alpha=10.0),
    dict(state_C=0.01, timing_alpha=3.0),
    dict(state_C=0.01, timing_alpha=30.0)
]
HIERARCHICAL_B_CHANGE_FRAC_GRID = (0.2, 0.3, 0.4)
HIERARCHICAL_C_THRESHOLD_GRID = (0.4, 0.5, 0.6)
HIERARCHICAL_LBR_GRID = [
    dict(radius=1.0, local_alpha=100.0, strength=0.5),
    dict(radius=0.5, local_alpha=100.0, strength=0.5),
    dict(radius=1.5, local_alpha=100.0, strength=0.5),
    dict(radius=1.0, local_alpha=30.0, strength=0.5),
    dict(radius=1.0, local_alpha=300.0, strength=0.5),
    dict(radius=1.0, local_alpha=100.0, strength=0.25),
    dict(radius=1.0, local_alpha=100.0, strength=0.75)
]
def hierarchical_inner_contexts(n4, caches, outer_target, outer_source_pids):
    """
    Build one fixed set of participant-grouped inner folds.

    Every hierarchical stage uses the SAME inner folds. All fitting/tuning is
    restricted to the outer-source participants.
    """
    folds = n4.grouped_folds(outer_source_pids, stable_seed('STRICT_HIERARCHICAL_INNER', COHERENT_SEED, outer_target, *outer_source_pids), INNER_FOLDS)
    flat = sorted((p for fold in folds for p in fold))
    if flat != sorted(outer_source_pids):
        raise RuntimeError(f'{outer_target}: inner participant coverage failed.')
    contexts = []
    for fi, held_pids in enumerate(folds, 1):
        tr_pids = [p for p in outer_source_pids if p not in held_pids]
        contexts.append({
            'fold': fi,
            'held_pids': list(held_pids),
            'train_pids': tr_pids,
            'train': training_rows(n4, caches, tr_pids),
            'validation_by_pid': {pid: caches[pid]['test'] for pid in held_pids},
            'forbidden': set(held_pids) | {outer_target}
        })
    return contexts
def hierarchical_choose(score_dict, ordered_candidates):
    ranked = []
    for idx, candidate in enumerate(ordered_candidates):
        key = canonical_cfg(candidate)
        vals = score_dict[key]
        if not len(vals):
            raise RuntimeError('Hierarchical candidate has no validation scores.')
        ranked.append((float(np.mean(vals)), idx, dict(candidate)))
    ranked.sort(key=lambda x: (x[0], x[1]))
    return (ranked[0][2], float(ranked[0][0]))
def select_coherent_configs(n4, caches, outer_target, outer_source_pids):
    """HIERARCHICAL FULLY-NESTED SELECTION."""
    contexts = hierarchical_inner_contexts(n4, caches, outer_target, outer_source_pids)
    score_rows = []
    audit_rows = []
    scores_A = {canonical_cfg(cfg): [] for cfg in HIERARCHICAL_A_GRID}
    for ctx in contexts:
        base_audit = n4.FitAudit(ctx['train_pids'], ctx['forbidden'], f"strict/outer={outer_target}/inner={ctx['fold']}/BASE_for_A")
        base = fit_coherent_base(n4, ctx['train'], base_audit, stage_prefix='STRICT_INNER_BASE_FOR_A')
        audit_rows.extend(base_audit.rows)
        for ci, cfg in enumerate(HIERARCHICAL_A_GRID):
            audit = n4.FitAudit(ctx['train_pids'], ctx['forbidden'], f"strict/outer={outer_target}/inner={ctx['fold']}/A={ci}")
            model = fit_coherent_A(n4, ctx['train'], cfg, audit, stage_prefix='STRICT_INNER_A')
            audit_rows.extend(audit.rows)
            loss = mean_e2e_loss(n4, ctx['validation_by_pid'], lambda x, b=base, m=model: predict_row_A(n4, b, m, x))
            scores_A[canonical_cfg(cfg)].append(loss)
            score_rows.append({
                'outer_target': outer_target,
                'inner_fold': ctx['fold'],
                'row': ROW_A,
                'selection_stage': 'A_ONLY',
                'inherited_cfg': 'FIXED_BASE',
                'candidate_cfg': canonical_cfg(cfg),
                'local_cfg': '',
                'loss': loss,
                'inner_validation_pids': '|'.join(ctx['held_pids'])
            })
    selected_A, loss_A = hierarchical_choose(scores_A, HIERARCHICAL_A_GRID)
    hierarchical_B_candidates = [{**selected_A, 'change_frac': float(frac)} for frac in HIERARCHICAL_B_CHANGE_FRAC_GRID]
    scores_B = {canonical_cfg(cfg): [] for cfg in hierarchical_B_candidates}
    for ctx in contexts:
        base_audit = n4.FitAudit(ctx['train_pids'], ctx['forbidden'], f"strict/outer={outer_target}/inner={ctx['fold']}/BASE_for_B")
        base = fit_coherent_base(n4, ctx['train'], base_audit, stage_prefix='STRICT_INNER_BASE_FOR_B')
        audit_rows.extend(base_audit.rows)
        for ci, cfg in enumerate(hierarchical_B_candidates):
            audit = n4.FitAudit(ctx['train_pids'], ctx['forbidden'], f"strict/outer={outer_target}/inner={ctx['fold']}/B={ci}")
            model = fit_coherent_AB(n4, ctx['train'], cfg, audit, stage_prefix='STRICT_INNER_AB')
            audit_rows.extend(audit.rows)
            loss = mean_e2e_loss(n4, ctx['validation_by_pid'], lambda x, b=base, m=model: predict_row_AB(n4, b, m, x))
            scores_B[canonical_cfg(cfg)].append(loss)
            score_rows.append({
                'outer_target': outer_target,
                'inner_fold': ctx['fold'],
                'row': ROW_AB,
                'selection_stage': 'B_ONLY',
                'inherited_cfg': canonical_cfg(selected_A),
                'candidate_cfg': canonical_cfg(cfg),
                'local_cfg': '',
                'loss': loss,
                'inner_validation_pids': '|'.join(ctx['held_pids'])
            })
    selected_AB, loss_AB = hierarchical_choose(scores_B, hierarchical_B_candidates)
    for key in ('state_C', 'timing_alpha'):
        if not np.isclose(float(selected_AB[key]), float(selected_A[key])):
            raise RuntimeError(f'Component B inherited settings mismatch for {key}.')
    hierarchical_C_candidates = [{**selected_AB, 'response_threshold': float(threshold)} for threshold in HIERARCHICAL_C_THRESHOLD_GRID]
    scores_C = {canonical_cfg(cfg): [] for cfg in hierarchical_C_candidates}
    for ctx in contexts:
        for ci, cfg in enumerate(hierarchical_C_candidates):
            audit = n4.FitAudit(ctx['train_pids'], ctx['forbidden'], f"strict/outer={outer_target}/inner={ctx['fold']}/C={ci}")
            model = fit_coherent_ABC(n4, ctx['train'], cfg, audit, stage_prefix='STRICT_INNER_ABC')
            audit_rows.extend(audit.rows)
            loss = mean_e2e_loss(n4, ctx['validation_by_pid'], lambda x, m=model: m.predict(x))
            scores_C[canonical_cfg(cfg)].append(loss)
            score_rows.append({
                'outer_target': outer_target,
                'inner_fold': ctx['fold'],
                'row': ROW_ABC,
                'selection_stage': 'C_ONLY',
                'inherited_cfg': canonical_cfg(selected_AB),
                'candidate_cfg': canonical_cfg(cfg),
                'local_cfg': '',
                'loss': loss,
                'inner_validation_pids': '|'.join(ctx['held_pids'])
            })
    selected_ABC, loss_ABC = hierarchical_choose(scores_C, hierarchical_C_candidates)
    for key in ('state_C', 'timing_alpha', 'change_frac'):
        if not np.isclose(float(selected_ABC[key]), float(selected_AB[key])):
            raise RuntimeError(f'Component C inherited settings mismatch for {key}.')
    scores_LBR = {canonical_cfg(cfg): [] for cfg in HIERARCHICAL_LBR_GRID}
    for ctx in contexts:
        for li, lcfg in enumerate(HIERARCHICAL_LBR_GRID):
            audit = n4.FitAudit(ctx['train_pids'], ctx['forbidden'], f"strict/outer={outer_target}/inner={ctx['fold']}/LBR={li}")
            model = fit_coherent_full(
                n4,
                ctx['train'],
                selected_ABC,
                lcfg,
                audit,
                outer_forbidden=ctx['forbidden'],
                context=f"strict/outer={outer_target}/inner={ctx['fold']}/LBR={li}"
            )
            audit_rows.extend(audit.rows)
            loss = mean_e2e_loss(n4, ctx['validation_by_pid'], lambda x, m=model: m.predict(x))
            scores_LBR[canonical_cfg(lcfg)].append(loss)
            score_rows.append({
                'outer_target': outer_target,
                'inner_fold': ctx['fold'],
                'row': ROW_FULL,
                'selection_stage': 'LBR_ONLY',
                'inherited_cfg': canonical_cfg(selected_ABC),
                'candidate_cfg': canonical_cfg(selected_ABC),
                'local_cfg': canonical_cfg(lcfg),
                'loss': loss,
                'inner_validation_pids': '|'.join(ctx['held_pids'])
            })
    selected_LBR, loss_FULL = hierarchical_choose(scores_LBR, HIERARCHICAL_LBR_GRID)
    selected = {
        ROW_A: {'global': dict(selected_A), 'local': None, 'inner_loss': float(loss_A), 'selection_stage': 'A_ONLY'},
        ROW_AB: {'global': dict(selected_AB), 'local': None, 'inner_loss': float(loss_AB), 'selection_stage': 'B_ONLY_INHERITS_A'},
        ROW_ABC: {'global': dict(selected_ABC), 'local': None, 'inner_loss': float(loss_ABC), 'selection_stage': 'C_ONLY_INHERITS_AB'},
        ROW_FULL: {'global': dict(selected_ABC), 'local': dict(selected_LBR), 'inner_loss': float(loss_FULL), 'selection_stage': 'LBR_ONLY_INHERITS_ABC'}
    }
    if canonical_cfg(selected[ROW_FULL]['global']) != canonical_cfg(selected[ROW_ABC]['global']):
        raise RuntimeError('The final model changed the selected A+B+C parent.')
    return (selected, pd.DataFrame(score_rows), pd.DataFrame(audit_rows))
def hierarchical_self_tests(ref, n4):
    protected_self_tests(ref, n4)
    assert len(HIERARCHICAL_A_GRID) == 5
    assert HIERARCHICAL_B_CHANGE_FRAC_GRID == (0.2, 0.3, 0.4)
    assert HIERARCHICAL_C_THRESHOLD_GRID == (0.4, 0.5, 0.6)
    assert len(HIERARCHICAL_LBR_GRID) == 7
    selected_A = dict(state_C=0.01, timing_alpha=10.0)
    selected_AB = {**selected_A, 'change_frac': 0.3}
    selected_ABC = {**selected_AB, 'response_threshold': 0.5}
    for key in ('state_C', 'timing_alpha'):
        assert selected_AB[key] == selected_A[key]
    for key in ('state_C', 'timing_alpha', 'change_frac'):
        assert selected_ABC[key] == selected_AB[key]
    print('Configuration selection checks: PASS')
def hierarchical_try_full_merge(n4, out):
    """
    Same result merge as the coherent pilot, but with a hierarchical
    manifest describing exactly which parameters each stage was allowed to tune.
    """
    ckdir = out / 'checkpoints'
    required = [(ckdir / f'{pid}_ROWS.csv.gz', ckdir / f'{pid}_SELECTED.json', ckdir / f'{pid}_INNER.csv.gz', ckdir / f'{pid}_AUDIT.csv.gz') for pid in ALL_PIDS]
    if not all((all((p.is_file() for p in bundle)) for bundle in required)):
        missing = [pid for pid, bundle in zip(ALL_PIDS, required) if not all((p.is_file() for p in bundle))]
        print('\nCOMPONENT SUMMARY MERGE: waiting for other shard.')
        print('Missing participants:', missing)
        return False
    lock = out / 'FULL_STRICT_MERGE.lock'
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        print('\nCOMPONENT SUMMARY MERGE: another shard is already merging.')
        return False
    try:
        all_rows = pd.concat([pd.read_csv(ckdir / f'{pid}_ROWS.csv.gz') for pid in ALL_PIDS], ignore_index=True)
        all_inner = pd.concat([pd.read_csv(ckdir / f'{pid}_INNER.csv.gz') for pid in ALL_PIDS], ignore_index=True)
        all_audit = pd.concat([pd.read_csv(ckdir / f'{pid}_AUDIT.csv.gz') for pid in ALL_PIDS], ignore_index=True)
        selected_by_outer = {pid: json.loads((ckdir / f'{pid}_SELECTED.json').read_text(encoding='utf-8')) for pid in ALL_PIDS}
        inheritance_rows = []
        for pid, selected in selected_by_outer.items():
            A = selected[ROW_A]['global']
            AB = selected[ROW_AB]['global']
            ABC = selected[ROW_ABC]['global']
            FULL = selected[ROW_FULL]['global']
            a_to_b = all((np.isclose(float(AB[key]), float(A[key])) for key in ('state_C', 'timing_alpha')))
            b_to_c = all((np.isclose(float(ABC[key]), float(AB[key])) for key in ('state_C', 'timing_alpha', 'change_frac')))
            c_to_full = canonical_cfg(ABC) == canonical_cfg(FULL)
            inheritance_rows.append({
                'outer_target': pid,
                'A_to_B_inheritance_pass': bool(a_to_b),
                'B_to_C_inheritance_pass': bool(b_to_c),
                'C_to_LBR_global_parent_lock_pass': bool(c_to_full)
            })
        inheritance = pd.DataFrame(inheritance_rows)
        if not inheritance[['A_to_B_inheritance_pass', 'B_to_C_inheritance_pass', 'C_to_LBR_global_parent_lock_pass']].to_numpy(bool).all():
            raise RuntimeError('Hierarchical setting inheritance verification failed.')
        pm = coherent_participant_metrics(n4, all_rows)
        summary = coherent_summary(pm)
        stats = coherent_incremental_stats(pm)
        selected_frame = selected_frame_from_dicts(selected_by_outer)
        atomic_csv(all_rows, out / 'results' / 'FULL_B0_STRICT_HIERARCHICAL_PER_TRIAL_SEED01.csv.gz')
        atomic_csv(pm, out / 'results' / 'FULL_B0_STRICT_HIERARCHICAL_PARTICIPANT_METRICS_SEED01.csv')
        atomic_csv(summary, out / 'tables' / 'Table1_B0_STRICT_HIERARCHICAL_ABLATION_SEED01.csv')
        atomic_csv(stats, out / 'tables' / 'Table2_B0_STRICT_HIERARCHICAL_INCREMENTAL_STATS_SEED01.csv')
        atomic_csv(selected_frame, out / 'tables' / 'Table3_B0_STRICT_HIERARCHICAL_SELECTED_CONFIGS_SEED01.csv')
        atomic_csv(inheritance, out / 'audit' / 'STRICT_HIERARCHY_INHERITANCE_AUDIT.csv')
        atomic_csv(all_inner, out / 'audit' / 'FULL_INNER_CANDIDATE_SCORES_SEED01.csv.gz')
        atomic_csv(all_audit, out / 'audit' / 'FULL_FIT_MEMBERSHIP_AUDIT_SEED01.csv.gz')
        atomic_json({
            'version': HIERARCHICAL_VERSION,
            'repeat_seed': 1,
            'complete': True,
            'participants': list(ALL_PIDS),
            'n_participants': 26,
            'outer_evaluation': 'participant LOSO',
            'inner_selection': '3-fold participant-grouped CV',
            'inner_training_bank': 'TRAIN',
            'inner_validation_bank': 'TEST',
            'outer_target_used_in_tuning': False,
            'seed_bank_reused': True,
            'seed_bank_regenerated': False,
            'all_five_rows_rerun': True,
            'old_current_ABC_reused': False,
            'old_current_HemoBoundary_reused': False,
            'hierarchical_selection': {'A': 'tune state_C + timing_alpha', 'B': 'inherit A exactly; tune change_frac only', 'C': 'inherit A+B exactly; add response classifier and tune response_threshold only', 'LBR': 'inherit ABC exactly; tune radius/local_alpha/strength only'},
            'A_grid': HIERARCHICAL_A_GRID,
            'B_change_frac_grid': list(HIERARCHICAL_B_CHANGE_FRAC_GRID),
            'C_threshold_grid': list(HIERARCHICAL_C_THRESHOLD_GRID),
            'LBR_grid': HIERARCHICAL_LBR_GRID,
            'LBR_source_participant_OOF_anchors': True,
            'strict_inheritance_audit_pass': True
        }, out / 'audit' / 'FULL_RUN_MANIFEST.json')
        print('\n' + '=' * 180)
        print('PARTICIPANT-LEVEL HEMOBOUNDARY EVALUATION')
        print('=' * 180)
        print(summary.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
        print('\n' + '=' * 180)
        print('INCREMENTAL COMPONENT STATISTICS')
        print('=' * 180)
        print(stats.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
        print('\nHIERARCHICAL SELECTION VERIFIED: 26/26')
        return True
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
def parse_hierarchical_args():
    p = argparse.ArgumentParser()
    p.add_argument('--shin-root', default=None)
    p.add_argument('--experiment-root', default=None)
    p.add_argument('--output', default=None)
    p.add_argument('--max-participants', type=int, default=0)
    p.add_argument('--force', action='store_true', help="Rerun only this shard's hierarchical checkpoints; never delete the other shard.")
    p.add_argument('--self-test-only', action='store_true')
    args, unknown = p.parse_known_args()
    if unknown:
        print('Ignoring notebook/runtime arguments:', unknown)
    return args
def main_hierarchical():
    args = parse_hierarchical_args()
    with tempfile.TemporaryDirectory(prefix='hemoboundary_strict_hierarchical_b0_') as td:
        td = Path(td)
        ref = import_embedded(REFINED_ENGINE_SOURCE, '_strict_b0_refined_engine', td)
        n4 = import_embedded(N4_ENGINE_SOURCE, '_strict_b0_n4_engine', td)
        configure_repeat_seed(ref, n4, COHERENT_SEED)
        hierarchical_self_tests(ref, n4)
        if args.self_test_only:
            return
        shin_root = find_root(args.shin_root)
        source_experiment_root = Path(args.experiment_root) if args.experiment_root else shin_root / OUT_DIRNAME
        cache_dir = source_experiment_root / 'seed_01' / 'fresh_cache'
        if not (cache_dir.is_dir() and all(((cache_dir / f'{pid}.npz').is_file() for pid in ALL_PIDS))):
            raise FileNotFoundError(f'The initial evaluation requires the existing first-seed bank. Bank not found: {cache_dir}')
        out = Path(args.output) if args.output else shin_root / HIERARCHICAL_OUT_DIRNAME
        for sub in ('checkpoints', 'results', 'tables', 'audit'):
            (out / sub).mkdir(parents=True, exist_ok=True)
        caches0 = {pid: ref.load_cache(cache_dir / f'{pid}.npz') for pid in ALL_PIDS}
        caches = add_pid_weight(caches0)
        targets = list(SHARD_TARGETS)
        if args.max_participants > 0:
            targets = targets[:args.max_participants]
        if args.force:
            for pid in targets:
                for suffix in ('_ROWS.csv.gz', '_SELECTED.json', '_INNER.csv.gz', '_AUDIT.csv.gz'):
                    p = out / 'checkpoints' / f'{pid}{suffix}'
                    if p.is_file():
                        p.unlink()
        print('=' * 180)
        print('HEMOBOUNDARY — HIERARCHICAL Base -> +A -> +B -> +C -> +LBR')
        print('=' * 180)
        print('Seed                         : 1')
        print('Shard                        :', SHARD_NAME)
        print('Outer targets                :', targets)
        print('Outer evaluation             : participant LOSO')
        print('Inner selection              : 3-fold participant-grouped CV')
        print('A tunes                      : state_C + timing_alpha')
        print('B tunes                      : change_frac ONLY; A locked')
        print('C tunes                      : response_threshold ONLY; A+B locked')
        print('LBR tunes                    : radius/alpha/strength ONLY; ABC locked')
        print('All five rows                : RERUN')
        print('Old ABC / old Full reused    : NO')
        print('LBR residual anchors         : source-participant OOF')
        print('Outer target in tuning       : NEVER')
        print('Seed-1 bank                  : REUSED, not regenerated')
        print('Output                       :', out)
        print()
        shard_rows = []
        shard_inner = []
        shard_audit = []
        selected_by_outer = {}
        for oi, target in enumerate(targets, 1):
            ck = out / 'checkpoints'
            cp_rows = ck / f'{target}_ROWS.csv.gz'
            cp_sel = ck / f'{target}_SELECTED.json'
            cp_inner = ck / f'{target}_INNER.csv.gz'
            cp_audit = ck / f'{target}_AUDIT.csv.gz'
            if cp_rows.is_file() and cp_sel.is_file() and cp_inner.is_file() and cp_audit.is_file() and (not args.force):
                print(f'[{oi:02d}/{len(targets):02d}] {target}: RESUME', flush=True)
                shard_rows.append(pd.read_csv(cp_rows))
                shard_inner.append(pd.read_csv(cp_inner))
                shard_audit.append(pd.read_csv(cp_audit))
                selected_by_outer[target] = json.loads(cp_sel.read_text(encoding='utf-8'))
                continue
            print(f'[{oi:02d}/{len(targets):02d}] {target}: hierarchical configuration selection', flush=True)
            source_pids = [p for p in ALL_PIDS if p != target]
            selected, inner_scores, inner_audit = select_coherent_configs(n4, caches, target, source_pids)
            print('    +A   :', selected[ROW_A])
            print('    +B   :', selected[ROW_AB])
            print('    +C   :', selected[ROW_ABC])
            print('    +LBR :', selected[ROW_FULL])
            rows, outer_audit = outer_all_rows(n4, caches, target, selected)
            combined_audit = pd.concat([inner_audit, outer_audit], ignore_index=True)
            atomic_csv(rows, cp_rows)
            atomic_csv(inner_scores, cp_inner)
            atomic_csv(combined_audit, cp_audit)
            atomic_json(selected, cp_sel)
            shard_rows.append(rows)
            shard_inner.append(inner_scores)
            shard_audit.append(combined_audit)
            selected_by_outer[target] = selected
        per_trial = pd.concat(shard_rows, ignore_index=True)
        pm = coherent_participant_metrics(n4, per_trial)
        summary = coherent_summary(pm)
        stats = coherent_incremental_stats(pm)
        selected_frame = selected_frame_from_dicts(selected_by_outer)
        atomic_csv(per_trial, out / 'results' / f'SHARD_{SHARD_NAME}_PER_TRIAL.csv.gz')
        atomic_csv(pm, out / 'results' / f'SHARD_{SHARD_NAME}_PARTICIPANT_METRICS.csv')
        atomic_csv(summary, out / 'tables' / f'SHARD_{SHARD_NAME}_Table1_ABLATION.csv')
        atomic_csv(stats, out / 'tables' / f'SHARD_{SHARD_NAME}_Table2_INCREMENTAL_STATS.csv')
        atomic_csv(selected_frame, out / 'tables' / f'SHARD_{SHARD_NAME}_Table3_SELECTED_CONFIGS.csv')
        atomic_csv(pd.concat(shard_inner, ignore_index=True), out / 'audit' / f'SHARD_{SHARD_NAME}_INNER_SCORES.csv.gz')
        atomic_csv(pd.concat(shard_audit, ignore_index=True), out / 'audit' / f'SHARD_{SHARD_NAME}_FIT_AUDIT.csv.gz')
        atomic_json({
            'version': HIERARCHICAL_VERSION,
            'repeat_seed': 1,
            'shard': SHARD_NAME,
            'targets': targets,
            'strict_hierarchical_selection': True,
            'A_tunes': ['state_C', 'timing_alpha'],
            'B_inherits_A': True,
            'B_tunes': ['change_frac'],
            'C_inherits_AB': True,
            'C_tunes': ['response_threshold'],
            'LBR_inherits_ABC': True,
            'LBR_tunes': ['radius', 'local_alpha', 'strength'],
            'all_five_rows_rerun': True,
            'old_ABC_reused': False,
            'old_HemoBoundary_reused': False,
            'seed_bank_reused': True,
            'outer_target_used_in_tuning': False,
            'LBR_source_participant_OOF_anchors': True
        }, out / 'audit' / f'SHARD_{SHARD_NAME}_MANIFEST.json')
        print('\n' + '=' * 180)
        print(f'PARTICIPANT SUBSET {SHARD_NAME} — CURRENT SUMMARY')
        print('=' * 180)
        print(summary.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
        print('\nAttempting 26-participant merge...')
        hierarchical_try_full_merge(n4, out)
SHARD_NAME = 'VP001_VP013'
SHARD_TARGETS = ('VP001', 'VP002', 'VP003', 'VP004', 'VP005', 'VP006', 'VP007', 'VP008', 'VP009', 'VP010', 'VP011', 'VP012', 'VP013')
COHERENT_VERSION = HIERARCHICAL_VERSION
COHERENT_OUT_DIRNAME = HIERARCHICAL_OUT_DIRNAME
import gc
MULTISEED_HIERARCHICAL_VERSION = 'HEMOBOUNDARY_B0_STRICT_HIERARCHICAL_10SEEDS_V1_20260916'
MULTISEED_HIERARCHICAL_OUT_DIRNAME = '99_B0_STRICT_HIERARCHICAL_10SEEDS_V1'
RUN_PART_NAME = 'PART1_SEEDS_02_04'
RUN_SEEDS = (2, 3, 4)
EXPECTED_RUN_PART_NAME = 'PART1_SEEDS_02_04'
EXPECTED_RUN_SEEDS = (2, 3, 4)

def hierarchical_repeat_seed_dir(parent: Path, seed: int) -> Path:
    return parent / f'seed_{int(seed):02d}'
def hierarchical_repeat_complete_marker(parent: Path, seed: int) -> Path:
    return hierarchical_repeat_seed_dir(parent, seed) / 'audit' / 'SEED_COMPLETE.json'
def hierarchical_repeat_inheritance_audit(selected_by_outer):
    rows = []
    for pid in ALL_PIDS:
        selected = selected_by_outer[pid]
        A = selected[ROW_A]['global']
        AB = selected[ROW_AB]['global']
        ABC = selected[ROW_ABC]['global']
        FULL = selected[ROW_FULL]['global']
        LBR = selected[ROW_FULL]['local']
        a_to_b = all((np.isclose(float(AB[key]), float(A[key])) for key in ('state_C', 'timing_alpha')))
        b_to_c = all((np.isclose(float(ABC[key]), float(AB[key])) for key in ('state_C', 'timing_alpha', 'change_frac')))
        c_to_lbr = canonical_cfg(ABC) == canonical_cfg(FULL)
        rows.append({
            'participant': pid,
            'A_cfg': canonical_cfg(A),
            'AB_cfg': canonical_cfg(AB),
            'ABC_cfg': canonical_cfg(ABC),
            'LBR_global_parent': canonical_cfg(FULL),
            'LBR_cfg': canonical_cfg(LBR),
            'A_to_B_inheritance_pass': bool(a_to_b),
            'AB_to_C_inheritance_pass': bool(b_to_c),
            'ABC_to_LBR_parent_lock_pass': bool(c_to_lbr),
            'pass': bool(a_to_b and b_to_c and c_to_lbr)
        })
    out = pd.DataFrame(rows)
    if not out['pass'].all():
        raise RuntimeError('Hierarchical setting inheritance verification failed:\n' + out[~out['pass']].to_string(index=False))
    return out
def hierarchical_repeat_selected_frame(selected_by_outer):
    rows = []
    for pid in ALL_PIDS:
        selected = selected_by_outer[pid]
        rows.append({
            'participant': pid,
            'row': ROW_BASE,
            'configuration': ROW_LABELS[ROW_BASE],
            'global_cfg': 'FIXED_CONVENTIONAL_BASE',
            'local_cfg': '',
            'inner_selected_loss': np.nan,
            'selection_stage': 'FIXED_BASE'
        })
        for row in (ROW_A, ROW_AB, ROW_ABC, ROW_FULL):
            item = selected[row]
            rows.append({
                'participant': pid,
                'row': row,
                'configuration': ROW_LABELS[row],
                'global_cfg': canonical_cfg(item['global']),
                'local_cfg': canonical_cfg(item['local']) if item.get('local') is not None else '',
                'inner_selected_loss': float(item['inner_loss']),
                'selection_stage': item.get('selection_stage', '')
            })
    return pd.DataFrame(rows)
def merge_one_hierarchical_repeat(n4, seed_out: Path, seed: int):
    """
    Merge one complete 26-participant repeat.
    No fitting or selection is performed here.
    """
    ck = seed_out / 'checkpoints'
    required = []
    for pid in ALL_PIDS:
        bundle = [ck / f'{pid}_ROWS.csv.gz', ck / f'{pid}_SELECTED.json', ck / f'{pid}_INNER.csv.gz', ck / f'{pid}_AUDIT.csv.gz']
        if not all((p.is_file() for p in bundle)):
            return False
        required.append((pid, bundle))
    all_rows = pd.concat([pd.read_csv(bundle[0]) for _pid, bundle in required], ignore_index=True)
    all_inner = pd.concat([pd.read_csv(bundle[2]) for _pid, bundle in required], ignore_index=True)
    all_audit = pd.concat([pd.read_csv(bundle[3]) for _pid, bundle in required], ignore_index=True)
    selected_by_outer = {pid: json.loads(bundle[1].read_text(encoding='utf-8')) for pid, bundle in required}
    inheritance = hierarchical_repeat_inheritance_audit(selected_by_outer)
    pm = coherent_participant_metrics(n4, all_rows)
    summary = coherent_summary(pm)
    stats = coherent_incremental_stats(pm)
    selected = hierarchical_repeat_selected_frame(selected_by_outer)
    for df in (all_rows, all_inner, all_audit, inheritance, pm, summary, stats, selected):
        if 'repeat_seed' not in df.columns:
            df.insert(0, 'repeat_seed', int(seed))
    atomic_csv(all_rows, seed_out / 'results' / 'STRICT_HIERARCHICAL_PER_TRIAL.csv.gz')
    atomic_csv(pm, seed_out / 'results' / 'STRICT_HIERARCHICAL_PARTICIPANT_METRICS.csv')
    atomic_csv(summary, seed_out / 'tables' / 'Table1_STRICT_HIERARCHICAL_ABLATION.csv')
    atomic_csv(stats, seed_out / 'tables' / 'Table2_STRICT_INCREMENTAL_STATS.csv')
    atomic_csv(selected, seed_out / 'tables' / 'Table3_STRICT_SELECTED_CONFIGS.csv')
    atomic_csv(inheritance, seed_out / 'audit' / 'STRICT_HIERARCHY_INHERITANCE_AUDIT.csv')
    atomic_csv(all_inner, seed_out / 'audit' / 'ALL_INNER_CANDIDATE_SCORES.csv.gz')
    atomic_csv(all_audit, seed_out / 'audit' / 'ALL_FIT_MEMBERSHIP_AUDIT.csv.gz')
    atomic_json({
        'version': MULTISEED_HIERARCHICAL_VERSION,
        'repeat_seed': int(seed),
        'complete': True,
        'n_outer_participants': 26,
        'outer_evaluation': 'participant LOSO',
        'inner_selection': '3-fold participant-grouped CV',
        'outer_target_used_in_tuning': False,
        'fresh_seed_specific_RAW_D_SCREEN_bank': True,
        'strict_hierarchical_selection': True,
        'A_tunes': ['state_C', 'timing_alpha'],
        'B_inherits_A': True,
        'B_tunes': ['change_frac'],
        'C_inherits_AB': True,
        'C_tunes': ['response_threshold'],
        'LBR_inherits_ABC': True,
        'LBR_tunes': ['radius', 'local_alpha', 'strength'],
        'LBR_source_participant_OOF_anchors': True,
        'strict_inheritance_audit_pass': bool(inheritance['pass'].all())
    }, seed_out / 'audit' / 'SEED_COMPLETE.json')
    print('\n' + '=' * 180)
    print(f'SEED {seed:02d} — FINAL COMPONENT EVALUATION')
    print('=' * 180)
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print('\n' + '=' * 180)
    print(f'SEED {seed:02d} — INCREMENTAL PAIRED RESULTS')
    print('=' * 180)
    print(stats.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print(f'\nSEED {seed:02d}: INHERITED SETTINGS VERIFIED 26/26')
    return True
def run_one_hierarchical_repeat(ref, n4, shin_root: Path, fmap, parent_out: Path, seed: int, args):
    """
    Run one fresh robustness repeat using the frozen initial evaluation architecture.
    """
    global COHERENT_SEED
    COHERENT_SEED = int(seed)
    configure_repeat_seed(ref, n4, int(seed))
    seed_out = hierarchical_repeat_seed_dir(parent_out, seed)
    complete_marker = hierarchical_repeat_complete_marker(parent_out, seed)
    if complete_marker.is_file() and (not args.force) and (args.max_participants <= 0):
        print(f'\nSEED {seed:02d}: COMPLETE marker found — skipping.')
        return True
    if args.force and seed_out.exists():
        print(f'SEED {seed:02d}: --force removes ONLY {seed_out}')
        shutil.rmtree(seed_out)
    for sub in ('fresh_cache', 'checkpoints', 'results', 'tables', 'audit'):
        (seed_out / sub).mkdir(parents=True, exist_ok=True)
    print('\n' + '=' * 180)
    print(f'HEMOBOUNDARY — HEMODYNAMIC RESPONSE EVALUATION SEED {seed:02d}')
    print('=' * 180)
    print('Runner part                   :', RUN_PART_NAME)
    print('Assigned seeds                :', list(RUN_SEEDS))
    print('Current seed                  :', int(seed))
    print('Outer targets                 : ALL 26 participants')
    print('Outer evaluation              : participant LOSO')
    print('Inner selection               : 3-fold participant-grouped CV')
    print('A                             : tune state_C + timing_alpha')
    print('B                             : inherit A; tune change_frac only')
    print('C                             : inherit A+B; tune response_threshold only')
    print('LBR                           : inherit ABC; tune radius/alpha/strength only')
    print('LBR residual anchors          : source-participant OOF')
    print('Outer target used in tuning   : NEVER')
    print('Fresh RAW_D_SCREEN bank       : YES')
    print('Seed-specific inner folds     : YES')
    print('Seed-specific LBR OOF folds   : YES')
    print('Seed-specific estimator state : YES')
    print('Saved predictions reused : NO')
    print('Output                        :', seed_out)
    caches0 = ref.prepare_all_caches(fmap, seed_out / 'fresh_cache', mode='B')
    caches = add_pid_weight(caches0)
    cache_audit = pd.DataFrame([{
        'repeat_seed': int(seed),
        'participant': pid,
        'path': str(seed_out / 'fresh_cache' / f'{pid}.npz'),
        'sha256': file_sha256(seed_out / 'fresh_cache' / f'{pid}.npz'),
        'train_rows': len(caches[pid]['train']['present']),
        'test_rows': len(caches[pid]['test']['present'])
    } for pid in ALL_PIDS])
    atomic_csv(cache_audit, seed_out / 'audit' / 'FRESH_CACHE_AUDIT.csv')
    targets = list(ALL_PIDS)
    if args.max_participants > 0:
        targets = targets[:args.max_participants]
    for oi, target in enumerate(targets, 1):
        ck = seed_out / 'checkpoints'
        cp_rows = ck / f'{target}_ROWS.csv.gz'
        cp_sel = ck / f'{target}_SELECTED.json'
        cp_inner = ck / f'{target}_INNER.csv.gz'
        cp_audit = ck / f'{target}_AUDIT.csv.gz'
        if cp_rows.is_file() and cp_sel.is_file() and cp_inner.is_file() and cp_audit.is_file() and (not args.force):
            print(f'[seed {seed:02d}] [{oi:02d}/{len(targets):02d}] {target}: RESUME', flush=True)
            continue
        print(f'[seed {seed:02d}] [{oi:02d}/{len(targets):02d}] {target}: hierarchical configuration selection', flush=True)
        source_pids = [p for p in ALL_PIDS if p != target]
        selected, inner_scores, inner_audit = select_coherent_configs(n4, caches, target, source_pids)
        print('    +A   :', selected[ROW_A])
        print('    +B   :', selected[ROW_AB])
        print('    +C   :', selected[ROW_ABC])
        print('    +LBR :', selected[ROW_FULL])
        rows, outer_audit = outer_all_rows(n4, caches, target, selected)
        rows.insert(0, 'repeat_seed', int(seed))
        inner_scores.insert(0, 'repeat_seed', int(seed))
        combined_audit = pd.concat([inner_audit, outer_audit], ignore_index=True)
        combined_audit.insert(0, 'repeat_seed', int(seed))
        atomic_csv(rows, cp_rows)
        atomic_csv(inner_scores, cp_inner)
        atomic_csv(combined_audit, cp_audit)
        atomic_json(selected, cp_sel)
        del (rows, inner_scores, inner_audit, outer_audit, combined_audit, selected)
        gc.collect()
    if args.max_participants > 0:
        print(f'\nSEED {seed:02d}: smoke/partial run complete ({len(targets)}/26 outer participants).')
        return False
    merged = merge_one_hierarchical_repeat(n4, seed_out, seed)
    if not merged:
        raise RuntimeError(f'SEED {seed:02d}: expected all 26 checkpoint bundles, but merge found an incomplete participant.')
    return True
def print_assigned_seed_summary(parent_out: Path):
    completed = [int(seed) for seed in RUN_SEEDS if hierarchical_repeat_complete_marker(parent_out, seed).is_file()]
    if not completed:
        return
    tables = []
    for seed in completed:
        path = hierarchical_repeat_seed_dir(parent_out, seed) / 'tables' / 'Table1_STRICT_HIERARCHICAL_ABLATION.csv'
        if path.is_file():
            q = pd.read_csv(path)
            if 'repeat_seed' not in q.columns:
                q.insert(0, 'repeat_seed', seed)
            tables.append(q)
    if not tables:
        return
    all_table = pd.concat(tables, ignore_index=True)
    metrics = ('response_balanced_accuracy', 'onset_mae_sec', 'peak_mae_sec', 'offset_mae_sec', 'duration_mae_sec', 'selection_loss')
    rows = []
    for row in ROW_ORDER:
        q = all_table[all_table['row'].astype(str) == row]
        if len(q) == 0:
            continue
        r = {'row': row, 'configuration': ROW_LABELS[row], 'n_completed_assigned_seeds': int(q['repeat_seed'].nunique())}
        for metric in metrics:
            x = pd.to_numeric(q[metric], errors='coerce').to_numpy(float)
            r[metric + '_mean'] = float(np.nanmean(x))
            r[metric + '_sd'] = float(np.nanstd(x, ddof=1)) if len(x) > 1 else 0.0
        rows.append(r)
    summary = pd.DataFrame(rows)
    out = parent_out / 'PART_SUMMARIES'
    out.mkdir(parents=True, exist_ok=True)
    atomic_csv(all_table, out / f'{RUN_PART_NAME}_PER_SEED_TABLE1.csv')
    atomic_csv(summary, out / f'{RUN_PART_NAME}_MEAN_SD.csv')
    print('\n' + '=' * 180)
    print(f'{RUN_PART_NAME} — CURRENT ASSIGNED-SEED MEAN / SD')
    print('=' * 180)
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
def parse_multiseed_hierarchical_args():
    p = argparse.ArgumentParser()
    p.add_argument('--shin-root', default=None)
    p.add_argument('--output', default=None)
    p.add_argument('--max-participants', type=int, default=0, help='Smoke-test only the first N outer participants of each assigned seed.')
    p.add_argument('--force', action='store_true', help="Delete and rerun ONLY this runner's assigned seed folders.")
    p.add_argument('--merge-only', action='store_true', help='Do no training; merge/print assigned seeds that already have all 26 participant checkpoints.')
    p.add_argument('--self-test-only', action='store_true')
    args, unknown = p.parse_known_args()
    if unknown:
        print('Ignoring notebook/runtime arguments:', unknown)
    return args
def main_multiseed_hierarchical():
    if RUN_PART_NAME != EXPECTED_RUN_PART_NAME:
        raise RuntimeError(f'Runner identity mismatch: {RUN_PART_NAME} != {EXPECTED_RUN_PART_NAME}')
    if tuple(RUN_SEEDS) != tuple(EXPECTED_RUN_SEEDS):
        raise RuntimeError(f'Runner seed assignment mismatch: {RUN_SEEDS} != {EXPECTED_RUN_SEEDS}')
    if len(set(RUN_SEEDS)) != 3:
        raise RuntimeError(f'Expected exactly 3 unique assigned seeds, got {RUN_SEEDS}.')
    if any((int(s) < 2 or int(s) > 10 for s in RUN_SEEDS)):
        raise RuntimeError(f'Assigned seed outside allowed 2..10 range: {RUN_SEEDS}.')
    print(f'Seed assignment check: PASS — {RUN_PART_NAME} owns seeds {list(RUN_SEEDS)}')
    args = parse_multiseed_hierarchical_args()
    with tempfile.TemporaryDirectory(prefix='hemoboundary_strict_multiseed_') as td:
        td = Path(td)
        ref = import_embedded(REFINED_ENGINE_SOURCE, '_strict_multiseed_refined_engine', td)
        n4 = import_embedded(N4_ENGINE_SOURCE, '_strict_multiseed_n4_engine', td)
        configure_repeat_seed(ref, n4, 1)
        global COHERENT_SEED
        COHERENT_SEED = 1
        hierarchical_self_tests(ref, n4)
        if args.self_test_only:
            return
        shin_root = find_root(args.shin_root)
        fmap = ref.participant_files(shin_root)
        if sorted(fmap) != list(ALL_PIDS):
            raise RuntimeError(f'Expected exactly SHIN VP001..VP026 preprocessing files. Found: {sorted(fmap)}')
        parent_out = Path(args.output) if args.output else shin_root / MULTISEED_HIERARCHICAL_OUT_DIRNAME
        parent_out.mkdir(parents=True, exist_ok=True)
        print('=' * 180)
        print('HEMOBOUNDARY — REPEATED EVALUATION')
        print('=' * 180)
        print('Runner part                  :', RUN_PART_NAME)
        print('Assigned seeds               :', list(RUN_SEEDS))
        print('Participants / seed          : 26')
        print('Fresh bank / seed            : YES')
        print('Architecture changed         : NO')
        print('Hyperparameter grids changed : NO')
        print('Inherited settings changed   : NO')
        print('Output root                  :', parent_out)
        if args.merge_only:
            for seed in RUN_SEEDS:
                seed_out = hierarchical_repeat_seed_dir(parent_out, seed)
                ok = merge_one_hierarchical_repeat(n4, seed_out, seed)
                print(f'Seed {seed:02d} merge:', 'COMPLETE' if ok else 'INCOMPLETE')
            print_assigned_seed_summary(parent_out)
            return
        if args.force:
            for seed in RUN_SEEDS:
                seed_out = hierarchical_repeat_seed_dir(parent_out, seed)
                if seed_out.exists():
                    print('--force removes assigned seed only:', seed_out)
                    shutil.rmtree(seed_out)
        for seed in RUN_SEEDS:
            run_one_hierarchical_repeat(ref, n4, shin_root, fmap, parent_out, int(seed), args)
            gc.collect()
        print_assigned_seed_summary(parent_out)
        atomic_json({
            'version': MULTISEED_HIERARCHICAL_VERSION,
            'runner_part': RUN_PART_NAME,
            'assigned_seeds': list(RUN_SEEDS),
            'participants_per_seed': 26,
            'architecture': 'Base -> +A Temporal Boundary Estimation -> +B Recovery Estimation -> +C Response Detection -> +LBR Local Boundary Refinement',
            'strict_hierarchical_selection': True,
            'A_tunes': ['state_C', 'timing_alpha'],
            'B_inherits_A_and_tunes': ['change_frac'],
            'C_inherits_AB_and_tunes': ['response_threshold'],
            'LBR_inherits_ABC_and_tunes': ['radius', 'local_alpha', 'strength'],
            'fresh_seed_specific_bank': True,
            'outer_target_used_in_tuning': False,
            'parallel_safe_non_overlapping_seed_folders': True
        }, parent_out / f'{RUN_PART_NAME}_RUNNER_MANIFEST.json')
        print('\n' + '=' * 180)
        print(f'{RUN_PART_NAME}: FINISHED')
        print('=' * 180)
COMPARISON_TABLE3_VERSION = 'HEMOBOUNDARY_OPTIONB_TABLE3_10SEED_EXTERNAL_V1_20260916'
COMPARISON_TABLE3_OUT_DIRNAME = '102_OPTIONB_TABLE3_10SEED_BALINT_KHAN_ZERO_SHOT_V1'
HIERARCHICAL_SEED1_RESULT_DIRNAME = '98_B0_STRICT_HIERARCHICAL_SEED01_FULLY_NESTED_V1'
HIERARCHICAL_MULTI_RESULT_DIRNAME = '99_B0_STRICT_HIERARCHICAL_10SEEDS_V1'
HIERARCHICAL_SEED1_BANK_PARENT = '95_N4_FUNCTIONAL_ABC_LBR_FULLY_NESTED_10SEEDS_V1'
EXT_MODEL_ABC = 'OPTIONB_ABC'
EXT_MODEL_FULL = 'OPTIONB_HEMOBOUNDARY'
EXT_MODEL_LABEL = {EXT_MODEL_ABC: 'Base + A + B + C', EXT_MODEL_FULL: 'HemoBoundary'}
EXT_ALL_SEEDS = tuple(range(1, 11))
EXT_BOOT_REPS = 20000
EXT_SIGNFLIP_REPS = 20000
EXT_TIMING = ('onset_mae_sec', 'peak_mae_sec', 'offset_mae_sec', 'duration_mae_sec')
EXT_METRICS = ('response_balanced_accuracy', *EXT_TIMING, 'selection_loss')

def ext_discover_shin_root(explicit: Optional[str]=None) -> Path:
    try_mount_drive()
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([Path('/content/drive/MyDrive/HonorProject/shin2018_word_generation_state_preparation_v1'), Path('/content/gdrive/MyDrive/HonorProject/shin2018_word_generation_state_preparation_v1')])
    for p in candidates:
        if p.is_dir():
            return p
    raise FileNotFoundError('Could not locate SHIN project root. Use --shin-root PATH.')
def ext_seed_result_root(shin_root: Path, seed: int) -> Path:
    seed = int(seed)
    if seed == 1:
        p = shin_root / HIERARCHICAL_SEED1_RESULT_DIRNAME
    else:
        p = shin_root / HIERARCHICAL_MULTI_RESULT_DIRNAME / f'seed_{seed:02d}'
    if not p.is_dir():
        raise FileNotFoundError(f'Seed {seed:02d}: evaluation results folder not found: {p}')
    return p
def ext_seed_cache_dir(shin_root: Path, seed: int) -> Path:
    seed = int(seed)
    if seed == 1:
        p = shin_root / HIERARCHICAL_SEED1_BANK_PARENT / 'seed_01' / 'fresh_cache'
    else:
        p = shin_root / HIERARCHICAL_MULTI_RESULT_DIRNAME / f'seed_{seed:02d}' / 'fresh_cache'
    if not (p.is_dir() and all(((p / f'{pid}.npz').is_file() for pid in ALL_PIDS))):
        raise FileNotFoundError(f'Seed {seed:02d}: complete source cache not found: {p}')
    return p
def ext_load_seed_caches(ref, shin_root: Path, seed: int):
    cache_dir = ext_seed_cache_dir(shin_root, seed)
    caches0 = {pid: ref.load_cache(cache_dir / f'{pid}.npz') for pid in ALL_PIDS}
    return (add_pid_weight(caches0), cache_dir)
def ext_seed_selected_json(shin_root: Path, seed: int, pid: str) -> Mapping[str, Any]:
    p = ext_seed_result_root(shin_root, seed) / 'checkpoints' / f'{pid}_SELECTED.json'
    if not p.is_file():
        raise FileNotFoundError(f'Seed {seed:02d} {pid}: selected config not found: {p}')
    return json.loads(p.read_text(encoding='utf-8'))
def ext_choose_modal_config(rows: pd.DataFrame, config_col: str, loss_col: str) -> Tuple[str, Dict[str, Any]]:
    """
    Rule:
      1. largest number of SHIN outer folds selecting exact config;
      2. lower mean saved SHIN inner loss;
      3. deterministic lexical tie-break.
    """
    summary = rows.groupby(config_col, dropna=False).agg(outer_folds_selected=('participant', 'nunique'), mean_inner_loss=(loss_col, 'mean')).reset_index().sort_values([
        'outer_folds_selected',
        'mean_inner_loss',
        config_col
    ], ascending=[False, True, True]).reset_index(drop=True)
    if len(summary) == 0:
        raise RuntimeError('No SHIN source configurations available for external freeze.')
    chosen = str(summary.iloc[0][config_col])
    audit = {
        'selection_rule': 'modal SHIN outer-selected configuration; tie-break by lower mean saved SHIN inner loss, then deterministic lexical order',
        'modal_count': int(summary.iloc[0]['outer_folds_selected']),
        'modal_fraction': float(summary.iloc[0]['outer_folds_selected'] / 26.0),
        'modal_mean_inner_loss': float(summary.iloc[0]['mean_inner_loss']),
        'all_counts': summary.to_dict(orient='records'),
        'external_data_used_for_selection': False
    }
    return (chosen, audit)
def ext_seed_frozen_config(shin_root: Path, seed: int):
    """Freeze the exact hierarchical HemoBoundary source model for one robustness seed."""
    records = []
    for pid in ALL_PIDS:
        selected = ext_seed_selected_json(shin_root, seed, pid)
        A = selected[ROW_A]['global']
        AB = selected[ROW_AB]['global']
        ABC = selected[ROW_ABC]['global']
        FULL = selected[ROW_FULL]['global']
        LBR = selected[ROW_FULL]['local']
        if not all((np.isclose(float(AB[key]), float(A[key])) for key in ('state_C', 'timing_alpha'))):
            raise RuntimeError(f'Seed {seed:02d} {pid}: A->B inheritance failed.')
        if not all((np.isclose(float(ABC[key]), float(AB[key])) for key in ('state_C', 'timing_alpha', 'change_frac'))):
            raise RuntimeError(f'Seed {seed:02d} {pid}: AB->C inheritance failed.')
        if canonical_cfg(ABC) != canonical_cfg(FULL):
            raise RuntimeError(f'Seed {seed:02d} {pid}: ABC parent changed in Full.')
        records.append({
            'participant': pid,
            'ABC_cfg': canonical_cfg(ABC),
            'ABC_inner_loss': float(selected[ROW_ABC]['inner_loss']),
            'FULL_global_cfg': canonical_cfg(FULL),
            'LBR_cfg': canonical_cfg(LBR),
            'FULL_inner_loss': float(selected[ROW_FULL]['inner_loss'])
        })
    q = pd.DataFrame(records)
    abc_key, abc_audit = ext_choose_modal_config(q, 'ABC_cfg', 'ABC_inner_loss')
    abc_cfg = json.loads(abc_key)
    lbr_source = q[q['FULL_global_cfg'].astype(str) == str(abc_key)].copy()
    conditional_used = True
    if len(lbr_source) == 0:
        lbr_source = q.copy()
        conditional_used = False
    lbr_key, lbr_audit = ext_choose_modal_config(lbr_source, 'LBR_cfg', 'FULL_inner_loss')
    lbr_cfg = json.loads(lbr_key)
    audit = {
        'repeat_seed': int(seed),
        'source_outer_folds': 26,
        'strict_inheritance_reaudited': True,
        'ABC_parent_selection': abc_audit,
        'LBR_selection': lbr_audit,
        'LBR_selection_conditioned_on_frozen_ABC_parent': bool(conditional_used),
        'LBR_selection_source_outer_folds': int(lbr_source['participant'].nunique()),
        'frozen_ABC_cfg': abc_cfg,
        'frozen_FULL_global_cfg': dict(abc_cfg),
        'frozen_LBR_cfg': lbr_cfg,
        'ABC_and_FULL_share_identical_global_parent': True,
        'external_data_used_for_selection': False
    }
    return (abc_cfg, lbr_cfg, audit)
def ext_resolve_external_path(explicit: Optional[str], default_path: Path) -> Path:
    p = Path(explicit) if explicit else Path(default_path)
    candidates = [p, Path(str(p).replace('/content/gdrive/', '/content/drive/')), Path(str(p).replace('/content/drive/', '/content/gdrive/'))]
    for q in candidates:
        if q.is_file():
            return q
    raise FileNotFoundError(p)
def ext_prepare_fixed_external_bank(n4, dataset_name: str, h5_path: Path):
    """
    Fixed target-physiology known-boundary bank.

    This is created ONCE before source-seed mutation, so all ten SHIN source
    seeds are evaluated on identical external known-boundary trials.
    """
    store = n4.load_external_hdf5(h5_path)
    pids = sorted(set(store['participant'].astype(str).tolist()))
    if len(pids) < 2:
        raise RuntimeError(f'{dataset_name}: expected multiple external participants.')
    banks = {}
    real_predictors = {}
    qc_rows = []
    for pid in pids:
        profile, valid_c, total_c = n4.noise_profile_from_external(store, pid)
        bank = n4.external_synthetic_test(profile, dataset_name, pid)
        banks[pid] = bank
        real_x, kept = n4.real_external_predictors(store, pid)
        real_predictors[pid] = (real_x, kept)
        qc_rows.append({
            'dataset': dataset_name,
            'participant': pid,
            'hdf5': str(h5_path),
            'h5_schema': store['schema'],
            'n_real_epochs': int(np.sum(store['participant'].astype(str) == str(pid))),
            'usable_channels': int(valid_c),
            'channels_before_validity_filter': int(total_c),
            'phi_hbo': float(profile.phi_hbo),
            'phi_hbr': float(profile.phi_hbr),
            'median_hbo_noise_scale': float(np.median(profile.hbo_scale)),
            'median_hbr_noise_scale': float(np.median(profile.hbr_scale)),
            'fixed_external_bank_across_source_seeds': True
        })
    return {'store': store, 'pids': pids, 'banks': banks, 'real_predictors': real_predictors, 'qc': pd.DataFrame(qc_rows)}
def ext_fit_source_models(n4, caches, abc_cfg, lbr_cfg, seed: int):
    """
    Refit ABC and Full on ALL 26 SHIN TRAIN banks.

    Full shares the exact same frozen ABC parent as ABC and adds only LBR.
    """
    train = training_rows(n4, caches, ALL_PIDS)
    abc_audit = n4.FitAudit(ALL_PIDS, [], f'optionb_table3/seed={seed:02d}/ABC/all_SHIN_source')
    abc_model = fit_coherent_ABC(n4, train, abc_cfg, abc_audit, stage_prefix='OPTIONB_EXT_ABC')
    full_audit = n4.FitAudit(ALL_PIDS, [], f'optionb_table3/seed={seed:02d}/FULL/all_SHIN_source')
    full_model = fit_coherent_full(n4, train, abc_cfg, lbr_cfg, full_audit, outer_forbidden=set(), context=f'optionb_table3/seed={seed:02d}/FULL/all_SHIN_source')
    audit = pd.DataFrame([{
        'repeat_seed': int(seed),
        'external_model': EXT_MODEL_ABC,
        **row
    } for row in abc_audit.rows] + [{'repeat_seed': int(seed), 'external_model': EXT_MODEL_FULL, **row} for row in full_audit.rows])
    return (abc_model, full_model, audit)
def ext_predict_dataset_seed(n4, seed: int, dataset_name: str, info, abc_model, full_model, include_real_descriptive: bool):
    trial_parts = []
    pm_rows = []
    real_parts = []
    for pid in info['pids']:
        truth = info['banks'][pid]
        x = n4.predictor_view(truth)
        pred_abc = abc_model.predict(x)
        pred_full = full_model.predict(x)
        for model_name, pred in ((EXT_MODEL_ABC, pred_abc), (EXT_MODEL_FULL, pred_full)):
            m = n4.trial_metric_summary(truth, pred)
            pm_rows.append({
                'repeat_seed': int(seed),
                'dataset': dataset_name,
                'participant': pid,
                'model': model_name,
                'model_label': EXT_MODEL_LABEL[model_name],
                'response_balanced_accuracy': float(m['response_balanced_accuracy']),
                'onset_mae_sec': float(m['onset_mae_sec']),
                'peak_mae_sec': float(m['peak_mae_sec']),
                'offset_mae_sec': float(m['offset_mae_sec']),
                'duration_mae_sec': float(m['duration_mae_sec']),
                'selection_loss': float(m['selection_loss'])
            })
            trial_parts.append(pd.DataFrame({
                'repeat_seed': int(seed),
                'dataset': dataset_name,
                'participant': pid,
                'model': model_name,
                'model_label': EXT_MODEL_LABEL[model_name],
                'synthetic_index': np.arange(len(pred), dtype=int),
                'truth_present': np.asarray(truth['present'], int),
                'truth_onset': np.asarray(truth['onset'], float),
                'truth_peak': np.asarray(truth['peak'], float),
                'truth_offset': np.asarray(truth['offset'], float),
                'truth_censored': np.asarray(truth['censored'], int),
                'pred_response': pred['pred_response'].to_numpy(int),
                'pred_onset': pred['pred_onset'].to_numpy(float),
                'pred_peak': pred['pred_peak'].to_numpy(float),
                'pred_offset': pred['pred_offset'].to_numpy(float)
            }))
        if include_real_descriptive:
            real_x, kept = info['real_predictors'][pid]
            if real_x is not None and len(kept):
                for model_name, model in ((EXT_MODEL_ABC, abc_model), (EXT_MODEL_FULL, full_model)):
                    pred = model.predict(real_x)
                    real_parts.append(pd.DataFrame({
                        'repeat_seed': int(seed),
                        'dataset': dataset_name,
                        'participant': pid,
                        'model': model_name,
                        'model_label': EXT_MODEL_LABEL[model_name],
                        'real_trial_index': kept,
                        'pred_response': pred['pred_response'].to_numpy(int),
                        'pred_onset': pred['pred_onset'].to_numpy(float),
                        'pred_peak': pred['pred_peak'].to_numpy(float),
                        'pred_offset': pred['pred_offset'].to_numpy(float),
                        'timing_truth_available': False,
                        'role': 'DESCRIPTIVE_ONLY'
                    }))
    return (pd.concat(trial_parts, ignore_index=True), pd.DataFrame(pm_rows), pd.concat(real_parts, ignore_index=True) if real_parts else pd.DataFrame())
def ext_bootstrap_ci(x: Sequence[float], seed: int):
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(a))
    if len(a) == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    vals = np.empty(EXT_BOOT_REPS, float)
    done = 0
    while done < EXT_BOOT_REPS:
        n = min(2000, EXT_BOOT_REPS - done)
        ix = rng.integers(0, len(a), size=(n, len(a)))
        vals[done:done + n] = np.mean(a[ix], axis=1)
        done += n
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return (mean, float(lo), float(hi))
def ext_signflip_p(x: Sequence[float], seed: int):
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan
    obs = abs(float(np.mean(a)))
    if obs < 1e-15:
        return 1.0
    rng = np.random.default_rng(seed)
    hits = 0
    done = 0
    while done < EXT_SIGNFLIP_REPS:
        n = min(2000, EXT_SIGNFLIP_REPS - done)
        signs = rng.choice([-1.0, 1.0], size=(n, len(a)))
        stat = np.abs(np.mean(signs * a[None, :], axis=1))
        hits += int(np.sum(stat >= obs - 1e-15))
        done += n
    return float((hits + 1) / (EXT_SIGNFLIP_REPS + 1))
def ext_holm(pvals: Sequence[float]):
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan, float)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return out
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        out[idx] = min(1.0, running)
    return out
def ext_per_seed_performance(pm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, dataset, model), q in pm.groupby(['repeat_seed', 'dataset', 'model'], sort=True):
        row = {
            'repeat_seed': int(seed),
            'dataset': dataset,
            'model': model,
            'model_label': EXT_MODEL_LABEL[model],
            'n_participants': int(q['participant'].nunique())
        }
        for metric in EXT_METRICS:
            row[metric] = float(np.mean(q[metric].to_numpy(float)))
        rows.append(row)
    return pd.DataFrame(rows)
def ext_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model), q in per_seed.groupby(['dataset', 'model'], sort=True):
        row = {
            'dataset': dataset,
            'model': model,
            'model_label': EXT_MODEL_LABEL[model],
            'n_source_seeds': int(q['repeat_seed'].nunique()),
            'n_external_participants': int(q['n_participants'].iloc[0])
        }
        for metric in EXT_METRICS:
            x = q[metric].to_numpy(float)
            row[metric + '_mean'] = float(np.mean(x))
            row[metric + '_sd'] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)
def ext_paper_ready(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in ('BALINT', 'KHAN'):
        for model in (EXT_MODEL_ABC, EXT_MODEL_FULL):
            q = summary[(summary['dataset'] == dataset) & (summary['model'] == model)]
            if len(q) != 1:
                continue
            q = q.iloc[0]
            rows.append({
                'dataset': dataset,
                'model': EXT_MODEL_LABEL[model],
                'n_external_participants': int(q['n_external_participants']),
                'response_BA': f"{float(q['response_balanced_accuracy_mean']):.4f} ± {float(q['response_balanced_accuracy_sd']):.4f}",
                'onset_MAE_s': f"{float(q['onset_mae_sec_mean']):.4f} ± {float(q['onset_mae_sec_sd']):.4f}",
                'peak_MAE_s': f"{float(q['peak_mae_sec_mean']):.4f} ± {float(q['peak_mae_sec_sd']):.4f}",
                'offset_MAE_s': f"{float(q['offset_mae_sec_mean']):.4f} ± {float(q['offset_mae_sec_sd']):.4f}",
                'duration_MAE_s': f"{float(q['duration_mae_sec_mean']):.4f} ± {float(q['duration_mae_sec_sd']):.4f}"
            })
    return pd.DataFrame(rows)
def ext_primary_stats(pm: pd.DataFrame) -> pd.DataFrame:
    """
    Primary inference:
      within each external participant, average ABC-minus-Full timing gain
      (or Full-minus-ABC BA gain) across 10 source seeds;
      then infer across external participants.
    """
    gain_rows = []
    for (dataset, participant, seed), q in pm.groupby(['dataset', 'participant', 'repeat_seed'], sort=True):
        p = q.set_index('model')
        if not {EXT_MODEL_ABC, EXT_MODEL_FULL}.issubset(set(p.index)):
            raise RuntimeError(f'{dataset} {participant} seed {seed}: missing model.')
        for metric in EXT_METRICS:
            a = float(p.loc[EXT_MODEL_ABC, metric])
            b = float(p.loc[EXT_MODEL_FULL, metric])
            if metric == 'response_balanced_accuracy':
                gain = b - a
            else:
                gain = a - b
            gain_rows.append({'dataset': dataset, 'participant': participant, 'repeat_seed': int(seed), 'metric': metric, 'gain_positive_HemoBoundary_better': gain})
    gains = pd.DataFrame(gain_rows)
    participant_gain = gains.groupby([
        'dataset',
        'participant',
        'metric'
    ], as_index=False)['gain_positive_HemoBoundary_better'].mean().rename(columns={'gain_positive_HemoBoundary_better': 'mean_gain_across_source_seeds'})
    seed_gain = gains.groupby([
        'dataset',
        'repeat_seed',
        'metric'
    ], as_index=False)['gain_positive_HemoBoundary_better'].mean().rename(columns={'gain_positive_HemoBoundary_better': 'mean_gain_across_external_participants'})
    rows = []
    for dataset in sorted(pm['dataset'].unique()):
        for metric in EXT_METRICS:
            q = participant_gain[(participant_gain['dataset'] == dataset) & (participant_gain['metric'] == metric)]
            x = q['mean_gain_across_source_seeds'].to_numpy(float)
            sq = seed_gain[(seed_gain['dataset'] == dataset) & (seed_gain['metric'] == metric)]
            sx = sq['mean_gain_across_external_participants'].to_numpy(float)
            mean, lo, hi = ext_bootstrap_ci(x, stable_seed('OPTIONB_EXT_BOOT', dataset, metric))
            p = ext_signflip_p(x, stable_seed('OPTIONB_EXT_SIGN', dataset, metric))
            rows.append({
                'dataset': dataset,
                'comparison': 'ABC -> HemoBoundary',
                'metric': metric,
                'n_external_participants': len(x),
                'n_source_seeds': len(sx),
                'mean_gain_positive_HemoBoundary_better': mean,
                'ci95_low': lo,
                'ci95_high': hi,
                'participants_better': int(np.sum(x > 1e-12)),
                'participants_equal': int(np.sum(np.isclose(x, 0.0, atol=1e-12))),
                'participants_worse': int(np.sum(x < -1e-12)),
                'source_seeds_positive': int(np.sum(sx > 1e-12)),
                'source_seeds_equal': int(np.sum(np.isclose(sx, 0.0, atol=1e-12))),
                'source_seeds_negative': int(np.sum(sx < -1e-12)),
                'signflip_p': p
            })
    out = pd.DataFrame(rows)
    out['holm_p_4_timing_within_dataset'] = np.nan
    for dataset in out['dataset'].unique():
        ix = out.index[(out['dataset'] == dataset) & out['metric'].isin(EXT_TIMING)].tolist()
        out.loc[ix, 'holm_p_4_timing_within_dataset'] = ext_holm(out.loc[ix, 'signflip_p'].to_numpy(float))
    out['holm_p_global_external_timing'] = np.nan
    ix = out.index[out['metric'].isin(EXT_TIMING)].tolist()
    out.loc[ix, 'holm_p_global_external_timing'] = ext_holm(out.loc[ix, 'signflip_p'].to_numpy(float))
    return (gains, participant_gain, seed_gain, out)
def ext_run_source_seed(ref, n4, shin_root: Path, out: Path, seed: int, external_sets, include_real_descriptive: bool, force: bool):
    cp = out / 'checkpoints'
    cp_trial = cp / f'seed_{seed:02d}_PER_TRIAL.csv.gz'
    cp_pm = cp / f'seed_{seed:02d}_PARTICIPANT_METRICS.csv'
    cp_real = cp / f'seed_{seed:02d}_REAL_DESCRIPTIVE.csv.gz'
    cp_fit = cp / f'seed_{seed:02d}_SOURCE_FIT_AUDIT.csv'
    cp_freeze = cp / f'seed_{seed:02d}_FREEZE_AUDIT.json'
    if cp_trial.is_file() and cp_pm.is_file() and cp_fit.is_file() and cp_freeze.is_file() and (not force):
        print(f'Seed {seed:02d}: RESUME', flush=True)
        return (pd.read_csv(cp_trial), pd.read_csv(cp_pm), pd.read_csv(cp_real) if cp_real.is_file() else pd.DataFrame())
    configure_repeat_seed(ref, n4, int(seed))
    global COHERENT_SEED
    COHERENT_SEED = int(seed)
    caches, cache_dir = ext_load_seed_caches(ref, shin_root, seed)
    abc_cfg, lbr_cfg, freeze_audit = ext_seed_frozen_config(shin_root, seed)
    print(f'Seed {seed:02d}: SHIN-only frozen ABC={abc_cfg}, LBR={lbr_cfg}', flush=True)
    abc_model, full_model, fit_audit = ext_fit_source_models(n4, caches, abc_cfg, lbr_cfg, seed)
    trial_parts = []
    pm_parts = []
    real_parts = []
    for dataset_name, info in external_sets.items():
        print(f'  {dataset_name}: zero-shot', flush=True)
        trial, pm, real = ext_predict_dataset_seed(n4, seed, dataset_name, info, abc_model, full_model, include_real_descriptive)
        trial_parts.append(trial)
        pm_parts.append(pm)
        if len(real):
            real_parts.append(real)
    trial = pd.concat(trial_parts, ignore_index=True)
    pm = pd.concat(pm_parts, ignore_index=True)
    real = pd.concat(real_parts, ignore_index=True) if real_parts else pd.DataFrame()
    atomic_csv(trial, cp_trial)
    atomic_csv(pm, cp_pm)
    if len(real):
        atomic_csv(real, cp_real)
    atomic_csv(fit_audit, cp_fit)
    atomic_json({
        'version': COMPARISON_TABLE3_VERSION,
        'repeat_seed': int(seed),
        'source_cache': str(cache_dir),
        'source_participants': list(ALL_PIDS),
        'source_participant_count': 26,
        'frozen_ABC_cfg': abc_cfg,
        'frozen_FULL_global_cfg': dict(abc_cfg),
        'frozen_LBR_cfg': lbr_cfg,
        'ABC_and_FULL_share_identical_global_parent': True,
        'freeze_selection_audit': freeze_audit,
        'source_fit_uses_all_26_SHIN_train_banks': True,
        'external_data_used_for_config_selection': False,
        'external_data_used_for_source_fit': False,
        'external_labels_used_for_tuning': False,
        'external_adaptation': False,
        'real_external_timing_truth_available': False,
        'real_external_predictions_role': 'DESCRIPTIVE_ONLY',
        'fixed_known_boundary_external_bank_across_source_seeds': True
    }, cp_freeze)
    return (trial, pm, real)
def comparison_table3_self_tests():
    q = pd.DataFrame({'participant': ['A', 'B', 'C'], 'cfg': ['{"x":1}', '{"x":1}', '{"x":2}'], 'loss': [1.0, 2.0, 0.1]})
    chosen, audit = ext_choose_modal_config(q, 'cfg', 'loss')
    assert chosen == '{"x":1}'
    assert audit['modal_count'] == 2
    mean, lo, hi = ext_bootstrap_ci([0.1, 0.2, 0.3], 123)
    assert mean > 0
    assert lo > 0
    assert hi > 0
    print('External evaluation checks: PASS')
def parse_comparison_table3_args():
    p = argparse.ArgumentParser()
    p.add_argument('--shin-root', default=None)
    p.add_argument('--output', default=None)
    p.add_argument('--balint-h5', default=None)
    p.add_argument('--khan-h5', default=None)
    p.add_argument('--datasets', default='BALINT,KHAN', help='BALINT, KHAN, or BALINT,KHAN')
    p.add_argument('--seeds', default='1,2,3,4,5,6,7,8,9,10')
    p.add_argument('--skip-real-descriptive', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('--self-test-only', action='store_true')
    args, unknown = p.parse_known_args()
    if unknown:
        print('Ignoring notebook/runtime arguments:', unknown)
    return args
def parse_ext_seed_list(text: str) -> List[int]:
    vals = sorted(set((int(x.strip()) for x in str(text).split(',') if x.strip())))
    if not vals:
        raise ValueError('No source seeds requested.')
    if any((s not in EXT_ALL_SEEDS for s in vals)):
        raise ValueError(f'Source seeds must be 1..10, got {vals}.')
    return vals
def main_comparison_table3():
    args = parse_comparison_table3_args()
    with tempfile.TemporaryDirectory(prefix='hemoboundary_optionb_table3_') as td:
        td = Path(td)
        ref = import_embedded(REFINED_ENGINE_SOURCE, '_optionb_table3_refined_engine', td)
        n4 = import_embedded(N4_ENGINE_SOURCE, '_optionb_table3_n4_engine', td)
        protected_self_tests(ref, n4)
        comparison_table3_self_tests()
        if args.self_test_only:
            return
        shin_root = ext_discover_shin_root(args.shin_root)
        out = Path(args.output) if args.output else shin_root / COMPARISON_TABLE3_OUT_DIRNAME
        for sub in ('checkpoints', 'results', 'tables', 'audit'):
            (out / sub).mkdir(parents=True, exist_ok=True)
        requested_datasets = [x.strip().upper() for x in str(args.datasets).split(',') if x.strip()]
        if not requested_datasets or not set(requested_datasets).issubset({'BALINT', 'KHAN'}):
            raise ValueError('--datasets must contain BALINT and/or KHAN.')
        requested_seeds = parse_ext_seed_list(args.seeds)
        configure_repeat_seed(ref, n4, 1)
        global COHERENT_SEED
        COHERENT_SEED = 1
        balint_path = ext_resolve_external_path(args.balint_h5, n4.BALINT_DEFAULT)
        khan_path = ext_resolve_external_path(args.khan_h5, n4.KHAN_DEFAULT)
        ext_paths = {'BALINT': balint_path, 'KHAN': khan_path}
        external_sets = {}
        print('=' * 180)
        print('HEMOBOUNDARY — EXTERNAL DATASET EVALUATION')
        print('=' * 180)
        print('SHIN root                       :', shin_root)
        print('Output                          :', out)
        print('Source robustness seeds         :', requested_seeds)
        print('External datasets               :', requested_datasets)
        print('Source participants / refit     : 26 SHIN')
        print('External config selection       : SHIN ONLY')
        print('ABC / Full global parent        : IDENTICAL')
        print('Full adds                       : LBR ONLY')
        print('External fitting                : NONE')
        print('External tuning                 : NONE')
        print('External adaptation             : NONE')
        print('Known-boundary external bank    : FIXED across all source seeds')
        print('Real external timing truth      : unavailable; descriptive only')
        for dataset_name in requested_datasets:
            print(f'Preparing fixed external bank: {dataset_name}', flush=True)
            external_sets[dataset_name] = ext_prepare_fixed_external_bank(n4, dataset_name, ext_paths[dataset_name])
        qc = pd.concat([external_sets[d]['qc'] for d in requested_datasets], ignore_index=True)
        atomic_csv(qc, out / 'audit' / 'EXTERNAL_FIXED_BANK_QC.csv')
        for seed in requested_seeds:
            ext_run_source_seed(ref, n4, shin_root, out, int(seed), external_sets, include_real_descriptive=not args.skip_real_descriptive, force=bool(args.force))
        trial_parts = []
        pm_parts = []
        real_parts = []
        complete_seeds = []
        for seed in EXT_ALL_SEEDS:
            cp_trial = out / 'checkpoints' / f'seed_{seed:02d}_PER_TRIAL.csv.gz'
            cp_pm = out / 'checkpoints' / f'seed_{seed:02d}_PARTICIPANT_METRICS.csv'
            cp_real = out / 'checkpoints' / f'seed_{seed:02d}_REAL_DESCRIPTIVE.csv.gz'
            if cp_trial.is_file() and cp_pm.is_file():
                trial_parts.append(pd.read_csv(cp_trial))
                pm_parts.append(pd.read_csv(cp_pm))
                if cp_real.is_file():
                    real_parts.append(pd.read_csv(cp_real))
                complete_seeds.append(int(seed))
        if not pm_parts:
            raise RuntimeError('No completed external source-seed checkpoints found.')
        all_trial = pd.concat(trial_parts, ignore_index=True)
        all_pm = pd.concat(pm_parts, ignore_index=True)
        all_real = pd.concat(real_parts, ignore_index=True) if real_parts else pd.DataFrame()
        atomic_csv(all_trial, out / 'results' / 'TABLE3_OPTIONB_KNOWN_BOUNDARY_PER_TRIAL_ALL_SEEDS.csv.gz')
        atomic_csv(all_pm, out / 'results' / 'TABLE3_OPTIONB_PARTICIPANT_METRICS_ALL_SEEDS.csv.gz')
        if len(all_real):
            atomic_csv(all_real, out / 'results' / 'TABLE3_OPTIONB_REAL_EXTERNAL_DESCRIPTIVE_ALL_SEEDS.csv.gz')
        if sorted(complete_seeds) != list(EXT_ALL_SEEDS):
            print('\n' + '=' * 180)
            print('EXTERNAL EVALUATION — PARTIAL RESULTS SAVED')
            print('=' * 180)
            print('Completed source seeds:', complete_seeds)
            atomic_json({
                'version': COMPARISON_TABLE3_VERSION,
                'complete': False,
                'completed_source_seeds': complete_seeds,
                'required_source_seeds': list(EXT_ALL_SEEDS),
                'external_tuning': False,
                'external_adaptation': False
            }, out / 'audit' / 'TABLE3_OPTIONB_RUN_MANIFEST.json')
            return
        for dataset_name in requested_datasets:
            expected_pids = set(external_sets[dataset_name]['pids'])
            q = all_pm[all_pm['dataset'] == dataset_name]
            for seed in EXT_ALL_SEEDS:
                for model in (EXT_MODEL_ABC, EXT_MODEL_FULL):
                    z = q[(q['repeat_seed'] == seed) & (q['model'] == model)]
                    if set(z['participant'].astype(str)) != expected_pids:
                        raise RuntimeError(f'{dataset_name} seed {seed:02d} {model}: participant coverage incomplete.')
        per_seed = ext_per_seed_performance(all_pm)
        summary = ext_summary(per_seed)
        paper = ext_paper_ready(summary)
        raw_gains, participant_gains, seed_gains, stats = ext_primary_stats(all_pm)
        atomic_csv(per_seed, out / 'results' / 'TABLE3_OPTIONB_PER_SOURCE_SEED_PERFORMANCE.csv')
        atomic_csv(summary, out / 'tables' / 'Table3_OPTIONB_10SEED_EXTERNAL_MEAN_SD.csv')
        atomic_csv(paper, out / 'tables' / 'Table3_OPTIONB_10SEED_EXTERNAL_PAPER_READY.csv')
        atomic_csv(stats, out / 'tables' / 'Table3_OPTIONB_ABC_VS_HEMOBOUNDARY_EXTERNAL_STATS.csv')
        atomic_csv(raw_gains, out / 'results' / 'D1_OPTIONB_EXTERNAL_GAINS_SEED_PARTICIPANT.csv.gz')
        atomic_csv(participant_gains, out / 'results' / 'D2_OPTIONB_EXTERNAL_GAINS_PARTICIPANT_AVERAGED_10SEEDS.csv')
        atomic_csv(seed_gains, out / 'results' / 'D3_OPTIONB_EXTERNAL_GAINS_SEED_AVERAGED_PARTICIPANTS.csv')
        freeze_audits = []
        for seed in EXT_ALL_SEEDS:
            p = out / 'checkpoints' / f'seed_{seed:02d}_FREEZE_AUDIT.json'
            freeze_audits.append(json.loads(p.read_text(encoding='utf-8')))
        atomic_json({
            'version': COMPARISON_TABLE3_VERSION,
            'complete': True,
            'source_seeds': list(EXT_ALL_SEEDS),
            'source_participants': list(ALL_PIDS),
            'source_participants_per_refit': 26,
            'datasets': requested_datasets,
            'external_paths': {d: str(ext_paths[d]) for d in requested_datasets},
            'fixed_known_boundary_bank_across_source_seeds': True,
            'external_config_selection': 'Shin-source modal A+B+C settings per seed; LBR modal selection conditioned on frozen ABC parent',
            'ABC_and_FULL_share_identical_global_parent': True,
            'Full_adds_only_LBR': True,
            'external_data_used_for_model_selection': False,
            'external_tuning': False,
            'external_adaptation': False,
            'external_labels_used_for_source_fit': False,
            'external_labels_used_for_model_selection': False,
            'ABC_external_model': 'refit once per source seed on all 26 SHIN TRAIN banks',
            'HemoBoundary_external_model': 'same frozen ABC parent refit on all 26 SHIN TRAIN banks; source-participant OOF LBR residual fitting; add LBR only',
            'real_external_timing_truth_available': False,
            'real_external_predictions_role': 'DESCRIPTIVE_ONLY',
            'primary_statistical_unit': 'external participant',
            'source_seed_participant_rows_treated_as_independent': False,
            'primary_inference': 'average ABC-to-Full gain across ten SHIN source seeds within each external participant, then bootstrap/sign-flip across external participants',
            'holm_within_dataset_timing_family': 4,
            'holm_global_external_timing_family': len(requested_datasets) * 4,
            'seed_freeze_audits': freeze_audits
        }, out / 'audit' / 'TABLE3_OPTIONB_RUN_MANIFEST.json')
        print('\n' + '=' * 180)
        print('EXTERNAL EVALUATION — SUMMARY')
        print('=' * 180)
        print(paper.to_string(index=False))
        print('\n' + '=' * 180)
        print('EXTERNAL EVALUATION — MEAN AND SD')
        print('=' * 180)
        print(summary.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
        print('\n' + '=' * 180)
        print('EXTERNAL EVALUATION — MODEL COMPARISON')
        print('=' * 180)
        print(stats.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
        print('\nFINAL AUDIT')
        print('  source robustness seeds         : 10 / 10')
        print('  SHIN source participants/refit  : 26 / 26')
        print('  external datasets               :', requested_datasets)
        print('  ABC / Full parent identical     : YES')
        print('  Full adds only LBR              : YES')
        print('  external model selection        : SHIN ONLY')
        print('  external fitting/tuning         : NO')
        print('  external adaptation             : NO')
        print('  external labels in source fit   : NO')
        print('  fixed ext bank across src seeds : YES')
        print('  real timing accuracy claim      : NO')
        print('  inference unit                  : external participant')
        print('  pseudo-replication across seeds : NO')
        print('  saved                           :', out)
if __name__ == '__main__':
    main_comparison_table3()
