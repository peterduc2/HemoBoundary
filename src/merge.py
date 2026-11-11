"""Combined participant-level evaluation summaries."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
VERSION = 'HEMOBOUNDARY_B0_STRICT_OPTION_B_FINAL_10SEED_MERGE_V1_20260916'
SEED1_DIRNAME = '98_B0_STRICT_HIERARCHICAL_SEED01_FULLY_NESTED_V1'
MULTISEED_DIRNAME = '99_B0_STRICT_HIERARCHICAL_10SEEDS_V1'
FINAL_DIRNAME = '100_B0_STRICT_HIERARCHICAL_FINAL_10SEED_MERGE_V1'
ALL_SEEDS = tuple(range(1, 11))
ALL_PIDS = tuple((f'VP{i:03d}' for i in range(1, 27)))
MISS_PENALTY_SEC = 4.0
BOOT_REPS = 20000
SIGNFLIP_REPS = 20000
ROW_BASE = 'BASE_CANONICAL_HRF_GLM_DERIVATIVES'
ROW_A = 'BASE_PLUS_A_TEMPORAL_BOUNDARY'
ROW_AB = 'BASE_PLUS_A_PLUS_B_RECOVERY'
ROW_ABC = 'BASE_PLUS_A_PLUS_B_PLUS_C_RESPONSE'
ROW_FULL = 'BASE_PLUS_A_PLUS_B_PLUS_C_PLUS_LBR'
ROW_ORDER = (ROW_BASE, ROW_A, ROW_AB, ROW_ABC, ROW_FULL)
ROW_SHORT = {ROW_BASE: 'Base', ROW_A: 'Base + A', ROW_AB: 'Base + A + B', ROW_ABC: 'Base + A + B + C', ROW_FULL: 'Base + A + B + C + LBR — HemoBoundary'}
TRANSITIONS = ((ROW_BASE, ROW_A, '+A'), (ROW_A, ROW_AB, '+B'), (ROW_AB, ROW_ABC, '+C'), (ROW_ABC, ROW_FULL, '+LBR'))
METRICS = ('response_balanced_accuracy', 'onset_mae_sec', 'peak_mae_sec', 'offset_mae_sec', 'duration_mae_sec', 'selection_loss')
TIMING_METRICS = ('onset_mae_sec', 'peak_mae_sec', 'offset_mae_sec', 'duration_mae_sec')

def stable_seed(*parts: Any) -> int:
    raw = '|'.join(map(str, parts)).encode('utf-8')
    return int(hashlib.sha256(raw).hexdigest()[:8], 16) % (2 ** 31 - 1)
def try_mount_drive():
    if Path('/content/drive/MyDrive').exists() or Path('/content/gdrive/MyDrive').exists():
        return
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
    except Exception:
        pass
def find_shin_root(explicit=None) -> Path:
    try_mount_drive()
    cands = []
    if explicit:
        cands.append(Path(explicit))
    cands += [Path('/content/drive/MyDrive/HonorProject/shin2018_word_generation_state_preparation_v1'), Path('/content/gdrive/MyDrive/HonorProject/shin2018_word_generation_state_preparation_v1')]
    for p in cands:
        if p.is_dir():
            return p
    raise FileNotFoundError('Could not locate SHIN project root; use --shin-root PATH.')
def seed_root(root: Path, seed: int) -> Path:
    return root / SEED1_DIRNAME if seed == 1 else root / MULTISEED_DIRNAME / f'seed_{seed:02d}'
def bundle(root: Path, seed: int, pid: str):
    ck = seed_root(root, seed) / 'checkpoints'
    return {'rows': ck / f'{pid}_ROWS.csv.gz', 'selected': ck / f'{pid}_SELECTED.json', 'inner': ck / f'{pid}_INNER.csv.gz', 'audit': ck / f'{pid}_AUDIT.csv.gz'}
def canonical_cfg(d: Mapping[str, Any]) -> str:
    return json.dumps({str(k): d[k] for k in sorted(d)}, sort_keys=True, separators=(',', ':'))
def atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    df.to_csv(tmp, index=False, compression='gzip' if path.suffix == '.gz' else None)
    os.replace(tmp, path)
def atomic_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding='utf-8')
    os.replace(tmp, path)
def completion(root: Path) -> pd.DataFrame:
    rows = []
    for seed in ALL_SEEDS:
        for pid in ALL_PIDS:
            b = bundle(root, seed, pid)
            f = {k: p.is_file() for k, p in b.items()}
            rows.append({
                'repeat_seed': seed,
                'participant': pid,
                'ROWS': f['rows'],
                'SELECTED': f['selected'],
                'INNER': f['inner'],
                'AUDIT': f['audit'],
                'COMPLETE': all(f.values())
            })
    return pd.DataFrame(rows)
def ba(y, pred):
    y, pred = (np.asarray(y, int), np.asarray(pred, int))
    pos, neg = (y == 1, y == 0)
    sens = np.mean(pred[pos] == 1) if np.any(pos) else np.nan
    spec = np.mean(pred[neg] == 0) if np.any(neg) else np.nan
    return float(np.nanmean([sens, spec]))
def metrics(q: pd.DataFrame):
    present = q['truth_present'].to_numpy(int) == 1
    cens = q['truth_censored'].to_numpy(int) == 1
    decision = q['pred_response'].to_numpy(int)
    out = {'response_balanced_accuracy': ba(q['truth_present'].to_numpy(int), decision)}
    for name in ('onset', 'peak', 'offset'):
        truth = q[f'truth_{name}'].to_numpy(float)
        pred = q[f'pred_{name}'].to_numpy(float)
        valid = present & np.isfinite(truth)
        if name == 'offset':
            valid &= ~cens
        idx = np.flatnonzero(valid)
        err = np.full(len(idx), MISS_PENALTY_SEC, float)
        ok = np.isfinite(pred[idx]) & (decision[idx] == 1)
        err[ok] = np.abs(pred[idx][ok] - truth[idx][ok])
        out[f'{name}_mae_sec'] = float(np.mean(err)) if len(err) else np.nan
    ton = q['truth_onset'].to_numpy(float)
    toff = q['truth_offset'].to_numpy(float)
    pon = q['pred_onset'].to_numpy(float)
    poff = q['pred_offset'].to_numpy(float)
    valid = present & ~cens & np.isfinite(ton) & np.isfinite(toff)
    idx = np.flatnonzero(valid)
    err = np.full(len(idx), MISS_PENALTY_SEC, float)
    ok = np.isfinite(pon[idx]) & np.isfinite(poff[idx]) & (decision[idx] == 1)
    err[ok] = np.abs(poff[idx][ok] - pon[idx][ok] - (toff[idx][ok] - ton[idx][ok]))
    out['duration_mae_sec'] = float(np.mean(err)) if len(err) else np.nan
    v = np.array([out['onset_mae_sec'], out['peak_mae_sec'], out['offset_mae_sec'], out['duration_mae_sec']], float)
    out['selection_loss'] = float(np.nanmean(v / np.array([1.0, 1.0, 1.5, 1.5])))
    return out
def validate_rows(q: pd.DataFrame, seed: int, pid: str):
    req = {
        'participant',
        'row',
        'synthetic_index',
        'truth_present',
        'truth_onset',
        'truth_peak',
        'truth_offset',
        'truth_censored',
        'pred_response',
        'pred_onset',
        'pred_peak',
        'pred_offset'
    }
    miss = sorted(req - set(q.columns))
    if miss:
        raise RuntimeError(f'Seed {seed:02d} {pid}: missing columns {miss}.')
    if set(q['row'].astype(str)) != set(ROW_ORDER):
        raise RuntimeError(f'Seed {seed:02d} {pid}: five-row set mismatch.')
    counts = q.groupby('row').size().to_dict()
    if len(set(counts.values())) != 1:
        raise RuntimeError(f'Seed {seed:02d} {pid}: unequal trial counts {counts}.')
    for row in ROW_ORDER:
        if q[q['row'].astype(str) == row]['synthetic_index'].duplicated().any():
            raise RuntimeError(f'Seed {seed:02d} {pid} {row}: duplicate trial index.')
    base = q[q['row'].astype(str) == ROW_BASE].sort_values('synthetic_index').reset_index(drop=True)
    for row in ROW_ORDER[1:]:
        z = q[q['row'].astype(str) == row].sort_values('synthetic_index').reset_index(drop=True)
        for col in ('truth_present', 'truth_onset', 'truth_peak', 'truth_offset', 'truth_censored'):
            a = pd.to_numeric(base[col], errors='coerce').to_numpy(float)
            b = pd.to_numeric(z[col], errors='coerce').to_numpy(float)
            if not np.allclose(a, b, atol=1e-12, rtol=0.0, equal_nan=True):
                raise RuntimeError(f'Seed {seed:02d} {pid}: truth parity failed for {col}.')
def inheritance(seed: int, pid: str, sel):
    A, AB, ABC, FULL = (sel[ROW_A]['global'], sel[ROW_AB]['global'], sel[ROW_ABC]['global'], sel[ROW_FULL]['global'])
    LBR = sel[ROW_FULL]['local']
    p1 = all((np.isclose(float(AB[k]), float(A[k])) for k in ('state_C', 'timing_alpha')))
    p2 = all((np.isclose(float(ABC[k]), float(AB[k])) for k in ('state_C', 'timing_alpha', 'change_frac')))
    p3 = canonical_cfg(ABC) == canonical_cfg(FULL)
    return {
        'repeat_seed': seed,
        'participant': pid,
        'A_cfg': canonical_cfg(A),
        'AB_cfg': canonical_cfg(AB),
        'ABC_cfg': canonical_cfg(ABC),
        'LBR_global_parent': canonical_cfg(FULL),
        'LBR_cfg': canonical_cfg(LBR),
        'A_to_B_pass': bool(p1),
        'AB_to_C_pass': bool(p2),
        'ABC_to_LBR_pass': bool(p3),
        'pass': bool(p1 and p2 and p3)
    }
def load_complete(root: Path, seeds: Sequence[int]):
    pm_rows, inh_rows, sel_rows, per_trial_parts = ([], [], [], [])
    for seed in seeds:
        print(f'Loading seed {seed:02d}...', flush=True)
        for pid in ALL_PIDS:
            b = bundle(root, seed, pid)
            q = pd.read_csv(b['rows'])
            if 'repeat_seed' in q.columns:
                q['repeat_seed'] = seed
            else:
                q.insert(0, 'repeat_seed', seed)
            q['participant'] = q['participant'].astype(str)
            if set(q['participant'].unique()) != {pid}:
                raise RuntimeError(f'Seed {seed:02d} {pid}: participant ID mismatch.')
            validate_rows(q, seed, pid)
            per_trial_parts.append(q)
            sel = json.loads(b['selected'].read_text(encoding='utf-8'))
            inh_rows.append(inheritance(seed, pid, sel))
            for row in ROW_ORDER:
                z = q[q['row'].astype(str) == row].sort_values('synthetic_index').reset_index(drop=True)
                pm_rows.append({'repeat_seed': seed, 'participant': pid, 'row': row, 'configuration': ROW_SHORT[row], **metrics(z)})
            sel_rows.append({
                'repeat_seed': seed,
                'participant': pid,
                'row': ROW_BASE,
                'configuration': ROW_SHORT[ROW_BASE],
                'global_cfg': 'FIXED_CONVENTIONAL_BASE',
                'local_cfg': '',
                'inner_selected_loss': np.nan,
                'selection_stage': 'FIXED_BASE'
            })
            for row in (ROW_A, ROW_AB, ROW_ABC, ROW_FULL):
                it = sel[row]
                sel_rows.append({
                    'repeat_seed': seed,
                    'participant': pid,
                    'row': row,
                    'configuration': ROW_SHORT[row],
                    'global_cfg': canonical_cfg(it['global']),
                    'local_cfg': canonical_cfg(it['local']) if it.get('local') is not None else '',
                    'inner_selected_loss': float(it['inner_loss']),
                    'selection_stage': it.get('selection_stage', '')
                })
    inh = pd.DataFrame(inh_rows)
    if not inh['pass'].all():
        raise RuntimeError('Configuration inheritance failed:\n' + inh[~inh['pass']].to_string(index=False))
    return (pd.concat(per_trial_parts, ignore_index=True), pd.DataFrame(pm_rows), inh, pd.DataFrame(sel_rows))
def per_seed_summary(pm):
    rows = []
    for seed in sorted(pm['repeat_seed'].unique()):
        for row in ROW_ORDER:
            q = pm[(pm['repeat_seed'] == seed) & (pm['row'] == row)]
            r = {'repeat_seed': int(seed), 'row': row, 'configuration': ROW_SHORT[row], 'n_participants': q['participant'].nunique()}
            for m in METRICS:
                r[m] = float(np.nanmean(q[m].to_numpy(float)))
            rows.append(r)
    return pd.DataFrame(rows)
def across_seed_mean_sd(ss):
    rows = []
    for row in ROW_ORDER:
        q = ss[ss['row'] == row]
        r = {'row': row, 'configuration': ROW_SHORT[row], 'n_seeds': q['repeat_seed'].nunique()}
        for m in METRICS:
            x = q[m].to_numpy(float)
            r[m + '_mean'] = float(np.nanmean(x))
            r[m + '_sd'] = float(np.nanstd(x, ddof=1)) if len(x) > 1 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)
def incremental_gains(pm):
    rows = []
    for seed in sorted(pm['repeat_seed'].unique()):
        for pid in ALL_PIDS:
            p = pm[(pm['repeat_seed'] == seed) & (pm['participant'] == pid)].set_index('row')
            for ref, cand, label in TRANSITIONS:
                a, b = (p.loc[ref], p.loc[cand])
                for m in METRICS:
                    g = float(b[m] - a[m]) if m == 'response_balanced_accuracy' else float(a[m] - b[m])
                    rows.append({'repeat_seed': int(seed), 'participant': pid, 'transition': label, 'metric': m, 'gain_positive_added_component_better': g})
    return pd.DataFrame(rows)
def average_gains(g):
    pg = g.groupby([
        'participant',
        'transition',
        'metric'
    ], as_index=False)['gain_positive_added_component_better'].mean().rename(columns={'gain_positive_added_component_better': 'mean_gain_across_seeds'})
    sg = g.groupby([
        'repeat_seed',
        'transition',
        'metric'
    ], as_index=False)['gain_positive_added_component_better'].mean().rename(columns={'gain_positive_added_component_better': 'mean_gain_across_participants'})
    return (pg, sg)
def bootstrap_ci(x, seed):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(x))
    rng = np.random.default_rng(seed)
    vals = np.empty(BOOT_REPS, float)
    done = 0
    while done < BOOT_REPS:
        n = min(2000, BOOT_REPS - done)
        ix = rng.integers(0, len(x), size=(n, len(x)))
        vals[done:done + n] = np.mean(x[ix], axis=1)
        done += n
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return (mean, float(lo), float(hi))
def signflip_p(x, seed):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan
    obs = abs(float(np.mean(x)))
    if obs < 1e-15:
        return 1.0
    rng = np.random.default_rng(seed)
    hits = done = 0
    while done < SIGNFLIP_REPS:
        n = min(2000, SIGNFLIP_REPS - done)
        signs = rng.choice([-1.0, 1.0], size=(n, len(x)))
        stat = np.abs(np.mean(signs * x[None, :], axis=1))
        hits += int(np.sum(stat >= obs - 1e-15))
        done += n
    return float((hits + 1) / (SIGNFLIP_REPS + 1))
def holm(pvals):
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    run, m = (0.0, len(order))
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx])
        out[idx] = min(1.0, run)
    return out
def incremental_stats(pg, sg):
    rows = []
    for tr in ('+A', '+B', '+C', '+LBR'):
        for m in METRICS:
            x = pg[(pg['transition'] == tr) & (pg['metric'] == m)]['mean_gain_across_seeds'].to_numpy(float)
            sx = sg[(sg['transition'] == tr) & (sg['metric'] == m)]['mean_gain_across_participants'].to_numpy(float)
            if len(x) != 26 or len(sx) != 10:
                raise RuntimeError(f'{tr} {m}: expected 26 participant means and 10 seed means; got {len(x)}, {len(sx)}.')
            mean, lo, hi = bootstrap_ci(x, stable_seed('inc_boot', tr, m))
            p = signflip_p(x, stable_seed('inc_sign', tr, m))
            rows.append({
                'transition': tr,
                'metric': m,
                'n_participants': 26,
                'n_seeds': 10,
                'mean_gain_positive_added_component_better': mean,
                'ci95_low': lo,
                'ci95_high': hi,
                'participants_better': int(np.sum(x > 1e-12)),
                'participants_equal': int(np.sum(np.isclose(x, 0.0, atol=1e-12))),
                'participants_worse': int(np.sum(x < -1e-12)),
                'seeds_positive': int(np.sum(sx > 1e-12)),
                'seeds_equal': int(np.sum(np.isclose(sx, 0.0, atol=1e-12))),
                'seeds_negative': int(np.sum(sx < -1e-12)),
                'signflip_p': p
            })
    out = pd.DataFrame(rows)
    out['holm_p_4_timing_within_transition'] = np.nan
    for tr in out['transition'].unique():
        ix = out.index[(out['transition'] == tr) & out['metric'].isin(TIMING_METRICS)].tolist()
        out.loc[ix, 'holm_p_4_timing_within_transition'] = holm(out.loc[ix, 'signflip_p'].to_numpy(float))
    out['holm_p_16_all_incremental_timing'] = np.nan
    ix = out.index[out['metric'].isin(TIMING_METRICS)].tolist()
    out.loc[ix, 'holm_p_16_all_incremental_timing'] = holm(out.loc[ix, 'signflip_p'].to_numpy(float))
    return out
def base_full_stats(pm):
    raw = []
    for seed in ALL_SEEDS:
        for pid in ALL_PIDS:
            p = pm[(pm['repeat_seed'] == seed) & (pm['participant'] == pid)].set_index('row')
            a, b = (p.loc[ROW_BASE], p.loc[ROW_FULL])
            for m in METRICS:
                g = float(b[m] - a[m]) if m == 'response_balanced_accuracy' else float(a[m] - b[m])
                raw.append({'repeat_seed': seed, 'participant': pid, 'metric': m, 'gain': g})
    raw = pd.DataFrame(raw)
    pg = raw.groupby(['participant', 'metric'], as_index=False)['gain'].mean()
    sg = raw.groupby(['repeat_seed', 'metric'], as_index=False)['gain'].mean()
    rows = []
    for m in METRICS:
        x = pg[pg['metric'] == m]['gain'].to_numpy(float)
        sx = sg[sg['metric'] == m]['gain'].to_numpy(float)
        mean, lo, hi = bootstrap_ci(x, stable_seed('bf_boot', m))
        p = signflip_p(x, stable_seed('bf_sign', m))
        rows.append({
            'metric': m,
            'mean_gain_positive_HemoBoundary_better': mean,
            'ci95_low': lo,
            'ci95_high': hi,
            'participants_better': int(np.sum(x > 1e-12)),
            'participants_equal': int(np.sum(np.isclose(x, 0.0, atol=1e-12))),
            'participants_worse': int(np.sum(x < -1e-12)),
            'seeds_positive': int(np.sum(sx > 1e-12)),
            'seeds_equal': int(np.sum(np.isclose(sx, 0.0, atol=1e-12))),
            'seeds_negative': int(np.sum(sx < -1e-12)),
            'signflip_p': p
        })
    st = pd.DataFrame(rows)
    st['holm_p_4_timing'] = np.nan
    ix = st.index[st['metric'].isin(TIMING_METRICS)].tolist()
    st.loc[ix, 'holm_p_4_timing'] = holm(st.loc[ix, 'signflip_p'].to_numpy(float))
    return (raw, pg, sg, st)
def config_freq(sel):
    parts = []
    for row in (ROW_A, ROW_AB, ROW_ABC, ROW_FULL):
        q = sel[sel['row'] == row]
        cols = ['global_cfg'] + (['local_cfg'] if row == ROW_FULL else [])
        f = q.groupby(cols, dropna=False).agg(
            n_outer_folds=('participant', 'size'),
            n_unique_participants=('participant', 'nunique'),
            n_unique_seeds=('repeat_seed', 'nunique')
        ).reset_index().sort_values('n_outer_folds', ascending=False).reset_index(drop=True)
        f.insert(0, 'configuration', ROW_SHORT[row])
        parts.append(f)
    return pd.concat(parts, ignore_index=True)
def show(df, title, cols=None):
    print('\n' + '=' * 190)
    print(title)
    print('=' * 190)
    x = df[list(cols)] if cols else df
    print(x.to_string(index=False, float_format=lambda z: f'{z:.6f}'))
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--shin-root', default=None)
    p.add_argument('--output', default=None)
    p.add_argument(
        '--allow-partial',
        action='store_true',
        help='Preview fully complete seeds only; final inferential statistics are suppressed unless all 10 are complete.'
    )
    args, unknown = p.parse_known_args()
    if unknown:
        print('Ignoring notebook/runtime arguments:', unknown)
    return args
def main():
    args = parse_args()
    root = find_shin_root(args.shin_root)
    outdir = Path(args.output) if args.output else root / FINAL_DIRNAME
    outdir.mkdir(parents=True, exist_ok=True)
    print('=' * 190)
    print('HEMOBOUNDARY — TEN-SEED SUMMARY')
    print('=' * 190)
    print('Seed 1 :', root / SEED1_DIRNAME)
    print('Seeds 2-10:', root / MULTISEED_DIRNAME)
    print('Training: NO | Tuning: NO | Checkpoint modification: NO')
    print('Primary inference unit: PARTICIPANT')
    c = completion(root)
    sc = c.groupby('repeat_seed', as_index=False).agg(completed_outer_folds=('COMPLETE', 'sum'), total_outer_folds=('COMPLETE', 'size'))
    sc['seed_complete'] = sc['completed_outer_folds'] == 26
    show(sc, 'SEED COMPLETION')
    complete = sc[sc['seed_complete']]['repeat_seed'].astype(int).tolist()
    incomplete = sc[~sc['seed_complete']]['repeat_seed'].astype(int).tolist()
    if incomplete and (not args.allow_partial):
        missing = c[~c['COMPLETE']][['repeat_seed', 'participant', 'ROWS', 'SELECTED', 'INNER', 'AUDIT']]
        show(missing, 'INCOMPLETE CHECKPOINT BUNDLES')
        raise RuntimeError(f'Not all 10 seeds are complete. Complete seeds={complete}; incomplete={incomplete}. Use --allow-partial for a descriptive preview.')
    seeds = complete if args.allow_partial else list(ALL_SEEDS)
    if not seeds:
        raise RuntimeError('No complete seeds available.')
    per_trial, pm, inh, sel = load_complete(root, seeds)
    print(f'\nHIERARCHICAL SELECTION VERIFIED: {len(inh)}/{len(inh)} outer folds')
    ss = per_seed_summary(pm)
    for seed in seeds:
        q = ss[ss['repeat_seed'] == seed]
        show(q, f'SEED {seed:02d} — PARTICIPANT-BALANCED ABLATION', [
            'configuration',
            'response_balanced_accuracy',
            'onset_mae_sec',
            'peak_mae_sec',
            'offset_mae_sec',
            'duration_mae_sec',
            'selection_loss'
        ])
    ms = across_seed_mean_sd(ss)
    show(ms, f"{('FINAL' if len(seeds) == 10 else 'PARTIAL')} MEAN ± SD ACROSS {len(seeds)} SEEDS", [
        'configuration',
        'n_seeds',
        'response_balanced_accuracy_mean',
        'response_balanced_accuracy_sd',
        'onset_mae_sec_mean',
        'onset_mae_sec_sd',
        'peak_mae_sec_mean',
        'peak_mae_sec_sd',
        'offset_mae_sec_mean',
        'offset_mae_sec_sd',
        'duration_mae_sec_mean',
        'duration_mae_sec_sd',
        'selection_loss_mean',
        'selection_loss_sd'
    ])
    freq = config_freq(sel)
    show(freq, 'CONFIGURATION STABILITY')
    atomic_csv(pm, outdir / 'D1_SEED_PARTICIPANT_METRICS.csv.gz')
    atomic_csv(ss, outdir / 'T1_PER_SEED_ABLATION.csv')
    atomic_csv(ms, outdir / 'T2_10SEED_MEAN_SD_ABLATION.csv')
    atomic_csv(inh, outdir / 'AUDIT_STRICT_INHERITANCE.csv.gz')
    atomic_csv(sel, outdir / 'D2_SELECTED_CONFIGS.csv.gz')
    atomic_csv(freq, outdir / 'T3_CONFIGURATION_FREQUENCIES.csv')
    if len(seeds) != 10:
        atomic_json({
            'version': VERSION,
            'status': 'PARTIAL_PREVIEW',
            'complete_seeds': seeds,
            'incomplete_seeds': incomplete,
            'strict_inheritance_pass': bool(inh['pass'].all()),
            'training_performed': False,
            'tuning_performed': False
        }, outdir / 'PARTIAL_PREVIEW_MANIFEST.json')
        print('\nPARTIAL PREVIEW COMPLETE — final participant-level inference not run.')
        return
    gains = incremental_gains(pm)
    pg, sg = average_gains(gains)
    st = incremental_stats(pg, sg)
    show(st, 'FINAL INCREMENTAL COMPONENT STATISTICS — PARTICIPANT IS INFERENCE UNIT', [
        'transition',
        'metric',
        'mean_gain_positive_added_component_better',
        'ci95_low',
        'ci95_high',
        'participants_better',
        'participants_equal',
        'participants_worse',
        'seeds_positive',
        'seeds_equal',
        'seeds_negative',
        'signflip_p',
        'holm_p_4_timing_within_transition',
        'holm_p_16_all_incremental_timing'
    ])
    bf_raw, bf_pg, bf_sg, bf_st = base_full_stats(pm)
    show(bf_st, 'BASE VS FINAL HEMOBOUNDARY — PARTICIPANT-LEVEL 10-SEED INFERENCE')
    atomic_csv(per_trial, outdir / 'D0_ALL_10SEED_PER_TRIAL.csv.gz')
    atomic_csv(gains, outdir / 'D3_INCREMENTAL_GAINS_SEED_PARTICIPANT.csv.gz')
    atomic_csv(pg, outdir / 'D4_INCREMENTAL_GAINS_PARTICIPANT_AVERAGED_10SEEDS.csv')
    atomic_csv(sg, outdir / 'D5_INCREMENTAL_GAINS_SEED_AVERAGED_26PARTICIPANTS.csv')
    atomic_csv(st, outdir / 'TABLE_FINAL_INCREMENTAL_COMPONENT_STATS.csv')
    atomic_csv(bf_raw, outdir / 'D6_BASE_VS_FULL_GAINS_SEED_PARTICIPANT.csv.gz')
    atomic_csv(bf_pg, outdir / 'D7_BASE_VS_FULL_GAINS_PARTICIPANT_AVERAGED_10SEEDS.csv')
    atomic_csv(bf_sg, outdir / 'D8_BASE_VS_FULL_GAINS_SEED_AVERAGED_26PARTICIPANTS.csv')
    atomic_csv(bf_st, outdir / 'TABLE_FINAL_BASE_VS_HEMOBOUNDARY_STATS.csv')
    atomic_json({
        'version': VERSION,
        'status': 'COMPLETE',
        'seeds': list(ALL_SEEDS),
        'n_seeds': 10,
        'participants': list(ALL_PIDS),
        'n_participants': 26,
        'n_seed_participant_outer_folds': 260,
        'strict_inheritance_pass': bool(inh['pass'].all()),
        'primary_inference_unit': 'participant',
        'participant_gain_definition': 'For each participant, component gain is averaged across 10 seeds before bootstrap/sign-flip inference.',
        'positive_gain_definition': 'BA uses candidate-reference; MAE/loss uses reference-candidate; positive always means better.',
        'bootstrap_reps': BOOT_REPS,
        'signflip_reps': SIGNFLIP_REPS,
        'miss_penalty_sec': MISS_PENALTY_SEC,
        'training_performed': False,
        'tuning_performed': False,
        'checkpoint_modification': False
    }, outdir / 'FINAL_10SEED_MANIFEST.json')
    print('\n' + '=' * 190)
    print('FINAL 10-SEED MERGE COMPLETE')
    print('=' * 190)
    print('Seeds: 10/10')
    print('Participants per seed: 26/26')
    print('Seed-participant outer folds: 260')
    print('Configuration inheritance verified: 260/260')
    print('Primary inference unit: participant (n=26)')
    print('Final output:', outdir)
if __name__ == '__main__':
    main()
