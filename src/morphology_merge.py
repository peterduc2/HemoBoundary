"""Morphology robustness summaries."""
from __future__ import annotations
import os
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
DRIVE_ROOT = Path(os.environ.get('HEMOBOUNDARY_MYDRIVE', '/content/gdrive/MyDrive'))
SHIN_ROOT = Path(os.environ.get('HEMOBOUNDARY_SHIN_ROOT', str(DRIVE_ROOT / 'HonorProject' / 'HemoBoundary' / 'SHIN')))
ROOT = SHIN_ROOT / '111_OOD_HRF_SHAPE_STRESS_10SEED_V1_STANDALONE'
UNIFIED_JOB = ROOT / 'all_seeds_01_10'
OUT = ROOT / 'FINAL_MERGE'
SEEDS = tuple(range(1, 11))
PIDS = tuple((f'VP{i:03d}' for i in range(1, 27)))
FAMILIES = ('PAPER_MATCHED_CACHE', 'MATCHED_HALF_COSINE_NEW', 'OOD_SKEWED_BETA', 'OOD_BIPHASIC_UNDERSHOOT', 'OOD_BIEXPONENTIAL_TAIL', 'OOD_SHOULDER_DOUBLE_PEAK')
OOD = ('OOD_SKEWED_BETA', 'OOD_BIPHASIC_UNDERSHOOT', 'OOD_BIEXPONENTIAL_TAIL', 'OOD_SHOULDER_DOUBLE_PEAK')
MATCHED = 'MATCHED_HALF_COSINE_NEW'
METHODS = ('FULL', 'ABC_PARENT')
METRICS = ('response_balanced_accuracy', 'onset_mae_sec', 'peak_mae_sec', 'offset_mae_sec', 'duration_mae_sec', 'selection_loss')
LABEL = {
    'response_balanced_accuracy': 'Response BA',
    'onset_mae_sec': 'Onset MAE (s)',
    'peak_mae_sec': 'Peak MAE (s)',
    'offset_mae_sec': 'Offset MAE (s)',
    'duration_mae_sec': 'Duration MAE (s)',
    'selection_loss': 'Selection loss'
}
LOWER_BETTER = {
    'response_balanced_accuracy': False,
    'onset_mae_sec': True,
    'peak_mae_sec': True,
    'offset_mae_sec': True,
    'duration_mae_sec': True,
    'selection_loss': True
}
BOOT_REPS = 20000
SIGNFLIP_REPS = 20000
TOL = 1e-12
def mount_drive():
    if DRIVE_ROOT.is_dir():
        return
    from google.colab import drive
    drive.mount(str(DRIVE_ROOT.parent), force_remount=False)
def job(seed):
    return UNIFIED_JOB
def p_metric(seed, pid):
    return job(seed) / 'checkpoints' / f'seed_{seed:02d}_{pid}_METRICS.csv'
def p_trial(seed, pid):
    return job(seed) / 'checkpoints' / f'seed_{seed:02d}_{pid}_TRIALS.csv.gz'
def p_shape(seed, pid):
    return job(seed) / 'checkpoints' / f'seed_{seed:02d}_{pid}_SHAPE_PARAMETERS.csv.gz'
def p_audit(seed, pid):
    return job(seed) / 'checkpoints' / f'seed_{seed:02d}_{pid}_AUDIT.csv.gz'
def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    comp = 'gzip' if str(path).endswith('.gz') else None
    df.to_csv(tmp, index=False, compression=comp)
    tmp.replace(path)
def hseed(*parts):
    raw = '|'.join(map(str, parts)).encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16) % (2 ** 31 - 1)
def bootstrap_ci(x, seed):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(x))
    if len(x) == 1:
        return (mean, mean, mean)
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
    if len(x) == 0:
        return np.nan
    obs = abs(float(np.mean(x)))
    if obs <= 1e-15:
        return 1.0
    rng = np.random.default_rng(seed)
    hits = 0
    done = 0
    while done < SIGNFLIP_REPS:
        n = min(2000, SIGNFLIP_REPS - done)
        signs = rng.choice([-1.0, 1.0], size=(n, len(x)))
        stat = np.abs(np.mean(signs * x[None, :], axis=1))
        hits += int(np.sum(stat >= obs - 1e-15))
        done += n
    return float((hits + 1) / (SIGNFLIP_REPS + 1))
def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
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
def diff_positive_reference_better(reference, candidate, metric):
    a = np.asarray(reference, float)
    b = np.asarray(candidate, float)
    if LOWER_BETTER[metric]:
        return b - a
    return a - b
def completion():
    rows = []
    for seed in SEEDS:
        for pid in PIDS:
            mp, tp, sp, ap = (p_metric(seed, pid), p_trial(seed, pid), p_shape(seed, pid), p_audit(seed, pid))
            rows.append({
                'repeat_seed': seed,
                'participant': pid,
                'metrics': mp.is_file(),
                'trials': tp.is_file(),
                'shape_params': sp.is_file(),
                'fit_audit': ap.is_file(),
                'complete': mp.is_file() and tp.is_file() and sp.is_file() and ap.is_file()
            })
    return pd.DataFrame(rows)
def load_metrics():
    parts = [pd.read_csv(p_metric(s, p)) for s in SEEDS for p in PIDS]
    d = pd.concat(parts, ignore_index=True)
    required = {'repeat_seed', 'participant', 'family', 'method', *METRICS}
    missing = required - set(d.columns)
    if missing:
        raise RuntimeError(f'Missing metric columns: {sorted(missing)}')
    if d.duplicated(['repeat_seed', 'participant', 'family', 'method']).any():
        raise RuntimeError('Duplicate participant metric rows.')
    expected = 10 * 26 * 6 * 2
    if len(d) != expected:
        raise RuntimeError(f'Expected {expected} metric rows, found {len(d)}.')
    if set(d['family'].astype(str)) != set(FAMILIES):
        raise RuntimeError('Family coverage mismatch.')
    if set(d['method'].astype(str)) != set(METHODS):
        raise RuntimeError('Method coverage mismatch.')
    return d
def load_trials():
    parts = [pd.read_csv(p_trial(s, p)) for s in SEEDS for p in PIDS]
    d = pd.concat(parts, ignore_index=True)
    keys = ['repeat_seed', 'participant', 'family', 'method', 'trial_index']
    if d.duplicated(keys).any():
        raise RuntimeError('Duplicate per-trial rows.')
    counts = d.groupby(keys[:-1])['trial_index'].nunique()
    if not np.all(counts.to_numpy() == 36):
        raise RuntimeError('Expected exactly 36 trials per seed-participant-family-method.')
    expected = 10 * 26 * 6 * 2 * 36
    if len(d) != expected:
        raise RuntimeError(f'Expected {expected} per-trial rows, found {len(d)}.')
    return d
def load_shapes():
    parts = [pd.read_csv(p_shape(s, p)) for s in SEEDS for p in PIDS]
    d = pd.concat(parts, ignore_index=True)
    keys = ['repeat_seed', 'participant', 'positive_trial_index', 'family']
    if d.duplicated(keys).any():
        raise RuntimeError('Duplicate stress-shape parameter rows.')
    expected = 10 * 26 * 24 * 5
    if len(d) != expected:
        raise RuntimeError(f'Expected {expected} shape rows, found {len(d)}.')
    return d
def participant_average(metrics):
    ignore = {'repeat_seed', 'participant', 'family', 'method', 'generator_class', 'selected_abc_cfg', 'selected_lbr_cfg'}
    numeric = [c for c in metrics.columns if c not in ignore and pd.api.types.is_numeric_dtype(metrics[c])]
    avg = metrics.groupby(['participant', 'family', 'method'], as_index=False)[numeric].mean()
    nseed = metrics.groupby(['participant', 'family', 'method'], as_index=False)['repeat_seed'].nunique().rename(columns={'repeat_seed': 'n_seeds_averaged'})
    avg = avg.merge(nseed, on=['participant', 'family', 'method'], validate='one_to_one')
    if not np.all(avg['n_seeds_averaged'].to_numpy() == 10):
        raise RuntimeError('Not all participant rows averaged exactly 10 seeds.')
    avg['generator_class'] = np.where(avg['family'].isin(OOD), 'OUT_OF_GENERATOR', 'MATCHED_REFERENCE')
    if len(avg) != 26 * 6 * 2:
        raise RuntimeError('Participant-averaged row count mismatch.')
    return avg
def full_means(avg):
    q = avg[avg['method'] == 'FULL'].copy()
    rows = []
    for family in FAMILIES:
        z = q[q['family'] == family]
        row = {'family': family, 'generator_class': 'OUT_OF_GENERATOR' if family in OOD else 'MATCHED_REFERENCE', 'n_participants': z['participant'].nunique()}
        for metric in METRICS:
            x = pd.to_numeric(z[metric], errors='coerce').to_numpy(float)
            row[f'{metric}_mean'] = float(np.nanmean(x))
            row[f'{metric}_sd'] = float(np.nanstd(x, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)
def shift_stats(avg):
    full = avg[avg['method'] == 'FULL'].copy()
    matched = full[full['family'] == MATCHED].set_index('participant').sort_index()
    rows = []
    for family in OOD:
        ood = full[full['family'] == family].set_index('participant').sort_index()
        common = matched.index.intersection(ood.index)
        if len(common) != 26:
            raise RuntimeError(f'{family}: not 26 paired participants.')
        for metric in METRICS:
            a = pd.to_numeric(matched.loc[common, metric], errors='coerce').to_numpy(float)
            b = pd.to_numeric(ood.loc[common, metric], errors='coerce').to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = (a[ok], b[ok])
            penalty = diff_positive_reference_better(a, b, metric)
            mean, lo, hi = bootstrap_ci(penalty, hseed('SHIFT_BOOT', family, metric))
            p = signflip_p(penalty, hseed('SHIFT_SIGN', family, metric))
            rows.append({
                'reference_family': MATCHED,
                'ood_family': family,
                'metric': metric,
                'metric_label': LABEL[metric],
                'n_participants': len(penalty),
                'matched_mean': float(np.mean(a)),
                'ood_mean': float(np.mean(b)),
                'mean_penalty_positive_ood_worse': mean,
                'ci95_low': lo,
                'ci95_high': hi,
                'participants_ood_worse': int(np.sum(penalty > TOL)),
                'participants_equal': int(np.sum(np.abs(penalty) <= TOL)),
                'participants_ood_better': int(np.sum(penalty < -TOL)),
                'raw_signflip_p': p
            })
    out = pd.DataFrame(rows)
    out['holm_p_within_family_6'] = np.nan
    for family in OOD:
        ix = out.index[out['ood_family'] == family]
        out.loc[ix, 'holm_p_within_family_6'] = holm(out.loc[ix, 'raw_signflip_p'].to_numpy(float))
    out['holm_p_global_24'] = holm(out['raw_signflip_p'].to_numpy(float))
    return out
def lbr_stats(avg):
    rows = []
    for family in FAMILIES:
        full = avg[(avg['family'] == family) & (avg['method'] == 'FULL')].set_index('participant').sort_index()
        abc = avg[(avg['family'] == family) & (avg['method'] == 'ABC_PARENT')].set_index('participant').sort_index()
        common = full.index.intersection(abc.index)
        if len(common) != 26:
            raise RuntimeError(f'{family}: not 26 FULL/ABC pairs.')
        for metric in METRICS:
            a = pd.to_numeric(full.loc[common, metric], errors='coerce').to_numpy(float)
            b = pd.to_numeric(abc.loc[common, metric], errors='coerce').to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = (a[ok], b[ok])
            gain = diff_positive_reference_better(a, b, metric)
            mean, lo, hi = bootstrap_ci(gain, hseed('LBR_BOOT', family, metric))
            p = signflip_p(gain, hseed('LBR_SIGN', family, metric))
            rows.append({
                'family': family,
                'generator_class': 'OUT_OF_GENERATOR' if family in OOD else 'MATCHED_REFERENCE',
                'metric': metric,
                'metric_label': LABEL[metric],
                'n_participants': len(gain),
                'full_mean': float(np.mean(a)),
                'abc_parent_mean': float(np.mean(b)),
                'mean_gain_positive_full_better': mean,
                'ci95_low': lo,
                'ci95_high': hi,
                'participants_full_better': int(np.sum(gain > TOL)),
                'participants_equal': int(np.sum(np.abs(gain) <= TOL)),
                'participants_abc_better': int(np.sum(gain < -TOL)),
                'raw_signflip_p': p
            })
    out = pd.DataFrame(rows)
    out['holm_p_within_family_6'] = np.nan
    for family in FAMILIES:
        ix = out.index[out['family'] == family]
        out.loc[ix, 'holm_p_within_family_6'] = holm(out.loc[ix, 'raw_signflip_p'].to_numpy(float))
    out['holm_p_global_36'] = holm(out['raw_signflip_p'].to_numpy(float))
    return out
def pooled_ood(avg):
    q = avg[avg['family'].isin(OOD)].copy()
    ignore = {'participant', 'family', 'method', 'generator_class'}
    numeric = [c for c in q.columns if c not in ignore and pd.api.types.is_numeric_dtype(q[c])]
    p = q.groupby(['participant', 'method'], as_index=False)[numeric].mean()
    p.insert(1, 'family', 'POOLED_OOD_4_FAMILIES')
    p.insert(2, 'generator_class', 'OUT_OF_GENERATOR')
    return p
def pooled_shift(avg, pooled):
    matched = avg[(avg['family'] == MATCHED) & (avg['method'] == 'FULL')].set_index('participant').sort_index()
    ood = pooled[pooled['method'] == 'FULL'].set_index('participant').sort_index()
    rows = []
    for metric in METRICS:
        a = pd.to_numeric(matched[metric], errors='coerce').to_numpy(float)
        b = pd.to_numeric(ood[metric], errors='coerce').to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = (a[ok], b[ok])
        d = diff_positive_reference_better(a, b, metric)
        mean, lo, hi = bootstrap_ci(d, hseed('POOL_SHIFT_BOOT', metric))
        p = signflip_p(d, hseed('POOL_SHIFT_SIGN', metric))
        rows.append({
            'metric': metric,
            'metric_label': LABEL[metric],
            'n_participants': len(d),
            'matched_mean': float(np.mean(a)),
            'pooled_ood_mean': float(np.mean(b)),
            'mean_penalty_positive_ood_worse': mean,
            'ci95_low': lo,
            'ci95_high': hi,
            'participants_ood_worse': int(np.sum(d > TOL)),
            'participants_equal': int(np.sum(np.abs(d) <= TOL)),
            'participants_ood_better': int(np.sum(d < -TOL)),
            'raw_signflip_p': p
        })
    out = pd.DataFrame(rows)
    out['holm_p_6'] = holm(out['raw_signflip_p'].to_numpy(float))
    return out
def pooled_lbr(pooled):
    full = pooled[pooled['method'] == 'FULL'].set_index('participant').sort_index()
    abc = pooled[pooled['method'] == 'ABC_PARENT'].set_index('participant').sort_index()
    rows = []
    for metric in METRICS:
        a = pd.to_numeric(full[metric], errors='coerce').to_numpy(float)
        b = pd.to_numeric(abc[metric], errors='coerce').to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = (a[ok], b[ok])
        d = diff_positive_reference_better(a, b, metric)
        mean, lo, hi = bootstrap_ci(d, hseed('POOL_LBR_BOOT', metric))
        p = signflip_p(d, hseed('POOL_LBR_SIGN', metric))
        rows.append({
            'metric': metric,
            'metric_label': LABEL[metric],
            'n_participants': len(d),
            'full_mean': float(np.mean(a)),
            'abc_parent_mean': float(np.mean(b)),
            'mean_gain_positive_full_better': mean,
            'ci95_low': lo,
            'ci95_high': hi,
            'participants_full_better': int(np.sum(d > TOL)),
            'participants_equal': int(np.sum(np.abs(d) <= TOL)),
            'participants_abc_better': int(np.sum(d < -TOL)),
            'raw_signflip_p': p
        })
    out = pd.DataFrame(rows)
    out['holm_p_6'] = holm(out['raw_signflip_p'].to_numpy(float))
    return out
def seedwise(metrics):
    q = metrics[metrics['method'] == 'FULL'].copy()
    return q.groupby(['repeat_seed', 'family'], as_index=False).agg(
        n_participants=('participant', 'nunique'),
        response_BA=('response_balanced_accuracy', 'mean'),
        onset_MAE=('onset_mae_sec', 'mean'),
        peak_MAE=('peak_mae_sec', 'mean'),
        offset_MAE=('offset_mae_sec', 'mean'),
        duration_MAE=('duration_mae_sec', 'mean'),
        selection_loss=('selection_loss', 'mean')
    ).sort_values(['repeat_seed', 'family']).reset_index(drop=True)
def self_tests():
    a = np.array([1.0, 1.0])
    b = np.array([2.0, 2.0])
    assert np.allclose(diff_positive_reference_better(a, b, 'onset_mae_sec'), 1.0)
    a = np.array([0.9, 0.9])
    b = np.array([0.5, 0.5])
    assert np.allclose(diff_positive_reference_better(a, b, 'response_balanced_accuracy'), 0.4)
    hh = holm(np.array([0.01, 0.02, 0.5]))
    assert np.all(np.isfinite(hh))
    assert np.all((hh >= 0) & (hh <= 1))
    print('OOD merge protected self-tests: PASS')
def main():
    mount_drive()
    self_tests()
    OUT.mkdir(parents=True, exist_ok=True)
    print('=' * 180)
    print('HEMOBOUNDARY — OOD HRF SHAPE STRESS 10-SEED MERGE + PRINT')
    print('=' * 180)
    print('Model fitting                  : NO')
    print('Nested selection               : NO')
    print('Stress-family tuning           : NO')
    print('Manual path input              : NO')
    print('Expected seeds                 : 10')
    print('Expected participants/seed     : 26')
    print('Expected families              : 6')
    print('Expected methods               : 2')
    print('Final statistical unit         : participant after averaging 10 seeds')
    print('Bootstrap reps                 :', BOOT_REPS)
    print('Sign-flip reps                 :', SIGNFLIP_REPS)
    print('Output                         :', OUT)
    print()
    audit = completion()
    save_csv(audit, OUT / 'COMPLETION_AUDIT.csv')
    missing = audit[~audit['complete']]
    if len(missing):
        print('INCOMPLETE:')
        print(missing[['repeat_seed', 'participant', 'metrics', 'trials', 'shape_params', 'fit_audit']].to_string(index=False))
        raise RuntimeError(f'{len(missing)} / 260 checkpoint bundles incomplete.')
    print('COMPLETION CHECK               : PASS — 260 / 260')
    metrics = load_metrics()
    save_csv(metrics.sort_values(['repeat_seed', 'participant', 'family', 'method']), OUT / 'MERGED_PARTICIPANT_METRICS_ALL_10_SEEDS.csv')
    print('Participant metric rows        :', len(metrics), '/ 3120')
    print('Merging per-trial predictions...')
    trials = load_trials()
    save_csv(trials.sort_values(['repeat_seed', 'participant', 'family', 'method', 'trial_index']), OUT / 'MERGED_PER_TRIAL_PREDICTIONS_ALL_10_SEEDS.csv.gz')
    print('Per-trial rows                 :', len(trials), '/ 112320')
    print('Merging stress-shape parameters...')
    shapes = load_shapes()
    save_csv(shapes.sort_values(['repeat_seed', 'participant', 'family', 'positive_trial_index']), OUT / 'MERGED_STRESS_SHAPE_PARAMETERS_ALL_10_SEEDS.csv.gz')
    print('Stress-shape parameter rows    :', len(shapes), '/ 31200')
    avg = participant_average(metrics)
    save_csv(avg.sort_values(['participant', 'family', 'method']), OUT / 'FINAL_PARTICIPANT_AVERAGED_METRICS.csv')
    t1 = full_means(avg)
    t2 = shift_stats(avg)
    t3 = lbr_stats(avg)
    t4 = seedwise(metrics)
    pool = pooled_ood(avg)
    t6 = pooled_shift(avg, pool)
    t7 = pooled_lbr(pool)
    save_csv(t1, OUT / 'TABLE1_FULL_HEMOBOUNDARY_BY_FAMILY.csv')
    save_csv(t2, OUT / 'TABLE2_FULL_GENERATOR_SHIFT_PAIRED_STATS.csv')
    save_csv(t3, OUT / 'TABLE3_LBR_FULL_VS_ABC_BY_FAMILY.csv')
    save_csv(t4, OUT / 'TABLE4_SEEDWISE_FULL_ROBUSTNESS.csv')
    save_csv(pool, OUT / 'TABLE5_POOLED_OOD_PARTICIPANT_METRICS.csv')
    save_csv(t6, OUT / 'TABLE6_POOLED_OOD_SHIFT_STATS.csv')
    save_csv(t7, OUT / 'TABLE7_POOLED_OOD_LBR_STATS.csv')
    compact1 = pd.DataFrame({
        'family': t1['family'],
        'class': t1['generator_class'],
        'BA': t1['response_balanced_accuracy_mean'],
        'onset_MAE': t1['onset_mae_sec_mean'],
        'peak_MAE': t1['peak_mae_sec_mean'],
        'offset_MAE': t1['offset_mae_sec_mean'],
        'duration_MAE': t1['duration_mae_sec_mean'],
        'selection_loss': t1['selection_loss_mean']
    })
    print('\n' + '=' * 180)
    print('TABLE 1 — FULL HEMOBOUNDARY PERFORMANCE BY SHAPE FAMILY')
    print('=' * 180)
    print(compact1.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print('\n' + '=' * 180)
    print('TABLE 2 — GENERATOR-SHIFT PENALTIES: FRESH MATCHED CONTROL vs OOD')
    print('=' * 180)
    print('Positive penalty = OOD is WORSE.')
    print('BA: matched - OOD. Timing/loss: OOD - matched.\n')
    print(t2[[
        'ood_family',
        'metric_label',
        'matched_mean',
        'ood_mean',
        'mean_penalty_positive_ood_worse',
        'ci95_low',
        'ci95_high',
        'participants_ood_worse',
        'participants_equal',
        'participants_ood_better',
        'raw_signflip_p',
        'holm_p_within_family_6',
        'holm_p_global_24'
    ]].to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print('\n' + '=' * 180)
    print('TABLE 3 — LBR ROBUSTNESS: FULL vs ABC_PARENT')
    print('=' * 180)
    print('Positive gain = FULL HemoBoundary is BETTER.\n')
    print(t3[[
        'family',
        'metric_label',
        'full_mean',
        'abc_parent_mean',
        'mean_gain_positive_full_better',
        'ci95_low',
        'ci95_high',
        'participants_full_better',
        'participants_equal',
        'participants_abc_better',
        'raw_signflip_p',
        'holm_p_within_family_6',
        'holm_p_global_36'
    ]].to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print('\n' + '=' * 180)
    print('POOLED 4-OOD-FAMILY GENERATOR-SHIFT SUMMARY')
    print('=' * 180)
    print(t6.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print('\n' + '=' * 180)
    print('POOLED 4-OOD-FAMILY LBR EFFECT')
    print('=' * 180)
    print(t7.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print('\n' + '=' * 180)
    print('SEEDWISE FULL-MODEL ROBUSTNESS')
    print('=' * 180)
    print(t4.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    summary = [
        'HEMOBOUNDARY — OOD HRF SHAPE STRESS — FINAL 10-SEED MERGE',
        '=' * 110,
        '',
        'COMPLETION',
        '10/10 seeds',
        '26/26 participants per seed',
        '6/6 shape families',
        '2/2 methods',
        f'Participant metric rows: {len(metrics)}/3120',
        f'Per-trial rows: {len(trials)}/112320',
        f'Stress-shape parameter rows: {len(shapes)}/31200',
        '',
        'FULL HEMOBOUNDARY BY FAMILY',
        compact1.to_string(index=False, float_format=lambda x: f'{x:.6f}'),
        '',
        'GENERATOR-SHIFT PAIRED TESTS',
        t2[['ood_family', 'metric_label', 'matched_mean', 'ood_mean', 'mean_penalty_positive_ood_worse', 'ci95_low', 'ci95_high', 'raw_signflip_p', 'holm_p_global_24']].to_string(index=False, float_format=lambda x: f'{x:.6f}'),
        '',
        'LBR FULL vs ABC_PARENT',
        t3[['family', 'metric_label', 'full_mean', 'abc_parent_mean', 'mean_gain_positive_full_better', 'ci95_low', 'ci95_high', 'raw_signflip_p', 'holm_p_global_36']].to_string(index=False, float_format=lambda x: f'{x:.6f}'),
        '',
        'POOLED OOD SHIFT',
        t6.to_string(index=False, float_format=lambda x: f'{x:.6f}'),
        '',
        'POOLED OOD LBR',
        t7.to_string(index=False, float_format=lambda x: f'{x:.6f}'),
        '',
        'INFERENCE',
        'Statistical unit: participant after averaging 10 seeds within participant.',
        f'Bootstrap reps: {BOOT_REPS}.',
        f'Sign-flip reps: {SIGNFLIP_REPS}.',
        'Generator-shift Holm: 6 metrics within each OOD family and 24 tests globally.',
        'LBR Holm: 6 metrics within each family and 36 tests globally.',
        '',
        'INTERPRETATION BOUNDARY',
        'These are known-boundary semi-synthetic generator-shift stress tests, not biological timing ground truth.'
    ]
    (OUT / 'FINAL_SUMMARY.txt').write_text('\n'.join(summary), encoding='utf-8')
    print('\n' + '=' * 180)
    print('FINAL OOD MERGE: PASS')
    print('=' * 180)
    print('Saved:', OUT)
if __name__ == '__main__':
    main()
