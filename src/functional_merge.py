"""Functional necessity summaries and paired tests."""
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
DRIVE_ROOT = Path(os.environ.get('HEMOBOUNDARY_MYDRIVE', '/content/gdrive/MyDrive'))
SHIN_ROOT = Path(os.environ.get('HEMOBOUNDARY_SHIN_ROOT', str(DRIVE_ROOT / 'HonorProject' / 'HemoBoundary' / 'SHIN')))
ANALYSIS_ROOT = SHIN_ROOT / '110_FUNCTIONAL_NECESSITY_ABLATION_10SEED_V2_STANDALONE'
FINAL_OUT = ANALYSIS_ROOT / 'FINAL_MERGE'
UNIFIED_JOB = ANALYSIS_ROOT / 'all_seeds_01_10'
SEED_TO_JOB = {seed: UNIFIED_JOB for seed in range(1, 11)}
ALL_SEEDS = tuple(range(1, 11))
ALL_PIDS = tuple((f'VP{i:03d}' for i in range(1, 27)))
VARIANTS = ('FULL', 'A_OUTPUT_NEUTRALIZED', 'B_OUTPUT_NEUTRALIZED', 'C_GATE_NEUTRALIZED', 'LBR_NEUTRALIZED')
COMPONENT = {
    'A_OUTPUT_NEUTRALIZED': 'A — Temporal Boundary Estimation',
    'B_OUTPUT_NEUTRALIZED': 'B — Recovery Estimation',
    'C_GATE_NEUTRALIZED': 'C — Response Detection',
    'LBR_NEUTRALIZED': 'LBR — Local Boundary Refinement'
}
METRICS = ('response_balanced_accuracy', 'e2e_onset_mae_sec', 'e2e_peak_mae_sec', 'e2e_offset_mae_sec', 'e2e_duration_mae_sec', 'e2e_selection_loss')
METRIC_LABEL = {
    'response_balanced_accuracy': 'Response BA',
    'e2e_onset_mae_sec': 'Onset MAE (s)',
    'e2e_peak_mae_sec': 'Peak MAE (s)',
    'e2e_offset_mae_sec': 'Offset MAE (s)',
    'e2e_duration_mae_sec': 'Duration MAE (s)',
    'e2e_selection_loss': 'Selection loss'
}
LOWER_IS_BETTER = {
    'response_balanced_accuracy': False,
    'e2e_onset_mae_sec': True,
    'e2e_peak_mae_sec': True,
    'e2e_offset_mae_sec': True,
    'e2e_duration_mae_sec': True,
    'e2e_selection_loss': True
}
BOOT_REPS = 20000
SIGNFLIP_REPS = 20000
TOL = 1e-12

def mount_drive():
    if DRIVE_ROOT.is_dir():
        return
    from google.colab import drive
    drive.mount(str(DRIVE_ROOT.parent), force_remount=False)
def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    compression = 'gzip' if str(path).endswith('.gz') else None
    df.to_csv(tmp, index=False, compression=compression)
    tmp.replace(path)
def metric_checkpoint(seed: int, pid: str) -> Path:
    return SEED_TO_JOB[seed] / 'checkpoints' / f'seed_{seed:02d}_{pid}_METRICS.csv'
def trial_checkpoint(seed: int, pid: str) -> Path:
    return SEED_TO_JOB[seed] / 'checkpoints' / f'seed_{seed:02d}_{pid}_TRIALS.csv.gz'
def bootstrap_ci(x: np.ndarray, seed: int) -> Tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(x))
    if len(x) == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    sims = np.empty(BOOT_REPS, float)
    done = 0
    while done < BOOT_REPS:
        n = min(2000, BOOT_REPS - done)
        ix = rng.integers(0, len(x), size=(n, len(x)))
        sims[done:done + n] = np.mean(x[ix], axis=1)
        done += n
    lo, hi = np.quantile(sims, [0.025, 0.975])
    return (mean, float(lo), float(hi))
def signflip_p(x: np.ndarray, seed: int) -> float:
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
        signs = rng.choice((-1.0, 1.0), size=(n, len(x)))
        stats = np.abs(np.mean(signs * x[None, :], axis=1))
        hits += int(np.sum(stats >= obs - 1e-15))
        done += n
    return float((hits + 1) / (SIGNFLIP_REPS + 1))
def holm_adjust(pvals: np.ndarray) -> np.ndarray:
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
def full_better_gain(full: np.ndarray, neutral: np.ndarray, metric: str) -> np.ndarray:
    full = np.asarray(full, float)
    neutral = np.asarray(neutral, float)
    if LOWER_IS_BETTER[metric]:
        return neutral - full
    return full - neutral
def load_metrics():
    parts: List[pd.DataFrame] = []
    audit = []
    for seed in ALL_SEEDS:
        for pid in ALL_PIDS:
            p = metric_checkpoint(seed, pid)
            ok = p.is_file()
            audit.append({'repeat_seed': seed, 'participant': pid, 'metrics_exists': ok, 'metrics_file': str(p)})
            if not ok:
                continue
            q = pd.read_csv(p)
            if 'repeat_seed' not in q.columns:
                q.insert(0, 'repeat_seed', seed)
            if 'participant' not in q.columns:
                q.insert(1, 'participant', pid)
            parts.append(q)
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), pd.DataFrame(audit))
def validate_metrics(metrics: pd.DataFrame, audit: pd.DataFrame):
    missing = audit[~audit['metrics_exists']].copy()
    if len(missing):
        print('\nMISSING / INCOMPLETE CHECKPOINTS')
        print(missing[['repeat_seed', 'participant']].to_string(index=False))
        raise RuntimeError(f'Jobs are not finished: {len(missing)} of 260 seed-participant metric checkpoints are missing.')
    required = {'repeat_seed', 'participant', 'variant', *METRICS}
    missing_cols = required - set(metrics.columns)
    if missing_cols:
        raise RuntimeError(f'Missing metric columns: {sorted(missing_cols)}')
    metrics['repeat_seed'] = pd.to_numeric(metrics['repeat_seed'], errors='raise').astype(int)
    metrics['participant'] = metrics['participant'].astype(str)
    metrics['variant'] = metrics['variant'].astype(str)
    dup = metrics.duplicated(['repeat_seed', 'participant', 'variant'], keep=False)
    if dup.any():
        raise RuntimeError('Duplicate seed-participant-variant rows:\n' + metrics.loc[dup, ['repeat_seed', 'participant', 'variant']].to_string(index=False))
    expected = 10 * 26 * 5
    if len(metrics) != expected:
        raise RuntimeError(f'Expected {expected} metric rows; found {len(metrics)}.')
    if set(metrics['repeat_seed']) != set(ALL_SEEDS):
        raise RuntimeError('Seed coverage is not exactly 1..10.')
    if set(metrics['participant']) != set(ALL_PIDS):
        raise RuntimeError('Participant coverage is not exactly VP001..VP026.')
    if set(metrics['variant']) != set(VARIANTS):
        raise RuntimeError('Variant coverage is not exactly the expected five variants.')
    coverage = metrics.groupby(['repeat_seed', 'participant'])['variant'].nunique()
    if not np.all(coverage.to_numpy() == 5):
        raise RuntimeError('At least one seed-participant is missing one or more variants.')
def load_trials():
    parts: List[pd.DataFrame] = []
    missing = []
    for seed in ALL_SEEDS:
        for pid in ALL_PIDS:
            p = trial_checkpoint(seed, pid)
            if not p.is_file():
                missing.append((seed, pid))
                continue
            q = pd.read_csv(p)
            if 'repeat_seed' not in q.columns:
                q.insert(0, 'repeat_seed', seed)
            if 'participant' not in q.columns:
                q.insert(1, 'participant', pid)
            parts.append(q)
    if missing:
        raise RuntimeError(f'Per-trial merge found {len(missing)} missing files. First missing: {missing[:10]}')
    out = pd.concat(parts, ignore_index=True)
    dup = out.duplicated(['repeat_seed', 'participant', 'variant', 'trial_index'], keep=False)
    if dup.any():
        raise RuntimeError('Duplicate per-trial rows found.')
    counts = out.groupby(['repeat_seed', 'participant', 'variant'])['trial_index'].nunique()
    bad = counts[counts != 36]
    if len(bad):
        raise RuntimeError('Expected exactly 36 test trials per seed-participant-variant.\n' + bad.to_string())
    expected = 10 * 26 * 5 * 36
    if len(out) != expected:
        raise RuntimeError(f'Expected {expected} trial rows; found {len(out)}.')
    return out
def participant_average(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in metrics.columns if c not in {
        'repeat_seed',
        'participant',
        'variant',
        'selected_abc_cfg',
        'selected_lbr_cfg'
    } and pd.api.types.is_numeric_dtype(metrics[c])]
    avg = metrics.groupby(['participant', 'variant'], as_index=False)[numeric_cols].mean()
    nseed = metrics.groupby(['participant', 'variant'], as_index=False)['repeat_seed'].nunique().rename(columns={'repeat_seed': 'n_seeds_averaged'})
    avg = avg.merge(nseed, on=['participant', 'variant'], validate='one_to_one')
    if not np.all(avg['n_seeds_averaged'].to_numpy() == 10):
        raise RuntimeError('Final participant averages did not use exactly 10 seeds everywhere.')
    return avg
def means_table(avg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        q = avg[avg['variant'] == variant]
        row = {
            'variant': variant,
            'component_intervention': 'Intact full model' if variant == 'FULL' else COMPONENT[variant],
            'n_participants': int(q['participant'].nunique())
        }
        for metric in METRICS:
            x = pd.to_numeric(q[metric], errors='coerce').to_numpy(float)
            row[f'{metric}_mean'] = float(np.nanmean(x))
            row[f'{metric}_sd'] = float(np.nanstd(x, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)
def paired_table(avg: pd.DataFrame) -> pd.DataFrame:
    full = avg[avg['variant'] == 'FULL'].set_index('participant').sort_index()
    rows = []
    for neutral_variant in VARIANTS[1:]:
        neutral = avg[avg['variant'] == neutral_variant].set_index('participant').sort_index()
        common = full.index.intersection(neutral.index)
        if len(common) != 26:
            raise RuntimeError(f'{neutral_variant}: expected 26 paired participants, got {len(common)}.')
        for metric in METRICS:
            a = pd.to_numeric(full.loc[common, metric], errors='coerce').to_numpy(float)
            b = pd.to_numeric(neutral.loc[common, metric], errors='coerce').to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = (a[ok], b[ok])
            gain = full_better_gain(a, b, metric)
            sid = VARIANTS.index(neutral_variant) * 100 + METRICS.index(metric)
            mean_gain, lo, hi = bootstrap_ci(gain, 202609260 + sid)
            p = signflip_p(gain, 202609261 + sid)
            rows.append({
                'component': COMPONENT[neutral_variant],
                'neutralized_variant': neutral_variant,
                'metric': metric,
                'metric_label': METRIC_LABEL[metric],
                'n_participants': int(len(gain)),
                'full_mean': float(np.mean(a)),
                'neutralized_mean': float(np.mean(b)),
                'mean_gain_positive_full_better': float(mean_gain),
                'ci95_low': float(lo),
                'ci95_high': float(hi),
                'participants_full_better': int(np.sum(gain > TOL)),
                'participants_equal': int(np.sum(np.abs(gain) <= TOL)),
                'participants_neutralized_better': int(np.sum(gain < -TOL)),
                'raw_signflip_p': float(p)
            })
    out = pd.DataFrame(rows)
    out['holm_p_within_component_6'] = np.nan
    for component in out['component'].unique():
        ix = out.index[out['component'] == component].tolist()
        out.loc[ix, 'holm_p_within_component_6'] = holm_adjust(out.loc[ix, 'raw_signflip_p'].to_numpy(float))
    out['holm_p_global_24'] = holm_adjust(out['raw_signflip_p'].to_numpy(float))
    return out
def seedwise_table(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby(['repeat_seed', 'variant'], as_index=False).agg(
        n_participants=('participant', 'nunique'),
        response_BA=('response_balanced_accuracy', 'mean'),
        onset_MAE=('e2e_onset_mae_sec', 'mean'),
        peak_MAE=('e2e_peak_mae_sec', 'mean'),
        offset_MAE=('e2e_offset_mae_sec', 'mean'),
        duration_MAE=('e2e_duration_mae_sec', 'mean'),
        selection_loss=('e2e_selection_loss', 'mean')
    ).sort_values(['repeat_seed', 'variant']).reset_index(drop=True)
def compact_means(means: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        'variant': means['variant'],
        'BA': means['response_balanced_accuracy_mean'],
        'onset_MAE': means['e2e_onset_mae_sec_mean'],
        'peak_MAE': means['e2e_peak_mae_sec_mean'],
        'offset_MAE': means['e2e_offset_mae_sec_mean'],
        'duration_MAE': means['e2e_duration_mae_sec_mean'],
        'selection_loss': means['e2e_selection_loss_mean']
    })
def compact_stats(stats: pd.DataFrame) -> pd.DataFrame:
    return stats[[
        'component',
        'metric_label',
        'full_mean',
        'neutralized_mean',
        'mean_gain_positive_full_better',
        'ci95_low',
        'ci95_high',
        'participants_full_better',
        'participants_equal',
        'participants_neutralized_better',
        'raw_signflip_p',
        'holm_p_within_component_6',
        'holm_p_global_24'
    ]]
def main():
    mount_drive()
    FINAL_OUT.mkdir(parents=True, exist_ok=True)
    print('=' * 180)
    print('HEMOBOUNDARY — FUNCTIONAL NECESSITY 10-SEED MERGE + PRINT')
    print('=' * 180)
    print('Model fitting                  : NO')
    print('Nested selection               : NO')
    print('Manual path input              : NO')
    print('Expected seeds                 : 10')
    print('Expected participants/seed     : 26')
    print('Expected variants              : 5')
    print('Final statistical unit         : participant after averaging 10 seeds')
    print('Bootstrap reps                 :', BOOT_REPS)
    print('Sign-flip reps                 :', SIGNFLIP_REPS)
    print('Output                         :', FINAL_OUT)
    print()
    metrics, audit = load_metrics()
    save_csv(audit, FINAL_OUT / 'COMPLETION_AUDIT.csv')
    validate_metrics(metrics, audit)
    print('COMPLETION CHECK               : PASS')
    print('Participant metric rows        :', len(metrics), '/ 1300')
    metrics = metrics.sort_values(['repeat_seed', 'participant', 'variant']).reset_index(drop=True)
    save_csv(metrics, FINAL_OUT / 'MERGED_PARTICIPANT_METRICS_ALL_10_SEEDS.csv')
    print('\nMerging per-trial predictions...')
    trials = load_trials()
    trials = trials.sort_values(['repeat_seed', 'participant', 'variant', 'trial_index']).reset_index(drop=True)
    save_csv(trials, FINAL_OUT / 'MERGED_PER_TRIAL_PREDICTIONS_ALL_10_SEEDS.csv.gz')
    print('Per-trial rows                 :', len(trials), '/ 46800')
    avg = participant_average(metrics)
    avg = avg.sort_values(['participant', 'variant']).reset_index(drop=True)
    save_csv(avg, FINAL_OUT / 'FINAL_PARTICIPANT_AVERAGED_METRICS.csv')
    means = means_table(avg)
    stats = paired_table(avg)
    seedwise = seedwise_table(metrics)
    save_csv(means, FINAL_OUT / 'TABLE1_FUNCTIONAL_NECESSITY_MEANS.csv')
    save_csv(stats, FINAL_OUT / 'TABLE2_FUNCTIONAL_NECESSITY_PAIRED_STATS.csv')
    save_csv(seedwise, FINAL_OUT / 'TABLE3_SEEDWISE_MEANS.csv')
    pmeans = compact_means(means)
    pstats = compact_stats(stats)
    print('\n' + '=' * 180)
    print('TABLE 1 — FINAL MEANS AFTER 10-SEED AVERAGING WITHIN PARTICIPANT')
    print('=' * 180)
    print(pmeans.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print('\nGain convention: POSITIVE = intact FULL HemoBoundary is better.')
    print('BA gain = FULL - neutralized.')
    print('Timing/loss gain = neutralized - FULL.')
    print('\n' + '=' * 180)
    print('TABLE 2 — PARTICIPANT-LEVEL PAIRED FUNCTIONAL NECESSITY TESTS')
    print('=' * 180)
    print(pstats.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print('\n' + '=' * 180)
    print('TABLE 3 — SEEDWISE ROBUSTNESS MEANS')
    print('=' * 180)
    print(seedwise.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    focus_metric = {
        'A — Temporal Boundary Estimation': 'e2e_onset_mae_sec',
        'B — Recovery Estimation': 'e2e_offset_mae_sec',
        'C — Response Detection': 'response_balanced_accuracy',
        'LBR — Local Boundary Refinement': 'e2e_selection_loss'
    }
    focus_parts = []
    for component, metric in focus_metric.items():
        q = stats[(stats['component'] == component) & (stats['metric'] == metric)]
        if len(q) == 1:
            focus_parts.append(q)
    focus = pd.concat(focus_parts, ignore_index=True) if focus_parts else pd.DataFrame()
    print('\n' + '=' * 180)
    print('COMPONENT-ROLE FOCUS TABLE')
    print('=' * 180)
    if len(focus):
        print(focus[[
            'component',
            'metric_label',
            'mean_gain_positive_full_better',
            'ci95_low',
            'ci95_high',
            'raw_signflip_p',
            'holm_p_within_component_6',
            'holm_p_global_24'
        ]].to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    summary = [
        'HEMOBOUNDARY — FUNCTIONAL NECESSITY 10-SEED FINAL MERGE',
        '=' * 100,
        '',
        'COMPLETION',
        'Seeds: 10/10',
        'Participants per seed: 26/26',
        'Variants: 5/5',
        f'Participant metric rows: {len(metrics)}/1300',
        f'Per-trial rows: {len(trials)}/46800',
        '',
        'PRIMARY MEANS',
        pmeans.to_string(index=False, float_format=lambda x: f'{x:.6f}'),
        '',
        'PAIRED PARTICIPANT-LEVEL TESTS',
        pstats.to_string(index=False, float_format=lambda x: f'{x:.6f}'),
        '',
        'GAIN CONVENTION',
        'Positive gain always means intact FULL HemoBoundary performed better.',
        'BA = FULL - neutralized.',
        'Timing MAE / selection loss = neutralized - FULL.',
        '',
        'INFERENCE',
        f'Bootstrap repetitions: {BOOT_REPS}',
        f'Monte-Carlo sign-flip repetitions: {SIGNFLIP_REPS}',
        'Statistical unit: participant after averaging all 10 seeds within participant.',
        'Holm: within each component across 6 primary metrics and globally across 24 tests.'
    ]
    (FINAL_OUT / 'FINAL_SUMMARY.txt').write_text('\n'.join(summary), encoding='utf-8')
    print('\n' + '=' * 180)
    print('FINAL MERGE: PASS')
    print('=' * 180)
    print('Saved:', FINAL_OUT)
if __name__ == '__main__':
    main()
