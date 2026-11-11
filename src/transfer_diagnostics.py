"""Exploratory detector-score, threshold and paired background controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import reproduction_utils as ru

THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
              0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80)
METRICS = ('sensitivity', 'specificity', 'balanced_accuracy',
           'onset_loss', 'peak_loss', 'offset_loss', 'duration_loss')


def load_checkpoint(completed, seed, kind):
    completed = Path(completed)
    if kind == 'source':
        filename = completed / 'nested' / f'SEED_{seed:02d}_VP001_PREDICTIONS.csv.gz'
        d = pd.read_csv(filename)
        d = d.loc[d.method.eq('HEMOBOUNDARY_FULL')].copy()
        if len(d) != 36 or d.participant.nunique() != 1:
            raise RuntimeError(f'Incomplete original Shin VP001 checkpoint at seed {seed}')
    else:
        filename = completed / 'ten_source' / f'SEED_{seed:02d}_REST_PREDICTIONS.csv.gz'
        d = pd.read_csv(filename)
        if len(d) != 224 or d.participant.nunique() != 14:
            raise RuntimeError(f'Incomplete recorded-background checkpoint at seed {seed}')
        audit = json.loads((completed / 'ten_source' /
                            f'SEED_{seed:02d}_FIT_AUDIT.json').read_text(encoding='utf-8'))
        context = audit.get('context', audit.get('fit_context'))
        if context != f'{ru.REST_CONTEXT}/seed{seed:02d}/heldVP001':
            raise RuntimeError(f'Seed {seed} recorded model context differs')
        if audit.get('source_sha256') != ru.MODEL_SHA256:
            raise RuntimeError(f'Seed {seed} source hash differs')
    d.sort_values(['participant', 'trial_index'], inplace=True)
    d.reset_index(drop=True, inplace=True)
    if set(d.seed) != {seed}:
        raise RuntimeError('Checkpoint has an unexpected source seed')
    return d


def distribution_rows(df, domain, seed, threshold):
    rows = []
    for label in (0, 1):
        x = df.loc[df.truth_present.eq(label), 'response_probability'].to_numpy(float)
        rows.append({'seed': seed, 'domain': domain, 'response_present': label,
                     'n_trials': len(x), 'median_probability': float(np.median(x)),
                     'p10': float(np.quantile(x, .1)), 'p90': float(np.quantile(x, .9)),
                     'fraction_over_original_threshold': float(np.mean(x >= threshold))})
    return rows


def paired_bootstrap(values, seed=23456, draws=20000):
    x = np.asarray(values, float)
    if len(x) != 14 or not np.isfinite(x).all():
        raise RuntimeError('Paired bootstrap requires fourteen participant contrasts')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 14, size=(draws, 14))
    lower, upper = np.percentile(x[idx].mean(axis=1), (2.5, 97.5))
    return float(x.mean()), float(lower), float(upper)


def source_vs_external(completed, output, seeds=range(1, 11)):
    out = Path(output) / 'diagnostics'
    out.mkdir(parents=True, exist_ok=True)
    probabilities, selected, scored = [], [], []
    seeds = tuple(seeds)
    for seed in seeds:
        source = load_checkpoint(completed, seed, 'source')
        external = load_checkpoint(completed, seed, 'external')
        t0 = float(source.threshold.iloc[0])
        if not np.allclose(source.threshold.to_numpy(float), t0):
            raise RuntimeError('Source threshold varies within a seed')
        if not np.allclose(external.threshold.to_numpy(float), t0):
            raise RuntimeError('Source/external threshold differs')
        probabilities.extend(distribution_rows(source, 'SHIN_HELD_VP001', seed, t0))
        probabilities.extend(distribution_rows(external, 'RECORDED_REST_14', seed, t0))
        truth = source.truth_present.to_numpy(int)
        proba = source.response_probability.to_numpy(float)
        candidates = []
        for threshold in THRESHOLDS:
            decision = proba >= threshold
            sensitivity = float(decision[truth == 1].mean())
            specificity = float((~decision[truth == 0]).mean())
            candidates.append((threshold, (sensitivity + specificity) / 2))
        chosen = sorted(candidates, key=lambda item: (-item[1], abs(item[0]-t0), -item[0]))[0]
        selected.append({'seed': seed, 'original_threshold': t0, 'source_selected_threshold': chosen[0],
                         'source_BA_selected': chosen[1],
                         'selection_dataset': 'Shin VP001 held-out source-domain trials',
                         'status': 'Exploratory, not independent threshold validation'})
        for pid, group in external.groupby('participant', sort=True):
            for label, threshold in (('ORIGINAL_FROZEN', t0), ('SHIN_VP001_SELECTED', chosen[0])):
                scored.append({'seed': seed, 'participant': pid, 'rule': label,
                               'threshold': threshold, **ru.participant_metric(group, threshold)})
        print(f'Checked source and external scores: seed {seed}')
    p = pd.DataFrame(probabilities)
    p.to_csv(out / 'SOURCE_VS_REST_QUANTILES_BY_SEED.csv', index=False)
    pd.DataFrame(selected).to_csv(out / 'SOURCE_ONLY_THRESHOLDS.csv', index=False)
    part = pd.DataFrame(scored).groupby(['rule', 'participant'], as_index=False)[list(METRICS)].mean()
    part.to_csv(out / 'EXTERNAL_SEED_AVERAGED_BY_PARTICIPANT.csv', index=False)
    summary = part.groupby('rule', as_index=False)[list(METRICS)].mean()
    summary.to_csv(out / 'THRESHOLD_COMPARISON.csv', index=False)
    print('\nResponse probability medians (mean over source seeds)')
    print(p.groupby(['domain', 'response_present']).median_probability.mean().round(5).to_string())
    print('\nExploratory threshold comparison')
    print(summary.round(4).to_string(index=False))
    return summary


def lag_corr(x):
    a = x[:, :-1] - x[:, :-1].mean(axis=1, keepdims=True)
    b = x[:, 1:] - x[:, 1:].mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(a*a, axis=1) * np.sum(b*b, axis=1))
    phi = np.divide(np.sum(a*b, axis=1), denominator,
                    out=np.zeros(len(a)), where=denominator > 1e-20)
    return np.clip(np.nan_to_num(phi, nan=0., posinf=0., neginf=0.), 0., .95)


def make_ar_surrogate(h, r, time, rng):
    """Scale-matched AR(1) control; it does not preserve the full physiology spectrum."""
    base = time < 0
    ph, pr = lag_corr(h), lag_corr(r)
    n, steps = h.shape

    def one(phi):
        shared = rng.standard_normal(steps)
        innovations = .75 * rng.standard_normal((n, steps)) + .25 * shared[None, :]
        signal = np.zeros((n, steps))
        signal[:, 0] = innovations[:, 0]
        for i in range(1, steps):
            signal[:, i] = phi * signal[:, i-1] + np.sqrt(1. - phi*phi) * innovations[:, i]
        return signal

    hh, rr = one(ph), one(pr)
    f = float(rng.uniform(.06, .12))
    phase = float(rng.uniform(0., 2. * np.pi))
    wave = np.sin(2. * np.pi * f * (time - time[0]) + phase)
    amp = float(rng.uniform(0., .8)) * float(np.median(np.std(h[:, base], axis=1)))
    hh += amp * wave[None, :]
    rr += amp * float(rng.uniform(.5, 1.)) * wave[None, :]
    for synthetic, recorded in ((hh, h), (rr, r)):
        rec_mean, rec_scale = recorded[:, base].mean(axis=1), recorded[:, base].std(axis=1)
        syn_mean, syn_scale = synthetic[:, base].mean(axis=1), synthetic[:, base].std(axis=1)
        synthetic -= syn_mean[:, None]
        synthetic *= (rec_scale / np.maximum(syn_scale, 1e-12))[:, None]
        synthetic += rec_mean[:, None]
    return hh, rr


def surrogate_control(shin_root, rest_root, completed, output):
    out = Path(output) / 'background_control'
    out.mkdir(parents=True, exist_ok=True)
    m = ru.load_model(shin_root)
    m.configure_seed(1)
    model, cfg, local, audit = ru.fit_frozen_source(m, 1)
    expected_cfg = {'change_frac': 0.3, 'response_threshold': 0.4,
                    'state_C': 0.03, 'timing_alpha': 10.0}
    if cfg != expected_cfg:
        raise RuntimeError('The Seed 1 original source configuration changed')
    original = load_checkpoint(completed, 1, 'external')
    participants = []
    for k, (pid, item, f) in enumerate(ru.locate_rest_files(rest_root)):
        h, r, fs = ru.read_rest(f)
        windows = ru.get_windows(h, r, fs, m)
        truth_recorded, meta = ru.make_trials(m, windows, k)
        source_rows = original.loc[original.participant.eq(pid)].sort_values('trial_index')
        if len(source_rows) != 16:
            raise RuntimeError('External reference lacks participant trial rows')
        surrogate_trials = []
        for window_index, (hh, rr) in enumerate(windows):
            rng = np.random.default_rng(73000 + 100*k + window_index)
            shape, on, peak, off, _ = m.original_half_cosine_shape(rng)
            surh, surr = make_ar_surrogate(hh, rr, np.asarray(m.FULL_T),
                                          np.random.default_rng(20261103 + 100*k + window_index))
            surrogate_trials.append(m.trial_to_predictors(surh.copy(), surr.copy(), False,
                                                          np.nan, np.nan, np.nan))
            scale = np.maximum(np.std(hh[:, m.FULL_T < 0], axis=1), 1e-12)
            for amp in (0.5, 1.0, 2.0):
                support = rng.uniform(.6, 1., size=len(scale))
                inject = (amp * scale * support)[:, None] * shape[None, :]
                hp, rp = surh.copy(), surr.copy()
                hp[:, m.POST_MASK] += inject
                rp[:, m.POST_MASK] -= 0.5 * inject
                surrogate_trials.append(m.trial_to_predictors(hp, rp, True, on, peak, off))
        keys = ('present', 'onset', 'peak', 'offset', 'censored', 'obs', 'state_feat')
        truth_surrogate = {key: np.asarray([t[key] for t in surrogate_trials]) for key in keys}
        raw_rec, p_rec, threshold = ru.predict_model(m, model, truth_recorded,
                                                     cfg['response_threshold'])
        rows_recorded = ru.prediction_frame(1, pid, truth_recorded, meta,
                                            raw_rec, p_rec, threshold)
        raw_sur, p_sur, threshold = ru.predict_model(m, model, truth_surrogate,
                                                     cfg['response_threshold'])
        rows_surrogate = ru.prediction_frame(1, pid, truth_surrogate, meta,
                                             raw_sur, p_sur, threshold)
        for key in ('truth_present', 'truth_onset', 'truth_peak', 'truth_offset',
                    'truth_censored', 'raw_onset', 'raw_peak', 'raw_offset',
                    'response_probability', 'native_response'):
            if not np.allclose(rows_recorded[key], source_rows[key], atol=1e-7,
                               rtol=1e-7, equal_nan=True):
                raise RuntimeError(f'Original Seed 1 replay differs for {pid}/{key}')
        for key in ('truth_present', 'truth_onset', 'truth_peak', 'truth_offset', 'truth_censored'):
            if not np.allclose(rows_surrogate[key], source_rows[key], atol=1e-7,
                               rtol=1e-7, equal_nan=True):
                raise RuntimeError(f'Surrogate injection truth differs for {pid}/{key}')
        participants.append({'participant': pid, 'arm': 'RECORDED',
                             **ru.participant_metric(rows_recorded)})
        participants.append({'participant': pid, 'arm': 'AR_SURROGATE_MATCHED_SCALE',
                             **ru.participant_metric(rows_surrogate)})
        print('Verified recorded Seed 1 replay and paired AR surrogate:', pid)
    data = pd.DataFrame(participants)
    data.to_csv(out / 'PARTICIPANT_PAIRED_METRICS.csv', index=False)
    summary = data.groupby('arm', as_index=False)[list(METRICS)].mean()
    summary.to_csv(out / 'CONTROL_SUMMARY.csv', index=False)
    contrasts = []
    pivot = data.pivot(index='participant', columns='arm', values=list(METRICS))
    for field in METRICS:
        difference = (pivot[(field, 'AR_SURROGATE_MATCHED_SCALE')] -
                      pivot[(field, 'RECORDED')]).to_numpy(float)
        mean, low, high = paired_bootstrap(difference)
        contrasts.append({'metric': field, 'surrogate_minus_recorded': mean,
                          'CI_low': low, 'CI_high': high,
                          'independent_participants': 14,
                          'interpretation': 'Exploratory paired difference, not a causal estimate'})
    pd.DataFrame(contrasts).to_csv(out / 'PAIRED_SURROGATE_MINUS_RECORDED.csv', index=False)
    print(summary.round(4).to_string(index=False))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('probabilities', 'background', 'all', 'selftest'))
    parser.add_argument('--completed', type=Path)
    parser.add_argument('--shin-root', type=Path)
    parser.add_argument('--rest-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'selftest':
        ru.self_test()
        assert np.allclose(lag_corr(np.tile(np.arange(12.), (4, 1))), .95)
        print('PASS: transfer diagnostics helper self-tests')
        return
    if args.completed is None or args.output is None:
        parser.error('--completed and --output are required')
    if args.action in ('probabilities', 'all'):
        source_vs_external(args.completed, args.output)
    if args.action in ('background', 'all'):
        if args.rest_root is None or args.shin_root is None:
            parser.error('Background control requires --shin-root and --rest-root')
        surrogate_control(args.shin_root, args.rest_root, args.completed, args.output)


if __name__ == '__main__':
    main()
