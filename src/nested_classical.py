"""Source-nested Ridge and ExtraTrees comparisons on Shin known-boundary banks."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src import reproduction_utils as ru

RIDGE_ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0)
ET_LEAVES = (1, 4, 8)
THRESHOLDS = (0.20, 0.40, 0.50, 0.60)
REPRESENTATIONS = ('RAW', 'STATE', 'MULTISCALE', 'COMBINED')
READOUTS = ('RIDGE', 'ET')
REP_CONFIG = {'state_C': 0.01, 'change_frac': 0.30,
              'timing_alpha': 10.0, 'response_threshold': 0.50}


class ClassicalHead:
    def __init__(self, kind, parameter, seed):
        self.kind, self.parameter, self.seed = kind, parameter, seed

    def fit(self, features, truth):
        X = np.asarray(features, float)
        if not np.isfinite(X).all():
            raise ValueError('Nonfinite source features')
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        present = np.asarray(truth['present'], int)
        observed = (present == 1) & (np.asarray(truth['censored'], int) == 0)
        if np.unique(present).size != 2 or observed.sum() < 15:
            raise ValueError('Insufficient source training examples')
        self.det = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1200,
                                      random_state=self.seed).fit(Z, present)
        self.reg = {}
        for j, name in enumerate(('onset', 'peak', 'offset')):
            allowed = observed if name == 'offset' else (present == 1)
            estimator = (Ridge(alpha=float(self.parameter)) if self.kind == 'RIDGE'
                         else ExtraTreesRegressor(n_estimators=128,
                                                  min_samples_leaf=int(self.parameter),
                                                  n_jobs=-1, random_state=self.seed+j))
            estimator.fit(Z[allowed], np.asarray(truth[name], float)[allowed])
            self.reg[name] = estimator
        return self

    def predict(self, features, m):
        Z = self.scaler.transform(np.asarray(features, float))
        probabilities = self.det.predict_proba(Z)[:, 1]
        raw = np.column_stack([self.reg[k].predict(Z) for k in ('onset', 'peak', 'offset')])
        raw = np.column_stack(m.enforce_geometry(raw[:, 0], raw[:, 1], raw[:, 2]))
        return raw, probabilities


def representation(m, state, truth):
    x = m.predictor_view(truth)
    raw = np.asarray(truth['state_feat'], float).reshape(len(truth['present']), -1)
    states = m.coherent_state_features(m.n4,
                                      m.coherent_state_probabilities(m.n4, state, x))
    multi = m.change_branch_features(x, float(REP_CONFIG['change_frac']))
    return {'RAW': raw, 'STATE': states, 'MULTISCALE': multi,
            'COMBINED': np.column_stack([states, multi])}


def test_rows_from_source(m, caches, pids):
    copies = {p: dict(caches[p], train=caches[p]['test']) for p in pids}
    return m.training_rows(copies, pids)


def candidate_loss(m, truth, raw, probability, threshold):
    pred = m.make_prediction_threshold(m.n4, raw, probability, float(threshold))
    pids = np.asarray(truth['pid']).astype(str)
    scores = []
    for pid in sorted(set(pids)):
        mask = pids == pid
        subset = m.subset(truth, mask)
        metrics = m.participant_metrics(subset, pred.loc[mask].reset_index(drop=True))
        scores.append(float(metrics['selection_loss']))
    return float(np.mean(scores))


def original_full(m, seed, held, caches, original_archive=None):
    """Regenerate the frozen source reference if a prior checkpoint is unavailable."""
    m.configure_seed(seed)
    original = None
    if original_archive is not None:
        original = ru.read_optional_csv(original_archive,
                                        f'checkpoints/seed_{seed:02d}_{held}.csv.gz')
        original = original.loc[original.family.eq('PAPER_MATCHED_CACHE') &
                                original.method.eq('HEMOBOUNDARY_FULL')].copy()
        original.sort_values('trial_index', inplace=True)
        original.reset_index(drop=True, inplace=True)
        if len(original) != 36:
            raise RuntimeError('Original matched checkpoint does not contain 36 trials')
    source = [p for p in ru.PIDS if p != held]
    cfg, local, _ = m.load_full_config(seed, held)
    train = m.training_rows(caches, source)
    context = f'{ru.SOURCE_CONTEXT}/seed={seed:02d}/outer={held}'
    audit = m.FitAudit(source, [held], context)
    model = m.fit_coherent_full(m.n4, train, cfg, local, audit,
                                outer_forbidden={held}, context=context)
    truth = caches[held]['test']
    raw, prob, threshold = ru.predict_model(m, model, truth, cfg['response_threshold'])
    meta = [{'trial_index': i} for i in range(len(truth['present']))]
    replay = ru.prediction_frame(seed, held, truth, meta, raw, prob, threshold)
    replay['method'] = 'HEMOBOUNDARY_FULL'
    replay['family'] = 'PAPER_MATCHED_CACHE'
    if len(replay) != 36:
        raise RuntimeError('Expected exactly 36 matched trials')
    if original is not None:
        fields = ('truth_present', 'truth_onset', 'truth_peak', 'truth_offset',
                  'truth_censored', 'raw_onset', 'raw_peak', 'raw_offset',
                  'response_probability', 'native_response')
        for key in fields:
            if not np.allclose(replay[key], original[key], atol=1e-7,
                               rtol=1e-7, equal_nan=True):
                raise RuntimeError(f'Original HemoBoundary replay differs: {seed}/{held}/{key}')
        return original
    return replay


def one_outer(m, seed, outer, caches, original_archive=None):
    source = [p for p in ru.PIDS if p != outer]
    groups = m.grouped_folds(source,
                             m.stable_seed('HB_ARRAY_NESTED_CLASSICAL_20261102', seed, outer), n=3)
    folds, audit_rows = [], []
    for number, held in enumerate(groups, 1):
        held = list(map(str, held))
        train_pids = [p for p in source if p not in held]
        train = m.training_rows(caches, train_pids)
        val = test_rows_from_source(m, caches, held)
        fit_audit = m.FitAudit(train_pids, held+[outer],
                               f'NESTED_CLASSICAL/seed{seed}/outer{outer}/inner{number}')
        state = m.fit_coherent_state(m.n4, train, float(REP_CONFIG['state_C']),
                                     fit_audit, stage_prefix=f'NESTED_CLASSICAL_INNER_{number}')
        if any(outer in r.get('fit_participants', '').split('|') for r in fit_audit.rows):
            raise RuntimeError('Outer participant used during inner fitting')
        audit_rows.extend(fit_audit.rows)
        folds.append((train, val, representation(m, state, train),
                      representation(m, state, val)))
        del state
    candidate_rows, chosen = [], {}
    for rep in REPRESENTATIONS:
        for kind in READOUTS:
            parameters = RIDGE_ALPHAS if kind == 'RIDGE' else ET_LEAVES
            scores = []
            for parameter in parameters:
                per_threshold = {t: [] for t in THRESHOLDS}
                for fold_number, (train, val, xtrain, xval) in enumerate(folds, 1):
                    model = ClassicalHead(kind, parameter, seed+fold_number).fit(xtrain[rep], train)
                    raw, probability = model.predict(xval[rep], m)
                    for t in THRESHOLDS:
                        per_threshold[t].append(candidate_loss(m, val, raw, probability, t))
                for t in THRESHOLDS:
                    score = float(np.mean(per_threshold[t]))
                    scores.append((score, float(parameter), float(t)))
                    candidate_rows.append({'seed': seed, 'outer': outer, 'representation': rep,
                                           'readout': kind, 'parameter': parameter, 'threshold': t,
                                           'mean_inner_selection_loss': score, 'inner_folds': 3})
            best = min(scores)
            chosen[(rep, kind)] = {'score': best[0], 'param': best[1], 'threshold': best[2]}
    del folds
    gc.collect()
    train = m.training_rows(caches, source)
    fit_audit = m.FitAudit(source, [outer],
                           f'NESTED_CLASSICAL_FINAL/seed{seed}/outer{outer}')
    state = m.fit_coherent_state(m.n4, train, float(REP_CONFIG['state_C']),
                                 fit_audit, stage_prefix='NESTED_CLASSICAL_FINAL_ENCODER')
    test = caches[outer]['test']
    xtrain, xtest = representation(m, state, train), representation(m, state, test)
    original = original_full(m, seed, outer, caches, original_archive)
    if not np.allclose(original.truth_onset, test['onset'], equal_nan=True, atol=1e-7):
        raise RuntimeError('Reference test trial order differs from outer test bank')
    frames = [original]
    for (rep, kind), opt in chosen.items():
        model = ClassicalHead(kind, opt['param'], seed).fit(xtrain[rep], train)
        raw, probability = model.predict(xtest[rep], m)
        result = original.copy()
        result['method'] = f'NESTED_{rep}_{kind}'
        result['response_probability'] = probability
        result['threshold'] = opt['threshold']
        result['native_response'] = (probability >= opt['threshold']).astype(int)
        result['raw_onset'], result['raw_peak'], result['raw_offset'] = raw.T
        frames.append(result)
    result = pd.concat(frames, ignore_index=True)
    if len(result) != 324 or result.method.nunique() != 9:
        raise RuntimeError('Nested comparison must contain nine models times 36 trials')
    return result, pd.DataFrame(candidate_rows), chosen, audit_rows+fit_audit.rows


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate(p[order] * (len(p)-np.arange(len(p))))
    result = np.empty(len(p))
    result[order] = np.minimum(adjusted, 1.0)
    return result


def paired_inference(participant, gate, output, seeds):
    if set(seeds) != set(range(1, 11)):
        print('Incomplete 10-source seed set: no Holm inference generated')
        return
    rng = np.random.default_rng(19043)
    sub = participant.loc[participant.gate.eq(gate)]
    baseline = sub.loc[sub.method.eq('HEMOBOUNDARY_FULL')].set_index('participant')
    rows = []
    for method in sorted(set(sub.method) - {'HEMOBOUNDARY_FULL'}):
        comp = sub.loc[sub.method.eq(method)].set_index('participant')
        if len(comp) != 26 or len(baseline) != 26 or set(comp.index) != set(baseline.index):
            raise RuntimeError('Nested inference requires the same 26 outer participants')
        outcomes = ('onset_loss', 'peak_loss', 'offset_loss', 'duration_loss')
        if gate == 'native':
            outcomes += ('balanced_accuracy',)
        for key in outcomes:
            x = comp.loc[baseline.index, key].to_numpy(float)
            y = baseline[key].to_numpy(float)
            difference = (y-x) if key == 'balanced_accuracy' else (x-y)
            if not np.isfinite(difference).all():
                raise RuntimeError('Nonfinite participant differences')
            observed = abs(float(difference.mean()))
            exceed = 0
            for _ in range(10):
                signs = rng.integers(0, 2, size=(10000, 26), dtype=np.int8) * 2 - 1
                exceed += int((np.abs((signs*difference).mean(axis=1)) >= observed-1e-14).sum())
            rawp = (exceed + 1) / 100001
            index = rng.integers(0, 26, size=(20000, 26))
            lower, upper = np.percentile(difference[index].mean(axis=1), (2.5, 97.5))
            rows.append({'gate': gate, 'method': method, 'outcome': key,
                         'mean_gain_pro_HB': float(difference.mean()),
                         'CI_low': float(lower), 'CI_high': float(upper),
                         'raw_signflip_p': rawp, 'participants': 26, 'seeds_averaged': 10})
    result = pd.DataFrame(rows)
    for fields in (('onset_loss', 'peak_loss', 'offset_loss', 'duration_loss'),
                   ('balanced_accuracy',)):
        mask = result.outcome.isin(fields)
        if mask.any():
            result.loc[mask, 'holm_p'] = holm(result.loc[mask, 'raw_signflip_p'].to_numpy())
            result.loc[mask, 'correction_family_size'] = int(mask.sum())
    result.to_csv(output / f'NESTED_{gate.upper()}_PAIRED_HOLM.csv', index=False)
    return result


def run(shin_root, output, seeds=range(1, 11), original_archive=None):
    out = Path(output).expanduser().resolve() / 'nested'
    out.mkdir(parents=True, exist_ok=True)
    m = ru.load_model(shin_root)
    seeds = tuple(int(s) for s in seeds)
    all_metrics = []
    for seed in seeds:
        m.configure_seed(seed)
        caches, _ = m.load_seed_caches(seed)
        for outer in ru.PIDS:
            result_path = out / f'SEED_{seed:02d}_{outer}_PREDICTIONS.csv.gz'
            audit_path = out / f'SEED_{seed:02d}_{outer}_AUDIT.json'
            if result_path.is_file() and audit_path.is_file():
                audit = json.loads(audit_path.read_text(encoding='utf-8'))
                if audit.get('source_sha256') != ru.MODEL_SHA256 or audit.get('seed') != seed:
                    raise RuntimeError(f'Unsafe nested checkpoint for {seed}/{outer}')
                frame = pd.read_csv(result_path)
                if len(frame) != 324 or frame.method.nunique() != 9:
                    raise RuntimeError('Incomplete nested comparison checkpoint')
                print(f'RESUMED nested {seed}/{outer}')
            else:
                print(f'FITTING nested seed {seed}/10, outer {outer}', flush=True)
                frame, grid, selected, fit_audit = one_outer(m, seed, outer, caches, original_archive)
                frame.to_csv(result_path, index=False, compression='gzip')
                grid.to_csv(out / f'SEED_{seed:02d}_{outer}_INNER_CANDIDATES.csv.gz',
                            index=False, compression='gzip')
                ru.write_json(audit_path,
                              {'source_sha256': ru.MODEL_SHA256, 'seed': seed, 'outer': outer,
                               'participant_grouped_inner_folds': 3,
                               'outer_label_selection': False,
                               'original_checkpoint_parity_checked': original_archive is not None,
                               'representation_settings_fixed': REP_CONFIG,
                               'tuning_grid': {'ridge_alpha': RIDGE_ALPHAS,
                                               'et_leaf': ET_LEAVES, 'thresholds': THRESHOLDS},
                               'selected': {str(k): v for k, v in selected.items()},
                               'fit_audit': fit_audit})
            hb_gate = frame.loc[frame.method.eq('HEMOBOUNDARY_FULL')].sort_values('trial_index').native_response.to_numpy(int)
            for method, group in frame.groupby('method', sort=True):
                group = group.sort_values('trial_index').copy()
                for gate in ('native', 'shared_hb', 'ungated_positive'):
                    g = group.copy()
                    if gate == 'shared_hb':
                        g['native_response'] = hb_gate
                    if gate == 'ungated_positive':
                        g['native_response'] = g.truth_present.to_numpy(int)
                    metrics = ru.participant_metric(g)
                    if gate != 'native':
                        metrics['balanced_accuracy'] = np.nan
                    all_metrics.append({'seed': seed, 'participant': outer,
                                        'method': method, 'gate': gate, **metrics})
        del caches
        gc.collect()
    result = pd.DataFrame(all_metrics)
    result.to_csv(out / 'NESTED_PARTICIPANT_BY_SEED.csv', index=False)
    participant = result.groupby(['participant', 'method', 'gate'], as_index=False).mean(numeric_only=True)
    participant.to_csv(out / 'NESTED_SEED_AVERAGED_BY_PARTICIPANT.csv', index=False)
    averages = participant.groupby(['method', 'gate'], as_index=False).mean(numeric_only=True)
    averages.to_csv(out / 'NESTED_METHOD_SUMMARY.csv', index=False)
    for gate in ('native', 'shared_hb', 'ungated_positive'):
        paired_inference(participant, gate, out, seeds)
    print(averages[['gate', 'method', 'balanced_accuracy', 'onset_loss', 'peak_loss',
                    'offset_loss', 'duration_loss']].round(4).to_string(index=False))
    return averages


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shin-root', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(1, 11)))
    p.add_argument('--original-archive', type=Path,
                   help='Optional saved matched checkpoints for strict replay verification')
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if a.self_test:
        assert np.allclose(holm([.01, .03, .04]), [.03, .06, .06])
        ru.self_test()
        print('PASS: source-nested classical helper self-tests')
        return
    if a.shin_root is None or a.output is None:
        p.error('--shin-root and --output are required')
    run(a.shin_root, a.output, a.seeds, a.original_archive)


if __name__ == '__main__':
    main()
