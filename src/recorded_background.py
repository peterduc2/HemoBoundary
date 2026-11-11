"""Frozen recorded-background evaluation and ten-source robustness study."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import reproduction_utils as ru

METRICS = ('sensitivity', 'specificity', 'balanced_accuracy',
           'onset_loss', 'peak_loss', 'offset_loss', 'duration_loss')


def archived_reference(base):
    if base is None:
        return None
    result = ru.read_optional_csv(base, 'REST_RECORDED_BACKGROUND_TRIALS.csv.gz')
    result = result.loc[result.method.eq('HEMOBOUNDARY_FULL')].copy()
    result.sort_values(['participant', 'trial_index'], inplace=True)
    result.reset_index(drop=True, inplace=True)
    if len(result) != 224 or result.participant.nunique() != 14:
        raise RuntimeError('Original reference must contain 224 HemoBoundary conditions')
    return result


def verify_parity(result, reference, fields, label):
    if reference is None:
        return False
    if not result[['participant', 'trial_index']].equals(reference[['participant', 'trial_index']]):
        raise RuntimeError(f'Archived participant/trial order differs for {label}')
    for col in fields:
        if not np.allclose(result[col].to_numpy(float), reference[col].to_numpy(float),
                           atol=1e-7, rtol=1e-7, equal_nan=True):
            difference = np.abs(result[col].to_numpy(float) - reference[col].to_numpy(float))
            raise RuntimeError(f'Archived replay differs for {label}/{col}: max={np.nanmax(difference)}')
    return True


def inspect_data(rest_root, output=None):
    records = []
    for pid, item, path in ru.locate_rest_files(rest_root):
        h, r, fs = ru.read_rest(path)
        if h.shape[0] != item['channels'] or not np.isclose(fs, 2.0):
            raise RuntimeError(f'Recorded preprocessing differs for {pid}')
        records.append({'participant': pid, 'channels': int(h.shape[0]),
                        'sample_rate_hz': fs, 'source_sha256': item['sha256'],
                        'source_subject_id': item['subject_id']})
    if output:
        ru.write_json(Path(output) / 'recorded_input_audit.json',
                      {'participants': records, 'n_participants': len(records),
                       'conditions_expected': 224})
    print(f'PASS: {len(records)} checksum-matched resting recordings, unaugmented channel checks')
    return records


def evaluate_seed(m, seed, banks, original=None):
    m.configure_seed(seed)
    model, cfg, local, audit = ru.fit_frozen_source(m, seed, held='VP001')
    if seed == 1:
        expected_cfg = {'change_frac': 0.3, 'response_threshold': 0.4,
                        'state_C': 0.03, 'timing_alpha': 10.0}
        expected_local = {'local_alpha': 100.0, 'radius': 1.0, 'strength': 0.75}
        if cfg != expected_cfg or local != expected_local:
            raise RuntimeError('Seed 1 source configuration is inconsistent with the verified model')
    frames = []
    for pid, item, _, _, truth, meta in banks:
        raw, probability, threshold = ru.predict_model(m, model, truth,
                                                        cfg['response_threshold'])
        rows = ru.prediction_frame(seed, pid, truth, meta, raw, probability, threshold, item)
        frames.append(rows)
    combined = pd.concat(frames, ignore_index=True).sort_values(['participant', 'trial_index'])
    combined.reset_index(drop=True, inplace=True)
    if len(combined) != 224 or combined.participant.nunique() != 14:
        raise RuntimeError('Incomplete recorded-background evaluation')
    if seed == 1 and original is not None:
        verify_parity(combined, original,
                      ('truth_present', 'truth_onset', 'truth_peak', 'truth_offset',
                       'truth_censored', 'raw_onset', 'raw_peak', 'raw_offset',
                       'response_probability', 'native_response'), 'original Seed 1')
    provenance = {'seed': seed, 'source_sha256': ru.MODEL_SHA256,
                  'context': f'{ru.REST_CONTEXT}/seed{seed:02d}/heldVP001',
                  'source_participants': 25, 'held_source_participant': 'VP001',
                  'external_participants': 14, 'external_labels_used_in_fit': False,
                  'original_seed1_replay_verified': bool(seed == 1 and original is not None),
                  'config': cfg, 'local': local, 'fit_audit': audit.rows}
    del model
    gc.collect()
    return combined, provenance


def run(shin_root, rest_root, output, seeds=range(1, 11), original_archive=None):
    out = Path(output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    reference = archived_reference(original_archive)
    m = ru.load_model(shin_root)
    banks = ru.build_rest_banks(m, rest_root, reference)
    print('PASS: 14 recorded source participants, 224 known injection/control conditions')
    frames = []
    seeds = tuple(int(x) for x in seeds)
    for seed in seeds:
        if seed not in range(1, 11):
            raise ValueError('Expected source seeds 1 through 10')
        result_path = out / 'ten_source' / f'SEED_{seed:02d}_REST_PREDICTIONS.csv.gz'
        audit_path = out / 'ten_source' / f'SEED_{seed:02d}_FIT_AUDIT.json'
        if result_path.is_file() and audit_path.is_file():
            old = json.loads(audit_path.read_text(encoding='utf-8'))
            if (old.get('source_sha256') != ru.MODEL_SHA256 or
                    old.get('seed', old.get('source_seed')) != seed):
                raise RuntimeError(f'Unsafe resume checkpoint for seed {seed}')
            if old.get('context', old.get('fit_context')) != f'{ru.REST_CONTEXT}/seed{seed:02d}/heldVP001':
                raise RuntimeError(f'Wrong source model fit context in seed {seed}')
            if reference is not None and seed == 1 and not old.get(
                    'original_seed1_replay_verified', old.get('seed1_original_replay_verified')):
                raise RuntimeError('Existing Seed 1 was not checked against the supplied archive')
            frame = pd.read_csv(result_path)
            if len(frame) != 224 or frame.participant.nunique() != 14:
                raise RuntimeError(f'Incomplete seed {seed} checkpoint')
            print(f'RESUMED seed {seed}/10')
        else:
            print(f'FITTING source seed {seed}/10', flush=True)
            frame, audit = evaluate_seed(m, seed, banks, reference)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(result_path, index=False, compression='gzip')
            ru.write_json(audit_path, audit)
        frames.append(frame)
    full = pd.concat(frames, ignore_index=True)
    summary_rows = []
    for (seed, pid), group in full.groupby(['seed', 'participant'], sort=True):
        summary_rows.append({'seed': seed, 'participant': pid, **ru.participant_metric(group)})
    part = pd.DataFrame(summary_rows)
    part.to_csv(out / 'ten_source' / 'PARTICIPANT_BY_SEED.csv', index=False)
    by_seed = part.groupby('seed', as_index=False)[list(METRICS)].mean()
    by_seed.to_csv(out / 'ten_source' / 'SEED_SUMMARY.csv', index=False)
    participant = part.groupby('participant', as_index=False)[list(METRICS)].mean()
    participant.to_csv(out / 'ten_source' / 'EXTERNAL_PARTICIPANT_SEED_AVERAGED.csv', index=False)
    ci_rows = []
    for key in METRICS:
        ci = ru.bootstrap_mean(participant[key].to_numpy())
        ci_rows.append({'metric': key, 'mean': float(participant[key].mean()),
                        'CI_low': ci[0], 'CI_high': ci[1],
                        'independent_participants': 14, 'repeated_source_models': len(seeds)})
    pd.DataFrame(ci_rows).to_csv(out / 'ten_source' / 'EXTERNAL_PARTICIPANT_BOOTSTRAP_CI.csv',
                                index=False)
    ru.write_json(out / 'recorded_background_protocol.json',
                  {'source_sha256': ru.MODEL_SHA256, 'seeds': list(seeds),
                   'held_shin_participant': 'VP001', 'external_participants': 14,
                   'expected_conditions': 224, 'original_reference_supplied': reference is not None,
                   'independent_validation': 'Unverified unless run from public data and checked',
                   'no_external_label_selection': True})
    print(by_seed.round(4).to_string(index=False))
    return part


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('inspect', 'run', 'selftest'))
    p.add_argument('--shin-root', type=Path)
    p.add_argument('--rest-root', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
    p.add_argument('--original-archive', type=Path,
                   help='Optional local original prediction archive for strict Seed 1 replay')
    p.add_argument('--download', action='store_true', help='Download provider Dataset I when missing')
    args = p.parse_args()
    if args.action == 'selftest':
        ru.self_test()
        print('PASS: recorded-background utility self-tests')
        return
    if args.rest_root is None:
        p.error('--rest-root is required')
    if args.download and not list(args.rest_root.rglob('resting_hrf_100.snirf')):
        ru.download_rest(args.rest_root)
    if args.action == 'inspect':
        inspect_data(args.rest_root, args.output)
        return
    if args.shin_root is None or args.output is None:
        p.error('run requires --shin-root and --output')
    run(args.shin_root, args.rest_root, args.output, args.seeds, args.original_archive)


if __name__ == '__main__':
    main()
