"""Shared, source-verified helpers for the recorded-background evaluations."""
from __future__ import annotations

import gzip
import hashlib
import importlib
import io
import json
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_SHA256 = '0e3bfaa393fdcc9a7f648db3079e239824dcd7c968d63a5bac3513a45cc4aaeb'
PIDS = tuple(f'VP{i:03d}' for i in range(1, 27))
REST_CONTEXT = 'REST_REPAIR_EXACT_SOURCE'
SOURCE_CONTEXT = 'OOD_HRF_STRESS'
DATASET_URL = 'https://www.nitrc.org/frs/downloadlink.php/11884'
MANIFEST = Path(__file__).resolve().parents[1] / 'docs' / 'RECORDED_BACKGROUND_FILES.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str, ensure_ascii=False) + '\n', encoding='utf-8')


def read_optional_archive(base, name):
    """Read a local result file or a member of its local ZIP. Never download outputs."""
    if base is None:
        return None
    base = Path(base)
    if base.is_file() and base.suffix.lower() == '.zip':
        with zipfile.ZipFile(base) as z:
            return z.read(name)
    if (base / name).is_file():
        return (base / name).read_bytes()
    if base.with_suffix('.zip').is_file():
        with zipfile.ZipFile(base.with_suffix('.zip')) as z:
            return z.read(name)
    raise FileNotFoundError(f'Archived input not found: {base} / {name}')


def read_optional_csv(base, name):
    raw = read_optional_archive(base, name)
    if raw is None:
        return None
    if name.endswith('.gz'):
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw))


def load_model(shin_root):
    """Load the unchanged model implementation and its original source-cache layout."""
    module_path = Path(__file__).resolve().parent / 'morphology.py'
    if digest(module_path) != MODEL_SHA256:
        raise RuntimeError('Pinned src/morphology.py does not match the verified implementation')
    m = importlib.import_module('src.morphology')
    shin = Path(shin_root).expanduser().resolve()
    if not shin.is_dir():
        raise FileNotFoundError(f'Shin source cache directory is missing: {shin}')
    m.SHIN_ROOT = shin
    m.N4_10SEED_DIR = shin / '95_N4_FUNCTIONAL_ABC_LBR_FULLY_NESTED_10SEEDS_V1'
    m.HIERARCHICAL_SEED1_DIR = shin / '98_B0_STRICT_HIERARCHICAL_SEED01_FULLY_NESTED_V1'
    m.HIERARCHICAL_10SEED_DIR = shin / '99_B0_STRICT_HIERARCHICAL_10SEEDS_V1'
    m.protected_self_tests()
    return m


def snirf_integer(dataset, name):
    """The SNIRF indices may be scalars or one-element NumPy arrays."""
    values = np.asarray(dataset[()])
    if values.size != 1:
        raise ValueError(f'{name} contains {values.size} values rather than one')
    value = values.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode('utf-8')
    number = float(str(value).strip().strip('\x00'))
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f'Invalid SNIRF {name}: {value!r}')
    return int(number)


def unaugmented_pairs(path):
    """Accept only pairs explicitly marked unaugmented with dataTypeLabel=0."""
    import h5py
    pairs = set()
    with h5py.File(path, 'r') as f:
        if 'nirs' not in f:
            raise RuntimeError('The SNIRF nirs group is missing')
        for data_group in f['nirs']:
            if not re.fullmatch(r'data\d+', data_group):
                continue
            labels = defaultdict(set)
            for name in f['nirs'][data_group]:
                if not re.fullmatch(r'measurementList\d+', name):
                    continue
                g = f['nirs'][data_group][name]
                if not all(k in g for k in ('sourceIndex', 'detectorIndex', 'dataTypeLabel')):
                    continue
                source = snirf_integer(g['sourceIndex'], 'sourceIndex')
                detector = snirf_integer(g['detectorIndex'], 'detectorIndex')
                label = snirf_integer(g['dataTypeLabel'], 'dataTypeLabel')
                labels[(source, detector)].add(label)
            pairs.update(f'S{s}_D{d}' for (s, d), found in labels.items() if found == {0})
    if not pairs:
        raise RuntimeError('No verified unaugmented SNIRF channel pairs')
    return pairs


def read_rest(path):
    import mne
    pairs = unaugmented_pairs(path)
    raw = mne.io.read_raw_snirf(str(path), preload=True, verbose='ERROR')
    raw.pick(picks='fnirs_cw_amplitude')
    raw = mne.preprocessing.nirs.optical_density(raw)
    raw = mne.preprocessing.nirs.beer_lambert_law(raw, ppf=6.0)
    raw.filter(0.01, 0.2, verbose='ERROR')
    raw.resample(2.0, verbose='ERROR')
    selected = [p for p in sorted(pairs)
                if f'{p} hbo' in raw.ch_names and f'{p} hbr' in raw.ch_names]
    if len(selected) < 4:
        raise RuntimeError('Fewer than four paired unaugmented HbO/HbR channels')
    hbo = raw.get_data(picks=[f'{p} hbo' for p in selected])
    hbr = raw.get_data(picks=[f'{p} hbr' for p in selected])
    good = np.isfinite(hbo).all(axis=1) & np.isfinite(hbr).all(axis=1)
    good &= (np.std(hbo, axis=1) > 1e-12) & (np.std(hbr, axis=1) > 1e-12)
    if good.sum() < 4:
        raise RuntimeError('Fewer than four finite, nonconstant HbO/HbR channels')
    return hbo[good], hbr[good], float(raw.info['sfreq'])


def get_manifest():
    data = json.loads(MANIFEST.read_text(encoding='utf-8'))
    participants = data.get('participants', [])
    if len(participants) != 14 or len({p['subject_id'] for p in participants}) != 14:
        raise RuntimeError('Expected 14 unique external source recordings')
    return participants


def download_rest(target):
    """Optional public data download. Original recordings remain outside the repository."""
    import requests
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    archive = target / 'resting_state_dataset_i.zip'
    if not zipfile.is_zipfile(archive):
        partial = archive.with_suffix('.partial')
        with requests.get(DATASET_URL, stream=True, timeout=(30, 180),
                          headers={'User-Agent': 'Mozilla/5.0'}) as response:
            response.raise_for_status()
            with partial.open('wb') as f:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
        if not zipfile.is_zipfile(partial):
            partial.unlink(missing_ok=True)
            raise RuntimeError('Downloaded file was not a valid ZIP. Obtain Dataset I from the provider.')
        partial.replace(archive)
    root = target.resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            dest = (root / member.filename).resolve()
            if not dest.is_relative_to(root) or dest == root:
                raise RuntimeError('Unsafe archive path')
            if member.is_dir():
                continue
            mode = (member.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise RuntimeError('Archive links are not accepted')
        z.extractall(root)
    return archive


def locate_rest_files(rest_root):
    """Use the fixed, checksum-verified recordings, not whichever files happen to exist."""
    rest_root = Path(rest_root)
    files = []
    for k, item in enumerate(get_manifest(), 1):
        candidates = [p for p in rest_root.rglob(item['filename'])
                      if p.parent.name == item['directory']]
        unique = {p.resolve() for p in candidates}
        if len(unique) != 1:
            raise FileNotFoundError(f'Expected one matching SNIRF for {item["subject_id"]}; found {len(unique)}')
        f = unique.pop()
        if digest(f) != item['sha256']:
            raise RuntimeError(f'SNIRF checksum mismatch for {item["subject_id"]}')
        files.append((f'REST_{k:03d}', item, f))
    return files


def get_windows(hbo, hbr, fs, m):
    if not np.isclose(fs, 2.0):
        raise RuntimeError('Recorded-background preprocessing must use 2 Hz')
    n = len(m.FULL_T)
    windows = []
    for start in range(0, hbo.shape[1] - n + 1, int(round(30 * fs))):
        if len(windows) == 4:
            break
        h = hbo[:, start:start+n]
        r = hbr[:, start:start+n]
        if np.isfinite(h).all() and np.isfinite(r).all():
            windows.append((h, r))
    if len(windows) != 4:
        raise RuntimeError('Exactly four complete recorded background windows are required')
    return windows


def make_trials(m, windows, participant_index):
    """Create exactly the 16 original conditions per participant, in original order."""
    trials, metadata = [], []
    for window_index, (h, r) in enumerate(windows):
        trials.append(m.trial_to_predictors(h, r, False, np.nan, np.nan, np.nan))
        metadata.append({'trial_index': 4 * window_index, 'window_index': window_index,
                         'amplitude_in_baseline_sd': np.nan})
        rng = np.random.default_rng(73000 + 100 * participant_index + window_index)
        shape, onset, peak, offset, _ = m.original_half_cosine_shape(rng)
        scale = np.maximum(np.std(h[:, m.FULL_T < 0], axis=1), 1e-12)
        for number, amplitude in enumerate((0.5, 1.0, 2.0), 1):
            support = rng.uniform(0.6, 1.0, size=len(scale))
            inject = (amplitude * scale * support)[:, None] * shape[None, :]
            hp, rp = h.copy(), r.copy()
            hp[:, m.POST_MASK] += inject
            rp[:, m.POST_MASK] -= 0.5 * inject
            trials.append(m.trial_to_predictors(hp, rp, True, onset, peak, offset))
            metadata.append({'trial_index': 4 * window_index + number, 'window_index': window_index,
                             'amplitude_in_baseline_sd': amplitude})
    keys = ('present', 'onset', 'peak', 'offset', 'censored', 'obs', 'state_feat')
    truth = {key: np.asarray([t[key] for t in trials]) for key in keys}
    assert len(trials) == len(metadata) == 16
    return truth, metadata


def build_rest_banks(m, rest_root, reference=None):
    """Reconstruct the 224 conditions; optional archived reference provides strict parity."""
    reference = None if reference is None else reference.sort_values(['participant', 'trial_index'])
    banks = []
    for index, (pid, item, file) in enumerate(locate_rest_files(rest_root)):
        h, r, fs = read_rest(file)
        if h.shape[0] != item['channels']:
            raise RuntimeError(f'Prepared channel count differs for {item["subject_id"]}')
        windows = get_windows(h, r, fs, m)
        truth, metadata = make_trials(m, windows, index)
        if reference is not None:
            g = reference.loc[reference.participant.eq(pid)].sort_values('trial_index')
            if len(g) != 16:
                raise RuntimeError(f'Archived reference lacks 16 trials for {pid}')
            for archive_col, key in [('truth_present', 'present'), ('truth_onset', 'onset'),
                                     ('truth_peak', 'peak'), ('truth_offset', 'offset'),
                                     ('truth_censored', 'censored')]:
                if not np.allclose(g[archive_col].to_numpy(float), truth[key],
                                   atol=1e-7, rtol=1e-7, equal_nan=True):
                    raise RuntimeError(f'Archived truth mismatch for {pid}/{archive_col}')
        banks.append((pid, item, file, windows, truth, metadata))
    if len(banks) != 14:
        raise RuntimeError('Recorded-background cohort incomplete')
    return banks


def fit_frozen_source(m, seed, held='VP001', context_prefix=REST_CONTEXT, caches=None):
    m.configure_seed(seed)
    if caches is None:
        caches, _ = m.load_seed_caches(seed)
    cfg, local, _ = m.load_full_config(seed, held)
    source = [p for p in PIDS if p != held]
    train = m.training_rows(caches, source)
    context = f'{context_prefix}/seed{seed:02d}/held{held}'
    audit = m.FitAudit(source, [held], context)
    model = m.fit_coherent_full(m.n4, train, cfg, local, audit,
                                outer_forbidden={held}, context=context)
    for row in audit.rows:
        if held in row.get('fit_participants', '').split('|'):
            raise RuntimeError('Held-out Shin participant entered a source fit')
    return model, cfg, local, audit


def predict_model(m, model, truth, threshold):
    x = m.predictor_view(truth)
    coarse, prob, state = model.abc.raw(x)
    raw = np.asarray(model.refined_raw(x, coarse, prob, state), float)
    probability = np.asarray(prob, float)
    if raw.shape != (len(truth['present']), 3) or probability.shape != (len(raw),):
        raise RuntimeError('Unexpected model prediction shapes')
    return raw, probability, float(threshold)


def prediction_frame(seed, pid, truth, metadata, raw, prob, threshold, item=None):
    d = pd.DataFrame(metadata).copy()
    d['seed'] = int(seed)
    d['participant'] = pid
    d['truth_present'] = np.asarray(truth['present'], int)
    for key in ('onset', 'peak', 'offset', 'censored'):
        d['truth_' + key] = np.asarray(truth[key])
    d['response_probability'] = np.asarray(prob, float)
    d['threshold'] = float(threshold)
    d['native_response'] = (np.asarray(prob) >= threshold).astype(int)
    for index, key in enumerate(('onset', 'peak', 'offset')):
        d['raw_' + key] = np.asarray(raw)[:, index]
    if item is not None:
        d['external_original_subject_id'] = item['subject_id']
        d['recording_sha256'] = item['sha256']
        d['background_channels'] = item['channels']
    return d


def participant_metric(g, threshold=None, penalty=4.0):
    y = g.truth_present.to_numpy(int)
    p = g.response_probability.to_numpy(float)
    detected = (g.native_response.to_numpy(int) == 1) if threshold is None else (p >= threshold)
    if not ((y == 1).any() and (y == 0).any()):
        raise RuntimeError('At least one positive and negative condition is required')
    sensitivity = float(detected[y == 1].mean())
    specificity = float((~detected[y == 0]).mean())
    result = {'sensitivity': sensitivity, 'specificity': specificity,
              'balanced_accuracy': (sensitivity + specificity) / 2}
    censored = g.truth_censored.to_numpy(int) == 1
    for key in ('onset', 'peak', 'offset', 'duration'):
        eligible = (y == 1) & (~censored if key in ('offset', 'duration') else True)
        if key == 'duration':
            truth = g.truth_offset.to_numpy(float) - g.truth_onset.to_numpy(float)
            pred = g.raw_offset.to_numpy(float) - g.raw_onset.to_numpy(float)
        else:
            truth = g['truth_' + key].to_numpy(float)
            pred = g['raw_' + key].to_numpy(float)
        err = np.abs(truth - pred)
        covered = eligible & detected & np.isfinite(err)
        result[key + '_loss'] = (float((err[covered].sum() + penalty * (eligible & ~covered).sum()) /
                                       eligible.sum()) if eligible.any() else np.nan)
        result[key + '_conditional_mae'] = float(err[covered].mean()) if covered.any() else np.nan
        finite = eligible & np.isfinite(err)
        result[key + '_ungated_mae'] = float(err[finite].mean()) if finite.any() else np.nan
        result[key + '_coverage'] = float(covered.sum() / eligible.sum()) if eligible.any() else np.nan
    return result


def bootstrap_mean(values, draws=20000, seed=891):
    x = np.asarray(values, float)
    if x.ndim != 1 or len(x) != 14 or not np.isfinite(x).all():
        raise ValueError('Bootstrap expects 14 independent external participant values')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(draws, len(x)))
    return tuple(float(v) for v in np.percentile(x[idx].mean(axis=1), (2.5, 97.5)))


def self_test():
    class Fake:
        def __init__(self, value):
            self.value = value
        def __getitem__(self, index):
            return self.value
    assert snirf_integer(Fake(np.array([3])), 'sourceIndex') == 3
    assert snirf_integer(Fake(np.array([[2]])), 'detectorIndex') == 2
    assert snirf_integer(Fake(np.array(0)), 'dataTypeLabel') == 0
    p = np.array([1e-6, 0.8, 0.01, 0.9])
    d = pd.DataFrame({'truth_present': [1, 1, 0, 0], 'response_probability': p,
                      'native_response': [0, 1, 0, 1], 'truth_censored': [0, 0, 0, 0],
                      'truth_onset': [1., 2., np.nan, np.nan],
                      'truth_peak': [3., 4., np.nan, np.nan],
                      'truth_offset': [7., 8., np.nan, np.nan],
                      'raw_onset': [1.1, 2.1, 2., 3.], 'raw_peak': [3.1, 4.1, 4., 5.],
                      'raw_offset': [7.1, 8.1, 6., 7.]})
    m = participant_metric(d)
    assert np.isclose(m['sensitivity'], .5) and np.isclose(m['specificity'], .5)
    assert np.isclose(m['onset_loss'], (4 + .1) / 2)
    return True


if __name__ == '__main__':
    self_test()
    print('PASS: recorded-background helper self-tests')
