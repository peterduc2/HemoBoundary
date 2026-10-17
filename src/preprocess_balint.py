"""Bálint fNIRS preprocessing."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Any

def _ensure_dependencies() -> None:
    required = {'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy', 'h5py': 'h5py', 'requests': 'requests', 'tqdm': 'tqdm', 'mne': 'mne>=1.10,<1.12'}
    missing = [pip_name for import_name, pip_name in required.items() if importlib.util.find_spec(import_name) is None]
    if missing:
        print('[setup] Installing missing packages:', ', '.join(missing))
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
_ensure_dependencies()
import numpy as np
import pandas as pd
import h5py
import requests
from tqdm.auto import tqdm
import mne
from mne.preprocessing.nirs import optical_density, source_detector_distances, temporal_derivative_distribution_repair
VERSION = 'BALINT_DOWNLOAD_PREPROCESS_FEATURES_v1_2026_09_05'
DRYAD_DOI = '10.5061/dryad.crjdfn3g9'
DRYAD_LANDING = 'https://datadryad.org/dataset/doi:10.5061/dryad.crjdfn3g9'
DRYAD_SNIRF_FILE_ID = 4161846
DRYAD_DIRECT_FILE_URL = f'https://datadryad.org/downloads/file_stream/{DRYAD_SNIRF_FILE_ID}'
DRYAD_API_FILE_URL = f'https://datadryad.org/api/v2/files/{DRYAD_SNIRF_FILE_ID}/download'
EXPECTED_ZIP_NAME = 'snirf_local-bids.zip'
MIN_EXPECTED_ZIP_BYTES = 600000000
AUTHOR_SFREQ_HZ = 14.0
FILTER_LOW_HZ = 0.01
FILTER_HIGH_HZ = 0.1
SHORT_CHANNEL_MAX_DIST_M = 0.01
WAVELENGTHS = (780, 805, 830)
HBO_COEFF = {780: -1.4887, 805: 0.597, 830: 1.4878}
HBR_COEFF = {780: 1.8545, 805: -0.2394, 830: -1.0947}
EPOCH_TMIN = -5.0
EPOCH_TMAX = 24.0
BASELINE_TMIN = -5.0
BASELINE_TMAX = 0.0
OUTPUT_SFREQ_HZ = 10.0
STIM_DURATION_MIN = 12.0
STIM_DURATION_MAX = 14.5
EXPECTED_STIMULI_PER_PARTICIPANT = 40
H5_COMPRESSION = 'gzip'
H5_COMPRESSION_LEVEL = 4
FEATURE_WINDOWS = {'early_0_5': (0.0, 5.0), 'stim_0_13': (0.0, 13.0), 'recovery_13_24': (13.0, 24.0), 'full_0_24': (0.0, 24.0)}

def stable_hash(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)
def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    compression = 'gzip' if path.suffix == '.gz' else None
    name = path.name.lower()
    sep = '\t' if name.endswith('.tsv') or name.endswith('.tsv.gz') else ','
    df.to_csv(tmp, index=False, compression=compression, sep=sep)
    tmp.replace(path)
def maybe_mount_drive() -> None:
    if not importlib.util.find_spec('google.colab'):
        return
    target = Path('/content/gdrive/MyDrive')
    if target.exists():
        return
    try:
        from google.colab import drive
        drive.mount('/content/gdrive')
    except Exception as exc:
        print('[warning] Google Drive mount failed:', repr(exc))
def default_project_root() -> Path:
    env = os.environ.get('BALINT_ROOT', '').strip()
    if env:
        return Path(env)
    if Path('/content/gdrive/MyDrive').exists():
        return Path('/content/gdrive/MyDrive/HonorProject/HemoBoundary/Balint')
    return Path.cwd() / 'HemoBoundary' / 'Balint'
def _safe_decode_array(values) -> List[str]:
    out = []
    for x in values:
        if isinstance(x, bytes):
            out.append(x.decode('utf-8', errors='replace'))
        else:
            out.append(str(x))
    return out
def _iqr(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.quantile(x, 0.75) - np.quantile(x, 0.25))
def _safe_slope(t: np.ndarray, y: np.ndarray) -> float:
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    if np.sum(m) < 3:
        return np.nan
    tt = t[m]
    yy = y[m]
    tt = tt - np.mean(tt)
    denom = float(np.dot(tt, tt))
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(tt, yy - np.mean(yy)) / denom)
def _interp_nonfinite_1d(y: np.ndarray) -> Tuple[np.ndarray, float]:
    y = np.asarray(y, float).copy()
    finite = np.isfinite(y)
    frac = float(np.mean(finite))
    if np.sum(finite) < 2:
        return (y, frac)
    if not finite.all():
        idx = np.arange(len(y))
        y[~finite] = np.interp(idx[~finite], idx[finite], y[finite])
    return (y, frac)
def _download_stream(url: str, dest: Path, token: Optional[str]=None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/150.0 Safari/537.36', 'Accept': '*/*', 'Referer': DRYAD_LANDING}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    existing = part.stat().st_size if part.exists() else 0
    if existing > 0:
        headers['Range'] = f'bytes={existing}-'
    with requests.get(url, headers=headers, stream=True, timeout=(30, 180), allow_redirects=True) as r:
        if r.status_code not in (200, 206):
            body = ''
            try:
                body = r.text[:300]
            except Exception:
                pass
            raise RuntimeError(f'HTTP {r.status_code} from Dryad download endpoint. {body}')
        if r.status_code == 200 and existing > 0:
            existing = 0
            if part.exists():
                part.unlink()
        mode = 'ab' if existing > 0 and r.status_code == 206 else 'wb'
        total_header = r.headers.get('Content-Length')
        total = int(total_header) + existing if total_header else None
        with open(part, mode) as f, tqdm(total=total, initial=existing, unit='B', unit_scale=True, unit_divisor=1024, desc=f'Downloading {dest.name}') as bar:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))
    part.replace(dest)
def _validate_balint_zip(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 100000000 and zipfile.is_zipfile(path)
    except Exception:
        return False
def _find_manual_balint_archive(download_dir: Path) -> Optional[Path]:
    """Locate a manually downloaded archive without forcing a 720 MB duplicate copy."""
    candidates = [
        download_dir / EXPECTED_ZIP_NAME,
        Path('/content') / EXPECTED_ZIP_NAME,
        Path('/content/gdrive/MyDrive') / EXPECTED_ZIP_NAME,
        Path('/content/drive/MyDrive') / EXPECTED_ZIP_NAME
    ]
    for p in candidates:
        if _validate_balint_zip(p):
            return p
    roots = [Path('/content/gdrive/MyDrive/HonorProject'), Path('/content/drive/MyDrive/HonorProject'), download_dir.parent.parent]
    seen = set()
    for root in roots:
        try:
            root = root.resolve()
        except Exception:
            pass
        if str(root) in seen or not root.exists():
            continue
        seen.add(str(root))
        patterns = [EXPECTED_ZIP_NAME, f'*/{EXPECTED_ZIP_NAME}', f'*/*/{EXPECTED_ZIP_NAME}', f'*/*/*/{EXPECTED_ZIP_NAME}']
        for pat in patterns:
            for p in root.glob(pat):
                if _validate_balint_zip(p):
                    return p
    return None
def ensure_dataset_archive(download_dir: Path) -> Path:
    """Dryad-compatible archive acquisition."""
    download_dir.mkdir(parents=True, exist_ok=True)
    existing = _find_manual_balint_archive(download_dir)
    if existing is not None:
        print('[download] Found valid Bálint archive:', existing, f'({existing.stat().st_size / 1000000.0:.1f} MB)')
        return existing
    token = os.environ.get('DRYAD_API_TOKEN', '').strip() or None
    persistent_dest = download_dir / EXPECTED_ZIP_NAME
    if token:
        print('[download] No local archive found.')
        print('[download] Using authenticated Dryad API file download.')
        try:
            _download_stream(DRYAD_API_FILE_URL, persistent_dest, token)
            if not _validate_balint_zip(persistent_dest):
                raise RuntimeError('Authenticated download completed but the result is not a valid Bálint ZIP archive.')
            print('[download] Authenticated download complete:', persistent_dest, f'({persistent_dest.stat().st_size / 1000000.0:.1f} MB)')
            return persistent_dest
        except Exception as exc:
            if persistent_dest.exists():
                try:
                    persistent_dest.unlink()
                except Exception:
                    pass
            raise RuntimeError(f'Authenticated Dryad download failed. Original error: {repr(exc)}') from exc
    raise RuntimeError(f'\nBÁLINT DATA ZIP IS NOT YET AVAILABLE LOCALLY.\n\nThis is expected with current Dryad access: anonymous scripted file downloads can be rejected with HTTP 403.\n\nDo this once in your normal web browser:\n  1) Open: {DRYAD_LANDING}\n  2) Under Data files, click: {EXPECTED_ZIP_NAME}\n     (published size ≈ 720.74 MB)\n  3) Put/upload that ZIP in EITHER of these locations:\n       {persistent_dest}\n     OR /content/{EXPECTED_ZIP_NAME}\n\nRecommended persistent location is the Google Drive path above.\nThen rerun the SAME Colab code. The script will find the ZIP automatically, extract it, and continue preprocessing.\n\nAlternative for an authenticated Dryad API account:\n       %env DRYAD_API_TOKEN=YOUR_TOKEN\nthen rerun the code.')
def find_bids_root(search_root: Path) -> Optional[Path]:
    if (search_root / 'participants.tsv').is_file():
        return search_root
    candidates = []
    for p in search_root.rglob('participants.tsv'):
        root = p.parent
        if list(root.glob('sub-*')):
            candidates.append(root)
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda x: len(x.parts))
    return candidates[0]
def ensure_extracted_bids(archive: Path, bids_extract_root: Path) -> Path:
    existing = find_bids_root(bids_extract_root)
    if existing is not None:
        print('[extract] Existing BIDS root:', existing)
        return existing
    bids_extract_root.mkdir(parents=True, exist_ok=True)
    print('[extract] Extracting:', archive)
    with zipfile.ZipFile(archive, 'r') as zf:
        zf.extractall(bids_extract_root)
    root = find_bids_root(bids_extract_root)
    if root is not None:
        print('[extract] BIDS root:', root)
        return root
    nested = list(bids_extract_root.rglob(EXPECTED_ZIP_NAME))
    if nested:
        print('[extract] Found nested SNIRF archive:', nested[0])
        nested_root = bids_extract_root / 'snirf_local_bids_contents'
        nested_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(nested[0], 'r') as zf:
            zf.extractall(nested_root)
        root = find_bids_root(nested_root)
        if root is not None:
            print('[extract] BIDS root:', root)
            return root
    raise RuntimeError(f'Could not locate participants.tsv + sub-* folders under {bids_extract_root}')
def selected_participants(bids_root: Path, group: str) -> List[str]:
    dirs = sorted([p for p in bids_root.glob('sub-*') if p.is_dir()])
    ids = [p.name.replace('sub-', '', 1) for p in dirs]
    if group == 'nh':
        ids = [x for x in ids if x.upper().startswith('C')]
    elif group == 'ci':
        ids = [x for x in ids if x.upper().startswith('S')]
    elif group == 'all':
        pass
    else:
        raise ValueError(group)
    return ids
def participant_files(bids_root: Path, pid: str) -> Tuple[Path, Path]:
    nirs_dir = bids_root / f'sub-{pid}' / 'nirs'
    snirf = nirs_dir / f'sub-{pid}_task-full_nirs.snirf'
    events = nirs_dir / f'sub-{pid}_task-full_events.tsv'
    if not snirf.is_file():
        candidates = list(nirs_dir.glob('*_nirs.snirf'))
        if len(candidates) == 1:
            snirf = candidates[0]
    if not events.is_file():
        candidates = list(nirs_dir.glob('*_events.tsv'))
        if len(candidates) == 1:
            events = candidates[0]
    if not snirf.is_file():
        raise FileNotFoundError(f'SNIRF missing for {pid}: {snirf}')
    if not events.is_file():
        raise FileNotFoundError(f'events.tsv missing for {pid}: {events}')
    return (snirf, events)
def choose_stimulus_events(events: pd.DataFrame) -> pd.DataFrame:
    if 'onset' not in events.columns:
        raise ValueError('events.tsv has no onset column.')
    d = events.copy()
    d['onset'] = pd.to_numeric(d['onset'], errors='coerce')
    if 'duration' in d.columns:
        duration = pd.to_numeric(d['duration'], errors='coerce')
    else:
        duration = pd.Series(np.nan, index=d.index)
    d['_duration_num'] = duration
    stim = d[d['_duration_num'].between(STIM_DURATION_MIN, STIM_DURATION_MAX, inclusive='both')].copy()
    if len(stim) < 20:
        label_col = 'trial_type' if 'trial_type' in d.columns else 'value' if 'value' in d.columns else None
        text = d[label_col].astype(str).str.lower() if label_col else pd.Series('', index=d.index)
        obvious_nonstim = text.str.contains('question|answer|start|stop|break|pause|response', regex=True, na=False)
        stim = d[~obvious_nonstim & d['_duration_num'].between(8.0, 18.0, inclusive='both')].copy()
    stim = stim[np.isfinite(stim['onset'])].copy()
    stim = stim.sort_values('onset').reset_index(drop=True)
    return stim
def event_label(row: Mapping[str, Any]) -> str:
    for key in ['trial_type', 'condition', 'value', 'stim_type', 'event_type']:
        if key in row and pd.notna(row[key]):
            return str(row[key])
    return 'unknown'
def map_modality(label: str) -> str:
    s = str(label).strip().lower()
    compact = re.sub('[^a-z0-9]+', '', s)
    if 'noise' in s or 'sin' in compact:
        return 'speech_in_noise'
    if 'audiovis' in s or 'audio_visual' in s or 'audio-visual' in s or (compact in {'av', 'audiovisual'}):
        return 'audiovisual'
    if 'visual' in s or 'lip' in s or compact in {'vo', 'visualonly'}:
        return 'visual_speech'
    if 'quiet' in s or compact in {'ao', 'audioonly', 'speechquiet', 'sq'}:
        return 'speech_in_quiet'
    return 'unmapped'
def row_explicitly_bad(row: Mapping[str, Any]) -> Tuple[bool, str]:
    """
    Only use fields that explicitly communicate exclusion/artifact status.
    Do not infer "bad" from comprehension accuracy.
    """
    for key, value in row.items():
        k = str(key).lower()
        if not any((token in k for token in ['exclude', 'artifact', 'interference', 'movement', 'motion', 'valid', 'quality', 'reject'])):
            continue
        text = str(value).strip().lower()
        if not text or text in {'nan', 'n/a', 'na', 'none'}:
            continue
        if 'valid' in k:
            if text in {'0', 'false', 'no', 'invalid', 'bad'}:
                return (True, f'{key}={value}')
        elif text in {'1', 'true', 'yes', 'bad', 'reject', 'rejected', 'artifact', 'movement', 'motion', 'interference', 'exclude', 'excluded'}:
            return (True, f'{key}={value}')
    return (False, '')
CHANNEL_RE = re.compile('^S(\\d+)_D(\\d+)\\s+(\\d+(?:\\.\\d+)?)$')

def parse_channel_name(name: str) -> Optional[Tuple[int, int, int]]:
    m = CHANNEL_RE.match(str(name).strip())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(round(float(m.group(3)))))
def load_absorbance_raw(snirf: Path) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_snirf(snirf, optode_frame='unknown', preload=True, verbose='ERROR')
    types = set(raw.get_channel_types())
    if 'fnirs_od' in types and 'fnirs_cw_amplitude' not in types:
        od = raw.copy()
        source_type = 'fnirs_od_from_snirf'
    elif 'fnirs_cw_amplitude' in types:
        od = optical_density(raw, verbose='ERROR')
        source_type = 'cw_amplitude_converted_to_od'
    else:
        raise RuntimeError(f'Unsupported SNIRF fNIRS channel types: {sorted(types)}')
    od.info['description'] = source_type
    return od
def infer_short_channel_threshold_m(od: mne.io.BaseRaw) -> Tuple[float, Dict[str, Any]]:
    """
    Prefer the MNE-NIRS 10-mm convention when it yields the published three
    short source-detector pairs. Otherwise use geometry to separate the three
    shortest complete source-detector pairs from the long channels.
    """
    distances = source_detector_distances(od.info)
    pair_dist: Dict[Tuple[int, int], float] = {}
    pair_waves: Dict[Tuple[int, int], set] = {}
    for idx, name in enumerate(od.ch_names):
        token = parse_channel_name(name)
        if token is None:
            continue
        src, det, wave = token
        pair = (src, det)
        pair_dist[pair] = float(distances[idx])
        pair_waves.setdefault(pair, set()).add(wave)
    complete = sorted([(pair, dist) for pair, dist in pair_dist.items() if all((w in pair_waves.get(pair, set()) for w in WAVELENGTHS)) and np.isfinite(dist) and (dist > 0)], key=lambda z: z[1])
    if len(complete) < 4:
        raise RuntimeError(f'Need >=4 complete source-detector pairs to infer short channels; found {len(complete)}.')
    conventional_n = sum((dist < SHORT_CHANNEL_MAX_DIST_M for _, dist in complete))
    if conventional_n == 3:
        threshold = SHORT_CHANNEL_MAX_DIST_M
        rule = 'fixed_10mm_yields_published_three_pairs'
    else:
        third = complete[2][1]
        fourth = complete[3][1]
        if not fourth > third:
            raise RuntimeError('Could not separate the three shortest SDS pairs from long pairs.')
        threshold = float((third + fourth) / 2.0)
        rule = 'geometry_midpoint_between_3rd_and_4th_shortest_pairs'
    short_pairs = [f'S{pair[0]}_D{pair[1]}' for pair, dist in complete if dist < threshold]
    meta = {
        'short_threshold_used_m': float(threshold),
        'short_threshold_rule': rule,
        'short_pairs_from_geometry': short_pairs,
        'three_short_pairs_recovered': int(len(short_pairs) == 3),
        'three_short_pair_distances_m': [float(dist) for _, dist in complete[:3]],
        'fourth_pair_distance_m': float(complete[3][1])
    }
    return (threshold, meta)
def _channel_midpoint_m(info: mne.Info, idx: int) -> np.ndarray:
    """
    Robust fNIRS channel midpoint. MNE normally stores the channel midpoint in
    loc[:3] and source/detector coordinates in loc[3:6] / loc[6:9].
    """
    loc = np.asarray(info['chs'][idx]['loc'], float)
    if len(loc) >= 3:
        p = loc[:3]
        if np.all(np.isfinite(p)) and np.linalg.norm(p) > 1e-09:
            return p.copy()
    if len(loc) >= 9:
        src = loc[3:6]
        det = loc[6:9]
        if np.all(np.isfinite(src)) and np.all(np.isfinite(det)) and (np.linalg.norm(src) > 1e-09 or np.linalg.norm(det) > 1e-09):
            return 0.5 * (src + det)
    return np.zeros(3, float)
def nearest_short_channel_regression_numpy(od: mne.io.BaseRaw, short_threshold_m: float) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    """Nearest-short-channel regression implemented only with NumPy."""
    out = od.copy().load_data()
    data = np.asarray(out.get_data(), float).copy()
    distances = np.asarray(source_detector_distances(out.info), float)
    parsed = [parse_channel_name(n) for n in out.ch_names]
    short_idx = [i for i, token in enumerate(parsed) if token is not None and token[2] in WAVELENGTHS and np.isfinite(distances[i]) and (distances[i] > 0) and (distances[i] < short_threshold_m)]
    long_idx = [i for i, token in enumerate(parsed) if token is not None and token[2] in WAVELENGTHS and np.isfinite(distances[i]) and (distances[i] >= short_threshold_m)]
    if not short_idx:
        raise RuntimeError('No short-separation channels were detected for short-channel regression.')
    midpoints = {i: _channel_midpoint_m(out.info, i) for i in short_idx + long_idx}
    mappings = []
    n_regressed = 0
    n_unmatched = 0
    for li in long_idx:
        token = parsed[li]
        assert token is not None
        wave = token[2]
        candidates = [si for si in short_idx if parsed[si] is not None and parsed[si][2] == wave]
        if not candidates:
            n_unmatched += 1
            mappings.append({
                'long_channel': out.ch_names[li],
                'short_channel': '',
                'wavelength_nm': int(wave),
                'distance_between_channel_midpoints_m': np.nan,
                'beta': np.nan,
                'status': 'no_same_wavelength_short_channel'
            })
            continue
        lp = midpoints[li]
        distances_to_short = [float(np.linalg.norm(lp - midpoints[si])) for si in candidates]
        best_pos = int(np.argmin(distances_to_short))
        si = candidates[best_pos]
        spatial_distance = distances_to_short[best_pos]
        y = np.asarray(data[li], float)
        x = np.asarray(data[si], float)
        finite = np.isfinite(x) & np.isfinite(y)
        if np.sum(finite) < 20:
            n_unmatched += 1
            mappings.append({
                'long_channel': out.ch_names[li],
                'short_channel': out.ch_names[si],
                'wavelength_nm': int(wave),
                'distance_between_channel_midpoints_m': spatial_distance,
                'beta': np.nan,
                'status': 'insufficient_finite_samples'
            })
            continue
        xf = x[finite]
        yf = y[finite]
        x_mean = float(np.mean(xf))
        y_mean = float(np.mean(yf))
        xc = xf - x_mean
        yc = yf - y_mean
        denom = float(np.dot(xc, xc))
        if not np.isfinite(denom) or denom <= 1e-20:
            beta = 0.0
        else:
            beta = float(np.dot(xc, yc) / denom)
        corrected = y.copy()
        corrected[finite] = yf - beta * (xf - x_mean)
        data[li] = corrected
        n_regressed += 1
        mappings.append({
            'long_channel': out.ch_names[li],
            'short_channel': out.ch_names[si],
            'wavelength_nm': int(wave),
            'distance_between_channel_midpoints_m': spatial_distance,
            'beta': beta,
            'status': 'regressed'
        })
    out._data[:] = data
    meta = {
        'implementation': 'numpy_nearest_same_wavelength_OLS',
        'n_short_wavelength_channels': int(len(short_idx)),
        'n_long_wavelength_channels': int(len(long_idx)),
        'n_long_channels_regressed': int(n_regressed),
        'n_long_channels_unmatched': int(n_unmatched),
        'mapping': mappings
    }
    return (out, meta)
def author_preprocess_absorbance(od: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, float, Dict[str, Any]]:
    """
    Author-matched first two processing operations:
      nearest short-channel regression -> 0.01-0.10 Hz band-pass.
    """
    short_threshold, short_meta = infer_short_channel_threshold_m(od)
    corrected, regression_meta = nearest_short_channel_regression_numpy(od.copy(), short_threshold)
    short_meta['short_channel_regression'] = regression_meta
    corrected.filter(l_freq=FILTER_LOW_HZ, h_freq=FILTER_HIGH_HZ, picks='all', method='fir', phase='zero', fir_design='firwin', verbose='ERROR')
    return (corrected, short_threshold, short_meta)
def three_wavelength_to_hb(absorbance: mne.io.BaseRaw, short_threshold_m: float) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, Dict[str, Any]]:
    """Convert the filtered 780/805/830 absorbance channels to HbO/HbR using the"""
    names = list(absorbance.ch_names)
    parsed = [parse_channel_name(n) for n in names]
    distances = source_detector_distances(absorbance.info)
    pairs: Dict[Tuple[int, int], Dict[int, int]] = {}
    pair_distance: Dict[Tuple[int, int], float] = {}
    pair_position: Dict[Tuple[int, int], np.ndarray] = {}
    for idx, token in enumerate(parsed):
        if token is None:
            continue
        s, d, w = token
        if w not in WAVELENGTHS:
            continue
        pairs.setdefault((s, d), {})[w] = idx
        pair_distance[s, d] = float(distances[idx])
        pair_position[s, d] = np.asarray(absorbance.info['chs'][idx]['loc'][:3], float).copy()
    long_pairs = []
    short_pairs = []
    incomplete_pairs = []
    for pair, wave_map in sorted(pairs.items()):
        if not all((w in wave_map for w in WAVELENGTHS)):
            incomplete_pairs.append(pair)
            continue
        dist = pair_distance[pair]
        if dist < short_threshold_m:
            short_pairs.append(pair)
        else:
            long_pairs.append(pair)
    data = absorbance.get_data()
    hbo_list = []
    hbr_list = []
    valid_pair_names = []
    valid_dist = []
    valid_pos = []
    invalid_due_to_nan = []
    for pair in long_pairs:
        wave_map = pairs[pair]
        A = {w: np.asarray(data[wave_map[w]], float) for w in WAVELENGTHS}
        hbo = sum((HBO_COEFF[w] * A[w] for w in WAVELENGTHS))
        hbr = sum((HBR_COEFF[w] * A[w] for w in WAVELENGTHS))
        hbo_i, f_hbo = _interp_nonfinite_1d(hbo)
        hbr_i, f_hbr = _interp_nonfinite_1d(hbr)
        if min(f_hbo, f_hbr) < 0.95:
            invalid_due_to_nan.append(f'S{pair[0]}_D{pair[1]}')
            hbo_list.append(np.full_like(hbo, np.nan, dtype=float))
            hbr_list.append(np.full_like(hbr, np.nan, dtype=float))
        else:
            hbo_list.append(hbo_i)
            hbr_list.append(hbr_i)
        valid_pair_names.append(f'S{pair[0]}_D{pair[1]}')
        valid_dist.append(pair_distance[pair])
        valid_pos.append(pair_position[pair])
    if not hbo_list:
        raise RuntimeError('No complete long source-detector pairs were found.')
    hbo_arr = np.asarray(hbo_list, float)
    hbr_arr = np.asarray(hbr_list, float)
    valid_mask = (np.isfinite(hbo_arr).mean(axis=1) >= 0.95) & (np.isfinite(hbr_arr).mean(axis=1) >= 0.95)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx):
        tddr_data = []
        tddr_names = []
        tddr_types = []
        for i in valid_idx:
            tddr_data.extend([hbo_arr[i], hbr_arr[i]])
            tddr_names.extend([f'{valid_pair_names[i]} hbo', f'{valid_pair_names[i]} hbr'])
            tddr_types.extend(['hbo', 'hbr'])
        info = mne.create_info(ch_names=tddr_names, sfreq=float(absorbance.info['sfreq']), ch_types=tddr_types)
        hb_raw = mne.io.RawArray(np.asarray(tddr_data, float), info, verbose='ERROR')
        hb_raw = temporal_derivative_distribution_repair(hb_raw, verbose='ERROR')
        clean = hb_raw.get_data()
        for j, i in enumerate(valid_idx):
            hbo_arr[i] = clean[2 * j]
            hbr_arr[i] = clean[2 * j + 1]
    meta = {
        'n_complete_long_pairs': int(len(long_pairs)),
        'n_complete_short_pairs': int(len(short_pairs)),
        'n_incomplete_pairs': int(len(incomplete_pairs)),
        'n_valid_long_pairs_after_nan_check': int(np.sum(valid_mask)),
        'invalid_long_pairs_due_to_nan': invalid_due_to_nan,
        'short_pairs': [f'S{s}_D{d}' for s, d in short_pairs],
        'incomplete_pairs': [f'S{s}_D{d}' for s, d in incomplete_pairs]
    }
    return (hbo_arr, hbr_arr, valid_pair_names, np.asarray(valid_dist, float), np.asarray(valid_pos, float), meta)
def preprocess_participant(snirf: Path) -> Dict[str, Any]:
    od = load_absorbance_raw(snirf)
    sfreq = float(od.info['sfreq'])
    od_clean, short_threshold, short_meta = author_preprocess_absorbance(od)
    hbo, hbr, pair_names, pair_dist, pair_pos, meta = three_wavelength_to_hb(od_clean, short_threshold)
    meta.update(short_meta)
    return {
        'sfreq': sfreq,
        'n_times': int(od_clean.n_times),
        'duration_sec': float((od_clean.n_times - 1) / sfreq),
        'hbo': hbo,
        'hbr': hbr,
        'pair_names': pair_names,
        'pair_distance_m': pair_dist,
        'pair_position_m': pair_pos,
        'valid_pair_mask': (np.isfinite(hbo).mean(axis=1) >= 0.95) & (np.isfinite(hbr).mean(axis=1) >= 0.95),
        'source_type': od.info.get('description', 'unknown'),
        'short_threshold_used_m': float(short_threshold),
        'preprocess_meta': meta
    }
def standard_epoch_times() -> np.ndarray:
    n = int(round((EPOCH_TMAX - EPOCH_TMIN) * OUTPUT_SFREQ_HZ)) + 1
    return np.linspace(EPOCH_TMIN, EPOCH_TMAX, n, dtype=float)
def interpolate_epoch(signal: np.ndarray, sfreq: float, onset_sec: float, rel_times: np.ndarray) -> Optional[np.ndarray]:
    signal = np.asarray(signal, float)
    abs_target = onset_sec + rel_times
    recording_times = np.arange(signal.shape[1], dtype=float) / sfreq
    if abs_target[0] < recording_times[0] - 1e-06:
        return None
    if abs_target[-1] > recording_times[-1] + 1e-06:
        return None
    out = np.full((signal.shape[0], len(rel_times)), np.nan, dtype=float)
    for ch in range(signal.shape[0]):
        y = signal[ch]
        finite = np.isfinite(y)
        if np.sum(finite) < 2:
            continue
        out[ch] = np.interp(abs_target, recording_times[finite], y[finite])
    return out
def baseline_correct(epoch: np.ndarray, times: np.ndarray) -> np.ndarray:
    out = np.asarray(epoch, float).copy()
    bm = (times >= BASELINE_TMIN - 1e-09) & (times <= BASELINE_TMAX + 1e-09)
    baseline = np.nanmean(out[:, bm], axis=1, keepdims=True)
    out -= baseline
    return out
def _window_features(times: np.ndarray, y: np.ndarray, lo: float, hi: float) -> Dict[str, float]:
    m = (times >= lo - 1e-09) & (times <= hi + 1e-09)
    t = times[m]
    x = np.asarray(y[m], float)
    finite = np.isfinite(x)
    if np.sum(finite) < 3:
        return {'mean': np.nan, 'std': np.nan, 'slope': np.nan, 'auc': np.nan, 'max': np.nan, 'min': np.nan, 'tmax': np.nan, 'tmin': np.nan}
    tt = t[finite]
    xx = x[finite]
    imax = int(np.argmax(xx))
    imin = int(np.argmin(xx))
    try:
        auc = float(np.trapezoid(xx, tt))
    except AttributeError:
        auc = float(np.trapz(xx, tt))
    return {
        'mean': float(np.mean(xx)),
        'std': float(np.std(xx)),
        'slope': _safe_slope(tt, xx),
        'auc': auc,
        'max': float(xx[imax]),
        'min': float(xx[imin]),
        'tmax': float(tt[imax]),
        'tmin': float(tt[imin])
    }
def channel_feature_row(
    participant: str,
    trial_index: int,
    trial_label: str,
    modality: str,
    onset_sec: float,
    channel: str,
    distance_m: float,
    hbo: np.ndarray,
    hbr: np.ndarray,
    times: np.ndarray
) -> Dict[str, Any]:
    row = {
        'participant': participant,
        'trial_index': int(trial_index),
        'trial_label': trial_label,
        'modality': modality,
        'event_onset_sec': float(onset_sec),
        'channel': channel,
        'source_detector_distance_m': float(distance_m),
        'channel_valid': int(np.isfinite(hbo).mean() >= 0.95 and np.isfinite(hbr).mean() >= 0.95)
    }
    bm = (times >= -5.0) & (times <= 0.0)
    row['hbo_baseline_sd'] = float(np.nanstd(hbo[bm]))
    row['hbr_baseline_sd'] = float(np.nanstd(hbr[bm]))
    for chrom, signal in [('hbo', hbo), ('hbr', hbr)]:
        for window_name, (lo, hi) in FEATURE_WINDOWS.items():
            f = _window_features(times, signal, lo, hi)
            for metric, value in f.items():
                row[f'{chrom}_{window_name}_{metric}'] = value
    fm = (times >= 0.0) & (times <= 24.0)
    x = hbo[fm]
    y = hbr[fm]
    ok = np.isfinite(x) & np.isfinite(y)
    row['hbo_hbr_corr_0_24'] = float(np.corrcoef(x[ok], y[ok])[0, 1]) if np.sum(ok) >= 5 and np.nanstd(x[ok]) > 1e-12 and (np.nanstd(y[ok]) > 1e-12) else np.nan
    row['hbo_recovery_minus_stim_mean'] = row['hbo_recovery_13_24_mean'] - row['hbo_stim_0_13_mean']
    row['hbr_recovery_minus_stim_mean'] = row['hbr_recovery_13_24_mean'] - row['hbr_stim_0_13_mean']
    return row
def aggregate_trial_features(channel_df: pd.DataFrame) -> pd.DataFrame:
    if channel_df.empty:
        return pd.DataFrame()
    id_cols = ['participant', 'trial_index', 'trial_label', 'modality', 'event_onset_sec']
    numeric_cols = [c for c in channel_df.columns if c not in id_cols and c not in {'channel'} and pd.api.types.is_numeric_dtype(channel_df[c]) and (c != 'source_detector_distance_m')]
    rows = []
    for keys, d in channel_df.groupby(id_cols, dropna=False):
        row = dict(zip(id_cols, keys))
        row['n_channels'] = int(len(d))
        row['n_valid_channels'] = int(d['channel_valid'].sum())
        row['valid_channel_fraction'] = float(d['channel_valid'].mean())
        for c in numeric_cols:
            if c == 'channel_valid':
                continue
            x = pd.to_numeric(d[c], errors='coerce').to_numpy(float)
            row[f'{c}__median_across_channels'] = float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan
            row[f'{c}__iqr_across_channels'] = _iqr(x)
        rows.append(row)
    return pd.DataFrame(rows)
def write_participant_h5(
    h5: h5py.File,
    participant: str,
    pre: Mapping[str, Any],
    epochs_hbo: np.ndarray,
    epochs_hbr: np.ndarray,
    times: np.ndarray,
    trial_rows: List[Dict[str, Any]]
) -> None:
    if participant in h5:
        del h5[participant]
    g = h5.create_group(participant)
    g.attrs['participant'] = participant
    g.attrs['native_sfreq_hz'] = float(pre['sfreq'])
    g.attrs['output_sfreq_hz'] = float(OUTPUT_SFREQ_HZ)
    g.attrs['filter_low_hz'] = float(FILTER_LOW_HZ)
    g.attrs['filter_high_hz'] = float(FILTER_HIGH_HZ)
    g.attrs['short_channel_max_dist_m'] = float(pre['short_threshold_used_m'])
    g.attrs['hemoglobin_units'] = 'uMolar*cm (device-specific published conversion)'
    g.attrs['baseline_tmin_sec'] = float(BASELINE_TMIN)
    g.attrs['baseline_tmax_sec'] = float(BASELINE_TMAX)
    g.attrs['epoch_tmin_sec'] = float(EPOCH_TMIN)
    g.attrs['epoch_tmax_sec'] = float(EPOCH_TMAX)
    g.create_dataset('hbo', data=np.asarray(epochs_hbo, np.float32), compression=H5_COMPRESSION, compression_opts=H5_COMPRESSION_LEVEL, shuffle=True)
    g.create_dataset('hbr', data=np.asarray(epochs_hbr, np.float32), compression=H5_COMPRESSION, compression_opts=H5_COMPRESSION_LEVEL, shuffle=True)
    g.create_dataset('times_sec', data=np.asarray(times, np.float32))
    str_dt = h5py.string_dtype(encoding='utf-8')
    g.create_dataset('channel_names', data=np.asarray(pre['pair_names'], dtype=object), dtype=str_dt)
    g.create_dataset('channel_distance_m', data=np.asarray(pre['pair_distance_m'], np.float32))
    g.create_dataset('channel_position_m', data=np.asarray(pre['pair_position_m'], np.float32))
    g.create_dataset('channel_valid_mask', data=np.asarray(pre['valid_pair_mask'], np.uint8))
    g.create_dataset('trial_onset_sec', data=np.asarray([x['event_onset_sec'] for x in trial_rows], np.float64))
    g.create_dataset('trial_duration_sec', data=np.asarray([x['event_duration_sec'] for x in trial_rows], np.float32))
    g.create_dataset('trial_label', data=np.asarray([x['trial_label'] for x in trial_rows], dtype=object), dtype=str_dt)
    g.create_dataset('trial_modality', data=np.asarray([x['modality'] for x in trial_rows], dtype=object), dtype=str_dt)
    g.create_dataset('event_row_json', data=np.asarray([x['event_row_json'] for x in trial_rows], dtype=object), dtype=str_dt)
def process_one_participant(
    bids_root: Path,
    participant: str,
    h5: h5py.File,
    force: bool=False
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if participant in h5 and (not force):
        raise RuntimeError('Internal resume should be handled before process_one_participant.')
    snirf, events_path = participant_files(bids_root, participant)
    events = pd.read_csv(events_path, sep='\t')
    stim = choose_stimulus_events(events)
    pre = preprocess_participant(snirf)
    rel_times = standard_epoch_times()
    epoch_hbo = []
    epoch_hbr = []
    trial_rows = []
    feature_rows = []
    skipped = []
    for j, row in stim.iterrows():
        row_dict = row.to_dict()
        onset = float(row['onset'])
        duration = float(row['_duration_num']) if np.isfinite(row['_duration_num']) else np.nan
        bad, reason = row_explicitly_bad(row_dict)
        if bad:
            skipped.append({'participant': participant, 'event_index': int(j), 'event_onset_sec': onset, 'reason': f'explicit_event_exclusion:{reason}'})
            continue
        ehbo = interpolate_epoch(pre['hbo'], pre['sfreq'], onset, rel_times)
        ehbr = interpolate_epoch(pre['hbr'], pre['sfreq'], onset, rel_times)
        if ehbo is None or ehbr is None:
            skipped.append({'participant': participant, 'event_index': int(j), 'event_onset_sec': onset, 'reason': 'epoch_outside_recording_bounds'})
            continue
        ehbo = baseline_correct(ehbo, rel_times)
        ehbr = baseline_correct(ehbr, rel_times)
        label = event_label(row_dict)
        modality = map_modality(label)
        trial_index = len(trial_rows)
        trial_meta = {
            'participant': participant,
            'trial_index': int(trial_index),
            'event_source_index': int(j),
            'event_onset_sec': onset,
            'event_duration_sec': duration,
            'trial_label': label,
            'modality': modality,
            'event_row_json': json.dumps({k: None if pd.isna(v) else float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else str(v) for k, v in row_dict.items() if k != '_duration_num'}, sort_keys=True)
        }
        epoch_hbo.append(ehbo)
        epoch_hbr.append(ehbr)
        trial_rows.append(trial_meta)
        for ch, ch_name in enumerate(pre['pair_names']):
            feature_rows.append(channel_feature_row(
                participant=participant,
                trial_index=trial_index,
                trial_label=label,
                modality=modality,
                onset_sec=onset,
                channel=ch_name,
                distance_m=float(pre['pair_distance_m'][ch]),
                hbo=ehbo[ch],
                hbr=ehbr[ch],
                times=rel_times
            ))
    if not trial_rows:
        raise RuntimeError(f'{participant}: no usable stimulus epochs.')
    epoch_hbo = np.stack(epoch_hbo, axis=0)
    epoch_hbr = np.stack(epoch_hbr, axis=0)
    write_participant_h5(h5, participant, pre, epoch_hbo, epoch_hbr, rel_times, trial_rows)
    modalities = pd.Series([x['modality'] for x in trial_rows]).value_counts().to_dict()
    labels = pd.Series([x['trial_label'] for x in trial_rows]).value_counts().to_dict()
    qc = {
        'participant': participant,
        'native_sfreq_hz': float(pre['sfreq']),
        'native_sfreq_diff_from_published_14hz': float(pre['sfreq'] - AUTHOR_SFREQ_HZ),
        'recording_duration_sec': float(pre['duration_sec']),
        'n_stimulus_events_detected': int(len(stim)),
        'n_epochs_saved': int(len(trial_rows)),
        'n_epochs_skipped': int(len(skipped)),
        'expected_40_stimuli_match': int(len(trial_rows) == EXPECTED_STIMULI_PER_PARTICIPANT),
        'n_long_pairs': int(len(pre['pair_names'])),
        'n_valid_long_pairs': int(np.sum(pre['valid_pair_mask'])),
        'valid_long_pair_fraction': float(np.mean(pre['valid_pair_mask'])),
        'n_short_pairs_detected': int(pre['preprocess_meta']['n_complete_short_pairs']),
        'short_threshold_used_m': float(pre['short_threshold_used_m']),
        'short_threshold_rule': str(pre['preprocess_meta'].get('short_threshold_rule', 'unknown')),
        'source_data_type': str(pre['source_type']),
        'modalities_json': json.dumps(modalities, sort_keys=True),
        'original_labels_json': json.dumps(labels, sort_keys=True),
        'unmapped_modality_trials': int(sum((x['modality'] == 'unmapped' for x in trial_rows))),
        'nan_fraction_saved_hbo': float(np.mean(~np.isfinite(epoch_hbo))),
        'nan_fraction_saved_hbr': float(np.mean(~np.isfinite(epoch_hbr)))
    }
    return (qc, trial_rows, feature_rows, skipped)
def self_tests() -> None:
    a780 = np.array([1.0, 2.0])
    a805 = np.array([0.5, 1.0])
    a830 = np.array([0.25, 0.5])
    hbo = HBO_COEFF[780] * a780 + HBO_COEFF[805] * a805 + HBO_COEFF[830] * a830
    hbr = HBR_COEFF[780] * a780 + HBR_COEFF[805] * a805 + HBR_COEFF[830] * a830
    assert np.isfinite(hbo).all()
    assert np.isfinite(hbr).all()
    import tempfile as _tempfile
    with _tempfile.TemporaryDirectory() as _td:
        _bad = Path(_td) / EXPECTED_ZIP_NAME
        _bad.write_text('<html>403 Forbidden</html>', encoding='utf-8')
        assert not _validate_balint_zip(_bad)
    e = pd.DataFrame({'onset': np.arange(42) * 30.0, 'duration': [0.0] + [13.0] * 40 + [0.0], 'trial_type': ['start'] + ['speech'] * 40 + ['stop']})
    s = choose_stimulus_events(e)
    assert len(s) == 40
    rng = np.random.default_rng(123)
    short = rng.normal(size=500)
    neural = rng.normal(scale=0.2, size=500)
    long = neural + 2.5 * short
    xc = short - np.mean(short)
    yc = long - np.mean(long)
    beta = float(np.dot(xc, yc) / np.dot(xc, xc))
    corrected = long - beta * xc
    assert abs(beta - 2.5) < 0.1
    assert abs(np.corrcoef(corrected, short)[0, 1]) < 1e-10
    tt = standard_epoch_times()
    assert abs(tt[0] - EPOCH_TMIN) < 1e-12
    assert abs(tt[-1] - EPOCH_TMAX) < 1e-12
    assert len(tt) == 291
    ep = np.ones((2, len(tt)), float) * 3.0
    ep[:, tt >= 0] += 2.0
    bc = baseline_correct(ep, tt)
    bm = (tt >= -5) & (tt <= 0)
    assert np.allclose(np.nanmean(bc[:, bm], axis=1), 0.0, atol=1e-12)
    print('Bálint preparation protected self-tests: PASS')
def parse_args(argv: Optional[Sequence[str]]=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        '--project-root',
        default=None,
        help='Output project root. Default: BALINT_ROOT env, otherwise /content/gdrive/MyDrive/HonorProject/HemoBoundary/Balint in Colab.'
    )
    p.add_argument('--group', choices=['nh', 'ci', 'all'], default='nh', help='Participant cohort. Default nh = normal hearing IDs beginning C.')
    p.add_argument('--max-participants', type=int, default=0, help='Optional smoke-test limit. 0 = all selected participants.')
    p.add_argument('--force', action='store_true', help='Reprocess participants already present in the HDF5 store.')
    p.add_argument('--skip-download', action='store_true', help='Require an already extracted BIDS dataset under raw/bids.')
    p.add_argument('--download-only', action='store_true', help='Download + extract only, then stop.')
    p.add_argument('--delete-zip-after-extract', action='store_true', help='Delete the ~720 MB source ZIP after a successful extraction.')
    p.add_argument('--self-test-only', action='store_true')
    args, unknown = p.parse_known_args(argv)
    if unknown:
        print('Ignoring notebook/runtime arguments:', unknown)
    return args
def main(argv: Optional[Sequence[str]]=None) -> None:
    args = parse_args(argv)
    self_tests()
    if args.self_test_only:
        return
    maybe_mount_drive()
    project_root = Path(args.project_root) if args.project_root else default_project_root()
    raw_download = project_root / 'raw' / 'download'
    raw_bids = project_root / 'raw' / 'bids'
    processed_dir = project_root / 'processed'
    features_dir = project_root / 'features'
    diagnostics_dir = project_root / 'diagnostics'
    checkpoint_dir = project_root / 'checkpoints'
    for d in [raw_download, raw_bids, processed_dir, features_dir, diagnostics_dir, checkpoint_dir]:
        d.mkdir(parents=True, exist_ok=True)
    if args.skip_download:
        bids_root = find_bids_root(raw_bids)
        if bids_root is None:
            raise FileNotFoundError(f'--skip-download was used but no BIDS root exists under {raw_bids}')
    else:
        archive = ensure_dataset_archive(raw_download)
        bids_root = ensure_extracted_bids(archive, raw_bids)
        if args.delete_zip_after_extract:
            archive.unlink(missing_ok=True)
            print('[cleanup] Deleted source ZIP:', archive)
    if args.download_only:
        print('Download/extraction complete:', bids_root)
        return
    participants_tsv = bids_root / 'participants.tsv'
    participants_df = pd.read_csv(participants_tsv, sep='\t')
    pids = selected_participants(bids_root, args.group)
    if args.max_participants > 0:
        pids = pids[:args.max_participants]
    if not pids:
        raise RuntimeError(f'No participants selected for group={args.group!r}.')
    if args.group == 'nh' and args.max_participants == 0 and (len(pids) != 26):
        warnings.warn(f'Expected 26 normal-hearing C* participants, found {len(pids)}.')
    print('=' * 132)
    print('BÁLINT EXTERNAL DATASET — DOWNLOAD + PREPROCESSING + FEATURE EXTRACTION')
    print('=' * 132)
    print('Version               :', VERSION)
    print('Dataset DOI           :', DRYAD_DOI)
    print('BIDS root             :', bids_root)
    print('Project root          :', project_root)
    print('Group                 :', args.group)
    print('Participants selected :', len(pids))
    print('Published sampling    :', '14 Hz')
    print('Preprocessing         :', 'SSR -> 0.01-0.10 Hz -> 3-wave Hb -> TDDR')
    print('Epoch                 :', '-5 to +24 s')
    print('Baseline              :', '-5 to 0 s')
    print('Output epoch sfreq    :', '10 Hz')
    print('Expected NH trials    :', '40 stimulus epochs / participant')
    pid_col = 'participant_id' if 'participant_id' in participants_df.columns else None
    if pid_col:
        keep_ids = {f'sub-{p}' for p in pids} | set(pids)
        sel_meta = participants_df[participants_df[pid_col].astype(str).isin(keep_ids)].copy()
    else:
        sel_meta = participants_df.copy()
    atomic_csv(sel_meta, processed_dir / 'participants_selected.tsv')
    manifest = {
        'version': VERSION,
        'dataset_doi': DRYAD_DOI,
        'dryad_file_id': DRYAD_SNIRF_FILE_ID,
        'group': args.group,
        'participants': pids,
        'acquisition': {'system': 'FOIRE-3000 Shimadzu', 'published_sfreq_hz': AUTHOR_SFREQ_HZ, 'wavelengths_nm': list(WAVELENGTHS), 'published_short_sds_channels': 3},
        'preprocessing': ['read SNIRF optical density; if cw amplitude, convert to optical density', f'nearest short-channel regression max_dist={SHORT_CHANNEL_MAX_DIST_M} m', f'MNE zero-phase FIR bandpass {FILTER_LOW_HZ}-{FILTER_HIGH_HZ} Hz', 'FOIRE-3000 published 3-wavelength HbO/HbR conversion', 'TDDR after Hb conversion', f'epoch {EPOCH_TMIN} to {EPOCH_TMAX} s', f'baseline {BASELINE_TMIN} to {BASELINE_TMAX} s', f'post-preprocessing interpolation to {OUTPUT_SFREQ_HZ} Hz'],
        'hbo_coefficients': HBO_COEFF,
        'hbr_coefficients': HBR_COEFF,
        'stimulus_selection': {'primary_rule': f'events.tsv duration between {STIM_DURATION_MIN} and {STIM_DURATION_MAX} seconds', 'expected_trials_per_participant': EXPECTED_STIMULI_PER_PARTICIPANT, 'wrong_comprehension_answer_auto_excluded': False, 'explicit_event_artifact_fields_auto_excluded': True},
        'feature_windows_sec': FEATURE_WINDOWS
    }
    manifest['identity'] = stable_hash(manifest)
    atomic_json(manifest, diagnostics_dir / 'preprocessing_manifest.json')
    h5_path = processed_dir / 'balint_epochs_10hz.h5'
    qc_rows: List[Dict[str, Any]] = []
    all_trial_rows: List[Dict[str, Any]] = []
    all_feature_rows: List[Dict[str, Any]] = []
    all_skipped: List[Dict[str, Any]] = []
    t0 = time.time()
    with h5py.File(h5_path, 'a') as h5:
        h5.attrs['version'] = VERSION
        h5.attrs['dataset_doi'] = DRYAD_DOI
        h5.attrs['group'] = args.group
        h5.attrs['output_sfreq_hz'] = OUTPUT_SFREQ_HZ
        for i, pid in enumerate(pids, 1):
            qc_cp = checkpoint_dir / f'{pid}_qc.csv'
            trial_cp = checkpoint_dir / f'{pid}_trials.csv.gz'
            feat_cp = checkpoint_dir / f'{pid}_channel_features.csv.gz'
            skip_cp = checkpoint_dir / f'{pid}_skipped.csv'
            resumable = pid in h5 and qc_cp.is_file() and trial_cp.is_file() and feat_cp.is_file() and skip_cp.is_file() and (not args.force)
            if resumable:
                print(f'[{i:02d}/{len(pids):02d}] {pid}: RESUME checkpoint')
                qc = pd.read_csv(qc_cp).iloc[0].to_dict()
                trials_df_pid = pd.read_csv(trial_cp)
                feats_df_pid = pd.read_csv(feat_cp)
                skipped_df_pid = pd.read_csv(skip_cp)
                qc_rows.append(qc)
                all_trial_rows.extend(trials_df_pid.to_dict('records'))
                all_feature_rows.extend(feats_df_pid.to_dict('records'))
                all_skipped.extend(skipped_df_pid.to_dict('records'))
                continue
            if pid in h5:
                del h5[pid]
            print(f'[{i:02d}/{len(pids):02d}] {pid}: preprocessing')
            qc, trials, feats, skipped = process_one_participant(bids_root=bids_root, participant=pid, h5=h5, force=True)
            h5.flush()
            qc_df_pid = pd.DataFrame([qc])
            trials_df_pid = pd.DataFrame(trials)
            feats_df_pid = pd.DataFrame(feats)
            skipped_df_pid = pd.DataFrame(skipped, columns=['participant', 'event_index', 'event_onset_sec', 'reason'])
            atomic_csv(qc_df_pid, qc_cp)
            atomic_csv(trials_df_pid, trial_cp)
            atomic_csv(feats_df_pid, feat_cp)
            atomic_csv(skipped_df_pid, skip_cp)
            qc_rows.append(qc)
            all_trial_rows.extend(trials)
            all_feature_rows.extend(feats)
            all_skipped.extend(skipped)
            print(f"    trials={qc['n_epochs_saved']} | long pairs={qc['n_valid_long_pairs']}/{qc['n_long_pairs']} | short pairs={qc['n_short_pairs_detected']} | sfreq={qc['native_sfreq_hz']:.4f} Hz | unmapped labels={qc['unmapped_modality_trials']}")
    qc_df = pd.DataFrame(qc_rows)
    trial_df = pd.DataFrame(all_trial_rows)
    channel_features = pd.DataFrame(all_feature_rows)
    skipped_df = pd.DataFrame(all_skipped)
    if len(qc_df):
        atomic_csv(qc_df, processed_dir / 'participant_qc.csv')
    if len(trial_df):
        atomic_csv(trial_df, processed_dir / 'trial_manifest.csv.gz')
    if len(channel_features):
        atomic_csv(channel_features, features_dir / 'balint_channel_features.csv.gz')
        trial_features = aggregate_trial_features(channel_features)
        atomic_csv(trial_features, features_dir / 'balint_trial_features.csv.gz')
    else:
        trial_features = pd.DataFrame()
    if len(skipped_df):
        atomic_csv(skipped_df, diagnostics_dir / 'skipped_trials.csv')
    else:
        atomic_csv(pd.DataFrame(columns=['participant', 'event_index', 'event_onset_sec', 'reason']), diagnostics_dir / 'skipped_trials.csv')
    if len(trial_df):
        event_dictionary = trial_df.groupby(['trial_label', 'modality'], dropna=False).agg(
            n_trials=('trial_index', 'size'),
            n_participants=('participant', 'nunique'),
            median_event_duration_sec=('event_duration_sec', 'median')
        ).reset_index().sort_values(['modality', 'trial_label'])
        atomic_csv(event_dictionary, processed_dir / 'event_dictionary.csv')
    else:
        event_dictionary = pd.DataFrame()
    if len(qc_df):
        median_trials = float(qc_df['n_epochs_saved'].median())
        median_short = float(qc_df['n_short_pairs_detected'].median())
        print('\nQuality summary')
        print('---------------')
        print('Processed participants        :', len(qc_df))
        print('Median saved stimulus epochs  :', median_trials)
        print('Participants with exactly 40  :', int(qc_df['expected_40_stimuli_match'].sum()))
        print('Median detected short pairs   :', median_short)
        print('Median valid long-pair fraction:', round(float(qc_df['valid_long_pair_fraction'].median()), 4))
        if abs(median_short - 3.0) > 0.5:
            warnings.warn('The median number of detected short pairs is not 3. Inspect participant_qc.csv before using the external validation.')
    if len(event_dictionary):
        print('\nDiscovered stimulus labels:')
        print(event_dictionary.to_string(index=False))
    guide = f'BÁLINT EXTERNAL PREPARATION RESULTS GUIDE\n================================================================================\n\nMAIN DATASET\n------------\nDryad DOI: {DRYAD_DOI}\nDefault cohort: normal-hearing IDs starting with C\nExpected cohort size: 26\n\nOUTPUT FOR EXTERNAL EVALUATION\n------------------------------------------\n{h5_path}\n\nHDF5 structure per participant:\n  hbo                    [trial, long_channel, time]\n  hbr                    [trial, long_channel, time]\n  times_sec              [-5 ... +24 s] at 10 Hz\n  channel_names\n  channel_distance_m\n  channel_position_m\n  channel_valid_mask\n  trial_onset_sec\n  trial_duration_sec\n  trial_label\n  trial_modality\n  event_row_json\n\nPREPROCESSING\n-------------\n1. SNIRF -> optical density if required by the raw data type\n2. nearest short-channel regression, max distance {SHORT_CHANNEL_MAX_DIST_M} m\n3. bandpass {FILTER_LOW_HZ}-{FILTER_HIGH_HZ} Hz\n4. published FOIRE-3000 780/805/830 nm conversion to HbO/HbR\n5. TDDR\n6. epoch {EPOCH_TMIN} to {EPOCH_TMAX} s\n7. baseline {BASELINE_TMIN} to {BASELINE_TMAX} s\n8. resample/interpolate only after preprocessing to {OUTPUT_SFREQ_HZ} Hz\n\nFEATURE TABLES\n--------------\nfeatures/balint_channel_features.csv.gz\n    One row per trial x long source-detector pair.\n\nfeatures/balint_trial_features.csv.gz\n    Participant/trial-level channel-median and channel-IQR features.\n\nDATA USE NOTES\n---------------------------\nUse balint_epochs_10hz.h5 for external evaluation.\nThe feature tables are descriptive; the analysis derives temporal\nevidence from epoch signals rather than relying on descriptive\nfeatures.\n\nEVENT LABELS\n------------\nCheck processed/event_dictionary.csv.\nThe script preserves every original BIDS event label. If a label could not be\nmapped by text to speech-in-quiet / speech-in-noise / audiovisual / visual speech,\nit is retained as modality="unmapped". Do not guess the mapping in a paper; inspect\nthe event dictionary first.\n\nRESUME NOTE\n-----------\nEvery participant is checkpointed independently (HDF5 + QC + trial manifest +\nchannel features). If Colab disconnects, rerun the same code: completed\nparticipants are loaded from checkpoints and only unfinished participants run.\n'
    (project_root / 'RESULTS_GUIDE.txt').write_text(guide, encoding='utf-8')
    package_root = project_root / 'BALINT_PREP_FEATURES_COMPLETE'
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True, exist_ok=True)
    for folder in ['processed', 'features', 'diagnostics']:
        src = project_root / folder
        if src.exists():
            shutil.copytree(src, package_root / folder)
    shutil.copy2(project_root / 'RESULTS_GUIDE.txt', package_root / 'RESULTS_GUIDE.txt')
    zip_base = project_root / 'BALINT_PREP_FEATURES_COMPLETE'
    zip_path = Path(str(zip_base) + '.zip')
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_base), 'zip', root_dir=package_root.parent, base_dir=package_root.name)
    print('\n' + '=' * 132)
    print('BÁLINT PREPARATION COMPLETE')
    print('=' * 132)
    print('Elapsed min     :', round((time.time() - t0) / 60.0, 2))
    print('Epoch HDF5      :', h5_path)
    print('Participant QC  :', processed_dir / 'participant_qc.csv')
    print('Event dictionary:', processed_dir / 'event_dictionary.csv')
    print('Channel features:', features_dir / 'balint_channel_features.csv.gz')
    print('Trial features  :', features_dir / 'balint_trial_features.csv.gz')
    print('Portable ZIP    :', zip_path)
if __name__ == '__main__':
    main()
