"""Khan finger-movement fNIRS preprocessing."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import warnings
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
VERSION = 'KHAN2026_FINGER_FNIRS_PREP_V7_SATORI_MULTI_REPRESENTATION_FIX_2026_09_10'
ARTICLE_DOI = '10.3389/fnhum.2026.1747655'
ARTICLE_URL = 'https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2026.1747655/full'
FIGSHARE_SHARE_TOKEN = os.environ.get('HEMOBOUNDARY_KHAN_FIGSHARE_TOKEN', '').strip()
FIGSHARE_FILE_ID = 59576999
FIGSHARE_SHARE_URL = f'https://figshare.com/s/{FIGSHARE_SHARE_TOKEN}?file={FIGSHARE_FILE_ID}'
FIGSHARE_DOWNLOAD_URLS = [f'https://figshare.com/ndownloader/files/{FIGSHARE_FILE_ID}?private_link={FIGSHARE_SHARE_TOKEN}', f'https://ndownloader.figshare.com/files/{FIGSHARE_FILE_ID}?private_link={FIGSHARE_SHARE_TOKEN}']
MIN_REASONABLE_DOWNLOAD_BYTES = 10000000
FIGSHARE_EXPECTED_FILENAME = 'FingerTappingDataset_Published2025.zip'
TARGET_SFREQ = 10.0
EPOCH_TMIN = -5.0
EPOCH_TMAX_EXCLUSIVE = 20.0
BASELINE_TMIN = -5.0
BASELINE_TMAX_EXCLUSIVE = 0.0
FILTER_L_FREQ = 0.01
FILTER_H_FREQ = 0.5
FILTER_ORDER = 4
PPF = 6.0
SD_DISTANCE_M = 0.03
SCI_THRESHOLD = 0.5
MIN_FINITE_FRACTION = 0.95
MIN_VALID_CHANNELS_WARN = 8
NEXT_TASK_GUARD_SEC = 0.15
TASK_CODES = (3, 4, 5, 6, 7)
REST_CODE = 2
EVENT_TO_LABEL = {3: 'thumb', 4: 'index', 5: 'middle', 6: 'ring', 7: 'little'}
LABEL_TO_EVENT = {
    'thumb': 3,
    'index': 4,
    'middle': 5,
    'ring': 6,
    'little': 7,
    'littlefinger': 7,
    'little_finger': 7,
    'pinky': 7,
    'pinkie': 7,
    'rest': 2,
    'baseline': 2
}
CANONICAL_CHANNELS: List[Tuple[str, int, int, str]] = [
    ('CH01', 1, 1, 'left'),
    ('CH02', 1, 3, 'left'),
    ('CH03', 2, 1, 'left'),
    ('CH04', 2, 2, 'left'),
    ('CH05', 2, 4, 'left'),
    ('CH06', 3, 1, 'left'),
    ('CH07', 3, 3, 'left'),
    ('CH08', 3, 4, 'left'),
    ('CH09', 3, 5, 'left'),
    ('CH10', 4, 2, 'left'),
    ('CH11', 4, 4, 'left'),
    ('CH12', 4, 6, 'left'),
    ('CH13', 5, 3, 'left'),
    ('CH14', 5, 5, 'left'),
    ('CH15', 5, 7, 'left'),
    ('CH16', 6, 4, 'left'),
    ('CH17', 6, 5, 'left'),
    ('CH18', 6, 6, 'left'),
    ('CH19', 6, 8, 'left'),
    ('CH20', 7, 5, 'left'),
    ('CH21', 7, 7, 'left'),
    ('CH22', 7, 8, 'left'),
    ('CH23', 8, 6, 'left'),
    ('CH24', 8, 8, 'left'),
    ('CH25', 9, 9, 'right'),
    ('CH26', 9, 10, 'right'),
    ('CH27', 9, 11, 'right'),
    ('CH28', 10, 10, 'right'),
    ('CH29', 10, 12, 'right'),
    ('CH30', 11, 9, 'right'),
    ('CH31', 11, 11, 'right'),
    ('CH32', 11, 13, 'right'),
    ('CH33', 12, 10, 'right'),
    ('CH34', 12, 11, 'right'),
    ('CH35', 12, 12, 'right'),
    ('CH36', 12, 14, 'right'),
    ('CH37', 13, 11, 'right'),
    ('CH38', 13, 13, 'right'),
    ('CH39', 13, 14, 'right'),
    ('CH40', 13, 15, 'right'),
    ('CH41', 14, 12, 'right'),
    ('CH42', 14, 14, 'right'),
    ('CH43', 14, 16, 'right'),
    ('CH44', 15, 13, 'right'),
    ('CH45', 15, 15, 'right'),
    ('CH46', 16, 14, 'right'),
    ('CH47', 16, 15, 'right'),
    ('CH48', 16, 16, 'right')
]
PAIR_TO_INDEX = {(s, d): i for i, (_, s, d, _) in enumerate(CANONICAL_CHANNELS)}
CHANNEL_NAMES = [ch for ch, _, _, _ in CANONICAL_CHANNELS]
HEMISPHERES = [hemi for _, _, _, hemi in CANONICAL_CHANNELS]
LEFT_MASK = np.array([h == 'left' for h in HEMISPHERES], bool)
RIGHT_MASK = ~LEFT_MASK

def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:16]
def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)
def atomic_csv(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    compression = 'gzip' if str(path).endswith('.gz') else None
    df.to_csv(tmp, index=False, compression=compression)
    tmp.replace(path)
def decode_any(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode('utf-8', errors='replace')
    if isinstance(x, np.bytes_):
        return bytes(x).decode('utf-8', errors='replace')
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return decode_any(x.reshape(-1)[0])
        return ' '.join((decode_any(v) for v in x.reshape(-1)))
    return str(x)
def robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))
def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
def safe_median(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan
def safe_max(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.nanmax(x)) if np.isfinite(x).any() else np.nan
def safe_min(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.nanmin(x)) if np.isfinite(x).any() else np.nan
def linear_slope(t: np.ndarray, y: np.ndarray) -> float:
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    if np.sum(ok) < 3:
        return np.nan
    x = t[ok]
    z = y[ok]
    x = x - np.mean(x)
    den = float(np.dot(x, x))
    if den <= 0:
        return np.nan
    return float(np.dot(x, z - np.mean(z)) / den)
def trapz_safe(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    if np.sum(ok) < 2:
        return np.nan
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(y[ok], x[ok]))
    return float(np.trapz(y[ok], x[ok]))
def mount_drive_if_available() -> None:
    candidates = [Path('/content/gdrive/MyDrive'), Path('/content/drive/MyDrive')]
    if any((p.exists() for p in candidates)):
        return
    try:
        from google.colab import drive
    except Exception:
        return
    print('[drive] Google Drive not mounted in this runtime. Mounting at /content/gdrive ...')
    drive.mount('/content/gdrive', force_remount=False)
def default_root() -> Path:
    mount_drive_if_available()
    if Path('/content/gdrive/MyDrive').exists():
        return Path('/content/gdrive/MyDrive/HonorProject/KHAN2026_FINGER_HEMOBOUNDARY_EXTERNAL_PREP')
    if Path('/content/drive/MyDrive').exists():
        return Path('/content/drive/MyDrive/HonorProject/KHAN2026_FINGER_HEMOBOUNDARY_EXTERNAL_PREP')
    return Path.cwd() / 'KHAN2026_FINGER_HEMOBOUNDARY_EXTERNAL_PREP'
def ensure_dependencies() -> None:
    """MNE 1.13.0 is pinned because it supports an explicit source-detector"""
    from packaging.version import Version
    need_mne = True
    try:
        import mne
        need_mne = Version(mne.__version__) < Version('1.13.0')
    except Exception:
        need_mne = True
    pkgs = []
    if need_mne:
        pkgs.append('mne==1.13.0')
    for mod, pkg in [
        ('h5py', 'h5py>=3.10'),
        ('pandas', 'pandas>=2.0'),
        ('requests', 'requests>=2.31'),
        ('tqdm', 'tqdm>=4.66'),
        ('packaging', 'packaging>=23'),
        ('curl_cffi', 'curl-cffi')
    ]:
        try:
            __import__(mod)
        except Exception:
            pkgs.append(pkg)
    if pkgs:
        print('[deps] Installing:', ' '.join(pkgs))
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *pkgs])
    import mne
    print('[deps] MNE:', mne.__version__)
def content_disposition_filename(header: Optional[str]) -> Optional[str]:
    if not header:
        return None
    m = re.search("filename\\*=UTF-8''([^;]+)", header, flags=re.I)
    if m:
        from urllib.parse import unquote
        return Path(unquote(m.group(1).strip())).name
    m = re.search('filename="?([^";]+)"?', header, flags=re.I)
    if m:
        return Path(m.group(1).strip()).name
    return None
def looks_like_html(path: Path) -> bool:
    try:
        b = path.read_bytes()[:4096].lower()
    except Exception:
        return False
    return b'<html' in b or b'<!doctype html' in b or b'insufficientpermissions' in b
def _stream_signed_url_to_file(signed_url: str, final: Path) -> Path:
    """
    Download the short-lived signed S3 URL returned by Figshare.
    This request no longer passes through Figshare/Cloudflare.
    """
    import requests
    from tqdm.auto import tqdm
    tmp = final.with_suffix(final.suffix + '.partial')
    if tmp.exists():
        tmp.unlink()
    with requests.get(signed_url, stream=True, allow_redirects=True, timeout=(30, 300), headers={'User-Agent': 'Mozilla/5.0', 'Accept': '*/*'}) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0) or 0)
        with open(tmp, 'wb') as f, tqdm(total=total or None, unit='B', unit_scale=True, desc='Dataset') as pbar:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
    if not tmp.is_file() or tmp.stat().st_size < MIN_REASONABLE_DOWNLOAD_BYTES or looks_like_html(tmp):
        size = tmp.stat().st_size if tmp.exists() else 0
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f'Signed S3 download was incomplete or invalid ({size} bytes).')
    if final.exists():
        final.unlink()
    tmp.replace(final)
    return final
def _resolve_figshare_redirect_curl_cffi(url: str) -> str:
    """
    Resolve Figshare's private-link downloader to its temporary signed S3 URL
    without downloading the whole archive through the Figshare frontend.
    """
    from curl_cffi import requests as cf_requests
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Referer': FIGSHARE_SHARE_URL,
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-site',
        'Upgrade-Insecure-Requests': '1'
    }
    profiles = ['chrome', 'chrome136', 'chrome131', 'safari']
    errors = []
    for profile in profiles:
        try:
            print(f'[download] Resolving Figshare redirect with browser profile: {profile}')
            r = cf_requests.get(url, headers=headers, impersonate=profile, allow_redirects=False, timeout=60)
            status = int(r.status_code)
            location = r.headers.get('location')
            if status in {301, 302, 303, 307, 308} and location:
                if 'amazonaws.com' in location or 'figshare' in location:
                    return str(location)
            if status == 200 and str(r.url).startswith('http') and ('amazonaws.com' in str(r.url)):
                return str(r.url)
            errors.append(f"{profile}: status={status}, location={location!r}, content-type={r.headers.get('content-type')!r}")
        except Exception as exc:
            errors.append(f'{profile}: {repr(exc)}')
    raise RuntimeError('curl-cffi could not resolve the Figshare redirect:\n' + '\n'.join(errors))
def _download_with_system_curl(url: str, final: Path) -> Optional[Path]:
    """
    Secondary fallback. curl often behaves differently from wget on Figshare.
    """
    curl = shutil.which('curl')
    if not curl:
        return None
    tmp = final.with_suffix(final.suffix + '.curl.partial')
    if tmp.exists():
        tmp.unlink()
    print('[download] Trying system curl fallback...')
    cmd = [
        curl,
        '-L',
        '--fail',
        '--retry',
        '4',
        '--retry-all-errors',
        '--connect-timeout',
        '30',
        '--max-time',
        '0',
        '--http1.1',
        '-A',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/152.0.0.0 Safari/537.36',
        '-e',
        FIGSHARE_SHARE_URL,
        '-o',
        str(tmp),
        url
    ]
    proc = subprocess.run(cmd, text=True)
    if proc.returncode == 0 and tmp.is_file() and (tmp.stat().st_size >= MIN_REASONABLE_DOWNLOAD_BYTES) and (not looks_like_html(tmp)):
        if final.exists():
            final.unlink()
        tmp.replace(final)
        return final
    if tmp.exists():
        tmp.unlink()
    return None
def download_figshare(download_dir: Path, force: bool=False) -> Path:
    """Colab-safe Figshare private-link downloader."""
    download_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        [p for p in download_dir.iterdir() if p.is_file() and p.stat().st_size >= MIN_REASONABLE_DOWNLOAD_BYTES and (not looks_like_html(p))],
        key=lambda p: p.stat().st_size,
        reverse=True
    )
    if existing and (not force):
        print('[download] Reusing:', existing[0], f'({existing[0].stat().st_size / 1024 ** 2:.1f} MB)')
        return existing[0]
    final = download_dir / FIGSHARE_EXPECTED_FILENAME
    direct = f'https://ndownloader.figshare.com/files/{FIGSHARE_FILE_ID}?private_link={FIGSHARE_SHARE_TOKEN}'
    errors = []
    try:
        signed = _resolve_figshare_redirect_curl_cffi(direct)
        print('[download] Figshare redirect resolved.')
        if 'amazonaws.com' in signed:
            print('[download] Downloading from temporary signed S3 URL...')
            out = _stream_signed_url_to_file(signed, final)
            print('[download] Saved:', out, f'({out.stat().st_size / 1024 ** 2:.2f} MB)')
            return out
        maybe = _download_with_system_curl(signed, final)
        if maybe is not None:
            print('[download] Saved:', maybe, f'({maybe.stat().st_size / 1024 ** 2:.2f} MB)')
            return maybe
    except Exception as exc:
        errors.append('curl-cffi/S3 route: ' + repr(exc))
    try:
        maybe = _download_with_system_curl(direct, final)
        if maybe is not None:
            print('[download] Saved:', maybe, f'({maybe.stat().st_size / 1024 ** 2:.2f} MB)')
            return maybe
    except Exception as exc:
        errors.append('system curl route: ' + repr(exc))
    try:
        import requests
        from tqdm.auto import tqdm
        print('[download] Trying ordinary requests as final automatic fallback...')
        tmp = final.with_suffix(final.suffix + '.requests.partial')
        if tmp.exists():
            tmp.unlink()
        with requests.get(
            direct,
            stream=True,
            allow_redirects=True,
            timeout=(30, 300),
            headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/152.0.0.0 Safari/537.36', 'Referer': FIGSHARE_SHARE_URL, 'Accept': '*/*'}
        ) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0) or 0)
            with open(tmp, 'wb') as f, tqdm(total=total or None, unit='B', unit_scale=True, desc='Figshare') as pbar:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))
        if tmp.is_file() and tmp.stat().st_size >= MIN_REASONABLE_DOWNLOAD_BYTES and (not looks_like_html(tmp)):
            if final.exists():
                final.unlink()
            tmp.replace(final)
            print('[download] Saved:', final, f'({final.stat().st_size / 1024 ** 2:.2f} MB)')
            return final
        if tmp.exists():
            tmp.unlink()
    except Exception as exc:
        errors.append('ordinary requests route: ' + repr(exc))
    raise RuntimeError(f'\nAutomatic Figshare download still failed after the Colab-safe browser-impersonation route.\n\nThe dataset link itself is valid, but Figshare can block cloud datacenter IPs intermittently.\n\nOfficial share page:\n{FIGSHARE_SHARE_URL}\n\nExpected archive name:\n{FIGSHARE_EXPECTED_FILENAME}\n\nIf this rare fallback is needed, download the archive once in your normal browser and upload/copy it to:\n{download_dir}\n\nThen rerun:\n%run /content/KHAN2026_FINGER_FNIRS_DOWNLOAD_PREPROCESS_FEATURES_COMPLETE_V2.py --skip-download\n\nAutomatic route errors:\n' + '\n'.join(errors))
def install_7zip_if_colab() -> None:
    if shutil.which('7z'):
        return
    if Path('/content').exists():
        print('[extract] Installing p7zip-full for archive fallback...')
        subprocess.run(['apt-get', 'update', '-qq'], check=False)
        subprocess.run(['apt-get', 'install', '-y', '-qq', 'p7zip-full'], check=False)
def extract_archive(archive: Path, extract_dir: Path, force: bool=False) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    existing_snirf = list(extract_dir.rglob('*.snirf')) + list(extract_dir.rglob('*.SNIRF'))
    if existing_snirf and (not force):
        print('[extract] Reusing extracted data:', len(existing_snirf), 'SNIRF files')
        return
    if force and extract_dir.exists():
        shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
    print('[extract] Extracting:', archive)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive, 'r') as z:
            z.extractall(extract_dir)
        return
    try:
        if tarfile.is_tarfile(archive):
            with tarfile.open(archive, 'r:*') as t:
                t.extractall(extract_dir)
            return
    except Exception:
        pass
    install_7zip_if_colab()
    seven = shutil.which('7z')
    if seven:
        proc = subprocess.run([seven, 'x', '-y', f'-o{extract_dir}', str(archive)], text=True)
        if proc.returncode == 0:
            return
    try:
        import h5py
        with h5py.File(archive, 'r') as h5:
            if any((str(k).lower().startswith('nirs') for k in h5.keys())):
                dst = extract_dir / 'downloaded_dataset.snirf'
                shutil.copy2(archive, dst)
                return
    except Exception:
        pass
    raise RuntimeError(f'Downloaded file is not a recognized ZIP/TAR/7z/RAR/SNIRF archive. File: {archive}')
def find_snirf_files(extract_dir: Path) -> Tuple[List[Path], List[Path]]:
    all_files = sorted(set(list(extract_dir.rglob('*.snirf')) + list(extract_dir.rglob('*.SNIRF'))))
    if not all_files:
        raise FileNotFoundError(f'No SNIRF files found under {extract_dir}')
    raw = []
    filtered = []
    for p in all_files:
        low = p.name.lower()
        if 'cc_filtered' in low or 'filtered' in low or 'processed' in low:
            filtered.append(p)
        else:
            raw.append(p)
    trim_raw = [p for p in raw if re.search('S\\d{2}R\\d+.*TRIM', p.stem, flags=re.I)]
    if trim_raw:
        raw = trim_raw
    return (sorted(raw), sorted(filtered))
def parse_subject_run(path: Path) -> Tuple[str, str]:
    text = path.stem
    m = re.search('(S\\d{2})R(\\d+)', text, flags=re.I)
    if m:
        return (m.group(1).upper(), f'R{int(m.group(2))}')
    text2 = path.parent.name + '_' + path.name
    m = re.search('(S\\d{2}).*?R(\\d+)', text2, flags=re.I)
    if m:
        return (m.group(1).upper(), f'R{int(m.group(2))}')
    raise ValueError(f'Could not parse subject/run from {path}')
def matching_author_filtered(raw_path: Path, filtered_files: Sequence[Path]) -> Optional[Path]:
    subj, run = parse_subject_run(raw_path)
    candidates = []
    for p in filtered_files:
        try:
            s, r = parse_subject_run(p)
        except Exception:
            continue
        if s == subj and r == run:
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: ('cc_filtered' not in p.name.lower(), len(p.name)))
    return candidates[0]
def event_code_from_text(text: Any) -> Optional[int]:
    s = decode_any(text).strip().lower()
    compact = re.sub('[^a-z0-9]+', '', s)
    for name, code in LABEL_TO_EVENT.items():
        key = re.sub('[^a-z0-9]+', '', name.lower())
        if key and key in compact:
            return int(code)
    nums = re.findall('(?<!\\d)([0-9]+)(?!\\d)', s)
    for token in reversed(nums):
        val = int(token)
        if 1 <= val <= 8:
            return val
    return None
def read_events_from_mne_annotations(raw) -> List[Dict[str, Any]]:
    out = []
    ann = raw.annotations
    for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
        code = event_code_from_text(desc)
        if code is None:
            continue
        out.append({'onset_sec': float(onset), 'duration_sec': float(duration), 'description': str(desc), 'event_code': int(code), 'event_source': 'mne_annotation'})
    out.sort(key=lambda x: x['onset_sec'])
    return out
def read_events_direct_h5(snirf_path: Path) -> List[Dict[str, Any]]:
    import h5py
    out = []
    with h5py.File(snirf_path, 'r') as h5:
        nirs_groups = [h5[k] for k in h5.keys() if str(k).lower().startswith('nirs')]
        for nirs in nirs_groups:
            stim_keys = sorted([k for k in nirs.keys() if str(k).lower().startswith('stim')])
            for sk in stim_keys:
                g = nirs[sk]
                if 'name' in g:
                    name = decode_any(g['name'][()])
                else:
                    name = str(sk)
                code_from_name = event_code_from_text(name)
                if 'data' not in g:
                    continue
                data = np.asarray(g['data'][()])
                if data.ndim == 1:
                    data = data[None, :]
                for row in data:
                    if len(row) < 1:
                        continue
                    onset = float(row[0])
                    duration = float(row[1]) if len(row) >= 2 and np.isfinite(row[1]) else 0.0
                    code = code_from_name
                    if code is None and len(row) >= 3 and np.isfinite(row[2]):
                        maybe = int(round(float(row[2])))
                        if 1 <= maybe <= 8:
                            code = maybe
                    if code is None:
                        continue
                    out.append({'onset_sec': onset, 'duration_sec': duration, 'description': name, 'event_code': int(code), 'event_source': 'snirf_h5_stim'})
    uniq = {}
    for e in out:
        key = (round(e['onset_sec'], 6), int(e['event_code']))
        if key not in uniq:
            uniq[key] = e
    return sorted(uniq.values(), key=lambda x: x['onset_sec'])
def get_events(raw, snirf_path: Path) -> List[Dict[str, Any]]:
    ev = read_events_from_mne_annotations(raw)
    n_task = sum((e['event_code'] in TASK_CODES for e in ev))
    if n_task > 0:
        return ev
    ev2 = read_events_direct_h5(snirf_path)
    n_task2 = sum((e['event_code'] in TASK_CODES for e in ev2))
    if n_task2 > 0:
        return ev2
    descriptions = sorted(set((str(x) for x in raw.annotations.description)))
    raise RuntimeError(f'No finger-task events (3..7) could be parsed from {snirf_path}. MNE descriptions were: {descriptions[:20]}')
def infer_task_duration(events: Sequence[Mapping[str, Any]], idx: int) -> float:
    e = events[idx]
    d = float(e.get('duration_sec', 0.0) or 0.0)
    if 5.0 <= d <= 20.0:
        return d
    onset = float(e['onset_sec'])
    for j in range(idx + 1, len(events)):
        nxt = events[j]
        dt = float(nxt['onset_sec']) - onset
        if dt <= 0:
            continue
        if dt > 20.0:
            break
        if int(nxt['event_code']) == REST_CODE:
            if 5.0 <= dt <= 20.0:
                return dt
    return 10.0
def next_task_onset(events: Sequence[Mapping[str, Any]], idx: int) -> Optional[float]:
    onset = float(events[idx]['onset_sec'])
    for j in range(idx + 1, len(events)):
        e = events[j]
        if int(e['event_code']) in TASK_CODES:
            t = float(e['onset_sec'])
            if t > onset:
                return t
    return None
PAIR_RE = re.compile('S0*(\\d+)[^0-9]+D0*(\\d+)', flags=re.I)

def source_detector_pair_from_name(name: str) -> Optional[Tuple[int, int]]:
    m = PAIR_RE.search(str(name))
    if not m:
        m = re.search('source\\s*0*(\\d+).*detector\\s*0*(\\d+)', str(name), flags=re.I)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))
@dataclass
class RunProcessed:
    subject: str
    run_id: str
    source_file: str
    processing_mode: str
    sfreq_original: float
    sfreq_final: float
    data_times: np.ndarray
    hbo48: np.ndarray
    hbr48: np.ndarray
    pair_sci48: np.ndarray
    pair_present48: np.ndarray
    events: List[Dict[str, Any]]
    mne_channel_types: str
def _h5_scalar_int(g, key: str, default: Optional[int]=None) -> Optional[int]:
    if key not in g:
        return default
    a = np.asarray(g[key][()]).reshape(-1)
    if len(a) == 0:
        return default
    return int(round(float(a[0])))
def _h5_scalar_text(g, key: str, default: str='') -> str:
    if key not in g:
        return default
    return decode_any(g[key][()]).strip()
def _natural_key(text: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split('(\\d+)', str(text))]
def _probe_positions(probe, base: str) -> Optional[np.ndarray]:
    for suffix in ('Pos3D', 'Pos2D'):
        key = base + suffix
        if key not in probe:
            continue
        arr = np.asarray(probe[key][()], float)
        if arr.ndim != 2:
            continue
        if arr.shape[1] == 2:
            arr = np.c_[arr, np.zeros(len(arr), float)]
        if arr.shape[1] >= 3:
            return arr[:, :3]
    return None
def read_satori_fraw_as_mne_cw_amplitude(path: Path):
    """Dataset-specific repair for Khan et al. 2026 Satori-converted raw SNIRF."""
    import h5py
    import mne
    canonical_pairs = [(int(s), int(d)) for _, s, d, _ in CANONICAL_CHANNELS]
    canonical_pair_set = set(canonical_pairs)
    with h5py.File(path, 'r') as h5:
        if 'nirs' not in h5:
            raise RuntimeError(f'{path.name}: missing /nirs group.')
        nirs = h5['nirs']
        if 'data1' not in nirs:
            raise RuntimeError(f'{path.name}: missing /nirs/data1.')
        data_g = nirs['data1']
        if 'dataTimeSeries' not in data_g:
            raise RuntimeError(f'{path.name}: missing dataTimeSeries.')
        if 'probe' not in nirs:
            raise RuntimeError(f'{path.name}: missing /nirs/probe.')
        probe = nirs['probe']
        if 'wavelengths' not in probe:
            raise RuntimeError(f'{path.name}: missing probe/wavelengths.')
        wavelengths = np.asarray(probe['wavelengths'][()], float).reshape(-1)
        if len(wavelengths) < 2:
            raise RuntimeError(f'{path.name}: fewer than two wavelengths.')
        wl_sorted = np.sort(wavelengths[np.isfinite(wavelengths)])
        if len(wl_sorted) < 2:
            raise RuntimeError(f'{path.name}: invalid probe wavelength array.')
        documented = np.asarray([760.0, 850.0], float)
        if not all((np.min(np.abs(wl_sorted - target)) <= 5.0 for target in documented)):
            raise RuntimeError(f'{path.name}: probe wavelengths {wl_sorted.tolist()} do not match the documented ~760/850-nm configuration.')
        if 'time' not in data_g:
            raise RuntimeError(f'{path.name}: missing /nirs/data1/time.')
        times = np.asarray(data_g['time'][()], float).reshape(-1)
        if len(times) < 3:
            raise RuntimeError(f'{path.name}: invalid time vector.')
        dt = np.diff(times)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt) == 0:
            raise RuntimeError(f'{path.name}: could not infer sampling rate.')
        sfreq = float(1.0 / np.median(dt))
        ml_names = sorted([k for k in data_g.keys() if str(k).lower().startswith('measurementlist') and str(k).lower() != 'measurementlists'], key=_natural_key)
        records = []
        skipped_invalid_wavelength = 0
        skipped_noncanonical_pair = 0
        skipped_incomplete_metadata = 0
        skipped_dod_representation = 0
        skipped_concentration_representation = 0
        skipped_other_representation = 0
        if ml_names:
            total_measurements = len(ml_names)
            for col, key in enumerate(ml_names):
                g = data_g[key]
                src = _h5_scalar_int(g, 'sourceIndex')
                det = _h5_scalar_int(g, 'detectorIndex')
                widx = _h5_scalar_int(g, 'wavelengthIndex')
                dtype = _h5_scalar_int(g, 'dataType')
                label = _h5_scalar_text(g, 'dataTypeLabel', '')
                if src is None or det is None or widx is None:
                    skipped_incomplete_metadata += 1
                    continue
                label_norm = str(label).strip().lower()
                if label_norm in {'dod', 'd_od', 'optical density', 'optical_density', 'od'}:
                    skipped_dod_representation += 1
                    continue
                if label_norm in {'hbo', 'hbr', 'hb', 'oxy', 'deoxy'}:
                    skipped_concentration_representation += 1
                    continue
                rawish = label_norm in {'', 'raw', 'fraw', 'cw', 'amplitude', 'raw amplitude'} or dtype == 1
                if not rawish:
                    skipped_other_representation += 1
                    continue
                if not 1 <= int(widx) <= len(wavelengths):
                    skipped_invalid_wavelength += 1
                    continue
                pair = (int(src), int(det))
                if pair not in canonical_pair_set:
                    skipped_noncanonical_pair += 1
                    continue
                records.append({
                    'original_col': int(col),
                    'measurement_list': str(key),
                    'source_index': int(src),
                    'detector_index': int(det),
                    'wavelength_index': int(widx),
                    'wavelength_nm': float(wavelengths[int(widx) - 1]),
                    'data_type': dtype,
                    'data_type_label': label
                })
        elif 'measurementLists' in data_g:
            mg = data_g['measurementLists']
            required = ['sourceIndex', 'detectorIndex', 'wavelengthIndex']
            for key in required:
                if key not in mg:
                    raise RuntimeError(f'{path.name}: compact measurementLists missing {key}.')
            srcs = np.asarray(mg['sourceIndex'][()]).reshape(-1)
            dets = np.asarray(mg['detectorIndex'][()]).reshape(-1)
            widxs = np.asarray(mg['wavelengthIndex'][()]).reshape(-1)
            total_measurements = len(srcs)
            if not (len(dets) == total_measurements and len(widxs) == total_measurements):
                raise RuntimeError(f'{path.name}: compact measurementLists length mismatch.')
            dtypes = np.asarray(mg['dataType'][()]).reshape(-1) if 'dataType' in mg else np.full(total_measurements, np.nan)
            labels = np.asarray(mg['dataTypeLabel'][()]).reshape(-1) if 'dataTypeLabel' in mg else np.asarray([b''] * total_measurements)
            for col in range(total_measurements):
                try:
                    src = int(srcs[col])
                    det = int(dets[col])
                    widx = int(widxs[col])
                except Exception:
                    skipped_incomplete_metadata += 1
                    continue
                label_norm = decode_any(labels[col]).strip().lower() if col < len(labels) else ''
                dtype_val = int(dtypes[col]) if col < len(dtypes) and np.isfinite(dtypes[col]) else None
                if label_norm in {'dod', 'd_od', 'optical density', 'optical_density', 'od'}:
                    skipped_dod_representation += 1
                    continue
                if label_norm in {'hbo', 'hbr', 'hb', 'oxy', 'deoxy'}:
                    skipped_concentration_representation += 1
                    continue
                rawish = label_norm in {'', 'raw', 'fraw', 'cw', 'amplitude', 'raw amplitude'} or dtype_val == 1
                if not rawish:
                    skipped_other_representation += 1
                    continue
                if not 1 <= widx <= len(wavelengths):
                    skipped_invalid_wavelength += 1
                    continue
                pair = (src, det)
                if pair not in canonical_pair_set:
                    skipped_noncanonical_pair += 1
                    continue
                records.append({
                    'original_col': int(col),
                    'measurement_list': f'measurementLists[{col}]',
                    'source_index': src,
                    'detector_index': det,
                    'wavelength_index': widx,
                    'wavelength_nm': float(wavelengths[widx - 1]),
                    'data_type': dtype_val,
                    'data_type_label': label_norm
                })
        else:
            raise RuntimeError(f'{path.name}: no measurementList metadata found.')
        if len(records) == 0:
            raise RuntimeError(f'{path.name}: zero published optical channels found after filtering auxiliary measurement entries.')
        labels_lower = {str(r['data_type_label']).strip().lower() for r in records if str(r['data_type_label']).strip()}
        valid_labels = {'', 'raw', 'fraw', 'cw', 'amplitude', 'raw amplitude'}
        unknown_labels = labels_lower - valid_labels
        if unknown_labels:
            raise RuntimeError(f'{path.name}: non-raw representation leaked into the retained optical set: {sorted(unknown_labels)}')
        dtype_values = {r['data_type'] for r in records if r['data_type'] is not None}
        allowed_types = {1, 99999}
        if dtype_values and (not dtype_values.issubset(allowed_types)):
            raise RuntimeError(f'{path.name}: unexpected retained optical SNIRF dataType values: {sorted(dtype_values)}')
        data = np.asarray(data_g['dataTimeSeries'][()], float)
        if data.ndim != 2:
            raise RuntimeError(f'{path.name}: dataTimeSeries must be 2D, got {data.shape}.')
        if data.shape[0] == len(times) and data.shape[1] == total_measurements:
            data_tc = data
        elif data.shape[1] == len(times) and data.shape[0] == total_measurements:
            data_tc = data.T
        else:
            raise RuntimeError(f'{path.name}: dataTimeSeries shape {data.shape} does not match time={len(times)} and total measurement lists={total_measurements}.')
        by_pair: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for r in records:
            pair = (int(r['source_index']), int(r['detector_index']))
            by_pair.setdefault(pair, []).append(r)
        ordered = []
        missing_pairs = []
        duplicate_entries_skipped = 0
        for pair in canonical_pairs:
            rr = by_pair.get(pair, [])
            if not rr:
                missing_pairs.append((pair, 'no measurements'))
                continue
            by_wavelength: Dict[int, List[Dict[str, Any]]] = {}
            for z in rr:
                by_wavelength.setdefault(int(z['wavelength_index']), []).append(z)
            expected_indices = []
            for target_nm in (760.0, 850.0):
                idx0 = int(np.argmin(np.abs(wavelengths - target_nm)))
                if abs(float(wavelengths[idx0]) - target_nm) > 5.0:
                    raise RuntimeError(f'{path.name}: could not identify the documented {target_nm:g}-nm wavelength.')
                expected_indices.append(idx0 + 1)
            pair_selected = []
            for widx in expected_indices:
                candidates = sorted(by_wavelength.get(int(widx), []), key=lambda z: int(z['original_col']))
                if not candidates:
                    pair_selected = []
                    break
                pair_selected.append(candidates[0])
                duplicate_entries_skipped += max(0, len(candidates) - 1)
            if len(pair_selected) != 2:
                available = sorted(((int(k), float(wavelengths[int(k) - 1])) for k in by_wavelength if 1 <= int(k) <= len(wavelengths)))
                missing_pairs.append((pair, available))
                continue
            ordered.extend(pair_selected)
        if missing_pairs:
            raise RuntimeError(f'{path.name}: the published 48-channel montage is incomplete after filtering Satori auxiliary entries. First missing: {missing_pairs[:8]}')
        if len(ordered) != 96:
            raise RuntimeError(f'{path.name}: expected exactly 96 optical wavelength measurements (48 pairs x 2 wavelengths), got {len(ordered)}.')
        selected_pairs = {(int(z['source_index']), int(z['detector_index'])) for z in ordered}
        if selected_pairs != canonical_pair_set:
            raise RuntimeError(f'{path.name}: retained source-detector montage does not exactly match the 48 published channels.')
        print(
            '  [reader] measurement lists:',
            total_measurements,
            '| optical kept:',
            len(ordered),
            '| dOD skipped:',
            skipped_dod_representation,
            '| concentration skipped:',
            skipped_concentration_representation,
            '| other representation skipped:',
            skipped_other_representation,
            '| invalid-wavelength skipped:',
            skipped_invalid_wavelength,
            '| non-published-pair skipped:',
            skipped_noncanonical_pair,
            '| incomplete skipped:',
            skipped_incomplete_metadata,
            '| duplicate raw optical skipped:',
            duplicate_entries_skipped
        )
        ch_names = []
        ch_data = []
        for r in ordered:
            src = int(r['source_index'])
            det = int(r['detector_index'])
            wl = float(r['wavelength_nm'])
            wl_name = str(int(round(wl))) if abs(wl - round(wl)) < 1e-06 else f'{wl:g}'
            ch_names.append(f'S{src}_D{det} {wl_name}')
            ch_data.append(data_tc[:, int(r['original_col'])])
        arr = np.asarray(ch_data, float)
        if arr.shape != (96, len(times)):
            raise RuntimeError(f'{path.name}: reconstructed optical array has unexpected shape {arr.shape}; expected (96, {len(times)}).')
        if not np.isfinite(arr).any():
            raise RuntimeError(f'{path.name}: retained optical data are all non-finite.')
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['fnirs_cw_amplitude'] * len(ch_names))
        source_pos = _probe_positions(probe, 'source')
        detector_pos = _probe_positions(probe, 'detector')
        for i, r in enumerate(ordered):
            src = int(r['source_index'])
            det = int(r['detector_index'])
            wl = float(r['wavelength_nm'])
            info['chs'][i]['loc'][9] = wl
            if source_pos is not None and 1 <= src <= len(source_pos):
                info['chs'][i]['loc'][3:6] = source_pos[src - 1]
            if detector_pos is not None and 1 <= det <= len(detector_pos):
                info['chs'][i]['loc'][6:9] = detector_pos[det - 1]
            if source_pos is not None and detector_pos is not None and (1 <= src <= len(source_pos)) and (1 <= det <= len(detector_pos)):
                info['chs'][i]['loc'][0:3] = (source_pos[src - 1] + detector_pos[det - 1]) / 2.0
        raw = mne.io.RawArray(arr, info, first_samp=0, verbose='ERROR')
    events = read_events_direct_h5(path)
    if events:
        raw.set_annotations(mne.Annotations(
            onset=[float(e['onset_sec']) for e in events],
            duration=[float(e.get('duration_sec', 0.0)) for e in events],
            description=[str(int(e['event_code'])) for e in events]
        ), emit_warning=False)
    raw.info['description'] = 'Khan2026 Satori raw repair: only dataTypeLabel=raw/CW-amplitude for the publication-defined 48-channel montage x 760/850 nm is retained; duplicated dOD/concentration and auxiliary entries ignored.'
    return raw
def snirf_is_satori_raw_labeled(path: Path) -> bool:
    """
    Detect the Khan/Satori malformed raw encoding before asking MNE to parse
    the file. This avoids MNE failing on any other header field first.
    """
    import h5py
    try:
        with h5py.File(path, 'r') as h5:
            if 'nirs' not in h5 or 'data1' not in h5['nirs']:
                return False
            data_g = h5['nirs']['data1']
            ml_names = sorted([k for k in data_g.keys() if str(k).lower().startswith('measurementlist') and str(k).lower() != 'measurementlists'], key=_natural_key)
            labels = []
            if ml_names:
                for key in ml_names[:min(8, len(ml_names))]:
                    g = data_g[key]
                    if 'dataTypeLabel' in g:
                        labels.append(decode_any(g['dataTypeLabel'][()]).strip().lower())
            elif 'measurementLists' in data_g:
                mg = data_g['measurementLists']
                if 'dataTypeLabel' in mg:
                    vals = np.asarray(mg['dataTypeLabel'][()]).reshape(-1)
                    labels.extend((decode_any(x).strip().lower() for x in vals[:min(8, len(vals))]))
            labels = [x for x in labels if x]
            if not labels:
                return False
            return all((x in {'raw', 'fraw', 'cw', 'amplitude', 'raw amplitude'} for x in labels))
    except Exception:
        return False
def raw_intensity_preprocess(path: Path) -> RunProcessed:
    import mne
    from mne.preprocessing.nirs import optical_density, scalp_coupling_index, temporal_derivative_distribution_repair, beer_lambert_law
    subject, run_id = parse_subject_run(path)
    reader_mode = 'MNE_STANDARD_SNIRF'
    if snirf_is_satori_raw_labeled(path):
        print("  [reader] Satori dataTypeLabel='raw' detected by HDF5 preflight; using dataset-specific CW-amplitude reconstruction.")
        raw = read_satori_fraw_as_mne_cw_amplitude(path)
        reader_mode = 'SATORI_FRAW_REINTERPRETED_AS_CW_AMPLITUDE'
    else:
        raw = mne.io.read_raw_snirf(str(path), preload=True, verbose='ERROR')
    sfreq_original = float(raw.info['sfreq'])
    types = raw.get_channel_types()
    if 'fnirs_cw_amplitude' not in set(types):
        raise RuntimeError(f'{path.name}: expected raw CW amplitude after reader repair, got channel types {sorted(set(types))}')
    events = get_events(raw, path)
    od = optical_density(raw, verbose='ERROR')
    try:
        sci = np.asarray(scalp_coupling_index(od, verbose='ERROR'), float)
    except Exception as exc:
        warnings.warn(f'{path.name}: SCI failed ({exc!r}); storing NaN SCI.')
        sci = np.full(len(od.ch_names), np.nan, float)
    pair_sci_values: Dict[Tuple[int, int], List[float]] = {}
    for ch_name, value in zip(od.ch_names, sci):
        pair = source_detector_pair_from_name(ch_name)
        if pair is None:
            continue
        pair_sci_values.setdefault(pair, []).append(float(value))
    od_corr = temporal_derivative_distribution_repair(od, verbose='ERROR')
    try:
        hb = beer_lambert_law(od_corr, ppf=PPF, sd_distances=SD_DISTANCE_M)
    except TypeError:
        od_for_mbll = od_corr.copy()
        for ch in od_for_mbll.info['chs']:
            ch['loc'][3:6] = np.array([0.0, 0.0, 0.0], float)
            ch['loc'][6:9] = np.array([SD_DISTANCE_M, 0.0, 0.0], float)
            ch['loc'][0:3] = np.array([SD_DISTANCE_M / 2.0, 0.0, 0.0], float)
        hb = beer_lambert_law(od_for_mbll, ppf=PPF)
    hb.filter(FILTER_L_FREQ, FILTER_H_FREQ, method='iir', iir_params={'order': FILTER_ORDER, 'ftype': 'butter'}, phase='zero', verbose='ERROR')
    if not np.isclose(float(hb.info['sfreq']), TARGET_SFREQ, atol=1e-06):
        hb.resample(TARGET_SFREQ, npad='auto', verbose='ERROR')
    data = hb.get_data()
    hbo48 = np.full((48, data.shape[1]), np.nan, float)
    hbr48 = np.full_like(hbo48, np.nan)
    pair_present = np.zeros(48, bool)
    for i, (ch_name, ch_type) in enumerate(zip(hb.ch_names, hb.get_channel_types())):
        pair = source_detector_pair_from_name(ch_name)
        if pair not in PAIR_TO_INDEX:
            continue
        ci = PAIR_TO_INDEX[pair]
        if ch_type == 'hbo':
            hbo48[ci] = data[i]
            pair_present[ci] = True
        elif ch_type == 'hbr':
            hbr48[ci] = data[i]
            pair_present[ci] = True
    pair_sci = np.full(48, np.nan, float)
    for pair, vals in pair_sci_values.items():
        if pair not in PAIR_TO_INDEX:
            continue
        finite = np.asarray(vals, float)
        finite = finite[np.isfinite(finite)]
        if len(finite):
            pair_sci[PAIR_TO_INDEX[pair]] = float(np.min(finite))
    n_hbo = int(np.sum(np.isfinite(hbo48).any(axis=1)))
    n_hbr = int(np.sum(np.isfinite(hbr48).any(axis=1)))
    if n_hbo < 40 or n_hbr < 40:
        raise RuntimeError(f'{path.name}: reconstructed Hb channel count too small (HbO={n_hbo}, HbR={n_hbr}); expected close to 48 each.')
    return RunProcessed(
        subject=subject,
        run_id=run_id,
        source_file=str(path),
        processing_mode=reader_mode + '_OD_TDDR_MBLL_BUTTERWORTH',
        sfreq_original=sfreq_original,
        sfreq_final=float(hb.info['sfreq']),
        data_times=np.asarray(hb.times, float),
        hbo48=hbo48,
        hbr48=hbr48,
        pair_sci48=pair_sci,
        pair_present48=pair_present,
        events=events,
        mne_channel_types=';'.join(sorted(set(hb.get_channel_types())))
    )
def author_filtered_preprocess(path: Path) -> RunProcessed:
    """
    Fallback only. Uses the authors' processed HbO/HbR SNIRF, which the paper
    describes as already spike-removed / converted / temporally filtered /
    normalized / baseline-zero-adjusted in Satori.
    """
    import mne
    subject, run_id = parse_subject_run(path)
    raw = mne.io.read_raw_snirf(str(path), preload=True, verbose='ERROR')
    events = get_events(raw, path)
    sfreq_original = float(raw.info['sfreq'])
    types = raw.get_channel_types()
    if not ('hbo' in types and 'hbr' in types):
        raise RuntimeError(f'{path.name}: author-filtered fallback did not contain HbO/HbR; types={sorted(set(types))}')
    hb = raw.copy()
    if not np.isclose(float(hb.info['sfreq']), TARGET_SFREQ, atol=1e-06):
        hb.resample(TARGET_SFREQ, npad='auto', verbose='ERROR')
    data = hb.get_data()
    hbo48 = np.full((48, data.shape[1]), np.nan, float)
    hbr48 = np.full_like(hbo48, np.nan)
    pair_present = np.zeros(48, bool)
    for i, (ch_name, ch_type) in enumerate(zip(hb.ch_names, hb.get_channel_types())):
        pair = source_detector_pair_from_name(ch_name)
        if pair not in PAIR_TO_INDEX:
            continue
        ci = PAIR_TO_INDEX[pair]
        if ch_type == 'hbo':
            hbo48[ci] = data[i]
            pair_present[ci] = True
        elif ch_type == 'hbr':
            hbr48[ci] = data[i]
            pair_present[ci] = True
    return RunProcessed(
        subject=subject,
        run_id=run_id,
        source_file=str(path),
        processing_mode='AUTHOR_SATORI_CC_FILTERED_FALLBACK',
        sfreq_original=sfreq_original,
        sfreq_final=float(hb.info['sfreq']),
        data_times=np.asarray(hb.times, float),
        hbo48=hbo48,
        hbr48=hbr48,
        pair_sci48=np.full(48, np.nan, float),
        pair_present48=pair_present,
        events=events,
        mne_channel_types=';'.join(sorted(set(hb.get_channel_types())))
    )
EPOCH_TIMES = np.arange(EPOCH_TMIN, EPOCH_TMAX_EXCLUSIVE, 1.0 / TARGET_SFREQ, dtype=float)
BASELINE_MASK = (EPOCH_TIMES >= BASELINE_TMIN) & (EPOCH_TIMES < BASELINE_TMAX_EXCLUSIVE)

def interpolate_epoch(data48: np.ndarray, source_times: np.ndarray, onset: float) -> np.ndarray:
    target_abs = onset + EPOCH_TIMES
    out = np.full((48, len(EPOCH_TIMES)), np.nan, float)
    for c in range(48):
        y = np.asarray(data48[c], float)
        ok = np.isfinite(source_times) & np.isfinite(y)
        if np.sum(ok) < 3:
            continue
        x = source_times[ok]
        z = y[ok]
        inside = (target_abs >= x[0]) & (target_abs <= x[-1])
        if not np.any(inside):
            continue
        out[c, inside] = np.interp(target_abs[inside], x, z)
    return out
def baseline_correct(epoch: np.ndarray) -> np.ndarray:
    out = np.asarray(epoch, float).copy()
    baseline = np.nanmean(out[:, BASELINE_MASK], axis=1)
    out -= baseline[:, None]
    return out
@dataclass
class TrialRecord:
    subject: str
    run_id: str
    source_file: str
    processing_mode: str
    trial_index_within_run: int
    event_code: int
    label: str
    onset_sec: float
    task_duration_sec: float
    hbo: np.ndarray
    hbr: np.ndarray
    pair_sci48: np.ndarray
    pair_present48: np.ndarray
def epochs_from_run(run: RunProcessed) -> Tuple[List[TrialRecord], List[Dict[str, Any]]]:
    trials: List[TrialRecord] = []
    excluded: List[Dict[str, Any]] = []
    task_events = [(i, e) for i, e in enumerate(run.events) if int(e['event_code']) in TASK_CODES]
    for within, (event_idx, event) in enumerate(task_events):
        code = int(event['event_code'])
        onset = float(event['onset_sec'])
        duration = infer_task_duration(run.events, event_idx)
        if onset + EPOCH_TMIN < run.data_times[0] - 1e-06:
            excluded.append({'subject': run.subject, 'run_id': run.run_id, 'event_code': code, 'onset_sec': onset, 'reason': 'insufficient_pre_event_data'})
            continue
        if onset + EPOCH_TMAX_EXCLUSIVE > run.data_times[-1] + 1.0 / TARGET_SFREQ:
            excluded.append({'subject': run.subject, 'run_id': run.run_id, 'event_code': code, 'onset_sec': onset, 'reason': 'insufficient_post_event_data'})
            continue
        nxt = next_task_onset(run.events, event_idx)
        if nxt is not None and nxt < onset + EPOCH_TMAX_EXCLUSIVE - NEXT_TASK_GUARD_SEC:
            excluded.append({
                'subject': run.subject,
                'run_id': run.run_id,
                'event_code': code,
                'onset_sec': onset,
                'reason': 'next_finger_task_enters_analysis_window',
                'next_task_onset_sec': nxt
            })
            continue
        hbo = interpolate_epoch(run.hbo48, run.data_times, onset)
        hbr = interpolate_epoch(run.hbr48, run.data_times, onset)
        hbo = baseline_correct(hbo)
        hbr = baseline_correct(hbr)
        finite_fraction = float(np.mean(np.isfinite(hbo) & np.isfinite(hbr)))
        if finite_fraction < 0.25:
            excluded.append({
                'subject': run.subject,
                'run_id': run.run_id,
                'event_code': code,
                'onset_sec': onset,
                'reason': 'epoch_too_sparse',
                'finite_fraction': finite_fraction
            })
            continue
        trials.append(TrialRecord(
            subject=run.subject,
            run_id=run.run_id,
            source_file=run.source_file,
            processing_mode=run.processing_mode,
            trial_index_within_run=int(within),
            event_code=code,
            label=EVENT_TO_LABEL[code],
            onset_sec=onset,
            task_duration_sec=float(duration),
            hbo=hbo,
            hbr=hbr,
            pair_sci48=np.asarray(run.pair_sci48, float),
            pair_present48=np.asarray(run.pair_present48, bool)
        ))
    return (trials, excluded)
def regional_average(data: np.ndarray, channel_mask: np.ndarray) -> np.ndarray:
    d = np.asarray(data, float)
    mask = np.asarray(channel_mask, bool)
    q = d[mask]
    if q.size == 0:
        return np.full(d.shape[1], np.nan, float)
    return np.nanmean(q, axis=0)
def morphology_for_signal(y: np.ndarray, prefix: str) -> Dict[str, float]:
    t = EPOCH_TIMES
    y = np.asarray(y, float)
    task = (t >= 0.0) & (t < 10.0)
    early = (t >= 0.0) & (t <= 5.0)
    peak_window = (t >= 0.0) & (t < 15.0)
    recovery = (t >= 10.0) & (t < 20.0)
    late = (t >= 15.0) & (t < 20.0)
    base = (t >= -5.0) & (t < 0.0)
    result = {
        f'{prefix}_baseline_sd': float(np.nanstd(y[base], ddof=1)) if np.isfinite(y[base]).sum() >= 2 else np.nan,
        f'{prefix}_task_mean_0_10': safe_mean(y[task]),
        f'{prefix}_early_slope_0_5': linear_slope(t[early], y[early]),
        f'{prefix}_auc_0_10': trapz_safe(y[task], t[task]),
        f'{prefix}_auc_10_20': trapz_safe(y[recovery], t[recovery]),
        f'{prefix}_late_mean_15_20': safe_mean(y[late])
    }
    q = y[peak_window]
    tq = t[peak_window]
    if np.isfinite(q).any():
        idx_max = int(np.nanargmax(q))
        idx_min = int(np.nanargmin(q))
        result[f'{prefix}_max_amp_0_15'] = float(q[idx_max])
        result[f'{prefix}_max_time_0_15'] = float(tq[idx_max])
        result[f'{prefix}_min_amp_0_15'] = float(q[idx_min])
        result[f'{prefix}_min_time_0_15'] = float(tq[idx_min])
    else:
        result[f'{prefix}_max_amp_0_15'] = np.nan
        result[f'{prefix}_max_time_0_15'] = np.nan
        result[f'{prefix}_min_amp_0_15'] = np.nan
        result[f'{prefix}_min_time_0_15'] = np.nan
    peak = result[f'{prefix}_max_amp_0_15']
    late_mean = result[f'{prefix}_late_mean_15_20']
    if np.isfinite(peak) and np.isfinite(late_mean):
        result[f'{prefix}_recovery_reserve'] = float((peak - late_mean) / max(abs(peak), 1e-12))
    else:
        result[f'{prefix}_recovery_reserve'] = np.nan
    return result
def trial_features(tr: TrialRecord) -> Dict[str, Any]:
    finite_channel = np.mean(np.isfinite(tr.hbo) & np.isfinite(tr.hbr), axis=1) >= MIN_FINITE_FRACTION
    sci_ok = ~np.isfinite(tr.pair_sci48) | (tr.pair_sci48 >= SCI_THRESHOLD)
    valid = finite_channel & tr.pair_present48 & sci_ok
    if np.sum(valid) < MIN_VALID_CHANNELS_WARN:
        valid = finite_channel & tr.pair_present48
    masks = {'global': valid, 'left': valid & LEFT_MASK, 'right': valid & RIGHT_MASK}
    row: Dict[str, Any] = {
        'participant': tr.subject,
        'run_id': tr.run_id,
        'trial_index_within_run': tr.trial_index_within_run,
        'event_code': tr.event_code,
        'trial_label': tr.label,
        'trial_onset_sec': tr.onset_sec,
        'task_duration_sec': tr.task_duration_sec,
        'processing_mode': tr.processing_mode,
        'n_valid_channels_trial': int(np.sum(valid)),
        'n_valid_left_channels_trial': int(np.sum(valid & LEFT_MASK)),
        'n_valid_right_channels_trial': int(np.sum(valid & RIGHT_MASK))
    }
    signals = {}
    for region, mask in masks.items():
        hbo = regional_average(tr.hbo, mask)
        hbr = regional_average(tr.hbr, mask)
        signals[region, 'hbo'] = hbo
        signals[region, 'hbr'] = hbr
        row.update(morphology_for_signal(hbo, f'{region}_hbo'))
        row.update(morphology_for_signal(hbr, f'{region}_hbr'))
    row['hbo_task_laterality_left_minus_right'] = row['left_hbo_task_mean_0_10'] - row['right_hbo_task_mean_0_10']
    row['hbo_peak_laterality_left_minus_right'] = row['left_hbo_max_amp_0_15'] - row['right_hbo_max_amp_0_15']
    return row
def participant_validity(trials: Sequence[TrialRecord]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not trials:
        return (np.zeros(48, bool), np.full(48, np.nan, float), np.zeros(48, float))
    hbo = np.stack([t.hbo for t in trials], axis=0)
    hbr = np.stack([t.hbr for t in trials], axis=0)
    finite_fraction = np.mean(np.isfinite(hbo) & np.isfinite(hbr), axis=(0, 2))
    sci_stack = np.stack([t.pair_sci48 for t in trials], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        med_sci = np.nanmedian(sci_stack, axis=0)
    has_sci = np.isfinite(med_sci)
    sci_ok = ~has_sci | (med_sci >= SCI_THRESHOLD)
    valid = (finite_fraction >= MIN_FINITE_FRACTION) & sci_ok
    if np.sum(valid) < MIN_VALID_CHANNELS_WARN:
        valid = finite_fraction >= MIN_FINITE_FRACTION
    return (valid.astype(bool), med_sci.astype(float), finite_fraction.astype(float))
def write_hdf5(participants: Mapping[str, Sequence[TrialRecord]], out_path: Path) -> 'pandas.DataFrame':
    import h5py
    import pandas as pd
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + '.tmp')
    if tmp.exists():
        tmp.unlink()
    participant_qc = []
    utf8 = h5py.string_dtype(encoding='utf-8')
    with h5py.File(tmp, 'w') as h5:
        h5.attrs['dataset'] = 'Khan2026 individual finger movement fNIRS'
        h5.attrs['version'] = VERSION
        h5.attrs['article_doi'] = ARTICLE_DOI
        h5.attrs['target_sfreq_hz'] = TARGET_SFREQ
        h5.attrs['epoch_tmin_sec'] = EPOCH_TMIN
        h5.attrs['epoch_tmax_exclusive_sec'] = EPOCH_TMAX_EXCLUSIVE
        h5.attrs['baseline_window'] = '[-5,0) seconds'
        h5.attrs['task_event_codes'] = '3,4,5,6,7'
        h5.attrs['nominal_task_duration_sec'] = 10.0
        h5.attrs['short_channel_regression'] = 'NO - reported montage uses 3-cm channels'
        h5.attrs['zero_shot_target_labels_used_for_preprocessing'] = 0
        h5.create_dataset('channel_names', data=np.asarray(CHANNEL_NAMES, dtype=object), dtype=utf8)
        h5.create_dataset('hemisphere', data=np.asarray(HEMISPHERES, dtype=object), dtype=utf8)
        h5.create_dataset('source_index', data=np.asarray([s for _, s, _, _ in CANONICAL_CHANNELS], np.int16))
        h5.create_dataset('detector_index', data=np.asarray([d for _, _, d, _ in CANONICAL_CHANNELS], np.int16))
        for pid in sorted(participants):
            trials = list(participants[pid])
            if not trials:
                continue
            hbo = np.stack([t.hbo for t in trials], axis=0).astype(np.float32)
            hbr = np.stack([t.hbr for t in trials], axis=0).astype(np.float32)
            valid, med_sci, finite_fraction = participant_validity(trials)
            g = h5.create_group(pid)
            g.create_dataset('hbo', data=hbo, compression='gzip', compression_opts=4, shuffle=True)
            g.create_dataset('hbr', data=hbr, compression='gzip', compression_opts=4, shuffle=True)
            g.create_dataset('times_sec', data=EPOCH_TIMES.astype(np.float32))
            g.create_dataset('channel_valid_mask', data=valid.astype(np.uint8))
            g.create_dataset('channel_median_sci', data=med_sci.astype(np.float32))
            g.create_dataset('channel_finite_fraction', data=finite_fraction.astype(np.float32))
            g.create_dataset('trial_label', data=np.asarray([t.label for t in trials], dtype=object), dtype=utf8)
            g.create_dataset('trial_event_code', data=np.asarray([t.event_code for t in trials], np.int16))
            g.create_dataset('trial_run_id', data=np.asarray([t.run_id for t in trials], dtype=object), dtype=utf8)
            g.create_dataset('trial_onset_sec', data=np.asarray([t.onset_sec for t in trials], np.float32))
            g.create_dataset('task_duration_sec', data=np.asarray([t.task_duration_sec for t in trials], np.float32))
            g.create_dataset('source_file', data=np.asarray([t.source_file for t in trials], dtype=object), dtype=utf8)
            g.create_dataset('processing_mode', data=np.asarray([t.processing_mode for t in trials], dtype=object), dtype=utf8)
            labels = [t.label for t in trials]
            modes = sorted(set((t.processing_mode for t in trials)))
            participant_qc.append({
                'participant': pid,
                'n_trials': int(len(trials)),
                'n_runs': int(len(set((t.run_id for t in trials)))),
                'n_valid_channels': int(np.sum(valid)),
                'n_left_valid_channels': int(np.sum(valid & LEFT_MASK)),
                'n_right_valid_channels': int(np.sum(valid & RIGHT_MASK)),
                'median_channel_sci': safe_median(med_sci),
                'median_channel_finite_fraction': safe_median(finite_fraction),
                'processing_modes': ';'.join(modes),
                **{f'n_{lab}_trials': int(sum((x == lab for x in labels))) for lab in EVENT_TO_LABEL.values()},
                'qc_warn_lt_8_valid_channels': int(np.sum(valid) < MIN_VALID_CHANNELS_WARN)
            })
    tmp.replace(out_path)
    return pd.DataFrame(participant_qc)
def run_one(raw_path: Path, filtered_files: Sequence[Path], allow_author_filtered_fallback: bool) -> Tuple[Optional[RunProcessed], Dict[str, Any]]:
    subject, run_id = parse_subject_run(raw_path)
    row: Dict[str, Any] = {'participant': subject, 'run_id': run_id, 'raw_file': str(raw_path), 'status': 'FAILED', 'fallback_used': 0}
    try:
        run = raw_intensity_preprocess(raw_path)
        row.update({
            'status': 'PASS',
            'processing_mode': run.processing_mode,
            'sfreq_original': run.sfreq_original,
            'sfreq_final': run.sfreq_final,
            'n_events': len(run.events),
            'n_task_events': sum((e['event_code'] in TASK_CODES for e in run.events)),
            'n_present_channels': int(np.sum(run.pair_present48)),
            'n_sci_pass_channels': int(np.sum(np.isfinite(run.pair_sci48) & (run.pair_sci48 >= SCI_THRESHOLD))),
            'median_sci': safe_median(run.pair_sci48)
        })
        return (run, row)
    except Exception as exc:
        row['raw_error'] = repr(exc)
        if not allow_author_filtered_fallback:
            return (None, row)
        fallback = matching_author_filtered(raw_path, filtered_files)
        if fallback is None:
            row['fallback_error'] = 'No matching author-filtered SNIRF'
            return (None, row)
        try:
            run = author_filtered_preprocess(fallback)
            row.update({
                'status': 'PASS_FALLBACK',
                'fallback_used': 1,
                'fallback_file': str(fallback),
                'processing_mode': run.processing_mode,
                'sfreq_original': run.sfreq_original,
                'sfreq_final': run.sfreq_final,
                'n_events': len(run.events),
                'n_task_events': sum((e['event_code'] in TASK_CODES for e in run.events)),
                'n_present_channels': int(np.sum(run.pair_present48)),
                'n_sci_pass_channels': np.nan,
                'median_sci': np.nan
            })
            return (run, row)
        except Exception as exc2:
            row['fallback_error'] = repr(exc2)
            return (None, row)
def build_inventory(raw_files: Sequence[Path], filtered_files: Sequence[Path]):
    import pandas as pd
    rows = []
    for kind, files in [('raw_candidate', raw_files), ('author_filtered', filtered_files)]:
        for p in files:
            try:
                s, r = parse_subject_run(p)
            except Exception:
                s, r = ('unknown', 'unknown')
            rows.append({'kind': kind, 'participant': s, 'run_id': r, 'path': str(p), 'size_mb': p.stat().st_size / 1024 ** 2})
    return pd.DataFrame(rows)
def main_pipeline(
    root: Path,
    archive_arg: Optional[str],
    skip_download: bool,
    force_download: bool,
    force_extract: bool,
    force_preprocess: bool,
    allow_author_filtered_fallback: bool
) -> None:
    import pandas as pd
    root = Path(root)
    download_dir = root / 'raw_download'
    extract_dir = root / 'raw_extracted'
    processed_dir = root / 'processed'
    feature_dir = root / 'features'
    audit_dir = root / 'audit'
    for d in [download_dir, extract_dir, processed_dir, feature_dir, audit_dir]:
        d.mkdir(parents=True, exist_ok=True)
    output_h5 = processed_dir / 'khan2026_finger_epochs_10hz.h5'
    feature_csv = feature_dir / 'trial_morphology_features.csv.gz'
    if output_h5.is_file() and feature_csv.is_file() and (not force_preprocess):
        print('[done] Existing processed outputs found.')
        print('H5      :', output_h5)
        print('Features:', feature_csv)
        print('Use --force-preprocess to rebuild.')
        return
    if archive_arg:
        archive = Path(archive_arg)
        if not archive.is_file():
            raise FileNotFoundError(archive)
        print('[input] Using supplied archive:', archive)
    else:
        existing_snirf = list(extract_dir.rglob('*.snirf')) + list(extract_dir.rglob('*.SNIRF'))
        if skip_download and existing_snirf:
            archive = None
            print('[input] --skip-download and extracted SNIRF already present.')
        elif skip_download:
            candidates = sorted(
                [p for p in download_dir.iterdir() if p.is_file() and p.stat().st_size >= MIN_REASONABLE_DOWNLOAD_BYTES],
                key=lambda p: p.stat().st_size,
                reverse=True
            )
            if not candidates:
                raise FileNotFoundError(f'--skip-download was requested but no extracted SNIRF files or substantial archive exists under {root}.')
            archive = candidates[0]
            print('[input] Reusing local downloaded archive:', archive)
        else:
            archive = download_figshare(download_dir, force=force_download)
    if archive is not None:
        extract_archive(archive, extract_dir, force=force_extract)
    raw_files, filtered_files = find_snirf_files(extract_dir)
    inventory = build_inventory(raw_files, filtered_files)
    atomic_csv(inventory, audit_dir / 'source_file_inventory.csv')
    print('\n[inventory]')
    print('Raw candidate SNIRF :', len(raw_files))
    print('Author-filtered SNIRF:', len(filtered_files))
    parsed_subjects = sorted(set((parse_subject_run(p)[0] for p in raw_files)))
    print('Participants:', len(parsed_subjects), parsed_subjects)
    participants: Dict[str, List[TrialRecord]] = {}
    run_qc = []
    excluded = []
    features = []
    total = len(raw_files)
    for i, raw_path in enumerate(raw_files, 1):
        try:
            subject, run_id = parse_subject_run(raw_path)
        except Exception as exc:
            run_qc.append({'raw_file': str(raw_path), 'status': 'FAILED_PARSE_NAME', 'error': repr(exc)})
            continue
        print(f'\n[run {i:02d}/{total:02d}] {subject} {run_id} — {raw_path.name}')
        run, qc = run_one(raw_path, filtered_files, allow_author_filtered_fallback)
        if run is None:
            print('  FAILED:', qc.get('raw_error', qc.get('fallback_error', 'unknown')))
            run_qc.append(qc)
            continue
        trials, exclusions = epochs_from_run(run)
        qc['n_trials_kept'] = int(len(trials))
        qc['n_trials_excluded'] = int(len(exclusions))
        run_qc.append(qc)
        excluded.extend(exclusions)
        participants.setdefault(subject, []).extend(trials)
        for tr in trials:
            features.append(trial_features(tr))
        print('  mode:', run.processing_mode)
        print('  sfreq:', f'{run.sfreq_original:.5f}', '->', f'{run.sfreq_final:.2f}')
        print('  task events:', sum((e['event_code'] in TASK_CODES for e in run.events)), '| epochs kept:', len(trials), '| excluded:', len(exclusions))
    if not participants:
        raise RuntimeError('No participants produced valid epochs. Check audit/run_qc.csv.')
    participant_qc = write_hdf5(participants, output_h5)
    feature_df = pd.DataFrame(features)
    run_qc_df = pd.DataFrame(run_qc)
    excluded_df = pd.DataFrame(excluded)
    atomic_csv(feature_df, feature_csv)
    atomic_csv(run_qc_df, audit_dir / 'run_qc.csv')
    atomic_csv(participant_qc, audit_dir / 'participant_qc.csv')
    atomic_csv(excluded_df, audit_dir / 'excluded_trials.csv')
    channel_map_df = pd.DataFrame([{
        'channel': ch,
        'source_index': s,
        'detector_index': d,
        'hemisphere': hemi,
        'source_detector_distance_m': SD_DISTANCE_M
    } for ch, s, d, hemi in CANONICAL_CHANNELS])
    atomic_csv(channel_map_df, audit_dir / 'channel_map.csv')
    event_df = pd.DataFrame([{
        'event_code': REST_CODE,
        'label': 'intermediate_rest',
        'used_as_task_epoch': 0
    }, *[{'event_code': code, 'label': label, 'used_as_task_epoch': 1} for code, label in EVENT_TO_LABEL.items()]])
    atomic_csv(event_df, audit_dir / 'event_dictionary.csv')
    import h5py
    with h5py.File(output_h5, 'r') as h5:
        h5_participants = sorted([k for k in h5.keys() if re.fullmatch('S\\d{2}', str(k))])
        shape_checks = []
        for pid in h5_participants:
            g = h5[pid]
            ok = g['hbo'].ndim == 3 and g['hbr'].shape == g['hbo'].shape and (g['hbo'].shape[1] == 48) and (g['hbo'].shape[2] == len(EPOCH_TIMES)) and (len(g['times_sec']) == len(EPOCH_TIMES))
            shape_checks.append(ok)
        if not all(shape_checks):
            raise AssertionError('Final HDF5 shape validation failed.')
    manifest = {
        'version': VERSION,
        'article_doi': ARTICLE_DOI,
        'article_url': ARTICLE_URL,
        'figshare_share_url': FIGSHARE_SHARE_URL,
        'figshare_file_id': FIGSHARE_FILE_ID,
        'n_raw_snirf_candidates': int(len(raw_files)),
        'n_author_filtered_snirf': int(len(filtered_files)),
        'n_participants_output': int(len(participants)),
        'n_trials_output': int(sum((len(v) for v in participants.values()))),
        'target_sfreq_hz': TARGET_SFREQ,
        'epoch_window_sec': [EPOCH_TMIN, EPOCH_TMAX_EXCLUSIVE],
        'baseline_window_sec': [BASELINE_TMIN, BASELINE_TMAX_EXCLUSIVE],
        'filter_hz': [FILTER_L_FREQ, FILTER_H_FREQ],
        'filter_type': f'zero-phase Butterworth order {FILTER_ORDER}',
        'motion_correction': 'TDDR in optical density',
        'satori_raw_snirf_repair': 'Satori TRIM files contain multiple representations; only raw/CW-amplitude entries for the publication-defined 48 source-detector pairs x 760/850 nm are retained, while dOD, concentration and auxiliary entries are ignored',
        'mbll_ppf': PPF,
        'source_detector_distance_m': SD_DISTANCE_M,
        'sci_threshold': SCI_THRESHOLD,
        'sci_role': 'channel QC metadata only',
        'short_channel_regression': False,
        'reason_no_short_channel_regression': 'Published montage reports 48 source-detector channels at approximately 3 cm; no short-separation montage is reported.',
        'task_codes': list(TASK_CODES),
        'event_mapping': EVENT_TO_LABEL,
        'author_filtered_fallback_enabled': bool(allow_author_filtered_fallback),
        'external_target_task_labels_used_for_parameter_tuning': False,
        'hemoboundary_run_here': False,
        'output_h5': str(output_h5),
        'feature_csv': str(feature_csv),
        'pipeline_hash': stable_hash({'version': VERSION, 'target_sfreq': TARGET_SFREQ, 'epoch': [EPOCH_TMIN, EPOCH_TMAX_EXCLUSIVE], 'baseline': [BASELINE_TMIN, BASELINE_TMAX_EXCLUSIVE], 'filter': [FILTER_L_FREQ, FILTER_H_FREQ, FILTER_ORDER], 'ppf': PPF, 'sd_distance': SD_DISTANCE_M, 'sci_threshold': SCI_THRESHOLD, 'channel_map': CANONICAL_CHANNELS, 'event_map': EVENT_TO_LABEL})
    }
    atomic_json(manifest, audit_dir / 'preprocessing_manifest.json')
    print('\n' + '=' * 150)
    print('KHAN2026 FINGER fNIRS PREPARATION COMPLETE')
    print('=' * 150)
    print('Participants:', len(participants))
    print('Trials      :', sum((len(v) for v in participants.values())))
    print('HDF5        :', output_h5)
    print('Features    :', feature_csv)
    print('Run QC      :', audit_dir / 'run_qc.csv')
    print('Participant QC:', audit_dir / 'participant_qc.csv')
    print('Manifest    :', audit_dir / 'preprocessing_manifest.json')
    print('\nParticipant summary:')
    print(participant_qc.to_string(index=False))
def self_tests() -> None:
    assert len(CANONICAL_CHANNELS) == 48
    assert len(set(((s, d) for _, s, d, _ in CANONICAL_CHANNELS))) == 48
    assert int(np.sum(LEFT_MASK)) == 24
    assert int(np.sum(RIGHT_MASK)) == 24
    for code, label in EVENT_TO_LABEL.items():
        assert event_code_from_text(str(code)) == code
        assert event_code_from_text(label) == code
        assert event_code_from_text(f'Stimulus/S {code}') == code
    assert event_code_from_text('rest') == 2
    assert source_detector_pair_from_name('S1_D1 760') == (1, 1)
    assert source_detector_pair_from_name('S09_D11 hbo') == (9, 11)
    assert PAIR_TO_INDEX[1, 1] == 0
    assert PAIR_TO_INDEX[16, 16] == 47
    assert len(EPOCH_TIMES) == 250
    assert np.isclose(EPOCH_TIMES[0], -5.0)
    assert EPOCH_TIMES[-1] < 20.0
    ev = [
        {'onset_sec': 20.0, 'duration_sec': 0.0, 'event_code': 3},
        {'onset_sec': 30.0, 'duration_sec': 0.0, 'event_code': 2},
        {'onset_sec': 40.0, 'duration_sec': 0.0, 'event_code': 4}
    ]
    assert np.isclose(infer_task_duration(ev, 0), 10.0)
    assert np.isclose(next_task_onset(ev, 0), 40.0)
    x = np.ones((48, len(EPOCH_TIMES)), float)
    x[:, EPOCH_TIMES >= 0] += 2.0
    y = baseline_correct(x)
    assert abs(float(np.nanmean(y[:, BASELINE_MASK]))) < 1e-10
    assert np.isclose(float(np.nanmean(y[:, (EPOCH_TIMES >= 0) & (EPOCH_TIMES < 10)])), 2.0)
    print('KHAN2026 finger-fNIRS preparation protected self-tests: PASS')
def parse_args(argv: Optional[Sequence[str]]=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--root', default=None, help='Project output root. Defaults to Google Drive HonorProject when running in Colab.')
    p.add_argument('--archive', default=None, help='Optional already-downloaded Figshare archive/file.')
    p.add_argument('--skip-download', action='store_true', help='Do not contact Figshare. Reuse extracted files or an archive already in raw_download/.')
    p.add_argument('--force-download', action='store_true')
    p.add_argument('--force-extract', action='store_true')
    p.add_argument('--force-preprocess', action='store_true')
    p.add_argument(
        '--allow-author-filtered-fallback',
        action='store_true',
        help="If raw SNIRF processing fails for a run, permit the matching authors' CC_filtered HbO/HbR SNIRF as a fallback."
    )
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
    root = Path(args.root) if args.root else default_root()
    print('=' * 150)
    print('KHAN2026 INDIVIDUAL-FINGER fNIRS — DOWNLOAD + PREPROCESS + FEATURES')
    print('=' * 150)
    print('Version     :', VERSION)
    print('Root        :', root)
    print('Article DOI :', ARTICLE_DOI)
    print('Figshare ID :', FIGSHARE_FILE_ID)
    print('Target sfreq:', TARGET_SFREQ, 'Hz')
    print('Epoch       :', f'[{EPOCH_TMIN}, {EPOCH_TMAX_EXCLUSIVE}) s')
    print('Filter      :', f'{FILTER_L_FREQ}-{FILTER_H_FREQ} Hz Butterworth order {FILTER_ORDER}')
    print('Motion      : TDDR in optical density')
    print('MBLL        : PPF', PPF, '| distance', SD_DISTANCE_M, 'm')
    print('SCI         :', SCI_THRESHOLD, '(QC only)')
    print('Author-filtered fallback:', bool(args.allow_author_filtered_fallback))
    ensure_dependencies()
    main_pipeline(
        root=root,
        archive_arg=args.archive,
        skip_download=args.skip_download,
        force_download=args.force_download,
        force_extract=args.force_extract,
        force_preprocess=args.force_preprocess,
        allow_author_filtered_fallback=args.allow_author_filtered_fallback
    )
if __name__ == '__main__':
    main()
