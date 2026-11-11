"""Recorded-response recovery in the Shin and Bálint datasets."""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import h5py
import numpy as np
import pandas as pd
VERSION = 'BALINT_LATE_WINDOW_RECOVERY_REAL_V1_20260922'
DEFAULT_H5 = Path('/content/drive/MyDrive/HonorProject/HemoBoundary/Balint/processed/balint_epochs_10hz.h5')
DEFAULT_OUTPUT = Path('/content/drive/MyDrive/HonorProject/HemoBoundary/Balint/late_window_recovery')
EXPECTED_PARTICIPANTS = 26
EXPECTED_COMPLETE_STIMULI = 40
STIMULI_PER_BLOCK = 4
EXPECTED_BLOCKS = 10
BASELINE_START = -5.0
BASELINE_END = 0.0
PRIMARY_END = 20.0
LATE_END = 24.0
STIMULUS_DURATION_SEC = 13.0
BASELINE_Q_LOW = 0.025
BASELINE_Q_HIGH = 0.975
SUSTAIN_SEC = 0.8
SECONDARY_PEAK_FRACTION = 0.2
MIN_BASELINE_SAMPLES = 20
MIN_VALID_CHANNELS = 2
MIN_FINITE_FRACTION = 0.8
BOOT_REPS = 20000
BOOT_SEED = 20260922
EPS = 1e-12
def _safe_is_file(path: Path) -> bool:
    """Path.is_file() that treats broken FUSE/Drive mounts as unavailable."""
    try:
        return path.is_file()
    except OSError:
        return False
def _safe_is_dir(path: Path) -> bool:
    """Path.is_dir() that treats broken FUSE/Drive mounts as unavailable."""
    try:
        return path.is_dir()
    except OSError:
        return False
def try_mount_drive(force: bool=False) -> None:
    """
    Ensure Google Drive is mounted at /content/drive.

    This project intentionally uses /content/drive/MyDrive only.
    A stale /content/gdrive alias is never used.
    """
    root = Path('/content/drive/MyDrive')
    if not force and _safe_is_dir(root):
        return
    try:
        from google.colab import drive
    except Exception as e:
        raise RuntimeError('Google Drive is not available. Run this script in Colab or mount /content/drive manually.') from e
    try:
        drive.mount('/content/drive', force_remount=bool(force))
    except Exception:
        try:
            drive.mount('/content/drive', force_remount=True)
        except Exception as e2:
            raise RuntimeError('Could not mount Google Drive at /content/drive. Restart the Colab runtime or remount Drive, then rerun.') from e2
    if not _safe_is_dir(root):
        raise RuntimeError('Google Drive mount did not become readable at /content/drive/MyDrive.')
def parse_args(argv: Optional[Sequence[str]]=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Bálint real-data +20 to +24 s late-window recovery analysis.')
    p.add_argument('--h5', default=str(DEFAULT_H5))
    p.add_argument('--output', default=str(DEFAULT_OUTPUT))
    p.add_argument('--self-test-only', action='store_true')
    args, unknown = p.parse_known_args(argv)
    if unknown:
        print('Ignoring notebook/runtime arguments:', unknown)
    return args
def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    compression = 'gzip' if str(path).endswith('.gz') else None
    df.to_csv(tmp, index=False, compression=compression)
    os.replace(tmp, path)
def atomic_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)
def decode_scalar(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode('utf-8', errors='replace')
    if isinstance(x, np.bytes_):
        return bytes(x).decode('utf-8', errors='replace')
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return decode_scalar(x.item())
    return str(x)
def decode_string_array(a: Any) -> np.ndarray:
    arr = np.asarray(a)
    return np.asarray([decode_scalar(x) for x in arr.reshape(-1)], dtype=str).reshape(arr.shape)
def participant_groups(h5: h5py.File) -> List[str]:
    groups = []
    for key in h5.keys():
        obj = h5[key]
        if isinstance(obj, h5py.Group) and re.fullmatch('(?i)C\\d{1,3}', key):
            groups.append(key)
    groups.sort(key=lambda x: int(re.findall('\\d+', x)[0]))
    return groups
def score_name(name: str, positive_tokens: Sequence[str]) -> int:
    low = name.lower()
    leaf = low.rsplit('/', 1)[-1]
    score = 0
    for token in positive_tokens:
        if leaf == token:
            score += 10
        elif token in leaf:
            score += 4
        elif token in low:
            score += 1
    return score
def find_time_vector(group: h5py.Group) -> Tuple[str, np.ndarray]:
    candidates = []
    for name, obj in group.items():
        if not isinstance(obj, h5py.Dataset) or obj.ndim != 1:
            continue
        if not np.issubdtype(obj.dtype, np.number):
            continue
        score = score_name(name, ('times_sec', 'time_sec', 'times', 'time', 'epoch_times'))
        if score <= 0:
            continue
        a = np.asarray(obj[:], float).reshape(-1)
        if len(a) < 50 or not np.isfinite(a).all():
            continue
        if not np.all(np.diff(a) > 0):
            continue
        if a[0] > BASELINE_START + 0.2 or a[-1] < LATE_END - 0.2:
            continue
        candidates.append((score, name, a))
    if not candidates:
        raise RuntimeError(f'{group.name}: could not identify a monotonic time vector covering approximately {BASELINE_START:g} to {LATE_END:g} s.')
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return (candidates[0][1], candidates[0][2])
def normalize_trial_channel_time(array: np.ndarray, n_time: int, label: str) -> np.ndarray:
    """
    Convert a 3D array to [trial, channel, time].
    The time axis is fixed by n_time; among the remaining axes the smaller
    channel-like dimension is preferred as channels.
    """
    x = np.asarray(array, float)
    if x.ndim != 3:
        raise RuntimeError(f'{label}: expected a 3D array, got {x.shape}.')
    t_axes = [i for i, s in enumerate(x.shape) if s == n_time]
    if len(t_axes) != 1:
        raise RuntimeError(f'{label}: cannot uniquely identify time axis length {n_time} in shape {x.shape}.')
    ta = t_axes[0]
    other = [i for i in range(3) if i != ta]
    a0, a1 = other
    if x.shape[a0] <= x.shape[a1]:
        ca, ra = (a0, a1)
    else:
        ca, ra = (a1, a0)
    if 40 in (x.shape[a0], x.shape[a1]):
        if x.shape[a0] == 40 and x.shape[a1] != 40:
            ra, ca = (a0, a1)
        elif x.shape[a1] == 40 and x.shape[a0] != 40:
            ra, ca = (a1, a0)
    out = np.transpose(x, (ra, ca, ta))
    return out
def find_hbo_hbr(group: h5py.Group, n_time: int) -> Tuple[str, str, np.ndarray, np.ndarray]:
    hbo_candidates = []
    hbr_candidates = []

    def visit(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        if obj.ndim != 3 or not np.issubdtype(obj.dtype, np.number):
            return
        low = name.lower()
        hbo_score = score_name(name, ('hbo', 'hbo_epochs', 'oxy', 'oxygenated'))
        hbr_score = score_name(name, ('hbr', 'hbr_epochs', 'deoxy', 'deoxygenated'))
        if 'deoxy' in low or 'deoxygenated' in low:
            hbo_score = 0
        if hbo_score > 0:
            hbo_candidates.append((hbo_score, name, obj))
        if hbr_score > 0:
            hbr_candidates.append((hbr_score, name, obj))
    group.visititems(visit)
    pairs = []
    for so, no, dso in hbo_candidates:
        for sr, nr, dsr in hbr_candidates:
            try:
                H = normalize_trial_channel_time(dso[:], n_time, group.name + '/' + no)
                R = normalize_trial_channel_time(dsr[:], n_time, group.name + '/' + nr)
            except RuntimeError:
                continue
            if H.shape[0] != R.shape[0] or H.shape[-1] != R.shape[-1]:
                continue
            c = min(H.shape[1], R.shape[1])
            H = H[:, :c, :]
            R = R[:, :c, :]
            pairs.append((so + sr, no, nr, H, R))
    if not pairs:
        raise RuntimeError(f'{group.name}: could not resolve matching 3D HbO/HbR datasets.')
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    _, no, nr, H, R = pairs[0]
    return (no, nr, H, R)
def optional_1d_dataset(group: h5py.Group, names: Sequence[str], n: int) -> Tuple[Optional[str], Optional[np.ndarray]]:
    lowered = {key.lower(): key for key in group.keys()}
    for wanted in names:
        if wanted.lower() in lowered:
            key = lowered[wanted.lower()]
            ds = group[key]
            if isinstance(ds, h5py.Dataset) and ds.ndim == 1 and (len(ds) == n):
                return (key, np.asarray(ds[:]))
    candidates = []
    for key, obj in group.items():
        if not isinstance(obj, h5py.Dataset) or obj.ndim != 1 or len(obj) != n:
            continue
        sc = score_name(key, names)
        if sc > 0:
            candidates.append((sc, key, np.asarray(obj[:])))
    if not candidates:
        return (None, None)
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return (candidates[0][1], candidates[0][2])
def load_participant(group: h5py.Group) -> Dict[str, Any]:
    time_name, times = find_time_vector(group)
    hbo_name, hbr_name, H, R = find_hbo_hbr(group, len(times))
    n_trials = H.shape[0]
    onset_name, onset = optional_1d_dataset(group, ('trial_onset_sec', 'trial_onset', 'onset_sec'), n_trials)
    json_name, event_json = optional_1d_dataset(group, ('event_row_json', 'event_json', 'trial_event_json'), n_trials)
    if onset is None:
        onset = np.full(n_trials, np.nan, float)
    else:
        onset = pd.to_numeric(pd.Series(onset.reshape(-1)), errors='coerce').to_numpy(float)
    if event_json is None:
        event_json = np.asarray([''] * n_trials, dtype=str)
    else:
        event_json = decode_string_array(event_json).reshape(-1)
    order = np.arange(n_trials)
    if np.isfinite(onset).all() and np.all(np.diff(onset) >= 0):
        pass
    elif np.isfinite(onset).all():
        order = np.argsort(onset, kind='stable')
    return dict(participant=group.name.rsplit('/', 1)[-1], times=times, hbo=H[order], hbr=R[order], trial_onset_sec=onset[order], event_row_json=event_json[order], original_index=order.astype(int), audit=dict(
        time_dataset=group.name + '/' + time_name,
        hbo_dataset=group.name + '/' + hbo_name,
        hbr_dataset=group.name + '/' + hbr_name,
        trial_onset_dataset=None if onset_name is None else group.name + '/' + onset_name,
        event_json_dataset=None if json_name is None else group.name + '/' + json_name,
        n_trials=int(n_trials),
        n_channels=int(H.shape[1]),
        n_timepoints=int(H.shape[2]),
        time_start=float(times[0]),
        time_end=float(times[-1]),
        median_dt=float(np.median(np.diff(times)))
    ))
def safe_json_loads(text: str) -> Any:
    text = (text or '').strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {'_unparsed_text': text}
def flatten_json(obj: Any, prefix: str='', out: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    if out is None:
        out = {}
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            new = f'{prefix}.{key}' if prefix else str(key)
            flatten_json(value, new, out)
    elif isinstance(obj, (list, tuple)):
        out[prefix] = json.dumps(obj, default=str)
    else:
        out[prefix] = obj
    return out
def leaf_key(path: str) -> str:
    return path.lower().rsplit('.', 1)[-1]
def parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        x = float(value)
        return x if np.isfinite(x) else None
    text = str(value)
    nums = re.findall('[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)', text)
    if not nums:
        return None
    values = []
    for token in nums:
        try:
            x = float(token)
            if np.isfinite(x):
                values.append(x)
        except Exception:
            pass
    if not values:
        return None
    positive = [x for x in values if x >= 0]
    return min(positive) if positive else values[0]
def find_exact_leaf(flat: Mapping[str, Any], candidate_leafs: Sequence[str]) -> Tuple[Optional[str], Any]:
    wanted = {x.lower() for x in candidate_leafs}
    for key, value in flat.items():
        if leaf_key(key) in wanted:
            return (key, value)
    return (None, None)
def find_key_with_tokens(flat: Mapping[str, Any], token_groups: Sequence[Sequence[str]]) -> Tuple[Optional[str], Any]:
    """
    Return first key whose lower-case path contains at least one token
    from every token group.
    """
    for key, value in flat.items():
        low = key.lower()
        ok = True
        for group in token_groups:
            if not any((token in low for token in group)):
                ok = False
                break
        if ok:
            return (key, value)
    return (None, None)
def extract_event_fields(event_json: str, current_onset: float) -> Dict[str, Any]:
    obj = safe_json_loads(event_json)
    flat = flatten_json(obj)
    _, condition = find_exact_leaf(flat, ('trial_type', 'condition', 'trial_label', 'stimulus', 'stimulus_type', 'event_type', 'description'))
    pos_key, pos_val = find_exact_leaf(flat, (
        'trial_in_block',
        'position_in_block',
        'stimulus_in_block',
        'stim_in_block',
        'within_block_position',
        'block_position',
        'order_in_block'
    ))
    position = parse_number(pos_val)
    block_key, block_val = find_exact_leaf(flat, ('block', 'block_id', 'block_number', 'block_index'))
    block = parse_number(block_val)
    duration_key, duration_val = find_exact_leaf(flat, ('duration', 'stimulus_duration', 'duration_sec'))
    duration = parse_number(duration_val)
    if duration is None or duration <= 0 or duration > 60:
        duration = STIMULUS_DURATION_SEC
    break_key, break_val = find_key_with_tokens(flat, (('break', 'iti', 'inter_trial', 'intertrial', 'pause', 'inter_stim', 'interstim'),))
    break_sec = parse_number(break_val)
    if break_sec is not None and (not 0 <= break_sec <= 120):
        break_sec = None
    gap_key, gap_val = find_key_with_tokens(flat, (('next',), ('gap', 'interval', 'delay')))
    next_gap = parse_number(gap_val)
    if next_gap is not None and (not 0 < next_gap <= 300):
        next_gap = None
    next_time_key, next_time_val = find_key_with_tokens(flat, (('next',), ('onset', 'start', 'time')))
    next_time = parse_number(next_time_val)
    if next_gap is None and next_time is not None:
        if np.isfinite(current_onset) and next_time > current_onset + 1.0:
            next_gap = float(next_time - current_onset)
        elif 0 < next_time <= 300:
            next_gap = float(next_time)
    compact_flat = {str(k): v.item() if isinstance(v, np.generic) else v for k, v in flat.items()}
    return dict(
        condition='' if condition is None else str(condition),
        position_in_block=position,
        position_key=pos_key,
        block=block,
        block_key=block_key,
        stimulus_duration_sec=float(duration),
        duration_key=duration_key,
        break_sec=break_sec,
        break_key=break_key,
        next_event_gap_sec=next_gap,
        next_event_key=gap_key or next_time_key,
        flat_json=json.dumps(compact_flat, default=str, ensure_ascii=False)
    )
def event_clean_decision(participant_n_trials: int, sequential_index: int, meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Conservative rule."""
    pos = meta.get('position_in_block')
    duration = float(meta.get('stimulus_duration_sec') or STIMULUS_DURATION_SEC)
    break_sec = meta.get('break_sec')
    next_gap = meta.get('next_event_gap_sec')
    if pos is not None and np.isfinite(pos):
        ipos = int(round(float(pos)))
        if 0 <= ipos <= 3 and ipos == 0:
            ipos_1 = 1
        elif 0 <= ipos <= 3 and ipos > 0:
            ipos_1 = ipos
        else:
            ipos_1 = ipos
        if ipos_1 == 4:
            return dict(event_clean=False, event_status='EXCLUDE_PREQUESTION_POSITION4', event_rule='explicit_position_in_block', inferred_block_position=4)
        if ipos_1 in (1, 2, 3):
            return dict(event_clean=True, event_status='SAFE_NONQUESTION_POSITION', event_rule='explicit_position_in_block', inferred_block_position=ipos_1)
    if break_sec is not None and np.isfinite(break_sec):
        event_gap = duration + float(break_sec)
        clean = event_gap > LATE_END + 1e-09
        return dict(
            event_clean=bool(clean),
            event_status='SAFE_EXPLICIT_BREAK' if clean else 'EXCLUDE_EVENT_WITHIN_24S_EXPLICIT_BREAK',
            event_rule='explicit_break_duration',
            inferred_block_position=np.nan,
            estimated_next_event_gap_sec=float(event_gap)
        )
    if next_gap is not None and np.isfinite(next_gap):
        clean = float(next_gap) > LATE_END + 1e-09
        return dict(
            event_clean=bool(clean),
            event_status='SAFE_EXPLICIT_NEXT_EVENT_GAP' if clean else 'EXCLUDE_EVENT_WITHIN_24S_EXPLICIT_GAP',
            event_rule='explicit_next_event_gap',
            inferred_block_position=np.nan,
            estimated_next_event_gap_sec=float(next_gap)
        )
    if participant_n_trials == EXPECTED_COMPLETE_STIMULI:
        pos1 = int(sequential_index) % STIMULI_PER_BLOCK + 1
        block1 = int(sequential_index) // STIMULI_PER_BLOCK + 1
        if pos1 == 4:
            return dict(
                event_clean=False,
                event_status='EXCLUDE_PREQUESTION_COMPLETE_BLOCK',
                event_rule='complete_40_trial_10x4_design',
                inferred_block_position=pos1,
                inferred_block=block1,
                estimated_next_event_gap_sec=STIMULUS_DURATION_SEC + 10.0
            )
        return dict(
            event_clean=True,
            event_status='SAFE_COMPLETE_BLOCK_NONQUESTION',
            event_rule='complete_40_trial_10x4_design',
            inferred_block_position=pos1,
            inferred_block=block1,
            estimated_next_event_gap_sec=STIMULUS_DURATION_SEC + 20.0
        )
    return dict(
        event_clean=False,
        event_status='UNKNOWN_EVENT_SPACING_EXCLUDED',
        event_rule='insufficient_metadata_in_incomplete_participant',
        inferred_block_position=np.nan,
        estimated_next_event_gap_sec=np.nan
    )
def robust_scale_channel(x: np.ndarray, baseline_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    HemoBoundary-style trial/channel baseline normalization.
    x is [channel, time].
    """
    x = np.asarray(x, float)
    base = x[:, baseline_mask]
    med = np.nanmedian(base, axis=1, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(base - med), axis=1, keepdims=True)
    sd = np.nanstd(base, axis=1, keepdims=True)
    scale = np.where(np.isfinite(mad) & (mad > 1e-08), mad, np.where(np.isfinite(sd) & (sd > 1e-08), sd, 1.0))
    z = (x - med) / scale
    return (z, med.reshape(-1), scale.reshape(-1))
def first_sustained_run(mask: np.ndarray, times: np.ndarray, start_index: int, sustain_sec: float, max_confirm_time: float) -> Tuple[float, float, int]:
    """
    Return (run_start_time, confirmation_time, n_samples) for the first
    qualifying consecutive True run after start_index.

    Confirmation time is the final sample needed to establish the sustained run.
    """
    mask = np.asarray(mask, bool)
    times = np.asarray(times, float)
    if len(times) < 2:
        return (np.nan, np.nan, 0)
    dt = float(np.median(np.diff(times)))
    n_need = max(1, int(math.ceil(float(sustain_sec) / dt - 1e-12)))
    i = max(0, int(start_index))
    while i < len(mask):
        if not mask[i] or times[i] > max_confirm_time + 1e-09:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        run_len = j - i
        if run_len >= n_need:
            confirm_idx = i + n_need - 1
            if times[confirm_idx] <= max_confirm_time + 1e-09:
                return (float(times[i]), float(times[confirm_idx]), int(n_need))
        i = max(j, i + 1)
    return (np.nan, np.nan, int(n_need))
def analyse_trial(participant: str, trial_index: int, times: np.ndarray, hbo: np.ndarray, hbr: np.ndarray) -> Dict[str, Any]:
    times = np.asarray(times, float)
    H = np.asarray(hbo, float)
    R = np.asarray(hbr, float)
    C = min(H.shape[0], R.shape[0])
    H = H[:C]
    R = R[:C]
    bm = (times >= BASELINE_START - 1e-09) & (times < BASELINE_END - 1e-09)
    if int(np.sum(bm)) < MIN_BASELINE_SAMPLES:
        return dict(signal_usable=False, signal_status='INSUFFICIENT_BASELINE_SAMPLES')
    baseline_finite = np.mean(np.isfinite(H[:, bm]) & np.isfinite(R[:, bm]), axis=1)
    full_finite = np.mean(np.isfinite(H) & np.isfinite(R), axis=1)
    Hbase = H[:, bm]
    Rbase = R[:, bm]
    Hsd = np.nanstd(Hbase, axis=1)
    Rsd = np.nanstd(Rbase, axis=1)
    valid_ch = (baseline_finite >= MIN_FINITE_FRACTION) & (full_finite >= MIN_FINITE_FRACTION) & np.isfinite(Hsd) & np.isfinite(Rsd)
    if int(np.sum(valid_ch)) < MIN_VALID_CHANNELS:
        return dict(signal_usable=False, signal_status='TOO_FEW_VALID_CHANNELS', n_valid_channels=int(np.sum(valid_ch)), n_channels_before=int(C))
    H = H[valid_ch]
    R = R[valid_ch]
    Hz, _, _ = robust_scale_channel(H, bm)
    Rz, _, _ = robust_scale_channel(R, bm)
    ghbo = np.nanmedian(Hz, axis=0)
    ghbr = np.nanmedian(Rz, axis=0)
    anti = 0.5 * (ghbo - ghbr)
    finite = np.isfinite(anti)
    if np.mean(finite) < 0.95:
        return dict(signal_usable=False, signal_status='NONFINITE_TRIAL_CONTRAST', n_valid_channels=int(np.sum(valid_ch)), n_channels_before=int(C))
    baseline_values = anti[bm & finite]
    if len(baseline_values) < MIN_BASELINE_SAMPLES:
        return dict(signal_usable=False, signal_status='INSUFFICIENT_FINITE_BASELINE_CONTRAST', n_valid_channels=int(np.sum(valid_ch)), n_channels_before=int(C))
    baseline_center = float(np.median(baseline_values))
    baseline_low = float(np.quantile(baseline_values, BASELINE_Q_LOW))
    baseline_high = float(np.quantile(baseline_values, BASELINE_Q_HIGH))
    task_mask = (times >= 0.0 - 1e-09) & (times <= PRIMARY_END + 1e-09) & finite
    task_idx = np.flatnonzero(task_mask)
    if len(task_idx) == 0:
        return dict(signal_usable=False, signal_status='NO_0_TO_20_WINDOW', n_valid_channels=int(np.sum(valid_ch)), n_channels_before=int(C))
    peak_idx = int(task_idx[np.argmax(anti[task_idx])])
    peak_time = float(times[peak_idx])
    peak_value = float(anti[peak_idx])
    peak_excursion = float(peak_value - baseline_center)
    response_evaluable = bool(np.isfinite(peak_value) and peak_value > baseline_high and (peak_excursion > 0.0))
    if not response_evaluable:
        return dict(
            signal_usable=True,
            signal_status='NO_POSITIVE_EXCURSION_ABOVE_BASELINE_ENVELOPE',
            response_evaluable=False,
            n_valid_channels=int(np.sum(valid_ch)),
            n_channels_before=int(C),
            baseline_center=baseline_center,
            baseline_low=baseline_low,
            baseline_high=baseline_high,
            peak_time_sec=peak_time,
            peak_value=peak_value,
            peak_excursion=peak_excursion
        )
    inside = finite & (anti >= baseline_low) & (anti <= baseline_high)
    p_start, p_confirm, n_need = first_sustained_run(inside, times, start_index=peak_idx + 1, sustain_sec=SUSTAIN_SEC, max_confirm_time=LATE_END)
    secondary_threshold = baseline_center + SECONDARY_PEAK_FRACTION * peak_excursion
    below20 = finite & (anti <= secondary_threshold)
    s_start, s_confirm, _ = first_sustained_run(below20, times, start_index=peak_idx + 1, sustain_sec=SUSTAIN_SEC, max_confirm_time=LATE_END)
    primary_recovered_by20 = bool(np.isfinite(p_confirm) and p_confirm <= PRIMARY_END + 1e-09)
    primary_late_recovery = bool(np.isfinite(p_confirm) and p_confirm > PRIMARY_END + 1e-09 and (p_confirm <= LATE_END + 1e-09))
    primary_unresolved20 = bool(not primary_recovered_by20)
    primary_persistent24 = bool(primary_unresolved20 and (not primary_late_recovery))
    secondary_recovered_by20 = bool(np.isfinite(s_confirm) and s_confirm <= PRIMARY_END + 1e-09)
    secondary_late_recovery = bool(np.isfinite(s_confirm) and s_confirm > PRIMARY_END + 1e-09 and (s_confirm <= LATE_END + 1e-09))
    secondary_unresolved20 = bool(not secondary_recovered_by20)
    secondary_persistent24 = bool(secondary_unresolved20 and (not secondary_late_recovery))
    consensus_late = bool(primary_unresolved20 and secondary_unresolved20 and primary_late_recovery and secondary_late_recovery)
    return dict(
        signal_usable=True,
        signal_status='OK',
        response_evaluable=True,
        n_valid_channels=int(np.sum(valid_ch)),
        n_channels_before=int(C),
        baseline_center=baseline_center,
        baseline_low=baseline_low,
        baseline_high=baseline_high,
        peak_time_sec=peak_time,
        peak_value=peak_value,
        peak_excursion=peak_excursion,
        sustain_samples=int(n_need),
        primary_recovery_start_sec=p_start,
        primary_recovery_confirm_sec=p_confirm,
        primary_recovered_by20=primary_recovered_by20,
        primary_unresolved_at20=primary_unresolved20,
        primary_late_recovery_20_24=primary_late_recovery,
        primary_persistent_through24=primary_persistent24,
        secondary_threshold=secondary_threshold,
        secondary_recovery_start_sec=s_start,
        secondary_recovery_confirm_sec=s_confirm,
        secondary_recovered_by20=secondary_recovered_by20,
        secondary_unresolved_at20=secondary_unresolved20,
        secondary_late_recovery_20_24=secondary_late_recovery,
        secondary_persistent_through24=secondary_persistent24,
        consensus_late_recovery_20_24=consensus_late
    )
def safe_fraction(num: int, den: int) -> float:
    return float(num / den) if den > 0 else np.nan
def participant_summary(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, q0 in trials.groupby('participant', sort=True):
        q = q0[q0['event_clean'].astype(bool) & q0['signal_usable'].astype(bool) & q0['response_evaluable'].astype(bool)].copy()
        n = len(q)
        u20 = int(q['primary_unresolved_at20'].astype(bool).sum()) if n else 0
        late = int(q['primary_late_recovery_20_24'].astype(bool).sum()) if n else 0
        pers = int(q['primary_persistent_through24'].astype(bool).sum()) if n else 0
        s_u20 = int(q['secondary_unresolved_at20'].astype(bool).sum()) if n else 0
        s_late = int(q['secondary_late_recovery_20_24'].astype(bool).sum()) if n else 0
        s_pers = int(q['secondary_persistent_through24'].astype(bool).sum()) if n else 0
        consensus = int(q['consensus_late_recovery_20_24'].astype(bool).sum()) if n else 0
        late_times = pd.to_numeric(q.loc[q['primary_late_recovery_20_24'].astype(bool), 'primary_recovery_start_sec'], errors='coerce').dropna().to_numpy(float)
        late_confirm = pd.to_numeric(q.loc[q['primary_late_recovery_20_24'].astype(bool), 'primary_recovery_confirm_sec'], errors='coerce').dropna().to_numpy(float)
        rows.append(dict(
            participant=pid,
            n_all_trials=int(len(q0)),
            n_event_clean=int(q0['event_clean'].astype(bool).sum()),
            n_signal_usable_clean=int((q0['event_clean'].astype(bool) & q0['signal_usable'].astype(bool)).sum()),
            n_recovery_evaluable=int(n),
            n_primary_unresolved_at20=u20,
            primary_unresolved20_fraction=safe_fraction(u20, n),
            n_primary_late_recovery=late,
            primary_late_recovery_fraction_among_unresolved20=safe_fraction(late, u20),
            n_primary_persistent_through24=pers,
            primary_persistent24_fraction_among_unresolved20=safe_fraction(pers, u20),
            n_secondary_unresolved_at20=s_u20,
            secondary_unresolved20_fraction=safe_fraction(s_u20, n),
            n_secondary_late_recovery=s_late,
            secondary_late_recovery_fraction_among_unresolved20=safe_fraction(s_late, s_u20),
            n_secondary_persistent_through24=s_pers,
            secondary_persistent24_fraction_among_unresolved20=safe_fraction(s_pers, s_u20),
            n_consensus_late_recovery=consensus,
            consensus_late_recovery_fraction_among_primary_unresolved20=safe_fraction(consensus, u20),
            median_primary_late_recovery_start_sec=float(np.median(late_times)) if len(late_times) else np.nan,
            median_primary_late_recovery_confirm_sec=float(np.median(late_confirm)) if len(late_confirm) else np.nan
        ))
    return pd.DataFrame(rows)
def bootstrap_mean_ci(values: Sequence[float], seed: int) -> Tuple[float, float, float]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(x))
    if len(x) == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    boot = np.empty(BOOT_REPS, float)
    done = 0
    while done < BOOT_REPS:
        n = min(2000, BOOT_REPS - done)
        idx = rng.integers(0, len(x), size=(n, len(x)))
        boot[done:done + n] = np.mean(x[idx], axis=1)
        done += n
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return (mean, float(lo), float(hi))
def bootstrap_median_ci(values: Sequence[float], seed: int) -> Tuple[float, float, float]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    med = float(np.median(x))
    if len(x) == 1:
        return (med, med, med)
    rng = np.random.default_rng(seed)
    boot = np.empty(BOOT_REPS, float)
    done = 0
    while done < BOOT_REPS:
        n = min(1000, BOOT_REPS - done)
        idx = rng.integers(0, len(x), size=(n, len(x)))
        boot[done:done + n] = np.median(x[idx], axis=1)
        done += n
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return (med, float(lo), float(hi))
def paper_table(participant_df: pd.DataFrame, trial_df: pd.DataFrame) -> pd.DataFrame:
    p = participant_df.copy()
    unresolved = pd.to_numeric(p['primary_unresolved20_fraction'], errors='coerce').to_numpy(float)
    late_frac = pd.to_numeric(p['primary_late_recovery_fraction_among_unresolved20'], errors='coerce').to_numpy(float)
    persistent_frac = pd.to_numeric(p['primary_persistent24_fraction_among_unresolved20'], errors='coerce').to_numpy(float)
    sec_late = pd.to_numeric(p['secondary_late_recovery_fraction_among_unresolved20'], errors='coerce').to_numpy(float)
    consensus = pd.to_numeric(p['consensus_late_recovery_fraction_among_primary_unresolved20'], errors='coerce').to_numpy(float)
    med_u, med_u_lo, med_u_hi = bootstrap_median_ci(unresolved, BOOT_SEED + 1)
    mean_l, mean_l_lo, mean_l_hi = bootstrap_mean_ci(late_frac, BOOT_SEED + 2)
    mean_p, mean_p_lo, mean_p_hi = bootstrap_mean_ci(persistent_frac, BOOT_SEED + 3)
    mean_s, mean_s_lo, mean_s_hi = bootstrap_mean_ci(sec_late, BOOT_SEED + 4)
    mean_c, mean_c_lo, mean_c_hi = bootstrap_mean_ci(consensus, BOOT_SEED + 5)
    eligible = trial_df[trial_df['event_clean'].astype(bool) & trial_df['signal_usable'].astype(bool) & trial_df['response_evaluable'].astype(bool)].copy()
    late_trials = eligible[eligible['primary_late_recovery_20_24'].astype(bool)].copy()
    late_start = pd.to_numeric(late_trials['primary_recovery_start_sec'], errors='coerce').dropna().to_numpy(float)
    late_confirm = pd.to_numeric(late_trials['primary_recovery_confirm_sec'], errors='coerce').dropna().to_numpy(float)
    n_participants_late = int(p['n_primary_late_recovery'].gt(0).sum())
    rows = [
        dict(outcome='Censored / unresolved at +20 s', estimate=med_u, ci95_low=med_u_lo, ci95_high=med_u_hi, summary='Median participant proportion', denominator='Event-clean, signal-usable, recovery-evaluable Bálint trials'),
        dict(outcome='Recovered between +20 and +24 s — primary baseline-envelope rule', estimate=mean_l, ci95_low=mean_l_lo, ci95_high=mean_l_hi, summary='Mean participant proportion', denominator='Primary +20-s unresolved subset'),
        dict(outcome='Still unresolved at +24 s — primary baseline-envelope rule', estimate=mean_p, ci95_low=mean_p_lo, ci95_high=mean_p_hi, summary='Mean participant proportion', denominator='Primary +20-s unresolved subset'),
        dict(outcome='Recovered between +20 and +24 s — secondary 20%-of-peak rule', estimate=mean_s, ci95_low=mean_s_lo, ci95_high=mean_s_hi, summary='Mean participant proportion', denominator='Secondary +20-s unresolved subset'),
        dict(outcome='Consensus late recovery — both rules', estimate=mean_c, ci95_low=mean_c_lo, ci95_high=mean_c_hi, summary='Mean participant proportion', denominator='Primary +20-s unresolved subset'),
        dict(outcome='Participants with observed primary late recovery', estimate=float(n_participants_late), ci95_low=np.nan, ci95_high=np.nan, summary='Count', denominator=f"Bálint participants with analyzable clean trials (n={p['participant'].nunique()})"),
        dict(outcome='Median primary late-recovery start time', estimate=float(np.median(late_start)) if len(late_start) else np.nan, ci95_low=np.nan, ci95_high=np.nan, summary='Seconds; pooled late-recovery trials', denominator='Observed primary late-recovery trials'),
        dict(outcome='Median primary late-recovery confirmation time', estimate=float(np.median(late_confirm)) if len(late_confirm) else np.nan, ci95_low=np.nan, ci95_high=np.nan, summary='Seconds; pooled late-recovery trials', denominator='Observed primary late-recovery trials')
    ]
    return pd.DataFrame(rows)
def protected_self_tests() -> None:
    t = np.arange(-5.0, 24.0 + 0.05, 0.1)
    rng = np.random.default_rng(123)
    base = 0.05 * rng.normal(size=len(t))
    resp = np.zeros(len(t))
    m1 = (t >= 2) & (t < 7)
    resp[m1] = np.linspace(0, 3, int(m1.sum()))
    m2 = (t >= 7) & (t < 18)
    resp[m2] = 3.0
    m3 = (t >= 18) & (t < 21.2)
    resp[m3] = np.linspace(3, 0, int(m3.sum()))
    anti = base + resp
    H = np.vstack([anti, anti * 0.95, anti * 1.05])
    R = -H
    H[:, t < 0] += 0.1 * rng.normal(size=(3, int((t < 0).sum())))
    R[:, t < 0] += 0.1 * rng.normal(size=(3, int((t < 0).sum())))
    row = analyse_trial('TEST', 0, t, H, R)
    assert row['signal_usable']
    assert row['response_evaluable']
    assert row['primary_unresolved_at20']
    assert row['primary_late_recovery_20_24']
    assert not row['primary_persistent_through24']
    safe = event_clean_decision(40, 0, {})
    unsafe = event_clean_decision(40, 3, {})
    assert safe['event_clean']
    assert not unsafe['event_clean']
    print('Bálint recovery checks: PASS')
import io
import zipfile
import warnings
from scipy import signal as sp_signal
from scipy.io import loadmat, savemat
COMBINED_VERSION = 'SHIN_BALINT_HARMONIZED_LATE_WINDOW_V2_20260922'
DEFAULT_SHIN_ROOT = Path('/content/drive/MyDrive/HonorProject/shin2018_word_generation_state_preparation_v1')
DEFAULT_BALINT_H5 = DEFAULT_H5
DEFAULT_COMBINED_OUTPUT = DEFAULT_SHIN_ROOT / '107_SHIN_BALINT_HARMONIZED_LATE_WINDOW_RECOVERY_V2'
SHIN_PIDS = tuple((f'VP{i:03d}' for i in range(1, 27)))
COMMON_GRID_HZ = 10.0
COMMON_TIMES = np.arange(BASELINE_START, LATE_END + 0.5 / COMMON_GRID_HZ, 1.0 / COMMON_GRID_HZ)
SHIN_LOWPASS_HZ = 0.2
SHIN_LOWPASS_ORDER = 3

def combined_parse_args(argv: Optional[Sequence[str]]=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Harmonized real-physiology +20 to +24 s recovery analysis for SHIN and Bálint.')
    p.add_argument('--shin-root', default=str(DEFAULT_SHIN_ROOT))
    p.add_argument(
        '--shin-nirs-dir',
        default=None,
        help='Exact directory containing VP001-NIRS ... VP026-NIRS folders or zips. If omitted, only a short deterministic list below the SHIN root is checked.'
    )
    p.add_argument('--balint-h5', default=str(DEFAULT_BALINT_H5))
    p.add_argument('--output', default=str(DEFAULT_COMBINED_OUTPUT))
    p.add_argument('--self-test-only', action='store_true')
    raw = list(sys.argv[1:] if argv is None else argv)
    cleaned: List[str] = []
    ignored: List[str] = []
    i = 0
    while i < len(raw):
        if raw[i] == '-f' and i + 1 < len(raw):
            ignored.extend(raw[i:i + 2])
            i += 2
        elif raw[i].startswith('--f='):
            ignored.append(raw[i])
            i += 1
        else:
            cleaned.append(raw[i])
            i += 1
    if ignored:
        print('Ignoring notebook/runtime arguments:', ignored)
    return p.parse_args(cleaned)
def drive_alias(path: Path) -> Path:
    """Compatibility helper retained for old call sites."""
    return Path(path)
def resolve_existing_exact(path: Path, kind: str='file') -> Path:
    """
    Resolve only the exact supplied path.

    If a /content/drive path is unexpectedly unreadable/missing, force-remount
    Google Drive once and retry. No recursive search and no /content/gdrive alias.
    """
    path = Path(path)
    checker = _safe_is_file if kind == 'file' else _safe_is_dir
    if checker(path):
        return path
    if str(path).startswith('/content/drive/'):
        print('[Drive] Exact path was not readable on first check. Force-remounting /content/drive once...')
        try_mount_drive(force=True)
        if checker(path):
            return path
    raise FileNotFoundError(f'Exact {kind} path does not exist or is not readable after Drive remount:\\n  {path}')
def has_shin_pid_source(base: Path, pid: str='VP001') -> bool:
    return _safe_is_dir(base / f'{pid}-NIRS') or _safe_is_file(base / f'{pid}-NIRS.zip')
def resolve_shin_nirs_dir(shin_root: Path, explicit: Optional[str]) -> Path:
    if explicit:
        p = resolve_existing_exact(Path(explicit), 'dir')
        if not has_shin_pid_source(p):
            raise FileNotFoundError(f'--shin-nirs-dir exists but {p} does not contain VP001-NIRS/ or VP001-NIRS.zip.')
        return p
    roots = [shin_root]
    candidates: List[Path] = []
    for root in roots:
        candidates.extend([root / 'rawdata' / 'NIRS', root / 'NIRS', root / 'raw' / 'NIRS', root / 'rawdata', root / 'raw', root])
    seen = set()
    checked = []
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        checked.append(key)
        if _safe_is_dir(p) and has_shin_pid_source(p):
            return p
    raise FileNotFoundError('Could not locate the SHIN MATLAB-compatible NIRS participant files using the deterministic direct-path checks. No recursive search was performed.\nChecked:\n  - ' + '\n  - '.join(checked) + '\nRun again with, for example:\n  %run /content/SHIN_BALINT_HARMONIZED_LATE_WINDOW_V2.py --shin-nirs-dir /content/drive/MyDrive/<YOUR_EXACT_NIRS_DIRECTORY>')
def _mat_public_variables(d: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if not str(k).startswith('__')}
def _load_mat_bytes(data: bytes) -> Dict[str, Any]:
    return _mat_public_variables(loadmat(io.BytesIO(data), simplify_cells=True))
def _load_mat_path(path: Path) -> Dict[str, Any]:
    return _mat_public_variables(loadmat(str(path), simplify_cells=True))
def _single_mat_variable(d: Mapping[str, Any], preferred: str) -> Any:
    if preferred in d:
        return d[preferred]
    if len(d) == 1:
        return next(iter(d.values()))
    matches = [v for k, v in d.items() if preferred.lower() in str(k).lower()]
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(f"MAT file did not contain an unambiguous '{preferred}' variable. Available: {sorted(map(str, d.keys()))}")
def _zip_member_by_basename(z: zipfile.ZipFile, basename: str) -> str:
    matches = [n for n in z.namelist() if Path(n).name.lower() == basename.lower()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one '{basename}' in {z.filename}; found {matches}.")
    return matches[0]
def load_shin_matlab_source(source: Path) -> Tuple[Any, Any, Dict[str, Any]]:
    name_pairs = [('cnt_wg.mat', 'mrk_wg.mat', 'cnt_wg', 'mrk_wg'), ('cnt_wc.mat', 'mrk_wc.mat', 'cnt_wc', 'mrk_wc')]
    if source.is_file() and source.suffix.lower() == '.zip':
        with zipfile.ZipFile(source, 'r') as z:
            names = {Path(n).name.lower(): n for n in z.namelist()}
            chosen = None
            for cnt_name, mrk_name, cnt_var, mrk_var in name_pairs:
                if cnt_name.lower() in names and mrk_name.lower() in names:
                    chosen = (names[cnt_name.lower()], names[mrk_name.lower()], cnt_var, mrk_var)
                    break
            if chosen is None:
                raise RuntimeError(f'{source}: expected either cnt_wg/mrk_wg or cnt_wc/mrk_wc MAT files. Available basenames include: {sorted(names)[:40]}')
            cnt_member, mrk_member, cnt_var, mrk_var = chosen
            cnt_d = _load_mat_bytes(z.read(cnt_member))
            mrk_d = _load_mat_bytes(z.read(mrk_member))
        audit = dict(source=str(source), source_type='zip', cnt_member=cnt_member, mrk_member=mrk_member)
    elif source.is_dir():
        chosen = None
        for cnt_name, mrk_name, cnt_var, mrk_var in name_pairs:
            cnt_path = source / cnt_name
            mrk_path = source / mrk_name
            if cnt_path.is_file() and mrk_path.is_file():
                chosen = (cnt_path, mrk_path, cnt_var, mrk_var)
                break
        if chosen is None:
            raise FileNotFoundError(f'{source}: expected exact files cnt_wg.mat + mrk_wg.mat, or cnt_wc.mat + mrk_wc.mat.')
        cnt_path, mrk_path, cnt_var, mrk_var = chosen
        cnt_d = _load_mat_path(cnt_path)
        mrk_d = _load_mat_path(mrk_path)
        audit = dict(source=str(source), source_type='directory', cnt_member=str(cnt_path), mrk_member=str(mrk_path))
    else:
        raise FileNotFoundError(source)
    cnt = _single_mat_variable(cnt_d, cnt_var)
    mrk = _single_mat_variable(mrk_d, mrk_var)
    return (cnt, mrk, audit)
def _field(obj: Any, name: str, default: Any=None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
        low = {str(k).lower(): k for k in obj.keys()}
        if name.lower() in low:
            return obj[low[name.lower()]]
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.void) and obj.dtype.names and (name in obj.dtype.names):
        return obj[name]
    return default
def _decode_mat_strings(value: Any) -> List[str]:
    if value is None:
        return []
    a = np.asarray(value, dtype=object).reshape(-1)
    out = []
    for x in a:
        if isinstance(x, bytes):
            out.append(x.decode('utf-8', errors='replace'))
        elif isinstance(x, np.ndarray) and x.dtype.kind in ('U', 'S'):
            out.append(''.join(map(str, x.reshape(-1))).strip())
        else:
            out.append(str(x))
    return out
def _time_channel_matrix(x: Any, label: str) -> np.ndarray:
    a = np.asarray(x, float)
    a = np.squeeze(a)
    if a.ndim != 2:
        raise RuntimeError(f'{label}: expected 2D continuous array, got {a.shape}.')
    if a.shape[0] < a.shape[1] and a.shape[0] <= 256:
        a = a.T
    if a.shape[0] < 1000:
        raise RuntimeError(f'{label}: continuous time dimension looks too short: {a.shape}.')
    return a
def extract_shin_cnt(cnt: Any) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    oxy = _field(cnt, 'oxy')
    deoxy = _field(cnt, 'deoxy')
    if oxy is None or deoxy is None:
        raise RuntimeError("cnt_wg must contain separate 'oxy' and 'deoxy' fields.")
    hbo = _time_channel_matrix(_field(oxy, 'x'), 'cnt_wg.oxy.x')
    hbr = _time_channel_matrix(_field(deoxy, 'x'), 'cnt_wg.deoxy.x')
    fs_o = float(np.asarray(_field(oxy, 'fs'), float).reshape(-1)[0])
    fs_r = float(np.asarray(_field(deoxy, 'fs'), float).reshape(-1)[0])
    if not np.isfinite(fs_o) or not np.isfinite(fs_r) or fs_o <= 0 or (fs_r <= 0):
        raise RuntimeError('Invalid SHIN NIRS sampling rate.')
    if abs(fs_o - fs_r) > 1e-06:
        raise RuntimeError(f'HbO/HbR sampling-rate mismatch: {fs_o} vs {fs_r}.')
    if hbo.shape[0] != hbr.shape[0]:
        raise RuntimeError(f'HbO/HbR time-length mismatch: {hbo.shape} vs {hbr.shape}.')
    c = min(hbo.shape[1], hbr.shape[1])
    hbo = hbo[:, :c]
    hbr = hbr[:, :c]
    return (hbo, hbr, fs_o, dict(n_samples=int(hbo.shape[0]), n_channels=int(c), fs_hz=float(fs_o)))
def extract_shin_markers(mrk: Any) -> Dict[str, Any]:
    times_ms = np.asarray(_field(mrk, 'time'), float).reshape(-1)
    y = np.asarray(_field(mrk, 'y'), float)
    class_names = _decode_mat_strings(_field(mrk, 'className'))
    if times_ms.ndim != 1 or len(times_ms) == 0 or (not np.isfinite(times_ms).all()):
        raise RuntimeError('Invalid mrk_wg.time.')
    if np.any(np.diff(times_ms) < 0):
        order = np.argsort(times_ms, kind='stable')
        times_ms = times_ms[order]
    else:
        order = np.arange(len(times_ms))
    n = len(times_ms)
    if y.ndim == 1:
        if len(y) != n:
            raise RuntimeError(f'mrk_wg.y shape {y.shape} incompatible with {n} events.')
        class_index = y[order].astype(int)
    elif y.ndim == 2:
        if y.shape[1] == n:
            yy = y[:, order]
            class_index = np.argmax(yy, axis=0)
        elif y.shape[0] == n:
            yy = y[order, :]
            class_index = np.argmax(yy, axis=1)
        else:
            raise RuntimeError(f'mrk_wg.y shape {y.shape} incompatible with {n} events.')
    else:
        raise RuntimeError(f'Unsupported mrk_wg.y shape {y.shape}.')
    wg_class = None
    for i, name in enumerate(class_names):
        key = re.sub('[^a-z0-9]+', '', name.lower())
        if key in {'wg', 'wordgeneration', 'wordgenerating', 'verbalfluency'} or 'word' in key:
            wg_class = i
            break
    if wg_class is None:
        wg_class = 0
    times_sec = times_ms / 1000.0
    if len(times_sec) >= 2:
        median_gap = float(np.median(np.diff(times_sec)))
        if not 5.0 <= median_gap <= 120.0:
            raise RuntimeError(f'Unexpected SHIN marker spacing after ms->s conversion: median={median_gap:.3f}s. Expected BBCI mrk.time in milliseconds.')
    wg_mask = class_index == int(wg_class)
    if not 20 <= int(wg_mask.sum()) <= 40:
        raise RuntimeError(f'Expected about 30 WG events; identified {int(wg_mask.sum())}. className={class_names}, wg_class_index={wg_class}.')
    return dict(times_sec=times_sec, class_index=class_index, class_names=class_names, wg_class_index=int(wg_class), wg_mask=wg_mask)
def _fill_and_lowpass_continuous(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan, dtype=float)
    valid = np.zeros(x.shape[1], dtype=bool)
    sos = sp_signal.butter(SHIN_LOWPASS_ORDER, SHIN_LOWPASS_HZ, btype='low', fs=float(fs), output='sos')
    idx = np.arange(x.shape[0], dtype=float)
    for c in range(x.shape[1]):
        a = x[:, c]
        fin = np.isfinite(a)
        if fin.mean() < MIN_FINITE_FRACTION or int(fin.sum()) < 50:
            continue
        if not fin.all():
            a = np.interp(idx, idx[fin], a[fin])
        try:
            out[:, c] = sp_signal.sosfiltfilt(sos, a)
            valid[c] = np.isfinite(out[:, c]).mean() >= MIN_FINITE_FRACTION
        except ValueError:
            continue
    return (out[:, valid], valid)
def _interpolate_epoch(continuous: np.ndarray, fs: float, onset_sec: float, rel_times: np.ndarray) -> np.ndarray:
    q = onset_sec + np.asarray(rel_times, float)
    total_end = (continuous.shape[0] - 1) / float(fs)
    if q[0] < -1e-09 or q[-1] > total_end + 1e-09:
        raise ValueError('epoch outside continuous recording')
    source_t = np.arange(continuous.shape[0], dtype=float) / float(fs)
    return np.vstack([np.interp(q, source_t, continuous[:, c]) for c in range(continuous.shape[1])])
def resolve_shin_participant_source(base: Path, pid: str) -> Path:
    folder = base / f'{pid}-NIRS'
    zipped = base / f'{pid}-NIRS.zip'
    if folder.is_dir():
        return folder
    if zipped.is_file():
        return zipped
    raise FileNotFoundError(f'Missing exact SHIN source for {pid}. Expected either:\n  {folder}\n  {zipped}')
def process_shin(shin_nirs_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    input_rows: List[Dict[str, Any]] = []
    for pi, pid in enumerate(SHIN_PIDS, 1):
        source = resolve_shin_participant_source(shin_nirs_dir, pid)
        cnt, mrk, source_audit = load_shin_matlab_source(source)
        hbo_raw, hbr_raw, fs, cnt_audit = extract_shin_cnt(cnt)
        marker = extract_shin_markers(mrk)
        hbo, valid_o = _fill_and_lowpass_continuous(hbo_raw, fs)
        hbr, valid_r = _fill_and_lowpass_continuous(hbr_raw, fs)
        common = valid_o & valid_r
        if int(common.sum()) < MIN_VALID_CHANNELS:
            raise RuntimeError(f'{pid}: too few channels survive continuous low-pass QC.')
        hbo_all, _ = _fill_and_lowpass_continuous(hbo_raw[:, common], fs)
        hbr_all, _ = _fill_and_lowpass_continuous(hbr_raw[:, common], fs)
        if hbo_all.shape[1] != hbr_all.shape[1]:
            raise RuntimeError(f'{pid}: paired HbO/HbR channel count mismatch after QC.')
        all_times = np.asarray(marker['times_sec'], float)
        wg_indices = np.flatnonzero(marker['wg_mask'])
        total_end = (hbo_all.shape[0] - 1) / float(fs)
        input_rows.append(dict(
            dataset='SHIN',
            participant=pid,
            **source_audit,
            **cnt_audit,
            n_common_filtered_channels=int(hbo_all.shape[1]),
            n_all_markers=int(len(all_times)),
            n_wg_markers=int(len(wg_indices)),
            class_names='|'.join(marker['class_names']),
            wg_class_index=int(marker['wg_class_index']),
            continuous_end_sec=float(total_end)
        ))
        clean = usable = unresolved = late = 0
        for local_idx, ei in enumerate(wg_indices):
            onset = float(all_times[ei])
            next_gap = float(all_times[ei + 1] - onset) if ei + 1 < len(all_times) else np.nan
            coverage_ok = onset + LATE_END <= total_end + 1e-09
            spacing_ok = bool(not np.isfinite(next_gap) or next_gap > LATE_END + 1e-09)
            event_clean = bool(coverage_ok and spacing_ok)
            if not coverage_ok:
                status = 'EXCLUDE_RECORDING_END_BEFORE_24S'
            elif not spacing_ok:
                status = 'EXCLUDE_NEXT_TRIAL_WITHIN_24S'
            else:
                status = 'SAFE_NEXT_TRIAL_AFTER_24S'
            erow = dict(
                dataset='SHIN',
                participant=pid,
                sequential_trial_index=int(local_idx),
                marker_event_index=int(ei),
                trial_onset_sec=onset,
                next_any_event_gap_sec=next_gap,
                continuous_coverage_to_24s=coverage_ok,
                event_clean=event_clean,
                event_status=status,
                event_rule='next_any_datasetC_marker_and_recording_coverage'
            )
            event_rows.append(erow)
            if event_clean:
                clean += 1
                try:
                    H = _interpolate_epoch(hbo_all, fs, onset, COMMON_TIMES)
                    R = _interpolate_epoch(hbr_all, fs, onset, COMMON_TIMES)
                    sig = analyse_trial(pid, local_idx, COMMON_TIMES, H, R)
                except Exception as exc:
                    sig = dict(signal_usable=False, signal_status=f'EPOCH_ANALYSIS_ERROR:{type(exc).__name__}:{exc}')
            else:
                sig = dict(signal_usable=False, signal_status='EVENT_WINDOW_NOT_CLEAN')
            row = dict(erow)
            row.update(sig)
            all_rows.append(row)
            if event_clean and bool(sig.get('signal_usable', False)) and bool(sig.get('response_evaluable', False)):
                usable += 1
                unresolved += int(bool(sig.get('primary_unresolved_at20', False)))
                late += int(bool(sig.get('primary_late_recovery_20_24', False)))
        print(f'[SHIN {pi:02d}/26] {pid}: WG={len(wg_indices)} | clean={clean} | recovery-evaluable={usable} | unresolved@20={unresolved} | late-rescue={late}', flush=True)
    return (pd.DataFrame(all_rows), pd.DataFrame(event_rows), pd.DataFrame(input_rows))
def process_balint(h5_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_trial_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    input_audit_rows: List[Dict[str, Any]] = []
    with h5py.File(h5_path, 'r') as h5:
        pids = participant_groups(h5)
        if not pids:
            raise RuntimeError('No Cxx Bálint participant groups were found.')
        if len(pids) != EXPECTED_PARTICIPANTS:
            print(f'[WARN] Expected 26 Bálint participants, found {len(pids)}: {pids}')
        for pi, pid in enumerate(pids, 1):
            data = load_participant(h5[pid])
            times = np.asarray(data['times'], float)
            H = np.asarray(data['hbo'], float)
            R = np.asarray(data['hbr'], float)
            onsets = np.asarray(data['trial_onset_sec'], float)
            event_json = np.asarray(data['event_row_json'], str)
            if times[0] > BASELINE_START + 0.11 or times[-1] < LATE_END - 0.05:
                raise RuntimeError(f'{pid}: prepared Bálint epoch does not cover -5 to +24 s: {times[0]:.3f} to {times[-1]:.3f}.')
            input_audit_rows.append(dict(dataset='BALINT', participant=pid, **data['audit']))
            n_trials = len(H)
            clean = usable = unresolved = late = 0
            for ti in range(n_trials):
                meta = extract_event_fields(event_json[ti], current_onset=onsets[ti])
                decision = event_clean_decision(n_trials, ti, meta)
                erow = dict(
                    dataset='BALINT',
                    participant=pid,
                    sequential_trial_index=int(ti),
                    original_trial_index=int(data['original_index'][ti]),
                    trial_onset_sec=float(onsets[ti]) if np.isfinite(onsets[ti]) else np.nan,
                    condition=meta.get('condition', ''),
                    event_json_present=bool(str(event_json[ti]).strip()),
                    event_clean=bool(decision['event_clean']),
                    event_status=decision['event_status'],
                    event_rule=decision['event_rule'],
                    inferred_block_position=decision.get('inferred_block_position', np.nan),
                    inferred_block=decision.get('inferred_block', np.nan),
                    estimated_next_event_gap_sec=decision.get('estimated_next_event_gap_sec', np.nan),
                    explicit_position_key=meta.get('position_key'),
                    explicit_position_in_block=meta.get('position_in_block'),
                    explicit_block_key=meta.get('block_key'),
                    explicit_block=meta.get('block'),
                    explicit_break_key=meta.get('break_key'),
                    explicit_break_sec=meta.get('break_sec'),
                    explicit_next_event_key=meta.get('next_event_key'),
                    explicit_next_event_gap_sec=meta.get('next_event_gap_sec'),
                    stimulus_duration_sec=meta.get('stimulus_duration_sec'),
                    event_metadata_flat_json=meta.get('flat_json', '{}')
                )
                event_rows.append(erow)
                if bool(decision['event_clean']):
                    clean += 1
                    sig = analyse_trial(pid, ti, times, H[ti], R[ti])
                else:
                    sig = dict(signal_usable=False, signal_status='EVENT_WINDOW_NOT_CLEAN')
                row = dict(erow)
                row.update(sig)
                all_trial_rows.append(row)
                if bool(decision['event_clean']) and bool(sig.get('signal_usable', False)) and bool(sig.get('response_evaluable', False)):
                    usable += 1
                    unresolved += int(bool(sig.get('primary_unresolved_at20', False)))
                    late += int(bool(sig.get('primary_late_recovery_20_24', False)))
            print(f'[BÁLINT {pi:02d}/{len(pids):02d}] {pid}: trials={n_trials} | clean={clean} | recovery-evaluable={usable} | unresolved@20={unresolved} | late-rescue={late}', flush=True)
    return (pd.DataFrame(all_trial_rows), pd.DataFrame(event_rows), pd.DataFrame(input_audit_rows))
def bool_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[name].fillna(False).astype(bool)
def harmonized_participant_summary(dataset: str, trials: pd.DataFrame) -> pd.DataFrame:
    base = participant_summary(trials)
    base.insert(0, 'dataset', dataset)
    base['participant_analyzable'] = base['n_recovery_evaluable'].fillna(0).astype(int) > 0
    return base
def _finite_values(series: pd.Series) -> np.ndarray:
    x = pd.to_numeric(series, errors='coerce').to_numpy(float)
    return x[np.isfinite(x)]
def dataset_summary_row(dataset: str, trials: pd.DataFrame, ps: pd.DataFrame, seed_offset: int) -> Dict[str, Any]:
    p = ps[ps['participant_analyzable'].astype(bool)].copy()
    if len(p) == 0:
        raise RuntimeError(f'{dataset}: no analyzable participants.')
    u = _finite_values(p['primary_unresolved20_fraction'])
    l = _finite_values(p['primary_late_recovery_fraction_among_unresolved20'])
    r = _finite_values(p['primary_persistent24_fraction_among_unresolved20'])
    u_est, u_lo, u_hi = bootstrap_median_ci(u, BOOT_SEED + seed_offset + 1)
    l_est, l_lo, l_hi = bootstrap_mean_ci(l, BOOT_SEED + seed_offset + 2)
    r_est, r_lo, r_hi = bootstrap_mean_ci(r, BOOT_SEED + seed_offset + 3)
    eligible = trials[bool_col(trials, 'event_clean') & bool_col(trials, 'signal_usable') & bool_col(trials, 'response_evaluable')].copy()
    late = eligible[bool_col(eligible, 'primary_late_recovery_20_24')].copy()
    starts = _finite_values(late['primary_recovery_start_sec']) if len(late) else np.array([])
    confirms = _finite_values(late['primary_recovery_confirm_sec']) if len(late) else np.array([])
    n_late_participants = int(p['n_primary_late_recovery'].fillna(0).gt(0).sum())
    n_participants = int(len(p))
    return dict(
        dataset=dataset,
        n_analyzable_participants=n_participants,
        n_recovery_evaluable_trials=int(len(eligible)),
        unresolved_at20_estimate=u_est,
        unresolved_at20_ci95_low=u_lo,
        unresolved_at20_ci95_high=u_hi,
        unresolved_at20_summary='median participant proportion',
        recovered_20_24_estimate=l_est,
        recovered_20_24_ci95_low=l_lo,
        recovered_20_24_ci95_high=l_hi,
        recovered_20_24_summary='mean participant proportion among +20-s unresolved',
        still_unresolved_at24_estimate=r_est,
        still_unresolved_at24_ci95_low=r_lo,
        still_unresolved_at24_ci95_high=r_hi,
        still_unresolved_at24_summary='mean participant proportion among +20-s unresolved',
        participants_with_late_recovery=n_late_participants,
        participants_with_late_recovery_denominator=n_participants,
        participants_with_late_recovery_fraction=float(n_late_participants / n_participants),
        median_late_recovery_start_sec=float(np.median(starts)) if len(starts) else np.nan,
        median_late_recovery_confirmation_sec=float(np.median(confirms)) if len(confirms) else np.nan
    )
def sensitivity_summary_row(dataset: str, trials: pd.DataFrame, ps: pd.DataFrame, seed_offset: int) -> Dict[str, Any]:
    p = ps[ps['participant_analyzable'].astype(bool)].copy()
    sec = _finite_values(p['secondary_late_recovery_fraction_among_unresolved20'])
    con = _finite_values(p['consensus_late_recovery_fraction_among_primary_unresolved20'])
    s_est, s_lo, s_hi = bootstrap_mean_ci(sec, BOOT_SEED + seed_offset + 11)
    c_est, c_lo, c_hi = bootstrap_mean_ci(con, BOOT_SEED + seed_offset + 12)
    return dict(
        dataset=dataset,
        n_analyzable_participants=int(len(p)),
        secondary_20pct_late_recovery_estimate=s_est,
        secondary_ci95_low=s_lo,
        secondary_ci95_high=s_hi,
        consensus_late_recovery_estimate=c_est,
        consensus_ci95_low=c_lo,
        consensus_ci95_high=c_hi
    )
def long_paper_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, q in wide.iterrows():
        d = str(q['dataset'])
        rows.extend([dict(
            dataset=d,
            outcome='Unresolved at +20 s',
            estimate=float(q['unresolved_at20_estimate']),
            ci95_low=float(q['unresolved_at20_ci95_low']),
            ci95_high=float(q['unresolved_at20_ci95_high']),
            summary='Median participant proportion',
            denominator='Event-clean, signal-usable, recovery-evaluable real trials'
        ), dict(dataset=d, outcome='Recovered between +20 and +24 s', estimate=float(q['recovered_20_24_estimate']), ci95_low=float(q['recovered_20_24_ci95_low']), ci95_high=float(q['recovered_20_24_ci95_high']), summary='Mean participant proportion', denominator='+20-s unresolved subset'), dict(dataset=d, outcome='Still unresolved at +24 s', estimate=float(q['still_unresolved_at24_estimate']), ci95_low=float(q['still_unresolved_at24_ci95_low']), ci95_high=float(q['still_unresolved_at24_ci95_high']), summary='Mean participant proportion', denominator='+20-s unresolved subset'), dict(dataset=d, outcome='Participants with observed late recovery', estimate=float(q['participants_with_late_recovery']), ci95_low=np.nan, ci95_high=np.nan, summary=f"Count / {int(q['participants_with_late_recovery_denominator'])}", denominator='Analyzable participants'), dict(dataset=d, outcome='Median late-recovery start time', estimate=float(q['median_late_recovery_start_sec']), ci95_low=np.nan, ci95_high=np.nan, summary='Seconds', denominator='Observed primary late-recovery trials')])
    return pd.DataFrame(rows)
def event_summary(dataset: str, events: pd.DataFrame) -> pd.DataFrame:
    q = events.groupby(
        ['event_status', 'event_rule'],
        dropna=False,
        as_index=False
    ).agg(n_trials=('participant', 'size'), n_participants=('participant', 'nunique')).sort_values(['event_status', 'event_rule']).reset_index(drop=True)
    q.insert(0, 'dataset', dataset)
    return q
def combined_protected_self_tests() -> None:
    protected_self_tests()
    import tempfile
    fs = 10.0
    n = 4000
    t = np.arange(n) / fs
    hbo = np.column_stack([np.sin(2 * np.pi * 0.05 * t), np.sin(2 * np.pi * 0.04 * t)])
    hbr = -0.5 * hbo
    times_ms = np.arange(20, 20 + 60 * 26, 26, dtype=float)[:60] * 1000.0
    y = np.zeros((2, len(times_ms)))
    y[0, ::2] = 1
    y[1, 1::2] = 1
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td) / 'VP001-NIRS'
        folder.mkdir()
        savemat(folder / 'cnt_wg.mat', {'cnt_wg': {'oxy': {'x': hbo, 'fs': fs}, 'deoxy': {'x': hbr, 'fs': fs}}})
        savemat(folder / 'mrk_wg.mat', {'mrk_wg': {'time': times_ms, 'y': y, 'className': np.array(['WG', 'BL'], dtype=object)}})
        cnt, mrk, _ = load_shin_matlab_source(folder)
        H, R, f, _ = extract_shin_cnt(cnt)
        mm = extract_shin_markers(mrk)
        assert H.shape == hbo.shape and R.shape == hbr.shape
        assert abs(f - fs) < 1e-09
        assert int(mm['wg_mask'].sum()) == 30
    print('Shin input parsing checks: PASS')
def combined_main(argv: Optional[Sequence[str]]=None) -> None:
    args = combined_parse_args(argv)
    combined_protected_self_tests()
    if args.self_test_only:
        return
    try_mount_drive()
    shin_root = Path(args.shin_root)
    balint_h5 = resolve_existing_exact(Path(args.balint_h5), 'file')
    shin_nirs_dir = resolve_shin_nirs_dir(shin_root, args.shin_nirs_dir)
    out = Path(args.output)
    for sub in ('audit', 'results', 'tables'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    print('=' * 160)
    print('HEMOBOUNDARY — HARMONIZED SHIN + BÁLINT REAL LATE-WINDOW RECOVERY')
    print('=' * 160)
    print('SHIN NIRS directory          :', shin_nirs_dir)
    print('Bálint HDF5                 :', balint_h5)
    print('Output                       :', out)
    print('Common grid                  : -5 to +24 s at 10 Hz')
    print('Primary window               : 0 to +20 s')
    print('Late window                  : +20 to +24 s')
    print('Primary recovery             : baseline 2.5-97.5% envelope, sustained >=0.8 s')
    print('Secondary recovery           : <=20% peak excursion, sustained >=0.8 s')
    print('Synthetic bank               : NOT USED')
    print('Model fitting/tuning         : NONE')
    print('Forced +24-s offset          : NO')
    print()
    print('[1/2] SHIN')
    shin_trials, shin_events, shin_inputs = process_shin(shin_nirs_dir)
    print('\n[2/2] BÁLINT')
    bal_trials, bal_events, bal_inputs = process_balint(balint_h5)
    atomic_csv(shin_trials, out / 'results' / 'SHIN_REAL_LATE_WINDOW_TRIALS.csv.gz')
    atomic_csv(bal_trials, out / 'results' / 'BALINT_REAL_LATE_WINDOW_TRIALS.csv.gz')
    atomic_csv(shin_events, out / 'audit' / 'SHIN_EVENT_SPACING_AUDIT.csv.gz')
    atomic_csv(bal_events, out / 'audit' / 'BALINT_EVENT_SPACING_AUDIT.csv.gz')
    atomic_csv(shin_inputs, out / 'audit' / 'SHIN_INPUT_AUDIT.csv')
    atomic_csv(bal_inputs, out / 'audit' / 'BALINT_INPUT_AUDIT.csv')
    shin_ps = harmonized_participant_summary('SHIN', shin_trials)
    bal_ps = harmonized_participant_summary('BALINT', bal_trials)
    ps = pd.concat([shin_ps, bal_ps], ignore_index=True)
    atomic_csv(ps, out / 'results' / 'SHIN_BALINT_PARTICIPANT_SUMMARY.csv')
    wide = pd.DataFrame([dataset_summary_row('SHIN', shin_trials, shin_ps, 100), dataset_summary_row('BALINT', bal_trials, bal_ps, 200)])
    long = long_paper_table(wide)
    sens = pd.DataFrame([sensitivity_summary_row('SHIN', shin_trials, shin_ps, 100), sensitivity_summary_row('BALINT', bal_trials, bal_ps, 200)])
    atomic_csv(wide, out / 'tables' / 'Table4_SHIN_BALINT_HARMONIZED_LATE_WINDOW_WIDE.csv')
    atomic_csv(long, out / 'tables' / 'Table4_SHIN_BALINT_HARMONIZED_LATE_WINDOW_LONG.csv')
    atomic_csv(sens, out / 'tables' / 'TableS_LATE_WINDOW_SENSITIVITY.csv')
    evsum = pd.concat([event_summary('SHIN', shin_events), event_summary('BALINT', bal_events)], ignore_index=True)
    atomic_csv(evsum, out / 'tables' / 'EVENT_SPACING_SUMMARY.csv')
    manifest = dict(
        version=COMBINED_VERSION,
        output=str(out),
        shin_nirs_dir=str(shin_nirs_dir),
        balint_h5=str(balint_h5),
        datasets=['SHIN', 'BALINT'],
        real_physiological_data=True,
        semi_synthetic_bank_used=False,
        model_fitting=False,
        model_tuning=False,
        common_grid_hz=COMMON_GRID_HZ,
        epoch_sec=[BASELINE_START, LATE_END],
        baseline_sec=[BASELINE_START, BASELINE_END],
        primary_window_sec=[0.0, PRIMARY_END],
        late_window_sec=[PRIMARY_END, LATE_END],
        primary_recovery_rule='post-peak return to trial empirical baseline 2.5th-97.5th percentile envelope, sustained >=0.8 s',
        secondary_recovery_rule='post-peak <=20% positive peak excursion, sustained >=0.8 s',
        shin_preprocessing='official HbO/HbR; continuous third-order zero-phase low-pass 0.2 Hz; common robust baseline normalization',
        shin_event_clean_rule='next ANY Dataset-C trial marker after +24 s and continuous coverage through +24 s',
        balint_event_clean_rule='exclude fourth/pre-question stimulus in complete 10x4 blocks; incomplete trials require explicit spacing metadata',
        exact_real_timing_truth_claim=False,
        endpoint_forced_at_24s=False,
        reporting_wording='still unresolved at +24 s / persistent through +24 s',
        shin_analyzable_participants=int(wide.loc[wide.dataset == 'SHIN', 'n_analyzable_participants'].iloc[0]),
        balint_analyzable_participants=int(wide.loc[wide.dataset == 'BALINT', 'n_analyzable_participants'].iloc[0])
    )
    atomic_json(manifest, out / 'audit' / 'RUN_MANIFEST.json')
    summary_lines = [
        'HARMONIZED SHIN + BÁLINT LATE-WINDOW RECOVERY',
        '=' * 100,
        '',
        'COMMON PRIMARY RULE',
        'Return to the empirical pre-stimulus 2.5th-97.5th percentile envelope',
        'for >=0.8 s after the positive task peak.',
        '',
        'TABLE 4 — WIDE',
        wide.to_string(index=False),
        '',
        'TABLE 4 — LONG',
        long.to_string(index=False),
        '',
        'SENSITIVITY',
        sens.to_string(index=False),
        '',
        'EVENT-SPACING AUDIT',
        evsum.to_string(index=False),
        '',
        'INTERPRETATION',
        'These are descriptive real-physiology late-window sensitivity results.',
        'They are not exact biological timing-accuracy results because real onset/peak/recovery truth is unavailable.',
        "Use 'still unresolved at +24 s' rather than 'beyond +24 s' in the harmonized comparison."
    ]
    (out / 'SUMMARY.txt').write_text('\n'.join(summary_lines), encoding='utf-8')
    print('\n' + '=' * 160)
    print('EVENT-SPACING SUMMARY')
    print('=' * 160)
    print(evsum.to_string(index=False))
    print('\n' + '=' * 160)
    print('TABLE 4 — HARMONIZED SHIN + BÁLINT')
    print('=' * 160)
    print(wide.to_string(index=False))
    print('\n' + '=' * 160)
    print('SUPPLEMENTARY SENSITIVITY')
    print('=' * 160)
    print(sens.to_string(index=False))
    print('\nFINAL AUDIT')
    print('  real physiology             : YES')
    print('  identical recovery rule     : YES')
    print('  identical -5..+24 grid      : YES')
    print('  dataset-specific preprocessing retained: YES')
    print('  SHIN next-event contamination screened : YES')
    print('  Bálint question contamination screened : YES')
    print('  semi-synthetic bank         : NO')
    print('  model fitting/tuning        : NO')
    print('  forced +24-s offset         : NO')
    print('  exact real timing truth     : NO')
    print('  output                      :', out)
    print('  read first                  :', out / 'SUMMARY.txt')
if __name__ == '__main__':
    combined_main()
