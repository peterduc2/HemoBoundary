"""Recorded-response recovery in the Khan dataset."""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import h5py
import numpy as np
import pandas as pd
VERSION = 'KHAN_REAL_END_OF_WINDOW_RECOVERY_V1_20260922'
DEFAULT_H5 = Path('/content/drive/MyDrive/HonorProject/KHAN2026_FINGER_HEMOBOUNDARY_EXTERNAL_PREP/processed/khan2026_finger_epochs_10hz.h5')
DEFAULT_OUTPUT = Path('/content/drive/MyDrive/HonorProject/KHAN2026_FINGER_HEMOBOUNDARY_EXTERNAL_PREP/108_KHAN_REAL_END_OF_WINDOW_RECOVERY_V1')
EXPECTED_PARTICIPANTS = 25
BASELINE_START = -5.0
BASELINE_END = 0.0
BASELINE_Q_LOW = 0.025
BASELINE_Q_HIGH = 0.975
SUSTAIN_SEC = 0.8
SECONDARY_PEAK_FRACTION = 0.2
MIN_BASELINE_SAMPLES = 20
MIN_VALID_CHANNELS = 2
MIN_FINITE_FRACTION = 0.8
BOOT_REPS = 20000
BOOT_SEED = 20260922

def parse_args(argv: Optional[Sequence[str]]=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Khan real-fNIRS end-of-window recovery audit.')
    p.add_argument('--h5', default=str(DEFAULT_H5))
    p.add_argument('--output', default=str(DEFAULT_OUTPUT))
    p.add_argument('--self-test-only', action='store_true')
    args, unknown = p.parse_known_args(argv)
    if unknown:
        print('Ignoring notebook/runtime arguments:', unknown)
    return args
def safe_is_file(path: Path) -> bool:
    try:
        return Path(path).is_file()
    except OSError:
        return False
def safe_is_dir(path: Path) -> bool:
    try:
        return Path(path).is_dir()
    except OSError:
        return False
def try_mount_drive(force: bool=False) -> None:
    root = Path('/content/drive/MyDrive')
    if not force and safe_is_dir(root):
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
    if not safe_is_dir(root):
        raise RuntimeError('Google Drive did not become readable at /content/drive/MyDrive.')
def resolve_exact_file(path: Path) -> Path:
    path = Path(path)
    if safe_is_file(path):
        return path
    if str(path).startswith('/content/drive/'):
        print('[Drive] Input was not readable. Force-remounting Drive once...')
        try_mount_drive(force=True)
        if safe_is_file(path):
            return path
    raise FileNotFoundError(f"Exact Khan HDF5 path does not exist or is not readable:\n  {path}\n\nThis script does not recursively search Drive. If your prepared HDF5 is elsewhere, rerun with --h5 '/content/drive/MyDrive/.../file.h5'.")
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
def decode_str_array(a: Any) -> np.ndarray:
    arr = np.asarray(a)
    vals = []
    for x in arr.reshape(-1):
        if isinstance(x, bytes):
            vals.append(x.decode('utf-8', errors='replace'))
        elif isinstance(x, np.bytes_):
            vals.append(bytes(x).decode('utf-8', errors='replace'))
        else:
            vals.append(str(x))
    return np.asarray(vals, dtype=str).reshape(arr.shape)
def score_name(name: str, tokens: Sequence[str]) -> int:
    low = str(name).lower()
    leaf = low.rstrip('/').split('/')[-1]
    score = 0
    for token in tokens:
        token = token.lower()
        if leaf == token:
            score += 10
        elif token in leaf:
            score += 4
        elif token in low:
            score += 1
    return score
def all_datasets(h5: h5py.File) -> Dict[str, h5py.Dataset]:
    out: Dict[str, h5py.Dataset] = {}

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            out['/' + name] = obj
    h5.visititems(visit)
    return out
def h5_inventory(path: Path) -> pd.DataFrame:
    rows = []
    with h5py.File(path, 'r') as h5:

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                rows.append(dict(path='/' + name, shape=str(tuple(obj.shape)), dtype=str(obj.dtype), ndim=int(obj.ndim)))
        h5.visititems(visit)
    return pd.DataFrame(rows)
def find_time_dataset(datasets: Mapping[str, h5py.Dataset]) -> Tuple[str, np.ndarray]:
    candidates = []
    for name, ds in datasets.items():
        if ds.ndim != 1 or not np.issubdtype(ds.dtype, np.number):
            continue
        sc = score_name(name, ('times', 'time', 'epoch_times', 'epoch_time', 'nirs_times'))
        if sc <= 0:
            continue
        a = np.asarray(ds[:], float).reshape(-1)
        if len(a) < 50:
            continue
        if not np.all(np.isfinite(a)):
            continue
        if not np.all(np.diff(a) > 0):
            continue
        if float(a[0]) > -4.0 or float(a[-1]) < 15.0:
            continue
        candidates.append((sc, name, a))
    if not candidates:
        raise RuntimeError('Could not identify a monotonic epoch time vector spanning the pre-task and post-task periods.')
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return (candidates[0][1], candidates[0][2])
def participant_group_candidates(h5: h5py.File) -> List[str]:
    groups: List[str] = []

    def visit(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Group):
            return
        leaf = name.rstrip('/').split('/')[-1]
        if re.fullmatch('(?i)(?:sub[-_]?)?S\\d{1,3}', leaf):
            groups.append(name)
    h5.visititems(visit)
    return sorted(set(groups), key=lambda x: int(re.findall('\\d+', x.rsplit('/', 1)[-1])[0]))
def normalize_group_pair(dso: h5py.Dataset, dsr: h5py.Dataset, times: np.ndarray, datasets: Mapping[str, h5py.Dataset], label: str) -> Tuple[np.ndarray, np.ndarray]:
    if dso.ndim != 3 or dsr.ndim != 3:
        raise RuntimeError(f'{label}: HbO/HbR must be 3D.')
    if tuple(dso.shape) != tuple(dsr.shape):
        raise RuntimeError(f'{label}: HbO/HbR shape mismatch {dso.shape} vs {dsr.shape}.')
    shape = tuple(dso.shape)
    n_time = len(times)
    t_axes = [i for i, s in enumerate(shape) if s == n_time]
    if len(t_axes) != 1:
        raise RuntimeError(f'{label}: could not uniquely identify time axis {n_time} in {shape}.')
    ta = t_axes[0]
    other = [i for i in range(3) if i != ta]
    trial_lengths = []
    for name2, ds2 in datasets.items():
        if ds2.ndim != 1:
            continue
        if score_name(name2, ('trial_index', 'trial', 'trials', 'label', 'labels', 'event', 'events')) > 0:
            trial_lengths.append(len(ds2))
    trial_axis = None
    for ntrial in trial_lengths:
        matching = [ax for ax in other if shape[ax] == ntrial]
        if len(matching) == 1:
            trial_axis = matching[0]
            break
    if trial_axis is None and 48 in (shape[other[0]], shape[other[1]]):
        if shape[other[0]] == 48 and shape[other[1]] != 48:
            channel_axis = other[0]
            trial_axis = other[1]
        elif shape[other[1]] == 48 and shape[other[0]] != 48:
            channel_axis = other[1]
            trial_axis = other[0]
        else:
            channel_axis = None
    else:
        channel_axis = None
    if trial_axis is None:
        a0, a1 = other
        if shape[a0] <= 128 < shape[a1]:
            channel_axis, trial_axis = (a0, a1)
        elif shape[a1] <= 128 < shape[a0]:
            channel_axis, trial_axis = (a1, a0)
        else:
            d0 = abs(shape[a0] - 48)
            d1 = abs(shape[a1] - 48)
            if d0 < d1:
                channel_axis, trial_axis = (a0, a1)
            elif d1 < d0:
                channel_axis, trial_axis = (a1, a0)
            else:
                raise RuntimeError(f'{label}: ambiguous trial/channel axes in shape {shape}.')
    if channel_axis is None:
        channel_axis = [ax for ax in other if ax != trial_axis][0]
    H = np.transpose(np.asarray(dso[:], float), (trial_axis, channel_axis, ta))
    R = np.transpose(np.asarray(dsr[:], float), (trial_axis, channel_axis, ta))
    return (H, R)
def find_hbo_hbr_group(group: h5py.Group, times: np.ndarray) -> Tuple[str, str, np.ndarray, np.ndarray]:
    datasets: Dict[str, h5py.Dataset] = {}
    gp = group.name.rstrip('/')

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets[gp + '/' + name] = obj
    group.visititems(visit)
    hbo = []
    hbr = []
    for name, ds in datasets.items():
        if ds.ndim != 3 or not np.issubdtype(ds.dtype, np.number):
            continue
        low = name.lower()
        so = score_name(name, ('hbo', 'hbo_epochs', 'oxy', 'oxygenated'))
        sr = score_name(name, ('hbr', 'hbr_epochs', 'deoxy', 'deoxygenated'))
        if 'deoxy' in low or 'deoxygenated' in low:
            so = 0
        if so > 0:
            hbo.append((so, name, ds))
        if sr > 0:
            hbr.append((sr, name, ds))
    pairs = []
    for so, no, dso in hbo:
        for sr, nr, dsr in hbr:
            try:
                H, R = normalize_group_pair(dso, dsr, times, datasets, label=group.name)
            except RuntimeError:
                continue
            if H.shape != R.shape:
                continue
            pairs.append((so + sr, no, nr, H, R))
    if not pairs:
        raise RuntimeError(f'{group.name}: could not identify matching prepared HbO/HbR arrays.')
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    _, no, nr, H, R = pairs[0]
    return (no, nr, H, R)
def find_flat_participant_dataset(datasets: Mapping[str, h5py.Dataset]) -> Tuple[Optional[str], Optional[np.ndarray]]:
    allowed = {'participant', 'participants', 'participant_id', 'participant_ids', 'subject', 'subject_id', 'subjects'}
    candidates = []
    for name, ds in datasets.items():
        if ds.ndim != 1:
            continue
        leaf = name.rstrip('/').split('/')[-1].lower()
        if leaf not in allowed:
            continue
        try:
            vals = decode_str_array(ds[:]).reshape(-1)
        except Exception:
            continue
        unique = sorted(set(vals.tolist()))
        if not 2 <= len(unique) <= 100:
            continue
        sc = 10 + (3 if 'participant' in leaf else 0)
        candidates.append((sc, name, vals))
    if not candidates:
        return (None, None)
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return (candidates[0][1], candidates[0][2])
def normalize_flat_array(arr: np.ndarray, n_trials: int, n_time: int, label: str) -> np.ndarray:
    x = np.asarray(arr, float)
    if x.ndim != 3:
        raise RuntimeError(f'{label}: expected 3D array, got {x.shape}.')
    t_axes = [i for i, s in enumerate(x.shape) if s == n_time]
    if len(t_axes) != 1:
        raise RuntimeError(f'{label}: could not identify time axis {n_time} in {x.shape}.')
    ta = t_axes[0]
    other = [i for i in range(3) if i != ta]
    tr_axes = [i for i in other if x.shape[i] == n_trials]
    if len(tr_axes) != 1:
        raise RuntimeError(f'{label}: could not identify trial axis {n_trials} in {x.shape}.')
    ra = tr_axes[0]
    ca = [i for i in other if i != ra][0]
    return np.transpose(x, (ra, ca, ta))
def load_flat_hdf5(path: Path) -> Optional[Dict[str, Any]]:
    with h5py.File(path, 'r') as h5:
        datasets = all_datasets(h5)
        try:
            time_path, times = find_time_dataset(datasets)
        except RuntimeError:
            return None
        participant_path, participants = find_flat_participant_dataset(datasets)
        if participants is None:
            return None
        n_trials = len(participants)
        n_time = len(times)
        hbo = []
        hbr = []
        for name, ds in datasets.items():
            if ds.ndim != 3 or not np.issubdtype(ds.dtype, np.number):
                continue
            low = name.lower()
            so = score_name(name, ('hbo', 'hbo_epochs', 'oxy', 'oxygenated'))
            sr = score_name(name, ('hbr', 'hbr_epochs', 'deoxy', 'deoxygenated'))
            if 'deoxy' in low or 'deoxygenated' in low:
                so = 0
            if so > 0:
                hbo.append((so, name, ds))
            if sr > 0:
                hbr.append((sr, name, ds))
        pairs = []
        for so, no, dso in hbo:
            for sr, nr, dsr in hbr:
                try:
                    H = normalize_flat_array(dso[:], n_trials, n_time, no)
                    R = normalize_flat_array(dsr[:], n_trials, n_time, nr)
                except RuntimeError:
                    continue
                if H.shape == R.shape:
                    pairs.append((so + sr, no, nr, H, R))
        if not pairs:
            return None
        pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
        _, no, nr, H, R = pairs[0]
        return dict(
            times=np.asarray(times, float),
            participant=np.asarray(participants, dtype=str),
            hbo=H,
            hbr=R,
            schema='FLAT_HDF5',
            audit=dict(time_dataset=time_path, participant_dataset=participant_path, hbo_source=no, hbr_source=nr)
        )
def load_grouped_hdf5(path: Path) -> Optional[Dict[str, Any]]:
    participant_blocks = []
    audit_rows = []
    with h5py.File(path, 'r') as h5:
        global_ds = all_datasets(h5)
        try:
            time_path, times = find_time_dataset(global_ds)
        except RuntimeError:
            time_path, times = (None, None)
        groups = participant_group_candidates(h5)
        for gp in groups:
            grp = h5[gp]
            if times is None:
                local = {}

                def visit(name: str, obj: Any) -> None:
                    if isinstance(obj, h5py.Dataset):
                        local[grp.name.rstrip('/') + '/' + name] = obj
                grp.visititems(visit)
                try:
                    local_time_path, local_times = find_time_dataset(local)
                except RuntimeError:
                    continue
                t = local_times
                t_source = local_time_path
            else:
                t = times
                t_source = time_path
            try:
                hbo_source, hbr_source, H, R = find_hbo_hbr_group(grp, t)
            except RuntimeError:
                continue
            pid = gp.rstrip('/').split('/')[-1]
            participant_blocks.append((pid, np.asarray(t, float), H, R))
            audit_rows.append(dict(
                participant=pid,
                group='/' + gp,
                time_source=t_source,
                hbo_source=hbo_source,
                hbr_source=hbr_source,
                n_trials=int(H.shape[0]),
                n_channels=int(H.shape[1]),
                n_timepoints=int(H.shape[2]),
                time_start=float(t[0]),
                time_end=float(t[-1]),
                median_dt=float(np.median(np.diff(t)))
            ))
    if not participant_blocks:
        return None
    ref_t = participant_blocks[0][1]
    for pid, t, _, _ in participant_blocks:
        if len(t) != len(ref_t) or not np.allclose(t, ref_t, atol=1e-08, rtol=0):
            raise RuntimeError(f'{pid}: participant time vector differs from the common Khan epoch.')
    min_c = min((min(H.shape[1], R.shape[1]) for _, _, H, R in participant_blocks))
    pids = []
    HbO = []
    HbR = []
    for pid, _, H, R in participant_blocks:
        pids.extend([pid] * len(H))
        HbO.append(H[:, :min_c, :])
        HbR.append(R[:, :min_c, :])
    return dict(
        times=np.asarray(ref_t, float),
        participant=np.asarray(pids, dtype=str),
        hbo=np.concatenate(HbO, axis=0),
        hbr=np.concatenate(HbR, axis=0),
        schema='GROUPED_HDF5',
        audit=dict(group_rows=audit_rows)
    )
def load_khan_hdf5(path: Path) -> Dict[str, Any]:
    flat = load_flat_hdf5(path)
    if flat is not None:
        return flat
    grouped = load_grouped_hdf5(path)
    if grouped is not None:
        return grouped
    inv = h5_inventory(path)
    raise RuntimeError('Could not resolve prepared Khan HbO/HbR/time/participant arrays.\nNo result was written because the script will not guess the schema.\n\n' + inv.to_string(index=False))
def robust_z(x: np.ndarray, baseline_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    base = x[:, baseline_mask]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        med = np.nanmedian(base, axis=1, keepdims=True)
        mad = 1.4826 * np.nanmedian(np.abs(base - med), axis=1, keepdims=True)
        sd = np.nanstd(base, axis=1, keepdims=True)
    scale = np.where(np.isfinite(mad) & (mad > 1e-08), mad, np.where(np.isfinite(sd) & (sd > 1e-08), sd, 1.0))
    return ((x - med) / scale, med.reshape(-1), scale.reshape(-1))
def first_sustained_run(mask: np.ndarray, times: np.ndarray, start_index: int, sustain_sec: float) -> Tuple[float, float, int]:
    mask = np.asarray(mask, bool)
    times = np.asarray(times, float)
    if len(times) < 2:
        return (np.nan, np.nan, 0)
    dt = float(np.median(np.diff(times)))
    n_need = max(1, int(math.ceil(float(sustain_sec) / dt - 1e-12)))
    i = max(0, int(start_index))
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        run_len = j - i
        if run_len >= n_need:
            confirm_idx = i + n_need - 1
            return (float(times[i]), float(times[confirm_idx]), int(n_need))
        i = max(j, i + 1)
    return (np.nan, np.nan, int(n_need))
def analyse_trial(times: np.ndarray, H: np.ndarray, R: np.ndarray) -> Dict[str, Any]:
    times = np.asarray(times, float)
    H = np.asarray(H, float)
    R = np.asarray(R, float)
    C = min(H.shape[0], R.shape[0])
    H = H[:C]
    R = R[:C]
    bm = (times >= BASELINE_START - 1e-09) & (times < BASELINE_END - 1e-09)
    if int(np.sum(bm)) < MIN_BASELINE_SAMPLES:
        return dict(signal_usable=False, response_evaluable=False, signal_status='INSUFFICIENT_BASELINE_SAMPLES')
    baseline_finite = np.mean(np.isfinite(H[:, bm]) & np.isfinite(R[:, bm]), axis=1)
    full_finite = np.mean(np.isfinite(H) & np.isfinite(R), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        Hsd = np.nanstd(H[:, bm], axis=1)
        Rsd = np.nanstd(R[:, bm], axis=1)
    valid = (baseline_finite >= MIN_FINITE_FRACTION) & (full_finite >= MIN_FINITE_FRACTION) & np.isfinite(Hsd) & np.isfinite(Rsd)
    if int(np.sum(valid)) < MIN_VALID_CHANNELS:
        return dict(
            signal_usable=False,
            response_evaluable=False,
            signal_status='TOO_FEW_VALID_CHANNELS',
            n_valid_channels=int(np.sum(valid)),
            n_channels_before=int(C)
        )
    H = H[valid]
    R = R[valid]
    Hz, _, _ = robust_z(H, bm)
    Rz, _, _ = robust_z(R, bm)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        ghbo = np.nanmedian(Hz, axis=0)
        ghbr = np.nanmedian(Rz, axis=0)
    anti = 0.5 * (ghbo - ghbr)
    finite = np.isfinite(anti)
    if np.mean(finite) < 0.95:
        return dict(
            signal_usable=False,
            response_evaluable=False,
            signal_status='NONFINITE_TRIAL_CONTRAST',
            n_valid_channels=int(np.sum(valid)),
            n_channels_before=int(C)
        )
    baseline_values = anti[bm & finite]
    if len(baseline_values) < MIN_BASELINE_SAMPLES:
        return dict(
            signal_usable=False,
            response_evaluable=False,
            signal_status='INSUFFICIENT_FINITE_BASELINE_CONTRAST',
            n_valid_channels=int(np.sum(valid)),
            n_channels_before=int(C)
        )
    baseline_center = float(np.median(baseline_values))
    baseline_low = float(np.quantile(baseline_values, BASELINE_Q_LOW))
    baseline_high = float(np.quantile(baseline_values, BASELINE_Q_HIGH))
    post_idx = np.flatnonzero((times >= 0.0) & finite)
    if len(post_idx) == 0:
        return dict(
            signal_usable=False,
            response_evaluable=False,
            signal_status='NO_POST_TASK_WINDOW',
            n_valid_channels=int(np.sum(valid)),
            n_channels_before=int(C)
        )
    peak_idx = int(post_idx[np.argmax(anti[post_idx])])
    peak_time = float(times[peak_idx])
    peak_value = float(anti[peak_idx])
    peak_excursion = float(peak_value - baseline_center)
    response_evaluable = bool(np.isfinite(peak_value) and peak_value > baseline_high and (peak_excursion > 0.0))
    common = dict(
        signal_usable=True,
        n_valid_channels=int(np.sum(valid)),
        n_channels_before=int(C),
        baseline_center=baseline_center,
        baseline_low=baseline_low,
        baseline_high=baseline_high,
        peak_time_sec=peak_time,
        peak_value=peak_value,
        peak_excursion=peak_excursion,
        epoch_end_sec=float(times[-1])
    )
    if not response_evaluable:
        return dict(**common, response_evaluable=False, signal_status='NO_POSITIVE_EXCURSION_ABOVE_BASELINE_ENVELOPE')
    inside = finite & (anti >= baseline_low) & (anti <= baseline_high)
    p_start, p_confirm, n_need = first_sustained_run(inside, times, start_index=peak_idx + 1, sustain_sec=SUSTAIN_SEC)
    primary_recovered = bool(np.isfinite(p_confirm))
    primary_unresolved = bool(not primary_recovered)
    secondary_threshold = baseline_center + SECONDARY_PEAK_FRACTION * peak_excursion
    below20 = finite & (anti <= secondary_threshold)
    s_start, s_confirm, _ = first_sustained_run(below20, times, start_index=peak_idx + 1, sustain_sec=SUSTAIN_SEC)
    secondary_recovered = bool(np.isfinite(s_confirm))
    secondary_unresolved = bool(not secondary_recovered)
    return dict(
        **common,
        response_evaluable=True,
        signal_status='OK',
        sustain_samples=int(n_need),
        primary_recovery_start_sec=p_start,
        primary_recovery_confirm_sec=p_confirm,
        primary_recovered_before_epoch_end=primary_recovered,
        primary_unresolved_at_epoch_end=primary_unresolved,
        secondary_threshold=secondary_threshold,
        secondary_recovery_start_sec=s_start,
        secondary_recovery_confirm_sec=s_confirm,
        secondary_recovered_before_epoch_end=secondary_recovered,
        secondary_unresolved_at_epoch_end=secondary_unresolved,
        both_rules_unresolved_at_epoch_end=bool(primary_unresolved and secondary_unresolved),
        rules_agree_on_recovery_status=bool(primary_unresolved == secondary_unresolved)
    )
def safe_fraction(num: int, den: int) -> float:
    return float(num / den) if den > 0 else np.nan
def participant_summary(trial_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, q0 in trial_df.groupby('participant', sort=True):
        q = q0[q0['signal_usable'].astype(bool) & q0['response_evaluable'].astype(bool)].copy()
        n = int(len(q))
        n_unresolved = int(q['primary_unresolved_at_epoch_end'].astype(bool).sum()) if n else 0
        n_recovered = int(q['primary_recovered_before_epoch_end'].astype(bool).sum()) if n else 0
        n_secondary_unresolved = int(q['secondary_unresolved_at_epoch_end'].astype(bool).sum()) if n else 0
        n_both_unresolved = int(q['both_rules_unresolved_at_epoch_end'].astype(bool).sum()) if n else 0
        recovery_start = pd.to_numeric(q.loc[q['primary_recovered_before_epoch_end'].astype(bool), 'primary_recovery_start_sec'], errors='coerce').dropna().to_numpy(float)
        recovery_confirm = pd.to_numeric(q.loc[q['primary_recovered_before_epoch_end'].astype(bool), 'primary_recovery_confirm_sec'], errors='coerce').dropna().to_numpy(float)
        rows.append(dict(
            participant=pid,
            n_all_trials=int(len(q0)),
            n_signal_usable=int(q0['signal_usable'].astype(bool).sum()),
            n_recovery_evaluable=n,
            n_primary_recovered=n_recovered,
            primary_recovered_fraction=safe_fraction(n_recovered, n),
            n_primary_unresolved=n_unresolved,
            primary_unresolved_fraction=safe_fraction(n_unresolved, n),
            n_secondary_unresolved=n_secondary_unresolved,
            secondary_unresolved_fraction=safe_fraction(n_secondary_unresolved, n),
            n_both_rules_unresolved=n_both_unresolved,
            both_rules_unresolved_fraction=safe_fraction(n_both_unresolved, n),
            median_primary_recovery_start_sec=float(np.median(recovery_start)) if len(recovery_start) else np.nan,
            median_primary_recovery_confirm_sec=float(np.median(recovery_confirm)) if len(recovery_confirm) else np.nan
        ))
    return pd.DataFrame(rows)
def bootstrap_stat_ci(values: Sequence[float], statistic: str, seed: int) -> Tuple[float, float, float]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    if statistic == 'mean':
        fn = np.mean
    elif statistic == 'median':
        fn = np.median
    else:
        raise ValueError(statistic)
    est = float(fn(x))
    if len(x) == 1:
        return (est, est, est)
    rng = np.random.default_rng(seed)
    boot = np.empty(BOOT_REPS, float)
    done = 0
    while done < BOOT_REPS:
        n = min(1000, BOOT_REPS - done)
        idx = rng.integers(0, len(x), size=(n, len(x)))
        if statistic == 'mean':
            boot[done:done + n] = np.mean(x[idx], axis=1)
        else:
            boot[done:done + n] = np.median(x[idx], axis=1)
        done += n
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return (est, float(lo), float(hi))
def make_tables(trial_df: pd.DataFrame, psummary: pd.DataFrame, epoch_end: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible_p = psummary[psummary['n_recovery_evaluable'] > 0].copy()
    unresolved = pd.to_numeric(eligible_p['primary_unresolved_fraction'], errors='coerce').to_numpy(float)
    recovered = pd.to_numeric(eligible_p['primary_recovered_fraction'], errors='coerce').to_numpy(float)
    secondary_unresolved = pd.to_numeric(eligible_p['secondary_unresolved_fraction'], errors='coerce').to_numpy(float)
    both_unresolved = pd.to_numeric(eligible_p['both_rules_unresolved_fraction'], errors='coerce').to_numpy(float)
    u, ulo, uhi = bootstrap_stat_ci(unresolved, 'median', BOOT_SEED + 1)
    r, rlo, rhi = bootstrap_stat_ci(recovered, 'median', BOOT_SEED + 2)
    su, sulo, suhi = bootstrap_stat_ci(secondary_unresolved, 'median', BOOT_SEED + 3)
    bu, bulo, buhi = bootstrap_stat_ci(both_unresolved, 'median', BOOT_SEED + 4)
    eligible_trials = trial_df[trial_df['signal_usable'].astype(bool) & trial_df['response_evaluable'].astype(bool)].copy()
    n_evaluable_trials = int(len(eligible_trials))
    n_unresolved_trials = int(eligible_trials['primary_unresolved_at_epoch_end'].astype(bool).sum())
    n_recovered_trials = int(eligible_trials['primary_recovered_before_epoch_end'].astype(bool).sum())
    n_participants = int(eligible_p['participant'].nunique())
    n_participants_unresolved = int(eligible_p['n_primary_unresolved'].gt(0).sum())
    recovered_trials = eligible_trials[eligible_trials['primary_recovered_before_epoch_end'].astype(bool)]
    recovery_start = pd.to_numeric(recovered_trials['primary_recovery_start_sec'], errors='coerce').dropna().to_numpy(float)
    recovery_confirm = pd.to_numeric(recovered_trials['primary_recovery_confirm_sec'], errors='coerce').dropna().to_numpy(float)
    main = pd.DataFrame([
        dict(outcome=f'Unresolved at available epoch end (~+{epoch_end:.1f} s)', estimate=u, ci95_low=ulo, ci95_high=uhi, summary='Median participant proportion', denominator='Recovery-evaluable real Khan trials'),
        dict(outcome=f'Recovered before available epoch end (~+{epoch_end:.1f} s)', estimate=r, ci95_low=rlo, ci95_high=rhi, summary='Median participant proportion', denominator='Recovery-evaluable real Khan trials'),
        dict(outcome='Participants with at least one unresolved response', estimate=float(n_participants_unresolved), ci95_low=np.nan, ci95_high=np.nan, summary=f'Count / {n_participants}', denominator='Participants with >=1 recovery-evaluable trial'),
        dict(outcome='Recovery-evaluable trials', estimate=float(n_evaluable_trials), ci95_low=np.nan, ci95_high=np.nan, summary='Count', denominator='All prepared real Khan trials'),
        dict(outcome='Unresolved recovery-evaluable trials', estimate=float(n_unresolved_trials), ci95_low=np.nan, ci95_high=np.nan, summary='Pooled count', denominator=f'{n_evaluable_trials} recovery-evaluable trials'),
        dict(outcome='Recovered recovery-evaluable trials', estimate=float(n_recovered_trials), ci95_low=np.nan, ci95_high=np.nan, summary='Pooled count', denominator=f'{n_evaluable_trials} recovery-evaluable trials'),
        dict(outcome='Median recovery start time among recovered trials', estimate=float(np.median(recovery_start)) if len(recovery_start) else np.nan, ci95_low=np.nan, ci95_high=np.nan, summary='Seconds', denominator='Recovered real Khan trials'),
        dict(outcome='Median recovery confirmation time among recovered trials', estimate=float(np.median(recovery_confirm)) if len(recovery_confirm) else np.nan, ci95_low=np.nan, ci95_high=np.nan, summary='Seconds', denominator='Recovered real Khan trials')
    ])
    sensitivity = pd.DataFrame([dict(
        criterion='Primary: baseline-envelope sustained return',
        unresolved_estimate=u,
        ci95_low=ulo,
        ci95_high=uhi,
        summary='Median participant unresolved proportion'
    ), dict(criterion='Secondary: <=20% of peak excursion sustained return', unresolved_estimate=su, ci95_low=sulo, ci95_high=suhi, summary='Median participant unresolved proportion'), dict(criterion='Both rules classify unresolved', unresolved_estimate=bu, ci95_low=bulo, ci95_high=buhi, summary='Median participant proportion')])
    table4_row = pd.DataFrame([dict(
        dataset='Khan',
        primary_window_end_sec=float(epoch_end),
        unresolved_at_window_end=u,
        unresolved_ci95_low=ulo,
        unresolved_ci95_high=uhi,
        late_followup_available='No',
        recovered_during_followup='N/A',
        unresolved_at_followup_end='N/A',
        participants_analyzable=n_participants,
        participants_with_unresolved=n_participants_unresolved,
        recovery_definition=f'Sustained return to empirical pre-task baseline envelope for >= {SUSTAIN_SEC:.1f} s after the response peak',
        interpretation='End-of-window sensitivity only; the prepared Khan epoch does not show what happens after the final sample.'
    )])
    return (main, sensitivity, table4_row)
def protected_self_tests() -> None:
    t = np.arange(-5.0, 19.9 + 0.05, 0.1)
    rng = np.random.default_rng(123)

    def make_trial(recover_time: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
        anti = 0.04 * rng.normal(size=len(t))
        rise = (t >= 2.0) & (t < 6.0)
        anti[rise] += np.linspace(0.0, 3.0, int(rise.sum()))
        active = (t >= 6.0) & (t < 12.0)
        anti[active] += 3.0
        if recover_time is None:
            decay = t >= 12.0
            anti[decay] += np.linspace(3.0, 1.3, int(decay.sum()))
        else:
            decay = (t >= 12.0) & (t < recover_time)
            if np.any(decay):
                anti[decay] += np.linspace(3.0, 0.0, int(decay.sum()))
        H = np.vstack([anti, 0.95 * anti, 1.05 * anti])
        R = -H
        bm = t < 0
        H[:, bm] += 0.1 * rng.normal(size=(3, int(bm.sum())))
        R[:, bm] += 0.1 * rng.normal(size=(3, int(bm.sum())))
        return (H, R)
    H1, R1 = make_trial(16.0)
    a = analyse_trial(t, H1, R1)
    assert a['signal_usable']
    assert a['response_evaluable']
    assert a['primary_recovered_before_epoch_end']
    assert not a['primary_unresolved_at_epoch_end']
    H2, R2 = make_trial(None)
    b = analyse_trial(t, H2, R2)
    assert b['signal_usable']
    assert b['response_evaluable']
    assert b['primary_unresolved_at_epoch_end']
    print('Khan recovery checks: PASS')
def main(argv: Optional[Sequence[str]]=None) -> None:
    args = parse_args(argv)
    protected_self_tests()
    if args.self_test_only:
        return
    try_mount_drive()
    h5_path = resolve_exact_file(Path(args.h5))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'audit').mkdir(parents=True, exist_ok=True)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    store = load_khan_hdf5(h5_path)
    t = np.asarray(store['times'], float)
    H = np.asarray(store['hbo'], float)
    R = np.asarray(store['hbr'], float)
    participant = np.asarray(store['participant'], str)
    if len(t) < 2:
        raise RuntimeError('Khan time vector has fewer than two samples.')
    if t[0] > BASELINE_START + 0.2:
        raise RuntimeError(f'Khan prepared epoch starts at {t[0]:.3f} s; the required -5..0 baseline is unavailable.')
    epoch_end = float(t[-1])
    dt = float(np.median(np.diff(t)))
    pids = sorted(set(participant.tolist()), key=lambda x: int(re.findall('\\d+', x)[0]) if re.findall('\\d+', x) else x)
    if len(pids) != EXPECTED_PARTICIPANTS:
        print(f'[WARN] Expected {EXPECTED_PARTICIPANTS} Khan participants, found {len(pids)}: {pids}')
    print('=' * 150)
    print('KHAN 2026 — REAL-fNIRS END-OF-WINDOW RECOVERY AUDIT')
    print('=' * 150)
    print('Input HDF5                 :', h5_path)
    print('Output                     :', out)
    print('Schema                     :', store['schema'])
    print('Participants               :', len(pids))
    print('Prepared epoch             :', f'{t[0]:.3f} to {t[-1]:.3f} s')
    print('Median sample interval     :', f'{dt:.4f} s')
    print('Semi-synthetic bank        : NOT USED')
    print('HemoBoundary fitting/tune  : NONE')
    print('Primary recovery rule      : baseline 2.5-97.5% envelope for >=0.8 s')
    print('Secondary recovery rule    : <=20% peak excursion for >=0.8 s')
    print('Interpretation             : recovery status AT AVAILABLE EPOCH END')
    print()
    rows: List[Dict[str, Any]] = []
    for pi, pid in enumerate(pids, 1):
        mask = participant == pid
        Hp = H[mask]
        Rp = R[mask]
        usable = 0
        evaluable = 0
        recovered = 0
        unresolved = 0
        for local_i in range(len(Hp)):
            res = analyse_trial(times=t, H=Hp[local_i], R=Rp[local_i])
            row = dict(participant=pid, participant_trial_index=int(local_i))
            row.update(res)
            rows.append(row)
            if bool(res.get('signal_usable', False)):
                usable += 1
            if bool(res.get('response_evaluable', False)):
                evaluable += 1
                if bool(res.get('primary_recovered_before_epoch_end', False)):
                    recovered += 1
                if bool(res.get('primary_unresolved_at_epoch_end', False)):
                    unresolved += 1
        print(f'[{pi:02d}/{len(pids):02d}] {pid}: trials={len(Hp)} | usable={usable} | recovery-evaluable={evaluable} | recovered={recovered} | unresolved@end={unresolved}', flush=True)
    trial_df = pd.DataFrame(rows)
    if not len(trial_df):
        raise RuntimeError('No Khan trials were loaded.')
    psummary = participant_summary(trial_df)
    main_table, sensitivity_table, table4_row = make_tables(trial_df, psummary, epoch_end=epoch_end)
    if store['schema'] == 'GROUPED_HDF5':
        input_audit = pd.DataFrame(store['audit']['group_rows'])
    else:
        input_audit = pd.DataFrame([store['audit']])
    atomic_csv(input_audit, out / 'audit' / 'INPUT_HDF5_AUDIT.csv')
    atomic_csv(trial_df, out / 'results' / 'KHAN_REAL_END_WINDOW_PER_TRIAL.csv.gz')
    atomic_csv(psummary, out / 'results' / 'KHAN_REAL_END_WINDOW_PARTICIPANT_SUMMARY.csv')
    atomic_csv(main_table, out / 'tables' / 'Table_KHAN_END_OF_WINDOW_RECOVERY.csv')
    atomic_csv(sensitivity_table, out / 'tables' / 'TableS_KHAN_RECOVERY_SENSITIVITY.csv')
    atomic_csv(table4_row, out / 'tables' / 'Table4_KHAN_ROW.csv')
    n_total = int(len(trial_df))
    n_usable = int(trial_df['signal_usable'].astype(bool).sum())
    n_evaluable = int(trial_df['response_evaluable'].astype(bool).sum())
    n_unresolved = int(trial_df.loc[trial_df['response_evaluable'].astype(bool), 'primary_unresolved_at_epoch_end'].astype(bool).sum())
    analyzable_p = psummary[psummary['n_recovery_evaluable'] > 0]
    n_analyzable_p = int(analyzable_p['participant'].nunique())
    n_unresolved_p = int(analyzable_p['n_primary_unresolved'].gt(0).sum())
    manifest = dict(
        version=VERSION,
        input_hdf5=str(h5_path),
        output=str(out),
        hdf5_schema=store['schema'],
        real_physiological_epochs=True,
        semi_synthetic_bank_used=False,
        model_fitting=False,
        model_tuning=False,
        baseline_window_sec=[BASELINE_START, BASELINE_END],
        prepared_epoch_start_sec=float(t[0]),
        prepared_epoch_end_sec=epoch_end,
        median_dt_sec=dt,
        primary_recovery_rule=f'after positive peak, anti-Hb contrast enters empirical {BASELINE_Q_LOW:.3f}-{BASELINE_Q_HIGH:.3f} baseline quantile envelope and remains for >= {SUSTAIN_SEC:.1f} s',
        secondary_recovery_rule=f'after positive peak, anti-Hb contrast falls to <= {SECONDARY_PEAK_FRACTION:.2f} of peak excursion and remains for >= {SUSTAIN_SEC:.1f} s',
        n_participants_loaded=int(len(pids)),
        n_participants_recovery_evaluable=n_analyzable_p,
        n_participants_with_unresolved=n_unresolved_p,
        n_total_trials=n_total,
        n_signal_usable_trials=n_usable,
        n_recovery_evaluable_trials=n_evaluable,
        n_primary_unresolved_trials=n_unresolved,
        inference='descriptive real-data end-of-window sensitivity analysis; participant-balanced bootstrap intervals; no exact real timing truth',
        reporting_warning='The prepared Khan epoch ends at approximately 19.9 s. An unresolved trial means recovery was not confirmed before the available epoch ended. It does NOT establish recovery beyond 19.9 s.'
    )
    atomic_json(manifest, out / 'audit' / 'RUN_MANIFEST.json')
    summary_lines = [
        'KHAN 2026 — REAL-fNIRS END-OF-WINDOW RECOVERY AUDIT',
        '=' * 92,
        '',
        f'Input: {h5_path}',
        f'Output: {out}',
        f'Prepared epoch: {t[0]:.3f} to {t[-1]:.3f} s',
        '',
        'PAPER-STYLE TABLE',
        main_table.to_string(index=False),
        '',
        'SENSITIVITY TABLE',
        sensitivity_table.to_string(index=False),
        '',
        'TABLE 4 KHAN ROW',
        table4_row.to_string(index=False),
        '',
        'CORE COUNTS',
        f'Participants loaded                      : {len(pids)}',
        f'Participants with evaluable trials       : {n_analyzable_p}',
        f'Participants with >=1 unresolved trial   : {n_unresolved_p}',
        f'Total real trials                        : {n_total}',
        f'Signal-usable trials                     : {n_usable}',
        f'Recovery-evaluable trials                : {n_evaluable}',
        f'Primary unresolved trials at epoch end   : {n_unresolved}',
        '',
        'INTERPRETATION',
        "Use: 'remained unresolved at the end of the available Khan epoch'.",
        'Do NOT write that these responses remained unresolved beyond 19.9 s,',
        'because the prepared data do not observe what happened afterward.',
        '',
        'This analysis uses real physiological epochs and does not use the',
        'semi-synthetic known-boundary bank.'
    ]
    summary_path = out / 'SUMMARY.txt'
    summary_path.write_text('\n'.join(summary_lines), encoding='utf-8')
    print('\n' + '=' * 150)
    print('KHAN END-OF-WINDOW RECOVERY TABLE')
    print('=' * 150)
    print(main_table.to_string(index=False))
    print('\n' + '=' * 150)
    print('SENSITIVITY')
    print('=' * 150)
    print(sensitivity_table.to_string(index=False))
    print('\n' + '=' * 150)
    print('TABLE 4 KHAN ROW')
    print('=' * 150)
    print(table4_row.to_string(index=False))
    print('\nFINAL AUDIT')
    print('  real physiological epochs       : YES')
    print('  semi-synthetic bank             : NO')
    print('  model fitting/tuning            : NO')
    print('  exact real recovery truth       : NO')
    print('  later-than-epoch claim          : NO')
    print('  prepared epoch end              :', f'{epoch_end:.3f} s')
    print('  results                         :', out)
    print('  read first                      :', summary_path)
if __name__ == '__main__':
    main()
