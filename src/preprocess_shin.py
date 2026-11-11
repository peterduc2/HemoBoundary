"""Shin word-generation fNIRS preprocessing."""
from __future__ import annotations
import json
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
VERSION = 'SHIN_02_PREPROCESS_WG_V1_0_2026_08_26'
EPOCH_TMIN = -5.0
EPOCH_TMAX = 25.0
BASELINE = (-5.0, -2.0)
TASK_END_SEC = 10.0
EXPECTED_TRIALS = 60
EXPECTED_TRIALS_PER_SESSION = 20
EXPECTED_WG_PER_SESSION = 10
EXPECTED_BL_PER_SESSION = 10
EEG_BP = (1.0, 40.0)
EEG_FILTER_ORDER = 6
NIRS_LP_HZ = 0.2
NIRS_FILTER_ORDER = 3

def ensure_package(import_name: str, pip_name: Optional[str]=None) -> None:
    try:
        __import__(import_name)
    except Exception:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pip_name or import_name])
ensure_package('scipy')
from scipy.io import loadmat
from scipy import signal
def maybe_mount_drive() -> None:
    if Path('/content/drive/MyDrive').exists() or Path('/content/gdrive/MyDrive').exists():
        return
    if not Path('/content').exists():
        return
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
    except Exception:
        pass
def output_root() -> Path:
    explicit = os.environ.get('SHIN_WG_ROOT', '').strip()
    if explicit:
        return Path(explicit)
    maybe_mount_drive()
    for p in (Path('/content/drive/MyDrive/HonorProject'), Path('/content/gdrive/MyDrive/HonorProject')):
        if p.exists():
            return p / 'shin2018_word_generation_state_preparation_v1'
    return Path('/content/shin2018_word_generation_state_preparation_v1')
def sarray(vals: Sequence[Any]) -> np.ndarray:
    return np.asarray([str(v) for v in vals], dtype=str)
def atomic_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')
    tmp.replace(path)
def mat_load(path: Path) -> Dict[str, Any]:
    try:
        return loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError as exc:
        raise RuntimeError(f'{path} is MATLAB v7.3/HDF5; this dataset is expected to use classic MAT files.') from exc
def mat_var(path: Path, expected: str) -> Any:
    d = {k: v for k, v in mat_load(path).items() if not k.startswith('__')}
    if expected in d:
        return d[expected]
    if len(d) == 1:
        return next(iter(d.values()))
    for k, v in d.items():
        if k.lower() == expected.lower() or expected.lower() in k.lower():
            return v
    raise KeyError(f'{path.name}: cannot find {expected}; variables={list(d)}')
def field(obj: Any, name: str, default: Any=None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.void) and obj.dtype.names and (name in obj.dtype.names):
        return obj[name]
    return default
def strings(x: Any) -> List[str]:
    if x is None:
        return []
    a = np.asarray(x, dtype=object).reshape(-1)
    out = []
    for v in a:
        if isinstance(v, bytes):
            out.append(v.decode(errors='replace'))
        elif isinstance(v, np.ndarray) and v.ndim == 0:
            out.append(str(v.item()))
        else:
            out.append(str(v))
    return out
def continuous_x_fs_clab(cnt: Any) -> Tuple[np.ndarray, float, List[str]]:
    x = np.asarray(field(cnt, 'x'), dtype=float)
    fs = float(np.asarray(field(cnt, 'fs')).reshape(-1)[0])
    clab = strings(field(cnt, 'clab'))
    if x.ndim != 2:
        raise RuntimeError(f'Expected continuous x [time,channel], got {x.shape}')
    if clab and x.shape[1] != len(clab) and (x.shape[0] == len(clab)):
        x = x.T
    if clab and x.shape[1] != len(clab):
        raise RuntimeError(f'Channel-label mismatch x={x.shape}, clab={len(clab)}')
    return (x, fs, clab)
def nirs_oxy_deoxy(cnt: Any) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
    oxy = field(cnt, 'oxy')
    deoxy = field(cnt, 'deoxy')
    if oxy is None or deoxy is None:
        raise RuntimeError('NIRS cnt_wg must contain .oxy and .deoxy structures')
    xo, fso, co = continuous_x_fs_clab(oxy)
    xd, fsd, cd = continuous_x_fs_clab(deoxy)
    if xo.shape != xd.shape or abs(fso - fsd) > 1e-09:
        raise RuntimeError(f'HbO/HbR mismatch: {xo.shape}/{xd.shape}, {fso}/{fsd}')
    clab = co if co else cd
    if co and cd and (co != cd):
        warnings.warn('HbO/HbR channel labels differ; using HbO labels')
    return (xo, xd, fso, clab)
def decode_markers(mrk: Any, modality: str) -> pd.DataFrame:
    times_ms = np.asarray(field(mrk, 'time'), dtype=float).reshape(-1)
    y = np.asarray(field(mrk, 'y'), dtype=float)
    class_names = [s.strip() for s in strings(field(mrk, 'className'))]
    n = len(times_ms)
    if y.ndim == 1:
        codes = y.reshape(-1)
        if len(codes) != n:
            raise RuntimeError(f'Marker y length {len(codes)} != times {n}')
        if modality == 'EEG':
            label = np.where(codes == 16, 'WG', np.where(codes == 32, 'BL', 'UNKNOWN'))
        else:
            label = np.where(codes == 1, 'WG', np.where(codes == 2, 'BL', 'UNKNOWN'))
    else:
        if y.shape[-1] == n:
            Y = y.reshape((-1, n))
        elif y.shape[0] == n:
            Y = y.reshape((n, -1)).T
        else:
            raise RuntimeError(f'Cannot align mrk.y shape {y.shape} to {n} markers')
        idx = np.argmax(Y, axis=0)
        if class_names and len(class_names) >= Y.shape[0]:
            normalized = []
            for c in class_names[:Y.shape[0]]:
                cu = c.upper()
                if 'WG' in cu or 'VERBAL' in cu or 'FLU' in cu:
                    normalized.append('WG')
                elif 'BL' in cu or 'BASE' in cu:
                    normalized.append('BL')
                else:
                    normalized.append(c)
            label = np.asarray([normalized[i] for i in idx], dtype=str)
        else:
            label = np.where(idx == 0, 'WG', 'BL')
    frame = pd.DataFrame({'marker_index': np.arange(n), 'onset_sec': times_ms / 1000.0, 'label': label})
    frame = frame[frame['label'].isin(['WG', 'BL'])].sort_values('onset_sec', kind='stable').reset_index(drop=True)
    if len(frame) != EXPECTED_TRIALS:
        raise RuntimeError(f'{modality}: expected {EXPECTED_TRIALS} WG/BL markers, found {len(frame)}')
    frame['trial_global'] = np.arange(EXPECTED_TRIALS, dtype=int)
    frame['session'] = frame['trial_global'] // EXPECTED_TRIALS_PER_SESSION + 1
    frame['trial_in_session'] = frame['trial_global'] % EXPECTED_TRIALS_PER_SESSION
    for session_id, g in frame.groupby('session'):
        counts = g['label'].value_counts().to_dict()
        if counts.get('WG', 0) != EXPECTED_WG_PER_SESSION or counts.get('BL', 0) != EXPECTED_BL_PER_SESSION:
            raise RuntimeError(f'{modality} session {session_id}: expected 10 WG/10 BL, got {counts}')
    return frame
def regress_eog(x: np.ndarray, n_scalp: int) -> Tuple[np.ndarray, Dict[str, object]]:
    scalp = np.asarray(x[:, :n_scalp], float)
    eog = np.asarray(x[:, n_scalp:], float)
    if eog.shape[1] == 0:
        return (scalp.copy(), {'eog_channels': 0, 'regression_applied': 0, 'variance_removed_fraction': 0.0})
    finite = np.isfinite(eog).all(axis=1) & np.isfinite(scalp).all(axis=1)
    if finite.sum() < 100:
        return (scalp.copy(), {'eog_channels': eog.shape[1], 'regression_applied': 0, 'variance_removed_fraction': np.nan})
    E = eog.copy()
    E -= np.nanmean(E[finite], axis=0, keepdims=True)
    S = scalp.copy()
    mu = np.nanmean(S[finite], axis=0, keepdims=True)
    Sd = S - mu
    beta = np.linalg.pinv(E[finite]) @ Sd[finite]
    cleaned = Sd - E @ beta + mu
    before = np.nanvar(S[finite])
    after = np.nanvar(cleaned[finite])
    frac = 1.0 - after / before if before > 0 else np.nan
    return (cleaned, {'eog_channels': eog.shape[1], 'regression_applied': 1, 'variance_removed_fraction': float(frac)})
def sos_forward(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return signal.sosfilt(sos, x, axis=0)
def sos_zero(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return signal.sosfiltfilt(sos, x, axis=0)
def epochs_from_continuous(x: np.ndarray, fs: float, onsets: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samp = int(round((EPOCH_TMAX - EPOCH_TMIN) * fs)) + 1
    times = np.arange(n_samp, dtype=float) / fs + EPOCH_TMIN
    eps = np.full((len(onsets), x.shape[1], n_samp), np.nan, dtype=np.float32)
    ok = np.zeros(len(onsets), dtype=bool)
    for i, onset in enumerate(onsets):
        center = int(round(float(onset) * fs))
        start = center + int(round(EPOCH_TMIN * fs))
        stop = start + n_samp
        if start < 0 or stop > x.shape[0]:
            continue
        seg = x[start:stop, :].T
        if seg.shape[1] == n_samp:
            eps[i] = seg.astype(np.float32)
            ok[i] = np.isfinite(seg).all()
    return (eps, times, ok)
def baseline_correct(ep: np.ndarray, times: np.ndarray) -> np.ndarray:
    mask = (times >= BASELINE[0] - 1e-09) & (times <= BASELINE[1] + 1e-09)
    if mask.sum() < 2:
        raise RuntimeError('Baseline mask too small')
    base = np.nanmean(ep[:, :, mask], axis=2, keepdims=True)
    return (ep - base).astype(np.float32)
def robust_z(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, float)
    med = np.nanmedian(a)
    mad = np.nanmedian(np.abs(a - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        scale = np.nanstd(a)
    if not np.isfinite(scale) or scale < 1e-12:
        return np.zeros_like(a)
    return (a - med) / scale
def trial_quality(eeg: np.ndarray, eeg_times: np.ndarray, hbo: np.ndarray, hbr: np.ndarray, nirs_times: np.ndarray, eeg_fs: float, nirs_fs: float) -> pd.DataFrame:
    em = (eeg_times >= 0) & (eeg_times <= TASK_END_SEC)
    nm = (nirs_times >= 0) & (nirs_times <= TASK_END_SEC)
    eeg_abs = np.nanmax(np.abs(eeg[:, :, em]), axis=(1, 2))
    eeg_ptp = np.nanmax(np.ptp(eeg[:, :, em], axis=2), axis=1)
    dhbo = np.diff(hbo[:, :, nm], axis=2) * nirs_fs
    dhbr = np.diff(hbr[:, :, nm], axis=2) * nirs_fs
    nirs_deriv = np.nanmax(np.maximum(np.abs(dhbo), np.abs(dhbr)), axis=(1, 2))
    out = pd.DataFrame({'eeg_max_abs': eeg_abs, 'eeg_max_ptp': eeg_ptp, 'nirs_max_abs_derivative': nirs_deriv})
    out['eeg_abs_robust_z'] = robust_z(eeg_abs)
    out['eeg_ptp_robust_z'] = robust_z(eeg_ptp)
    out['nirs_deriv_robust_z'] = robust_z(nirs_deriv)
    out['artifact_review_flag'] = ((np.abs(out['eeg_abs_robust_z']) > 6) | (np.abs(out['eeg_ptp_robust_z']) > 6) | (np.abs(out['nirs_deriv_robust_z']) > 6)).astype(int)
    return out
def discover_subject_dirs(raw: Path) -> List[Tuple[str, Path, Path]]:
    rows = []
    for eeg_dir in sorted(raw.glob('VP*-EEG')):
        m = re.search('VP(\\d{3})-EEG', eeg_dir.name)
        if not m:
            continue
        vp = f'VP{int(m.group(1)):03d}'
        nirs_dir = raw / f'{vp}-NIRS'
        if nirs_dir.is_dir():
            rows.append((vp, eeg_dir, nirs_dir))
    return rows
def self_tests() -> None:

    class Dummy:
        pass
    m = Dummy()
    m.time = np.arange(60) * 25000.0
    m.y = np.zeros((2, 60))
    for s in range(3):
        ids = np.arange(s * 20, (s + 1) * 20)
        m.y[0, ids[:10]] = 1
        m.y[1, ids[10:]] = 1
    m.className = np.array(['WG', 'BL'], dtype=object)
    f = decode_markers(m, 'EEG')
    assert len(f) == 60 and f.groupby('session')['label'].value_counts().min() == 10
    x = np.arange(200 * 100 * 2, dtype=float).reshape(200 * 100, 2)
    ep, t, ok = epochs_from_continuous(x, 200, [10, 40, 70])
    assert ep.shape[1] == 2 and ok.all()
    print('Shin 02 preprocessing self-tests: PASS')
def _process_shin_participants(alignment_all, eeg_audit, failures, nirs_audit, quality_all, stage, subjects, summary):
    for idx, (vp, eeg_dir, nirs_dir) in enumerate(subjects, 1):
        print(f'\n[{idx:02d}/{len(subjects):02d}] {vp}')
        try:
            eeg_cnt = mat_var(eeg_dir / 'cnt_wg.mat', 'cnt_wg')
            eeg_mrk = mat_var(eeg_dir / 'mrk_wg.mat', 'mrk_wg')
            eeg_mnt = mat_var(eeg_dir / 'mnt_wg.mat', 'mnt_wg')
            nirs_cnt = mat_var(nirs_dir / 'cnt_wg.mat', 'cnt_wg')
            nirs_mrk = mat_var(nirs_dir / 'mrk_wg.mat', 'mrk_wg')
            nirs_mnt = mat_var(nirs_dir / 'mnt_wg.mat', 'mnt_wg')
            eeg_x, eeg_fs, eeg_clab = continuous_x_fs_clab(eeg_cnt)
            mnt_eeg_clab = strings(field(eeg_mnt, 'clab'))
            if not eeg_clab:
                eeg_clab = mnt_eeg_clab
            if len(eeg_clab) < 28:
                raise RuntimeError(f'{vp}: expected >=28 EEG/EOG labels, got {len(eeg_clab)}')
            n_scalp = len(mnt_eeg_clab) - 2 if len(mnt_eeg_clab) >= 30 else min(28, len(eeg_clab) - 2)
            if n_scalp != 28:
                warnings.warn(f'{vp}: inferred scalp count {n_scalp}; expected 28')
            scalp_names = eeg_clab[:n_scalp]
            eog_names = eeg_clab[n_scalp:]
            eeg_mark = decode_markers(eeg_mrk, 'EEG')
            nirs_mark = decode_markers(nirs_mrk, 'NIRS')
            if eeg_mark['label'].tolist() != nirs_mark['label'].tolist():
                raise RuntimeError(f'{vp}: EEG/NIRS WG/BL marker sequence mismatch')
            meta = eeg_mark[['trial_global', 'session', 'trial_in_session', 'label']].copy()
            meta['participant'] = vp
            meta['trial_uid'] = [f'{vp}__S{int(s):02d}__T{int(t):02d}' for s, t in zip(meta.session, meta.trial_in_session)]
            meta['eeg_onset_sec'] = eeg_mark['onset_sec'].to_numpy(float)
            meta['nirs_onset_sec'] = nirs_mark['onset_sec'].to_numpy(float)
            meta['task_end_sec_relative'] = TASK_END_SEC
            eg = np.r_[np.diff(meta['eeg_onset_sec']), np.nan]
            ng = np.r_[np.diff(meta['nirs_onset_sec']), np.nan]
            meta['eeg_next_task_gap_sec'] = eg
            meta['nirs_next_task_gap_sec'] = ng
            eeg_reg, eog_info = regress_eog(eeg_x, n_scalp)
            if not 150 <= eeg_fs <= 250:
                warnings.warn(f'{vp}: EEG fs={eeg_fs}, expected ~200 Hz MATLAB resource')
            eeg_sos = signal.butter(EEG_FILTER_ORDER, EEG_BP, btype='bandpass', fs=eeg_fs, output='sos')
            eeg_causal = sos_forward(eeg_reg, eeg_sos)
            eeg_zero = sos_zero(eeg_reg, eeg_sos)
            ep_ec, t_e, ok_ec = epochs_from_continuous(eeg_causal, eeg_fs, meta.eeg_onset_sec)
            ep_ez, _, ok_ez = epochs_from_continuous(eeg_zero, eeg_fs, meta.eeg_onset_sec)
            ep_ec = baseline_correct(ep_ec, t_e)
            ep_ez = baseline_correct(ep_ez, t_e)
            hbo, hbr, nirs_fs, nirs_clab = nirs_oxy_deoxy(nirs_cnt)
            if not nirs_clab:
                nirs_clab = strings(field(nirs_mnt, 'clab'))
            if hbo.shape[1] != 36:
                warnings.warn(f'{vp}: NIRS channels={hbo.shape[1]}, paper describes 36')
            ep_hbo_raw, t_n, ok_no = epochs_from_continuous(hbo, nirs_fs, meta.nirs_onset_sec)
            ep_hbr_raw, _, ok_nd = epochs_from_continuous(hbr, nirs_fs, meta.nirs_onset_sec)
            ep_hbo_raw = baseline_correct(ep_hbo_raw, t_n)
            ep_hbr_raw = baseline_correct(ep_hbr_raw, t_n)
            n_sos = signal.butter(NIRS_FILTER_ORDER, NIRS_LP_HZ, btype='lowpass', fs=nirs_fs, output='sos')
            hbo_c = sos_forward(hbo, n_sos)
            hbr_c = sos_forward(hbr, n_sos)
            hbo_z = sos_zero(hbo, n_sos)
            hbr_z = sos_zero(hbr, n_sos)
            ep_hbo_c, _, ok_nc = epochs_from_continuous(hbo_c, nirs_fs, meta.nirs_onset_sec)
            ep_hbr_c, _, ok_ncr = epochs_from_continuous(hbr_c, nirs_fs, meta.nirs_onset_sec)
            ep_hbo_z, _, ok_nz = epochs_from_continuous(hbo_z, nirs_fs, meta.nirs_onset_sec)
            ep_hbr_z, _, ok_nzr = epochs_from_continuous(hbr_z, nirs_fs, meta.nirs_onset_sec)
            for arr in (ep_hbo_c, ep_hbr_c, ep_hbo_z, ep_hbr_z):
                arr[:] = baseline_correct(arr, t_n)
            all_ok = ok_ec & ok_ez & ok_no & ok_nd & ok_nc & ok_ncr & ok_nz & ok_nzr
            if all_ok.sum() < 50:
                raise RuntimeError(f'{vp}: only {all_ok.sum()}/60 trials have complete multimodal epochs')
            quality = trial_quality(ep_ec, t_e, ep_hbo_raw, ep_hbr_raw, t_n, eeg_fs, nirs_fs)
            quality.insert(0, 'trial_uid', meta.trial_uid)
            quality.insert(0, 'participant', vp)
            quality['complete_multimodal_epoch'] = all_ok.astype(int)
            quality_all.append(quality)
            meta['complete_multimodal_epoch'] = all_ok.astype(int)
            align = meta.copy()
            alignment_all.append(align)
            np.savez_compressed(
                stage / f'{vp}_WG_ALIGNED_PREPROCESSED.npz',
                eeg_causal_epochs=ep_ec.astype(np.float32),
                eeg_zero_phase_epochs=ep_ez.astype(np.float32),
                eeg_times=t_e.astype(np.float64),
                eeg_sfreq=np.asarray([eeg_fs], np.float64),
                eeg_channel_names=sarray(scalp_names),
                eeg_eog_names=sarray(eog_names),
                hbo_raw_epochs=ep_hbo_raw.astype(np.float32),
                hbr_raw_epochs=ep_hbr_raw.astype(np.float32),
                hbo_causal_lp02_epochs=ep_hbo_c.astype(np.float32),
                hbr_causal_lp02_epochs=ep_hbr_c.astype(np.float32),
                hbo_zero_lp02_epochs=ep_hbo_z.astype(np.float32),
                hbr_zero_lp02_epochs=ep_hbr_z.astype(np.float32),
                nirs_times=t_n.astype(np.float64),
                nirs_sfreq=np.asarray([nirs_fs], np.float64),
                nirs_channel_names=sarray(nirs_clab if nirs_clab else [f'NIRS{i + 1:02d}' for i in range(hbo.shape[1])]),
                participant=sarray([vp] * len(meta)),
                trial_uid=sarray(meta.trial_uid),
                session=meta.session.to_numpy(int),
                trial_in_session=meta.trial_in_session.to_numpy(int),
                trial_global=meta.trial_global.to_numpy(int),
                label=sarray(meta.label),
                y=(meta.label.to_numpy(str) == 'WG').astype(int),
                eeg_onset_sec=meta.eeg_onset_sec.to_numpy(float),
                nirs_onset_sec=meta.nirs_onset_sec.to_numpy(float),
                task_end_sec=np.full(len(meta), TASK_END_SEC, dtype=float),
                complete_multimodal_epoch=all_ok.astype(np.uint8),
                artifact_review_flag=quality.artifact_review_flag.to_numpy(np.uint8),
                preprocessing_version=sarray([VERSION] * len(meta))
            )
            eeg_audit.append({
                'participant': vp,
                'eeg_fs': eeg_fs,
                'continuous_samples': len(eeg_x),
                'continuous_channels': eeg_x.shape[1],
                'scalp_channels': n_scalp,
                'eog_channels': len(eog_names),
                'eog_regression_applied': eog_info['regression_applied'],
                'variance_removed_fraction': eog_info['variance_removed_fraction'],
                'primary_filter': '6th-order forward Butterworth 1-40 Hz',
                'sensitivity_filter': '6th-order zero-phase Butterworth 1-40 Hz',
                'baseline': '-5 to -2 s',
                'epoch': '-5 to +25 s'
            })
            nirs_audit.append({
                'participant': vp,
                'nirs_fs': nirs_fs,
                'continuous_samples': len(hbo),
                'channels': hbo.shape[1],
                'primary_branch': 'provided HbO/HbR, no temporal filter',
                'causal_sensitivity': '3rd-order forward Butterworth low-pass 0.2 Hz',
                'paper_sensitivity': '3rd-order zero-phase Butterworth low-pass 0.2 Hz',
                'baseline': '-5 to -2 s',
                'epoch': '-5 to +25 s',
                'SCI_available': 0
            })
            summary.append({
                'participant': vp,
                'trials': len(meta),
                'WG': int((meta.label == 'WG').sum()),
                'BL': int((meta.label == 'BL').sum()),
                'complete_epochs': int(all_ok.sum()),
                'artifact_review_flags': int(quality.artifact_review_flag.sum())
            })
            print(f'  saved trials={len(meta)} complete={all_ok.sum()} review_flags={quality.artifact_review_flag.sum()}')
        except Exception as exc:
            failures.append({'participant': vp, 'error_type': type(exc).__name__, 'error': repr(exc)})
            print('  FAILED:', repr(exc))
def main() -> None:
    print('=' * 136)
    print('SHIN 02 — PREPROCESS WORD-GENERATION EEG–NIRS')
    print('=' * 136)
    self_tests()
    root = output_root()
    raw = root / '01_download' / 'raw_matlab'
    stage = root / '02_preprocessed'
    audit = stage / 'audits'
    stage.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    if not raw.is_dir():
        raise FileNotFoundError(f'Run Stage 01 first. Missing {raw}')
    subjects = discover_subject_dirs(raw)
    if not subjects:
        raise FileNotFoundError(f'No VPxxx-EEG/NIRS pairs found in {raw}')
    alignment_all = []
    eeg_audit = []
    nirs_audit = []
    quality_all = []
    summary = []
    failures = []
    _process_shin_participants(alignment_all, eeg_audit, failures, nirs_audit, quality_all, stage, subjects, summary)
    if alignment_all:
        pd.concat(alignment_all, ignore_index=True).to_csv(audit / 'trial_alignment_audit.csv', index=False)
    pd.DataFrame(summary).to_csv(audit / 'participant_alignment_summary.csv', index=False)
    pd.DataFrame(eeg_audit).to_csv(audit / 'eeg_preprocessing_audit.csv', index=False)
    pd.DataFrame(nirs_audit).to_csv(audit / 'nirs_preprocessing_audit.csv', index=False)
    if quality_all:
        pd.concat(quality_all, ignore_index=True).to_csv(audit / 'trial_quality_review.csv', index=False)
    pd.DataFrame(failures, columns=['participant', 'error_type', 'error']).to_csv(audit / 'failures.csv', index=False)
    atomic_json({
        'version': VERSION,
        'completed_at_unix': time.time(),
        'participants_succeeded': len(summary),
        'participants_failed': len(failures)
    }, audit / 'COMPLETED.json')
    print('\n' + '=' * 136)
    print('SHIN 02 COMPLETE')
    print('=' * 136)
    print('Succeeded:', len(summary), 'Failed:', len(failures))
    print('Output:', stage)
    print('Next: run SHIN_03_RICH_FEATURE_EXTRACTION_WORD_GENERATION_COMPLETE.py')
if __name__ == '__main__':
    main()
