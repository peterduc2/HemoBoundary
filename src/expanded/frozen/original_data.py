"""Data-generation functions recovered from the user's original HemoBoundary source.
Only data loading, noise-profile estimation and deterministic generation are included.
No HemoBoundary fitting or configuration-selection routine is executed here.
The equations and constants are retained, not refitted to benchmark outcomes.
"""
from __future__ import annotations
import hashlib, math, json, re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple, List, Optional, Sequence
import numpy as np
import h5py
GRID_DT=0.5
GRID_END=20.0
BASELINE=(-5.0,0.0)
FULL_T=np.arange(-5.0,20.0+0.25,0.5)
POST_MASK=FULL_T>=0
GRID=FULL_T[POST_MASK]

def stable_seed(*parts):
    return int(hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:8],16) % (2**31-1)

import pandas as pd  # original HDF5 schema/inventory reporting only

def load_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    required = ["nirs_times", "hbo_causal_lp02_epochs", "hbr_causal_lp02_epochs"]
    missing = [k for k in required if k not in d]
    if missing:
        raise RuntimeError(f"{path.name}: missing {missing}")
    return d

@dataclass
class NoiseProfile:
    n_channels: int
    hbo_scale: np.ndarray
    hbr_scale: np.ndarray
    phi_hbo: float
    phi_hbr: float

def safe_lag1(a: np.ndarray) -> float:
    vals = []
    for row in a:
        x, y = row[:-1], row[1:]
        if len(x) >= 3 and np.std(x) > 1e-10 and np.std(y) > 1e-10:
            vals.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.clip(np.nanmedian(vals) if vals else 0.5, 0.0, 0.95))

def estimate_noise_profile(path: Path) -> NoiseProfile:
    d = load_npz(path)
    t = np.asarray(d["nirs_times"], float)
    H = np.asarray(d["hbo_causal_lp02_epochs"], float)
    R = np.asarray(d["hbr_causal_lp02_epochs"], float)
    bm = (t >= BASELINE[0]) & (t < BASELINE[1])
    if bm.sum() < 4:
        raise RuntimeError(f"{path.name}: insufficient pre-cue samples.")
    C = min(H.shape[1], R.shape[1])
    H = H[:, :C, :][:, :, bm]
    R = R[:, :C, :][:, :, bm]
    H = H - np.nanmedian(H, axis=2, keepdims=True)
    R = R - np.nanmedian(R, axis=2, keepdims=True)
    hs = np.nanmedian(np.nanstd(H, axis=2), axis=0)
    rs = np.nanmedian(np.nanstd(R, axis=2), axis=0)
    hs = np.where(np.isfinite(hs) & (hs > 1e-8), hs, np.nanmedian(hs[hs > 0]) if np.any(hs > 0) else 1.0)
    rs = np.where(np.isfinite(rs) & (rs > 1e-8), rs, np.nanmedian(rs[rs > 0]) if np.any(rs > 0) else 1.0)
    sample_H = H[: min(24, len(H))].reshape(-1, H.shape[-1])
    sample_R = R[: min(24, len(R))].reshape(-1, R.shape[-1])
    return NoiseProfile(C, hs, rs, safe_lag1(sample_H), safe_lag1(sample_R))

def ar_noise(scale: np.ndarray, phi: float, rng: np.random.Generator, n_time: int) -> np.ndarray:
    C = len(scale)
    x = np.zeros((C, n_time), float)
    common = rng.normal(size=n_time)
    for k in range(n_time):
        eps = 0.75 * rng.normal(size=C) + 0.25 * common[k]
        if k == 0:
            x[:, k] = eps
        else:
            x[:, k] = phi * x[:, k - 1] + math.sqrt(max(1.0 - phi**2, 1e-6)) * eps
    return x * scale[:, None]

def piecewise_shape(onset: float, rise: float, active: float, decay: float) -> Tuple[np.ndarray, float, float, np.ndarray]:
    peak = onset + rise
    active_end = peak + active
    offset = active_end + decay
    y = np.zeros(len(GRID), float)
    states = np.zeros(len(GRID), int)
    for i, tt in enumerate(GRID):
        if tt < onset:
            y[i] = 0.0; states[i] = 0
        elif tt < peak:
            u = (tt - onset) / max(rise, 1e-8)
            y[i] = 0.5 - 0.5 * np.cos(np.pi * u); states[i] = 1
        elif tt < active_end:
            y[i] = 1.0; states[i] = 2
        elif tt < offset:
            u = (tt - active_end) / max(decay, 1e-8)
            y[i] = 0.5 + 0.5 * np.cos(np.pi * u); states[i] = 3
        else:
            y[i] = 0.0; states[i] = 4
    return y, peak, offset, states

def synthetic_trial(profile: NoiseProfile, rng: np.random.Generator, positive: bool) -> Dict[str, Any]:
    H = ar_noise(profile.hbo_scale, profile.phi_hbo, rng, len(FULL_T))
    R = ar_noise(profile.hbr_scale, profile.phi_hbr, rng, len(FULL_T))
    systemic = rng.uniform(0.0, 0.8) * np.nanmedian(profile.hbo_scale) * np.sin(
        2 * np.pi * rng.uniform(0.06, 0.12) * FULL_T + rng.uniform(0, 2*np.pi)
    )
    H += systemic[None, :]
    R += rng.uniform(0.5, 1.0) * systemic[None, :]

    if not positive:
        return dict(
            hbo=H, hbr=R, present=0, onset=np.nan, peak=np.nan, offset=np.nan,
            censored=0, clean_shape=np.zeros(len(GRID), float),
            states=np.zeros(len(GRID), int),
        )

    onset = float(rng.uniform(0.5, 4.5))
    rise = float(rng.uniform(1.0, 3.5))
    active = float(rng.uniform(0.5, 3.0))
    if rng.random() < 0.18:
        decay = float(rng.uniform(7.0, 12.0))
    else:
        decay = float(rng.uniform(2.0, 8.0))
    shape, peak, offset, states = piecewise_shape(onset, rise, active, decay)

    post_shape = np.zeros(len(FULL_T), float)
    post_shape[POST_MASK] = shape
    support = rng.uniform(0.55, 1.0, size=profile.n_channels)
    amp = rng.uniform(1.5, 4.5) * profile.hbo_scale
    H += (amp * support)[:, None] * post_shape[None, :]
    R -= (amp * rng.uniform(0.35, 0.70) * rng.uniform(0.7, 1.1, size=profile.n_channels))[:, None] * post_shape[None, :]

    return dict(
        hbo=H, hbr=R, present=1, onset=onset, peak=peak, offset=offset,
        censored=int(offset > GRID_END - 0.25), clean_shape=shape, states=states,
    )

def baseline_z(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, float)
    bm = FULL_T < 0
    med = np.nanmedian(arr[:, bm], axis=1, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(arr[:, bm] - med), axis=1, keepdims=True)
    sd = np.nanstd(arr[:, bm], axis=1, keepdims=True)
    scale = np.where(np.isfinite(mad) & (mad > 1e-8), mad, np.where(sd > 1e-8, sd, 1.0))
    return (arr - med) / scale

def evidence(hbo: np.ndarray, hbr: np.ndarray) -> Dict[str, np.ndarray]:
    H = baseline_z(hbo)[:, POST_MASK]
    R = baseline_z(hbr)[:, POST_MASK]
    ghbo = np.nanmedian(H, axis=0)
    ghbr = np.nanmedian(R, axis=0)
    anti = 0.5 * (ghbo - ghbr)
    dhbo = np.gradient(ghbo, GRID_DT)
    dhbr = np.gradient(ghbr, GRID_DT)
    danti = np.gradient(anti, GRID_DT)
    dH = np.gradient(H, GRID_DT, axis=1)
    dR = np.gradient(R, GRID_DT, axis=1)
    cons_hbo = np.mean(np.sign(dH) == np.sign(dhbo)[None, :], axis=0)
    cons_hbr = np.mean(np.sign(dR) == np.sign(dhbr)[None, :], axis=0)
    iqr_hbo = np.nanquantile(H, .75, axis=0) - np.nanquantile(H, .25, axis=0)
    iqr_hbr = np.nanquantile(R, .75, axis=0) - np.nanquantile(R, .25, axis=0)
    return dict(H=H, R=R, ghbo=ghbo, ghbr=ghbr, anti=anti, dhbo=dhbo, dhbr=dhbr,
                danti=danti, cons_hbo=cons_hbo, cons_hbr=cons_hbr,
                iqr_hbo=iqr_hbo, iqr_hbr=iqr_hbr)

def whole_vector(ev: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([ev["ghbo"], ev["ghbr"], ev["anti"], ev["danti"]]).astype(np.float32)

def state_features(ev: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([
        ev["ghbo"], ev["ghbr"], ev["anti"], ev["dhbo"], ev["dhbr"],
        ev["danti"], ev["cons_hbo"], ev["cons_hbr"],
        0.5 * (ev["iqr_hbo"] + ev["iqr_hbr"]),
    ]).astype(np.float32)

def channel_anti(ev: Mapping[str, np.ndarray], common_c: int) -> np.ndarray:
    H = ev["H"][:common_c]
    R = ev["R"][:common_c]
    return (0.5 * (H - R)).astype(np.float32)

def _decode_str_array(a):
    a = np.asarray(a)
    out = []
    for x in a.reshape(-1):
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return np.asarray(out, dtype=str)

def _h5_dataset_inventory(path):
    rows = []
    with h5py.File(path, "r") as h5:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                rows.append(
                    dict(
                        path="/" + name,
                        shape=tuple(obj.shape),
                        dtype=str(obj.dtype),
                        ndim=int(obj.ndim),
                    )
                )
        h5.visititems(visit)
    return pd.DataFrame(rows)

def _all_datasets(h5):
    out = {}
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out["/" + name] = obj
    h5.visititems(visit)
    return out

def _score_name(name, tokens):
    low = name.lower()
    return sum(5 if low.endswith("/" + t) or low.split("/")[-1] == t else 1 for t in tokens if t in low)

def _find_time_dataset(datasets):
    candidates = []
    for name, ds in datasets.items():
        if ds.ndim != 1 or not np.issubdtype(ds.dtype, np.number):
            continue
        if len(ds) < 20 or len(ds) > 5000:
            continue
        score = _score_name(name, ("time", "times", "epoch_time", "t"))
        if score <= 0:
            continue
        a = np.asarray(ds[:], float).reshape(-1)
        if not np.isfinite(a).all() or not np.all(np.diff(a) > 0):
            continue
        if np.nanmin(a) > -0.1 or np.nanmax(a) < 10.0:
            continue
        candidates.append((score, name, a))
    if not candidates:
        raise RuntimeError(
            "Could not identify a monotonic HDF5 epoch time vector spanning pre/post stimulus."
        )
    candidates.sort(key=lambda x: (-x[0], len(x[2]), x[1]))
    return candidates[0][1], candidates[0][2]

def _find_participant_dataset(datasets):
    candidates = []

    allowed_leafs = {
        "participant",
        "participants",
        "participant_id",
        "participant_ids",
        "subject",
        "subjects",
        "subject_id",
        "subject_ids",
        "sub",
        "sub_id",
    }

    for name, ds in datasets.items():
        if ds.ndim != 1:
            continue

        leaf = name.rstrip("/").split("/")[-1].lower()

        # Deliberately inspect the DATASET NAME, not parent group names.
        # This prevents /participants/C01/trial_index from being interpreted
        # as a participant vector merely because "participants" appears above it.
        if leaf not in allowed_leafs:
            continue

        try:
            vals = _decode_str_array(ds[:])
        except Exception:
            continue

        unique = sorted(set(vals.tolist()))

        if not (2 <= len(unique) <= 200):
            continue

        # Require at least some participant-like values, or an explicitly
        # named participant/subject dataset.
        looks_like = sum(
            bool(
                re.fullmatch(
                    r"(?i)(?:sub[-_]?)?[A-Z]*\d{1,4}",
                    v.strip(),
                )
            )
            for v in unique
        )

        if looks_like == 0 and leaf in {"sub", "sub_id"}:
            continue

        score = 10
        if "participant" in leaf:
            score += 3
        if leaf.endswith("_id") or leaf.endswith("_ids"):
            score += 1

        candidates.append((score, name, vals))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1], candidates[0][2]

def _normalize_trial_channel_time(a, n_trials, n_time, label):
    x = np.asarray(a, float)
    if x.ndim != 3:
        raise RuntimeError(f"{label}: expected 3D trial/channel/time data, got {x.shape}")

    t_axes = [i for i, s in enumerate(x.shape) if s == n_time]
    if len(t_axes) != 1:
        raise RuntimeError(
            f"{label}: could not uniquely identify time axis length {n_time} in shape {x.shape}"
        )
    x = np.moveaxis(x, t_axes[0], -1)

    first_two = x.shape[:2]
    tr_axes = [i for i, s in enumerate(first_two) if s == n_trials]
    if len(tr_axes) != 1:
        raise RuntimeError(
            f"{label}: could not uniquely identify trial axis length {n_trials} in shape {x.shape}"
        )
    if tr_axes[0] == 1:
        x = np.swapaxes(x, 0, 1)

    if x.shape[0] != n_trials or x.shape[-1] != n_time:
        raise RuntimeError(f"{label}: normalization failed, shape={x.shape}")

    return x

def _find_separate_chromophore_pair(datasets, n_trials, n_time):
    hbo = []
    hbr = []

    for name, ds in datasets.items():
        if ds.ndim != 3 or not np.issubdtype(ds.dtype, np.number):
            continue

        low = name.lower()
        score_o = _score_name(name, ("hbo", "oxy", "oxygenated"))
        score_r = _score_name(name, ("hbr", "deoxy", "deoxygenated"))

        # Prevent "deoxy" from being accidentally scored as oxy.
        if "deoxy" in low or "deoxygenated" in low:
            score_o = 0

        if score_o > 0:
            hbo.append((score_o, name, ds))
        if score_r > 0:
            hbr.append((score_r, name, ds))

    pairs = []

    for so, no, dso in hbo:
        for sr, nr, dsr in hbr:
            try:
                O = _normalize_trial_channel_time(dso[:], n_trials, n_time, no)
                R = _normalize_trial_channel_time(dsr[:], n_trials, n_time, nr)
            except RuntimeError:
                continue
            if O.shape == R.shape:
                pairs.append((so + sr, no, nr, O, R))

    if not pairs:
        return None

    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    _, no, nr, O, R = pairs[0]
    return no, nr, O, R

def _find_channel_names(datasets):
    candidates = []
    for name, ds in datasets.items():
        if ds.ndim != 1:
            continue
        score = _score_name(name, ("channel_names", "channels", "ch_names", "channel"))
        if score <= 0:
            continue
        try:
            vals = _decode_str_array(ds[:])
        except Exception:
            continue
        if len(vals) >= 2:
            candidates.append((score, name, vals))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1], candidates[0][2]

def _normalize_combined_epochs(a, n_trials, n_time, n_channels, label):
    x = np.asarray(a, float)
    if x.ndim != 3:
        raise RuntimeError(f"{label}: combined epoch dataset must be 3D.")

    axes = list(range(3))
    ta = [i for i, s in enumerate(x.shape) if s == n_time]
    ca = [i for i, s in enumerate(x.shape) if s == n_channels]
    ra = [i for i, s in enumerate(x.shape) if s == n_trials]

    if len(ta) != 1 or len(ca) != 1 or len(ra) != 1:
        raise RuntimeError(
            f"{label}: could not uniquely map trial/channel/time axes in shape {x.shape}"
        )

    x = np.transpose(x, (ra[0], ca[0], ta[0]))
    return x

def _split_combined_by_channel_names(datasets, n_trials, n_time):
    cn_path, names = _find_channel_names(datasets)
    if names is None:
        return None

    idx_o = []
    idx_r = []
    keys_o = []
    keys_r = []

    for i, name in enumerate(names):
        low = name.lower()
        is_r = ("hbr" in low) or ("deoxy" in low) or ("deoxygenated" in low)
        is_o = ("hbo" in low) or (("oxy" in low or "oxygenated" in low) and not is_r)
        key = re.sub(
            r"(?i)(hbo|hbr|deoxy(?:genated)?|oxy(?:genated)?)",
            "",
            low,
        )
        key = re.sub(r"[^a-z0-9]+", "", key)

        if is_o:
            idx_o.append(i)
            keys_o.append(key)
        elif is_r:
            idx_r.append(i)
            keys_r.append(key)

    if not idx_o or not idx_r:
        return None

    # Pair HbO/HbR channels by normalized channel identity when possible.
    map_o = {k: i for k, i in zip(keys_o, idx_o) if k}
    map_r = {k: i for k, i in zip(keys_r, idx_r) if k}
    common = sorted(set(map_o) & set(map_r))

    if common:
        io = np.asarray([map_o[k] for k in common], int)
        ir = np.asarray([map_r[k] for k in common], int)
    else:
        c = min(len(idx_o), len(idx_r))
        io = np.asarray(idx_o[:c], int)
        ir = np.asarray(idx_r[:c], int)

    candidates = []

    for path, ds in datasets.items():
        if ds.ndim != 3 or not np.issubdtype(ds.dtype, np.number):
            continue
        score = _score_name(path, ("epoch", "epochs", "data", "hb", "nirs"))
        if score <= 0:
            continue
        try:
            X = _normalize_combined_epochs(
                ds[:],
                n_trials=n_trials,
                n_time=n_time,
                n_channels=len(names),
                label=path,
            )
        except RuntimeError:
            continue

        O = X[:, io, :]
        R = X[:, ir, :]
        candidates.append((score, path, cn_path, O, R))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[0], x[1]))
    _, data_path, _, O, R = candidates[0]
    return data_path + "::HbO/HbR channel split", cn_path, O, R

def _load_flat_hdf5(path):
    with h5py.File(path, "r") as h5:
        datasets = _all_datasets(h5)

        time_path, times = _find_time_dataset(datasets)
        participant_path, participants = _find_participant_dataset(datasets)

        if participants is None:
            return None

        n_trials = len(participants)
        n_time = len(times)

        pair = _find_separate_chromophore_pair(
            datasets,
            n_trials=n_trials,
            n_time=n_time,
        )

        if pair is None:
            pair = _split_combined_by_channel_names(
                datasets,
                n_trials=n_trials,
                n_time=n_time,
            )

        if pair is None:
            return None

        hbo_path, hbr_path, H, R = pair

    return dict(
        times=times,
        participant=participants,
        hbo=H,
        hbr=R,
        schema="FLAT_HDF5",
        audit=dict(
            time_dataset=time_path,
            participant_dataset=participant_path,
            hbo_source=hbo_path,
            hbr_source=hbr_path,
        ),
    )

def _participant_group_candidates(h5):
    candidates = []

    def visit(name, obj):
        if not isinstance(obj, h5py.Group):
            return
        leaf = name.rstrip("/").split("/")[-1]
        if re.fullmatch(r"(?i)(?:sub[-_])?[CS]?\d{1,3}", leaf):
            candidates.append(name)

    h5.visititems(visit)
    return sorted(set(candidates))

def _load_grouped_hdf5(path):
    participant_blocks = []
    audit_rows = []

    with h5py.File(path, "r") as h5:
        global_ds = _all_datasets(h5)

        try:
            global_time_path, global_times = _find_time_dataset(global_ds)
        except RuntimeError:
            global_time_path, global_times = None, None

        groups = _participant_group_candidates(h5)

        for gp in groups:
            grp = h5[gp]
            datasets = {}

            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets["/" + gp + "/" + name] = obj

            grp.visititems(visit)

            if not datasets:
                continue

            if global_times is not None:
                times = global_times
                time_path = global_time_path
            else:
                try:
                    time_path, times = _find_time_dataset(datasets)
                except RuntimeError:
                    continue

            # Infer trial count from a strong HbO/HbR pair. Here we allow trial axis
            # discovery by searching all shape pairs that contain time length.
            hbo_ds = []
            hbr_ds = []

            for name, ds in datasets.items():
                if ds.ndim != 3 or not np.issubdtype(ds.dtype, np.number):
                    continue
                low = name.lower()
                so = _score_name(name, ("hbo", "oxy", "oxygenated"))
                sr = _score_name(name, ("hbr", "deoxy", "deoxygenated"))
                if "deoxy" in low or "deoxygenated" in low:
                    so = 0
                if so > 0:
                    hbo_ds.append((so, name, ds))
                if sr > 0:
                    hbr_ds.append((sr, name, ds))

            best = None

            for so, no, dso in hbo_ds:
                for sr, nr, dsr in hbr_ds:
                    if tuple(dso.shape) != tuple(dsr.shape):
                        continue
                    shape = dso.shape
                    t_axes = [i for i, s in enumerate(shape) if s == len(times)]
                    if len(t_axes) != 1:
                        continue
                    ta = t_axes[0]
                    other = [i for i in range(3) if i != ta]
                    # Trial is usually the larger non-time axis for participant groups;
                    # if an explicit trial vector exists, prefer its length.
                    trial_lengths = []
                    for n2, ds2 in datasets.items():
                        if ds2.ndim == 1 and _score_name(n2, ("trial", "trials", "label", "event")) > 0:
                            trial_lengths.append(len(ds2))
                    trial_axis = None
                    for length in trial_lengths:
                        matching = [ax for ax in other if shape[ax] == length]
                        if len(matching) == 1:
                            trial_axis = matching[0]
                            break
                    if trial_axis is None:
                        # Channels in these prepared fNIRS stores are typically <= 64;
                        # choose the other axis as trials when only one looks channel-like.
                        channel_like = [ax for ax in other if shape[ax] <= 128]
                        if len(channel_like) == 1:
                            trial_axis = [ax for ax in other if ax != channel_like[0]][0]
                        else:
                            trial_axis = other[0]

                    channel_axis = [ax for ax in other if ax != trial_axis][0]

                    O = np.transpose(np.asarray(dso[:], float), (trial_axis, channel_axis, ta))
                    R = np.transpose(np.asarray(dsr[:], float), (trial_axis, channel_axis, ta))

                    score = so + sr
                    if best is None or score > best[0]:
                        best = (score, no, nr, O, R)

            if best is None:
                continue

            _, no, nr, O, R = best
            pid = gp.rstrip("/").split("/")[-1]
            participant_blocks.append((pid, times, O, R))
            audit_rows.append(
                dict(
                    participant=pid,
                    group="/" + gp,
                    hbo_source=no,
                    hbr_source=nr,
                    time_source=time_path,
                    n_trials=int(len(O)),
                    n_channels=int(O.shape[1]),
                )
            )

    if not participant_blocks:
        return None

    # Require common time vectors; concatenate into the flat internal structure.
    ref_t = participant_blocks[0][1]
    for pid, t, _, _ in participant_blocks:
        if len(t) != len(ref_t) or not np.allclose(t, ref_t, atol=1e-8, rtol=0):
            raise RuntimeError(
                f"Grouped HDF5 participant {pid} has a different time vector."
            )

    participants = []
    H = []
    R = []
    min_c = min(min(x[2].shape[1], x[3].shape[1]) for x in participant_blocks)

    # Different external participants can have different valid channel counts;
    # retaining min_c here is only for concatenation. Later participant-level
    # extraction applies validity filtering. Prepared Khan data is usually fixed 48.
    for pid, _, h, r in participant_blocks:
        participants.extend([pid] * len(h))
        H.append(h[:, :min_c, :])
        R.append(r[:, :min_c, :])

    return dict(
        times=np.asarray(ref_t, float),
        participant=np.asarray(participants, dtype=str),
        hbo=np.concatenate(H, axis=0),
        hbr=np.concatenate(R, axis=0),
        schema="GROUPED_HDF5",
        audit=dict(group_rows=audit_rows),
    )

def load_external_hdf5(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    flat = _load_flat_hdf5(path)
    if flat is not None:
        return flat

    grouped = _load_grouped_hdf5(path)
    if grouped is not None:
        return grouped

    inventory = _h5_dataset_inventory(path)
    raise RuntimeError(
        "Could not resolve HbO/HbR/time/participant arrays in external HDF5.\n"
        "The script wrote no results and will not guess a schema.\n"
        "HDF5 inventory:\n"
        + inventory.to_string(index=False)
    )

def _participant_arrays(store, pid):
    mask = store["participant"].astype(str) == str(pid)
    if not np.any(mask):
        raise RuntimeError(f"External participant {pid} has no epochs.")

    H = np.asarray(store["hbo"][mask], float)
    R = np.asarray(store["hbr"][mask], float)
    t = np.asarray(store["times"], float)

    C = min(H.shape[1], R.shape[1])
    H = H[:, :C, :]
    R = R[:, :C, :]

    bm = (t >= BASELINE[0]) & (t < BASELINE[1])
    if bm.sum() < 4:
        raise RuntimeError(f"{pid}: external store has insufficient -5..0 s baseline.")

    finite_fraction = np.mean(
        np.isfinite(H[:, :, bm]) & np.isfinite(R[:, :, bm]),
        axis=(0, 2),
    )

    hsd = np.nanmedian(np.nanstd(H[:, :, bm], axis=2), axis=0)
    rsd = np.nanmedian(np.nanstd(R[:, :, bm], axis=2), axis=0)

    valid = (
        (finite_fraction >= 0.80)
        & np.isfinite(hsd)
        & np.isfinite(rsd)
        & (hsd > 1e-12)
        & (rsd > 1e-12)
    )

    if valid.sum() < 2:
        # Allow robust use of channels that are mostly finite even if one
        # chromophore has extremely low baseline variance.
        valid = finite_fraction >= 0.80

    if valid.sum() < 2:
        raise RuntimeError(f"{pid}: fewer than two usable HbO/HbR channels.")

    return t, H[:, valid, :], R[:, valid, :], valid

def noise_profile_from_external(store, pid):
    t, H, R, valid = _participant_arrays(store, pid)

    bm = (t >= BASELINE[0]) & (t < BASELINE[1])

    Hb = H[:, :, bm]
    Rb = R[:, :, bm]

    Hb = Hb - np.nanmedian(Hb, axis=2, keepdims=True)
    Rb = Rb - np.nanmedian(Rb, axis=2, keepdims=True)

    hs = np.nanmedian(np.nanstd(Hb, axis=2), axis=0)
    rs = np.nanmedian(np.nanstd(Rb, axis=2), axis=0)

    hs_positive = hs[np.isfinite(hs) & (hs > 0)]
    rs_positive = rs[np.isfinite(rs) & (rs > 0)]

    h_fallback = float(np.nanmedian(hs_positive)) if len(hs_positive) else 1.0
    r_fallback = float(np.nanmedian(rs_positive)) if len(rs_positive) else 1.0

    hs = np.where(np.isfinite(hs) & (hs > 1e-12), hs, h_fallback)
    rs = np.where(np.isfinite(rs) & (rs > 1e-12), rs, r_fallback)

    sample_H = Hb[: min(24, len(Hb))].reshape(-1, Hb.shape[-1])
    sample_R = Rb[: min(24, len(Rb))].reshape(-1, Rb.shape[-1])

    profile = NoiseProfile(
        n_channels=int(len(hs)),
        hbo_scale=hs,
        hbr_scale=rs,
        phi_hbo=safe_lag1(sample_H),
        phi_hbr=safe_lag1(sample_R),
    )

    return profile, int(valid.sum()), int(len(valid))

