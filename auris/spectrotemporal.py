"""A 24-step by 32-place auditory image of a sound's trajectory over time, matched by dynamic time warping."""
from __future__ import annotations

import numpy as np

from .gallery import chi2

T_STEPS = 24
N_PLACE = 32


def trim_to_voice(percepts):
    if not percepts:
        return 0, 0
    energy = np.array([float(p.energy) for p in percepts], dtype=np.float64)
    if energy.max() <= 0:
        return 0, len(percepts)
    floor = max(0.06 * energy.max(), 1.5 * float(np.median(energy)))
    above = np.where(energy > floor)[0]
    if len(above) == 0:
        return 0, len(percepts)
    start = max(0, int(above[0]) - 1)
    end = min(len(percepts), int(above[-1]) + 2)
    return start, end


def _resample_rows(P, T=T_STEPS):
    n = P.shape[0]
    if n == 0:
        return np.zeros((T, P.shape[1]), dtype=np.float32)
    if n == 1:
        return np.repeat(P, T, axis=0).astype(np.float32)
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, T)
    out = np.empty((T, P.shape[1]), dtype=np.float32)
    for c in range(P.shape[1]):
        out[:, c] = np.interp(dst, src, P[:, c])
    return out


def _l1_rows(M):
    s = M.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (M / s).astype(np.float32)


def trajectory(percepts, T=T_STEPS, extra=None):
    if not percepts:
        return {"traj": np.zeros((T, N_PLACE), dtype=np.float32),
                "n_frames": 0, "seg": (0, 0), "silent": True}
    P = np.array([p.place for p in percepts], dtype=np.float32)
    a, b = trim_to_voice(percepts)
    seg = P[a:b] if b > a else P
    silent = not (b > a)
    if extra is not None:
        E = np.asarray(extra, dtype=np.float32)[a:b] if b > a else np.asarray(extra, np.float32)
        seg = np.concatenate([_l1_rows(seg), E], axis=1)
        traj = _resample_rows(seg, T)
    else:
        traj = _l1_rows(_resample_rows(_l1_rows(seg), T))
    return {"traj": traj.astype(np.float32), "n_frames": int(b - a) or len(percepts),
            "seg": (int(a), int(b)), "silent": bool(silent)}


def segment_utterances(samples, sr, hop=512, win=1024,
                       min_ms=80.0, gap_ms=120.0):
    x = np.asarray(samples, dtype=np.float32)
    if len(x) < win:
        return []
    starts = np.arange(0, len(x) - win, hop)
    E = np.array([float(np.mean(x[i:i + win] ** 2)) for i in starts])
    if E.max() <= 0:
        return []
    floor = max(0.05 * E.max(), 2.0 * float(np.median(E)))
    voiced = E > floor

    runs = []
    a = None
    for i, v in enumerate(voiced):
        if v and a is None:
            a = i
        if (not v or i == len(voiced) - 1) and a is not None:
            b = i + 1 if v else i
            runs.append([a, b])
            a = None
    if not runs:
        return []

    gap_frames = gap_ms * sr / 1000.0 / hop
    merged = [runs[0]]
    for a, b in runs[1:]:
        if a - merged[-1][1] <= gap_frames:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    min_samp = min_ms * sr / 1000.0
    out = []
    for a, b in merged:
        s = int(starts[a])
        e = int(starts[min(b, len(starts) - 1)] + win)
        if e - s >= min_samp:
            out.append((s, min(e, len(x))))
    return out


def dtw(a, b, band=None):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return float("inf")
    if band is None:
        band = max(1, max(n, m) // 3)
    INF = float("inf")
    D = np.full((n + 1, m + 1), INF)
    L = np.zeros((n + 1, m + 1))
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        jlo = max(1, i - band)
        jhi = min(m, i + band)
        ai = a[i - 1]
        for j in range(jlo, jhi + 1):
            cost = 0.5 * np.sum((ai - b[j - 1]) ** 2 / (ai + b[j - 1] + 1e-10))

            best, bl = D[i - 1, j - 1], L[i - 1, j - 1]
            if D[i - 1, j] < best:
                best, bl = D[i - 1, j], L[i - 1, j]
            if D[i, j - 1] < best:
                best, bl = D[i, j - 1], L[i, j - 1]
            D[i, j] = cost + best
            L[i, j] = bl + 1
    total, length = D[n, m], L[n, m]
    if not np.isfinite(total) or length == 0:
        return float("inf")
    return float(total / length)


def _shift_freq(traj, s):
    traj = np.asarray(traj, dtype=np.float64)
    if s == 0:
        return traj.copy()
    T, F = traj.shape
    out = np.zeros_like(traj)
    if s > 0:
        if s < F:
            out[:, s:] = traj[:, :F - s]
    else:
        if -s < F:
            out[:, :F + s] = traj[:, -s:]
    rs = out.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return out / rs


def dtw_pitch_invariant(a, b, band=None, max_shift=4):
    best = float("inf")
    for s in range(-max_shift, max_shift + 1):
        d = dtw(a, _shift_freq(b, s) if s else b, band=band)
        if d < best:
            best = d
    return best
