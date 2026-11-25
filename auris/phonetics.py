"""Phonetic features per frame: voicing, vowel, fricative, plosive and nasal fractions, formants, and voice onset time."""
from __future__ import annotations

import numpy as np

from . import audio_io, formants
from .cochlea import BLOCK


FEATURES = ["voiced", "vowel", "fricative", "plosive", "nasal",
            "f1n", "f2n", "f3n", "centroid", "flatness"]
K = len(FEATURES)
_FMAX_NORM = 3500.0


def _nrm(f):
    return 0.0 if (f is None or np.isnan(f)) else float(np.clip(f / _FMAX_NORM, 0.0, 1.0))


def frame_features(percepts, samples, sr):
    window = np.hanning(BLOCK).astype(np.float32)
    freqs = np.fft.rfftfreq(BLOCK, 1.0 / sr)
    blocks = list(audio_io.blocks(np.asarray(samples, np.float32), BLOCK))
    feats, ftrack = [], []
    maxE = max((float(p.energy) for p in percepts), default=1.0) or 1.0
    for i, p in enumerate(percepts):
        if i < len(blocks):
            F, C = formants.spectrum_peaks(
                np.abs(np.fft.rfft(blocks[i] * window)), freqs)
        else:
            F, C = np.full(3, np.nan), np.zeros(3)
        voiced = 1.0 if (p.f0 > 0 and p.flatness <= 0.3) else 0.0
        has_form = 0.0 if np.isnan(F[0]) else 1.0
        vowel = 1.0 if (voiced and has_form and p.flatness < 0.2) else 0.0
        fricative = 1.0 if (not voiced and p.high > 0.3 and p.flatness > 0.5) else 0.0
        plosive = 0.0
        if i > 0 and p.flux > 0.4 and float(percepts[i - 1].energy) < 0.25 * maxE \
                and float(p.energy) > 0.4 * maxE:
            plosive = 1.0
        nasal = 1.0 if (voiced and p.bass > 0.4 and
                        (np.isnan(F[1]) or C[1] < 0.2)) else 0.0
        feats.append([voiced, vowel, fricative, plosive, nasal,
                      _nrm(F[0]), _nrm(F[1]), _nrm(F[2]),
                      float(p.centroid), float(p.flatness)])
        ftrack.append(F)
    return np.array(feats, dtype=np.float32), np.array(ftrack, dtype=np.float64)


def _periodicity(seg, sr):
    seg = np.asarray(seg, dtype=np.float64)
    seg = seg - seg.mean()
    N = len(seg)
    if float(np.sum(seg ** 2)) < 1e-9:
        return 0.0
    lo, hi = int(sr / 400), min(int(sr / 80), N - 1)
    best = 0.0
    for lag in range(lo, hi + 1):
        a, b = seg[:N - lag], seg[lag:]
        denom = np.sqrt(float(np.sum(a ** 2)) * float(np.sum(b ** 2))) + 1e-9
        val = float(np.sum(a * b)) / denom
        if val > best:
            best = val
    return best


def voice_onset_time(samples, sr, hop=256, win=512, boundary_ms=30.0):
    x = np.asarray(samples, dtype=np.float32)
    if len(x) < win * 2:
        return {"vot_ms": None, "phon_class": None, "boundary_ms": boundary_ms}
    starts = list(range(0, len(x) - win, hop))
    E = np.array([float(np.mean(x[i:i + win] ** 2)) for i in starts])
    if E.max() <= 0:
        return {"vot_ms": None, "phon_class": None, "boundary_ms": boundary_ms}
    thr = 0.15 * E.max()
    above = np.where(E > thr)[0]
    if len(above) == 0:
        return {"vot_ms": None, "phon_class": None, "boundary_ms": boundary_ms}
    burst = int(above[0])


    vwin = max(win, 1024)
    voice = None
    for k in range(burst, len(starts)):
        seg = x[starts[k]:starts[k] + vwin]
        if E[k] > thr and len(seg) >= vwin and _periodicity(seg, sr) > 0.6:
            voice = k
            break
    if voice is None:
        return {"vot_ms": None, "phon_class": None, "burst_frame": burst,
                "boundary_ms": boundary_ms}
    vot = (starts[voice] - starts[burst]) / sr * 1000.0
    return {"vot_ms": round(vot, 1),
            "phon_class": "voiceless" if vot >= boundary_ms else "voiced",
            "boundary_ms": boundary_ms}


def summarize(feat_track):
    if feat_track.size == 0:
        return "no phonetic content"
    frac = feat_track.mean(axis=0)
    parts = []
    for name, f in zip(FEATURES[:5], frac[:5]):
        if f >= 0.25:
            parts.append(f"{int(round(f * 100))}% {name}")
    return ", ".join(parts) if parts else "unvoiced / non-speech"
