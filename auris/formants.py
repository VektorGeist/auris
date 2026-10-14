"""Model-free formant tracking — the resonances that make a vowel a vowel.

A vowel's identity is carried by its formants: the 2-3 lowest resonant peaks of the
vocal tract (F1, F2, F3). "ee" vs "oo" vs "ah" differ almost entirely in where F1
and F2 sit, and consonant identity shows up as how those peaks *move* into and out
of the vowel (formant transitions). The brain's superior temporal gyrus reads these.

We find them the honest DSP way — peak-pick the smoothed magnitude spectrum — NOT
with a trained model and NOT with LPC (which fits an all-pole vocal-tract model; we
stay purely descriptive). Fragile by nature: at high f0 the harmonics are far apart
and can be mistaken for formants, so we smooth first and report a confidence, NaN
when nothing resonant stands out. The matcher down-weights low-confidence frames.
"""
from __future__ import annotations

import numpy as np

FMIN, FMAX = 200.0, 3500.0       # F1..F3 live here for speech
MIN_GAP_HZ = 180.0               # don't call two peaks the same formant
N_FORMANTS = 3


def _smooth_logmag(mag):
    """Smoothed log-magnitude = a cheap spectral-envelope proxy (no LPC/model)."""
    logm = np.log(np.asarray(mag, dtype=np.float64) + 1e-9)
    k = np.hanning(9)
    k /= k.sum()
    return np.convolve(logm, k, mode="same")


def spectrum_peaks(mag, freqs, fmin=FMIN, fmax=FMAX, n=N_FORMANTS,
                   min_gap_hz=MIN_GAP_HZ):
    """Return (formants[n], confs[n]) — F1<F2<F3 in Hz, NaN where none found.

    Peaks are local maxima of the smoothed log-spectrum inside [fmin,fmax]; kept by
    prominence above the in-band median, spaced at least `min_gap_hz` apart, then
    sorted ascending by frequency. `conf` = prominence normalized to ~0..1.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    sm = _smooth_logmag(mag)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any():
        return (np.full(n, np.nan), np.zeros(n))
    base = float(np.median(sm[band]))
    cand = []
    idx = np.where(band)[0]
    for i in idx:
        if i <= 0 or i >= len(sm) - 1:
            continue
        if sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1] and sm[i] > base:
            cand.append((freqs[i], sm[i] - base))
    # strongest first, enforce spacing
    cand.sort(key=lambda c: c[1], reverse=True)
    kept: list = []
    for f, prom in cand:
        if all(abs(f - kf) >= min_gap_hz for kf, _ in kept):
            kept.append((f, prom))
        if len(kept) >= n:
            break
    kept.sort(key=lambda c: c[0])                       # ascending -> F1,F2,F3
    F = np.full(n, np.nan)
    C = np.zeros(n)
    if kept:
        scale = max(kept[0][1], 1e-6) if False else max(p for _, p in kept)
        for k, (f, prom) in enumerate(kept):
            F[k] = f
            C[k] = float(np.clip(prom / (scale + 1e-9), 0.0, 1.0))
    return F, C


def block_formants(block, window, freqs, **kw):
    """Formants of one raw audio block (recomputes the FFT the cochlea also does)."""
    b = np.asarray(block, dtype=np.float32)
    if len(b) < len(window):
        b = np.pad(b, (0, len(window) - len(b)))
    mag = np.abs(np.fft.rfft(b[: len(window)] * window))
    return spectrum_peaks(mag, freqs, **kw)


def transitions(formant_track):
    """Frame-to-frame delta of an (n_frames x n_formants) track = the movement cue.

    NaNs (unvoiced/unclear frames) propagate as NaN deltas — a real 'no transition
    measurable here', not a fake zero."""
    F = np.asarray(formant_track, dtype=np.float64)
    if F.shape[0] < 2:
        return np.zeros_like(F)
    d = np.diff(F, axis=0)
    return np.vstack([np.zeros((1, F.shape[1])), d])
