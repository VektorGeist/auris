"""Phonetic features — the mid-level cues the superior temporal gyrus reads.

Between raw sound and whole words, the brain represents *phonetic features*: is
this frame voiced, a vowel, a fricative (s/sh/f), a plosive burst (p/t/k), a nasal
(m/n)? Plus where the formants sit. We compute these model-free from the per-frame
`Percept` scalars Auris already has, adding formant peaks (formants.py). The result
is a per-frame feature track that can enrich the spectrotemporal trajectory.

Voice-onset-time (VOT) — the gap between a plosive burst and the start of voicing —
is the classic categorical-perception cue (/b/ vs /p/ flips at ~30 ms). The 46 ms
Percept frames are too coarse to resolve it, so VOT gets its OWN fine-hop analysis
on the raw samples. The sharp threshold on a smooth continuum IS the categorical
boundary — pure geometry, no model.
"""
from __future__ import annotations

import numpy as np

from . import audio_io, formants
from .cochlea import BLOCK

# per-frame feature layout (all bounded ~0..1)
FEATURES = ["voiced", "vowel", "fricative", "plosive", "nasal",
            "f1n", "f2n", "f3n", "centroid", "flatness"]
K = len(FEATURES)
_FMAX_NORM = 3500.0


def _nrm(f):
    return 0.0 if (f is None or np.isnan(f)) else float(np.clip(f / _FMAX_NORM, 0.0, 1.0))


def frame_features(percepts, samples, sr):
    """Return (n_frames x K feature track, n_frames x 3 formant track in Hz).

    Manner cues come from the Percept scalars; formants are recomputed per block.
    Plosive detection needs the previous frame, so it's computed over the sequence.
    """
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
    """Peak normalized cross-correlation (NCCF) in the 80-400 Hz voicing band.

    Per-lag normalization (divide by the overlapping segments' own energies) gives
    ~1.0 at the true period for a clean tone regardless of lag — unlike a plain
    autocorrelation, whose triangular decay understates long-period (low-pitch)
    voicing."""
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
    """Time from a plosive burst to voicing onset -> categorical voiced/voiceless.

    Fine-hop (≈6 ms) so it can resolve the ~30 ms boundary the 46 ms Percept frames
    cannot. Returns vot_ms + the class the sharp boundary assigns.
    """
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
    # voicing uses a LONGER window than the energy frame: over ~23 ms a real tone
    # still peaks near 1.0 while transient burst noise averages well below the bar.
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
    """A compact human-readable read of a phonetic track (for describe_word)."""
    if feat_track.size == 0:
        return "no phonetic content"
    frac = feat_track.mean(axis=0)
    parts = []
    for name, f in zip(FEATURES[:5], frac[:5]):
        if f >= 0.25:
            parts.append(f"{int(round(f * 100))}% {name}")
    return ", ".join(parts) if parts else "unvoiced / non-speech"
