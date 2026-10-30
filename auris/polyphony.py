"""Multi-pitch candidate list for AURIS token packs.

Harmonic-stack scores for MIDI 21-108, top peaks with non-maximum
suppression of plus or minus one semitone, reported as the top 5.

This is a candidate list only. It does not replace the single-f0
reading in a token pack, and it is not a clean voicing:
octave-below duplicates can appear in the later slots, it fails on
noisy sounds, and it fails on sequential notes inside one window.
Scores are raw FFT-magnitude sums; compare score_norm
(candidate / top score), not the raw score, across segments.
"""

from __future__ import annotations

import numpy as np


def stack_scores(samples, sr):
    """Harmonic-stack scores keyed by MIDI note number."""
    seg = samples[: sr * 2] if len(samples) >= sr * 2 else samples
    if len(seg) == 0:
        return {}
    window = np.hanning(len(seg))
    spectrum = np.abs(np.fft.rfft(seg * window))
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)

    def energy_at(f):
        if f <= 0 or f >= freqs[-1]:
            return 0.0
        i = int(np.argmin(np.abs(freqs - f)))
        return float(np.max(spectrum[max(0, i - 1) : i + 2]))

    scores = {}
    for midi in range(21, 109):
        f = 440.0 * 2 ** ((midi - 69) / 12)
        scores[midi] = sum(
            w * energy_at(f * k) for k, w in [(1, 1.0), (2, 0.6), (3, 0.4), (4, 0.3)]
        )
    return scores


def top_peaks(scores, n=5):
    """Top peaks, suppressing neighbours within one semitone."""
    order = sorted(scores, key=lambda m: -scores[m])
    picked = []
    for m in order:
        if all(abs(m - p) > 1 for p in picked):
            picked.append(m)
        if len(picked) >= n:
            break
    return picked
