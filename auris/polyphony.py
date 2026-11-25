"""Multi-pitch candidate list: harmonic-stack scores for MIDI 21 to 108, top 5 with semitone non-maximum suppression."""

from __future__ import annotations

import numpy as np


def stack_scores(samples, sr):
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
    order = sorted(scores, key=lambda m: -scores[m])
    picked = []
    for m in order:
        if all(abs(m - p) > 1 for p in picked):
            picked.append(m)
        if len(picked) >= n:
            break
    return picked
