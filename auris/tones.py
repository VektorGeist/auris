"""Deterministic synthetic sounds — so Auris can be RUN and tested on real audio
without a microphone or external files. Pure numpy; seeded noise.
"""
from __future__ import annotations

import numpy as np

SR = 44100


def _t(dur, sr):
    return np.linspace(0.0, dur, int(dur * sr), endpoint=False)


def tone(freq=440.0, dur=1.0, sr=SR, amp=0.6, harmonics=0):
    """A sine (optionally with `harmonics` decaying overtones -> a timbre)."""
    t = _t(dur, sr)
    x = np.sin(2 * np.pi * freq * t)
    for k in range(2, 2 + harmonics):
        x += (1.0 / k) * np.sin(2 * np.pi * freq * k * t)
    x /= np.max(np.abs(x)) + 1e-9
    return (amp * x).astype(np.float32)


def chord(freqs=(261.63, 329.63, 392.0), dur=1.0, sr=SR, amp=0.6):
    """Sum of tones -> a chord (a sound whose identity is its relationships)."""
    t = _t(dur, sr)
    x = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    x /= np.max(np.abs(x)) + 1e-9
    return (amp * x).astype(np.float32)


def sweep(f0=110.0, f1=1760.0, dur=1.5, sr=SR, amp=0.6):
    """A rising glide (log chirp) — exercises pitch-trend detection."""
    t = _t(dur, sr)
    k = (f1 / f0) ** (1.0 / dur)
    phase = 2 * np.pi * f0 * (k ** t - 1) / np.log(k)
    return (amp * np.sin(phase)).astype(np.float32)


def noise(dur=1.0, sr=SR, amp=0.5, seed=0):
    """Seeded white noise -> an unpitched sound (deterministic)."""
    rng = np.random.RandomState(seed)
    return (amp * rng.uniform(-1.0, 1.0, int(dur * sr))).astype(np.float32)


def clicks(rate=4.0, dur=2.0, sr=SR, amp=0.8):
    """Regular impulses -> rhythm (onset-rate) test signal."""
    x = np.zeros(int(dur * sr), dtype=np.float32)
    step = int(sr / rate)
    for i in range(0, len(x), step):
        x[i: i + 64] = amp * np.hanning(min(64, len(x) - i))
    return x
