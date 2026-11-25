"""Model-free speech synthesizer for tests."""
from __future__ import annotations

import numpy as np

SR = 44100


VOWELS = {
    "ae": (660, 1720, 2410),
    "aw": (570, 840, 2410),
}


FINALS = {
    "t": {"f2_locus": 1750, "band": (3000, 7000), "seed": 101},
    "p": {"f2_locus": 1050, "band": (200, 1200), "seed": 202},
}
INITIALS = {"k": {"band": (1200, 2600), "seed": 303}}


def _formant_gain(freq, formants, bw=70.0):
    g = sum(1.0 / (1.0 + ((freq - Ff) / bw) ** 2) for Ff in formants)
    return 0.02 + g ** 3


def vowel(vname, dur=0.32, f0=120.0, sr=SR, amp=0.6, f2_end=None):
    F1, F2, F3 = VOWELS[vname]
    n = int(dur * sr)
    t = np.arange(n) / sr
    G = 48
    prog = np.linspace(0, 1, G)
    def f2_at(p):
        if f2_end is None:
            return F2

        return F2 if p < 0.6 else F2 + (f2_end - F2) * (p - 0.6) / 0.4
    x = np.zeros(n, dtype=np.float64)
    k = 1
    while k * f0 < 5000:
        fk = k * f0
        env_grid = np.array([_formant_gain(fk, (F1, f2_at(p), F3)) for p in prog])
        env = np.interp(np.linspace(0, 1, n), prog, env_grid)
        x += env * np.sin(2 * np.pi * fk * t)
        k += 1
    x *= np.hanning(n) ** 0.25
    x /= np.max(np.abs(x)) + 1e-9
    return (amp * x).astype(np.float32)


def burst(band, dur=0.018, sr=SR, amp=0.7, seed=0):
    n = int(dur * sr)
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1, 1, n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    lo, hi = band
    X *= (f >= lo) & (f <= hi)
    y = np.fft.irfft(X, n)
    y *= np.hanning(n)
    y /= np.max(np.abs(y)) + 1e-9
    return (amp * y).astype(np.float32)


def _sil(ms, sr=SR):
    return np.zeros(int(sr * ms / 1000.0), dtype=np.float32)


def word(initial, vname, final, dur=0.32, f0=120.0, sr=SR):
    fin, ini = FINALS[final], INITIALS[initial]
    v = vowel(vname, dur=dur, f0=f0, sr=sr, f2_end=fin["f2_locus"])
    return np.concatenate([
        _sil(25, sr),
        burst(ini["band"], sr=sr, seed=ini["seed"]),
        v,
        _sil(18, sr),
        burst(fin["band"], sr=sr, seed=fin["seed"]),
        _sil(30, sr),
    ]).astype(np.float32)


CAT = ("k", "ae", "t")
CAP = ("k", "ae", "p")
COT = ("k", "aw", "t")


def take(spec, i=0):
    init, v, fin = spec


    dur = 0.30 + 0.02 * (i % 4)
    f0 = 120.0 + 1.0 * (i % 2)
    return word(init, v, fin, dur=dur, f0=f0)
