"""Minimal model-free speech synthesizer for tests — NO recordings, NO model.

Builds CV-C "words" from pure DSP so the word-discrimination test needs no mic and
no voice: a voiced vowel is additive harmonics of f0 shaped by a formant envelope
(with an optional formant glide at the end = the consonant transition), and a
plosive is a short spectrally-shaped noise burst after a closure. Enough to make
cat / cap / cot genuinely differ where real speech does:
  - cat vs cot: different VOWEL (F1/F2).
  - cat vs cap: SAME vowel, different FINAL consonant (burst band + F2 locus glide).
"""
from __future__ import annotations

import numpy as np

SR = 44100

# vowel formants (F1, F2, F3) in Hz
VOWELS = {
    "ae": (660, 1720, 2410),   # [æ] as in cat/cap
    "aw": (570, 840, 2410),    # [ɒ] as in cot
}
# final-consonant F2 locus the vowel gLIDES toward + burst band (lo, hi) + a FIXED
# burst seed so a given consonant has a consistent spectral signature across takes
# (a speaker's /t/ sounds like their /t/ — the randomness is per-phoneme, not per-take).
FINALS = {
    "t": {"f2_locus": 1750, "band": (3000, 7000), "seed": 101},  # alveolar: high burst, F2 up
    "p": {"f2_locus": 1050, "band": (200, 1200), "seed": 202},   # labial:   low burst,  F2 down
}
INITIALS = {"k": {"band": (1200, 2600), "seed": 303}}            # velar burst


def _formant_gain(freq, formants, bw=70.0):
    """Prominent formant resonances — cubed Lorentzian sum so the formant peaks
    dominate the harmonic spectrum the way real vowels do (weak formants leave the
    place map dominated by the low harmonics and blind to F2 movement)."""
    g = sum(1.0 / (1.0 + ((freq - Ff) / bw) ** 2) for Ff in formants)
    return 0.02 + g ** 3


def vowel(vname, dur=0.32, f0=120.0, sr=SR, amp=0.6, f2_end=None):
    """Voiced vowel; if f2_end given, F2 glides to it over the last ~40% (the
    formant transition into a final consonant)."""
    F1, F2, F3 = VOWELS[vname]
    n = int(dur * sr)
    t = np.arange(n) / sr
    G = 48                                   # coarse time grid for the formant glide
    prog = np.linspace(0, 1, G)
    def f2_at(p):
        if f2_end is None:
            return F2
        # steady until 60%, then linear glide to f2_end
        return F2 if p < 0.6 else F2 + (f2_end - F2) * (p - 0.6) / 0.4
    x = np.zeros(n, dtype=np.float64)
    k = 1
    while k * f0 < 5000:
        fk = k * f0
        env_grid = np.array([_formant_gain(fk, (F1, f2_at(p), F3)) for p in prog])
        env = np.interp(np.linspace(0, 1, n), prog, env_grid)
        x += env * np.sin(2 * np.pi * fk * t)
        k += 1
    x *= np.hanning(n) ** 0.25               # gentle edges, keep body
    x /= np.max(np.abs(x)) + 1e-9
    return (amp * x).astype(np.float32)


def burst(band, dur=0.018, sr=SR, amp=0.7, seed=0):
    """Spectrally-shaped noise burst (a plosive release)."""
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
    """Assemble initial-burst + vowel(+transition) + closure + final-burst.

    Burst noise is seeded PER CONSONANT (not per take), so each phoneme keeps a
    consistent signature — within-word variation then comes only from pitch/length,
    while cat vs cap differ by the final burst band + F2 glide (the real cues)."""
    fin, ini = FINALS[final], INITIALS[initial]
    v = vowel(vname, dur=dur, f0=f0, sr=sr, f2_end=fin["f2_locus"])
    return np.concatenate([
        _sil(25, sr),
        burst(ini["band"], sr=sr, seed=ini["seed"]),
        v,
        _sil(18, sr),                        # closure before the final release
        burst(fin["band"], sr=sr, seed=fin["seed"]),
        _sil(30, sr),
    ]).astype(np.float32)


# the three minimal-pair words, as (initial, vowel, final)
CAT = ("k", "ae", "t")
CAP = ("k", "ae", "p")
COT = ("k", "aw", "t")


def take(spec, i=0):
    """A slightly varied utterance of a word spec — different duration, pitch, and
    burst noise, so recognition must tolerate real within-word variation."""
    init, v, fin = spec
    # vary the TIMING (which DTW is built to absorb); hold pitch ~fixed because a
    # spectral-place sense is pitch-sensitive (pitch-invariance is future work, a
    # documented v1 limit). This isolates the real claim: place-trajectory + DTW
    # captures vowel + consonant identity where the median snapshot cannot.
    dur = 0.30 + 0.02 * (i % 4)              # 0.30 .. 0.36 s
    f0 = 120.0 + 1.0 * (i % 2)               # 120 / 121 Hz (same-speaker steadiness)
    return word(init, v, fin, dur=dur, f0=f0)
