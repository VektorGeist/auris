"""The cochlea — the ear that lays sound out in space.

Reuses Soundgarden's `Analyzer` for the smoothed feature vector (bass/mid/high/
loud/centroid/flux/f0) and ADDS the tonotopic place map: a log-frequency layout of
the block's energy. Tonotopy is how the biological cochlea works — low pitch at one
end, high at the other — so the place map IS "sound laid out in space", the thing
the rest of Auris perceives and remembers. Pure DSP; deterministic per block.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _ear

SR = 44100
BLOCK = 2048
N_PLACE = 32                 # tonotopic bins (cochlear "places")
FMIN, FMAX = 40.0, 16000.0


@dataclass
class Percept:
    """One frame of hearing."""
    t: float                 # frame start time, seconds
    bass: float
    mid: float
    high: float
    loud: float
    centroid: float          # timbre brightness, 0..1
    flux: float              # onset strength, 0..1
    f0: float                # fundamental pitch, Hz (0 = unpitched)
    place: np.ndarray        # N_PLACE tonotopic energies, L1-normalized
    energy: float            # raw spectral energy (pre-normalization)
    flatness: float          # spectral flatness, 0=tonal .. 1=noise


class Cochlea:
    """Feed mono float32 blocks, get Percepts. Built for one sample rate."""

    def __init__(self, sr: int = SR, block: int = BLOCK, n_place: int = N_PLACE):
        self.sr, self.block, self.n_place = sr, block, n_place
        audio = _ear.soundgarden_audio()
        self.analyzer = audio.Analyzer(sr=sr, block=block)        # the ear
        self.window = np.hanning(block).astype(np.float32)
        self.freqs = np.fft.rfftfreq(block, 1.0 / sr)
        edges = np.geomspace(FMIN, min(FMAX, sr / 2.0), n_place + 1)
        self.edges = edges
        self.centers = np.sqrt(edges[:-1] * edges[1:])            # bin centre Hz
        self._bin_idx = [
            np.where((self.freqs >= edges[i]) & (self.freqs < edges[i + 1]))[0]
            for i in range(n_place)
        ]

    def _place_map(self, block: np.ndarray):
        mag = np.abs(np.fft.rfft(block * self.window))
        v = np.array([mag[idx].sum() if idx.size else 0.0
                      for idx in self._bin_idx], dtype=np.float32)
        s = float(v.sum())
        # spectral flatness (geometric/arithmetic mean, skip DC): ~0 for a tone,
        # ~1 for noise. This is the tone-vs-noise discriminator the pitch gate lacks.
        band = mag[1:]
        gmean = float(np.exp(np.mean(np.log(band + 1e-9))))
        flatness = gmean / (float(np.mean(band)) + 1e-9)
        return (v / s if s > 0 else v), float(mag.sum()), float(flatness)

    def hear_block(self, block, t: float) -> Percept:
        b = np.asarray(block, dtype=np.float32)
        if len(b) < self.block:
            b = np.pad(b, (0, self.block - len(b)))
        b = b[: self.block]
        f = self.analyzer.analyze(b)                              # soundgarden
        place, energy, flatness = self._place_map(b)
        return Percept(t=t, bass=f.bass, mid=f.mid, high=f.high, loud=f.loud,
                       centroid=f.centroid, flux=f.flux, f0=f.f0,
                       place=place, energy=energy, flatness=flatness)
