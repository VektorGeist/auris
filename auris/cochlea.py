"""Tonotopic place map: a log-frequency layout of a block's energy across 32 places from 20 Hz to 16,000 Hz, with the ear's feature vector."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _ear

SR = 44100
BLOCK = 2048
N_PLACE = 32


FMIN, FMAX = 20.0, 16000.0


NFFT = 16384


@dataclass
class Percept:
    t: float
    bass: float
    mid: float
    high: float
    loud: float
    centroid: float
    flux: float
    f0: float
    place: np.ndarray
    energy: float
    flatness: float


class Cochlea:

    def __init__(self, sr: int = SR, block: int = BLOCK, n_place: int = N_PLACE):
        self.sr, self.block, self.n_place = sr, block, n_place
        audio = _ear.ear_audio()
        self.analyzer = audio.Analyzer(sr=sr, block=block)
        self.window = np.hanning(block).astype(np.float32)
        self.freqs = np.fft.rfftfreq(NFFT, 1.0 / sr)
        edges = np.geomspace(FMIN, min(FMAX, sr / 2.0), n_place + 1)
        self.edges = edges
        self.centers = np.sqrt(edges[:-1] * edges[1:])


        logc = np.log(self.centers)
        step = float(logc[1] - logc[0])
        logf = np.log(np.maximum(self.freqs, 1e-9))
        w = 1.0 - np.abs(logf[:, None] - logc[None, :]) / step
        w = np.clip(w, 0.0, None)
        w[self.freqs < FMIN] = 0.0
        self._place_w = w.astype(np.float32)

    def _place_map(self, block: np.ndarray):
        mag = np.abs(np.fft.rfft(block * self.window, n=NFFT))


        v = (self._place_w.T @ (mag ** 2)).astype(np.float32)
        s = float(v.sum())


        band = mag[1:]
        gmean = float(np.exp(np.mean(np.log(band + 1e-9))))
        flatness = gmean / (float(np.mean(band)) + 1e-9)
        return (v / s if s > 0 else v), float(mag.sum()), float(flatness)

    def hear_block(self, block, t: float) -> Percept:
        b = np.asarray(block, dtype=np.float32)
        if len(b) < self.block:
            b = np.pad(b, (0, self.block - len(b)))
        b = b[: self.block]
        f = self.analyzer.analyze(b)
        place, energy, flatness = self._place_map(b)
        return Percept(t=t, bass=f.bass, mid=f.mid, high=f.high, loud=f.loud,
                       centroid=f.centroid, flux=f.flux, f0=f.f0,
                       place=place, energy=energy, flatness=flatness)
