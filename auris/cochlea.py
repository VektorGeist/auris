"""The cochlea — the ear that lays sound out in space.

Uses AURIS's own `Analyzer` (in `audio_engine.py`) for the smoothed feature vector (bass/mid/high/
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
# The floor is 20 Hz, below the lowest key (A0 = 27.5 Hz). At the old 40 Hz
# floor the bottom keys (A0 through D#1) sat under the first place entirely
# and all piled onto a single place reading.
FMIN, FMAX = 20.0, 16000.0
# The place map is computed on a zero-padded FFT, like the Analyzer's
# spectrum: the raw 21.5 Hz bin grid makes the map's fine structure depend
# on where a tone happens to fall between bins.
NFFT = 16384


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
        audio = _ear.ear_audio()
        self.analyzer = audio.Analyzer(sr=sr, block=block)        # the ear
        self.window = np.hanning(block).astype(np.float32)
        self.freqs = np.fft.rfftfreq(NFFT, 1.0 / sr)
        edges = np.geomspace(FMIN, min(FMAX, sr / 2.0), n_place + 1)
        self.edges = edges
        self.centers = np.sqrt(edges[:-1] * edges[1:])            # bin centre Hz
        # Soft (triangular) assignment of spectral power to places, on the
        # log-frequency axis. Hard bin sums discard WHERE inside a place a
        # tone's energy sits, and a place spans about 3.6 semitones, so at
        # high frequencies two different notes inside one place produced
        # identical place vectors while two notes straddling an edge sat at
        # maximum distance: a step function, not a hearing. With triangular
        # weights each bin's power is shared between the two nearest place
        # centres in proportion to its log-distance from them, so the place
        # vector moves smoothly as frequency moves and carries the tone's
        # position inside the place as well as the place itself.
        logc = np.log(self.centers)
        step = float(logc[1] - logc[0])
        logf = np.log(np.maximum(self.freqs, 1e-9))
        w = 1.0 - np.abs(logf[:, None] - logc[None, :]) / step
        w = np.clip(w, 0.0, None)
        w[self.freqs < FMIN] = 0.0
        self._place_w = w.astype(np.float32)

    def _place_map(self, block: np.ndarray):
        mag = np.abs(np.fft.rfft(block * self.window, n=NFFT))
        # Place energies are POWER sums (magnitude squared), not magnitude
        # sums. A windowed tone's leakage floor sits about 31 dB down in
        # magnitude but 62 dB down in power; with magnitude sums the small
        # places were dominated by that leakage, whose pattern depends on
        # where the tone falls between FFT bins, so at high frequencies the
        # same note repeated could sit farther apart than two different
        # notes. Power sums make the map read the tone's own energy
        # distribution instead of the window's leakage.
        v = (self._place_w.T @ (mag ** 2)).astype(np.float32)
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
        f = self.analyzer.analyze(b)                              # the ear
        place, energy, flatness = self._place_map(b)
        return Percept(t=t, bass=f.bass, mid=f.mid, high=f.high, loud=f.loud,
                       centroid=f.centroid, flux=f.flux, f0=f.f0,
                       place=place, energy=energy, flatness=flatness)
