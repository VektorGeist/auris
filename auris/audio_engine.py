"""AURIS's own audio engine - the ear, inside the package.

Audio capture + feature extraction. Turns a mono audio block into a small
feature vector: band energies (bass/mid/high), loudness, spectral centroid
(timbre brightness), spectral flux (onset/rhythm), and fundamental frequency.
Pure DSP - no model, no AI.

This code was originally written for SoundGarden and has been moved into
AURIS itself, so AURIS is self-contained: no sibling project, no file-path
lookup, and no environment variable is needed to hear.

Two sources:
  - LiveSource: microphone via sounddevice (non-deterministic timing).
  - FileSource: a WAV, yielded in fixed blocks (deterministic).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

BLOCK = 2048          # samples per analysis frame
SR = 44100            # sample rate
BANDS = {"bass": (20, 100), "mid": (200, 2000), "high": (5000, 20000)}


@dataclass
class Features:
    """Normalized per-frame features, each roughly in 0..1."""
    bass: float = 0.0
    mid: float = 0.0
    high: float = 0.0
    loud: float = 0.0
    centroid: float = 0.0   # spectral centroid / nyquist -> timbre brightness
    flux: float = 0.0       # positive spectral flux -> onset strength
    f0: float = 0.0         # fundamental freq (Hz), 0 if none


class Envelope:
    """Per-feature attack/decay smoother + slow-decaying auto-gain to 0..1."""

    def __init__(self, attack=0.5, decay=0.08, gain_decay=0.999):
        self.attack, self.decay, self.gain_decay = attack, decay, gain_decay
        self.value = 0.0
        self.peak = 1e-6

    def push(self, raw: float) -> float:
        self.peak = max(raw, self.peak * self.gain_decay, 1e-6)
        norm = raw / self.peak
        a = self.attack if norm > self.value else self.decay
        self.value += a * (norm - self.value)
        return float(np.clip(self.value, 0.0, 1.0))


class Analyzer:
    """Stateful analyzer: feed mono float32 blocks, get smoothed Features."""

    # The spectrum is computed on a zero-padded FFT: the raw 2048-point grid
    # is 21.5 Hz wide, which at the bottom of the piano (A0 = 27.5 Hz, a
    # semitone = 1.6 Hz) cannot locate a peak at all (A0 read 25.4 Hz and
    # was named G#0). Padding to 16384 points samples the same windowed
    # spectrum on a 2.7 Hz grid, fine enough for the peak interpolation to
    # land inside a semitone across the whole range of the keys.
    NFFT = 16384
    # Pitch search band: the full range of the keys, A0 (27.5 Hz) through
    # C8 (4186 Hz), with margin. The old search stopped below 1000 Hz, so
    # every key from C6 up returned no pitch at all.
    PITCH_LO, PITCH_HI = 20.0, 5000.0

    def __init__(self, sr=SR, block=BLOCK):
        self.sr, self.block = sr, block
        self.window = np.hanning(block).astype(np.float32)
        self.freqs = np.fft.rfftfreq(self.NFFT, 1.0 / sr)
        self.masks = {n: (self.freqs >= lo) & (self.freqs < hi)
                      for n, (lo, hi) in BANDS.items()}
        self.prev_mag = np.zeros(len(self.freqs), dtype=np.float32)
        self.env = {k: Envelope() for k in
                    ("bass", "mid", "high", "loud", "flux")}

    def analyze(self, block: np.ndarray) -> Features:
        if len(block) < self.block:
            block = np.pad(block, (0, self.block - len(block)))
        block = block[: self.block].astype(np.float32)
        mag = np.abs(np.fft.rfft(block * self.window, n=self.NFFT))

        total = mag.sum() + 1e-9
        f = Features()
        f.bass = self.env["bass"].push(mag[self.masks["bass"]].sum())
        f.mid = self.env["mid"].push(mag[self.masks["mid"]].sum())
        f.high = self.env["high"].push(mag[self.masks["high"]].sum())
        f.loud = self.env["loud"].push(np.sqrt(np.mean(block ** 2)))
        f.centroid = float(np.clip(
            (self.freqs * mag).sum() / total / (self.sr / 2), 0.0, 1.0))
        f.flux = self.env["flux"].push(
            np.maximum(mag - self.prev_mag, 0.0).sum())
        self.prev_mag = mag

        idx = np.where((self.freqs >= self.PITCH_LO)
                       & (self.freqs <= self.PITCH_HI))[0]
        if idx.size and mag[idx].max(initial=0.0) > total * 0.02:
            k = int(idx[int(np.argmax(mag[idx]))])
            # Parabolic interpolation around the peak bin, on the padded
            # grid: recovers the peak's position between bins, which is
            # what lets a 440 Hz tone read 439.3 Hz instead of 431 Hz and
            # a 27.5 Hz tone read inside its own semitone at all.
            f.f0 = float(self._interp_peak(k, mag))
        return f

    def _interp_peak(self, k: int, mag) -> float:
        if k <= 0 or k >= len(mag) - 1:
            return float(self.freqs[k])
        a, b, c = float(mag[k - 1]), float(mag[k]), float(mag[k + 1])
        denom = a - 2.0 * b + c
        if denom == 0.0:
            return float(self.freqs[k])
        delta = 0.5 * (a - c) / denom
        delta = float(np.clip(delta, -1.0, 1.0))
        bin_hz = float(self.freqs[1] - self.freqs[0]) if len(self.freqs) > 1 \
            else self.sr / self.block
        return float(self.freqs[k] + delta * bin_hz)


def file_blocks(path: str, block=BLOCK):
    """Yield fixed-size mono float32 blocks from a WAV — deterministic."""
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    was_int = np.issubdtype(data.dtype, np.integer)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if was_int or np.abs(data).max() > 1.5:
        data = data / (np.abs(data).max() + 1e-9)
    for i in range(0, len(data) - block, block):
        yield data[i: i + block], sr
