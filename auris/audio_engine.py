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

    def __init__(self, sr=SR, block=BLOCK):
        self.sr, self.block = sr, block
        self.window = np.hanning(block).astype(np.float32)
        self.freqs = np.fft.rfftfreq(block, 1.0 / sr)
        self.masks = {n: (self.freqs >= lo) & (self.freqs < hi)
                      for n, (lo, hi) in BANDS.items()}
        self.prev_mag = np.zeros(len(self.freqs), dtype=np.float32)
        self.env = {k: Envelope() for k in
                    ("bass", "mid", "high", "loud", "flux")}

    def analyze(self, block: np.ndarray) -> Features:
        if len(block) < self.block:
            block = np.pad(block, (0, self.block - len(block)))
        block = block[: self.block].astype(np.float32)
        mag = np.abs(np.fft.rfft(block * self.window))

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

        sub = self.freqs < 1000                     # crude pitch: peak below 1kHz
        if mag[sub].max(initial=0.0) > total * 0.02:
            f.f0 = float(self.freqs[sub][np.argmax(mag[sub])])
        return f


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
