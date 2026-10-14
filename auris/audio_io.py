"""WAV read/write + block iteration, stdlib `wave` only (no scipy dependency).

Kept deliberately small and defensive: real WAVs vary (8/16/32-bit, stereo,
odd lengths). We normalize to mono float32 in -1..1.
"""
from __future__ import annotations

import wave

import numpy as np

_INT_DTYPE = {1: np.uint8, 2: "<i2", 4: "<i4"}   # 8-bit WAV is UNSIGNED


def read_wav(path: str):
    """Return (sample_rate, mono float32 in -1..1)."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw not in _INT_DTYPE:
        raise ValueError(f"unsupported WAV sample width: {sw*8}-bit")
    data = np.frombuffer(raw, dtype=_INT_DTYPE[sw]).astype(np.float32)
    if ch > 1:                                   # interleaved -> mono mean
        data = data[: len(data) // ch * ch].reshape(-1, ch).mean(axis=1)
    if sw == 1:                                  # unsigned 0..255 -> -1..1
        data = (data - 128.0) / 128.0
    else:
        data = data / float(2 ** (8 * sw - 1))
    return sr, data


def write_wav(path: str, samples, sr: int) -> None:
    """Write mono float32 (-1..1) as 16-bit PCM."""
    s = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    ints = (s * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(ints.tobytes())


def blocks(samples, block: int = 2048):
    """Yield non-overlapping fixed-size blocks (drops a short final remainder)."""
    n = len(samples)
    for i in range(0, n - block + 1, block):
        yield samples[i: i + block]
