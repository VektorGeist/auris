"""Live microphone as an ear — capture real sound so Auris hears the room, not a file.

The rest of Auris is deterministic DSP; a microphone is not, so this module is
kept SMALL, OPTIONAL and DEFENSIVE:

- The capture backend (``sounddevice`` / PortAudio) is imported lazily inside the
  functions, so the core package still imports and all file-based tests still run
  on a machine with no audio stack at all.
- Nothing about a device is hardcoded: the sample rate defaults to the input
  device's own default rate (the portability rule), and the device is selectable.
- Every capture is normalized to the SAME shape the file path produces — mono
  float32 in -1..1 — so ``Ear.hear_samples`` cannot tell a mic from a WAV.

A ``_capture`` hook is injectable so the normalization/downmix logic is unit-tested
without any hardware (see tests/test_listen.py).
"""
from __future__ import annotations

import numpy as np


def available() -> bool:
    """True if a microphone capture backend is importable on this machine."""
    import importlib.util
    return importlib.util.find_spec("sounddevice") is not None


def list_inputs() -> list:
    """Input devices as [{index, name, channels, default_samplerate, default}].

    Returns [] (never raises) if no backend / no devices — a sense with no ear
    simply hears nothing rather than crashing the tool.
    """
    try:
        import sounddevice as sd
    except Exception:
        return []
    try:
        default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
        out = []
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                out.append({
                    "index": i,
                    "name": d.get("name", "?"),
                    "channels": int(d["max_input_channels"]),
                    "default_samplerate": int(d.get("default_samplerate", 0) or 0),
                    "default": (i == default_in),
                })
        return out
    except Exception:
        return []


def _default_samplerate(device) -> int:
    """The chosen (or default) input device's own default sample rate."""
    try:
        import sounddevice as sd
        info = sd.query_devices(device, "input") if device is not None else sd.query_devices(
            sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else None, "input")
        sr = int(info.get("default_samplerate", 0) or 0)
        return sr if sr > 0 else 44100
    except Exception:
        return 44100


def _sd_capture(frames: int, sr: int, channels: int, device):
    """Default backend: blocking one-shot capture via PortAudio, float32 -1..1."""
    import sounddevice as sd
    rec = sd.rec(frames, samplerate=sr, channels=channels, dtype="float32", device=device)
    sd.wait()
    return rec


def _to_mono(rec) -> np.ndarray:
    """Any (frames,) or (frames, ch) float buffer -> mono float32 in -1..1."""
    a = np.asarray(rec, dtype=np.float32)
    if a.ndim == 2 and a.shape[1] > 1:
        a = a.mean(axis=1)
    a = a.reshape(-1)
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def record(seconds: float = 2.0, sr: int | None = None, device=None,
           channels: int = 1, gain: float = 1.0, _capture=None):
    """Capture ``seconds`` of live audio -> (sr, mono float32 in -1..1).

    sr defaults to the device's own default rate. ``gain`` optionally boosts a
    quiet mic (the Realtek array is known low-gain) BEFORE clipping; it never
    changes the sample rate. ``_capture`` is injectable for hardware-free tests.
    """
    if seconds <= 0:
        raise ValueError("seconds must be > 0")
    cap = _capture or _sd_capture
    if sr is None:
        sr = _default_samplerate(device)
    sr = int(sr)
    frames = int(round(seconds * sr))
    if frames < 1:
        raise ValueError("capture window too short for this sample rate")
    rec = cap(frames, sr, channels, device)
    mono = _to_mono(rec)
    if gain != 1.0:
        mono = np.clip(mono * float(gain), -1.0, 1.0).astype(np.float32)
    return sr, mono


def windows(window: float = 2.0, sr: int | None = None, device=None,
            channels: int = 1, gain: float = 1.0, _capture=None):
    """Yield (sr, mono float32) forever, one capture ``window`` at a time.

    A continuously-hearing sense, built by re-capturing back-to-back windows
    (small gap between windows is acceptable for v1 and documented). The caller
    decides when to stop (break / seconds budget). Afferent-only: this only
    listens, it never emits sound.
    """
    if sr is None:
        sr = _default_samplerate(device)
    sr = int(sr)
    while True:
        yield record(window, sr=sr, device=device, channels=channels,
                     gain=gain, _capture=_capture)
