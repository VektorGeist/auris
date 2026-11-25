"""Live microphone capture through an optional sounddevice backend."""
from __future__ import annotations

import numpy as np


def available() -> bool:
    import importlib.util
    return importlib.util.find_spec("sounddevice") is not None


def list_inputs() -> list:
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
    try:
        import sounddevice as sd
        info = sd.query_devices(device, "input") if device is not None else sd.query_devices(
            sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else None, "input")
        sr = int(info.get("default_samplerate", 0) or 0)
        return sr if sr > 0 else 44100
    except Exception:
        return 44100


def _sd_capture(frames: int, sr: int, channels: int, device):
    import sounddevice as sd
    rec = sd.rec(frames, samplerate=sr, channels=channels, dtype="float32", device=device)
    sd.wait()
    return rec


def _to_mono(rec) -> np.ndarray:
    a = np.asarray(rec, dtype=np.float32)
    if a.ndim == 2 and a.shape[1] > 1:
        a = a.mean(axis=1)
    a = a.reshape(-1)
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def record(seconds: float = 2.0, sr: int | None = None, device=None,
           channels: int = 1, gain: float = 1.0, _capture=None):
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
    if sr is None:
        sr = _default_samplerate(device)
    sr = int(sr)
    while True:
        yield record(window, sr=sr, device=device, channels=channels,
                     gain=gain, _capture=_capture)
