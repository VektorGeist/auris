"""Bridge to Soundgarden's audio engine — Auris reuses it as the ear, not a fork.

Soundgarden's `Analyzer` is the transducer (FFT -> bass/mid/high/loud/centroid/
flux/f0). We load its `audio.py` DIRECTLY by file path (it needs only numpy and has
no intra-package imports), which avoids the `soundgarden` namespace-package collision
that a plain `from soundgarden import audio` hits. The path is derived from THIS
file (never hardcoded to one machine — the portability rule) and overridable via
AURIS_SOUNDGARDEN_AUDIO.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_cached = None


def _load_from_file(path: Path):
    spec = importlib.util.spec_from_file_location("soundgarden_audio", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # so @dataclass can resolve cls.__module__
    spec.loader.exec_module(mod)
    return mod


def soundgarden_audio():
    """Return Soundgarden's audio module (the ear). Cached."""
    global _cached
    if _cached is not None:
        return _cached

    override = os.environ.get("AURIS_SOUNDGARDEN_AUDIO")
    candidates = []
    if override:
        candidates.append(Path(override))
    # .../Projects/auris/auris/_ear.py -> parents[2] == .../Projects
    projects = Path(__file__).resolve().parents[2]
    candidates.append(projects / "soundgarden" / "soundgarden" / "audio.py")

    for c in candidates:
        if c and c.is_file():
            _cached = _load_from_file(c)
            return _cached

    # last resort: a real pip install exposing the submodule
    try:
        from soundgarden import audio  # type: ignore
        _cached = audio
        return audio
    except Exception as e:  # pragma: no cover - clear failure, not a crash
        raise ImportError(
            "Auris needs Soundgarden's audio engine (the ear). Expected "
            f"{projects / 'soundgarden' / 'soundgarden' / 'audio.py'} or set "
            "AURIS_SOUNDGARDEN_AUDIO to soundgarden's audio.py. Original: %s" % e
        ) from e
