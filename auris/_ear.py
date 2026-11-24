"""The ear, inside AURIS.

AURIS's `Analyzer` lives in this package, in `audio_engine.py`. This module
only hands that internal engine to the cochlea. There is no external lookup:
no sibling project, no file path, and no environment variable.
"""
from __future__ import annotations

from . import audio_engine


def ear_audio():
    """Return AURIS's own audio engine module (the ear)."""
    return audio_engine


def soundgarden_audio():
    """Legacy name kept for existing callers. Returns the internal engine.

    The name is historical only - nothing is loaded from SoundGarden.
    """
    return audio_engine
