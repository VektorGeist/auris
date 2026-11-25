"""Hands the package's internal audio engine to the cochlea."""
from __future__ import annotations

from . import audio_engine


def ear_audio():
    return audio_engine


def soundgarden_audio():
    return audio_engine
