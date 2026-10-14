"""Auris — Vanta's deterministic, model-free sense of hearing.

Sound -> tonotopic place map (the ear, via Soundgarden's FFT) -> structured text
(perception, Irix-style) -> episodic memory (hippocampus) -> recognition (gallery).
No AI, no model: pure DSP + geometry + remembered exemplars. Afferent-only (AGM):
it perceives and remembers, it never acts.

Sibling to Irix (the eyes). Working name; Flouk may rename.
"""
from __future__ import annotations

from .sense import Ear

__all__ = ["Ear", "hear_file"]
__version__ = "0.1.0"


def hear_file(wav, **kw):
    """Convenience: perceive a WAV file with a throwaway Ear on the default store."""
    return Ear().hear_file(wav, **kw)
