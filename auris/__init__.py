"""A deterministic, model-free sense of hearing."""
from __future__ import annotations

from .sense import Ear
from . import chladni

__all__ = ["Ear", "hear_file", "chladni"]
__version__ = "0.3.0"


def hear_file(wav, **kw):
    return Ear().hear_file(wav, **kw)
