"""Chladni patterns for AURIS — SoundGarden's cymatics mode as a Python module.

This bridges SoundGarden's Chladni visualization (mode 1, the sand plate)
into AURIS. Pure numpy, no GPU needed. Does not modify existing AURIS code.

The math is ported directly from SoundGarden's GLSL shader:
    soundgarden/shaders/soundgarden.frag, function chladni() and cymatics()

Usage:
    from auris.chladni import chladni_timeline
    images = chladni_timeline(percepts)  # list of PIL Images, one per time block
"""
from __future__ import annotations

import numpy as np

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def chladni_field(n: int, m: int, size: int = 128) -> np.ndarray:
    """Generate a Chladni figure for mode (n, m).

    Ported from SoundGarden's GLSL:
        float chladni(vec2 p, float n, float m) {
            return cos(n * PI * p.x) * cos(m * PI * p.y)
                 - cos(m * PI * p.x) * cos(n * PI * p.y);
        }

    Returns a 2D array (size x size) with values 0-1,
    where 1 = nodal line (sand gathers here).
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    px, py = np.meshgrid(x, y)

    field = (np.cos(n * np.pi * px) * np.cos(m * np.pi * py)
             - np.cos(m * np.pi * px) * np.cos(n * np.pi * py))

    # Nodal lines are where field ~ 0
    nodal = 1.0 - np.clip(np.abs(field) * 3.0, 0, 1)
    return nodal


def mode_numbers(pitch_norm: float, centroid_norm: float, mid_norm: float = 0.5) -> tuple[int, int]:
    """Compute Chladni mode numbers (n, m) from audio features.

    Ported from SoundGarden's GLSL:
        float s = 1.0 + u_pitch * 9.0;
        float n = floor(s);
        float m = n + 1.0 + floor(clamp(u_centroid * 3.0 + u_mid * 2.0, 0.0, 4.0));

    Args:
        pitch_norm: normalized pitch 0-1 (from AURIS f0)
        centroid_norm: normalized spectral centroid 0-1 (from AURIS centroid, timbre)
        mid_norm: normalized mid-band energy 0-1 (optional)

    Returns:
        (n, m) mode numbers, clamped to reasonable range
    """
    s = 1.0 + pitch_norm * 9.0
    n = int(s)  # floor
    m_offset = int(min(max(centroid_norm * 3.0 + mid_norm * 2.0, 0.0), 4.0))
    m = n + 1 + m_offset

    # Clamp to sane range (avoid degenerate patterns)
    n = max(1, min(n, 10))
    m = max(1, min(m, 14))

    return n, m


def pitch_to_norm(f0_hz: float, fmin: float = 50.0, fmax: float = 2000.0) -> float:
    """Normalize a frequency in Hz to 0-1 on a log scale."""
    if f0_hz <= 0:
        return 0.5
    return float(np.clip(np.log2(f0_hz / fmin) / np.log2(fmax / fmin), 0, 1))


def centroid_to_norm(centroid_hz: float, fmin: float = 50.0, fmax: float = 8000.0) -> float:
    """Normalize a spectral centroid in Hz to 0-1 on a log scale."""
    if centroid_hz <= 0:
        return 0.5
    return float(np.clip(np.log2(centroid_hz / fmin) / np.log2(fmax / fmin), 0, 1))


def percept_to_chladni(percept, size: int = 128) -> np.ndarray:
    """Generate a Chladni figure from a single AURIS percept.

    Args:
        percept: AURIS percept object with .f0 (pitch in Hz) and .centroid (Hz)
        size: output image size in pixels

    Returns:
        2D numpy array (size x size), values 0-1
    """
    pitch_norm = pitch_to_norm(getattr(percept, 'f0', 0))
    centroid_norm = centroid_to_norm(getattr(percept, 'centroid', 1000))

    n, m = mode_numbers(pitch_norm, centroid_norm)
    return chladni_field(n, m, size)


def chladni_timeline(percepts, size: int = 128, as_images: bool = False):
    """Generate Chladni figures for a sequence of AURIS percepts.

    This is the "whole song over time" view — one Chladni figure per
    time block, showing how the vibration pattern evolves.

    Args:
        percepts: list of AURIS percept objects (from Ear._transduce)
        size: each figure's size in pixels
        as_images: if True and PIL available, return PIL Images; else numpy arrays

    Returns:
        List of figures (numpy arrays or PIL Images)
    """
    figures = []
    for p in percepts:
        field = percept_to_chladni(p, size)
        if as_images and _HAS_PIL:
            # Convert to PIL, white on dark
            img_arr = (field * 255).astype(np.uint8)
            img = Image.fromarray(img_arr, mode='L')
            # Colorize
            rgb = Image.new('RGB', (size, size), (10, 10, 25))
            white = Image.new('RGB', (size, size), (220, 200, 255))
            rgb.paste(white, (0, 0), img)
            figures.append(rgb)
        else:
            figures.append(field)
    return figures


def chladni_strip(percepts, size: int = 128, gap: int = 2):
    """Create a horizontal strip image of Chladni figures over time.

    Args:
        percepts: list of AURIS percept objects
        size: each figure's size in pixels
        gap: pixels between figures

    Returns:
        PIL Image of the timeline, or None if PIL not available
    """
    if not _HAS_PIL:
        return None

    figures = chladni_timeline(percepts, size=size, as_images=True)
    if not figures:
        return None

    total_width = len(figures) * (size + gap) - gap
    strip = Image.new('RGB', (total_width, size), (5, 5, 15))

    for i, fig in enumerate(figures):
        strip.paste(fig, (i * (size + gap), 0))

    return strip
