"""Chladni patterns for AURIS — a square plate model, not a picture mapping.

The plate: a fixed table of modes of a square plate. Each mode pair (n, m),
n < m, carries its own resonant frequency, rising with n^2 + m^2 from the
reference mode (0, 1):

    f(n, m) = PLATE_REFERENCE_HZ * (n^2 + m^2)

A sound excites the plate through its whole spectrum: each mode's weight is
the spectral power the sound carries near that mode's resonant frequency,
under a Gaussian resonance window of RESONANCE_WIDTH_OCTAVES. The figure is
the weighted sum of the excited mode shapes, a mix, the way a real plate
answers — no mode is chosen by pitch. Pitch is one contributor among the
spectrum's whole shape, because a mode rings when the spectrum has energy
at the mode's frequency, whatever produced it.

The mode shapes are the standard idealized square-plate modes,
    cos(n pi x) cos(m pi y) - cos(m pi x) cos(n pi y),
whose nodal lines are where sand gathers on a real plate.

Coverage, stated in numbers: the table runs from the (0, 1) mode at
PLATE_REFERENCE_HZ to the (11, 12) mode at 14,575 Hz. Energy below about
40 Hz excites no mode measurably (the lowest resonance is 55 Hz and the
resonance window is a quarter octave wide), and modes above 14,575 Hz are
not in the table, so spectral detail above that reaches no mode either.

The legacy shader mapping further down (mode_numbers and its helpers) is a
VISUALIZATION OF PITCH AND BRIGHTNESS ported from SoundGarden's GLSL
shader. It is not the plate model, it is not used by the plate path, and it
carries no information beyond the pitch and centroid numbers it is fed.
"""
from __future__ import annotations

import numpy as np

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# --- the plate ----------------------------------------------------------------

PLATE_REFERENCE_HZ = 55.0     # resonant frequency of the (0, 1) mode
PLATE_MAX_MODE = 12           # modes are the pairs (n, m), 0 <= n < m <= 12
RESONANCE_WIDTH_OCTAVES = 0.25  # Gaussian sigma of a mode's resonance window

# The fixed mode table: ((n, m), resonant frequency in Hz), ordered by n, m.
PLATE_MODES = tuple(
    ((n, m), PLATE_REFERENCE_HZ * (n * n + m * m))
    for n in range(0, PLATE_MAX_MODE)
    for m in range(n + 1, PLATE_MAX_MODE + 1)
)
_MODE_FREQS = np.array([f for _, f in PLATE_MODES], dtype=np.float64)
_SIGMA_LN = RESONANCE_WIDTH_OCTAVES * np.log(2.0)


def _default_centers() -> np.ndarray:
    """The cochlea's place centres, rebuilt from its own constants."""
    from .cochlea import FMIN, FMAX, N_PLACE
    edges = np.geomspace(FMIN, FMAX, N_PLACE + 1)
    return np.sqrt(edges[:-1] * edges[1:])


def plate_weights(place, centers=None) -> np.ndarray:
    """Mode weights for one place field (the ear's spectrum, power shares).

    Each mode is weighted by the spectral power near its resonant
    frequency: the place shares are summed under a Gaussian window in
    log-frequency centred on the mode. The result is normalized to sum 1;
    a silent field (no power anywhere) gives all-zero weights.
    """
    place = np.asarray(place, dtype=np.float64)
    total = float(place.sum())
    if total <= 0:
        return np.zeros(len(PLATE_MODES), dtype=np.float64)
    place = place / total
    if centers is None:
        centers = _default_centers()
    log_ratio = (np.log(_MODE_FREQS)[:, None]
                 - np.log(np.asarray(centers, dtype=np.float64))[None, :])
    window = np.exp(-(log_ratio ** 2) / (2.0 * _SIGMA_LN ** 2))
    w = window @ place
    s = float(w.sum())
    return w / s if s > 0 else w


def weights_for_percept(percept) -> np.ndarray:
    """Mode weights excited by one percept's own place field."""
    return plate_weights(np.asarray(percept.place, dtype=np.float64))


def top_modes(weights, k: int = 5) -> list:
    """The strongest modes as [n, m, weight] triples, strongest first."""
    w = np.asarray(weights, dtype=np.float64)
    order = np.argsort(-w)[:k]
    return [[int(PLATE_MODES[i][0][0]), int(PLATE_MODES[i][0][1]),
             round(float(w[i]), 4)] for i in order if w[i] > 0]


# --- the figures ----------------------------------------------------------------

def mode_shape(n: int, m: int, size: int = 128) -> np.ndarray:
    """The signed vibration field of one mode of the square plate."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    px, py = np.meshgrid(x, y)
    return (np.cos(n * np.pi * px) * np.cos(m * np.pi * py)
            - np.cos(m * np.pi * px) * np.cos(n * np.pi * py))


_shape_cache: dict = {}


def _shapes(size: int) -> np.ndarray:
    if size not in _shape_cache:
        _shape_cache[size] = np.stack(
            [mode_shape(n, m, size) for (n, m), _ in PLATE_MODES])
    return _shape_cache[size]


def plate_field(weights, size: int = 128) -> np.ndarray:
    """The plate's figure under a mix of modes: the weighted sum of the
    excited mode shapes, rendered by its nodal lines (1 = nodal line,
    where sand gathers)."""
    w = np.asarray(weights, dtype=np.float64)
    field = np.tensordot(w, _shapes(size), axes=1)
    peak = float(np.abs(field).max())
    if peak > 0:
        field = field / peak
    return 1.0 - np.clip(np.abs(field) * 3.0, 0, 1)


def chladni_field(n: int, m: int, size: int = 128) -> np.ndarray:
    """One mode's figure alone, nodal lines as in plate_field."""
    field = mode_shape(n, m, size)
    return 1.0 - np.clip(np.abs(field) * 3.0, 0, 1)


def percept_to_chladni(percept, size: int = 128) -> np.ndarray:
    """The plate's figure for a single AURIS percept: the weighted mix of
    the modes the percept's spectrum excites."""
    return plate_field(weights_for_percept(percept), size)


def chladni_timeline(percepts, size: int = 128, as_images: bool = False):
    """Plate figures for a sequence of percepts, one per time block."""
    figures = []
    for p in percepts:
        field = percept_to_chladni(p, size)
        if as_images and _HAS_PIL:
            img_arr = (field * 255).astype(np.uint8)
            img = Image.fromarray(img_arr, mode='L')
            rgb = Image.new('RGB', (size, size), (10, 10, 25))
            white = Image.new('RGB', (size, size), (220, 200, 255))
            rgb.paste(white, (0, 0), img)
            figures.append(rgb)
        else:
            figures.append(field)
    return figures


def chladni_strip(percepts, size: int = 128, gap: int = 2):
    """A horizontal strip image of plate figures over time, or None."""
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


# --- legacy shader mapping: a visualization of pitch and brightness only --------
#
# Ported from SoundGarden's GLSL shader (soundgarden.frag, chladni() and
# cymatics()). mode_numbers maps a normalized pitch and a normalized
# centroid onto ONE mode pair by arithmetic on those two numbers. No plate
# is modelled, no spectrum is consulted, and the result carries no
# information beyond the pitch and brightness it was fed. Kept only where a
# display wants that look; the plate model above is the hearing path.

def mode_numbers(pitch_norm: float, centroid_norm: float, mid_norm: float = 0.5) -> tuple[int, int]:
    """LEGACY: one mode pair from pitch and centroid, for display only."""
    s = 1.0 + pitch_norm * 9.0
    n = int(s)  # floor
    m_offset = int(min(max(centroid_norm * 3.0 + mid_norm * 2.0, 0.0), 4.0))
    m = n + 1 + m_offset
    n = max(1, min(n, 10))
    m = max(1, min(m, 14))
    return n, m


def pitch_to_norm(f0_hz: float, fmin: float = 50.0, fmax: float = 2000.0) -> float:
    """LEGACY helper for mode_numbers: log-normalize a frequency to 0-1."""
    if f0_hz <= 0:
        return 0.5
    return float(np.clip(np.log2(f0_hz / fmin) / np.log2(fmax / fmin), 0, 1))


def centroid_to_norm(centroid_hz: float, fmin: float = 50.0, fmax: float = 8000.0) -> float:
    """LEGACY helper for mode_numbers: log-normalize a centroid to 0-1."""
    if centroid_hz <= 0:
        return 0.5
    return float(np.clip(np.log2(centroid_hz / fmin) / np.log2(fmax / fmin), 0, 1))
