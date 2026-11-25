"""Square plate model: a fixed table of plate modes with resonant frequencies, weighted by a sound's spectrum."""
from __future__ import annotations

import numpy as np

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


PLATE_REFERENCE_HZ = 55.0
PLATE_MAX_MODE = 12
RESONANCE_WIDTH_OCTAVES = 0.25


PLATE_MODES = tuple(
    ((n, m), PLATE_REFERENCE_HZ * (n * n + m * m))
    for n in range(0, PLATE_MAX_MODE)
    for m in range(n + 1, PLATE_MAX_MODE + 1)
)
_MODE_FREQS = np.array([f for _, f in PLATE_MODES], dtype=np.float64)
_SIGMA_LN = RESONANCE_WIDTH_OCTAVES * np.log(2.0)


def _default_centers() -> np.ndarray:
    from .cochlea import FMIN, FMAX, N_PLACE
    edges = np.geomspace(FMIN, FMAX, N_PLACE + 1)
    return np.sqrt(edges[:-1] * edges[1:])


def plate_weights(place, centers=None) -> np.ndarray:
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
    return plate_weights(np.asarray(percept.place, dtype=np.float64))


def top_modes(weights, k: int = 5) -> list:
    w = np.asarray(weights, dtype=np.float64)
    order = np.argsort(-w)[:k]
    return [[int(PLATE_MODES[i][0][0]), int(PLATE_MODES[i][0][1]),
             round(float(w[i]), 4)] for i in order if w[i] > 0]


def mode_shape(n: int, m: int, size: int = 128) -> np.ndarray:
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
    w = np.asarray(weights, dtype=np.float64)
    field = np.tensordot(w, _shapes(size), axes=1)
    peak = float(np.abs(field).max())
    if peak > 0:
        field = field / peak
    return 1.0 - np.clip(np.abs(field) * 3.0, 0, 1)


def chladni_field(n: int, m: int, size: int = 128) -> np.ndarray:
    field = mode_shape(n, m, size)
    return 1.0 - np.clip(np.abs(field) * 3.0, 0, 1)


def percept_to_chladni(percept, size: int = 128) -> np.ndarray:
    return plate_field(weights_for_percept(percept), size)


def chladni_timeline(percepts, size: int = 128, as_images: bool = False):
    figures = []
    for p in percepts:
        field = percept_to_chladni(p, size)
        if as_images and _HAS_PIL:
            img_arr = (field * 255).astype(np.uint8)
            img = Image.fromarray(img_arr)
            rgb = Image.new('RGB', (size, size), (10, 10, 25))
            white = Image.new('RGB', (size, size), (220, 200, 255))
            rgb.paste(white, (0, 0), img)
            figures.append(rgb)
        else:
            figures.append(field)
    return figures


def chladni_strip(percepts, size: int = 128, gap: int = 2):
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


def mode_numbers(pitch_norm: float, centroid_norm: float, mid_norm: float = 0.5) -> tuple[int, int]:
    s = 1.0 + pitch_norm * 9.0
    n = int(s)
    m_offset = int(min(max(centroid_norm * 3.0 + mid_norm * 2.0, 0.0), 4.0))
    m = n + 1 + m_offset
    n = max(1, min(n, 10))
    m = max(1, min(m, 14))
    return n, m


def pitch_to_norm(f0_hz: float, fmin: float = 50.0, fmax: float = 2000.0) -> float:
    if f0_hz <= 0:
        return 0.5
    return float(np.clip(np.log2(f0_hz / fmin) / np.log2(fmax / fmin), 0, 1))


def centroid_to_norm(centroid_hz: float, fmin: float = 50.0, fmax: float = 8000.0) -> float:
    if centroid_hz <= 0:
        return 0.5
    return float(np.clip(np.log2(centroid_hz / fmin) / np.log2(fmax / fmin), 0, 1))
