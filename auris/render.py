"""The spatial field — sound made visible, and read back.

Builds the tonotopic field (frequency x time, weighted by loudness), saves it as a
stdlib PGM image (no PIL/cv2 needed to WRITE), and reads it structurally back into
numbers. `irix_read` optionally hands the image to Irix — the literal "combine Irix
and Soundgarden" path — but Auris never depends on Irix being present.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_field(field, centers) -> dict:
    """Read a (freq x time) field: where the bright band sits, how much it moves,
    and overall loudness. This is the percept in render-then-read mode."""
    f = np.asarray(field, dtype=np.float32)
    if f.ndim != 2 or f.size == 0:
        return {"peak_hz": 0.0, "motion": 0.0, "brightness": 0.0}
    rowmean = f.mean(axis=1)
    peak = int(np.argmax(rowmean))
    peak_hz = float(centers[peak]) if peak < len(centers) else 0.0
    brightness = float(f.sum(axis=0).mean())            # mean per-frame loudness
    motion = float(f.std(axis=1).mean() / (f.mean() + 1e-6))
    return {"peak_hz": peak_hz, "motion": motion, "brightness": brightness}


def save_field(field, out_dir) -> str | None:
    """Write the field as a PGM (portable graymap, stdlib) — low freq at bottom."""
    f = np.asarray(field, dtype=np.float32)
    if f.size == 0:
        return None
    img = np.clip(f / (f.max() + 1e-9) * 255.0, 0, 255).astype(np.uint8)
    img = np.flipud(img)                                # low frequency at bottom
    p = Path(out_dir) / "last_field.pgm"
    with open(p, "wb") as fh:
        fh.write(b"P5\n%d %d\n255\n" % (img.shape[1], img.shape[0]))
        fh.write(img.tobytes())
    return str(p)


def irix_read(path) -> dict | None:
    """Hand the field image to Irix's model-free perception, if Irix is available.
    Fully guarded: any absence/error returns None (Irix is a bonus, not a dep)."""
    try:
        try:
            from irix import physics
        except ImportError:
            import sys
            src = Path(__file__).resolve().parents[2] / "irix" / "src"
            if src.is_dir() and str(src) not in sys.path:
                sys.path.insert(0, str(src))
            from irix import physics
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return None
        regions = physics.salient_regions(img, max_n=1)
        if not regions:
            return None
        attrs = physics.region_attributes(img, regions[0])
        return {"description": physics.physical_description(attrs),
                "box": [int(v) for v in regions[0]]}
    except Exception:
        return None
