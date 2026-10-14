"""Sound recognition gallery — "memory, not inference" (Irix's principle).

A growing zip-backed store of remembered sound exemplars. A sound is fingerprinted
by its tonotopic descriptor (the aggregated place map — an L1-normalized histogram),
and recognition is nearest-neighbour by chi-square distance over everything ever
heard. No model. More exemplars -> strictly better recognition; no ceiling.

Zip layout mirrors Irix's gallery (<id>.npy descriptor + <id>.json meta) so the
idiom is shared, but Auris's descriptor is a sound, not a picture.
"""
from __future__ import annotations

import io
import json
import os
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np

DEFAULT_DIST = float(os.environ.get("AURIS_MATCH_DIST", "0.35"))
# 0.35 = synthetic placeholder (same-sound chi2 is small, different-sound larger).
# Calibrate on real captures before trusting in the wild — same status as Irix's
# face/object thresholds started in.


def chi2(a, b) -> float:
    """Chi-square distance between two L1-normalized histograms. 0 = identical."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-10)))


class SoundGallery:
    """Append-only zip of remembered sound exemplars; cached in memory for matching."""

    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict | None = None

    def _load(self) -> dict:
        if self._cache is not None:
            return self._cache
        out: dict = {}
        if Path(self.path).exists():
            with zipfile.ZipFile(self.path, "r") as z:
                names = z.namelist()
                for n in names:
                    if not n.endswith(".json"):
                        continue
                    eid = n[:-5]
                    meta = json.loads(z.read(n))
                    npy = eid + ".npy"
                    if npy in names:
                        desc = np.load(io.BytesIO(z.read(npy)))
                        out[eid] = {**meta, "desc": desc}
        self._cache = out
        return out

    def _append(self, eid: str, desc, meta: dict) -> None:
        nb = io.BytesIO()
        np.save(nb, np.asarray(desc, dtype=np.float32))
        mode = "a" if Path(self.path).exists() else "w"
        with zipfile.ZipFile(self.path, mode, zipfile.ZIP_DEFLATED) as z:
            z.writestr(eid + ".npy", nb.getvalue())
            z.writestr(eid + ".json", json.dumps(meta))
        if self._cache is not None:
            self._cache[eid] = {**meta, "desc": np.asarray(desc, dtype=np.float32)}

    def remember(self, desc, name: str, meta: dict | None = None) -> str:
        """Store one sound exemplar's descriptor under `name`. Returns entry id."""
        eid = uuid.uuid4().hex
        m = {"name": name, "ts": time.time()}
        if meta:
            m.update(meta)
        self._append(eid, desc, m)
        return eid

    def match(self, desc, threshold: float | None = None) -> dict:
        """Nearest remembered sound — minimum chi-square over the gallery."""
        thr = DEFAULT_DIST if threshold is None else threshold
        best_name, best = None, 1e9
        for e in self._load().values():
            d = chi2(desc, e["desc"])
            if d < best:
                best, best_name = d, e["name"]
        known = best_name is not None and best <= thr
        return {"name": best_name if known else None,
                "dist": round(best, 4) if best_name is not None else None,
                "known": known, "nearest": best_name}

    def identities(self) -> set:
        return {e["name"] for e in self._load().values()}

    def stats(self) -> dict:
        gal = self._load()
        by: dict = {}
        for e in gal.values():
            by[e["name"]] = by.get(e["name"], 0) + 1
        return {"exemplars": len(gal), "identities": len(by), "by_name": by,
                "path": self.path}
