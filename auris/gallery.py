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

DEFAULT_DIST = float(os.environ.get("AURIS_MATCH_DIST", "0.0587"))
# Descriptor (median place vector, chi2) match threshold. MEASURED BASIS:
# the equal-error point of the same-class vs different-class chi2
# distributions on the 2026-11-23 exam teach/test sets (modal-synthesis
# containers, 8 classes, 4 teach + 4 test takes each), measured
# 2026-11-24 AFTER the ear fixes of that day: 16.3% false-accept and
# 16.4% false-reject at this value — the classes overlap even on
# synthetic material, so this is a balance point, not a separation.
# NOT calibrated on real recordings: on real music pieces (the
# composing drafts) same-class descriptor distances averaged 0.358 in
# the post-fix leave-one-out measurement, far above this value, so on
# real material this threshold mostly rejects. A real-capture
# calibration is still outstanding; do not read 0.0587 as universal.

DEFAULT_TRAJ_DIST = float(os.environ.get("AURIS_TRAJ_DIST", "0.0533"))
# Trajectory (DTW) match threshold, on the same mean-per-frame chi2 scale
# as DEFAULT_DIST. MEASURED BASIS: the equal-error point of the
# same-class vs different-class DTW distributions on the same
# 2026-11-23 exam sets, measured 2026-11-24 AFTER the ear fixes of that
# day (the fixes compressed the distance scale; the earlier 0.11 was
# measured before them and false-accepted 31% of different-class pairs
# on these sets afterwards): 9.4% false-accept and 9.4% false-reject at
# this value. NOT calibrated on real recordings: on the composing
# drafts same-class DTW distances averaged 0.489 in the post-fix
# leave-one-out measurement, far above this value, so on real material
# this threshold mostly rejects. A real-capture calibration is still
# outstanding; do not read 0.0533 as universal.


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
                        entry = {**meta, "desc": desc}
                        traj_npy = eid + ".traj.npy"
                        if traj_npy in names:
                            entry["traj"] = np.load(io.BytesIO(z.read(traj_npy)))
                        out[eid] = entry
        self._cache = out
        return out

    def _append(self, eid: str, desc, meta: dict, traj=None) -> None:
        nb = io.BytesIO()
        np.save(nb, np.asarray(desc, dtype=np.float32))
        tb = None
        if traj is not None:
            tb = io.BytesIO()
            np.save(tb, np.asarray(traj, dtype=np.float32))
        mode = "a" if Path(self.path).exists() else "w"
        with zipfile.ZipFile(self.path, mode, zipfile.ZIP_DEFLATED) as z:
            z.writestr(eid + ".npy", nb.getvalue())
            if tb is not None:
                z.writestr(eid + ".traj.npy", tb.getvalue())
            z.writestr(eid + ".json", json.dumps(meta))
        if self._cache is not None:
            entry = {**meta, "desc": np.asarray(desc, dtype=np.float32)}
            if traj is not None:
                entry["traj"] = np.asarray(traj, dtype=np.float32)
            self._cache[eid] = entry

    def remember(self, desc, name: str, meta: dict | None = None,
                 traj=None) -> str:
        """Store one sound exemplar's descriptor under `name`. Returns entry id.

        `traj` (optional) is the sound's spectrotemporal trajectory (T x F);
        when stored, the exemplar can also be matched by trajectory (DTW),
        which keeps the sound's shape over time that the aggregated
        descriptor collapses."""
        eid = uuid.uuid4().hex
        m = {"name": name, "ts": time.time()}
        if meta:
            m.update(meta)
        self._append(eid, desc, m, traj=traj)
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

    def match_traj(self, traj, threshold: float | None = None) -> dict:
        """Nearest remembered sound by trajectory — DTW over stored trajectories.

        Only exemplars remembered with a trajectory take part. The distance is
        spectrotemporal.dtw's mean per-frame chi2, on the same scale as the
        descriptor distance, so the two thresholds are comparable."""
        from . import spectrotemporal
        thr = DEFAULT_TRAJ_DIST if threshold is None else threshold
        best_name, best = None, 1e9
        for e in self._load().values():
            if "traj" not in e:
                continue
            d = spectrotemporal.dtw(traj, e["traj"])
            if d < best:
                best, best_name = d, e["name"]
        known = best_name is not None and best <= thr
        return {"name": best_name if known else None,
                "dist": round(best, 4) if best_name is not None else None,
                "known": known, "nearest": best_name, "via": "trajectory"}

    def has_traj(self) -> bool:
        return any("traj" in e for e in self._load().values())

    def match_any(self, desc, traj=None, threshold: float | None = None,
                  traj_threshold: float | None = None) -> dict:
        """Best available recognition: trajectory when the query carries one
        and the gallery holds trajectory exemplars, else the descriptor."""
        if traj is not None and self.has_traj():
            return self.match_traj(traj, threshold=traj_threshold)
        out = self.match(desc, threshold=threshold)
        out["via"] = "descriptor"
        return out

    def identities(self) -> set:
        return {e["name"] for e in self._load().values()}

    def stats(self) -> dict:
        gal = self._load()
        by: dict = {}
        for e in gal.values():
            by[e["name"]] = by.get(e["name"], 0) + 1
        return {"exemplars": len(gal), "identities": len(by), "by_name": by,
                "path": self.path}
