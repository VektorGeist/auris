"""A store of remembered sound exemplars; recognition by nearest-neighbour chi-square distance over tonotopic descriptors and spectrotemporal trajectories."""
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


DEFAULT_TRAJ_DIST = float(os.environ.get("AURIS_TRAJ_DIST", "0.0533"))


def chi2(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-10)))


class SoundGallery:

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
        eid = uuid.uuid4().hex
        m = {"name": name, "ts": time.time()}
        if meta:
            m.update(meta)
        self._append(eid, desc, m, traj=traj)
        return eid

    def match(self, desc, threshold: float | None = None) -> dict:
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
