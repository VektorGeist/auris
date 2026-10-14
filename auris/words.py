"""Word lexicon — a remembered store of spoken-word sound-forms.

The brain keeps an "auditory lexicon": a catalog of the sound-shapes of words it
has heard, and recognises a new utterance by matching it to that memory (not by a
trained classifier). This is Auris's gallery idiom applied to WORDS: each exemplar
is a T x F spectrotemporal trajectory (spectrotemporal.trajectory), matched by DTW.
More exemplars per word -> strictly better recognition; no model, no ceiling.

Storage mirrors SoundGallery exactly (zip of <id>.npy + <id>.json) so the on-disk
idiom is shared; the payload is a 2-D trajectory instead of a 1-D histogram.
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

from .spectrotemporal import dtw, dtw_pitch_invariant

# Measured on real captures via calibrate.py; synthetic placeholder until then.
DEFAULT_WORD_DIST = float(os.environ.get("AURIS_WORD_DIST", "0.20"))


class WordLexicon:
    """Append-only zip of remembered word trajectories; cached for matching."""

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
                        traj = np.load(io.BytesIO(z.read(npy)))
                        out[eid] = {**meta, "traj": traj}
        self._cache = out
        return out

    def _append(self, eid: str, traj, meta: dict) -> None:
        nb = io.BytesIO()
        np.save(nb, np.asarray(traj, dtype=np.float32))
        mode = "a" if Path(self.path).exists() else "w"
        with zipfile.ZipFile(self.path, mode, zipfile.ZIP_DEFLATED) as z:
            z.writestr(eid + ".npy", nb.getvalue())
            z.writestr(eid + ".json", json.dumps(meta))
        if self._cache is not None:
            self._cache[eid] = {**meta, "traj": np.asarray(traj, dtype=np.float32)}

    def remember(self, traj, name: str, n_frames: int = 0, meta: dict | None = None) -> str:
        """Store one word exemplar's trajectory under `name`. Returns entry id."""
        eid = uuid.uuid4().hex
        m = {"name": name, "ts": time.time(), "n_frames": int(n_frames)}
        if meta:
            m.update(meta)
        self._append(eid, traj, m)
        return eid

    def match(self, traj, threshold: float | None = None, band=None,
              pitch_invariant: bool = False, max_shift: int = 4) -> dict:
        """Nearest remembered word by minimum DTW; per-name best distances too.

        pitch_invariant=True matches modulo a bounded tonotopic shift (the Hodos pitch/shift
        quotient) so the same word at a different pitch still recognizes — closing the documented
        pitch-sensitivity limit. Opt-in; default keeps the exact prior behaviour.
        """
        thr = DEFAULT_WORD_DIST if threshold is None else threshold
        best_name, best = None, float("inf")
        per_name: dict = {}
        _match = ((lambda x, y: dtw_pitch_invariant(x, y, band=band, max_shift=max_shift))
                  if pitch_invariant else (lambda x, y: dtw(x, y, band=band)))
        for e in self._load().values():
            d = _match(traj, e["traj"])
            nm = e["name"]
            if nm not in per_name or d < per_name[nm]:
                per_name[nm] = d
            if d < best:
                best, best_name = d, nm
        known = best_name is not None and best <= thr
        ranked = sorted(per_name.items(), key=lambda kv: kv[1])
        return {"name": best_name if known else None,
                "dist": round(best, 4) if best_name is not None else None,
                "known": known, "nearest": best_name,
                "cohort": [{"name": n, "dist": round(d, 4)} for n, d in ranked]}

    def identities(self) -> set:
        return {e["name"] for e in self._load().values()}

    def stats(self) -> dict:
        lex = self._load()
        by: dict = {}
        for e in lex.values():
            by[e["name"]] = by.get(e["name"], 0) + 1
        return {"exemplars": len(lex), "words": len(by), "by_name": by,
                "path": self.path}
