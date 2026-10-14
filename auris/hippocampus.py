"""The hippocampus — episodic memory of what was heard.

The gallery is recognition (semantic: "this sound is called X"). The hippocampus is
the part that forms a memory of a MOMENT and later brings it back:

  encode / bind   imprint(): pack (percept + tonotopic signature + time + context)
                  into an episode with a sparse pattern-separation key (similar
                  moments stay distinguishable).
  hold            recent(): a buffer of recent, full-detail episodes.
  consolidate     consolidate(): replay — promote salient or repeated episodes into
                  the durable gallery; the rest simply age in the log.
  recall          recall(): pattern completion — nearest stored episodes to a partial
                  or whole cue signature.

Deterministic, model-free, stdlib + numpy. Persisted as JSONL under ~/.auris.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

import numpy as np

from .gallery import chi2

_DUP_DIST = 0.1              # descriptors closer than this count as "the same sound"


class Hippocampus:
    def __init__(self, path, buffer_size: int = 64):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self.episodes = self._load()

    def _load(self) -> list:
        eps: list = []
        p = Path(self.path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    d["desc"] = np.asarray(d["desc"], dtype=np.float32)
                    eps.append(d)
                except Exception:
                    continue                # skip a corrupt line, don't crash recall
        return eps

    @staticmethod
    def _sep_key(desc) -> str:
        """Pattern-separation key: a distinct bucket from the tonotopic shape."""
        top = int(np.argmax(desc))
        h = hashlib.sha1(np.round(desc, 2).tobytes()).hexdigest()[:8]
        return f"{top:02d}-{h}"

    def imprint(self, desc, summary: str, context: str = "", salience: float = 0.0,
                source: str = "", t: float | None = None, note=None) -> str:
        """Bind one percept into an episode and store it. Returns the episode id."""
        t = time.time() if t is None else float(t)
        desc = np.asarray(desc, dtype=np.float32)
        ep = {"id": uuid.uuid4().hex, "t": t, "summary": summary, "context": context,
              "salience": round(float(salience), 4), "source": source, "note": note,
              "key": self._sep_key(desc), "consolidated": False,
              "desc": [round(float(x), 6) for x in desc]}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ep) + "\n")
        self.episodes.append({**ep, "desc": desc})
        return ep["id"]

    def recall(self, cue_desc, k: int = 3) -> list:
        """Pattern completion: the k nearest stored episodes to a cue signature."""
        cue = np.asarray(cue_desc, dtype=np.float32)
        scored = sorted(((chi2(cue, e["desc"]), e) for e in self.episodes),
                        key=lambda x: x[0])
        keep = ("id", "t", "summary", "context", "note", "salience", "source")
        return [{"dist": round(d, 4), **{f: e.get(f) for f in keep}}
                for d, e in scored[:k]]

    def recent(self, n: int | None = None) -> list:
        n = self.buffer_size if n is None else n
        return self.episodes[-n:]

    def consolidate(self, gallery, salience_thresh: float = 0.5,
                    repeat_thresh: int = 3, name: str | None = None) -> dict:
        """Replay: promote salient OR repeated unconsolidated episodes into the
        gallery (long-term recognition). The rest just age in the log."""
        promoted = []
        for e in [x for x in self.episodes if not x.get("consolidated")]:
            repeats = sum(1 for o in self.episodes
                          if chi2(e["desc"], o["desc"]) < _DUP_DIST)
            if e["salience"] >= salience_thresh or repeats >= repeat_thresh:
                nm = name or e.get("note") or f"sound-{e['key']}"
                gid = gallery.remember(e["desc"], nm,
                                       meta={"from_episode": e["id"],
                                             "summary": e["summary"]})
                e["consolidated"] = True
                promoted.append({"episode": e["id"], "name": nm, "gallery_id": gid,
                                 "salience": e["salience"], "repeats": repeats})
        if promoted:
            self._rewrite()
        return {"promoted": promoted, "n": len(promoted)}

    def _rewrite(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            for e in self.episodes:
                d = {k: v for k, v in e.items() if k != "desc"}
                d["desc"] = [round(float(x), 6) for x in e["desc"]]
                f.write(json.dumps(d) + "\n")

    def stats(self) -> dict:
        con = sum(1 for e in self.episodes if e.get("consolidated"))
        return {"episodes": len(self.episodes), "consolidated": con,
                "unconsolidated": len(self.episodes) - con, "path": self.path}
