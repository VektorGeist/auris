"""Sets recognition thresholds from measured separation between labelled exemplars."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import audio_io
from .spectrotemporal import trajectory, dtw


def _trajs_from_dir(ear, directory):
    by_word: dict = {}
    for p in sorted(Path(directory).glob("*.wav")):
        word = p.stem.split("_")[0]
        sr, samples = audio_io.read_wav(str(p))
        percepts, _ = ear._transduce(samples, sr)
        by_word.setdefault(word, []).append(trajectory(percepts)["traj"])
    return by_word


def separation(by_word: dict, band=None) -> dict:
    same, diff = [], []
    words = list(by_word)
    for w in words:
        ts = by_word[w]
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                same.append(dtw(ts[i], ts[j], band=band))
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            for a in by_word[words[i]]:
                for b in by_word[words[j]]:
                    diff.append(dtw(a, b, band=band))
    same, diff = np.array(same), np.array(diff)
    if same.size == 0 or diff.size == 0:
        return {"ok": False, "reason": "need >=2 takes of >=2 words",
                "n_same": int(same.size), "n_diff": int(diff.size)}

    hi_same, lo_diff = float(same.max()), float(diff.min())
    separable = hi_same < lo_diff
    if separable:
        thr = 0.5 * (hi_same + lo_diff)
    else:

        cands = np.unique(np.concatenate([same, diff]))
        best_thr, best_err = float(cands[0]), 1e9
        for c in cands:
            fr = float((same > c).mean())
            fa = float((diff <= c).mean())
            if fr + fa < best_err:
                best_err, best_thr = fr + fa, float(c)
        thr = best_thr
    return {"ok": True, "threshold": round(thr, 4), "separable": separable,
            "same": {"n": int(same.size), "mean": round(float(same.mean()), 4),
                     "max": round(hi_same, 4)},
            "diff": {"n": int(diff.size), "mean": round(float(diff.mean()), 4),
                     "min": round(lo_diff, 4)},
            "margin": round(lo_diff - hi_same, 4)}


def calibrate_dir(ear, directory, band=None) -> dict:
    rep = separation(_trajs_from_dir(ear, directory), band=band)
    if rep.get("ok"):
        rep["env"] = f"AURIS_WORD_DIST={rep['threshold']}"
        rep["note"] = ("clean separation - recognition trustworthy"
                       if rep["separable"] else
                       "distributions OVERLAP - some confusions expected at this threshold")
    return rep
