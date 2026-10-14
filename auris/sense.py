"""Ear — the whole sense wired together.

capture -> cochlea (ear) -> aggregate -> perceive (text) -> gallery (recognise) ->
hippocampus (form an episodic memory). Afferent-only: hearing perceives and
remembers; it never acts.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import audio_io, perceive, spectrotemporal
from .cochlea import Cochlea, BLOCK
from .gallery import SoundGallery, DEFAULT_DIST
from .hippocampus import Hippocampus
from .words import WordLexicon


def _data_dir(d=None) -> Path:
    if d:
        return Path(d)
    return Path(os.environ.get("AURIS_DIR", str(Path.home() / ".auris")))


class Ear:
    """Vanta's sense of hearing over a persistent memory store."""

    def __init__(self, data_dir=None):
        self.dir = _data_dir(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.gallery = SoundGallery(self.dir / "gallery.zip")
        self.hippocampus = Hippocampus(self.dir / "episodes.jsonl")
        self.words = WordLexicon(self.dir / "words.zip")

    # --- transduce + aggregate --------------------------------------------------
    def _transduce(self, samples, sr):
        cochlea = Cochlea(sr=sr)                     # built for this clip's rate
        dt = cochlea.block / cochlea.sr
        percepts, t = [], 0.0
        for block in audio_io.blocks(samples, cochlea.block):
            percepts.append(cochlea.hear_block(block, t))
            t += dt
        return percepts, cochlea

    def _aggregate(self, percepts, cochlea) -> dict:
        P = np.array([p.place for p in percepts], dtype=np.float32)     # F x N
        place = np.median(P, axis=0)
        s = float(place.sum())
        place = (place / s if s > 0 else place).astype(np.float32)
        louds = np.array([p.loud for p in percepts], dtype=np.float32)
        loud = float(np.percentile(louds, 90))
        centroid = float(np.median([p.centroid for p in percepts]))
        flatness = float(np.median([p.flatness for p in percepts]))
        # tone-vs-noise gate: a flat spectrum is noise/percussion, not a pitch,
        # even when Soundgarden's peak-picker fires on a spiky noise bin.
        tonal = flatness <= 0.3
        voiced = [p.f0 for p in percepts if p.f0 > 0] if tonal else []
        frac_voiced = len(voiced) / len(percepts)
        f0 = float(np.median(voiced)) if frac_voiced >= 0.3 else 0.0
        flux = np.array([p.flux for p in percepts], dtype=np.float32)
        thr = float(flux.mean() + flux.std())
        onsets = int(np.sum((flux[1:] > thr) & (flux[:-1] <= thr))) if len(flux) > 1 else 0
        dur = len(percepts) * cochlea.block / cochlea.sr
        onset_rate = onsets / dur if dur > 0 else 0.0
        pitch_trend = None
        if len(voiced) >= 3:
            k = max(1, len(voiced) // 3)
            a, b = np.median(voiced[:k]), np.median(voiced[-k:])
            pitch_trend = ("rising" if b > a * 1.03 else
                           "falling" if b < a * 0.97 else "steady")
        lk = max(1, len(louds) // 3)
        la, lb = float(louds[:lk].mean()), float(louds[-lk:].mean())
        loud_trend = ("swelling" if lb > la * 1.15 else
                      "fading" if lb < la * 0.85 else "steady")
        peak_hz = float(cochlea.centers[int(np.argmax(place))])
        return {"place": place, "loud": loud, "centroid": centroid, "f0": f0,
                "flatness": round(flatness, 3), "tonal": tonal,
                "frac_voiced": round(frac_voiced, 3), "onset_rate": onset_rate,
                "pitch_trend": pitch_trend, "loud_trend": loud_trend,
                "peak_hz": peak_hz, "duration": round(dur, 3),
                "centers": cochlea.centers}

    def _trajectory(self, percepts, extra=None):
        """Time-resolved descriptor (T x F) over the voiced span — keeps the
        trajectory the median `_aggregate` collapses. `extra` = optional per-frame
        phonetic features (Layer B) concatenated on."""
        return spectrotemporal.trajectory(percepts, extra=extra)

    # --- the sense ---------------------------------------------------------------
    def hear_samples(self, samples, sr, mode="features", context="",
                     remember_as=None, t=None, imprint=True,
                     word=False, word_as=None, phon=False) -> dict:
        samples = np.asarray(samples, dtype=np.float32)
        percepts, cochlea = self._transduce(samples, sr)
        if not percepts:
            return {"error": "audio too short (need at least one 2048-sample block)"}
        agg = self._aggregate(percepts, cochlea)
        desc = agg["place"]
        if mode == "render":
            p = perceive.describe_render(percepts, agg, out_dir=self.dir)
        else:
            p = perceive.describe(agg)

        rec = self.gallery.match(desc)
        novelty = 1.0 if rec["dist"] is None else float(
            min(1.0, rec["dist"] / max(DEFAULT_DIST, 1e-6)))
        salience = round(float(0.5 * agg["loud"] + 0.5 * novelty), 4)

        if remember_as:
            self.gallery.remember(desc, remember_as,
                                  meta={"summary": p["text"], "note": p.get("note")})
            rec = {"name": remember_as, "dist": 0.0, "known": True,
                   "nearest": remember_as}

        episode_id = None
        if imprint:
            episode_id = self.hippocampus.imprint(
                desc, p["text"], context=context, salience=salience,
                source="samples", t=t, note=p.get("note"))
        out = {"heard": p["text"], "note": p.get("note"), "perception": p,
               "recognised": rec["name"], "recognition": rec,
               "salience": salience, "episode_id": episode_id,
               "sr": int(sr), "frames": len(percepts), "mode": p["mode"],
               "descriptor": desc}

        # word path (opt-in): keep the trajectory, recognise/remember a word
        if word or word_as:
            traj = self._trajectory(percepts)
            if word_as:
                self.words.remember(traj["traj"], word_as, n_frames=traj["n_frames"])
                out["word"] = {"name": word_as, "dist": 0.0, "known": True,
                               "remembered": True}
            else:
                out["word"] = self.words.match(traj["traj"])
            out["trajectory"] = traj
            if phon:
                from . import phonetics
                ft, _ = phonetics.frame_features(percepts, samples, sr)
                vot = phonetics.voice_onset_time(samples, sr)
                summary = phonetics.summarize(ft)
                out["word"]["phonetics"] = {"summary": summary, "vot": vot}
                out["word"]["read"] = perceive.describe_word(
                    out["word"].get("cohort"), summary, vot)
        return out

    def hear_file(self, wav, mode="features", context="", remember_as=None,
                  t=None, imprint=True) -> dict:
        sr, samples = audio_io.read_wav(str(wav))
        out = self.hear_samples(samples, sr, mode=mode, context=context,
                                remember_as=remember_as, t=t, imprint=imprint)
        out["source"] = str(wav)
        return out

    # --- words (time-resolved recognition) --------------------------------------
    def remember_word(self, wav, name: str) -> dict:
        """Hear a WAV and store its trajectory as a named word exemplar."""
        sr, samples = audio_io.read_wav(str(wav))
        out = self.hear_samples(samples, sr, imprint=False, word_as=name)
        out["source"] = str(wav)
        return out

    def hear_word(self, wav, phon=True) -> dict:
        """Hear a WAV and name it against the word lexicon (DTW, no model)."""
        sr, samples = audio_io.read_wav(str(wav))
        out = self.hear_samples(samples, sr, imprint=False, word=True, phon=phon)
        out["source"] = str(wav)
        return out

    def hear_words(self, samples, sr, phon=False) -> list:
        """Segment a buffer into silence-bracketed utterances and name each word."""
        samples = np.asarray(samples, dtype=np.float32)
        spans = spectrotemporal.segment_utterances(samples, sr)
        results = []
        for (a, b) in spans:
            out = self.hear_samples(samples[a:b], sr, imprint=False,
                                    word=True, phon=phon)
            w = out["word"]
            w["span"] = [int(a), int(b)]
            results.append(w)
        return results

    # --- live microphone (the room, not a file) ---------------------------------
    def hear_mic(self, seconds=2.0, sr=None, device=None, gain=1.0,
                 mode="features", context="", remember_as=None, t=None,
                 imprint=True) -> dict:
        """Capture `seconds` of live sound and perceive it like any WAV."""
        from . import listen
        sr, samples = listen.record(seconds, sr=sr, device=device, gain=gain)
        out = self.hear_samples(samples, sr, mode=mode, context=context,
                                remember_as=remember_as, t=t, imprint=imprint)
        out["source"] = "mic"
        return out

    def listen_stream(self, window=2.0, sr=None, device=None, gain=1.0,
                      seconds=None, mode="features", context="", imprint=True):
        """Generator: perceive the mic continuously, one `window` at a time.

        Yields a hear_samples dict per window (afferent-only — only listens).
        Stops after `seconds` of listening if given, else runs until the caller
        breaks. Blocks with an audible descriptor are still yielded (silence is
        a percept too); a window too short to transduce yields its {"error": ...}.
        """
        from . import listen
        elapsed = 0.0
        for wsr, samples in listen.windows(window, sr=sr, device=device, gain=gain):
            out = self.hear_samples(samples, wsr, mode=mode, context=context,
                                    imprint=imprint)
            out["source"] = "mic"
            out["t"] = round(elapsed, 3)
            yield out
            elapsed += window
            if seconds is not None and elapsed >= seconds:
                return

    def listen_words(self, window=1.0, sr=None, device=None, gain=1.0,
                     seconds=None, phon=False, _capture=None):
        """Generator: hear WORDS live off the mic, one committed utterance at a time.

        Captures windows (reusing listen.windows + the injectable _capture hook),
        accumulates a buffer, and yields a word percept the moment an utterance is
        fully bracketed by silence — so a word straddling a capture boundary isn't
        split. Afferent-only. Stops after `seconds` if given, else until broken.
        """
        from . import listen
        buf = np.zeros(0, dtype=np.float32)
        emitted_to = 0            # sample index consumed so far
        wsr = None
        elapsed = 0.0
        for csr, chunk in listen.windows(window, sr=sr, device=device, gain=gain,
                                         _capture=_capture):
            wsr = csr
            buf = np.concatenate([buf, np.asarray(chunk, dtype=np.float32)])
            guard = int(0.3 * csr)        # require 0.3 s of room after a word to commit
            for (a, b) in spectrotemporal.segment_utterances(buf, csr):
                if a >= emitted_to and b <= len(buf) - guard:
                    out = self.hear_samples(buf[a:b], csr, imprint=False,
                                            word=True, phon=phon)
                    w = out["word"]
                    w["t"] = round(a / csr, 3)
                    emitted_to = b
                    yield w
            # bound memory: drop fully-consumed prefix
            if emitted_to > csr:          # keep ~1 s of context before the cursor
                keep_from = max(0, emitted_to - csr)
                buf = buf[keep_from:]
                emitted_to -= keep_from
            elapsed += window
            if seconds is not None and elapsed >= seconds:
                return
        _ = wsr

    # --- memory ------------------------------------------------------------------
    def recall_file(self, wav, k=3) -> list:
        sr, samples = audio_io.read_wav(str(wav))
        percepts, cochlea = self._transduce(samples, sr)
        if not percepts:
            return []
        agg = self._aggregate(percepts, cochlea)
        return self.hippocampus.recall(agg["place"], k=k)

    def consolidate(self, **kw) -> dict:
        return self.hippocampus.consolidate(self.gallery, **kw)

    def stats(self) -> dict:
        return {"gallery": self.gallery.stats(),
                "hippocampus": self.hippocampus.stats(),
                "data_dir": str(self.dir)}
