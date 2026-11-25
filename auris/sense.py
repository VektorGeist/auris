"""The whole sense wired together: capture, cochlea, aggregation, perception, gallery recognition, and episodic memory."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import audio_io, chladni, perceive, spectrotemporal
from .cochlea import Cochlea, BLOCK
from .gallery import SoundGallery, DEFAULT_DIST
from .hippocampus import Hippocampus
from .words import WordLexicon


def _data_dir(d=None) -> Path:
    if d:
        return Path(d)
    return Path(os.environ.get("AURIS_DIR", str(Path.home() / ".auris")))


class Ear:

    def __init__(self, data_dir=None):
        self.dir = _data_dir(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.gallery = SoundGallery(self.dir / "gallery.zip")
        self.hippocampus = Hippocampus(self.dir / "episodes.jsonl")
        self.words = WordLexicon(self.dir / "words.zip")


    def _transduce(self, samples, sr):
        cochlea = Cochlea(sr=sr)
        dt = cochlea.block / cochlea.sr
        percepts, t = [], 0.0
        for block in audio_io.blocks(samples, cochlea.block):
            percepts.append(cochlea.hear_block(block, t))
            t += dt
        return percepts, cochlea

    def _aggregate(self, percepts, cochlea, rms=None) -> dict:
        P = np.array([p.place for p in percepts], dtype=np.float32)
        place = np.median(P, axis=0)
        s = float(place.sum())
        place = (place / s if s > 0 else place).astype(np.float32)
        louds = np.array([p.loud for p in percepts], dtype=np.float32)
        loud = float(np.percentile(louds, 90))
        energies = np.array([p.energy for p in percepts], dtype=np.float32)
        centroid = float(np.median([p.centroid for p in percepts]))


        _g = perceive.gate(percepts, rms=rms)
        silent = _g["silent"]
        flatness = _g["flatness"]
        tonal = _g["tonal"]
        voiced = _g["voiced"]
        frac_voiced = _g["frac_voiced"]
        f0 = _g["f0"]
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


        ek = max(1, len(energies) // 3)
        ea, eb = float(energies[:ek].mean()), float(energies[-ek:].mean())
        loud_trend = ("swelling" if eb > ea * 1.15 else
                      "fading" if eb < ea * 0.85 else "steady")
        peak_hz = float(cochlea.centers[int(np.argmax(place))])
        _note, _cents = perceive.note_detail(f0)
        _cw = chladni.plate_weights(place, centers=cochlea.centers)
        return {"place": place, "loud": loud, "centroid": centroid, "f0": f0,
                "note": _note,
                "cents_off": (round(float(_cents), 1)
                              if _cents is not None else None),
                "chladni_modes": chladni.top_modes(_cw, k=5),
                "flatness": round(flatness, 3), "tonal": tonal,
                "frac_voiced": round(frac_voiced, 3), "onset_rate": onset_rate,
                "pitch_trend": pitch_trend, "loud_trend": loud_trend,
                "peak_hz": peak_hz, "duration": round(dur, 3),
                "silent": silent,
                "centers": cochlea.centers}

    def _trajectory(self, percepts, extra=None):
        return spectrotemporal.trajectory(percepts, extra=extra)


    def hear_samples(self, samples, sr, mode="features", context="",
                     remember_as=None, t=None, imprint=True,
                     word=False, word_as=None, phon=False) -> dict:
        samples = np.asarray(samples, dtype=np.float32)
        percepts, cochlea = self._transduce(samples, sr)
        if not percepts:
            return {"error": "audio too short (need at least one 2048-sample block)"}
        rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
        agg = self._aggregate(percepts, cochlea, rms=rms)
        desc = agg["place"]
        if mode == "render":
            p = perceive.describe_render(percepts, agg, out_dir=self.dir)
        else:
            p = perceive.describe(agg)

        traj = self._trajectory(percepts)
        rec = self.gallery.match_any(desc, traj["traj"])
        novelty = 1.0 if rec["dist"] is None else float(
            min(1.0, rec["dist"] / max(DEFAULT_DIST, 1e-6)))
        salience = round(float(0.5 * agg["loud"] + 0.5 * novelty), 4)

        if remember_as:
            self.gallery.remember(desc, remember_as,
                                  meta={"summary": p["text"], "note": p.get("note")},
                                  traj=traj["traj"])
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


    def remember_word(self, wav, name: str) -> dict:
        sr, samples = audio_io.read_wav(str(wav))
        out = self.hear_samples(samples, sr, imprint=False, word_as=name)
        out["source"] = str(wav)
        return out

    def hear_word(self, wav, phon=True) -> dict:
        sr, samples = audio_io.read_wav(str(wav))
        out = self.hear_samples(samples, sr, imprint=False, word=True, phon=phon)
        out["source"] = str(wav)
        return out

    def hear_words(self, samples, sr, phon=False) -> list:
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


    def hear_mic(self, seconds=2.0, sr=None, device=None, gain=1.0,
                 mode="features", context="", remember_as=None, t=None,
                 imprint=True) -> dict:
        from . import listen
        sr, samples = listen.record(seconds, sr=sr, device=device, gain=gain)
        out = self.hear_samples(samples, sr, mode=mode, context=context,
                                remember_as=remember_as, t=t, imprint=imprint)
        out["source"] = "mic"
        return out

    def listen_stream(self, window=2.0, sr=None, device=None, gain=1.0,
                      seconds=None, mode="features", context="", imprint=True):
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
        from . import listen
        buf = np.zeros(0, dtype=np.float32)
        emitted_to = 0
        wsr = None
        elapsed = 0.0
        for csr, chunk in listen.windows(window, sr=sr, device=device, gain=gain,
                                         _capture=_capture):
            wsr = csr
            buf = np.concatenate([buf, np.asarray(chunk, dtype=np.float32)])
            guard = int(0.3 * csr)
            for (a, b) in spectrotemporal.segment_utterances(buf, csr):
                if a >= emitted_to and b <= len(buf) - guard:
                    out = self.hear_samples(buf[a:b], csr, imprint=False,
                                            word=True, phon=phon)
                    w = out["word"]
                    w["t"] = round(a / csr, 3)
                    emitted_to = b
                    yield w

            if emitted_to > csr:
                keep_from = max(0, emitted_to - csr)
                buf = buf[keep_from:]
                emitted_to -= keep_from
            elapsed += window
            if seconds is not None and elapsed >= seconds:
                return
        _ = wsr


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


    def chladni(self, samples, sr, size=128, as_images=False):
        from . import chladni as _chladni
        percepts, _ = self._transduce(samples, sr)
        return _chladni.chladni_timeline(percepts, size=size, as_images=as_images)

    def chladni_file(self, wav, size=128, as_images=False):
        from . import audio_io as _audio_io
        from . import chladni as _chladni
        sr, samples = _audio_io.read_wav(str(wav))
        percepts, _ = self._transduce(samples, sr)
        return _chladni.chladni_timeline(percepts, size=size, as_images=as_images)
