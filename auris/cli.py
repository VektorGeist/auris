"""auris CLI — hear a sound, remember it, recall it, consolidate memory.

  auris hear FILE [--mode features|render] [--remember NAME] [--context STR]
  auris remember FILE NAME
  auris recall FILE [-k N]
  auris consolidate [--name NAME] [--salience F] [--repeat N]
  auris stats
  auris tone [--freq HZ] [--dur S] [--kind tone|chord|sweep|noise|clicks] --out FILE
  auris devices
  auris listen [--seconds S] [--device D] [--gain G] [--mode features|render] [--remember NAME] [--context STR] [--no-imprint]
  auris stream [--window S] [--for S] [--device D] [--gain G] [--mode features|render]
"""
from __future__ import annotations

import argparse
import json

from .sense import Ear


def _print(obj):
    def default(o):
        try:
            import numpy as np
            if isinstance(o, np.ndarray):
                return [round(float(x), 5) for x in o]
        except Exception:
            pass
        return str(o)
    print(json.dumps(obj, indent=2, default=default))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="auris", description="Vanta's deterministic sense of hearing")
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("hear", help="perceive a WAV into structured text + a memory")
    h.add_argument("file")
    h.add_argument("--mode", choices=["features", "render"], default="features")
    h.add_argument("--remember", metavar="NAME", default=None)
    h.add_argument("--context", default="")
    h.add_argument("--no-imprint", action="store_true", help="perceive without forming a memory")

    r = sub.add_parser("remember", help="hear a WAV and store it as a named exemplar")
    r.add_argument("file")
    r.add_argument("name")

    rc = sub.add_parser("recall", help="what earlier sounds does this WAV most resemble")
    rc.add_argument("file")
    rc.add_argument("-k", type=int, default=3)

    c = sub.add_parser("consolidate", help="replay: promote salient/repeated episodes to the gallery")
    c.add_argument("--name", default=None)
    c.add_argument("--salience", type=float, default=0.5)
    c.add_argument("--repeat", type=int, default=3)

    sub.add_parser("stats", help="gallery + hippocampus stats")

    sub.add_parser("devices", help="list microphone input devices")

    ls = sub.add_parser("listen", help="capture live mic audio and perceive it (one shot)")
    ls.add_argument("--seconds", type=float, default=2.0)
    ls.add_argument("--device", default=None, help="input device index or name (default: system default)")
    ls.add_argument("--gain", type=float, default=1.0, help="boost a quiet mic before clipping")
    ls.add_argument("--mode", choices=["features", "render"], default="features")
    ls.add_argument("--remember", metavar="NAME", default=None)
    ls.add_argument("--context", default="")
    ls.add_argument("--no-imprint", action="store_true", help="perceive without forming a memory")

    st = sub.add_parser("stream", help="perceive the mic continuously, one window at a time")
    st.add_argument("--window", type=float, default=2.0, help="seconds per perception window")
    st.add_argument("--for", dest="duration", type=float, default=None, help="stop after this many seconds")
    st.add_argument("--device", default=None)
    st.add_argument("--gain", type=float, default=1.0)
    st.add_argument("--mode", choices=["features", "render"], default="features")

    rw = sub.add_parser("remember-word", help="hear a WAV and store its trajectory as a named word")
    rw.add_argument("file")
    rw.add_argument("name")

    hw = sub.add_parser("hear-word", help="name a WAV against the word lexicon (DTW, no model)")
    hw.add_argument("file")

    lw = sub.add_parser("listen-words", help="hear WORDS live off the mic, one utterance at a time")
    lw.add_argument("--window", type=float, default=1.0)
    lw.add_argument("--for", dest="duration", type=float, default=None)
    lw.add_argument("--device", default=None)
    lw.add_argument("--gain", type=float, default=1.0)

    cal = sub.add_parser("calibrate", help="measure a real word-recognition threshold from <word>_<take>.wav files")
    cal.add_argument("dir")

    t = sub.add_parser("tone", help="generate a test sound (WAV)")
    t.add_argument("--kind", choices=["tone", "chord", "sweep", "noise", "clicks"], default="tone")
    t.add_argument("--freq", type=float, default=440.0)
    t.add_argument("--dur", type=float, default=1.0)
    t.add_argument("--harmonics", type=int, default=0)
    t.add_argument("--out", required=True)

    args = ap.parse_args(argv)

    if args.cmd == "tone":
        from . import tones, audio_io
        if args.kind == "tone":
            x = tones.tone(args.freq, args.dur, harmonics=args.harmonics)
        elif args.kind == "chord":
            x = tones.chord(dur=args.dur)
        elif args.kind == "sweep":
            x = tones.sweep(dur=args.dur)
        elif args.kind == "noise":
            x = tones.noise(dur=args.dur)
        else:
            x = tones.clicks(rate=args.freq, dur=args.dur)
        audio_io.write_wav(args.out, x, tones.SR)
        _print({"ok": True, "wrote": args.out, "kind": args.kind,
                "seconds": args.dur, "sr": tones.SR})
        return 0

    ear = Ear()
    if args.cmd == "hear":
        _print(_clean(ear.hear_file(args.file, mode=args.mode,
                                    remember_as=args.remember, context=args.context,
                                    imprint=not args.no_imprint)))
    elif args.cmd == "remember":
        _print(_clean(ear.hear_file(args.file, remember_as=args.name)))
    elif args.cmd == "recall":
        _print({"recall": ear.recall_file(args.file, k=args.k)})
    elif args.cmd == "consolidate":
        _print(ear.consolidate(salience_thresh=args.salience,
                               repeat_thresh=args.repeat, name=args.name))
    elif args.cmd == "stats":
        _print(ear.stats())
    elif args.cmd == "devices":
        from . import listen
        if not listen.available():
            _print({"backend": "sounddevice", "available": False,
                    "hint": "pip install sounddevice (PortAudio) to hear a live mic"})
        else:
            _print({"backend": "sounddevice", "available": True,
                    "inputs": listen.list_inputs()})
    elif args.cmd == "listen":
        _print(_clean(ear.hear_mic(seconds=args.seconds, device=_dev(args.device),
                                   gain=args.gain, mode=args.mode,
                                   remember_as=args.remember, context=args.context,
                                   imprint=not args.no_imprint)))
    elif args.cmd == "stream":
        try:
            for out in ear.listen_stream(window=args.window, seconds=args.duration,
                                         device=_dev(args.device), gain=args.gain,
                                         mode=args.mode):
                _print(_clean(out))
        except KeyboardInterrupt:
            _print({"stopped": "keyboard-interrupt"})
    elif args.cmd == "remember-word":
        _print(_clean(ear.remember_word(args.file, args.name)))
    elif args.cmd == "hear-word":
        _print(_clean(ear.hear_word(args.file)))
    elif args.cmd == "listen-words":
        try:
            for w in ear.listen_words(window=args.window, seconds=args.duration,
                                      device=_dev(args.device), gain=args.gain,
                                      phon=True):
                _print(w)
        except KeyboardInterrupt:
            _print({"stopped": "keyboard-interrupt"})
    elif args.cmd == "calibrate":
        from . import calibrate
        _print(calibrate.calibrate_dir(ear, args.dir))
    return 0


def _dev(d):
    """A --device arg may be a numeric index or a name substring."""
    if d is None:
        return None
    return int(d) if str(d).isdigit() else d


def _clean(d: dict) -> dict:
    """Drop the raw descriptor/trajectory arrays from CLI output (keep it readable)."""
    return {k: v for k, v in d.items() if k not in ("descriptor", "trajectory")}


if __name__ == "__main__":
    raise SystemExit(main())
