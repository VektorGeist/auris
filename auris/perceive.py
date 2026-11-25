"""Turns the ear's field into structured text."""
from __future__ import annotations

import math

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(f0: float):
    if not f0 or f0 <= 0:
        return None
    midi = int(round(69 + 12 * math.log2(f0 / 440.0)))
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def note_detail(f0: float):
    if not f0 or f0 <= 0:
        return None, None
    midi = int(round(69 + 12 * math.log2(f0 / 440.0)))
    exact = 440.0 * 2 ** ((midi - 69) / 12)
    cents = 1200.0 * math.log2(f0 / exact)
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}", cents


def gate(percepts, rms=None) -> dict:
    flatness = float(np.median([p.flatness for p in percepts]))
    if rms is not None:
        silent = bool(rms < 1e-4)
    else:
        energies = np.array([p.energy for p in percepts], dtype=np.float32)
        silent = bool(float(np.percentile(energies, 90)) == 0.0)
    tonal = flatness <= 0.3
    voiced = [p.f0 for p in percepts if p.f0 > 0] if tonal else []
    frac_voiced = len(voiced) / len(percepts)
    f0 = float(np.median(voiced)) if frac_voiced >= 0.15 else 0.0
    return {"tonal": tonal, "silent": silent, "flatness": flatness,
            "f0": f0, "frac_voiced": frac_voiced, "voiced": voiced}


def _loud_word(x):
    return ("silent" if x < 0.03 else "very quiet" if x < 0.12 else
            "quiet" if x < 0.3 else "moderate" if x < 0.6 else
            "loud" if x < 0.85 else "very loud")


def _bright_word(c):
    return ("dark" if c < 0.1 else "warm" if c < 0.25 else
            "bright" if c < 0.5 else "brilliant")


def region_word(hz):
    return ("sub-bass" if hz < 80 else "bass" if hz < 160 else
            "low-mid" if hz < 500 else "mid / voiced" if hz < 2000 else
            "high" if hz < 6000 else "very high / airy")


def describe_word(cohort, phon_summary=None, vot=None) -> str:
    parts = []
    if cohort:
        top = cohort[0]
        if len(cohort) > 1:
            gap = cohort[1]["dist"] - top["dist"]
            conf = "clearly" if gap > 0.05 else "narrowly"
            parts.append(f"sounds {conf} like '{top['name']}' "
                         f"(nearest '{cohort[1]['name']}' at +{gap:.3f})")
        else:
            parts.append(f"sounds like '{top['name']}'")
    if phon_summary:
        parts.append(f"made of: {phon_summary}")
    if vot and vot.get("vot_ms") is not None:
        parts.append(f"onset {vot['vot_ms']:.0f} ms -> {vot['phon_class']}")
    return "; ".join(parts) if parts else "an unrecognised word-sound"


def describe(agg: dict) -> dict:
    if agg.get("silent"):


        return {
            "text": "silence.",
            "note": None,
            "cents_off": None,
            "chladni_modes": [],
            "f0": 0.0,
            "loud": round(float(agg["loud"]), 3),
            "centroid": round(float(agg["centroid"]), 3),
            "peak_hz": 0.0,
            "region": region_word(0),
            "onset_rate": 0.0,
            "pitch_trend": None,
            "loud_trend": "steady",
            "silent": True,
            "mode": "features-to-text",
        }
    if "tonal" not in agg:


        raise ValueError(
            "describe() needs the gate fields: build agg with "
            "perceive.gate() so it carries 'tonal' and 'silent'")
    note, cents = note_detail(agg["f0"])
    parts = []
    if note:
        parts.append(f"a pitched tone, {note} at {agg['f0']:.1f} Hz "
                     f"({cents:+.0f} cents from exact {note})")
    elif agg.get("onset_rate", 0) >= 1.5:
        parts.append("an unpitched, percussive sound")
    elif agg.get("tonal"):


        parts.append("an unpitched, tonal sound")
    else:
        parts.append("an unpitched, noise-like sound")
    parts.append(_loud_word(agg["loud"]))
    parts.append(f"{_bright_word(agg['centroid'])} in timbre")
    parts.append(f"energy centred around {agg['peak_hz']:.0f} Hz "
                 f"({region_word(agg['peak_hz'])})")
    parts.append(f"rhythmic (~{agg['onset_rate']:.1f} onsets/s)"
                 if agg["onset_rate"] >= 0.5 else "steady / sustained")
    if agg.get("pitch_trend") and agg["pitch_trend"] != "steady":
        parts.append(f"pitch {agg['pitch_trend']}")
    if agg.get("loud_trend") and agg["loud_trend"] != "steady":
        parts.append(agg["loud_trend"])
    modes = agg.get("chladni_modes") or []
    if modes:
        parts.append("plate modes ringing strongest: " + ", ".join(
            f"({n},{m}) {w:.2f}" for n, m, w in modes[:3]))
    return {
        "text": ", ".join(parts) + ".",
        "note": note,
        "cents_off": (round(float(cents), 1) if cents is not None else None),
        "chladni_modes": modes,
        "f0": round(float(agg["f0"]), 1),
        "loud": round(float(agg["loud"]), 3),
        "centroid": round(float(agg["centroid"]), 3),
        "peak_hz": round(float(agg["peak_hz"]), 1),
        "region": region_word(agg["peak_hz"]),
        "onset_rate": round(float(agg["onset_rate"]), 2),
        "pitch_trend": agg.get("pitch_trend"),
        "loud_trend": agg.get("loud_trend"),
        "mode": "features-to-text",
    }


def describe_render(percepts, agg, out_dir=None, want_irix=True) -> dict:
    from . import render as _render

    field = np.array([p.place for p in percepts], dtype=np.float32).T
    read = _render.read_field(field, agg["centers"])
    note = note_name(read["peak_hz"])
    text = (f"seen as a field: brightest band at {read['peak_hz']:.0f} Hz "
            f"({region_word(read['peak_hz'])}"
            + (f", ~{note}" if note else "") + "), "
            f"{'moving / rhythmic' if read['motion'] >= 0.15 else 'a steady horizontal band'}, "
            f"{_loud_word(read['brightness'])} overall.")
    out = {"text": text, "note": note, "peak_hz": round(read["peak_hz"], 1),
           "region": region_word(read["peak_hz"]), "motion": round(read["motion"], 3),
           "brightness": round(read["brightness"], 3), "mode": "render-then-read"}
    if out_dir is not None:
        out["image"] = _render.save_field(field, out_dir)
        if want_irix and out.get("image"):
            irix = _render.irix_read(out["image"])
            if irix:
                out["irix"] = irix
    return out
