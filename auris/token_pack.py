"""AURIS token pack (file input).

One pack per audio file, one block per time segment, everything
together:
- time stamp, start and end
- structured text (perceive.describe, AURIS's own words, unchanged)
- pitch, loudness, centroid, peak Hz, region, onset proxy,
  pitch and loudness trends
- Chladni mode numbers (n, m) plus a Chladni figure image per segment
- spectrogram slice image per segment (tonotopic field, dB)
- oscilloscope waveform per segment: amplitude over time, tokenized
  as peak, RMS, zero-crossing rate and a 20-bin envelope, plus a
  waveform line image. AURIS itself does NOT use an oscilloscope.
  This is an extra, separate channel added to the pack, alongside
  Chladni and spectrogram, not a claim about AURIS internals.
  A waveform line shows amplitude over time, a Chladni figure shows
  the vibration shape a frequency makes. They are different views;
  neither replaces the other.
- gallery nearest name and distance, read-only (no memory writes
  here), with a calibration warning: the gallery default threshold
  is a synthetic placeholder; calibrate on real captures before
  trusting a name.
- polyphony candidates per segment: top-5 pitch candidates with
  harmonic-stack scores, in addition to the single f0 and
  harmonic_f0 fields. The original AURIS text is unchanged;
  candidates are a candidate list only, not a replacement pitch,
  and they still fail on noise and on sequential notes inside one
  window.

This is a wrapper around existing AURIS code. It uses equal
segments over the file and Cochlea directly, so the Ear's
gallery and hippocampus memory are never written by this script.
Pillow is optional and only needed for image output.

Usage:
  python -m auris.token_pack input.wav --segments 20 --out out_dir
  python -m auris.token_pack input.wav --segments 20 --out out_dir --no-images
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

from . import polyphony as _polyphony
from .polyphony import stack_scores, top_peaks

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def load_wav(path):
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    samples = raw.astype(np.float32).reshape(-1, ch).mean(axis=1) / 32768.0
    return samples, sr


def harmonic_f0(samples, sr):
    """Harmonic-stack pitch: score MIDI 21-108 by f+2f+3f+4f energy.

    Fixes octave errors where a harmonic is louder than the
    fundamental. Returns (f_hz, midi) or (0.0, 0).
    """
    if samples is None or len(samples) < 1024 or sr <= 0:
        return 0.0, 0
    seg = samples[: sr * 2]
    spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)

    def energy_at(f):
        if f <= 0 or f >= freqs[-1]:
            return 0.0
        i = int(np.argmin(np.abs(freqs - f)))
        return float(np.max(spectrum[max(0, i - 1) : i + 2]))

    best = (0.0, 0, -1.0)
    for midi in range(21, 109):
        f = 440.0 * 2 ** ((midi - 69) / 12)
        score = sum(w * energy_at(f * k) for k, w in [(1, 1.0), (2, 0.6), (3, 0.4), (4, 0.3)])
        sub = sum(w * energy_at((f / 2) * k) for k, w in [(1, 1.0), (2, 0.6), (3, 0.4)])
        if sub > score * 0.9:
            score *= 0.5
        if score > best[2]:
            best = (f, midi, score)
    return round(float(best[0]), 1), int(best[1])


def polyphony_candidates(samples, sr, n=5):
    """Top-N pitch candidates with harmonic-stack scores for one segment.

    Harmonic-stack scores over MIDI 21-108, plus-or-minus one
    semitone non-maximum suppression, first 2 s of the segment
    (stack_scores truncates). Scores are raw FFT-magnitude sums,
    so score_norm (candidate / top score) is the comparable
    number, not the raw score. Returns [] on short or empty input.
    It does not apply the sub-octave suppression in harmonic_f0,
    so octave-below candidates can and do appear in the list;
    that is a known failure mode, reported, not hidden.
    """
    if samples is None or len(samples) < 1024 or sr <= 0:
        return []
    try:
        scores = stack_scores(samples, sr)
        peaks = top_peaks(scores, n)
    except Exception:
        return []
    if not peaks:
        return []
    top = max(float(scores[p]) for p in peaks) or 1.0
    out = []
    for midi in peaks:
        f = 440.0 * 2 ** ((midi - 69) / 12)
        out.append(
            {
                "midi": int(midi),
                "note": f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}",
                "hz": round(float(f), 1),
                "score": round(float(scores[midi]), 1),
                "score_norm": round(float(scores[midi]) / top, 4),
            }
        )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--segments", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument(
        "--labeled-images",
        action="store_true",
        help="Also write *_labeled.png versions: Chladni n/m plus pitch text, "
        "spectrogram with Hz/time/dB axes (flipped, low at bottom), "
        "waveform with time/amplitude axes plus peak/RMS. "
        "Original PNGs are left unchanged.",
    )
    ap.add_argument(
        "--stream",
        action="store_true",
        help="Emit each segment as a JSON line as it completes. "
        "This is file-streaming output, not live microphone capture.",
    )
    args = ap.parse_args(argv)

    from . import audio_io, chladni, phonetics as _phon
    from .cochlea import Cochlea
    from .gallery import SoundGallery
    from .perceive import describe, region_word

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    samples, sr = load_wav(args.wav)
    duration = len(samples) / sr if sr else 0.0
    cochlea = Cochlea(sr=sr)
    percepts = []
    t = 0.0
    dt = cochlea.block / cochlea.sr
    for block in audio_io.blocks(samples, cochlea.block):
        percepts.append(cochlea.hear_block(block, t))
        t += dt
    if not percepts:
        print(json.dumps({"available": False}))
        return 1

    # Read-only gallery, if present. Never writes.
    gallery = None
    try:
        gallery = SoundGallery(Path.home() / ".auris" / "gallery.zip")
        gallery._load()
    except Exception:
        gallery = None

    n = len(percepts)
    edges = np.linspace(0, n, args.segments + 1).astype(int)
    segments = []

    for i in range(args.segments):
        seg = percepts[edges[i] : edges[i + 1]]
        if not seg:
            continue
        t_start = float(seg[0].t)
        t_end = float(seg[-1].t + dt)
        voiced = [p.f0 for p in seg if p.f0 > 0]
        f0 = float(np.median(voiced)) if voiced else 0.0
        loud = float(np.mean([p.loud for p in seg]))
        cen = float(np.median([p.centroid for p in seg]))
        mid = float(np.median([p.mid for p in seg]))
        # Peak place: mean place map, argmax -> centre Hz.
        mean_place = np.mean([p.place for p in seg], axis=0)
        peak_idx = int(np.argmax(mean_place))
        peak_hz = float(cochlea.centers[peak_idx])
        # Trends: first half vs second half median.
        half = max(1, len(seg) // 2)
        f_first = (
            np.median([p.f0 for p in seg[:half] if p.f0 > 0])
            if any(p.f0 > 0 for p in seg[:half])
            else f0
        )
        f_second = (
            np.median([p.f0 for p in seg[half:] if p.f0 > 0])
            if any(p.f0 > 0 for p in seg[half:])
            else f0
        )
        pitch_trend = (
            "rising" if f_second > f_first * 1.05 else "falling" if f_second < f_first * 0.95 else "steady"
        )
        l_first = float(np.mean([p.loud for p in seg[:half]]))
        l_second = float(np.mean([p.loud for p in seg[half:]]))
        loud_trend = (
            "getting louder"
            if l_second > l_first * 1.1
            else "fading"
            if l_second < l_first * 0.9
            else "steady"
        )
        onset_rate = float(np.mean([p.flux for p in seg]))

        agg = {
            "f0": f0,
            "loud": loud,
            "centroid": cen,
            "peak_hz": peak_hz,
            "onset_rate": onset_rate,
            "pitch_trend": pitch_trend,
            "loud_trend": loud_trend,
        }
        text = describe(agg)["text"]

        if f0 > 0:
            cn, cm = chladni.mode_numbers(
                chladni.pitch_to_norm(f0), cen, mid_norm=min(1.0, max(0.0, mid))
            )
            cn, cm = int(cn), int(cm)
        else:
            cn, cm = 0, 0

        rec = None
        if gallery is not None:
            try:
                rec = gallery.match(mean_place)
            except Exception:
                rec = None

        seg_out = {
            "index": i,
            "t_start": round(t_start, 2),
            "t_end": round(t_end, 2),
            "text": text,
            "f0_hz": round(f0, 1),
            "loud": round(loud, 3),
            "centroid": round(cen, 3),
            "peak_hz": round(peak_hz, 1),
            "region": region_word(peak_hz),
            "onset_flux_mean": round(onset_rate, 3),
            "pitch_trend": pitch_trend,
            "loud_trend": loud_trend,
            "chladni": {"n": cn, "m": cm},
            "harmonic_f0_hz": 0.0,
            "harmonic_midi": 0,
            "oscilloscope": None,
            "gallery": rec,
            "gallery_calibration_warning": "The gallery default threshold is a synthetic placeholder; "
            "calibrate on real captures before trusting a name.",
        }

        # Oscilloscope: raw amplitude vs time for this segment.
        s0 = int(t_start * sr)
        s1 = int(min(len(samples), t_end * sr))
        wseg = samples[s0:s1] if s1 > s0 else np.array([0.0], dtype=np.float32)
        hf, hm = harmonic_f0(wseg, sr)
        seg_out["harmonic_f0_hz"] = hf
        seg_out["harmonic_midi"] = hm
        seg_out["polyphony"] = {
            "method": "Harmonic-stack scores MIDI 21-108, plus-or-minus one semitone "
            "non-maximum suppression, first 2 s of segment; candidate list only, "
            "not a replacement pitch.",
            "top_n": 5,
            "candidates": polyphony_candidates(wseg, sr, n=5),
        }
        seg_out["octave_corrected"] = bool(hf and f0 and abs(hf - f0) / max(f0, 1) > 0.4)
        # Never silently replace AURIS text. "text" stays the original
        # perceive.describe wording. When the harmonic stack disagrees
        # by more than 40% (octave error), add a flagged note field
        # carrying the harmonic pitch alongside, not instead.
        seg_out["auris_text_original"] = text
        if seg_out["octave_corrected"]:
            _hn = f"{NOTE_NAMES[hm % 12]}{hm // 12 - 1} ({hf:.0f} Hz)" if hm else f"{hf:.0f} Hz"
            seg_out["harmonic_note"] = (
                f"NOTE: harmonic stack disagrees with AURIS f0 above: "
                f"harmonic pitch {_hn}, AURIS f0 {f0:.1f} Hz. "
                f"Original AURIS text is kept unchanged in text / auris_text_original; "
                f"do not read the harmonic pitch as the AURIS reading."
            )
        else:
            seg_out["harmonic_note"] = ""
        peak_amp = float(np.max(np.abs(wseg))) if wseg.size else 0.0
        rms_amp = float(np.sqrt(np.mean(wseg**2))) if wseg.size else 0.0
        zc = int(np.sum(np.diff(np.signbit(wseg)))) if wseg.size > 1 else 0
        zc_rate = round(zc / (t_end - t_start), 1) if t_end > t_start else 0
        # Envelope: max abs in 20 equal bins, tokenizable as numbers.
        bins = np.array_split(np.abs(wseg), 20) if wseg.size else []
        envelope = [round(float(np.max(b)) if b.size else 0.0, 3) for b in bins]
        seg_out["oscilloscope"] = {
            "note": "Separate channel, NOT an AURIS internal. Waveform = amplitude over time; "
            "Chladni = vibration shape; spectrogram = frequency over time.",
            "peak": round(peak_amp, 3),
            "rms": round(rms_amp, 3),
            "zero_crossing_rate_per_s": zc_rate,
            "envelope_20": envelope,
            "text": f"waveform peak {peak_amp:.2f}, RMS {rms_amp:.2f}, {zc_rate} zero crossings/s, "
            f"envelope {'rising' if envelope and envelope[-1] > envelope[0] else 'falling' if envelope and envelope[-1] < envelope[0] else 'steady'}.",
        }

        if not args.no_images:
            try:
                from PIL import Image, ImageDraw

                wimg = Image.new("RGB", (320, 128), (5, 5, 15))
                draw = ImageDraw.Draw(wimg)
                draw.line([(0, 64), (320, 64)], fill=(80, 80, 120), width=1)
                pts = np.linspace(0, len(wseg) - 1, 320).astype(int) if wseg.size else []
                coords = []
                for x, idx in enumerate(pts):
                    y = 64 - int(np.clip(wseg[idx], -1, 1) * 60)
                    coords.append((x, y))
                if len(coords) > 1:
                    draw.line(coords, fill=(220, 200, 255), width=1)
                wp = out_dir / "images" / f"seg{i:02d}_waveform.png"
                wimg.save(wp)
                seg_out["oscilloscope"]["waveform_image"] = str(wp.name)
            except Exception as e:
                seg_out["oscilloscope"]["image_error"] = str(e)

        if not args.no_images and cn > 0:
            try:
                from PIL import Image

                field = chladni.chladni_field(cn, cm, size=128)
                img = Image.fromarray((field * 255).astype(np.uint8), mode="L")
                p = out_dir / "images" / f"seg{i:02d}_chladni_n{cn}_m{cm}.png"
                img.save(p)
                seg_out["chladni_image"] = str(p.name)
                # Spectrogram slice: mean place x time for this segment, as dB image.
                spec_field = np.array(
                    [p.place * max(p.energy, 0.0) for p in seg], dtype=np.float64
                ).T
                peak = float(spec_field.max()) if spec_field.size else 0.0
                db = (
                    np.clip(20.0 * np.log10(spec_field / peak + 1e-9), -80, 0)
                    if peak > 0
                    else np.full(spec_field.shape, -80)
                )
                norm = ((db + 80) / 80 * 255).astype(np.uint8)
                simg = Image.fromarray(norm, mode="L")
                sp = out_dir / "images" / f"seg{i:02d}_spectrogram.png"
                simg.save(sp)
                seg_out["spectrogram_image"] = str(sp.name)
            except Exception as e:
                seg_out["image_error"] = str(e)

        segments.append(seg_out)
        if args.stream:
            print(json.dumps(seg_out), flush=True)

    # Phonetics, whole file, model-free manner cues (NOT transcription).
    # All fractions are reported, top-first, and no threshold hides
    # the raw numbers.
    phonetics_out = None
    try:
        _feats, _ftrack = _phon.frame_features(percepts, samples, sr)
        _frac = _feats.mean(axis=0) if _feats.size else np.zeros(10)
        _names = _phon.FEATURES
        _cues = {n: round(float(v), 3) for n, v in zip(_names[:5], _frac[:5])}
        _top = sorted(_cues.items(), key=lambda kv: -kv[1])[:3]
        phonetics_out = {
            "note": "Manner cues only, NOT word transcription. "
            "AURIS phonetics does not turn speech into words.",
            "fractions": _cues,
            "top_cues": _top,
            "corrected_summary": ", ".join(
                f"{int(round(v * 100))}% {k}" for k, v in _top if v > 0.05
            )
            or "no manner cue above 5%",
            "auris_summarize_original": _phon.summarize(_feats),
            "formants_mean_hz": [
                round(float(x), 1) if not np.isnan(x) else None
                for x in np.nanmean(_ftrack, axis=0)
            ]
            if _ftrack.size
            else [],
            "voice_onset_time": _phon.voice_onset_time(samples, sr),
        }
    except Exception as e:
        phonetics_out = {"error": str(e)}

    pack = {
        "source": Path(args.wav).name,
        "sr": sr,
        "phonetics": phonetics_out,
        "duration": round(duration, 2),
        "segments": args.segments,
        "model": "AURIS deterministic, model-free. Text tokens are perceive.describe words; "
        "image tokens are the Chladni plus spectrogram PNGs in images/. "
        "A model tokenizes those; AURIS does not produce model tokens directly.",
        "segments_data": segments,
    }
    (out_dir / "token_pack.json").write_text(json.dumps(pack, indent=2))
    if args.labeled_images and not args.no_images:
        try:
            from . import labeled_images as _lab

            pack["labeled_images"] = _lab.label_pack(out_dir)
            (out_dir / "token_pack.json").write_text(json.dumps(pack, indent=2))
        except Exception as e:
            print(json.dumps({"labeled_images_error": str(e)}))
    print(
        json.dumps(
            {
                "out": str(out_dir / "token_pack.json"),
                "segments": len(segments),
                "images": len(list((out_dir / "images").glob("*.png"))),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
