"""Add readable labels and scales to AURIS token-pack images.

Original PNGs are never overwritten. Labeled versions are written
alongside them as ``*_labeled.png``.

What is drawn, and what is not claimed:
- Chladni: "n=X m=Y" plus pitch in Hz and note name, drawn as text.
  The pattern itself is unchanged (upscaled with NEAREST so nodal
  lines stay crisp). The (n, m) numbers come from token_pack.json,
  not from reading the pattern. A vision model given the unlabeled
  pattern alone still cannot uniquely identify (n, m) without a
  reference chart; the label is what makes it readable.
- Spectrogram: the original is (width x 32 places), row 0 = lowest
  cochlea place. The labeled version is flipped vertically so low
  frequencies are at the bottom (conventional orientation),
  enlarged to 640x320, with a log frequency axis in Hz, a time axis
  in seconds (segment start/end), and a dB legend (-80 to 0 dB,
  white = loud). Frequency ticks are interpolated on the cochlea
  centre scale for the pack sample rate when computable, otherwise
  log ticks 50..8000 Hz; the axis is approximate, not a per-pixel
  measurement.
- Waveform: the original 320x128 line is copied (upscaled) from the
  original PNG, so no new amplitude data are invented; peak and RMS
  come from token_pack.json oscilloscope fields. The labeled version
  adds a time axis (segment start/end, seconds), an amplitude axis
  from -1.0 to +1.0, and peak/RMS values as text.

Pillow is optional: these functions are only used when image output
is requested, and they raise a clear error if Pillow is missing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(f0):
    if not f0 or f0 <= 0:
        return "unpitched"
    midi = int(round(69 + 12 * math.log2(f0 / 440.0)))
    return f"{NAMES[midi % 12]}{midi // 12 - 1}"


def font(size, bold=False):
    """Font by name only, with a default fallback. No absolute path."""
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except Exception:
        return ImageFont.load_default()


def centers_for(sr):
    try:
        from .cochlea import Cochlea

        return [float(x) for x in Cochlea(sr=sr).centers]
    except Exception:
        return None


def label_chladni(src, dst, seg):
    from PIL import Image, ImageDraw

    n, m = seg["chladni"]["n"], seg["chladni"]["m"]
    f0 = seg.get("f0_hz", 0.0)
    hf = seg.get("harmonic_f0_hz", 0.0)
    im = Image.open(src).convert("RGB")
    pat = im.resize((384, 384), Image.NEAREST)
    width, height = 560, 384 + 120
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(pat, (88, 80))
    d = ImageDraw.Draw(canvas)
    d.text((88, 8), f"Chladni  n={n}  m={m}", fill=(0, 0, 0), font=font(28, True))
    d.text(
        (88, 42),
        f"pitch {note_name(f0)}  {f0:.1f} Hz (AURIS f0)"
        + (f"  |  harmonic {hf:.1f} Hz" if hf else ""),
        fill=(0, 0, 0),
        font=font(18),
    )
    d.text(
        (88, height - 36),
        "Pattern = vibration shape. (n,m) drawn here from JSON;",
        fill=(60, 60, 60),
        font=font(13),
    )
    d.text(
        (88, height - 18),
        "pattern alone does not uniquely identify (n,m).",
        fill=(60, 60, 60),
        font=font(13),
    )
    canvas.save(dst)
    return dst


def label_spectrogram(src, dst, seg, sr):
    from PIL import Image, ImageDraw

    im = Image.open(src).convert("RGB")
    # Flip: original row 0 (top) = lowest place; labeled = low at bottom.
    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    w0, h0 = 640, 320
    big = im.resize((w0, h0), Image.BILINEAR)
    left, top, right, bottom = 70, 40, 20, 62
    width, height = left + w0 + right, top + h0 + bottom
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(big, (left, top))
    d = ImageDraw.Draw(canvas)
    d.text(
        (left, 6),
        "Spectrogram  (frequency over time, dB, white = loud)",
        fill=(0, 0, 0),
        font=font(18, True),
    )
    d.text(
        (left, height - 22),
        f"time: {seg['t_start']:.2f} s  to  {seg['t_end']:.2f} s   "
        f"|  -80 dB (black) to 0 dB (white)   |  flipped: low freq at bottom",
        fill=(0, 0, 0),
        font=font(14),
    )
    centers = centers_for(sr)
    lo = centers[0] if centers else 40.0
    hi = centers[-1] if centers else 8000.0
    ticks = [t for t in [50, 100, 200, 400, 800, 1600, 3200, 6400] if lo <= t <= hi]
    for t in ticks:
        frac = (math.log(t) - math.log(lo)) / (math.log(hi) - math.log(lo))
        y = top + h0 - int(frac * h0)
        d.line([(left - 6, y), (left, y)], fill=(0, 0, 0), width=1)
        d.text((2, y - 8), f"{t}", fill=(0, 0, 0), font=font(13))
    d.text((2, top - 14), "Hz", fill=(0, 0, 0), font=font(13))
    for frac, lab in [
        (0.0, f"{seg['t_start']:.1f}s"),
        (0.5, f"{(seg['t_start'] + seg['t_end']) / 2:.1f}s"),
        (1.0, f"{seg['t_end']:.1f}s"),
    ]:
        x = left + int(frac * w0)
        d.line([(x, top + h0), (x, top + h0 + 6)], fill=(0, 0, 0), width=1)
        tx = x - 14 if frac < 1.0 else x - 38
        d.text((tx, top + h0 + 8), lab, fill=(0, 0, 0), font=font(13))
    canvas.save(dst)
    return dst


def label_waveform(src, dst, seg):
    from PIL import Image, ImageDraw

    im = Image.open(src).convert("RGB")
    w0, h0 = 640, 256
    big = im.resize((w0, h0), Image.NEAREST)
    left, top, right, bottom = 56, 40, 20, 62
    width, height = left + w0 + right, top + h0 + bottom
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(big, (left, top))
    d = ImageDraw.Draw(canvas)
    osc = seg.get("oscilloscope") or {}
    peak, rms = osc.get("peak", 0.0), osc.get("rms", 0.0)
    d.text(
        (left, 6),
        f"Waveform  amplitude over time  |  peak {peak:.3f}  RMS {rms:.3f}",
        fill=(0, 0, 0),
        font=font(18, True),
    )
    for amp, lab in [(1.0, "+1.0"), (0.5, "+0.5"), (0.0, "0"), (-0.5, "-0.5"), (-1.0, "-1.0")]:
        y = top + int((1.0 - amp) / 2.0 * h0)
        d.line([(left - 6, y), (left, y)], fill=(0, 0, 0), width=1)
        d.text((4, y - 8), lab, fill=(0, 0, 0), font=font(13))
    for frac, lab in [
        (0.0, f"{seg['t_start']:.1f}s"),
        (0.5, f"{(seg['t_start'] + seg['t_end']) / 2:.1f}s"),
        (1.0, f"{seg['t_end']:.1f}s"),
    ]:
        x = left + int(frac * w0)
        d.line([(x, top + h0), (x, top + h0 + 6)], fill=(0, 0, 0), width=1)
        tx = x - 14 if frac < 1.0 else x - 38
        d.text((tx, top + h0 + 8), lab, fill=(0, 0, 0), font=font(13))
    d.text(
        (left, height - 22),
        "Amplitude scale -1 to +1. Peak/RMS above are from token_pack.json.",
        fill=(60, 60, 60),
        font=font(13),
    )
    canvas.save(dst)
    return dst


def label_pack(pack_dir, indices=None):
    """Label every image in a token-pack output directory."""
    pack_dir = Path(pack_dir)
    pack = json.loads((pack_dir / "token_pack.json").read_text())
    imgs = pack_dir / "images"
    sr = pack.get("sr", 44100)
    out = []
    for seg in pack["segments_data"]:
        if indices is not None and seg["index"] not in indices:
            continue
        for kind, fn in [
            ("chladni", seg.get("chladni_image")),
            ("spectrogram", seg.get("spectrogram_image")),
            ("waveform", (seg.get("oscilloscope") or {}).get("waveform_image")),
        ]:
            if not fn:
                continue
            src = imgs / fn
            if not src.exists():
                continue
            dst = imgs / (src.stem + "_labeled.png")
            if kind == "chladni":
                label_chladni(src, dst, seg)
            elif kind == "spectrogram":
                label_spectrogram(src, dst, seg, sr)
            else:
                label_waveform(src, dst, seg)
            out.append(str(dst.name))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack_dir")
    ap.add_argument("--segments", default="")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    idx = None
    if not args.all and args.segments:
        idx = set(int(x) for x in args.segments.split(",") if x.strip() != "")
    made = label_pack(args.pack_dir, idx)
    print(json.dumps({"labeled": made, "count": len(made)}))


if __name__ == "__main__":
    main()
