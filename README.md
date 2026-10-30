# Auris — a deterministic sense of hearing

Auris gives an agent the **concept of hearing** without any AI model. It is how the
brain actually does it: the ear does not record sound, it **lays it out in space**
(the cochlea is a physical frequency map — tonotopy). Auris turns sound into a
spatial field, reads that field into **structured text** a text-brain can reason on,
remembers what it hears, and learns to name it. Pure DSP + geometry + remembered
exemplars. Sibling to **Irix** (the eyes).

## The organism (model-free end to end)

| Organ | Module | What it does |
|---|---|---|
| Ear | `cochlea.py` (+ Soundgarden's `Analyzer`) | sound → features + a **tonotopic place map** (sound in space) |
| Perception | `perceive.py` (Irix idiom) | field → **structured text** — `features` mode (fast) or `render` mode (see-then-read) |
| Hippocampus | `hippocampus.py` | **episodic memory**: bind → hold → consolidate (replay) → recall (pattern completion) |
| Recognition | `gallery.py` | a growing store of remembered sounds → names by nearest-neighbour ("no ceiling = memory") |

Wired **afferent-only** (AGM): Auris perceives and remembers, it never acts.

## Try it

```bash
python -m auris.cli tone --kind tone --freq 440 --dur 1 --out a.wav
python -m auris.cli hear a.wav                 # perceive -> text + a memory
python -m auris.cli hear a.wav --mode render   # see the sound, read it back
python -m auris.cli remember a.wav "the A"     # name it
python -m auris.cli hear a.wav                 # now recognised
python -m auris.cli recall a.wav               # what earlier sounds it resembles
python -m auris.cli consolidate                # replay salient/repeated -> long-term
python -m auris.cli stats
```

## Chladni patterns (v0.2.0)

Auris can now visualize sound as **Chladni figures** — the vibration patterns
that form on a resonant plate. Integrated from SoundGarden's cymatics mode.

Each figure is determined by the pitch and timbre at that moment: the same
physics Beethoven felt through his jawbone when composing deaf.

```python
from auris import Ear

ear = Ear()
# From audio samples
figures = ear.chladni(samples, sr, size=128)

# From a WAV file
figures = ear.chladni_file("song.wav", size=128, as_images=True)
```

Or use the module directly:

```python
from auris import chladni

# Single figure for a mode (n, m)
field = chladni.chladni_field(n=3, m=4, size=128)

# Timeline from percepts
figures = chladni.chladni_timeline(percepts)

# Horizontal strip image
strip = chladni.chladni_strip(percepts, size=128)
```


## Token packs (v0.3.0)

New in v0.3.0, as a wrapper around the unchanged core above:

- **Harmonic pitch correction** (`auris/token_pack.py`, `harmonic_f0`):
  a harmonic-stack pitch alongside the original AURIS f0. It fixes
  octave errors where a harmonic is louder than the fundamental.
  The original AURIS text is never replaced; when the harmonic
  stack disagrees by more than 40%, a flagged `harmonic_note` field
  is added alongside, not instead.
- **Polyphony top-5 candidates** (`auris/polyphony.py`): harmonic-stack
  candidates for MIDI 21-108, with plus-or-minus one semitone
  non-maximum suppression. A candidate list only, not a clean voicing.
- **Labeled images** (`auris/labeled_images.py`, Pillow optional):
  Chladni images labeled with (n, m) and pitch, spectrograms with
  Hz, time and dB axes, and waveforms with time, amplitude, peak
  and RMS. Original PNGs are never overwritten; labeled versions
  are written alongside as `*_labeled.png`.

```bash
python -m auris.token_pack input.wav --segments 20 --out out_dir
python -m auris.token_pack input.wav --segments 20 --out out_dir --no-images
python -m auris.token_pack input.wav --segments 20 --out out_dir --labeled-images
```

### Limits, plainly

- **Gallery names at the 0.35 default must not be trusted without
  calibration.** On general, non-musical sounds the gallery
  false-accepts: it names sounds it never heard. The default
  threshold is a synthetic placeholder. No single threshold
  generalises across clean piano renders, real instruments, and
  noisy recordings, so no threshold is set by this package.
- **Speech is prosody and manner cues only, not transcription.**
  Phonetics reports voiced, vowel, fricative, plosive and nasal
  fractions, formants and voice-onset time. It does not turn
  speech into words.
- **Polyphony candidates are not clean voicings.** Octave-below
  duplicates can appear in the later slots, candidates fail on
  noisy sounds, and sequential notes inside one window are not
  recovered as a full set.
- **Labeled (n, m) and Hz numbers come from JSON, not from reading
  the pattern alone.** A Chladni pattern alone does not uniquely
  identify (n, m); the label copies those numbers from the pack
  onto the image. Spectrogram frequency ticks are interpolated on
  the cochlea centre scale and are approximate, not a per-pixel
  frequency measurement.
- **The oscilloscope waveform is a separate channel, not an AURIS
  internal.** AURIS does not use an oscilloscope. Waveform =
  amplitude over time, Chladni = vibration shape, spectrogram =
  frequency over time. Different views, neither replaces the other.

## Test

```bash
python -m pytest tests/ -v
```

## Rules

- **No AI / no model, ever.** DSP + geometry + remembered exemplars only.
- **Afferent-only** — a sense, not a controller.
- Reuse, don't fork: the ear is Soundgarden's audio engine; Irix (optional) reads the
  rendered field in `render` mode. Auris owns the hippocampus and the sound gallery.

Data lives in `~/.auris/` (override with `AURIS_DIR`).
