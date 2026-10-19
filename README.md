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
