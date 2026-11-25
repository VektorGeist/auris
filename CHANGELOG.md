# Changelog

## Unreleased

### Added

- Package author metadata: VektorGeist, vektorgeistllc@gmail.com.
- The `images` optional dependency set (Pillow) is declared in the
  package metadata.

### Fixed

- Token pack runs on silent input no longer print a "Mean of empty
  slice" warning from the formant means.
- Image writes no longer pass a mode to `Image.fromarray`.
- A token pack run with images requested and Pillow missing prints
  that images were skipped and how to install Pillow, instead of
  writing no images silently.

## v0.3.0

### Added

- Token pack (`auris/token_pack.py`): one pack per audio file,
  one block per time segment, with structured text, pitch,
  loudness, centroid, peak Hz, region, trends, Chladni mode
  numbers, spectrogram slices, and an oscilloscope waveform
  channel (peak, RMS, zero-crossing rate, 20-bin envelope).
- Harmonic pitch correction (`harmonic_f0`): a harmonic-stack
  pitch alongside the original f0. A flagged `harmonic_note`
  field is added when the two disagree by more than 40%.
- Polyphony top-5 candidates (`auris/polyphony.py`):
  harmonic-stack candidates for MIDI 21-108, with plus-or-minus
  one semitone non-maximum suppression.
- Labeled images (`auris/labeled_images.py`, Pillow optional):
  Chladni images labeled with (n, m) and pitch, spectrograms
  with Hz, time and dB axes, and waveforms with time, amplitude,
  peak and RMS.
- Whole-file phonetics in the pack: manner-cue fractions, top
  cues, formant means, and voice-onset time.
- Cochlea floor at 20 Hz. Pitch is searched from 20 Hz to
  5,000 Hz on a zero-padded 16,384-point FFT and read with
  parabolic interpolation.
- Pitch readouts carry a note name and a signed cents value.

### Fixed

- One shared gate helper, `perceive.gate`, called by `sense.Ear`
  and `token_pack`. Pack segments carry tonal, silent, and
  flatness. `perceive.describe` raises an error when the gate
  fields are missing.
- Silent pack segments skip the harmonic stack and the polyphony
  scan. Non-silent harmonic and polyphony outputs carry a
  confidence: the top score against the median score of the
  whole scan.
- `labeled_images` labels from the modes list and skips a
  segment with no modes.
- A missing input file and a too-short clip print a one-line
  error to stderr and exit non-zero.

### Dependencies

- Core is numpy only. Pillow is optional, for image and
  labeled-image output.

## v0.2.0

- Chladni patterns: a square plate model with a fixed table of
  modes, weighted by a sound's spectrum.
