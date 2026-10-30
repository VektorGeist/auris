# Changelog

## v0.3.0

### Added

- Token pack wrapper (`auris/token_pack.py`): one pack per audio
  file, one block per time segment, with structured AURIS text,
  pitch, loudness, centroid, peak Hz, region, trends, Chladni
  mode numbers, spectrogram slices, and an oscilloscope waveform
  channel (peak, RMS, zero-crossing rate, 20-bin envelope).
- Harmonic pitch correction (`harmonic_f0`): a harmonic-stack
  pitch alongside the original AURIS f0, fixing octave errors
  where a harmonic is louder than the fundamental. The original
  AURIS text is never replaced; a flagged `harmonic_note` field
  is added alongside when the two disagree by more than 40%.
- Polyphony top-5 candidates (`auris/polyphony.py`): harmonic-stack
  candidates for MIDI 21-108, with plus-or-minus one semitone
  non-maximum suppression. Candidate list only.
- Labeled images (`auris/labeled_images.py`, Pillow optional):
  Chladni images labeled with (n, m) and pitch, spectrograms with
  Hz, time and dB axes, and waveforms with time, amplitude, peak
  and RMS. Original PNGs are never overwritten.
- Whole-file phonetics in the pack: manner-cue fractions, top
  cues, formant means and voice-onset time, reported without
  threshold hiding.

### Limits carried with this release, plainly

- Gallery names at the 0.35 default must not be trusted without
  calibration. On general, non-musical sounds the gallery
  false-accepts, naming sounds it never heard. No single
  threshold generalises across levels, so none is set here.
- Speech is prosody and manner cues only, not transcription.
- Polyphony candidates are not clean voicings: octave-below
  duplicates can appear, and candidates fail on noise and on
  sequential notes inside one window.
- Labeled (n, m) and Hz numbers come from JSON, not from reading
  the pattern alone. Spectrogram frequency ticks are approximate.
- The oscilloscope waveform is a separate channel, not an AURIS
  internal.

### Dependencies

- Core remains numpy-only. Pillow is optional, only for image
  and labeled-image output, guarded so a missing Pillow install
  degrades to a reported image error rather than a broken pack.
  The harmonic and polyphony code uses numpy only; no scipy
  dependency was added.

## v0.2.0

- Chladni patterns integrated from SoundGarden's cymatics mode.
