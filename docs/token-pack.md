# Token pack format

One pack per audio file, written as `token_pack.json`, with one block
per time segment.

## Pack fields

- `source`: the input file name.
- `sr`: the sample rate.
- `duration`: the file length in seconds.
- `segments`: the number of segments.
- `segments_data`: the segment blocks.
- `phonetics`: whole-file phonetics: `fractions` (voiced, vowel,
  fricative, plosive, nasal), `top_cues`, `formants_mean_hz`, and
  `voice_onset_time`.
- `labeled_images`: the file names of the labeled images, when image
  output is enabled.
- `model`: a fixed line stating that the pack's text tokens are
  `perceive.describe` words and its image tokens are the Chladni and
  spectrogram PNGs.

## Segment fields

- `index`, `t_start`, `t_end`: the segment number and its time span
  in seconds.
- `text`: the structured text for the segment.
- `auris_text_original`: the structured text before any correction
  fields are applied.
- `f0_hz`, `note`, `cents_off`: the dominant pitch, its note name,
  and its signed distance in cents from the exact note.
- `harmonic_f0_hz`, `harmonic_midi`, `harmonic_note`,
  `harmonic_confidence`: the harmonic-stack pitch reading and its
  confidence.
- `octave_corrected`: whether the harmonic reading corrected the
  base pitch by an octave.
- `loud`, `centroid`, `peak_hz`, `region`: loudness, spectral
  centroid, peak frequency, and frequency region word.
- `pitch_trend`, `loud_trend`, `onset_flux_mean`: direction of
  change across the segment, and mean onset flux.
- `chladni`: `modes`, the excited plate modes as (n, m, weight)
  triples, and `weights`, the full weight vector.
- `chladni_image`, `spectrogram_image`: the segment's image file
  names.
- `oscilloscope`: the waveform channel: `peak`, `rms`,
  `zero_crossing_rate_per_s`, `envelope_20` (a 20-bin envelope),
  `text`, and `waveform_image`.
- `gallery`: the nearest remembered sound: `name`, `dist`, `known`,
  `nearest`, and `via`.
- `gallery_calibration_warning`: a fixed line stating that the
  gallery thresholds were measured on synthetic exam sets and must
  be calibrated on real captures before a name is trusted.
- `polyphony`: `candidates` (up to 5 pitches), `confidence`,
  `method`, and `top_n`.

## Images

With image output enabled, each segment also produces a spectrogram
image, a waveform image, and a Chladni image in the pack's `images/`
folder. Labeled versions add mode numbers and pitch to the Chladni
image, Hz, time and dB axes to the spectrogram, and time, amplitude,
peak and RMS to the waveform. Original images are never overwritten;
labeled versions are written alongside as `*_labeled.png`.
