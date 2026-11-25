# How the ear works

Sound moves through the package in one direction. Each stage reads the
previous stage's output and adds its own.

## Stages

1. Capture. `audio_io.py` reads a WAV file into mono float samples, or
   `listen.py` captures a block from a microphone.
2. Features. `audio_engine.py` computes band energies (bass, mid,
   high), loudness, spectral centroid, spectral flux, and fundamental
   frequency for each frame.
3. Place map. `cochlea.py` lays the block's energy on 32 places on a
   log-frequency scale from 20 Hz to 16,000 Hz, using a zero-padded
   16,384-point FFT. Pitch is searched from 20 Hz to 5,000 Hz and read
   with parabolic interpolation, then named with its signed distance
   in cents from the nearest note.
4. Field. `render.py` builds the spatial field, frequency by time
   weighted by loudness, and can save it as an image and read it back.
   Its `irix_read` function optionally reads the saved field through
   an external `irix` package when one is importable, including from
   a sibling `irix/src` folder next to this repository. Without it,
   the field is read numerically by `read_field`.
5. Trajectory. `spectrotemporal.py` keeps a 24-step by 32-place image
   of how the energy moves over time.
6. Text. `perceive.py` turns the aggregated field into structured
   text: note name and cents, loudness and brightness words, frequency
   region, trends, and Chladni mode weights.
7. Recognition. `gallery.py` compares the descriptor and the
   trajectory against remembered sounds by chi-square distance and
   dynamic time warping. Thresholds are 0.0587 (descriptor) and
   0.0533 (trajectory).
8. Memory. `hippocampus.py` stores the percept with its tonotopic
   signature, time, and context. Consolidation keeps an episode when
   its salience is at least 0.5, or when it repeats at least 3 times
   within duplicate distance 0.1. Recall completes a partial signature
   to the nearest stored episode.
9. Plate model. `chladni.py` weights a fixed table of 78 square plate
   modes by the energy the sound's whole spectrum carries near each
   mode's resonant frequency, 55 x (n^2 + m^2) Hz.

## Boundaries

- The base path reads one dominant pitch at a time.
- No stage calls an AI model. Every stage is arithmetic over the
  previous stage's numbers.
- The package perceives and remembers. It does not act on what it
  hears.
