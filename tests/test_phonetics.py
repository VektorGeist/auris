"""Layer B tests — formants, VOT categorical boundary, manner cues (synthetic)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auris import tones, formants, phonetics       # noqa: E402
from auris.cochlea import BLOCK, Cochlea            # noqa: E402
from auris import audio_io                          # noqa: E402

SR = tones.SR


def _transduce(x, sr=SR):
    c = Cochlea(sr=sr); dt = c.block / c.sr; ps = []; t = 0.0
    for b in audio_io.blocks(x, c.block):
        ps.append(c.hear_block(b, t)); t += dt
    return ps


def _vowel(f_formants=(700, 1200, 2600), dur=0.5, sr=SR):
    """A synthetic vowel: strong energy at the formant frequencies."""
    return tones.chord(freqs=f_formants, dur=dur, sr=sr)


def test_spectrum_peaks_find_formants():
    x = _vowel((700, 1200, 2600), dur=0.3)
    window = np.hanning(BLOCK).astype(np.float32)
    freqs = np.fft.rfftfreq(BLOCK, 1.0 / SR)
    mid = x[len(x) // 2 - BLOCK // 2: len(x) // 2 + BLOCK // 2]
    F, C = formants.spectrum_peaks(np.abs(np.fft.rfft(mid * window)), freqs)
    assert abs(F[0] - 700) < 100
    assert abs(F[1] - 1200) < 120
    # F3 near 2600 (allow it to be missed/weaker, but if present it's in range)
    assert np.isnan(F[2]) or abs(F[2] - 2600) < 200


def test_formants_ascending():
    x = _vowel((500, 1500, 2500), dur=0.3)
    window = np.hanning(BLOCK).astype(np.float32)
    freqs = np.fft.rfftfreq(BLOCK, 1.0 / SR)
    mid = x[BLOCK: 2 * BLOCK]
    F, _ = formants.spectrum_peaks(np.abs(np.fft.rfft(mid * window)), freqs)
    got = [f for f in F if not np.isnan(f)]
    assert got == sorted(got)                        # F1 < F2 < F3


def _burst_then_voice(gap_ms, sr=SR):
    """A plosive-like burst, a silent gap, then a voiced tone (a synthetic CV)."""
    burst = tones.noise(0.006, sr=sr, amp=0.9, seed=1)
    gap = np.zeros(int(sr * gap_ms / 1000.0), dtype=np.float32)
    voiced = tones.tone(150, 0.35, sr=sr, amp=0.6, harmonics=4)
    return np.concatenate([burst, gap, voiced]).astype(np.float32)


def test_vot_categorical_boundary():
    short = phonetics.voice_onset_time(_burst_then_voice(10), SR)
    long = phonetics.voice_onset_time(_burst_then_voice(70), SR)
    assert short["vot_ms"] is not None and long["vot_ms"] is not None
    assert short["vot_ms"] < 30 <= long["vot_ms"]
    assert short["phon_class"] == "voiced"           # /b,d,g/
    assert long["phon_class"] == "voiceless"         # /p,t,k/


def test_frame_features_voiced_vs_noise():
    voiced = tones.tone(180, 0.4, harmonics=5)
    ft_v, _ = phonetics.frame_features(_transduce(voiced), voiced, SR)
    ft_n_sig = tones.noise(0.4, seed=3)
    ft_n, _ = phonetics.frame_features(_transduce(ft_n_sig), ft_n_sig, SR)
    # voiced tone reads far more 'voiced' than white noise does
    vi = phonetics.FEATURES.index("voiced")
    assert ft_v[:, vi].mean() > ft_n[:, vi].mean()


def test_fricative_on_noise():
    x = tones.noise(0.4, seed=7)                      # broadband, unvoiced -> fricative-like
    ft, _ = phonetics.frame_features(_transduce(x), x, SR)
    fi = phonetics.FEATURES.index("fricative")
    assert ft[:, fi].mean() > 0.3


def test_feature_track_width():
    x = tones.tone(200, 0.3, harmonics=3)
    ft, ftrack = phonetics.frame_features(_transduce(x), x, SR)
    assert ft.shape[1] == phonetics.K
    assert ftrack.shape[1] == 3
