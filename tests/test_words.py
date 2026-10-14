"""Layer C tests — utterance segmentation, streaming word hearing, calibration."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auris import tones, spectrotemporal, calibrate, listen   # noqa: E402
from auris.sense import Ear                                    # noqa: E402
from auris import audio_io                                     # noqa: E402
from auris.words import WordLexicon                            # noqa: E402

SR = tones.SR


def _sil(ms):
    return np.zeros(int(SR * ms / 1000.0), dtype=np.float32)


def _vowel(ff, dur=0.4):
    return tones.chord(freqs=ff, dur=dur)


EE = (300, 2300, 3000)
AH = (700, 1200, 2600)


def test_segment_utterances_counts_words():
    stream = np.concatenate([_sil(300), _vowel(EE), _sil(300), _vowel(AH), _sil(300)])
    spans = spectrotemporal.segment_utterances(stream, SR)
    assert len(spans) == 2
    # spans are ordered and non-overlapping
    assert spans[0][1] <= spans[1][0]


def test_hear_words_names_each(tmp_path):
    e = Ear(data_dir=tmp_path)
    for nm, ff in [("ee", EE), ("ah", AH)]:
        p = tmp_path / f"{nm}.wav"; audio_io.write_wav(str(p), _vowel(ff, 0.5), SR)
        e.remember_word(p, nm)
    stream = np.concatenate([_sil(300), _vowel(EE), _sil(300), _vowel(AH), _sil(300)])
    words = e.hear_words(stream, SR)
    assert [w["nearest"] for w in words] == ["ee", "ah"]


def test_listen_words_streaming(tmp_path):
    e = Ear(data_dir=tmp_path)
    for nm, ff in [("ee", EE), ("ah", AH)]:
        p = tmp_path / f"{nm}.wav"; audio_io.write_wav(str(p), _vowel(ff, 0.5), SR)
        e.remember_word(p, nm)
    # a stream delivered in fixed windows through the injectable capture hook
    stream = np.concatenate([_sil(300), _vowel(EE), _sil(400),
                             _vowel(AH), _sil(500)]).reshape(-1, 1)
    win = int(0.5 * SR)
    chunks = [stream[i:i + win] for i in range(0, len(stream), win)]
    it = iter(chunks)

    def cap(frames, sr, channels, device):
        try:
            c = next(it)
        except StopIteration:
            raise KeyboardInterrupt      # stop the infinite windows() generator
        # pad the final short chunk
        if len(c) < frames:
            c = np.vstack([c, np.zeros((frames - len(c), 1), np.float32)])
        return c[:frames]

    got = []
    try:
        for w in e.listen_words(window=0.5, _capture=cap):
            got.append(w["nearest"])
    except KeyboardInterrupt:
        pass
    assert got == ["ee", "ah"]


def test_calibrate_separates(tmp_path):
    e = Ear(data_dir=tmp_path)
    # write two takes each of two clearly-different vowels
    for nm, ff in [("ee", EE), ("ah", AH)]:
        for k, dur in enumerate((0.5, 0.42)):
            p = tmp_path / f"{nm}_{k}.wav"
            audio_io.write_wav(str(p), _vowel(ff, dur), SR)
    rep = calibrate.calibrate_dir(e, tmp_path)
    assert rep["ok"] and rep["separable"]
    assert rep["same"]["max"] < rep["threshold"] < rep["diff"]["min"]


def _wtraj(centers, F=spectrotemporal.N_PLACE, T=spectrotemporal.T_STEPS, width=1.5):
    """A held T x F 'vowel' trajectory: formant bumps at given tonotopic bins."""
    x = np.arange(F)
    frame = sum(np.exp(-0.5 * ((x - c) / width) ** 2) for c in centers)
    frame = frame / frame.sum()
    return np.tile(frame, (T, 1)).astype(np.float64)


def test_pitch_invariant_dtw_collapses_transposition_not_identity():
    A = _wtraj([4, 10, 13])                             # a vowel (formant spacing = its identity)
    A_hi = spectrotemporal._shift_freq(A, 3)            # SAME vowel, pitched up 3 tonotopic bins
    B = _wtraj([4, 15, 19])                             # a DIFFERENT vowel (different spacing)
    d_plain = spectrotemporal.dtw(A, A_hi)             # pitch-sensitive: reads far apart
    d_inv = spectrotemporal.dtw_pitch_invariant(A, A_hi, max_shift=4)
    assert d_inv < 0.2 * d_plain + 1e-9                # invariant collapses the transposition
    assert spectrotemporal.dtw_pitch_invariant(A, B, max_shift=4) > 5 * d_inv + 1e-6  # not the different vowel


def test_word_lexicon_pitch_invariant_recognizes_transposed_word(tmp_path):
    lex = WordLexicon(tmp_path / "lex.zip")
    lex.remember(_wtraj([4, 10, 13]), "aa")
    q = spectrotemporal._shift_freq(_wtraj([4, 10, 13]), 3)     # the word, said higher
    plain = lex.match(q, threshold=1e9)
    inv = lex.match(q, threshold=1e9, pitch_invariant=True)
    assert inv["dist"] < plain["dist"]                 # invariant matches the transposition closer
    assert inv["nearest"] == "aa"


def test_word_lexicon_multiple_exemplars(tmp_path):
    e = Ear(data_dir=tmp_path)
    for k, dur in enumerate((0.5, 0.4, 0.45)):
        p = tmp_path / f"ee_{k}.wav"; audio_io.write_wav(str(p), _vowel(EE, dur), SR)
        e.remember_word(p, "ee")
    st = e.words.stats()
    assert st["words"] == 1 and st["exemplars"] == 3
