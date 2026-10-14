"""End-to-end proof: Auris discriminates the minimal-pair words cat / cap / cot.

All synthetic (tests/_speech.py) — no microphone, no voice, no model. The two
things this proves:
  1. cat vs cot — different VOWEL — separate (Auris hears which vowel it is).
  2. cat vs cap — SAME vowel, different FINAL consonant — separate too, AND the old
     time-averaged median descriptor is blind to that difference. This is the whole
     point of the spectrotemporal upgrade: keeping time lets Auris hear the part of
     the word the median throws away.

Recognition is 1-NN over remembered exemplars (memory, not inference); a single
global threshold is deliberately NOT the criterion here — minimal pairs are exactly
where nearest-neighbour beats a fixed cutoff (see the note in the calibration test).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _speech as S                                    # noqa: E402
from auris import audio_io, calibrate                  # noqa: E402
from auris.gallery import chi2                          # noqa: E402
from auris.sense import Ear                             # noqa: E402
from auris.spectrotemporal import trajectory           # noqa: E402

SR = S.SR
SPECS = {"cat": S.CAT, "cap": S.CAP, "cot": S.COT}


def _enroll(ear, tmp_path, takes=(0, 1, 2)):
    for w, spec in SPECS.items():
        for i in takes:
            p = tmp_path / f"{w}_{i}.wav"
            audio_io.write_wav(str(p), S.take(spec, i), SR)
            ear.remember_word(p, w)


def _heldout(ear, tmp_path, w, i=3):
    p = tmp_path / f"{w}_held.wav"
    audio_io.write_wav(str(p), S.take(SPECS[w], i), SR)
    return ear.hear_word(p)


def test_all_three_words_named_correctly(tmp_path):
    e = Ear(data_dir=tmp_path)
    _enroll(e, tmp_path)
    for w in SPECS:
        out = _heldout(e, tmp_path, w)
        assert out["word"]["nearest"] == w, f"{w} misnamed as {out['word']['nearest']}"


def test_per_query_margin_is_comfortable(tmp_path):
    # each held-out word must be clearly closer to its own kind than to any other
    e = Ear(data_dir=tmp_path)
    _enroll(e, tmp_path)
    for w in SPECS:
        cohort = _heldout(e, tmp_path, w)["word"]["cohort"]
        best, second = cohort[0], cohort[1]
        assert best["name"] == w
        assert best["dist"] < 0.5 * second["dist"]      # >2x margin to the runner-up


def test_cat_vs_cot_separate_on_the_vowel(tmp_path):
    # cat and cot share consonants; they must separate because the VOWEL differs
    e = Ear(data_dir=tmp_path)
    _enroll(e, tmp_path)
    d = {c["name"]: c["dist"] for c in _heldout(e, tmp_path, "cot")["word"]["cohort"]}
    assert d["cot"] < d["cat"]


def test_cat_vs_cap_the_median_is_blind_but_dtw_is_not(tmp_path):
    # THE proof: same vowel, differ only in the final consonant.
    e = Ear(data_dir=tmp_path)
    _enroll(e, tmp_path)
    d = {c["name"]: c["dist"] for c in _heldout(e, tmp_path, "cap")["word"]["cohort"]}
    assert d["cap"] < d["cat"]                           # DTW hears the /p/ vs /t/

    # the old median snapshot cannot tell cat from cap at all
    def median_desc(spec):
        sr, x = SR, S.take(spec, 0)
        percepts, _ = e._transduce(x, sr)
        m = np.median(np.array([p.place for p in percepts]), axis=0)
        return m / m.sum()
    median_gap = chi2(median_desc(S.CAT), median_desc(S.CAP))
    assert median_gap < 0.02                             # blind: cat ~= cap for the median
    # and the DTW gap between cat and cap is far larger than that median blur
    assert (d["cat"] - d["cap"]) > median_gap


def test_determinism(tmp_path):
    e = Ear(data_dir=tmp_path)
    x = S.take(S.CAT, 0)
    t1 = trajectory(e._transduce(x, SR)[0])["traj"]
    t2 = trajectory(e._transduce(x, SR)[0])["traj"]
    assert np.array_equal(t1, t2)


def test_calibration_separates_different_vowels(tmp_path):
    # A single global threshold works when words differ by vowel (cat vs cot). It
    # deliberately does NOT for the same-vowel minimal pair (cat/cap) — that's why
    # recognition is 1-NN/cohort, not a fixed cutoff. Here we verify the separable
    # case calibrates cleanly.
    e = Ear(data_dir=tmp_path)
    caldir = tmp_path / "cal"; caldir.mkdir()
    # controlled timing (fixed duration, only micro pitch variation) so the measure
    # reflects word identity, not DTW's length-warp residual
    for w, spec in [("cat", S.CAT), ("cot", S.COT)]:
        init, v, fin = spec
        for i, f0 in enumerate((119.0, 120.0, 121.0)):
            sig = S.word(init, v, fin, dur=0.34, f0=f0)
            audio_io.write_wav(str(caldir / f"{w}_{i}.wav"), sig, SR)
    rep = calibrate.calibrate_dir(e, caldir)
    assert rep["ok"] and rep["separable"]
    assert rep["same"]["max"] < rep["threshold"] < rep["diff"]["min"]
