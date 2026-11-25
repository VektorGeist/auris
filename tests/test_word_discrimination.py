"""Tests that synthetic minimal-pair words are discriminated."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _speech as S
from auris import audio_io, calibrate
from auris.gallery import chi2
from auris.sense import Ear
from auris.spectrotemporal import trajectory

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

    e = Ear(data_dir=tmp_path)
    _enroll(e, tmp_path)
    for w in SPECS:
        cohort = _heldout(e, tmp_path, w)["word"]["cohort"]
        best, second = cohort[0], cohort[1]
        assert best["name"] == w
        assert best["dist"] < 0.5 * second["dist"]


def test_cat_vs_cot_separate_on_the_vowel(tmp_path):

    e = Ear(data_dir=tmp_path)
    _enroll(e, tmp_path)
    d = {c["name"]: c["dist"] for c in _heldout(e, tmp_path, "cot")["word"]["cohort"]}
    assert d["cot"] < d["cat"]


def test_cat_vs_cap_the_median_is_blind_but_dtw_is_not(tmp_path):

    e = Ear(data_dir=tmp_path)
    _enroll(e, tmp_path)
    d = {c["name"]: c["dist"] for c in _heldout(e, tmp_path, "cap")["word"]["cohort"]}
    assert d["cap"] < d["cat"]


    def median_desc(spec):
        sr, x = SR, S.take(spec, 0)
        percepts, _ = e._transduce(x, sr)
        m = np.median(np.array([p.place for p in percepts]), axis=0)
        return m / m.sum()
    median_gap = chi2(median_desc(S.CAT), median_desc(S.CAP))
    assert median_gap < 0.02

    assert (d["cat"] - d["cap"]) > median_gap


def test_determinism(tmp_path):
    e = Ear(data_dir=tmp_path)
    x = S.take(S.CAT, 0)
    t1 = trajectory(e._transduce(x, SR)[0])["traj"]
    t2 = trajectory(e._transduce(x, SR)[0])["traj"]
    assert np.array_equal(t1, t2)


def test_calibration_separates_different_vowels(tmp_path):


    e = Ear(data_dir=tmp_path)
    caldir = tmp_path / "cal"; caldir.mkdir()


    for w, spec in [("cat", S.CAT), ("cot", S.COT)]:
        init, v, fin = spec
        for i, f0 in enumerate((119.0, 120.0, 121.0)):
            sig = S.word(init, v, fin, dur=0.34, f0=f0)
            audio_io.write_wav(str(caldir / f"{w}_{i}.wav"), sig, SR)
    rep = calibrate.calibrate_dir(e, caldir)
    assert rep["ok"] and rep["separable"]
    assert rep["same"]["max"] < rep["threshold"] < rep["diff"]["min"]
