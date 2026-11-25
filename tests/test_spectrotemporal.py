"""Tests for time-resolved hearing."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auris import tones, spectrotemporal
from auris.gallery import chi2
from auris.sense import Ear
from auris.spectrotemporal import trajectory, dtw, trim_to_voice

SR = tones.SR


def _transduce_only(x, sr):
    from auris.cochlea import Cochlea
    from auris import audio_io
    c = Cochlea(sr=sr)
    dt = c.block / c.sr
    ps, t = [], 0.0
    for blk in audio_io.blocks(x, c.block):
        ps.append(c.hear_block(blk, t)); t += dt
    return ps


def test_trajectory_shape_and_norm():
    tr = trajectory(_transduce_only(tones.sweep(110, 1760, 1.0), SR))
    assert tr["traj"].shape == (spectrotemporal.T_STEPS, spectrotemporal.N_PLACE)
    row_sums = tr["traj"].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-4)
    assert not tr["silent"]


def test_dtw_identical_is_zero():
    tr = trajectory(_transduce_only(tones.sweep(110, 1760, 1.0), SR))["traj"]
    assert dtw(tr, tr) < 1e-9


def test_rising_vs_falling_median_collapses_dtw_separates():
    rise = _transduce_only(tones.sweep(110, 1760, 1.2), SR)
    rise2 = _transduce_only(tones.sweep(110, 1760, 0.9), SR)
    fall = _transduce_only(tones.sweep(1760, 110, 1.2), SR)


    med = lambda ps: np.median(np.array([p.place for p in ps]), axis=0)
    med_r, med_f = med(rise), med(fall)
    med_r /= med_r.sum(); med_f /= med_f.sum()
    median_gap = chi2(med_r, med_f)

    tr_r = trajectory(rise)["traj"]
    tr_r2 = trajectory(rise2)["traj"]
    tr_f = trajectory(fall)["traj"]
    same = dtw(tr_r, tr_r2)
    diff = dtw(tr_r, tr_f)

    assert same < diff


    assert (diff - same) > median_gap


def test_trim_to_voice_excludes_silence():
    sig = np.concatenate([np.zeros(SR // 2, np.float32),
                          tones.tone(440, 0.6), np.zeros(SR // 2, np.float32)])
    ps = _transduce_only(sig, SR)
    a, b = trim_to_voice(ps)
    assert b > a

    assert a > 0 and b < len(ps)


def test_ear_word_remember_and_match(tmp_path):
    e = Ear(data_dir=tmp_path)
    from auris import audio_io
    for nm, sig in [("rising", tones.sweep(110, 1760, 1.1)),
                    ("falling", tones.sweep(1760, 110, 1.1))]:
        p = tmp_path / f"{nm}.wav"; audio_io.write_wav(str(p), sig, SR)
        e.remember_word(p, nm)

    q = tmp_path / "q.wav"; audio_io.write_wav(str(q), tones.sweep(110, 1760, 0.85), SR)
    out = e.hear_word(q)
    assert out["word"]["nearest"] == "rising"
    names = [c["name"] for c in out["word"]["cohort"]]
    assert names[0] == "rising"


def test_word_path_is_opt_in(tmp_path):

    e = Ear(data_dir=tmp_path)
    out = e.hear_samples(tones.tone(440, 0.6), SR, imprint=False)
    assert "word" not in out and "trajectory" not in out


def test_determinism(tmp_path):
    x = tones.sweep(110, 1760, 1.0)
    t1 = trajectory(_transduce_only(x, SR))["traj"]
    t2 = trajectory(_transduce_only(x, SR))["traj"]
    assert np.array_equal(t1, t2)
    assert dtw(t1, t2) < 1e-9
