import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auris import tones, audio_io
from auris.sense import Ear
from auris.perceive import note_name

SR = tones.SR


def ear(tmp_path):
    return Ear(data_dir=tmp_path)


def test_note_name():
    assert note_name(440.0) == "A4"
    assert note_name(261.63) == "C4"
    assert note_name(0) is None


def test_wav_roundtrip(tmp_path):
    p = tmp_path / "t.wav"
    x = tones.tone(440, 0.5)
    audio_io.write_wav(str(p), x, SR)
    sr, y = audio_io.read_wav(str(p))
    assert sr == SR
    assert abs(len(y) - len(x)) <= 1
    assert np.corrcoef(x[:len(y)], y[:len(x)])[0, 1] > 0.99


def test_pitched_vs_noise(tmp_path):
    e = ear(tmp_path)
    r = e.hear_samples(tones.tone(440, 1.0), SR, imprint=False)
    assert r["note"] is not None
    assert 250 < r["perception"]["peak_hz"] < 800
    n = e.hear_samples(tones.noise(1.0, seed=1), SR, imprint=False)
    assert n["note"] is None


def test_determinism(tmp_path):
    e = ear(tmp_path)
    x = tones.tone(330, 1.0)
    a = e.hear_samples(x, SR, imprint=False)
    b = e.hear_samples(x, SR, imprint=False)
    assert a["heard"] == b["heard"]
    assert np.allclose(a["descriptor"], b["descriptor"])


def test_gallery_recognises_and_discriminates(tmp_path):
    e = ear(tmp_path)
    e.hear_samples(tones.tone(440, 1.0), SR, remember_as="A4-tone", imprint=False)
    same = e.hear_samples(tones.tone(440, 1.0), SR, imprint=False)
    assert same["recognised"] == "A4-tone"
    diff = e.hear_samples(tones.tone(110, 1.0), SR, imprint=False)
    assert diff["recognised"] != "A4-tone"


def test_hippocampus_recall(tmp_path):
    e = ear(tmp_path)
    e.hear_samples(tones.tone(440, 1.0), SR, t=1.0)
    e.hear_samples(tones.tone(880, 1.0), SR, t=2.0)
    e.hear_samples(tones.chord(dur=1.0), SR, t=3.0)
    cue = e.hear_samples(tones.tone(440, 1.0), SR, imprint=False)["descriptor"]
    got = e.hippocampus.recall(cue, k=1)
    assert got and got[0]["note"] == "A4"
    assert got[0]["dist"] < 0.05


def test_hippocampus_consolidates_to_gallery(tmp_path):
    e = ear(tmp_path)
    e.hear_samples(tones.tone(440, 1.5), SR, t=1.0)
    out = e.consolidate(salience_thresh=0.5, repeat_thresh=99)
    assert out["n"] >= 1
    r = e.hear_samples(tones.tone(440, 1.0), SR, imprint=False)
    assert r["recognised"] is not None


def test_render_mode_runs(tmp_path):
    e = ear(tmp_path)
    r = e.hear_samples(tones.sweep(dur=1.0), SR, mode="render", imprint=False)
    assert r["mode"] == "render-then-read"
    assert r["perception"]["peak_hz"] > 0
    assert (Path(tmp_path) / "last_field.pgm").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
