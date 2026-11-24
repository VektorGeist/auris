"""Token pack tests — the pack describes sounds through the same gate
as the Ear (perceive.gate), stays silent about silence, and fails
cleanly on bad input."""
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import auris in-tree

from auris import token_pack as tp  # noqa: E402

SR = 44100


def _harmonic(freq, seconds=2.0, amp=0.25):
    t = np.arange(int(SR * seconds)) / SR
    return amp * sum(
        a * np.sin(2 * np.pi * freq * k * t)
        for k, a in [(1, 1.0), (2, 0.6), (3, 0.35), (4, 0.2)]
    )


def _write_wav(path, samples):
    x = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(x.tobytes())
    return str(path)


def _run_pack(tmp_path, wav, *extra):
    out = tmp_path / "pack"
    rc = tp.main([wav, "--segments", "2", "--out", str(out), *extra])
    assert rc == 0
    return json.loads((out / "token_pack.json").read_text())


@pytest.fixture(autouse=True)
def _no_home_gallery(monkeypatch, tmp_path):
    monkeypatch.setenv("AURIS_GALLERY", str(tmp_path / "no-gallery.zip"))


def test_chord_pack_is_tonal_not_noise_like(tmp_path):
    chord = (_harmonic(261.63) + _harmonic(329.63) + _harmonic(392.00)) / 3.0
    pack = _run_pack(tmp_path, _write_wav(tmp_path / "chord.wav", chord),
                     "--no-images")
    midis = set()
    for seg in pack["segments_data"]:
        assert "noise-like" not in seg["text"]
        assert "tonal" in seg["text"]
        midis |= {c["midi"] for c in seg["polyphony"]["candidates"]}
    assert {60, 64, 67} <= midis


def test_silence_pack_is_silence(tmp_path, capsys):
    pack = _run_pack(tmp_path, _write_wav(tmp_path / "silence.wav",
                                          np.zeros(SR * 2)),
                     "--labeled-images")
    out = capsys.readouterr().out
    assert "labeled_images_error" not in out
    for seg in pack["segments_data"]:
        assert seg["text"].startswith("silence")
        assert seg["polyphony"]["candidates"] == []
        assert seg["harmonic_f0_hz"] == 0


def test_tone_pack_unchanged_apart_from_confidence(tmp_path):
    pack = _run_pack(tmp_path, _write_wav(tmp_path / "tone.wav",
                                          _harmonic(440.0)),
                     "--no-images")
    for seg in pack["segments_data"]:
        assert seg["text"].startswith("a pitched tone, A4 at 440")
        assert abs(seg["f0_hz"] - 440.0) < 2.0
        assert seg["harmonic_f0_hz"] == 440.0
        assert seg["harmonic_confidence"] > 1.0
        assert seg["polyphony"]["confidence"] > 1.0
        assert seg["polyphony"]["candidates"]


def test_missing_file_is_a_clean_error(tmp_path, capsys):
    rc = tp.main([str(tmp_path / "nope.wav"), "--out", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert rc != 0
    assert "Traceback" not in err
    assert err.count("\n") <= 1


def test_short_file_is_a_clean_error(tmp_path, capsys):
    wav = _write_wav(tmp_path / "short.wav", np.zeros(512))
    rc = tp.main([wav, "--out", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert rc != 0
    assert "Traceback" not in err
    assert "too short" in err
