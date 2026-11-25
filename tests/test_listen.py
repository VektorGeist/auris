"""Live microphone path tests with an injected capture backend."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auris import listen, tones
from auris.sense import Ear

SR = tones.SR


def _fake_capture(buf):
    def cap(frames, sr, channels, device):
        return buf
    return cap


def test_to_mono_downmix_and_clip():
    stereo = np.array([[1.0, -1.0], [0.5, 0.5], [2.0, 2.0]], dtype=np.float32)
    mono = listen._to_mono(stereo)
    assert mono.shape == (3,)
    assert mono.dtype == np.float32
    assert np.allclose(mono[:2], [0.0, 0.5])
    assert mono[2] == 1.0


def test_record_shape_and_rate():
    buf = np.zeros((SR // 2, 2), dtype=np.float32)
    sr, mono = listen.record(0.5, sr=SR, _capture=_fake_capture(buf))
    assert sr == SR
    assert mono.ndim == 1 and len(mono) == SR // 2
    assert mono.dtype == np.float32


def test_record_gain_boosts_then_clips():
    quiet = (np.ones((10, 1), dtype=np.float32) * 0.1)
    _, mono = listen.record(10 / SR, sr=SR, gain=20.0, _capture=_fake_capture(quiet))
    assert np.all(mono == 1.0)


def test_mic_feeds_hear_samples_like_a_wav(tmp_path, monkeypatch):

    tone = tones.tone(440, 1.0)
    monkeypatch.setattr(listen, "record", lambda *a, **k: (SR, tone))
    e = Ear(data_dir=tmp_path)
    out = e.hear_mic(seconds=1.0, imprint=False)
    assert out["source"] == "mic"
    assert out["note"] is not None
    assert "perception" in out


def test_listen_stream_stops_after_budget(tmp_path, monkeypatch):
    tone = tones.tone(440, 0.5)
    monkeypatch.setattr(listen, "windows",
                        lambda *a, **k: iter([(SR, tone)] * 10))
    e = Ear(data_dir=tmp_path)
    outs = list(e.listen_stream(window=0.5, seconds=1.0, imprint=False))
    assert len(outs) == 2
    assert all(o["source"] == "mic" for o in outs)
    assert outs[0]["t"] == 0.0 and outs[1]["t"] == 0.5


def test_list_inputs_never_raises():
    got = listen.list_inputs()
    assert isinstance(got, list)
