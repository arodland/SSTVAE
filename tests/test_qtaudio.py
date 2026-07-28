"""The Qt capture backend's decodable parts.

Only the Qt-free logic is exercised here, on purpose. Constructing a
`QApplication` and running an event loop inside pytest has hung this
project's test runs before, and the interesting content is not in the Qt
glue anyway: it is the byte conversion and the device matching, both of
which are pure functions taking plain values.

What is *not* covered here, and was verified by hand against a real
device instead (K4 RX A at 48 kHz, with a thread deliberately holding the
GIL): that capture keeps every sample through 800 ms of GIL starvation,
where the PortAudio callback lost 0.35% of the stream at ~30 ms.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6.QtMultimedia",
                    reason="QtMultimedia is in pyside6-addons")

from sstvae.gui.qtaudio import bytes_to_mono, match_device  # noqa: E402


# --- sample conversion ---------------------------------------------------

def test_float_passes_through_unscaled():
    x = np.array([-1.0, -0.5, 0.0, 0.25, 1.0], dtype=np.float32)
    got = bytes_to_mono(x.tobytes(), "Float", 1)
    assert got.dtype == np.float64
    assert np.allclose(got, x)


def test_int16_is_scaled_to_plus_minus_one():
    x = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
    got = bytes_to_mono(x.tobytes(), "Int16", 1)
    assert np.isclose(got[0], -1.0)
    assert np.isclose(got[2], 0.0)
    assert 0.99 < got[4] < 1.0
    assert np.abs(got).max() <= 1.0


def test_int32_is_scaled_to_plus_minus_one():
    x = np.array([-(2 ** 31), 0, 2 ** 31 - 1], dtype=np.int32)
    got = bytes_to_mono(x.tobytes(), "Int32", 1)
    assert np.isclose(got[0], -1.0)
    assert np.isclose(got[1], 0.0)
    assert np.abs(got).max() <= 1.0


def test_uint8_is_centred():
    """Unsigned 8-bit is offset-binary: silence is 128, not 0."""
    x = np.array([0, 128, 255], dtype=np.uint8)
    got = bytes_to_mono(x.tobytes(), "UInt8", 1)
    assert np.isclose(got[0], -1.0)
    assert np.isclose(got[1], 0.0)
    assert 0.9 < got[2] <= 1.0


def test_stereo_is_mixed_down_not_decimated():
    """Averaging, not "take channel 0".

    A radio interface that puts the audio on the right channel only would
    otherwise capture silence -- and silence that syncs to nothing looks
    like a dead device rather than a wiring choice.
    """
    left = np.zeros(4, dtype=np.float32)
    right = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
    inter = np.empty(8, dtype=np.float32)
    inter[0::2], inter[1::2] = left, right
    got = bytes_to_mono(inter.tobytes(), "Float", 2)
    assert len(got) == 4
    assert np.allclose(got, right / 2)


def test_a_trailing_partial_frame_is_dropped():
    """Qt can hand us a byte count that is not a whole number of frames.

    Reshaping that would misalign every sample after it -- far worse than
    losing one.
    """
    x = np.array([1.0, 1.0, 2.0, 2.0, 3.0], dtype=np.float32)  # 5 for 2 ch
    got = bytes_to_mono(x.tobytes(), "Float", 2)
    assert np.allclose(got, [1.0, 2.0])


def test_an_unknown_format_is_refused():
    with pytest.raises(ValueError, match="unsupported sample format"):
        bytes_to_mono(b"\0\0", "Float64", 1)


def test_empty_input_is_harmless():
    assert bytes_to_mono(b"", "Float", 1).size == 0
    assert bytes_to_mono(b"", "Int16", 2).size == 0


# --- device matching -----------------------------------------------------

DEVICES = ["K4 RX A", "K4 RX B", "Razer Kiyo Pro Ultra Analog Stereo"]


def test_none_means_the_system_default():
    assert match_device(DEVICES, None) is None
    assert match_device(DEVICES, "") is None


def test_exact_match_wins():
    assert match_device(DEVICES, "K4 RX A") == 0
    assert match_device(DEVICES, "K4 RX B") == 1


def test_a_unique_substring_matches_case_insensitively():
    assert match_device(DEVICES, "razer") == 2


def test_an_ambiguous_substring_does_not_guess():
    """"K4 RX" matches two devices; picking one would be a coin flip
    between the radio's main and sub receiver."""
    assert match_device(DEVICES, "K4 RX") is None


def test_a_missing_device_reports_no_match():
    assert match_device(DEVICES, "Elecraft K3") is None


def test_exact_match_beats_a_substring_of_another_name():
    """A device whose full name is a substring of another's must still
    match itself exactly rather than being called ambiguous."""
    devices = ["Mic", "Mic Array", "Mic Array (rear)"]
    assert match_device(devices, "Mic") == 0
