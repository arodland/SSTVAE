"""The C++ `sstvae-decode` against the Python one, on the same WAV.

Phase 2's exit criterion, and a different question from
`test_native_parity.py`. That file compares modules; this one compares
the *program* -- WAV parsing, rate conversion, argument handling, the
order things are wired in. Every module could be right and the picture
still wrong, and the modem port already produced one bug of exactly
that shape (a compiled-in constant that a test's monkeypatch could not
reach).

Skipped when the binary has not been built. CI sets SSTVAE_REQUIRE_CODEC
after building it, which turns that skip into a failure -- the same
reasoning as the codec parity tests, and for the same reason: this is
the check most worth having and the one most able to disappear quietly.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.codec]

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "native" / "build" / "sstvae-decode"


def _require(reason):
    if os.environ.get("SSTVAE_REQUIRE_CODEC"):
        pytest.fail(f"SSTVAE_REQUIRE_CODEC is set but the CLI test cannot run: {reason}")
    pytest.skip(reason)


@pytest.fixture(scope="module")
def cli():
    if not BINARY.exists():
        _require(f"{BINARY} not built (cmake --build native/build)")
    return BINARY


@pytest.fixture(scope="module")
def model_dir():
    """The directory holding the cached fp16 parts."""
    pytest.importorskip("onnxruntime")
    from sstvae import checkpoint

    try:
        return str(Path(checkpoint.resolve_onnx("decoder", None, "fp16")).parent)
    except (Exception, SystemExit):
        _require("published ONNX artifacts are not cached")


@pytest.fixture(scope="module")
def transmission(tmp_path_factory):
    """A real mode A transmission through a noisy channel.

    Noise is the point. On a clean loopback the equalizer is doing
    almost nothing and every weight is 1, so the comparison would not
    exercise erasures, confidence weighting, or the drift tracker --
    the parts most likely to diverge between implementations.
    """
    from sstvae import wavio
    from sstvae.codec import load_codec
    from sstvae.config import MODES
    from sstvae.hfchannel import awgn
    from sstvae.modem import Modem

    d = tmp_path_factory.mktemp("tx")
    yy, xx = np.mgrid[0:480, 0:640].astype(np.float32)
    img = np.stack([
        0.5 + 0.5 * np.sin(xx / 37.0),
        0.5 + 0.5 * np.cos(yy / 29.0),
        ((xx.astype(int) ^ yy.astype(int)) % 256) / 255.0,
    ]).astype(np.float32)

    mode = MODES["A"]
    latents = load_codec(precision="fp16").encode(img)[: mode.n_latents]
    wave = Modem().modulate(latents, mode, callsign="KC2G")
    wavio.write_wav(str(d / "rx.wav"), awgn(wave, 12.0, seed=11))
    return d


def _python_decode(wav: Path, out: Path):
    """The reference decode -- always in a *subprocess*.

    This is not incidental. The slow suite runs under `--native`, which
    substitutes the C++ implementations into the reference modules by
    attribute assignment. An in-process `Modem().demodulate(...)` here
    would therefore be the C++ modem, and this file would be comparing
    C++ against C++ and passing while testing nothing at all -- the
    exact hazard the substitution mechanism is otherwise designed to
    avoid. A fresh interpreter has no substitutions applied.

    `sys.executable` rather than "python" for the same class of reason:
    under CI the two can differ, and the reference must come from the
    package under test.
    """
    subprocess.run(
        [sys.executable, str(ROOT / "sstvae_decode.py"), str(wav), str(out),
         "--precision", "fp16"],
        check=True, capture_output=True, cwd=ROOT)


# Blind decode has no CLI flag on the Python side, so the reference runs
# as a short script -- still a subprocess, for the reason above.
_BLIND_REFERENCE = """
import sys
from sstvae import wavio
from sstvae.codec import load_codec, pad_to_full, reconstruct
from sstvae.modem import Modem

r = Modem().demodulate_blind(wavio.read_wav(sys.argv[1]))
reconstruct(load_codec(precision="fp16"), pad_to_full(r.latents),
            pad_to_full(r.weights)).save(sys.argv[2])
print(r.beacon.frame_index if r.beacon else -1)
"""


def _python_blind_decode(wav: Path, out: Path) -> int:
    result = subprocess.run([sys.executable, "-c", _BLIND_REFERENCE, str(wav), str(out)],
                            check=True, capture_output=True, text=True, cwd=ROOT)
    return int(result.stdout.strip().splitlines()[-1])


def _pixels(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _assert_same_picture(a: Path, b: Path):
    x, y = _pixels(a), _pixels(b)
    assert x.shape == y.shape
    diff = np.abs(x.astype(int) - y.astype(int))
    assert not diff.any(), (
        f"{int((diff > 0).sum())} of {diff.size} subpixels differ, "
        f"max delta {int(diff.max())}"
    )


def test_decodes_the_same_picture(cli, model_dir, transmission, tmp_path):
    """The whole chain, byte for byte: sync, CFO, EQ, beacon, codec."""
    wav = transmission / "rx.wav"
    cpp, py = tmp_path / "cpp.png", tmp_path / "py.png"

    result = subprocess.run([str(cli), str(wav), str(cpp), "--model", model_dir],
                            check=True, capture_output=True, text=True)
    _python_decode(wav, py)
    _assert_same_picture(cpp, py)

    # The report the operator reads has to agree too -- a picture that
    # matched while the SNR or the frame count did not would mean one of
    # them was not being computed from what was decoded.
    assert "220/220 frames" in result.stdout
    assert "callsign 'KC2G'" in result.stdout


def test_decodes_a_44k_recording_identically(cli, model_dir, transmission, tmp_path):
    """Forces `dsp::resample_poly`, which nothing else here exercises.

    44.1 kHz is the ratio (160/441) that CLAUDE.md records as having
    cost 4.7 dB of SNR when resampling was done per-chunk -- while still
    syncing and reporting every frame, which is why it looked like a
    decoder bug.
    """
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    rate, data = wavfile.read(transmission / "rx.wav")
    up = resample_poly(data.astype(np.float64) / 32767.0, 441, 80)
    wav44 = tmp_path / "rx44.wav"
    wavfile.write(wav44, 44100, np.round(up * 32767).astype(np.int16))

    cpp, py = tmp_path / "cpp44.png", tmp_path / "py44.png"
    subprocess.run([str(cli), str(wav44), str(cpp), "--model", model_dir],
                   check=True, capture_output=True)
    _python_decode(wav44, py)
    _assert_same_picture(cpp, py)


def test_stereo_integer_wav_matches_the_mono_original(cli, model_dir, transmission,
                                                      tmp_path):
    """The scale-before-mixdown rule, which the reference once got wrong.

    Scaling after the mixdown returns samples in the +-32767 range, and
    the modem is scale-invariant enough to decode them anyway -- so the
    only way to catch it is to require the *same* picture, not a good
    one.
    """
    from scipy.io import wavfile

    rate, data = wavfile.read(transmission / "rx.wav")
    stereo = tmp_path / "rx_stereo.wav"
    wavfile.write(stereo, rate, np.stack([data, data], axis=1))

    mono_png, stereo_png = tmp_path / "mono.png", tmp_path / "stereo.png"
    subprocess.run([str(cli), str(transmission / "rx.wav"), str(mono_png),
                    "--model", model_dir], check=True, capture_output=True)
    subprocess.run([str(cli), str(stereo), str(stereo_png), "--model", model_dir],
                   check=True, capture_output=True)
    _assert_same_picture(mono_png, stereo_png)


def test_blind_decode_matches(cli, model_dir, transmission, tmp_path):
    """A recording that never contained the start of the transmission.

    Position comes from the beacon's absolute counter alone, so this
    checks the one path where the two implementations have to agree
    about *where they are* rather than merely what they received.
    """
    from scipy.io import wavfile

    rate, data = wavfile.read(transmission / "rx.wav")
    mid = tmp_path / "rx_mid.wav"
    wavfile.write(mid, rate, data[80000:200000])

    cpp = tmp_path / "cpp_blind.png"
    result = subprocess.run(
        [str(cli), str(mid), str(cpp), "--model", model_dir, "--blind"],
        check=True, capture_output=True, text=True)

    py = tmp_path / "py_blind.png"
    frame = _python_blind_decode(mid, py)
    _assert_same_picture(cpp, py)

    # Both must have located themselves, and at the same frame -- a
    # matching picture with no beacon would mean both had failed the
    # same way, which is not the same as both being right.
    assert frame >= 0, "the reference found no beacon; the fixture is not blind-decodable"
    assert f"frame {frame}," in result.stdout
