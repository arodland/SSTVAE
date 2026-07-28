"""Backend selection, and that both backends are the same codec.

The runtime is ONNX and torch is the reference implementation, but a
`.pt` checkpoint is still a documented `--model` input — it is how you
try a freshly trained checkpoint before exporting it. torch is no longer
installed by the app extras, though, so this path is easy to break
without noticing: nothing on a normal user's machine exercises it.

The numeric agreement between the two backends is checked at export time
by `scripts/export_onnx.py`; what is checked here is the plumbing around
them — which backend a given `--model` selects, and that each produces a
usable picture with unit-RMS latents.
"""

import numpy as np
import pytest

from sstvae import checkpoint
from sstvae.codec import OnnxCodec, TorchCodec, load_codec
from sstvae.images import IMG_H, IMG_W


def _probe() -> np.ndarray:
    yy = np.linspace(0, 1, IMG_H)[:, None]
    xx = np.linspace(0, 1, IMG_W)[None, :]
    img = np.stack([
        np.sin(6.283 * (1.5 * xx + 0.5 * yy)),
        np.sin(6.283 * (0.7 * xx - 1.1 * yy) + 1.0),
        np.cos(6.283 * (2.3 * yy + 0.3 * xx) + 2.0),
    ]).astype(np.float32)
    return (img - img.min()) / (img.max() - img.min())


def _cached(filename: str) -> str | None:
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(checkpoint.DEFAULT_REPO, filename,
                               local_files_only=True)
    except Exception:
        return checkpoint._any_cached_revision(filename)


# --- backend selection: no model files needed, so these always run ---------

def test_auto_backend_follows_the_suffix(tmp_path, monkeypatch):
    """`.pt` means torch, everything else means ONNX.

    Selection looks at the suffix only and must not require the file to
    exist -- reporting a missing file is the loader's job, and keeping
    it that way is what lets this test run without any artifacts.
    """
    from sstvae import codec

    chosen: list[str] = []
    monkeypatch.setattr(codec, "TorchCodec",
                        lambda path: chosen.append("torch"))
    monkeypatch.setattr(codec, "OnnxCodec",
                        lambda path, precision: chosen.append("onnx"))

    codec.load_codec(str(tmp_path / "somewhere" / "my.pt"))
    codec.load_codec(str(tmp_path / "a-directory"))
    codec.load_codec(str(tmp_path / "v1-encoder-int8.onnx"))
    codec.load_codec(None)

    assert chosen == ["torch", "onnx", "onnx", "onnx"]


def test_an_explicit_backend_overrides_the_suffix(tmp_path, monkeypatch):
    from sstvae import codec

    chosen: list[str] = []
    monkeypatch.setattr(codec, "OnnxCodec",
                        lambda path, precision: chosen.append("onnx"))
    codec.load_codec(str(tmp_path / "my.pt"), backend="onnx")
    assert chosen == ["onnx"]


def test_precision_is_rejected_for_the_torch_backend(tmp_path):
    """Better than silently ignoring it: a .pt has no precision."""
    with pytest.raises(ValueError, match="ONNX backend only"):
        load_codec(str(tmp_path / "my.pt"), precision="fp16")


def test_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="unknown backend"):
        load_codec(None, backend="tensorflow")


# --- round trips, which need the real artifacts ----------------------------

@pytest.mark.slow
def test_onnx_backend_round_trips():
    if _cached(checkpoint.onnx_filename("encoder")) is None:
        pytest.skip("published ONNX artifacts not cached")
    codec = load_codec()
    assert codec.backend == "onnx"

    flat = codec.encode(_probe())
    assert np.isclose(np.sqrt(np.mean(flat ** 2)), 1.0, atol=1e-3), (
        "latents must leave the encoder at unit RMS -- the on-air contract"
    )
    picture = codec.decode(flat, np.ones_like(flat))
    assert picture.size == (IMG_W, IMG_H) and picture.mode == "RGB"


@pytest.mark.slow
def test_torch_backend_round_trips():
    """The `.pt` path still works, on a machine that has torch."""
    pytest.importorskip("torch", reason="torch is not an app-extra dependency")
    pt = _cached(checkpoint.DEFAULT_FILE)
    if pt is None:
        pytest.skip(f"{checkpoint.DEFAULT_FILE} not cached")

    codec = load_codec(pt)
    assert codec.backend == "torch"

    flat = codec.encode(_probe())
    assert np.isclose(np.sqrt(np.mean(flat ** 2)), 1.0, atol=1e-3)
    picture = codec.decode(flat, np.ones_like(flat))
    assert picture.size == (IMG_W, IMG_H) and picture.mode == "RGB"


@pytest.mark.slow
def test_the_two_backends_agree():
    """Not a tolerance study -- that lives in scripts/export_onnx.py.

    This only asserts they are recognisably the same codec, so a wiring
    mistake (transposed latents, wrong normalisation, swapped parts)
    cannot pass.
    """
    pytest.importorskip("torch")
    pt = _cached(checkpoint.DEFAULT_FILE)
    if pt is None or _cached(checkpoint.onnx_filename("encoder")) is None:
        pytest.skip("need both the checkpoint and the ONNX artifacts cached")

    probe = _probe()
    a = load_codec(pt).encode(probe)
    b = load_codec().encode(probe)
    rms = float(np.sqrt(np.mean((a - b) ** 2)))
    assert rms < 5e-3, f"backends disagree by {rms:.3e} RMS on unit-RMS latents"
