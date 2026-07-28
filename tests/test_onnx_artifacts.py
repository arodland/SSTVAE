"""The published ONNX artifacts must be the same codec as the `.pt`.

The codec *is* the on-air format, so this is not a "does it run" test.
If an artifact drifted from the checkpoint it was exported from, two
stations running different precisions would quietly disagree about what
a picture looks like, and nothing would raise.

Two independent checks, because they fail differently:

- **Provenance.** Every artifact stamps its source checkpoint's sha256
  into `metadata_props`. Comparing that against the actual `v1.pt`
  catches a stale or mismatched artifact exactly, with no tolerance to
  argue about — and catches it even if the numbers happen to look fine.
- **Numerics.** fp32 must match torch essentially exactly; the quantised
  variants get a bound tied to the channel noise, which is the only
  yardstick that means anything here (see `docs/onnx.md`).

Marked slow: six onnxruntime sessions plus torch is ~20 s, and the fast
suite is meant to stay a ~10 s check. Skipped entirely unless both the
checkpoint and the artifacts are already cached — **the tests must never
reach the network.**
"""

import hashlib

import numpy as np
import pytest

from sstvae import checkpoint
from sstvae.images import IMG_H, IMG_W
from sstvae.latents import latents_to_flat

pytestmark = pytest.mark.slow

torch = pytest.importorskip("torch", reason="torch is the reference side")
pytest.importorskip("onnxruntime")

# Latent RMS error against torch, on unit-RMS latents. The channel puts
# 0.367 RMS on them, so even the int8 bound is a "something broke" gate
# rather than a quality judgement.
#
# The int8 bound is tight enough to catch the export losing its
# sensitivity tuning: an untuned int8 encoder measures 1.88e-01 and
# would fail this, a tuned one measures 7.31e-02 and passes. That is
# deliberate — silently shipping the untuned encoder would cost every
# receiver ~1 dB of effective SNR instead of 0.17 dB.
LATENT_RMS_BOUND = {"fp32": 1e-4, "fp16": 5e-3, "int8": 0.12}


def _cached(filename: str) -> str | None:
    """Path to an already-downloaded Hub file, or None. Never fetches.

    Falls back the same way production does, across cached revisions --
    otherwise this suite silently skips itself whenever the repo gets a
    commit, which is precisely when it is most worth running.
    """
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(checkpoint.DEFAULT_REPO, filename,
                               local_files_only=True)
    except Exception:
        return checkpoint._any_cached_revision(filename)


@pytest.fixture(scope="module")
def pt_path():
    path = _cached(checkpoint.DEFAULT_FILE)
    if path is None:
        pytest.skip(f"{checkpoint.DEFAULT_FILE} not cached; run a decode once")
    return path


@pytest.fixture(scope="module")
def pt_sha(pt_path):
    h = hashlib.sha256()
    with open(pt_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def reference(pt_path):
    """torch latents and picture for one deterministic probe image."""
    from sstvae.codec import load_torch_model

    model = load_torch_model(pt_path)
    img = _probe()
    with torch.no_grad():
        z = model.encoder(torch.from_numpy(img)[None])
        picture = model.decoder(z, torch.ones_like(z))[0].numpy()
    return img, z.numpy(), picture


def _probe() -> np.ndarray:
    """A smooth, deterministic (3, H, W) image in [0,1].

    Low-frequency rather than white noise: the model is far off its
    training distribution on broadband noise, which makes quantisation
    error behave unlike anything a real picture would produce.
    """
    yy = np.linspace(0, 1, IMG_H)[:, None]
    xx = np.linspace(0, 1, IMG_W)[None, :]
    planes = [
        np.sin(6.283 * (1.5 * xx + 0.5 * yy)) + 0.5 * np.cos(6.283 * 2.0 * yy),
        np.sin(6.283 * (0.7 * xx - 1.1 * yy) + 1.0),
        np.cos(6.283 * (2.3 * yy + 0.3 * xx) + 2.0),
    ]
    img = np.stack(planes).astype(np.float32)
    return (img - img.min()) / (img.max() - img.min())


def _session(path: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def _artifact(part: str, precision: str) -> str:
    name = checkpoint.onnx_filename(part, precision)
    path = _cached(name)
    if path is None:
        pytest.skip(f"{name} not cached; run scripts/export_onnx.py or a decode")
    return path


@pytest.mark.parametrize("precision", checkpoint.PRECISIONS)
@pytest.mark.parametrize("part", ["encoder", "decoder"])
def test_artifact_records_the_checkpoint_it_came_from(part, precision, pt_sha):
    """Provenance, exactly -- the check that catches a stale artifact."""
    import onnx

    model = onnx.load(_artifact(part, precision), load_external_data=False)
    props = {p.key: p.value for p in model.metadata_props}

    assert props.get("sstvae.source_checkpoint") == checkpoint.DEFAULT_FILE
    assert props.get("sstvae.source_sha256") == pt_sha, (
        f"{part}/{precision} was exported from a different checkpoint than "
        f"the {checkpoint.DEFAULT_FILE} in the cache -- re-run "
        "scripts/export_onnx.py"
    )
    assert props.get("sstvae.part") == part
    assert props.get("sstvae.precision") == precision


@pytest.mark.parametrize("precision", checkpoint.PRECISIONS)
def test_encoder_matches_the_checkpoint(precision, reference):
    img, ref_z, _ = reference
    sess = _session(_artifact("encoder", precision))
    got = sess.run(None, {sess.get_inputs()[0].name: img[None]})[0]

    rms = float(np.sqrt(np.mean((got - ref_z) ** 2)))
    assert rms <= LATENT_RMS_BOUND[precision], (
        f"{precision} encoder is {rms:.3e} RMS from torch, over the "
        f"{LATENT_RMS_BOUND[precision]:.0e} bound"
    )


@pytest.mark.parametrize("precision", checkpoint.PRECISIONS)
def test_decoder_matches_the_checkpoint(precision, reference):
    _, ref_z, ref_picture = reference
    sess = _session(_artifact("decoder", precision))
    names = [i.name for i in sess.get_inputs()]
    got = sess.run(None, {names[0]: ref_z, names[1]: np.ones_like(ref_z)})[0][0]

    mse = float(np.mean((got - ref_picture) ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
    # Difference-PSNR has no natural scale and is strongly
    # image-dependent, so the int8 floor is an *integrity* gate ("the
    # graph is intact and roughly the right function"), not a quality
    # claim: a genuinely broken decoder lands in the single digits. The
    # decoder is deliberately left fully quantised (see
    # scripts/export_onnx.py's INT8_TUNE_PARTS), costing 0.074 dB of a
    # picture only its own operator sees, so this floor stays loose.
    floor = {"fp32": 80.0, "fp16": 55.0, "int8": 15.0}[precision]
    assert psnr >= floor, f"{precision} decoder only {psnr:.1f} dB from torch"


def test_the_codec_round_trips_through_the_published_artifacts(reference):
    """End to end through the real codec class, not raw sessions."""
    from sstvae.codec import load_codec

    img, ref_z, _ = reference
    _artifact("encoder", "fp16"), _artifact("decoder", "fp16")  # skip if absent

    codec = load_codec()
    flat = codec.encode(img)
    assert flat.shape == latents_to_flat(ref_z)[0].shape
    assert np.isclose(np.sqrt(np.mean(flat ** 2)), 1.0, atol=1e-3), (
        "latents must come off the encoder at unit RMS -- that is the "
        "on-air contract"
    )

    picture = codec.decode(flat, np.ones_like(flat))
    assert picture.size == (IMG_W, IMG_H)
    assert picture.mode == "RGB"
