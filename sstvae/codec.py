"""Loading the model and turning pictures into latents and back.

The encoder/decoder network *is* the codec, so every tool -- the CLI
scripts, the live listener, the GUI -- needs it. These used to live in
`sstvae_encode.py` / `sstvae_decode.py` at the top level, which meant
anything inside the package that wanted them had to import a *script*.
They live here now; the scripts re-export them so their command lines
and any existing imports are unchanged.

**The runtime backend is ONNX, and torch is for training.** `torch` is
336 MB against `onnxruntime`'s 27 MB and exists here solely to run two
convolutional passes; `docs/onnx.md` has the measurements, including the
result that matters -- fp32 ONNX and torch are the *same codec* to
~2e-06 on unit-RMS latents, about 105 dB below the channel noise. A
station running either decodes the other's transmissions exactly.

The torch backend is still available (`backend="torch"`) as the
reference implementation, and is what `scripts/export_onnx.py` verifies
against. It is not what a receiving station installs.

**Parts load lazily and independently.** Encoding touches only the
encoder; decoding and listening only the decoder. So a receive-only
station never downloads, loads, or pays memory for the encoder -- which
is also why `--model` can accept a single `.onnx` file and still be
unambiguous. See `checkpoint.resolve_onnx`.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from . import checkpoint
from .config import MODES
from .images import image_to_array
from .latents import flat_to_latents, latents_to_flat

MODEL_HELP = (
    "model to use: a directory of exported .onnx files, a single .onnx "
    "(its sibling part is found beside it), or a .pt checkpoint (needs "
    "torch, which the app extras do not install). Defaults to the "
    "published ONNX artifacts, downloaded and cached on first use"
)
PRECISION_HELP = (
    "ONNX precision when using the published artifacts; fp16 is the default "
    "and is measured identical to fp32 end to end (see docs/onnx.md)"
)


class OnnxCodec:
    """The shipping codec: onnxruntime, CPU, no torch anywhere."""

    backend = "onnx"

    def __init__(self, path: str | None = None,
                 precision: str = checkpoint.DEFAULT_PRECISION):
        self._path = path
        self._precision = precision
        self._sessions: dict[str, object] = {}
        self._sources: dict[str, str] = {}

    def _session(self, part: str):
        if part not in self._sessions:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            # Measured best of 1/4/24 on x86-64; 1 was ~5x worse. The
            # receive loop reconstructs once per 5-second poll, so this
            # is about not being gratuitously slow, not about latency.
            opts.intra_op_num_threads = 4
            opts.log_severity_level = 3
            resolved = checkpoint.resolve_onnx(part, self._path, self._precision)
            sess = ort.InferenceSession(resolved, opts,
                                        providers=["CPUExecutionProvider"])
            self._sessions[part] = sess
            self._check_same_checkpoint(part, resolved, sess)
        return self._sessions[part]

    def _check_same_checkpoint(self, part: str, resolved: str, sess) -> None:
        """Both parts must come from one training run.

        An encoder and decoder from different checkpoints are not a
        codec: they will run, produce a picture, and the picture will be
        garbage, with nothing anywhere reporting a problem. That is the
        worst failure mode available here, and it is entirely avoidable
        because `scripts/export_onnx.py` stamps each artifact with its
        source checkpoint's sha256.

        Precisions may differ freely -- fp16 encoder with an int8 decoder
        is fine, they are the same codec (docs/onnx.md). Only the source
        checkpoint has to agree.
        """
        meta = sess.get_modelmeta().custom_metadata_map or {}
        sha = meta.get("sstvae.source_sha256")
        if sha is None:
            return  # third-party or hand-rolled export; nothing to check
        self._sources[part] = sha
        mismatched = {p: s for p, s in self._sources.items() if s != sha}
        if mismatched:
            other, other_sha = next(iter(mismatched.items()))
            raise SystemExit(
                f"{part} and {other} come from different checkpoints "
                f"({sha[:12]} vs {other_sha[:12]}).\n"
                f"{resolved}\nis not paired with the {other} in use. Both "
                "halves must be exported from the same training run, or the "
                "decoded picture will be silently wrong."
            )

    def encode(self, image) -> np.ndarray:
        """PIL image or (3, H, W) array in [0,1] -> flat latents, float64.

        Returns mode C's full-length vector; callers truncate to their
        mode. Latents are unit RMS -- the on-air contract -- and that
        normalization happens inside the graph, not here.
        """
        arr = image_to_array(image) if isinstance(image, Image.Image) else image
        arr = np.ascontiguousarray(arr, dtype=np.float32)[None]
        sess = self._session("encoder")
        z = sess.run(None, {sess.get_inputs()[0].name: arr})[0]
        return latents_to_flat(z)[0].astype(np.float64)

    def decode(self, latents: np.ndarray, weights: np.ndarray) -> Image.Image:
        """Full-length (mode C sized) latent/weight vectors -> PIL image."""
        z = flat_to_latents(np.asarray(latents, dtype=np.float32)[None])
        w = flat_to_latents(np.asarray(weights, dtype=np.float32)[None])
        sess = self._session("decoder")
        names = [i.name for i in sess.get_inputs()]
        # Erased latents must be zeroed, not merely down-weighted.
        img = sess.run(None, {names[0]: z * (w > 0), names[1]: w})[0][0]
        arr = (img.transpose(1, 2, 0) * 255).round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)


class TorchCodec:
    """Reference backend. Training and `scripts/export_onnx.py` use this."""

    backend = "torch"

    def __init__(self, path: str | None = None):
        self.model = load_torch_model(path)

    def encode(self, image) -> np.ndarray:
        import torch

        arr = image_to_array(image) if isinstance(image, Image.Image) else image
        with torch.no_grad():
            z = self.model.encoder(torch.from_numpy(np.asarray(arr, np.float32))[None])
        return latents_to_flat(z.numpy())[0].astype(np.float64)

    def decode(self, latents: np.ndarray, weights: np.ndarray) -> Image.Image:
        import torch

        z = torch.from_numpy(flat_to_latents(np.asarray(latents, np.float32)[None]))
        w = torch.from_numpy(flat_to_latents(np.asarray(weights, np.float32)[None]))
        with torch.no_grad():
            img = self.model.decoder(z * (w > 0), w)[0]
        arr = (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        return Image.fromarray(arr)


def load_codec(path: str | None = None, *, precision: str | None = None,
               backend: str = "auto"):
    """The codec a tool should use.

    `backend="auto"` picks torch for a `.pt` and ONNX for everything
    else, which makes `--model something.pt` keep working exactly as it
    did while the default path needs no flag at all.
    """
    if backend == "auto":
        backend = "torch" if path is not None and Path(path).suffix == ".pt" \
            else "onnx"
    if backend == "torch":
        if precision is not None:
            raise ValueError("precision applies to the ONNX backend only")
        return TorchCodec(path)
    if backend == "onnx":
        return OnnxCodec(path, precision or checkpoint.DEFAULT_PRECISION)
    raise ValueError(f"unknown backend {backend!r}")


def load_torch_model(path: str | None = None):
    """The raw `SSTVAE` module. **Training and export only.**

    Always loads onto the CPU: one 640x480 pass is a few milliseconds
    against ~270 ms of NumPy DSP in the same operation, so a GPU buys
    nothing and would drag ROCm/CUDA initialization into short-lived CLI
    runs and into the GUI process.

    Works, and is checked against the ONNX path by
    `scripts/export_onnx.py` — but **torch is not installed by the app
    extras**, so this raises a pointed error rather than a bare
    ImportError on a normal user's machine.
    """
    try:
        import torch
    except ImportError as e:
        raise SystemExit(
            "a .pt checkpoint needs torch, which the cli/listen/gui extras no "
            "longer install — the codec runs on onnxruntime now "
            "(see docs/onnx.md).\n"
            "Either point --model at exported .onnx artifacts (or leave it "
            "unset for the published ones), or install torch:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from e

    from .models import SSTVAE

    ckpt = torch.load(checkpoint.resolve(path), map_location="cpu")
    model = SSTVAE(width=ckpt.get("width", 128))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def reconstruct(codec, latents: np.ndarray, weights: np.ndarray) -> Image.Image:
    """Full-length latent/weight vectors -> PIL image.

    Kept as a free function with this exact signature because
    `sstvae/rx/engine.py` calls it in two places and that file is
    load-bearing -- see CLAUDE.md.
    """
    return codec.decode(latents, weights)


def pad_to_full(vec: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Extend a mode A/B latent (or weight) vector to mode C's length.

    The modes are nested -- mode A's latents are a prefix of mode C's --
    so a shorter mode is just a full-length vector whose tail never
    arrived, which is exactly what weight 0 means to the decoder.
    """
    full = MODES["C"].n_latents
    out = np.full(full, fill)
    out[: len(vec)] = vec
    return out
