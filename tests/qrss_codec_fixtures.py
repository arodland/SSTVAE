"""Inputs for the codec glue's parity tests (tests/test_qrss_native_codec.py,
tests/test_qrss_native_cli.py). Not a test module.

- `tiny_onnx(sha)`: a serialized ONNX model -- y = Identity(x), float[1],
  opset 13, a few hundred bytes -- carrying `sstvae.source_sha256` = sha in
  its metadata (none for None). onnxruntime loads it, so the codec ID can be
  read from any metadata a case needs without the published artifacts.
  Written out field by field (protobuf varints and length-delimited
  fields), as the `onnx` package is not a dependency.
- `scene(h, w, seed)`: a deterministic uint8 RGB picture with edges,
  gradients and texture, saved as PNG where a test needs an image file (a
  640x480 PNG decodes identically in PIL and the native image layer).
- `v5_dir()`: the directory holding the v5 encoder and decoder at fp16 and
  fp32, or a codec skip (qrss_helpers.codec_skip).
"""

from pathlib import Path

import numpy as np

from qrss_helpers import codec_skip, model_dir


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | 0x80 if n else b)
        if not n:
            return bytes(out)


def _field(num: int, payload) -> bytes:
    if isinstance(payload, int):
        return _varint(num << 3) + _varint(payload)
    if isinstance(payload, str):
        payload = payload.encode()
    return _varint(num << 3 | 2) + _varint(len(payload)) + payload


def tiny_onnx(sha: str | None = None) -> bytes:
    dim = _field(1, _field(1, 1))                        # TensorShapeProto.dim { dim_value: 1 }
    ttype = _field(1, _field(1, 1) + _field(2, dim))     # TypeProto.tensor_type { FLOAT, shape }

    def value_info(name):
        return _field(1, name) + _field(2, ttype)

    node = _field(1, "x") + _field(2, "y") + _field(4, "Identity")
    graph = _field(1, node) + _field(2, "g") + _field(11, value_info("x")) + _field(12, value_info("y"))
    model = _field(1, 7) + _field(7, graph) + _field(8, _field(2, 13))   # ir_version 7, opset 13
    if sha is not None:
        model += _field(14, _field(1, "sstvae.source_sha256") + _field(2, sha))
    return model


def scene(h: int = 480, w: int = 640, seed: int = 5) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    rng = np.random.default_rng(seed)
    img = np.stack([
        0.5 + 0.4 * np.sin(x / 37.0) * np.cos(y / 23.0),
        y / h,
        0.5 + 0.5 * ((x - w / 2) ** 2 + (y - h / 2) ** 2 < (h / 3) ** 2),
    ], axis=-1)
    img += 0.08 * rng.standard_normal(img.shape)
    return np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)


def v5_dir() -> Path:
    d = model_dir()
    missing = [f"{part}-{prec}" for part in ("encoder", "decoder") for prec in ("fp16", "fp32")
               if d is None or not list(Path(d).glob(f"*-{part}-{prec}.onnx"))]
    if missing:
        codec_skip(f"no model directory with the v5 {', '.join(missing)}")
    return Path(d)
