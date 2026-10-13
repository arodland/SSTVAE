"""sstvae.qrss.picture.codec_id_from_onnx -> sstvae_native.qrss.picture
(native/core/qrss/codec, the codec glue).

Bound only in a build with SSTVAE's codec (SSTVAE_BUILD_CODEC, which
brings onnxruntime): the glue library sstvae_qrss_codec exists only then,
and the module's `qrss.codec` submodule with it. Without the codec this
table is empty and the reference keeps its own function -- a build
configuration, not a partial build, so it is not the stale-row error.
With the codec, a missing `picture.codec_id_from_onnx` is that error.

The rest of `qrss.codec` (encode_picture, load_image, resolve_onnx, the
pathlib helpers) is qrss_encode.py's work, which has no single reference
function to stand in for; tests/test_qrss_native_codec.py compares it
directly, and tests/test_qrss_native_cli.py the CLI built on it.
"""

STAGE = "tx"


def rows(q):
    if not hasattr(q, "codec"):
        return {}
    return {
        ("sstvae.qrss.picture", "codec_id_from_onnx"): getattr(q.picture, "codec_id_from_onnx", None),
    }
