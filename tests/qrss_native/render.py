"""sstvae.qrss.render's arithmetic -> sstvae_native.qrss.render
(native/core/qrss/render): g(W), W~ and the decoder's two planes.

`render`, `render_accumulator` and the model lookup stay Python (they are
the codec call); `render` reaches `decoder_planes` through the module, so
it runs on the native planes.
"""

STAGE = "rx"


def rows(q):
    r = q.render
    t = "sstvae.qrss.render"
    return {
        (t, "g_shrink"): r.g_shrink,
        (t, "reference_w"): r.reference_w,
        (t, "decoder_planes"): r.decoder_planes,
    }
