"""sstvae.qrss.polar (CA-SCL decoder) -> sstvae_native.qrss.polar.

`polar_decode(accept=None)` is the header's own check in C++ (header::unpack
with the default versions), which is what the reference's default does
through `header.unpack`; a Python `accept` is called back per list path.
"""

STAGE = "rx"


def rows(q):
    p = q.polar
    t = "sstvae.qrss.polar"
    return {
        (t, "fold"): p.fold,
        (t, "decode_list"): p.decode_list,
        (t, "polar_decode"): p.polar_decode,
    }
