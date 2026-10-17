"""sstvae.qrss.polar (encoder) -> sstvae_native.qrss.polar (native/core/qrss/polar).

The decoder's rows are in polar_rx.py (stage rx). INFO_SET is not
substituted: it is data, generated into qrss/qrss_config.hpp from the tuple.
"""

STAGE = "tx"


def rows(q):
    p = q.polar
    t = "sstvae.qrss.polar"
    return {
        (t, "polar_transform"): p.polar_transform,
        (t, "rate_match"): p.rate_match,
        (t, "polar_encode"): p.polar_encode,
    }
