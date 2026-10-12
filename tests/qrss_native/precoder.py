"""sstvae.qrss.precoder -> sstvae_native.qrss.precoder (native/core/qrss/precoder).

The transmit side: the block layout, the WHT and the precoder itself.
`unprecode`, `block_mean` and `block_index` are the receiver's and are
substituted at stage rx (precoder_rx.py).
"""

STAGE = "tx"


def rows(q):
    p = q.precoder
    t = "sstvae.qrss.precoder"
    return {
        (t, "block_sizes"): p.block_sizes,
        (t, "wht"): p.wht,
        (t, "precode"): p.precode,
    }
