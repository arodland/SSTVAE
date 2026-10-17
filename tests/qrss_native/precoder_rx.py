"""sstvae.qrss.precoder, receive side -> sstvae_native.qrss.precoder.

`unprecode` undoes the precoder on received symbols; `block_index` and
`block_mean` are the demodulator's and EM's per-block bookkeeping.
"""

STAGE = "rx"


def rows(q):
    p = q.precoder
    t = "sstvae.qrss.precoder"
    return {
        (t, "unprecode"): p.unprecode,
        (t, "block_mean"): p.block_mean,
        (t, "block_index"): p.block_index,
    }
