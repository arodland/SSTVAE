"""sstvae.qrss.sequences -> sstvae_native.qrss.sequences (native/core/qrss/sequences)."""

STAGE = "tx"


def rows(q):
    s = q.sequences
    t = "sstvae.qrss.sequences"
    return {
        (t, "sha_bits"): s.sha_bits,
        (t, "pm1"): s.pm1,
        (t, "qpsk"): s.qpsk,
        (t, "preamble_ce"): s.preamble_ce,
        (t, "references_ce"): s.references_ce,
        (t, "preamble_l"): s.preamble_l,
        (t, "references_l"): s.references_l,
        (t, "spread_whitener"): s.spread_whitener,
        (t, "scrambler"): s.scrambler,
    }
