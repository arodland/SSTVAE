"""sstvae.qrss.lwave, the receive side -> sstvae_native.qrss.lwave.

Only `to_data_centre` so far: it fell out of the transmit port (it is
mix_turns and a phasor). `matched_filter`, `PULSE_NORM` and
`loopback_stats` with its masks are not ported yet.
"""

STAGE = "rx"


def rows(q):
    return {
        ("sstvae.qrss.lwave", "to_data_centre"): q.lwave.to_data_centre,
    }
