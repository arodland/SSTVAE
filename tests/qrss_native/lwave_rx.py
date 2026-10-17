"""sstvae.qrss.lwave, the receive side -> sstvae_native.qrss.lwave and
sstvae_native.qrss.lwave_rxf.

`to_data_centre` fell out of the transmit port (it is mix_turns and a
phasor) and lives in the transmit module; `matched_filter` and
`PULSE_NORM` (which acquire and track read at call time) are lwave_rxf's,
a module of its own so the transmit bindings stay untouched.
`loopback_stats` and the masks behind it are diagnostics nothing in the
receiver calls, and are not ported.
"""

STAGE = "rx"


def rows(q):
    t = "sstvae.qrss.lwave"
    return {
        (t, "to_data_centre"): q.lwave.to_data_centre,
        (t, "matched_filter"): q.lwave_rxf.matched_filter,
        (t, "PULSE_NORM"): q.lwave_rxf.PULSE_NORM,
    }
