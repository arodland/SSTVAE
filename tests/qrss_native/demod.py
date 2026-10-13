"""sstvae.qrss.demod -> sstvae_native.qrss.demod (native/core/qrss/demod):
waveform CE's symbol estimates, kappa calibration, block estimators and
header LLRs, the whole module.

Every row is the native function itself: it reads the track by attribute,
only what the reference reads (native/bindings/module/qrss/demod_rx.hpp),
so the tracker's TrackResult and the suites' stubs both work, and
`extract` replaces `track.m` for a given m exactly as the reference does.
`ldemod`'s from-imports of `block_estimate` and `spread_llr` are found by
the harness. The pinned constants are not rows: other modules read them
as values the generated header already equals (checked in the parity
test).
"""

STAGE = "rx"


def rows(q):
    d = q.demod
    t = "sstvae.qrss.demod"
    return {
        (t, "symbol_estimates"): d.symbol_estimates,
        (t, "data_estimates"): d.data_estimates,
        (t, "spread_llr"): d.spread_llr,
        (t, "calibrate"): d.calibrate,
        (t, "block_estimate"): d.block_estimate,
        (t, "extract"): d.extract,
    }
