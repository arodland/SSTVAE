"""sstvae.qrss.ldemod -> sstvae_native.qrss.ldemod (native/core/qrss/ldemod):
waveform L's symbol estimates, kappa calibration and header LLRs, the
whole module.

Every row is the native function itself, reading the track as the
reference does (GenieTrack-style stubs included: only spec, pos, m, u, P,
N, psi, and getattr for meas and classes.hdr_bits). `_hdr_bits`,
`_measured` and `_l_only` are folded into their callers; nothing outside
the module calls them. `block_estimate` and `spread_llr` come from demod
(demod.py's rows).
"""

STAGE = "rx"


def rows(q):
    d = q.ldemod
    t = "sstvae.qrss.ldemod"
    return {
        (t, "d_pass_x"): d.d_pass_x,
        (t, "symbol_estimates_l"): d.symbol_estimates_l,
        (t, "data_estimates_l"): d.data_estimates_l,
        (t, "header_estimates_l"): d.header_estimates_l,
        (t, "_combine"): d._combine,
        (t, "kappa_estimates_l"): d.kappa_estimates_l,
        (t, "calibrate_l"): d.calibrate_l,
        (t, "extract_l"): d.extract_l,
    }
