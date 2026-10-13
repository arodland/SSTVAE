"""sstvae.qrss.ce, the receive side -> sstvae_native.qrss.ce_rx
(native/core/qrss/ce_rx): the mean waveform (EM's and the tracker's
template) and the exact symbol matched filter (`track.templates`).

`symbol_classes` and the loopback diagnostics (`loopback_mf`,
`loopback_stats`) are not ported -- nothing in the receiver calls them --
and stay Python, running on the native matched filter.
"""

STAGE = "rx"


def rows(q):
    c = q.ce_rx
    t = "sstvae.qrss.ce"
    return {
        (t, "mean_template"): c.mean_template,
        (t, "_mean_template_grid"): c._mean_template_grid,
        (t, "matched_filter"): c.matched_filter,
    }
