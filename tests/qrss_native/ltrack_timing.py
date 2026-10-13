"""The rest of sstvae.qrss.ltrack -> sstvae_native.qrss.ltrack_timing
(native/core/qrss/ltrack_timing).

L's coarse timing fit (`fit_timing_l`, `_segment_timing_l`, the mix's
phases `_lag_phase` and `_clock_phase`) and gate V (`refine_path_l`,
`verify_track_l`, `resid_ratio`, `verify_l`). With
tests/qrss_native/ltrack.py every function of ltrack.py is a row but
`genie_track_l` (a test aid that stays Python, over these rows).

Results come back as the reference's objects (`_rx.to_ref`), and as the
very objects the reference would return where it returns its argument:
`refine_path_l` hands back `path` itself when nothing was measured, and
`verify_track_l` puts back the capture, spec and (with fit=False) the
candidate's own timing. ltrack's from-imports of track's names
(`_parabola`, `_flat_path`, `derotate`, ...) are other files' rows, found
by the harness.
"""

STAGE = "rx"


def rows(q):
    from sstvae.qrss.ltrack import SEG_TIMING_S

    from . import _rx

    t = q.ltrack_timing

    def fit_timing_l(ch, ch_fs, path, spec, tmpl_fn=None, tau0_init=None, ppm_range=250.0, *,
                     t0_index, tau_range=None, z_d=None, n_best=1):
        return _rx.to_ref(t.fit_timing_l(ch, ch_fs, path, spec, tmpl_fn, tau0_init, ppm_range,
                                         t0_index=t0_index, tau_range=tau_range, z_d=z_d,
                                         n_best=n_best))

    def _segment_timing_l(x, sig2, kn, t0abs, delta, sps, n_pos, n_ref, seg_s=SEG_TIMING_S):
        return t._segment_timing_l(x, sig2, kn, t0abs, delta, sps, n_pos, n_ref, seg_s)

    def refine_path_l(chan, spec, path):
        d, unchanged = t.refine_path_l(chan, spec, path)
        return path if unchanged else _rx.to_ref(d)

    def verify_track_l(chan, spec, cand, *, fit=True, tau_range=None, erase_s=()):
        d, same_timing = t.verify_track_l(chan, spec, cand, fit=fit, tau_range=tau_range,
                                          erase_s=erase_s)
        tr = _rx.to_ref(d)
        tr.spec = spec
        tr.chan = chan
        if same_timing:
            tr.timing = cand.timing
        return tr

    def verify_l(ch, ch_fs, spec, cand, *, fit=True, tau_range=None, erase_s=()):
        return _rx.to_ref(t.verify_l(ch, ch_fs, spec, cand, fit=fit, tau_range=tau_range,
                                     erase_s=erase_s))

    m = "sstvae.qrss.ltrack"
    return {
        (m, "_lag_phase"): t._lag_phase,
        (m, "_clock_phase"): t._clock_phase,
        (m, "fit_timing_l"): fit_timing_l,
        (m, "_segment_timing_l"): _segment_timing_l,
        (m, "refine_path_l"): refine_path_l,
        (m, "verify_track_l"): verify_track_l,
        (m, "resid_ratio"): t.resid_ratio,
        (m, "verify_l"): verify_l,
    }
