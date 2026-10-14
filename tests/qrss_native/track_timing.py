"""The rest of sstvae.qrss.track -> sstvae_native.qrss.track_timing
(native/core/qrss/track_timing).

The coarse timing fit (`fit_timing`, `_segment_timing`, `_parabola`),
gate V (`verify`, `verify_track`) and the frequency-path measurements
(`_carrier_lines`, `refine_path`, `line_path`, `_flat_path`). With
tests/qrss_native/track.py every function of track.py is a row but
`genie_track` (a test aid that stays Python) and `_timing_peaks` and
`_timings`, the grid search and the sort `fit_timing` and
`_fit_timing_tries` share (ltrack's L versions use `_timings` too), which
nothing else calls.

`_parabola` is acquire_stats' binding: track's is the same function as
acquire's, and that binding types its arithmetic as numpy does (float32
for three float32s, which ltrack's L pilot hands it). ltrack's
from-imports (`_parabola`, `_flat_path`) are found by the harness.

Results come back as the reference's objects (`_rx.to_ref`), and as the
very objects the reference would return where it returns its argument:
`refine_path` / `line_path` hand back `path` itself when nothing was
measured (test_qrss_lunits checks `is`), and `verify_track` puts back the
capture, spec and (with fit=False) the candidate's own timing.
"""

STAGE = "rx"


def rows(q):
    from . import _rx

    from sstvae.qrss.track import (PATH_FINE_RATIO, PATH_FINE_SHARE, PATH_SEG_S,
                                   PATH_SPAN_HZ, SEG_TIMING_S)

    t = q.track_timing

    def fit_timing(ch, ch_fs, path, spec, tmpl_fn=None, tau0_init=None, ppm_range=250.0, *,
                   t0_index, tau_range=None, z_d=None, n_best=1):
        return _rx.to_ref(t.fit_timing(ch, ch_fs, path, spec, tmpl_fn, tau0_init, ppm_range,
                                       t0_index=t0_index, tau_range=tau_range, z_d=z_d,
                                       n_best=n_best))

    def _fit_timing_tries(ch, ch_fs, path, spec, tmpl_fn=None, tau0_init=None, ppm_range=250.0, *,
                          t0_index, tau_range=None, z_d=None, n_best=1):
        best, tries = t._fit_timing_tries(ch, ch_fs, path, spec, tmpl_fn, tau0_init, ppm_range,
                                          t0_index=t0_index, tau_range=tau_range, z_d=z_d,
                                          n_best=n_best)
        return _rx.to_ref(best), _rx.to_ref(tries)

    def verify(ch, ch_fs, spec, cand, *, fit=True, tau_range=None, erase_s=()):
        return _rx.to_ref(t.verify(ch, ch_fs, spec, cand, fit=fit, tau_range=tau_range,
                                   erase_s=erase_s))

    def refine_path(chan, spec, path, seg_s=PATH_SEG_S, span_hz=PATH_SPAN_HZ, *, med_band=None,
                    ratio=10.0, fine_ratio=PATH_FINE_RATIO, fine_share=PATH_FINE_SHARE):
        d, unchanged = t.refine_path(chan, spec, path, seg_s, span_hz, med_band=med_band,
                                     ratio=ratio, fine_ratio=fine_ratio, fine_share=fine_share)
        return path if unchanged else _rx.to_ref(d)

    def line_path(chan, spec, path):
        d, unchanged = t.line_path(chan, spec, path)
        return path if unchanged else _rx.to_ref(d)

    def _flat_path(path):
        return _rx.to_ref(t._flat_path(path))

    def verify_track(chan, spec, cand, *, fit=True, tau_range=None, erase_s=()):
        d, same_timing = t.verify_track(chan, spec, cand, fit=fit, tau_range=tau_range,
                                        erase_s=erase_s)
        tr = _rx.to_ref(d)
        tr.spec = spec
        tr.chan = chan
        if same_timing:
            tr.timing = cand.timing
        return tr

    def _segment_timing(x, sig2, kn, t0abs, delta, sps, n_pos, seg_s=SEG_TIMING_S):
        return t._segment_timing(x, sig2, kn, t0abs, delta, sps, n_pos, seg_s)

    m = "sstvae.qrss.track"
    return {
        (m, "fit_timing"): fit_timing,
        (m, "_fit_timing_tries"): _fit_timing_tries,
        (m, "_segment_timing"): _segment_timing,
        (m, "_parabola"): q.acquire_stats._parabola,
        (m, "verify"): verify,
        (m, "_carrier_lines"): t._carrier_lines,
        (m, "refine_path"): refine_path,
        (m, "line_path"): line_path,
        (m, "_flat_path"): _flat_path,
        (m, "verify_track"): verify_track,
    }
