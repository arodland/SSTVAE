"""The waveform L tracker of sstvae.qrss.ltrack -> sstvae_native.qrss.ltrack
(native/core/qrss/ltrack).

Every piece the L tracker run uses is a row: the pilot filter and its
table, the data-band mix and matched filter, L's classes and templates,
the pilot's template, `_Model` (a class here whose methods are the C++
Model's, reading the model's arrays and its track when called, so code
that edits them -- or the reference's own `_Model` handed to `_fit_delta`
-- is read as the reference would read it), `_pu_local`, `_hold_ends`,
`_lead_supported_l`, `_fit_delta`, `track_l` itself, `refine_timing_l`,
`report_l` and `known_symbols_l`. ltrack's from-imports of track's names
are other files' rows, found by the harness: `derotate`, `kalman_rts`,
`select_q`, `_smooth` and `_keying_tuple` are track.py's, `_parabola` and
`_flat_path` track_timing.py's, `PathFn` and `freq_path_spline`
trackbase.py's.

Results come back as the reference's objects (`_rx.to_ref`). `pilot_table`
keeps one MfTable, as the reference's cache does; its taps are a read-only
view of the C++ table. `track_l` hands back the caller's own objects where
the reference would: the capture, the classes, the keying array, the path,
timing and z_d when the run kept them, and spec. `_q_near` slices the
caller's grid, as the reference does (its elements, not copies).

Not rows here (the coarse timing and gate V, another lane): `_lag_phase`,
`_clock_phase`, `fit_timing_l`, `_segment_timing_l`, `refine_path_l`,
`verify_track_l`, `resid_ratio`, `verify_l`; `genie_track_l` (a test aid)
stays Python, over these rows.
"""

STAGE = "rx"


def rows(q):
    from . import _rx

    lt = q.ltrack
    tables: list = []

    def pilot_table():
        if not tables:
            tables.append(_rx.to_ref(lt.pilot_table()))
        return tables[0]

    def make_classes_l(spec, hdr_bits=None, prior=None):
        return _rx.to_ref(lt.make_classes_l(spec, hdr_bits, prior))

    def known_symbols_l(spec, hdr_bits=None):
        return _rx.to_ref(lt.known_symbols_l(spec, hdr_bits))

    def _q_near(q_, grid):
        lo, hi = lt._q_near_span(q_, grid)
        return tuple(grid[lo:hi])

    class _Model:
        """sstvae.qrss.ltrack._Model on native/core/qrss/ltrack's Model."""

        def __init__(self, tr, spec, classes, withhold_known, keying, lead, n_lead):
            self.tr, self.spec, self.classes = tr, spec, classes
            self.keying, self.lead = keying, lead
            d = lt._model_init(tr, spec, classes, withhold_known, keying, lead, n_lead)
            self.c_all = d["c_all"]
            self.c, self.nu, self.gain, self.dist, self.is_sym = d["c"], d["nu"], d["gain"], d["dist"], d["is_sym"]
            self.is_known = d["is_known"]
            self.meas_d = d["meas_d"]
            self.need = d["need"]
            self.meas_p = d["meas_p"]
            self.measured = d["measured"]
            self.k_pl = d["k_pl"]
            tr.meas, tr.meas_pl = self.meas_d, self.meas_p

        def streams(self, Pu_loc, delta=1.0 + 0.0j, data=True, pilot=True, data_mask=None):
            return lt._model_streams(self, Pu_loc, delta, data, pilot, data_mask)

        def fused(self, Pu_loc, delta=1.0 + 0.0j, data=True, pilot=True, data_mask=None):
            return lt._model_fused(self, Pu_loc, delta, data, pilot, data_mask)

        def pu_initial(self) -> float:
            return lt._model_pu_initial(self)

    def track_l(chan, spec, det, classes=None, cw_known=None, prior=None, *, timing=None, freq=None,
                withhold_known=False, outer=2, refine=True, lead_in_s=None, z_d=None):
        d, keying, same_freq, same_timing, same_zd = lt.track_l(
            chan, spec, det, classes, cw_known, prior, timing=timing, freq=freq,
            withhold_known=withhold_known, outer=outer, refine=refine, lead_in_s=lead_in_s,
            z_d=z_d)
        tr = _rx.to_ref(d)
        tr.spec = spec
        tr.chan = chan
        if classes is not None:
            tr.classes = classes
        tr.keying = keying
        if same_freq:
            tr.freq = freq
        if same_timing:
            tr.timing = timing if timing is not None else det.timing
        if same_zd:
            tr.z_d = z_d
        return tr

    def refine_timing_l(tr, iters=3, u_ref=None, P_ref=None):
        return _rx.to_ref(lt.refine_timing_l(tr, iters, u_ref, P_ref))

    def report_l(tr, kappa=1.0, suspect=False):
        return _rx.to_ref(lt.report_l(tr, kappa, suspect))

    m = "sstvae.qrss.ltrack"
    return {
        (m, "_l_only"): lt._l_only,
        (m, "pilot_table"): pilot_table,
        (m, "_pilot_taps"): lt._pilot_taps,
        (m, "pilot_filter"): lt.pilot_filter,
        (m, "data_mix"): lt.data_mix,
        (m, "data_mf"): lt.data_mf,
        (m, "make_classes_l"): make_classes_l,
        (m, "_distortion"): lt._distortion,
        (m, "data_templates"): lt.data_templates,
        (m, "_pilot_gain"): lt._pilot_gain,
        (m, "pilot_wave"): lt.pilot_wave,
        (m, "_q_near"): _q_near,
        (m, "_Model"): _Model,
        (m, "_pu_local"): lt._pu_local,
        (m, "_hold_ends"): lt._hold_ends,
        (m, "_lead_supported_l"): lt._lead_supported_l,
        (m, "_fit_delta"): lt._fit_delta,
        (m, "track_l"): track_l,
        (m, "refine_timing_l"): refine_timing_l,
        (m, "report_l"): report_l,
        (m, "known_symbols_l"): known_symbols_l,
    }
