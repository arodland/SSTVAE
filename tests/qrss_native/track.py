"""The tracker core of sstvae.qrss.track -> sstvae_native.qrss.track
(native/core/qrss/track).

Every function the tracker run uses is a row: derotation, the symbol
matched filter and its tables, the CE templates, the Kalman filter and
RTS smoother with the q search, `track` itself, `refine_timing`, Z_ref,
`erase` and `report`. The C++ reads a track by attribute and only what
the reference reads, so a TrackResult, an LTrackResult (ltrack calls
`_recenter`, `report`, `erase` and `z_ref_stat` on its own) and the
suites' fakes all work. ltrack's from-imports (`derotate`, `kalman_rts`,
`select_q`, `_smooth`, `_keying_tuple`) are found by the harness.

Results come back as the reference's objects (`_rx.to_ref`). `mf_table`
keeps one MfTable per period, as the reference's cache does (`T.mf_table()
is T.mf_table(T_SYM)`); its taps are a read-only view of the C++ table,
which `symbol_mf` recognises and uses without a copy. `track` hands back
the caller's own objects where the reference would: the capture, the
classes, the path and timing when the run kept them, and spec.

fit_timing, _segment_timing, _parabola, verify, verify_track,
_carrier_lines, refine_path, line_path and _flat_path are rows in
tests/qrss_native/track_timing.py; genie_track (a test aid) stays Python. `_layout_of` is a row
in Python: it only dispatches to `frame.layout` / `lframe.layout`, which
are whatever those modules hold (the C++ under their own rows).
"""

STAGE = "rx"


def rows(q):
    from . import _rx

    from sstvae.qrss.constants import T_SYM, WAVEFORM_CE

    t = q.track
    tables: dict = {}

    def mf_table(t_sym=T_SYM):
        key = float(t_sym)
        tb = tables.get(key)
        if tb is None:
            tb = tables[key] = _rx.to_ref(t.mf_table(t_sym))
        return tb

    def known_symbols(spec, hdr_bits=None):
        return _rx.to_ref(t.known_symbols(spec, hdr_bits))

    def make_classes(spec, hdr_bits=None, prior=None):
        return _rx.to_ref(t.make_classes(spec, hdr_bits, prior))

    def _recenter(tr):
        return _rx.to_ref(t._recenter(tr))

    def refine_timing(tr, iters=3, u_ref=None, P_ref=None):
        return _rx.to_ref(t.refine_timing(tr, iters, u_ref, P_ref))

    def report(tr, kappa=1.0, suspect=False):
        return _rx.to_ref(t.report(tr, kappa, suspect))

    def _layout_of(spec):
        from sstvae.qrss import frame, lframe

        return frame.layout(spec) if spec.waveform == WAVEFORM_CE else lframe.layout(spec)

    def track(chan, spec, det, classes=None, cw_known=None, prior=None, *, timing=None, freq=None,
              withhold_known=False, outer=2, refine=True, lead_in_s=None, z_d=None):
        d, same_freq, same_timing = t.track(
            chan, spec, det, classes, cw_known, prior, timing=timing, freq=freq,
            withhold_known=withhold_known, outer=outer, refine=refine, lead_in_s=lead_in_s,
            z_d=z_d)
        tr = _rx.to_ref(d)
        tr.spec = spec
        tr.chan = chan
        if classes is not None:
            tr.classes = classes
        tr.keying = t._keying_tuple(cw_known)[0]
        if same_freq:
            tr.freq = freq
        if same_timing:
            tr.timing = timing if timing is not None else det.timing
        return tr

    m = "sstvae.qrss.track"
    return {
        (m, "derotate"): t.derotate,
        (m, "_pulse_at"): t._pulse_at,
        (m, "_pulse_g0"): t._pulse_g0,
        (m, "mf_table"): mf_table,
        (m, "symbol_mf"): t.symbol_mf,
        (m, "psi_at"): t.psi_at,
        (m, "_lowpass"): t._lowpass,
        (m, "_mf_taps"): t._mf_taps,
        (m, "known_symbols"): known_symbols,
        (m, "make_classes"): make_classes,
        (m, "remove_spread"): t.remove_spread,
        (m, "templates"): t.templates,
        (m, "_kf_forward"): t._kf_forward,
        (m, "kalman_rts"): t.kalman_rts,
        (m, "select_q"): t.select_q,
        (m, "_window_unknown"): t._window_unknown,
        (m, "_smooth"): t._smooth,
        (m, "_measurements"): t._measurements,
        (m, "_pu_initial"): t._pu_initial,
        (m, "_recenter"): _recenter,
        (m, "_keying_tuple"): t._keying_tuple,
        (m, "_lead_supported"): t._lead_supported,
        (m, "track"): track,
        (m, "refine_timing"): refine_timing,
        (m, "_layout_of"): _layout_of,
        (m, "z_ref_stat"): t.z_ref_stat,
        (m, "erase"): t.erase,
        (m, "report"): report,
    }
