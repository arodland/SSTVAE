"""sstvae.qrss.acquire's A1 (carrier lines), A2 (preamble search) and
lead-in measurement -> sstvae_native.qrss.acquire_a12
(native/core/qrss/acquire_a12).

Each function is assigned over acquire's attribute of the same name
through a wrapper with the reference's exact signature (defaults
included), which turns the records the binding hands back -- "__type__"
dicts -- into the reference's own dataclasses: `CarrierStats`,
`CarrierCandidate`, `_A2Input`, `_A2Channel`, `PreambleHit`. Arguments go
the other way untouched; the binding reads the reference's dataclasses
field by field.

`CE_PRE`, `L_PRE` and `PREAMBLE_KINDS` are tables computed at import (the
preamble sequences and the matched filters), so they are rows of their
own: the reference's `PreambleKind` holding the C++'s sequence and taps,
read-only as the reference's are. `PreambleHit.to_detection` becomes the
C++'s too.

What the wrappers do not reach: acquire's own helpers called *inside* a
native function (`preamble_search`'s `_a2_input`, `_preamble_sums`,
`CubicSpline`, ...; `carrier_lines`'s `carrier_line_stats` when `stats` is
None) run as the C++, whatever acquire's attribute holds -- a test that
patches one of them alone and calls its caller would see no effect. None
does; the patched ones (`tests/test_qrss_lacquire.py`'s arbitration test)
are the callers themselves. The one exception is `lead_in_span`'s noise
floor, which is whatever `frontend.noise_psd` is when it is called (the
C++ itself while that is the native function; a spy, or the reference,
called with the reference's arguments otherwise), as in chan.py.
"""

STAGE = "rx"


def rows(q):
    import numpy as np

    from sstvae.qrss import acquire as A

    from . import _rx

    a = q.acquire_a12
    native_noise_psd = q.frontend.noise_psd

    def _noise_psd():
        from sstvae.qrss import frontend

        fn = frontend.noise_psd
        return None if fn is native_noise_psd else fn

    # --- records (rx_types.hpp's shared converters; _rx.to_ref builds the dataclasses) ------

    to_ref = _rx.to_ref

    def _read_only(x):
        x = np.array(x)
        x.setflags(write=False)
        return x

    def _kind(n):
        return A.PreambleKind(int(n.waveform), _read_only(n.seq), float(n.t_sym), int(n.chunk),
                              float(n.data_offset_hz), float(n.phase_offset), _read_only(n.mf_taps))

    CE, L = _kind(a.CE_PRE), _kind(a.L_PRE)
    kinds = {int(w): (CE if k is a.CE_PRE else L) for w, k in a.PREAMBLE_KINDS.items()}

    # --- A1 --------------------------------------------------------------------------------

    def carrier_line_stats(fe, t0_index, fs=A.FE_FS, *, span_s=A.A1_SPAN_S, drifts=A.A1_DRIFTS,
                           band_hz=A.BAND_HZ):
        return to_ref(a.carrier_line_stats(fe, t0_index, fs, span_s=span_s, drifts=drifts,
                                           band_hz=band_hz))

    def carrier_lines(fe, t0_index, fs=A.FE_FS, *, pfa=A.A1_PFA, max_cands=A.A1_MAX_CANDS,
                      min_sep_hz=A.A1_MIN_SEP_HZ, band_hz=A.BAND_HZ, stats=None):
        return _rx.to_ref(a.carrier_lines(fe, t0_index, fs, pfa=pfa, max_cands=max_cands,
                                          min_sep_hz=min_sep_hz, band_hz=band_hz, stats=stats))

    def hint_line(cs, f_hz, fs=A.FE_FS, *, pfa=A.A1_PFA, span_hz=A.HINT_SPAN_HZ):
        return _rx.to_ref(a.hint_line(cs, f_hz, fs, pfa=pfa, span_hz=span_hz))

    def _cell_line(cs, b, d, v):
        return _rx.to_ref(a._cell_line(cs, b, d, v))

    def pilot_lines(cs, lines, *, sep_hz=A.A2_ALIAS_HZ):
        return _rx.to_ref(a.pilot_lines(cs, lines, sep_hz=sep_hz))

    # --- A2 -------------------------------------------------------------------------------

    def preamble_channel(fe, t0_index, f_audio_hz, drift_hz_per_min=0.0, fs=A.FE_FS, *, kind=CE):
        return to_ref(a.preamble_channel(fe, t0_index, f_audio_hz, drift_hz_per_min, fs, kind=kind))

    def _a2_input(fe, t0_index, f_audio_hz, drift_hz_per_min=0.0, fs=A.FE_FS):
        return to_ref(a._a2_input(fe, t0_index, f_audio_hz, drift_hz_per_min, fs))

    def _a2_mf(inp, kind=CE):
        return to_ref(a._a2_mf(inp, kind))

    def _templates(df_hz, kind=CE):
        return a._templates(df_hz, kind)

    def preamble_lambda(chan, df_hz=A.A2_DF_HZ, tau_s=A.A2_TAU_S):
        return a.preamble_lambda(chan, df_hz, tau_s)

    def _preamble_sums(chan, df_hz=A.A2_DF_HZ, tau_s=A.A2_TAU_S):
        return a._preamble_sums(chan, df_hz, tau_s)

    def _fine_sums(spline, chan, tau, df, rate=0.0):
        return a._fine_sums(spline, chan, tau, df, rate)

    def preamble_search(fe, t0_index, cand, fs=A.FE_FS, *, pfa=A.A2_PFA, kind=CE, inp=None):
        return to_ref(a.preamble_search(fe, t0_index, cand, fs, pfa=pfa, kind=kind, inp=inp))

    def to_detection(self, duration_s=1800.0):
        return _rx.to_ref(a.to_detection(self, duration_s))

    # --- lead-in ----------------------------------------------------------------------------

    def lead_in_span(fe, t0_index, f_hz, drift_hz_per_min=0.0, tau0_s=0.0, fs=A.FE_FS, *,
                     z_min=A.LEAD_Z, t_sym=A.T_SYM):
        return a.lead_in_span(fe, t0_index, f_hz, drift_hz_per_min, tau0_s, fs, z_min=z_min,
                              t_sym=t_sym, _noise_psd=_noise_psd())

    # The native acquire() (tests/qrss_native/acquire.py) calls the C++ stage
    # directly while acquire's attribute is one of these, and the attribute
    # (a test's replacement) otherwise.
    for fn in (carrier_line_stats, carrier_lines, hint_line, _cell_line, pilot_lines, preamble_channel,
               _a2_input, _a2_mf, _templates, preamble_lambda, _preamble_sums, _fine_sums,
               preamble_search, to_detection, lead_in_span):
        fn.__qrss_native__ = True

    t = "sstvae.qrss.acquire"
    return {
        (t, "carrier_line_stats"): carrier_line_stats,
        (t, "carrier_lines"): carrier_lines,
        (t, "hint_line"): hint_line,
        (t, "_cell_line"): _cell_line,
        (t, "pilot_lines"): pilot_lines,
        (t, "_notch_taps"): a._notch_taps,
        (t, "_mf_taps"): a._mf_taps,
        (t, "_l_mf_taps"): a._l_mf_taps,
        (t, "CE_PRE"): CE,
        (t, "L_PRE"): L,
        (t, "PREAMBLE_KINDS"): kinds,
        (t, "preamble_channel"): preamble_channel,
        (t, "_a2_input"): _a2_input,
        (t, "_a2_mf"): _a2_mf,
        (t, "_chunks"): a._chunks,
        (t, "_norm"): a._norm,
        (t, "_templates"): _templates,
        (t, "preamble_lambda"): preamble_lambda,
        (t, "_preamble_sums"): _preamble_sums,
        (t, "_fine_sums"): _fine_sums,
        (t, "_coherent"): a._coherent,
        (t, "_phase_line"): a._phase_line,
        (t, "preamble_search"): preamble_search,
        (t + ".PreambleHit", "to_detection"): to_detection,
        (t, "lead_in_span"): lead_in_span,
    }
