"""The tracker's shared vocabulary in sstvae.qrss.track ->
sstvae_native.qrss.trackbase (native/core/qrss/trackbase).

PathFn is replaced by `_rx.NativePathFn`, a subclass of the reference's
class whose fit and evaluation are the C++'s (so `isinstance(fn,
PathFn)` still holds and the from-import sites -- ltrack, acquire, em,
... -- get it too). `q_of_spread`, `spread_of_q` and the timing maps are
the native functions; `Q_GRID` is a value computed at import, so it is
not replaced (trackbase's own Q_GRID is checked against it in the
parity test).
"""

STAGE = "rx"


def rows(q):
    from . import _rx

    tb = q.trackbase

    def freq_path_spline(path, knots_s=tb.KNOTS_S):
        return _rx.NativePathFn.from_native(tb.freq_path_spline(path, knots_s))

    t = "sstvae.qrss.track"
    return {
        (t, "q_of_spread"): tb.q_of_spread,
        (t, "spread_of_q"): tb.spread_of_q,
        (t, "rx_index"): tb.rx_index,
        (t, "_tau_of_rx"): tb._tau_of_rx,
        (t, "PathFn"): _rx.NativePathFn,
        (t, "freq_path_spline"): freq_path_spline,
    }
