"""Shared by the receive-side rows (tests/qrss_native/*_rx.py, types.py,
trackbase.py) and the receive-side parity tests: the native records back
as the reference's dataclasses, and the native-backed PathFn.

Not a row module (the leading underscore keeps `apply` from loading it).

The C++ hands records to Python as plain dicts with a "__type__" key
(native/bindings/module/qrss/rx_types.hpp); `to_ref` turns such a tree
into the reference's own classes -- `types.Timing`, `types.PassResult`,
`header.HeaderFields`, `track.ChanCapture`, `track.TrackResult`,
`ltrack.LTrackResult`, the acquisition's `acquire.CarrierStats`,
`PreambleKind`, `PreambleHit`, `_A2Input`, `_A2Channel`, `TbdHit` and
`Acquisition`, ... -- so a substituted function returns exactly what the
reference's callers expect. A native FrameSpec becomes the
reference's spec (the module's own preset object when equal to one), and
a native PathFn a `NativePathFn`.
"""

from __future__ import annotations

import numpy as np


def _native():
    import sstvae_native

    return sstvae_native.qrss


# --- PathFn -----------------------------------------------------------------------------------

def _reference_pathfn():
    """The reference's PathFn class, even once a row has replaced the
    module attribute with NativePathFn (its base)."""
    from sstvae.qrss import track

    for cls in track.PathFn.__mro__:
        if cls.__module__ == "sstvae.qrss.track" and cls.__name__ == "PathFn":
            return cls
    raise LookupError("sstvae.qrss.track.PathFn not found")


def _knots_s():
    from sstvae.qrss import track

    return track.KNOTS_S


class NativePathFn(_reference_pathfn()):
    """`track.PathFn` fitted and evaluated by the C++
    (`sstvae_native.qrss.trackbase.PathFn`), an instance of the reference
    class. `lo`, `hi`, `_line` and `_spl` are the native fit's, so code
    that reads the reference's attributes sees them; `_native` is the
    native object, which `rx_types.hpp` takes as it is (no refit)."""

    def __init__(self, t, f, w=None, knots_s: float | None = None):
        tb = _native().trackbase
        self._adopt(tb.PathFn(t, f, w, _knots_s() if knots_s is None else knots_s))

    @classmethod
    def from_native(cls, n) -> NativePathFn:
        obj = cls.__new__(cls)
        obj._adopt(n)
        return obj

    def _adopt(self, n) -> None:
        from scipy import interpolate

        self._native = n
        self.lo, self.hi = float(n.lo), float(n.hi)
        self._line = np.array(n.line)
        spl = n.spline
        self._spl = (None if spl is None else
                     interpolate.BSpline.construct_fast(np.array(spl[0]), np.array(spl[1]), int(spl[2])))

    def _inside(self, t):
        if self._spl is None:
            return _native().math.polyval(self._line, t)
        return _native().math.bspline_eval(self._spl.t, self._spl.c, self._spl.k, t)

    def __call__(self, t):
        return self._native(t)

    def __reduce__(self):
        return (NativePathFn.from_native, (self._native,))


def path_fn(n):
    """A native PathFn (or None) as a NativePathFn."""
    if n is None or isinstance(n, NativePathFn):
        return n
    return NativePathFn.from_native(n)


# --- frame specs ------------------------------------------------------------------------------

def pyspec(n):
    """The reference spec for a native FrameSpec: the module's own preset
    object when equal to one, else a new one. A reference spec is returned
    as it is."""
    from sstvae.qrss import frame as F
    from sstvae.qrss import lframe as LF

    if isinstance(n, (F.FrameSpec, LF.LFrameSpec)):
        return n
    cls = LF.LFrameSpec if n.waveform == LF.WAVEFORM_L else F.FrameSpec
    s = cls(n.name, n.n_pre, n.n_hdr, n.n_data, n.cw_after, n.hdr_rho)
    for table in (F.PRESETS, F.LEGACY, F.PROTOTYPES, LF.PRESETS):
        for v in table.values():
            if v == s:
                return v
    return s


# --- records ----------------------------------------------------------------------------------

def _fields(d):
    return {k: v for k, v in d.items() if k != "__type__"}


def _types():
    from sstvae.qrss import types

    return types


def _timing(d):
    return _types().Timing(**_fields(d))


def _freq_path(d):
    return _types().FreqPath(**_fields(d))


def _detection(d):
    f = _fields(d)
    f["path"] = to_ref(f["path"])
    f["timing"] = to_ref(f["timing"])
    return _types().Detection(**f)


def _header(d):
    from sstvae.qrss.header import HeaderFields

    return HeaderFields(**_fields(d))


def _pass_result(d):
    f = _fields(d)
    for k in ("timing", "report", "header", "cw"):
        f[k] = to_ref(f[k])
    return _types().PassResult(**f)


def _chan(d):
    from sstvae.qrss import track

    f = _fields(d)
    f["exclude_hz"] = tuple(f["exclude_hz"])
    return track.ChanCapture(**f)


def _mf_table(d):
    from sstvae.qrss import track

    return track.MfTable(**_fields(d))


def _known(d):
    from sstvae.qrss import track

    return track._Known(**_fields(d))


def _classes(d):
    from sstvae.qrss import track

    return track.Classes(**_fields(d))


def _lclasses(d):
    from sstvae.qrss import ltrack

    return ltrack.LClasses(**_fields(d))


def _track_result(d):
    from sstvae.qrss import ltrack, track

    f = _fields(d)
    f["spec"] = pyspec(f["spec"])
    f["chan"] = to_ref(f["chan"])
    f["timing"] = to_ref(f["timing"])
    f["freq"] = path_fn(f["freq"])
    f["classes"] = to_ref(f["classes"])
    f["report"] = to_ref(f["report"])
    cls = ltrack.LTrackResult if d["__type__"] == "LTrackResult" else track.TrackResult
    return cls(**f)


def _simple(name):
    return lambda d: getattr(_types(), name)(**_fields(d))


# --- the acquisition's records (acquire.py) ---------------------------------------------------

def _acquire():
    from sstvae.qrss import acquire

    return acquire


def _acquire_simple(name):
    return lambda d: getattr(_acquire(), name)(**_fields(d))


def _a2_channel(d):
    f = _fields(d)
    f["kind"] = to_ref(f["kind"])
    return _acquire()._A2Channel(**f)


def _tbd_hit(d):
    f = _fields(d)
    f["path"] = to_ref(f["path"])
    f["l_path"] = to_ref(f["l_path"])
    return _acquire().TbdHit(**f)


def _acquisition(d):
    f = {k: to_ref(v) for k, v in _fields(d).items()}
    return _acquire().Acquisition(**f)


_CONVERT = {
    "Timing": _timing,
    "FreqPath": _freq_path,
    "CarrierCandidate": _simple("CarrierCandidate"),
    "Detection": _detection,
    "TrackReport": _simple("TrackReport"),
    "CwIdResult": _simple("CwIdResult"),
    "EmPrior": _simple("EmPrior"),
    "HeaderFields": _header,
    "PassResult": _pass_result,
    "ChanCapture": _chan,
    "MfTable": _mf_table,
    "Known": _known,
    "Classes": _classes,
    "LClasses": _lclasses,
    "TrackResult": _track_result,
    "LTrackResult": _track_result,
    "CarrierStats": _acquire_simple("CarrierStats"),
    "PreambleKind": _acquire_simple("PreambleKind"),
    "PreambleHit": _acquire_simple("PreambleHit"),
    "_A2Input": _acquire_simple("_A2Input"),
    "_A2Channel": _a2_channel,
    "TbdHit": _tbd_hit,
    "Acquisition": _acquisition,
}


def to_ref(x):
    """A native record (a "__type__" dict, a FrameSpec, a PathFn; lists and
    tuples of them) as the reference's objects; anything else as it is."""
    if isinstance(x, dict) and "__type__" in x:
        return _CONVERT[x["__type__"]](x)
    if isinstance(x, list):
        return [to_ref(v) for v in x]
    if isinstance(x, tuple):
        return tuple(to_ref(v) for v in x)
    q = _native()
    if isinstance(x, q.trackbase.PathFn):
        return path_fn(x)
    if isinstance(x, q.frame.FrameSpec):
        return pyspec(x)
    return x
