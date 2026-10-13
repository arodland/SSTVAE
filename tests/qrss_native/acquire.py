"""sstvae.qrss.acquire's arbitration and acquire() -> sstvae_native.qrss.acquire
(native/core/qrss/acquire): `acquire`, `_acquire_joint`, `_hint_items`,
`drop_symbol_rate_aliases`, `drop_band_owned`, `merge_hits`, `live_end_s`,
and `Acquisition.detections` / `_joint_detections`.

The C++ acquisition calls its own stages (A1, A2, the lead-in, A3, the
front end's blanker) while acquire's (and frontend's) attributes are the
native rows -- the acquire_a12 / acquire_tbd wrappers, which carry
`__qrss_native__`, or the frontend / acquire_stats / acquire_tbd functions
themselves. An attribute that is anything else (a test's replacement, or
the reference when those rows are not applied) is handed to the C++ as a
seam (`_seams`) and called with the reference's arguments, so a test that
monkeypatches `carrier_lines`, `preamble_search`, `lead_in_span`,
`shape_frames`, ... still sees its replacement called. `_a2_input` and
`preamble_search` go as a pair, as do `frontend.blank` and
`frontend.normalise`. Arbitration inside the native acquire() is the C++'s
whatever acquire's `drop_*` / `merge_hits` / `_hint_items` / `frame_map`
attributes hold; `acquire` itself defers to acquire's `_acquire_joint`
when that is not this row (a test's replacement).

The Acquisition comes back as the reference's own dataclass
(`_rx.to_ref`); lines, hits and A3 hits a replaced stage made are its own
objects. The arbitration functions return the caller's own hit objects.
`detections` restores a hint detection's f_hz to the hint line's own
object (an np.float64 when hint_line refined it).
"""

STAGE = "rx"

# acquire's stages, by the name the C++ seam takes
_ACQUIRE_STAGES = ("carrier_line_stats", "carrier_lines", "pilot_lines", "hint_line", "lead_in_span",
                   "track_before_detect")


def rows(q):
    from sstvae.qrss import acquire as A
    from sstvae.qrss import frontend

    from . import _rx

    n = q.acquire
    to_ref = _rx.to_ref
    fe_native = q.frontend
    a1_threshold_native = q.acquire_stats.a1_threshold
    shape_frames_native = q.acquire_tbd.shape_frames

    def _native(fn):
        return getattr(fn, "__qrss_native__", False)

    def seams():
        """The stages acquire() is to call in Python: those that are not native now."""
        s = {}
        if not (frontend.blank is fe_native.blank and frontend.normalise is fe_native.normalise):
            s["blank"], s["normalise"] = frontend.blank, frontend.normalise
        for name in _ACQUIRE_STAGES:
            fn = getattr(A, name)
            if not _native(fn):
                s[name] = fn
        if not (_native(A._a2_input) and _native(A.preamble_search)):
            s["_a2_input"], s["preamble_search"] = A._a2_input, A.preamble_search
        if A.a1_threshold is not a1_threshold_native:
            s["a1_threshold"] = A.a1_threshold
        if frontend.noise_psd is not fe_native.noise_psd:
            s["noise_psd"] = frontend.noise_psd
        if A.shape_frames is not shape_frames_native:
            s["shape_frames"] = A.shape_frames
        return s or None

    def acquire(cap, spec, *, live_only=False, preprocessed=False, hints=()):
        if isinstance(spec, (tuple, list)) and not _native(A._acquire_joint):
            return A._acquire_joint(cap, A.frame_map(spec), live_only=live_only,
                                    preprocessed=preprocessed, hints=hints)
        return to_ref(n.acquire(cap, spec, live_only=live_only, preprocessed=preprocessed,
                                hints=hints, _seams=seams()))

    def _acquire_joint(cap, specs, *, live_only, preprocessed, hints):
        return to_ref(n._acquire_joint(cap, specs, live_only=live_only, preprocessed=preprocessed,
                                       hints=hints, _seams=seams()))

    def _hint_f(acq, out):
        """A hint detection's f_hz as the hint line holds it (same value)."""
        own = [c.f_hz for c in acq.hinted]
        for d in out:
            if d.method == "hint":
                for f in own:
                    if f == d.f_hz:
                        d.f_hz = f
                        break
        return out

    def detections(self, spec):
        if isinstance(spec, (tuple, list)) and not _native(type(self)._joint_detections):
            return self._joint_detections(A.frame_map(spec))
        return _hint_f(self, to_ref(n._detections(self, spec)))

    def _joint_detections(self, specs):
        return _hint_f(self, to_ref(n._joint_detections(self, specs)))

    for fn in (acquire, _acquire_joint, detections, _joint_detections):
        fn.__qrss_native__ = True

    t = "sstvae.qrss.acquire"
    acq = t + ".Acquisition"
    return {
        (t, "acquire"): acquire,
        (t, "_acquire_joint"): _acquire_joint,
        (t, "_hint_items"): n._hint_items,
        (t, "drop_symbol_rate_aliases"): n.drop_symbol_rate_aliases,
        (t, "drop_band_owned"): n.drop_band_owned,
        (t, "merge_hits"): n.merge_hits,
        (t, "live_end_s"): n.live_end_s,
        (acq, "detections"): detections,
        (acq, "_joint_detections"): _joint_detections,
    }
