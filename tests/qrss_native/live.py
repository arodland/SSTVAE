"""sstvae.qrss.live up to LiveListener -> sstvae_native.qrss.live
(native/core/qrss/live): `utc_iso`, `frame_seconds`, `in_band`,
`centre_of`, `_waveform_id`, `_due_waveform`, `FeRing`, `_spans`,
`capture_read`, `capture_prepare`, `slot_capture`, `finish_receive`,
`DialLog`, `same_dial` and `restrict`, so that LiveListener (which stays
the reference's) runs on them. StreamFE is frontend.py's to port (its C++
twin is there for the parity test), the end-of-slot child process
(`_FinishProcess`) is not ported, and `Tile`, `SlotState` and `LiveConfig`
stay the reference's dataclasses: the listener sets their fields freely
and keeps them in its own state, which a pybind class would not stand in
for. Their C++ twins are tested against them (test_qrss_native_live.py).

`FeRing` and `DialLog` are the native classes themselves. The capture and
the end-of-slot receive go one of two ways, as receiver_slot.py's rows do:

- **In C++, whole** (`typed=True`), when every stage the call reaches is
  the native row or the reference's own: `frontend.blank` and
  `.normalise`, `live.frame_seconds`, `live.slot_t0`, the live rows this
  file installs, and for `finish_receive` the slot receiver's rows with
  their own `_typed()` true (read out of receiver_slot.py's
  `receive_slot` row, so the two can never disagree about what "every
  stage is native" means) and `receiver.is_joint` / `centre_hz` the rows'
  or the reference's.
- **Over the Python objects**, otherwise: every stage through live's
  module attributes, resolved at each call, with the reference's
  arguments (`live.capture_prepare`, `live.receiver.receive_slot`, ...),
  so a test's replacement or spy sees the reference's calls.

What was not heard is filled with `_live.noise()`: numpy's
`default_rng(q)` normals, so every capture is the reference's bit for bit
(`QRSS_LIVE_NOISE=cpp`: the C++ generator, the same distribution).
"""

STAGE = "rx"


def rows(q):
    from sstvae.qrss.frontend import PB_LEAD_S, PB_TAIL_S

    from . import _live, _rx
    from .receiver_slot import _defined_in, _reference

    n = q.live
    nr = q.receiver
    fe_native = q.frontend
    frame_seconds_native = n.frame_seconds
    to_ref = _rx.to_ref

    def _live_module():
        from sstvae.qrss import live as LV

        return LV

    def _fs_ok(LV):
        f = LV.frame_seconds
        return f is frame_seconds_native or _reference(f, "live", "frame_seconds")

    def _frontend_ok(LV):
        from sstvae.qrss import frontend

        if LV.frontend is not frontend:
            return False
        return all(getattr(frontend, k) is getattr(fe_native, k) or _reference(getattr(frontend, k), "frontend", k)
                   for k in ("blank", "normalise"))

    def _read_typed():
        LV = _live_module()
        return _fs_ok(LV) and _reference(LV.slot_t0, "frontend", "slot_t0")

    def _prepare_typed():
        LV = _live_module()
        return _fs_ok(LV) and _frontend_ok(LV)

    def _slot_capture_typed():
        LV = _live_module()
        return (_defined_in(LV.capture_read, "live", "capture_read")
                and _defined_in(LV.capture_prepare, "live", "capture_prepare")
                and _read_typed() and _prepare_typed())

    def _closure(fn, name):
        return fn.__closure__[fn.__code__.co_freevars.index(name)].cell_contents

    def _receiver_typed():
        """The slot receiver would run whole in C++: receive_slot and detect
        are receiver_slot.py's rows and their `_typed()` holds."""
        from sstvae.qrss import receiver as R

        rs, det = R.receive_slot, R.detect
        if not (_defined_in(rs, "receiver_slot", "receive_slot") and _defined_in(det, "receiver_slot", "detect")):
            return False
        for k in ("is_joint", "centre_hz"):
            fn = getattr(R, k)
            if not (fn is getattr(nr, k) or _reference(fn, "receiver", k)):
                return False
        return bool(_closure(rs, "_typed")())

    def _finish_typed():
        from sstvae.qrss import receiver as R

        LV = _live_module()
        return (LV.receiver is R and _defined_in(LV.capture_prepare, "live", "capture_prepare")
                and _prepare_typed() and _receiver_typed())

    def capture_read(ring, q, spec, lead_s=PB_LEAD_S, tail_s=PB_TAIL_S):
        return n.capture_read(ring, q, spec, lead_s, tail_s, _read_typed())

    def capture_prepare(x, ok, origin, q, spec, seed=None):
        return to_ref(n.capture_prepare(x, ok, origin, q, spec, seed, _live.noise(), _prepare_typed()))

    def slot_capture(ring, q, spec, seed=None, lead_s=PB_LEAD_S, tail_s=PB_TAIL_S):
        return to_ref(n.slot_capture(ring, q, spec, seed, lead_s, tail_s, _live.noise(), _slot_capture_typed()))

    def finish_receive(x, ok, origin, q, spec, estimator, dets, hints=()):
        passes, heard = n.finish_receive(x, ok, origin, q, spec, estimator, dets, hints, _live.noise(),
                                         _finish_typed())
        return to_ref(passes), heard

    t = "sstvae.qrss.live"
    return {
        (t, "utc_iso"): n.utc_iso,
        (t, "frame_seconds"): n.frame_seconds,
        (t, "in_band"): n.in_band,
        (t, "centre_of"): n.centre_of,
        (t, "_waveform_id"): n._waveform_id,
        (t, "_due_waveform"): n._due_waveform,
        (t, "FeRing"): n.FeRing,
        (t, "_spans"): n._spans,
        (t, "capture_read"): capture_read,
        (t, "capture_prepare"): capture_prepare,
        (t, "slot_capture"): slot_capture,
        (t, "finish_receive"): finish_receive,
        (t, "DialLog"): n.DialLog,
        (t, "same_dial"): n.same_dial,
        (t, "restrict"): n.restrict,
    }
