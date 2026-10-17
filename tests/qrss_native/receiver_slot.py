"""The slot half of sstvae.qrss.receiver -> sstvae_native.qrss.receiver_slot
(native/core/qrss/receiver_slot): `candidates`, `detect`, `_detect_joint`,
`_gate`, `_ce_resid_ratio`, `receive_slot`, `_receive_slot_joint`,
`capture_from_wav`, `receive_wav`, and the rules they apply (`_aliases`,
`_ghosts`, `_ghost_of`, `_suspect_neighbour`, `_suspect_beside`,
`_a3_lines`, `_owned`).

One orchestration (receiver_slot_flow.hpp), run two ways:

- **In C++, whole** (`typed=True`), when every stage the slot receiver
  reaches is the native row as this harness installs it -- receiver's own
  rows (this file's and tests/qrss_native/receiver.py's), `track.verify` /
  `verify_track`, `ltrack.verify_l`, `acquire.acquire` with every stage
  under it (acquire.py's own test, repeated here), `acquire.lead_in_span`,
  `frontend.noise_psd`, `acquire.frame_map` and `lframe.twin` (row or
  reference), `rules.old` (row or reference) and the clock (receiver's
  `time.monotonic`, the real one). Found by where each was defined
  (`_defined_in`) or by identity, never by `__qrss_native__` alone: a spy
  that copies that mark (test_qrss_native_receiver.py's `_counted`) must
  still be called, so it sends the slot receiver the second way. The
  arguments then become the C++ records; one that does not convert (a
  stub prep, a detection missing a field, one object listed twice, an
  np.int64 max_candidates) also goes the second way.
- **Over the Python objects**, otherwise: the same code calling every
  stage through receiver's own globals as the reference does
  (`receiver.detect`, `receiver.acquire.acquire`, `receiver.track.verify`,
  `receiver.time.monotonic`, `receiver.rules.old`, ...), resolved at each
  call, with the reference's arguments, so a test's replacement or spy
  sees exactly the reference's calls, and whatever a stage returns is
  handed on as it is.

The rules are the C++ either way (no test replaces one, and the
orchestration applies them itself), as are receiver's constants (MERGE_HZ,
GHOST_HZ, ...: generated from the reference, never read from the module).

Results come back as the reference's objects (`_rx.to_ref`): the native
records as its dataclasses, a stage's own objects as they are, and the
rules hand back the caller's own detections and passes.
`capture_from_wav` reads the WAV as `sstvae.wavio.read_wav` does
(native/core/qrss/wavread) while that and `frontend.audio_to_fe` are the
reference's or the rows', and through them otherwise.
"""

STAGE = "rx"

# receiver's own rows, by the rows file that defines them
_SLOT_ROWS = ("candidates", "detect", "_detect_joint", "_gate", "_ce_resid_ratio", "_receive_slot_joint")
_PASS_ROWS = ("prepare", "channel_for", "receive_pass")
_PLAIN_ROWS = ("frames", "is_joint", "_single", "centre_hz", "_centres", "_ce_hints")


def _defined_in(fn, module, name):
    """Whether fn is the row `name` of tests/qrss_native/<module>.py (a
    closure of its rows())."""
    return ((getattr(fn, "__module__", None) or "").rsplit(".", 1)[-1] == module
            and getattr(fn, "__qualname__", None) == f"rows.<locals>.{name}")


def _reference(fn, module, name):
    """Whether fn is the reference's own sstvae.qrss.<module>.<name>."""
    return getattr(fn, "__module__", None) == f"sstvae.qrss.{module}" and \
        getattr(fn, "__qualname__", None) == name


def rows(q):
    import time

    from . import _rx
    from .acquire import _ACQUIRE_STAGES

    n = q.receiver_slot
    nr = q.receiver
    to_ref = _rx.to_ref
    fe_native = q.frontend
    a1_threshold_native = q.acquire_stats.a1_threshold
    shape_frames_native = q.acquire_tbd.shape_frames
    frame_map_native = q.acquire_tbd.frame_map
    rules_old_native = q.rules.old
    monotonic = time.monotonic

    def _native(fn):
        return getattr(fn, "__qrss_native__", False)

    def _acquire_native():
        """acquire's own rows would hand the C++ no seams (acquire.py's
        seams() is None), and acquire / detections are those rows."""
        from sstvae.qrss import acquire as A
        from sstvae.qrss import frontend

        if not (frontend.blank is fe_native.blank and frontend.normalise is fe_native.normalise):
            return False
        if not all(_native(getattr(A, name)) for name in _ACQUIRE_STAGES):
            return False
        if not (_native(A._a2_input) and _native(A.preamble_search)):
            return False
        if A.a1_threshold is not a1_threshold_native or A.shape_frames is not shape_frames_native:
            return False
        return (_defined_in(A.acquire, "acquire", "acquire")
                and _defined_in(A._acquire_joint, "acquire", "_acquire_joint")
                and _defined_in(A.Acquisition.detections, "acquire", "detections")
                and _defined_in(A.Acquisition._joint_detections, "acquire", "_joint_detections")
                and _defined_in(A.lead_in_span, "acquire_a12", "lead_in_span"))

    def _typed():
        """Whether the whole slot can run in C++: every stage it reaches is
        the native row (module docstring)."""
        from sstvae.qrss import frontend, lframe, ltrack, rules, track
        from sstvae.qrss import receiver as R

        if not all(_defined_in(getattr(R, k), "receiver_slot", k) for k in _SLOT_ROWS):
            return False
        if not all(_defined_in(getattr(R, k), "receiver", k) for k in _PASS_ROWS):
            return False
        if not all(getattr(R, k) is getattr(nr, k) for k in _PLAIN_ROWS):
            return False
        if not (_defined_in(track.verify, "track_timing", "verify")
                and _defined_in(track.verify_track, "track_timing", "verify_track")
                and _defined_in(ltrack.verify_l, "ltrack_timing", "verify_l")):
            return False
        if R.track is not track or R.ltrack is not ltrack or R.frontend is not frontend or R.lframe is not lframe:
            return False
        if frontend.noise_psd is not fe_native.noise_psd:
            return False
        if R.acquire.frame_map is not frame_map_native and not _reference(R.acquire.frame_map, "acquire",
                                                                           "frame_map"):
            return False
        if not (_defined_in(lframe.twin, "frame", "twin") or _reference(lframe.twin, "lframe", "twin")):
            return False
        if R.rules is not rules or not (rules.old is rules_old_native or _reference(rules.old, "rules", "old")):
            return False
        if R.time is not time or time.monotonic is not monotonic:
            return False
        return _acquire_native()

    def _wav_typed():
        """capture_from_wav reads in C++: wavio.read_wav and audio_to_fe are
        the reference's or the rows'."""
        from sstvae import wavio
        from sstvae.qrss import frontend
        from sstvae.qrss import receiver as R

        fn = wavio.read_wav
        own = getattr(fn, "__module__", None) == "sstvae.wavio" and getattr(fn, "__qualname__", None) == "read_wav"
        a2f = frontend.audio_to_fe
        return own and R.frontend is frontend and (a2f is fe_native.audio_to_fe or _reference(a2f, "frontend",
                                                                                             "audio_to_fe"))

    # --- the rules ------------------------------------------------------------------------------

    from sstvae.qrss.constants import T_SYM

    def _aliases(dets, t_sym=T_SYM):
        return n._aliases(dets, t_sym)

    def _ghosts(dets, t_sym=T_SYM):
        return n._ghosts(dets, t_sym)

    def _ghost_of(f_hz, tau0, g_f_hz, g_tau0, t_sym=T_SYM):
        return n._ghost_of(f_hz, tau0, g_f_hz, g_tau0, t_sym)

    def _owned(d, found, a3=()):
        return n._owned(d, found, a3)

    # --- the stages -----------------------------------------------------------------------------

    def candidates(prep, spec, live_only=False, hints=()):
        return to_ref(n.candidates(prep, spec, live_only, hints, _typed()))

    def detect(prep, spec, live_only=False, max_candidates=None, budget_s=None, erase_s=(), cands=None,
               skip_hz=(), hints=()):
        return to_ref(n.detect(prep, spec, live_only, max_candidates, budget_s, erase_s, cands, skip_hz,
                               hints, _typed()))

    def _detect_joint(prep, specs, live_only, max_candidates, budget_s, erase_s, cands, skip_hz, hints):
        return to_ref(n._detect_joint(prep, specs, live_only, max_candidates, budget_s, erase_s, cands,
                                      skip_hz, hints, _typed()))

    def _gate(prep, spec, d, others, erase_s):
        return to_ref(n._gate(prep, spec, d, others, erase_s, _typed()))

    def _ce_resid_ratio(tr):
        return n._ce_resid_ratio(tr)

    from sstvae.qrss.frame import FULL

    def receive_slot(cap, spec=FULL, live_only=False, estimator="joint", *, erase_s=(), round_b=True,
                     dets=None, max_candidates=None, budget_s=None, hints=()):
        return to_ref(n.receive_slot(cap, spec, live_only, estimator, erase_s, round_b, dets, max_candidates,
                                     budget_s, hints, _typed()))

    def _receive_slot_joint(prep, specs, live_only, estimator, *, erase_s, round_b, dets, max_candidates,
                            budget_s, hints):
        return to_ref(n._receive_slot_joint(prep, specs, live_only, estimator, erase_s, round_b, dets,
                                            max_candidates, budget_s, hints, _typed()))

    def capture_from_wav(path, q, start_unix=None):
        return n.capture_from_wav(path, q, start_unix, _wav_typed())

    def receive_wav(path, q, start_unix=None, spec=FULL, estimator="joint"):
        from sstvae.qrss import receiver as R

        if (_typed() and _wav_typed() and _defined_in(R.receive_slot, "receiver_slot", "receive_slot")
                and _defined_in(R.capture_from_wav, "receiver_slot", "capture_from_wav")):
            return to_ref(n.receive_wav(path, q, start_unix, spec, estimator))
        return R.receive_slot(R.capture_from_wav(path, q, start_unix), spec, estimator=estimator)

    t = "sstvae.qrss.receiver"
    return {
        (t, "GHOST_HZ"): n.GHOST_HZ,
        (t, "GHOST_T"): n.GHOST_T,
        (t, "GHOST_Z_RATIO"): n.GHOST_Z_RATIO,
        (t, "ALIAS_HZ"): n.ALIAS_HZ,
        (t, "ALIAS_RATIO"): n.ALIAS_RATIO,
        (t, "L_OWN_BELOW_HZ"): n.L_OWN_BELOW_HZ,
        (t, "L_OWN_ABOVE_HZ"): n.L_OWN_ABOVE_HZ,
        (t, "CE_RESID_MIN"): n.CE_RESID_MIN,
        (t, "_aliases"): _aliases,
        (t, "_ghosts"): _ghosts,
        (t, "_ghost_of"): _ghost_of,
        (t, "_suspect_neighbour"): n._suspect_neighbour,
        (t, "_suspect_beside"): n._suspect_beside,
        (t, "_a3_lines"): n._a3_lines,
        (t, "_owned"): _owned,
        (t, "candidates"): candidates,
        (t, "detect"): detect,
        (t, "_detect_joint"): _detect_joint,
        (t, "_gate"): _gate,
        (t, "_ce_resid_ratio"): _ce_resid_ratio,
        (t, "receive_slot"): receive_slot,
        (t, "_receive_slot_joint"): _receive_slot_joint,
        (t, "capture_from_wav"): capture_from_wav,
        (t, "receive_wav"): receive_wav,
    }
