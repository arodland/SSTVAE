"""The per-pass half of sstvae.qrss.receiver -> sstvae_native.qrss.receiver
(native/core/qrss/receiver): `prepare`, `frames`, `is_joint`, `_single`,
`centre_hz`, `_centres`, `_ce_hints`, `channel_for`, `_psi_frame`,
`_cw_result`, `receive_pass`, `decode_header`, `_decode_header_l`,
`detection_from_pass`, `capture_from_pass`, `mean_w_db` and
`confidence_db`. The slot half (`candidates`, `detect`, `receive_slot`,
the ghost and alias rules, `_gate`, ...) stays the reference's for now, and
so does `_receive_pass_l`, which the native receive_pass no longer calls (it
dispatches on the frame's waveform itself, as the reference's does).

The native receive_pass calls its own C++ for what receiver's (and track's,
ltrack's, frontend's) attributes hold the native rows of: `prepare` and
`channel_for` (this file's, marked `__qrss_native__`), `track.verify` and
`ltrack.verify_l` (tests/qrss_native/track_timing.py's and
ltrack_timing.py's, found by where they were defined) and
`frontend.noise_psd` (the native function itself). Whatever else one holds
-- a test's replacement or spy, or the reference when those rows are not
applied -- is handed to the C++ as a seam (`_seams`) and called with the
reference's arguments, resolved at each call.

Results come back as the reference's objects (`_rx.to_ref`), and as the
very objects the reference would return where it returns an argument:
`prepare` hands back a Prepared, `frames` / `_single` the caller's frames,
`_ce_hints` the caller's hints, `decode_header` the caller's llr and spec
(or the block's LLRs in llr's dtype and `frame.plain(spec)`),
`detection_from_pass` the pass's own timing and `capture_from_pass` the
pass's own arrays. With return_tracks, the tracks share one channel object
and the PassResult holds the last track's timing object, as the
reference's do.
"""

STAGE = "rx"


def _defined_in(fn, module, name):
    """Whether fn is the row `name` of tests/qrss_native/<module>.py (a
    closure of its rows())."""
    return ((getattr(fn, "__module__", None) or "").rsplit(".", 1)[-1] == module
            and getattr(fn, "__qualname__", None) == f"rows.<locals>.{name}")


def rows(q):
    from . import _rx

    n = q.receiver
    to_ref = _rx.to_ref
    native_noise_psd = q.frontend.noise_psd

    def _native(fn):
        return getattr(fn, "__qrss_native__", False)

    def seams():
        """What receive_pass is to call in Python: the names whose attribute
        is not the native row now."""
        from sstvae.qrss import frontend, ltrack, track
        from sstvae.qrss import receiver as R

        s = {}
        if not _native(R.prepare):
            s["prepare"] = R.prepare
        if not _native(R.channel_for):
            fn = R.channel_for
            s["channel_for"] = lambda prep, spec, det, ex, fm: fn(to_ref(prep), spec, det, ex, fm)
        if not _defined_in(track.verify, "track_timing", "verify"):
            v = track.verify
            s["verify"] = lambda chan, fs, spec, det, erase_s=(): v(to_ref(chan), fs, spec, det, erase_s=erase_s)
        if not _defined_in(ltrack.verify_l, "ltrack_timing", "verify_l"):
            vl = ltrack.verify_l
            s["verify_l"] = lambda chan, fs, spec, det, erase_s=(): vl(to_ref(chan), fs, spec, det,
                                                                       erase_s=erase_s)
        if frontend.noise_psd is not native_noise_psd:
            s["noise_psd"] = frontend.noise_psd
        return s or None

    def prepare(cap):
        return to_ref(n.prepare(cap))

    def channel_for(prep, spec, det, exclude_hz=(), f_mix_hz=None):
        return to_ref(n.channel_for(prep, spec, det, exclude_hz, f_mix_hz))

    def _psi_frame(tr):
        return n._psi_frame(tr)

    def _cw_result(tr, spec, hdr):
        cw, known = n._cw_result(tr, spec, hdr)
        return to_ref(cw), known

    def receive_pass(cap, spec, det, prior=None, estimator="joint", *, exclude_hz=(), f_mix_hz=None,
                     round_b=True, return_tracks=False, erase_s=()):
        r = n.receive_pass(cap, spec, det, prior, estimator, exclude_hz=exclude_hz, f_mix_hz=f_mix_hz,
                           round_b=round_b, return_tracks=return_tracks, erase_s=erase_s,
                           _seams=seams())
        if r is None or not return_tracks:
            return to_ref(r)
        p, trs = r
        chan = to_ref(trs[0]["chan"])
        tracks = []
        for d in trs:
            d["chan"] = chan
            tracks.append(to_ref(d))
        pr = to_ref(p)
        pr.timing = tracks[-1].timing
        return pr, tracks

    def decode_header(llr, diag, spec):
        hdr, kept, sp = n.decode_header(llr, diag, spec)
        return to_ref(hdr), kept, to_ref(sp)

    def _decode_header_l(llr, diag, spec):
        return to_ref(n._decode_header_l(llr, diag, spec))

    def detection_from_pass(p, lead_in_s=0.0):
        return to_ref(n.detection_from_pass(p, lead_in_s))

    def capture_from_pass(p):
        return to_ref(n.capture_from_pass(p))

    for fn in (prepare, channel_for, receive_pass):
        fn.__qrss_native__ = True

    t = "sstvae.qrss.receiver"
    return {
        (t, "prepare"): prepare,
        (t, "frames"): n.frames,
        (t, "is_joint"): n.is_joint,
        (t, "_single"): n._single,
        (t, "centre_hz"): n.centre_hz,
        (t, "_centres"): n._centres,
        (t, "_ce_hints"): n._ce_hints,
        (t, "channel_for"): channel_for,
        (t, "_psi_frame"): _psi_frame,
        (t, "_cw_result"): _cw_result,
        (t, "receive_pass"): receive_pass,
        (t, "decode_header"): decode_header,
        (t, "_decode_header_l"): _decode_header_l,
        (t, "detection_from_pass"): detection_from_pass,
        (t, "capture_from_pass"): capture_from_pass,
        (t, "mean_w_db"): n.mean_w_db,
        (t, "confidence_db"): n.confidence_db,
    }
