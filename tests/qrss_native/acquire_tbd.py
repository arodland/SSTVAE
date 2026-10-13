"""sstvae.qrss.acquire's A3 (track-before-detect) -> sstvae_native.qrss.acquire_tbd
(native/core/qrss/acquire_tbd), with frame_map and _path_of.

The C++ returns TbdHits and FreqPaths as "__type__" dicts; the adapters
here build the reference's own `acquire.TbdHit` and `types.FreqPath`, so
acquire()'s arbitration (and every test) handles exactly the objects it
always did. TbdHit's methods -- `f_hz`, `line`, `passed`, `to_detection`
-- become the C++'s, reading the hit's fields, so a hit built in Python
(the tests construct them directly) runs them too.

`track_before_detect` finds the shape-test frames through whatever
`acquire.shape_frames` is when it is called -- the native function
directly, or (a test monkeypatches it to spy on its arguments) the
replacement, called with the reference's arguments.

The shared statistics A3 calls (spectrogram, cfar_normalise, gumbel_p,
_band_cov) are acquire_stats' rows; the C++ calls its own copies of them
directly.
"""

STAGE = "rx"


def _tbd_hit(d):
    from sstvae.qrss import acquire

    from . import _rx

    f = {k: v for k, v in d.items() if k != "__type__"}
    f["path"] = _rx.to_ref(f["path"])
    f["l_path"] = _rx.to_ref(f["l_path"])
    return acquire.TbdHit(**f)


def rows(q):
    from . import _rx

    a = q.acquire_tbd
    native_shape_frames = a.shape_frames

    def track_before_detect(fe, t0_index, spec, cands=(), fs=None, *, pfa=None, band_hz=None,
                            all_hits=False, shape_specs=None):
        from sstvae.qrss import acquire

        kw = {"all_hits": all_hits, "shape_specs": shape_specs}
        if fs is not None:
            kw["fs"] = fs
        if pfa is not None:
            kw["pfa"] = pfa
        if band_hz is not None:
            kw["band_hz"] = band_hz
        sf = acquire.shape_frames
        if sf is not native_shape_frames:
            kw["_shape_frames"] = sf
        return [_tbd_hit(d) for d in a.track_before_detect(fe, t0_index, spec, cands, **kw)]

    def _bins_path(E, f, t, bins):
        return _rx.to_ref(a._bins_path(E, f, t, bins))

    def _path_of(f_hz, drift_hz_per_min, dur):
        return _rx.to_ref(a._path_of(f_hz, drift_hz_per_min, dur))

    def to_detection(self, waveform=None):
        return _rx.to_ref(a._hit_to_detection(self, waveform))

    def line(self):
        return a._hit_line(self)

    m = "sstvae.qrss.acquire"
    hit = m + ".TbdHit"
    return {
        (m, "tbd_frame_starts"): a.tbd_frame_starts,
        (m, "tbd_n_frames"): a.tbd_n_frames,
        (m, "tbd_viterbi"): a.tbd_viterbi,
        (m, "_path_line"): a._path_line,
        (m, "tbd_channels"): a.tbd_channels,
        (m, "tbd_emissions"): a.tbd_emissions,
        (m, "shape_frames"): a.shape_frames,
        (m, "shape_z"): a.shape_z,
        (m, "shape_waveform"): a.shape_waveform,
        (m, "track_before_detect"): track_before_detect,
        (m, "_shape_verdict"): a._shape_verdict,
        (m, "_bins_path"): _bins_path,
        (m, "l_pilot_bins"): a.l_pilot_bins,
        (m, "frame_map"): a.frame_map,
        (m, "_path_of"): _path_of,
        (hit, "f_hz"): property(a._hit_f_hz),
        (hit, "line"): line,
        (hit, "passed"): property(a._hit_passed),
        (hit, "to_detection"): to_detection,
    }
