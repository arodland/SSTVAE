"""sstvae.qrss.lwave, the transmit side -> sstvae_native.qrss.lwave
(native/core/qrss/lwave: waveform L's data grid, pilot, composite,
clipper, clipped frame and 8 kHz interpolation).

The native functions take the reference's own objects -- `LFrameSpec`s,
keying masks, and anything with `z`, `n0` and `fs` as a Baseband -- so
every row but one is the native callable itself. `frame_baseband` returns
the native module's Baseband, so its row hands back the reference's frozen
dataclass around the same (read-only) samples.

The derived levels (A_PILOT, A_CLIP, A_ID, G0) and the frozen pulse table
are rows too: other modules read them at call time, and the port computes
them itself. The receive side (`matched_filter`, `to_data_centre`,
`loopback_stats` and the masks behind it, `PULSE_NORM`) is not here;
`to_data_centre` is in lwave_rx.py.
"""

STAGE = "tx"


def rows(q):
    from sstvae.qrss import lwave as LW

    n = q.lwave
    Baseband = LW.Baseband

    def frame_baseband(sym, spec, *, keying=None, lead_in_s=0.0, fs=n.CLIP_FS,
                       ramp_s=n.KEY_RAMP_S, margin_s=4.0):
        b = n.frame_baseband(sym, spec, keying=keying, lead_in_s=lead_in_s, fs=fs,
                             ramp_s=ramp_s, margin_s=margin_s)
        return Baseband(z=b.z, n0=b.n0, fs=b.fs)

    t = "sstvae.qrss.lwave"
    return {
        (t, "PULSE_TABLE"): n.PULSE_TABLE,
        (t, "G0"): n.G0,
        (t, "A_PILOT"): n.A_PILOT,
        (t, "A_CLIP"): n.A_CLIP,
        (t, "A_ID"): n.A_ID,
        (t, "_grid_step"): n._grid_step,
        (t, "_grid_sum"): n._grid_sum,
        (t, "at_positions"): n.at_positions,
        (t, "data_grid"): n.data_grid,
        (t, "mix_turns"): n.mix_turns,
        (t, "tau_grid"): n.tau_grid,
        (t, "keyed_span_pos"): n.keyed_span_pos,
        (t, "window_keying"): n.window_keying,
        (t, "_raised"): n._raised,
        (t, "cw_turns"): n.cw_turns,
        (t, "id_weight"): n.id_weight,
        (t, "pilot"): n.pilot,
        (t, "composite"): n.composite,
        (t, "band_mask"): n.band_mask,
        (t, "clip"): n.clip,
        (t, "frame_baseband"): frame_baseband,
        (t, "_interp_taps"): n._interp_taps,
        (t, "resample"): n.resample,
    }
