"""sstvae.qrss.ce, the transmit side -> sstvae_native.qrss.ce (native/core/qrss/ce).

Everything the transmitter and the channel simulator call is substituted:
the pulse (`_rrc`, `pulse`), phase synthesis on the 16 kHz grid and in
closed form (`_grid_sum`, `phase_grid`, `var_grid`, `phase_at`,
`_pulse_offsets`), the callsign windows (`window_keying`, `window_units`,
`_hann_int`, `cw_phase_turns`), the keyed span and its ramp, `baseband`,
`to_audio`, `check_carrier` and `carrier_fraction`. The module values the
C++ computes are rows too (`PULSE_NORM`, `G0`, and `PULSE_TABLE`, the
frozen table), so the receive-side Python left in the module
(`matched_filter`, `mean_template`, `_mean_template_grid`, the loopback
diagnostics) runs on the native pieces it calls. Those receive-only
functions are stage 2's.

`_pulse_offsets` is a generator in the reference and a list here; its one
caller iterates it. `lwave`, `track`, `ltrack` and `acquire` reach these
through the module (`ce._rrc`, `ce._turns_to_phasor`, ...), and channel.py's
`from .ce import check_carrier` is a from-import site the harness finds.
"""

STAGE = "tx"


def rows(q):
    c = q.ce
    t = "sstvae.qrss.ce"
    return {
        (t, "PULSE_NORM"): c.PULSE_NORM,
        (t, "PULSE_TABLE"): c.PULSE_TABLE,
        (t, "G0"): c.G0,
        (t, "_rrc"): c._rrc,
        (t, "pulse"): c.pulse,
        (t, "_grid_step"): c._grid_step,
        (t, "_check_sym"): c._check_sym,
        (t, "at_positions"): c.at_positions,
        (t, "_pulse_offsets"): c._pulse_offsets,
        (t, "phase_at"): c.phase_at,
        (t, "_grid_sum"): c._grid_sum,
        (t, "phase_grid"): c.phase_grid,
        (t, "var_grid"): c.var_grid,
        (t, "_hann_int"): c._hann_int,
        (t, "window_keying"): c.window_keying,
        (t, "window_units"): c.window_units,
        (t, "cw_phase_turns"): c.cw_phase_turns,
        (t, "keyed_span_pos"): c.keyed_span_pos,
        (t, "_ramp_envelope"): c._ramp_envelope,
        (t, "_turns_to_phasor"): c._turns_to_phasor,
        (t, "baseband"): c.baseband,
        (t, "to_audio"): c.to_audio,
        (t, "check_carrier"): c.check_carrier,
        (t, "carrier_fraction"): c.carrier_fraction,
    }
