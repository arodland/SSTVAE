"""Si5351 synthesis model: CE's phase made by frequency steps with error
feedback (design 2.10, spec 2.5 "Making it with an Si5351").

A **format module** in the design's sense: it is the reference for how a
clock-chip beacon turns the CE phase into register writes, and the C
reference (`qrss_beacon_c/`, test C3) must produce the same steps.

An Si5351 cannot set its phase on the fly, but its fractional dividers
can be rewritten while it runs. Every Tu = 1/f_update the controller aims
at the target phase at the *end* of the next interval and sets

    f_u = round((Phi_tgt(t_{u+1}) - Phi_reached) / (2 pi Tu) / step_hz)

in whole synthesizer steps, then advances Phi_reached by 2 pi f_u step Tu.
Aiming from the phase actually reached (error feedback) keeps the
rounding from accumulating. This is exactly the recipe of the spec's
`sims/ce/pm_spectrum_synth.synth`.

Update u happens at t_u = t0 + u0_pos*T + u/f_update; the default
u0_pos = -8 is where keying starts (the target there is already
beta p(8) x_0, about 5 mrad: the edge of the pulse cut at 8 symbols). The
target is the total CE phase phi + 2 pi theta_cw with the window phase
unwrapped (`ce.cw_phase_turns`), so an FSK callsign window is just more
frequency steps. On-off-keyed windows switch only the output, so the
model leaves the phase running.

Rounding is half away from zero, the usual integer rounding of a C
`(x + sign(x)/2)` truncation, so a port can reproduce it bit for bit.
"""

from fractions import Fraction

import numpy as np

from .constants import T_SYM
from .frame import FrameSpec
from . import ce


def _round_half_away(x: float) -> int:
    return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(-x + 0.5))


def target_phase(tau_pos, sym, spec: FrameSpec, keying=None, ook: bool = False) -> np.ndarray:
    """Total CE phase in radians, phi + 2 pi theta_cw (window phase unwrapped).

    On-off keying has no window phase (theta_cw = 0): the output is gated,
    the synthesizer keeps running on the carrier.
    """
    phi, _ = ce.phase_at(tau_pos, sym, spec)
    theta, _ = ce.cw_phase_turns(tau_pos, spec, keying, ook=ook)
    return phi + 2 * np.pi * theta


def update_times_pos(f_update, n: int, u0_pos: float = -8.0) -> np.ndarray:
    """tau (symbols from t0) of updates 0..n-1: u0_pos + u / (f_update T)."""
    f = Fraction(f_update)
    u = np.arange(n, dtype=np.float64)
    return u0_pos + u / (float(f) * T_SYM)


def start_phase(sym, spec: FrameSpec, keying=None, ook: bool = False,
                u0_pos: float = -8.0, phase_fn=None) -> float:
    """The target phase at update 0, where the feedback loop starts (rad)."""
    t = np.array([float(u0_pos)])
    ph = target_phase(t, sym, spec, keying, ook) if phase_fn is None else phase_fn(t)
    return float(np.asarray(ph, dtype=np.float64)[0])


def si5351_steps(sym, spec: FrameSpec, q_unused, f_update, step_hz: float, n_updates: int,
                 keying=None, ook: bool = False, u0_pos: float = -8.0,
                 phase_fn=None) -> np.ndarray:
    """int32[n_updates]: the frequency of each update interval, in synthesizer steps.

    Interval u runs from t_u to t_{u+1}; its frequency offset from the
    carrier is steps[u] * step_hz. `sym` is the assembled frame (the
    scrambler and precoder are already in it, hence q is unused); with
    `ook` the windows add no phase (the output is gated instead). `phase_fn(tau) -> rad` replaces the
    closed-form target, so a fixed-point port can drive the same feedback
    loop with its own phase.
    """
    n = int(n_updates)
    tu = 1.0 / float(Fraction(f_update))
    tau_end = update_times_pos(f_update, n + 1, u0_pos)[1:]
    if phase_fn is None:
        tgt = target_phase(tau_end, sym, spec, keying, ook)
    else:
        tgt = np.asarray(phase_fn(tau_end), dtype=np.float64)
    start = start_phase(sym, spec, keying, ook, u0_pos, phase_fn)
    out = np.empty(n, dtype=np.int32)
    k = 2 * np.pi * tu * float(step_hz)            # phase per step per interval
    reached = start
    for u in range(n):
        f = _round_half_away((tgt[u] - reached) / k)
        out[u] = f
        reached += f * k
    return out


def synth_phase(steps, f_update, step_hz: float, fs: int, n: int,
                u0_pos: float = -8.0, phase0: float = 0.0) -> np.ndarray:
    """float64[n]: the synthesized phase (rad) at t_i = t0 + u0_pos*T + i/fs.

    Piecewise linear from `phase0` at t_0 (pass `start_phase` to compare
    with the ideal phase): interval u adds 2 pi steps[u] step_hz (t - t_u).
    After the last update the phase holds (the synthesizer is back on the
    carrier).
    """
    steps = np.asarray(steps, dtype=np.float64)
    fu = float(Fraction(f_update))
    t = np.arange(int(n), dtype=np.float64) / fs          # seconds since t_0
    u_pos = t * fu
    u = np.floor(u_pos).astype(np.int64)
    frac = u_pos - u
    w = 2 * np.pi * steps * float(step_hz) / fu            # rad per interval
    reached = phase0 + np.concatenate([[0.0], np.cumsum(w)])   # phase at t_u
    nu = len(steps)
    inside = u < nu
    out = np.where(inside,
                   reached[np.clip(u, 0, nu)] + w[np.clip(u, 0, nu - 1)] * frac,
                   reached[-1])
    return out


def step_frequencies_hz(steps, step_hz: float) -> np.ndarray:
    """The commanded frequency offsets in Hz."""
    return np.asarray(steps, dtype=np.float64) * float(step_hz)
