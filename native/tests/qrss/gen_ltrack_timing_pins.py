#!/usr/bin/env python3
"""The pins of native/tests/qrss/test_ltrack_timing.cpp, from the Python reference.

Writes native/tests/qrss/ltrack_timing_pins.hpp, which is committed so the
ctest needs no Python. Re-run it from the repository root whenever
sstvae/qrss/ltrack.py (or anything it reads) changes on purpose:

    python native/tests/qrss/gen_ltrack_timing_pins.py

test_ltrack_timing.cpp rebuilds every input below from the same formulas
(exact binary fractions, correctly rounded decimal literals, or a function
the port computes to the bit: lwave.composite, lframe.assemble,
known_symbols_l, make_classes_l, data_templates), so both sides start from
the same doubles. Digests are hashlib.sha256 of the arrays' bytes; doubles
are written as hex literals.

What is pinned bitwise is what ltrack_timing.hpp lists as bitwise
(_lag_phase, _clock_phase, resid_ratio) and the coarse grid against the
C++'s own summation order, restated here in numpy (each k sum a
`(g[:, None] * v).sum(axis=0)`, numpy's sequential reduction over the
first axis); the rest -- the FFT convolutions, the fit, the path, gate V --
to rounding, its decisions exactly.
"""

from __future__ import annotations

import hashlib
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from sstvae.qrss import frame, lframe, lwave  # noqa: E402
from sstvae.qrss import ltrack as LT  # noqa: E402
from sstvae.qrss import track as T  # noqa: E402
from sstvae.qrss.constants import CH_FS, L_T_SYM, SPAN, WAVEFORM_L  # noqa: E402
from sstvae.qrss.types import Detection, FreqPath, Timing  # noqa: E402

OUT = Path(__file__).with_name("ltrack_timing_pins.hpp")
lines: list[str] = []


def dig(name: str, a) -> None:
    a = np.ascontiguousarray(a)
    lines.append(f'inline constexpr const char* {name} = "{hashlib.sha256(a.tobytes()).hexdigest()}";')


def flt(name: str, x) -> None:
    lines.append(f"inline constexpr double {name} = {float(x).hex()};")


def num(name: str, x) -> None:
    lines.append(f"inline constexpr std::int64_t {name} = {int(x)};")


def arr(name: str, xs) -> None:
    lines.append(f"inline constexpr double {name}[] = {{{', '.join(float(x).hex() for x in xs)}}};")


def iarr(name: str, xs) -> None:
    lines.append(f"inline constexpr std::int64_t {name}[] = {{{', '.join(str(int(x)) for x in xs)}}};")


def text(name: str, s: str) -> None:
    esc = s.replace("\\", "\\\\").replace('"', '\\"')
    lines.append(f'inline constexpr const char* {name} = "{esc}";')


def note(t: str) -> None:
    lines.append("")
    lines.append(f"// {t}")


def raised(fn, *a, **k) -> str:
    try:
        fn(*a, **k)
    except Exception as e:  # noqa: BLE001 - the reference's own text, whatever its type
        return str(e)
    raise AssertionError("did not raise")


TINY, SHORT = lframe.get("l-tiny"), lframe.get("l-short")
SPS = L_T_SYM * CH_FS

# --- the mix's phases (bitwise) -------------------------------------------------------------
note("_lag_phase(np.arange(-3000, 3000) * 0.375 + 0.0625), and at 0.0, -0.0 and 12.5 (signed zeros)")
dig("LAG_PHASE", LT._lag_phase(np.arange(-3000, 3000) * 0.375 + 0.0625))
for tag, v in (("LP_ZERO", 0.0), ("LP_NZERO", -0.0), ("LP_HALF", 12.5)):
    z = complex(LT._lag_phase(v))
    arr(tag, [z.real, z.imag])
note("_clock_phase(np.arange(0, 30000, 7), d) for d in (0.0, 2.5e-4, -1.75e-4)")
for tag, d in (("CP_0", 0.0), ("CP_P", 2.5e-4), ("CP_M", -1.75e-4)):
    dig(tag, LT._clock_phase(np.arange(0, 30000, 7), d))

# --- the statistic (to rounding: FFT convolutions) ------------------------------------------
n = np.arange(16000)
zc = (((n * 7) % 19 - 9) / 16.0) + 1j * (((n * 3) % 17 - 8) / 32.0)
N_REF = 12.375
note("zc = ((n*7 % 19 - 9) / 16) + j ((n*3 % 17 - 8) / 32), n < 16000; fit_mixed(zc, 12.375) and "
     "fit_statistic_l(zc, 12.375, 0.3) at FX_IDX, with max |.|")
y = LT.signal.oaconvolve(zc * np.conj(LT._lag_phase(np.arange(len(zc)) - N_REF)), T._mf_taps(CH_FS, L_T_SYM),
                         mode="same")
x = y * np.conj(T._lowpass(zc, 0.3, CH_FS))
fx_idx = [0, 7, 500, 8000, 12345, 15999]
iarr("FX_IDX", fx_idx)
arr("FM_RE", y.real[fx_idx])
arr("FM_IM", y.imag[fx_idx])
flt("FM_SCALE", np.max(np.abs(y)))
arr("FS_RE", x.real[fx_idx])
arr("FS_IM", x.imag[fx_idx])
flt("FS_SCALE", np.max(np.abs(x)))

# --- the coarse grid --------------------------------------------------------------------------
kn = LT.known_symbols_l(TINY)
G_BASE, G_NLAG, G_FRAC0, G_NORM = -40, 300, 0.375, 3.5
g_deltas = np.arange(-3, 4) * 2.5e-5
note("fit_grid_l(zc, known_symbols_l(l-tiny), base -40, nlag 300, frac0 0.375, np.arange(-3, 4) * 2.5e-5, "
     "L_T_SYM * 250, norm 3.5): bitwise against the C++'s order, and cells of the reference's direct path")


def grid(order: str) -> np.ndarray:
    a = np.asarray(kn.val, dtype=np.complex128)
    pos = np.asarray(kn.pos, dtype=np.float64)
    lags = np.arange(G_NLAG)
    lagph = LT._lag_phase(lags)
    xr, xi = zc.real.copy(), zc.imag.copy()
    Z = np.empty((len(g_deltas), G_NLAG))
    for i, d in enumerate(g_deltas):
        o = pos * SPS * (1 + d) + G_FRAC0
        idx = G_BASE + o[:, None] + lags[None, :]
        i0 = np.floor(idx).astype(np.int64)
        f = idx - i0
        ok = (i0 >= 0) & (i0 + 1 < len(zc))
        i0c = np.clip(i0, 0, len(zc) - 2)
        g = np.conj(a) * LT._clock_phase(pos, d)
        if order == "reference":
            v = (zc[i0c] * (1 - f) + zc[i0c + 1] * f) * ok
            Z[i] = (lagph * (g @ v)).real / G_NORM
        else:
            vr = (xr[i0c] * (1 - f) + xr[i0c + 1] * f) * ok
            vi = (xi[i0c] * (1 - f) + xi[i0c + 1] * f) * ok
            A = (g.real[:, None] * vr).sum(axis=0) / 1.0
            B = ((-g.imag)[:, None] * vi).sum(axis=0) / 1.0
            C = (g.real[:, None] * vi).sum(axis=0) / 1.0
            D = (g.imag[:, None] * vr).sum(axis=0) / 1.0
            w = np.empty(G_NLAG, dtype=np.complex128)
            w.real, w.imag = A + B, C + D
            Z[i] = (lagph * w).real / G_NORM
    return Z


Zk, Zr = grid("k"), grid("reference")
assert np.max(np.abs(Zk - Zr)) <= 1e-13 * np.max(np.abs(Zr))
dig("GRID", Zk)
cells = [(0, 0), (3, 150), (6, 299), (1, 77)]
iarr("GRID_CELLS", [c for ij in cells for c in ij])
arr("GRID_REF", [Zr[i, j] for i, j in cells])
flt("GRID_SCALE", np.max(np.abs(Zr)))

# --- a synthetic l-tiny pass ------------------------------------------------------------------
N_CH = 17500
xs = np.where((np.arange(TINY.n_data) * 5 + 1) % 3 == 0, 0.5, -0.5)
sym = lframe.assemble(TINY, None, xs)
ii = np.arange(N_CH)
sig = lwave.composite(sym, TINY, CH_FS, -3000, N_CH)
noise = (((ii * 7) % 19 - 9) / 4.0) + 1j * (((ii * 3) % 17 - 8) / 4.0)


def chan_of(z, n_keep=None):
    z = z if n_keep is None else z[:n_keep]
    return T.ChanCapture(ch=z.astype(np.complex64), fs=CH_FS, t0_index=3000.0, f_mix=Fraction(0))


chan, noise_chan = chan_of(sig + noise), chan_of(noise)
flat = FreqPath(t_s=np.array([-10.0, 0.0, 1800.0]), f_hz=np.full(3, 1500.0), weight=np.ones(3))
det = Detection(f_hz=1500.0, path=flat, timing=None, z_ref=np.nan, method="tbd", lead_in_s=0.0,
                waveform=WAVEFORM_L)
fn = T.freq_path_spline(flat)
z_d = T.derotate(chan, fn)
note("the synthetic l-tiny capture: lwave.composite(assemble(l-tiny, x = +-0.5), 250, -3000, 17500) + "
     "((i*7 % 19 - 9) + j (i*3 % 17 - 8)) / 4, complex64, t0_index 3000; and the noise alone")
dig("CAP", chan.ch)
dig("CAP_NOISE", noise_chan.ch)

note("fit_timing_l blind, n_best 8: the trace (base, nlag, nd, frac0, step exact; sig2 per carrier "
     "estimate), each peak's cell (lp, i, j), and the timings")
grids, segs = [], []
orig_full, orig_seg = np.full, LT._segment_timing_l


class _Np:
    def __getattr__(self, name):
        return getattr(np, name)

    def full(self, shape, fill, *a, **k):
        out = orig_full(shape, fill, *a, **k)
        if isinstance(shape, tuple) and len(shape) == 2:
            grids.append(out)
        return out


def _seg_spy(*a, **k):
    r = orig_seg(*a, **k)
    segs.append((a, r))
    return r


LT.np, LT._segment_timing_l = _Np(), _seg_spy
try:
    tms = LT.fit_timing_l(chan, CH_FS, fn, TINY, t0_index=chan.t0_index, z_d=z_d, n_best=8)
finally:
    LT.np, LT._segment_timing_l = np, orig_seg
n_pos = TINY.n_pos
step = 0.2 / n_pos
nd = int(math.floor(250.0 * 1e-6 / step))
base = int(math.floor(3000.0)) + int(math.floor(-2.5 * CH_FS - 2))
nlag = int(math.ceil(2.5 * CH_FS + 2)) - int(math.floor(-2.5 * CH_FS - 2)) + 1
num("FT_BASE", base)
num("FT_NLAG", nlag)
num("FT_ND", nd)
flt("FT_STEP", step)
sig2s = [s[0][1] for s in segs[::8]]
assert len(segs) == 8 * len(LT.CARRIER_LP_HZ_L)
arr("FT_SIG2", sig2s)
cells = []
for lp_i, Z in enumerate(grids):
    Zl, il = Z.max(axis=0), Z.argmax(axis=0)
    for k in range(8):
        j = int(np.argmax(Zl))
        if k and not np.isfinite(Zl[j]):
            break
        cells += [lp_i, int(il[j]), j]
        Zl[max(j - int(math.ceil(SPS)), 0):j + int(math.ceil(SPS)) + 1] = -np.inf
assert len(cells) == 3 * len(segs)
iarr("FT_CELLS", cells)
num("FT_N", len(tms))
arr("FT_TAU0", [t.tau0 for t in tms])
arr("FT_PPM", [t.ppm for t in tms])
arr("FT_Z", [t.z for t in tms])

note("fit_timing_l local: tau0_init = FT_TAU0[0] + 0.75 (n_best 1); the same with a real known set "
     "(pos[::3], sign(Re val[::3])) and tau_range 5")
t1 = LT.fit_timing_l(chan, CH_FS, fn, TINY, None, tms[0].tau0 + 0.75, t0_index=3000.0, z_d=z_d)
arr("FL", [t1.tau0, t1.ppm, t1.z])
kn_real = T._Known(pos=kn.pos[::3], val=np.sign(kn.val.real[::3]) + 0.0)
t2 = LT.fit_timing_l(chan, CH_FS, fn, TINY, kn_real, tms[0].tau0 + 0.75, t0_index=3000.0, tau_range=5.0, z_d=z_d)
arr("FLR", [t2.tau0, t2.ppm, t2.z])

note("_segment_timing_l(fit_statistic_l(z_d, base + frac0, 0.3), FT_SIG2[1], known_symbols_l(l-tiny), "
     "t0abs, delta, sps, n_pos, n_ref) for (t0abs, delta, seg_s) in SEG_IN")
n_ref = base + 0.0
xs03 = LT.signal.oaconvolve(z_d * np.conj(LT._lag_phase(np.arange(len(z_d)) - n_ref)),
                            T._mf_taps(CH_FS, L_T_SYM), mode="same") * np.conj(T._lowpass(z_d, 0.3, CH_FS))
seg_in = [(2999.75, 1e-5, 20.0), (3000.5, -2e-5, 7.0), (2999.0, 3e-5, 1000.0)]
arr("SEG_IN", [v for t in seg_in for v in t])
out = []
for t0abs, d, seg_s in seg_in:
    out += list(LT._segment_timing_l(xs03, sig2s[1], kn, t0abs, d, SPS, n_pos, n_ref, seg_s))
arr("SEG_OUT", out)

note("refine_path_l(chan, l-tiny, flat): length, ends, f at RP_IDX; and on the capture cut at t0 + 15 s, "
     "the path itself")
rp = LT.refine_path_l(chan, TINY, flat)
assert rp is not flat
num("RP_N", len(rp.t_s))
arr("RP_T_ENDS", [rp.t_s[0], rp.t_s[-1]])
rp_idx = [0, 10, 36, len(rp.t_s) - 1]
iarr("RP_IDX", rp_idx)
arr("RP_F", rp.f_hz[rp_idx])
assert np.all(rp.weight == 1.0) and np.allclose(np.diff(rp.t_s), 1.0)
cut = chan_of(sig + noise, 3000 + 15 * CH_FS)
assert LT.refine_path_l(cut, TINY, flat) is flat

note("verify_track_l(chan, l-tiny, det) (a blind fit): q, z_ref, resid_ratio, tau0, ppm; with the timing "
     "known (VT_TM, fit=False); and that with (10 s, 20 s) erased")
vt = LT.verify_track_l(chan, TINY, det)
arr("VT", [vt.q, vt.z_ref, vt.resid_ratio, vt.timing.tau0, vt.timing.ppm])
tm_known = Timing(tau0=-0.015625, ppm=7.5, gamma=0.0, z=0.0, cov=np.zeros((3, 3)))
arr("VT_TM", [tm_known.tau0, tm_known.ppm])
dk = Detection(f_hz=1500.0, path=flat, timing=tm_known, z_ref=np.nan, method="tbd", lead_in_s=0.0,
               waveform=WAVEFORM_L)
vk = LT.verify_track_l(chan, TINY, dk, fit=False)
assert vk.timing is tm_known
arr("VTK", [vk.q, vk.z_ref, vk.resid_ratio])
ve = LT.verify_track_l(chan, TINY, dk, fit=False, erase_s=((10.0, 20.0),))
arr("VTE", [ve.q, ve.z_ref, ve.resid_ratio])
vn = LT.verify_track_l(noise_chan, TINY, det)
arr("VTN", [vn.z_ref, vn.resid_ratio])

note("verify_l(chan, l-tiny, det): z_ref, tau0, ppm, the path's length and f at RP_IDX; noise refused")
v = LT.verify_l(chan, CH_FS, TINY, det)
assert v is not None and v.waveform == WAVEFORM_L
arr("V", [v.z_ref, v.timing.tau0, v.timing.ppm])
num("V_PATH_N", len(v.path.t_s))
arr("V_PATH_F", v.path.f_hz[rp_idx])
assert LT.verify_l(noise_chan, CH_FS, TINY, det) is None

# --- resid_ratio on a synthetic track (bitwise) -----------------------------------------------
note("resid_ratio on the synthetic l-short track of test_ltrack_timing.cpp (round A classes, "
     "data_templates(l-short, grid, classes)): bitwise; and with every psi 0, 1.0")
grid_s = np.arange(-SPAN, SHORT.n_pos, dtype=np.int64)
i = np.arange(len(grid_s))
cl = LT.make_classes_l(SHORT)
tr = LT.LTrackResult(
    spec=SHORT, chan=None, timing=tm_known, freq=None, pos=grid_s, z_d=None,
    m=(((i * 7) % 13 - 6) * 0.125) + 1j * (((i * 5) % 11 - 5) * 0.0625), c=None,
    u=(0.75 + (i % 3) * 0.25) + 1j * (((i % 4) - 1.5) * 0.125), P=0.001 * (1 + i % 5),
    N=0.25 + 0.0625 * (i % 7), psi=np.where(i % 97 == 0, 0.25, 1.0), meas=None, pu=None, classes=cl)
c_known, _, gain, dist, _ = LT.data_templates(SHORT, grid_s, cl)
rr = LT.resid_ratio(tr, c_known, gain, dist)
assert rr != 1.0
flt("RR", rr)
import dataclasses  # noqa: E402

assert LT.resid_ratio(dataclasses.replace(tr, psi=np.zeros(len(i))), c_known, gain, dist) == 1.0

note("the reference's refusals: a CE frame")
text("ERR_FIT", raised(LT.fit_timing_l, None, CH_FS, None, frame.TINY, t0_index=0.0))
text("ERR_VERIFY", raised(LT.verify_l, None, CH_FS, frame.TINY, None))
text("ERR_VTRACK", raised(LT.verify_track_l, None, frame.TINY, None))

head = """// Generated by native/tests/qrss/gen_ltrack_timing_pins.py from the Python
// reference (sstvae/qrss/ltrack.py). Do not edit: re-run the script.

#pragma once

#include <cstdint>

namespace ltrack_timing_pins {
"""
OUT.write_text(head + "\n".join(lines) + "\n\n}  // namespace ltrack_timing_pins\n")
print(f"wrote {OUT} ({len(lines)} lines)")
