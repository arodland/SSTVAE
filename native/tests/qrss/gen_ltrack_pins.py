#!/usr/bin/env python3
"""The pins of native/tests/qrss/test_ltrack.cpp, from the Python reference.

Writes native/tests/qrss/ltrack_pins.hpp, which is committed so the ctest
needs no Python. Re-run it from the repository root whenever
sstvae/qrss/ltrack.py (or anything it reads) changes on purpose:

    python native/tests/qrss/gen_ltrack_pins.py

test_ltrack.cpp rebuilds every input below from the same formulas (exact
binary fractions, correctly rounded divisions, or a function the port
computes to the bit), so both sides start from the same doubles. Digests
are hashlib.sha256 of the arrays' bytes; doubles are written as hex
literals.
"""

from __future__ import annotations

import hashlib
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from sstvae.qrss import ldemod, lframe, lwave  # noqa: E402
from sstvae.qrss import ltrack as LT  # noqa: E402
from sstvae.qrss import track as T  # noqa: E402
from sstvae.qrss.constants import CH_FS, L_T_SYM, SPAN, WAVEFORM_L  # noqa: E402
from sstvae.qrss.types import Detection, FreqPath, Timing  # noqa: E402

OUT = Path(__file__).with_name("ltrack_pins.hpp")
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


def note(text: str) -> None:
    lines.append("")
    lines.append(f"// {text}")


SHORT, TINY = lframe.get("l-short"), lframe.get("l-tiny")

# --- the exact-fraction inputs ---------------------------------------------------------------
n = np.arange(6000)
zr = ((n * 7) % 19 - 9) / 16.0
zc = zr + 1j * (((n * 3) % 17 - 8) / 32.0)
tm = Timing(tau0=3.25, ppm=12.0, gamma=0.5, z=0.0, cov=np.zeros((3, 3)))
chan0 = T.ChanCapture(ch=zc.astype(np.complex64), fs=CH_FS, t0_index=-100.5, f_mix=Fraction(0))
bits = ((np.arange(2474) * 7 + 3) % 5 < 2).astype(np.uint8)
key = ((np.arange(192) * 5 + 2) % 7 < 3).astype(np.uint8)

note("pilot_table, _pilot_taps")
tb = LT.pilot_table()
flt("TABLE_SPS", tb.sps)
num("TABLE_HALF", tb.half)
flt("TABLE_G0", tb.g0)
dig("TABLE_TAB", tb.tab)
dig("TAPS", LT._pilot_taps())

note("pilot_filter(zc | zr, tp), tp = np.arange(-400, 7200) * 0.875 + 0.0625: samples, and max |y|")
tp = np.arange(-400, 7200) * 0.875 + 0.0625
y = LT.pilot_filter(zc, tp)
yr = LT.pilot_filter(zr, tp)
pf_idx = [0, 399, 400, 401, 3000, 5123, 7228, 7229, 7599]
iarr("PF_IDX", pf_idx)
arr("PF_RE", y.real[pf_idx])
arr("PF_IM", y.imag[pf_idx])
flt("PF_SCALE", np.max(np.abs(y)))
arr("PFR", yr[pf_idx])
flt("PFR_SCALE", np.max(np.abs(yr)))

note("data_mix(zc, t0 -100.5, tm, l-short) all, and at idx [-5, 0, 7, 5999, 1234, -6000]")
dig("MIX_ALL", LT.data_mix(zc, chan0, tm, SHORT.n_pos))
dig("MIX_IDX", LT.data_mix(zc, chan0, tm, SHORT.n_pos, np.array([-5, 0, 7, 5999, 1234, -6000])))

note("data_mf: dense at np.arange(-20, 300), sparse at [0, 200.5, 377.25]")
dig("MF_DENSE", LT.data_mf(zc, chan0, tm, np.arange(-20, 300).astype(np.float64), SHORT.n_pos))
sparse = np.array([0.0, 200.5, 377.25])
tps = chan0.t0_index + T.rx_index(tm, sparse, SHORT.n_pos, CH_FS, t_sym=L_T_SYM)
half = T.mf_table(L_T_SYM).half
assert len(sparse) * (2 * half + 1) < min(int(np.ceil(tps.max())) + half + 2, 6000) - max(
    int(np.floor(tps.min())) - half - 1, 0), "the sparse pins must take the sparse path"
dig("MF_SPARSE", LT.data_mf(zc, chan0, tm, sparse, SHORT.n_pos))

note("make_classes_l: l-short A, B (bits), C (bits, prior), D (prior); l-tiny A")
k = np.arange(SHORT.n_data)
prior = SimpleNamespace(x_hat=((k * 5) % 9 - 4) / 8.0, v=((k * 3) % 7) / 5.0)
classes = {"A": LT.make_classes_l(SHORT), "B": LT.make_classes_l(SHORT, bits),
           "C": LT.make_classes_l(SHORT, bits, prior), "D": LT.make_classes_l(SHORT, None, prior),
           "TINY": LT.make_classes_l(TINY)}
for tag, c in classes.items():
    dig(f"CL_{tag}_MU", c.mu)
    dig(f"CL_{tag}_NU", c.nu)
    dig(f"CL_{tag}_KNOWN", c.known)
    dig(f"CL_{tag}_GAIN", c.gain)
    flt(f"CL_{tag}_RHO", c.spread_rho)

note("_distortion(l-short); data_templates(l-short, np.arange(-40, n_pos), classes B[, withhold])")
dig("DIST", LT._distortion(SHORT))
grid40 = np.arange(-40, SHORT.n_pos)
for tag, wk in (("", False), ("W", True)):
    c, nu, gain, dist, is_sym = LT.data_templates(SHORT, grid40, classes["B"], wk)
    dig(f"TM{tag}_C", c)
    dig(f"TM{tag}_NU", nu)
    dig(f"TM{tag}_GAIN", gain)
    dig(f"TM{tag}_DIST", dist)
    dig(f"TM{tag}_SYM", is_sym)

note("_pilot_gain / pilot_wave at tau = np.arange(-2400, 2 n_pos + 400) * 0.5 on l-short; keying "
     "((np.arange(192) * 5 + 2) % 7 < 3)")
tau = np.arange(-2400, 2 * SHORT.n_pos + 400) * 0.5
dig("PG_0", LT._pilot_gain(tau, SHORT))
dig("PG_5", LT._pilot_gain(tau, SHORT, 5.0))
dig("PW_0", LT.pilot_wave(tau, SHORT))
dig("PW_K", LT.pilot_wave(tau, SHORT, key))
dig("PW_K5", LT.pilot_wave(tau, SHORT, key, 5.0))
dig("PW_TINY", LT.pilot_wave(tau, TINY, None, 2.5))

note("q_grid(L_T_SYM), and _q_near's [lo, hi) for q = grid[0], grid[3], grid[-1], 1e-12, 1.0, "
     "1.3 grid[5]")
qg = T.q_grid(L_T_SYM)
dig("Q_GRID", np.array(qg))
qs = [qg[0], qg[3], qg[-1], 1e-12, 1.0, qg[5] * 1.3]
arr("QN_Q", qs)
spans = []
for q in qs:
    got = LT._q_near(q, qg)
    lo = qg.index(got[0])
    spans += [lo, lo + len(got)]
iarr("QN_SPAN", spans)

note("known_symbols_l: l-short, with bits, l-tiny")
for tag, kn in (("S", LT.known_symbols_l(SHORT)), ("SB", LT.known_symbols_l(SHORT, bits)),
                ("T", LT.known_symbols_l(TINY))):
    num(f"KN_{tag}_N", len(kn.pos))
    dig(f"KN_{tag}_POS", np.asarray(kn.pos, dtype=np.int64))
    dig(f"KN_{tag}_VAL", np.asarray(kn.val))

# --- the model on a synthetic track ---------------------------------------------------------
grid = np.arange(-SPAN, SHORT.n_pos, dtype=np.int64)
i = np.arange(len(grid))


def synth_track(cl):
    return LT.LTrackResult(
        spec=SHORT, chan=None, timing=tm, freq=None, pos=grid, z_d=None,
        m=(((i * 7) % 13 - 6) * 0.125) + 1j * (((i * 5) % 11 - 5) * 0.0625), c=None, u=None, P=None,
        N=0.25 + 0.0625 * (i % 7), psi=np.where(i % 97 == 0, 0.25, 1.0), meas=None, pu=None,
        classes=cl, m_pl=(((i * 3) % 11 - 5) * 0.0625) + 1j * (((i * 7) % 9 - 4) * 0.125),
        c_pl=0.375 + 1j * (((i % 5) - 2) * 0.0625), N_pl=0.125 + 0.03125 * (i % 5),
        psi_pl=np.where(i % 89 == 0, 0.125, 1.0))


note("_Model(tr, l-short, classes A, False, None, 0, 0) and (classes B, True, keying, 0, 1) on the "
     "synthetic track of test_ltrack.cpp")
pu = 0.5 + 0.125 * (i % 3)
v = (0.75 + (i % 3) * 0.25) + 1j * (((i % 4) - 1.5) * 0.125)
Pv = 0.001 * (1 + i % 5)
mask = (i % 4) != 1
for tag, (cl, wk, kk, nl) in (("M1", (classes["A"], False, None, 0)),
                              ("M2", (classes["B"], True, key, 1))):
    tr = synth_track(cl)
    md = LT._Model(tr, SHORT, cl, wk, kk, 0.0, nl)
    for f in ("c_all", "c", "nu", "gain", "dist", "k_pl", "is_sym", "is_known", "meas_d", "need",
              "meas_p", "measured"):
        dig(f"{tag}_{f.upper()}", getattr(md, f))
    (a, b), (c, d) = md.streams(pu)
    for f, x in (("MUD", a), ("RD", b), ("MUP", c), ("RP", d)):
        dig(f"{tag}_S_{f}", x)
    (a, b), (c, d) = md.streams(pu, 0.75 + 0.5j, data_mask=mask)
    for f, x in (("MUD", a), ("RD", b), ("MUP", c), ("RP", d)):
        dig(f"{tag}_SD_{f}", x)
    for f, kw in (("F", {}), ("FD", dict(delta=0.75 + 0.5j, data_mask=mask)), ("FP", dict(data=False)),
                  ("FN", dict(pilot=False))):
        mu, R = md.fused(pu, **kw)
        dig(f"{tag}_{f}_MU", mu)
        dig(f"{tag}_{f}_R", R)
    flt(f"{tag}_PU0", md.pu_initial())
    out = []
    for d0 in (1 + 0j, 0.75 + 0.5j, -1.25 - 0.25j):
        r = LT._fit_delta(md, v, Pv, pu, d0)
        out += [r.real, r.imag]
    arr(f"{tag}_DELTA", out)

note("_pu_local(v, 1000 Pv, model 1's measured | none): samples and Pu_g")
tr = synth_track(classes["A"])
md = LT._Model(tr, SHORT, classes["A"], False, None, 0.0, 0)
pl_idx = [0, 1, 17, 2000, len(grid) - 1]
iarr("PL_IDX", pl_idx)
for tag, meas in (("", md.measured), ("N", np.zeros(len(grid), dtype=bool))):
    loc, g = LT._pu_local(v, 1000 * Pv, meas)
    arr(f"PL{tag}_LOC", loc[pl_idx])
    flt(f"PL{tag}_G", g)

note("_hold_ends(v, Pv, R), R inf at i < 37, i > len - 50 and i % 11 == 0")
R = np.where((i < 37) | (i > len(grid) - 50) | (i % 11 == 0), np.inf, 1.0)
hu, hP = LT._hold_ends(v, Pv, R)
dig("HOLD_U", hu)
dig("HOLD_P", hP)

# --- synthetic l-tiny passes ------------------------------------------------------------------
N_CH = 17500


def capture(lead: float):
    x = np.where((np.arange(TINY.n_data) * 5 + 1) % 3 == 0, 0.5, -0.5)
    sym = lframe.assemble(TINY, None, x)
    ii = np.arange(N_CH)
    sig = lwave.composite(sym, TINY, CH_FS, -3000, N_CH, lead_in_s=lead)
    noise = (((ii * 7) % 19 - 9) / 16.0) + 1j * (((ii * 3) % 17 - 8) / 16.0)
    return T.ChanCapture(ch=(sig + noise).astype(np.complex64), fs=CH_FS, t0_index=3000.0, f_mix=Fraction(0))


flat = FreqPath(t_s=np.array([-10.0, 0.0, 1800.0]), f_hz=np.full(3, 1500.0), weight=np.ones(3))
tm0 = Timing(tau0=0.75, ppm=20.0, gamma=0.0, z=0.0, cov=np.zeros((3, 3)))
det = Detection(f_hz=1500.0, path=flat, timing=tm0, z_ref=np.nan, method="tbd", lead_in_s=0.0,
                waveform=WAVEFORM_L)
chan = capture(0.0)
note("the synthetic l-tiny capture (test_ltrack.cpp's synthetic_chan(0)), and track_l on it")
dig("CAP", chan.ch)
tr = LT.track_l(chan, TINY, det)
flt("TR_Q", tr.q)
flt("TR_LEAD", tr.lead_in_s)
num("TR_NPOS", len(tr.pos))
num("TR_P0", tr.pos[0])
dig("TR_MEAS", tr.meas)
dig("TR_MEAS_PL", tr.meas_pl)
dig("TR_C", tr.c)
flt("TR_LOGLIK", tr.loglik)
flt("TR_DF_MAX", tr.df_max)
arr("TR_TIMING", [tr.timing.tau0, tr.timing.ppm, tr.timing.gamma, tr.timing.z])
arr("TR_DELTA", [tr.delta.real, tr.delta.imag])
tr_idx = [0, 100, 400, len(tr.pos) - 1]
iarr("TR_IDX", tr_idx)
arr("TR_U", np.column_stack([tr.u.real[tr_idx], tr.u.imag[tr_idx]]).ravel())
arr("TR_P", tr.P[tr_idx])
arr("TR_UPL", np.column_stack([tr.u_pl.real[tr_idx], tr.u_pl.imag[tr_idx]]).ravel())
arr("TR_MPL", np.column_stack([tr.m_pl.real[tr_idx], tr.m_pl.imag[tr_idx]]).ravel())
arr("TR_M", np.column_stack([tr.m.real[tr_idx], tr.m.imag[tr_idx]]).ravel())
arr("TR_FREQ", tr.freq(np.array([0.0, 20.0, 50.0])))
rt = LT.refine_timing_l(tr)
assert rt is not None
arr("RT", [rt.tau0, rt.ppm, rt.gamma, rt.z])
rep = LT.report_l(tr, 0.9, False)
arr("REP", [rep.offset_hz, rep.drift_hz_per_min, rep.wander_hz_rms, rep.doppler_hz, rep.ppm, rep.snr2500_db])
flt("KEYED_DB_TINY", 10 * np.log10(LT.L_KEYED_POWER["l-tiny"] / ldemod.G_REF_ABS ** 2))

note("track_l(withhold_known=True), and track_l(refine=False, outer=0)")
tw = LT.track_l(chan, TINY, det, withhold_known=True)
flt("TW_Q", tw.q)
dig("TW_MEAS", tw.meas)
flt("TW_LOGLIK", tw.loglik)
arr("TW_TIMING", [tw.timing.tau0, tw.timing.ppm])
arr("TW_U", np.column_stack([tw.u.real[tr_idx], tw.u.imag[tr_idx]]).ravel())
tn = LT.track_l(chan, TINY, det, refine=False, outer=0)
flt("TN_Q", tn.q)
flt("TN_LOGLIK", tn.loglik)
arr("TN_DELTA", [tn.delta.real, tn.delta.imag])

note("a 3 s lead-in: the capture with it (synthetic_chan(3)) keeps the claim, the one without drops it")
chan_lead = capture(3.0)
dig("CAP_LEAD", chan_lead.ch)
det_lead = Detection(f_hz=1500.0, path=flat, timing=tm0, z_ref=np.nan, method="tbd", lead_in_s=3.0,
                     waveform=WAVEFORM_L)
tl = LT.track_l(chan_lead, TINY, det_lead)
flt("TL_LEAD", tl.lead_in_s)
num("TL_P0", tl.pos[0])
flt("TL_Q", tl.q)
dig("TL_MEAS_PL", tl.meas_pl)
arr("TL_TIMING", [tl.timing.tau0, tl.timing.ppm])
td = LT.track_l(chan, TINY, det_lead)
flt("TD_LEAD", td.lead_in_s)
num("TD_P0", td.pos[0])
z_lead = T.derotate(chan_lead, T.freq_path_spline(flat))
z0 = T.derotate(chan, T.freq_path_spline(flat))
num("LS_LEAD", LT._lead_supported_l(chan_lead, z_lead, tm0, TINY, 3.0))
num("LS_NONE", LT._lead_supported_l(chan, z0, tm0, TINY, 3.0))

head = """// Generated by native/tests/qrss/gen_ltrack_pins.py from the Python reference
// (sstvae/qrss/ltrack.py). Do not edit: re-run the script.

#pragma once

#include <cstdint>

namespace ltrack_pins {
"""
OUT.write_text(head + "\n".join(lines) + "\n\n}  // namespace ltrack_pins\n")
print(f"wrote {OUT} ({len(lines)} lines)")
