#!/usr/bin/env python3
"""The pins of native/tests/qrss/test_receiver.cpp, from the Python reference.

Writes native/tests/qrss/receiver_pins.hpp, which is committed so the
ctest needs no Python. Re-run it from the repository root whenever
sstvae/qrss/receiver.py (or anything it reads) changes on purpose:

    python native/tests/qrss/gen_receiver_pins.py

test_receiver.cpp rebuilds every input below from the same formulas (exact
binary fractions, integer hashes, or a function the port computes to the
bit: frame.assemble, ce.phase_grid, lframe.assemble, lwave.composite,
header.encode), so both sides start from the same doubles; the captures'
digests say so. Digests are hashlib.sha256 of the arrays' bytes; doubles
are written as hex literals.

What is pinned bitwise is what receiver.hpp lists as bitwise: decode_header
(the header and which copy and frame it leaves), confidence_db, mean_w_db,
centre_hz, detection_from_pass and the refusals' texts. A pass through
receive_pass is pinned to its primitives' rounding (the tracker, the fit,
the demodulator) and its decisions exactly: whether there is a pass, the
header, the callsign verdict, the trackers' q, the report's flags.

The passes are synthetic CH-rate captures (a stored pass's form, fs 250,
receive_pass's EM path) of TINY, SHORT, l-tiny and l-short at about -6 dB
in 2500 Hz: the frame of latents x = +-1 (L +-0.5) with the test header,
plus a hash noise (`noise_of`, a sum of four uniforms per component), and
a 0.02 Hz offset drifting at 0.001 Hz/s on the CE ones.
"""

from __future__ import annotations

import hashlib
import math
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from sstvae.qrss import ce, frame, frontend, header, lframe, lwave  # noqa: E402
from sstvae.qrss import receiver as RX  # noqa: E402
from sstvae.qrss.constants import CH_FS, FE_FS, WAVEFORM_CE, WAVEFORM_L  # noqa: E402
from sstvae.qrss.types import Detection, FreqPath, PassResult, Timing, TrackReport  # noqa: E402

OUT = Path(__file__).with_name("receiver_pins.hpp")
lines: list[str] = []


def dig(name: str, a) -> None:
    a = np.ascontiguousarray(a)
    lines.append(f'inline constexpr const char* {name} = "{hashlib.sha256(a.tobytes()).hexdigest()}";')


def flt(name: str, x) -> None:
    lines.append(f"inline constexpr double {name} = {float(x).hex()};")


def num(name: str, x) -> None:
    lines.append(f"inline constexpr std::int64_t {name} = {int(x)};")


def arr(name: str, xs) -> None:
    xs = list(xs)
    body = ", ".join(float(x).hex() if not math.isnan(float(x)) else "NAN_PIN" for x in xs)
    lines.append(f"inline constexpr double {name}[] = {{{body}}};")
    lines.append(f"inline constexpr std::size_t {name}_N = {len(xs)};")


def iarr(name: str, xs) -> None:
    xs = list(xs)
    lines.append(f"inline constexpr std::int64_t {name}[] = {{{', '.join(str(int(x)) for x in xs)}}};")
    lines.append(f"inline constexpr std::size_t {name}_N = {len(xs)};")


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


HDR = header.HeaderFields(callsign="K1ABC", grid="FN42", picture_id=0x12345678, mode=0,
                          segment=0, codec_id=0xD1D8)
HDR1 = header.HeaderFields(callsign="W1AW", grid=None, picture_id=0x0BADF00D, mode=1,
                           segment=1, codec_id=0x1234, version=1)
Q = 1954752
FLAT = FreqPath(t_s=np.array([-10.0, 0.0, 1800.0]), f_hz=np.full(3, 1500.0), weight=np.ones(3))
M32 = np.uint64(0xFFFFFFFF)


def hnoise(n: int, seed: int) -> np.ndarray:
    """A uniform on [-1/2, 1/2) per index: a 32-bit integer hash of (i, seed)
    over 2^32 (exact)."""
    h = (np.arange(n, dtype=np.uint64) * np.uint64(0x9E3779B1) + np.uint64(seed)) & M32
    h ^= h >> np.uint64(15)
    h = (h * np.uint64(0x85EBCA77)) & M32
    h ^= h >> np.uint64(13)
    h = (h * np.uint64(0xC2B2AE3D)) & M32
    h ^= h >> np.uint64(16)
    return h.astype(np.float64) / 4294967296.0 - 0.5


def noise_of(n: int, scale: float) -> np.ndarray:
    """(u1 + u2 + u3 + u4) + j (u5 + u6 + u7 + u8), times scale (exact sums)."""
    re = ((hnoise(n, 1) + hnoise(n, 2)) + hnoise(n, 3)) + hnoise(n, 4)
    im = ((hnoise(n, 5) + hnoise(n, 6)) + hnoise(n, 7)) + hnoise(n, 8)
    return (re + 1j * im) * scale


def ce_cap(spec, n: int) -> RX.Prepared:
    x = np.where((np.arange(spec.n_data) * 5 + 1) % 3 == 0, 1.0, -1.0)
    sym = frame.assemble(spec, header.encode(HDR) if spec.n_hdr else None, x)
    i = np.arange(n)
    phi = ce.phase_grid(250, -3000, n, sym, spec)
    sig = np.exp(1j * (phi + 2 * np.pi * (0.02 * i / 250 + 0.5 * 0.001 * (i / 250) ** 2)))
    return RX.Prepared(fe=(sig + noise_of(n, 0.75)).astype(np.complex64), keep=None, q=Q,
                       t0_index=3002.5, fs=CH_FS)


def l_cap(spec, n: int) -> RX.Prepared:
    xs = np.where((np.arange(spec.n_data) * 5 + 1) % 3 == 0, 0.5, -0.5)
    sym = lframe.assemble(spec, header.encode(HDR) if spec.n_hdr else None, xs)
    sig = lwave.composite(sym, spec, CH_FS, -3000, n)
    return RX.Prepared(fe=(sig + noise_of(n, 0.75)).astype(np.complex64), keep=None, q=Q,
                       t0_index=3000.0, fs=CH_FS)


def det_of(wf: int) -> Detection:
    return Detection(f_hz=1500.0, path=FLAT, timing=None, z_ref=np.nan, method="tbd", lead_in_s=0.0,
                     waveform=wf)


note("noise_of(17000, 0.75), complex128: the hash noise")
dig("NOISE", noise_of(17000, 0.75))

# --- the passes -----------------------------------------------------------------------------------
IDX = [0, 1, 7, 100, 511, 1023]


def pin_pass(tag: str, prep, spec, **kw) -> None:
    res = RX.receive_pass(prep, spec, det_of(spec.waveform), f_mix_hz=Fraction(0), return_tracks=True, **kw)
    if res is None:
        num(f"{tag}_HAS", 0)
        return
    p, trs = res
    num(f"{tag}_HAS", 1)
    num(f"{tag}_NTRACK", len(trs))
    arr(f"{tag}_TRACK_Q", [t.q for t in trs])
    # the keying a track keeps: cw_known's own (192,) mask on round B, None on A
    arr(f"{tag}_TRACK_KEYING_N", [-1 if t.keying is None else t.keying.size for t in trs])
    num(f"{tag}_PLAIN", int(p.frame != spec.name))
    h = p.header
    num(f"{tag}_HDR", int(h is not None))
    if h is not None:
        text(f"{tag}_HDR_CALL", h.callsign)
        text(f"{tag}_HDR_GRID", h.grid or "")
        iarr(f"{tag}_HDR_INTS", [h.picture_id, h.mode, h.segment, h.codec_id, h.version, h.reserved])
    num(f"{tag}_CW", int(p.cw is not None))
    if p.cw is not None:
        num(f"{tag}_CW_HAS_TEXT", int(p.cw.text is not None))
        text(f"{tag}_CW_TEXT", p.cw.text or "")
        text(f"{tag}_CW_KEYING", p.cw.keying)
        num(f"{tag}_CW_AGREES", -1 if p.cw.agrees is None else int(p.cw.agrees))
        flt(f"{tag}_CW_Z", p.cw.z_match)
    arr(f"{tag}_F", [p.f_hz, p.timing.tau0, p.timing.ppm, p.timing.gamma, p.timing.z])
    r = p.report
    arr(f"{tag}_REPORT", [r.offset_hz, r.drift_hz_per_min, r.wander_hz_rms, r.doppler_hz, r.ppm,
                          r.snr2500_db, r.z_ref, r.kappa])
    iarr(f"{tag}_FLAGS", [int(r.slip_free), int(r.suspect)])
    num(f"{tag}_NZ", len(p.z))
    arr(f"{tag}_Z_AT", p.z[IDX].astype(np.float64))
    arr(f"{tag}_W_AT", p.w[IDX].astype(np.float64))
    flt(f"{tag}_Z_SUM", float(np.sum(p.z.astype(np.float64))))
    flt(f"{tag}_W_SUM", float(np.sum(p.w.astype(np.float64))))
    num(f"{tag}_NLLR", len(p.hdr_llr))
    flt(f"{tag}_LLR_ABS", float(np.sum(np.abs(p.hdr_llr.astype(np.float64)))))
    num(f"{tag}_NPSI", len(p.psi))
    num(f"{tag}_PSI_ZERO", int(np.sum(p.psi == 0)))
    flt(f"{tag}_PSI_SUM", float(np.sum(p.psi.astype(np.float64))))
    flt(f"{tag}_CH_T0", p.ch_t0_index)
    iarr(f"{tag}_F_MIX", [p.f_mix_hz.numerator, p.f_mix_hz.denominator])
    text(f"{tag}_UID_HEAD", p.uid.rsplit("_", 1)[0])
    text(f"{tag}_FRAME", p.frame)
    flt(f"{tag}_MEAN_W_DB", RX.mean_w_db(p))
    arr(f"{tag}_CONF", [math.nan if c is None else c for c in RX.confidence_db(p)])


CAPS = {}
for tag, spec, mk, n in (("TINY", frame.TINY, ce_cap, 17000), ("SHORT", frame.SHORT, ce_cap, 72000),
                         ("LTINY", lframe.L_TINY, l_cap, 17500), ("LSHORT", lframe.L_SHORT, l_cap, 72000)):
    prep = mk(spec, n)
    CAPS[tag] = (prep, spec)
    note(f"the synthetic {spec.name} capture ({n} CH samples; module docstring) and its pass: "
         f"receive_pass(prep, {spec.name}, flat 1500 Hz det, f_mix_hz=Fraction(0), return_tracks=True)")
    dig(f"CAP_{tag}", prep.fe)
    pin_pass(tag, prep, spec)

note("TINY with erase_s=((10.0, 20.0),) and estimator 'plain'; SHORT with round_b=False; l-tiny with "
     "exclude_hz=(1530.0,) and erase_s=((5.0, 12.5),)")
pin_pass("TINY_ERASED", *CAPS["TINY"], erase_s=((10.0, 20.0),), estimator="plain")
pin_pass("SHORT_A", *CAPS["SHORT"], round_b=False)
pin_pass("LTINY_EX", *CAPS["LTINY"], exclude_hz=(1530.0,), erase_s=((5.0, 12.5),))

note("the noise alone as a TINY and an l-tiny capture: gate V refuses")
for tag, spec, t0 in (("NOISE_TINY", frame.TINY, 3002.5), ("NOISE_LTINY", lframe.L_TINY, 3000.0)):
    prep = RX.Prepared(fe=noise_of(17500, 0.75).astype(np.complex64), keep=None, q=Q, t0_index=t0, fs=CH_FS)
    assert RX.receive_pass(prep, spec, det_of(spec.waveform), f_mix_hz=Fraction(0)) is None
    num(f"{tag}_HAS", 0)

note("the waveform refusals: a CE detection as l-tiny, an L one as TINY, and a waveform 7 one")
text("ERR_CE_AS_L", raised(RX.receive_pass, None, lframe.L_TINY, det_of(WAVEFORM_CE)))
text("ERR_L_AS_CE", raised(RX.receive_pass, None, frame.TINY, det_of(WAVEFORM_L)))
text("ERR_7_AS_CE", raised(RX.receive_pass, None, frame.TINY, det_of(7)))

# --- a raw FE capture: prepare and channel_for ----------------------------------------------------
N_FE = 80 * FE_FS
k = np.arange(N_FE)
fe = 0.5 * np.exp(1j * (2 * np.pi * 0.25 * k / FE_FS)) + noise_of(N_FE, 0.25)
fe[100000] += 50.0
fe[200001] += 40.0 - 30.0j
cap = frontend.Capture(fe=fe.astype(np.complex64), q=Q, t0_index=48000.0)
note("a raw FE capture: 0.5 exp(j 2 pi 0.25 k / 4000) + noise_of(320000, 0.25), plus 50 at 100000 and "
     "40 - 30j at 200001, complex64, t0_index 48000; prepare() (blank, normalise), then channel_for at "
     "TINY with the flat det, and with det.timing tau0 250.5 and exclude_hz (1520.0, 1480.5)")
dig("FE_CAP", cap.fe)
prep = RX.prepare(cap)
FE_IDX = [0, 99999, 100000, 150000, 200001, 319999]
iarr("FE_IDX", FE_IDX)
arr("PREP_RE", prep.fe[FE_IDX].real.astype(np.float64))
arr("PREP_IM", prep.fe[FE_IDX].imag.astype(np.float64))
flt("PREP_POWER", float(np.sum(np.abs(prep.fe.astype(np.complex128)) ** 2)))
arr("PREP_KEEP_AT", prep.keep[FE_IDX].astype(np.float64))
flt("PREP_KEEP_SUM", float(np.sum(prep.keep.astype(np.float64))))
chan = RX.channel_for(prep, frame.TINY, det_of(WAVEFORM_CE))
num("CHAN_N", len(chan.ch))
flt("CHAN_T0", chan.t0_index)
iarr("CHAN_MIX", [chan.f_mix.numerator, chan.f_mix.denominator])
CH_IDX = [0, 1, 500, 5000, len(chan.ch) - 1]
iarr("CH_IDX", CH_IDX)
arr("CHAN_RE", chan.ch[CH_IDX].real.astype(np.float64))
arr("CHAN_IM", chan.ch[CH_IDX].imag.astype(np.float64))
flt("CHAN_POWER", float(np.sum(np.abs(chan.ch.astype(np.complex128)) ** 2)))
tm = Timing(tau0=250.5, ppm=0.0, gamma=0.0, z=0.0, cov=np.eye(2))
det_t = Detection(f_hz=1500.0, path=FLAT, timing=tm, z_ref=np.nan, method="tbd", lead_in_s=0.0)
chan2 = RX.channel_for(prep, frame.TINY, det_t, (1520.0, 1480.5))
num("CHAN2_N", len(chan2.ch))
flt("CHAN2_T0", chan2.t0_index)
iarr("CHAN2_MIX", [chan2.f_mix.numerator, chan2.f_mix.denominator])
flt("CHAN2_POWER", float(np.sum(np.abs(chan2.ch.astype(np.complex128)) ** 2)))
chan3 = RX.channel_for(RX.Prepared(fe=np.zeros(10, np.complex64), keep=None, q=Q, t0_index=2.5, fs=CH_FS),
                       frame.TINY, Detection(f_hz=1503.3, path=FLAT, timing=None, z_ref=np.nan, method="tbd",
                                             lead_in_s=0.0))
note("channel_for of a CH-rate capture at a 1503.3 Hz detection: the default mix, quantised")
iarr("CHAN3_MIX", [chan3.f_mix.numerator, chan3.f_mix.denominator])
note("receive_pass of the raw capture at TINY (a carrier, no frame): refused")
assert RX.receive_pass(cap, frame.TINY, det_of(WAVEFORM_CE)) is None

# --- decode_header (bitwise) -----------------------------------------------------------------------
kk = np.arange(2474)


def noisy(h) -> np.ndarray:
    s = 1.0 - 2.0 * header.encode(h).astype(np.float64)
    return 4.0 * s + ((kk * 7) % 19 - 9) * 0.5


garbage = ((kk * 5) % 11 - 5) * 1.0
note("decode_header on LLRs built from header.encode: noisy(h) = 4 (1 - 2 bits) + ((k*7 % 19) - 9) / 2, "
     "garbage = (k*5 % 11) - 5; each case (frame, sum, spread, block) -> header ints (picture_id, mode, "
     "segment, codec_id, version, reserved; -1 for none), block, plain")
CASES = [
    ("short", "noisy2", None, None),
    ("short", "garbage", "noisy2", None),
    ("short", "garbage", "garbage", "noisy1"),
    ("short", "garbage", "garbage", "noisy2"),
    ("short", "garbage", None, "noisy1"),
    ("short", "garbage", None, None),
    ("short", "garbage", "garbage", "garbage"),
    ("short-v1", "noisy1", None, None),
    ("short-v1", "garbage", "noisy2", "noisy1"),
    ("full-spreadonly", "noisy2", None, None),
    ("full-spreadonly", "garbage", "noisy2", "noisy2"),
    ("tiny", "noisy2", None, None),
    ("l-short", "noisy2", None, None),
    ("l-short", "garbage", "noisy2", None),
    ("l-short", "garbage", "garbage", "noisy1"),
    ("l-short", "garbage", "garbage", "noisy2"),
    ("l-short", "noisy1", None, None),
    ("l-tiny", "noisy2", None, None),
]
VEC = {"noisy2": noisy(HDR), "noisy1": noisy(HDR1), "garbage": garbage}
out = []
for name, s_, sp_, bl_ in CASES:
    spec = lframe.get_any(name)
    diag = {}
    if sp_ is not None:
        diag["llr_spread"] = VEC[sp_]
    if bl_ is not None:
        diag["llr_block"] = VEC[bl_]
    llr = VEC[s_].astype(np.float32)
    hdr, kept, sp = RX.decode_header(llr, diag, spec)
    ints = ([-1] * 6 if hdr is None else
            [hdr.picture_id, hdr.mode, hdr.segment, hdr.codec_id, hdr.version, hdr.reserved])
    block = kept is not llr
    plain = sp is not spec
    if block:
        assert kept.dtype == np.float32 and np.array_equal(kept, VEC[bl_].astype(np.float32))
    if plain:
        assert sp == frame.plain(spec)
    out += ints + [int(block), int(plain)]
    if hdr is not None:
        assert hdr.callsign.rstrip() in ("K1ABC", "W1AW")
iarr("HDR_CASES", out)
lines.append("inline constexpr const char* HDR_CASE_FRAMES[] = {"
             + ", ".join(f'"{c[0]}"' for c in CASES) + "};")
lines.append("inline constexpr const char* HDR_CASE_VECS[] = {"
             + ", ".join(f'"{c[1]}", "{c[2] or ""}", "{c[3] or ""}"' for c in CASES) + "};")

# --- confidence_db and mean_w_db (bitwise) --------------------------------------------------------


def synthetic_pass(spec) -> PassResult:
    nd, npos = spec.n_data, spec.n_pos
    i = np.arange(nd)
    w = np.where(i % 13 == 0, 0.0, ((i * 5) % 7) / 4.0 + 0.125).astype(np.float32)
    j = np.arange(2474)
    llr = (((j * 3) % 23 - 11) * 0.75).astype(np.float32)
    ps = np.arange(npos)
    psi = np.where((ps // 50) % 9 == 4, 0.0, 1.0).astype(np.float32)
    tm = Timing(tau0=0.0, ppm=0.0, gamma=0.0, z=0.0, cov=np.eye(2))
    rep = TrackReport(offset_hz=1500.0, drift_hz_per_min=0.0, wander_hz_rms=0.0, doppler_hz=0.0, ppm=0.0,
                      snr2500_db=0.0, z_ref=0.0, kappa=1.0, slip_free=True, suspect=False)
    return PassResult(uid="u", q=Q, frame=spec.name, waveform=spec.waveform, f_hz=1500.0, timing=tm,
                      report=rep, z=np.zeros(nd, np.float32), w=w, hdr_llr=llr, header=None, cw=None,
                      ch=np.zeros(0, np.complex64), ch_fs=CH_FS, ch_t0_index=0.0, f_mix_hz=Fraction(0),
                      psi=psi, estimator="joint")


note("confidence_db of a synthetic pass: w = (i*5 % 7)/4 + 1/8, 0 at every 13th; hdr_llr = "
     "((j*3 % 23) - 11) * 3/4; psi 0 on every ninth run of 50 positions (from the fifth); for SHORT and "
     "l-short, with (elapsed, n) in CONF_CASES; NaN for None")
CONF_CASES = [(None, 120), (100.0, 120), (None, 7), (1e9, 1), (0.0, 120)]
arr("CONF_ELAPSED", [math.nan if e is None else e for e, _ in CONF_CASES])
iarr("CONF_N", [n for _, n in CONF_CASES])
for tag, name in (("SHORT", "short"), ("LSHORT", "l-short"), ("TINY", "tiny"), ("FULL", "full")):
    p = synthetic_pass(lframe.get_any(name))
    for c, (e, n) in enumerate(CONF_CASES):
        arr(f"CONF_{tag}_{c}", [math.nan if v is None else v for v in RX.confidence_db(p, None, e, n)])
    flt(f"MEAN_W_DB_{tag}", RX.mean_w_db(p))
    p_nopsi = synthetic_pass(lframe.get_any(name))
    p_nopsi.psi = None
    p_nopsi.hdr_llr = None
    arr(f"CONF_{tag}_BARE", [math.nan if v is None else v for v in RX.confidence_db(p_nopsi)])
note("confidence_db's refusals: n = -1 (np.zeros), n = -2 (np.linspace), and n = 0 with latents to "
     "place (np.bincount)")
ps = synthetic_pass(frame.SHORT)
text("ERR_CONF_M1", raised(RX.confidence_db, ps, None, None, -1))
text("ERR_CONF_M2", raised(RX.confidence_db, ps, None, None, -2))
text("ERR_CONF_0", raised(RX.confidence_db, ps, None, None, 0))
note("mean_w_db of an all-zero w (the floor) and of [1e-31, 3.0, 0.5]")
flt("MEAN_W_DB_ZERO", RX.mean_w_db(SimpleNamespace(w=np.zeros(5))))
flt("MEAN_W_DB_MIX", RX.mean_w_db(SimpleNamespace(w=np.array([1e-31, 3.0, 0.5]))))

# --- small things ----------------------------------------------------------------------------------
note("centre_hz of a CE and an L signal at 1500.25 Hz; detection_from_pass's path at offset 1500.5, "
     "drift 0.75 Hz/min")
flt("CENTRE_CE", RX.centre_hz(SimpleNamespace(f_hz=1500.25)))
flt("CENTRE_L", RX.centre_hz(SimpleNamespace(waveform=WAVEFORM_L, f_hz=1500.25)))
p = synthetic_pass(frame.SHORT)
p.report = TrackReport(offset_hz=1500.5, drift_hz_per_min=0.75, wander_hz_rms=0.0, doppler_hz=0.0, ppm=0.0,
                       snr2500_db=0.0, z_ref=12.5, kappa=1.0, slip_free=True, suspect=False)
d = RX.detection_from_pass(p, 2.5)
arr("DFP_F", d.path.f_hz)
arr("DFP_T", d.path.t_s)

head = """// Generated by native/tests/qrss/gen_receiver_pins.py from the Python
// reference (sstvae/qrss/receiver.py). Do not edit: re-run the script.

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace receiver_pins {

inline constexpr double NAN_PIN = std::numeric_limits<double>::quiet_NaN();
"""
OUT.write_text(head + "\n".join(lines) + "\n\n}  // namespace receiver_pins\n")
print(f"wrote {OUT} ({len(lines)} lines)")
