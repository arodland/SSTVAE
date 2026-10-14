#!/usr/bin/env python3
"""The pins of native/tests/qrss/test_em_refine.cpp (em.py's pass_path,
rereceive and em_refine), from the Python reference.

Writes native/tests/qrss/em_refine_pins.hpp, which is committed so the
ctest needs no Python. Re-run it from the repository root whenever
sstvae/qrss/em.py (or anything under it: the receiver, the tracker, the
demodulator, the store) changes on purpose:

    PYTHONPATH=. python native/tests/qrss/gen_em_refine_pins.py

The fixture is gen_receiver_pins.py's synthetic CH-rate capture -- a
stored pass's form (fs 250), so every re-receive takes the CH branch of
channel_for -- built here and in the ctest from the same formulas, with
three changes: the data symbols are precode(a, q) of one latent vector
a[k] = +-1 ((5 k + 1) % 3 == 0 gives +1), so the passes of one picture
agree and their accumulator is a real reference; pass k (k = 0, 1, 2)
sits in slot q = Q + 7 k with its own noise, noise_of(n, 1.0, k) taking
the hash seeds 8 k + 1 .. 8 k + 8; and the noise is 1.0, not 0.75 (about
-8.5 dB in 2500 Hz). Each capture is received by receive_pass(prep,
spec, flat 1500 Hz det, f_mix_hz=Fraction(0)) and the TINY ones stored
and attached, in k order, to the accumulator ("K1ABC", 0x12345678)
(segment 0, mode 0, codec 0xD1D8) of a store whose clock reads T0.

What is pinned (each a value of the reference, the C++ held to its
primitives' rounding: the tracker, the fit and the demodulator agree to
~1e-7, so a value is held to 1e-5 of max(1, |value|), the decisions
exactly):

  * pass_path on TINY pass 0: both candidates' Z_ref, which won, the path
    chosen (its length, first and last time, frequencies at a few indices
    and their sum);
  * rereceive of TINY pass 0 without a prior (the receive_pass branch),
    and with em_prior(acc, pass 0) (its leave-one-out prior, split): the
    pass (f_hz, timing, report, flags, z and w at a few indices and their
    sums, psi's sum, hdr_llr's length, estimator, em_round) and the path
    it ended on (length and the frequency sum);
  * rereceive of a SHORT pass (the test header, the callsign windows
    unkeyed) with the formula prior prior_from(S, W, q) (S = 2 a + 0.25
    noise-free offsets, W = 2: exact in both), the header and the callsign
    keying (keying_units("K1ABC"), FSK): split, no split, and with no
    prior (one track, no split);
  * em_refine(store, key, rounds=3) on the three TINY passes: the history,
    the log lines (their numbers to 0.011 dB, printed to 0.01), each
    member's em_round and mean W afterwards; and with rounds=0;
  * refusals: pass_path of an L pass, em_refine of a key with no
    accumulator (their texts).

A pass uid is never pinned: it hashes the pass's float32 z, which the two
implementations agree on only to rounding.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from sstvae.qrss import ce, em, frame, header, morse  # noqa: E402
from sstvae.qrss import receiver as RX  # noqa: E402
from sstvae.qrss import store as S  # noqa: E402
from sstvae.qrss import track as T  # noqa: E402
from sstvae.qrss.constants import CH_FS, WAVEFORM_CE, WAVEFORM_L  # noqa: E402
from sstvae.qrss.precoder import precode  # noqa: E402
from sstvae.qrss.types import Detection, FreqPath, PassResult  # noqa: E402

OUT = Path(__file__).with_name("em_refine_pins.hpp")
lines: list[str] = []

HDR = header.HeaderFields(callsign="K1ABC", grid="FN42", picture_id=0x12345678, mode=0,
                          segment=0, codec_id=0xD1D8)
KEY = ("K1ABC", 0x12345678)
Q = 1954752
T0 = 1760000000.0
NOISE = 1.0
FLAT = FreqPath(t_s=np.array([-10.0, 0.0, 1800.0]), f_hz=np.full(3, 1500.0), weight=np.ones(3))
M32 = np.uint64(0xFFFFFFFF)
IDX = [0, 1, 7, 100, 511, 1023]


def flt(name: str, x) -> None:
    lines.append(f"inline constexpr double {name} = {float(x).hex()};")


def num(name: str, x) -> None:
    lines.append(f"inline constexpr std::int64_t {name} = {int(x)};")


def arr(name: str, xs) -> None:
    xs = [float(x) for x in xs]
    assert all(math.isfinite(x) for x in xs), name
    lines.append(f"inline constexpr double {name}[] = {{{', '.join(x.hex() for x in xs)}}};")
    lines.append(f"inline constexpr std::size_t {name}_N = {len(xs)};")


def text(name: str, s: str) -> None:
    esc = s.replace("\\", "\\\\").replace('"', '\\"')
    lines.append(f'inline constexpr const char* {name} = "{esc}";')


def dig(name: str, a) -> None:
    a = np.ascontiguousarray(a)
    lines.append(f'inline constexpr const char* {name} = "{hashlib.sha256(a.tobytes()).hexdigest()}";')


def note(t: str) -> None:
    lines.append("")
    lines.append(f"// {t}")


def raised(fn, *a, **k) -> str:
    try:
        fn(*a, **k)
    except Exception as e:  # noqa: BLE001 - the reference's own text, whatever its type
        return str(e)
    raise AssertionError("did not raise")


# --- the fixture ------------------------------------------------------------------------------

def hnoise(n: int, seed: int) -> np.ndarray:
    h = (np.arange(n, dtype=np.uint64) * np.uint64(0x9E3779B1) + np.uint64(seed)) & M32
    h ^= h >> np.uint64(15)
    h = (h * np.uint64(0x85EBCA77)) & M32
    h ^= h >> np.uint64(13)
    h = (h * np.uint64(0xC2B2AE3D)) & M32
    h ^= h >> np.uint64(16)
    return h.astype(np.float64) / 4294967296.0 - 0.5


def noise_of(n: int, scale: float, k: int) -> np.ndarray:
    s = 8 * k
    re_ = ((hnoise(n, s + 1) + hnoise(n, s + 2)) + hnoise(n, s + 3)) + hnoise(n, s + 4)
    im_ = ((hnoise(n, s + 5) + hnoise(n, s + 6)) + hnoise(n, s + 7)) + hnoise(n, s + 8)
    return (re_ + 1j * im_) * scale


def air(spec) -> np.ndarray:
    return np.where((np.arange(spec.n_data) * 5 + 1) % 3 == 0, 1.0, -1.0)


def cap(spec, n: int, k: int) -> RX.Prepared:
    q = Q + 7 * k
    sym = frame.assemble(spec, header.encode(HDR) if spec.n_hdr else None, precode(air(spec), q))
    i = np.arange(n)
    phi = ce.phase_grid(250, -3000, n, sym, spec)
    sig = np.exp(1j * (phi + 2 * np.pi * (0.02 * i / 250 + 0.5 * 0.001 * (i / 250) ** 2)))
    return RX.Prepared(fe=(sig + noise_of(n, NOISE, k)).astype(np.complex64), keep=None, q=q,
                       t0_index=3002.5, fs=CH_FS)


def det_of() -> Detection:
    return Detection(f_hz=1500.0, path=FLAT, timing=None, z_ref=np.nan, method="tbd", lead_in_s=0.0,
                     waveform=WAVEFORM_CE)


def receive(spec, n, k) -> PassResult:
    prep = cap(spec, n, k)
    p = RX.receive_pass(prep, spec, det_of(), f_mix_hz=Fraction(0))
    assert p is not None, (spec.name, k)
    return prep, p


def pin_pass(tag: str, p: PassResult, path: FreqPath | None = None) -> None:
    arr(f"{tag}_F", [p.f_hz, p.timing.tau0, p.timing.ppm, p.timing.gamma, p.timing.z])
    r = p.report
    arr(f"{tag}_REPORT", [r.offset_hz, r.drift_hz_per_min, r.wander_hz_rms, r.doppler_hz, r.ppm,
                          r.snr2500_db, r.z_ref, r.kappa])
    num(f"{tag}_FLAGS", int(r.slip_free) + 2 * int(r.suspect))
    num(f"{tag}_NZ", len(p.z))
    arr(f"{tag}_Z_AT", p.z[IDX].astype(np.float64))
    arr(f"{tag}_W_AT", p.w[IDX].astype(np.float64))
    flt(f"{tag}_Z_SUM", float(np.sum(p.z.astype(np.float64))))
    flt(f"{tag}_W_SUM", float(np.sum(p.w.astype(np.float64))))
    flt(f"{tag}_PSI_SUM", float(np.sum(p.psi.astype(np.float64))))
    num(f"{tag}_NLLR", len(p.hdr_llr))
    flt(f"{tag}_LLR_ABS", float(np.sum(np.abs(p.hdr_llr.astype(np.float64)))))
    text(f"{tag}_ESTIMATOR", p.estimator)
    num(f"{tag}_EM_ROUND", p.em_round)
    num(f"{tag}_HDR", int(p.header is not None))
    flt(f"{tag}_MEAN_W_DB", RX.mean_w_db(p))
    if path is not None:
        num(f"{tag}_PATH_N", len(path.t_s))
        arr(f"{tag}_PATH_T", [path.t_s[0], path.t_s[-1]])
        flt(f"{tag}_PATH_F_SUM", float(np.sum(path.f_hz)))


N_TINY, N_SHORT = 17000, 72000
note("the TINY captures, k = 0, 1, 2 (module docstring), and the SHORT one (k = 0)")
tiny = [receive(frame.TINY, N_TINY, k) for k in range(3)]
for k, (prep, _p) in enumerate(tiny):
    dig(f"CAP_TINY_{k}", prep.fe)
short_prep, short = receive(frame.SHORT, N_SHORT, 0)
dig("CAP_SHORT", short_prep.fe)
for k, (_prep, p) in enumerate(tiny):
    flt(f"TINY_{k}_MEAN_W_DB", RX.mean_w_db(p))
num("SHORT_HAS_HDR", int(short.header is not None))

p0 = tiny[0][1]
note("pass_path(TINY pass 0): Z_ref of the line and of its refinement, which won (0 the line), and the path")
spec0 = em.pass_spec(p0)
det0 = RX.detection_from_pass(p0)
chan0 = RX.channel_for(RX.capture_from_pass(p0), spec0, det0, f_mix_hz=p0.f_mix_hz)
line = T.line_path(chan0, spec0, em._flat(p0.f_hz))
refined = T.refine_path(chan0, spec0, line)
trl = T.verify_track(chan0, spec0, dataclasses.replace(det0, path=line), fit=False)
trr = T.verify_track(chan0, spec0, dataclasses.replace(det0, path=refined), fit=False)
assert trl is not None and trr is not None
zl, zr = trl.z_ref, trr.z_ref
best = em.pass_path(p0)
choice = 1 if zr > zl else 0
assert np.array_equal(best.f_hz, (line, refined)[choice].f_hz)
arr("PP_Z", [zl, zr])
num("PP_CHOICE", choice)
num("PP_N", len(best.t_s))
arr("PP_T", [best.t_s[0], best.t_s[-1]])
PP_IDX = [0, 1, len(best.t_s) // 2, len(best.t_s) - 1]
lines.append(f"inline constexpr std::size_t PP_IDX[] = {{{', '.join(str(i) for i in PP_IDX)}}};")
arr("PP_F_AT", best.f_hz[PP_IDX])
flt("PP_F_SUM", float(np.sum(best.f_hz)))

with tempfile.TemporaryDirectory() as tmp:
    st = S.Store(Path(tmp) / "qrss", clock=lambda: T0)
    for _prep, p in tiny:
        st.add_pass(p)
        st.attach(KEY, p.uid, 0, "test", mode=0, codec_id=0xD1D8)
    acc = st.accumulator(KEY)
    flt("ACC_MEAN_W_DB", em.acc_mean_w_db(acc))

    note("rereceive(TINY pass 0, None, return_path=True): the receive_pass branch, its own path")
    out, pth = em.rereceive(p0, None, return_path=True)
    pin_pass("RR_PLAIN", out, pth)
    note("rereceive(TINY pass 0, em_prior(acc, pass 0), em_round=1, return_path=True)")
    prior0 = em.em_prior(acc, p0.uid, st, p0.q, spec0)
    flt("PRIOR0_V_SUM", float(np.sum(prior0.v)))
    flt("PRIOR0_X_SUM", float(np.sum(prior0.x_hat)))
    out, pth = em.rereceive(p0, prior0, em_round=1, return_path=True)
    pin_pass("RR_PRIOR", out, pth)

    note("SHORT: the formula prior, the header, the keying; split, no split, and no prior")
    a = air(frame.SHORT)
    W = np.full(frame.SHORT.n_data, 2.0)
    Sv = W * (a + np.where(np.arange(frame.SHORT.n_data) % 4 == 0, 0.25, -0.25))
    pr_s = em.prior_from(Sv, W, short.q)
    cw = (morse.keying_units("K1ABC"), False)
    for tag, kw in (("RR_SHORT_SPLIT", {"prior": pr_s}), ("RR_SHORT_NOSPLIT", {"prior": pr_s, "split": False}),
                    ("RR_SHORT_NOPRIOR", {"prior": None})):
        prior = kw.pop("prior")
        out, pth = em.rereceive(short, prior, HDR, cw, None, 2, return_path=True, **kw)
        pin_pass(tag, out, pth)

    note("em_refine(store, KEY, rounds=3): the history, the log, each member's em_round and mean W after")
    log: list[str] = []
    hist = em.em_refine(st, KEY, 3, log=log.append)
    arr("EMR_HIST", hist)
    num("EMR_NLOG", len(log))
    pat = re.compile(r"^round (\d+) pass (\S+): mean W ([+-]\d+\.\d\d) -> ([+-]\d+\.\d\d) dB$")
    uids = list(st.accumulator(KEY).uids)
    rounds, members, before, after = [], [], [], []
    for x in log:
        m = pat.match(x)
        assert m, x
        rounds.append(int(m.group(1)))
        members.append(uids.index(m.group(2)))
        before.append(float(m.group(3)))
        after.append(float(m.group(4)))
    lines.append(f"inline constexpr std::int64_t EMR_LOG_ROUND[] = {{{', '.join(map(str, rounds))}}};")
    lines.append(f"inline constexpr std::int64_t EMR_LOG_MEMBER[] = {{{', '.join(map(str, members))}}};")
    arr("EMR_LOG_BEFORE", before)
    arr("EMR_LOG_AFTER", after)
    acc = st.accumulator(KEY)
    lines.append(f"inline constexpr std::int64_t EMR_EM_ROUND[] = "
                 f"{{{', '.join(str(int(m['em_round'])) for m in acc.members)}}};")
    arr("EMR_MEMBER_W_DB", [m["mean_w_db"] for m in acc.members])
    note("em_refine(store, KEY, rounds=0): the history alone")
    arr("EMR0_HIST", em.em_refine(st, KEY, 0))

    note("refusals: em_refine of a key with no accumulator")
    text("ERR_NO_ACC", raised(em.em_refine, st, ("NOPE", 7)))

note("refusals: pass_path of an L pass (l-tiny, otherwise pass 0)")
pl = dataclasses.replace(p0, frame="l-tiny", waveform=WAVEFORM_L)
text("ERR_L_PATH", raised(em.pass_path, pl))
assert em.rereceive(pl, None) is None

HEAD = f'''// Generated by native/tests/qrss/gen_em_refine_pins.py from the Python reference
// (sstvae/qrss/em.py's pass_path, rereceive and em_refine). Do not edit.
// The formulas are in that script's docstring.
#pragma once

#include <cstddef>
#include <cstdint>

namespace em_refine_pins {{

inline constexpr std::int64_t Q = {Q};
inline constexpr double T0 = {T0.hex()};
inline constexpr double NOISE = {NOISE.hex()};
inline constexpr std::int64_t N_TINY = {N_TINY};
inline constexpr std::int64_t N_SHORT = {N_SHORT};
inline constexpr std::size_t IDX[] = {{{', '.join(map(str, IDX))}}};
'''
OUT.write_text(HEAD + "\n".join(lines) + "\n\n}  // namespace em_refine_pins\n")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
