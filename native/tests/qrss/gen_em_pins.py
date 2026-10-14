#!/usr/bin/env python3
"""Writes native/tests/qrss/em_pins.hpp: what test_em.cpp and test_special.cpp
check the C++ port of sstvae/qrss/em.py's shared pieces (and the special
functions under its tail statistics) against, with no Python at test time.

  * special: ndtri and norm.isf (scipy 1.18.1) on q = u 2^-j, j = 0, 40, ...,
    1000, and on 1 - u (u = unit(splitmix(400 + j, 32)), infra_testgen.hpp),
    through one SHA-256 of both arrays' bytes (bit for bit: cephes' ndtri
    calls only the C library's log and sqrt); math.lgamma (CPython's own
    Lanczos) on four ranges the same way; gamma.logsf(x, a) as a table of
    values checked to 1e-12 relative (numpy's SIMD logs inside scipy may
    differ from the C library's in the last place).
  * the tail statistics: _gauss_z(log_p) and template_z(lam, K, n_cells) as
    tables (1e-12 relative; their refusals as texts).
  * the references: lmmse_latents and prior_from on formula (S, W) (u from
    splitmix, a NaN and an infinity placed in some), each case's x_hat and v
    hashed with NaN written as numpy's np.nan (0x7ff8000000000000), so a
    NaN's payload is not part of the pin.
  * the store-backed pieces: formula passes (below) stored and attached to
    one accumulator per scenario, and each scenario's queries (em_prior,
    acc_mean_w_db, known_header, known_keying) as result texts. A pass uid
    is never pinned: em_prior's refusal names it, and both sides write it
    as <uid>.

A pass of picture `pic` holds n latents, at air position i

    a = ((h32(i + 1000003 pic + 77) >> 27) - 15.5) / 8
    z = a + ((h32(i + 1000003 (seed + 7)) >> 28) - 7.5) / 8    where w > 0, else 0

(h32: gen_store_pins.h32, lowbias32), weight w up to erase_from and 0 after,
q and f_hz as listed, frame "tiny"/"short" by n ("l-tiny"/"l-short" for
waveform L). Its header LLRs (over the encoded header `hdr`) are

    zero     2474 zeros
    strong   4 (1 - 2 bit)
    weak     0.1875 (1 - 2 bit) + ((h32(i + 9000 + 13 seed) >> 24) - 127.5) / 64
    short    ten ones (not a header's length)

all exact in float32 (multiples of 1/128): one weak pass, or two, or three
summed, decode to nothing, six to the header. `decoded` reports `hdr` as
decoded on the pass; `cw` is (keying, z_match, agrees) or None.

    PYTHONPATH=. python native/tests/qrss/gen_em_pins.py      # rewrites the header
"""

import dataclasses
import hashlib
import math
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import special, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sstvae.qrss import em, frame, header, lframe  # noqa: E402
from sstvae.qrss import store as S  # noqa: E402
from sstvae.qrss.constants import N_HDR_BITS, WAVEFORM_CE, WAVEFORM_L  # noqa: E402
from sstvae.qrss.types import CwIdResult, PassResult, Timing, TrackReport, pass_uid  # noqa: E402

OUT = Path(__file__).with_name("em_pins.hpp")
M32 = 0xFFFFFFFF
U64 = (1 << 64) - 1
WEAK_A = 0.1875
T0 = 1760000000.0
Q0 = 1955555


# --- the formulas ---------------------------------------------------------------------------

def splitmix(seed, n):
    with np.errstate(over="ignore"):
        s = np.uint64(seed) + np.arange(1, n + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        z = (s ^ (s >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def unit(z):
    return (z >> np.uint64(11)).astype(np.float64) * 2.0 ** -53


def h32(x):
    x = np.asarray(x, dtype=np.uint64) & np.uint64(M32)
    x ^= x >> np.uint64(16)
    x = (x * np.uint64(0x7FEB352D)) & np.uint64(M32)
    x ^= x >> np.uint64(15)
    x = (x * np.uint64(0x846CA68B)) & np.uint64(M32)
    x ^= x >> np.uint64(16)
    return x


def canon_nan(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.isnan(x), np.nan, x)


def sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()


# --- special ----------------------------------------------------------------------------------

NDTRI_J = list(range(0, 1001, 40))


def ndtri_inputs():
    qs = [unit(splitmix(400 + j, 32)) * 2.0 ** -j for j in NDTRI_J]
    qs.append(1.0 - unit(splitmix(450, 64)))
    return np.concatenate(qs)


def ndtri_digest():
    q = ndtri_inputs()
    return sha(special.ndtri(q), stats.norm.isf(q))


def lgamma_inputs():
    return np.concatenate([unit(splitmix(500, 256)) * 60.0 - 20.0, unit(splitmix(501, 64)) * 2e6,
                           unit(splitmix(502, 64)) * 1e-3, np.arange(1, 201, dtype=np.float64) * 0.5 + 0.25])


def lgamma_digest():
    return sha(np.array([math.lgamma(float(x)) for x in lgamma_inputs()]))


LOGSF_A = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 33.0, 100.0, 300.0, 891.0, 1000.0]
LOGSF_X = [0.01, 0.3, 0.9, 1.0, 1.1, 2.0, 5.0, 20.0]


def logsf_table():
    rows = []
    for a in LOGSF_A:
        xs = [a * f for f in LOGSF_X] + [a + 40.0 * math.sqrt(a) + 40.0, float(special.gammaincinv(a, 0.5))]
        for x in xs:
            rows.append((x, a, float(stats.gamma.logsf(x, a))))
    return rows


# --- the tail statistics ------------------------------------------------------------------------

GAUSS_LOG_P = [3.0, 0.0, -0.1, math.log(0.5), math.nextafter(math.log(0.5), -1.0), -0.7, -1.0, -2.5, -5.0,
               -10.0, -23.025850929940457, -50.0, -100.0, -300.0, -699.9, -700.0, -700.0000001, -745.0,
               -1000.0, -1e4, -1e6, -math.inf, math.nan]


def gauss_table():
    return [(lp, em._gauss_z(lp)) for lp in GAUSS_LOG_P]


TEMPLATE = [
    (33.0, 33, 1.0), (50.0, 33, 1.0), (60.0, 33, 1e6), (80.0, 33, 4.2e5), (120.0, 33, 1e7),
    (10.0, 1, 1.0), (1.0, 1, 1.0), (0.5, 4, 100.0), (25.0, 4, 3000.0), (150.0, 66, 1e5),
    (3000.0, 33, 1e6), (5000.0, 33, 1.0), (40000.0, 900, 1e8), (2e5, 33, 1e6),
    (0.0, 33, 1e6), (-3.0, 33, 1.0), (40.0, 33, 0.5), (40.0, 33, -math.inf), (40.0, 33, math.inf),
    (40.0, 33, math.nan), (math.nan, 33, 1.0), (math.inf, 33, 1.0),
    (40.0, 0, 1.0), (40.0, -2, 1.0), (0.0, 0, 1.0),
]


def template_table():
    rows = []
    for lam, k, cells in TEMPLATE:
        try:
            rows.append((lam, k, cells, em.template_z(lam, k, cells), ""))
        except (ValueError, OverflowError) as e:
            rows.append((lam, k, cells, 0.0, str(e)))
    return rows


# --- the references -----------------------------------------------------------------------------

PRIOR_N = [7, 33, 1000, 1024, 4096, 50600]
PRIOR_Q = [0, 1, Q0, 2 ** 40, U64]


def prior_case(k, n):
    s = (unit(splitmix(600 + k, n)) - 0.5) * 8.0
    w = unit(splitmix(700 + k, n)) * 5.0 - 1.0
    if k % 2 == 1:
        s[3] = np.nan
    if k % 3 == 0:
        w[5] = np.inf
    if k == 4:
        w[2] = np.nan
    return s, w


def prior_table():
    rows = []
    for k, n in enumerate(PRIOR_N):
        s, w = prior_case(k, n)
        a, r = em.lmmse_latents(s, w)
        lm = sha(canon_nan(a), canon_nan(r))
        for q in PRIOR_Q:
            p = em.prior_from(s, w, q)
            rows.append((k, n, q, lm, sha(canon_nan(p.x_hat), canon_nan(p.v))))
    return rows


def prior_refusals():
    out = []
    for s_n, w_n, q in ((3, 2, 0), (4, 4, -1), (2, 3, -1), (1, 5, 0), (5, 1, 0)):
        s, w = np.ones(s_n), np.ones(w_n)
        try:
            p = em.prior_from(s, w, q)
            out.append((s_n, w_n, q, sha(p.x_hat, p.v)))
        except ValueError as e:
            out.append((s_n, w_n, q, "error: " + str(e)))
    return out


# --- the store-backed pieces --------------------------------------------------------------------

PICS = [
    # callsign, grid, picture_id, mode, codec_id
    ("K1ABC", "FN42", 0xA1B2C3D4, 2, 0xD1D8),
    ("W9XYZ/P", None, 0x00000042, 1, 0xD1D8),
    ("G4ABC", "IO91", 0x0BADCAFE, 2, 0x1234),
]


@dataclasses.dataclass(frozen=True)
class P:
    name: str
    n: int
    pic: int
    seed: int
    q: int
    w: float = 1.0
    waveform: int = WAVEFORM_CE
    f_hz: float = 1500.0
    erase_from: int = -1
    hdr_pic: int = -1          # -1: no header was sent (llr "zero" or "short" only)
    hdr_seg: int = 0
    llr: str = "zero"
    decoded: bool = False
    cw_keying: str = ""        # "": no cw
    cw_z: float = 8.0
    cw_agrees: int = 1         # 1 True, 0 False, -1 None


PASSES = [
    # em_prior / acc_mean_w_db
    P("p0", 4096, 0, 1, Q0 - 1, w=0.5),
    P("p1", 4096, 0, 2, Q0 - 2, w=1.5),
    P("p2", 4096, 0, 3, Q0 - 3, w=2.5, erase_from=1000),
    P("t0", 1024, 0, 4, Q0 - 4, w=1.0),
    P("o0", 4096, 0, 9, Q0 - 9, w=1.0),
] + [
    # known_header
    P(f"w{k}", 1024, 0, 10 + k, Q0 - 10 - k, w=0.5, hdr_pic=0, llr="weak") for k in range(6)
] + [
    P(f"lw{k}", 1024, 0, 20 + k, Q0 - 20 - k, w=0.5, waveform=WAVEFORM_L, f_hz=1509.5, hdr_pic=0, llr="weak")
    for k in range(6)
] + [
    P("d1", 1024, 0, 30, Q0 - 30, hdr_pic=1, llr="strong", decoded=True),
    P("d1s1", 1024, 0, 31, Q0 - 31, hdr_pic=1, hdr_seg=1, llr="strong", decoded=True),
    P("fd", 1024, 0, 32, Q0 - 32, hdr_pic=2, llr="strong", decoded=True),
    P("fl0", 1024, 0, 40, Q0 - 40, hdr_pic=2, llr="strong"),
    P("fl1", 1024, 0, 41, Q0 - 41, hdr_pic=2, llr="weak"),
    P("n0", 1024, 0, 50, Q0 - 50),
    P("n1", 1024, 0, 51, Q0 - 51),
    P("sh", 1024, 0, 60, Q0 - 60, llr="short"),
    P("st", 1024, 0, 61, Q0 - 61, hdr_pic=0, llr="strong"),
    P("d0", 1024, 0, 62, Q0 - 62, hdr_pic=0, llr="strong", decoded=True),
]

CW_VARIANTS = {
    "fsk": [("fsk", 8.0, 1)],
    "ook": [("ook", 8.0, 1)],
    "ook2-fsk1": [("ook", 8.0, 1), ("ook", 8.0, 1), ("fsk", 8.0, 1)],
    "tie": [("ook", 8.0, 1), ("fsk", 8.0, 1)],
    "weak": [("ook", 6.0, 1), ("ook", math.nan, 1)],
    "disagree": [("ook", 8.0, 0), ("ook", 8.0, -1)],
    "none": [("", 8.0, 1), ("", 8.0, 1)],
    "other-keying": [("xyz", 8.0, 1), ("xyz", 8.0, 1)],
    "other+fsk": [("xyz", 8.0, 1), ("xyz", 8.0, 1), ("fsk", 8.0, 1)],
}
for _name, _cws in CW_VARIANTS.items():
    for _wfm in ("ce", "l", "mixed"):
        for _k, (_keying, _z, _agrees) in enumerate(_cws):
            _wf = (WAVEFORM_CE, WAVEFORM_L)[_k % 2] if _wfm == "mixed" else (WAVEFORM_L if _wfm == "l" else WAVEFORM_CE)
            _seed = 100 + 10 * len(PASSES) + _k
            PASSES.append(P(f"cw-{_name}-{_wfm}-{_k}", 1024, 0, _seed, Q0 - 100 - _k, waveform=_wf,
                            f_hz=1509.5 if _wf == WAVEFORM_L else 1500.0, cw_keying=_keying, cw_z=_z,
                            cw_agrees=_agrees))
BY_NAME = {p.name: p for p in PASSES}
assert len(BY_NAME) == len(PASSES)

H_SEGS = [0, 1, 2, -1, 5]
CALLS = [None, "K1ABC", "K1ABC  ", "k1abc", "", "TOOLONGCALL", "K1/ABC", "K 1", "00000000"]
WAVEFORMS = [WAVEFORM_CE, WAVEFORM_L, 7]
PRIOR_SPECS = ["short", "tiny", "l-short"]

# name, key (PICS index, or -1 - k: ("K1ABC", 5000 + k)), attach mode, members [(pass, segment)], queries
SCENARIOS = [
    ("prior", 0, 2, [("p0", 0), ("p1", 0), ("p2", 0), ("t0", 0)], ["prior", "mean_w"]),
    ("soft", 0, 2, [(f"w{k}", 0) for k in range(6)], ["header"]),
    ("soft-L", 0, 2, [(f"lw{k}", 0) for k in range(6)], ["header"]),
    ("soft-mixed", 0, 2, [(f"w{k}", 0) for k in range(3)] + [(f"lw{k}", 0) for k in range(3)], ["header"]),
    ("soft-split", 0, 2, [(f"w{k}", k % 2) for k in range(6)], ["header"]),
    ("soft-short", 0, 2, [("w0", 0), ("w1", 0)], ["header"]),
    # six that decode summed, as three segments of two that do not: the soft header is per segment
    ("soft-split3", 0, 2, [(f"w{k}", k // 2) for k in range(6)], ["header"]),
    ("decoded", 1, 2, [("d1", 0)], ["header"]),
    ("decoded-later", 1, 2, [("n0", 0), ("d1s1", 1)], ["header"]),
    ("decoded-after-soft", 0, 2, [(f"w{k}", 0) for k in range(6)] + [("d0", 1)], ["header"]),
    ("foreign", 0, 2, [("fd", 0)], ["header"]),
    ("foreign-llr", 0, 2, [("fl0", 0), ("fl1", 0)], ["header"]),
    ("foreign+soft", 0, 2, [("fd", 0)] + [(f"w{k}", 0) for k in range(6)], ["header"]),
    ("nothing", 0, 2, [("n0", 0), ("n1", 0), ("sh", 0)], ["header", "mean_w"]),
    ("strong-after-weak", 0, 2, [("w0", 1), ("st", 0)], ["header"]),
]
for _k, _name in enumerate(CW_VARIANTS):
    for _j, _wfm in enumerate(("ce", "l", "mixed")):
        _members = [(f"cw-{_name}-{_wfm}-{i}", 0) for i in range(len(CW_VARIANTS[_name]))]
        SCENARIOS.append((f"cw-{_name}-{_wfm}", -1 - (3 * _k + _j), 2, _members, ["keying"]))


def key_of(k):
    if k >= 0:
        return (PICS[k][0], PICS[k][2])
    return ("K1ABC", 5000 + (-1 - k))


def hdr_of(pic, segment):
    call, grid, pid, mode, codec = PICS[pic]
    return header.HeaderFields(callsign=call, grid=grid, picture_id=pid, mode=mode, segment=segment,
                               codec_id=codec)


def frame_name(p: P) -> str:
    base = {1024: "tiny", 4096: "short"}[p.n]
    return ("l-" + base) if p.waveform == WAVEFORM_L else base


def make_pass(p: P) -> PassResult:
    i = np.arange(p.n, dtype=np.int64)
    a = ((h32(i + 1000003 * p.pic + 77) >> np.uint64(27)).astype(np.float64) - 15.5) / 8.0
    noise = ((h32(i + 1000003 * (p.seed + 7)) >> np.uint64(28)).astype(np.float64) - 7.5) / 8.0
    w = np.full(p.n, p.w)
    if p.erase_from >= 0:
        w[p.erase_from:] = 0.0
    z = np.where(w > 0, a + noise, 0.0).astype(np.float32)
    h = hdr_of(p.hdr_pic, p.hdr_seg) if p.hdr_pic >= 0 else None
    if p.llr == "zero":
        llr = np.zeros(N_HDR_BITS)
    elif p.llr == "short":
        llr = np.ones(10)
    else:
        b = header.encode(h).astype(np.float64)
        if p.llr == "strong":
            llr = 4.0 * (1.0 - 2.0 * b)
        else:
            j = np.arange(b.size, dtype=np.int64)
            llr = WEAK_A * (1.0 - 2.0 * b) + ((h32(j + 9000 + 13 * p.seed) >> np.uint64(24)).astype(np.float64)
                                              - 127.5) / 64.0
    cw = None
    if p.cw_keying:
        cw = CwIdResult(text="K1ABC", z_match=p.cw_z, keying=p.cw_keying,
                        agrees={1: True, 0: False, -1: None}[p.cw_agrees])
    timing = Timing(tau0=0.0, ppm=0.0, gamma=0.0, z=10.0, cov=np.eye(2))
    report = TrackReport(offset_hz=p.f_hz, drift_hz_per_min=0.0, wander_hz_rms=0.0, doppler_hz=0.0,
                         ppm=0.0, snr2500_db=-20.0, z_ref=10.0, kappa=1.0, slip_free=True, suspect=False)
    return PassResult(uid=pass_uid(p.q, p.f_hz, z), q=int(p.q), frame=frame_name(p), waveform=int(p.waveform),
                      f_hz=float(p.f_hz), timing=timing, report=report, z=z, w=w.astype(np.float32),
                      hdr_llr=llr.astype(np.float32), header=h if p.decoded else None, cw=cw,
                      ch=np.zeros(0, dtype=np.complex64), ch_fs=250, ch_t0_index=0.0, f_mix_hz=Fraction(1500),
                      psi=np.ones(4, dtype=np.float32), estimator="joint", em_round=0)


def check_weak():
    """One weak pass, two or three summed decode to nothing; six to the header."""
    ps = [make_pass(BY_NAME[f"w{k}"]) for k in range(6)]
    llr = [p.hdr_llr.astype(np.float64) for p in ps]
    h = hdr_of(0, 0)
    assert all(header.decode(x) is None for x in llr)
    assert header.decode(llr[0] + llr[1]) is None
    assert header.decode(llr[0] + llr[1] + llr[2]) is None
    assert all(header.decode(llr[k] + llr[k + 1]) is None for k in (0, 2, 4))
    got = header.decode(sum(llr))
    assert got is not None and (got.callsign, got.picture_id) == (h.callsign, h.picture_id)


def units_text(units):
    """keying_units' 192 bytes as the first 16 hex digits of their SHA-256."""
    return hashlib.sha256(np.asarray(units, dtype=np.uint8).tobytes()).hexdigest()[:16]


def header_text(h):
    if h is None:
        return "None"
    return "|".join([h.callsign, "<none>" if h.grid is None else h.grid, str(h.picture_id), str(h.mode),
                     str(h.segment), str(h.codec_id), str(h.version), str(h.reserved)])


class Clock:
    def __call__(self):
        return T0


def scenario_results(root: Path):
    out = []
    for name, k, mode, members, queries in SCENARIOS:
        st = S.Store(root / name, clock=Clock())
        key = key_of(k)
        built = {}
        for pname, g in members:
            p = make_pass(BY_NAME[pname])
            built[pname] = p
            st.add_pass(p)
        for pname, g in members:
            st.attach(key, built[pname].uid, g, "test", mode=mode, codec_id=0xD1D8)
        acc = st.accumulator(key)
        assert len(acc.members) == len(members), name
        for kind in queries:
            if kind == "prior":
                for pname, _ in members:
                    for spec_name in PRIOR_SPECS:
                        spec = lframe.get_any(spec_name)
                        for q in (0, Q0, 2 ** 40):
                            try:
                                pr = em.em_prior(acc, built[pname].uid, st, q, spec)
                                res = sha(canon_nan(pr.x_hat), canon_nan(pr.v))
                            except ValueError as e:
                                res = "error: " + str(e).replace(built[pname].uid, "<uid>")
                            out.append((name, f"em_prior|{pname}|{spec_name}|{q}", res))
            elif kind == "mean_w":
                out.append((name, "mean_w", float(em.acc_mean_w_db(acc)).hex()))
            elif kind == "header":
                for g in H_SEGS:
                    out.append((name, f"known_header|{g}", header_text(em.known_header(st, acc, g))))
            elif kind == "keying":
                for call in CALLS:
                    for wf in WAVEFORMS:
                        r = em.known_keying(st, acc, call, wf)
                        res = "None" if r is None else f"{units_text(r[0])}|{int(r[1])}"
                        c = "<none>" if call is None else call
                        out.append((name, f"known_keying|{c}|{wf}", res))
            else:
                raise ValueError(kind)
    return out


SPEC_CASES = [("short", -1), ("full", 1), ("full", 2), ("full", -1), ("tiny", 1), ("l-short", 1), ("l-full", -1),
              ("medium", 1), ("nope", -1), ("l-nope", -1)]


def spec_results():
    out = []
    for name, version in SPEC_CASES:
        h = None
        if version >= 0:
            h = dataclasses.replace(hdr_of(0, 0), version=version)
        p = make_pass(BY_NAME["t0"])
        p = dataclasses.replace(p, frame=name, header=h)
        try:
            spec = em.pass_spec(p)
            out.append((name, version, f"{spec.name}|{spec.n_data}|{int(em.is_l(p))}"))
        except (KeyError, ValueError) as e:
            out.append((name, version, f"error: {type(e).__name__}: {e.args[0]}"))
    return out


# --- writing --------------------------------------------------------------------------------------

def cstr(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def cdouble(x: float) -> str:
    if math.isnan(x):
        return "NAN_"
    if math.isinf(x):
        return "INF_" if x > 0 else "-INF_"
    return float(x).hex()


def main():
    check_weak()
    lines = [
        "// Generated by native/tests/qrss/gen_em_pins.py from sstvae/qrss/em.py and scipy "
        f"{__import__('scipy').__version__}; do not edit.",
        "// See that script for the formulas test_em.cpp and test_special.cpp rebuild.",
        "",
        "#pragma once",
        "",
        "#include <cstdint>",
        "#include <limits>",
        "",
        "namespace em_pins {",
        "",
        "inline constexpr double INF_ = std::numeric_limits<double>::infinity();",
        "inline constexpr double NAN_ = std::numeric_limits<double>::quiet_NaN();",
        "",
        "// --- special ---",
        f"inline constexpr int NDTRI_J[] = {{{', '.join(str(j) for j in NDTRI_J)}}};",
        f"inline constexpr const char* NDTRI_DIGEST = {cstr(ndtri_digest())};",
        f"inline constexpr const char* LGAMMA_DIGEST = {cstr(lgamma_digest())};",
        "struct LogSf { double x, a, logsf; };",
        "inline const LogSf GAMMA_LOGSF[] = {",
    ]
    for x, a, v in logsf_table():
        lines.append(f"    {{{cdouble(x)}, {cdouble(a)}, {cdouble(v)}}},")
    lines += ["};", "", "// --- the tail statistics ---", "struct GaussZ { double log_p, z; };",
              "inline const GaussZ GAUSS_Z[] = {"]
    for lp, z in gauss_table():
        lines.append(f"    {{{cdouble(lp)}, {cdouble(z)}}},")
    lines += ["};", "struct TemplateZ { double lam; std::int64_t k; double n_cells, z; const char* error; };",
              "inline const TemplateZ TEMPLATE_Z[] = {"]
    for lam, k, cells, z, err in template_table():
        lines.append(f"    {{{cdouble(lam)}, {k}, {cdouble(cells)}, {cdouble(z)}, {cstr(err)}}},")
    lines += ["};", "", "// --- the references ---",
              "struct Prior { int k; int n; std::uint64_t q; const char* lmmse; const char* prior; };",
              "inline const Prior PRIOR[] = {"]
    for k, n, q, lm, pr in prior_table():
        lines.append(f"    {{{k}, {n}, {q}ULL, {cstr(lm)}, {cstr(pr)}}},")
    lines += ["};", "struct PriorRefusal { int s_n, w_n; std::int64_t q; const char* result; };",
              "inline const PriorRefusal PRIOR_REFUSALS[] = {"]
    for s_n, w_n, q, res in prior_refusals():
        lines.append(f"    {{{s_n}, {w_n}, {q}, {cstr(res)}}},")
    lines += ["};", "", "// --- the store-backed pieces ---",
              "struct Pic { const char* callsign; const char* grid; std::int64_t picture_id, mode, codec_id; };",
              "inline const Pic PICS[] = {"]
    for call, grid, pid, mode, codec in PICS:
        lines.append(f"    {{{cstr(call)}, {'nullptr' if grid is None else cstr(grid)}, {pid}, {mode}, {codec}}},")
    lines += ["};",
              f"inline constexpr double WEAK_A = {WEAK_A!r};",
              f"inline constexpr double T0 = {T0!r};",
              "struct Pass {",
              "    const char* name; int n, pic, seed; std::int64_t q; double w; int waveform; double f_hz;",
              "    int erase_from, hdr_pic, hdr_seg; const char* llr; bool decoded;",
              "    const char* cw_keying; double cw_z; int cw_agrees;",
              "};",
              "inline const Pass PASSES[] = {"]
    for p in PASSES:
        lines.append(f"    {{{cstr(p.name)}, {p.n}, {p.pic}, {p.seed}, {p.q}, {cdouble(p.w)}, {p.waveform}, "
                     f"{cdouble(p.f_hz)}, {p.erase_from}, {p.hdr_pic}, {p.hdr_seg}, {cstr(p.llr)}, "
                     f"{'true' if p.decoded else 'false'}, {cstr(p.cw_keying)}, {cdouble(p.cw_z)}, {p.cw_agrees}}},")
    lines += ["};",
              "// key: a PICS index, or -1 - k for (\"K1ABC\", 5000 + k); members \"pass:segment ...\"",
              "struct Scenario { const char* name; int key; std::int64_t mode; const char* members; };",
              "inline const Scenario SCENARIOS[] = {"]
    for name, k, mode, members, _ in SCENARIOS:
        lines.append(f"    {{{cstr(name)}, {k}, {mode}, {cstr(' '.join(f'{m}:{g}' for m, g in members))}}},")
    lines += ["};", "struct Result { const char* scenario; const char* query; const char* result; };",
              "inline const Result RESULTS[] = {"]
    with tempfile.TemporaryDirectory() as td:
        for scen, query, res in scenario_results(Path(td)):
            lines.append(f"    {{{cstr(scen)}, {cstr(query)}, {cstr(res)}}},")
    lines += ["};", "struct SpecCase { const char* frame; int version; const char* result; };",
              "inline const SpecCase SPEC_CASES[] = {"]
    for name, version, res in spec_results():
        lines.append(f"    {{{cstr(name)}, {version}, {cstr(res)}}},")
    lines += ["};", "", "}  // namespace em_pins", ""]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
