#!/usr/bin/env python3
"""Writes native/tests/qrss/store_pins.hpp: a scripted multi-pass store
session run through sstvae/qrss/store.py and associate.py, for
test_store.cpp and test_associate.cpp to replay through the C++ port.

Every pass is a formula, so the C++ builds the same PassResult with no
random numbers and no file from Python: picture `pic` holds, at canonical
offset c of segment g,

    ((h32(c + 65536 g + 1000003 pic) >> 27) - 15.5) / 8

and a pass of segment g with n latents carries, at air position i (canonical
offset c = canonical_index(g, n)[i]), that value plus

    ((h32(i + 1000003 (seed + 7)) >> 28) - 7.5) / 8

where h32 is a 32-bit integer hash (Wellons' lowbias32), so that pictures
and noise of different seeds do not correlate (multiples of 1/16, so exact
in float32 and in float64), with weight
`w` up to `erase_from` and 0 after it (z then 0). The weights are chosen so
that each pass's mean weight is a power of ten: a member's mean_w_db is
10 log10 of it, and numpy's SIMD log10 and the C library's agree there (they
can differ in the last place elsewhere; see store.hpp). A pass's header
LLRs are 4 (1 - 2 bit) over the header's coded bits, or all zero.

The session (OPS) runs on a fake clock set before each op, and pins each
op's result as json.dumps text (the C++ builds the same structure and dumps
it with pyjson), plus, at each "tree" op, the SHA-256 of every file under the
store's root. Those cover the pass files, the accumulators (npz and JSON),
the provisional clusters and log.jsonl, so a replay that matches them wrote
the same bytes as store.py.

    PYTHONPATH=. python native/tests/qrss/gen_store_pins.py    # rewrites the header
"""

import hashlib
import json
import os
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sstvae.qrss import associate as asc  # noqa: E402
from sstvae.qrss import header  # noqa: E402
from sstvae.qrss import store as S  # noqa: E402
from sstvae.qrss.types import PassResult, Timing, TrackReport, pass_uid  # noqa: E402

OUT = Path(__file__).with_name("store_pins.hpp")

T0 = 1760000000.0
Q0 = 1955555                       # slot_time(Q0) = T0 - 500
N = 26000                          # latents per pass (more than half a segment: rule 1 validates)
M32 = 0xFFFFFFFF
PICS = [
    # callsign, grid, picture_id, mode, codec_id
    ("K1ABC", "FN42", 0xA1B2C3D4, 2, 0xD1D8),
    ("W9XYZ/P", None, 0x00000042, 1, 0xD1D8),
    ("K1ABC", "FN42", 0xA1B2C3D4, 1, 0xD1D8),     # picture 0's key with another mode
]
DAY = 86400.0

# name, pic, segment, q, f_hz, waveform, w, erase_from (-1: none), seed, decoded, llr, em_round
PASSES = [
    ("a0", 0, 0, Q0, 1500.0, 0, 10.0, -1, 1, True, True, 0),
    ("a0b", 0, 0, Q0 + 1, 1500.0, 0, 1.0, -1, 2, False, False, 0),
    ("b0", 1, 0, Q0 + 2, 1700.0, 0, 10.0, -1, 3, False, True, 0),
    ("b0b", 1, 0, Q0 + 3, 1700.0, 0, 10.0, -1, 4, False, True, 0),
    ("a1", 0, 1, Q0 + 4, 1500.0, 0, 20.0, N // 2, 5, False, False, 0),
    ("a1h", 0, 1, Q0 + 6, 1500.0, 0, 10.0, -1, 6, True, True, 0),
    ("a0t", 0, 0, Q0 + 1, 1500.5, 0, 1.0, -1, 7, False, False, 0),
    ("a0L", 0, 0, Q0 + 1, 1509.5, 1, 100.0, -1, 8, False, False, 0),
    ("a0e", 0, 0, Q0, 1500.0, 0, 100.0, -1, 1, False, False, 1),   # a0 again, EM round 1
    ("a0f", 0, 0, Q0 + 9, 1500.0, 0, 10.0, -1, 9, False, False, 0),
    ("a0g", 0, 0, Q0 + 10, 1500.0, 0, 1.0, -1, 10, False, False, 0),
    ("b1", 1, 1, Q0 + 12, 1700.0, 0, 10.0, -1, 11, False, False, 0),
    ("lone", 1, 1, Q0 - 40, 1300.0, 0, 1.0, -1, 12, False, False, 0),
    ("x0", 1, 0, Q0 + 7, 1600.0, 0, 10.0, -1, 13, True, True, 0),     # B's latents, A's header
    ("xm", 0, 0, Q0 + 8, 1550.0, 0, 10.0, -1, 14, True, True, 0),     # A's latents, mode 1
]
# a pass's header (and LLRs) when not its own picture's: (PICS index, segment)
HDR = {"x0": (0, 0), "xm": (2, 0)}

# kind, arg, extra, clock (the fake clock's value from this op on)
OPS = [
    ("associate", "a0", "", T0 + 1.0),
    ("associate", "a0b", "", T0 + 2.0),
    ("corr", "a0b", "", T0 + 2.0),
    ("associate", "b0", "", T0 + 3.0),
    ("tree", "", "", T0 + 3.0),
    ("associate", "b0b", "", T0 + 4.0),
    ("associate", "a1", "", T0 + 5.0),
    ("associate", "a1h", "", T0 + 6.0),
    ("associate", "a0", "", T0 + 7.0),
    ("associate", "a0t", "", T0 + 8.0),
    ("associate", "a0L", "", T0 + 9.0),
    ("associate", "x0", "", T0 + 9.0),           # rule 1's correlation check refuses it
    ("associate", "xm", "", T0 + 9.0),           # rule 1: the mode does not match
    ("agrees", "b0", "a0", T0 + 9.0),
    ("agrees", "b0", "b0", T0 + 9.0),
    ("agrees", "a0b", "a0", T0 + 9.0),
    ("tree", "", "", T0 + 9.0),
    ("update", "a0e", "", T0 + 10.0),
    ("merge", "a0f", "rx2", T0 + 11.0),
    ("merge", "a0f", "rx2", T0 + 12.0),           # the same arrays again: refused
    ("merge", "a0g", "rx2", T0 + 13.0),           # replaces rx2's earlier merge
    ("merge", "b1", "rx3", T0 + 14.0),            # another picture: does not correlate
    ("merge", "a0g", "nan", T0 + 15.0),           # not finite: refused, not logged
    ("merge", "a0g", "negative", T0 + 16.0),      # a negative weight: refused
    ("detach", "a1", "", T0 + 3 * DAY),
    ("prov_add", "a1", "", T0 + 3 * DAY),
    ("add", "lone", "", T0 + 3 * DAY),
    ("prov_add", "lone", "", T0 + 3 * DAY),
    ("add", "b1", "", T0 + 3 * DAY),
    ("prov_add", "b1", "lone", T0 + 3 * DAY),          # into lone's cluster
    ("prov_remove", "b1", "", T0 + 3 * DAY),
    ("log", "", "", T0 + 3 * DAY + 0.5),
    ("due", str(Q0 + 4), "", T0 + 3 * DAY),
    ("due", str(Q0 + 5), "", T0 + 3 * DAY),
    ("due", str(Q0 + 8), "", T0 + 3 * DAY),
    ("due", str(Q0 + 6), "", T0 + 3 * DAY),
    ("resolve", "", "", T0 + 3 * DAY + 1.0),
    ("tree", "", "", T0 + 3 * DAY + 1.0),
    ("expire", str(T0 + 7.5 * DAY), "", T0 + 3 * DAY + 2.0),
    ("tree", "", "", T0 + 3 * DAY + 2.0),
    ("expire", str(T0 + 30 * DAY), "", T0 + 3 * DAY + 3.0),
    ("tree", "", "", T0 + 3 * DAY + 3.0),
]


def hdr_of(pic: int, segment: int) -> header.HeaderFields:
    call, grid, pid, mode, codec = PICS[pic]
    return header.HeaderFields(callsign=call, grid=grid, picture_id=pid, mode=mode,
                               segment=segment, codec_id=codec)


def h32(x):
    """lowbias32 on uint64 arrays (values mod 2^32)."""
    x = np.asarray(x, dtype=np.uint64) & np.uint64(M32)
    x ^= x >> np.uint64(16)
    x = (x * np.uint64(0x7FEB352D)) & np.uint64(M32)
    x ^= x >> np.uint64(15)
    x = (x * np.uint64(0x846CA68B)) & np.uint64(M32)
    x ^= x >> np.uint64(16)
    return x


def make_pass(spec) -> PassResult:
    name, pic, g, q, f, wf, w0, erase, seed, decoded, llr, em_round = spec
    i = np.arange(N, dtype=np.int64)
    c = S.canonical_index(g, N).astype(np.int64)
    a = ((h32(c + 65536 * g + 1000003 * pic) >> np.uint64(27)).astype(np.float64) - 15.5) / 8.0
    noise = ((h32(i + 1000003 * (seed + 7)) >> np.uint64(28)).astype(np.float64) - 7.5) / 8.0
    w = np.full(N, w0)
    if erase >= 0:
        w[erase:] = 0.0
    z = np.where(w > 0, a + noise, 0.0).astype(np.float32)
    h = hdr_of(*HDR.get(name, (pic, g)))
    bits = header.encode(h).astype(np.float64)
    hdr_llr = (4.0 * (1.0 - 2.0 * bits) if llr else np.zeros(bits.size)).astype(np.float32)
    timing = Timing(tau0=0.0, ppm=0.0, gamma=0.0, z=10.0, cov=np.eye(2))
    report = TrackReport(offset_hz=f, drift_hz_per_min=0.0, wander_hz_rms=0.0, doppler_hz=0.0,
                         ppm=0.0, snr2500_db=-20.0, z_ref=10.0, kappa=1.0, slip_free=True,
                         suspect=False)
    return PassResult(uid=pass_uid(q, f, z), q=int(q), frame="l-short" if wf else "short",
                      waveform=int(wf), f_hz=float(f), timing=timing, report=report, z=z,
                      w=w.astype(np.float32), hdr_llr=hdr_llr, header=h if decoded else None,
                      cw=None, ch=np.zeros(0, dtype=np.complex64), ch_fs=250,
                      ch_t0_index=0.0, f_mix_hz=Fraction(1500), psi=np.ones(4, dtype=np.float32),
                      estimator="joint", em_round=int(em_round))


def foreign(p: PassResult, segment: int, variant: str):
    """A foreign (S, W): pass p laid on the canonical grid as its own sums."""
    Sf = np.zeros((3, S.GROUP_LATENTS), dtype=np.float32)
    Wf = np.zeros((3, S.GROUP_LATENTS), dtype=np.float32)
    c = S.canonical_index(segment, p.z.size)
    w = p.w.astype(np.float64)
    Sf[segment, c] = (w * p.z.astype(np.float64)).astype(np.float32)
    Wf[segment, c] = p.w
    if variant == "nan":
        Sf[0, 0] = np.nan
    elif variant == "negative":
        Wf[2, 7] = -1.0
    return Sf, Wf


def key_json(key):
    return None if key is None else [key[0], int(key[1])]


def tree(root: Path):
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = Path(dirpath) / f
            out.append((p.relative_to(root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()))
    return sorted(out)


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t


def run(root: Path):
    passes = {s[0]: (s, make_pass(s)) for s in PASSES}
    clock = Clock()
    st = S.Store(root, clock=clock)
    results = []
    for kind, arg, extra, t in OPS:
        clock.t = t
        if kind == "associate":
            key, rule = asc.associate(passes[arg][1], st)
            res = json.dumps([key_json(key), rule])
        elif kind == "corr":
            p = passes[arg][1]
            g = passes[arg][0][2]
            acc = st.accumulator(st.find_member(p.uid))
            m, W = asc._acc_reference(acc, g, p.z.size)
            z, w = p.z.astype(np.float64), p.w.astype(np.float64)
            # without the pass itself: the reference as it was before it joined
            Sl, Wl = acc.loo(p.uid, st)
            ml = np.divide(Sl, Wl, out=np.zeros_like(Wl), where=Wl > 0)
            t, M = asc.corr_stat(z, w, m, W)
            tl, Ml = asc.corr_stat(z, w, ml, Wl)
            res = json.dumps([float(t), int(M), asc.expected_t(w, W), bool(asc.corr_accept(t, asc.expected_t(w, W))),
                              float(tl), int(Ml), asc.expected_t(w, Wl)])
        elif kind == "agrees":
            spec = passes[extra][0]
            h = hdr_of(*HDR.get(extra, (spec[1], spec[2])))
            res = json.dumps(bool(asc.header_agrees(h, passes[arg][1].hdr_llr)))
        elif kind == "tree":
            res = json.dumps(tree(root))
        elif kind == "update":
            res = json.dumps([key_json(k) for k in st.update_pass(passes[arg][1])])
        elif kind == "merge":
            spec, p = passes[arg]
            Sf, Wf = foreign(p, spec[2], extra)
            key = (PICS[0][0], PICS[0][2])
            source = extra if extra not in ("nan", "negative") else "rx2"
            res = json.dumps(bool(st.merge(key, Sf, Wf, source=source)))
        elif kind == "detach":
            key = st.find_member(passes[arg][1].uid)
            acc = st.detach(key, passes[arg][1].uid)
            res = json.dumps([key_json(key), acc.uids])
        elif kind == "add":
            st.add_pass(passes[arg][1])
            res = json.dumps(st.pass_uids())
        elif kind == "prov_add":
            cluster = f"c_{passes[extra][1].uid}" if extra else None
            res = json.dumps(st.prov_add(passes[arg][1].uid, cluster))
        elif kind == "prov_remove":
            st.prov_remove(passes[arg][1].uid)
            res = json.dumps(st.prov_clusters())
        elif kind == "log":
            st.log({"event": "note", "v": 0.1, "n": [1, 2.5, None], "s": "caf\u00e9"})
            res = json.dumps(None)
        elif kind == "due":
            res = json.dumps([dict(d, key=key_json(d["key"])) for d in st.due(int(arg))])
        elif kind == "resolve":
            res = json.dumps([[c, key_json(k)] for c, k in asc.resolve_provisional(st)])
        elif kind == "expire":
            r = st.expire(now=float(arg))
            res = json.dumps({"acc": [key_json(k) for k in r["acc"]], "prov": r["prov"],
                              "passes": r["passes"]})
        else:
            raise ValueError(kind)
        results.append(res)
    return passes, results


# --- associate's statistics on pinned inputs (test_associate.cpp) ---------------------

# n_z, n_m, family ("same": m correlates with z, "other": it does not), edits
STATS = [
    (5000, 5000, "same", ""),
    (5000, 5000, "other", ""),
    (300, 500, "same", ""),            # m longer: the common prefix
    (500, 300, "same", ""),            # z longer
    (1000, 1000, "same", "nonfinite"),  # NaN, inf and negative weights dropped
    (400, 400, "same", "w0"),           # no weight anywhere: (0.0, 0)
    (0, 0, "same", ""),                 # nothing at all
    (20, 20, "same", ""),               # too few to pass t > 6
    (3000, 3000, "same", "weak"),       # m mostly noise: t below half its expectation
    (129, 129, "same", ""),             # past one pairwise block (128)
]


def stats_case(k, nz, nm, family, edits):
    i = np.arange(max(nz, nm), dtype=np.int64)
    z = ((h32(i + 7 * k + 11) >> np.uint64(24)).astype(np.float64) - 127.5) / 32.0
    w = (h32(i + 1000 + 7 * k) >> np.uint64(28)).astype(np.float64) / 4.0
    m = ((h32(i + 2000 + 7 * k) >> np.uint64(24)).astype(np.float64) - 127.5) / 32.0
    W = (h32(i + 3000 + 7 * k) >> np.uint64(28)).astype(np.float64) / 2.0
    if family == "same":
        m = m * (8.0 if edits == "weak" else 0.5) + z
    if edits == "nonfinite":
        z[i % 97 == 0] = np.nan
        m[i % 89 == 0] = np.inf
        W[i % 83 == 0] = np.nan
        w[i % 79 == 0] = -1.0
    if edits == "w0":
        w[:] = 0.0
    return z[:nz], w[:nz], m[:nm], W[:nm]


def stats():
    out = []
    for k, (nz, nm, family, edits) in enumerate(STATS):
        z, w, m, W = stats_case(k, nz, nm, family, edits)
        t, M = asc.corr_stat(z, w, m, W)
        te = asc.expected_t(w, W)
        out.append(json.dumps([float(t), int(M), type(t) is not float, te,
                               asc.expected_t(w, W, kurtosis=0.0), bool(asc.corr_accept(t, te))]))
    return out


# true (PICS index, segment), tested (PICS index, segment), signal A, noise B, zero share /16
AGREE = [
    (0, 0, 0, 0, 1.0, 1.0, 0),
    (0, 0, 0, 1, 1.0, 1.0, 0),     # another segment
    (0, 0, 1, 0, 1.0, 1.0, 0),     # another picture
    (0, 0, 2, 0, 1.0, 1.0, 4),     # another mode, a quarter not heard
    (0, 0, 0, 0, 0.3, 1.0, 8),     # half not heard
    (0, 0, 0, 0, 0.0, 0.0, 16),    # nothing heard: agrees
    (1, 1, 1, 1, 0.3, 1.0, 0),
    (1, 1, 1, 1, 0.38, 1.0, 0),
    (1, 1, 1, 1, 0.42, 1.0, 0),
    (1, 1, 1, 1, 0.45, 1.0, 0),
    (1, 1, 1, 1, 0.5, 1.0, 0),
    (1, 1, 1, 1, 0.7, 1.0, 2),
    (1, 1, 1, 1, 0.42, 1.0, 14),     # two sixteenths heard: few bits, a wide margin
    (1, 1, 1, 0, 0.3, 1.0, 0),
    (1, 1, 1, 0, 2.0, 1.0, 0),
    (1, 1, 0, 1, 0.5, 1.0, 12),
]


def agree_llr(k, tp, ts, a, b, zeros):
    c = header.encode(hdr_of(tp, ts)).astype(np.float64)
    i = np.arange(c.size, dtype=np.int64)
    noise = ((h32(i + 9000 + 13 * k) >> np.uint64(24)).astype(np.float64) - 127.5) / 64.0
    llr = a * (1.0 - 2.0 * c) + b * noise
    llr[(h32(i + 5000 + k) >> np.uint64(28)) < np.uint64(zeros)] = 0.0
    return llr


def agrees():
    out = []
    for k, (tp, ts, hp, hs, a, b, zeros) in enumerate(AGREE):
        out.append(bool(asc.header_agrees(hdr_of(hp, hs), agree_llr(k, tp, ts, a, b, zeros))))
    return out


def seam(root: Path):
    """The session's associations up to "xm" with rule 2 replaced by a
    search that finds nothing: [results], calls."""
    passes = {s[0]: make_pass(s) for s in PASSES}
    clock = Clock()
    st = S.Store(root, clock=clock)
    calls = []
    real = asc._soft_header
    asc._soft_header = lambda store, uid, llr: calls.append(uid)  # returns None
    try:
        out = []
        for kind, arg, _, t in OPS:
            if kind != "associate":
                continue
            clock.t = t
            key, rule = asc.associate(passes[arg], st)
            out.append(json.dumps([key_json(key), rule]))
            if arg == "xm":
                break
    finally:
        asc._soft_header = real
    return out, len(calls)


def errors(root: Path):
    """(id, exception type, args[0]) of the reference's refusals."""
    passes = {s[0]: make_pass(s) for s in PASSES}
    st = S.Store(root, clock=Clock())
    asc.associate(passes["a0"], st)
    key = (PICS[0][0], PICS[0][2])
    cases = {
        "canonical_index(0, 0)": lambda: S.canonical_index(0, 0),
        "canonical_index(0, 50601)": lambda: S.canonical_index(0, 50601),
        "canonical_index(3, 10)": lambda: S.canonical_index(3, 10),
        "loo(nope)": lambda: st.accumulator(key).loo("nope", st),
        "pass_path(a/b)": lambda: st._pass_path("a/b"),
        "pass_path(.x)": lambda: st._pass_path(".x"),
        "accumulator(NOPE, 1)": lambda: st.accumulator(("NOPE", 1)),
        "attach(nope)": lambda: st.attach(key, "nope", 0, "x"),
        "attach(a0, 3)": lambda: st.attach(key, passes["a0"].uid, 3, "x"),
        "has_data(3)": lambda: st.accumulator(key).has_data(3),
    }
    out = []
    for name, f in cases.items():
        try:
            f()
        except Exception as e:  # noqa: BLE001
            out.append((name, type(e).__name__, str(e.args[0]) if e.args else ""))
        else:
            raise AssertionError(name)
    values = [
        ("key_name(A/B, -5)", S.key_name(("A/B", -5))),
        ("key_name(K1ABC, 0xA1B2C3D4)", S.key_name(key)),
        ("slot_time(Q0)", repr(S.slot_time(Q0))),
        ("mean_w_db([])", repr(S.mean_w_db(np.zeros(0)))),
        ("mean_w_db([0, 0])", repr(S.mean_w_db(np.zeros(2)))),
        ("mean_w_db([10, 10, 10])", repr(S.mean_w_db(np.full(3, 10.0)))),
        ("mean_w_db([nan])", repr(S.mean_w_db(np.array([np.nan])))),
        ("mean_w_db([-1, 0.5])", repr(S.mean_w_db(np.array([-1.0, 0.5])))),
        ("mean_w_db([1000, 0, 0, 0, 0, 0, 0, 0, 0, 0])", repr(S.mean_w_db(np.array([1000.0] + [0.0] * 9)))),
    ]
    return out, values


def c_str(s: str) -> str:
    out = []
    for b in s.encode("utf-8"):
        if 32 <= b < 127 and chr(b) not in '"\\?':
            out.append(chr(b))
        else:
            out.append(f"\\x{b:02x}\"\"")
    return '"' + "".join(out) + '"'


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        passes, results = run(Path(d) / "qrss")
        seam_results, seam_calls = seam(Path(d) / "seam")
        errs, values = errors(Path(d) / "errors")
    stat_results = stats()
    agree_results = agrees()
    lines = [
        "// GENERATED by native/tests/qrss/gen_store_pins.py from sstvae/qrss/store.py and",
        "// associate.py. Do not edit by hand.",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace store_pins {",
        "",
        f"inline constexpr double T0 = {T0!r};",
        f"inline constexpr std::int64_t Q0 = {Q0};",
        f"inline constexpr int N = {N};",
        "",
        "struct Pic {",
        "    const char* callsign;",
        "    const char* grid;   // nullptr: None",
        "    std::int64_t picture_id;",
        "    int mode;",
        "    std::int64_t codec_id;",
        "};",
        "inline const Pic PICS[] = {",
    ]
    for call, grid, pid, mode, codec in PICS:
        lines.append(f"    {{{c_str(call)}, {c_str(grid) if grid else 'nullptr'}, {pid}, {mode}, {codec}}},")
    lines += [
        "};",
        "",
        "struct Pass {",
        "    const char* name;",
        "    int pic, segment;",
        "    std::int64_t q;",
        "    double f_hz;",
        "    int waveform;",
        "    double w;",
        "    int erase_from;   // -1: none",
        "    int seed;",
        "    bool decoded, llr;",
        "    int em_round;",
        "    int hdr_pic, hdr_seg;   // the header it carries",
        "    const char* uid;  // types.pass_uid, as Python made it",
        "};",
        "inline const Pass PASSES[] = {",
    ]
    for name, (spec, p) in passes.items():
        _, pic, g, q, f, wf, w0, erase, seed, decoded, llr, em = spec
        lines.append(f"    {{{c_str(name)}, {pic}, {g}, {q}, {f!r}, {wf}, {w0!r}, {erase}, {seed}, "
                     f"{'true' if decoded else 'false'}, {'true' if llr else 'false'}, {em}, "
                     f"{', '.join(map(str, HDR.get(name, (pic, g))))}, "
                     f"{c_str(p.uid)}}},")
    lines += [
        "};",
        "",
        "struct Op {",
        "    const char* kind;",
        "    const char* arg;",
        "    const char* extra;",
        "    double clock;",
        "    const char* want;   // json.dumps of the op's result",
        "};",
        "inline const Op OPS[] = {",
    ]
    for (kind, arg, extra, t), res in zip(OPS, results):
        lines.append(f"    {{{c_str(kind)}, {c_str(arg)}, {c_str(extra)}, {t!r},")
        lines.append(f"     {c_str(res)}}},")
    lines += [
        "};",
        "",
        "// The associations up to \"xm\" with rule 2's search finding nothing, and how",
        "// many times it was asked.",
        "inline const char* SEAM[] = {",
    ]
    lines += [f"    {c_str(r)}," for r in seam_results]
    lines += [
        "};",
        f"inline constexpr int SEAM_CALLS = {seam_calls};",
        "",
        "struct Stat {",
        "    int n_z, n_m;",
        "    const char* family;",
        "    const char* edits;",
        "    const char* want;   // [t, M, t is np.float64, expected_t, expected_t(kurtosis=0), corr_accept]",
        "};",
        "inline const Stat STATS[] = {",
    ]
    for (nz, nm, family, edits), res in zip(STATS, stat_results):
        lines.append(f"    {{{nz}, {nm}, {c_str(family)}, {c_str(edits)}, {c_str(res)}}},")
    lines += [
        "};",
        "",
        "struct Agree {",
        "    int true_pic, true_seg, pic, seg;",
        "    double a, b;",
        "    int zeros;",
        "    bool want;",
        "};",
        "inline const Agree AGREE[] = {",
    ]
    for (tp, ts, hp, hs, a, b, zeros), res in zip(AGREE, agree_results):
        lines.append(f"    {{{tp}, {ts}, {hp}, {hs}, {a!r}, {b!r}, {zeros}, {'true' if res else 'false'}}},")
    lines += [
        "};",
        "",
        "struct Error {",
        "    const char* call;",
        "    const char* type;   // the Python exception's",
        "    const char* text;   // its args[0]",
        "};",
        "inline const Error ERRORS[] = {",
    ]
    for name, typ, text in errs:
        lines.append(f"    {{{c_str(name)}, {c_str(typ)}, {c_str(text)}}},")
    lines += [
        "};",
        "",
        "struct Text {",
        "    const char* call;",
        "    const char* want;   // str, or repr of a float",
        "};",
        "inline const Text VALUES[] = {",
    ]
    for name, v in values:
        lines.append(f"    {{{c_str(name)}, {c_str(v)}}},")
    lines += ["};", "", "}  // namespace store_pins", ""]
    OUT.write_text("\n".join(lines))
    for (kind, arg, extra, _), res in zip(OPS, results):
        print(kind, arg, extra, res[:300])
    print("seam", seam_calls, seam_results)
    print("stats", stat_results)
    print("agree", agree_results)
    print("errors", errs)
    print("values", values)


if __name__ == "__main__":
    main()
