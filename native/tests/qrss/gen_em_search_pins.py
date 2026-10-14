#!/usr/bin/env python3
"""Writes native/tests/qrss/em_search_pins.hpp: what test_em_search.cpp checks
the C++ port of sstvae/qrss/em.py's template search against, with no Python
at test time. The formulas below are rebuilt there; splitmix, unit, h32 and
the double/str writers are gen_em_pins.py's.

  * _search_chan on formula captures and templates (SEARCH_CHAN): z_n =
    (u_2n - 1/2) + j (u_2n+1 - 1/2), u = unit(splitmix(seed, 2 n_z)); chunk
    k of the template E_kj = (v_2i - 1/2) + j (v_2i+1 - 1/2), v =
    unit(splitmix(seed + 1, 2 K chunk)), i = k chunk + j, its energy en_k =
    1 + k / 4 (any positive number serves: _search_chan takes it as given);
    with `embed` the template is added into z at `embed_lag`, times
    `embed` (z[start_k + lag + j] += embed E_kj where inside z). The cell
    (lag, m) exactly, Lambda to 1e-12 relative.
  * _chunk_template on formula (mu, nu) (CHUNK_TEMPLATE): mu_i = 2 u_i - 1,
    nu_i = v_i / 2 (u = unit(splitmix(800 + c, n_sym)), v = unit(splitmix(900
    + c, n_sym)), every seventh nu 0), keying none, keying_units("K1ABC")
    as FSK or as OOK: starts exactly, every energy to 1e-12 relative and
    three samples (first, middle, last) of every eighth chunk and the last
    to 1e-12 of the largest |E|, and the empty span's refusal.
  * template_search on a formula store (TEMPLATE_SEARCH): picture key
    ("K1ABC", PICTURE_ID), mode 1, two SHORT passes attached as segments 0
    and 1, pass g's air latents

        a = ((h32(i + 1000003 (10 + g) + 77) >> 27) - 15.5) / 8
        z = float32(a + ((h32(i + 1000003 (27 + g)) >> 28) - 7.5) / 8),  w = 3

    its header (call K1ABC, grid FN42, mode 1, segment g, codec 0xD1D8)
    decoded and its LLRs 4 (1 - 2 bit), q = Q0 - 1 - g, f_hz and
    report.offset_hz PASS_F[g], drift PASS_DRIFT[g], timing.ppm
    PASS_PPM[g]. The capture is a 250 Hz PassResult (ch, ch_t0_index T0_CH,
    f_mix_hz, q Q0 + 7):

        ch_n = float32(A (s_n c_n - s'_n d_n) + noise_re, A (s_n d_n + s'_n c_n) + noise_im)

    with noise the z formula above (seed), s + j s' the search's own
    template of segment g (the accumulator's prior, the segment's known
    header and the call's FSK keying on SHORT, as template_search builds
    it) laid out by _chunk_template(..., t0_index = T0_CH + L, ppm =
    ppm_g + k ppm_step, n_ch, chunk 1) and zero elsewhere, and (c, d) =
    (cos, sin)(2 pi r), r = x - floor(x), x = f_ch t + ((drift / 60) t) t
    / 2 + m0 n / 2048, t = (n - T0_CH) / 250, f_ch the candidate's offset
    from the mix: a pass of that segment from candidate cand, L samples
    late, k clock steps off and m0 bins up. A = 0 is noise alone. One case
    (FE) is a raw FE-rate noise capture (the z formula at 4 kHz, t0
    T0_FE). Each case: the detection or none, Z and Lambda to 1e-9
    relative (the template is libm's exp where numpy's SIMD exp is 1 ulp
    off on a few percent of arguments, and the capture is rounded to
    float32), the chunks, cells and cell (segment, f_hz, ppm, tau0,
    drift) exactly.

    PYTHONPATH=. python native/tests/qrss/gen_em_search_pins.py      # rewrites the header
    PYTHONPATH=. python native/tests/qrss/gen_em_search_pins.py --probe   # prints, writes nothing
"""

import dataclasses
import math
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

import gen_em_pins as G  # noqa: E402

from sstvae.qrss import em, frame, header, track  # noqa: E402
from sstvae.qrss import frontend  # noqa: E402
from sstvae.qrss import store as S  # noqa: E402
from sstvae.qrss.constants import CH_FS, FE_FS, N_HDR_BITS, T_SYM, WAVEFORM_CE  # noqa: E402
from sstvae.qrss.morse import keying_units  # noqa: E402
from sstvae.qrss.types import PassResult, Timing, TrackReport, pass_uid  # noqa: E402

OUT = HERE / "em_search_pins.hpp"
Q0 = 1955555
T0 = 1760000000.0
CALL = "K1ABC"
GRID = "FN42"
PICTURE_ID = 0xA1B2C3D4
CODEC_ID = 0xD1D8
MODE = 1
PASS_F = [1500.0, 1500.5]
PASS_DRIFT = [0.0, 0.3]
PASS_PPM = [0.0, 2.0]
PASS_W = 3.0
T0_CH = 2500.25
T0_FE = 40000.5
CAP_Q = Q0 + 7


def noise(seed, n):
    u = G.unit(G.splitmix(seed, 2 * n))
    return (u[0::2] - 0.5) + 1j * (u[1::2] - 0.5)


# --- _search_chan -----------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Chan:
    name: str
    seed: int
    n_z: int
    starts: tuple
    chunk: int
    lag_max: int
    m_max: int
    n0: float = 0.37
    drift: tuple = ()
    embed: float = 0.0
    embed_lag: int = 0


CHANS = [
    Chan("embedded", 10, 3000, (100, 600, 1100, 1600, 2100), 200, 40, 4, embed=0.6, embed_lag=13),
    Chan("embedded-late", 11, 3000, (100, 600, 1100, 1600, 2100), 200, 40, 4, embed=0.4, embed_lag=-29),
    Chan("drift", 12, 3000, (100, 600, 1100, 1600, 2100), 200, 40, 2, drift=(0, 1, -2, 3, 0), embed=0.6,
         embed_lag=5),
    Chan("noise", 13, 2600, (50, 700, 1350, 2000), 300, 60, 6),
    Chan("nfft-doubled", 14, 4000, (-100, 2500), 1500, 300, 3, embed=0.5, embed_lag=-7),
    Chan("edges", 15, 900, (-150, 500, 850), 100, 120, 1, embed=0.8, embed_lag=2),
    Chan("lag0-m0", 16, 1000, (0, 400), 300, 0, 0),
    Chan("no-chunk", 17, 500, (), 64, 10, 3),
]


def chan_inputs(c: Chan):
    z = noise(c.seed, c.n_z)
    K = len(c.starts)
    v = G.unit(G.splitmix(c.seed + 1, 2 * K * c.chunk))
    E = ((v[0::2] - 0.5) + 1j * (v[1::2] - 0.5)).reshape(K, c.chunk)
    en = 1.0 + np.arange(K) / 4.0
    if c.embed:
        for k, s in enumerate(c.starts):
            for j in range(c.chunk):
                n = s + c.embed_lag + j
                if 0 <= n < c.n_z:
                    z[n] = z[n] + c.embed * E[k, j]
    drift = np.asarray(c.drift if c.drift else [0] * K, dtype=np.int64)
    return z, np.asarray(c.starts, dtype=np.int64), E, en, drift


def chan_results():
    out = []
    for c in CHANS:
        z, starts, E, en, drift = chan_inputs(c)
        lam, lag, m = em._search_chan(z, c.n0, starts, E, en, c.lag_max, c.m_max, drift)
        out.append((c, float(lam), int(lag), int(m)))
    return out


# --- _chunk_template ----------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Ct:
    name: str
    spec: str
    keying: str        # "" | "fsk" | "ook"
    t0: float
    ppm: float
    n_ch: int
    chunk: int


CTS = [
    Ct("tiny", "tiny", "", 3.5, 0.0, 14000, 500),
    Ct("tiny-ppm", "tiny", "", 2600.25, -37.5, 20000, 500),
    Ct("tiny-early", "tiny", "", -500.5, 12.0, 13000, 64),
    Ct("short-fsk", "short", "fsk", 100.25, 37.5, 60000, 500),
    Ct("short-ook", "short", "ook", 61.0, -12.0, 70000, 333),
    Ct("tiny-one-chunk", "tiny", "", 0.0, 0.0, 600, 500),
    Ct("tiny-chunk-past-span", "tiny", "", 0.0, 0.0, 300, 500),
    Ct("tiny-empty", "tiny", "", 0.0, 0.0, 0, 500),
    Ct("tiny-after", "tiny", "", 20000.0, 0.0, 15000, 500),
]


def sampled_chunks(K: int):
    return [k for k in range(K) if k % 8 == 0 or k == K - 1]


def mu_nu(c: int, n: int):
    mu = 2.0 * G.unit(G.splitmix(800 + c, n)) - 1.0
    nu = G.unit(G.splitmix(900 + c, n)) / 2.0
    nu[::7] = 0.0
    return mu, nu


def ct_results():
    out = []
    for c, ct in enumerate(CTS):
        spec = frame.get(ct.spec)
        mu, nu = mu_nu(c, spec.n_sym)
        keying = keying_units(CALL) if ct.keying else None
        try:
            starts, E, en = em._chunk_template(spec, mu, nu, keying, ct.keying == "ook", ct.t0, ct.ppm, ct.n_ch,
                                               ct.chunk)
        except IndexError as e:
            out.append((ct, None, str(e)))
            continue
        out.append((ct, (starts, E, en), ""))
    return out


# --- template_search ----------------------------------------------------------------------------

def hdr_of(g):
    return header.HeaderFields(callsign=CALL, grid=GRID, picture_id=PICTURE_ID, mode=MODE, segment=g,
                               codec_id=CODEC_ID)


def make_pass(g) -> PassResult:
    n = frame.SHORT.n_data
    i = np.arange(n, dtype=np.int64)
    a = ((G.h32(i + 1000003 * (10 + g) + 77) >> np.uint64(27)).astype(np.float64) - 15.5) / 8.0
    nz = ((G.h32(i + 1000003 * (27 + g)) >> np.uint64(28)).astype(np.float64) - 7.5) / 8.0
    z = (a + nz).astype(np.float32)
    w = np.full(n, PASS_W, dtype=np.float32)
    h = hdr_of(g)
    llr = (4.0 * (1.0 - 2.0 * header.encode(h).astype(np.float64))).astype(np.float32)
    q = Q0 - 1 - g
    timing = Timing(tau0=0.0, ppm=PASS_PPM[g], gamma=0.0, z=10.0, cov=np.eye(2))
    report = TrackReport(offset_hz=PASS_F[g], drift_hz_per_min=PASS_DRIFT[g], wander_hz_rms=0.0,
                         doppler_hz=0.0, ppm=0.0, snr2500_db=-20.0, z_ref=10.0, kappa=1.0, slip_free=True,
                         suspect=False)
    return PassResult(uid=pass_uid(q, PASS_F[g], z), q=q, frame="short", waveform=WAVEFORM_CE, f_hz=PASS_F[g],
                      timing=timing, report=report, z=z, w=w, hdr_llr=llr, header=h, cw=None,
                      ch=np.zeros(0, dtype=np.complex64), ch_fs=250, ch_t0_index=0.0, f_mix_hz=Fraction(1500),
                      psi=np.ones(4, dtype=np.float32), estimator="joint", em_round=0)


def make_store(root: Path):
    st = S.Store(root, clock=lambda: T0)
    for g in (0, 1):
        p = make_pass(g)
        st.add_pass(p)
        st.attach((CALL, PICTURE_ID), p.uid, g, "test", mode=MODE, codec_id=CODEC_ID)
    return st


@dataclasses.dataclass(frozen=True)
class Ts:
    name: str
    spec: str
    seed: int
    amp: float = 0.0          # 0: noise alone
    seg: int = 0
    cand: int = 0             # PASS_F index: the signal's candidate
    lag: int = 0
    k: int = 0
    m0: int = 0
    f_mix: int = 0            # Hz inside FE (a 1/64 grid value would do; integers here)
    fe: bool = False          # a raw FE-rate noise capture instead
    freqs: tuple = ()
    segments: tuple | None = None
    tau_s: float = em.TS_TAU_S
    df_hz: float = em.TS_DF_HZ
    ppm_steps: int = em.TS_PPM_STEPS
    z_accept: float = 6.0


TSS = [
    Ts("tiny-signal", "tiny", 1001, amp=0.09, seg=1, cand=1, lag=37, k=1, m0=3),
    Ts("tiny-signal-seg0", "tiny", 1002, amp=0.09, seg=0, cand=0, lag=-120, k=-2, m0=-4),
    Ts("tiny-noise-0", "tiny", 1100),
    Ts("tiny-noise-1", "tiny", 1101),
    Ts("tiny-noise-2", "tiny", 1102),
    Ts("tiny-noise-3", "tiny", 1103),
    Ts("tiny-fe-noise", "tiny", 1300, fe=True),
    Ts("tiny-options", "tiny", 1001, amp=0.09, seg=1, cand=1, lag=37, k=1, m0=3, freqs=(1503.0,),
       segments=(1,), tau_s=1.0, df_hz=0.5, ppm_steps=1, z_accept=1e3),
    Ts("tiny-reach", "tiny", 1001, amp=0.09, seg=1, cand=1, lag=37, k=1, m0=3, f_mix=200),
    Ts("tiny-reach-freqs", "tiny", 1001, amp=0.09, seg=1, cand=1, lag=37, k=1, m0=3, f_mix=200, freqs=(1650.0,)),
    Ts("short-signal", "short", 1002, amp=0.035, seg=0, cand=0, lag=-50, k=-2, m0=-5),
    Ts("short-noise", "short", 1200, segments=(1,)),
]


def classes_of(st, spec, g, q):
    acc = st.accumulator((CALL, PICTURE_ID))
    idx = S.canonical_index(g, spec.n_data)
    prior = em.prior_from(acc.S[g, idx], acc.W[g, idx], q)
    h = em.known_header(st, acc, g)
    bits = header.encode(h) if (h is not None and spec.has_header) else None
    return track.make_classes(spec, bits, prior)


def ch_capture(st, ts: Ts):
    spec = frame.get(ts.spec)
    dur = spec.keyed_end_pos * T_SYM
    n_ch = int(T0_CH + dur * CH_FS + 2500)
    nz = noise(ts.seed, n_ch)
    if ts.amp:
        cl = classes_of(st, spec, ts.seg, CAP_Q)
        keying = keying_units(CALL) if spec.n_win else None
        ppm_step = em.TS_SLIP_SAMPLES / (dur * CH_FS) * 1e6
        ppm = PASS_PPM[ts.cand] + ts.k * ppm_step
        starts, E, _ = em._chunk_template(spec, cl.mu, cl.nu, keying, False, T0_CH + ts.lag, ppm, n_ch, 1)
        sig = np.zeros(n_ch, dtype=np.complex128)
        sig[starts] = E[:, 0]
        n = np.arange(n_ch, dtype=np.int64)
        t = (n - T0_CH) / 250.0
        f_ch = PASS_F[ts.cand] - (1500.0 + ts.f_mix)
        x = f_ch * t + ((PASS_DRIFT[ts.cand] / 60.0) * t) * t / 2.0 + (ts.m0 * n) / 2048.0
        r = x - np.floor(x)
        c = np.array([math.cos(2.0 * math.pi * v) for v in r])
        d = np.array([math.sin(2.0 * math.pi * v) for v in r])
        re = ts.amp * (sig.real * c - sig.imag * d) + nz.real
        im = ts.amp * (sig.real * d + sig.imag * c) + nz.imag
    else:
        re, im = nz.real, nz.imag
    ch = (re.astype(np.float32) + 1j * im.astype(np.float32)).astype(np.complex64)
    p = make_pass(0)
    return dataclasses.replace(p, uid=f"cap-{ts.name}", q=CAP_Q, ch=ch, ch_fs=CH_FS, ch_t0_index=T0_CH,
                               f_mix_hz=Fraction(ts.f_mix))


def fe_capture(ts: Ts):
    spec = frame.get(ts.spec)
    n = int(T0_FE + spec.keyed_end_pos * T_SYM * FE_FS + 40000)
    nz = noise(ts.seed, n)
    fe = (nz.real.astype(np.float32) + 1j * nz.imag.astype(np.float32)).astype(np.complex64)
    return frontend.Capture(fe, CAP_Q, T0_FE)


def ts_results(root: Path):
    st = make_store(root)
    out = []
    for ts in TSS:
        cap = fe_capture(ts) if ts.fe else ch_capture(st, ts)
        kw = dict(freqs=ts.freqs, segments=None if ts.segments is None else list(ts.segments), tau_s=ts.tau_s,
                  df_hz=ts.df_hz, ppm_steps=ts.ppm_steps, z_accept=ts.z_accept)
        t = time.perf_counter()
        det, z, info = em.template_search(st, (CALL, PICTURE_ID), cap, frame.get(ts.spec), return_stats=True, **kw)
        out.append((ts, det, float(z), info, time.perf_counter() - t))
    return out


# --- writing --------------------------------------------------------------------------------------

def ints(xs):
    return "{" + ", ".join(str(int(x)) for x in xs) + "}"


def dbls(xs):
    return "{" + ", ".join(G.cdouble(float(x)) for x in xs) + "}"


def main():
    probe = "--probe" in sys.argv
    t_all = time.perf_counter()
    chans = chan_results()
    cts = ct_results()
    with tempfile.TemporaryDirectory() as td:
        tss = ts_results(Path(td))
    if probe:
        for c, lam, lag, m in chans:
            print(f"chan {c.name}: lam {lam:.6g} lag {lag} m {m}")
        for ct, r, err in cts:
            print(f"ct {ct.name}: " + (err if r is None else f"K {len(r[0])} starts {r[0][:3]}..."))
        for ts, det, z, info, dt in tss:
            print(f"ts {ts.name}: det {det is not None} Z {z:.3f} {info} ({dt:.2f} s)")
        print(f"total {time.perf_counter() - t_all:.1f} s")
        return
    lines = [
        "// Generated by native/tests/qrss/gen_em_search_pins.py from sstvae/qrss/em.py; do not edit.",
        "// See that script for the formulas test_em_search.cpp rebuilds.",
        "",
        "#pragma once",
        "",
        "#include <cstdint>",
        "#include <vector>",
        "",
        "namespace em_search_pins {",
        "",
        "// --- _search_chan ---",
        "struct Chan {",
        "    const char* name; std::uint64_t seed; std::int64_t n_z; std::vector<std::int64_t> starts;",
        "    std::int64_t chunk, lag_max, m_max; double n0; std::vector<std::int64_t> drift; double embed;",
        "    std::int64_t embed_lag; double lam; std::int64_t lag, m;",
        "};",
        "inline const Chan CHANS[] = {",
    ]
    for c, lam, lag, m in chans:
        lines.append(f"    {{{G.cstr(c.name)}, {c.seed}, {c.n_z}, {ints(c.starts)}, {c.chunk}, {c.lag_max}, "
                     f"{c.m_max}, {G.cdouble(c.n0)}, {ints(c.drift)}, {G.cdouble(c.embed)}, {c.embed_lag}, "
                     f"{G.cdouble(lam)}, {lag}, {m}}},")
    lines += [
        "};",
        "",
        "// --- _chunk_template (case c's mu/nu seeds are 800 + c and 900 + c) ---",
        "struct ChunkT {",
        "    const char* name; const char* spec; const char* keying; double t0, ppm; std::int64_t n_ch, chunk;",
        "    const char* error; std::vector<std::int64_t> starts; std::vector<double> en;",
        "    // chunks k % 8 == 0 and the last: E[k][0], E[k][chunk/2], E[k][chunk-1], each (re, im)",
        "    std::vector<double> samples;",
        "};",
        "inline const ChunkT CHUNK_TEMPLATE[] = {",
    ]
    for ct, r, err in cts:
        if r is None:
            starts, en, samples = [], [], []
        else:
            starts, E, en = r
            samples = []
            for k in sampled_chunks(len(starts)):
                for j in (0, ct.chunk // 2, ct.chunk - 1):
                    samples += [E[k, j].real, E[k, j].imag]
        lines.append(f"    {{{G.cstr(ct.name)}, {G.cstr(ct.spec)}, {G.cstr(ct.keying)}, {G.cdouble(ct.t0)}, "
                     f"{G.cdouble(ct.ppm)}, {ct.n_ch}, {ct.chunk}, {G.cstr(err)},")
        lines.append(f"     {ints(starts)},")
        lines.append(f"     {dbls(en)},")
        lines.append(f"     {dbls(samples)}}},")
    lines += [
        "};",
        "",
        "// --- template_search on the formula store ---",
        f"inline constexpr std::int64_t Q0 = {Q0};",
        f"inline constexpr double T0 = {T0!r};",
        f"inline constexpr const char* CALL = {G.cstr(CALL)};",
        f"inline constexpr const char* GRID = {G.cstr(GRID)};",
        f"inline constexpr std::int64_t PICTURE_ID = {PICTURE_ID};",
        f"inline constexpr std::int64_t CODEC_ID = {CODEC_ID};",
        f"inline constexpr std::int64_t MODE = {MODE};",
        f"inline constexpr double PASS_F[] = {{{dbls(PASS_F)[1:-1]}}};",
        f"inline constexpr double PASS_DRIFT[] = {{{dbls(PASS_DRIFT)[1:-1]}}};",
        f"inline constexpr double PASS_PPM[] = {{{dbls(PASS_PPM)[1:-1]}}};",
        f"inline constexpr double PASS_W = {G.cdouble(PASS_W)};",
        f"inline constexpr double T0_CH = {G.cdouble(T0_CH)};",
        f"inline constexpr double T0_FE = {G.cdouble(T0_FE)};",
        f"inline constexpr std::int64_t CAP_Q = {CAP_Q};",
        "struct Search {",
        "    const char* name; const char* spec; std::uint64_t seed; double amp; std::int64_t seg, cand, lag, k,",
        "        m0, f_mix; bool fe; std::vector<double> freqs; bool has_segments; std::vector<std::int64_t>",
        "        segments; double tau_s, df_hz; std::int64_t ppm_steps; double z_accept;",
        "    // the reference's answer",
        "    bool det; double z; bool has_info; double lam; std::int64_t chunks; double cells;",
        "    std::int64_t segment; double f_hz, ppm, tau0, drift;",
        "};",
        "inline const Search TEMPLATE_SEARCH[] = {",
    ]
    for ts, det, z, info, _ in tss:
        segs = [] if ts.segments is None else list(ts.segments)
        if info:
            got = (f"{G.cdouble(info['lam'])}, {int(info['chunks'])}, {G.cdouble(info['cells'])}, "
                   f"{int(info['segment'])}, {G.cdouble(info['f_hz'])}, {G.cdouble(info['ppm'])}, "
                   f"{G.cdouble(info['tau0'])}, {G.cdouble(info['drift'])}")
        else:
            got = "0.0, 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0"
        lines.append(f"    {{{G.cstr(ts.name)}, {G.cstr(ts.spec)}, {ts.seed}, {G.cdouble(ts.amp)}, {ts.seg}, "
                     f"{ts.cand}, {ts.lag}, {ts.k}, {ts.m0}, {ts.f_mix}, {'true' if ts.fe else 'false'}, "
                     f"{dbls(ts.freqs)}, {'false' if ts.segments is None else 'true'}, {ints(segs)}, "
                     f"{G.cdouble(ts.tau_s)}, {G.cdouble(ts.df_hz)}, {ts.ppm_steps}, {G.cdouble(ts.z_accept)},")
        lines.append(f"     {'true' if det is not None else 'false'}, {G.cdouble(z)}, "
                     f"{'true' if info else 'false'}, {got}}},")
    lines += ["};", "", "}  // namespace em_search_pins", ""]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT} ({time.perf_counter() - t_all:.1f} s)")


if __name__ == "__main__":
    main()
