"""The portable C reference beacon, `qrss_beacon_c/` (design 9, 10.7: C1-C3).

The C is built once per session with whatever C compiler is on the PATH
(`$CC`, then cc, gcc, clang) into a temporary directory, and driven
through its `qrss_ce_test` command-line driver. Without a compiler every
test here except the table check skips.

- C1: SHA-256 against FIPS 180-4 and hashlib; the preamble, references and
  scrambler (three slots) bit for bit; the stream symbols of a beacon file
  equal `tx.slot_symbols` on the same file.
- C2: the integer phase is within 2e-4 rad rms and 1e-3 rad max of
  `si5351.target_phase` over [-8T, 600 s] of a SHORT beacon file, for FSK
  and on-off keyed callsign windows, at 990, 250 and 8000 Hz updates.
- C3: the C Si5351 steps equal `si5351.si5351_steps` driven by the same
  phase (its `phase_fn` seam) for three slots at 990 Hz.

Plus: the C has no undefined behaviour (a UBSan build gives the same
bytes) and no dependence on int's width (no shift of a bare literal, for
16-bit-int AVR ports); fractional update rates (fu_den != 1) and the
Si5351 arithmetic up to fu_num = 2^21; TINY's keyed end; and refusal of
a callsign outside its room or unknown flags. `qrss_tables.h` is what `tools/gen_qrss_tables.py` generates, and
the C derives the same frame lengths as `frame.FrameSpec` for every preset
(the slot timing lives in one place).
"""

import hashlib
import os
import shutil
import struct
import subprocess
import sys
import zlib
from fractions import Fraction

import numpy as np
import pytest

from qrss_helpers import Q_TEST, REPO_ROOT, synthetic_full_latents
from sstvae.qrss import beaconfile, frame, picture, sequences, si5351, tx
from sstvae.qrss.constants import CW_UNITS, T_SYM
from sstvae.qrss.header import HeaderFields

C_DIR = REPO_ROOT / "qrss_beacon_c"
SOURCES = ("qrss_ce.c", "sha256.c", "test_main.c")
CALL = "K1ABC/P"
Q_SLOTS = (Q_TEST, Q_TEST + 1, (1 << 40) + 12345)    # C1/C3: three slots, one past 2^32


def _compiler() -> str | None:
    for cand in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if cand and shutil.which(cand):
            return cand
    return None


@pytest.fixture(scope="session")
def exe(tmp_path_factory):
    """Path to a freshly built qrss_ce_test, or a skip."""
    cc = _compiler()
    if cc is None:
        pytest.skip("no C compiler on the PATH")
    out = tmp_path_factory.mktemp("qrss_c")
    if shutil.which("make"):
        cmd = ["make", "-s", "-C", str(C_DIR), f"BUILD={out}", f"CC={cc}"]
    else:
        cmd = [cc, "-O2", "-std=c99", "-Wall", "-Wextra", "-pedantic", "-o",
               str(out / "qrss_ce_test")] + [str(C_DIR / s) for s in SOURCES]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"build failed:\n{r.stdout}\n{r.stderr}"
    assert "warning" not in (r.stdout + r.stderr).lower(), r.stdout + r.stderr
    return str(out / "qrss_ce_test")


def _run(exe, *args, text=False) -> bytes | str:
    r = subprocess.run([exe, *map(str, args)], capture_output=True, text=text)
    assert r.returncode == 0, f"{args}: rc {r.returncode}: {r.stderr}"
    return r.stdout


@pytest.fixture(scope="module")
def sp():
    return picture.StoredPicture.from_latents(synthetic_full_latents(3), 0xD1D8, 1)


def _bf(sp, ook=False, segment=1):
    h = HeaderFields(callsign=CALL, grid="FN42", picture_id=sp.picture_id, mode=sp.mode,
                     segment=segment, codec_id=sp.codec_id)
    return beaconfile.from_picture(sp, segment, h, ook=ook)


@pytest.fixture(scope="module")
def bin_files(sp, tmp_path_factory):
    """{ook: (path, BeaconFile)} for a full segment, FSK and on-off keyed."""
    d = tmp_path_factory.mktemp("qrss_bin")
    out = {}
    for ook in (False, True):
        bf = _bf(sp, ook)
        path = d / f"seg_{'ook' if ook else 'fsk'}.bin"
        beaconfile.write(path, bf)
        out[ook] = (str(path), bf)
    return out


# --- tables ------------------------------------------------------------------------------

def test_tables_are_current():
    """qrss_tables.h is exactly what tools/gen_qrss_tables.py generates now."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        import gen_qrss_tables
    finally:
        sys.path.pop(0)
    assert (C_DIR / "qrss_tables.h").read_text() == gen_qrss_tables.render(), \
        "qrss_tables.h is stale: run tools/gen_qrss_tables.py"
    r = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "gen_qrss_tables.py"),
                        "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def _duration(show_text: str) -> float:
    return float(show_text.split("duration ")[1].split()[0])


def test_c_frame_lengths_follow_frame_spec(exe, bin_files):
    """The C derives n_sym, n_pos and the duration from the preset table, and
    they agree with FrameSpec for every preset (the slot timing is one edit)."""
    path, _ = bin_files[False]
    for spec in frame.PRESETS.values():
        txt = _run(exe, "show", path, Q_TEST, spec.name, text=True)
        f = dict(zip(txt.split()[3::2], txt.split()[4::2]))
        assert int(f["n_sym"]) == spec.n_sym and int(f["n_pos"]) == spec.n_pos, txt
        assert _duration(txt) == pytest.approx(spec.duration_s, abs=1e-6)
    # Without a frame argument the file's own latent count makes the FULL frame.
    txt = _run(exe, "show", path, Q_TEST, text=True)
    assert f"n_pos {frame.FULL.n_pos} " in txt


# --- C1: SHA-256 and the sequences -------------------------------------------------------

def test_c1_sha256_fips_vectors(exe):
    """C1: the C SHA-256 passes the FIPS 180-4 vectors (incl. one million 'a')."""
    assert "selftest ok" in _run(exe, "selftest", text=True)


def test_c1_sha256_equals_hashlib(exe):
    """C1: digests of messages across the 55/56/64-byte padding edges match hashlib."""
    rng = np.random.default_rng(7)
    for n in (0, 1, 31, 54, 55, 56, 57, 63, 64, 65, 119, 120, 128, 200, 1000):
        msg = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        got = _run(exe, "sha256", msg.hex() if n else "", text=True).strip()
        assert got == hashlib.sha256(msg).hexdigest(), n


def _seq(exe, which, q, start, n) -> np.ndarray:
    s = _run(exe, "seq", which, q, start, n, text=True).strip()
    return np.frombuffer(s.encode(), np.uint8) - ord("0")


def test_c1_sequences_bit_for_bit(exe):
    """C1: preamble (660), references (FULL's 3,539) and the scrambler of
    three slots (50,600 each) equal Python's sha_bits streams."""
    n_ref = frame.FULL.n_ref
    assert np.array_equal(_seq(exe, "pre", 0, 0, 660),
                          sequences.sha_bits(sequences.PREAMBLE_DOMAIN, 660))
    assert np.array_equal(_seq(exe, "ref", 0, 0, n_ref),
                          sequences.sha_bits(sequences.REFERENCE_DOMAIN, n_ref))
    for q in Q_SLOTS:
        c = _seq(exe, "scr", q, 0, 50600)
        assert np.array_equal(1 - 2 * c.astype(np.int8), sequences.scrambler(q, 50600)), q
    # An offset start lands on the same bits (the beacon reads them piecemeal).
    assert np.array_equal(_seq(exe, "scr", Q_TEST, 1000, 300),
                          (1 - sequences.scrambler(Q_TEST, 1300)[1000:]) // 2)


def _c_symbols(exe, path, q, spec) -> np.ndarray:
    vc = np.frombuffer(_run(exe, "sym", path, q, spec.name), np.int32).reshape(-1, 2)
    assert len(vc) == spec.n_sym
    m = 2.0 ** (vc[:, 1] - 1)
    return np.where(vc[:, 1] == 0, 1.0, 1.0 / (20 * np.sqrt(m))) * vc[:, 0]


@pytest.mark.parametrize("name", ["full", "short", "tiny"])
def test_c1_stream_symbols_equal_python(exe, bin_files, name):
    """C1: preamble, references, header bits and spare, and the precoded int8
    data (sign flips, then the WHT in integers) equal tx.slot_symbols."""
    spec = frame.PRESETS[name]
    path, bf = bin_files[False]
    for q in Q_SLOTS[:2] if name == "short" else Q_SLOTS[:1]:
        np.testing.assert_allclose(_c_symbols(exe, path, q, spec),
                                   tx.slot_symbols(bf, q, spec), rtol=0, atol=1e-12)


def test_c1_tail_blocks_and_custom_frames(exe, bin_files):
    """Every precoder tail size (n_data % 64 = 47 -> 32, 8, 4, 2, 1) and a
    frame with its last window ending it, given to the C as an explicit
    N_HDR,N_DATA,CW... frame, equal tx.slot_symbols."""
    path, bf = bin_files[False]
    spec = frame.FrameSpec("tail", n_data=4096 + 47, cw_after=())
    arg = f"{spec.n_hdr},{spec.n_data}"
    vc = np.frombuffer(_run(exe, "sym", path, Q_TEST, arg), np.int32).reshape(-1, 2)
    assert sorted(set(vc[:, 1].tolist())) == [0, 1, 2, 3, 4, 6, 7]
    m = 2.0 ** (vc[:, 1] - 1)
    got = np.where(vc[:, 1] == 0, 1.0, 1.0 / (20 * np.sqrt(m))) * vc[:, 0]
    np.testing.assert_allclose(got, tx.slot_symbols(bf, Q_TEST, spec), rtol=0, atol=1e-12)

    n_sym = frame.FrameSpec("end", n_data=2000, cw_after=()).n_sym
    spec = frame.FrameSpec("end", n_data=2000, cw_after=(4000, n_sym))
    arg = ",".join(map(str, (spec.n_hdr, spec.n_data) + spec.cw_after))
    txt = _run(exe, "show", path, Q_TEST, arg, text=True)
    assert f"n_pos {spec.n_pos} " in txt
    assert _duration(txt) == pytest.approx(spec.duration_s, abs=1e-6)
    # Its phase, through the window that ends the frame, and its keyed span.
    fu, n = 250, int((spec.n_pos + 30) * T_SYM * 250)
    ph_c = np.frombuffer(_run(exe, "phase", path, Q_TEST, arg, fu, 1, 0, n, 1),
                         np.uint32).astype(np.float64) * (2 * np.pi / 2 ** 32)
    tau = -8.0 + np.arange(n) / (fu * T_SYM)
    ph_p = si5351.target_phase(tau, tx.slot_symbols(bf, Q_TEST, spec), spec, bf.keying)
    assert np.max(np.abs(np.angle(np.exp(1j * (ph_c - ph_p))))) <= 1e-3
    keyed = np.frombuffer(_run(exe, "keyed", path, Q_TEST, arg, fu, 1, 0, n, 1), np.uint8)
    assert np.array_equal(keyed, (tau < spec.keyed_end_pos).astype(np.uint8))


# --- C2: phase ----------------------------------------------------------------------------

def _c_phase(exe, path, q, spec, fu, n, step=1) -> np.ndarray:
    out = _run(exe, "phase", path, q, spec.name, fu, 1, 0, n, step)
    return np.frombuffer(out, np.uint32).astype(np.float64) * (2 * np.pi / 2 ** 32)


@pytest.mark.parametrize("ook", [False, True], ids=["fsk", "ook"])
def test_c2_phase_matches_python(exe, bin_files, ook):
    """C2: C phase vs si5351.target_phase (phi + 2 pi theta_cw) over [-8T, 600 s]
    of a SHORT beacon file: <= 2e-4 rad rms, <= 1e-3 rad max, at 990, 250 and
    8000 Hz updates. At 8000 Hz every 4th update is compared (4 is coprime to
    485, so every 1/485-symbol grid phase is still visited)."""
    spec = frame.SHORT
    path, bf = bin_files[ook]
    sym = tx.slot_symbols(bf, Q_TEST, spec)
    for fu, step in ((990, 1), (250, 1), (8000, 4)):
        n_all = int((600.0 + 8 * T_SYM) * fu) + 1
        n = -(-n_all // step)
        ph_c = _c_phase(exe, path, Q_TEST, spec, fu, n, step)
        tau = -8.0 + np.arange(n) * step / (fu * T_SYM)
        ph_p = si5351.target_phase(tau, sym, spec, bf.keying, ook=ook)
        err = np.angle(np.exp(1j * (ph_c - ph_p)))
        rms, mx = np.sqrt(np.mean(err ** 2)), np.max(np.abs(err))
        print(f"C2 {'ook' if ook else 'fsk'} {fu} Hz: {rms:.2e} rad rms, {mx:.2e} max")
        assert rms <= 2e-4 and mx <= 1e-3, (fu, rms, mx)
    # Past the end of the frame the FSK windows have left whole turns behind.
    assert abs(np.angle(np.exp(1j * ph_c[-1]))) < 1e-6


def test_c2_keying_envelope(exe, bin_files):
    """qrss_ce_keyed follows the keyed span and, on-off keyed, the Morse in
    units 5..183 of each window (ce.cw_phase_turns' amplitude)."""
    from sstvae.qrss import ce
    spec = frame.SHORT
    fu = 250
    n = int((spec.keyed_end_pos + 20) * T_SYM * fu)
    tau = -8.0 + np.arange(n) / (fu * T_SYM)
    lo, hi = ce.keyed_span_pos(spec)
    for ook in (False, True):
        path, bf = bin_files[ook]
        k_c = np.frombuffer(_run(exe, "keyed", path, Q_TEST, "short", fu, 1, 0, n, 1), np.uint8)
        a = ce.cw_phase_turns(tau, spec, bf.keying, ook=ook)[1]
        k_p = ((tau >= lo) & (tau < hi)) * a
        assert np.array_equal(k_c, k_p.astype(np.uint8)), ook
        if ook:
            assert k_c.sum() < n - 100           # the call is keyed off in places


# --- C3: Si5351 steps -----------------------------------------------------------------------

@pytest.mark.parametrize("ook", [False, True], ids=["fsk", "ook"])
def test_c3_si5351_steps_identical(exe, bin_files, ook):
    """C3: C steps (0.4 Hz, 990 Hz updates, the whole SHORT frame) equal
    si5351_steps driven by the C phase through `phase_fn`, for three slots.
    The phase they reach stays within half a step (plus C2's 1e-3 rad) of
    Python's own closed-form target."""
    spec = frame.SHORT
    path, bf = bin_files[ook]
    fu = 990
    n = int((spec.keyed_end_pos + 8) * T_SYM * fu) + 100
    for q in Q_SLOTS if not ook else Q_SLOTS[:1]:
        turns = np.frombuffer(_run(exe, "phase", path, q, "short", fu, 1, 0, n + 1, 1),
                              np.uint32).astype(np.int64)
        d = (np.diff(turns) + 2 ** 31) % 2 ** 32 - 2 ** 31          # unwrap
        unwrapped = (turns[0] + np.concatenate([[0], np.cumsum(d)])) * (2 * np.pi / 2 ** 32)

        def phase_fn(tau):
            u = np.rint((np.asarray(tau) + 8.0) * fu * T_SYM).astype(np.int64)
            return unwrapped[u]

        sym = tx.slot_symbols(bf, q, spec)
        want = si5351.si5351_steps(sym, spec, q, Fraction(fu), 0.4, n, keying=bf.keying,
                                   ook=ook, phase_fn=phase_fn)
        got = np.frombuffer(_run(exe, "si5351", path, q, "short", fu, 1, 400, n), np.int32)
        assert np.array_equal(got, want), (q, np.flatnonzero(got != want)[:10])
        if q == Q_TEST:
            # The phase the synthesizer reaches tracks the *closed-form* target
            # to half a step plus the C phase's own error, through the windows.
            k = 2 * np.pi * 0.4 / fu
            reached = unwrapped[0] + np.cumsum(got * k)
            tgt = si5351.target_phase(si5351.update_times_pos(fu, n + 1)[1:], sym, spec,
                                      bf.keying, ook=ook)
            err = np.max(np.abs(reached - tgt))
            print(f"C3 {'ook' if ook else 'fsk'}: reached phase within {err:.2e} rad of "
                  f"the closed-form target (half a step is {k / 2:.2e})")
            assert err <= k / 2 + 1e-3


# --- init refusals ----------------------------------------------------------------------

def _recrc(body: bytes) -> bytes:
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def test_init_refuses_bad_files(exe, bin_files, tmp_path):
    """Bad magic, a flipped byte (CRC-32), an empty or misplaced callsign keying
    mask (the windows are a legal requirement) and a frame larger than the
    file are refused with their own codes."""
    path, _ = bin_files[False]
    good = open(path, "rb").read()

    def rc_of(data: bytes, frame_name="short") -> str:
        p = tmp_path / "bad.bin"
        p.write_bytes(data)
        r = subprocess.run([exe, "sym", str(p), str(Q_TEST), frame_name], capture_output=True,
                           text=True)
        assert r.returncode != 0
        return r.stderr.strip().split()[-1]

    assert rc_of(b"XRSB" + good[4:]) == "-2"
    flipped = bytearray(good)
    flipped[1000] ^= 0x10
    assert rc_of(bytes(flipped)) == "-3"
    no_call = bytearray(good[:-4])
    no_call[32:32 + CW_UNITS // 8] = bytes(CW_UNITS // 8)
    assert rc_of(_recrc(bytes(no_call))) == "-6"
    early = bytearray(good[:-4])
    early[32] |= 0x80                                   # key-down in unit 0
    assert rc_of(_recrc(bytes(early))) == "-6"
    assert rc_of(good[:-10]) == "-1"
    # TINY has no windows, so it needs no callsign; SHORT does.
    p = tmp_path / "nocall.bin"
    p.write_bytes(_recrc(bytes(no_call)))
    assert len(_run(exe, "sym", p, Q_TEST, "tiny")) == 8 * frame.TINY.n_sym


def test_show_prints_phases_and_steps(exe, bin_files):
    """test_main reads a beacon file and prints phases and steps."""
    txt = _run(exe, "show", bin_files[False][0], Q_TEST, "short", text=True)
    assert "FSK" in txt and "phase(deg)" in txt and len(txt.splitlines()) >= 23


# --- portability: no undefined behaviour, no dependence on int's width --------------------

def test_no_shift_of_a_bare_int_literal():
    """A shift of a plain literal (`1 << 18`) is done in `int`, which is 16
    bits on the AVR targets this reference is meant to port to: there
    `1 << 18` is undefined (avr-gcc gives 0). Every shift wider than 15 bits
    must name its type, as `(int64_t)1 << n` does."""
    import re
    for name in ("qrss_ce.c", "sha256.c", "qrss_ce.h", "qrss_tables.h"):
        for i, line in enumerate((C_DIR / name).read_text().splitlines(), 1):
            code = line.split("//")[0].split("/*")[0]
            assert not re.search(r"(?<![\w)])\d+[uU]?\s*<<", code), f"{name}:{i}: {line.strip()}"


@pytest.fixture(scope="session")
def ubsan_exe(tmp_path_factory):
    """qrss_ce_test built with -fsanitize=undefined, aborting on the first report."""
    cc = _compiler()
    if cc is None:
        pytest.skip("no C compiler on the PATH")
    out = tmp_path_factory.mktemp("qrss_ubsan") / "qrss_ce_ubsan"
    cmd = [cc, "-O1", "-std=c99", "-fsanitize=undefined", "-fno-sanitize-recover=all",
           "-o", str(out)] + [str(C_DIR / s) for s in SOURCES]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"no UBSan runtime: {r.stderr.strip()[:200]}")
    return str(out)


@pytest.mark.parametrize("ook", [False, True], ids=["fsk", "ook"])
def test_ubsan_clean(ubsan_exe, exe, bin_files, ook):
    """The phase over a whole SHORT frame (both sides of every window) and the
    Si5351 steps, including at fu_num near the documented 2^21 limit, run
    under -fsanitize=undefined without a report, and give the same bytes as
    the optimised build."""
    path, _ = bin_files[ook]
    n = int((frame.SHORT.keyed_end_pos + 16) * T_SYM * 250)
    for args in (("phase", path, Q_TEST, "short", 250, 1, 0, n, 1),
                 ("keyed", path, Q_TEST, "short", 250, 1, 0, n, 1),
                 ("si5351", path, Q_TEST, "short", 2000000, 2000, 400, 20000),
                 ("si5351", path, Q_TEST, "short", 1 << 21, 3000, 400, 5000)):
        assert _run(ubsan_exe, *args) == _run(exe, *args), args[0]


def test_c3_steps_do_not_depend_on_how_the_rate_is_written(exe, bin_files):
    """1000/1, 1050000/1050, 1100000/1100 and 2000000/2000 Hz are the same
    update rate and must give the same steps: the Si5351 arithmetic is exact
    up to fu_num = 2^21 (it once overflowed past ~1.07e6)."""
    path, _ = bin_files[False]
    n = 40000
    ref = _run(exe, "si5351", path, Q_TEST, "short", 1000, 1, 400, n)
    for num, den in ((1050000, 1050), (1100000, 1100), (2000000, 2000)):
        assert _run(exe, "si5351", path, Q_TEST, "short", num, den, 400, n) == ref, num


# --- fractional update rates (fu_den != 1) ---------------------------------------------------

@pytest.mark.parametrize("num,den", [(1000, 3), (100000, 101)])
def test_c2_c3_fractional_update_rate(exe, bin_files, num, den):
    """At a rate that is not a whole number of Hz the C phase follows
    target_phase at t = u * den / num, and the Si5351 steps equal
    si5351_steps(Fraction(num, den)) driven by that phase, through a callsign
    window (FSK, so its phase is in the steps)."""
    spec = frame.SHORT
    path, bf = bin_files[False]
    fu = Fraction(num, den)
    sym = tx.slot_symbols(bf, Q_TEST, spec)
    # From the start to past the first window's end.
    n = int((spec.win_start_pos[0] + 400) * T_SYM * float(fu))
    turns = np.frombuffer(_run(exe, "phase", path, Q_TEST, "short", num, den, 0, n + 1, 1),
                          np.uint32).astype(np.int64)
    tau = -8.0 + np.arange(n + 1) * den / (num * T_SYM)
    ph_p = si5351.target_phase(tau, sym, spec, bf.keying)
    err = np.angle(np.exp(1j * (turns * (2 * np.pi / 2 ** 32) - ph_p)))
    assert np.sqrt(np.mean(err ** 2)) <= 2e-4 and np.max(np.abs(err)) <= 1e-3

    d = (np.diff(turns) + 2 ** 31) % 2 ** 32 - 2 ** 31
    unwrapped = (turns[0] + np.concatenate([[0], np.cumsum(d)])) * (2 * np.pi / 2 ** 32)

    def phase_fn(t):
        return unwrapped[np.rint((np.asarray(t) + 8.0) * float(fu) * T_SYM).astype(np.int64)]

    want = si5351.si5351_steps(sym, spec, Q_TEST, fu, 0.4, n, keying=bf.keying,
                               phase_fn=phase_fn)
    got = np.frombuffer(_run(exe, "si5351", path, Q_TEST, "short", num, den, 400, n), np.int32)
    assert np.array_equal(got, want), np.flatnonzero(got != want)[:10]


# --- keyed span without windows, and the callsign's room ------------------------------------

def test_keyed_span_tiny_and_fractional(exe, bin_files):
    """TINY has no windows: keying ends 8 symbols after its last symbol's
    position (keyed_span_pos). Checked at an integer and a fractional rate."""
    from sstvae.qrss import ce
    spec = frame.TINY
    path, _ = bin_files[False]
    lo, hi = ce.keyed_span_pos(spec)
    for num, den in ((250, 1), (1000, 3)):
        n = int((hi + 20) * T_SYM * num / den)
        tau = -8.0 + np.arange(n) * den / (num * T_SYM)
        k_c = np.frombuffer(_run(exe, "keyed", path, Q_TEST, "tiny", num, den, 0, n, 1),
                            np.uint8)
        assert np.array_equal(k_c, ((tau >= lo) & (tau < hi)).astype(np.uint8)), (num, den)


def test_init_refuses_callsign_outside_its_room_and_unknown_flags(exe, bin_files, tmp_path):
    """A key-down in units 184..191 (after the call's room) is refused like
    one in the lead (units 0..7); unknown flag bits and a nonzero reserved
    byte are refused as a format error."""
    path, _ = bin_files[False]
    body = open(path, "rb").read()[:-4]

    def rc_of(data: bytes) -> str:
        p = tmp_path / "bad.bin"
        p.write_bytes(_recrc(data))
        r = subprocess.run([exe, "sym", str(p), str(Q_TEST), "short"], capture_output=True,
                           text=True)
        assert r.returncode != 0
        return r.stderr.strip().split()[-1]

    for unit in (7, 184, 188, 191):
        b = bytearray(body)
        b[32 + unit // 8] |= 0x80 >> (unit % 8)
        assert rc_of(bytes(b)) == "-6", unit
    for off, val in ((30, 0x02), (30, 0x80), (31, 0x01)):
        b = bytearray(body)
        b[off] |= val
        assert rc_of(bytes(b)) == "-4", (off, val)
    # The last unit of the room (183) is allowed.
    b = bytearray(body)
    b[32 + 183 // 8] |= 0x80 >> (183 % 8)
    p = tmp_path / "ok.bin"
    p.write_bytes(_recrc(bytes(b)))
    assert len(_run(exe, "sym", p, Q_TEST, "short")) == 8 * frame.SHORT.n_sym
