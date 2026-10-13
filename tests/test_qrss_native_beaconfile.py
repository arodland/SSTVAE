"""C++ beacon files (native/core/qrss/beaconfile) against sstvae.qrss.beaconfile.

Bit-exact: a beacon file is integers, packed bits, int8 latents and a
CRC-32, so the bytes agree exactly -- segments 0 and 1, FSK and on-off
keyed, with a grid and without. Every refusal of `from_bytes` and of
`check` is compared by type and text, including every single-byte flip of
a whole file, and the C reference beacon (qrss_beacon_c/) reads a file the
C++ wrote and sends the same symbols as the reference does from it.
BeaconFile stays the reference's dataclass, wrapped by the adapters in
tests/qrss_native/beaconfile.py (what --native-qrss installs).
"""

import dataclasses
import os
import shutil
import struct
import subprocess
import zlib

import numpy as np
import pytest

from qrss_helpers import Q_TEST, REPO_ROOT, synthetic_full_latents
from qrss_native import beaconfile as rows_mod
from qrss_native import require
from sstvae.qrss import beaconfile as ref
from sstvae.qrss import frame, picture, tx
from sstvae.qrss.constants import CW_UNITS, N_HDR_BITS, SPREAD_RHO
from sstvae.qrss.header import HeaderFields

CALL = "K1ABC/P"


@pytest.fixture(scope="module")
def nat(native):
    return require(native, "beaconfile")


@pytest.fixture(scope="module")
def ad(native):
    """The --native-qrss adapters by attribute name."""
    require(native, "beaconfile")
    return {attr: f for (_, attr), f in rows_mod.rows(native.qrss).items()}


@pytest.fixture(scope="module")
def sp():
    return picture.StoredPicture.from_latents(synthetic_full_latents(3), 0xD1D8, 1)


def _header(sp, segment=1, call=CALL, grid="FN42"):
    return HeaderFields(callsign=call, grid=grid, picture_id=sp.picture_id, mode=sp.mode,
                        segment=segment, codec_id=sp.codec_id)


def _outcome(f, *args, **kw):
    try:
        return ("ok", f(*args, **kw))
    except Exception as e:      # noqa: BLE001 -- any refusal is compared
        return ("raise", type(e), str(e))


def _same_bf(want, got):
    assert isinstance(got, ref.BeaconFile)
    for f in ("waveform", "mode", "segment", "picture_id", "codec_id", "callsign", "grid", "ook"):
        assert getattr(got, f) == getattr(want, f) and type(getattr(got, f)) is type(
            getattr(want, f)), f
    for f in ("keying", "hdr_bits", "latents_i8"):
        a, b = np.asarray(getattr(want, f)), np.asarray(getattr(got, f))
        assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), f


def _same_outcome(f_ref, f_nat, *args, **kw):
    want, got = _outcome(f_ref, *args, **kw), _outcome(f_nat, *args, **kw)
    assert got[0] == want[0], (want, got)
    if want[0] == "raise":
        assert got[1:] == want[1:]
    elif isinstance(want[1], ref.BeaconFile):
        _same_bf(want[1], got[1])
    else:
        assert got[1] == want[1] and type(got[1]) is type(want[1])
    return want


def _recrc(body: bytes) -> bytes:
    """A file body (without its CRC) with a valid CRC-32 appended."""
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


# --- writing ---------------------------------------------------------------------------------

@pytest.mark.parametrize("segment", [0, 1])
@pytest.mark.parametrize("ook", [False, True])
@pytest.mark.parametrize("grid", ["FN42", None])
def test_bytes_identical(nat, ad, sp, segment, ook, grid, tmp_path):
    """from_picture and to_bytes give the reference's bytes; both read either."""
    h = _header(sp, segment, grid=grid)
    want = ref.from_picture(sp, segment, h, ook=ook)
    got = ad["from_picture"](sp, segment, h, ook)
    _same_bf(want, got)
    data = want.to_bytes()
    assert len(data) == 50970 == nat.file_size() == ref.file_size()
    assert nat.to_bytes(want) == data == nat.to_bytes(got) == ref.BeaconFile.to_bytes(got)
    _same_bf(want, ad["from_bytes"].__func__(ref.BeaconFile, data))
    _same_bf(ref.BeaconFile.from_bytes(data), ad["from_bytes"].__func__(ref.BeaconFile,
                                                                           bytearray(data)))
    p_ref, p_nat = tmp_path / "r.bin", tmp_path / "n.bin"
    ref.write(p_ref, want)
    nat.write(p_nat, got)
    assert p_ref.read_bytes() == p_nat.read_bytes()
    _same_bf(ref.read(p_nat), ad["read"](str(p_ref)))
    assert ad["header_fields"](want) == want.header_fields()


def test_constants(nat):
    for name in ("MAGIC", "FILE_VERSION", "FLAG_OOK", "HEAD_BYTES", "KEYING_BYTES",
                 "HDR_BYTES", "LATENT_OFFSET"):
        assert getattr(nat, name) == getattr(ref, name), name
    for n in (1, 100, 50600, 0, 7.0, np.int16(5)):
        assert nat.file_size(n) == ref.file_size(n)


def test_shorter_latents_round_trip(nat, ad, sp):
    """A file of fewer latents than a full segment is valid; one of 1 too."""
    full = ref.from_picture(sp, 0, _header(sp, 0))
    for n in (1, 2, 1000, 50599):
        bf = dataclasses.replace(full, latents_i8=full.latents_i8[:n].copy())
        data = ref.BeaconFile.to_bytes(bf)
        assert nat.to_bytes(bf) == data
        _same_outcome(ref.BeaconFile.from_bytes,
                      lambda d: ad["from_bytes"].__func__(ref.BeaconFile, d), data)


# --- reading: refusals ------------------------------------------------------------------------

def test_every_single_byte_flip_is_refused(nat, sp):
    """Every byte of a whole file flipped (one bit, varying): the same refusal."""
    data = bytearray(ref.BeaconFile.to_bytes(ref.from_picture(sp, 1, _header(sp, 1))))
    for i in range(len(data)):
        mask = 1 << (i % 8)
        data[i] ^= mask
        want, got = _outcome(ref.BeaconFile.from_bytes, data), _outcome(nat.from_bytes, data)
        data[i] ^= mask
        assert want[0] == got[0] == "raise", (i, want, got)
        assert got[1:] == want[1:], (i, want, got)


def test_refusals_under_a_valid_crc(nat, ad, sp):
    """Refusals the CRC does not catch: each field edited and the CRC redone."""
    good = ref.BeaconFile.to_bytes(ref.from_picture(sp, 1, _header(sp, 1), ook=True))
    body = bytearray(good[:-4])
    lo = ref.LATENT_OFFSET
    ko, ho = ref.HEAD_BYTES, ref.HEAD_BYTES + ref.KEYING_BYTES

    def edit(fn):
        b = bytearray(body)
        fn(b)
        return _recrc(bytes(b))

    def setb(off, val):
        def f(b):
            b[off] = val
        return f

    cases = {
        "magic": b"XRSB" + good[4:],
        "short": good[:lo + 4],
        "empty": b"",
        "minimal_bad": good[:lo + 5],
        "version": edit(setb(4, 2)),
        "waveform": edit(setb(5, 1)),
        "mode": edit(setb(6, 2)),
        "mode_big": edit(setb(6, 7)),
        "segment": edit(setb(7, 0)),
        "segment_big": edit(setb(7, 9)),
        "pid": edit(setb(8, body[8] ^ 1)),
        "codec": edit(setb(12, body[12] ^ 1)),
        "call_nonascii": edit(setb(14, 0xC3)),
        "call_lower": edit(setb(14, ord("k"))),
        "call_space_inside": edit(setb(15, ord(" "))),
        "call_bad_char": edit(setb(16, ord("#"))),
        "call_blank": edit(lambda b: b.__setitem__(slice(14, 22), b" " * 8)),
        "grid": edit(setb(22, 0xFF)),
        "grid_none": edit(lambda b: b.__setitem__(slice(22, 24), b"\0\0")),
        "n_lat": edit(lambda b: b.__setitem__(slice(24, 28), struct.pack("<I", 50599))),
        "n_lat_zero": edit(lambda b: b.__setitem__(slice(24, 28), struct.pack("<I", 0))),
        "scale": edit(setb(28, 21)),
        "scale_hi": edit(setb(29, 1)),
        "flags": edit(setb(30, 0x03)),
        "flags_ook_off": edit(setb(30, 0x00)),
        "reserved": edit(setb(31, 1)),
        "flags_and_reserved": edit(lambda b: b.__setitem__(slice(30, 32), b"\x80\xff")),
        "keying": edit(setb(ko + 3, body[ko + 3] ^ 0x40)),
        "keying_blank": edit(lambda b: b.__setitem__(slice(ko, ho), bytes(ref.KEYING_BYTES))),
        "header_bit": edit(setb(ho + 10, body[ho + 10] ^ 0x08)),
        "header_pad": edit(setb(lo - 1, body[lo - 1] | 0x01)),
        "latent_m128": edit(setb(lo + 17, 0x80)),
        "latent_ok": edit(setb(lo + 17, 0x7F)),
        "too_long": _recrc(bytes(body) + b"\0"),
    }
    from_bytes = ad["from_bytes"].__func__
    for name, data in cases.items():
        want = _same_outcome(ref.BeaconFile.from_bytes,
                             lambda d: from_bytes(ref.BeaconFile, d), data)
        assert want[0] == ("ok" if name in ("latent_ok", "flags_ook_off") else "raise"), \
            (name, want)
    # The ones the task names, by their texts.
    for name, text in (("magic", "not a QRSSTVAE beacon file"), ("latent_m128", "-128"),
                       ("keying", "keying mask"), ("header_bit", "header coded bits"),
                       ("header_pad", "header pad bits"), ("reserved", "reserved byte"),
                       ("flags", "unknown flags 0x03")):
        with pytest.raises(ValueError, match=text):
            nat.from_bytes(cases[name])
    # Arguments as from_bytes takes them: bytes(), memoryview, a list of ints.
    for data in (memoryview(good), list(good[:500]), good):
        _same_outcome(ref.BeaconFile.from_bytes, lambda d: from_bytes(ref.BeaconFile, d), data)


def test_check_refusals(nat, ad, sp, tmp_path):
    """BeaconFile.check (and so to_bytes and write) on edited dataclasses."""
    bf = ref.from_picture(sp, 1, _header(sp, 1))
    rep = dataclasses.replace
    lat = bf.latents_i8
    cases = [
        bf,
        rep(bf, waveform=1),
        rep(bf, callsign="K1ABC/P "),
        rep(bf, callsign="K1ABC/Q"),
        rep(bf, callsign="k1abc/p"),
        rep(bf, callsign="TOOLONGCALL"),
        rep(bf, callsign=""),
        rep(bf, keying=bf.keying[:-1]),
        rep(bf, keying=np.zeros(CW_UNITS, np.uint8)),
        rep(bf, keying=bf.keying.astype(np.float64)),
        rep(bf, keying=list(bf.keying)),
        rep(bf, hdr_bits=bf.hdr_bits[:-1]),
        rep(bf, hdr_bits=1 - bf.hdr_bits),
        rep(bf, hdr_bits=bf.hdr_bits.reshape(2, -1)),
        rep(bf, grid="FN43"),
        rep(bf, grid=None),
        rep(bf, mode=2),
        rep(bf, segment=0),
        rep(bf, picture_id=bf.picture_id ^ 1),
        rep(bf, latents_i8=lat.astype(np.int16)),
        rep(bf, latents_i8=lat.astype(np.float64)),
        rep(bf, latents_i8=lat[:0]),
        rep(bf, latents_i8=np.concatenate([lat, lat[:1]])),
        rep(bf, latents_i8=lat.reshape(2, -1)),
        rep(bf, latents_i8=np.where(np.arange(len(lat)) == 9, np.int8(-128), lat)),
        rep(bf, latents_i8=list(lat[:10])),
        rep(bf, waveform=1, latents_i8=lat.astype(np.float32)),
        rep(bf, ook=True),
    ]
    for i, b in enumerate(cases):
        _same_outcome(ref.BeaconFile.check, ad["check"], b)
        _same_outcome(ref.BeaconFile.to_bytes, ad["to_bytes"], b)
        want = _outcome(ref.write, tmp_path / f"r{i}.bin", b)
        got = _outcome(nat.write, tmp_path / f"n{i}.bin", b)
        assert got[0] == want[0] and got[1:] == want[1:]
        # A refused write leaves an empty file in both (the file is opened first).
        assert (tmp_path / f"r{i}.bin").read_bytes() == (tmp_path / f"n{i}.bin").read_bytes()


def test_from_picture_refusals(nat, ad, sp):
    h = _header(sp, 1)
    rep = dataclasses.replace
    bad_sp = picture.StoredPicture(sp.mode, sp.codec_id, sp.segs, sp.picture_id ^ 1)
    for args in ((sp, 2, h), (sp, -1, h), (sp, 0, h), (sp, 1, rep(h, mode=2)),
                 (sp, 1, rep(h, picture_id=1)), (sp, 1, rep(h, codec_id=1)),
                 (sp, 1, rep(h, callsign="K1ABC/P  ")), (sp, 1, rep(h, callsign="bad call")),
                 (sp, 1, rep(h, grid="ZZ99")), (sp, 1, rep(h, grid="fn42")),
                 (bad_sp, 1, h), (sp, np.int64(1), h), (sp, 1.0, h), (sp, "1", h),
                 (sp, 1, rep(h, callsign="W1AW")), (sp, 1, h, 1)):
        _same_outcome(ref.from_picture, ad["from_picture"], *args)


# --- the C reference reads a C++-written file ----------------------------------------------------

def _compiler():
    for cand in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if cand and shutil.which(cand):
            return cand
    return None


@pytest.fixture(scope="module")
def c_exe(tmp_path_factory):
    """qrss_beacon_c's test driver, built as tests/test_qrss_c_ref.py builds it."""
    cc = _compiler()
    if cc is None:
        pytest.skip("no C compiler on the PATH")
    out = tmp_path_factory.mktemp("qrss_c_native")
    c_dir = REPO_ROOT / "qrss_beacon_c"
    if shutil.which("make"):
        cmd = ["make", "-s", "-C", str(c_dir), f"BUILD={out}", f"CC={cc}"]
    else:
        cmd = [cc, "-O2", "-std=c99", "-o", str(out / "qrss_ce_test")] + [
            str(c_dir / s) for s in ("qrss_ce.c", "sha256.c", "test_main.c")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"build failed:\n{r.stdout}\n{r.stderr}"
    return str(out / "qrss_ce_test")


def _c_values(exe, path, q, name):
    r = subprocess.run([exe, "sym", str(path), str(q), name], capture_output=True)
    assert r.returncode == 0, r.stderr
    vc = np.frombuffer(r.stdout, np.int32).reshape(-1, 3)
    m = 2.0 ** (vc[:, 1] - 1)
    return (np.where(vc[:, 1] == 0, 1.0, 1.0 / (20 * np.sqrt(m))) * vc[:, 0]
            + np.sqrt(SPREAD_RHO) * vc[:, 2])


@pytest.mark.parametrize("ook", [False, True])
def test_c_beacon_reads_a_native_file(nat, ad, sp, c_exe, ook, tmp_path):
    bf = ad["from_picture"](sp, 1, _header(sp, 1), ook)
    path = tmp_path / "native.bin"
    nat.write(path, bf)
    back = ref.read(path)
    for name in ("short", "tiny"):
        spec = frame.PRESETS[name]
        np.testing.assert_allclose(_c_values(c_exe, path, Q_TEST, name),
                                   tx.slot_symbols(back, Q_TEST, spec), rtol=0, atol=1e-12)
    txt = subprocess.run([c_exe, "show", str(path), str(Q_TEST), "short"], capture_output=True,
                         text=True)
    assert txt.returncode == 0 and ("on-off keyed" if ook else "FSK") in txt.stdout


def test_header_pad_is_the_only_slack(sp):
    """N_HDR_BITS is not a whole number of bytes, so the pad test above is real."""
    assert N_HDR_BITS % 8 != 0


# --- review probes -------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sp_c():
    """A mode-C picture, so segment 2 and the mode byte's top value are covered."""
    return picture.StoredPicture.from_latents(synthetic_full_latents(9), 0x0F0F, 2)


def test_from_bytes_field_values_under_a_valid_crc(ad, sp_c):
    """Every value class of every header byte, CRC redone: magic quoting,
    callsign bytes Python's repr escapes, every grid code class, latent
    counts, flags and reserved, scale, version, waveform, mode, segment,
    lengths around the minimum, and the argument types bytes() reads."""
    good = ref.BeaconFile.to_bytes(ref.from_picture(sp_c, 2, _header(sp_c, 2)))
    body = bytearray(good[:-4])
    from_bytes = ad["from_bytes"].__func__

    def edit(off, val):
        b = bytearray(body)
        b[off:off + len(val)] = val
        return _recrc(bytes(b))

    datas = [m + good[4:] for m in (b"'AB\"", b"'ABC", b"\"ABC", b"\\\t\n\r", b"\x7f\x80\xff\x00",
                                    b"QRS\x00", b"qrsb")]
    datas += [edit(14, c) for c in (b"K1\x00ABC  ", b"K1'ABC  ", b"K1\"ABC  ", b"K1\\ABC  ",
                                    b"K1\tABC  ", b"K1\x7fABC  ", b"K1'\"ABC ", b"ABCDEFGH",
                                    b" K1ABC  ", b"K1ABC/P\x00", b"K1\x80'ABC ")]
    datas += [edit(22, struct.pack("<H", g)) for g in (0, 1, 32399, 32400, 32766, 32767, 32768, 65535)]
    datas += [edit(24, struct.pack("<I", n)) for n in (2 ** 32 - 1, 50601, 50600, 1)]
    datas += [edit(30, bytes([fl, r])) for fl in (0x01, 0x02, 0x80, 0xFF) for r in (0, 1, 255)]
    datas += [edit(28, struct.pack("<H", v)) for v in (0, 65535, 276)]
    datas += [edit(4, bytes([v])) for v in (0, 255)] + [edit(5, bytes([255]))]
    datas += [edit(6, bytes([v])) for v in (3, 255)] + [edit(7, bytes([3]))]
    for n in (0, 1, 369, 370, 371, len(good) - 1, len(good) + 1):
        d = (good * 2)[:n]
        datas += [d, _recrc(d[:-4]) if n >= 4 else d]
    datas += [bytearray(good), memoryview(good), np.frombuffer(good, np.uint8), "abc", 5, None, [256]]
    for d in datas:
        _same_outcome(ref.BeaconFile.from_bytes, lambda x: from_bytes(ref.BeaconFile, x), d)


def test_check_on_other_field_types(ad, sp_c, tmp_path):
    """check / to_bytes / header_fields on dataclasses holding other types:
    bits as bool, int, complex or out of range, latents of another dtype or
    shape, grids and callsigns the header refuses, fields out of range."""
    import warnings
    bf = ref.from_picture(sp_c, 2, _header(sp_c, 2), ook=True)
    rep = dataclasses.replace
    k, b, lat = bf.keying, bf.hdr_bits, bf.latents_i8
    cases = [rep(bf, keying=k.astype(bool)), rep(bf, keying=k.astype(np.int64) * 2),
             rep(bf, keying=k.astype(np.complex128)), rep(bf, keying=k + 1j),
             rep(bf, keying=k.astype(np.int64) + 256), rep(bf, hdr_bits=b.astype(bool)),
             rep(bf, hdr_bits=b.astype(np.float32)), rep(bf, latents_i8=lat.astype(np.uint8)),
             rep(bf, latents_i8=lat[::2]), rep(bf, latents_i8=np.int8(3)), rep(bf, grid="fn42"),
             rep(bf, grid="AR99"), rep(bf, grid="RR99"), rep(bf, callsign="K1ABC/P\x00"),
             rep(bf, callsign="K1ÄBC"), rep(bf, picture_id=-1), rep(bf, picture_id=1 << 32),
             rep(bf, codec_id=1 << 16), rep(bf, mode=-1), rep(bf, segment=-1),
             rep(bf, mode=3, segment=3), rep(bf, ook=0), rep(bf, ook=np.True_),
             rep(bf, latents_i8=np.full(50600, 127, np.int8)), rep(bf, latents_i8=np.zeros(50601, np.int8)),
             rep(bf, mode=np.int64(2), segment=np.uint8(2), picture_id=np.uint32(bf.picture_id))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")       # the complex casts
        for i, c in enumerate(cases):
            _same_outcome(ref.BeaconFile.check, ad["check"], c)
            _same_outcome(ref.BeaconFile.to_bytes, ad["to_bytes"], c)
            _same_outcome(ref.BeaconFile.header_fields, ad["header_fields"], c)


def test_from_picture_other_arguments(ad, sp_c):
    rep = dataclasses.replace
    h = _header(sp_c, 2)
    for args in ((sp_c, 2, h), (sp_c, 2, rep(h, grid=None)), (sp_c, 0, _header(sp_c, 0, call="A")),
                 (sp_c, 1, _header(sp_c, 1, call="ABCDEFGH")), (sp_c, 1, _header(sp_c, 1, call="00000000")),
                 (sp_c, 2, rep(h, version=1)), (sp_c, 2, rep(h, reserved=1)), (sp_c, 2, rep(h, version=3)),
                 (sp_c, 2, h, np.True_), (sp_c, 2, h, "x"), (sp_c, 2, h, 0), (sp_c, 3, h), (sp_c, 2.7, h),
                 (sp_c, 2, rep(h, callsign="K1ABC/P\x00")), (sp_c, 2, rep(h, grid="")),
                 (sp_c, 2, rep(h, picture_id=float(sp_c.picture_id))),
                 (sp_c, 2, rep(h, segment=np.int64(2)))):
        _same_outcome(ref.from_picture, ad["from_picture"], *args)


def test_file_size_any_int(nat):
    for n in (2 ** 63 - 1, 2 ** 63 - 371, 2 ** 63 - 370, 2 ** 63 - 369, 2 ** 70, -2 ** 63,
              -2 ** 63 + 1, -2 ** 70, True, 3.9, "12"):
        _same_outcome(ref.file_size, nat.file_size, n)


def test_beacon_files_at_random(ad, tmp_path):
    """Random pictures, segments, callsigns, grids and keying: the same bytes,
    and each implementation reads the other's file back to the same fields."""
    rng = np.random.default_rng(21)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/"
    for i in range(5):
        mode = int(rng.integers(0, 3))
        sp_r = picture.StoredPicture.from_latents(
            rng.standard_normal(3 * picture.GROUP_LATENTS) * rng.uniform(0.1, 10),
            int(rng.integers(0, 1 << 16)), mode)
        seg = int(rng.integers(0, mode + 1))
        call = "".join(rng.choice(list(alphabet), int(rng.integers(1, 9))))
        grid = None if i % 3 == 0 else (chr(65 + int(rng.integers(0, 18))) + chr(65 + int(rng.integers(0, 18)))
                                        + str(int(rng.integers(0, 100))).zfill(2))
        h = _header(sp_r, seg, call=call, grid=grid)
        ook = bool(i % 2)
        want = _outcome(ref.from_picture, sp_r, seg, h, ook)
        got = _outcome(ad["from_picture"], sp_r, seg, h, ook)
        assert want[0] == got[0] and (want[0] == "ok" or want[1:] == got[1:]), (call, want, got)
        if want[0] != "ok":
            continue
        _same_bf(want[1], got[1])
        data = ref.BeaconFile.to_bytes(want[1])
        assert ad["to_bytes"](got[1]) == data
        _same_bf(ref.BeaconFile.from_bytes(data), ad["from_bytes"].__func__(ref.BeaconFile, data))
