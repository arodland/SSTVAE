"""C++ stored pictures (native/core/qrss/picture) against sstvae.qrss.picture.

Bit-exact: the air <-> canonical maps are the frozen interleaver table,
prepare_segment's RMS is numpy's pairwise mean in air order and its
float16 is one rounding from float64 (so the picture ID, a SHA-256 of
those float16 bytes, agrees), to_int8 rounds half to even, and a `.qrsp`
is the same npz byte for byte. Refusal texts are compared character for
character. StoredPicture stays the reference's dataclass; the adapters in
tests/qrss_native/picture.py (what --native-qrss installs) are used here.
"""

import numpy as np
import pytest

from qrss_helpers import synthetic_full_latents
from qrss_native import picture as rows_mod
from qrss_native import require
from sstvae.qrss import picture as ref
from sstvae.qrss.picture import GROUP_LATENTS, N_GROUPS, SENT, StoredPicture


def _closed_form_full() -> np.ndarray:
    """The tensor of test_qrss_format's frozen picture ID."""
    g = np.arange(N_GROUPS * GROUP_LATENTS, dtype=np.float64)
    return np.sin(0.001 * g * g % 6.283) + 0.3 * np.cos(g)


@pytest.fixture(scope="module")
def nat(native):
    return require(native, "picture")


@pytest.fixture(scope="module")
def ad(native):
    """The --native-qrss adapters by attribute name."""
    require(native, "picture")
    return {attr: f for (_, attr), f in rows_mod.rows(native.qrss).items()}


def _outcome(f, *args, **kw):
    try:
        return ("ok", f(*args, **kw))
    except Exception as e:      # noqa: BLE001 -- any refusal is compared
        return ("raise", type(e), str(e))


def _same(want, got):
    if isinstance(want, np.ndarray):
        assert isinstance(got, np.ndarray)
        assert got.dtype == want.dtype and got.shape == want.shape, (want.dtype, got.dtype)
        assert got.tobytes() == want.tobytes()        # bitwise (NaN, -0.0 included)
    elif isinstance(want, StoredPicture):
        assert (got.mode, got.codec_id, got.picture_id) == (want.mode, want.codec_id,
                                                             want.picture_id)
        assert len(got.segs) == len(want.segs)
        for a, b in zip(want.segs, got.segs):
            _same(a, b)
    else:
        assert got == want and type(got) is type(want), (want, got)


def _same_outcome(f_ref, f_nat, *args, **kw):
    want, got = _outcome(f_ref, *args, **kw), _outcome(f_nat, *args, **kw)
    assert got[0] == want[0], (want, got)
    if want[0] == "raise":
        assert got[1:] == want[1:]
    else:
        _same(want[1], got[1])
    return want


def _from_latents(nat, full, codec_id, mode):
    return StoredPicture(**nat.from_latents(full, codec_id, mode))


def test_constants(nat):
    for name in ("GROUP_LATENTS", "SENT", "N_GROUPS", "MODE_NAMES", "QRSP_VERSION"):
        assert getattr(nat, name) == getattr(ref, name), name


# --- the frozen picture ID ----------------------------------------------------------

def test_closed_form_picture_id(nat):
    """test_qrss_format's frozen ID, 0xD349C671 (mode A, codec 0xD1D8), and the
    stored float16 tensor bit for bit in every mode."""
    full = _closed_form_full()
    got = _from_latents(nat, full, 0xD1D8, 0)
    assert got.picture_id == 0xD349C671 == ref.StoredPicture.from_latents(full, 0xD1D8, 0).picture_id
    for mode in range(N_GROUPS):
        for codec in (0, 0xD1D8, 0xFFFF):
            _same(ref.StoredPicture.from_latents(full, codec, mode),
                  _from_latents(nat, full, codec, mode))


def test_picture_id_on_other_tensors(nat):
    for seed in range(3):
        full = synthetic_full_latents(seed)
        mode = seed % 3
        sp = ref.StoredPicture.from_latents(full, 0x1234 + seed, mode)
        _same(sp, _from_latents(nat, full, 0x1234 + seed, mode))
        # picture_id itself, on the stored segments and on float32/float64 copies
        for segs in (sp.segs, [s.astype(np.float32) for s in sp.segs],
                     [s.astype(np.float64) for s in sp.segs], tuple(sp.segs)):
            assert nat.picture_id(sp.codec_id, mode, segs) == ref.picture_id(sp.codec_id, mode, segs)


def test_picture_id_refusals(nat):
    sp = ref.StoredPicture.from_latents(_closed_form_full(), 0xD1D8, 1)
    for args in ((0xD1D8, 3, sp.segs), (0xD1D8, -1, sp.segs), (1 << 16, 1, sp.segs),
                 (-1, 1, sp.segs), (0xD1D8, 1, sp.segs[:1]), (0xD1D8, 0, sp.segs),
                 (0xD1D8, 1, [sp.segs[0], sp.segs[1][:100]]),
                 (0xD1D8, 1, [sp.segs[0], sp.segs[1].reshape(2, -1)]),
                 (0xD1D8, np.int64(1), sp.segs), (np.uint16(0xD1D8), 1, sp.segs),
                 (0xD1D8, 7.0, sp.segs), (1 << 80, 1, sp.segs)):
        _same_outcome(ref.picture_id, nat.picture_id, *args)


# --- maps -----------------------------------------------------------------------------

def test_maps(nat):
    for g in range(N_GROUPS):
        a2c = nat.air_to_canonical(g)
        want = ref.air_to_canonical(g)
        _same(want, a2c)
        assert not a2c.flags.writeable
        ns = nat.never_sent(g)
        _same(ref.never_sent(g), ns)
        assert ns.sum() == 2200 == GROUP_LATENTS - SENT
    for g in (3, -1, np.int64(2), 2.0, True):
        _same_outcome(ref.air_to_canonical, nat.air_to_canonical, g)
        _same_outcome(ref.never_sent, nat.never_sent, g)
        _same_outcome(ref._check_group, nat._check_group, g)
        _same_outcome(ref._check_mode, nat._check_mode, g)


# --- prepare_segment ----------------------------------------------------------------

def test_prepare_segment(nat):
    rng = np.random.default_rng(5)
    inputs = [synthetic_full_latents(7), _closed_form_full(),
              rng.standard_normal(N_GROUPS * GROUP_LATENTS).astype(np.float32),
              (rng.standard_normal(N_GROUPS * GROUP_LATENTS) * 1e-3).reshape(3, 528, 100),
              (rng.standard_normal(N_GROUPS * GROUP_LATENTS) * 7e4),     # float16 overflow -> inf
              rng.integers(-3, 4, N_GROUPS * GROUP_LATENTS)]
    for full in inputs:
        for g in range(N_GROUPS):
            _same_outcome(ref.prepare_segment, nat.prepare_segment, full, g)


def test_prepare_segment_refusals(nat):
    full = synthetic_full_latents(1)
    zeros_g1 = full.copy()
    zeros_g1[GROUP_LATENTS:2 * GROUP_LATENTS] = 0.0
    nan_g2 = full.copy()
    nan_g2[2 * GROUP_LATENTS + int(ref.air_to_canonical(2)[0])] = np.nan
    inf_g0 = full.copy()
    inf_g0[int(ref.air_to_canonical(0)[5])] = np.inf
    never = full.copy()            # energy only where nothing is sent
    never[:GROUP_LATENTS] = 0.0
    never[:GROUP_LATENTS][ref.never_sent(0)] = 1.0
    for args in ((full[:-1], 0), (full, 3), (full, -1), (zeros_g1, 1), (nan_g2, 2),
                 (inf_g0, 0), (never, 0), (full[:-1], 7)):
        _same_outcome(ref.prepare_segment, nat.prepare_segment, *args)


# --- air_values / canonical_from_air -------------------------------------------------

def test_air_values_and_canonical_from_air(nat):
    sp = ref.StoredPicture.from_latents(synthetic_full_latents(2), 0xD1D8, 2)
    rng = np.random.default_rng(9)
    for g in range(N_GROUPS):
        seg = sp.segs[g]
        for s in (seg, seg.astype(np.float32), seg.astype(np.float64), list(seg[:0]) or seg,
                  np.stack([seg, seg[::-1]]), rng.integers(-5, 5, GROUP_LATENTS)):
            _same_outcome(ref.air_values, nat.air_values, s, g)
        air = ref.air_values(seg, g)
        for a in (air, air.astype(np.float16), air.astype(np.float32),
                  np.stack([air, -air]).reshape(1, 2, SENT), rng.integers(-5, 5, SENT),
                  air.astype(np.longdouble), air.astype(">f8")):
            _same_outcome(ref.canonical_from_air, nat.canonical_from_air, a, g)
    for args in ((sp.segs[0][:-1], 0), (sp.segs[0], 3), (sp.segs[0][:5], 9),
                 (np.zeros((2, 3)), 0), (np.float64(1.0), 0)):
        _same_outcome(ref.air_values, nat.air_values, *args)
    for args in ((np.zeros(SENT + 1), 0), (np.zeros(SENT), 3), (np.zeros(SENT), -1),
                 (np.zeros(GROUP_LATENTS), 0)):
        _same_outcome(ref.canonical_from_air, nat.canonical_from_air, *args)


# --- int8 ---------------------------------------------------------------------------------

def test_int8(nat):
    edges = np.array([0.0, -0.0, 0.025, -0.025, 0.075, -0.075, 0.125, 6.35, -6.35, 6.375,
                      -6.375, 6.4, -6.4, 1e9, -1e9, np.inf, -np.inf, 0.0249999999, 0.0250000001,
                      1.0, -1.0, 3.14159])
    rng = np.random.default_rng(3)
    for a in (edges, rng.standard_normal(50600) * 2.0, rng.standard_normal((4, 7)),
              edges.astype(np.float32), list(edges[:5]), 0.075, np.arange(-200, 200)):
        _same_outcome(ref.to_int8, nat.to_int8, a)
    q = np.arange(-128, 128, dtype=np.int8)
    for a in (q, q.astype(np.int64), q.reshape(16, 16), list(range(-3, 4)), np.int8(7),
              q.astype(np.float32) / 3):
        _same_outcome(ref.from_int8, nat.from_int8, a)


# --- StoredPicture -----------------------------------------------------------------------

def test_check_and_id_hex(nat, ad):
    sp = ref.StoredPicture.from_latents(_closed_form_full(), 0xD1D8, 0)
    _same_outcome(ref.StoredPicture.check, ad["check"], sp)
    assert ad["id_hex"].fget(sp) == sp.id_hex == "d349c671"
    for pid in (sp.picture_id ^ 1, 0, 0xFFFFFFFF, -5, 1 << 40):
        bad = ref.StoredPicture(sp.mode, sp.codec_id, sp.segs, pid)
        _same_outcome(ref.StoredPicture.check, ad["check"], bad)
        assert ad["id_hex"].fget(bad) == bad.id_hex
    for bad in (ref.StoredPicture(1, sp.codec_id, sp.segs, sp.picture_id),
                ref.StoredPicture(0, 1 << 16, sp.segs, sp.picture_id),
                ref.StoredPicture(5, sp.codec_id, sp.segs, sp.picture_id)):
        _same_outcome(ref.StoredPicture.check, ad["check"], bad)
    for args in ((_closed_form_full(), 0xD1D8, 3), (_closed_form_full(), 1 << 16, 1),
                 (_closed_form_full()[:10], 0xD1D8, 0), (_closed_form_full(), -1, 0)):
        _same_outcome(ref.StoredPicture.from_latents, lambda *a: _from_latents(nat, *a), *args)


# --- .qrsp ----------------------------------------------------------------------------------

def test_qrsp_both_ways(nat, ad, tmp_path):
    """A .qrsp written by either reads back in the other, byte-identical."""
    for mode in range(N_GROUPS):
        sp = ref.StoredPicture.from_latents(synthetic_full_latents(mode), 0xD1D8, mode)
        p_ref, p_nat = tmp_path / f"r{mode}.qrsp", tmp_path / f"n{mode}.qrsp"
        ref.save_qrsp(p_ref, sp)
        nat.save_qrsp(p_nat, sp)
        assert p_ref.read_bytes() == p_nat.read_bytes() == nat._qrsp_bytes(sp)
        _same(ref.load_qrsp(p_nat), ad["load_qrsp"](p_ref))
        _same(sp, ad["load_qrsp"](str(p_ref)))


def _npz(path, **members):
    with open(path, "wb") as f:
        np.savez(f, **members)


def test_qrsp_refusals(nat, ad, tmp_path):
    sp = ref.StoredPicture.from_latents(_closed_form_full(), 0xD1D8, 1)
    segs = np.stack(sp.segs)
    good = dict(version=np.int64(1), mode=np.int64(1), codec_id=np.int64(0xD1D8), segs=segs,
                picture_id=np.int64(sp.picture_id))
    cases = {
        "wrong_id": dict(good, picture_id=np.int64(sp.picture_id ^ 0x100)),
        "version": dict(good, version=np.int64(2)),
        "no_segs": {k: v for k, v in good.items() if k != "segs"},
        "no_version": {k: v for k, v in good.items() if k != "version"},
        "no_pid": {k: v for k, v in good.items() if k != "picture_id"},
        "f4": dict(good, segs=segs.astype(np.float32)),
        "short": dict(good, segs=segs[:1]),
        "mode": dict(good, mode=np.int64(2)),
        "mode_bad": dict(good, mode=np.int64(5), segs=np.zeros((6, GROUP_LATENTS), np.float16)),
        "codec": dict(good, codec_id=np.int64(1 << 16)),
        "flipped": dict(good, segs=np.where(np.arange(GROUP_LATENTS) == 7, np.float16(0.5),
                                            segs)),
        "int32": dict(good, version=np.int32(1), mode=np.int32(1)),
    }
    for name, members in cases.items():
        p = tmp_path / f"{name}.qrsp"
        _npz(p, **members)
        _same_outcome(ref.load_qrsp, ad["load_qrsp"], p)
    # A wrong ID is refused, with its text.
    p = tmp_path / "wrong_id.qrsp"
    with pytest.raises(ValueError, match="does not match the stored latents"):
        ad["load_qrsp"](p)
    # A missing file fails as Python's open() does; save refuses a bad picture first.
    _same_outcome(ref.load_qrsp, ad["load_qrsp"], tmp_path / "absent.qrsp")
    bad = ref.StoredPicture(sp.mode, sp.codec_id, sp.segs, sp.picture_id ^ 1)
    _same_outcome(ref.save_qrsp, nat.save_qrsp, tmp_path / "bad.qrsp", bad)
    # Not an npz at all: both refuse (the texts are numpy's, not compared).
    p = tmp_path / "junk.qrsp"
    p.write_bytes(b"not a zip file at all")
    with pytest.raises(Exception):
        ref.load_qrsp(p)
    with pytest.raises(Exception):
        ad["load_qrsp"](p)


# --- review probes: inputs past the ones the callers use -------------------------------------

def _quiet_outcome(f, *args):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # numpy's cast warnings (complex, overflow)
        return _outcome(f, *args)


def _same_quiet(f_ref, f_nat, *args):
    want, got = _quiet_outcome(f_ref, *args), _quiet_outcome(f_nat, *args)
    assert got[0] == want[0], (want, got)
    if want[0] == "raise":
        assert got[1:] == want[1:]
    else:
        _same(want[1], got[1])


def test_picture_id_rounds_like_astype_f2(nat):
    """picture_id's s.astype("<f2") of float64 / float32 segments that are not
    float16 values: the rounding edges (overflow to inf at 65520, the
    subnormal ties, mantissa ties), NaN payloads, other byte orders,
    strides, ints and lists, and the ID's own argument forms."""
    rng = np.random.default_rng(11)
    seg = rng.standard_normal(GROUP_LATENTS)
    edge = [65519.0, 65520.0, 65504.0, 65505.0, 2.0 ** -25, 3 * 2.0 ** -26, 2.0 ** -24, -0.0,
            np.nan, np.inf, -np.inf, 1e-30, 2.0 ** -14 - 2.0 ** -25, 1.0 + 2.0 ** -11,
            1.0 + 3 * 2.0 ** -11, -(1.0 + 2.0 ** -11), 2.0 ** -15 + 2.0 ** -25]
    seg[:len(edge)] = edge
    nan64 = np.zeros(GROUP_LATENTS)
    nan64.view(np.uint64)[:4] = [0x7FF0000000000001, 0x7FF8000000000001, 0xFFF4000000000000,
                                 0x7FF0040000000000]
    nan32 = np.zeros(GROUP_LATENTS, np.float32)
    nan32.view(np.uint32)[:4] = [0x7F800001, 0x7FC00001, 0xFFA00000, 0x7F802000]
    for s in (seg, seg.astype(np.float32), rng.standard_normal(GROUP_LATENTS).astype(np.float32) * 1e-6,
              nan64, nan32, rng.standard_normal(GROUP_LATENTS).astype(">f2"),
              rng.standard_normal(GROUP_LATENTS).astype(">f8"),
              rng.integers(-70000, 70000, GROUP_LATENTS),
              rng.standard_normal(2 * GROUP_LATENTS).astype(np.float16)[::2],
              rng.standard_normal(2 * GROUP_LATENTS)[::2], list(rng.standard_normal(GROUP_LATENTS))):
        _same_quiet(ref.picture_id, nat.picture_id, 7, 0, [s])
    two = rng.standard_normal((2, GROUP_LATENTS)).astype(np.float16)
    for args in ((7, 1, two), (7, 0, [np.float16(1)]), (7, 0, []), (True, 0, [seg]), (1.5, 0, [seg]),
                 (65535.9, 0, [seg]), (-0.5, 0, [seg]), (7, -0.5, [seg]), (7, 2.9, [seg] * 3),
                 (7, "1", [seg] * 2), ("7", 0, [seg])):
        _same_quiet(ref.picture_id, nat.picture_id, *args)


def test_maps_on_other_arrays(nat):
    """air_values / canonical_from_air on NaN payloads, other layouts and
    byte orders, empty batches, and every dtype kind."""
    rng = np.random.default_rng(12)
    h = rng.standard_normal(GROUP_LATENTS).astype(np.float16)
    h.view(np.uint16)[:6] = [0x7C01, 0x7E00, 0xFE01, 0x7DFF, 0x8000, 0x0001]
    with np.errstate(invalid="ignore"):         # the signalling NaN cast to complex / long double
        segs = (h, h.astype(">f2"), np.asfortranarray(np.stack([h, h, h])), h[None, None, :],
                np.zeros((0, GROUP_LATENTS), np.float16), np.zeros((2, 0, GROUP_LATENTS)),
                h.astype(bool), h.astype(np.complex64), h.astype(np.longdouble),
                rng.integers(0, 2 ** 63, GROUP_LATENTS, dtype=np.uint64))
    for s in segs:
        for g in (0, 2):
            _same_quiet(ref.air_values, nat.air_values, s, g)
    a = rng.standard_normal(SENT)
    airs = (a.astype(np.float16), np.zeros((0, SENT)), np.zeros((3, 0, SENT), np.float32),
            a.astype(bool), a.astype(np.complex128), a.astype(np.int8), a.astype(">f2"),
            a.astype(">f4"), np.asfortranarray(np.stack([a, a]).astype(np.float32)), a[::-1])
    for s in airs:
        for g in (1, 2):
            _same_quiet(ref.canonical_from_air, nat.canonical_from_air, s, g)
    for g in (np.int64(1), 1.9, "2", True, "x", None, np.uint8(0)):
        _same_quiet(ref.air_values, nat.air_values, h, g)
        _same_quiet(ref.canonical_from_air, nat.canonical_from_air, a, g)
        for f in ("_check_group", "_check_mode", "never_sent", "air_to_canonical"):
            _same_quiet(getattr(ref, f), getattr(nat, f), g)


def test_int8_on_other_inputs(nat):
    """Every boundary k/20 +- 1/40 in float64, other dtypes, 0-d and empty."""
    ties = (np.arange(-2600, 2600) + 0.5) / 20.0
    for x in (ties, np.nextafter(ties, np.inf), np.nextafter(ties, -np.inf), np.arange(-130, 130) / 20.0,
              np.float16([0.025, 0.075, 6.35]), np.array(0.125), np.float64(0.025), [[0.1, 0.2]],
              np.zeros((0, 3)), np.array([2.0 ** 63, -2.0 ** 63]), np.array([True, False]),
              np.array([1 + 2j])):
        _same_quiet(ref.to_int8, nat.to_int8, x)
    for x in (np.array([True, False]), np.arange(256, dtype=np.uint8), np.int8(-128),
              np.arange(-128, 128, dtype=np.int8)[::3], np.float16([1.5]), np.zeros((0,), np.int8), 7,
              [1, 2], np.array([1 + 1j])):
        _same_quiet(ref.from_int8, nat.from_int8, x)


def test_prepare_segment_other_inputs(nat):
    """Other dtypes and layouts of `full`, and scales where the squares
    underflow, overflow or go subnormal."""
    full = synthetic_full_latents(4)
    one = np.zeros_like(full)
    one[int(ref.air_to_canonical(0)[3])] = 1e-200
    lone = np.zeros_like(full)
    lone[int(ref.air_to_canonical(0)[3])] = -3.0
    nan_never = full.copy()
    nan_never[:GROUP_LATENTS][ref.never_sent(0)] = np.nan
    cases = [full.astype(np.float16), full.astype(np.float32).reshape(-1, 2), full.astype(np.complex128),
             full > 0, np.asfortranarray(full.reshape(3, -1)), full[::-1], full * 1e-160, full * 1e160,
             full * 1e-310, one, lone, nan_never]
    for f in cases:
        for g in (0, 2):
            _same_quiet(ref.prepare_segment, nat.prepare_segment, f, g)
    for g in ("1", 1.5, 3):
        _same_quiet(ref.prepare_segment, nat.prepare_segment, full, g)


def test_qrsp_crafted_members(nat, ad, tmp_path):
    """.qrsp files whose members np.savez wrote with other dtypes and shapes."""
    sp = ref.StoredPicture.from_latents(synthetic_full_latents(1), 0xD1D8, 1)
    segs = np.stack(sp.segs)
    good = dict(version=np.int64(1), mode=np.int64(1), codec_id=np.int64(0xD1D8), segs=segs,
                picture_id=np.int64(sp.picture_id))
    cases = {
        "mode_f": dict(good, mode=np.float64(1.7)),
        "mode_b": dict(good, mode=np.bool_(True)),
        "pid_u4": dict(good, picture_id=np.uint32(sp.picture_id)),
        "pid_f8": dict(good, picture_id=np.float64(sp.picture_id)),
        "codec_u2": dict(good, codec_id=np.uint16(0xD1D8)),
        "segs_fortran": dict(good, segs=np.asfortranarray(segs)),
        "extra": dict(good, extra=np.zeros(3)),
        "segs_3d": dict(good, segs=segs[None]),
        "segs_1d": dict(good, segs=segs[0]),
        "segs_0d": dict(good, segs=np.float16(1)),
        "segs_bool": dict(good, segs=segs != 0),
        "mode_neg": dict(good, mode=np.int64(-1), segs=segs[:0]),
        "mode_max": dict(good, mode=np.int64(2 ** 63 - 1)),
        "mode_min": dict(good, mode=np.int64(-2 ** 63)),
        "mode_3": dict(good, mode=np.int64(3), segs=np.zeros((4, GROUP_LATENTS), np.float16)),
        "pid_neg": dict(good, picture_id=np.int64(-5)),
        "pid_big": dict(good, picture_id=np.int64(1 << 40)),
        "codec_neg": dict(good, codec_id=np.int64(-1)),
    }
    for name, members in cases.items():
        p = tmp_path / f"{name}.qrsp"
        _npz(p, **members)
        _same_outcome(ref.load_qrsp, ad["load_qrsp"], p)
    _same_outcome(ref.load_qrsp, ad["load_qrsp"], str(tmp_path / "extra.qrsp"))
    _same_outcome(ref.load_qrsp, ad["load_qrsp"], tmp_path)          # a directory


def test_stored_pictures_at_random(nat, ad, tmp_path):
    """Random pictures, modes and codec IDs: the stored tensor, the ID and the
    .qrsp bytes agree, and each loads the other's file."""
    rng = np.random.default_rng(13)
    for i in range(4):
        full = rng.standard_normal(N_GROUPS * GROUP_LATENTS) * rng.uniform(1e-3, 1e3)
        full = np.where(rng.random(full.size) < 0.01, 0.0, full)
        mode, codec = int(rng.integers(0, 3)), int(rng.integers(0, 1 << 16))
        want = ref.StoredPicture.from_latents(full, codec, mode)
        got = _from_latents(nat, full, codec, mode)
        _same(want, got)
        assert nat._qrsp_bytes(want) == (ref.save_qrsp(tmp_path / f"{i}.qrsp", want)
                                         or (tmp_path / f"{i}.qrsp").read_bytes())
        _same(want, ad["load_qrsp"](tmp_path / f"{i}.qrsp"))


def test_prepare_segment_rms_to_the_last_bit(nat):
    """The RMS is numpy's pairwise mean of the squares in air order, to the
    last bit. Random inputs pass even with a sum in another order, since the
    float16 rounding hides an ulp of RMS; so these are inputs where both a
    naive and an exact sum give another RMS than numpy's, with values
    placed on float16 rounding ties (t = rms * 2^-25 is exactly half the
    smallest subnormal) that an RMS one ulp either way moves."""
    import math
    found = 0
    for seed in range(400):
        g = seed % 3
        r = np.random.default_rng(1000 + seed)
        x = r.standard_normal(SENT) * np.exp(r.uniform(-1, 1, SENT))
        ties = r.choice(SENT, 8, replace=False)
        x[ties] = 0.0
        rms = np.sqrt(np.mean(x ** 2))
        sq = (x ** 2).tolist()
        if math.sqrt(sum(sq) / SENT) == rms or math.sqrt(math.fsum(sq) / SENT) == rms:
            continue
        t = rms * 2.0 ** -25
        x[ties] = [t, -t, np.nextafter(t, np.inf), -np.nextafter(t, np.inf), np.nextafter(t, 0),
                   3 * t, 5 * t, np.nextafter(3 * t, np.inf)]
        rms = np.sqrt(np.mean(x ** 2))
        full = r.standard_normal(N_GROUPS * GROUP_LATENTS)
        full[g * GROUP_LATENTS + ref.air_to_canonical(g)] = x
        want = ref.prepare_segment(full, g)
        # The test has teeth: an RMS one ulp off either way stores other values.
        for off in (np.nextafter(rms, np.inf), np.nextafter(rms, 0)):
            assert (x / off).astype(np.float16).tobytes() != ref.air_values(want, g).astype(
                np.float16).tobytes()
        _same(want, nat.prepare_segment(full, g))
        found += 1
        if found == 4:
            break
    assert found == 4
