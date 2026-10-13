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
