"""C++ frames (native/core/qrss/frame) against sstvae.qrss.frame and .lframe.

Bit-exact throughout: the layouts and lengths are integer arithmetic, and
`assemble`, `data_latents`, `pair` and `unpair` reproduce numpy's operation
order (including `pair`'s complex division, which is a multiplication by
1/sqrt(2)), so every comparison here is of bytes, signed zeros and NaNs
included. Skips when the extension is not built; `pytest
--native-qrss=tx` runs the rest of the QRSS suites against the same code.
"""

import dataclasses

import numpy as np
import pytest

from qrss_native import require
from sstvae.qrss import frame as F
from sstvae.qrss import lframe as LF
from sstvae.qrss.constants import N_HDR_BITS, SPREAD_RHO, WAVEFORM_CE, WAVEFORM_L

ALL = [*F.PRESETS.values(), *F.LEGACY.values(), *F.PROTOTYPES.values(), *LF.PRESETS.values()]


def _custom():
    """Frames the suites build that are not presets, both waveforms."""
    end = F.FrameSpec("end", n_data=2000, cw_after=()).n_sym
    lend = LF.LFrameSpec("l-end", n_data=2002, cw_after=()).n_sym
    return [
        F.FrameSpec("edge", n_data=4096, cw_after=(3301, 7670)),
        F.FrameSpec("tail", n_data=4096 + 47, cw_after=()),
        F.FrameSpec("end", n_data=2000, cw_after=(4000, end)),
        F.FrameSpec("custom", n_data=4096, cw_after=(5000,)),
        F.FrameSpec("test-spreadonly", n_hdr=0, n_data=8192, cw_after=(), hdr_rho=0.3),
        F.FrameSpec("one", n_hdr=0, n_data=1, cw_after=(661,)),
        F.FrameSpec("rho", n_data=3000, cw_after=(), hdr_rho=0.5),
        LF.LFrameSpec("l-edge", n_data=4096, cw_after=(1651, 3834)),
        LF.LFrameSpec("l-end", n_data=2002, cw_after=(2000, lend)),
        LF.LFrameSpec("l-odd", n_data=2, cw_after=()),
        LF.LFrameSpec("l-spread", n_hdr=0, n_data=6000, cw_after=(), hdr_rho=0.2),
        LF.LFrameSpec("l-mid", n_data=1000, cw_after=(1700,)),
    ]


SPECS = ALL + _custom()
IDS = [s.name for s in SPECS]


@pytest.fixture(scope="module")
def nat(native):
    return require(native, "frame")


@pytest.fixture(scope="module")
def lnat(native):
    return require(native, "lframe")


def _same(got, want, what=""):
    """Equal bytes, dtype and shape (so -0.0 != 0.0 and NaNs compare)."""
    got, want = np.asarray(got), np.asarray(want)
    assert got.dtype == want.dtype, (what, got.dtype, want.dtype)
    assert got.shape == want.shape, (what, got.shape, want.shape)
    assert np.ascontiguousarray(got).tobytes() == np.ascontiguousarray(want).tobytes(), what


def _ref_layout(spec):
    return (LF.layout if spec.waveform == WAVEFORM_L else F.layout)(spec)


def _nat_layout(nat, lnat, spec):
    return (lnat.layout if spec.waveform == WAVEFORM_L else nat.layout)(spec)


def _bits(rng):
    return rng.integers(0, 2, N_HDR_BITS).astype(np.uint8)


# --- specs: fields, derived lengths, positions --------------------------------------------

DERIVED = ("waveform", "name", "n_pre", "n_hdr", "n_data", "cw_after", "hdr_rho", "t_sym",
           "cw_window", "has_header", "n_data_sym", "n_sym", "n_win", "n_pos", "n_ref",
           "duration_s", "win_start_pos", "keyed_end_pos", "keyed_start_pos")


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_derived_properties(nat, spec):
    n = nat.of(spec)
    for name in DERIVED + (("n_data_payload",) if spec.waveform == WAVEFORM_L else ()):
        got, want = getattr(n, name), getattr(spec, name)
        assert got == want and type(got) is type(want), (name, got, want)
    assert n.window_spans_s() == spec.window_spans_s()
    assert n.fields() == (spec.waveform, *dataclasses.astuple(spec))
    assert repr(n) == repr(spec)
    # rebuilt from its fields, it is the same frame
    assert nat.make(*n.fields()) == n and nat.of(n) == n


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_positions(nat, spec):
    n = nat.of(spec)
    s = np.arange(-5, spec.n_sym + 5)
    _same(n.pos_of(s), spec.pos_of(s), "pos_of array")
    _same(n.pos_of(s.reshape(-1, 1)[: (len(s) // 2) * 2].reshape(2, -1)),
          spec.pos_of(s.reshape(-1, 1)[: (len(s) // 2) * 2].reshape(2, -1)), "pos_of 2-D")
    for v in (0, spec.n_sym - 1, *spec.cw_after, np.int64(17), 3.7):
        got, want = n.pos_of(v), spec.pos_of(v)
        assert got == want and type(got) is type(want) is int, v
    p = np.arange(spec.n_pos, dtype=np.int64)
    _same(n.time_s(p), spec.time_s(p), "time_s")
    got, want = n.time_s(1234), spec.time_s(1234)
    assert got == want and type(got) is type(want)


def test_preset_tables(nat, lnat):
    for table, ref in ((nat.PRESETS, F.PRESETS), (nat.LEGACY, F.LEGACY),
                       (nat.PROTOTYPES, F.PROTOTYPES), (lnat.PRESETS, LF.PRESETS)):
        assert list(table) == list(ref)
        for name, n in table.items():
            assert n.fields() == (ref[name].waveform, *dataclasses.astuple(ref[name])), name
    assert lnat.N_HDR_SLOTS == LF.N_HDR_SLOTS


def test_counting_helpers(nat, lnat):
    for n in range(-300, 3000):
        assert nat._data_symbols(n) == F._data_symbols(n), n
    for n in (4095, 4096, 16384, 50600, np.int64(1024)):
        assert nat._data_symbols(n) == F._data_symbols(n), n
    for j in range(-100, 100):
        assert lnat._refs_below(j) == LF._refs_below(j), j
    for n in range(-40, 200, 3):
        for j0 in range(-40, 1400, 37):
            assert lnat._data_symbols(n, j0) == LF._data_symbols(n, j0), (n, j0)
    for n in (2048, 8192, 25300):
        for j0 in (0, 1320, 1321, 1327, 1328):
            assert lnat._data_symbols(n, j0) == LF._data_symbols(n, j0), (n, j0)


# --- layouts ------------------------------------------------------------------------------

@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_layout(nat, lnat, spec):
    want = _ref_layout(spec)
    got = _nat_layout(nat, lnat, spec)
    names = [f.name for f in dataclasses.fields(want)]
    assert sorted(got) == sorted(names)
    for name in names:
        _same(got[name], getattr(want, name), name)
        assert not got[name].flags.writeable, name


def test_layout_refuses_the_other_waveform(nat, lnat):
    with pytest.raises(AssertionError):
        F.layout(LF.L_SHORT)
    with pytest.raises(AssertionError, match="frame.layout is CE's"):
        nat.layout(LF.L_SHORT)
    with pytest.raises(AssertionError):
        LF.layout(F.SHORT)
    with pytest.raises(AssertionError, match="lframe.layout is L's"):
        lnat.layout(F.SHORT)


# --- assembling --------------------------------------------------------------------------

@pytest.mark.parametrize("spec", SPECS, ids=IDS)
@pytest.mark.parametrize("seed", [0, 1])
def test_assemble(nat, lnat, spec, seed):
    rng = np.random.default_rng(seed)
    b = _bits(rng) if spec.has_header else None
    x = rng.standard_normal(spec.n_data)
    if spec.waveform == WAVEFORM_L:
        _same(lnat.assemble(spec, b, x), LF.assemble(spec, b, x))
    else:
        _same(nat.assemble(spec, b, x), F.assemble(spec, b, x))


@pytest.mark.parametrize("rho", [0.0, SPREAD_RHO, 0.3, 0.999])
@pytest.mark.parametrize("wave", ["ce", "l"])
def test_assemble_at_every_rho(nat, lnat, rho, wave):
    """hdr_rho 0 and 0.054 as the plan names them, and two more."""
    rng = np.random.default_rng(int(rho * 1000) + 7)
    b = _bits(rng)
    if wave == "ce":
        spec = dataclasses.replace(F.SHORT, name=f"short-{rho}", hdr_rho=rho)
        x = rng.standard_normal(spec.n_data) * 3
        _same(nat.assemble(spec, b, x), F.assemble(spec, b, x))
        _same(nat.spread_symbols(spec, b), F.spread_symbols(spec, b))
    else:
        spec = dataclasses.replace(LF.L_SHORT, name=f"l-short-{rho}", hdr_rho=rho)
        x = rng.standard_normal(spec.n_data) * 3
        _same(lnat.assemble(spec, b, x), LF.assemble(spec, b, x))
        _same(lnat.data_latents(spec, x, b), LF.data_latents(spec, x, b))
    _same(nat.spread_signs(spec, b), F.spread_signs(spec, b))


def test_assemble_input_dtypes(nat, lnat):
    """Bits and latents as the suites hand them over: lists, bools, int64, float32."""
    rng = np.random.default_rng(5)
    b = _bits(rng)
    x = rng.standard_normal(F.SHORT.n_data).astype(np.float32)
    for bb in (b, b.astype(bool), b.astype(np.int64), b.astype(np.float32), b.tolist()):
        _same(nat.assemble(F.SHORT, bb, x), F.assemble(F.SHORT, bb, x))
        _same(lnat.assemble(LF.L_SHORT, bb, x), LF.assemble(LF.L_SHORT, bb, x))
    _same(nat.assemble(F.TINY, None, list(range(1024))), F.assemble(F.TINY, None, list(range(1024))))
    _same(nat.assemble(F.TINY, [], np.zeros(1024)), F.assemble(F.TINY, [], np.zeros(1024)))
    # spread_signs does not check the values, only the count
    odd = rng.random(N_HDR_BITS)
    _same(nat.spread_signs(F.SHORT, odd), F.spread_signs(F.SHORT, odd))
    _same(nat.spread_signs(LF.L_FULL, odd), F.spread_signs(LF.L_FULL, odd))


def test_data_latents(lnat):
    rng = np.random.default_rng(9)
    b = _bits(rng)
    x = rng.standard_normal(LF.L_TINY.n_data)
    out = lnat.data_latents(LF.L_TINY, x)                 # rho 0: x itself
    assert out is x and LF.data_latents(LF.L_TINY, x) is x
    x = rng.standard_normal(4096)
    _same(lnat.data_latents(LF.L_SHORT, x, b), LF.data_latents(LF.L_SHORT, x, b))
    _same(lnat.data_latents(F.SHORT, x, b), LF.data_latents(F.SHORT, x, b))   # any spec, as Python
    _same(lnat.data_latents(LF.L_SHORT, x.tolist(), b.tolist()), LF.data_latents(LF.L_SHORT, x.tolist(), b.tolist()))


SPECIALS = np.array([0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, np.inf, 1.0, -np.inf, 2.0,
                     np.nan, 3.0, 1.0, np.inf, 1.0, -np.inf, 1.0, np.nan, 5e-324, -5e-324,
                     1e308, 1e308, -1e308, 2.5e-308, 1.0, -1.0])


def test_pair_and_unpair(lnat):
    rng = np.random.default_rng(3)
    for v in (rng.standard_normal(20000), rng.standard_normal(20000) * 1e-310, SPECIALS,
              np.arange(-50, 50, dtype=np.int64), [1, 2, 3, 4], np.zeros(0)):
        _same(lnat.pair(v), LF.pair(v), "pair")
        p = LF.pair(v)
        _same(lnat.unpair(p), LF.unpair(p), "unpair")
    z = (rng.standard_normal((3, 5, 40)) + 1j * rng.standard_normal((3, 5, 40)))
    for zz in (z, z.astype(np.complex64), z.real, z.real.astype(np.float32),
               np.arange(12, dtype=np.int16).reshape(3, 4), np.array([True, False]),
               np.zeros((2, 0), np.complex128), [1 + 2j, -0.0 - 0.0j]):
        _same(lnat.unpair(zz), LF.unpair(zz), "unpair dtype")
    # unpair(pair(v)) is v within one rounding each way
    v = rng.standard_normal(1000)
    assert np.max(np.abs(lnat.unpair(lnat.pair(v)) - v)) <= 4 * np.spacing(np.max(np.abs(v)))


@pytest.mark.parametrize("spec", [F.SHORT, F.TINY, F.FULL_SPREADONLY, LF.L_SHORT, LF.L_TINY],
                         ids=lambda s: s.name)
def test_disassemble(nat, lnat, spec):
    rng = np.random.default_rng(11)
    ref, mine = (LF.disassemble, lnat.disassemble) if spec.waveform else (F.disassemble, nat.disassemble)
    y = rng.standard_normal((2, 3, spec.n_sym))
    batches = [y, y[0, 0], y.astype(np.float32), (y + 1j * y[::-1]).astype(np.complex64),
               y + 1j * rng.standard_normal(y.shape), (y * 10).astype(np.int8), y[..., ::-1],
               y[0, 0].tolist()]
    for yy in batches:
        got, want = mine(yy, spec), ref(yy, spec)
        _same(got[0], want[0], "header")
        _same(got[1], want[1], "data")


# --- names ---------------------------------------------------------------------------------

def test_get_and_friends(nat, lnat):
    for name in [*F.PRESETS, *F.LEGACY, *F.PROTOTYPES]:
        assert nat.get(name).fields() == nat.of(F.get(name)).fields()
        assert lnat.get_any(name).fields() == nat.of(LF.get_any(name)).fields()
    for name in LF.PRESETS:
        assert lnat.get(name).fields() == nat.of(LF.get(name)).fields()
        assert lnat.get_any(name).fields() == nat.of(LF.get_any(name)).fields()
    for spec in ALL:
        for w in (WAVEFORM_CE, WAVEFORM_L, 2, -1):
            try:
                want = nat.of(LF.twin(spec, w)).fields()
            except ValueError as e:
                with pytest.raises(ValueError) as got:
                    lnat.twin(spec, w)
                assert str(got.value) == str(e), (spec.name, w)
                continue
            assert lnat.twin(spec, w).fields() == want, (spec.name, w)
    for spec in SPECS:
        if spec.waveform == WAVEFORM_CE:
            assert nat.plain(spec).fields() == nat.of(F.plain(spec)).fields(), spec.name
    assert nat.plain(LF.L_SHORT).fields() == nat.of(F.plain(LF.L_SHORT)).fields()
    # a native spec passes through; one of the other waveform is an unknown name
    assert nat.get(nat.get("short")) == nat.get("short")


@pytest.mark.parametrize("call", [
    lambda m, lm: m.get("huge"),
    lambda m, lm: m.get(""),
    lambda m, lm: m.get("it's"),
    lambda m, lm: m.get("l-short"),
    lambda m, lm: m.get(5),
    lambda m, lm: m.get(None),
    lambda m, lm: m.get(LF.L_TINY),
    lambda m, lm: lm.get("short"),
    lambda m, lm: lm.get("l-huge"),
    lambda m, lm: lm.get(F.TINY),
    lambda m, lm: lm.get_any("l-x"),
    lambda m, lm: lm.get_any("x\n"),
    lambda m, lm: lm.get_any(3.5),
    lambda m, lm: lm.twin(F.FULL_V1, WAVEFORM_L),
    lambda m, lm: lm.twin(F.FULL_SPREADONLY, WAVEFORM_L),
], ids=lambda c: "")
def test_name_refusals(nat, lnat, call):
    with pytest.raises(ValueError) as want:
        call(F, LF)
    with pytest.raises(ValueError) as got:
        call(nat, lnat)
    assert str(got.value) == str(want.value)


# --- validation ----------------------------------------------------------------------------

BAD = [
    dict(cw_after=(5000, 4000)),
    dict(cw_after=(5000, 5000)),
    dict(cw_after=(3300,)),
    dict(cw_after=(7671,)),
    dict(cw_after=(-1,)),
    dict(cw_after=(5000.9, 5000)),
    dict(n_hdr=1000),
    dict(n_hdr=-2640),
    dict(n_pre=600),
    dict(n_pre=np.int64(661)),
    dict(n_data=0),
    dict(n_data=-7),
    dict(hdr_rho=1.0),
    dict(hdr_rho=-0.5),
    dict(hdr_rho=float("nan")),
    dict(hdr_rho=1e-5 - 1),
    dict(hdr_rho=2),
    dict(hdr_rho=True),
    dict(hdr_rho=1e300),
    dict(n_hdr=1000, n_pre=600),
    dict(hdr_rho=2.0, n_pre=600),
]
LBAD = BAD + [dict(n_data=4095), dict(n_data=-2), dict(n_hdr=2640), dict(n_pre=660),
              dict(cw_after=(2500, 3835)), dict(cw_after=(1650,))]


def _outcome(make, waveform=None):
    """("refused", text) or ("made", fields)."""
    try:
        s = make()
    except ValueError as e:
        return "refused", str(e)
    fields = s.fields() if waveform is None else (waveform, *dataclasses.astuple(s))
    return "made", fields


@pytest.mark.parametrize("kwargs", BAD, ids=[str(k) for k in BAD])
def test_ce_validation(nat, kwargs):
    base = dict(name="bad", n_pre=660, n_hdr=2640, n_data=4096, cw_after=(5000, 7670), hdr_rho=None)
    base.update(kwargs)
    want = _outcome(lambda: F.FrameSpec(**base), WAVEFORM_CE)
    assert want[0] == "refused"
    assert _outcome(lambda: nat.make(WAVEFORM_CE, **base)) == want


@pytest.mark.parametrize("kwargs", LBAD, ids=[str(k) for k in LBAD])
def test_l_validation(nat, kwargs):
    """The same table on L's numbers (some of CE's refusals are L frames)."""
    base = dict(name="l-bad", n_pre=330, n_hdr=1320, n_data=4096, cw_after=(2500, 3834), hdr_rho=None)
    base.update(kwargs)
    want = _outcome(lambda: LF.LFrameSpec(**base), WAVEFORM_L)
    assert _outcome(lambda: nat.make(WAVEFORM_L, **base)) == want


def test_accepted_forms(nat):
    """What __post_init__ normalises: int(c) of each window, float(rho), the default rho."""
    for kw in (dict(cw_after=[5000.7, np.int64(7670)]), dict(hdr_rho=0), dict(hdr_rho="0.25"),
               dict(n_hdr=0, cw_after=()), dict(hdr_rho=np.float32(0.1))):
        base = dict(name="ok", n_pre=660, n_hdr=2640, n_data=4096, cw_after=(5000, 7670), hdr_rho=None)
        base.update(kw)
        want = F.FrameSpec(**base)
        got = nat.make(WAVEFORM_CE, **base)
        assert got.fields() == (WAVEFORM_CE, *dataclasses.astuple(want)), kw
        assert type(got.hdr_rho) is float and all(type(c) is int for c in got.cw_after)
    with pytest.raises(ValueError, match="waveform must be"):
        nat.make(2, "x", 660, 2640, 4096, ())


# --- refusals of the assembling functions -----------------------------------------------

def _both(ref_call, nat_call, exc=ValueError):
    with pytest.raises(exc) as want:
        ref_call()
    with pytest.raises(exc) as got:
        nat_call()
    assert str(got.value) == str(want.value)


@pytest.mark.parametrize("wave", ["ce", "l"])
def test_assemble_refusals(nat, lnat, wave):
    ref, mine = (F, nat) if wave == "ce" else (LF, lnat)
    spec = F.SHORT if wave == "ce" else LF.L_SHORT
    tiny = F.TINY if wave == "ce" else LF.L_TINY
    b = np.zeros(N_HDR_BITS, np.uint8)
    x = np.zeros(spec.n_data)
    cases = [
        (spec, None, x),
        (spec, np.zeros(10, np.uint8), x),
        (spec, np.full(N_HDR_BITS, 2, np.uint8), x),
        (spec, np.zeros((2, N_HDR_BITS), np.uint8), x),
        (spec, np.full(N_HDR_BITS, 0.5), x),
        (spec, np.full(N_HDR_BITS, np.nan), x),
        (spec, b, np.zeros(spec.n_data + 1)),
        (spec, b, np.zeros((2, spec.n_data))),
        (spec, b, 3.0),
        (spec, None, np.zeros(5)),                       # which complaint comes first
        (tiny, b, np.zeros(tiny.n_data)),
        (tiny, np.zeros((2, 3)), np.zeros(tiny.n_data)),
        (tiny, b, np.zeros(3)),
    ]
    for s, bb, xx in cases:
        _both(lambda: ref.assemble(s, bb, xx), lambda: mine.assemble(s, bb, xx))
    _both(lambda: F.spread_signs(spec, None), lambda: nat.spread_signs(spec, None))
    _both(lambda: F.spread_signs(spec, b[:-1]), lambda: nat.spread_signs(spec, b[:-1]))
    _both(lambda: ref.disassemble(np.zeros(spec.n_sym + 1), spec),
          lambda: mine.disassemble(np.zeros(spec.n_sym + 1), spec))
    _both(lambda: ref.disassemble(np.float64(1.0), spec), lambda: mine.disassemble(np.float64(1.0), spec),
          IndexError)


def test_pair_refusals(lnat):
    for v in ([1.0, 2.0, 3.0], np.zeros((2, 2)), 1.0):
        _both(lambda: LF.pair(v), lambda: lnat.pair(v))
    _both(lambda: LF.data_latents(LF.L_SHORT, np.zeros(3)), lambda: lnat.data_latents(LF.L_SHORT, np.zeros(3)))
    _both(lambda: LF.data_latents(LF.L_SHORT, np.zeros(4096)),
          lambda: lnat.data_latents(LF.L_SHORT, np.zeros(4096)))       # no bits for the spread copy


def test_wrong_waveform_is_an_assertion(nat, lnat):
    b = np.zeros(N_HDR_BITS, np.uint8)
    for call in (lambda m: m.assemble(LF.L_SHORT, b, np.zeros(4096)),
                 lambda m: m.disassemble(np.zeros(LF.L_SHORT.n_sym), LF.L_SHORT),
                 lambda m: m.spread_symbols(LF.L_SHORT, b)):
        with pytest.raises(AssertionError):
            call(F)
        with pytest.raises(AssertionError):
            call(nat)
    with pytest.raises((AssertionError, AttributeError)):
        LF.assemble(F.TINY, None, np.zeros(1024))
    with pytest.raises(AssertionError):
        lnat.assemble(F.TINY, None, np.zeros(1024))


# --- the refusal texts' formatting -------------------------------------------------------

def test_float_repr(nat):
    """Python's float repr (the hdr_rho refusal prints it), on random bit
    patterns, decimal round numbers and the powers of two whose rounding
    interval is lopsided."""
    rng = np.random.default_rng(13)
    bits = rng.integers(0, 2**63, 20000, dtype=np.uint64) | (rng.integers(0, 2, 20000, dtype=np.uint64) << 63)
    vals = list(bits.view(np.float64)) + [2.0 ** k for k in range(-1074, 1024)]
    vals += [float(f"{m}e{e}") for m in (1, 5, 123, 9999, 1234567) for e in range(-330, 310, 7)]
    vals += [0.054, 0.1, 1e16, 1e15, 9.999999999999999e15, 1e-4, 1e-5, 5e-324, 1.7976931348623157e308,
             -0.0, 0.0, float("inf"), float("-inf"), float("nan"), 0.3, 2 / 3, 1e22, 1e23]
    for v in vals:
        v = float(v)
        assert nat._repr_float(v) == repr(v), v.hex()


@pytest.mark.parametrize("s", ["short", "", "it's", 'say "x"', "both ' and \"", "a\\b", "tab\there",
                               "nl\n", "\x00\x7f", "café", " ­", "☃"])
def test_str_repr(nat, s):
    assert nat._repr_str(s) == repr(s)
