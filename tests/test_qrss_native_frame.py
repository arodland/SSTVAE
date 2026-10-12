"""C++ frames (native/core/qrss/frame) against sstvae.qrss.frame and .lframe.

Bit-exact throughout: the layouts and lengths are integer arithmetic, and
`assemble`, `data_latents`, `pair` and `unpair` reproduce numpy's operation
order (including `pair`'s complex division, which is a multiplication by
1/sqrt(2)), so every comparison here is of bytes, signed zeros and NaNs
included. Skips when the extension is not built; `pytest
--native-qrss=tx` runs the rest of the QRSS suites against the same code.
"""

import dataclasses
import warnings

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


# --- adversarial probes (review) ----------------------------------------------------------
#
# Each probe runs the reference and the port on the same input and requires the
# same outcome: equal bytes, dtype, shape and type, or the same exception class
# with the same text. The inputs are what a caller could plausibly hand over
# (lists, tuples, ranges, every dtype numpy has) and what it should not
# (strings, complex bits with an imaginary part, long double that rounds to
# 0/1, scalars where a sequence belongs), because a port that accepts what the
# reference refuses is the direction that hides a bug upstream.

def _run(call):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return "ok", call()
        except Exception as e:  # noqa: BLE001 -- the class is part of the comparison
            return type(e), str(e)


def _agree(ref_call, nat_call, what=""):
    want, got = _run(ref_call), _run(nat_call)
    assert got[0] == want[0], (what, got, want)
    if want[0] != "ok":
        assert got[1] == want[1], what
        return
    w, g = want[1], got[1]
    if isinstance(w, tuple):
        assert isinstance(g, tuple) and len(g) == len(w), what
        for k, (a, b) in enumerate(zip(g, w)):
            _same(a, b, f"{what}[{k}]")
    else:
        assert type(g) is type(w), (what, type(g), type(w))
        _same(g, w, what)


def _bit_forms(b):
    """hdr_bits as callers hand them over, valid and not."""
    return {
        "uint8": b, "bool": b.astype(bool), "int64": b.astype(np.int64), "float16": b.astype(np.float16),
        "uint64": b.astype(np.uint64), "list": b.tolist(), "tuple": tuple(b.tolist()), "range": range(N_HDR_BITS),
        "object": np.array(b.tolist(), dtype=object), "strided": np.repeat(b, 2)[::2], "minus-zero": np.where(b, 1.0, -0.0),
        "complex": b.astype(complex), "complex-imag": b + 1j, "longdouble": b.astype(np.longdouble),
        "longdouble-off": b.astype(np.longdouble) + np.longdouble(2.0) ** -62, "close": b + 1e-12,
        "int8-neg": -b.astype(np.int8), "nan": np.full(N_HDR_BITS, np.nan), "inf": np.full(N_HDR_BITS, np.inf),
        "str": "0" * N_HDR_BITS, "list-str": ["0"] * N_HDR_BITS, "row": b[None, :], "short": b[:-1],
        "scalar": 0, "numpy-scalar": np.uint8(1), "0-d": np.array(1), "empty": [], "empty-2d": np.zeros((0, 3)),
        "empty-dict": {}, "empty-str": "", "dict": {1: 2}, "None": None,
    }


@pytest.mark.parametrize("spec", [F.SHORT, F.SHORT_V1, F.TINY, F.FULL_SPREADONLY, LF.L_SHORT, LF.L_TINY,
                                  LF.LFrameSpec("l-spread", n_hdr=0, n_data=6000, cw_after=(), hdr_rho=0.2),
                                  LF.LFrameSpec("l-v1", n_data=4096, cw_after=(2500, 3834), hdr_rho=0.0)],
                         ids=lambda s: s.name)
def test_probe_hdr_bit_forms(nat, lnat, spec):
    """assemble checks hdr_bits as the reference does: its own (b != 0) & (b != 1)
    on np.asarray(hdr_bits) and len() of the argument itself; spread_signs and
    data_latents read np.asarray(hdr_bits, dtype=np.float64)."""
    rng = np.random.default_rng(21)
    x = rng.standard_normal(spec.n_data)
    ref_asm, nat_asm = (LF.assemble, lnat.assemble) if spec.waveform else (F.assemble, nat.assemble)
    for name, bb in _bit_forms(_bits(rng)).items():
        _agree(lambda: ref_asm(spec, bb, x), lambda: nat_asm(spec, bb, x), f"assemble {name}")
        _agree(lambda: F.spread_signs(spec, bb), lambda: nat.spread_signs(spec, bb), f"spread_signs {name}")
        if spec.waveform:
            _agree(lambda: LF.data_latents(spec, x, bb), lambda: lnat.data_latents(spec, x, bb), f"data_latents {name}")
        else:
            _agree(lambda: F.spread_symbols(spec, bb), lambda: nat.spread_symbols(spec, bb), f"spread_symbols {name}")


@pytest.mark.parametrize("spec", [F.SHORT, F.TINY, LF.L_SHORT, LF.L_TINY], ids=lambda s: s.name)
def test_probe_x_forms(nat, lnat, spec):
    """x in every form, alone and together with bad bits (which complaint wins
    is the reference's operation order: CE checks x first, L the header)."""
    rng = np.random.default_rng(22)
    x = rng.standard_normal(spec.n_data)
    b = _bits(rng) if spec.has_header else None
    ref_asm, nat_asm = (LF.assemble, lnat.assemble) if spec.waveform else (F.assemble, nat.assemble)
    forms = {"float32": x.astype(np.float32), "list": x.tolist(), "strided": np.repeat(x, 2)[::2],
             "int": np.arange(spec.n_data), "object": np.array(x.tolist(), dtype=object), "complex": x + 0j,
             "longdouble": x.astype(np.longdouble), "row": x[None, :], "scalar": 1.0, "short": x[:-2],
             "None": None, "str": "abc", "big-int": [2 ** 80] * spec.n_data}
    for name, xx in forms.items():
        _agree(lambda: ref_asm(spec, b, xx), lambda: nat_asm(spec, b, xx), f"x {name}")
        _agree(lambda: ref_asm(spec, np.zeros(5), xx), lambda: nat_asm(spec, np.zeros(5), xx), f"x {name}, bad bits")
        if spec.waveform:
            _agree(lambda: LF.data_latents(spec, xx, b), lambda: lnat.data_latents(spec, xx, b), f"data_latents {name}")


def test_probe_data_latents_is_x(lnat):
    """At rho 0 the reference returns np.asarray(x, float64) itself -- a view
    stays that view (no contiguous copy), a float64 array is returned as is."""
    x = np.random.default_rng(23).standard_normal(2 * LF.L_TINY.n_data)
    view = x[::2]
    assert LF.data_latents(LF.L_TINY, view) is view
    assert lnat.data_latents(LF.L_TINY, view) is view
    ro = x[:LF.L_TINY.n_data].copy()
    ro.setflags(write=False)
    assert lnat.data_latents(LF.L_TINY, ro) is ro


ODD_DTYPES = {
    "object": lambda v: np.array(v.tolist(), dtype=object),
    "object-complex": lambda v: np.array((v + 2j).tolist(), dtype=object),
    "str": lambda v: np.array(["a"] * v.size).reshape(v.shape),
    "bytes": lambda v: np.array([b"ab"] * v.size).reshape(v.shape),
    "datetime": lambda v: np.zeros(v.shape, "M8[s]"),
    "timedelta": lambda v: np.arange(v.size).reshape(v.shape).astype("m8[s]"),
    "longdouble": lambda v: v.astype(np.longdouble) / 3,
    "clongdouble": lambda v: ((v + 1j) / 3).astype(np.clongdouble),
    "big-endian-f8": lambda v: v.astype(">f8"),
    "big-endian-c16": lambda v: (v + 1j * v[..., ::-1]).astype(">c16"),
    "uint64": lambda v: (np.abs(v) * 2 ** 60).astype(np.uint64),
    "int64-inexact": lambda v: (v * 2 ** 61).astype(np.int64) | 1,
    "float16": lambda v: v.astype(np.float16),
    "bool": lambda v: v > 0,
}


@pytest.mark.parametrize("spec", [F.SHORT, F.TINY, LF.L_SHORT, LF.L_TINY], ids=lambda s: s.name)
@pytest.mark.parametrize("dtype", list(ODD_DTYPES))
def test_probe_disassemble_any_dtype(nat, lnat, spec, dtype):
    """disassemble is a gather, so it takes any dtype the reference's y[..., idx]
    takes; L's goes on through unpair, which runs numpy's own arithmetic for
    dtypes that do not become complex128 exactly (long double rounds once,
    in extended precision)."""
    ref, mine = (LF.disassemble, lnat.disassemble) if spec.waveform else (F.disassemble, nat.disassemble)
    v = np.random.default_rng(24).standard_normal((2, spec.n_sym))
    y = ODD_DTYPES[dtype](v)
    _agree(lambda: ref(y, spec), lambda: mine(y, spec), dtype)
    _agree(lambda: ref(y[0], spec), lambda: mine(y[0], spec), dtype + " 1-D")


@pytest.mark.parametrize("dtype", list(ODD_DTYPES))
def test_probe_unpair_any_dtype(lnat, dtype):
    v = np.random.default_rng(25).standard_normal((3, 400))
    z = ODD_DTYPES[dtype](v)
    _agree(lambda: LF.unpair(z), lambda: lnat.unpair(z), dtype)
    _agree(lambda: LF.pair(z[0]), lambda: lnat.pair(z[0]), dtype + " pair")


@pytest.mark.parametrize("spec", [F.SHORT, F.FULL_V1, F.TINY, LF.L_SHORT, LF.L_FULL], ids=lambda s: s.name)
def test_probe_twin_compares_like_python(nat, lnat, spec):
    """twin compares `waveform` with ==, so 1.0, True and numpy scalars select
    like the ints they equal and anything else (None, "1", 2**70) is neither."""
    for w in (0, 1, 1.0, 0.0, True, False, np.int64(1), np.float64(0.0), np.array(1), None, "1", 2, -1,
              2 ** 70, 1 + 0j, 0.5):
        _agree(lambda: nat.of(LF.twin(spec, w)).fields(), lambda: lnat.twin(spec, w).fields(), repr(w))
    _agree(lambda: LF.twin(spec, np.array([1, 0])), lambda: lnat.twin(spec, np.array([1, 0])), "array")


def test_probe_twin_slices_code_points(nat, lnat):
    """spec.name[2:] counts code points: "é-short" is SHORT's twin, "éshort"
    is no frame's (a byte slice would answer the other way round)."""
    specs = [LF.LFrameSpec("é-short", n_data=4096, cw_after=(2500, 3834)),
             LF.LFrameSpec("éshort", n_data=4096, cw_after=(2500, 3834)),
             LF.LFrameSpec("😀étiny", n_hdr=0, n_data=1024, cw_after=()),
             LF.LFrameSpec("l-", n_data=4096, cw_after=()), LF.LFrameSpec("x", n_data=4096, cw_after=()),
             F.FrameSpec("ésshort", n_data=4096, cw_after=(5000, 7670)), F.FrameSpec("", n_data=4096, cw_after=()),
             F.FrameSpec(" x", n_data=4096, cw_after=())]
    for spec in specs:
        for w in (WAVEFORM_CE, WAVEFORM_L, 7):
            _agree(lambda: nat.of(LF.twin(spec, w)).fields(), lambda: lnat.twin(spec, w).fields(),
                   (spec.name, w))


@pytest.mark.parametrize("name", [" ", "​", "x\x85", "é", "퟿", "\U000e0001", "😀", "l- ",
                                  "l-​", "­", "\x7f"])
def test_probe_name_refusals_use_python_repr(nat, lnat, name):
    """The unknown-name refusal shows the name as Python's repr does, which
    escapes every non-printable code point, not only ASCII and Latin-1."""
    for ref_call, nat_call in ((lambda: F.get(name), lambda: nat.get(name)),
                               (lambda: LF.get(name), lambda: lnat.get(name)),
                               (lambda: LF.get_any(name), lambda: lnat.get_any(name))):
        _agree(ref_call, nat_call, repr(name))
    spec = F.FrameSpec(name, n_data=4096, cw_after=())
    _agree(lambda: F.assemble(spec, None, np.zeros(4096)), lambda: nat.assemble(spec, None, np.zeros(4096)))
    _agree(lambda: LF.twin(spec, WAVEFORM_L), lambda: lnat.twin(spec, WAVEFORM_L))


def test_probe_unhashable_name(nat, lnat):
    """`name in PRESETS` hashes it first: a list is a TypeError, not an unknown frame."""
    for v in ([], [1], {}, {1}):
        _agree(lambda: F.get(v), lambda: nat.get(v), repr(v))
        _agree(lambda: LF.get(v), lambda: lnat.get(v), repr(v))
        _agree(lambda: LF.get_any(v), lambda: lnat.get_any(v), repr(v))


def _random_specs(seed, count):
    """Valid frames of either waveform with random lengths, headers, rho and windows."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(count):
        is_l = bool(rng.integers(2))
        cls, n_pre, n_hdr_full = (LF.LFrameSpec, 330, 1320) if is_l else (F.FrameSpec, 660, 2640)
        n_hdr = int(rng.choice([0, n_hdr_full]))
        n_data = int(rng.integers(1, 3000)) * (2 if is_l else 1)
        rho = [None, 0.0, float(rng.random() * 0.99), SPREAD_RHO][int(rng.integers(4))]
        probe = cls(f"r{k}", n_pre=n_pre, n_hdr=n_hdr, n_data=n_data, cw_after=(), hdr_rho=rho)
        lo, hi = n_pre + n_hdr, probe.n_sym
        n_win = int(rng.integers(0, 5))
        cw = sorted(set(int(c) for c in rng.integers(lo + 1, hi + 1, n_win)))
        if cw and rng.integers(2):
            cw[-1] = hi                                    # a window that ends the frame
        out.append(cls(f"r{k}", n_pre=n_pre, n_hdr=n_hdr, n_data=n_data, cw_after=tuple(cw), hdr_rho=rho))
    return out


RANDOM_SPECS = _random_specs(31, 40)


@pytest.mark.parametrize("spec", RANDOM_SPECS, ids=[s.name for s in RANDOM_SPECS])
def test_probe_random_frames(nat, lnat, spec):
    """Random frames of both waveforms: every derived property, every layout
    array, pos_of across the frame and past both ends, and assemble/disassemble
    bitwise."""
    n = nat.of(spec)
    for name in DERIVED + (("n_data_payload",) if spec.waveform == WAVEFORM_L else ()):
        got, want = getattr(n, name), getattr(spec, name)
        assert got == want and type(got) is type(want), (name, got, want)
    assert n.window_spans_s() == spec.window_spans_s()
    s = np.arange(-3, spec.n_sym + 3)
    _same(n.pos_of(s), spec.pos_of(s), "pos_of")
    want, got = _ref_layout(spec), _nat_layout(nat, lnat, spec)
    for f in dataclasses.fields(want):
        _same(got[f.name], getattr(want, f.name), f.name)
    rng = np.random.default_rng(spec.n_data)
    b = _bits(rng) if spec.has_header else None
    x = rng.standard_normal(spec.n_data) * 2
    ref_asm, nat_asm = (LF.assemble, lnat.assemble) if spec.waveform else (F.assemble, nat.assemble)
    ref_dis, nat_dis = (LF.disassemble, lnat.disassemble) if spec.waveform else (F.disassemble, nat.disassemble)
    sym = ref_asm(spec, b, x)
    _same(nat_asm(spec, b, x), sym, "assemble")
    for a, w in zip(nat_dis(sym, spec), ref_dis(sym, spec)):
        _same(a, w, "disassemble")


def test_probe_names_with_lone_surrogates(nat, lnat):
    """Any str the reference takes as a name, lone surrogates included, survives
    the trip through the C++: a lookup refuses it with the reference's
    ValueError, a spec carries it (repr, fields, refusal texts), and twin
    slices it by code points ("\\udc80\\udc80short"[2:] is "short")."""
    for name in ("\udc80", "l-\udc80", "sh\ud800ort"):
        _agree(lambda: F.get(name), lambda: nat.get(name), ascii(name))
        _agree(lambda: LF.get(name), lambda: lnat.get(name), ascii(name))
        _agree(lambda: LF.get_any(name), lambda: lnat.get_any(name), ascii(name))
        _agree(lambda: F.FrameSpec(name, n_data=4096, cw_after=(1,)),
               lambda: nat.make(WAVEFORM_CE, name, 660, 2640, 4096, (1,)), ascii(name))
    specs = [F.FrameSpec("\udc80\udc80short", n_data=4096, cw_after=(5000, 7670)),
             LF.LFrameSpec("\ud800-short", n_data=4096, cw_after=(2500, 3834)),
             LF.LFrameSpec("l-\udc80", n_data=4096, cw_after=(2500, 3834)),
             F.FrameSpec("a\udc80", n_data=4096, cw_after=())]
    for spec in specs:
        n = nat.of(spec)
        assert n.name == spec.name and repr(n) == repr(spec)
        assert n.fields() == (spec.waveform, *dataclasses.astuple(spec))
        for w in (WAVEFORM_CE, WAVEFORM_L, 3):
            _agree(lambda: nat.of(LF.twin(spec, w)).fields(), lambda: lnat.twin(spec, w).fields(), ascii(spec.name))
        _agree(lambda: nat.of(F.plain(spec)).fields(), lambda: nat.plain(spec).fields(), ascii(spec.name))
        ref_asm, nat_asm = (LF.assemble, lnat.assemble) if spec.waveform else (F.assemble, nat.assemble)
        _agree(lambda: ref_asm(spec, None, np.zeros(spec.n_data)), lambda: nat_asm(spec, None, np.zeros(spec.n_data)))
