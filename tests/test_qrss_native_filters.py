"""C++ filters (native/core/qrss/filters) against the scipy they replace.

sstvae.qrss takes upfirdn, firwin, resample_poly, windows.hann and
special.i0e from scipy. The port copies scipy's own arithmetic (the loop
of _upfirdn_apply.pyx, numpy's operation order in firwin and the windows,
Cephes i0/i0e), so the bar here is **bit-exact**: same dtype, same shape,
same bits (NaN compared as NaN, zeros by sign). It rests on libm's
sin/cos/exp/sqrt being the ones numpy and scipy call, which holds on
x86-64 Linux; if a platform ever breaks it, the failures will be single
ulps in firwin and the windows, and the honest fix is a tolerance here,
not a change to the port.

Skips when the extension is not built; `pytest --native-qrss=tx` (and
`=rx` for the receiver's overlays) runs the QRSS suites against the same
code.
"""

import numpy as np
import pytest
from scipy import signal, special

from qrss_native import require
from sstvae.qrss import acquire, ce, cwid, frame, frontend, lframe, lwave
from sstvae.qrss.constants import CH_DECIM, CH_FS, FE_FS, FS


@pytest.fixture(scope="module")
def nat(native):
    return require(native, "filters")


def same(got, want, what=""):
    got, want = np.asarray(got), np.asarray(want)
    assert got.dtype == want.dtype, f"{what}: dtype {got.dtype} != {want.dtype}"
    assert got.shape == want.shape, f"{what}: shape {got.shape} != {want.shape}"
    for g, w in ((got.real, want.real), (got.imag, want.imag)) if np.iscomplexobj(want) else ((got, want),):
        nan = np.isnan(w)
        assert np.array_equal(np.isnan(g), nan), f"{what}: NaN positions differ"
        ok = (g == w) & (np.signbit(g) == np.signbit(w))
        bad = ~(ok | nan)
        assert not bad.any(), (f"{what}: {bad.sum()} of {bad.size} differ, "
                               f"max |diff| {np.max(np.abs(g[bad] - w[bad]))}")


def raises_same(ref, nat_call):
    """The same exception type and arguments (so the same message) from both."""
    with pytest.raises(Exception) as e_ref:
        ref()
    with pytest.raises(type(e_ref.value)) as e_nat:
        nat_call()
    assert str(e_nat.value) == str(e_ref.value)
    assert e_nat.value.args == e_ref.value.args
    return e_nat.value


class Overlay:
    """`base` with some names replaced (what tests/qrss_native/filters_rx.py does)."""

    def __init__(self, base, **names):
        self.__dict__["_base"] = base
        self.__dict__.update(names)

    def __getattr__(self, name):
        return getattr(self.__dict__["_base"], name)


# --- i0, i0e ------------------------------------------------------------------------------

def _special_grid():
    rng = np.random.default_rng(7)
    edges = [0.0, -0.0, 8.0, np.nextafter(8.0, 0), np.nextafter(8.0, 9), 709.0, 709.78,
             np.nextafter(709.782712893384, 0), 709.782712893384, 709.79, 713.0, 714.0, 1e300,
             5e-324, 1e-300, np.inf, -np.inf, np.nan]
    return np.concatenate([np.linspace(-40, 40, 80001), rng.uniform(-750, 750, 200000),
                           rng.standard_normal(20000) * 1e-3, edges, -np.array(edges)])


@pytest.mark.parametrize("name", ["i0", "i0e"])
def test_i0_and_i0e_are_cephes_bit_for_bit(nat, name):
    x = _special_grid()
    same(getattr(nat, name)(x), getattr(special, name)(x), name)


@pytest.mark.parametrize("name", ["i0", "i0e"])
def test_i0_follows_the_ufunc_types(nat, name):
    """Loops f->f and d->d: bool and small ints take float32, as scipy's do."""
    ref, got = getattr(special, name), getattr(nat, name)
    for x in (1.5, 3, True, np.float32(2.5), np.float16(1.25), np.array(2.0), np.int8(4),
              np.arange(12, dtype=np.int16).reshape(3, 4), np.arange(5, dtype=np.int64),
              np.linspace(0, 20, 7, dtype=np.float32), [[1.0, 2.0], [30.0, -9.0]], np.array([])):
        w, g = ref(x), got(x)
        assert type(g) is type(w), (x, type(g), type(w))
        same(g, w, f"{name}({x!r})")
    raises_same(lambda: ref(1 + 2j), lambda: got(1 + 2j))


# --- windows -------------------------------------------------------------------------------

@pytest.mark.parametrize("sym", [True, False])
def test_windows(nat, sym):
    for M in list(range(0, 70)) + [255, 321, 481, 2001, 16000]:
        same(nat.hann(M, sym=sym), signal.windows.hann(M, sym=sym), f"hann({M})")
        same(nat.hamming(M, sym=sym), signal.windows.hamming(M, sym=sym), f"hamming({M})")
        for beta in (0.0, 5.0, 8, 12.0, 0.3, 38.5):
            same(nat.kaiser(M, beta, sym=sym), signal.windows.kaiser(M, beta, sym=sym), f"kaiser({M}, {beta})")


def test_get_window(nat):
    for w in ["hann", "hamming", "ham", "hamm", "han", ("kaiser", 4.0), ("ksr", 7), 3.5, np.float64(2.0),
              "hann_periodic", "hamming_symmetric", ("kaiser_periodic", 6.0)]:
        for n in (1, 2, 9, 64, 257):
            for fftbins in (True, False):
                same(nat.get_window(w, n, fftbins=fftbins), signal.get_window(w, n, fftbins=fftbins),
                     f"get_window({w!r}, {n}, {fftbins})")


WINDOW_REFUSALS = [
    ("get_window", ("hann", 0), {}),
    ("get_window", ("hann", -2), {}),
    ("get_window", ("hann", 4.0), {}),
    ("get_window", ("hann", 8), dict(fftbins=1)),
    ("get_window", ("nosuch", 8), {}),
    ("get_window", ("nosuch_periodic", 8), {}),
    ("get_window", (("hann", 3), 8), {}),
    ("get_window", (("kaiser",), 8), {}),
    ("get_window", (("kaiser", 1, 2), 8), {}),
    ("get_window", ((1, 2), 8), {}),
    ("get_window", (object(), 8), {}),
    ("get_window", ("hann", True), {}),
    ("get_window", ("nope", True), {}),
    ("hann", (True,), {}),
    ("kaiser", (False, 3.0), {}),
    ("hann", (-1,), {}),
    ("hann", (2.5,), {}),
    ("hamming", (-1,), {}),
    ("kaiser", (-3, 2.0), {}),
]


@pytest.mark.parametrize("name,args,kw", WINDOW_REFUSALS)
def test_window_refusals(nat, name, args, kw):
    ref = getattr(signal.windows if name in ("hann", "hamming", "kaiser") else signal, name)
    raises_same(lambda: ref(*args, **kw), lambda: getattr(nat, name)(*args, **kw))


def test_a_window_scipy_has_but_the_port_does_not_says_so(nat):
    for w in ("tukey", ("tukey", 0.3), "blackman", ("general_hamming", 0.6)):
        signal.get_window(w, 16)          # scipy has it
        with pytest.raises(NotImplementedError, match="not ported"):
            nat.get_window(w, 16)
        with pytest.raises(NotImplementedError, match="not ported"):
            nat.firwin(15, 0.3, window=w)


# --- firwin ----------------------------------------------------------------------------------

# Every firwin sstvae/qrss makes, by where it makes it.
SSTVAE_FIRWIN = {
    "lwave._interp_taps": ((2 * lwave.INTERP_HALF + 1, 125.0), dict(fs=FS, window=("kaiser", 12.0))),
    "frontend._FE_TAPS": ((255, 1250.0), dict(fs=FS, window=("kaiser", 8.0))),
    "frontend decimator": ((2 * 10 * CH_DECIM + 1, 1.0 / CH_DECIM), dict(window=("kaiser", 10.0))),
    "acquire._notch_taps": ((acquire.NOTCH_TAPS, acquire.NOTCH_HZ), dict(fs=CH_FS, window="hann")),
}


@pytest.mark.parametrize("where", sorted(SSTVAE_FIRWIN))
def test_every_firwin_in_sstvae(nat, where):
    args, kw = SSTVAE_FIRWIN[where]
    same(nat.firwin(*args, **kw), signal.firwin(*args, **kw), where)


def test_firwin_is_the_reference_tables(nat):
    same(nat.firwin(2 * lwave.INTERP_HALF + 1, 125.0, fs=FS, window=("kaiser", 12.0)), lwave._interp_taps(1),
         "lwave._interp_taps(1)")
    same(nat.firwin(255, 1250.0, fs=FS, window=("kaiser", 8.0)), frontend._FE_TAPS, "frontend._FE_TAPS")
    same(nat.firwin(acquire.NOTCH_TAPS, acquire.NOTCH_HZ, fs=CH_FS, window="hann"), acquire._notch_taps(),
         "acquire._notch_taps")


@pytest.mark.parametrize("max_rate", [2, 3, 4, 5, 8, 16, 32, 64, 160])
@pytest.mark.parametrize("window", [("kaiser", 5.0), ("kaiser", 10.0), "hann", "hamming"])
def test_firwin_as_resample_poly_designs_it(nat, max_rate, window):
    n = 2 * 10 * max_rate + 1
    same(nat.firwin(n, 1.0 / max_rate, window=window), signal.firwin(n, 1.0 / max_rate, window=window))


def test_firwin_random_designs(nat):
    rng = np.random.default_rng(3)
    windows = ["hamming", "hann", "ham", ("kaiser", 0.5), ("kaiser", 14.0), 6.0, "hann_periodic", ("ksr", 3)]
    checked = refused = 0
    for _ in range(600):
        numtaps = int(rng.integers(1, 400))
        k = int(rng.integers(1, 5))
        fs = [None, 2.0, 8000, 250.0][int(rng.integers(0, 4))]
        nyq = 1.0 if fs is None else fs / 2
        cutoff = np.sort(rng.uniform(0.01, 0.99, k)) * nyq
        kw = dict(window=windows[int(rng.integers(0, len(windows)))],
                  pass_zero=[True, False, "lowpass", "highpass", "bandpass", "bandstop"][int(rng.integers(0, 6))],
                  scale=bool(rng.integers(0, 2)))
        if fs is not None:
            kw["fs"] = fs
        if rng.uniform() < 0.15:
            kw["width"] = float(rng.uniform(0.01, 0.3)) * nyq
        arg = cutoff if k > 1 or rng.uniform() < 0.5 else float(cutoff[0])
        try:
            want = signal.firwin(numtaps, arg, **kw)
        except ValueError as e:
            with pytest.raises(ValueError) as got:
                nat.firwin(numtaps, arg, **kw)
            assert str(got.value) == str(e)
            refused += 1
            continue
        same(nat.firwin(numtaps, arg, **kw), want, f"firwin({numtaps}, {arg}, {kw})")
        checked += 1
    assert checked > 300 and refused > 30


FIRWIN_REFUSALS = [
    ((31, 0.0), {}),
    ((31, 1.0), {}),
    ((31, 4000.0), dict(fs=8000)),
    ((31, []), {}),
    ((31, [[0.1, 0.2]]), {}),
    ((31, [0.3, 0.2]), {}),
    ((31, [0.2, 0.2]), {}),
    ((31, 0.2), dict(fs=np.array([8000.0]))),
    ((31, [0.1, 0.4]), dict(pass_zero="lowpass")),
    ((31, 0.1), dict(pass_zero="bandstop")),
    ((31, [0.1, 0.4]), dict(pass_zero="highpass")),
    ((31, 0.1), dict(pass_zero="bandpass")),
    ((31, 0.1), dict(pass_zero="nope")),
    ((31, 0.1), dict(pass_zero=1)),
    ((31, 0.1), dict(pass_zero=np.True_)),
    ((30, 0.1), dict(pass_zero=False)),
    ((30, [0.1, 0.4]), {}),
    ((0, 0.1), {}),
    ((-3, 0.1), {}),
    ((np.int64(0), 0.1), {}),
    ((4.0, 0.1), {}),
    ((5.5, 0.1), {}),
    ((31, 0.1), dict(window="nope")),
    ((31, 0.1), dict(window=("hann", 2))),
    ((31, 0.1), dict(window=("kaiser",))),
    ((31, 0.1), dict(window=(2, 3))),
    ((31, 0.1), dict(window=object())),
    ((31, 0.1), dict(width="abc")),
    # the order scipy checks in: the earlier refusal wins
    ((31, 2.0), dict(pass_zero="nope")),
    ((30, [0.2, 0.4]), dict(pass_zero="nope")),
    ((0, [0.2, 0.4]), dict(window="nope")),
    ((30, [0.2, 0.4]), dict(window="nope")),
    ((4.0, [0.2, 0.4]), dict(window="nope")),
    ((4.5, 0.2), dict(window="nope")),
    ((31, 2.0), dict(width="abc")),
    ((31, 0.2), dict(width="abc", pass_zero="nope")),
    ((30, [0.2, 0.4]), dict(width=0.1, window="nope")),
    ((0, 0.2), dict(pass_zero="lowpass", window=object())),
    ((True, 0.2), {}),
    ((False, 0.2), {}),
    ((True, 0.2), dict(window="nope")),
]


@pytest.mark.parametrize("args,kw", FIRWIN_REFUSALS)
def test_firwin_refusals(nat, args, kw):
    raises_same(lambda: signal.firwin(*args, **kw), lambda: nat.firwin(*args, **kw))


def test_firwin_takes_the_order_dependent_cases_scipy_takes(nat):
    """width replaces any window, valid or not; a NaN cutoff passes the checks."""
    same(nat.firwin(31, [0.2, 0.4], width=0.1, window="nope"), signal.firwin(31, [0.2, 0.4], width=0.1, window="nope"))
    with np.errstate(invalid="ignore"):
        same(nat.firwin(11, np.nan), signal.firwin(11, np.nan))
    same(nat.firwin(np.int16(33), np.float32(0.25)), signal.firwin(np.int16(33), np.float32(0.25)))


def test_kaiser_helpers(nat):
    for a in (0.0, 10.0, 21.0, 21.5, 35.0, 50.0, 50.5, 120.0):
        assert nat.kaiser_beta(a) == signal.kaiser_beta(a)
    for n, w in ((31, 0.1), (481, 0.02), (2, 0.5)):
        assert nat.kaiser_atten(n, w) == signal.kaiser_atten(n, w)


# --- upfirdn ---------------------------------------------------------------------------------

def _x(rng, n, dtype):
    x = rng.standard_normal(n) * 3
    if np.dtype(dtype).kind == "c":
        x = x + 1j * rng.standard_normal(n)
    return x.astype(dtype)


@pytest.mark.parametrize("cplx", [False, True])
@pytest.mark.parametrize("table,up,down", [("ce", 485, 2), ("ce", 485, 4), ("ce", 485, 64),
                                            ("l", 970, 2), ("l", 970, 64), ("l", 970, 4)])
def test_upfirdn_on_the_pulse_tables(nat, table, up, down, cplx):
    """`_grid_sum`'s call: the pulse table behind z0 zeros, upsampled by T."""
    rng = np.random.default_rng(up + down)
    t = ce.PULSE_TABLE if table == "ce" else lwave.PULSE_TABLE
    for n in (1, 2, 17, 40, 333):
        for z0 in (0, 1, down - 1):
            h = np.concatenate([np.zeros(z0), t])
            x = _x(rng, n, np.complex128 if cplx else np.float64)
            same(nat.upfirdn(h, x, up=up, down=down), signal.upfirdn(h, x, up=up, down=down),
                 f"upfirdn({table}, n={n}, z0={z0})")


def test_upfirdn_interpolates_as_lwave_resample_does(nat):
    rng = np.random.default_rng(5)
    h = lwave._interp_taps(32)
    for n in (1, 5, 16, 200):
        x = _x(rng, n, np.complex128)
        same(nat.upfirdn(h, x, up=32), signal.upfirdn(h, x, up=32), f"n={n}")


def test_upfirdn_random(nat):
    rng = np.random.default_rng(11)
    dtypes = [(np.float64, np.float64), (np.float64, np.complex128), (np.float32, np.float32),
              (np.float32, np.complex64), (np.complex128, np.float64), (np.float64, np.int64),
              (np.float32, np.int8), (np.float64, np.float32)]
    for i in range(1500):
        hd, xd = dtypes[i % len(dtypes)]
        h = _x(rng, int(rng.integers(1, 40)), hd)
        n = int(rng.integers(0, 50))
        x = _x(rng, n, xd) if np.dtype(xd).kind != "i" else rng.integers(-50, 50, n).astype(xd)
        up, down = int(rng.integers(1, 13)), int(rng.integers(1, 13))
        kw = {}
        if rng.uniform() < 0.2:
            kw["cval"] = float(rng.standard_normal())
        try:
            want = signal.upfirdn(h, x, up, down, **kw)
        except ValueError as e:
            with pytest.raises(ValueError) as got:
                nat.upfirdn(h, x, up, down, **kw)
            assert str(got.value) == str(e)
            continue
        same(nat.upfirdn(h, x, up, down, **kw), want, f"case {i}: h{h.shape} x{x.shape} {up}/{down} {kw}")


def test_upfirdn_complex_cval_and_signed_zeros(nat):
    h = np.array([1.0, -2.0, 0.5, -0.0])
    for x in (np.array([-0.0, 0.0, -0.0]), np.array([0j, -0.0 - 0.0j, 1 - 0.0j])):
        for up, down in ((1, 1), (3, 2), (2, 5)):
            for cval in (0, -0.0, 0.5):
                same(nat.upfirdn(h, x, up, down, cval=cval), signal.upfirdn(h, x, up, down, cval=cval))
    x = np.arange(6) * (1 + 1j)
    same(nat.upfirdn(h, x, 2, 3, cval=1 - 2j), signal.upfirdn(h, x, 2, 3, cval=1 - 2j))


@pytest.mark.parametrize("axis", [0, 1, -1, 2, -3])
def test_upfirdn_along_an_axis(nat, axis):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 7, 5)) + 1j * rng.standard_normal((3, 7, 5))
    h = rng.standard_normal(9)
    same(nat.upfirdn(h, x, 3, 2, axis=axis), signal.upfirdn(h, x, 3, 2, axis=axis))
    same(nat.upfirdn(h, x.real, 1, 4, axis=axis), signal.upfirdn(h, x.real, 1, 4, axis=axis))


@pytest.mark.parametrize("call", [
    lambda f: f([], [1.0, 2.0]),
    lambda f: f([[1.0]], [1.0, 2.0]),
    lambda f: f([1.0], [1.0, 2.0], up=0),
    lambda f: f([1.0], [1.0, 2.0], down=-1),
    lambda f: f([1.0], [1.0, 2.0], mode="nosuch"),
    lambda f: f([1.0], [1.0, 2.0], axis=1),
    lambda f: f([1.0], [], up=3),
    lambda f: f([1.0], 2.0),
])
def test_upfirdn_refusals(nat, call):
    raises_same(lambda: call(signal.upfirdn), lambda: call(nat.upfirdn))


def test_upfirdn_takes_scipy_s_loose_arguments(nat):
    x = np.arange(10.0)
    same(nat.upfirdn([1, 2, 3], x, 2.9, np.int64(3)), signal.upfirdn([1, 2, 3], x, 2.9, np.int64(3)))
    same(nat.upfirdn([1, 2, 3], x, mode="CONSTANT"), signal.upfirdn([1, 2, 3], x, mode="CONSTANT"))
    for mode in ("reflect", "line", "wrap"):
        signal.upfirdn([1, 2, 3], x, 2, mode=mode)      # scipy has it
        with pytest.raises(NotImplementedError, match="not ported"):
            nat.upfirdn([1, 2, 3], x, 2, mode=mode)


# --- resample_poly ---------------------------------------------------------------------------

@pytest.mark.parametrize("fs", [250, 500, 1000, 2000, 4000, 16000, 3000])
def test_resample_poly_as_ce_to_audio_calls_it(nat, fs):
    import math
    rng = np.random.default_rng(fs)
    g = math.gcd(FS, fs)
    for n in (0, 1, 5, 64, 1000):
        z = _x(rng, n, np.complex128)
        same(nat.resample_poly(z, FS // g, fs // g), signal.resample_poly(z, FS // g, fs // g), f"n={n}")


@pytest.mark.parametrize("dtype", [np.complex128, np.float64])
def test_resample_poly_as_the_channeliser_calls_it(nat, dtype):
    rng = np.random.default_rng(1)
    for n in (0, 3, 15, 16, 17, 321, 5000):
        y = _x(rng, n, dtype)
        same(nat.resample_poly(y, 1, CH_DECIM, window=("kaiser", 10.0)),
             signal.resample_poly(y, 1, CH_DECIM, window=("kaiser", 10.0)), f"n={n}")


def test_resample_poly_random(nat):
    rng = np.random.default_rng(9)
    windows = [("kaiser", 5.0), ("kaiser", 10.0), "hann", "hamming", 3.0]
    dtypes = [np.float64, np.complex128, np.float32, np.complex64, np.int64, np.int16, np.bool_]
    for i in range(400):
        dt = dtypes[i % len(dtypes)]
        n = int(rng.integers(0, 300))
        x = _x(rng, n, dt) if np.dtype(dt).kind in "fc" else rng.integers(0, 2 if dt is np.bool_ else 40, n).astype(dt)
        up, down = int(rng.integers(1, 20)), int(rng.integers(1, 20))
        w = windows[i % len(windows)]
        kw = dict(window=w)
        if rng.uniform() < 0.2:
            kw["cval"] = float(rng.standard_normal())
        same(nat.resample_poly(x, up, down, **kw), signal.resample_poly(x, up, down, **kw),
             f"case {i}: {dt.__name__}[{n}] {up}/{down} {kw}")


def test_resample_poly_with_given_taps(nat):
    rng = np.random.default_rng(4)
    x = rng.standard_normal(100)
    xc = x + 1j * rng.standard_normal(100)
    for w in ([0.25, 0.5, 0.25], [1, 2, 1], np.hanning(11), np.hanning(12).astype(np.float32), [], [3.0]):
        for xx in (x, xc, x.astype(np.float32)):
            for up, down in ((3, 2), (1, 4), (5, 1)):
                same(nat.resample_poly(xx, up, down, window=w), signal.resample_poly(xx, up, down, window=w),
                     f"window={w!r} {xx.dtype} {up}/{down}")


def test_resample_poly_copies_when_the_rates_agree(nat):
    for x in (np.arange(7), np.arange(7.0), np.ones(3, dtype=np.complex64)):
        got = nat.resample_poly(x, 3, 3)
        same(got, signal.resample_poly(x, 3, 3))
        assert got is not x


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_resample_poly_along_an_axis(nat, axis):
    rng = np.random.default_rng(6)
    x = rng.standard_normal((40, 30)) + 1j * rng.standard_normal((40, 30))
    same(nat.resample_poly(x, 3, 2, axis=axis), signal.resample_poly(x, 3, 2, axis=axis))


@pytest.mark.parametrize("call", [
    lambda f: f(np.ones(8), 2.5, 1),
    lambda f: f(np.ones(8), 2, 1.5),
    lambda f: f(np.ones(8), 0, 1),
    lambda f: f(np.ones(8), 2, -1),
    lambda f: f(np.ones(8), 2, 3, padtype="nosuch"),
    lambda f: f(np.ones(8), 2, 3, padtype="line", cval=1.0),
    lambda f: f(np.ones(8), 2, 3, window=np.ones((2, 2))),
    lambda f: f(np.ones(8), 2, 3, window=np.float64(5.0)),
    lambda f: f(np.ones(8), 2, 3, window="nosuch"),
    lambda f: f(np.ones(8), 2, 3, window=("kaiser",)),
    lambda f: f(np.ones(8), 2, 3, axis=1),
])
def test_resample_poly_refusals(nat, call):
    raises_same(lambda: call(signal.resample_poly), lambda: call(nat.resample_poly))


def test_resample_poly_says_what_it_does_not_port(nat):
    for kw in (dict(padtype="mean"), dict(padtype="line"), dict(window="tukey")):
        signal.resample_poly(np.ones(8), 2, 3, **kw)       # scipy has it
        with pytest.raises(NotImplementedError, match="not ported"):
            nat.resample_poly(np.ones(8), 2, 3, **kw)
    with pytest.raises(NotImplementedError, match="not ported"):
        nat.resample_poly(np.ones(8, dtype=np.float16), 2, 3)


# --- the sstvae.qrss functions, with the native filters in -----------------------------------

def _ce_sym(spec, seed=0):
    rng = np.random.default_rng(seed)
    hdr = rng.integers(0, 2, 2474).astype(np.uint8) if spec.has_header else None
    return frame.assemble(spec, hdr, rng.standard_normal(spec.n_data))


def _l_sym(spec, seed=0):
    rng = np.random.default_rng(seed)
    hdr = rng.integers(0, 2, 2474).astype(np.uint8) if spec.has_header else None
    return lframe.assemble(spec, hdr, rng.standard_normal(spec.n_data))


@pytest.mark.parametrize("fs,n0", [(8000, -1940), (8000, 123457), (4000, 0), (250, -77)])
def test_ce_phase_grid(nat, monkeypatch, fs, n0):
    sym = _ce_sym(frame.SHORT)
    n = 10 * fs
    want = ce.phase_grid(fs, n0, n, sym, frame.SHORT)
    monkeypatch.setattr(ce, "upfirdn", nat.upfirdn)
    same(ce.phase_grid(fs, n0, n, sym, frame.SHORT), want, "phase_grid")


def test_ce_to_audio_from_a_lower_rate(nat, monkeypatch):
    rng = np.random.default_rng(2)
    z = _x(rng, 2500, np.complex128)
    want = ce.to_audio(z, 250, 3001, 2, n0=17)
    monkeypatch.setattr(ce, "resample_poly", nat.resample_poly)
    same(ce.to_audio(z, 250, 3001, 2, n0=17), want, "to_audio")


@pytest.mark.parametrize("fs,n0", [(250, -1000), (8000, 4321), (4000, 0)])
def test_lwave_data_grid(nat, monkeypatch, fs, n0):
    sym = _l_sym(lframe.L_SHORT)
    n = 8 * fs
    want = lwave.data_grid(fs, n0, n, sym, lframe.L_SHORT)
    monkeypatch.setattr(lwave, "upfirdn", nat.upfirdn)
    same(lwave.data_grid(fs, n0, n, sym, lframe.L_SHORT), want, "data_grid")


def test_lwave_resample(nat, monkeypatch):
    rng = np.random.default_rng(8)
    bb = lwave.Baseband(z=_x(rng, 3000, np.complex128), n0=-250, fs=250)
    cases = [(-9000, 4000), (-100, 333), (0, 8000), (90000, 9000), (95000, 100)]
    want = [lwave.resample(bb, 8000, i0, m) for i0, m in cases]
    monkeypatch.setattr(lwave, "upfirdn", nat.upfirdn)
    monkeypatch.setattr(lwave, "firwin", nat.firwin)
    for (i0, m), w in zip(cases, want):
        same(lwave.resample(bb, 8000, i0, m), w, f"resample({i0}, {m})")


def test_frontend_channelise(nat, monkeypatch):
    rng = np.random.default_rng(12)
    fe = _x(rng, 20000, np.complex64)
    keep = (rng.uniform(size=20000) > 0.1).astype(np.float32)
    want = frontend.channelise(fe, FE_FS, 37.3, n0=-5)
    want_keep = frontend.channelise_keep(keep)
    monkeypatch.setattr(frontend, "signal", Overlay(signal, resample_poly=nat.resample_poly, firwin=nat.firwin))
    got = frontend.channelise(fe, FE_FS, 37.3, n0=-5)
    same(got[0], want[0], "channelise")
    assert got[1] == want[1]
    same(frontend.channelise_keep(keep), want_keep, "channelise_keep")


def test_acquire_notch_and_cwid_ln_i0(nat, monkeypatch):
    want = acquire._notch_taps()
    x = np.concatenate([np.linspace(0, 40, 4001), [0.0, 1e-9, 700.0, 1e5]])
    want_ln = cwid._ln_i0(x)
    monkeypatch.setattr(acquire, "signal", Overlay(signal, firwin=nat.firwin))
    monkeypatch.setattr(cwid, "special", Overlay(special, i0e=nat.i0e))
    same(acquire._notch_taps(), want, "_notch_taps")
    same(cwid._ln_i0(x), want_ln, "_ln_i0")
