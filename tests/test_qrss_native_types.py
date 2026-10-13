"""Parity of native/core/qrss/types (sstvae_native.qrss.types) with
sstvae/qrss/types.py.

- every record crosses Python -> C++ -> Python unchanged (rx_types.hpp's
  converters, `_rx.to_ref`), NaN and infinities included, header and cw
  None or set;
- pass_uid is the reference's string for the same inputs, its refusals
  included;
- a PassResult file either side writes, the other loads with every field
  equal -- and the two files are the same bytes.
"""

import dataclasses
import json
import math
from fractions import Fraction

import numpy as np
import pytest

from qrss_helpers import Q_TEST
from qrss_native import _rx, require
from sstvae.qrss import constants as C
from sstvae.qrss import frame
from sstvae.qrss import types as qt
from sstvae.qrss.header import HeaderFields


@pytest.fixture(scope="module")
def nt(native, request):
    if request.config.getoption("--native-qrss", default=None):
        pytest.skip("--native-qrss replaces the reference this file compares against")
    return require(native, "types")


def same(a, b, path="") -> None:
    """Equal values of equal types: arrays by dtype, shape and bytes,
    floats bit for bit (NaN equal to NaN), dataclasses field by field."""
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        assert type(a) is type(b), (path, type(a), type(b))
        for f in dataclasses.fields(a):
            same(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
        return
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype, (path, a.dtype, b.dtype)
        assert a.shape == b.shape, (path, a.shape, b.shape)
        assert a.tobytes() == b.tobytes(), path
        return
    if isinstance(a, float) or isinstance(b, float):
        assert isinstance(a, float) and isinstance(b, float), (path, a, b)
        assert (math.isnan(a) and math.isnan(b)) or (a == b and math.copysign(1, a) == math.copysign(1, b)), \
            (path, a, b)
        return
    assert type(a) is type(b), (path, type(a), type(b))
    assert a == b, (path, a, b)


def fake_pass(header=None, cw=True, **over):
    n = 4096
    rng = np.random.default_rng(11)
    z = rng.standard_normal(n).astype(np.float32)
    fields = dict(
        uid=qt.pass_uid(Q_TEST, 1500.25, z), q=Q_TEST, frame="short", waveform=0, f_hz=1500.25,
        timing=qt.Timing(tau0=3000.5, ppm=12.0, gamma=0.0, z=40.0, cov=np.eye(2) * 1e-3),
        report=qt.TrackReport(offset_hz=0.25, drift_hz_per_min=0.1, wander_hz_rms=0.02, doppler_hz=0.1,
                              ppm=12.0, snr2500_db=-12.0, z_ref=40.0, kappa=1.02, slip_free=True,
                              suspect=False),
        z=z, w=np.full(n, 3.0, np.float32), hdr_llr=rng.standard_normal(2474).astype(np.float32),
        header=header,
        cw=qt.CwIdResult(text="K1ABC", z_match=9.5, keying="fsk", agrees=True) if cw else None,
        ch=(rng.standard_normal(1000) + 1j * rng.standard_normal(1000)).astype(np.complex64),
        ch_fs=C.CH_FS, ch_t0_index=3000.5, f_mix_hz=Fraction(96017, 64),
        psi=np.ones(frame.SHORT.n_pos, np.float32), estimator="joint",
    )
    fields.update(over)
    return qt.PassResult(**fields)


HEADER = HeaderFields(callsign="K1ABC", grid="FN42", picture_id=0xD349C671, mode=1, segment=0,
                      codec_id=0xD1D8)
WEIRD_REPORT = qt.TrackReport(offset_hz=float("nan"), drift_hz_per_min=float("inf"),
                              wander_hz_rms=float("-inf"), doppler_hz=-0.0, ppm=1e-320,
                              snr2500_db=-1.7976931348623157e308, z_ref=0.1 + 0.2, kappa=1e22,
                              slip_free=False, suspect=True)

PASSES = {
    "plain": lambda: fake_pass(),
    "header": lambda: fake_pass(HEADER),
    "header_no_grid": lambda: fake_pass(dataclasses.replace(HEADER, grid=None, version=2, reserved=5)),
    "no_cw": lambda: fake_pass(cw=False),
    "cw_unknown": lambda: fake_pass(cw=qt.CwIdResult(text=None, z_match=float("nan"), keying="ook",
                                                     agrees=None)),
    "nan_inf": lambda: fake_pass(HEADER, report=WEIRD_REPORT, f_hz=float("nan"),
                                 timing=qt.Timing(tau0=-0.0, ppm=float("inf"), gamma=1e-300,
                                                  z=float("nan"), cov=np.full((3, 3), np.nan))),
    "unicode": lambda: fake_pass(estimator="jöint   \U0001f600 \x7f \"q\" \\ \t", frame="l-short",
                                 waveform=1, em_round=3),
    "empty_cov": lambda: fake_pass(timing=qt.Timing(tau0=1.0, ppm=0.0, gamma=0.0, z=0.0,
                                                    cov=np.zeros((0, 0)))),
    "big": lambda: fake_pass(q=2 ** 40, ch_fs=250, ch_t0_index=1e15, f_mix_hz=Fraction(-7, 3),
                             em_round=-1),
    "empty_arrays": lambda: fake_pass(z=np.zeros(0, np.float32), w=np.zeros(0, np.float32),
                                      hdr_llr=np.zeros(0, np.float32), ch=np.zeros(0, np.complex64),
                                      psi=np.zeros(0, np.float32)),
}


# --- records both ways ----------------------------------------------------------------------


RECORDS = [
    qt.Timing(tau0=1.5, ppm=-3.0, gamma=0.25, z=7.0, cov=np.arange(9.0).reshape(3, 3)),
    qt.Timing(tau0=float("nan"), ppm=float("inf"), gamma=-0.0, z=0.0, cov=np.diag([1.0, 2.0])),
    qt.FreqPath(t_s=np.linspace(0, 10, 7), f_hz=np.full(7, 1500.5), weight=np.ones(7)),
    qt.CarrierCandidate(f_hz=1499.0, drift_hz_per_min=0.5, metric=3.0, waveform=None),
    qt.CarrierCandidate(f_hz=1499.0, drift_hz_per_min=0.5, metric=float("nan"), waveform=1),
    qt.Detection(f_hz=1501.0, path=qt.FreqPath(np.arange(3.0), np.arange(3.0), np.ones(3)), timing=None,
                 z_ref=5.0, method="preamble", lead_in_s=2.0),
    qt.Detection(f_hz=1501.0, path=qt.FreqPath(np.arange(3.0), np.arange(3.0), np.ones(3)),
                 timing=qt.Timing(1.0, 2.0, 3.0, 4.0, np.eye(3)), z_ref=5.0, method="template",
                 lead_in_s=0.0, waveform=1, a3_z=8.5),
    WEIRD_REPORT,
    qt.CwIdResult(text="K1ABC", z_match=1.0, keying="fsk", agrees=False),
    qt.CwIdResult(text=None, z_match=1.0, keying="ook", agrees=None),
    qt.EmPrior(x_hat=np.linspace(-1, 1, 5), v=np.full(5, 0.5)),
    HEADER,
    dataclasses.replace(HEADER, grid=None),
]


@pytest.mark.parametrize("rec", RECORDS, ids=lambda r: type(r).__name__)
def test_records_both_ways(nt, rec):
    back = _rx.to_ref(nt.roundtrip(rec))
    same(back, rec)
    # and from the dict form the C++ hands out
    same(_rx.to_ref(nt.roundtrip(nt.roundtrip(rec))), rec)


@pytest.mark.parametrize("name", sorted(PASSES))
def test_pass_result_both_ways(nt, name):
    p = PASSES[name]()
    same(_rx.to_ref(nt.roundtrip(p)), p)


# --- pass_uid -------------------------------------------------------------------------------


def test_pass_uid(nt):
    rng = np.random.default_rng(3)
    z = rng.standard_normal(100).astype(np.float32)
    for q in (0, 1, Q_TEST, -5, 2 ** 40):
        for f in (1500.25, 0.0, -0.0, 1e-4, 0.0005, 0.0015, 0.0025, 2.5e-3, 1234.5675, -2.0005, 1e12,
                  *rng.uniform(0, 3000, 50), *(np.arange(0, 20) + 0.5) / 1000):
            assert nt.pass_uid(q, float(f), z) == qt.pass_uid(q, float(f), z), (q, f)
    # z is hashed as float32, whatever it arrived as
    zz = rng.standard_normal(33)
    assert nt.pass_uid(7, 1.0, zz) == qt.pass_uid(7, 1.0, zz)
    assert nt.pass_uid(7, 1.0, []) == qt.pass_uid(7, 1.0, [])
    assert nt.pass_uid(np.int64(9), np.float32(2.5), z) == qt.pass_uid(np.int64(9), np.float32(2.5), z)
    for bad, exc in ((float("nan"), ValueError), (float("inf"), OverflowError), (-float("inf"), OverflowError)):
        with pytest.raises(exc) as want:
            qt.pass_uid(1, bad, z)
        with pytest.raises(exc) as got:
            nt.pass_uid(1, bad, z)
        assert str(got.value) == str(want.value)


# --- the file -------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PASSES))
def test_pass_files_are_the_same_bytes(nt, name, tmp_path):
    p = PASSES[name]()
    p.save(tmp_path / "py.npz")
    py_bytes = (tmp_path / "py.npz").read_bytes()
    nt.save_pass(p, str(tmp_path / "cpp.npz"))
    assert (tmp_path / "cpp.npz").read_bytes() == py_bytes
    assert nt.pass_bytes(p) == py_bytes
    with np.load(tmp_path / "py.npz") as d:
        assert nt.meta_json(p) == d["meta"].tobytes().decode()


@pytest.mark.parametrize("name", sorted(PASSES))
def test_pass_files_cross_load(nt, name, tmp_path):
    p = PASSES[name]()
    p.save(tmp_path / "py.npz")
    nt.save_pass(p, str(tmp_path / "cpp.npz"))
    from_py = _rx.to_ref(nt.load_pass(str(tmp_path / "py.npz")))
    from_cpp = qt.PassResult.load(tmp_path / "cpp.npz")
    same(from_py, qt.PassResult.load(tmp_path / "py.npz"))
    same(from_cpp, from_py)
    same(_rx.to_ref(nt.parse_pass((tmp_path / "py.npz").read_bytes())), from_py)
    # what load hands back is what was saved (Python's load gives back the
    # cov's shape; a 0x0 one is (0, 0) both ways)
    same(dataclasses.replace(from_py, timing=p.timing), p)


def test_python_int_fields_are_written_as_floats(nt, tmp_path):
    """A Python int in a float field: json.dumps writes "0", the C++ (whose
    field is a double) "0.0" -- the one way the files can differ, and why
    the receiver's builders pass floats. Both load back equal."""
    p = fake_pass(f_hz=1500, timing=qt.Timing(tau0=3000, ppm=0, gamma=0, z=40, cov=np.eye(2)))
    p.save(tmp_path / "py.npz")
    meta_py = json.loads(np.load(tmp_path / "py.npz")["meta"].tobytes())
    meta_cpp = json.loads(nt.meta_json(p))
    assert meta_py == meta_cpp
    assert nt.pass_bytes(p) != (tmp_path / "py.npz").read_bytes()
    assert type(meta_py["timing"]["tau0"]) is int and type(meta_cpp["timing"]["tau0"]) is float


def test_load_is_strict_about_dtypes(nt, tmp_path):
    """Python's load hands back whatever dtypes the file holds; the C++
    record is typed, so it refuses a file whose arrays are not the
    receiver's (a deliberate deviation, ValueError naming the member)."""
    p = fake_pass(z=np.zeros(8, np.float64))
    p.save(tmp_path / "p.npz")
    assert qt.PassResult.load(tmp_path / "p.npz").z.dtype == np.float64
    with pytest.raises(ValueError, match="member z"):
        nt.load_pass(str(tmp_path / "p.npz"))


def test_load_refusals(nt, tmp_path):
    p = fake_pass()
    p.save(tmp_path / "p.npz")
    good = (tmp_path / "p.npz").read_bytes()
    with pytest.raises(RuntimeError, match="cannot open"):
        nt.load_pass(str(tmp_path / "missing.npz"))
    with pytest.raises(ValueError):
        nt.parse_pass(good[:100])
    # meta that is not UTF-8, or not JSON: Python raises UnicodeDecodeError
    # / JSONDecodeError, both ValueErrors
    for bad_meta, exc in ((b"\xff{}", ValueError), (b"{not json", ValueError), (b'{"uid": "x"}', KeyError)):
        arrays = {k: getattr(p, k) for k in ("z", "w", "hdr_llr", "ch", "psi")}
        np.savez(tmp_path / "bad.npz", meta=np.frombuffer(bad_meta, np.uint8), **arrays,
                 timing_cov=np.eye(2))
        with pytest.raises(exc) as want:
            qt.PassResult.load(tmp_path / "bad.npz")
        with pytest.raises(exc) as got:
            nt.load_pass(str(tmp_path / "bad.npz"))
        if exc is KeyError:
            assert str(got.value) == str(want.value) == "'header'"
