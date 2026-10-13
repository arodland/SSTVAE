"""C++ slot functions (native/core/qrss/tx) against sstvae.qrss.tx.

Bit-exact: a slot's stream symbols are the stored latents (float16 from a
.qrsp, int8 from a beacon file, exact as float64), precoded with an exact
WHT and assembled from exact tables, so CE (float64) and L (complex128)
symbols agree bit for bit -- every preset of both waveforms, from both
kinds of source, at three slots, with and without use_int8. slot_samples
is one ceil of the same float expression; check_pilot and every refusal
are compared by type and text. transmit_audio stays Python (the CE and L
audio are not ported yet). The adapters in tests/qrss_native/tx.py (what
--native-qrss installs) are used here.
"""

import dataclasses
from fractions import Fraction

import numpy as np
import pytest

from qrss_helpers import Q_TEST, synthetic_full_latents
from qrss_native import require
from qrss_native import tx as rows_mod
from sstvae.qrss import beaconfile, frame, lframe, picture
from sstvae.qrss import tx as ref
from sstvae.qrss.constants import L_EDGE_HZ, L_PILOT_HZ
from sstvae.qrss.header import HeaderFields

CALL = "K1ABC/P"
Q_SLOTS = (Q_TEST, Q_TEST + 1, (1 << 40) + 12345)
SPECS = [frame.PRESETS[n] for n in ("full", "short", "tiny")] + \
    [lframe.PRESETS[n] for n in ("l-full", "l-short", "l-tiny")]


@pytest.fixture(scope="module")
def nat(native):
    return require(native, "tx")


@pytest.fixture(scope="module")
def ad(native):
    """The --native-qrss adapters by attribute name."""
    require(native, "tx")
    return {attr: f for (_, attr), f in rows_mod.rows(native.qrss).items()}


def _header(sp, segment=1, call=CALL, grid="FN42"):
    return HeaderFields(callsign=call, grid=grid, picture_id=sp.picture_id, mode=sp.mode,
                        segment=segment, codec_id=sp.codec_id)


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    """A stored picture read from a .qrsp and a beacon file read from disk."""
    d = tmp_path_factory.mktemp("qrss_tx_native")
    sp0 = picture.StoredPicture.from_latents(synthetic_full_latents(3), 0xD1D8, 1)
    picture.save_qrsp(d / "p.qrsp", sp0)
    sp = picture.load_qrsp(d / "p.qrsp")
    beaconfile.write(d / "b.bin", beaconfile.from_picture(sp, 1, _header(sp, 1), ook=True))
    bf = beaconfile.read(d / "b.bin")
    return sp, bf


def _outcome(f, *args, **kw):
    try:
        return ("ok", f(*args, **kw))
    except Exception as e:      # noqa: BLE001 -- any refusal is compared
        return ("raise", type(e), str(e))


def _same(want, got):
    if isinstance(want, np.ndarray):
        assert isinstance(got, np.ndarray), got
        assert got.dtype == want.dtype and got.shape == want.shape, (want.dtype, got.dtype)
        assert got.tobytes() == want.tobytes()                     # bit for bit
    elif isinstance(want, tuple):
        assert isinstance(got, tuple) and len(got) == len(want)
        for a, b in zip(want, got):
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


# --- slot_symbols ----------------------------------------------------------------------------

@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_slot_symbols_bit_exact(nat, sources, spec):
    sp, bf = sources
    for q in Q_SLOTS:
        for use_int8 in (None, True):
            want = ref.slot_symbols(sp, q, spec, segment=1, header=_header(sp, 1),
                                    use_int8=use_int8)
            assert want.dtype == (np.complex128 if spec.waveform == lframe.WAVEFORM_L
                                  else np.float64)
            _same(want, nat.slot_symbols(sp, q, spec, segment=1, header=_header(sp, 1),
                                         use_int8=use_int8))
            _same(ref.slot_symbols(bf, q, spec, use_int8=use_int8),
                  nat.slot_symbols(bf, q, spec, use_int8=use_int8))
        if not spec.has_header:    # a .qrsp needs no header for a frame without one
            _same(ref.slot_symbols(sp, q, spec), nat.slot_symbols(sp, q, spec))
            _same(ref.slot_symbols(sp, q, spec, segment=0), nat.slot_symbols(sp, q, spec,
                                                                             segment=0))


def test_slot_symbols_defaults_and_spec_forms(nat, sources):
    sp, bf = sources
    _same(ref.slot_symbols(bf, Q_TEST), nat.slot_symbols(bf, Q_TEST))
    _same(ref.slot_symbols(bf, np.int64(Q_TEST), frame.SHORT),
          nat.slot_symbols(bf, Q_TEST, nat_spec := frame.get("short")))
    assert nat_spec is frame.SHORT
    # A frame given by the native spec object itself.
    import sstvae_native
    _same(ref.slot_symbols(bf, Q_TEST, lframe.PRESETS["l-short"]),
          nat.slot_symbols(bf, Q_TEST, sstvae_native.qrss.lframe.get("l-short")))
    # q as a float and as a numpy integer (int(q), as the reference reads it)
    _same(ref.slot_symbols(bf, float(Q_TEST), frame.TINY),
          nat.slot_symbols(bf, float(Q_TEST), frame.TINY))


def test_slot_symbols_refusals(nat, sources):
    sp, bf = sources
    h = _header(sp, 1)
    rep = dataclasses.replace
    short_bf = dataclasses.replace(bf, latents_i8=bf.latents_i8[:1000].copy())
    cases = [
        ((sp, Q_TEST, frame.SHORT), {}),                                       # no header
        ((sp, Q_TEST, lframe.PRESETS["l-full"]), {}),
        ((sp, Q_TEST, frame.SHORT), {"segment": 2, "header": h}),
        ((sp, Q_TEST, frame.SHORT), {"segment": -1}),
        ((sp, Q_TEST, frame.SHORT), {"segment": 0, "header": h}),             # header seg 1
        ((sp, Q_TEST, frame.SHORT), {"segment": 1, "header": rep(h, codec_id=1)}),
        ((sp, Q_TEST, frame.SHORT), {"segment": 1, "header": rep(h, picture_id=7)}),
        ((sp, Q_TEST, frame.SHORT), {"segment": 1, "header": rep(h, callsign="bad call")}),
        ((sp, Q_TEST, frame.SHORT), {"segment": 1, "header": rep(h, grid="ZZ99")}),
        ((sp, Q_TEST, frame.SHORT), {"segment": "1", "header": h}),
        ((sp, -1, frame.SHORT), {"segment": 1, "header": h}),
        ((sp, 1 << 64, frame.SHORT), {"segment": 1, "header": h}),
        ((sp, "abc", frame.TINY), {}),
        ((bf, Q_TEST, frame.SHORT), {"segment": 0}),
        ((bf, Q_TEST, frame.SHORT), {"segment": 1.5}),
        ((bf, Q_TEST, frame.SHORT), {"header": rep(h, grid="FN43")}),
        ((bf, Q_TEST, frame.SHORT), {"header": h}),                             # matches: ok
        ((bf, Q_TEST, frame.SHORT), {"segment": 1, "use_int8": True}),          # ok
        ((bf, Q_TEST, frame.SHORT), {"use_int8": False}),
        ((bf, Q_TEST, frame.SHORT), {"use_int8": np.False_}),                   # not `is False`
        ((bf, -5, frame.TINY), {}),
        ((short_bf, Q_TEST, frame.SHORT), {}),
        ((short_bf, Q_TEST, frame.TINY), {}),                                   # 1024 <= 1000? no
        ((short_bf, Q_TEST, lframe.PRESETS["l-short"]), {}),
        ((42, Q_TEST, frame.TINY), {}),
        ((None, Q_TEST, frame.TINY), {}),
    ]
    for args, kw in cases:
        _same_outcome(ref.slot_symbols, nat.slot_symbols, *args, **kw)


# --- the other slot functions --------------------------------------------------------------

def test_segment_and_header(ad, sources):
    sp, bf = sources
    h = _header(sp, 1)
    rep = dataclasses.replace
    f = ad["_segment_and_header"]
    for args in ((sp, None, None), (sp, 0, None), (sp, 1, h), (sp, np.int64(1), h),
                 (sp, 1.0, None), (sp, 2, None), (sp, -1, None), (sp, 0, h),
                 (sp, 1, rep(h, mode=0)), (bf, None, None), (bf, 1, h), (bf, 1, None),
                 (bf, 0, None), (bf, None, rep(h, callsign="W1AW")), ("x", None, None)):
        want = _same_outcome(ref._segment_and_header, f, *args)
        if want[0] == "ok" and args[0] is bf:
            assert isinstance(f(*args)[1], HeaderFields)


def test_slot_air_latents(nat, sources):
    sp, bf = sources
    for args in ((sp,), (sp, 0), (sp, 1), (sp, 1, True), (sp, 0, 1), (sp, 1, 0),
                 (sp, 1, False), (sp, -1), (sp, -2), (sp, -3), (sp, 2), (sp, np.int64(1)),
                 (bf,), (bf, 0), (bf, None, True), (bf, None, False), (bf, 5, np.False_)):
        _same_outcome(ref.slot_air_latents, nat.slot_air_latents, *args)


def test_slot_keying(nat, sources):
    sp, bf = sources
    for args in ((bf,), (bf, _header(sp, 0)), (sp,), (sp, _header(sp, 1)),
                 (sp, _header(sp, 1, call="W1AW")), (sp, _header(sp, 1, call="bad call")),
                 (object(), None)):
        _same_outcome(ref.slot_keying, nat.slot_keying, *args)


def test_slot_samples(nat):
    specs = [*frame.PRESETS.values(), *frame.LEGACY.values(), *frame.PROTOTYPES.values(),
             *lframe.PRESETS.values()]
    pres = (12.0, 0, 3, 12.5, 0.1, Fraction(1, 3), 600.0, -2.0)
    posts = (3.0, 0, 7.25, 1e-9)
    for spec in specs:
        _same_outcome(ref.slot_samples, nat.slot_samples, spec)
        for pre in pres:
            for post in posts:
                _same_outcome(ref.slot_samples, nat.slot_samples, spec, pre, post)
        for fs in (8000, 48000, 44100.0, 11025):
            _same_outcome(ref.slot_samples, nat.slot_samples, spec, fs=fs)
        for pre, post in ((float("nan"), 3.0), (12.0, float("inf")), (float("-inf"), 0.0)):
            _same_outcome(ref.slot_samples, nat.slot_samples, spec, pre, post)


def test_check_pilot(nat):
    edge = float(L_PILOT_HZ + L_EDGE_HZ)
    for f in (1500, 1500.0, Fraction(3001, 2), 310, 300 + edge, 300 + edge - 1e-12,
              np.nextafter(300 + edge, 0), 322.6, 322.7, 2700, 2700.0001, 299, 300, 0, -5,
              float("nan"), float("inf"), "1500", np.float64(1500.5), np.int64(2000), 4000):
        _same_outcome(ref.check_pilot, nat.check_pilot, f)
