"""QRSSTVAE receiver front end, design 6.2 and 6.9, acceptance A-1 to A-3
and A-8.

- A-1: `audio_to_fe` / `fe_to_audio` round trip (in-band error < -60 dB,
  power within 0.05 dB); `channelise` exact on its 1/64 Hz grid.
- A-2: the blanker never touches a CE signal up to +20 dB, and catches
  >= 99% of the sample energy of 30 dB clicks.
- A-3: `normalise` undoes a fast receive AGC to within 1 dB rms at
  SNR2500 <= -10 dB.
- `noise_psd`: the out-of-band noise floor is the in-band one.
- A-8: `PassbandStore` across an hour boundary, int16 within 1 LSB,
  48 h expiry, restart, gaps as zeros with a mask.
"""

import json
import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import ndimage

from sstvae.qrss import ce, frame, frontend
from sstvae.qrss import channel as ch
from sstvae.qrss.channel import Agc, ChannelConfig, Impulses
from sstvae.qrss.constants import FE_FS
from sstvae.qrss.morse import keying_units

from qrss_helpers import Q_TEST

KEY = keying_units("K1ABC")


def _sym(spec=frame.SHORT, seed=0):
    rng = np.random.default_rng(seed)
    hdr = rng.integers(0, 2, 2474).astype(np.uint8) if spec.n_hdr else None
    return frame.assemble(spec, hdr, rng.standard_normal(spec.n_data))


def _noise(n, var, rng):
    return math.sqrt(var / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def _noise_var(snr_db):
    """E|n|^2 at FE for a unit-power signal at SNR2500 = snr_db."""
    return FE_FS / (2500 * 10 ** (snr_db / 10))


# --- A-1 --------------------------------------------------------------------------------------

def test_a1_audio_round_trip():
    """fe -> audio -> fe: in-band error < -60 dB, audio power = FE power within 0.05 dB.

    "In band" is the front-end filter's passband, 400-2600 Hz audio: the
    design's 255-tap Kaiser-8 low-pass at 1250 Hz is down 0.3 dB at 300
    and 2700 Hz (6 dB at 250/2750), which is a level error there, not a
    loss of anything a signal 50 Hz wide needs.
    """
    rng = np.random.default_rng(1)
    n = 40000
    X = np.fft.fft(_noise(n, 1.0, rng))
    f = np.fft.fftfreq(n, 1 / FE_FS)
    X[np.abs(f) > 1100] = 0                         # 400-2600 Hz audio
    fe = np.fft.ifft(X)
    audio = frontend.fe_to_audio(fe)
    back = frontend.audio_to_fe(audio)
    sl = slice(500, -500)                           # away from the filter's edge effects
    err = np.mean(np.abs(back[sl] - fe[sl]) ** 2) / np.mean(np.abs(fe[sl]) ** 2)
    assert 10 * np.log10(err) < -60
    p_db = 10 * np.log10(np.mean(audio ** 2) / np.mean(np.abs(fe) ** 2))
    assert abs(p_db) < 0.05
    assert back.dtype == np.complex64 and len(back) == n


def test_a1_audio_to_fe_power_of_a_tone():
    """A real sine of power P becomes a complex tone of power P at f - 1500."""
    n = 8000 * 4
    t = np.arange(n) / 8000
    x = np.sqrt(2) * 0.3 * np.cos(2 * np.pi * 1730 * t + 0.4)       # power 0.09
    fe = frontend.audio_to_fe(x)[2000:-2000]
    assert 10 * np.log10(np.mean(np.abs(fe) ** 2) / 0.09) == pytest.approx(0, abs=0.01)
    k = np.fft.fft(fe)
    f = np.fft.fftfreq(len(fe), 1 / FE_FS)
    assert f[np.argmax(np.abs(k))] == pytest.approx(230, abs=0.5)


def test_channel_uses_this_front_end():
    """channel.py delegates its audio <-> FE conversions here once frontend exists."""
    x = np.random.default_rng(2).standard_normal(4000)
    assert np.array_equal(ch.audio_to_fe(x), frontend.audio_to_fe(x))
    fe = frontend.audio_to_fe(x)
    assert np.array_equal(ch.fe_to_audio(fe), frontend.fe_to_audio(fe))


def test_a1_channelise_exact_on_its_grid():
    """A tone exactly on the 1/64 Hz grid comes out as a constant phasor
    (to the decimation filter's ripple), however far into the stream it
    starts; the mixing phase is exact integer turns."""
    f_mix = Fraction(-31337, 64)                     # -489.640625 Hz
    n0 = 7_654_321                                   # deep into a slot
    n = 4000 * 20
    idx = np.arange(n0, n0 + n, dtype=np.int64)
    D = 64 * FE_FS
    turns = ((idx % D) * (f_mix.numerator % D) % D) / D
    tone = np.exp(2j * np.pi * (turns + 0.123))
    y, fq = frontend.channelise(tone, FE_FS, float(f_mix), n0=n0)
    assert fq == f_mix and isinstance(fq, Fraction)
    mid = y[200:-200].astype(np.complex128)
    assert np.max(np.abs(mid - np.exp(2j * np.pi * 0.123))) < 1e-5
    assert len(y) == n // 16 and y.dtype == np.complex64
    # the mixer itself: exact, and stream-position consistent
    a = frontend.mix_phasor(n0, 1000, f_mix, FE_FS)
    b = frontend.mix_phasor(n0 - 500, 1500, f_mix, FE_FS)[500:]
    assert np.array_equal(a, b)
    assert np.allclose(a, np.exp(-2j * np.pi * float(f_mix) * idx[:1000] / FE_FS), atol=1e-6)
    # off-grid frequencies are quantised; an off-grid Fraction is refused
    assert frontend.quantise_mix(100.004) == Fraction(6400, 64)
    with pytest.raises(ValueError):
        frontend.mix_phasor(0, 10, Fraction(1, 3), FE_FS)


def test_channelise_alignment_and_band():
    """ch[k] is fe[16k] for a slow signal; a tone 200 Hz away is rejected."""
    n = 4000 * 10
    t = np.arange(n) / FE_FS
    slow = np.exp(2j * np.pi * (3.0 * t + 0.5 * np.sin(2 * np.pi * 0.7 * t)))
    y, _ = frontend.channelise(slow * np.exp(2j * np.pi * 400 * t), FE_FS, 400.0)
    ref = slow[::16]
    assert np.max(np.abs(y[100:-100] - ref[100:-100])) < 1e-4
    far, _ = frontend.channelise(np.exp(2j * np.pi * 600 * t), FE_FS, 400.0)
    assert np.max(np.abs(far[100:-100])) < 1e-4


# --- A-2 --------------------------------------------------------------------------------------

@pytest.mark.parametrize("snr_db", [-20.0, 0.0, 20.0])
def test_a2_blanker_never_touches_ce(snr_db):
    """A constant-envelope signal, lead-in and callsign windows included,
    is never blanked at any SNR up to +20 dB."""
    spec = frame.SHORT
    sim = ch.simulate(_sym(spec), spec, Q_TEST, ChannelConfig(snr_db=snr_db, seed=3),
                      keying=KEY, lead_in_s=10.0)
    _, keep = frontend.blank(sim.fe)
    assert np.all(keep == 1.0)


@pytest.mark.parametrize("snr_db", [0.0, 10.0])
def test_a2_blanker_catches_clicks(snr_db):
    """30 dB clicks (10/s, +-6 dB, 1 ms decay) on CE + noise: >= 99% of
    their sample energy blanked, while little else is."""
    spec = frame.SHORT
    n = 60 * FE_FS
    rng = np.random.default_rng(int(snr_db) + 5)
    z = ce.baseband(_sym(spec), spec, FE_FS, -12.0, n, keying=KEY, lead_in_s=10.0)
    imp, ev = ch.impulse_noise(n, FE_FS, 1.0, Impulses(rate_hz=10.0), rng)
    x = (z + _noise(n, _noise_var(snr_db), rng) + imp).astype(np.complex64)
    y, keep = frontend.blank(x)
    e = np.abs(imp) ** 2
    caught = float(np.sum(e * (1 - keep)) / np.sum(e))
    assert len(ev) > 400 and caught >= 0.99
    assert np.mean(keep < 1) < 0.1                  # ~13 ms per click
    assert np.array_equal(y, (x * keep).astype(np.complex64))


def test_blank_guard_and_taper():
    """One sample impulse: 2 ms guard each side, then a 1 ms raised-cosine taper."""
    x = np.ones(8000, dtype=np.complex64)
    x[4000] = 100
    _, keep = frontend.blank(x)
    g, L = 8, 4                                     # samples at 4 kHz
    assert np.all(keep[4000 - g:4000 + g + 1] == 0)
    assert np.all(keep[:4000 - g - L] == 1) and np.all(keep[4000 + g + L + 1:] == 1)
    ramp = keep[4000 + g + 1:4000 + g + L + 1]
    assert np.all(np.diff(ramp) > 0) and 0 < ramp[0] < ramp[-1] < 1
    assert np.allclose(ramp, keep[4000 - g - 1:4000 - g - L - 1:-1])


def test_channelise_keep_is_aligned():
    """The keep indicator decimates like the signal: a blanked stretch
    stays where it was, at CH rate, with values in [0, 1] away from edges."""
    keep = np.ones(4000 * 10, dtype=np.float32)
    keep[20000:22000] = 0.0
    k = frontend.channelise_keep(keep)
    assert len(k) == 2500
    assert np.all(k[1260:1365] < 0.01) and np.all(np.abs(k[100:1200] - 1) < 0.01)
    assert abs(np.sum(1 - k[100:-100]) * 16 - 2000) < 20


# --- A-3 --------------------------------------------------------------------------------------

@pytest.mark.parametrize("snr_db", [-10.0, -20.0])
def test_a3_normalise_removes_fast_agc(snr_db):
    """Fast AGC (2 ms / 200 ms) pumped by clicks and crashes: after blank
    and normalise, the gain left on the background noise (truth AGC gain
    over the normaliser's sqrt(level)) varies < 1 dB rms, against several
    dB before. Measured where the background is noise: kept samples away
    from the impulses themselves (a crash's own energy is not noise the
    AGC scaled, and normalise rightly levels it too)."""
    spec = frame.TINY
    cfg = ChannelConfig(snr_db=snr_db, agc=Agc(), seed=3,
                        impulses=Impulses(rate_hz=2.0, crash_per_min=4.0))
    sim = ch.simulate(_sym(spec), spec, Q_TEST, cfg)
    y, keep = frontend.blank(sim.fe)
    yn, level = frontend.normalise(y, keep)
    g = sim.truth.agc_gain.astype(np.float64)
    near = ndimage.maximum_filter1d(sim.truth.impulse_mask.view(np.uint8), 401).astype(bool)
    ok = (keep >= 1) & ~near
    assert ok.mean() > 0.6
    before = np.std(20 * np.log10(g[ok]))
    r = 20 * np.log10(g[ok] / np.sqrt(level[ok]))
    after = float(np.sqrt(np.mean((r - np.median(r)) ** 2)))
    assert after < 1.0 and after < 0.7 * before
    # the output noise floor is flat at median power 1
    assert np.median(np.abs(yn[ok]) ** 2) == pytest.approx(1.0, rel=0.05)


def test_normalise_tracks_a_gain_step():
    """A 20 dB gain step in noise: the sliding median switches within a
    hop, so the noise is level on both sides beyond 20 ms of the step."""
    rng = np.random.default_rng(6)
    n = 4 * FE_FS
    x = _noise(n, 1.0, rng)
    x[n // 2:] *= 10.0
    y, level = frontend.normalise(x.astype(np.complex64))
    p = np.abs(y) ** 2
    w = int(0.02 * FE_FS)
    left, right = p[:n // 2 - w], p[n // 2 + w:]
    assert 10 * np.log10(np.mean(right) / np.mean(left)) == pytest.approx(0, abs=0.3)
    assert np.median(left) == pytest.approx(1.0, rel=0.05)


# --- noise floor ------------------------------------------------------------------------------

def test_noise_psd_is_the_in_band_floor():
    """White FE noise of variance s2 reads s2/FE_FS per Hz from 85-120 Hz off
    the carrier (the channeliser's skirt corrected), with or without a
    strong CE signal in the channel; an excluded neighbour is left out."""
    rng = np.random.default_rng(4)
    n = 60 * FE_FS
    s2 = 4.0
    psd = frontend.noise_psd(frontend.channelise(_noise(n, s2, rng), FE_FS, 10.3)[0])
    assert len(psd) == 6 and psd.dtype == np.float32
    assert np.mean(psd) == pytest.approx(s2 / FE_FS, rel=0.04)
    # a +20 dB CE signal at the channel centre does not lift it
    spec = frame.TINY
    z = ce.baseband(_sym(spec), spec, FE_FS, -1.0, n)
    sig = z * math.sqrt(s2 * 100 * 2500 / FE_FS)       # SNR2500 = +20 dB
    other = ce.baseband(_sym(spec, seed=9), spec, FE_FS, -1.0, n)
    nb = other * np.exp(2j * np.pi * 100.0 * np.arange(n) / FE_FS) * math.sqrt(
        s2 * 100 * 2500 / FE_FS)                       # a CE neighbour 100 Hz up, +20 dB
    x = sig + _noise(n, s2, np.random.default_rng(5)) + nb
    c, _ = frontend.channelise(x, FE_FS, 0.0)
    with_nb = frontend.noise_psd(c)
    excl = frontend.noise_psd(c, exclude_hz=[100.0])
    assert np.mean(excl) == pytest.approx(s2 / FE_FS, rel=0.06)
    assert np.mean(with_nb) > np.mean(excl)


# --- A-8 --------------------------------------------------------------------------------------

def test_a8_passband_store(tmp_path):
    """Hourly int16 files: write across an hour boundary, read an arbitrary
    span within 1 LSB, gaps as zeros with a mask, restart, 48 h expiry."""
    store = frontend.PassbandStore(tmp_path, scale=8192.0)
    rng = np.random.default_rng(7)
    hour0 = 490_000                                     # an hour count (2025-11-24)
    t_a = hour0 * 3600 + 3600 - 100.0                   # 100 s before the boundary
    x = (_noise(4 * 60 * FE_FS, 0.5, rng)).astype(np.complex64)
    store.write(t_a, x, expire=False)
    assert store.hours() == [hour0, hour0 + 1]
    lsb = 1 / 8192
    t1, t2 = t_a + 37.25, t_a + 171.5                   # across the boundary
    y = store.read(t1, t2)
    a = int(round(37.25 * FE_FS))
    ref = x[a:a + len(y)]
    assert len(y) == int(round((t2 - t1) * FE_FS))
    assert np.max(np.abs(y.real - ref.real)) <= 0.5 * lsb + 1e-9
    assert np.max(np.abs(y.imag - ref.imag)) <= 0.5 * lsb + 1e-9
    # a gap, then more data: zeros with a mask
    t_b = t_a + 4 * 60 + 30.0
    x2 = np.full(10 * FE_FS, 0.25 + 0.5j, dtype=np.complex64)
    store.write(t_b, x2, expire=False)
    z, mask = store.read(t_a + 230.0, t_b + 5.0, return_mask=True)
    gap = slice(int(10 * FE_FS), int(40 * FE_FS))
    assert np.all(z[gap] == 0) and not mask[gap].any()
    assert mask[:int(10 * FE_FS)].all() and mask[-5 * FE_FS:].all()
    assert np.allclose(z[-5 * FE_FS:], 0.25 + 0.5j, atol=lsb)
    # never written at all: zeros, empty mask
    z, mask = store.read(t_a - 7200, t_a - 7100, return_mask=True)
    assert not mask.any() and not np.any(z)
    # restart: a new store on the same root reads the same (scale from meta.json)
    again = frontend.PassbandStore(tmp_path, scale=1.0)
    assert again.scale == 8192.0
    assert np.array_equal(again.read(t1, t2), y)
    assert json.loads((tmp_path / "meta.json").read_text())["fs"] == FE_FS
    # expiry: 48 h after the end of an hour, its files go
    now = (hour0 + 1) * 3600 + 48 * 3600 + 1.0
    assert again.expire(now) == [hour0]
    assert again.hours() == [hour0 + 1]
    assert not np.any(again.read(t1, t_a + 100.0))
    # writing data expires automatically by its own time
    again.write((hour0 + 60) * 3600.0, x2)
    assert again.hours() == [hour0 + 60]


def test_passband_store_clips_to_int16(tmp_path):
    store = frontend.PassbandStore(tmp_path, scale=8192.0)
    store.write(1000.0, np.array([10 + 0j, -10j, 1 + 1j], dtype=np.complex64), expire=False)
    y = store.read(1000.0, 1000.0 + 3 / FE_FS)
    assert y[0].real == pytest.approx(32767 / 8192) and y[1].imag == pytest.approx(-32767 / 8192)
    assert y[2] == pytest.approx(1 + 1j)
