import numpy as np

from sstvae.config import NC, NCP, MODES
from sstvae.modem import framing, ofdm
from sstvae.modem.dsp import to_baseband


def test_symbol_loopback_via_baseband():
    """Modulate random symbols, convert to baseband, demod each window."""
    rng = np.random.default_rng(1)
    n_sym = 20
    s = (rng.normal(size=(n_sym, NC)) + 1j * rng.normal(size=(n_sym, NC))) / np.sqrt(2)
    # Guard symbols on both ends so the baseband FIR transient stays clear.
    pad = np.zeros((2, NC), dtype=complex)
    x = ofdm.modulate_symbols(np.vstack([pad, s, pad]))
    z = to_baseband(x)
    for i in range(n_sym):
        start = (2 + i) * (160 + NCP) + NCP
        got = ofdm.demod_window(z, start)
        err = np.abs(got - s[i]) ** 2
        snr = 10 * np.log10(np.mean(np.abs(s[i]) ** 2) / np.mean(err))
        assert snr > 35, f"symbol {i}: {snr:.1f} dB"


def test_interleaver_roundtrip():
    for mode in MODES.values():
        rng = np.random.default_rng(mode.index)
        lat = rng.normal(size=mode.n_latents)
        assert np.array_equal(
            framing.deinterleave(framing.interleave(lat, mode), mode), lat
        )


def test_slots_symbols_roundtrip():
    from sstvae.config import LATENTS_PER_FRAME

    rng = np.random.default_rng(3)
    slots = rng.normal(size=LATENTS_PER_FRAME)
    sym = framing.slots_to_symbols(slots)
    assert np.allclose(framing.symbols_to_slots(sym), slots)
    # unit-RMS slots -> unit-power symbols
    assert abs(np.mean(np.abs(sym) ** 2) / np.mean(slots**2) - 1) < 1e-9


def test_header_roundtrip():
    for mode in MODES.values():
        soft = np.real(framing.header_symbol(mode)).astype(float)
        assert framing.decode_header(soft) is mode


def test_header_rejects_garbage():
    rng = np.random.default_rng(4)
    rejects = sum(
        framing.decode_header(rng.normal(size=NC)) is None for _ in range(50)
    )
    assert rejects >= 45
