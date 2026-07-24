import numpy as np
import pytest

from sstvae import hfchannel
from sstvae.config import CHIPS_PER_FRAME, MODES
from sstvae.modem import Modem, beacon


def test_encode_decode_roundtrip():
    chips = beacon.encode_chips(0, "K6ABC/P")
    r = beacon.decode(chips)
    assert r is not None
    assert r.chip_offset == 0
    assert r.frame_index == 0
    assert r.callsign == "K6ABC/P"


def test_frame_counter_range():
    for f in [0, 1, 500, beacon.MAX_FRAME_COUNTER]:
        r = beacon.decode(beacon.encode_chips(f, "W1AW"))
        assert r.frame_index == f
    with pytest.raises(ValueError):
        beacon.encode_chips(beacon.MAX_FRAME_COUNTER + 1, "W1AW")


def test_callsign_charset_and_padding():
    r = beacon.decode(beacon.encode_chips(3, "n0call"))  # lowercase, short
    assert r.callsign == "N0CALL"


def test_crc_rejects_garbage_no_false_lock():
    rng = np.random.default_rng(0)
    false_locks = 0
    for seed in range(300):
        junk = np.random.default_rng(seed).normal(size=400)
        if beacon.decode(junk) is not None:
            false_locks += 1
    assert false_locks == 0


def test_decode_finds_sync_anywhere_in_a_longer_stream():
    chips = beacon.encode_chips(42, "N0CALL")
    rng = np.random.default_rng(1)
    pad_a = rng.choice([-1.0, 1.0], size=37)
    pad_b = rng.choice([-1.0, 1.0], size=61)
    stream = np.concatenate([pad_a, chips, pad_b])
    r = beacon.decode(stream)
    assert r is not None
    assert r.chip_offset == len(pad_a)
    assert r.frame_index == 42
    assert r.callsign == "N0CALL"


def test_chip_stream_recovers_absolute_frame_index_from_mid_stream_window():
    """The core resync property: a window that starts mid-transmission
    (not superframe-aligned) still recovers the true absolute frame
    index of wherever the sync word happens to land inside it."""
    cs = beacon.chip_stream(0, 400, "W1AW")
    start_frame = 137
    win = cs[start_frame * CHIPS_PER_FRAME : start_frame * CHIPS_PER_FRAME + 4 * beacon.SUPERFRAME_LEN]
    r = beacon.decode(win)
    assert r is not None
    assert r.callsign == "W1AW"
    global_chip = start_frame * CHIPS_PER_FRAME + r.chip_offset
    assert global_chip // CHIPS_PER_FRAME == r.frame_index


def test_noise_tolerant_decode():
    rng = np.random.default_rng(2)
    chips = beacon.encode_chips(9, "KJ7ABC")
    noisy = chips + rng.normal(scale=0.5, size=chips.shape)
    r = beacon.decode(noisy)
    assert r is not None
    assert r.callsign == "KJ7ABC"


# --- integration with the real modem ----------------------------------------


def test_modem_reports_callsign_on_clean_channel():
    modem = Modem()
    lat = np.random.default_rng(3).normal(size=MODES["A"].n_latents)
    lat /= np.sqrt(np.mean(lat**2))
    x = modem.modulate(lat, "A", callsign="N0CALL")
    r = modem.demodulate(x)
    assert r.callsign == "N0CALL"
    assert r.beacon is not None
    assert r.beacon.frame_index >= 0


def test_modem_beacon_survives_awgn():
    modem = Modem()
    lat = np.random.default_rng(4).normal(size=MODES["A"].n_latents)
    lat /= np.sqrt(np.mean(lat**2))
    x = modem.modulate(lat, "A", callsign="W1AW")
    y = hfchannel.apply_channel(x, snr_db=10.0)
    r = modem.demodulate(y)
    assert r.callsign == "W1AW"


def test_modem_default_callsign_is_blank_but_resync_still_works():
    modem = Modem()
    lat = np.random.default_rng(5).normal(size=MODES["A"].n_latents)
    lat /= np.sqrt(np.mean(lat**2))
    x = modem.modulate(lat, "A")  # no callsign passed
    r = modem.demodulate(x)
    assert r.callsign == ""
    assert r.beacon is not None  # sync word + frame counter still decode
