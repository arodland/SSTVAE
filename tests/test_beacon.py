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


# --- multi-repetition combining ----------------------------------------------


def test_combining_decodes_where_no_single_repetition_can():
    """The core claim of the multi-repetition fallback: chip noise too
    heavy for any single ~5 s repetition's own Golay+CRC decode to
    survive can still be resolved once enough repetitions (here, mode
    C's ~18) are combined -- see beacon.py's module docstring on why
    (invariant chunks summed by sign, the counter and CRC-mixed chunks
    resolved by a joint search across repetitions, not per-repetition
    voting). Regression for a real, measured gain: this exact scenario
    (mode C, mpp fading, SNR -6..0 dB) went from 12-88% end-to-end
    success to 100% once every case where the pilot itself locks also
    got a beacon decode.

    Ported to native/ too -- `beacon._decode_payload` below is always
    the Python reference regardless of --native (it isn't in
    NATIVE_SUBSTITUTIONS, and doesn't need to be: it's only this test's
    own diagnostic for "did combining actually have to do anything",
    not the thing under test), but `beacon.decode` itself resolves to
    the native binding under --native.
    """
    n_frames = MODES["C"].n_frames
    chips = beacon.chip_stream(0, n_frames, "TEST")
    rng = np.random.default_rng(6)
    # Calibrated so most *individual* repetitions fail on their own --
    # asserted below -- while the combined decode still succeeds.
    noisy = chips + rng.normal(scale=1.3, size=chips.shape)

    n_solo_ok = 0
    for off in beacon.find_sync(noisy, threshold=0.3, max_candidates=30):
        end = off + beacon.SYNC_LEN + beacon.CODED_LEN
        if end > len(noisy):
            continue
        if beacon._decode_payload(noisy[off + beacon.SYNC_LEN : end]) is not None:
            n_solo_ok += 1
    assert n_solo_ok <= 2, (
        f"{n_solo_ok} individual repetitions already decoded on their own -- "
        "noise scale needs raising so this test actually exercises combining"
    )

    r = beacon.decode(noisy)
    assert r is not None, "combining should have rescued this from noise too heavy for any single repetition"
    assert r.frame_index == r.chip_offset // CHIPS_PER_FRAME
    assert r.callsign == "TEST"


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
