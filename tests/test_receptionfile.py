"""The on-disk / on-the-wire reception payload."""

import io
import json

import numpy as np
import pytest

from sstvae import __version__
from sstvae.config import MODES, PROTOCOL_VERSION
from sstvae.modem.modem import BlindDemodResult, DemodResult
from sstvae.rx import receptionfile
from sstvae.rx.receptionfile import ReceptionFileError

MODE_A = MODES["A"]
MODE_C = MODES["C"]


def _header_result(seed=0):
    rng = np.random.default_rng(seed)
    n = MODE_A.n_latents
    return DemodResult(
        latents=rng.uniform(-10, 10, n),
        weights=rng.uniform(0, 1, n),
        mode=MODE_A,
        freq_offset=-3.25,
        sync_metric=0.87,
        frames_received=137,
        beacon=None,
        callsign="BALLOON1",
        preamble_start=4242,
        snr_db=6.5,
    )


def _blind_result(seed=1):
    rng = np.random.default_rng(seed)
    n = MODE_C.n_latents
    return BlindDemodResult(
        latents=rng.uniform(-10, 10, n),
        weights=rng.uniform(0, 1, n),
        freq_offset=11.5,
        beacon=None,
        callsign="BALLOON1",
        frame_offset=88,
        n_frames=250,
        frame0_start=-9000,
        snr_db=2.25,
    )


def _roundtrip(result, **kw):
    kw.setdefault("utc_start", 1_770_000_000.125)
    kw.setdefault("station_callsign", "n0call")
    kw.setdefault("dial_freq_hz", 14_233_000.0)
    return receptionfile.read(receptionfile.to_bytes(result, **kw))


def test_a_header_reception_round_trips():
    sent = _header_result()
    got, meta = _roundtrip(sent)

    assert isinstance(got, DemodResult)
    # float32 is the storage precision; the values must survive it exactly.
    assert np.array_equal(got.latents, sent.latents.astype(np.float32))
    assert np.array_equal(got.weights, sent.weights.astype(np.float32))
    assert got.mode is MODE_A
    assert got.frames_received == 137
    assert got.callsign == "BALLOON1"
    assert got.snr_db == pytest.approx(6.5)
    assert got.freq_offset == pytest.approx(-3.25)

    assert meta["path"] == "header"
    assert meta["station_callsign"] == "N0CALL"  # normalized like the rest of the app
    assert meta["dial_freq_hz"] == 14_233_000.0
    assert meta["utc_start"] == "2026-02-02T02:40:00.125Z"
    assert meta["utc_start_epoch"] == pytest.approx(1_770_000_000.125)
    assert meta["protocol_version"] == PROTOCOL_VERSION
    assert meta["software"] == f"sstvae {__version__}"


def test_a_blind_reception_round_trips():
    sent = _blind_result()
    got, meta = _roundtrip(sent)

    assert isinstance(got, BlindDemodResult)
    assert np.array_equal(got.latents, sent.latents.astype(np.float32))
    assert np.array_equal(got.weights, sent.weights.astype(np.float32))
    assert got.frame_offset == 88
    assert got.n_frames == 250
    assert len(got.latents) == MODE_C.n_latents
    assert meta["path"] == "blind"
    # A blind reception has no header, so no mode at all.
    assert meta.get("mode_name") is None


def test_erasures_survive_exactly():
    """Weight 0 means "never received", and the combiner and decoder
    both depend on it being exactly 0 rather than nearly."""
    sent = _header_result()
    sent.weights[::3] = 0.0
    sent.latents[::3] = 0.0
    got, _ = _roundtrip(sent)
    assert np.all(got.weights[::3] == 0.0)
    assert np.all(got.latents[::3] == 0.0)


def test_a_dial_frequency_is_optional():
    """A headless skimmer may have no rig control at all, so the field
    that is least reliable is also never required."""
    got, meta = _roundtrip(_header_result(), dial_freq_hz=None)
    assert meta["dial_freq_hz"] is None
    assert isinstance(got, DemodResult)


def _tamper(result, **changes):
    """Re-emit a payload with its metadata edited, as a corrupt or
    mismatched sender would."""
    raw = receptionfile.to_bytes(
        result,
        utc_start=1_770_000_000.0,
        station_callsign="N0CALL",
        dial_freq_hz=None,
    )
    with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    meta = json.loads(str(arrays["meta"]))
    meta.update(changes)
    arrays["meta"] = np.array(json.dumps(meta))
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


@pytest.mark.parametrize(
    "changes, expect",
    [
        ({"format": "something-else"}, "not a sstvae-reception"),
        ({"format_version": 99}, "format_version"),
        ({"protocol_version": PROTOCOL_VERSION + 1}, "protocol_version"),
        ({"mode_name": "Z"}, "unknown mode"),
        ({"mode_name": "C"}, "wants"),  # mode C's size against mode A's arrays
        ({"path": "sideways"}, "unknown path"),
        ({"utc_start": "not a time"}, "bad utc_start"),
    ],
)
def test_a_payload_that_does_not_describe_itself_is_rejected(changes, expect):
    with pytest.raises(ReceptionFileError, match=expect):
        receptionfile.read(_tamper(_header_result(), **changes))


def test_garbage_is_rejected_rather_than_raising_something_unhelpful():
    with pytest.raises(ReceptionFileError, match="unreadable"):
        receptionfile.read(b"this is not an npz")


def test_a_picture_is_not_a_reception():
    with pytest.raises(ReceptionFileError, match="DemodResult"):
        receptionfile.to_bytes(
            object(), utc_start=0.0, station_callsign="N0CALL"
        )


def test_write_produces_a_readable_file(tmp_path):
    p = tmp_path / "rx.npz"
    receptionfile.write(
        p, _header_result(), utc_start=1_770_000_000.0, station_callsign="N0CALL"
    )
    got, meta = receptionfile.read(p.read_bytes())
    assert got.mode is MODE_A
    assert meta["station_callsign"] == "N0CALL"
