"""The reception aggregation server."""

import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
# TestClient drives the app in-process and needs an HTTP client of its
# own; without it the import raises RuntimeError rather than ImportError,
# which importorskip would not catch.
pytest.importorskip("httpx2")
from fastapi.testclient import TestClient  # noqa: E402

from sstvae import hfchannel  # noqa: E402
from sstvae.config import MODES  # noqa: E402
from sstvae.modem.modem import BlindDemodResult, DemodResult, Modem  # noqa: E402
from sstvae.rx import receptionfile  # noqa: E402
from sstvae.server.app import create_app  # noqa: E402
from sstvae.server.config import ServerConfig  # noqa: E402
from sstvae.server.db import Database  # noqa: E402

from conftest import latent_snr_db, unit_latents  # noqa: E402

MODE_A = MODES["A"]
MODE_C = MODES["C"]
BASE_UTC = 1_770_000_000.0


class _StubCodec:
    """Stands in for the ONNX decoder: the server's job is to combine,
    and the decode itself is covered by the codec-marked test below."""

    def __init__(self):
        self.calls = []

    def decode(self, latents, weights):
        from PIL import Image

        self.calls.append((np.asarray(latents), np.asarray(weights)))
        return Image.new("RGB", (640, 480), (int(np.clip(len(self.calls) * 10, 0, 255)), 0, 0))


@pytest.fixture
def server(tmp_path):
    config = ServerConfig(data_dir=tmp_path / "data")
    codec = _StubCodec()
    db = Database(config.db_path)
    app = create_app(config, codec=codec, db=db)
    client = TestClient(app)
    client.codec = codec
    client.db = db
    client.config = config
    yield client
    db.close()


def _demod(seed=0, snr_db=5.0, frames=200, callsign="BALLOON1", mode=MODE_A):
    rng = np.random.default_rng(seed)
    n = mode.n_latents
    return DemodResult(
        latents=rng.uniform(-3, 3, n),
        weights=rng.uniform(0, 1, n),
        mode=mode,
        freq_offset=0.0,
        sync_metric=0.9,
        frames_received=frames,
        beacon=None,
        callsign=callsign,
        preamble_start=0,
        snr_db=snr_db,
    )


def _blind(seed=0, snr_db=5.0, callsign="BALLOON1"):
    rng = np.random.default_rng(seed)
    n = MODE_C.n_latents
    return BlindDemodResult(
        latents=rng.uniform(-3, 3, n),
        weights=rng.uniform(0, 1, n),
        freq_offset=0.0,
        beacon=None,
        callsign=callsign,
        frame_offset=0,
        n_frames=220,
        frame0_start=0,
        snr_db=snr_db,
    )


def _upload(client, key, result, utc_start=BASE_UTC, station="STA1", dial=14_233_000.0):
    body = receptionfile.to_bytes(
        result, utc_start=utc_start, station_callsign=station, dial_freq_hz=dial
    )
    return client.post(
        "/api/v1/receptions",
        content=body,
        headers={"Authorization": f"Bearer {key}"},
    )


# -- auth ------------------------------------------------------------

def test_health_needs_no_key(server):
    r = server.get("/healthz")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_an_upload_without_a_key_is_refused(server):
    r = server.post("/api/v1/receptions", content=b"whatever")
    assert r.status_code == 401


def test_an_unknown_key_is_refused(server):
    r = _upload(server, "not-a-real-key", _demod())
    assert r.status_code == 401


def test_a_revoked_key_stops_working(server):
    key = server.db.issue_key("STA1")
    assert _upload(server, key, _demod()).status_code == 200
    server.db.revoke("STA1")
    assert _upload(server, key, _demod()).status_code == 401


def test_a_key_cannot_upload_in_another_stations_name(server):
    """The key names the station. Otherwise any key would let its holder
    file receptions as anyone."""
    key = server.db.issue_key("STA1")
    r = _upload(server, key, _demod(), station="STA2")
    assert r.status_code == 400
    assert "STA1" in r.json()["detail"]


# -- payload validation ----------------------------------------------

def test_garbage_is_rejected_as_a_bad_request(server):
    key = server.db.issue_key("STA1")
    r = server.post(
        "/api/v1/receptions",
        content=b"not an npz at all",
        headers={"Authorization": f"Bearer {key}"},
    )
    assert r.status_code == 400


def test_a_reception_from_the_future_is_refused(server):
    """A transmission that has not finished yet cannot already have been
    decoded and uploaded: the station's clock is fast."""
    key = server.db.issue_key("STA1")
    r = _upload(server, key, _demod(), utc_start=time.time() + 3600)
    assert r.status_code == 400
    assert "clock" in r.json()["detail"]


def test_a_late_upload_is_still_accepted(server):
    """The other direction is indistinguishable from a queued retry, so
    it must never be refused."""
    key = server.db.issue_key("STA1")
    r = _upload(server, key, _demod(), utc_start=time.time() - 86_400)
    assert r.status_code == 200


# -- transmission matching -------------------------------------------

def test_two_stations_at_the_same_time_are_one_transmission(server):
    k1 = server.db.issue_key("STA1")
    k2 = server.db.issue_key("STA2")
    a = _upload(server, k1, _demod(seed=1), utc_start=BASE_UTC, station="STA1")
    b = _upload(server, k2, _demod(seed=2), utc_start=BASE_UTC + 0.4, station="STA2")
    assert a.json()["transmission_id"] == b.json()["transmission_id"]
    assert b.json()["n_receptions"] == 2


def test_frequency_never_splits_a_transmission(server):
    """The decision this server is built on: dial frequency is recorded,
    never matched on. A station with no rig control, or an operator who
    typed the wrong band, must still combine -- splitting the bucket
    would silently forfeit exactly the gain being collected."""
    k1 = server.db.issue_key("STA1")
    k2 = server.db.issue_key("STA2")
    a = _upload(server, k1, _demod(seed=1), station="STA1", dial=14_233_000.0)
    b = _upload(server, k2, _demod(seed=2), station="STA2", dial=21_000_000.0)
    assert a.json()["transmission_id"] == b.json()["transmission_id"]
    assert b.json()["n_receptions"] == 2


def test_a_missing_dial_frequency_buckets_normally(server):
    k1 = server.db.issue_key("STA1")
    k2 = server.db.issue_key("STA2")
    a = _upload(server, k1, _demod(seed=1), station="STA1", dial=None)
    b = _upload(server, k2, _demod(seed=2), station="STA2", dial=14_233_000.0)
    assert a.json()["transmission_id"] == b.json()["transmission_id"]


def test_starts_far_apart_are_different_transmissions(server):
    k1 = server.db.issue_key("STA1")
    k2 = server.db.issue_key("STA2")
    a = _upload(server, k1, _demod(seed=1), utc_start=BASE_UTC, station="STA1")
    b = _upload(server, k2, _demod(seed=2), utc_start=BASE_UTC + 600, station="STA2")
    assert a.json()["transmission_id"] != b.json()["transmission_id"]


def test_a_different_transmitter_is_a_different_transmission(server):
    k1 = server.db.issue_key("STA1")
    k2 = server.db.issue_key("STA2")
    a = _upload(server, k1, _demod(seed=1, callsign="BALLOON1"), station="STA1")
    b = _upload(server, k2, _demod(seed=2, callsign="BALLOON2"), station="STA2")
    assert a.json()["transmission_id"] != b.json()["transmission_id"]


def test_a_station_uploading_twice_replaces_rather_than_doubles(server):
    """Combining two copies of one reception would break the
    independence the weighting assumes and claim confidence the signal
    does not have."""
    key = server.db.issue_key("STA1")
    _upload(server, key, _demod(seed=1))
    r = _upload(server, key, _demod(seed=1))
    assert r.json()["n_receptions"] == 1


# -- combining --------------------------------------------------------

def test_combining_beats_either_station_alone(server):
    """The whole point, end to end through the HTTP path: two real
    degraded receptions of one transmission, combined server-side,
    recovering the latents better than either station did."""
    modem = Modem()
    lat = unit_latents("A")
    x = modem.modulate(lat, "A")
    r1 = modem.demodulate(hfchannel.apply_channel(x, snr_db=6.0, seed=10))
    r2 = modem.demodulate(hfchannel.apply_channel(x, snr_db=6.0, seed=20))
    s1 = latent_snr_db(lat, r1.latents, r1.weights)
    s2 = latent_snr_db(lat, r2.latents, r2.weights)

    k1 = server.db.issue_key("STA1")
    k2 = server.db.issue_key("STA2")
    _upload(server, k1, r1, station="STA1")
    reply = _upload(server, k2, r2, station="STA2").json()

    assert reply["n_receptions"] == 2
    assert reply["combined_snr_db"] > max(r1.snr_db, r2.snr_db)

    # What the server actually handed the decoder.
    latents, weights = server.codec.calls[-1]
    combined_snr = latent_snr_db(lat, latents[: MODE_A.n_latents],
                                 weights[: MODE_A.n_latents])
    assert combined_snr > max(s1, s2) + 1.5, (
        f"combined {combined_snr:.2f} dB against branches {s1:.2f}/{s2:.2f} -- "
        "the server combined nothing useful"
    )


def test_contributions_are_recorded_per_station(server):
    k1 = server.db.issue_key("STA1")
    k2 = server.db.issue_key("STA2")
    _upload(server, k1, _demod(seed=1, snr_db=3.0), station="STA1")
    reply = _upload(server, k2, _demod(seed=2, snr_db=9.0), station="STA2").json()

    tx = server.get(f"/api/v1/transmissions/{reply['transmission_id']}").json()
    shares = {s["callsign"]: s["contrib_frac"] for s in tx["stations"]}
    assert set(shares) == {"STA1", "STA2"}
    assert all(v is not None for v in shares.values())
    assert sum(shares.values()) == pytest.approx(1.0, abs=0.02)
    assert shares["STA2"] > shares["STA1"], "the stronger station should carry more"


def test_a_blind_only_transmission_still_makes_a_picture(server):
    """No header anywhere, so no mode: the picture is a full mode-C
    decode, exactly as the live blind path produces."""
    k1 = server.db.issue_key("STA1")
    k2 = server.db.issue_key("STA2")
    _upload(server, k1, _blind(seed=1), station="STA1")
    reply = _upload(server, k2, _blind(seed=2), station="STA2").json()
    assert reply["mode"] is None
    assert reply["n_receptions"] == 2
    tx = server.get(f"/api/v1/transmissions/{reply['transmission_id']}").json()
    assert tx["picture_url"] is not None
    assert server.get(tx["picture_url"].split("?")[0]).status_code == 200


def test_a_header_and_a_blind_station_combine(server):
    k1 = server.db.issue_key("STA1")
    k2 = server.db.issue_key("STA2")
    _upload(server, k1, _demod(seed=1), station="STA1")
    reply = _upload(server, k2, _blind(seed=2), station="STA2").json()
    assert reply["n_receptions"] == 2
    assert reply["mode"] == "A", "the header branch is what knows the mode"


def test_one_station_disagreeing_about_the_mode_does_not_lose_the_picture(server):
    """A mode mismatch is fatal to the combiner by design. The server
    cannot ask, so the majority wins and the odd one out is dropped --
    rather than every station losing their picture to one bad report."""
    k = {c: server.db.issue_key(c) for c in ("STA1", "STA2", "STA3")}

    _upload(server, k["STA1"], _demod(seed=1, mode=MODE_A), station="STA1")
    _upload(server, k["STA2"], _demod(seed=2, mode=MODE_A), station="STA2")
    reply = _upload(
        server, k["STA3"], _demod(seed=3, mode=MODES["B"]), station="STA3"
    ).json()

    assert reply["mode"] == "A"
    assert reply["n_receptions"] == 2, "the disagreeing station should sit out"
    assert reply["n_stations"] == 3, "it is still recorded as having uploaded"
    assert reply["note"] and "mode B" in reply["note"], (
        "the station got a 200 and no way to know it contributed nothing"
    )


def test_a_later_upload_can_reverse_the_mode_vote(server):
    """The vote is taken when the picture is made, not at the door.

    Nothing is rejected on arrival, and the whole combine re-runs on
    every upload, so the count is over everything received so far. A
    station outvoted at one moment is counted again as soon as later
    arrivals agree with it -- which is what lets the server decide with
    no quorum and nothing to undo.
    """
    k = {c: server.db.issue_key(c) for c in ("STA1", "STA2", "STA3")}

    first = _upload(
        server, k["STA1"], _demod(seed=1, mode=MODE_A, snr_db=9.0), station="STA1"
    ).json()
    tx = first["transmission_id"]
    assert first["mode"] == "A"

    # A lone dissenter loses to the incumbent.
    second = _upload(
        server, k["STA2"], _demod(seed=2, mode=MODES["B"], snr_db=3.0), station="STA2"
    ).json()
    assert second["mode"] == "A"
    assert second["note"], "the outvoted station should be told it is not counted"

    # Another station agrees with the dissenter, and the picture changes.
    third = _upload(
        server, k["STA3"], _demod(seed=3, mode=MODES["B"], snr_db=3.0), station="STA3"
    ).json()
    assert third["mode"] == "B", "the later uploads should have carried the vote"
    assert third["n_receptions"] == 2
    assert third["note"] is None, "STA3 is in the picture, so it has nothing to explain"

    stations = {
        s["callsign"]: s
        for s in server.get(f"/api/v1/transmissions/{tx}").json()["stations"]
    }
    assert stations["STA2"]["excluded_reason"] is None, (
        "a station that was outvoted and then vindicated is still shown as excluded"
    )
    assert stations["STA2"]["contrib_frac"] is not None


def test_a_station_that_loses_the_vote_stops_claiming_a_share(server):
    """The contribution of every reception is rewritten on each combine,
    not only that of the ones which won.

    Otherwise a station counted a moment ago keeps whatever share it was
    last told, and goes on claiming to have supplied a picture it
    contributed nothing to -- with the shares summing past 1."""
    k = {c: server.db.issue_key(c) for c in ("STA1", "STA2", "STA3")}

    first = _upload(
        server, k["STA1"], _demod(seed=1, mode=MODE_A, snr_db=9.0), station="STA1"
    ).json()
    tx = first["transmission_id"]
    assert (
        server.get(f"/api/v1/transmissions/{tx}").json()["stations"][0]["contrib_frac"]
        == 1.0
    )

    _upload(server, k["STA2"], _demod(seed=2, mode=MODES["B"], snr_db=3.0), station="STA2")
    _upload(server, k["STA3"], _demod(seed=3, mode=MODES["B"], snr_db=3.0), station="STA3")

    stations = {
        s["callsign"]: s
        for s in server.get(f"/api/v1/transmissions/{tx}").json()["stations"]
    }
    assert stations["STA1"]["contrib_frac"] is None, (
        "an outvoted station still claims a share of a picture it is not in"
    )
    assert stations["STA1"]["excluded_reason"], "and no reason is given for its absence"
    counted = [s["contrib_frac"] for s in stations.values() if s["contrib_frac"] is not None]
    assert sum(counted) == pytest.approx(1.0, abs=0.02), (
        f"shares over the counted stations should sum to 1, got {sum(counted)}"
    )


# -- read-only surface ------------------------------------------------

def test_the_transmission_list_reports_the_stations(server):
    key = server.db.issue_key("STA1")
    _upload(server, key, _demod())
    listed = server.get("/api/v1/transmissions").json()["transmissions"]
    assert len(listed) == 1
    assert listed[0]["callsign"] == "BALLOON1"
    assert [s["callsign"] for s in listed[0]["stations"]] == ["STA1"]


def test_the_picture_url_changes_when_a_station_improves_it(server):
    """A browser that cached the one-station picture must not go on
    showing it after a second station arrives."""
    k1 = server.db.issue_key("STA1")
    k2 = server.db.issue_key("STA2")
    first = _upload(server, k1, _demod(seed=1), station="STA1").json()
    url_one = server.get(f"/api/v1/transmissions/{first['transmission_id']}").json()[
        "picture_url"
    ]
    time.sleep(0.01)
    _upload(server, k2, _demod(seed=2), station="STA2")
    url_two = server.get(f"/api/v1/transmissions/{first['transmission_id']}").json()[
        "picture_url"
    ]
    assert url_one != url_two


def test_an_unknown_transmission_is_a_404(server):
    assert server.get("/api/v1/transmissions/999").status_code == 404
    assert server.get("/pictures/999.png").status_code == 404


def test_the_gallery_is_served(server):
    r = server.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "transmissions" in r.text


def test_the_reply_carries_the_server_clock(server):
    """What lets a station notice its own clock is wrong -- otherwise it
    uploads happily and never combines with anyone."""
    key = server.db.issue_key("STA1")
    reply = _upload(server, key, _demod()).json()
    assert abs(reply["server_time"] - time.time()) < 30


# -- against the real decoder ----------------------------------------

@pytest.mark.codec
def test_the_combined_picture_beats_either_stations_own(tmp_path):
    """The stub codec above proves the server combines the right
    arrays; this proves the result is a better *picture*, through the
    decoder that actually ships.

    Skipped without the downloaded artifacts, like every codec test --
    CI prefetches and sets SSTVAE_REQUIRE_CODEC=1, so a skip there is a
    failure.
    """
    import os

    from PIL import Image

    from sstvae.codec import load_codec, pad_to_full, reconstruct
    from sstvae.images import IMG_H, IMG_W

    def _unavailable(why):
        if os.environ.get("SSTVAE_REQUIRE_CODEC"):
            pytest.fail(f"SSTVAE_REQUIRE_CODEC is set but the codec is unusable: {why}")
        pytest.skip(f"codec unavailable: {why}")

    # The codec loads its parts lazily, so a missing runtime or an
    # uncached artifact surfaces at first *use* rather than at load.
    # Force it here, where it can still be a skip.
    try:
        codec = load_codec(None)
        probe = np.zeros(MODE_C.n_latents)
        reconstruct(codec, probe, probe)
    except (Exception, SystemExit) as exc:
        # SystemExit is deliberate, not a stray: checkpoint.py raises it
        # rather than an ImportError so a missing artifact reads as
        # advice instead of a traceback. It descends from BaseException,
        # so `except Exception` would let it past and turn an
        # unavailable codec into a failing test rather than a skipped
        # one.
        _unavailable(exc)

    modem = Modem()
    lat = unit_latents("A")
    x = modem.modulate(lat, "A")
    r1 = modem.demodulate(hfchannel.apply_channel(x, snr_db=4.0, seed=31))
    r2 = modem.demodulate(hfchannel.apply_channel(x, snr_db=4.0, seed=41))

    config = ServerConfig(data_dir=tmp_path / "data")
    db = Database(config.db_path)
    try:
        client = TestClient(create_app(config, codec=codec, db=db))
        k1 = db.issue_key("STA1")
        k2 = db.issue_key("STA2")
        _upload(client, k1, r1, station="STA1")
        reply = _upload(client, k2, r2, station="STA2").json()
        assert reply["n_receptions"] == 2

        truth = reconstruct(codec, pad_to_full(lat), np.ones(len(pad_to_full(lat))))
        singles = [
            reconstruct(codec, pad_to_full(r.latents), pad_to_full(r.weights))
            for r in (r1, r2)
        ]
        combined = Image.open(
            config.pictures_dir / f"{reply['transmission_id']}.png"
        )
        assert combined.size == (IMG_W, IMG_H)

        def psnr(img):
            a = np.asarray(img, dtype=np.float64)
            b = np.asarray(truth, dtype=np.float64)
            mse = np.mean((a - b) ** 2)
            return 10 * np.log10(255.0**2 / mse) if mse > 0 else float("inf")

        assert psnr(combined) > max(psnr(s) for s in singles), (
            "the combined picture is no better than the better single station's"
        )
    finally:
        db.close()
