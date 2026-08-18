"""UploadSink: the station half of reception aggregation."""

import io
import json
import urllib.error

import numpy as np
import pytest

from sstvae.config import MODES
from sstvae.modem.modem import DemodResult
from sstvae.rx import receptionfile
from sstvae.rx.engine import Reception
from sstvae.upload import UploadSink

MODE_A = MODES["A"]


class _InnerSink:
    def __init__(self):
        self.seen = []

    def on_reception(self, rec):
        self.seen.append(rec)
        return "/tmp/saved.png"


class _FakeServer:
    """Stands in for urlopen. Records requests, replies as told."""

    def __init__(self, fail_times=0, server_time=None):
        self.requests = []
        self.fail_times = fail_times
        self.server_time = server_time

    def __call__(self, req, timeout=None):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise urllib.error.URLError("connection refused")
        self.requests.append(req)
        body = {
            "transmission_id": 7,
            "n_receptions": len(self.requests),
            "combined_snr_db": 9.5,
        }
        if self.server_time is not None:
            body["server_time"] = self.server_time
        return _FakeResponse(json.dumps(body).encode())


class _FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _reception(utc_start=1_770_000_000.5, seed=0):
    rng = np.random.default_rng(seed)
    n = MODE_A.n_latents
    result = DemodResult(
        latents=rng.uniform(-3, 3, n),
        weights=rng.uniform(0, 1, n),
        mode=MODE_A,
        freq_offset=0.0,
        sync_metric=0.9,
        frames_received=180,
        beacon=None,
        callsign="BALLOON1",
        preamble_start=0,
        snr_db=4.5,
    )
    return Reception(
        image=None,
        mode_name="A",
        callsign="BALLOON1",
        snr_db=4.5,
        frames_received=180,
        n_frames_expected=MODE_A.n_frames,
        result=result,
        utc_start=utc_start,
    )


def _sink(tmp_path, monkeypatch, server, **kw):
    monkeypatch.setattr("urllib.request.urlopen", server)
    inner = _InnerSink()
    kw.setdefault("station_callsign", "n0call")
    kw.setdefault("dial_freq_hz", 14_233_000.0)
    return inner, UploadSink(
        inner,
        url="https://example.invalid/",
        key="secret-key",
        queue_dir=tmp_path / "queue",
        verbose=False,
        **kw,
    )


def test_the_local_save_happens_first_and_is_what_the_loop_sees(tmp_path, monkeypatch):
    inner, sink = _sink(tmp_path, monkeypatch, _FakeServer())
    rec = _reception()
    assert sink.on_reception(rec) == "/tmp/saved.png"
    assert inner.seen == [rec]


def test_the_uploaded_body_is_the_reception(tmp_path, monkeypatch):
    server = _FakeServer()
    _, sink = _sink(tmp_path, monkeypatch, server)
    rec = _reception()
    sink.on_reception(rec)

    assert len(server.requests) == 1
    req = server.requests[0]
    assert req.get_header("Authorization") == "Bearer secret-key"
    assert req.get_header("Content-type") == "application/octet-stream"
    assert req.full_url.endswith("/api/v1/receptions")

    got, meta = receptionfile.read(req.data)
    assert np.array_equal(got.latents, rec.result.latents.astype(np.float32))
    assert meta["station_callsign"] == "N0CALL"
    assert meta["dial_freq_hz"] == 14_233_000.0
    assert meta["utc_start_epoch"] == pytest.approx(rec.utc_start)


def test_a_failed_upload_keeps_the_reception_and_retries_later(tmp_path, monkeypatch):
    """The queue is the whole point: a station that hears something
    while its link is down must not lose it."""
    server = _FakeServer(fail_times=99)
    _, sink = _sink(tmp_path, monkeypatch, server)

    sink.on_reception(_reception(utc_start=1_770_000_000.0, seed=1))
    sink.on_reception(_reception(utc_start=1_770_000_100.0, seed=2))
    queued = sorted((tmp_path / "queue").glob("*.npz"))
    assert len(queued) == 2, "a failed upload dropped the reception on the floor"
    assert server.requests == []

    # The link comes back.
    server.fail_times = 0
    assert sink.flush() == 2
    assert not list((tmp_path / "queue").glob("*.npz"))
    # Oldest first, so the server sees them in the order they happened.
    sent = [receptionfile.read(r.data)[1]["utc_start_epoch"] for r in server.requests]
    assert sent == sorted(sent)


def test_the_queue_survives_a_restart(tmp_path, monkeypatch):
    """Anything left over goes out when the next session starts, not
    when the next reception happens to arrive."""
    server = _FakeServer(fail_times=99)
    _, sink = _sink(tmp_path, monkeypatch, server)
    sink.on_reception(_reception())
    assert len(list((tmp_path / "queue").glob("*.npz"))) == 1

    working = _FakeServer()
    _, fresh = _sink(tmp_path, monkeypatch, working)
    assert len(working.requests) == 1, "a queued reception waited for a new one"
    assert not list((tmp_path / "queue").glob("*.npz"))


def test_a_reception_with_no_latents_is_saved_but_not_uploaded(tmp_path, monkeypatch):
    """An older sink, or a path that never kept the demodulator output:
    the picture is still saved and nothing is sent."""
    server = _FakeServer()
    inner, sink = _sink(tmp_path, monkeypatch, server)
    rec = _reception()
    rec.result = None
    assert sink.on_reception(rec) == "/tmp/saved.png"
    assert inner.seen == [rec]
    assert server.requests == []


def test_an_http_rejection_does_not_lose_the_reception(tmp_path, monkeypatch):
    def refuse(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b"bad key"))

    _, sink = _sink(tmp_path, monkeypatch, refuse)
    sink.on_reception(_reception())
    assert len(list((tmp_path / "queue").glob("*.npz"))) == 1


def test_a_skewed_clock_is_reported_to_the_operator(tmp_path, monkeypatch, capsys):
    """A wrong clock uploads successfully and then silently fails to
    combine, so the only place it can be caught is here."""
    server = _FakeServer(server_time=1_000.0)  # decades off
    monkeypatch.setattr("urllib.request.urlopen", server)
    sink = UploadSink(
        _InnerSink(),
        url="https://example.invalid/",
        key="k",
        station_callsign="N0CALL",
        queue_dir=tmp_path / "queue",
        verbose=True,
    )
    sink.on_reception(_reception())
    out = capsys.readouterr().out
    assert "clock" in out.lower() and "NTP" in out


def test_a_good_clock_says_nothing(tmp_path, monkeypatch, capsys):
    import time

    server = _FakeServer(server_time=time.time())
    monkeypatch.setattr("urllib.request.urlopen", server)
    sink = UploadSink(
        _InnerSink(),
        url="https://example.invalid/",
        key="k",
        station_callsign="N0CALL",
        queue_dir=tmp_path / "queue",
        verbose=True,
    )
    sink.on_reception(_reception())
    assert "clock" not in capsys.readouterr().out.lower()
