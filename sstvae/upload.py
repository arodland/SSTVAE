"""Send finished receptions to an aggregation server.

Many stations hear one transmission, each of them partially and at a
different quality; a server that has all of their latents can combine
them into a better picture than any one station decoded
(`sstvae.modem.diversity`, and docs/reception-aggregation.md). This is
the sending half: a sink that wraps another sink.

Two properties are load-bearing.

**A failed upload never loses a reception.** The inner sink runs first
and its return value is what the decode loop sees, so saving locally is
never at risk from anything network-shaped. The payload is then spooled
to disk, and only deleted once the server has acknowledged it -- so a
reception heard while the link is down is still on disk twice, and goes
out with the next one.

**Nothing here reaches into the decode loop.** `on_reception` is called
after the reception has been handed off, so the worst a slow server can
do is delay the next poll; the ring buffer goes on filling behind it
with ~130 s of headroom. That is why this is synchronous with a
timeout rather than another thread with another queue to reason about.

The HTTP client is `urllib` from the standard library. The request is
one POST with a bearer token and a binary body -- `requests` would be
the package's first HTTP dependency and would buy nothing at that size.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .rx.engine import Reception
from .rx.receptionfile import to_bytes

__all__ = ["UploadSink", "UploadError", "CLOCK_WARN_S"]

# How far the station's clock may sit from the server's before we say
# so. The server groups receptions of one transmission by their
# reported start time, so a clock this far out stops a station's
# contribution from being recognized as the same transmission at all --
# it would upload successfully and silently never combine.
CLOCK_WARN_S = 1.0


class UploadError(RuntimeError):
    """An upload that did not reach the server, or that it refused."""


class UploadSink:
    """Wrap a sink so every reception it handles is also uploaded.

    `inner` is called first and its result is passed straight back to
    the decode loop, so this is transparent to the loop and to the GUI.
    """

    def __init__(
        self,
        inner,
        url: str,
        key: str,
        station_callsign: str,
        dial_freq_hz: float | None = None,
        queue_dir: str | os.PathLike = "upload-queue",
        timeout_s: float = 15.0,
        verbose: bool = True,
    ):
        self.inner = inner
        self.url = url.rstrip("/") + "/api/v1/receptions"
        self.key = key
        self.station_callsign = (station_callsign or "").strip().upper()
        self.dial_freq_hz = dial_freq_hz
        self.queue_dir = Path(queue_dir)
        self.timeout_s = timeout_s
        self.verbose = verbose
        # Anything left over from a previous run goes out now, before
        # this session adds to it.
        self.flush()

    # -- the sink protocol ------------------------------------------

    def on_reception(self, rec: Reception) -> str | None:
        saved = self.inner.on_reception(rec)
        if rec.result is None or rec.utc_start is None:
            # A sink upstream of this one produced a picture with no
            # demodulator output behind it: there is nothing to upload,
            # and that is not a failure of the reception.
            self._say("not uploading: this reception carries no latents")
            return saved
        try:
            self._spool(rec)
        except OSError as exc:
            self._say(f"could not spool for upload: {exc}")
        self.flush()
        return saved

    # -- the queue ---------------------------------------------------

    def _spool(self, rec: Reception) -> Path:
        """Write the payload to the queue, atomically."""
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        payload = to_bytes(
            rec.result,
            utc_start=rec.utc_start,
            station_callsign=self.station_callsign,
            dial_freq_hz=self.dial_freq_hz,
        )
        stem = f"{rec.utc_start:.3f}_{rec.callsign or 'unknown'}"
        final = self.queue_dir / f"{stem}.npz"
        part = final.with_suffix(".part")
        part.write_bytes(payload)
        part.replace(final)
        return final

    def flush(self) -> int:
        """Send everything queued, oldest first. Returns how many went.

        Stops at the first failure rather than working through the rest:
        whatever stopped one upload will almost certainly stop the next,
        and the queue's order is worth keeping.
        """
        if not self.queue_dir.is_dir():
            return 0
        sent = 0
        for path in sorted(self.queue_dir.glob("*.npz")):
            try:
                reply = self._post(path.read_bytes())
            except (UploadError, OSError) as exc:
                self._say(f"upload deferred ({exc}); {path.name} stays queued")
                break
            path.unlink(missing_ok=True)
            sent += 1
            self._report(reply)
        return sent

    # -- the wire ----------------------------------------------------

    def _post(self, payload: bytes) -> dict:
        req = urllib.request.Request(
            self.url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/octet-stream",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                if resp.status != 200:
                    raise UploadError(f"HTTP {resp.status}")
                body = resp.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass
            raise UploadError(f"HTTP {exc.code} {detail}".strip()) from exc
        except urllib.error.URLError as exc:
            raise UploadError(f"{exc.reason}") from exc
        try:
            return json.loads(body)
        except ValueError:
            return {}

    def _report(self, reply: dict) -> None:
        n = reply.get("n_receptions")
        snr = reply.get("combined_snr_db")
        where = f"transmission {reply.get('transmission_id', '?')}"
        detail = f"{n} station(s)" if n else ""
        if isinstance(snr, (int, float)):
            detail += f", combined {snr:.1f} dB"
        self._say(f"uploaded to {where}" + (f" -- {detail}" if detail else ""))
        self._check_clock(reply.get("server_time"))

    def _check_clock(self, server_time) -> None:
        """Compare clocks using the reply we already asked for.

        A station whose clock is off does not fail: it uploads
        successfully and is then filed as a transmission of its own,
        contributing nothing to the picture it should have improved.
        That is invisible from here, so it is worth saying out loud to
        the one person who can fix it.
        """
        if not isinstance(server_time, (int, float)):
            return
        skew = time.time() - server_time
        if abs(skew) > CLOCK_WARN_S:
            self._say(
                f"WARNING: this station's clock is {skew:+.1f}s from the server's. "
                "Receptions are matched to a transmission by their start time, so "
                "a clock this far out may stop them combining. Check NTP."
            )

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(f"[upload] {msg}")
