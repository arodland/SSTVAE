"""The HTTP surface: uploads in, pictures and a gallery out."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from ..rx.receptionfile import ReceptionFileError, read
from .combine import combine_transmission
from .config import ServerConfig
from .db import Database, clock_complaint

__all__ = ["create_app"]

_GALLERY = Path(__file__).with_name("static") / "index.html"


def create_app(config: ServerConfig | None = None, codec=None, db=None) -> FastAPI:
    """Build the app.

    `codec` and `db` are injected for tests. The codec is loaded lazily
    on first use rather than at startup, so the server comes up (and
    answers /healthz) without waiting on a model download, and a station
    can be issued a key before any picture has been decoded.
    """
    config = config or ServerConfig()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.receptions_dir.mkdir(parents=True, exist_ok=True)
    config.pictures_dir.mkdir(parents=True, exist_ok=True)
    database = db or Database(config.db_path)

    app = FastAPI(title="SSTVAE reception aggregator")
    app.state.config = config
    app.state.db = database
    app.state.codec = codec

    def get_codec():
        if app.state.codec is None:
            from ..codec import load_codec

            app.state.codec = load_codec(config.model, precision=config.precision)
        return app.state.codec

    def station(request: Request):
        auth = request.headers.get("authorization", "")
        scheme, _, token = auth.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise HTTPException(401, "a bearer token is required")
        row = database.station_for_key(token.strip())
        if row is None:
            raise HTTPException(401, "unknown or revoked key")
        return row

    # -- upload ------------------------------------------------------

    @app.post("/api/v1/receptions")
    async def post_reception(request: Request, who=Depends(station)):
        payload = await request.body()
        if len(payload) > config.max_upload_bytes:
            raise HTTPException(413, "payload too large")
        try:
            result, meta = read(payload)
        except ReceptionFileError as exc:
            raise HTTPException(400, str(exc)) from exc

        claimed = (meta.get("station_callsign") or "").strip().upper()
        if claimed and claimed != who["callsign"]:
            # The key names the station; a payload claiming another one
            # is refused rather than believed.
            raise HTTPException(
                400,
                f"this key belongs to {who['callsign']}, but the payload says {claimed}",
            )

        utc_start = float(meta["utc_start_epoch"])
        now = time.time()
        complaint = clock_complaint(
            utc_start, meta.get("mode_name"), now, config.future_slack_s
        )
        if complaint:
            raise HTTPException(400, complaint)
        database.record_skew(who["id"], now - utc_start)

        transmitter = (meta.get("callsign") or "").strip().upper() or "UNKNOWN"
        dial = meta.get("dial_freq_hz")
        tx_id = database.find_or_create_transmission(
            transmitter, utc_start, dial,
            config.utc_tolerance_s, config.freq_split_khz,
        )

        dest = config.receptions_dir / str(tx_id)
        dest.mkdir(parents=True, exist_ok=True)
        file_path = dest / f"{who['callsign']}.npz"
        file_path.write_bytes(payload)

        database.upsert_reception(
            tx_id,
            who["id"],
            file_path=str(file_path),
            utc_start=utc_start,
            snr_db=meta.get("snr_db"),
            path=meta.get("path"),
            mode_name=meta.get("mode_name"),
            frames_received=meta.get("frames_received"),
            dial_freq_hz=dial,
        )

        combined = _recombine(app, tx_id)
        rows = database.receptions_for(tx_id)
        # Two different numbers, and the difference matters to the
        # station asking: how many uploaded, and how many went into the
        # picture. A station whose own reception was left out is told
        # why -- otherwise it gets a 200 and never learns it contributed
        # nothing.
        skipped = dict(combined.skipped) if combined else {}
        return {
            "transmission_id": tx_id,
            "n_receptions": len(combined.used) if combined else 0,
            "n_stations": len(rows),
            "combined_snr_db": None if combined is None else combined.snr_db,
            "mode": None if combined is None else combined.mode_name,
            "note": skipped.get(who["id"]),
            # The station compares this with its own clock: a wrong
            # clock uploads fine and then never combines, which is
            # otherwise invisible from both ends.
            "server_time": time.time(),
        }

    def _recombine(app, tx_id: int):
        rows = database.receptions_for(tx_id)
        if not rows:
            return None
        combined = combine_transmission(rows, get_codec())
        if combined is None:
            return None
        image_path = config.pictures_dir / f"{tx_id}.png"
        combined.image.save(image_path)
        database.update_transmission(
            tx_id,
            mode_name=combined.mode_name,
            n_receptions=len(combined.used),
            combined_snr_db=combined.snr_db,
            image_path=str(image_path),
        )
        database.set_contributions(tx_id, combined.contributions)
        return combined

    # -- read-only ---------------------------------------------------

    def _as_json(row):
        stations = [
            {
                "callsign": r["station_callsign"],
                "snr_db": r["snr_db"],
                "path": r["path"],
                "frames_received": r["frames_received"],
                "dial_freq_hz": r["dial_freq_hz"],
                "contrib_frac": r["contrib_frac"],
            }
            for r in database.receptions_for(row["id"])
        ]
        return {
            "id": row["id"],
            "callsign": row["callsign"],
            "utc_start": row["utc_start"],
            "dial_freq_hz": row["dial_freq_hz"],
            "mode": row["mode_name"],
            "n_receptions": row["n_receptions"],
            "combined_snr_db": row["combined_snr_db"],
            "updated_utc": row["updated_utc"],
            # Millisecond resolution, not seconds: two stations
            # uploading within the same second would otherwise produce
            # the same URL, and a browser would go on showing the
            # one-station picture that the second station just improved.
            "picture_url": (
                f"/pictures/{row['id']}.png?v={row['updated_utc']:.3f}"
                if row["image_path"] else None
            ),
            "stations": stations,
        }

    @app.get("/api/v1/transmissions")
    def list_transmissions(limit: int = 50):
        return {"transmissions": [_as_json(r) for r in database.transmissions(limit)]}

    @app.get("/api/v1/transmissions/{tx_id}")
    def get_transmission(tx_id: int):
        row = database.transmission(tx_id)
        if row is None:
            raise HTTPException(404, "no such transmission")
        return _as_json(row)

    @app.get("/pictures/{tx_id}.png")
    def get_picture(tx_id: int):
        row = database.transmission(tx_id)
        if row is None or not row["image_path"]:
            raise HTTPException(404, "no picture for that transmission")
        return FileResponse(row["image_path"], media_type="image/png")

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "server_time": time.time()}

    @app.get("/", response_class=HTMLResponse)
    def gallery():
        return _GALLERY.read_text(encoding="utf-8")

    return app
