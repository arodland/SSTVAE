"""Storage, and the rule that decides what a transmission is.

SQLite through the standard library. One process-wide lock guards every
write and every recombine: FastAPI runs sync endpoints in a threadpool,
uploads are rare (at most one per station per transmission, and a
transmission lasts 32-95 s), and a lock is far cheaper to reason about
than the alternative.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from ..config import MODES
from .auth import hash_key, key_matches, new_key

__all__ = [
    "Database",
    "clock_complaint",
    "ClockError",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS stations (
    id           INTEGER PRIMARY KEY,
    callsign     TEXT NOT NULL UNIQUE,
    key_hash     TEXT NOT NULL,
    note         TEXT NOT NULL DEFAULT '',
    revoked      INTEGER NOT NULL DEFAULT 0,
    clock_skew_s REAL,
    created_utc  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS transmissions (
    id              INTEGER PRIMARY KEY,
    callsign        TEXT NOT NULL,
    utc_start       REAL NOT NULL,
    dial_freq_hz    REAL,
    mode_name       TEXT,
    n_receptions    INTEGER NOT NULL DEFAULT 0,
    combined_snr_db REAL,
    image_path      TEXT,
    updated_utc     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS transmissions_lookup
    ON transmissions (callsign, utc_start);
CREATE TABLE IF NOT EXISTS receptions (
    id              INTEGER PRIMARY KEY,
    transmission_id INTEGER NOT NULL REFERENCES transmissions(id),
    station_id      INTEGER NOT NULL REFERENCES stations(id),
    file_path       TEXT NOT NULL,
    utc_start       REAL NOT NULL,
    snr_db          REAL,
    path            TEXT NOT NULL,
    mode_name       TEXT,
    frames_received INTEGER,
    dial_freq_hz    REAL,
    contrib_frac    REAL,
    received_utc    REAL NOT NULL,
    UNIQUE (transmission_id, station_id)
);
"""


class ClockError(ValueError):
    """A reported start time that cannot be true."""


def clock_complaint(utc_start: float, mode_name: str | None, now: float,
                    slack_s: float) -> str | None:
    """Why this start time is impossible, or None if it might be true.

    Only one direction is decidable. A payload claiming a transmission
    that has *not finished yet* cannot be honest: the station decoded it
    before uploading, so the whole transmission is already in its past,
    and every delay between there and here pushes the other way. A start
    time that looks too old, by contrast, is exactly what a queued or
    retried upload looks like, so it is recorded and never refused.
    """
    spec = MODES.get(mode_name) if mode_name else MODES["C"]
    duration = spec.duration_s if spec else MODES["C"].duration_s
    ends_at = utc_start + duration
    if ends_at > now + slack_s:
        ahead = ends_at - now
        return (
            f"this reception says it ends {ahead:.0f}s from now, which cannot be "
            "true of something already decoded -- the station's clock is fast"
        )
    return None


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- stations ----------------------------------------------------

    def issue_key(self, callsign: str, note: str = "") -> str:
        """Create or re-key a station. The plaintext is returned once."""
        callsign = callsign.strip().upper()
        if not callsign:
            raise ValueError("a station needs a callsign")
        key = new_key()
        with self.lock:
            self.conn.execute(
                """INSERT INTO stations (callsign, key_hash, note, revoked, created_utc)
                   VALUES (?, ?, ?, 0, ?)
                   ON CONFLICT(callsign) DO UPDATE SET
                       key_hash=excluded.key_hash,
                       note=excluded.note,
                       revoked=0""",
                (callsign, hash_key(key), note, time.time()),
            )
            self.conn.commit()
        return key

    def revoke(self, callsign: str) -> bool:
        with self.lock:
            cur = self.conn.execute(
                "UPDATE stations SET revoked=1 WHERE callsign=?",
                (callsign.strip().upper(),),
            )
            self.conn.commit()
        return cur.rowcount > 0

    def stations(self) -> list[sqlite3.Row]:
        with self.lock:
            return list(
                self.conn.execute("SELECT * FROM stations ORDER BY callsign")
            )

    def station_for_key(self, key: str) -> sqlite3.Row | None:
        """The station this key belongs to, if any.

        Every non-revoked station is checked, because the key names the
        station rather than the other way round -- there is nothing to
        look up by.
        """
        if not key:
            return None
        with self.lock:
            rows = list(self.conn.execute("SELECT * FROM stations WHERE revoked=0"))
        for row in rows:
            if key_matches(key, row["key_hash"]):
                return row
        return None

    def record_skew(self, station_id: int, skew_s: float) -> None:
        with self.lock:
            self.conn.execute(
                "UPDATE stations SET clock_skew_s=? WHERE id=?", (skew_s, station_id)
            )
            self.conn.commit()

    # -- transmissions -----------------------------------------------

    def find_or_create_transmission(
        self,
        callsign: str,
        utc_start: float,
        dial_freq_hz: float | None,
        utc_tolerance_s: float,
        freq_split_khz: float = 0.0,
    ) -> int:
        """Which transmission this reception belongs to.

        Callsign plus start time, within tolerance. The nearest
        candidate in time wins, so a burst of uploads cannot chain
        themselves into one over-wide bucket.
        """
        callsign = (callsign or "").strip().upper()
        with self.lock:
            rows = list(
                self.conn.execute(
                    """SELECT id, utc_start, dial_freq_hz FROM transmissions
                       WHERE callsign=? AND ABS(utc_start - ?) <= ?
                       ORDER BY ABS(utc_start - ?)""",
                    (callsign, utc_start, utc_tolerance_s, utc_start),
                )
            )
            for row in rows:
                if freq_split_khz > 0 and dial_freq_hz and row["dial_freq_hz"]:
                    if abs(dial_freq_hz - row["dial_freq_hz"]) > freq_split_khz * 1000.0:
                        continue
                # Keep the earliest start reported for this
                # transmission: later joiners heard it late, and the
                # transmission did not start when they tuned in.
                if utc_start < row["utc_start"]:
                    self.conn.execute(
                        "UPDATE transmissions SET utc_start=? WHERE id=?",
                        (utc_start, row["id"]),
                    )
                    self.conn.commit()
                return int(row["id"])

            cur = self.conn.execute(
                """INSERT INTO transmissions
                       (callsign, utc_start, dial_freq_hz, updated_utc)
                   VALUES (?, ?, ?, ?)""",
                (callsign, utc_start, dial_freq_hz, time.time()),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def upsert_reception(self, transmission_id: int, station_id: int, **fields) -> int:
        """Record one station's reception, replacing its previous one.

        A station that uploads twice for one transmission replaces
        rather than adds. Combining two copies of one reception would
        break the independence the maximal-ratio weighting assumes and
        report a confidence the signal does not have.
        """
        with self.lock:
            self.conn.execute(
                "DELETE FROM receptions WHERE transmission_id=? AND station_id=?",
                (transmission_id, station_id),
            )
            cur = self.conn.execute(
                """INSERT INTO receptions
                       (transmission_id, station_id, file_path, utc_start, snr_db,
                        path, mode_name, frames_received, dial_freq_hz, received_utc)
                   VALUES (:transmission_id, :station_id, :file_path, :utc_start,
                           :snr_db, :path, :mode_name, :frames_received,
                           :dial_freq_hz, :received_utc)""",
                {
                    "transmission_id": transmission_id,
                    "station_id": station_id,
                    "received_utc": time.time(),
                    **fields,
                },
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def receptions_for(self, transmission_id: int) -> list[sqlite3.Row]:
        with self.lock:
            return list(
                self.conn.execute(
                    """SELECT r.*, s.callsign AS station_callsign
                       FROM receptions r JOIN stations s ON s.id = r.station_id
                       WHERE r.transmission_id=?
                       ORDER BY r.snr_db DESC""",
                    (transmission_id,),
                )
            )

    def update_transmission(self, transmission_id: int, **fields) -> None:
        if not fields:
            return
        fields["updated_utc"] = time.time()
        assigns = ", ".join(f"{k}=?" for k in fields)
        with self.lock:
            self.conn.execute(
                f"UPDATE transmissions SET {assigns} WHERE id=?",
                (*fields.values(), transmission_id),
            )
            self.conn.commit()

    def set_contributions(self, transmission_id: int, by_station: dict[int, float]) -> None:
        with self.lock:
            for station_id, frac in by_station.items():
                self.conn.execute(
                    """UPDATE receptions SET contrib_frac=?
                       WHERE transmission_id=? AND station_id=?""",
                    (float(frac), transmission_id, station_id),
                )
            self.conn.commit()

    def transmission(self, transmission_id: int) -> sqlite3.Row | None:
        with self.lock:
            return self.conn.execute(
                "SELECT * FROM transmissions WHERE id=?", (transmission_id,)
            ).fetchone()

    def transmissions(self, limit: int = 50) -> list[sqlite3.Row]:
        with self.lock:
            return list(
                self.conn.execute(
                    """SELECT * FROM transmissions
                       ORDER BY utc_start DESC LIMIT ?""",
                    (limit,),
                )
            )
