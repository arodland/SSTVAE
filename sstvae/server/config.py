"""Server settings, in one place."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["ServerConfig"]


@dataclass
class ServerConfig:
    """How one aggregation server is set up.

    `utc_tolerance_s` is the only field that decides *what a
    transmission is*, and it is deliberately generous. Two stations are
    taken to have heard the same transmission when their reported start
    times agree within it, and the errors either side of that are not
    symmetric: too wide can only merge transmissions that one
    transmitter could not physically have made (the shortest mode lasts
    32 s), while too narrow splits a real transmission into two buckets
    and silently forfeits exactly the diversity gain this server exists
    to collect. 5 s is five times the ~1 s that stations are asked to
    hold, and six times under the 32 s floor.

    Frequency is not part of the rule. It is the least reliable field a
    payload carries -- a skimmer may have no rig control at all and the
    operator types it in -- so requiring it to match would cost real
    receptions. `freq_split_khz` exists for a multi-band aggregator
    that wants it after all, and is off by default.
    """

    data_dir: Path = Path("aggregator-data")
    db_path: Path | None = None  # defaults to data_dir/server.db
    model: str | None = None
    precision: str | None = None
    utc_tolerance_s: float = 5.0
    freq_split_khz: float = 0.0  # 0 = frequency never splits a transmission
    # How far into the future a reported start may sit before the
    # payload is refused; see `db.clock_complaint`.
    future_slack_s: float = 60.0
    max_upload_bytes: int = 8 * 1024 * 1024

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        if self.db_path is None:
            self.db_path = self.data_dir / "server.db"
        self.db_path = Path(self.db_path)

    @property
    def receptions_dir(self) -> Path:
        return self.data_dir / "receptions"

    @property
    def pictures_dir(self) -> Path:
        return self.data_dir / "pictures"
