#!/usr/bin/env python3
"""Run the reception aggregation server, and manage its station keys.

Many stations hear one balloon transmission, each partially; this
server collects their latents and combines them into the best picture
the network as a whole could hear. See docs/reception-aggregation.md.

    sstvae_server.py issue-key --callsign N0CALL
    sstvae_server.py run --data-dir ./aggregator-data

Install the extra first: `pip install -e '.[server]'`.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from sstvae.server.config import ServerConfig
from sstvae.server.db import Database


def _config(args) -> ServerConfig:
    return ServerConfig(
        data_dir=Path(args.data_dir),
        db_path=Path(args.db) if args.db else None,
        model=getattr(args, "model", None),
        precision=getattr(args, "precision", None),
        utc_tolerance_s=getattr(args, "utc_tolerance", 5.0),
        freq_split_khz=getattr(args, "freq_split_khz", 0.0),
    )


def cmd_run(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "the server needs its extra: pip install -e '.[server]'",
            file=sys.stderr,
        )
        return 1
    from sstvae.server.app import create_app

    config = _config(args)
    print(f"data in {config.data_dir}, matching within +-{config.utc_tolerance_s:g}s")
    uvicorn.run(create_app(config), host=args.host, port=args.port)
    return 0


def cmd_issue_key(args) -> int:
    db = Database(_config(args).db_path)
    key = db.issue_key(args.callsign, args.note or "")
    call = args.callsign.strip().upper()
    print(f"station {call}")
    print(f"key:     {key}")
    print()
    print("This is the only time the key is shown -- only its hash is stored.")
    print("Give it to the operator, who passes it as:")
    print(f"  sstvae_listen.py --upload-url <url> --station-call {call} \\")
    print("      --upload-key-file <path to a file holding the key>")
    return 0


def cmd_revoke_key(args) -> int:
    db = Database(_config(args).db_path)
    if db.revoke(args.callsign):
        print(f"revoked {args.callsign.strip().upper()}")
        return 0
    print(f"no station {args.callsign.strip().upper()}", file=sys.stderr)
    return 1


def cmd_list_stations(args) -> int:
    db = Database(_config(args).db_path)
    rows = db.stations()
    if not rows:
        print("no stations yet -- issue-key creates one")
        return 0
    print(f"{'callsign':<10} {'state':<8} {'clock':>9}  note")
    for row in rows:
        skew = row["clock_skew_s"]
        # The skew includes however long the upload took to arrive, so
        # it is a bound rather than a measurement -- big numbers are
        # meaningful, small ones are not.
        clock = "—" if skew is None else f"{skew:+.1f}s"
        state = "revoked" if row["revoked"] else "active"
        print(f"{row['callsign']:<10} {state:<8} {clock:>9}  {row['note']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="aggregator-data",
                    help="where receptions, pictures and the database live")
    ap.add_argument("--db", default=None, help="database path (default <data-dir>/server.db)")
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="serve the API and gallery")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8000)
    run.add_argument("--model", default=None, help="codec path, as for the other tools")
    run.add_argument("--precision", default=None)
    run.add_argument(
        "--utc-tolerance", type=float, default=5.0, metavar="SECONDS",
        help="how far two stations' reported start times may differ and still "
        "be one transmission (default 5). Generous on purpose: too narrow "
        "splits a transmission and forfeits the diversity gain, while too "
        "wide cannot merge two, since no transmitter sends twice in 32s.",
    )
    run.add_argument(
        "--freq-split-khz", type=float, default=0.0, metavar="KHZ",
        help="also split transmissions whose reported dial frequencies differ "
        "by more than this. Off by default: a skimmer may have no rig "
        "control, so frequency is the least reliable field a payload carries.",
    )
    run.set_defaults(func=cmd_run)

    issue = sub.add_parser("issue-key", help="create or re-key a station")
    issue.add_argument("--callsign", required=True)
    issue.add_argument("--note", default="", help="free text, shown by list-stations")
    issue.set_defaults(func=cmd_issue_key)

    revoke = sub.add_parser("revoke-key", help="disable a station's key")
    revoke.add_argument("--callsign", required=True)
    revoke.set_defaults(func=cmd_revoke_key)

    listing = sub.add_parser("list-stations", help="show known stations")
    listing.set_defaults(func=cmd_list_stations)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
