"""sstvae.qrss.sequences' slot clock -> sstvae_native.qrss.slot
(native/core/qrss/slot): parse_utc, parse_slot, quarter_hour_count and
quarter_hour_start. parse_utc is CPython 3.13's datetime.fromisoformat
ported, so qrss_transmit.py, qrss_receive.py and qrss_simulate.py read
--slot through the C++ under --native-qrss=tx.

Times cross as real datetime objects (timezone.utc for a zero offset, as
fromisoformat returns it), so callers' .timestamp() and comparisons are
unchanged.
"""

STAGE = "tx"


def rows(q):
    s = q.slot
    t = "sstvae.qrss.sequences"
    return {
        (t, "parse_utc"): s.parse_utc,
        (t, "parse_slot"): s.parse_slot,
        (t, "quarter_hour_count"): s.quarter_hour_count,
        (t, "quarter_hour_start"): s.quarter_hour_start,
    }
