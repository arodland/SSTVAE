"""sstvae.qrss.morse -> sstvae_native.qrss.morse (native/core/qrss/morse).

The table itself (`MORSE`, `MAX_CALL_CHARS`) is data the generator copies
into qrss/qrss_config.hpp, so the parity test compares it rather than
substituting it.
"""

STAGE = "tx"


def rows(q):
    m = q.morse
    t = "sstvae.qrss.morse"
    return {
        (t, "check_callsign"): m.check_callsign,
        (t, "morse_keying"): m.morse_keying,
        (t, "keying_units"): m.keying_units,
    }
