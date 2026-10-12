"""sstvae.qrss.header (fields, pack/unpack, encode) -> sstvae_native.qrss.header.

HeaderFields stays the reference's dataclass: the native functions take its
eight fields as arguments and return them as a tuple, and the adapters here
convert. `decode` is in header_rx.py (stage rx).
"""

from sstvae.qrss.constants import FORMAT_VERSION

STAGE = "tx"


def fields(h):
    """HeaderFields (or anything with its attributes) -> the native arguments."""
    return (h.callsign, h.grid, h.picture_id, h.mode, h.segment, h.codec_id, h.version, h.reserved)


def wrap(t):
    """A native result tuple -> HeaderFields; None stays None."""
    from sstvae.qrss.header import HeaderFields

    return None if t is None else HeaderFields(*t)


def rows(q):
    n = q.header
    t = "sstvae.qrss.header"

    def pack(h):
        return n.pack(*fields(h))

    def unpack(bits, versions=(FORMAT_VERSION,)):
        return wrap(n.unpack(bits, versions))

    def encode(h):
        return n.encode(*fields(h))

    return {
        (t, "crc16"): n.crc16,
        (t, "grid_encode"): n.grid_encode,
        (t, "grid_decode"): n.grid_decode,
        (t, "pack"): pack,
        (t, "unpack"): unpack,
        (t, "encode"): encode,
    }
