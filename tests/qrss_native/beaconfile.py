"""sstvae.qrss.beaconfile -> sstvae_native.qrss.beaconfile
(native/core/qrss/beaconfile).

BeaconFile, StoredPicture and HeaderFields stay the reference's
dataclasses: the native functions take any object with their attributes
and return a BeaconFile's fields as a dict (HeaderFields' as a tuple),
which the adapters here wrap. BeaconFile's methods are replaced by plain
functions, as a native function assigned to a class does not bind.
"""

STAGE = "tx"


def rows(q):
    from sstvae.qrss import beaconfile as B
    from sstvae.qrss.header import HeaderFields

    n = q.beaconfile
    t, c = "sstvae.qrss.beaconfile", "sstvae.qrss.beaconfile.BeaconFile"

    def header_fields(self):
        return HeaderFields(*n.header_fields(self))

    def check(self):
        n.check(self)

    def to_bytes(self):
        return n.to_bytes(self)

    def from_bytes(cls, data):
        return cls(**n.from_bytes(data))

    def read(path):
        return B.BeaconFile(**n.read(path))

    def from_picture(sp, segment, header, ook=False):
        return B.BeaconFile(**n.from_picture(sp, segment, header, ook))

    return {
        (t, "file_size"): n.file_size,
        (t, "write"): n.write,
        (t, "read"): read,
        (t, "from_picture"): from_picture,
        (c, "header_fields"): header_fields,
        (c, "check"): check,
        (c, "to_bytes"): to_bytes,
        (c, "from_bytes"): classmethod(from_bytes),
    }
