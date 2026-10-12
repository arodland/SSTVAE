"""sstvae.qrss.header.decode (polar CA-SCL plus the field checks, with the
version fallback) -> sstvae_native.qrss.header.decode."""

from sstvae.qrss.constants import FORMAT_VERSION

from .header import wrap

STAGE = "rx"


def rows(q):
    n = q.header

    def decode(llr, list_size=8, versions=(FORMAT_VERSION,)):
        return wrap(n.decode(llr, list_size, versions))

    return {("sstvae.qrss.header", "decode"): decode}
