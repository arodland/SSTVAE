"""sstvae.qrss.live's `Stamper` -> sstvae_native.qrss.sources.Stamper
(native/core/qrss/sources). The reference's StdinSource looks `Stamper` up
in live's namespace when it is made, so it stamps with the C++ one too.

`StdinSource` and `WavSource` are not assigned: their C++ twins read a
file descriptor (not a Python stream) and replay at the C++'s own clock
unless given one, and nothing in the suites drives the reference's
through the listener; tests/test_qrss_native_sources.py holds each
against the reference instead.
"""

STAGE = "rx"


def rows(q):
    return {("sstvae.qrss.live", "Stamper"): q.sources.Stamper}
