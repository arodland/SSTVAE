"""Association rules 1 to 4, sstvae.qrss.associate ->
sstvae_native.qrss.associate (native/core/qrss/associate).

Every function is the native one, two through tests/qrss_native/_store.py:
`associate` hands the C++ the module's current `_soft_header` as its rule-2
seam when a test has replaced it (test_qrss_multipass.py does), and
`_soft_header` returns the reference's HeaderFields. The store they are
given is the native Store (tests/qrss_native/store.py), which the C++
reads directly; the rule strings are the reference's (a GUI contract).
"""

STAGE = "rx"


def rows(q):
    from . import _store

    return _store.associate_rows(q)
