"""The multi-pass store, sstvae.qrss.store -> sstvae_native.qrss.store
(native/core/qrss/store).

`Store` is tests/qrss_native/_store.py's facade: the native class with the
reference's os.replace and ZW_CACHE as its seams and load_pass returning
the reference's PassResult; everything else is the C++. `Accumulator` is
the native class (its arrays numpy views of the C++'s). The module's
functions are the native ones. Every Store any reference module or test
constructs through `sstvae.qrss.store.Store` (live.py, em.py, the qrss_*.py
scripts run in-process, the tests' fixtures) is then the C++ store; a
script run as a subprocess is Python, which makes test_qrss_cli.py an
interop check of the files as well.
"""

STAGE = "rx"


def rows(q):
    from . import _store

    s = q.store
    t = "sstvae.qrss.store"
    return {
        (t, "Store"): _store.store_class(q),
        (t, "Accumulator"): s.Accumulator,
        (t, "default_root"): s.default_root,
        (t, "key_name"): s.key_name,
        (t, "slot_time"): s.slot_time,
        (t, "mean_w_db"): s.mean_w_db,
        (t, "canonical_index"): s.canonical_index,
    }
