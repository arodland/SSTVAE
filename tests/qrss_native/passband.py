"""The 48 h passband store and the directory lock -> sstvae_native.qrss.passband
(native/core/qrss/passband, flac, dirlock).

`frontend.PassbandStore` is replaced whole, as a class: the native one
keeps the reference's constructor, methods and the attributes callers
read (`root` as a pathlib.Path, `fs`, `scale`, `keep_hours`, `per_hour`,
`pack`, and `_packer`, None until a packer thread starts, then something
with `join(timeout)`), packs with its own FLAC encoder (no soundfile
needed) and reads the reference's packed hours too. em._from_passband
imports the class at call time, so its isinstance test sees this one.
`_merge_ranges` and `slot_t0` go with it (live.py's from-import of
slot_t0 is found and replaced by the mechanism), and `live.dir_lock` is
the native lock: flock / LockFileEx on the same byte as the reference's,
so a Python and a C++ listener exclude each other. `_atomic_write` is
not replaced (the store's JSON writes are the C++'s own, with the same
temporary names).
"""

STAGE = "rx"


def rows(q):
    p = q.passband
    f = "sstvae.qrss.frontend"
    return {
        (f, "PassbandStore"): p.PassbandStore,
        (f, "_merge_ranges"): p._merge_ranges,
        (f, "slot_t0"): p.slot_t0,
        ("sstvae.qrss.live", "dir_lock"): p.dir_lock,
    }
