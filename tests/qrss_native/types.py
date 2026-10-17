"""sstvae.qrss.types (pass_uid, PassResult.save/.load) ->
sstvae_native.qrss.types (native/core/qrss/types).

PassResult stays the reference's dataclass: `save` hands it to the C++,
which builds the meta JSON and the .npz byte for byte as np.savez of
json.dumps would; `load` parses the file in C++ and `_rx.to_ref` builds
the reference's PassResult (with header.HeaderFields, types.CwIdResult,
...) from what comes back. A file the C++ cannot open is reported as the
OSError Python's own open raises for it.
"""

import os

STAGE = "rx"


def rows(q):
    from . import _rx

    t = q.types

    def save(self, path):
        p = os.fspath(path)
        try:
            t.save_pass(self, p)
        except RuntimeError as e:
            open(p, "ab").close()        # the OSError Python's open raises, when it is one
            raise OSError(str(e)) from e

    def load(cls, path):
        if hasattr(path, "read"):
            return _rx.to_ref(t.parse_pass(path.read()))
        p = os.fspath(path)
        try:
            d = t.load_pass(p)
        except RuntimeError:
            open(p, "rb").close()        # as above
            raise
        return _rx.to_ref(d)

    target = "sstvae.qrss.types"
    return {
        (target, "pass_uid"): t.pass_uid,
        (f"{target}.PassResult", "save"): save,
        (f"{target}.PassResult", "load"): classmethod(load),
    }
