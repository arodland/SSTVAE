"""Helpers for the multi-pass store's rows (store.py, associate.py here):
a thin facade over sstvae_native.qrss.store.Store, and the association
entry points that honour a test's replacement of associate._soft_header.

The facade is a Python subclass of the native Store and adds nothing but
the reference's three Python-side seams, which the C++ cannot see by
itself:

- **os.replace**, which store.py calls at every atomic write (and
  test_qrss_multipass.py's test_p1_atomic_writes monkeypatches): the
  facade hands the C++ a `replace` that looks os.replace up at each call.
- **store.ZW_CACHE**, read at every pass_zw (test_pass_cache_is_bounded
  monkeypatches it): the facade hands the C++ a `zw_cache` that reads the
  reference module's value at each call.
- **load_pass** returns the reference's PassResult (`_rx.to_ref` of the
  native record), since em.py hands it to dataclasses.replace.

Everything else -- every file, every sum, the caches, the locking, `root`
and `clock` -- is the native class's, inherited unchanged.

`associate_rows(q)` gives associate's rows. `associate` is the native one,
handed the module's current `_soft_header` as its rule-2 seam when that is
no longer this file's row (a test's monkeypatch), so the C++ calls it
exactly where the reference would; `_soft_header` returns the reference's
HeaderFields.
"""

from __future__ import annotations

import os
import time

from . import _rx

_CLASSES: dict[int, type] = {}


def _replace(src, dst):
    os.replace(src, dst)        # looked up at each call: a monkeypatch reaches it


def store_class(q):
    """sstvae.qrss.store.Store as the C++ (built once per native module)."""
    cls = _CLASSES.get(id(q))
    if cls is not None:
        return cls
    import sstvae.qrss.store as ref

    def zw_cache():
        return ref.ZW_CACHE

    class Store(q.store.Store):
        """The on-disk multi-pass store: sstvae_native.qrss.store.Store with
        store.py's os.replace and ZW_CACHE as its seams (tests/qrss_native/
        _store.py)."""

        def __init__(self, root=None, clock=time.time):
            super().__init__(root, clock, replace=_replace, zw_cache=zw_cache)

        def load_pass(self, uid):
            return _rx.to_ref(super().load_pass(uid))

    Store.__qualname__ = "Store"
    _CLASSES[id(q)] = Store
    return Store


def associate_rows(q):
    import sstvae.qrss.associate as ref

    n = q.associate

    def _soft_header(store, uid, llr):
        r = n._soft_header(store, uid, llr)
        return None if r is None else (_rx.to_ref(r[0]), r[1])

    def associate(p, store):
        sh = ref._soft_header
        return n.associate(p, store, soft_header=None if sh is _soft_header else sh)

    t = "sstvae.qrss.associate"
    return {
        (t, "corr_stat"): n.corr_stat,
        (t, "expected_t"): n.expected_t,
        (t, "corr_accept"): n.corr_accept,
        (t, "_acc_reference"): n._acc_reference,
        (t, "_cluster_sums"): n._cluster_sums,
        (t, "_best_accumulator"): n._best_accumulator,
        (t, "validate_header_match"): n.validate_header_match,
        (t, "_attach_by_header"): n._attach_by_header,
        (t, "header_agrees"): n.header_agrees,
        (t, "_soft_header"): _soft_header,
        (t, "_rule"): n._rule,
        (t, "associate"): associate,
        (t, "resolve_provisional"): n.resolve_provisional,
    }


# --- for the parity tests (tests/test_qrss_native_store.py, _associate.py) ------------------


def apply_store_rows(native, mp, wrap=None):
    """This lane's rows (store.py's and associate.py's here) over their
    targets and every from-import site in sstvae.qrss, through `mp` (a
    pytest MonkeyPatch); `wrap(key, repl)` may replace each row first (a
    counting spy, installed before the rows so it is what the module
    attributes and the from-import sites hold)."""
    import qrss_native

    from . import associate as rows_associate
    from . import store as rows_store

    q = native.qrss
    reference = qrss_native._import_reference()
    replaced = {}
    for mod in (rows_store, rows_associate):
        for key, repl in mod.rows(q).items():
            if wrap is not None:
                repl = wrap(key, repl)
            owner = qrss_native._owner(key[0])
            orig = getattr(owner, key[1])
            mp.setattr(owner, key[1], repl)
            if callable(orig) or hasattr(orig, "__array__"):
                replaced[id(orig)] = (orig, repl)
    for m in reference:
        for k, v in list(vars(m).items()):
            hit = replaced.get(id(v))
            if hit is not None and hit[0] is v:
                mp.setattr(m, k, hit[1])


def counting(used):
    """A `wrap` for apply_store_rows counting the calls into each row in
    `used` (by name); a class row (Store, Accumulator) is wrapped in a
    subclass counting its constructions."""
    def wrap(key, repl):
        name = key[1]
        if isinstance(repl, type):
            def __init__(self, *a, **k):
                used[name] = used.get(name, 0) + 1
                repl.__init__(self, *a, **k)
            return type(repl.__name__, (repl,), {"__init__": __init__})

        def f(*a, **k):
            used[name] = used.get(name, 0) + 1
            return repl(*a, **k)
        return f
    return wrap


def tree(root):
    """{relative path: bytes} of every file under a store's root."""
    from pathlib import Path

    root = Path(root)
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def compare_trees(py_root, cpp_root):
    """Two store trees, file by file: the same names and the same bytes,
    except where a member's mean_w_db was written from numpy's log10 (the
    reference) and from the C library's (the C++), which differ in the
    last place for a few percent of arguments on a machine where numpy
    uses its own SIMD log10. Each such difference is checked to be exactly
    that -- the reference's value is 10 * np.log10(m), the C++'s
    10 * math.log10(m), m the pass's mean weight (read from the pass file
    in the reference's tree), and the file is otherwise the same text --
    and returned as (path, uid, reference, C++)."""
    import io
    import json
    import math

    import numpy as np

    a, b = tree(py_root), tree(cpp_root)
    assert sorted(a) == sorted(b), sorted(set(a) ^ set(b))
    justified = []
    for name in a:
        if a[name] == b[name]:
            continue
        assert name.startswith("acc/") and name.endswith(".json"), f"{name} differs"
        ja, jb = json.loads(a[name]), json.loads(b[name])
        assert len(ja["members"]) == len(jb["members"]), name
        for x, y in zip(ja["members"], jb["members"]):
            if x == y:
                continue
            assert {k: v for k, v in x.items() if k != "mean_w_db"} == \
                {k: v for k, v in y.items() if k != "mean_w_db"}, (name, x, y)
            with np.load(io.BytesIO(a[f"passes/{x['uid']}.npz"])) as d:
                m = float(np.mean(d["w"].astype(np.float64)))
            assert x["mean_w_db"] == float(10 * np.log10(m)), (name, x)
            assert y["mean_w_db"] == 10 * math.log10(m), (name, y)
            justified.append((name, x["uid"], x["mean_w_db"], y["mean_w_db"]))
            y["mean_w_db"] = x["mean_w_db"]
        assert json.dumps(jb, indent=1, sort_keys=True).encode() == a[name], f"{name}: more than mean_w_db"
    return justified
