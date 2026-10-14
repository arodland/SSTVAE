"""Leave-one-out EM, the loop: sstvae.qrss.em's `pass_path`, `rereceive`
and `em_refine` -> sstvae_native.qrss.em_refine (native/core/qrss/em_refine).

Everything under them is the C++'s: rereceive runs the native tracker,
demodulator and receive_pass whatever sstvae.qrss.track, demod or receiver
hold (a test spying on one of those while calling em.rereceive would not
see the call -- none does), and pass_path the native line_path,
refine_path and verify_track.

em_refine calls the native rereceive when `em.rereceive` holds this file's
row (marked `__qrss_native__`); anything else there -- a test's spy, as
test_qrss_em.py's P4 smoke and test_qrss_native_store.py's EM run install,
or the reference when this row is not applied -- is handed to the C++ as
the seam `_seams["rereceive"]` and called the way the reference calls it,
`rereceive(p, prior, hdr, cw_known, estimator, r, path=..., return_path=True)`,
with the reference's objects (`_rx.to_ref`). It is resolved once per
em_refine call, which is when the reference reads the module attribute
too (at each call inside the loop, but nothing in a loop changes it).

The store is the native one (tests/qrss_native/store.py's facade), which
the C++ reads directly; results come back as the reference's objects: a
PassResult and FreqPath dataclass, and em_refine's history a list of
floats. `log` is the reference's callable; `progress` and `cancel` (the
C++'s hooks, sstvae_native.qrss.em.CancelFlag) are keywords the reference
does not have and the rows do not pass.
"""

STAGE = "rx"


def rows(q):
    from sstvae.qrss import em as E

    from . import _rx

    n = q.em_refine
    to_ref = _rx.to_ref

    def _native(fn):
        return getattr(fn, "__qrss_native__", False)

    def seams():
        """What em_refine is to call in Python: rereceive, unless the
        attribute is this file's row."""
        if _native(E.rereceive):
            return None
        fn = E.rereceive

        def rr(p, prior, hdr, cw_known, estimator, r, path=None, return_path=True):
            return fn(to_ref(p), to_ref(prior), to_ref(hdr), cw_known, estimator, r, path=to_ref(path),
                      return_path=return_path)
        return {"rereceive": rr}

    def pass_path(p, chan=None):
        return to_ref(n.pass_path(p, chan))

    def rereceive(p, prior, hdr=None, cw_known=None, estimator=None, em_round=None, path=None,
                  return_path=False, split=True):
        r = n.rereceive(p, prior, hdr, cw_known, estimator, em_round, path, return_path, split)
        if return_path:
            new, pth = r
            return to_ref(new), to_ref(pth)
        return to_ref(r)

    def em_refine(store, key, rounds=E.EM_ROUNDS, tol_db=E.EM_TOL_DB, *, estimator=None, use_header=True,
                  log=None):
        return n.em_refine(store, key, rounds, tol_db, estimator=estimator, use_header=use_header, log=log,
                           _seams=seams())

    for fn in (pass_path, rereceive, em_refine):
        fn.__qrss_native__ = True

    t = "sstvae.qrss.em"
    return {
        (t, "pass_path"): pass_path,
        (t, "rereceive"): rereceive,
        (t, "em_refine"): em_refine,
    }
