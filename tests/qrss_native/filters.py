"""scipy filters the transmitter takes by name -> sstvae_native.qrss.filters
(native/core/qrss/filters).

`ce` and `lwave` import `upfirdn`, `resample_poly` and `firwin` from
scipy.signal by name, so each of those module attributes is a row. The
receiver's uses go through the `scipy.signal` module object instead and
are in filters_rx.py.
"""

STAGE = "tx"


def rows(q):
    f = q.filters
    return {
        ("sstvae.qrss.ce", "upfirdn"): f.upfirdn,
        ("sstvae.qrss.ce", "resample_poly"): f.resample_poly,
        ("sstvae.qrss.lwave", "upfirdn"): f.upfirdn,
        ("sstvae.qrss.lwave", "firwin"): f.firwin,
    }
