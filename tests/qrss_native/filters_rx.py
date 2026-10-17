"""The receiver's scipy filters -> sstvae_native.qrss.filters
(native/core/qrss/filters).

`frontend`, `acquire` and `cwid` call scipy through the module object
(`signal.firwin(...)`, `signal.resample_poly(...)`,
`signal.windows.hann(...)`, `special.i0e(...)`), so assigning over a
function cannot reach them without patching scipy itself for everyone.
Each of those modules' `signal`/`special` attribute is replaced instead by
an overlay: the ported functions are the native ones and every other name
falls through to scipy unchanged (oaconvolve, freqz, gammaincc, ...).

`frontend._FE_TAPS` is computed by firwin at import, before any row is
applied, so it is a row of its own.
"""

import scipy.signal
import scipy.special

STAGE = "rx"


class _Overlay:
    """A module stand-in: `names` first, then `base`."""

    def __init__(self, base, **names):
        self.__dict__["_base"] = base
        self.__dict__.update(names)

    def __getattr__(self, name):
        return getattr(self.__dict__["_base"], name)

    def __repr__(self):
        return f"<native overlay of {self.__dict__['_base'].__name__}>"


def rows(q):
    from sstvae.qrss.constants import FS

    f = q.filters
    windows = _Overlay(scipy.signal.windows, hann=f.hann, hamming=f.hamming, kaiser=f.kaiser,
                       get_window=f.get_window)
    signal = _Overlay(scipy.signal, firwin=f.firwin, resample_poly=f.resample_poly,
                      upfirdn=f.upfirdn, get_window=f.get_window, windows=windows)
    special = _Overlay(scipy.special, i0=f.i0, i0e=f.i0e)
    return {
        ("sstvae.qrss.frontend", "signal"): signal,
        ("sstvae.qrss.frontend", "_FE_TAPS"): f.firwin(255, 1250.0, fs=FS, window=("kaiser", 8.0)),
        ("sstvae.qrss.acquire", "signal"): signal,
        ("sstvae.qrss.cwid", "special"): special,
    }
