"""sstvae.qrss.acquire's shared statistics -> sstvae_native.qrss.acquire_stats
(native/core/qrss/acquire_stats), and the CubicSpline preamble_search
builds -> sstvae_native.qrss.cubic_spline (native/core/math/cubic_spline).

The pure-array functions A1, A2 and A3 have in common: the STFT frames
and their CFAR normalisation, the A1/A2 Gamma thresholds and A2's
p-values (scipy.special's gammaincc/gammainccinv, ported in
native/core/math/special), A3's Gumbel table and band covariance, and the
vertex parabola. Each is assigned over acquire's attribute of the same
name, so acquire's own callers (and receiver's, through the module) run
the C++.

`CubicSpline` is the name acquire.py imports from scipy.interpolate; only
preamble_search calls it (complex values, not-a-knot), and the native
class refuses anything else with NotImplementedError.

`acquire.special` and `acquire.ndimage` are not overlaid: every use of
them in acquire.py is inside a function replaced here (gammaincc in
preamble_p, gammainccinv in a1_threshold/a2_threshold, median_filter in
spectral_shape), so an overlay would be dead code. The ports themselves
are checked against scipy by tests/test_qrss_native_special.py and
test_qrss_native_ndfilter.py.
"""

STAGE = "rx"


def rows(q):
    a = q.acquire_stats
    return {
        ("sstvae.qrss.acquire", "_frame_len"): a._frame_len,
        ("sstvae.qrss.acquire", "frame_window"): a.frame_window,
        ("sstvae.qrss.acquire", "spectrogram"): a.spectrogram,
        ("sstvae.qrss.acquire", "band_bins"): a.band_bins,
        ("sstvae.qrss.acquire", "cfar_normalise"): a.cfar_normalise,
        ("sstvae.qrss.acquire", "spectral_shape"): a.spectral_shape,
        ("sstvae.qrss.acquire", "_parabola"): a._parabola,
        ("sstvae.qrss.acquire", "a1_gamma"): a.a1_gamma,
        ("sstvae.qrss.acquire", "a1_threshold"): a.a1_threshold,
        ("sstvae.qrss.acquire", "a2_threshold"): a.a2_threshold,
        ("sstvae.qrss.acquire", "preamble_scales"): a.preamble_scales,
        ("sstvae.qrss.acquire", "preamble_p"): a.preamble_p,
        ("sstvae.qrss.acquire", "tbd_gumbel"): a.tbd_gumbel,
        ("sstvae.qrss.acquire", "gumbel_p"): a.gumbel_p,
        ("sstvae.qrss.acquire", "_band_cov"): a._band_cov,
        ("sstvae.qrss.acquire", "CubicSpline"): q.cubic_spline.CubicSpline,
    }
