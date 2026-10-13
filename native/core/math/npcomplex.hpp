// numpy's elementwise complex arithmetic as numpy 2.5 computes it on the
// machines the reference runs on (x86-64 with AVX2 and FMA3, numpy's
// X86_V3 dispatch target), for the receive-side ports that are held to the
// bit against numpy.
//
// Measured, not assumed (numpy 2.5.3, this container's CPU):
//
// * multiply (complex128 and complex64): numpy's SIMD loop fuses one
//   product of each part: re = fma(ar, br, -(ai*bi)), im = fma(ar, bi,
//   ai*br). The textbook (ar*br - ai*bi) disagrees in about a quarter of
//   the results. A real operand is promoted to (x, 0) first, so real times
//   complex comes out as the plain products except for the sign of an
//   underflowed zero -- `mul` below does the fused form throughout.
// * divide: the scalar loop (Smith's method, no fusion): for |br| >= |bi|,
//   rat = bi/br, scl = 1/(br + bi*rat), re = (ar + ai*rat)*scl, im =
//   (ai - ar*rat)*scl -- so dividing by a real is a multiply by its
//   reciprocal, not two divisions (which differ in ~43% of samples).
// * absolute: numpy's SIMD hypot, larger * sqrt(fma(r, r, 1)) with r =
//   smaller/larger (0 when larger is 0 or smaller is inf), inf for an inf
//   part, NaN for a NaN part otherwise. Not libm's hypot (35% differ).
//
// A host whose numpy takes a non-FMA path (an x86-64 without AVX2, say)
// would compute the unfused forms; these are what the reference produced
// where the tests were measured, and std::fma makes them the same on
// every platform the C++ runs on.

#ifndef SSTVAE_MATH_NPCOMPLEX_HPP
#define SSTVAE_MATH_NPCOMPLEX_HPP

#include <cmath>
#include <complex>
#include <limits>

namespace sstvae::math::np {

// a * b, numpy's complex multiply.
template <typename T>
inline std::complex<T> mul(std::complex<T> a, std::complex<T> b) {
    const T ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
    return {std::fma(ar, br, -(ai * bi)), std::fma(ar, bi, ai * br)};
}

// a * x for a real x promoted to (x, 0) first, as numpy promotes it.
template <typename T>
inline std::complex<T> mul(std::complex<T> a, T x) {
    return mul(a, std::complex<T>(x, T(0)));
}

// a / b, numpy's complex divide (Smith's method).
template <typename T>
inline std::complex<T> div(std::complex<T> a, std::complex<T> b) {
    const T ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
    const T br_abs = std::fabs(br), bi_abs = std::fabs(bi);
    if (br_abs >= bi_abs) {
        if (br_abs == T(0) && bi_abs == T(0)) return {ar / br_abs, ai / br_abs};
        const T rat = bi / br;
        const T scl = T(1) / (br + bi * rat);
        return {(ar + ai * rat) * scl, (ai - ar * rat) * scl};
    }
    const T rat = br / bi;
    const T scl = T(1) / (bi + br * rat);
    return {(ar * rat + ai) * scl, (ai * rat - ar) * scl};
}

// a / x for a real x promoted to (x, 0).
template <typename T>
inline std::complex<T> div(std::complex<T> a, T x) {
    return div(a, std::complex<T>(x, T(0)));
}

// np.abs(a): numpy's SIMD complex absolute value.
template <typename T>
inline T abs(std::complex<T> a) {
    constexpr T inf = std::numeric_limits<T>::infinity();
    constexpr T nan = std::numeric_limits<T>::quiet_NaN();
    T re = std::fabs(a.real()), im = std::fabs(a.imag());
    const bool re_inf = re == inf, im_inf = im == inf;
    if (re_inf) im = inf;
    if (im_inf) re = inf;
    const bool re_ok = !std::isnan(re), im_ok = !std::isnan(im);
    if (!re_ok) im = nan;
    if (!im_ok) re = nan;
    // npyv_max/min: the second operand where the comparison fails (NaN)
    const T larger = re > im ? re : im;
    const T smaller = im < re ? im : re;
    const bool div_ok = !(larger == T(0) || smaller == inf);
    const T ratio = div_ok ? smaller / larger : T(0);
    return std::sqrt(std::fma(ratio, ratio, T(1))) * larger;
}

}  // namespace sstvae::math::np

#endif  // SSTVAE_MATH_NPCOMPLEX_HPP
