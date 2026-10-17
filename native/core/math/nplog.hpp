// np.log on float32 as numpy 2.5 computes it on x86-64 with AVX2/FMA3 or
// AVX-512F (numpy/_core/src/umath/loops_exponent_log.dispatch.c.src,
// simd_log_FLOAT): not libm's logf but numpy's own SIMD algorithm, which
// numpy runs for every float32 element on those machines -- a lone scalar
// included ("We use the AVX function to compute exp/log for scalar
// elements as well"). Its results differ from a correctly rounded logf in
// about 5% of inputs, so a port that calls std::log disagrees with the
// reference wherever a float32 log feeds a decision (acquire.carrier_lines'
// vertex parabola on the log statistic).
//
// The algorithm, transcribed:
//
//   x = m * 2^e with m in [0.5, 1) (frexp; numpy's getmant/getexp, and
//   its 2^100 rescale of a denormal, give the same m and e);
//   if m <= 1/sqrt(2): m = m + m, e = e - 1;
//   t = m - 1;
//   log(x) = fma(e, ln 2, P(t) / Q(t)), P and Q degree-5 polynomials by
//   Horner's rule in fused multiply-adds, everything in float32;
//
// then the special cases, applied in numpy's order: NaN -> NaN, x < 0
// (-inf included) -> -NaN, +-0 -> -inf, +inf -> +inf.
//
// A numpy built without those kernels (a non-x86 machine, or an x86-64
// without AVX2) uses npy_logf, i.e. libm; this is what the reference
// computes where its tests were measured, and std::fma makes it the same
// number on every platform the C++ runs on.

#ifndef SSTVAE_MATH_NPLOG_HPP
#define SSTVAE_MATH_NPLOG_HPP

#include <cmath>
#include <limits>

namespace sstvae::math::np {

inline float log_f32(float x) {
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    constexpr float inf = std::numeric_limits<float>::infinity();
    if (std::isnan(x)) return nan;
    if (x < 0.0f) return -nan;
    if (x == 0.0f) return -inf;
    if (x == inf) return inf;
    // NPY_COEFF_{P,Q}{0..5}_LOGf, NPY_LOGE2f, NPY_SQRT1_2f (npy_simd_data.h,
    // npy_math.h), as float literals exactly as numpy spells them.
    constexpr float P0 = 0.000000000000000000000e+00f;
    constexpr float P1 = 9.999999999999998702752e-01f;
    constexpr float P2 = 2.112677543073053063722e+00f;
    constexpr float P3 = 1.480000633576506585156e+00f;
    constexpr float P4 = 3.808837741388407920751e-01f;
    constexpr float P5 = 2.589979117907922693523e-02f;
    constexpr float Q0 = 1.000000000000000000000e+00f;
    constexpr float Q1 = 2.612677543073109236779e+00f;
    constexpr float Q2 = 2.453006071784736363091e+00f;
    constexpr float Q3 = 9.864942958519418960339e-01f;
    constexpr float Q4 = 1.546476374983906719538e-01f;
    constexpr float Q5 = 5.875095403124574342950e-03f;
    constexpr float LOGE2 = 0.693147180559945309417232121458176568f;
    constexpr float SQRT1_2 = 0.707106781186547524400844362104849039f;
    int e = 0;
    float m = std::frexp(x, &e);
    float exponent = static_cast<float>(e);
    if (m <= SQRT1_2) {
        m = m + m;
        exponent = exponent - 1.0f;
    }
    const float t = m - 1.0f;
    float num = std::fma(P5, t, P4);
    num = std::fma(num, t, P3);
    num = std::fma(num, t, P2);
    num = std::fma(num, t, P1);
    num = std::fma(num, t, P0);
    float den = std::fma(Q5, t, Q4);
    den = std::fma(den, t, Q3);
    den = std::fma(den, t, Q2);
    den = std::fma(den, t, Q1);
    den = std::fma(den, t, Q0);
    const float poly = num / den;
    return std::fma(exponent, LOGE2, poly);
}

}  // namespace sstvae::math::np

#endif  // SSTVAE_MATH_NPLOG_HPP
