// scipy.interpolate.CubicSpline of complex values, bc_type "not-a-knot",
// extrapolating, as acquire.preamble_search builds and evaluates one:
// `CubicSpline(np.arange(lo, hi), chan.m[lo:hi])` called at fractional
// sample positions (and its derivatives, spline(x, nu)).
//
// The arithmetic is scipy 1.18.1's, operation for operation, so the
// result is scipy's to the bit (tests/test_qrss_native_cubic_spline.py)
// for n != 3 points:
//   - the slopes and Hermite coefficients divide a complex128 by a
//     float64 the way numpy does -- promoted to complex and divided by
//     numpy's Smith loop, which multiplies by a reciprocal
//     (re = (a + b*rat) * (1/c)) rather than dividing;
//   - the not-a-knot tridiagonal system is solved by an emulation of
//     LAPACK zgtsv (scipy.linalg.solve_banded((1, 1), ...) on complex
//     right-hand sides): partial pivoting on |re| + |im|, Fortran complex
//     products and divisions, the same back substitution;
//   - evaluation is _ppoly.evaluate's evaluate_poly1: a power series in
//     s = x - x[i] accumulated in C complex arithmetic.
// For n == 3 scipy solves a 3x3 dense system (scipy.linalg.solve); that
// is done here by Gaussian elimination with partial pivoting and agrees
// to rounding only (the receiver never builds a 3-point spline).
//
// The checks and their ValueError texts are scipy's prepare_input's, in
// its order; a singular system raises math::LinAlgError("singular
// matrix") as solve_banded does.
//
// Immutable once built: evaluation is thread-safe.

#ifndef SSTVAE_MATH_CUBIC_SPLINE_HPP
#define SSTVAE_MATH_CUBIC_SPLINE_HPP

#include <complex>
#include <cstddef>
#include <span>
#include <vector>

namespace sstvae::math {

class CubicSpline {
public:
    // CubicSpline(x, y) with bc_type="not-a-knot", extrapolate=True.
    // x strictly increasing and finite, y finite, len(x) == len(y) >= 2
    // (std::invalid_argument with scipy's message otherwise).
    CubicSpline(std::span<const double> x, std::span<const std::complex<double>> y);

    // spline(xv, nu): the nu-th derivative (nu >= 0; std::invalid_argument
    // "Order of derivative cannot be negative" otherwise) at xv, the
    // first/last piece extended outside [x[0], x[-1]], NaN (+0j) at a NaN.
    std::complex<double> operator()(double xv, int nu = 0) const;
    std::vector<std::complex<double>> operator()(std::span<const double> xv, int nu = 0) const;

    // The breakpoints and scipy's PPoly coefficients c[k, i] (k = 0 the
    // cubic term), row-major: c()[k * (size() - 1) + i].
    const std::vector<double>& x() const { return x_; }
    const std::vector<std::complex<double>>& c() const { return c_; }
    std::size_t size() const { return x_.size(); }

private:
    std::size_t interval(double xv) const;
    std::vector<double> x_;
    std::vector<std::complex<double>> c_;
};

}  // namespace sstvae::math

#endif  // SSTVAE_MATH_CUBIC_SPLINE_HPP
