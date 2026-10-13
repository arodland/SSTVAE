// scipy.interpolate's B-splines as the QRSSTVAE receiver uses them:
// make_lsq_spline (method "qr", scipy 1.18.1) and BSpline evaluation and
// differentiation (extrapolate=True), for 1-D float64 data.
//
// The same algorithms in the same order as scipy's C++ (scipy/interpolate/
// src/__fitpack.cc: _deBoor_D, _find_interval, data_matrix, qr_reduce,
// fpback, evaluate_spline) and Python (_bsplines.make_lsq_spline,
// _fitpack_impl.splder), with LAPACK 3.12's dlartg (the scaled f90 Givens
// rotation scipy's OpenBLAS ships) -- so the coefficients are scipy's to
// the bit wherever OpenBLAS's dlartg is the reference one, and evaluation
// is bit-exact given the coefficients. Built with -ffp-contract=off.
//
// Behaviour kept from scipy because callers see it:
//
// * The checks, in scipy's order, with its ValueError texts
//   (std::invalid_argument): "Array must not contain infs or nans." (x, y,
//   t, w), "Need more x points.", "Expect non-negative k.", "Expect t to be
//   a 1D strictly increasing sequence." (t may repeat: only a decrease is
//   refused), "Shapes of x (m,) and y (p,) are incompatible", "Out of
//   bounds w/ x = [...]." (x outside [t[k], t[-k]]; the array is printed as
//   numpy would for short arrays of round values only -- the text is
//   approximate, the condition exact), "Shapes of x (m,) and w (p,) are
//   incompatible", "Expect x to be a 1D non-decreasing sequence.", and
//   fpback's "nc = <n> > m = <m>" when there are more coefficients than
//   points. A t too short to index t[k] is std::out_of_range (IndexError).
// * Knot intervals without data are not refused: the QR reduction leaves a
//   zero on R's diagonal and back substitution divides by it, so the
//   coefficients come out inf/NaN -- which is how track.PathFn notices a
//   Schoenberg-Whitney violation. A row whose first nonzero column is past
//   its own row index is rotated as if it were not (scipy's qr_reduce sets
//   offset[i] = i regardless), so some violations give finite, wrong
//   coefficients instead; that is scipy's behaviour and is reproduced.
// * Evaluation outside [t[k], t[n]] extrapolates the end polynomial; a NaN
//   x gives NaN. derivative() is splder: c' = (c[1:] - c[:-1]) * k / dt,
//   refusing a zero dt with scipy's "The spline has internal repeated
//   knots and is not differentiable 1 times".

#ifndef SSTVAE_MATH_BSPLINE_HPP
#define SSTVAE_MATH_BSPLINE_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace sstvae::math {

// BSpline.construct_fast(t, c, k) with extrapolate=True: n = len(t) - k - 1
// coefficients are used; c may be longer (FITPACK's padding), never shorter.
struct BSpline {
    std::vector<double> t;
    std::vector<double> c;
    int k = 3;

    // spl(x)
    double operator()(double x) const;
    std::vector<double> operator()(std::span<const double> x) const;
    // spl.derivative(nu): a spline of degree k - nu (std::invalid_argument
    // for nu > k, or a zero knot span under differentiation).
    BSpline derivative(int nu = 1) const;
    bool operator==(const BSpline&) const = default;
};

// scipy.interpolate.make_lsq_spline(x, y, t, k, w) (axis 0, check_finite,
// method "qr"). w nullopt is all ones.
BSpline make_lsq_spline(std::span<const double> x, std::span<const double> y, std::span<const double> t,
                        int k = 3, std::optional<std::span<const double>> w = std::nullopt);

// The pieces, exposed for the tests and for code that evaluates many
// points against one knot vector.

// _find_interval(t, k, x, prev_l, extrapolate): t[l] <= x < t[l+1] with l
// in [k, n-1]; -1 for NaN, or outside [t[k], t[n]] without extrapolation.
std::int64_t find_interval(std::span<const double> t, int k, double x, std::int64_t prev_l, bool extrapolate);

// _deBoor_D(t, x, k, ell, m, result): result (2k + 2 doubles) receives the
// k + 1 nonzero B-splines (m-th derivative) at x in interval ell.
void deboor_d(std::span<const double> t, double x, int k, std::int64_t ell, int m, double* result);

// LAPACK's dlartg (3.10+): c, s, r with [c s; -s c] [f; g] = [r; 0].
struct Givens {
    double c, s, r;
};
Givens dlartg(double f, double g);

}  // namespace sstvae::math

#endif  // SSTVAE_MATH_BSPLINE_HPP
