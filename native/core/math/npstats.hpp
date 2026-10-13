// numpy's statistics and 1-D fitting as the QRSSTVAE receiver calls them:
// np.median, np.mean, np.interp, np.polyfit and np.polyval on 1-D float
// arrays (numpy 2.5, the version tests/ pins).
//
// Bit-exact against numpy: median, mean, interp and polyval (the same
// operations in the same order, no FMA -- core/math is built with
// -ffp-contract=off). polyfit is numpy's algorithm (Vandermonde columns by
// repeated multiplication, weighting, column scaling, lstsq at numpy's
// rcond, unscaling) over linalg.hpp's lstsq, which is a Jacobi SVD rather
// than LAPACK's gelsd: agreement is to rounding (~1e-15 relative on a
// well-conditioned fit), not bitwise.
//
// What numpy does that is copied here, because callers can see it:
//
// * median: NaN for an empty array and for any array holding a NaN; the
//   middle one or two values averaged through np.mean's own arithmetic
//   (so a float32 median is float32(float64(a + b) / 2), and -0.0 alone
//   comes back as 0.0, as numpy's sum onto 0 does).
// * interp: numpy's binary_search_with_guess (its guess carried from one x
//   to the next, which matters only where xp repeats a value), fp[j] where
//   x hits xp[j] exactly, the slope retried from the right end when the
//   left one gives NaN, NaN for a NaN x, and left/right default fp[0] and
//   fp[-1]. "array of sample points is empty" and "fp and xp are not of
//   the same length." are std::invalid_argument (ValueError).
// * polyfit: x, y and w go through `+ 0.0` first (so -0.0 becomes 0.0);
//   "expected deg >= 0" is std::invalid_argument; "expected non-empty
//   vector for x", "expected x and y to have same length" and "expected w
//   and y to have the same length" are math::TypeError; a non-finite
//   value reaching the scaled matrix (NaN or inf in x or w, or a column
//   whose scale is 0) is LinAlgError "SVD did not converge in Linear Least
//   Squares", while a non-finite y only makes the coefficients NaN.
// * polyval: y = 0, then y = y * x + p for each coefficient, so an
//   infinite x gives NaN (0 * inf) exactly as numpy's does.

#ifndef SSTVAE_MATH_NPSTATS_HPP
#define SSTVAE_MATH_NPSTATS_HPP

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace sstvae::math {

// np.median(x) of a 1-D float64 / float32 array.
double median(std::span<const double> x);
float median(std::span<const float> x);

// np.mean(x): numpy's pairwise sum over the count (NaN for an empty array).
// float32: float32(float64(sum32) / n), numpy's two roundings.
double mean(std::span<const double> x);
float mean(std::span<const float> x);

// np.interp(x, xp, fp, left, right) for real float64 arrays (no period).
// xp is expected increasing, as numpy expects it; nothing checks it.
double interp(double x, std::span<const double> xp, std::span<const double> fp,
              std::optional<double> left = std::nullopt, std::optional<double> right = std::nullopt);
std::vector<double> interp(std::span<const double> x, std::span<const double> xp,
                           std::span<const double> fp, std::optional<double> left = std::nullopt,
                           std::optional<double> right = std::nullopt);

// np.polyfit(x, y, deg, rcond=rcond, w=w): deg + 1 coefficients, highest
// power first. rcond nullopt is numpy's default, len(x) * eps.
std::vector<double> polyfit(std::span<const double> x, std::span<const double> y, int deg,
                            std::optional<std::span<const double>> w = std::nullopt,
                            std::optional<double> rcond = std::nullopt);

// np.polyval(p, x), p highest power first.
double polyval(std::span<const double> p, double x);
std::vector<double> polyval(std::span<const double> p, std::span<const double> x);

}  // namespace sstvae::math

#endif  // SSTVAE_MATH_NPSTATS_HPP
