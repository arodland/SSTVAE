// scipy.special's regularised incomplete Gamma functions as the QRSSTVAE
// receiver calls them: gammaincc and gammainccinv (acquire.py's A1 and A2
// false-alarm thresholds and A2's per-cell p-values), with gammainc and
// gammaincinv beside them because the inverses call each other.
//
// A port of what scipy 1.18.1 runs for the float64 loops of these ufuncs:
// xsf (scipy's special-function library) `gammaincc` = cephes::igamc,
// `gammainccinv` = cephes::igamci, and everything they reach -- igam's
// power series, the continued fraction, Temme's uniform asymptotic
// expansion (with its 25x25 coefficient table), DiDonato & Morris's
// initial guess for the inverse and three Halley steps, cephes lgam,
// Gamma, erfc, zeta, log1pmx, expm1, lgam1p and the Lanczos sum. The same
// operations in the same order, built without FMA contraction, over the
// same libm calls (std::exp, log, pow, sqrt, log1p) that scipy's compiled
// code makes: on Linux/glibc x86-64 the results are scipy's to the bit
// (tests/test_qrss_native_special.py), elsewhere to the libm's own ulp.
//
// The integer-shape closed form (gammaincc(k, x) = e^-x sum x^i / i!) is
// not used: it agrees only to ~1e-14, and the receiver's argmin over
// p-values deserves the reference's own numbers.
//
// What scipy does that is kept: NaN for a NaN argument; NaN for a domain
// error (a < 0, x < 0, q outside [0, 1]) -- scipy only records an error
// flag, which is off by default; gammaincc(0, x) = 0 for x > 0, NaN at 0;
// gammaincc(a, 0) = 1; gammaincc(inf, x) = 1 (NaN at x = inf);
// gammaincc(a, inf) = 0; gammainccinv(a, 0) = inf, gammainccinv(a, 1) = 0.
// The inverse's domain errors fall through into the computation, exactly
// as xsf's igamci does, and come out NaN the same way.
//
// Thread-safe: pure functions, constant tables only.

#ifndef SSTVAE_MATH_SPECIAL_HPP
#define SSTVAE_MATH_SPECIAL_HPP

#include <span>
#include <vector>

namespace sstvae::math {

// scipy.special.gammaincc(a, x): the regularised upper incomplete Gamma
// function Q(a, x) = Gamma(a, x) / Gamma(a).
double gammaincc(double a, double x);
std::vector<double> gammaincc(double a, std::span<const double> x);

// scipy.special.gammainc(a, x): P(a, x) = 1 - Q(a, x).
double gammainc(double a, double x);

// scipy.special.gammainccinv(a, q): x with gammaincc(a, x) = q.
double gammainccinv(double a, double q);

// scipy.special.gammaincinv(a, p): x with gammainc(a, x) = p.
double gammaincinv(double a, double p);

// The cephes pieces underneath, for tests and later stages (stage 4's
// gamma.logsf): log|Gamma(x)| (cephes lgam), Gamma(x), erfc(x), and the
// Hurwitz zeta(x, q).
double lgam(double x);
double gamma(double x);
double erfc(double x);
double zeta(double x, double q);

}  // namespace sstvae::math

#endif  // SSTVAE_MATH_SPECIAL_HPP
