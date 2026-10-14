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

// --- stage 4: the template search's tail statistics (em.template_z) ----------

// scipy.special.ndtri(y): x with ndtr(x) = y, xsf's cephes::ndtri (its
// three rational approximations, the same polevl/p1evl). -inf at 0, +inf
// at 1, NaN outside [0, 1] and for NaN.
double ndtri(double y);

// scipy.stats.norm.isf(q) as rv_continuous.isf computes it for norm:
// -ndtri(q) * 1 + 0 for 0 < q < 1 (so isf(0.5) is +0.0), +inf at q = 0,
// -inf at q = 1, NaN (norm.badvalue) outside [0, 1] and for NaN.
double norm_isf(double q);

// scipy.stats.gamma.logsf(x, a) with loc 0 and scale 1, as scipy 1.18.1
// computes it: gamma_gen has no _logsf of its own, so this is
// rv_continuous.logsf's argument handling (NaN for a NaN x or a shape
// that is not > 0; 0 for x <= 0; -inf where x is not inside the open
// support, i.e. x = +inf) around rv_continuous._logsf, which splits at the
// median gammaincinv(a, 0.5): log(gammaincc(a, x)) above it,
// log1p(-gammainc(a, x)) at or below it. -inf where gammaincc underflows
// to 0 (x a few thousand past a, e.g. a = 300, x = 4800). The logs are the
// C library's where numpy's are its own SIMD ones on an AVX-512 machine
// (np.log, and SVML's np.log1p): those differ in the last place for some
// arguments, so this agrees with scipy to a few ulp, not to the bit.
double gamma_logsf(double x, double a);

// Python's math.lgamma(x): CPython's own Lanczos approximation
// (Modules/mathmodule.c m_lgamma, not the C library's lgamma, from which
// it differs in the last place for about half of all arguments), with its
// special cases: NaN for NaN, +inf for +-inf, 0 for 1 and 2. Where
// Python raises -- ValueError "math domain error" at an integer <= 0,
// OverflowError "math range error" when a finite x overflows -- this
// returns +inf; the caller that wants Python's exception raises it
// (em::template_z does).
double py_lgamma(double x);

}  // namespace sstvae::math

#endif  // SSTVAE_MATH_SPECIAL_HPP
