// scipy.interpolate.CubicSpline (complex values, not-a-knot): see
// cubic_spline.hpp. Follows scipy 1.18.1's _cubic.py (prepare_input,
// CubicSpline.__init__, CubicHermiteSpline.__init__), linalg/_basic.py's
// solve_banded -> LAPACK zgtsv, and interpolate/_ppoly.pyx's evaluate /
// evaluate_poly1. SciPy: Copyright (c) 2001-2002 Enthought, Inc.
// 2003, SciPy Developers, BSD-3-Clause. LAPACK zgtsv: Univ. of
// Tennessee, Univ. of California Berkeley, Univ. of Colorado Denver and
// NAG Ltd., BSD-3-Clause.
//
// Complex arithmetic is spelled out on a plain {re, im} pair, never
// std::complex's operators, because three different conventions meet
// here and each must be the reference's own:
//   numpy's complex division (Smith, multiplying by a reciprocal),
//   Fortran's (gfortran -fcx-fortran-rules: naive products, Smith
//     division that divides), and
//   C's (Cython evaluate_poly1: naive products).
// The file is built without FMA contraction (-ffp-contract=off on
// sstvae_qrss), which every one of those formulas assumes.

#include "math/cubic_spline.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

#include "math/errors.hpp"

namespace sstvae::math {
namespace {

struct Cx {
    double r;
    double i;
};

Cx add(Cx a, Cx b) { return {a.r + b.r, a.i + b.i}; }
Cx sub(Cx a, Cx b) { return {a.r - b.r, a.i - b.i}; }
Cx neg(Cx a) { return {-a.r, -a.i}; }

// (a.r + i a.i)(b.r + i b.i), naive: numpy's, C's and gfortran's product.
Cx mul(Cx a, Cx b) { return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r}; }

// A float64 promoted to complex128, as numpy does for `real * complex`.
Cx real(double v) { return {v, 0.0}; }

// numpy's complex128 divide loop (numpy/_core/src/umath/loops.c.src).
Cx np_div(Cx a, Cx b) {
    const double br_abs = std::fabs(b.r), bi_abs = std::fabs(b.i);
    if (br_abs >= bi_abs) {
        if (br_abs == 0 && bi_abs == 0) return {a.r / br_abs, a.i / br_abs};
        const double rat = b.i / b.r;
        const double scl = 1.0 / (b.r + b.i * rat);
        return {(a.r + a.i * rat) * scl, (a.i - a.r * rat) * scl};
    }
    const double rat = b.r / b.i;
    const double scl = 1.0 / (b.i + b.r * rat);
    return {(a.r * rat + a.i) * scl, (a.i * rat - a.r) * scl};
}

// gfortran's complex division (GCC expand_complex_div_wide).
Cx f_div(Cx a, Cx b) {
    if (std::fabs(b.r) < std::fabs(b.i)) {
        const double ratio = b.r / b.i;
        const double div = (b.r * ratio) + b.i;
        const double tr = (a.r * ratio) + a.i;
        const double ti = (a.i * ratio) - a.r;
        return {tr / div, ti / div};
    }
    const double ratio = b.i / b.r;
    const double div = (b.i * ratio) + b.r;
    const double tr = (a.i * ratio) + a.r;
    const double ti = a.i - (a.r * ratio);
    return {tr / div, ti / div};
}

double cabs1(Cx a) { return std::fabs(a.r) + std::fabs(a.i); }
bool is_zero(Cx a) { return a.r == 0.0 && a.i == 0.0; }

// LAPACK zgtsv, one right-hand side: solve the tridiagonal system with
// sub-diagonal dl (n-1), diagonal d (n), super-diagonal du (n-1), in place
// in b. Returns INFO (0, or k > 0 when U(k, k) is exactly zero).
int zgtsv(std::vector<Cx>& dl, std::vector<Cx>& d, std::vector<Cx>& du, std::vector<Cx>& b) {
    const std::size_t n = d.size();
    if (n == 0) return 0;
    for (std::size_t k = 0; k + 1 < n; ++k) {
        if (is_zero(dl[k])) {
            if (is_zero(d[k])) return static_cast<int>(k + 1);
        } else if (cabs1(d[k]) >= cabs1(dl[k])) {
            const Cx mult = f_div(dl[k], d[k]);
            d[k + 1] = sub(d[k + 1], mul(mult, du[k]));
            b[k + 1] = sub(b[k + 1], mul(mult, b[k]));
            if (k + 2 < n) dl[k] = {0.0, 0.0};
        } else {
            const Cx mult = f_div(d[k], dl[k]);
            d[k] = dl[k];
            Cx temp = d[k + 1];
            d[k + 1] = sub(du[k], mul(mult, temp));
            if (k + 2 < n) {
                dl[k] = du[k + 1];
                du[k + 1] = neg(mul(mult, dl[k]));
            }
            du[k] = temp;
            temp = b[k];
            b[k] = b[k + 1];
            b[k + 1] = sub(temp, mul(mult, b[k + 1]));
        }
    }
    if (is_zero(d[n - 1])) return static_cast<int>(n);
    b[n - 1] = f_div(b[n - 1], d[n - 1]);
    if (n > 1) b[n - 2] = f_div(sub(b[n - 2], mul(du[n - 2], b[n - 1])), d[n - 2]);
    for (std::size_t kk = n - 2; kk-- > 0;)
        b[kk] = f_div(sub(sub(b[kk], mul(du[kk], b[kk + 1])), mul(dl[kk], b[kk + 2])), d[kk]);
    return 0;
}

// The 3-point not-a-knot system (scipy.linalg.solve(A, b) of a dense
// real 3x3 A): Gaussian elimination with partial pivoting. Agrees with
// LAPACK to rounding, not to the bit.
std::vector<Cx> solve3(double A[3][3], std::vector<Cx> b) {
    int perm[3] = {0, 1, 2};
    for (int col = 0; col < 3; ++col) {
        int p = col;
        for (int r = col + 1; r < 3; ++r)
            if (std::fabs(A[perm[r]][col]) > std::fabs(A[perm[p]][col])) p = r;
        std::swap(perm[col], perm[p]);
        const double piv = A[perm[col]][col];
        if (piv == 0.0) throw LinAlgError("Matrix is singular.");
        for (int r = col + 1; r < 3; ++r) {
            const double f = A[perm[r]][col] / piv;
            for (int c = col; c < 3; ++c) A[perm[r]][c] -= f * A[perm[col]][c];
            b[static_cast<std::size_t>(perm[r])] =
                sub(b[static_cast<std::size_t>(perm[r])], mul(real(f), b[static_cast<std::size_t>(perm[col])]));
        }
    }
    std::vector<Cx> x(3);
    for (int r = 2; r >= 0; --r) {
        Cx acc = b[static_cast<std::size_t>(perm[r])];
        for (int c = r + 1; c < 3; ++c) acc = sub(acc, mul(real(A[perm[r]][c]), x[static_cast<std::size_t>(c)]));
        x[static_cast<std::size_t>(r)] = np_div(acc, real(A[perm[r]][r]));
    }
    return x;
}

bool finite(Cx a) { return std::isfinite(a.r) && std::isfinite(a.i); }

// `dxr[0]**2` in CubicSpline.__init__ squares an np.float64 *scalar*, and
// numpy's scalar power calls libm pow, which is not x*x for every x (about
// 1 in 1200 uniform draws differ by an ulp; an array's ** 2 does square).
// GCC folds pow(x, 2.0) into x*x, so the exponent is read through a
// volatile to keep the call.
double scalar_square(double x) {
    volatile double two = 2.0;
    return std::pow(x, two);
}

}  // namespace

CubicSpline::CubicSpline(std::span<const double> x, std::span<const std::complex<double>> y) {
    const std::size_t n = x.size();
    // prepare_input, in its order (x is 1-D by construction).
    if (n < 2) throw std::invalid_argument("`x` must contain at least 2 elements.");
    if (y.size() != n) throw std::invalid_argument("The length of `y` along `axis`=0 doesn't match the length of `x`");
    for (const double v : x)
        if (!std::isfinite(v)) throw std::invalid_argument("`x` must contain only finite values.");
    for (const std::complex<double>& v : y)
        if (!std::isfinite(v.real()) || !std::isfinite(v.imag()))
            throw std::invalid_argument("`y` must contain only finite values.");
    std::vector<double> dx(n - 1);
    for (std::size_t k = 0; k + 1 < n; ++k) dx[k] = x[k + 1] - x[k];
    for (const double v : dx)
        if (v <= 0) throw std::invalid_argument("`x` must be strictly increasing sequence.");

    std::vector<Cx> yv(n);
    for (std::size_t k = 0; k < n; ++k) yv[k] = {y[k].real(), y[k].imag()};
    // slope = np.diff(y, axis=0) / dxr
    std::vector<Cx> slope(n - 1);
    for (std::size_t k = 0; k + 1 < n; ++k) slope[k] = np_div(sub(yv[k + 1], yv[k]), real(dx[k]));

    std::vector<Cx> s;
    if (n == 3) {
        double A[3][3] = {{1.0, 1.0, 0.0}, {dx[1], 2 * (dx[0] + dx[1]), dx[0]}, {0.0, 1.0, 1.0}};
        std::vector<Cx> b(3);
        b[0] = mul(real(2.0), slope[0]);
        b[1] = mul(real(3.0), add(mul(real(dx[0]), slope[1]), mul(real(dx[1]), slope[0])));
        b[2] = mul(real(2.0), slope[1]);
        s = solve3(A, std::move(b));
    } else {
        // The banded A of CubicSpline.__init__ as zgtsv's three diagonals:
        // du = A[0, 1:], d = A[1, :], dl = A[2, :-1].
        std::vector<Cx> du(n - 1), d(n), dl(n - 1), b(n);
        if (n == 2) {
            // bc (1, slope[0]) at both ends.
            d[0] = real(1.0);
            du[0] = real(0.0);
            b[0] = slope[0];
            d[1] = real(1.0);
            dl[0] = real(0.0);
            b[1] = slope[0];
        } else {
            for (std::size_t k = 1; k + 1 < n; ++k) {
                d[k] = real(2 * (dx[k - 1] + dx[k]));
                b[k] = mul(real(3.0), add(mul(real(dx[k]), slope[k - 1]), mul(real(dx[k - 1]), slope[k])));
            }
            for (std::size_t k = 2; k < n; ++k) du[k - 1] = real(dx[k - 2]);  // A[0, 2:] = dx[:-1]
            for (std::size_t k = 0; k + 2 < n; ++k) dl[k] = real(dx[k + 1]);  // A[-1, :-2] = dx[1:]
            // not-a-knot start
            {
                d[0] = real(dx[1]);
                du[0] = real(x[2] - x[0]);
                const double dd = x[2] - x[0];
                const Cx t1 = mul(real((dx[0] + 2 * dd) * dx[1]), slope[0]);
                const Cx t2 = mul(real(scalar_square(dx[0])), slope[1]);
                b[0] = np_div(add(t1, t2), real(dd));
            }
            // not-a-knot end
            {
                d[n - 1] = real(dx[n - 3]);
                dl[n - 2] = real(x[n - 1] - x[n - 3]);
                const double dd = x[n - 1] - x[n - 3];
                const Cx t1 = mul(real(scalar_square(dx[n - 2])), slope[n - 3]);
                const Cx t2 = mul(real((2 * dd + dx[n - 2]) * dx[n - 3]), slope[n - 2]);
                b[n - 1] = np_div(add(t1, t2), real(dd));
            }
        }
        if (zgtsv(dl, d, du, b) > 0) throw LinAlgError("singular matrix");
        s = std::move(b);
    }

    // CubicHermiteSpline.__init__ (its prepare_input re-checks dydx).
    for (const Cx& v : s)
        if (!finite(v)) throw std::invalid_argument("`dydx` must contain only finite values.");
    const std::size_t m = n - 1;
    c_.assign(4 * m, {});
    for (std::size_t k = 0; k < m; ++k) {
        const Cx dxk = real(dx[k]);
        const Cx t = np_div(sub(add(s[k], s[k + 1]), mul(real(2.0), slope[k])), dxk);
        const Cx c0 = np_div(t, dxk);
        const Cx c1 = sub(np_div(sub(slope[k], s[k]), dxk), t);
        c_[k] = {c0.r, c0.i};
        c_[m + k] = {c1.r, c1.i};
        c_[2 * m + k] = {s[k].r, s[k].i};
        c_[3 * m + k] = {yv[k].r, yv[k].i};
    }
    x_.assign(x.begin(), x.end());
}

std::size_t CubicSpline::interval(double xv) const {
    // find_interval_ascending with extrapolation: x[i] <= xv < x[i+1],
    // 0 below x[0], n-2 at or above x[-1].
    const std::size_t n = x_.size();
    if (!(xv >= x_[0])) return 0;
    if (xv >= x_[n - 1]) return n - 2;
    return static_cast<std::size_t>(std::upper_bound(x_.begin(), x_.end(), xv) - x_.begin()) - 1;
}

std::complex<double> CubicSpline::operator()(double xv, int nu) const {
    if (nu < 0) throw std::invalid_argument("Order of derivative cannot be negative");
    if (std::isnan(xv)) return {std::numeric_limits<double>::quiet_NaN(), 0.0};
    const std::size_t m = x_.size() - 1;
    const std::size_t i = interval(xv);
    const double s = xv - x_[i];
    Cx res{0.0, 0.0};
    Cx z{1.0, 0.0};
    for (int kp = 0; kp < 4; ++kp) {
        double prefactor = 1.0;
        if (nu > 0) {
            if (kp < nu) continue;
            for (int k = kp; k > kp - nu; --k) prefactor *= k;
        }
        const std::complex<double>& cc = c_[static_cast<std::size_t>(3 - kp) * m + i];
        res = add(res, mul(mul(Cx{cc.real(), cc.imag()}, z), real(prefactor)));
        if (kp < 3 && kp >= nu) z = mul(z, real(s));
    }
    return {res.r, res.i};
}

std::vector<std::complex<double>> CubicSpline::operator()(std::span<const double> xv, int nu) const {
    if (nu < 0) throw std::invalid_argument("Order of derivative cannot be negative");
    std::vector<std::complex<double>> out(xv.size());
    for (std::size_t k = 0; k < xv.size(); ++k) out[k] = (*this)(xv[k], nu);
    return out;
}

}  // namespace sstvae::math
