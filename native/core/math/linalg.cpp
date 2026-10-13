#include "math/linalg.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sstvae::math {

namespace {

constexpr double EPS = std::numeric_limits<double>::epsilon();
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

void require_square(const Matrix& a) {
    if (a.rows != a.cols) throw LinAlgError("Last 2 dimensions of the array must be square");
}

bool all_finite(const std::vector<double>& v) {
    return std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); });
}

// The thin SVD of a (m x n, m >= n) by one-sided Jacobi (Hestenes):
// columns of `u` are rotated until mutually orthogonal, `v` accumulates
// the rotations; then s_j = ||u_j|| and u_j /= s_j (a zero column stays 0).
struct Svd {
    Matrix u;               // m x n
    std::vector<double> s;  // n, descending
    Matrix v;               // n x n
};

Svd jacobi_svd(Matrix u) {
    const std::size_t m = u.rows, n = u.cols;
    Matrix v = Matrix::identity(n);
    auto col_dot = [&](std::size_t p, std::size_t q) {
        double d = 0.0;
        for (std::size_t i = 0; i < m; ++i) d += u(i, p) * u(i, q);
        return d;
    };
    for (int sweep = 0; sweep < 80; ++sweep) {
        bool rotated = false;
        for (std::size_t p = 0; p + 1 < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                const double alpha = col_dot(p, p);
                const double beta = col_dot(q, q);
                const double gamma = col_dot(p, q);
                if (alpha == 0.0 || beta == 0.0) continue;
                if (std::abs(gamma) <= EPS * std::sqrt(alpha) * std::sqrt(beta)) continue;
                rotated = true;
                const double zeta = (beta - alpha) / (2.0 * gamma);
                const double t = std::copysign(1.0, zeta) / (std::abs(zeta) + std::hypot(1.0, zeta));
                const double c = 1.0 / std::hypot(1.0, t);
                const double s = c * t;
                for (std::size_t i = 0; i < m; ++i) {
                    const double up = u(i, p), uq = u(i, q);
                    u(i, p) = c * up - s * uq;
                    u(i, q) = s * up + c * uq;
                }
                for (std::size_t i = 0; i < n; ++i) {
                    const double vp = v(i, p), vq = v(i, q);
                    v(i, p) = c * vp - s * vq;
                    v(i, q) = s * vp + c * vq;
                }
            }
        }
        if (!rotated) break;
    }
    std::vector<double> s(n);
    for (std::size_t j = 0; j < n; ++j) {
        double ss = 0.0;
        // a scaled norm, so columns near the overflow threshold survive
        double big = 0.0;
        for (std::size_t i = 0; i < m; ++i) big = std::max(big, std::abs(u(i, j)));
        if (big > 0.0) {
            for (std::size_t i = 0; i < m; ++i) {
                const double r = u(i, j) / big;
                ss += r * r;
            }
            s[j] = big * std::sqrt(ss);
            for (std::size_t i = 0; i < m; ++i) u(i, j) /= s[j];
        } else {
            s[j] = 0.0;
        }
    }
    // descending, columns of u and v alongside
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return s[a] > s[b]; });
    Svd out{Matrix(m, n), std::vector<double>(n), Matrix(n, n)};
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t j = idx[k];
        out.s[k] = s[j];
        for (std::size_t i = 0; i < m; ++i) out.u(i, k) = u(i, j);
        for (std::size_t i = 0; i < n; ++i) out.v(i, k) = v(i, j);
    }
    return out;
}

Matrix transpose(const Matrix& a) {
    Matrix t(a.cols, a.rows);
    for (std::size_t i = 0; i < a.rows; ++i)
        for (std::size_t j = 0; j < a.cols; ++j) t(j, i) = a(i, j);
    return t;
}

// LU with partial pivoting in place (getf2's order: the first largest |a|
// in the column is the pivot, the column below it is scaled by the
// reciprocal pivot where that is safe, then the trailing update). Returns
// the row permutation; `singular` is set when a pivot is exactly zero
// (getrf's info > 0), and elimination carries on past it, as getrf does.
std::vector<std::size_t> lu_inplace(Matrix& a, bool& singular) {
    const std::size_t n = a.rows;
    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), std::size_t{0});
    singular = false;
    const double sfmin = std::numeric_limits<double>::min();
    for (std::size_t j = 0; j < n; ++j) {
        std::size_t p = j;
        double best = std::abs(a(j, j));
        for (std::size_t i = j + 1; i < n; ++i)
            if (std::abs(a(i, j)) > best) {
                best = std::abs(a(i, j));
                p = i;
            }
        if (a(p, j) != 0.0) {
            if (p != j) {
                for (std::size_t k = 0; k < n; ++k) std::swap(a(p, k), a(j, k));
                std::swap(perm[p], perm[j]);
            }
            const double piv = a(j, j);
            if (std::abs(piv) >= sfmin) {
                const double r = 1.0 / piv;
                for (std::size_t i = j + 1; i < n; ++i) a(i, j) *= r;
            } else {
                for (std::size_t i = j + 1; i < n; ++i) a(i, j) /= piv;
            }
        } else {
            singular = true;
        }
        for (std::size_t i = j + 1; i < n; ++i) {
            const double l = a(i, j);
            for (std::size_t k = j + 1; k < n; ++k) a(i, k) -= l * a(j, k);
        }
    }
    return perm;
}

// Solve LU x = P b for one column.
std::vector<double> lu_solve(const Matrix& lu, const std::vector<std::size_t>& perm, std::span<const double> b) {
    const std::size_t n = lu.rows;
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = b[perm[i]];
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < i; ++k) x[i] -= lu(i, k) * x[k];
    for (std::size_t ii = n; ii-- > 0;) {
        for (std::size_t k = ii + 1; k < n; ++k) x[ii] -= lu(ii, k) * x[k];
        x[ii] /= lu(ii, ii);
    }
    return x;
}

// Cyclic Jacobi eigenvalue iteration on a symmetric matrix (both triangles
// filled from the lower one).
EighResult jacobi_eigh(Matrix a) {
    const std::size_t n = a.rows;
    Matrix v = Matrix::identity(n);
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0, diag = 0.0;
        for (std::size_t p = 0; p < n; ++p) {
            diag += a(p, p) * a(p, p);
            for (std::size_t q = p + 1; q < n; ++q) off += a(p, q) * a(p, q);
        }
        if (off == 0.0 || off <= EPS * EPS * 1e-4 * diag) break;
        for (std::size_t p = 0; p + 1 < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                const double apq = a(p, q);
                if (apq == 0.0) continue;
                const double app = a(p, p), aqq = a(q, q);
                const double theta = (aqq - app) / (2.0 * apq);
                double t;
                if (std::abs(theta) > 1e150)
                    t = 0.5 / theta;
                else
                    t = std::copysign(1.0, theta) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                for (std::size_t k = 0; k < n; ++k) {
                    const double akp = a(k, p), akq = a(k, q);
                    a(k, p) = c * akp - s * akq;
                    a(k, q) = s * akp + c * akq;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    const double apk = a(p, k), aqk = a(q, k);
                    a(p, k) = c * apk - s * aqk;
                    a(q, k) = s * apk + c * aqk;
                }
                a(p, q) = a(q, p) = 0.0;
                for (std::size_t k = 0; k < n; ++k) {
                    const double vkp = v(k, p), vkq = v(k, q);
                    v(k, p) = c * vkp - s * vkq;
                    v(k, q) = s * vkp + c * vkq;
                }
            }
        }
    }
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t x, std::size_t y) { return a(x, x) < a(y, y); });
    EighResult out{std::vector<double>(n), Matrix(n, n)};
    for (std::size_t k = 0; k < n; ++k) {
        out.w[k] = a(idx[k], idx[k]);
        for (std::size_t i = 0; i < n; ++i) out.v(i, k) = v(i, idx[k]);
    }
    return out;
}

}  // namespace

Matrix::Matrix(std::size_t r, std::size_t c, std::vector<double> v) : rows(r), cols(c), data(std::move(v)) {
    if (data.size() != r * c)
        throw std::invalid_argument("Matrix: " + std::to_string(data.size()) + " values for " + std::to_string(r) +
                                    "x" + std::to_string(c));
}

Matrix Matrix::identity(std::size_t n) {
    Matrix m(n, n);
    for (std::size_t i = 0; i < n; ++i) m(i, i) = 1.0;
    return m;
}

LstsqResult lstsq(const Matrix& a, std::span<const double> b, std::optional<double> rcond) {
    const std::size_t m = a.rows, n = a.cols;
    if (b.size() != m) throw LinAlgError("Incompatible dimensions");
    LstsqResult out;
    out.x.assign(n, 0.0);
    const std::size_t k = std::min(m, n);
    if (!all_finite(a.data)) throw LinAlgError("SVD did not converge in Linear Least Squares");
    if (k == 0) return out;   // numpy: x[...] = 0 when m == 0; no singular values
    // A = U S V^T, via the transpose when m < n.
    Matrix U, V;
    std::vector<double> s;
    if (m >= n) {
        Svd d = jacobi_svd(a);
        U = std::move(d.u);
        V = std::move(d.v);
        s = std::move(d.s);
    } else {
        Svd d = jacobi_svd(transpose(a));   // A^T = U' S V'^T, so A = V' S U'^T
        U = std::move(d.v);
        V = std::move(d.u);
        s = std::move(d.s);
    }
    out.s = s;
    double r = rcond ? *rcond : EPS * static_cast<double>(std::max(m, n));
    if (r <= 0.0 || r >= 1.0) r = EPS;   // dlalsd's RCND
    const double tol = r * s[0];
    std::vector<bool> keep(k);
    for (std::size_t i = 0; i < k; ++i) keep[i] = (k == 1) ? (s[i] != 0.0) : (s[i] > tol);
    out.rank = static_cast<int>(std::count(keep.begin(), keep.end(), true));
    if (!std::all_of(b.begin(), b.end(), [](double v) { return std::isfinite(v); })) {
        std::fill(out.x.begin(), out.x.end(), NaN);
    } else {
        for (std::size_t i = 0; i < k; ++i) {
            if (!keep[i]) continue;
            double ub = 0.0;
            for (std::size_t r2 = 0; r2 < m; ++r2) ub += U(r2, i) * b[r2];
            const double coef = ub / s[i];
            for (std::size_t j = 0; j < n; ++j) out.x[j] += V(j, i) * coef;
        }
    }
    if (static_cast<std::size_t>(out.rank) == n && m > n) {
        double ss = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            double ax = 0.0;
            for (std::size_t j = 0; j < n; ++j) ax += a(i, j) * out.x[j];
            const double d = b[i] - ax;
            ss += d * d;
        }
        out.residuals.push_back(ss);
    }
    return out;
}

std::vector<double> solve(const Matrix& a, std::span<const double> b) {
    require_square(a);
    if (b.size() != a.rows)
        throw std::invalid_argument("solve: b has " + std::to_string(b.size()) + " values for a " +
                                    std::to_string(a.rows) + "x" + std::to_string(a.cols) + " matrix");
    Matrix lu = a;
    bool singular = false;
    const auto perm = lu_inplace(lu, singular);
    if (singular) throw LinAlgError("Singular matrix");
    return lu_solve(lu, perm, b);
}

Matrix inv(const Matrix& a) {
    require_square(a);
    const std::size_t n = a.rows;
    Matrix lu = a;
    bool singular = false;
    const auto perm = lu_inplace(lu, singular);
    if (singular) throw LinAlgError("Singular matrix");
    Matrix out(n, n);
    std::vector<double> e(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        std::fill(e.begin(), e.end(), 0.0);
        e[j] = 1.0;
        const auto col = lu_solve(lu, perm, e);
        for (std::size_t i = 0; i < n; ++i) out(i, j) = col[i];
    }
    return out;
}

EighResult eigh(const Matrix& a) {
    require_square(a);
    const std::size_t n = a.rows;
    if (n == 0) return {};
    if (n == 1) return {{a(0, 0)}, Matrix::identity(1)};
    Matrix sym(n, n);
    bool finite = true;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            sym(i, j) = sym(j, i) = a(i, j);
            finite = finite && std::isfinite(a(i, j));
        }
    if (!finite) {
        EighResult bad{std::vector<double>(n, NaN), Matrix(n, n)};
        std::fill(bad.v.data.begin(), bad.v.data.end(), NaN);
        return bad;
    }
    return jacobi_eigh(std::move(sym));
}

std::vector<double> eigvalsh(const Matrix& a) { return eigh(a).w; }

}  // namespace sstvae::math
