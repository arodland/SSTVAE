// native/core/math/linalg against numpy.linalg: lstsq, solve, inv, eigh
// and eigvalsh on small systems pinned from numpy (to ~1e-14: LAPACK's
// summation orders are not reproduced), their structural properties, and
// the refusals. tests/test_qrss_native_math.py compares against numpy
// directly, on random systems. The pins:
//
//   A = [[1, 1], [1, 2], [1, 3]]; b = [1, 2, 2]
//   np.linalg.lstsq(A, b, rcond=None)
//   M = [[4, -2, 1], [3, 6, -4], [2, 1, 8]]
//   np.linalg.solve(M, [12, -25, 32]); np.linalg.inv(M)
//   S = [[4, 1, 2], [1, 3, 0.5], [2, 0.5, 5]]; np.linalg.eigh(S)

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "math/linalg.hpp"
#include "qrss_check.hpp"

using namespace sstvae;
using math::Matrix;

template <typename E, typename F>
static void throws(F f, const std::string& text, const std::string& what) {
    try {
        f();
    } catch (const E& e) {
        check::equal(std::string(e.what()), text, what);
        return;
    }
    check::fail(what, "did not throw");
}

static void test_lstsq() {
    const Matrix a(3, 2, {1.0, 1.0, 1.0, 2.0, 1.0, 3.0});
    const auto r = math::lstsq(a, std::vector<double>{1.0, 2.0, 2.0});
    check::close(r.x, std::vector<double>{0.6666666666666659, 0.5000000000000004}, 1e-14, "lstsq x");
    check::close(r.residuals, std::vector<double>{0.1666666666666669}, 1e-14, "lstsq residuals");
    check::equal(r.rank, 2, "lstsq rank");
    check::close(r.s, std::vector<double>{4.079143328941734, 0.6004912172131635}, 1e-14, "lstsq singular values");
    // rank-deficient: the minimum-norm solution, no residuals
    const auto d = math::lstsq(Matrix(3, 2, {1.0, 1.0, 2.0, 2.0, 3.0, 3.0}), std::vector<double>{1.0, 2.0, 3.0});
    check::close(d.x, std::vector<double>{0.5, 0.5}, 1e-14, "rank 1: min-norm");
    check::equal(d.rank, 1, "rank 1");
    check::equal(d.residuals.size(), std::size_t{0}, "rank 1: no residuals");
    // underdetermined 1x2: rank 1, min-norm
    const auto u = math::lstsq(Matrix(1, 2, {1.0, 2.0}), std::vector<double>{3.0});
    check::close(u.x, std::vector<double>{0.6, 1.2}, 1e-14, "1x2 min-norm");
    // no rows: x = 0, rank 0
    const auto e = math::lstsq(Matrix(0, 2), std::vector<double>{});
    check::close(e.x, std::vector<double>{0.0, 0.0}, 0.0, "m = 0");
    check::equal(e.rank, 0, "m = 0: rank 0");
    // a NaN in b: x NaN, rank from A alone
    const auto n = math::lstsq(Matrix::identity(2), std::vector<double>{std::nan(""), 1.0});
    check::is_true(std::isnan(n.x[0]) && std::isnan(n.x[1]), "NaN b: NaN x");
    check::equal(n.rank, 2, "NaN b: rank 2");
    // rcond: explicit, and gelsd's "<= 0 or >= 1 means eps"
    const Matrix dg(3, 3, {1.0, 0, 0, 0, 1e-9, 0, 0, 0, 1e-17});
    const std::vector<double> ones = {1.0, 1.0, 1.0};
    check::equal(math::lstsq(dg, ones, 1e-8).rank, 1, "rcond 1e-8");
    check::equal(math::lstsq(dg, ones, 1e-12).rank, 2, "rcond 1e-12");
    check::equal(math::lstsq(dg, ones, -1.0).rank, 2, "rcond < 0: eps");
    check::equal(math::lstsq(dg, ones, 2.0).rank, 2, "rcond >= 1: eps");
    throws<math::LinAlgError>(
        [] { math::lstsq(Matrix(2, 2, {std::nan(""), 1.0, 0.0, 1.0}), std::vector<double>{1.0, 1.0}); },
        "SVD did not converge in Linear Least Squares", "NaN in A");
    throws<math::LinAlgError>(
        [] { math::lstsq(Matrix(2, 2, {HUGE_VAL, 1.0, 0.0, 1.0}), std::vector<double>{1.0, 1.0}); },
        "SVD did not converge in Linear Least Squares", "inf in A");
}

static void test_solve_inv() {
    const Matrix m(3, 3, {4.0, -2.0, 1.0, 3.0, 6.0, -4.0, 2.0, 1.0, 8.0});
    check::close(math::solve(m, std::vector<double>{12.0, -25.0, 32.0}),
                 std::vector<double>{1.0, -1.9999999999999996, 4.000000000000001}, 1e-14, "solve");
    check::close(math::inv(m).data,
                 std::vector<double>{0.19771863117870722, 0.06463878326996197, 0.007604562737642592,
                                     -0.12167300380228137, 0.11406844106463877, 0.07224334600760458,
                                     -0.034220532319391636, -0.030418250950570346, 0.1140684410646388},
                 1e-15, "inv");
    check::equal(math::inv(Matrix()).rows, std::size_t{0}, "inv of 0x0 is 0x0");
    throws<math::LinAlgError>([] { math::solve(Matrix(2, 2, {1.0, 2.0, 2.0, 4.0}), std::vector<double>{1.0, 1.0}); },
                              "Singular matrix", "solve, singular");
    throws<math::LinAlgError>([] { math::inv(Matrix(2, 2, {1.0, 2.0, 2.0, 4.0})); }, "Singular matrix",
                              "inv, singular");
    throws<math::LinAlgError>([] { math::inv(Matrix(2, 3)); }, "Last 2 dimensions of the array must be square",
                              "inv, not square");
    const auto s = math::solve(Matrix(2, 2, {std::nan(""), 1.0, 0.0, 1.0}), std::vector<double>{1.0, 1.0});
    check::is_true(std::isnan(s[0]), "solve with a NaN: NaN, no error");
}

static void test_eigh() {
    const Matrix s(3, 3, {4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 5.0});
    const auto r = math::eigh(s);
    check::close(r.w, std::vector<double>{2.0965216172967187, 3.0722239520050754, 6.831254430698205}, 1e-14,
                 "eigh w");
    const std::vector<double> want_v = {-0.7167430662893416, -0.3257672733027812, -0.616567158200162,
                                        0.5748726248246245,  -0.7765018813719532, -0.2580044446381949,
                                        0.39471615390480125, 0.5393704773814453,  -0.743827026919507};
    // eigenvectors up to sign, column by column
    for (std::size_t j = 0; j < 3; ++j) {
        double dot = 0.0;
        for (std::size_t i = 0; i < 3; ++i) dot += r.v(i, j) * want_v[i * 3 + j];
        check::is_true(std::abs(std::abs(dot) - 1.0) < 1e-14, "eigh v column " + std::to_string(j));
    }
    check::close(math::eigvalsh(s), r.w, 0.0, "eigvalsh is eigh's w");
    // only the lower triangle is read
    const Matrix upper_nan(2, 2, {1.0, std::nan(""), 0.0, 1.0});
    check::close(math::eigvalsh(upper_nan), std::vector<double>{1.0, 1.0}, 0.0, "upper triangle ignored");
    const auto ln = math::eigvalsh(Matrix(2, 2, {1.0, 0.0, std::nan(""), 1.0}));
    check::is_true(std::isnan(ln[0]) && std::isnan(ln[1]), "NaN in the lower triangle: NaN, no error");
    check::equal(math::eigvalsh(Matrix()).size(), std::size_t{0}, "0x0");
    const auto one = math::eigh(Matrix(1, 1, {-3.0}));
    check::equal(one.w[0], -3.0, "1x1 value");
    check::equal(one.v(0, 0), 1.0, "1x1 vector");
}

int main() {
    try {
        test_lstsq();
        test_solve_inv();
        test_eigh();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return check::report("qrss_linalg");
}
