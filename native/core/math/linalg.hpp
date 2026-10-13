// numpy.linalg as the QRSSTVAE receiver calls it: lstsq, solve, inv, eigh
// and eigvalsh on small dense float64 matrices (the tracker's 2x2 and 3x3
// Hessians, its (n, 2) timing regressions, polyfit's Vandermonde matrix).
//
// Not LAPACK, and not bitwise: lstsq is a one-sided Jacobi SVD where numpy
// calls gelsd, solve and inv an LU with partial pivoting where numpy calls
// gesv (OpenBLAS's getrf), eigh a cyclic Jacobi where numpy calls syevd.
// Every one is backward stable, so results agree with numpy's to rounding
// on a well-conditioned problem (~1e-15 relative; tests/
// test_qrss_native_math.py measures it). What is numpy's exactly is every
// decision a caller can branch on:
//
// * lstsq's rank: singular values s_i > rcond * max(s) count, with gelsd's
//   (dlalsd's) rule that rcond <= 0 or >= 1 means eps, and that a
//   one-column or one-row problem has rank 1 unless its one singular value
//   is exactly 0. rcond nullopt is numpy's default, eps * max(m, n). The
//   solution is the minimum-norm one. A non-finite value in `a` is
//   LinAlgError "SVD did not converge in Linear Least Squares" (numpy's
//   gelsd fails there); a non-finite value in `b` makes every coefficient
//   NaN (gelsd scales B by its max norm). m == 0 gives x = 0 and rank 0.
//   residuals is empty unless rank == n and m > n, as numpy documents;
//   then it is ||a x - b||^2 computed directly (gelsd sums its
//   transformed right-hand side: the same number to rounding).
// * solve and inv: LinAlgError "Singular matrix" exactly when elimination
//   meets an exact zero pivot (getrf's info > 0); NaN propagates without
//   an error, as numpy's does -- though not always to the same entries:
//   OpenBLAS's dscal by 1/NaN leaves a zero a zero, so numpy solves
//   [[nan, 1], [0, 1]] x = [1, 1] as [nan, 1] where plain elimination
//   (this) gives [nan, nan].
// * eigh / eigvalsh read the lower triangle only (UPLO='L'); eigenvalues
//   ascending. A NaN or infinity in the lower triangle of a matrix larger
//   than 1x1 makes every eigenvalue (and eigenvector entry) NaN, which is
//   what numpy's syevd returns there -- no exception, so the tracker's
//   `except LinAlgError` around eigh never fires on a NaN Hessian in
//   either implementation. A 1x1 matrix is its own eigenvalue, inf
//   included, with eigenvector [1]. Eigenvector signs are Jacobi's, not
//   LAPACK's: compare eigenvectors up to sign (the receiver uses only the
//   eigenvalues).
// * A non-square matrix for solve/inv/eigh is LinAlgError "Last 2
//   dimensions of the array must be square"; mismatched lengths are
//   std::invalid_argument.

#ifndef SSTVAE_MATH_LINALG_HPP
#define SSTVAE_MATH_LINALG_HPP

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "math/errors.hpp"

namespace sstvae::math {

// A dense row-major float64 matrix. 0x0 is the empty matrix.
struct Matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<double> data;   // rows * cols, row-major

    Matrix() = default;
    Matrix(std::size_t r, std::size_t c) : rows(r), cols(c), data(r * c, 0.0) {}
    // From row-major values; std::invalid_argument unless v.size() == r * c.
    Matrix(std::size_t r, std::size_t c, std::vector<double> v);
    static Matrix identity(std::size_t n);

    double& operator()(std::size_t i, std::size_t j) { return data[i * cols + j]; }
    double operator()(std::size_t i, std::size_t j) const { return data[i * cols + j]; }
    bool empty() const { return data.empty(); }
    bool operator==(const Matrix&) const = default;
};

struct LstsqResult {
    std::vector<double> x;           // n coefficients
    std::vector<double> residuals;   // empty, or one value (see above)
    int rank = 0;
    std::vector<double> s;           // min(m, n) singular values, descending
};

// np.linalg.lstsq(a, b, rcond) for one right-hand side (b of length a.rows).
LstsqResult lstsq(const Matrix& a, std::span<const double> b, std::optional<double> rcond = std::nullopt);

// np.linalg.solve(a, b) for one right-hand side.
std::vector<double> solve(const Matrix& a, std::span<const double> b);

// np.linalg.inv(a).
Matrix inv(const Matrix& a);

struct EighResult {
    std::vector<double> w;   // eigenvalues, ascending
    Matrix v;                // eigenvectors as columns: v(:, i) belongs to w[i]
};

// np.linalg.eigh(a) / np.linalg.eigvalsh(a), UPLO='L'.
EighResult eigh(const Matrix& a);
std::vector<double> eigvalsh(const Matrix& a);

}  // namespace sstvae::math

#endif  // SSTVAE_MATH_LINALG_HPP
