#include "math/npstats.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "math/errors.hpp"
#include "math/linalg.hpp"
#include "qrss/npsum.hpp"
#include "qrss/rational.hpp"

namespace sstvae::math {

namespace npsum = sstvae::qrss::npsum;

namespace {

// numpy's _median on a 1-D array: partition at the middle (and at -1 for
// the NaN check), then np.mean of the middle one or two values. The values
// at the partition points are the sorted order's whatever the algorithm,
// and only a signed zero could tell two equal elements apart -- which the
// mean's 0 + sum then erases.
template <typename T>
T median_of(std::span<const T> x) {
    if (x.empty()) return std::numeric_limits<T>::quiet_NaN();
    for (const T v : x)
        if (std::isnan(v)) return std::numeric_limits<T>::quiet_NaN();
    std::vector<T> v(x.begin(), x.end());
    const std::size_t n = v.size();
    const std::size_t half = n / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(half), v.end());
    if (n % 2 == 1) {
        const T mid[1] = {v[half]};
        return mean(std::span<const T>(mid, 1));
    }
    const T hi = v[half];
    const T lo = *std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(half));
    const T mid[2] = {lo, hi};
    return mean(std::span<const T>(mid, 2));
}

// numpy's binary_search_with_guess (numpy/_core/src/multiarray/compiled_base.c).
std::ptrdiff_t binary_search_with_guess(double key, const double* arr, std::ptrdiff_t len, std::ptrdiff_t guess) {
    constexpr std::ptrdiff_t LIKELY_IN_CACHE_SIZE = 8;
    std::ptrdiff_t imin = 0;
    std::ptrdiff_t imax = len;
    if (key > arr[len - 1]) return len;
    if (key < arr[0]) return -1;
    if (len <= 4) {
        std::ptrdiff_t i = 1;
        for (; i < len && key >= arr[i]; ++i) {
        }
        return i - 1;
    }
    if (guess > len - 3) guess = len - 3;
    if (guess < 1) guess = 1;
    if (key < arr[guess]) {
        if (key < arr[guess - 1]) {
            imax = guess - 1;
            if (guess > LIKELY_IN_CACHE_SIZE && key >= arr[guess - LIKELY_IN_CACHE_SIZE])
                imin = guess - LIKELY_IN_CACHE_SIZE;
        } else {
            return guess - 1;
        }
    } else {
        if (key < arr[guess + 1]) return guess;
        if (key < arr[guess + 2]) return guess + 1;
        imin = guess + 2;
        if (guess < len - LIKELY_IN_CACHE_SIZE - 1 && key < arr[guess + LIKELY_IN_CACHE_SIZE])
            imax = guess + LIKELY_IN_CACHE_SIZE;
    }
    while (imin < imax) {
        const std::ptrdiff_t imid = imin + ((imax - imin) >> 1);
        if (key >= arr[imid])
            imin = imid + 1;
        else
            imax = imid;
    }
    return imin - 1;
}

void check_interp(std::span<const double> xp, std::span<const double> fp) {
    if (xp.empty()) throw std::invalid_argument("array of sample points is empty");
    if (fp.size() != xp.size()) throw std::invalid_argument("fp and xp are not of the same length.");
}

// One value of numpy's arr_interp loop; `j` is the carried guess.
double interp_one(double x, std::span<const double> xp, std::span<const double> fp, double lval, double rval,
                  std::ptrdiff_t& j) {
    const auto len = static_cast<std::ptrdiff_t>(xp.size());
    const double* dx = xp.data();
    const double* dy = fp.data();
    if (len == 1) return x < dx[0] ? lval : (x > dx[0] ? rval : dy[0]);
    if (std::isnan(x)) return x;
    j = binary_search_with_guess(x, dx, len, j);
    if (j == -1) return lval;
    if (j == len) return rval;
    if (j == len - 1) return dy[j];
    if (dx[j] == x) return dy[j];
    const double slope = (dy[j + 1] - dy[j]) / (dx[j + 1] - dx[j]);
    double r = slope * (x - dx[j]) + dy[j];
    if (std::isnan(r)) {
        r = slope * (x - dx[j + 1]) + dy[j + 1];
        if (std::isnan(r) && dy[j] == dy[j + 1]) r = dy[j];
    }
    return r;
}

}  // namespace

double median(std::span<const double> x) { return median_of(x); }
float median(std::span<const float> x) { return median_of(x); }

double mean(std::span<const double> x) { return npsum::mean(x); }
float mean(std::span<const float> x) { return npsum::mean(x); }

double average(std::span<const double> a, std::span<const double> w) {
    if (a.size() != w.size()) throw TypeError("Axis must be specified when shapes of a and weights differ.");
    const double scl = npsum::sum(w);
    if (scl == 0.0) throw qrss::ZeroDivision("Weights sum to zero, can't be normalized");
    std::vector<double> aw(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) aw[i] = a[i] * w[i];
    return npsum::sum(std::span<const double>(aw)) / scl;
}

double interp(double x, std::span<const double> xp, std::span<const double> fp, std::optional<double> left,
              std::optional<double> right) {
    check_interp(xp, fp);
    std::ptrdiff_t j = 0;
    return interp_one(x, xp, fp, left.value_or(fp.front()), right.value_or(fp.back()), j);
}

std::vector<double> interp(std::span<const double> x, std::span<const double> xp, std::span<const double> fp,
                           std::optional<double> left, std::optional<double> right) {
    check_interp(xp, fp);
    const double lval = left.value_or(fp.front());
    const double rval = right.value_or(fp.back());
    std::vector<double> out(x.size());
    std::ptrdiff_t j = 0;
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = interp_one(x[i], xp, fp, lval, rval, j);
    return out;
}

std::vector<double> polyfit(std::span<const double> x_in, std::span<const double> y_in, int deg,
                            std::optional<std::span<const double>> w_in, std::optional<double> rcond) {
    // order = int(deg) + 1; x = asarray(x) + 0.0; y = asarray(y) + 0.0
    if (deg < 0) throw std::invalid_argument("expected deg >= 0");
    if (x_in.empty()) throw TypeError("expected non-empty vector for x");
    if (x_in.size() != y_in.size()) throw TypeError("expected x and y to have same length");
    const std::size_t m = x_in.size();
    const std::size_t order = static_cast<std::size_t>(deg) + 1;
    std::vector<double> x(m), rhs(m);
    for (std::size_t i = 0; i < m; ++i) {
        x[i] = x_in[i] + 0.0;
        rhs[i] = y_in[i] + 0.0;
    }
    const double rc = rcond ? *rcond : static_cast<double>(m) * std::numeric_limits<double>::epsilon();
    // lhs = vander(x, order): column order-1 is 1, then x, x*x, (x*x)*x, ...
    // leftwards, by multiply.accumulate.
    Matrix lhs(m, order);
    for (std::size_t i = 0; i < m; ++i) {
        double p = 1.0;
        lhs(i, order - 1) = p;
        if (order > 1) {
            p = x[i];
            lhs(i, order - 2) = p;
            for (std::size_t k = 2; k < order; ++k) {
                p = p * x[i];
                lhs(i, order - 1 - k) = p;
            }
        }
    }
    if (w_in) {
        const std::span<const double> w = *w_in;
        if (w.size() != m) throw TypeError("expected w and y to have the same length");
        for (std::size_t i = 0; i < m; ++i) {
            const double wi = w[i] + 0.0;
            for (std::size_t j = 0; j < order; ++j) lhs(i, j) *= wi;
            rhs[i] *= wi;
        }
    }
    // scale = sqrt((lhs*lhs).sum(axis=0)): a reduction along the slow axis,
    // which numpy adds row after row (not pairwise).
    std::vector<double> scale(order, 0.0);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < order; ++j) scale[j] += lhs(i, j) * lhs(i, j);
    for (double& s : scale) s = std::sqrt(s);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < order; ++j) lhs(i, j) /= scale[j];
    std::vector<double> c = lstsq(lhs, rhs, rc).x;
    for (std::size_t j = 0; j < order; ++j) c[j] /= scale[j];
    return c;
}

double polyval(std::span<const double> p, double x) {
    double y = 0.0;
    for (const double pv : p) y = y * x + pv;
    return y;
}

std::vector<double> polyval(std::span<const double> p, std::span<const double> x) {
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = polyval(p, x[i]);
    return out;
}

}  // namespace sstvae::math
