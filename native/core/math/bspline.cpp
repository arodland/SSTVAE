#include "math/bspline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "qrss/frame.hpp"

namespace sstvae::math {

namespace {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

bool all_finite(std::span<const double> v) {
    return std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); });
}

std::string shape1(std::size_t n) { return "(" + std::to_string(n) + ",)"; }

// str(x) for the "Out of bounds" refusal: Python reprs between brackets.
// numpy pads and trims differently; this is the approximate text.
std::string array_text(std::span<const double> x) {
    std::string s = "[";
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (i) s += ' ';
        s += sstvae::qrss::frame::py_repr(x[i]);
    }
    return s + "]";
}

// fprota(c, s, a, b) = (c*a + s*b, -s*a + c*b)
inline void fprota(double c, double s, double& a, double& b) {
    const double na = c * a + s * b;
    const double nb = -s * a + c * b;
    a = na;
    b = nb;
}

}  // namespace

Givens dlartg(double f, double g) {
    // LAPACK 3.12 dlartg.f90, la_constants: safmin = 2^-1022, safmax = 1/safmin.
    const double safmin = std::numeric_limits<double>::min();
    const double safmax = 1.0 / safmin;
    const double rtmin = std::sqrt(safmin);
    const double rtmax = std::sqrt(safmax / 2);
    const double f1 = std::abs(f);
    const double g1 = std::abs(g);
    Givens out{};
    if (g == 0.0) {
        out.c = 1.0;
        out.s = 0.0;
        out.r = f;
    } else if (f == 0.0) {
        out.c = 0.0;
        out.s = std::copysign(1.0, g);
        out.r = g1;
    } else if (f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax) {
        const double d = std::sqrt(f * f + g * g);
        out.c = f1 / d;
        out.r = std::copysign(d, f);
        out.s = g / out.r;
    } else {
        const double u = std::min(safmax, std::max(safmin, std::max(f1, g1)));
        const double fs = f / u;
        const double gs = g / u;
        const double d = std::sqrt(fs * fs + gs * gs);
        out.c = std::abs(fs) / d;
        out.r = std::copysign(d, f);
        out.s = gs / out.r;
        out.r = out.r * u;
    }
    return out;
}

std::int64_t find_interval(std::span<const double> t, int k, double x, std::int64_t prev_l, bool extrapolate) {
    const auto len_t = static_cast<std::int64_t>(t.size());
    const std::int64_t n = len_t - k - 1;
    const double tb = t[static_cast<std::size_t>(k)];
    const double te = t[static_cast<std::size_t>(n)];
    if (x != x) return -1;
    if (((x < tb) || (x > te)) && !extrapolate) return -1;
    std::int64_t l = (k < prev_l) && (prev_l < n) ? prev_l : k;
    while ((x < t[static_cast<std::size_t>(l)]) && (l != k)) l -= 1;
    l += 1;
    while ((x >= t[static_cast<std::size_t>(l)]) && (l != n)) l += 1;
    return l - 1;
}

void deboor_d(std::span<const double> t, double x, int k, std::int64_t ell, int m, double* result) {
    double* hh = result + k + 1;
    double* h = result;
    result[0] = 1.0;
    for (int j = 1; j <= k - m; ++j) {
        std::copy(h, h + j, hh);
        h[0] = 0.0;
        for (int n = 1; n <= j; ++n) {
            const std::int64_t ind = ell + n;
            const double xb = t[static_cast<std::size_t>(ind)];
            const double xa = t[static_cast<std::size_t>(ind - j)];
            if (xb == xa) {
                h[n] = 0.0;
                continue;
            }
            const double w = hh[n - 1] / (xb - xa);
            h[n - 1] += w * (xb - x);
            h[n] = w * (x - xa);
        }
    }
    for (int j = k - m + 1; j <= k; ++j) {
        std::copy(h, h + j, hh);
        h[0] = 0.0;
        for (int n = 1; n <= j; ++n) {
            const std::int64_t ind = ell + n;
            const double xb = t[static_cast<std::size_t>(ind)];
            const double xa = t[static_cast<std::size_t>(ind - j)];
            if (xb == xa) {
                h[n] = 0.0;
                continue;
            }
            const double w = j * hh[n - 1] / (xb - xa);
            h[n - 1] -= w;
            h[n] = w;
        }
    }
}

double BSpline::operator()(double x) const {
    // evaluate_spline (nu = 0, extrapolate) for one point, interval search from k.
    std::vector<double> wrk(2 * static_cast<std::size_t>(k) + 2);
    const std::int64_t interval = find_interval(t, k, x, k, true);
    if (interval < 0) return NaN;
    deboor_d(t, x, k, interval, 0, wrk.data());
    double out = 0.0;
    for (int a = 0; a < k + 1; ++a) out += c[static_cast<std::size_t>(interval + a - k)] * wrk[static_cast<std::size_t>(a)];
    return out;
}

std::vector<double> BSpline::operator()(std::span<const double> x) const {
    std::vector<double> out(x.size());
    std::vector<double> wrk(2 * static_cast<std::size_t>(k) + 2);
    std::int64_t interval = k;
    for (std::size_t ip = 0; ip < x.size(); ++ip) {
        interval = find_interval(t, k, x[ip], interval, true);
        if (interval < 0) {
            out[ip] = NaN;
            continue;
        }
        deboor_d(t, x[ip], k, interval, 0, wrk.data());
        double v = 0.0;
        for (int a = 0; a < k + 1; ++a)
            v += c[static_cast<std::size_t>(interval + a - k)] * wrk[static_cast<std::size_t>(a)];
        out[ip] = v;
    }
    return out;
}

BSpline BSpline::derivative(int nu) const {
    if (nu < 0) throw std::invalid_argument("derivative: nu must be >= 0 (antiderivatives are not ported)");
    if (nu > k)
        throw std::invalid_argument("Order of derivative (n = " + std::to_string(nu) +
                                    ") must be <= order of spline (k = " + std::to_string(k) + ")");
    // BSpline.derivative: pad c with zeros to len(t), then splder.
    std::vector<double> tt = t;
    std::vector<double> cc = c;
    if (cc.size() < tt.size()) cc.resize(tt.size(), 0.0);
    int kk = k;
    for (int j = 0; j < nu; ++j) {
        const std::size_t N = tt.size();
        const std::size_t nk = static_cast<std::size_t>(kk);
        // dt = t[k+1:-1] - t[1:-k-1]; c = (c[1:-1-k] - c[:-2-k]) * k / dt
        const std::size_t len = N >= nk + 2 ? N - nk - 2 : 0;
        std::vector<double> nc(len + nk, 0.0);
        for (std::size_t i = 0; i < len; ++i) {
            const double dt = tt[nk + 1 + i] - tt[1 + i];
            const double diff = cc[1 + i] - cc[i];
            const double num = diff * kk;
            // np.errstate(invalid='raise', divide='raise'): an operation that
            // makes a NaN out of non-NaN operands (inf - inf, 0/0, inf/inf)
            // or divides a finite number by zero raises (IEEE's
            // divide-by-zero flag: inf/0 is an exact inf and raises
            // nothing); NaN operands pass quietly.
            const bool invalid = (std::isnan(diff) && !std::isnan(cc[1 + i]) && !std::isnan(cc[i])) ||
                                 (std::isnan(dt) && !std::isnan(tt[nk + 1 + i]) && !std::isnan(tt[1 + i])) ||
                                 (dt == 0.0 && std::isfinite(num)) || (std::isinf(num) && std::isinf(dt));
            if (invalid)
                throw std::invalid_argument("The spline has internal repeated knots and is not differentiable " +
                                            std::to_string(nu) + " times");
            nc[i] = num / dt;
        }
        cc = std::move(nc);
        tt = std::vector<double>(tt.begin() + 1, tt.end() - 1);
        kk -= 1;
    }
    return BSpline{std::move(tt), std::move(cc), kk};
}

BSpline make_lsq_spline(std::span<const double> x, std::span<const double> y, std::span<const double> t, int k,
                        std::optional<std::span<const double>> w_in) {
    const char* nonfinite = "Array must not contain infs or nans.";
    if (!all_finite(x) || !all_finite(y) || !all_finite(t)) throw std::invalid_argument(nonfinite);
    std::vector<double> w = w_in ? std::vector<double>(w_in->begin(), w_in->end()) : std::vector<double>(x.size(), 1.0);
    if (w_in && !all_finite(w)) throw std::invalid_argument(nonfinite);
    const auto m = static_cast<std::int64_t>(x.size());
    if (m < k + 1) throw std::invalid_argument("Need more x points.");
    if (k < 0) throw std::invalid_argument("Expect non-negative k.");
    for (std::size_t i = 1; i < t.size(); ++i)
        if (t[i] - t[i - 1] < 0) throw std::invalid_argument("Expect t to be a 1D strictly increasing sequence.");
    if (x.size() != y.size())
        throw std::invalid_argument("Shapes of x " + shape1(x.size()) + " and y " + shape1(y.size()) +
                                    " are incompatible");
    if (k > 0) {
        const auto len_t = static_cast<std::int64_t>(t.size());
        if (len_t <= k)   // t[k] (and t[-k]) do not exist
            throw std::out_of_range("index " + std::to_string(k) + " is out of bounds for axis 0 with size " +
                                    std::to_string(len_t));
        const double lo = t[static_cast<std::size_t>(k)];
        const double hi = t[static_cast<std::size_t>(len_t - k)];
        for (const double v : x)
            if (v < lo || v > hi) throw std::invalid_argument("Out of bounds w/ x = " + array_text(x) + ".");
    }
    if (x.size() != w.size())
        throw std::invalid_argument("Shapes of x " + shape1(x.size()) + " and w " + shape1(w.size()) +
                                    " are incompatible");
    for (std::size_t i = 1; i < x.size(); ++i)
        if (x[i] - x[i - 1] < 0) throw std::invalid_argument("Expect x to be a 1D non-decreasing sequence.");

    const auto len_t = static_cast<std::int64_t>(t.size());
    const std::int64_t nc = len_t - k - 1;   // number of coefficients
    const std::int64_t nz = k + 1;
    if (nc <= 0)
        throw std::invalid_argument("make_lsq_spline: " + std::to_string(len_t) + " knots give no coefficients at k = " +
                                    std::to_string(k));

    // y_w = y * w
    std::vector<double> yw(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) yw[i] = y[i] * w[i];

    // data_matrix (extrapolate=False): A(i, j) = B_j(x_i) * w_i, offset = ind - k
    std::vector<double> R(static_cast<std::size_t>(m * nz));
    auto at = [&](std::int64_t i, std::int64_t j) -> double& { return R[static_cast<std::size_t>(i * nz + j)]; };
    std::vector<std::int64_t> offset(static_cast<std::size_t>(m));
    std::vector<double> wrk(2 * static_cast<std::size_t>(k) + 2);
    std::int64_t ind = k;
    for (std::int64_t i = 0; i < m; ++i) {
        const double xval = x[static_cast<std::size_t>(i)];
        ind = find_interval(t, k, xval, ind, false);
        if (ind < 0) throw std::runtime_error("find_interval: out of bounds with x = " + std::to_string(xval));
        offset[static_cast<std::size_t>(i)] = ind - k;
        deboor_d(t, xval, k, ind, 0, wrk.data());
        for (std::int64_t j = 0; j < nz; ++j)
            at(i, j) = wrk[static_cast<std::size_t>(j)] * w[static_cast<std::size_t>(i)];
    }

    // qr_reduce(startrow=1): Givens rotations row by row, FITPACK's order.
    for (std::int64_t i = 1; i < m; ++i) {
        const std::int64_t oi = offset[static_cast<std::size_t>(i)];
        for (std::int64_t j = oi; j < nc; ++j) {
            if (j >= std::min(i, nc)) break;
            const Givens g = dlartg(at(j, 0), at(i, 0));
            at(j, 0) = g.r;
            for (std::int64_t l = 1; l < nz; ++l) {
                double a = at(j, l), b = at(i, l);
                fprota(g.c, g.s, a, b);
                at(j, l) = a;
                at(i, l - 1) = b;
            }
            at(i, nz - 1) = 0.0;
            fprota(g.c, g.s, yw[static_cast<std::size_t>(j)], yw[static_cast<std::size_t>(i)]);
        }
        if (i < nc) offset[static_cast<std::size_t>(i)] = i;
    }

    // fpback
    if (nc > m) throw std::invalid_argument("nc = " + std::to_string(nc) + " > m = " + std::to_string(m));
    std::vector<double> c(static_cast<std::size_t>(nc));
    c[static_cast<std::size_t>(nc - 1)] = yw[static_cast<std::size_t>(nc - 1)] / at(nc - 1, 0);
    for (std::int64_t i = nc - 2; i >= 0; --i) {
        const std::int64_t nel = std::min(nz, nc - i);
        double ssum = yw[static_cast<std::size_t>(i)];
        for (std::int64_t j = 1; j < nel; ++j) ssum -= at(i, j) * c[static_cast<std::size_t>(i + j)];
        ssum /= at(i, 0);
        c[static_cast<std::size_t>(i)] = ssum;
    }
    return BSpline{std::vector<double>(t.begin(), t.end()), std::move(c), k};
}

}  // namespace sstvae::math
