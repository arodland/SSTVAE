// Port of the rest of sstvae/qrss/track.py (fit_timing, _segment_timing,
// verify, the frequency-path measurements); see track_timing.hpp.

#include "qrss/track_timing.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "math/convolve.hpp"
#include "math/linalg.hpp"
#include "math/npcomplex.hpp"
#include "math/npfft.hpp"
#include "math/npstats.hpp"
#include "qrss/acquire_stats.hpp"
#include "qrss/npsum.hpp"
#include "qrss/rational.hpp"

namespace sstvae::qrss::track {

namespace {

namespace np = math::np;
namespace rt = rxconfig::track;
using frame::FrameSpec;
using types::Detection;
using types::FreqPath;
using types::Timing;

constexpr double PI = 3.141592653589793;   // np.pi
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

// --- Python's scalar semantics -------------------------------------------------

// max(a, b) / min(a, b): the first argument unless the second compares
// greater (less), so a NaN first argument is kept.
inline double py_max(double a, double b) { return b > a ? b : a; }
inline double py_min(double a, double b) { return b < a ? b : a; }

// int() of an integral float, with Python's errors for NaN and infinity.
std::int64_t int_of_integral(double v) {
    if (std::isnan(v)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(v)) throw std::overflow_error("cannot convert float infinity to integer");
    if (!(v >= -9223372036854775808.0 && v < 9223372036854775808.0))
        throw std::overflow_error("Python int too large to convert to C long");
    return static_cast<std::int64_t>(v);
}
// math.floor, math.ceil, int() and round() (half to even) to an int.
std::int64_t py_floor(double v) { return int_of_integral(std::isfinite(v) ? std::floor(v) : v); }
std::int64_t py_ceil(double v) { return int_of_integral(std::isfinite(v) ? std::ceil(v) : v); }
std::int64_t py_trunc(double v) { return int_of_integral(std::isfinite(v) ? std::trunc(v) : v); }
std::int64_t py_round(double v) { return int_of_integral(std::isfinite(v) ? std::nearbyint(v) : v); }

// Python's floor division and modulo of ints.
inline std::int64_t floordiv(std::int64_t a, std::int64_t b) {
    std::int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}
inline std::int64_t floormod(std::int64_t a, std::int64_t b) {
    const std::int64_t r = a % b;
    return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r;
}

// x ** 2 for a Python float: C pow, which the compiler must not fold into
// x * x (the exponent goes through a volatile).
double py_sq(double x) {
    volatile double two = 2.0;
    return std::pow(x, two);
}

// --- numpy's elementwise semantics ---------------------------------------------

// float64 -> int64 as numpy casts on x86-64 (the "integer indefinite"
// INT64_MIN for NaN and anything out of range).
inline std::int64_t np_i64(double x) {
    if (!(x >= -9223372036854775808.0 && x < 9223372036854775808.0)) return std::numeric_limits<std::int64_t>::min();
    return static_cast<std::int64_t>(x);
}
// int64 addition wrapping as numpy's does.
inline std::int64_t add_wrap(std::int64_t a, std::int64_t b) {
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(a) + static_cast<std::uint64_t>(b));
}
// |z|^2 as np.abs(z) ** 2.
inline double abs2(cplx z) {
    const double a = np::abs(z);
    return a * a;
}

[[noreturn]] void index_error(std::int64_t i, std::size_t size) {
    throw std::out_of_range("index " + std::to_string(i) + " is out of bounds for axis 0 with size " +
                            std::to_string(size));
}
[[noreturn]] void empty_argmax() { throw std::invalid_argument("attempt to get argmax of an empty sequence"); }

// x[i] with numpy's negative indexing (IndexError outside).
inline double at(std::span<const double> x, std::int64_t i) {
    const auto n = static_cast<std::int64_t>(x.size());
    const std::int64_t k = i < 0 ? i + n : i;
    if (k < 0 || k >= n) index_error(i, x.size());
    return x[static_cast<std::size_t>(k)];
}

// np.argmax of a float64 array: the first maximum, a NaN winning first.
std::size_t np_argmax(std::span<const double> x) {
    if (x.empty()) empty_argmax();
    std::size_t best = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (std::isnan(x[i])) return i;
        if (x[i] > x[best]) best = i;
    }
    return best;
}

// np.arange(start, stop, step) of floats.
std::vector<double> arange_f(double start, double stop, double step) {
    if (step == 0.0) throw ZeroDivision("float division by zero");
    const double len_f = (stop - start) / step;
    if (std::isnan(len_f)) throw std::invalid_argument("arange: cannot compute length");
    const double c = std::ceil(len_f);
    if (!(c > 0)) return {};
    if (c > 9.2e18) throw std::invalid_argument("Maximum allowed size exceeded");
    const auto n = static_cast<std::size_t>(c);
    std::vector<double> out(n);
    out[0] = start;
    if (n == 1) return out;
    out[1] = start + step;
    const double delta = out[1] - out[0];
    for (std::size_t i = 2; i < n; ++i) out[i] = start + static_cast<double>(i) * delta;
    return out;
}

// np.linspace(start, stop, num) (endpoint=True, float64).
std::vector<double> linspace(double start, double stop, std::int64_t num) {
    std::vector<double> y(static_cast<std::size_t>(num));
    const std::int64_t div = num - 1;
    const double delta = stop - start;
    for (std::int64_t i = 0; i < num; ++i) y[static_cast<std::size_t>(i)] = static_cast<double>(i);
    if (div > 0) {
        const double step = delta / static_cast<double>(div);
        if (step == 0.0) {
            for (auto& v : y) v = (v / static_cast<double>(div)) * delta;
        } else {
            for (auto& v : y) v = v * step;
        }
    } else {
        for (auto& v : y) v = v * delta;
    }
    for (auto& v : y) v = v + start;
    if (num > 1) y[static_cast<std::size_t>(num - 1)] = stop;
    return y;
}

// np.hanning(M): 0.5 + 0.5 cos(pi n / (M - 1)), n = arange(1 - M, M, 2).
std::vector<double> hanning(std::int64_t M) {
    if (M < 1) return {};
    if (M == 1) return {1.0};
    const auto Md = static_cast<double>(M);
    const std::vector<double> n = arange_f(1.0 - Md, Md, 2.0);
    std::vector<double> w(n.size());
    for (std::size_t i = 0; i < n.size(); ++i) w[i] = 0.5 + 0.5 * std::cos(PI * n[i] / (Md - 1.0));
    return w;
}

std::vector<double> as_doubles(std::span<const std::int64_t> v) {
    std::vector<double> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = static_cast<double>(v[i]);
    return out;
}

// np.sum(a * a) (pairwise) of float64 values.
double sum_sq(std::span<const double> a) {
    std::vector<double> sq(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) sq[i] = a[i] * a[i];
    return npsum::sum(std::span<const double>(sq));
}

const std::vector<double>& real_known(const Known& kn) {
    if (kn.is_complex()) throw std::invalid_argument("fit_timing needs real known symbols (a CE frame's)");
    return kn.real();
}

// --- the gather -----------------------------------------------------------------

#if defined(__GNUC__) || defined(__clang__)
#define SSTVAE_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define SSTVAE_ALWAYS_INLINE inline
#endif

// acc[j] += a ((xp[j] w0) + (xp[j + 1] w1)) for j < n: one known symbol's
// terms at the lags where its interpolation weights are the same. Plain
// multiplies and adds in that order (no fused operations; this file is
// built with -ffp-contract=off), so every clone computes the same bits.
SSTVAE_ALWAYS_INLINE void gather_impl(double* acc, const double* xp, std::size_t n, double a, double w0, double w1) {
#if defined(__GNUC__) || defined(__clang__)
    typedef double v4 __attribute__((vector_size(32)));
    const v4 va = {a, a, a, a}, vw0 = {w0, w0, w0, w0}, vw1 = {w1, w1, w1, w1};
    std::size_t j = 0;
    for (; j + 4 <= n; j += 4) {
        v4 x0, x1, s;
        std::memcpy(&x0, xp + j, sizeof x0);
        std::memcpy(&x1, xp + j + 1, sizeof x1);
        std::memcpy(&s, acc + j, sizeof s);
        s += va * (x0 * vw0 + x1 * vw1);
        std::memcpy(acc + j, &s, sizeof s);
    }
    for (; j < n; ++j) acc[j] += a * (xp[j] * w0 + xp[j + 1] * w1);
#else
    for (std::size_t j = 0; j < n; ++j) acc[j] += a * (xp[j] * w0 + xp[j + 1] * w1);
#endif
}

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#define SSTVAE_GATHER_CLONE 1
bool cpu_has_avx2() {
    static const bool has = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") != 0;
    }();
    return has;
}
// (An AVX-512F clone of the same loop measured no faster on an AVX-512
// Xeon, eight-wide or four-wide, and an AVX-512 clock is the slower one.)
__attribute__((target("avx2"))) void gather_avx2(double* acc, const double* xp, std::size_t n, double a, double w0,
                                                 double w1) {
    gather_impl(acc, xp, n, a, w0, w1);
}
#endif

void gather(double* acc, const double* xp, std::size_t n, double a, double w0, double w1) {
#ifdef SSTVAE_GATHER_CLONE
    if (cpu_has_avx2()) {
        gather_avx2(acc, xp, n, a, w0, w1);
        return;
    }
#endif
    gather_impl(acc, xp, n, a, w0, w1);
}

// One term of the reference's direct path at receive index idx: a ((1 -
// f) x[i0c] + f x[i0c + 1]) ok, i0 = floor(idx), f = idx - i0, ok = i0 >= 0
// and i0 + 1 < len(x), i0c = clip(i0, 0, len(x) - 2).
inline double term_at(std::span<const double> x, double idx) {
    const auto nx = static_cast<std::int64_t>(x.size());
    const std::int64_t i0 = np_i64(std::floor(idx));
    const double f = idx - static_cast<double>(i0);
    const bool ok = i0 >= 0 && i0 + 1 < nx;
    const std::int64_t i0c = std::min(std::max(i0, std::int64_t{0}), nx - 2);   // np.clip
    const double v = at(x, i0c) * (1.0 - f) + at(x, i0c + 1) * f;
    return v * (ok ? 1.0 : 0.0);
}

// The 1-D @ 2-D product of the reference's direct path: out[j] = sum_k a_k
// term(y_k + lags[j]), in k order, for y_k given.
void correlate_rows(std::span<const double> x, std::span<const double> a, std::span<const double> y,
                    std::span<const double> lags, double* out) {
    const std::size_t nl = lags.size();
    std::fill(out, out + nl, 0.0);
    for (std::size_t k = 0; k < a.size(); ++k)
        for (std::size_t j = 0; j < nl; ++j) out[j] += a[k] * term_at(x, y[k] + lags[j]);
}

// _segment_timing's per-segment pieces.
struct SegData {
    std::vector<double> pos;   // the segment's positions (float)
    std::vector<double> a;     // their symbols
    double sum_a2 = 0.0;       // np.sum(a[sel] ** 2)
};

}  // namespace

// --- the coarse timing fit ------------------------------------------------------

double parabola(double ym, double y0, double yp) { return acquire::parabola(ym, y0, yp); }

std::vector<double> fit_statistic(std::span<const cplx> z_d, double lp_hz, int fs) {
    const std::vector<cplx> uc = lowpass(z_d, lp_hz, static_cast<double>(fs));
    std::vector<cplx> prod(z_d.size());
    for (std::size_t i = 0; i < z_d.size(); ++i) prod[i] = np::mul(z_d[i], std::conj(uc[i]));
    const std::vector<double> taps = mf_taps(static_cast<double>(fs));
    const std::vector<cplx> c =
        math::oaconvolve(std::span<const cplx>(prod), std::span<const double>(taps), math::ConvMode::Same);
    std::vector<double> x(c.size());
    for (std::size_t i = 0; i < c.size(); ++i) x[i] = c[i].imag();
    return x;
}

std::vector<double> fit_grid(std::span<const double> x, const Known& known, std::int64_t base, std::int64_t nlag,
                             double frac0, std::span<const double> deltas, double sps, double norm) {
    if (nlag < 0) throw std::invalid_argument("negative dimensions are not allowed");
    const std::vector<double>& av = real_known(known);
    const std::span<const double> a(av);
    if (known.pos.size() != a.size())
        throw std::invalid_argument("operands could not be broadcast together with shapes (" +
                                    std::to_string(a.size()) + ",) (" + std::to_string(known.pos.size()) + ",nlag) ");
    const auto nl = static_cast<std::size_t>(nlag);
    std::vector<double> Z(deltas.size() * nl);
    if (nl == 0) return Z;
    const auto nx = static_cast<std::int64_t>(x.size());
    if (nx == 0 && !a.empty()) index_error(-2, 0);
    // The fast path skips the terms outside x (each a_k * 0); it needs the
    // values the clip would read there to be finite for that to be exact.
    const bool edges_finite = nx >= 2 && std::isfinite(x[0]) && std::isfinite(x[1]) &&
                              std::isfinite(x[static_cast<std::size_t>(nx - 2)]) &&
                              std::isfinite(x[static_cast<std::size_t>(nx - 1)]);
    const auto based = static_cast<double>(base);
    constexpr double TWO50 = 1125899906842624.0;
    constexpr std::uint64_t EXP_MASK = 0x7ff0000000000000ULL, EXP_ONE = 0x0010000000000000ULL;
    std::vector<double> acc(nl);
    for (std::size_t i = 0; i < deltas.size(); ++i) {
        std::fill(acc.begin(), acc.end(), 0.0);
        const double s = 1.0 + deltas[i];
        for (std::size_t k = 0; k < a.size(); ++k) {
            const double o = static_cast<double>(known.pos[k]) * sps * s + frac0;
            const double y = based + o;   // idx = y + j
            const double ak = a[k];
            if (!edges_finite) {
                for (std::size_t j = 0; j < nl; ++j) acc[j] += ak * term_at(x, y + static_cast<double>(j));
                continue;
            }
            // Runs of lags over which y + j (rounded, as the reference adds
            // it) is idx + (j - j0): within one binade of ulp <= 1/4 the
            // rounding of y + j is the same for every integer j, so floor
            // steps by one and f is constant -- a contiguous gather. A term
            // outside x is a finite value times 0 and is skipped.
            std::size_t j = 0;
            while (j < nl) {
                const double idx = y + static_cast<double>(j);
                if (!(idx >= 1.0 && idx < TWO50)) {
                    if (idx < 1.0) {   // below x, or its first sample: one at a time
                        acc[j] += ak * term_at(x, idx);
                        ++j;
                        continue;
                    }
                    for (; j < nl; ++j) acc[j] += ak * term_at(x, y + static_cast<double>(j));
                    break;
                }
                // B = 2^(e + 1) for idx in [2^e, 2^(e + 1)): idx's exponent bits, plus one
                const double B = std::bit_cast<double>((std::bit_cast<std::uint64_t>(idx) & EXP_MASK) + EXP_ONE);
                const double room = std::ceil(B - idx);   // B - idx is exact (idx >= B / 2)
                const std::size_t run = room >= static_cast<double>(nl - j) ? nl - j : static_cast<std::size_t>(room);
                const double fl = std::floor(idx);
                const auto i0 = static_cast<std::int64_t>(fl);
                const double f = idx - fl;
                const std::int64_t lim = nx - 1 - i0;   // valid while i0 + (j' - j) + 1 < len(x)
                const std::size_t n_ok = lim <= 0 ? 0 : std::min<std::size_t>(run, static_cast<std::size_t>(lim));
                if (n_ok > 0) gather(acc.data() + j, x.data() + i0, n_ok, ak, 1.0 - f, f);
                j += run;
            }
        }
        double* row = Z.data() + i * nl;
        for (std::size_t j = 0; j < nl; ++j) row[j] = acc[j] / norm;
    }
    return Z;
}

SegmentTiming segment_timing(std::span<const double> x, double sig2, const Known& known, double t0abs, double delta,
                             double sps, std::int64_t n_pos, double seg_s) {
    const std::vector<double>& av = real_known(known);
    const std::vector<double> pos = as_doubles(known.pos);
    if (seg_s == 0.0) throw ZeroDivision("float division by zero");
    if (pos.empty()) throw std::invalid_argument("zero-size array to reduction operation minimum which has no identity");
    const std::int64_t nseg = std::max<std::int64_t>(1, py_ceil(static_cast<double>(n_pos) * config::T_SYM / seg_s));
    double pmin = pos[0], pmax = pos[0];
    for (const double p : pos) {
        pmin = std::min(pmin, p);
        pmax = std::max(pmax, p);
    }
    const std::vector<double> edges = linspace(pmin, pmax + 1, nseg + 1);
    const std::vector<double> lags = arange_f(-sps, sps + 1e-9, 0.1);
    std::vector<SegData> segs;
    for (std::int64_t s = 0; s < nseg; ++s) {
        const double s0 = edges[static_cast<std::size_t>(s)], s1 = edges[static_cast<std::size_t>(s + 1)];
        SegData sd;
        for (std::size_t q = 0; q < pos.size(); ++q)
            if (pos[q] >= s0 && pos[q] < s1) {
                sd.pos.push_back(pos[q]);
                sd.a.push_back(av[q]);
            }
        if (sd.pos.size() < 8) continue;
        std::vector<double> a2(sd.a.size());
        for (std::size_t q = 0; q < sd.a.size(); ++q) a2[q] = sd.a[q] * sd.a[q];
        sd.sum_a2 = npsum::sum(std::span<const double>(a2));
        segs.push_back(std::move(sd));
    }

    std::vector<double> c(lags.size());
    std::vector<double> yk;
    // local(sel, t0, d): (offset, Z^2) or nullopt
    auto local = [&](const SegData& sd, double t0, double d) -> std::optional<std::pair<double, double>> {
        yk.resize(sd.pos.size());
        for (std::size_t q = 0; q < sd.pos.size(); ++q) yk[q] = t0 + sd.pos[q] * sps * (1 + d);
        correlate_rows(x, sd.a, yk, lags, c.data());
        const double den = std::sqrt(sig2 * sd.sum_a2);
        for (double& v : c) v = v / den;
        for (const double v : c)
            if (!std::isfinite(v)) return std::nullopt;
        const std::size_t k = np_argmax(c);
        if (c[k] < 4.0) return std::nullopt;
        const double dk = (k > 0 && k + 1 < c.size()) ? parabola(c[k - 1], c[k], c[k + 1]) : 0.0;
        return std::make_pair(lags[k] + dk * 0.1, py_sq(c[k]));
    };

    // fit(tp, ab, wt, t0_, d_)
    auto fit = [&](const std::vector<double>& tp, const std::vector<double>& ab, const std::vector<double>& wt,
                   double t0_, double d_) -> std::pair<double, double> {
        const std::size_t n = tp.size();
        auto ptp = [](std::span<const double> v) {
            double lo = v[0], hi = v[0];
            for (const double e : v) {
                if (std::isnan(e)) return NaN;
                lo = std::min(lo, e);
                hi = std::max(hi, e);
            }
            return hi - lo;
        };
        if (n == 1 || ptp(tp) < 300) {
            std::vector<double> r(n);
            for (std::size_t q = 0; q < n; ++q) r[q] = ab[q] - tp[q] * sps * (1 + delta);
            const double t0 = math::average(std::span<const double>(r), std::span<const double>(wt));
            return {t0, delta};
        }
        std::vector<double> sw(n);
        for (std::size_t q = 0; q < n; ++q) sw[q] = std::sqrt(wt[q]);
        auto solve = [&](const std::vector<std::size_t>& rows) {
            math::Matrix A(rows.size(), 2);
            std::vector<double> b(rows.size());
            for (std::size_t r = 0; r < rows.size(); ++r) {
                const std::size_t q = rows[r];
                A(r, 0) = 1.0 * sw[q];
                A(r, 1) = (tp[q] * sps) * sw[q];
                b[r] = ab[q] * sw[q];
            }
            return math::lstsq(A, b).x;
        };
        std::vector<std::size_t> all(n);
        for (std::size_t q = 0; q < n; ++q) all[q] = q;
        std::vector<double> coef = solve(all);
        std::vector<double> res(n), res2(n);
        for (std::size_t q = 0; q < n; ++q) {
            res[q] = ab[q] - (coef[0] + (tp[q] * sps) * coef[1]);
            res2[q] = res[q] * res[q];
        }
        const double cut =
            py_max(3 * std::sqrt(math::average(std::span<const double>(res2), std::span<const double>(wt))), 0.3);
        std::vector<std::size_t> keep;
        std::vector<double> tpk;
        for (std::size_t q = 0; q < n; ++q)
            if (std::abs(res[q]) < cut) {
                keep.push_back(q);
                tpk.push_back(tp[q]);
            }
        if (keep.size() >= 2 && keep.size() < n && ptp(tpk) >= 300) coef = solve(keep);
        const double t0n = coef[0], dn = coef[1] - 1.0;
        if (!(std::isfinite(t0n) && std::isfinite(dn)) || std::abs(dn) > 300e-6) return {t0_, d_};
        return {t0n, dn};
    };

    double t0 = t0abs, d = delta;
    std::vector<double> tp, ab, wt;
    for (const SegData& sd : segs) {
        const auto r = local(sd, t0, d);
        if (!r) continue;
        const double pm = math::mean(std::span<const double>(sd.pos));
        tp.push_back(pm);
        ab.push_back(t0 + pm * sps * (1 + d) + r->first);
        wt.push_back(r->second);
        std::tie(t0, d) = fit(tp, ab, wt, t0, d);
    }
    if (tp.empty()) return {t0abs, delta, 0.0};
    // the evidence for the final line: every segment's Z at it (no search)
    const auto nx = static_cast<std::int64_t>(x.size());
    double score = 0.0;
    for (const SegData& sd : segs) {
        double dot = 0.0;
        for (std::size_t q = 0; q < sd.pos.size(); ++q) {
            const double o = t0 + sd.pos[q] * sps * (1 + d);
            const double fo = std::floor(o);
            const std::int64_t i0 = std::min(std::max(np_i64(fo), std::int64_t{0}), nx - 2);
            const double f = o - fo;
            const double v = at(x, i0) * (1 - f) + at(x, i0 + 1) * f;
            dot += sd.a[q] * v;
        }
        score += py_sq(py_max(dot / std::sqrt(sig2 * sd.sum_a2), 0.0));
    }
    return {t0, d, score};
}

std::vector<Timing> fit_timing(std::span<const cplx> z_d, int ch_fs, const FrameSpec& spec, double t0_index,
                               const FitTimingArgs& args, FitTimingTrace* trace) {
    ce_only(spec, "track.fit_timing", "ltrack.fit_timing_l");
    const int fs = ch_fs;
    const Known own = args.known ? Known{} : known_symbols(spec);
    const Known& kn = args.known ? *args.known : own;
    const std::vector<double>& a = real_known(kn);
    const std::int64_t n_pos = spec.n_pos();
    const double sps = config::T_SYM * fs;
    const double dur = static_cast<double>(n_pos) * sps;
    double lo = 0.0, hi = 0.0;
    if (!args.tau0_init) {
        lo = -2.5 * fs - 2;
        hi = 2.5 * fs + 2;
    } else {
        const double r = args.tau_range ? *args.tau_range : 3.0;
        lo = *args.tau0_init - r;
        hi = *args.tau0_init + r;
    }
    const double step = 0.2 / static_cast<double>(n_pos);
    const std::int64_t nd = py_floor(args.ppm_range * 1e-6 / step);
    std::vector<double> deltas;
    for (std::int64_t i = -nd; i <= nd; ++i) deltas.push_back(static_cast<double>(i) * step);
    const std::int64_t base = py_floor(t0_index) + py_floor(lo);
    const std::int64_t nlag = py_ceil(hi) - py_floor(lo) + 1;
    const double frac0 = t0_index - std::floor(t0_index);
    if (nlag < 0) throw std::invalid_argument("negative dimensions are not allowed");
    const double sum_a2 = sum_sq(a);
    if (trace) {
        *trace = FitTimingTrace{};
        trace->base = base;
        trace->nlag = nlag;
        trace->nd = nd;
        trace->frac0 = frac0;
        trace->step = step;
    }

    struct Peak {
        double score, zmax, t0abs, delta;
    };
    std::vector<Peak> peaks;
    const auto nl = static_cast<std::size_t>(nlag);
    const std::size_t nrow = deltas.size();
    const std::int64_t csps = py_ceil(sps);
    for (std::size_t lpi = 0; lpi < rt::CARRIER_LP_HZ.size(); ++lpi) {
        const std::vector<double> x = fit_statistic(z_d, rt::CARRIER_LP_HZ[lpi], fs);
        const auto nx = static_cast<std::int64_t>(x.size());
        // sel = slice(max(base, 0), max(min(base + nlag + int(dur) + 1, len(x)), max(base, 0) + 1))
        const std::int64_t s0 = std::max<std::int64_t>(base, 0);
        const std::int64_t s1 = std::max(std::min(base + nlag + py_trunc(dur) + 1, nx), s0 + 1);
        std::vector<double> sq;
        for (std::int64_t q = s0; q < std::min(s1, nx); ++q) {
            const double v = x[static_cast<std::size_t>(q)];
            sq.push_back(v * v);
        }
        const double sig2 = py_max(math::median(std::span<const double>(sq)) / 0.4549, 1e-30);
        const double norm = std::sqrt(sig2 * sum_a2);
        if (trace) trace->sig2.push_back(sig2);
        const std::vector<double> Z = fit_grid(x, kn, base, nlag, frac0, deltas, sps, norm);
        // Zl = Z.max(axis=0), il = Z.argmax(axis=0)
        if (nrow == 0 && nl > 0)
            throw std::invalid_argument("zero-size array to reduction operation maximum which has no identity");
        std::vector<double> Zl(nl);
        std::vector<std::int64_t> il(nl, 0);
        for (std::size_t j = 0; j < nl; ++j) {
            double m = Z[j];
            double best = Z[j];
            std::size_t bi = 0;
            bool nan_found = std::isnan(best);
            for (std::size_t i = 1; i < nrow; ++i) {
                const double v = Z[i * nl + j];
                if (std::isnan(v)) {
                    if (!std::isnan(m)) m = v;
                    if (!nan_found) {
                        nan_found = true;
                        bi = i;
                    }
                    continue;
                }
                if (!std::isnan(m) && v > m) m = v;
                if (!nan_found && v > best) {
                    best = v;
                    bi = i;
                }
            }
            Zl[j] = m;
            il[j] = static_cast<std::int64_t>(bi);
        }
        const std::int64_t n_iter = std::max<std::int64_t>(1, args.n_best);
        for (std::int64_t kk = 0; kk < n_iter; ++kk) {
            const std::size_t j = np_argmax(Zl);
            if (kk && !std::isfinite(Zl[j])) break;
            const auto i = static_cast<std::size_t>(il[j]);
            double di = 0.0, dj = 0.0;
            if (i > 0 && i + 1 < nrow) di = parabola(Z[(i - 1) * nl + j], Z[i * nl + j], Z[(i + 1) * nl + j]);
            if (j > 0 && j + 1 < nl) dj = parabola(Z[i * nl + j - 1], Z[i * nl + j], Z[i * nl + j + 1]);
            const double delta_in =
                (static_cast<double>(static_cast<std::int64_t>(i) - nd) + di) * step;
            const double t0abs_in = static_cast<double>(base + static_cast<std::int64_t>(j)) + dj + frac0;
            const SegmentTiming st = segment_timing(x, sig2, kn, t0abs_in, delta_in, sps, n_pos);
            peaks.push_back({st.score, Z[i * nl + j], st.t0abs, st.delta});
            if (trace)
                trace->peaks.push_back(FitPeak{lpi, static_cast<std::int64_t>(i), static_cast<std::int64_t>(j), di,
                                               dj, Z[i * nl + j], t0abs_in, delta_in, st.t0abs, st.delta,
                                               st.score});
            const std::int64_t lo_j = std::max<std::int64_t>(static_cast<std::int64_t>(j) - csps, 0);
            const std::int64_t hi_j = std::min<std::int64_t>(static_cast<std::int64_t>(j) + csps + 1,
                                                             static_cast<std::int64_t>(nl));
            for (std::int64_t q = lo_j; q < hi_j; ++q) Zl[static_cast<std::size_t>(q)] = -INF;
        }
    }
    // peaks.sort(key=lambda b: -b[0]): stable; a NaN score (only from a
    // capture holding NaN) is put last here
    std::stable_sort(peaks.begin(), peaks.end(), [](const Peak& p, const Peak& q) {
        if (std::isnan(q.score)) return !std::isnan(p.score);
        if (std::isnan(p.score)) return false;
        return -p.score < -q.score;
    });
    std::vector<Timing> out;
    std::vector<double> seen;
    const double sd_ppm = py_sq(step * 1e6);
    for (const Peak& p : peaks) {
        bool dup = false;
        for (const double t : seen)
            if (std::abs(p.t0abs - t) < sps) {
                dup = true;
                break;
            }
        if (dup) continue;
        seen.push_back(p.t0abs);
        Timing tm;
        tm.tau0 = p.t0abs - t0_index;
        tm.ppm = p.delta * 1e6;
        tm.gamma = 0.0;
        tm.z = p.zmax;
        tm.cov = math::Matrix(3, 3, {0.25, 0.0, 0.0, 0.0, sd_ppm, 0.0, 0.0, 0.0, 0.0});
        out.push_back(std::move(tm));
    }
    const std::int64_t keep = args.n_best <= 1 ? 1 : args.n_best;
    if (static_cast<std::int64_t>(out.size()) > keep) out.resize(static_cast<std::size_t>(keep));
    return out;
}

std::vector<Timing> fit_timing(const ChanCapture& chan, int ch_fs, const PathFn& fn, const FrameSpec& spec,
                               double t0_index, const FitTimingArgs& args, FitTimingTrace* trace) {
    ce_only(spec, "track.fit_timing", "ltrack.fit_timing_l");
    const std::vector<cplx> z_d = derotate(chan, fn);
    return fit_timing(std::span<const cplx>(z_d), ch_fs, spec, t0_index, args, trace);
}

// --- the frequency path, measured on the frame ----------------------------------

namespace {

// int(chan.t0_index + spec.n_pos * spec.t_sym * fs), the frame's last CH
// index (exclusive) in a capture
std::int64_t frame_end(const ChanCapture& chan, const FrameSpec& spec, double fs) {
    return py_trunc(chan.t0_index + static_cast<double>(spec.n_pos()) * spec.t_sym() * fs);
}

// 1 << math.ceil(math.log2(m))
std::int64_t pow2_at_least(std::int64_t m) {
    if (m <= 0) throw std::invalid_argument("math domain error");
    const std::int64_t e = py_ceil(std::log2(static_cast<double>(m)));
    if (e < 0) throw std::invalid_argument("negative shift count");
    return std::int64_t{1} << e;
}

// |fft(seg * win[:len(seg)], nfft)| ** 2
std::vector<double> power_spectrum(std::span<const cplx> seg, std::span<const double> win, std::int64_t nfft) {
    std::vector<cplx> w(seg.size());
    for (std::size_t i = 0; i < seg.size(); ++i) w[i] = np::mul(seg[i], cplx(win[i], 0.0));
    const std::vector<cplx> F = math::fft(std::span<const cplx>(w), nfft);
    std::vector<double> S(F.size());
    for (std::size_t i = 0; i < F.size(); ++i) S[i] = abs2(F[i]);
    return S;
}

}  // namespace

CarrierLines carrier_lines(std::span<const cplx> z, const ChanCapture& chan, const FrameSpec& spec, double seg_s,
                           double span_hz, std::optional<std::pair<double, double>> med_band, double ratio) {
    const auto fs = static_cast<double>(chan.fs);
    const std::int64_t n = py_round(seg_s * fs);
    const std::int64_t i0 = std::max<std::int64_t>(py_ceil(chan.t0_index), 0);
    const std::int64_t i1 = std::min<std::int64_t>(frame_end(chan, spec, fs), static_cast<std::int64_t>(z.size()));
    const std::int64_t nfft = pow2_at_least(8 * n);
    const std::vector<double> f = math::fftfreq(nfft, 1.0 / fs);
    const std::vector<double> win = hanning(n);
    std::vector<std::size_t> in_med;
    for (std::size_t q = 0; q < f.size(); ++q) {
        const bool in = med_band ? (f[q] >= med_band->first && f[q] <= med_band->second)
                                 : (std::abs(f[q]) <= 4 * span_hz + 10);
        if (in) in_med.push_back(q);
    }
    CarrierLines out;
    double prev = 0.0;
    const auto nz = static_cast<std::int64_t>(z.size());
    const std::int64_t stop = std::max(i1 - floordiv(n, 2), i0 + 1);
    if (n <= 0) throw std::invalid_argument("range() arg 3 must not be zero");
    std::vector<double> sm;
    for (std::int64_t a = i0; a < stop; a += n) {
        const std::int64_t b = std::min(a + n, nz);
        const std::int64_t len = std::max<std::int64_t>(b - a, 0);
        if (len < floordiv(n, 2)) break;
        const std::span<const cplx> seg = z.subspan(static_cast<std::size_t>(a), static_cast<std::size_t>(len));
        const std::vector<double> S = power_spectrum(seg, win, nfft);
        // k: the first maximum of S over |f - prev| <= span_hz
        std::optional<std::size_t> k;
        for (std::size_t q = 0; q < f.size(); ++q) {
            if (!(std::abs(f[q] - prev) <= span_hz)) continue;
            if (!k) {
                k = q;
                if (std::isnan(S[q])) break;
                continue;
            }
            if (std::isnan(S[q])) {
                k = q;
                break;
            }
            if (S[q] > S[*k]) k = q;
        }
        if (!k) empty_argmax();
        sm.clear();
        for (const std::size_t q : in_med) sm.push_back(S[q]);
        const double med = math::median(std::span<const double>(sm));
        if (!(med > 0 && S[*k] >= ratio * med)) continue;   // a blanked (all-zero) segment has med 0
        const auto kk = static_cast<std::int64_t>(*k);
        const double d = parabola(S[static_cast<std::size_t>(floormod(kk - 1, nfft))], S[*k],
                                  S[static_cast<std::size_t>(floormod(kk + 1, nfft))]);
        const double fr = f[*k] + d * fs / static_cast<double>(nfft);
        out.t.push_back(chan.t_s(static_cast<double>(a) + static_cast<double>(len) / 2));
        out.f.push_back(fr);
        out.w.push_back(S[*k] / med);
        prev = fr;
    }
    return out;
}

PathResult refine_path(const ChanCapture& chan, const FrameSpec& spec, const FreqPath& path,
                       const RefinePathArgs& args) {
    const PathFn fn(path.t_s, path.f_hz, std::span<const double>(path.weight), 60.0);
    const std::vector<cplx> z = derotate(chan, fn);
    const CarrierLines L = carrier_lines(z, chan, spec, args.seg_s, args.span_hz, args.med_band, args.ratio);
    if (L.t.size() < 2) return {path, true};
    const PathFn res(L.t, L.f, std::span<const double>(L.w), py_max(3 * args.seg_s, 60.0));
    const double step = args.seg_s;
    std::vector<double> ts;
    ts.reserve(L.t.size() + 2);
    ts.push_back(L.t.front() - 10);
    ts.insert(ts.end(), L.t.begin(), L.t.end());
    ts.push_back(L.t.back() + 10);
    const std::vector<double> f_ts = fn(ts), r_ts = res(ts);
    std::vector<double> f1(ts.size());
    for (std::size_t q = 0; q < ts.size(); ++q) f1[q] = f_ts[q] + r_ts[q];
    const PathFn fn1(ts, f1, std::nullopt, 60.0);
    const std::vector<cplx> z1 = derotate(chan, fn1);
    const CarrierLines L2 = carrier_lines(z1, chan, spec, rt::PATH_FINE_S, 1.0, args.med_band, args.ratio);
    const double n_fine = static_cast<double>(spec.n_pos()) * spec.t_sym() / rt::PATH_FINE_S;
    FreqPath out;
    if (static_cast<double>(L2.t.size()) >= py_max(8.0, args.fine_share * n_fine) &&
        math::median(std::span<const double>(L2.w)) >= args.fine_ratio) {
        const PathFn res2(L2.t, L2.f, std::span<const double>(L2.w), 2 * rt::PATH_FINE_S);
        out.t_s = arange_f(py_min(L2.t.front(), 0.0) - 10.0, L2.t.back() + 2.0 * rt::PATH_FINE_S, 1.0);
        const std::vector<double> a = fn1(out.t_s), b = res2(out.t_s);
        out.f_hz.resize(out.t_s.size());
        for (std::size_t q = 0; q < out.t_s.size(); ++q) out.f_hz[q] = a[q] + b[q];
    } else {
        out.t_s = arange_f(py_min(L.t.front(), 0.0) - 10.0, L.t.back() + 2.0 * step, 2.0);
        const std::vector<double> a = fn(out.t_s), b = res(out.t_s);
        out.f_hz.resize(out.t_s.size());
        for (std::size_t q = 0; q < out.t_s.size(); ++q) out.f_hz[q] = a[q] + b[q];
    }
    out.weight.assign(out.t_s.size(), 1.0);
    return {std::move(out), false};
}

PathResult line_path(const ChanCapture& chan, const FrameSpec& spec, const FreqPath& path) {
    const PathFn fn(path.t_s, path.f_hz, std::span<const double>(path.weight), 60.0);
    const std::vector<double> t_frame{0.0, static_cast<double>(spec.n_pos()) * spec.t_sym()};
    const std::vector<double> f_lin = fn(t_frame);
    const double d0 = (f_lin[1] - f_lin[0]) / t_frame[1];   // Hz/s
    const double tm = t_frame[1] / 2;
    const double f_mid = f_lin[0] + d0 * tm;
    const std::vector<double> tl{-10.0, 0.0, 10.0};
    std::vector<double> ft(3), ff(3);
    for (std::size_t q = 0; q < 3; ++q) {
        ft[q] = tm + tl[q];
        ff[q] = f_mid + d0 * tl[q];
    }
    const PathFn flat(ft, ff);
    const std::vector<cplx> z = derotate(chan, flat);   // the path's line removed
    const auto fs = static_cast<double>(chan.fs);
    const std::int64_t n = py_round(rt::LINE_SEG_S * fs);
    const std::int64_t i0 = std::max<std::int64_t>(py_ceil(chan.t0_index), 0);
    const std::int64_t i1 = std::min<std::int64_t>(frame_end(chan, spec, fs), static_cast<std::int64_t>(z.size()));
    const std::int64_t nfft = pow2_at_least(4 * n);
    const std::vector<double> win = hanning(n);
    const auto nz = static_cast<std::int64_t>(z.size());
    if (n <= 0) throw std::invalid_argument("range() arg 3 must not be zero");
    std::vector<std::vector<double>> S;
    std::vector<double> ts;
    for (std::int64_t a = i0; a < i1 - floordiv(n, 2); a += n) {
        const std::int64_t b = std::min(a + n, nz);
        const std::int64_t len = std::max<std::int64_t>(b - a, 0);
        if (len < floordiv(n, 2)) break;
        const std::span<const cplx> seg = std::span<const cplx>(z).subspan(static_cast<std::size_t>(a), static_cast<std::size_t>(len));
        std::vector<double> P = power_spectrum(seg, win, nfft);
        const double m = py_max(math::median(std::span<const double>(P)), 1e-300);
        for (double& v : P) v = v / m;
        S.push_back(std::move(P));
        ts.push_back(chan.t_s(static_cast<double>(a) + static_cast<double>(len) / 2) - tm);
    }
    if (S.size() < 3) return {path, true};
    const double df = fs / static_cast<double>(nfft);
    const std::int64_t kspan = py_ceil(rt::LINE_SPAN_HZ / df);
    std::vector<std::int64_t> k;
    for (std::int64_t q = 0; q < kspan + 1; ++q) k.push_back(q);
    for (std::int64_t q = nfft - kspan; q < nfft; ++q) k.push_back(q);
    std::vector<double> fk(k.size());
    for (std::size_t q = 0; q < k.size(); ++q)
        fk[q] = static_cast<double>(k[q] > floordiv(nfft, 2) ? k[q] - nfft : k[q]) * df;

    struct Best {
        double tot, foff, dd;
    };
    std::vector<double> tot(k.size());
    std::vector<std::int64_t> sh(S.size());
    auto search = [&](std::span<const double> drifts) {
        Best best{-INF, 0.0, 0.0};
        for (const double dd : drifts) {
            for (std::size_t r = 0; r < S.size(); ++r) sh[r] = np_i64(std::nearbyint(dd * ts[r] / df));
            // S[rows, (k + sh) % nfft].sum(axis=0): the rows added in order
            for (std::size_t r = 0; r < S.size(); ++r) {
                const std::vector<double>& row = S[r];
                for (std::size_t q = 0; q < k.size(); ++q) {
                    const auto col = static_cast<std::size_t>(floormod(add_wrap(k[q], sh[r]), nfft));
                    tot[q] = r == 0 ? row[col] : tot[q] + row[col];
                }
            }
            const std::size_t j = np_argmax(tot);
            if (tot[j] > best.tot) best = {tot[j], fk[j], dd};
        }
        return best;
    };
    // coarse drift steps, then steps fine enough that the line moves under
    // a bin across the frame
    std::vector<double> coarse = arange_f(-rt::LINE_DRIFT_HZ_MIN, rt::LINE_DRIFT_HZ_MIN + 1e-9, rt::LINE_DRIFT_STEP);
    for (double& v : coarse) v = v / 60.0;
    const double dd1 = search(coarse).dd;
    double maxabs = std::abs(ts[0]);
    for (const double v : ts) {
        if (std::isnan(v)) {
            maxabs = v;
            break;
        }
        maxabs = std::max(maxabs, std::abs(v));
    }
    const double fine = 0.5 * df / py_max(maxabs, 1.0);
    std::vector<double> drifts = arange_f(-rt::LINE_DRIFT_STEP / 60.0, rt::LINE_DRIFT_STEP / 60.0 + 1e-12, fine);
    for (double& v : drifts) v = dd1 + v;
    const Best b = search(drifts);
    FreqPath out;
    out.t_s = arange_f(-rt::LEAD_LINE_S, static_cast<double>(spec.n_pos()) * spec.t_sym() + 10.0, 10.0);
    out.f_hz.resize(out.t_s.size());
    const double f_base = f_mid + b.foff;
    for (std::size_t q = 0; q < out.t_s.size(); ++q) out.f_hz[q] = f_base + (d0 + b.dd) * (out.t_s[q] - tm);
    out.weight.assign(out.t_s.size(), 1.0);
    return {std::move(out), false};
}

FreqPath flat_path(const FreqPath& path) {
    const std::vector<double> t0{0.0};
    const double f0 = freq_path_spline(path)(std::span<const double>(t0))[0];
    return FreqPath{{-10.0, 0.0, 1800.0}, {f0, f0, f0}, {1.0, 1.0, 1.0}};
}

// --- gate V ------------------------------------------------------------------------

TrackResult verify_track(std::shared_ptr<const ChanCapture> chan, const FrameSpec& spec, const Detection& cand,
                         const VerifyArgs& args) {
    ce_only(spec, "track.verify_track", "ltrack.verify_track_l");
    const PathFn fn = freq_path_spline(cand.path);
    auto z_d = std::make_shared<const std::vector<cplx>>(derotate(*chan, fn));
    std::optional<Timing> tm = cand.timing;
    if (args.fit || !tm) {
        std::optional<double> tau0;
        if (tm) tau0 = tm->tau0;
        std::optional<double> rng = args.tau_range;
        if (tau0 && !rng) {
            const double sd = !tm->cov.empty() ? std::sqrt(py_max(tm->cov(0, 0), 0.0)) : 1.0;
            rng = py_max(3.0, 4.0 * sd);
        }
        FitTimingArgs fa;
        fa.tau0_init = tau0;
        fa.tau_range = rng;
        tm = fit_timing(std::span<const cplx>(*z_d), config::CH_FS, spec, chan->t0_index, fa).front();
    }
    const Classes classes = make_classes(spec);
    TrackArgs ta;
    ta.classes = classes;
    ta.timing = tm;
    ta.freq = fn;
    ta.withhold_known = true;
    ta.outer = 0;
    ta.refine = false;
    ta.z_d = z_d;
    ta.noise_psd = args.noise_psd;
    TrackResult tr = track(chan, spec, cand, ta);
    erase(tr, args.erase_s);
    const double keyed_from = spec.keyed_start_pos() - tr.lead_in_s / config::T_SYM;
    const Templates tp = templates(spec, tr.pos, classes.mu, classes.nu, nullptr, false, keyed_from);
    tr.z_ref = z_ref_stat(tr, tp.c);
    return tr;
}

std::optional<Detection> verify(std::shared_ptr<const ChanCapture> chan, const FrameSpec& spec, const Detection& cand,
                                const VerifyArgs& args) {
    ce_only(spec, "track.verify", "ltrack.verify_l");
    // the acquisition's drift can be wrong (a preamble's estimate on a fading
    // or warming-up signal): the path is measured on the frame itself
    // first, from the candidate's path and, failing that, its flat version
    std::optional<TrackResult> tr;
    FreqPath path = cand.path;
    const FreqPath flat = flat_path(cand.path);
    const FreqPath* bases[3] = {&cand.path, &flat, nullptr};
    auto accepted = [&] { return tr && tr->z_ref > config::Z_ACCEPT; };
    for (const FreqPath* base : bases) {
        // the last try, a line over the whole frame, is for weak candidates
        FreqPath pth = base ? refine_path(*chan, spec, *base).path : line_path(*chan, spec, cand.path).path;
        Detection c2 = cand;
        c2.path = pth;
        VerifyArgs va;
        va.fit = args.fit;
        va.tau_range = args.tau_range;
        va.erase_s = args.erase_s;
        va.noise_psd = args.noise_psd;
        TrackResult alt = verify_track(chan, spec, c2, va);
        if (!tr || alt.z_ref > tr->z_ref) {
            tr = std::move(alt);
            path = pth;
        }
        if (accepted()) break;
        if (!base && args.fit && !cand.timing && rt::V_TIMING_TRIES > 1) {
            // a blind (A3) candidate near the floor, on its whole-frame line:
            // the runners-up of the coarse timing statistic are tracked too
            const PathFn fn = freq_path_spline(pth);
            const std::vector<cplx> z_d = derotate(*chan, fn);
            FitTimingArgs fa;
            fa.n_best = rt::V_TIMING_TRIES;
            const std::vector<Timing> tms =
                fit_timing(std::span<const cplx>(z_d), config::CH_FS, spec, chan->t0_index, fa);
            VerifyArgs vb;
            vb.fit = false;
            vb.erase_s = args.erase_s;
            vb.noise_psd = args.noise_psd;
            for (std::size_t q = 1; q < tms.size(); ++q) {
                Detection c3 = cand;
                c3.path = pth;
                c3.timing = tms[q];
                TrackResult alt3 = verify_track(chan, spec, c3, vb);
                if (!tr || alt3.z_ref > tr->z_ref) {
                    tr = std::move(alt3);
                    path = pth;
                }
                if (accepted()) break;
            }
        }
        if (accepted()) break;
    }
    if (!tr || !(tr->z_ref > config::Z_ACCEPT)) return std::nullopt;
    Detection out;
    out.f_hz = cand.f_hz;
    out.path = std::move(path);
    out.timing = tr->timing;
    out.z_ref = tr->z_ref;
    out.method = cand.method;
    out.lead_in_s = cand.lead_in_s;
    out.waveform = cand.waveform;
    return out;
}

}  // namespace sstvae::qrss::track
