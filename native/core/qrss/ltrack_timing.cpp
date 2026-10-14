// Port of sstvae/qrss/ltrack.py's coarse timing and gate V (fit_timing_l,
// _segment_timing_l, _lag_phase, _clock_phase, refine_path_l,
// verify_track_l, resid_ratio, verify_l); see ltrack_timing.hpp.

#include "qrss/ltrack_timing.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "math/convolve.hpp"
#include "math/linalg.hpp"
#include "math/npcomplex.hpp"
#include "math/npstats.hpp"
#include "qrss/npsum.hpp"
#include "qrss/rational.hpp"
#include "qrss/track.hpp"

namespace sstvae::qrss::ltrack {

namespace {

namespace np = math::np;
namespace rt = rxconfig::track;
namespace rl = rxconfig::ltrack;
using frame::FrameSpec;
using track::ChanCapture;
using track::Known;
using track::LClasses;
using track::LTrackResult;
using track::PathFn;
using track::SegmentTiming;
using types::Detection;
using types::FreqPath;
using types::Timing;

constexpr double PI = 3.141592653589793;   // np.pi
constexpr double TWO_PI = 2 * PI;          // 2 * np.pi, exact
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr double L_T_SYM = config::L_T_SYM;
// _lag_phase's 12207.0 / 242500.0 (f_p / CH_FS in turns per CH sample)
constexpr double LAG_TURN = 12207.0 / 242500.0;
static_assert(config::L_PILOT_HZ.num == 12207 && config::L_PILOT_HZ.den * config::CH_FS == 242500);
static_assert(rl::FP_NUM_ == 12207 && rl::FP_DEN_ == 16000);

// --- Python's scalar semantics -------------------------------------------------

// max(a, b): the first argument unless the second compares greater, so a NaN
// first argument is kept.
inline double py_max(double a, double b) { return b > a ? b : a; }

// int() of an integral float, with Python's errors for NaN and infinity.
std::int64_t int_of_integral(double v) {
    if (std::isnan(v)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(v)) throw std::overflow_error("cannot convert float infinity to integer");
    if (!(v >= -9223372036854775808.0 && v < 9223372036854775808.0))
        throw std::overflow_error("Python int too large to convert to C long");
    return static_cast<std::int64_t>(v);
}
// math.floor, math.ceil and int() to an int.
std::int64_t py_floor(double v) { return int_of_integral(std::isfinite(v) ? std::floor(v) : v); }
std::int64_t py_ceil(double v) { return int_of_integral(std::isfinite(v) ? std::ceil(v) : v); }
std::int64_t py_trunc(double v) { return int_of_integral(std::isfinite(v) ? std::trunc(v) : v); }

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
// |z|^2 as np.abs(z) ** 2.
inline double abs2(cplx z) {
    const double a = np::abs(z);
    return a * a;
}
// np.exp(-2j * np.pi * t) for a float t: (cos, sin) of the imaginary part
// through libm. CPython (3.13) multiplies -2j (-0.0, -2) by the float pi as
// by (pi, 0), so -2j * np.pi is (+0.0, -2 pi); numpy's product of that with
// (t, 0) is (+0.0, 0 * 0 + (-2 pi) t): the + 0.0 makes t = 0 give +0.0, not
// -0.0, and sin keeps the sign. (CPython 3.14's mixed-mode rule scales both
// parts instead, (-0.0, -2 pi), which would make it -0.0.)
inline cplx exp_neg_i(double t) {
    const double y = 0.0 + -TWO_PI * t;
    return {std::cos(y), std::sin(y)};
}

[[noreturn]] void index_error(std::int64_t i, std::size_t size) {
    throw std::out_of_range("index " + std::to_string(i) + " is out of bounds for axis 0 with size " +
                            std::to_string(size));
}
[[noreturn]] void empty_argmax() { throw std::invalid_argument("attempt to get argmax of an empty sequence"); }

// x[i] with numpy's negative indexing (IndexError outside).
template <typename T>
inline T at(std::span<const T> x, std::int64_t i) {
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

std::vector<double> as_doubles(std::span<const std::int64_t> v) {
    std::vector<double> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = static_cast<double>(v[i]);
    return out;
}

// np.asarray(kn.val, dtype=np.complex128): a real Known's values as (v, 0).
std::vector<cplx> complex_vals(const Known& kn) {
    if (kn.is_complex()) return kn.cplx();
    const std::vector<double>& r = kn.real();
    std::vector<cplx> out(r.size());
    for (std::size_t i = 0; i < r.size(); ++i) out[i] = {r[i], 0.0};
    return out;
}

void same_count(const Known& kn, std::size_t n_val) {
    if (kn.pos.size() != n_val)
        throw std::invalid_argument("operands could not be broadcast together with shapes (" + std::to_string(n_val) +
                                    ",) (" + std::to_string(kn.pos.size()) + ",) ");
}

// np.sum(np.abs(a) ** 2) (pairwise).
double sum_abs2(std::span<const cplx> a) {
    std::vector<double> sq(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) sq[i] = abs2(a[i]);
    return npsum::sum(std::span<const double>(sq));
}

// One term of the reference's complex interpolation at receive index idx:
// (x[i0c] (1 - f) + x[i0c + 1] f) ok, i0 = floor(idx), f = idx - i0, ok =
// i0 >= 0 and i0 + 1 < len(x), i0c = clip(i0, 0, len(x) - 2), every
// product numpy's complex one with the float promoted to (v, 0).
inline cplx term_c(std::span<const cplx> x, double idx) {
    const auto nx = static_cast<std::int64_t>(x.size());
    const std::int64_t i0 = np_i64(std::floor(idx));
    const double f = idx - static_cast<double>(i0);
    const bool ok = i0 >= 0 && i0 + 1 < nx;
    const std::int64_t i0c = std::min(std::max(i0, std::int64_t{0}), nx - 2);   // np.clip
    const cplx p = np::mul(at(x, i0c), 1.0 - f), q = np::mul(at(x, i0c + 1), f);
    const cplx v{p.real() + q.real(), p.imag() + q.imag()};
    return np::mul(v, ok ? 1.0 : 0.0);
}

// _segment_timing_l's per-segment pieces.
struct SegL {
    std::vector<double> pos;   // the segment's positions (float)
    std::vector<cplx> a;       // their symbols
    double den = 0.0;          // math.sqrt(sig2 / 2 * np.sum(np.abs(a[sel]) ** 2))
};

// The positions of a boolean mask over a layout's stream symbols:
// np.asarray(lay.pos)[known] (IndexError when the lengths differ).
std::vector<std::int64_t> masked_pos(const frame::Layout& lay, std::span<const std::uint8_t> known) {
    if (known.size() != lay.pos.size())
        throw std::out_of_range("boolean index did not match indexed array along axis 0; size of axis is " +
                                std::to_string(lay.pos.size()) + " but size of corresponding boolean axis is " +
                                std::to_string(known.size()));
    std::vector<std::int64_t> out;
    for (std::size_t s = 0; s < known.size(); ++s)
        if (known[s]) out.push_back(lay.pos[s]);
    return out;
}

}  // namespace

// --- the receiver-time mix's phases ----------------------------------------------

cplx lag_phase(double lag) {
    double turns = lag * LAG_TURN;
    turns = turns - std::floor(turns);
    return exp_neg_i(turns);
}

std::vector<cplx> lag_phase(std::span<const double> lag) {
    std::vector<cplx> out(lag.size());
    for (std::size_t i = 0; i < lag.size(); ++i) out[i] = lag_phase(lag[i]);
    return out;
}

std::vector<cplx> clock_phase(std::span<const double> pos, double d) {
    // pos * (d * _FP_NUM / _FP_DEN): (d * 12207) / 16000 first, a Python float
    const double k = (d * static_cast<double>(rl::FP_NUM_)) / static_cast<double>(rl::FP_DEN_);
    std::vector<cplx> out(pos.size());
    for (std::size_t i = 0; i < pos.size(); ++i) {
        double turns = pos[i] * k;
        turns = turns - std::floor(turns);
        out[i] = exp_neg_i(turns);
    }
    return out;
}

// --- the coarse timing fit ----------------------------------------------------------

std::vector<cplx> fit_mixed(std::span<const cplx> z_d, double n_ref, int fs) {
    std::vector<cplx> prod(z_d.size());
    for (std::size_t i = 0; i < z_d.size(); ++i)
        prod[i] = np::mul(z_d[i], std::conj(lag_phase(static_cast<double>(i) - n_ref)));
    const std::vector<double> taps = track::mf_taps(static_cast<double>(fs), L_T_SYM);
    return math::oaconvolve(std::span<const cplx>(prod), std::span<const double>(taps), math::ConvMode::Same);
}

std::vector<cplx> fit_statistic_l(std::span<const cplx> z_d, std::span<const cplx> y, double lp_hz, int fs) {
    const std::vector<cplx> uc = track::lowpass(z_d, lp_hz, static_cast<double>(fs));
    if (uc.size() != y.size())
        throw std::invalid_argument("operands could not be broadcast together with shapes (" +
                                    std::to_string(y.size()) + ",) (" + std::to_string(uc.size()) + ",) ");
    std::vector<cplx> x(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) x[i] = np::mul(y[i], std::conj(uc[i]));
    return x;
}

std::vector<cplx> fit_statistic_l(std::span<const cplx> z_d, double n_ref, double lp_hz, int fs) {
    const std::vector<cplx> y = fit_mixed(z_d, n_ref, fs);
    return fit_statistic_l(z_d, std::span<const cplx>(y), lp_hz, fs);
}

std::vector<double> fit_grid_l(std::span<const cplx> x, const Known& known, std::int64_t base, std::int64_t nlag,
                               double frac0, std::span<const double> deltas, double sps, double norm) {
    if (nlag < 0) throw std::invalid_argument("negative dimensions are not allowed");
    const std::vector<cplx> a = complex_vals(known);
    same_count(known, a.size());
    const auto nl = static_cast<std::size_t>(nlag);
    std::vector<double> Z(deltas.size() * nl);
    if (nl == 0) return Z;
    std::vector<double> xr(x.size()), xi(x.size());
    for (std::size_t n = 0; n < x.size(); ++n) {
        xr[n] = x[n].real();
        xi[n] = x[n].imag();
    }
    const std::vector<double> pos = as_doubles(known.pos);
    std::vector<double> lags(nl);
    for (std::size_t j = 0; j < nl; ++j) lags[j] = static_cast<double>(j);
    const std::vector<cplx> lagph = lag_phase(std::span<const double>(lags));
    // The complex sum over k as four real gathers (track::fit_grid's kernel):
    // Re = sum gr xr + sum (-gi) xi, Im = sum gr xi + sum gi xr, each term
    // the reference's interpolation of one part, outside x 0.
    Known kr{known.pos, std::vector<double>(a.size())};
    Known ki{known.pos, std::vector<double>(a.size())};
    Known kmi{known.pos, std::vector<double>(a.size())};
    auto& gr = std::get<0>(kr.val);
    auto& gi = std::get<0>(ki.val);
    auto& gmi = std::get<0>(kmi.val);
    for (std::size_t i = 0; i < deltas.size(); ++i) {
        const std::vector<cplx> cp = clock_phase(std::span<const double>(pos), deltas[i]);
        for (std::size_t k = 0; k < a.size(); ++k) {
            const cplx g = np::mul(std::conj(a[k]), cp[k]);
            gr[k] = g.real();
            gi[k] = g.imag();
            gmi[k] = -g.imag();
        }
        const std::span<const double> d1(deltas.data() + i, 1);
        const std::vector<double> A = track::fit_grid(xr, kr, base, nlag, frac0, d1, sps, 1.0);
        const std::vector<double> B = track::fit_grid(xi, kmi, base, nlag, frac0, d1, sps, 1.0);
        const std::vector<double> C = track::fit_grid(xi, kr, base, nlag, frac0, d1, sps, 1.0);
        const std::vector<double> D = track::fit_grid(xr, ki, base, nlag, frac0, d1, sps, 1.0);
        double* row = Z.data() + i * nl;
        for (std::size_t j = 0; j < nl; ++j)
            row[j] = np::mul(lagph[j], cplx(A[j] + B[j], C[j] + D[j])).real() / norm;
    }
    return Z;
}

SegmentTiming segment_timing_l(std::span<const cplx> x, double sig2, const Known& known, double t0abs, double delta,
                               double sps, std::int64_t n_pos, double n_ref, double seg_s) {
    const std::vector<double> pos = as_doubles(known.pos);
    const std::vector<cplx> av = complex_vals(known);
    if (seg_s == 0.0) throw ZeroDivision("float division by zero");
    const std::int64_t nseg = std::max<std::int64_t>(1, py_ceil(static_cast<double>(n_pos) * L_T_SYM / seg_s));
    if (pos.empty()) throw std::invalid_argument("zero-size array to reduction operation minimum which has no identity");
    double pmin = pos[0], pmax = pos[0];
    for (const double p : pos) {
        pmin = std::min(pmin, p);
        pmax = std::max(pmax, p);
    }
    const std::vector<double> edges = linspace(pmin, pmax + 1, nseg + 1);
    const std::vector<double> lags = arange_f(-sps, sps + 1e-9, 0.1);
    same_count(known, av.size());
    std::vector<SegL> segs;
    for (std::int64_t s = 0; s < nseg; ++s) {
        const double s0 = edges[static_cast<std::size_t>(s)], s1 = edges[static_cast<std::size_t>(s + 1)];
        SegL sd;
        for (std::size_t q = 0; q < pos.size(); ++q)
            if (pos[q] >= s0 && pos[q] < s1) {
                sd.pos.push_back(pos[q]);
                sd.a.push_back(av[q]);
            }
        if (sd.pos.size() < 8) continue;
        sd.den = std::sqrt(sig2 / 2 * sum_abs2(sd.a));
        segs.push_back(std::move(sd));
    }

    // stat(sel, t0, d, lg) into c: Re(lag_phase(t0 + lg - n_ref) (g @ v)) / den
    std::vector<cplx> r;
    auto stat = [&](const SegL& sd, double t0, double d, std::span<const double> lg, std::vector<double>& c) {
        const std::vector<cplx> cp = clock_phase(std::span<const double>(sd.pos), d);
        r.assign(lg.size(), cplx(0.0, 0.0));
        for (std::size_t k = 0; k < sd.pos.size(); ++k) {
            const double o = t0 + sd.pos[k] * sps * (1 + d);
            const cplx g = np::mul(std::conj(sd.a[k]), cp[k]);
            const double gr = g.real(), gi = g.imag();
            for (std::size_t j = 0; j < lg.size(); ++j) {
                const cplx v = term_c(x, o + lg[j]);
                r[j] = {r[j].real() + (gr * v.real() - gi * v.imag()), r[j].imag() + (gr * v.imag() + gi * v.real())};
            }
        }
        c.resize(lg.size());
        for (std::size_t j = 0; j < lg.size(); ++j)
            c[j] = np::mul(lag_phase((t0 + lg[j]) - n_ref), r[j]).real() / sd.den;
    };

    std::vector<double> c;
    // local(sel, t0, d): (offset, Z^2) or nullopt
    auto local = [&](const SegL& sd, double t0, double d) -> std::optional<std::pair<double, double>> {
        stat(sd, t0, d, std::span<const double>(lags), c);
        for (const double v : c)
            if (!std::isfinite(v)) return std::nullopt;
        const std::size_t k = np_argmax(c);
        if (c[k] < 4.0) return std::nullopt;
        const double dk = (k > 0 && k + 1 < c.size()) ? track::parabola(c[k - 1], c[k], c[k + 1]) : 0.0;
        return std::make_pair(lags[k] + dk * 0.1, py_sq(c[k]));
    };

    // fit(tp, ab, wt, t0_, d_): track's, the outer `delta` in its short branch
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
            std::vector<double> rr(n);
            for (std::size_t q = 0; q < n; ++q) rr[q] = ab[q] - tp[q] * sps * (1 + delta);
            const double t0 = math::average(std::span<const double>(rr), std::span<const double>(wt));
            return {t0, delta};
        }
        std::vector<double> sw(n);
        for (std::size_t q = 0; q < n; ++q) sw[q] = std::sqrt(wt[q]);
        auto solve = [&](const std::vector<std::size_t>& rows) {
            math::Matrix A(rows.size(), 2);
            std::vector<double> b(rows.size());
            for (std::size_t i = 0; i < rows.size(); ++i) {
                const std::size_t q = rows[i];
                A(i, 0) = 1.0 * sw[q];
                A(i, 1) = (tp[q] * sps) * sw[q];
                b[i] = ab[q] * sw[q];
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
    for (const SegL& sd : segs) {
        const auto rr = local(sd, t0, d);
        if (!rr) continue;
        const double pm = math::mean(std::span<const double>(sd.pos));
        tp.push_back(pm);
        ab.push_back(t0 + pm * sps * (1 + d) + rr->first);
        wt.push_back(rr->second);
        std::tie(t0, d) = fit(tp, ab, wt, t0, d);
    }
    if (tp.empty()) return {t0abs, delta, 0.0};
    // the evidence for the final line: every segment's Z at it (no search)
    const double zero[1] = {0.0};
    double score = 0.0;
    for (const SegL& sd : segs) {
        stat(sd, t0, d, std::span<const double>(zero, 1), c);
        score += py_sq(py_max(c[0], 0.0));
    }
    return {t0, d, score};
}

namespace {

struct PeaksL {
    std::vector<track::TimingPeak> peaks;   // carrier estimate by carrier estimate, rank by rank
    double sps = 0.0, step = 0.0;
};

// fit_timing_l's grid search (the reference's _timing_peaks_l).
PeaksL timing_peaks_l(std::span<const cplx> z_d, int ch_fs, const FrameSpec& spec, double t0_index,
                      const FitTimingLArgs& args, track::FitTimingTrace* trace) {
    const int fs = ch_fs;
    const Known own = args.known ? Known{} : known_symbols_l(spec);
    const Known& kn = args.known ? *args.known : own;
    const std::int64_t n_pos = spec.n_pos();
    const double sps = L_T_SYM * fs;
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
    const double n_ref = static_cast<double>(base) + frac0;
    if (nlag < 0) throw std::invalid_argument("negative dimensions are not allowed");
    const std::vector<cplx> a = complex_vals(kn);
    same_count(kn, a.size());
    const double sa2 = sum_abs2(a);
    if (trace) {
        *trace = track::FitTimingTrace{};
        trace->base = base;
        trace->nlag = nlag;
        trace->nd = nd;
        trace->frac0 = frac0;
        trace->step = step;
    }
    const std::vector<cplx> y = fit_mixed(z_d, n_ref, fs);

    PeaksL res;
    res.sps = sps;
    res.step = step;
    std::vector<track::TimingPeak>& peaks = res.peaks;
    const auto nl = static_cast<std::size_t>(nlag);
    const std::size_t nrow = deltas.size();
    const std::int64_t csps = py_ceil(sps);
    const double ln2 = std::log(2.0);
    for (std::size_t lpi = 0; lpi < rl::CARRIER_LP_HZ_L.size(); ++lpi) {
        const std::vector<cplx> x = fit_statistic_l(z_d, std::span<const cplx>(y), rl::CARRIER_LP_HZ_L[lpi], fs);
        const auto nx = static_cast<std::int64_t>(x.size());
        // sel = slice(max(base, 0), max(min(base + nlag + int(dur) + 1, len(x)), max(base, 0) + 1))
        const std::int64_t s0 = std::max<std::int64_t>(base, 0);
        const std::int64_t s1 = std::max(std::min(base + nlag + py_trunc(dur) + 1, nx), s0 + 1);
        std::vector<double> sq;
        for (std::int64_t q = s0; q < std::min(s1, nx); ++q) sq.push_back(abs2(x[static_cast<std::size_t>(q)]));
        const double sig2 = py_max(math::median(std::span<const double>(sq)) / ln2, 1e-30);
        const double norm = std::sqrt(sig2 / 2 * sa2);
        if (trace) trace->sig2.push_back(sig2);
        const std::vector<double> Z = fit_grid_l(x, kn, base, nlag, frac0, deltas, sps, norm);
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
            if (i > 0 && i + 1 < nrow) di = track::parabola(Z[(i - 1) * nl + j], Z[i * nl + j], Z[(i + 1) * nl + j]);
            if (j > 0 && j + 1 < nl) dj = track::parabola(Z[i * nl + j - 1], Z[i * nl + j], Z[i * nl + j + 1]);
            const double delta_in = (static_cast<double>(static_cast<std::int64_t>(i) - nd) + di) * step;
            const double t0abs_in = static_cast<double>(base + static_cast<std::int64_t>(j)) + dj + frac0;
            const SegmentTiming st = segment_timing_l(x, sig2, kn, t0abs_in, delta_in, sps, n_pos, n_ref);
            peaks.push_back({kk, st.score, Z[i * nl + j], st.t0abs, st.delta});
            if (trace)
                trace->peaks.push_back(track::FitPeak{lpi, static_cast<std::int64_t>(i), static_cast<std::int64_t>(j),
                                                      di, dj, Z[i * nl + j], t0abs_in, delta_in, st.t0abs, st.delta,
                                                      st.score});
            const std::int64_t lo_j = std::max<std::int64_t>(static_cast<std::int64_t>(j) - csps, 0);
            const std::int64_t hi_j = std::min<std::int64_t>(static_cast<std::int64_t>(j) + csps + 1,
                                                             static_cast<std::int64_t>(nl));
            for (std::int64_t q = lo_j; q < hi_j; ++q) Zl[static_cast<std::size_t>(q)] = -INF;
        }
    }
    return res;
}

}  // namespace

std::vector<Timing> fit_timing_l(std::span<const cplx> z_d, int ch_fs, const FrameSpec& spec, double t0_index,
                                 const FitTimingLArgs& args, track::FitTimingTrace* trace) {
    l_only(spec, "ltrack.fit_timing_l");
    PeaksL p = timing_peaks_l(z_d, ch_fs, spec, t0_index, args, trace);
    return track::timings_of(std::move(p.peaks), args.n_best, p.sps, p.step, t0_index);
}

track::FitTimingTries fit_timing_tries_l(std::span<const cplx> z_d, int ch_fs, const FrameSpec& spec,
                                         double t0_index, const FitTimingLArgs& args, track::FitTimingTrace* trace) {
    l_only(spec, "ltrack.fit_timing_l");
    PeaksL p = timing_peaks_l(z_d, ch_fs, spec, t0_index, args, trace);
    return track::tries_of(std::move(p.peaks), args.n_best, p.sps, p.step, t0_index);
}

std::vector<Timing> fit_timing_l(const ChanCapture& chan, int ch_fs, const PathFn& fn, const FrameSpec& spec,
                                 double t0_index, const FitTimingLArgs& args, track::FitTimingTrace* trace) {
    l_only(spec, "ltrack.fit_timing_l");
    const std::vector<cplx> z_d = track::derotate(chan, fn);
    return fit_timing_l(std::span<const cplx>(z_d), ch_fs, spec, t0_index, args, trace);
}

// --- gate V --------------------------------------------------------------------------

track::PathResult refine_path_l(const ChanCapture& chan, const FrameSpec& spec, const FreqPath& path) {
    const double span = rt::PATH_SPAN_HZ;
    track::RefinePathArgs a;
    a.seg_s = rt::PATH_SEG_S;
    a.span_hz = span;
    a.med_band = std::make_pair(rl::PATH_MED_LO_HZ, 4 * span + 10.0);
    return track::refine_path(chan, spec, path, a);
}

double resid_ratio(const LTrackResult& tr, std::span<const cplx> c_known, std::span<const double> gain,
                   std::span<const double> dist) {
    const LClasses* cl = tr.l_classes();
    if (!cl) throw std::invalid_argument("resid_ratio: the track has no L classes");
    const auto lay = frame::layout(tr.spec);
    const std::vector<std::int64_t> kp = masked_pos(*lay, cl->known);
    const std::int64_t p_lo = tr.p_lo();
    const auto npos = static_cast<std::int64_t>(tr.pos.size());
    std::vector<std::size_t> gk;
    for (const std::int64_t p : kp) {
        const std::int64_t g = p - p_lo;
        if (g < 0 || g >= npos) continue;
        if (static_cast<std::size_t>(g) >= tr.psi.size()) index_error(g, tr.psi.size());
        if (tr.psi[static_cast<std::size_t>(g)] >= rt::PSI_MIN) gk.push_back(static_cast<std::size_t>(g));
    }
    if (gk.empty()) return 1.0;
    std::vector<double> num, den;
    for (const std::size_t g : gk) {
        const cplx m = tr.m.at(g), u = tr.u.at(g);
        const double N = tr.N.at(g), P = tr.P.at(g);
        if (g >= c_known.size()) index_error(static_cast<std::int64_t>(g), c_known.size());
        const cplx c = c_known[g];
        const double u2 = abs2(u), c2 = abs2(c);
        const double V = c2 * (N * u2 + c2 * u2 * P + N * P) / 2;
        if (!(V > 0)) continue;
        if (g >= gain.size()) index_error(static_cast<std::int64_t>(g), gain.size());
        if (g >= dist.size()) index_error(static_cast<std::int64_t>(g), dist.size());
        const double w = c2 * u2 / V;
        const double om = w * w * c2 * u2;
        const cplx uc = np::mul(u, c);
        const double e2 = abs2(cplx(m.real() - uc.real(), m.imag() - uc.imag()));
        const double gg = gain[g];
        const double e0 = N + c2 * P + u2 * (gg * gg) * dist[g];
        num.push_back(om * e2);
        den.push_back(om * e0);
    }
    if (num.empty()) return 1.0;
    const double d = npsum::sum(std::span<const double>(den));
    return d > 0 ? npsum::sum(std::span<const double>(num)) / d : 1.0;
}

LTrackResult verify_track_l(std::shared_ptr<const ChanCapture> chan, const FrameSpec& spec, const Detection& cand,
                            const track::VerifyArgs& args) {
    l_only(spec, "ltrack.verify_track_l");
    const PathFn fn = track::freq_path_spline(cand.path);
    auto z_d = std::make_shared<const std::vector<cplx>>(track::derotate(*chan, fn));
    std::optional<Timing> tm = cand.timing;
    if (args.fit || !tm) {
        std::optional<double> tau0;
        if (tm) tau0 = tm->tau0;
        std::optional<double> rng = args.tau_range;
        if (tau0 && !rng) {
            const double sd = !tm->cov.empty() ? std::sqrt(py_max(tm->cov(0, 0), 0.0)) : 1.0;
            rng = py_max(3.0, 4.0 * sd);
        }
        FitTimingLArgs fa;
        fa.tau0_init = tau0;
        fa.tau_range = rng;
        tm = fit_timing_l(std::span<const cplx>(*z_d), config::CH_FS, spec, chan->t0_index, fa).front();
    }
    const LClasses classes = make_classes_l(spec);
    TrackLArgs ta;
    ta.classes = classes;
    ta.timing = tm;
    ta.freq = fn;
    ta.withhold_known = true;
    ta.outer = 0;
    ta.refine = false;
    ta.z_d = z_d;
    ta.noise_psd = args.noise_psd;
    LTrackResult tr = track_l(chan, spec, cand, ta);
    track::erase(tr, args.erase_s);
    const DataTemplates dt = data_templates(spec, tr.pos, classes);
    const double z = track::z_ref_stat(tr, dt.c, std::span<const cplx>(dt.c));
    tr.resid_ratio = resid_ratio(tr, dt.c, dt.gain, dt.dist);
    tr.z_ref = z / std::sqrt(py_max(tr.resid_ratio, 1.0));
    return tr;
}

std::optional<Detection> verify_l(std::shared_ptr<const ChanCapture> chan, const FrameSpec& spec,
                                  const Detection& cand, const track::VerifyArgs& args) {
    l_only(spec, "ltrack.verify_l");
    std::optional<LTrackResult> tr;
    FreqPath path = cand.path;
    const FreqPath flat = track::flat_path(cand.path);
    const FreqPath* bases[3] = {&cand.path, &flat, nullptr};
    auto accepted = [&] { return tr && tr->z_ref > config::Z_ACCEPT; };
    for (const FreqPath* base : bases) {
        FreqPath pth = base ? refine_path_l(*chan, spec, *base).path : track::line_path(*chan, spec, cand.path).path;
        Detection c2 = cand;
        c2.path = pth;
        track::VerifyArgs va;
        va.fit = args.fit;
        va.tau_range = args.tau_range;
        va.erase_s = args.erase_s;
        va.noise_psd = args.noise_psd;
        // a blind candidate on its whole-frame line: the runners-up of the
        // coarse timing statistic are tracked too, from the grid that gives
        // the best timing (fit_timing_tries_l)
        const bool tries = !base && args.fit && !cand.timing && rt::V_TIMING_TRIES > 1;
        std::vector<Timing> tms;
        if (tries) {
            const PathFn fn = track::freq_path_spline(pth);
            const std::vector<cplx> z_d = track::derotate(*chan, fn);
            FitTimingLArgs fa;
            fa.n_best = rt::V_TIMING_TRIES;
            track::FitTimingTries ft =
                fit_timing_tries_l(std::span<const cplx>(z_d), config::CH_FS, spec, chan->t0_index, fa);
            c2.timing = std::move(ft.best);   // the timing verify_track_l would fit
            va.fit = false;
            va.tau_range.reset();
            tms = std::move(ft.tries);
        }
        LTrackResult alt = verify_track_l(chan, spec, c2, va);
        if (!tr || alt.z_ref > tr->z_ref) {
            tr = std::move(alt);
            path = pth;
        }
        if (accepted()) break;
        if (tries) {
            track::VerifyArgs vb;
            vb.fit = false;
            vb.erase_s = args.erase_s;
            vb.noise_psd = args.noise_psd;
            for (std::size_t q = 1; q < tms.size(); ++q) {
                Detection c3 = cand;
                c3.path = pth;
                c3.timing = tms[q];
                LTrackResult alt3 = verify_track_l(chan, spec, c3, vb);
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
    out.waveform = config::WAVEFORM_L;
    return out;
}

}  // namespace sstvae::qrss::ltrack
