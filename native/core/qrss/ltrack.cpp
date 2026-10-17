// Port of sstvae/qrss/ltrack.py's L tracker; see ltrack.hpp.

#include "qrss/ltrack.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "math/convolve.hpp"
#include "math/errors.hpp"
#include "math/linalg.hpp"
#include "math/npcomplex.hpp"
#include "math/npstats.hpp"
#include "qrss/ce.hpp"
#include "qrss/npsum.hpp"
#include "qrss/pytext.hpp"
#include "qrss/rational.hpp"
#include "qrss/sequences.hpp"
#include "qrss/track.hpp"

namespace sstvae::qrss::ltrack {

namespace {

namespace np = math::np;
namespace rt = rxconfig::track;
namespace rl = rxconfig::ltrack;
namespace rd = rxconfig::ldemod;
using frame::FrameSpec;
using track::ChanCapture;
using track::LClasses;
using track::LTrackResult;
using types::Timing;

constexpr double PI = 3.141592653589793;   // np.pi
constexpr double TWO_PI = 2 * PI;          // 2 * np.pi, exact
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr std::int64_t MF_RES = rt::MF_RES;
constexpr std::int64_t SPAN = config::SPAN;
constexpr std::int64_t PILOT_K = rl::PILOT_K;
constexpr std::int64_t PILOT_REACH = rl::PILOT_REACH;
constexpr double L_T_SYM = config::L_T_SYM;
constexpr std::int64_t I64_MIN = std::numeric_limits<std::int64_t>::min();
static_assert(PILOT_REACH == SPAN * PILOT_K);

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
// math.floor / int() to an int.
std::int64_t py_floor(double v) { return int_of_integral(std::isfinite(v) ? std::floor(v) : v); }
std::int64_t py_trunc(double v) { return int_of_integral(std::isfinite(v) ? std::trunc(v) : v); }
// math.floor / math.ceil as a float, with their errors for NaN and infinity
// (Python's ints are unbounded, so a huge finite value is not an error).
double py_floor_f(double v) {
    if (std::isnan(v)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(v)) throw std::overflow_error("cannot convert float infinity to integer");
    return std::floor(v);
}
double py_ceil_f(double v) {
    if (std::isnan(v)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(v)) throw std::overflow_error("cannot convert float infinity to integer");
    return std::ceil(v);
}

// x ** 2 for a Python float: C pow (the exponent kept out of the
// optimiser's reach so it is not rewritten as x * x).
double py_sq(double x) {
    volatile double two = 2.0;
    return std::pow(x, static_cast<double>(two));
}

// CPython 3.13's complex arithmetic (a float operand is complex(f, 0.0);
// no mixed-mode forms before 3.14), written out so neither std::complex's
// operators nor contraction reorders it.
inline cplx py_add(cplx a, cplx b) { return {a.real() + b.real(), a.imag() + b.imag()}; }
inline cplx py_mul(cplx a, cplx b) {
    return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
}
// _Py_c_quot: Smith's method dividing by the denominator, with C11 Annex
// G's recovery of infinities and zeros; a zero divisor is ZeroDivisionError.
cplx py_div(cplx a, cplx b) {
    const double ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
    const double abs_br = br < 0 ? -br : br;
    const double abs_bi = bi < 0 ? -bi : bi;
    double rr = 0.0, ri = 0.0;
    if (abs_br >= abs_bi) {
        if (abs_br == 0.0) throw ZeroDivision("complex division by zero");
        const double ratio = bi / br;
        const double denom = br + bi * ratio;
        rr = (ar + ai * ratio) / denom;
        ri = (ai - ar * ratio) / denom;
    } else if (abs_bi >= abs_br) {
        const double ratio = br / bi;
        const double denom = br * ratio + bi;
        rr = (ar * ratio + ai) / denom;
        ri = (ai * ratio - ar) / denom;
    } else {
        rr = ri = std::numeric_limits<double>::quiet_NaN();
    }
    if (std::isnan(rr) && std::isnan(ri)) {
        if ((std::isinf(ar) || std::isinf(ai)) && std::isfinite(br) && std::isfinite(bi)) {
            const double x = std::copysign(std::isinf(ar) ? 1.0 : 0.0, ar);
            const double y = std::copysign(std::isinf(ai) ? 1.0 : 0.0, ai);
            rr = INF * (x * br + y * bi);
            ri = INF * (y * br - x * bi);
        } else if ((std::isinf(abs_br) || std::isinf(abs_bi)) && std::isfinite(ar) && std::isfinite(ai)) {
            const double x = std::copysign(std::isinf(br) ? 1.0 : 0.0, br);
            const double y = std::copysign(std::isinf(bi) ? 1.0 : 0.0, bi);
            rr = 0.0 * (ar * x + ai * y);
            ri = 0.0 * (ai * x - ar * y);
        }
    }
    return {rr, ri};
}
// abs() of a Python complex: hypot, an infinite part's magnitude, NaN, and
// OverflowError when finite parts overflow.
double py_cabs(cplx z) {
    const double re = z.real(), im = z.imag();
    if (!std::isfinite(re) || !std::isfinite(im)) {
        if (std::isinf(re)) return std::fabs(re);
        if (std::isinf(im)) return std::fabs(im);
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double r = std::hypot(re, im);
    if (!std::isfinite(r)) throw std::overflow_error("absolute value too large");
    return r;
}
inline bool py_cfinite(cplx z) { return std::isfinite(z.real()) && std::isfinite(z.imag()); }

// --- numpy's elementwise semantics ---------------------------------------------

// float64 -> int64 as numpy casts on x86-64 (INT64_MIN for NaN and anything
// out of range).
inline std::int64_t np_i64(double x) {
    if (!(x >= -9223372036854775808.0 && x < 9223372036854775808.0)) return I64_MIN;
    return static_cast<std::int64_t>(x);
}
inline std::int64_t add_wrap(std::int64_t a, std::int64_t b) {
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(a) + static_cast<std::uint64_t>(b));
}
inline double np_clip(double x, double lo, double hi) {
    if (std::isnan(x)) return x;
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
// np.maximum / np.minimum against a scalar (NaN on either side propagates).
inline double np_maximum(double x, double b) {
    if (std::isnan(x)) return x;
    if (std::isnan(b)) return b;
    return x > b ? x : b;
}
inline double np_minimum(double x, double b) {
    if (std::isnan(x)) return x;
    if (std::isnan(b)) return b;
    return x < b ? x : b;
}
// |z|^2 as np.abs(z) ** 2.
inline double abs2(cplx z) {
    const double a = np::abs(z);
    return a * a;
}
// exp of (+0, x) as numpy's complex exp computes it: (cos x, sin x).
inline cplx exp_i(double x) { return {std::cos(x), std::sin(x)}; }

inline double np_sum(const std::vector<double>& v) { return npsum::sum(std::span<const double>(v)); }
inline cplx np_sum(const std::vector<cplx>& v) { return npsum::sum(std::span<const cplx>(v)); }
inline double np_mean(const std::vector<double>& v) { return math::mean(std::span<const double>(v)); }

// np.min / np.max of a non-empty float64 array (the first NaN wins).
double np_min(std::span<const double> x) {
    double m = x[0];
    for (const double v : x) {
        if (std::isnan(v)) return v;
        if (v < m) m = v;
    }
    return m;
}
double np_max(std::span<const double> x) {
    double m = x[0];
    for (const double v : x) {
        if (std::isnan(v)) return v;
        if (v > m) m = v;
    }
    return m;
}

[[noreturn]] void index_error(std::int64_t i, std::size_t size) {
    throw std::out_of_range("index " + std::to_string(i) + " is out of bounds for axis 0 with size " +
                            std::to_string(size));
}
// A numpy index into an axis of `size`: negative wraps, outside is IndexError.
std::size_t np_index(std::int64_t i, std::size_t size) {
    const auto n = static_cast<std::int64_t>(size);
    if (i < -n || i >= n) index_error(i, size);
    return static_cast<std::size_t>(i < 0 ? i + n : i);
}

[[noreturn]] void broadcast_error(std::size_t a, std::size_t b) {
    throw std::invalid_argument("operands could not be broadcast together with shapes (" + std::to_string(a) +
                                ",) (" + std::to_string(b) + ",) ");
}
// The length two 1-D operands broadcast to.
std::size_t bcast(std::size_t a, std::size_t b) {
    if (a == b || b == 1) return a;
    if (a == 1) return b;
    broadcast_error(a, b);
}
inline std::size_t at(std::size_t size, std::size_t i) { return size == 1 ? 0 : i; }
void same_length(std::size_t a, std::size_t b) {
    if (a != b) broadcast_error(a, b);
}
// a[idx] = v's shape check: v of idx's length or of one element.
void check_assign(std::size_t values, std::size_t targets) {
    if (values != targets && values != 1)
        throw std::invalid_argument("shape mismatch: value array of shape (" + std::to_string(values) +
                                    ",) could not be broadcast to indexing result of shape (" +
                                    std::to_string(targets) + ",)");
}

// np.asarray(lay.pos)[mask] (IndexError when the lengths differ).
std::vector<std::int64_t> masked_pos(const frame::Layout& lay, std::span<const std::uint8_t> mask) {
    if (mask.size() != lay.pos.size())
        throw std::out_of_range("boolean index did not match indexed array along axis 0; size of axis is " +
                                std::to_string(lay.pos.size()) + " but size of corresponding boolean axis is " +
                                std::to_string(mask.size()));
    std::vector<std::int64_t> out;
    for (std::size_t s = 0; s < mask.size(); ++s)
        if (mask[s]) out.push_back(lay.pos[s]);
    return out;
}

std::vector<double> as_doubles(std::span<const std::int64_t> v) {
    std::vector<double> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = static_cast<double>(v[i]);
    return out;
}

// np.argmin with numpy's NaN rule (the first NaN wins); x is not empty.
std::size_t np_argmin(std::span<const double> x) {
    std::size_t best = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (std::isnan(x[i])) return i;
        if (x[i] < x[best]) best = i;
    }
    return best;
}

// t0_index + rx_index(timing, pos) at L's period: the CH indices positions
// are read at.
std::vector<double> read_times(const Timing& timing, std::span<const double> pos, std::int64_t n_pos,
                               double t0_index) {
    std::vector<double> tp = track::rx_index(timing, pos, n_pos, config::CH_FS, L_T_SYM);
    for (double& v : tp) v = t0_index + v;
    return tp;
}

// _tau_of_rx(timing, np.arange(n) - t0_index): the sender time of every CH
// sample.
std::vector<double> tau_all(const Timing& timing, std::size_t n, double t0_index, std::int64_t n_pos) {
    std::vector<double> rel(n);
    for (std::size_t i = 0; i < n; ++i) rel[i] = static_cast<double>(i) - t0_index;
    return track::tau_of_rx(timing, rel, n_pos, L_T_SYM);
}

// The pilot filter's taps, once.
const std::vector<double>& taps() {
    static const std::vector<double> t = pilot_taps();
    return t;
}

template <typename T>
std::vector<T> pilot_filter_impl(std::span<const T> x, std::span<const double> tp) {
    const std::vector<double>& h = taps();
    const std::vector<T> y = math::oaconvolve(x, std::span<const double>(h), math::ConvMode::Same);
    const auto n = static_cast<std::int64_t>(y.size());
    std::vector<T> out(tp.size());
    if (tp.empty()) return out;
    if (n == 0) index_error(-1, 0);   // y[np.clip(i0, 0, -1)]
    for (std::size_t p = 0; p < tp.size(); ++p) {
        const std::int64_t i0 = np_i64(std::floor(tp[p]));
        const double f = tp[p] - static_cast<double>(i0);
        const std::int64_t i1 = add_wrap(i0, 1);
        const T zero{};
        const T a = (i0 >= 0 && i0 < n) ? y[static_cast<std::size_t>(i0)] : zero;
        const T b = (i1 >= 0 && i1 < n) ? y[static_cast<std::size_t>(i1)] : zero;
        if constexpr (std::is_same_v<T, cplx>) {
            const cplx s = np::mul(a, cplx(1 - f, 0.0));
            const cplx e = np::mul(b, cplx(f, 0.0));
            out[p] = cplx(s.real() + e.real(), s.imag() + e.imag());
        } else {
            out[p] = a * (1 - f) + b * f;
        }
    }
    return out;
}

// A 0/1 mask as the reference's ok.any().
bool any(std::span<const std::uint8_t> m) {
    return std::any_of(m.begin(), m.end(), [](std::uint8_t v) { return v != 0; });
}

}  // namespace

// --- refusals -------------------------------------------------------------------

void l_only(const FrameSpec& spec, std::string_view what) {
    if (spec.waveform() == frame::Waveform::L) return;
    throw frame::WrongWaveform(std::string(what) + " is L's; " + pytext::repr(spec.name()) + " is waveform " +
                               std::to_string(static_cast<int>(spec.waveform())) + ": use track's");
}

// --- the filters -------------------------------------------------------------

namespace {

track::MfTable build_pilot_table() {
    const double t2 = static_cast<double>(PILOT_K) * L_T_SYM;
    const double sps = t2 * config::CH_FS;
    const std::int64_t half = int_of_integral(std::ceil(static_cast<double>(SPAN) * sps)) + 1;
    const std::int64_t mmax = SPAN * PILOT_K * 970;
    const auto den = static_cast<double>(PILOT_K * 970);
    std::vector<double> sq(static_cast<std::size_t>(2 * mmax + 1)), h(sq.size());
    for (std::int64_t m = -mmax; m <= mmax; ++m) {
        const double v = ce::rrc(static_cast<double>(m) / den);
        h[static_cast<std::size_t>(m + mmax)] = v;   // np.where(|m| <= mmax, ...): every m
        sq[static_cast<std::size_t>(m + mmax)] = v * v;
    }
    const double norm = std::sqrt(npsum::sum(std::span<const double>(sq)) / den);
    const std::int64_t lo = -half * MF_RES, hi = half * MF_RES + 1;
    std::vector<double> tab(static_cast<std::size_t>(hi - lo));
    for (std::int64_t k = lo; k < hi; ++k) {
        const double d = static_cast<double>(k) / static_cast<double>(MF_RES);
        const double x = d / sps;
        tab[static_cast<std::size_t>(k - lo)] = std::fabs(x) <= static_cast<double>(SPAN) ? ce::rrc(x) / norm : 0.0;
    }
    const double g0 = npsum::sum(std::span<const double>(h)) / norm / den;
    return track::MfTable(L_T_SYM, sps, static_cast<int>(half), std::move(tab), g0);
}

}  // namespace

const track::MfTable& pilot_table() {
    static const track::MfTable t = build_pilot_table();
    return t;
}

std::vector<double> pilot_taps() {
    const track::MfTable& tb = pilot_table();
    const std::int64_t half = tb.half();
    const std::span<const double> tab = tb.tab();
    std::vector<double> out(static_cast<std::size_t>(2 * half + 1));
    for (std::int64_t k = -half; k <= half; ++k)
        out[static_cast<std::size_t>(k + half)] = tab[static_cast<std::size_t>(k * MF_RES + half * MF_RES)] / tb.sps();
    return out;
}

std::vector<cplx> pilot_filter(std::span<const cplx> x, std::span<const double> tp) {
    return pilot_filter_impl<cplx>(x, tp);
}

std::vector<double> pilot_filter(std::span<const double> x, std::span<const double> tp) {
    return pilot_filter_impl<double>(x, tp);
}

std::vector<cplx> data_mix(std::span<const cplx> z_d, double t0_index, const Timing& timing, std::int64_t n_pos,
                           std::optional<std::span<const std::int64_t>> idx) {
    const std::size_t n = idx ? idx->size() : z_d.size();
    std::vector<double> rel(n);
    for (std::size_t i = 0; i < n; ++i)
        rel[i] = (idx ? static_cast<double>((*idx)[i]) : static_cast<double>(i)) - t0_index;
    const std::vector<double> tau = track::tau_of_rx(timing, rel, n_pos, L_T_SYM);
    const double k = static_cast<double>(rl::FP_NUM_) / static_cast<double>(rl::FP_DEN_);
    std::vector<cplx> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        double turns = tau[i] * k;
        turns -= std::floor(turns);
        const cplx z = idx ? z_d[np_index((*idx)[i], z_d.size())] : z_d[i];
        out[i] = np::mul(z, exp_i(TWO_PI * turns));
    }
    return out;
}

std::optional<std::pair<std::int64_t, std::int64_t>> data_mf_reach(std::size_t n_z, double t0_index,
                                                                   const Timing& timing, std::span<const double> pos,
                                                                   std::int64_t n_pos) {
    const track::MfTable& tb = track::mf_table(L_T_SYM);
    if (pos.empty()) return std::nullopt;
    const std::vector<double> tp = read_times(timing, pos, n_pos, t0_index);
    const auto half = static_cast<double>(tb.half());
    const auto len = static_cast<double>(n_z);
    // max(int(math.floor(tp.min())) - half - 1, 0) and min(int(math.ceil(tp.max())) + half + 2, len),
    // exact in doubles wherever they can decide anything (|values| far under 2^53)
    const double lo_f = py_max(py_floor_f(np_min(tp)) - half - 1, 0.0);
    const double hi_f = py_min(py_ceil_f(np_max(tp)) + half + 2, len);
    if (hi_f <= lo_f) return std::nullopt;
    return std::pair{static_cast<std::int64_t>(lo_f), static_cast<std::int64_t>(hi_f)};
}

std::vector<cplx> data_mf(std::span<const cplx> z_d, double t0_index, const Timing& timing,
                          std::span<const double> pos, std::int64_t n_pos) {
    const track::MfTable& tb = track::mf_table(L_T_SYM);
    if (pos.empty()) return {};
    const auto reach = data_mf_reach(z_d.size(), t0_index, timing, pos, n_pos);
    if (!reach) return std::vector<cplx>(pos.size(), cplx(0.0, 0.0));
    const std::vector<double> tp = read_times(timing, pos, n_pos, t0_index);
    std::vector<cplx> zz(z_d.size(), cplx(0.0, 0.0));
    const auto [lo, hi] = *reach;
    const std::int64_t width = 2 * tb.half() + 1;
    std::vector<std::int64_t> idx;
    if (static_cast<std::int64_t>(pos.size()) * width < hi - lo) {
        // a few positions: mix only the samples their filters reach (np.unique: ascending)
        std::vector<std::uint8_t> mark(z_d.size(), 0);
        const auto nz = static_cast<std::int64_t>(z_d.size());
        for (const double t : tp) {
            const std::int64_t base = np_i64(std::floor(t));
            for (std::int64_t k = -tb.half(); k < tb.half() + 2; ++k) {
                const std::int64_t i = add_wrap(base, k);
                if (i >= 0 && i < nz) mark[static_cast<std::size_t>(i)] = 1;
            }
        }
        for (std::size_t i = 0; i < mark.size(); ++i)
            if (mark[i]) idx.push_back(static_cast<std::int64_t>(i));
    } else {
        for (std::int64_t i = lo; i < hi; ++i) idx.push_back(i);
    }
    const std::vector<cplx> mixed = data_mix(z_d, t0_index, timing, n_pos, std::span<const std::int64_t>(idx));
    for (std::size_t j = 0; j < idx.size(); ++j) zz[static_cast<std::size_t>(idx[j])] = mixed[j];
    return track::symbol_mf(std::span<const cplx>(zz), timing, pos, t0_index, n_pos, &tb);
}

// --- classes and templates ---------------------------------------------------------

LClasses make_classes_l(const FrameSpec& spec, const std::vector<std::uint8_t>* hdr_bits,
                        const types::EmPrior* prior) {
    l_only(spec, "ltrack.make_classes_l");
    const auto lay = frame::layout(spec);
    const auto n = static_cast<std::size_t>(spec.n_sym());
    LClasses c;
    c.mu.assign(n, cplx(0.0, 0.0));
    c.nu.assign(n, 1.0);
    c.known.assign(n, 0);
    c.gain.assign(n, rd::G_DATA_L);
    for (std::size_t k = 0; k < lay->known_idx.size(); ++k) {
        const auto i = static_cast<std::size_t>(lay->known_idx[k]);
        c.mu[i] = lay->known_val_l[k];
        c.nu[i] = 0.0;
        c.known[i] = 1;
        c.gain[i] = rd::G_REF_L;
    }
    for (const std::int64_t i : lay->hdr_sym) c.gain[static_cast<std::size_t>(i)] = rd::G_HDR_L;
    if (hdr_bits && spec.n_hdr()) {
        const std::vector<cplx> q = sequences::qpsk(*hdr_bits);
        check_assign(q.size(), lay->hdr_sym.size());
        for (std::size_t k = 0; k < lay->hdr_sym.size(); ++k) {
            const auto i = static_cast<std::size_t>(lay->hdr_sym[k]);
            c.mu[i] = q[at(q.size(), k)];
            c.nu[i] = 0.0;
            c.known[i] = 1;
        }
    }
    const double rho = spec.hdr_rho();
    const double s = std::sqrt(1.0 + rho);
    std::optional<std::vector<double>> h;
    if (hdr_bits && rho > 0) {
        const frame::HdrBits b = frame::HdrBits::of(*hdr_bits);
        h = frame::spread_signs(spec, &b);
    }
    const std::vector<std::int64_t>& data = lay->data;
    if (prior) {
        const std::vector<double>& x = prior->x_hat;
        std::vector<double> v(prior->v.size());
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = np_clip(prior->v[i], 0.0, 1.0);
        std::vector<double> lm, lv(v.size());
        if (h) {
            const double sr = std::sqrt(rho);
            lm.resize(bcast(x.size(), h->size()));
            for (std::size_t i = 0; i < lm.size(); ++i)
                lm[i] = (x[at(x.size(), i)] + sr * (*h)[at(h->size(), i)]) / s;
            for (std::size_t i = 0; i < v.size(); ++i) lv[i] = v[i] / (1.0 + rho);
        } else {
            lm.resize(x.size());
            for (std::size_t i = 0; i < x.size(); ++i) lm[i] = x[i] / s;
            for (std::size_t i = 0; i < v.size(); ++i) lv[i] = (v[i] + rho) / (1.0 + rho);
        }
        const std::vector<cplx> pm = frame::pair(lm);
        check_assign(pm.size(), data.size());
        for (std::size_t k = 0; k < data.size(); ++k) c.mu[static_cast<std::size_t>(data[k])] = pm[at(pm.size(), k)];
        const std::size_t ne = (lv.size() + 1) / 2, no = lv.size() / 2;
        const std::size_t nn = bcast(ne, no);
        std::vector<double> pv(nn);
        for (std::size_t k = 0; k < nn; ++k) pv[k] = 0.5 * (lv[2 * at(ne, k)] + lv[2 * at(no, k) + 1]);
        check_assign(pv.size(), data.size());
        for (std::size_t k = 0; k < data.size(); ++k) c.nu[static_cast<std::size_t>(data[k])] = pv[at(pv.size(), k)];
    } else if (h) {
        const double sr = std::sqrt(rho);
        std::vector<double> lm(h->size());
        for (std::size_t i = 0; i < lm.size(); ++i) lm[i] = sr * (*h)[i] / s;
        const std::vector<cplx> pm = frame::pair(lm);
        check_assign(pm.size(), data.size());
        for (std::size_t k = 0; k < data.size(); ++k) {
            c.mu[static_cast<std::size_t>(data[k])] = pm[at(pm.size(), k)];
            c.nu[static_cast<std::size_t>(data[k])] = 1.0 / (1.0 + rho);
        }
    }
    if (hdr_bits) c.hdr_bits = *hdr_bits;
    c.hdr_known = hdr_bits != nullptr;
    c.spread_rho = hdr_bits ? 0.0 : rho;
    return c;
}

std::vector<double> distortion(const FrameSpec& spec) {
    const auto lay = frame::layout(spec);
    std::vector<double> d(static_cast<std::size_t>(spec.n_sym()), rd::D_PASS_L);
    for (const std::int64_t i : lay->hdr_sym) d[static_cast<std::size_t>(i)] = rd::D_HDR_L;
    for (const std::int64_t i : lay->known_idx) d[static_cast<std::size_t>(i)] = rd::D_REF_L;
    return d;
}

DataTemplates data_templates(const FrameSpec& spec, std::span<const std::int64_t> grid, const LClasses& classes,
                             bool withhold_known) {
    const auto lay = frame::layout(spec);
    if (grid.empty()) index_error(0, 0);
    const std::int64_t p_lo = grid[0];
    const std::size_t n = grid.size();
    const std::size_t ns = lay->pos.size();
    DataTemplates t;
    t.c.assign(n, cplx(0.0, 0.0));
    t.nu.assign(n, 0.0);
    t.gain.assign(n, 0.0);
    t.dist.assign(n, 0.0);
    t.is_sym.assign(n, 0);
    std::vector<cplx> mu_s = classes.mu;
    std::vector<double> nu_s = classes.nu;
    if (withhold_known) {
        const std::size_t km = bcast(classes.known.size(), classes.mu.size());
        mu_s.resize(km);
        for (std::size_t i = 0; i < km; ++i)
            mu_s[i] = classes.known[at(classes.known.size(), i)] ? cplx(0.0, 0.0) : classes.mu[at(classes.mu.size(), i)];
        const std::size_t kn = bcast(classes.known.size(), classes.nu.size());
        nu_s.resize(kn);
        for (std::size_t i = 0; i < kn; ++i)
            nu_s[i] = classes.known[at(classes.known.size(), i)] ? 1.0 : classes.nu[at(classes.nu.size(), i)];
    }
    const std::size_t gm = bcast(classes.gain.size(), mu_s.size());
    std::vector<cplx> gmu(gm);
    for (std::size_t i = 0; i < gm; ++i)
        gmu[i] = np::mul(cplx(classes.gain[at(classes.gain.size(), i)], 0.0), mu_s[at(mu_s.size(), i)]);
    const std::vector<double> d = distortion(spec);
    std::vector<std::size_t> gi(ns);
    for (std::size_t s = 0; s < ns; ++s) gi[s] = np_index(lay->pos[s] - p_lo, n);
    check_assign(gmu.size(), ns);
    for (std::size_t s = 0; s < ns; ++s) t.c[gi[s]] = gmu[at(gmu.size(), s)];
    check_assign(nu_s.size(), ns);
    for (std::size_t s = 0; s < ns; ++s) t.nu[gi[s]] = nu_s[at(nu_s.size(), s)];
    check_assign(classes.gain.size(), ns);
    for (std::size_t s = 0; s < ns; ++s) t.gain[gi[s]] = classes.gain[at(classes.gain.size(), s)];
    for (std::size_t s = 0; s < ns; ++s) t.dist[gi[s]] = d[s];
    for (std::size_t s = 0; s < ns; ++s) t.is_sym[gi[s]] = 1;
    return t;
}

std::vector<double> pilot_gain(std::span<const double> tau, const FrameSpec& spec, double lead_in_s) {
    const auto lay = frame::layout(spec);
    const std::int64_t n_sym = spec.n_sym(), n_pre = spec.n_pre(), n_hdr = spec.n_hdr();
    const double hdr0 = n_sym > n_pre ? static_cast<double>(lay->pos[static_cast<std::size_t>(n_pre)]) - 0.5 : INF;
    const double dat0 =
        n_sym > n_pre + n_hdr ? static_cast<double>(lay->pos[static_cast<std::size_t>(n_pre + n_hdr)]) - 0.5 : INF;
    const std::vector<double> W = lwave::id_weight(tau, spec, lead_in_s);
    std::vector<double> out(tau.size());
    for (std::size_t i = 0; i < tau.size(); ++i) {
        double g = rd::PILOT_GAIN_DATA;
        if (tau[i] < dat0) g = rd::PILOT_GAIN_HDR;
        if (tau[i] < hdr0) g = rd::PILOT_GAIN_PRE;
        out[i] = W[i] * rd::PILOT_GAIN_ID + (1.0 - W[i]) * g;
    }
    return out;
}

std::vector<cplx> pilot_wave(std::span<const double> tau, const FrameSpec& spec, const lwave::Keying* keying,
                             double lead_in_s) {
    const std::vector<double> g = pilot_gain(tau, spec, lead_in_s);
    const std::vector<cplx> p = lwave::pilot(tau, spec, keying, lead_in_s);
    const cplx d(lwave::a_id() * rd::G_REF_ABS, 0.0);
    std::vector<cplx> out(tau.size());
    for (std::size_t i = 0; i < tau.size(); ++i) out[i] = np::div(np::mul(cplx(g[i], 0.0), p[i]), d);
    return out;
}

std::pair<std::size_t, std::size_t> q_near_span(double q, std::span<const double> grid) {
    if (grid.empty()) throw std::invalid_argument("attempt to get argmin of an empty sequence");
    if (!(q > 0) && !std::isnan(q)) throw std::invalid_argument("math domain error");
    const double lq = std::log(q);
    std::vector<double> d(grid.size());
    for (std::size_t i = 0; i < grid.size(); ++i) d[i] = std::fabs(std::log(grid[i]) - lq);
    const auto i = static_cast<std::int64_t>(np_argmin(d));
    const auto n = static_cast<std::int64_t>(grid.size());
    const std::int64_t lo = std::max<std::int64_t>(0, i - 2), hi = std::min<std::int64_t>(i + 3, n);
    return {static_cast<std::size_t>(lo), static_cast<std::size_t>(hi)};
}

std::vector<double> q_near(double q, std::span<const double> grid) {
    const auto [lo, hi] = q_near_span(q, grid);
    return std::vector<double>(grid.begin() + static_cast<std::ptrdiff_t>(lo),
                               grid.begin() + static_cast<std::ptrdiff_t>(hi));
}

// --- _Model -----------------------------------------------------------------------

Model::Model(LTrackResult& tr, const FrameSpec& spec, const LClasses& classes, bool withhold_known,
             const lwave::Keying* keying, double lead, std::int64_t n_lead) {
    const std::vector<std::int64_t>& grid = tr.pos;
    DataTemplates t = data_templates(spec, grid, classes, withhold_known);
    c_all = data_templates(spec, grid, classes, false).c;
    c = std::move(t.c);
    nu = std::move(t.nu);
    gain = std::move(t.gain);
    dist = std::move(t.dist);
    is_sym = std::move(t.is_sym);
    const std::size_t n = grid.size();
    const auto lay = frame::layout(spec);
    is_known.assign(n, 0);
    for (const std::int64_t p : masked_pos(*lay, classes.known)) is_known[np_index(p - grid[0], n)] = 1;
    meas_d.assign(n, 0);
    need.assign(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const bool nz = np::abs(c[i]) > 0;
        meas_d[i] = (is_sym[i] && nz && !(withhold_known && is_known[i])) ? 1 : 0;
        need[i] = (is_sym[i] && (nz || is_known[i])) ? 1 : 0;
    }
    meas_p.assign(n, 0);
    for (std::size_t i = 0; i < n; ++i) meas_p[i] = grid[i] % 2 == 0 ? 1 : 0;
    if (!keying && spec.n_win()) {
        const std::vector<std::uint8_t> wu = track::window_unknown(spec, grid, false, PILOT_REACH);
        for (std::size_t i = 0; i < n; ++i) meas_p[i] = (meas_p[i] && !wu[i]) ? 1 : 0;
    }
    if (n_lead == 0)
        for (std::size_t i = 0; i < n; ++i) meas_p[i] = (meas_p[i] && grid[i] >= PILOT_REACH - SPAN) ? 1 : 0;
    measured.assign(n, 0);
    for (std::size_t i = 0; i < n; ++i) measured[i] = (meas_d[i] || meas_p[i]) ? 1 : 0;
    const std::vector<double> W = lwave::id_weight(as_doubles(grid), spec, lead);
    k_pl.resize(n);
    for (std::size_t i = 0; i < n; ++i) k_pl[i] = rl::K_PL_DATA * (1.0 - W[i]) + rl::K_PL_ID * W[i];
    tr.meas = meas_d;
    tr.meas_pl = meas_p;
}

namespace {

[[noreturn]] void none_operand() {
    throw math::TypeError("unsupported operand type(s) for *: 'NoneType' and 'NoneType'");
}

template <typename T>
const std::vector<T>& need_pl(const std::optional<std::vector<T>>& v) {
    if (!v) none_operand();
    return *v;
}

// tr.m_pl[ok], tr.N_pl[ok] with the field None.
template <typename T>
const std::vector<T>& need_sub(const std::optional<std::vector<T>>& v) {
    if (!v) throw math::TypeError("'NoneType' object is not subscriptable");
    return *v;
}

}  // namespace

Streams Model::streams(const LTrackResult& tr, std::span<const double> pu_loc, cplx delta, bool data, bool pilot,
                       const std::vector<std::uint8_t>* data_mask) const {
    const std::size_t n = tr.pos.size();
    Streams s;
    s.mu_d.assign(n, cplx(0.0, 0.0));
    s.R_d.assign(n, INF);
    s.mu_p.assign(n, cplx(0.0, 0.0));
    s.R_p.assign(n, INF);
    if (data) {
        const std::size_t nc = c.size();
        same_length(nc, tr.psi.size());
        same_length(meas_d.size(), nc);
        if (data_mask) same_length(meas_d.size(), data_mask->size());
        for (std::size_t i = 0; i < nc; ++i) {
            const cplx cp = np::mul(c[i], cplx(tr.psi[i], 0.0));
            const bool md = meas_d[i] && (!data_mask || (*data_mask)[i]);
            if (!(md && tr.psi[i] >= rt::PSI_MIN && np::abs(cp) >= rt::C_MIN)) continue;
            if (i >= n || i >= tr.m.size() || i >= tr.N.size() || i >= pu_loc.size())
                index_error(static_cast<std::int64_t>(i), std::min({n, tr.m.size(), tr.N.size(), pu_loc.size()}));
            s.mu_d[i] = np::div(tr.m[i], cp);
            const double g2 = gain[i] * gain[i];
            const double a = np::abs(cp);
            s.R_d[i] = (tr.N[i] + pu_loc[i] * g2 * (nu[i] + dist[i])) / (a * a);
        }
    }
    if (pilot) {
        const std::vector<cplx>& c_pl = need_pl(tr.c_pl);
        const std::vector<double>& psi_pl = need_pl(tr.psi_pl);
        const std::size_t np_ = c_pl.size();
        same_length(np_, psi_pl.size());
        same_length(meas_p.size(), np_);
        for (std::size_t i = 0; i < np_; ++i) {
            const cplx cp = np::mul(np::mul(c_pl[i], cplx(psi_pl[i], 0.0)), delta);
            if (!(meas_p[i] && psi_pl[i] >= rt::PSI_MIN && np::abs(cp) >= rt::C_MIN)) continue;
            const std::vector<cplx>& m_pl = need_sub(tr.m_pl);
            const std::vector<double>& N_pl = need_sub(tr.N_pl);
            if (i >= n || i >= m_pl.size() || i >= N_pl.size() || i >= pu_loc.size())
                index_error(static_cast<std::int64_t>(i), std::min({n, m_pl.size(), N_pl.size(), pu_loc.size()}));
            s.mu_p[i] = np::div(m_pl[i], cp);
            const double a = np::abs(cp);
            s.R_p[i] = (N_pl[i] + pu_loc[i] * k_pl[i]) / (a * a);
        }
    }
    return s;
}

Fused Model::fused(const LTrackResult& tr, std::span<const double> pu_loc, cplx delta, bool data, bool pilot,
                   const std::vector<std::uint8_t>* data_mask) const {
    const Streams s = streams(tr, pu_loc, delta, data, pilot, data_mask);
    const std::size_t n = s.R_d.size();
    Fused f;
    f.mu.resize(n);
    f.R.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double wd = 1.0 / s.R_d[i], wp = 1.0 / s.R_p[i];
        const double W = wd + wp;
        const double R = W > 0 ? 1.0 / W : INF;
        const cplx a = np::mul(s.mu_d[i], cplx(wd, 0.0));
        const cplx b = np::mul(s.mu_p[i], cplx(wp, 0.0));
        const cplx m = np::mul(cplx(a.real() + b.real(), a.imag() + b.imag()), cplx(R, 0.0));
        f.mu[i] = W > 0 ? m : cplx(0.0, 0.0);
        f.R[i] = std::isfinite(R) ? R + rt::R_FLOOR * pu_loc[i] : R;
    }
    return f;
}

double Model::pu_initial(const LTrackResult& tr) const {
    double num = 0.0, den = 0.0;
    {
        const std::vector<cplx>& c_pl = need_pl(tr.c_pl);
        const std::vector<double>& psi_pl = need_pl(tr.psi_pl);
        same_length(c_pl.size(), psi_pl.size());
        same_length(meas_p.size(), c_pl.size());
        std::vector<double> a, b;
        for (std::size_t i = 0; i < c_pl.size(); ++i) {
            const cplx cp = np::mul(c_pl[i], cplx(psi_pl[i], 0.0));
            if (!(meas_p[i] && psi_pl[i] >= rt::PSI_MIN && np::abs(cp) >= rt::C_MIN)) continue;
            const std::vector<cplx>& m_pl = need_sub(tr.m_pl);
            const std::vector<double>& N_pl = need_sub(tr.N_pl);
            a.push_back(abs2(m_pl.at(i)) - N_pl.at(i));
            b.push_back(abs2(cp) + k_pl[i]);
        }
        if (!a.empty()) {
            num += np_sum(a);
            den += np_sum(b);
        }
    }
    {
        same_length(c.size(), tr.psi.size());
        same_length(meas_d.size(), c.size());
        std::vector<double> a, b;
        for (std::size_t i = 0; i < c.size(); ++i) {
            const cplx cd = np::mul(c[i], cplx(tr.psi[i], 0.0));
            if (!(meas_d[i] && tr.psi[i] >= rt::PSI_MIN && np::abs(cd) >= rt::C_MIN)) continue;
            // tr.m[okd], tr.N[okd] with the field None (empty here: the grid is not)
            if (tr.m.empty() || tr.N.empty()) throw math::TypeError("'NoneType' object is not subscriptable");
            a.push_back(abs2(tr.m.at(i)) - tr.N.at(i));
            b.push_back(abs2(cd) + gain[i] * gain[i] * (nu[i] + dist[i]));
        }
        if (!a.empty()) {
            num += np_sum(a);
            den += np_sum(b);
        }
    }
    if (den <= 0) return 1e-30;
    const double floor = tr.N_pl ? 1e-3 * math::median(std::span<const double>(*tr.N_pl)) : 0.0;
    return py_max(py_max(num / den, floor), 1e-30);
}

// --- the tracker's pieces ------------------------------------------------------------

PuLocal pu_local(std::span<const cplx> u, std::span<const double> P, std::span<const std::uint8_t> measured) {
    const std::size_t n = u.size();
    same_length(n, P.size());
    const bool all = !any(measured);
    if (!all) same_length(measured.size(), n);
    std::vector<double> raw_sel;
    for (std::size_t i = 0; i < n; ++i)
        if (all || measured[i]) raw_sel.push_back(abs2(u[i]) + P[i]);
    const double cap = py_max(np_mean(raw_sel), 1e-30);
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = abs2(u[i]) + np_minimum(P[i], cap);
    PuLocal out;
    out.loc = track::smooth(x, py_trunc(rt::PU_SMOOTH_S / L_T_SYM));
    for (double& v : out.loc) v = np_maximum(v, 1e-30);
    std::vector<double> sel;
    for (std::size_t i = 0; i < n; ++i)
        if (all || measured[i]) sel.push_back(out.loc[i]);
    out.g = py_max(np_mean(sel), 1e-30);
    return out;
}

bool hold_ends(std::vector<cplx>& u, std::vector<double>& P, std::span<const double> R) {
    std::optional<std::size_t> first, last;
    for (std::size_t i = 0; i < R.size(); ++i)
        if (std::isfinite(R[i])) {
            if (!first) first = i;
            last = i;
        }
    if (!first) return false;
    const std::size_t a = *first, b = *last;
    if (a >= u.size() || a >= P.size()) index_error(static_cast<std::int64_t>(a), std::min(u.size(), P.size()));
    if (b >= u.size() || b >= P.size()) index_error(static_cast<std::int64_t>(b), std::min(u.size(), P.size()));
    const cplx ua = u[a], ub = u[b];
    const double pa = P[a], pb = P[b];
    for (std::size_t i = 0; i < a; ++i) {
        u[i] = ua;
        P[i] = pa;
    }
    for (std::size_t i = b + 1; i < u.size(); ++i) u[i] = ub;
    for (std::size_t i = b + 1; i < P.size(); ++i) P[i] = pb;
    return true;
}

bool lead_supported_l(const ChanCapture& chan, std::span<const cplx> z_d, const Timing& tm, const FrameSpec& spec,
                      double lead, const track::NoisePsdFn& noise_psd) {
    const std::int64_t n = py_floor(py_min(lead, rt::LEAD_KEEP_S) / L_T_SYM);
    if (n < 4 + PILOT_REACH) return true;
    const std::int64_t n_pos = spec.n_pos();
    std::vector<double> p;
    for (std::int64_t q = -SPAN - n; q < -SPAN - PILOT_REACH; q += 2) p.push_back(static_cast<double>(q));
    const std::size_t k = p.size();
    for (std::int64_t q = PILOT_REACH; q < std::min<std::int64_t>(300, n_pos); q += 2) p.push_back(static_cast<double>(q));
    const std::vector<double> rel = track::rx_index(tm, p, n_pos, config::CH_FS, L_T_SYM);
    std::vector<double> tp(rel.size()), ts(rel.size());
    for (std::size_t i = 0; i < rel.size(); ++i) {
        tp[i] = chan.t0_index + rel[i];
        ts[i] = rel[i] / config::CH_FS;
    }
    const std::vector<cplx> m = pilot_filter(z_d, tp);
    const std::vector<double> tau = tau_all(tm, z_d.size(), chan.t0_index, n_pos);
    const std::vector<cplx> c = pilot_filter(pilot_wave(tau, spec, nullptr, lead), tp);
    std::vector<double> N = track::noise_psd_at(chan, ts, noise_psd);
    for (double& v : N) v /= static_cast<double>(PILOT_K) * L_T_SYM;
    std::vector<double> dl, df, cl, cf;
    for (std::size_t i = 0; i < p.size(); ++i) {
        if (i < k) {
            dl.push_back(abs2(m[i]) - N[i]);
            cl.push_back(abs2(c[i]));
        } else {
            df.push_back(abs2(m[i]) - N[i]);
            cf.push_back(abs2(c[i]));
        }
    }
    const double el = np_mean(dl);
    const double ef = np_mean(df);
    const double ratio = np_mean(cl) / py_max(np_mean(cf), 1e-30);
    return el > ratio * ef / 3;
}

cplx fit_delta(const Model& model, const LTrackResult& tr, std::span<const cplx> v, std::span<const double> Pv,
               std::span<const double> pu_loc, cplx delta0) {
    const Streams s = model.streams(tr, pu_loc, {1.0, 0.0}, true, false);
    std::vector<std::size_t> ok;
    for (std::size_t i = 0; i < s.R_d.size(); ++i)
        if (std::isfinite(s.R_d[i])) ok.push_back(i);
    if (ok.empty()) return delta0;
    const cplx r0 = py_div({1.0, 0.0}, delta0);
    const double a2 = py_sq(py_cabs(r0));
    std::vector<double> w(ok.size()), wv(ok.size());
    std::vector<cplx> t(ok.size());
    for (std::size_t j = 0; j < ok.size(); ++j) {
        const std::size_t i = ok[j];
        if (i >= v.size() || i >= Pv.size() || i >= pu_loc.size())
            index_error(static_cast<std::int64_t>(i), std::min({v.size(), Pv.size(), pu_loc.size()}));
        w[j] = 1.0 / (s.R_d[i] + a2 * Pv[i] + rt::R_FLOOR * pu_loc[i]);
        wv[j] = w[j] * abs2(v[i]);
        t[j] = np::mul(np::mul(cplx(w[j], 0.0), std::conj(v[i])), s.mu_d[i]);
    }
    const double den = np_sum(wv);
    if (den <= 0) return delta0;
    const cplx r_ls = py_div(np_sum(t), {den, 0.0});
    const double inv = 1.0 / py_sq(rl::DELTA_PRIOR_SD);
    const cplx r = py_div(py_add(py_mul(r_ls, {den, 0.0}), {inv, 0.0}), {den + inv, 0.0});
    if (!py_cfinite(r) || py_cabs(r) < 1e-3) return delta0;
    return py_div({1.0, 0.0}, r);
}

// --- the tracker -----------------------------------------------------------------------

namespace {

// track_l()'s closures over one run: measure, pu_loc, run, pilot_track, fit_delta.
class LRun {
public:
    LRun(LTrackResult& tr, const ChanCapture& chan, const track::NoisePsdFn& noise_psd, const Model& model,
         const lwave::Keying* keying, double lead)
        : tr_(tr), chan_(chan), noise_psd_(noise_psd), model_(model), keying_(keying), lead_(lead),
          grid_d_(as_doubles(tr.pos)) {}

    // tr.z_d was replaced (Python keys its pilot cache on id(tr.z_d)).
    void zd_changed() { ++zd_gen_; }

    void measure(bool full = false) {
        const std::int64_t n_pos = tr_.spec.n_pos();
        const std::array<double, 3> key{tr_.timing.tau0, tr_.timing.ppm, tr_.timing.gamma};
        const std::vector<std::uint8_t>& sel = full ? model_.is_sym : model_.need;
        std::vector<double> gs;
        std::vector<std::size_t> is;
        for (std::size_t i = 0; i < sel.size(); ++i)
            if (sel[i]) {
                gs.push_back(grid_d_[i]);
                is.push_back(i);
            }
        const std::vector<cplx> mm = data_mf(*tr_.z_d, chan_.t0_index, tr_.timing, gs, n_pos);
        tr_.m.assign(grid_d_.size(), cplx(0.0, 0.0));
        for (std::size_t j = 0; j < is.size(); ++j) tr_.m[is[j]] = mm[j];
        if (!psi_key_ || *psi_key_ != key) {
            psi_ = track::psi_at(chan_, tr_.timing, grid_d_, n_pos, &track::mf_table(L_T_SYM));
            if (!chan_.keep) {
                psi_pl_.assign(grid_d_.size(), 1.0);
            } else {
                const std::vector<double> tp = read_times(tr_.timing, grid_d_, n_pos, chan_.t0_index);
                const std::vector<double> k(chan_.keep->begin(), chan_.keep->end());
                psi_pl_ = pilot_filter(std::span<const double>(k), tp);
                const double g0 = pilot_table().g0();
                for (double& v : psi_pl_) v = np_clip(v / g0, 0.0, 1.0);
            }
            psi_key_ = key;
        }
        tr_.psi = psi_;
        tr_.psi_pl = psi_pl_;
        const std::vector<double> psd = track::noise_psd_at(chan_, tr_.t_pos_s(), noise_psd_);
        tr_.N.resize(psd.size());
        std::vector<double> npl(psd.size());
        for (std::size_t i = 0; i < psd.size(); ++i) {
            tr_.N[i] = psd[i] / L_T_SYM;
            npl[i] = psd[i] / (static_cast<double>(PILOT_K) * L_T_SYM);
        }
        tr_.N_pl = std::move(npl);
        if (!pl_key_ || pl_key_->first != zd_gen_ || pl_key_->second != key) {
            const std::vector<double> tp = read_times(tr_.timing, grid_d_, n_pos, chan_.t0_index);
            const std::vector<double> tau = tau_all(tr_.timing, tr_.z_d->size(), chan_.t0_index, n_pos);
            c_pl_ = pilot_filter(pilot_wave(tau, tr_.spec, keying_, lead_), tp);
            m_pl_ = pilot_filter(*tr_.z_d, tp);
            pl_key_ = std::make_pair(zd_gen_, key);
        }
        tr_.m_pl = m_pl_;
        tr_.c_pl = c_pl_;
    }

    PuLocal pu_loc() const {
        if (!have_u_) {
            const double g = model_.pu_initial(tr_);
            return {std::vector<double>(tr_.pos.size(), g), g};
        }
        return pu_local(tr_.u, tr_.P, model_.measured);
    }

    double run(std::span<const double> q_grid) {
        PuLocal pl = pu_loc();
        const Fused f = model_.fused(tr_, pl.loc, tr_.delta);
        const double q = q_grid.size() == 1 ? q_grid[0] : track::select_q(f.mu, f.R, pl.g, q_grid).value().first;
        track::KalmanResult k = track::kalman_rts(f.mu, f.R, q, pl.g);
        hold_ends(k.u, k.P, f.R);
        tr_.u = std::move(k.u);
        tr_.P = std::move(k.P);
        tr_.q = q;
        tr_.loglik = k.loglik;
        tr_.pu = std::move(pl.loc);
        have_u_ = true;
        return q;
    }

    // The pilot-only track (of u delta) at q, from the current measurements.
    struct PilotTrack {
        std::vector<cplx> v;
        std::vector<double> Pv;
        std::vector<double> loc;
    };
    PilotTrack pilot_track(double q) {
        PuLocal pl = pu_loc();
        const Fused f = model_.fused(tr_, pl.loc, {1.0, 0.0}, false);
        track::KalmanResult k = track::kalman_rts(f.mu, f.R, q, pl.g);
        hold_ends(k.u, k.P, f.R);
        tr_.u_pl = k.u;
        tr_.P_pl = k.P;
        return {std::move(k.u), std::move(k.P), std::move(pl.loc)};
    }

    void fit(double q) {
        const PilotTrack p = pilot_track(q);
        tr_.delta = fit_delta(model_, tr_, p.v, p.Pv, p.loc, tr_.delta);
    }

private:
    LTrackResult& tr_;
    const ChanCapture& chan_;
    const track::NoisePsdFn& noise_psd_;
    const Model& model_;
    const lwave::Keying* keying_;
    double lead_;
    std::vector<double> grid_d_;
    std::optional<std::array<double, 3>> psi_key_;
    std::vector<double> psi_, psi_pl_;
    std::uint64_t zd_gen_ = 0;
    std::optional<std::pair<std::uint64_t, std::array<double, 3>>> pl_key_;
    std::vector<cplx> m_pl_, c_pl_;
    bool have_u_ = false;
};

std::vector<std::uint8_t> keying_bytes(const lwave::Keying& k) {
    std::vector<std::uint8_t> out(k.values.size());
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<std::uint8_t>(k.values[i]);
    return out;
}

}  // namespace

LTrackResult track_l(std::shared_ptr<const ChanCapture> chan, const FrameSpec& spec, const types::Detection& det,
                     const TrackLArgs& args) {
    l_only(spec, "ltrack.track_l");
    if (!chan) throw std::invalid_argument("track_l: no channel");
    LClasses classes = args.classes ? *args.classes : make_classes_l(spec, nullptr, args.prior ? &*args.prior : nullptr);
    const lwave::Keying* keying = args.keying ? &*args.keying : nullptr;
    if (args.ook) throw std::invalid_argument("waveform L keys its callsign by FSK only: cw_known must say ook=False");
    track::PathFn fn = args.freq ? *args.freq : track::freq_path_spline(det.path);
    if (!args.timing && !det.timing) throw std::invalid_argument("track_l needs a timing (fit_timing_l, or gate V's)");
    const Timing tm = args.timing ? *args.timing : *det.timing;
    double lead = args.lead_in_s ? *args.lead_in_s : det.lead_in_s;
    lead = py_min(py_max(lead, 0.0), 10.0);
    std::shared_ptr<const std::vector<cplx>> z_d =
        args.z_d ? args.z_d : std::make_shared<const std::vector<cplx>>(track::derotate(*chan, fn));
    if (lead > 0 && !lead_supported_l(*chan, *z_d, tm, spec, lead, args.noise_psd)) lead = 0.0;
    const std::int64_t n_lead = py_floor(py_min(lead, rt::LEAD_KEEP_S) / L_T_SYM);
    const std::int64_t p_lo = -SPAN - n_lead;
    const std::int64_t n_pos = spec.n_pos();
    const auto qg = track::q_grid(L_T_SYM);
    const std::span<const double> qgs(qg);

    LTrackResult tr(spec, chan, tm, fn);
    for (std::int64_t p = p_lo; p < n_pos; ++p) tr.pos.push_back(p);
    tr.z_d = z_d;
    tr.classes = classes;
    if (keying) tr.keying = keying_bytes(*keying);
    tr.ook = false;
    tr.lead_in_s = lead;
    const Model model(tr, spec, classes, args.withhold_known, keying, lead, n_lead);
    tr.c = args.withhold_known ? model.c : model.c_all;

    LRun run(tr, *chan, args.noise_psd, model, keying, lead);
    run.measure();
    double q = run.run(qgs);
    if (!args.withhold_known) {
        run.fit(q);
        const std::vector<double> nq = q_near(q, qgs);
        q = run.run(nq);
    }
    if (args.refine && !args.withhold_known) {
        std::vector<std::uint8_t> unknown(model.is_known.size());
        for (std::size_t i = 0; i < unknown.size(); ++i) unknown[i] = model.is_known[i] ? 0 : 1;
        for (int it = 0; it < 3; ++it) {
            const PuLocal pl = run.pu_loc();
            const Fused fw = model.fused(tr, pl.loc, tr.delta, true, true, &unknown);
            const double qw = track::select_q(fw.mu, fw.R, pl.g, qgs).value().first;
            const LRun::PilotTrack pt = run.pilot_track(qw);
            const std::optional<Timing> new_tm = refine_timing_l(tr, 3, &pt.v, &pt.Pv);
            if (!new_tm) break;
            const double moved = std::fabs(new_tm->tau0 - tr.timing.tau0) +
                                 std::fabs((new_tm->ppm - tr.timing.ppm) * 1e-6 * static_cast<double>(n_pos) *
                                           (L_T_SYM * config::CH_FS));
            tr.timing = *new_tm;
            run.measure();
            run.fit(qw);
            if (moved < 0.02) break;
        }
    }
    q = run.run(qgs);
    double df_max = 0.0;
    for (int it = 0; it < args.outer + 1; ++it) {
        if (it == args.outer && df_max <= rt::RECENTER_MIN_HZ) break;
        track::Recenter rc = track::recenter(tr);
        df_max = rc.df_max;
        if (df_max < 0.002) break;
        if (rc.fn) tr.freq = std::move(*rc.fn);
        tr.z_d = std::make_shared<const std::vector<cplx>>(track::derotate(*chan, tr.freq));
        run.zd_changed();
        run.measure();
        const std::vector<double> nq = q_near(q, qgs);
        q = run.run(nq);
    }
    tr.df_max = df_max;
    if (!args.withhold_known) run.measure(true);
    return tr;
}

// --- refine_timing_l ----------------------------------------------------------------------

std::optional<Timing> refine_timing_l(const LTrackResult& tr, int iters, const std::vector<cplx>* u_ref,
                                      const std::vector<double>* P_ref) {
    const FrameSpec& spec = tr.spec;
    l_only(spec, "ltrack.refine_timing_l");
    const std::int64_t n_pos = spec.n_pos();
    const double sps = L_T_SYM * config::CH_FS;
    const auto lay = frame::layout(spec);
    const LClasses* lc = tr.l_classes();
    std::vector<std::uint8_t> known;
    if (lc) {
        known = lc->known;
    } else if (const auto* cc = tr.ce_classes()) {
        known = cc->known;
    } else {
        known.assign(static_cast<std::size_t>(spec.n_sym()), 0);
        for (const std::int64_t i : lay->known_idx) known[static_cast<std::size_t>(i)] = 1;
    }
    const std::vector<std::int64_t> kpos = masked_pos(*lay, known);
    const std::int64_t p_lo = tr.p_lo();
    const auto npos = static_cast<std::int64_t>(tr.pos.size());
    // sel, then ok, as index lists into the known symbols
    std::vector<std::size_t> ks;        // index into the known list
    std::vector<std::size_t> gk;        // grid index
    for (std::size_t j = 0; j < kpos.size(); ++j) {
        const std::int64_t g = kpos[j] - p_lo;
        if (g < 0 || g >= npos) continue;
        const auto gu = static_cast<std::size_t>(g);
        if (gu >= tr.psi.size()) index_error(g, tr.psi.size());
        if (gu >= tr.c.size()) index_error(g, tr.c.size());
        if (tr.psi[gu] >= rt::PSI_MIN && np::abs(tr.c[gu]) > 0) {
            ks.push_back(j);
            gk.push_back(gu);
        }
    }
    if (gk.size() < 8) return std::nullopt;
    std::vector<double> gain_known;
    if (lc) {
        if (lc->gain.size() != known.size())
            throw std::out_of_range("boolean index did not match indexed array along axis 0; size of axis is " +
                                    std::to_string(lc->gain.size()) + " but size of corresponding boolean axis is " +
                                    std::to_string(known.size()));
        for (std::size_t s = 0; s < known.size(); ++s)
            if (known[s]) gain_known.push_back(lc->gain[s]);
    } else if (tr.ce_classes()) {
        throw std::invalid_argument("'Classes' object has no attribute 'gain'");
    } else {
        gain_known.assign(kpos.size(), rd::G_REF_L);
    }
    std::vector<double> dist_known;
    {
        const std::vector<double> d = distortion(spec);
        for (std::size_t s = 0; s < known.size(); ++s)
            if (known[s]) dist_known.push_back(d[s]);
    }
    const bool have_pl = tr.u_pl.has_value();
    const std::vector<cplx>& U = u_ref ? *u_ref : (have_pl ? *tr.u_pl : tr.u);
    // PP follows tr.u_pl, not tr.P_pl: u_pl with P_pl None is None
    const bool pp_none = !P_ref && have_pl && !tr.P_pl;
    const std::vector<double>& PP = P_ref ? *P_ref : (have_pl ? (tr.P_pl ? *tr.P_pl : tr.P) : tr.P);
    // U[gk], then tr.N[gk], tr.pu[gk], PP[gk]: a field the tracker has not
    // filled (None, empty here) is a TypeError there, in that order
    const auto not_subscriptable = [] { throw math::TypeError("'NoneType' object is not subscriptable"); };
    if (!u_ref && !have_pl && tr.u.empty()) not_subscriptable();
    if (tr.N.empty() || tr.pu.empty() || pp_none || (!P_ref && !have_pl && tr.P.empty())) not_subscriptable();
    const std::size_t k = gk.size();
    std::vector<double> pk(k), var(k), Nk(k), a2(k);
    std::vector<cplx> tgt(k);
    for (std::size_t j = 0; j < k; ++j) {
        const std::size_t g = gk[j];
        pk[j] = static_cast<double>(tr.pos[g]);
        const cplx cg = tr.c[g];
        tgt[j] = np::mul(np::mul(U.at(g), cg), cplx(tr.psi[g], 0.0));
        const double gn = gain_known[ks[j]];
        const double ac = np::abs(cg);
        var[j] = tr.N.at(g) + gn * gn * dist_known[ks[j]] * tr.pu.at(g) + ac * ac * PP.at(g);
        Nk[j] = tr.N[g];
        a2[j] = abs2(tgt[j]);
    }
    double A = 0.0;
    {
        std::vector<double> r(k);
        for (std::size_t j = 0; j < k; ++j) r[j] = a2[j] / var[j];
        A = np_sum(r);
    }
    if (!(A > 0)) return std::nullopt;
    double r_abs = 1.0;
    if (u_ref || have_pl) {
        const double ad = py_cabs(tr.delta);
        if (ad == 0.0) throw ZeroDivision("float division by zero");
        r_abs = 1.0 / ad;
    }
    const double r_abs2 = py_sq(r_abs);
    const double prior = 1.0 / py_sq(rl::DELTA_PRIOR_SD);
    if (!tr.z_d) throw std::invalid_argument("refine_timing_l: the track has no z_d");
    if (!tr.chan) throw std::invalid_argument("refine_timing_l: the track has no channel");
    const std::vector<cplx>& z_d = *tr.z_d;
    const double t0_index = tr.chan->t0_index;
    const double slip_scale = static_cast<double>(n_pos) * sps;

    auto timing_of = [&](const std::array<double, 3>& th) {
        Timing t;
        t.tau0 = th[0];
        t.ppm = th[1] / slip_scale * 1e6;
        t.gamma = th[2];
        t.z = 0.0;
        t.cov = math::Matrix(3, 3);
        return t;
    };
    auto corr = [&](const std::array<double, 3>& th) {
        const std::vector<cplx> m = data_mf(z_d, t0_index, timing_of(th), pk, n_pos);
        std::vector<cplx> t(k);
        for (std::size_t j = 0; j < k; ++j) t[j] = np::div(np::mul(std::conj(tgt[j]), m[j]), cplx(var[j], 0.0));
        return np_sum(t);
    };
    auto ell = [&](const std::array<double, 3>& th) {
        const cplx B = corr(th);
        return r_abs2 * A - 2 * r_abs * py_cabs(py_add(B, {prior, 0.0}));
    };
    auto zk = [&](const std::array<double, 3>& th) {
        const cplx B = corr(th);
        std::vector<double> d(k);
        for (std::size_t j = 0; j < k; ++j) d[j] = a2[j] * Nk[j] / (var[j] * var[j]);
        const double den = std::sqrt(py_max(np_sum(d), 1e-300));
        return py_cabs(B) / den;
    };
    struct Newton {
        std::array<double, 3> th;
        std::optional<math::Matrix> H;
    };
    auto newton = [&](std::array<double, 3> th, const std::vector<std::size_t>& dims) {
        const double h = 0.3;
        const double h2 = py_sq(h);
        std::optional<math::Matrix> H;
        for (int it = 0; it < iters; ++it) {
            const double f0 = ell(th);
            const std::size_t kd = dims.size();
            std::vector<double> g(kd, 0.0), fp(kd, 0.0), fm(kd, 0.0);
            math::Matrix Hm(kd, kd);
            for (std::size_t a = 0; a < kd; ++a) {
                std::array<double, 3> tp{}, tm{};
                for (std::size_t q = 0; q < 3; ++q) {
                    const double e = q == dims[a] ? h : 0.0;
                    tp[q] = th[q] + e;
                    tm[q] = th[q] - e;
                }
                fp[a] = ell(tp);
                fm[a] = ell(tm);
                g[a] = (fp[a] - fm[a]) / (2 * h);
                Hm(a, a) = (fp[a] - 2 * f0 + fm[a]) / h2;
            }
            for (std::size_t a = 0; a < kd; ++a)
                for (std::size_t b = a + 1; b < kd; ++b) {
                    std::array<double, 3> tp{};
                    for (std::size_t q = 0; q < 3; ++q) {
                        const double e = (q == dims[a] || q == dims[b]) ? h : 0.0;
                        tp[q] = th[q] + e;
                    }
                    const double fpp = ell(tp);
                    const double v = (fpp - fp[a] - fp[b] + f0) / h2;
                    Hm(a, b) = v;
                    Hm(b, a) = v;
                }
            H = Hm;
            std::vector<double> w;
            try {
                w = math::eigvalsh(Hm);
            } catch (const math::LinAlgError&) {
                H.reset();
                break;
            }
            bool nonpos = false;
            for (const double x : w) nonpos = nonpos || x <= 0;
            if (nonpos) {
                // outside the peak's concave core: a gradient step of up to 1.5
                // samples instead, kept only if it gains; no covariance
                H.reset();
                double gn = std::fabs(g[0]);
                for (const double x : g) {
                    const double ax = std::fabs(x);
                    if (std::isnan(ax)) {
                        gn = ax;
                        break;
                    }
                    if (ax > gn) gn = ax;
                }
                if (!(gn > 0)) break;
                std::vector<double> step(kd);
                for (std::size_t a = 0; a < kd; ++a) step[a] = -g[a] * (1.5 / gn);
                bool gained = false;
                std::array<double, 3> trial{};
                for (int s = 0; s < 4; ++s) {
                    trial = th;
                    for (std::size_t a = 0; a < kd; ++a) trial[dims[a]] = th[dims[a]] + step[a];
                    if (ell(trial) < f0) {
                        gained = true;
                        break;
                    }
                    for (double& x : step) x = x / 2;
                }
                if (!gained) break;
                th = trial;
                continue;
            }
            std::vector<double> step = math::solve(Hm, g);
            double smax = 0.0;
            bool snan = false;
            for (double& s : step) {
                s = np_clip(-s, -1.5, 1.5);
                if (std::isnan(s)) snan = true;
                smax = std::max(smax, std::fabs(s));
            }
            for (std::size_t a = 0; a < kd; ++a) th[dims[a]] += step[a];
            if (!snan && smax < 0.01) break;
        }
        return Newton{th, H};
    };
    auto finite = [](const std::array<double, 3>& th) {
        return std::isfinite(th[0]) && std::isfinite(th[1]) && std::isfinite(th[2]);
    };

    const Timing& tm = tr.timing;
    const std::array<double, 3> th0{tm.tau0, tm.ppm * 1e-6 * static_cast<double>(n_pos) * sps, tm.gamma};
    Newton r = newton(th0, {0, 1});
    if (!finite(r.th)) return std::nullopt;
    std::vector<std::size_t> dims{0, 1};
    if (r.H && static_cast<double>(n_pos) * L_T_SYM > 10000 * config::T_SYM) {
        Newton r3 = newton(r.th, {0, 1, 2});
        if (r3.H && finite(r3.th) && zk(r3.th) > zk(r.th) + 1.0) {
            r = r3;
            dims = {0, 1, 2};
        }
    }
    std::optional<math::Matrix> cov;
    if (r.H) {
        std::optional<math::Matrix> cov_s;
        try {
            cov_s = math::inv(*r.H);
        } catch (const math::LinAlgError&) {
            cov_s.reset();
        }
        bool pos = cov_s.has_value();
        if (cov_s)
            for (std::size_t a = 0; a < dims.size(); ++a) pos = pos && (*cov_s)(a, a) > 0;
        if (pos) {
            const double jdiag[3] = {1.0, 1e6 / (static_cast<double>(n_pos) * sps), 1.0};
            const std::size_t kd = dims.size();
            math::Matrix J(kd, kd);
            for (std::size_t a = 0; a < kd; ++a)
                for (std::size_t b = 0; b < kd; ++b) J(a, b) = dims[a] == dims[b] ? jdiag[dims[a]] : 0.0;
            // J @ cov_s @ J.T
            math::Matrix A1(kd, kd), C(kd, kd);
            for (std::size_t i = 0; i < kd; ++i)
                for (std::size_t j = 0; j < kd; ++j) {
                    double s = 0.0;
                    for (std::size_t q = 0; q < kd; ++q) s += J(i, q) * (*cov_s)(q, j);
                    A1(i, j) = s;
                }
            for (std::size_t i = 0; i < kd; ++i)
                for (std::size_t j = 0; j < kd; ++j) {
                    double s = 0.0;
                    for (std::size_t q = 0; q < kd; ++q) s += A1(i, q) * J(j, q);
                    C(i, j) = s;
                }
            cov.emplace(3, 3);
            for (std::size_t a = 0; a < kd; ++a)
                for (std::size_t b = 0; b < kd; ++b) (*cov)(dims[a], dims[b]) = C(a, b);
        }
    }
    if (!cov) {
        // np.allclose(th, th0): |th - th0| <= 1e-8 + 1e-5 |th0| (th0 finite), or equal
        bool close = true;
        for (std::size_t i = 0; i < 3; ++i) {
            const double x = r.th[i], y = th0[i];
            const bool c = (std::fabs(x - y) <= 1e-8 + 1e-5 * std::fabs(y) && std::isfinite(y)) || x == y;
            close = close && c;
        }
        if (close) return std::nullopt;
        cov = tm.cov;   // moved by gradient steps only
    }
    Timing out = timing_of(r.th);
    out.z = zk(r.th);
    out.cov = *cov;
    return out;
}

// --- the report and the known symbols -----------------------------------------------------

double keyed_power_db(const FrameSpec& spec) {
    double p = 1.0;
    for (const auto& nv : rl::L_KEYED_POWER)
        if (nv.name == spec.name()) p = nv.value;
    return 10 * std::log10(p / py_sq(rd::G_REF_ABS));
}

types::TrackReport report_l(const LTrackResult& tr, double kappa, bool suspect, double noise_floor) {
    types::TrackReport rep = track::report(tr, kappa, suspect, noise_floor);
    rep.snr2500_db = rep.snr2500_db + keyed_power_db(tr.spec);
    return rep;
}

types::TrackReport report_l(const LTrackResult& tr, double kappa, bool suspect, const track::NoisePsdFn& noise_psd) {
    if (!tr.chan) throw std::invalid_argument("report_l: the track has no channel");
    return report_l(tr, kappa, suspect, track::noise_floor(*tr.chan, noise_psd));
}

track::Known known_symbols_l(const FrameSpec& spec, const std::vector<std::uint8_t>* hdr_bits) {
    l_only(spec, "ltrack.known_symbols_l");
    const auto lay = frame::layout(spec);
    std::vector<std::int64_t> idx = lay->known_idx;
    std::vector<cplx> val = lay->known_val_l;
    if (hdr_bits && spec.n_hdr()) {
        idx.insert(idx.end(), lay->hdr_sym.begin(), lay->hdr_sym.end());
        const std::vector<cplx> q = sequences::qpsk(*hdr_bits);
        val.insert(val.end(), q.begin(), q.end());
    }
    track::Known k;
    k.pos.reserve(idx.size());
    for (const std::int64_t i : idx) k.pos.push_back(lay->pos[static_cast<std::size_t>(i)]);
    k.val = std::move(val);
    return k;
}

}  // namespace sstvae::qrss::ltrack
