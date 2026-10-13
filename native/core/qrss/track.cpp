// Port of sstvae/qrss/track.py's tracker core; see track.hpp.

#include "qrss/track.hpp"

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
#include "qrss/ce_rx.hpp"
#include "qrss/lwave.hpp"
#include "qrss/lwave_rx.hpp"
#include "qrss/npsum.hpp"
#include "qrss/pytext.hpp"
#include "qrss/rational.hpp"

namespace sstvae::qrss::track {

namespace {

namespace np = math::np;
namespace rt = rxconfig::track;
using frame::FrameSpec;
using types::Timing;

constexpr double PI = 3.141592653589793;   // np.pi
constexpr double TWO_PI = 2 * PI;          // 2 * np.pi, exact
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr std::int64_t MF_RES = rt::MF_RES;
constexpr std::int64_t SPAN = config::SPAN;
constexpr std::int64_t I64_MIN = std::numeric_limits<std::int64_t>::min();

// _ce_only: CE's model on an L frame gives plausible, wrong numbers.
void ce_only(const FrameSpec& spec, std::string_view what, std::string_view instead) {
    if (spec.waveform() == frame::Waveform::CE) return;
    throw frame::WrongWaveform(std::string(what) + " is CE's; " + pytext::repr(spec.name()) + " is waveform " +
                               std::to_string(static_cast<int>(spec.waveform())) + ": use " + std::string(instead));
}

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

// Python's floor division of ints.
inline std::int64_t floordiv(std::int64_t a, std::int64_t b) {
    std::int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

// k * z for a float k and a complex z: CPython 3.13 makes k complex(k, 0)
// and multiplies unfused.
inline cplx py_rmul(double k, cplx z) {
    return {k * z.real() - 0.0 * z.imag(), k * z.imag() + 0.0 * z.real()};
}

// --- numpy's elementwise semantics ---------------------------------------------

// float64 -> int64 as numpy casts on x86-64 (cvttsd2si: the "integer
// indefinite" INT64_MIN for NaN and anything out of range).
inline std::int64_t np_i64(double x) {
    if (!(x >= -9223372036854775808.0 && x < 9223372036854775808.0)) return I64_MIN;
    return static_cast<std::int64_t>(x);
}
// int64 addition wrapping as numpy's does.
inline std::int64_t add_wrap(std::int64_t a, std::int64_t b) {
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(a) + static_cast<std::uint64_t>(b));
}
// np.clip(x, lo, hi) on float64 (a NaN stays NaN, -0.0 stays -0.0 at lo 0).
inline double np_clip(double x, double lo, double hi) {
    if (std::isnan(x)) return x;
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
// np.maximum(x, b) with a scalar b (NaN in x kept; equal values give b).
inline double np_maximum(double x, double b) { return (std::isnan(x) || x > b) ? x : b; }
// np.max of a float64 array (NaN propagates); the array is not empty.
double np_max(std::span<const double> x) {
    double m = x[0];
    for (const double v : x) {
        if (std::isnan(v)) return v;
        if (v > m) m = v;
    }
    return m;
}
// |z|^2 as np.abs(z) ** 2.
inline double abs2(cplx z) {
    const double a = np::abs(z);
    return a * a;
}
// exp of (+0, x) as numpy's complex exp computes it: (cos x, sin x).
inline cplx exp_i(double x) { return {std::cos(x), std::sin(x)}; }

// np.sum (pairwise) of float64 / complex128 values.
inline double np_sum(const std::vector<double>& v) { return npsum::sum(std::span<const double>(v)); }
inline cplx np_sum(const std::vector<cplx>& v) { return npsum::sum(std::span<const cplx>(v)); }
inline double np_mean(const std::vector<double>& v) { return math::mean(std::span<const double>(v)); }

[[noreturn]] void broadcast_error(std::size_t a, std::size_t b) {
    throw std::invalid_argument("operands could not be broadcast together with shapes (" + std::to_string(a) +
                                ",) (" + std::to_string(b) + ",) ");
}
void same_length(std::size_t a, std::size_t b) {
    if (a != b) broadcast_error(a, b);
}

[[noreturn]] void index_error(std::int64_t i, std::size_t size) {
    throw std::out_of_range("index " + std::to_string(i) + " is out of bounds for axis 0 with size " +
                            std::to_string(size));
}

// a[idx] = v for a fancy index, v of one element broadcast.
void assign_at(std::vector<double>& a, std::span<const std::int64_t> idx, std::span<const double> v) {
    if (v.size() != idx.size() && v.size() != 1)
        throw std::invalid_argument("shape mismatch: value array of shape (" + std::to_string(v.size()) +
                                    ",) could not be broadcast to indexing result of shape (" +
                                    std::to_string(idx.size()) + ",)");
    for (std::size_t k = 0; k < idx.size(); ++k) a[static_cast<std::size_t>(idx[k])] = v[v.size() == 1 ? 0 : k];
}

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

const std::vector<std::uint8_t>* known_of(const TrackResult& tr) {
    if (const auto* c = tr.ce_classes()) return &c->known;
    if (const auto* l = tr.l_classes()) return &l->known;
    return nullptr;
}

std::vector<double> as_doubles(std::span<const std::int64_t> v) {
    std::vector<double> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = static_cast<double>(v[i]);
    return out;
}

// --- symbol_mf -----------------------------------------------------------------

// einsum("ij,ij->i") of two float64 rows: numpy's SSE2 sum of products,
// two lanes over blocks of eight (each lane folding its four products
// from the last), the pairs left over, then the lanes added (onto 0.0).
double einsum_real(const double* a, const double* b, std::size_t n) {
    double acc0 = 0.0, acc1 = 0.0;
    std::size_t i = 0;
    for (; n - i >= 8; i += 8) {
        double x0 = acc0, x1 = acc1;
        for (int k = 3; k >= 0; --k) {
            x0 = a[i + 2 * k] * b[i + 2 * k] + x0;
            x1 = a[i + 2 * k + 1] * b[i + 2 * k + 1] + x1;
        }
        acc0 = x0;
        acc1 = x1;
    }
    for (; i < n; i += 2) {
        acc0 = a[i] * b[i] + acc0;
        if (i + 1 < n) acc1 = a[i + 1] * b[i + 1] + acc1;
    }
    return 0.0 + (acc0 + acc1);
}

// einsum("ij,ij->i") with a complex operand: the products summed in order,
// (vr tr - vi ti, vr ti + vi tr), onto 0.
cplx einsum_complex(const cplx* a, const cplx* b, std::size_t n) {
    double re = 0.0, im = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double vr = a[i].real(), vi = a[i].imag(), tr = b[i].real(), ti = b[i].imag();
        re += vr * tr - vi * ti;
        im += vr * ti + vi * tr;
    }
    return {0.0 + re, 0.0 + im};
}

template <typename Z>
std::vector<cplx> symbol_mf_impl(std::span<const Z> z, const Timing& timing, std::span<const double> pos,
                                 double t0_index, std::int64_t n_pos, const MfTable* table) {
    const MfTable& tb = table ? *table : mf_table();
    const std::int64_t half = tb.half();
    const auto len = static_cast<std::int64_t>(tb.size());
    const auto nz = static_cast<std::int64_t>(z.size());
    const auto width = static_cast<std::size_t>(2 * half + 1);
    std::vector<cplx> out(pos.size());
    if (pos.empty()) return out;
    if (nz == 0) index_error(-1, 0);   // z[np.clip(nn, 0, -1)]
    const std::vector<double> rel = rx_index(timing, pos, n_pos, config::CH_FS, tb.t_sym());
    const bool real_tab = !tb.is_complex();
    std::span<const double> rtab;
    std::span<const cplx> ctab;
    if (real_tab)
        rtab = tb.tab();
    else
        ctab = tb.ctab();
    constexpr bool real_z = std::is_same_v<Z, double>;
    std::vector<double> va(width), ta(width);
    std::vector<cplx> vc(width), tc(width);
    const double hoff = static_cast<double>(half * MF_RES);
    const cplx sps(tb.sps(), 0.0);
    for (std::size_t p = 0; p < pos.size(); ++p) {
        const double t = t0_index + rel[p];
        const std::int64_t base = np_i64(std::floor(t));
        for (std::size_t j = 0; j < width; ++j) {
            const std::int64_t nn = add_wrap(base, static_cast<std::int64_t>(j) - half);
            const double d = (static_cast<double>(nn) - t) * static_cast<double>(MF_RES) + hoff;
            const std::int64_t i = np_i64(std::floor(d));
            const double fr = d - static_cast<double>(i);
            const bool ok = i >= 0 && i < len - 1 && nn >= 0 && nn < nz;
            const std::int64_t ic = std::min(std::max(i, std::int64_t(0)), len - 2);
            const double okd = ok ? 1.0 : 0.0;
            const std::int64_t zi = std::min(std::max(nn, std::int64_t(0)), nz - 1);
            const auto ics = static_cast<std::size_t>(ic);
            if (real_tab) {
                const double tap = (rtab[ics] * (1.0 - fr) + rtab[ics + 1] * fr) * okd;
                if constexpr (real_z) {
                    va[j] = z[static_cast<std::size_t>(zi)];
                    ta[j] = tap;
                } else {
                    vc[j] = z[static_cast<std::size_t>(zi)];
                    tc[j] = cplx(tap, 0.0);
                }
            } else {
                const cplx s = np::mul(ctab[ics], cplx(1.0 - fr, 0.0));
                const cplx e = np::mul(ctab[ics + 1], cplx(fr, 0.0));
                tc[j] = np::mul(cplx(s.real() + e.real(), s.imag() + e.imag()), cplx(okd, 0.0));
                if constexpr (real_z)
                    vc[j] = cplx(z[static_cast<std::size_t>(zi)], 0.0);
                else
                    vc[j] = z[static_cast<std::size_t>(zi)];
            }
        }
        cplx acc;
        if (real_tab && real_z)
            acc = cplx(einsum_real(va.data(), ta.data(), width), 0.0);
        else
            acc = einsum_complex(vc.data(), tc.data(), width);
        out[p] = np::div(acc, sps);
    }
    return out;
}

// --- Kalman --------------------------------------------------------------------------

template <bool Store>
double kf_impl(std::span<const cplx> mu, std::span<const double> R, double q, double Pu, KfPass* out) {
    const std::size_t n = mu.size();
    const double qa = q * Pu / 3.0, qb = q * Pu / 2.0, qc = q * Pu;
    // meas = np.isfinite(R); first = argmax(meas)
    bool any = false;
    std::size_t first = 0;
    for (std::size_t i = 0; i < R.size(); ++i)
        if (std::isfinite(R[i])) {
            any = true;
            first = i;
            break;
        }
    if (R.size() < n) throw std::out_of_range("list index out of range");
    if (any && first >= n) index_error(static_cast<std::int64_t>(first), n);
    cplx x1 = any ? mu[first] : cplx(0.0, 0.0);
    cplx x2(0.0, 0.0);
    const double big = 10.0 * Pu + 1e-30;
    double a = big, b = 0.0, c = big * 1e-2;
    double ll = 0.0;
    if constexpr (Store) {
        out->xa.assign(n, cplx());
        out->xb.assign(n, cplx());
        out->pa.assign(n, 0.0);
        out->pb.assign(n, 0.0);
        out->pc.assign(n, 0.0);
        out->fa.assign(n, 0.0);
        out->fb.assign(n, 0.0);
        out->fc.assign(n, 0.0);
        out->fx1.assign(n, cplx());
        out->fx2.assign(n, cplx());
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (i) {
            a = a + 2 * b + c + qa;
            b = b + c + qb;
            c = c + qc;
            x1 = cplx(x1.real() + x2.real(), x1.imag() + x2.imag());
        }
        if constexpr (Store) {
            out->fa[i] = a;
            out->fb[i] = b;
            out->fc[i] = c;
            out->fx1[i] = x1;
            out->fx2[i] = x2;
        }
        if (std::isfinite(R[i])) {
            const double r = R[i];
            const double S = a + r;
            if (S == 0.0) throw ZeroDivision("float division by zero");
            const double k1 = a / S;
            const double k2 = b / S;
            const cplx inn(mu[i].real() - x1.real(), mu[i].imag() - x1.imag());
            if (S < 0.0) throw std::invalid_argument("math domain error");
            ll -= std::log(S) + (inn.real() * inn.real() + inn.imag() * inn.imag()) / S;
            const cplx d1 = py_rmul(k1, inn), d2 = py_rmul(k2, inn);
            x1 = cplx(x1.real() + d1.real(), x1.imag() + d1.imag());
            x2 = cplx(x2.real() + d2.real(), x2.imag() + d2.imag());
            c = c - k2 * b;
            b = b * (1 - k1);
            a = a * (1 - k1);
        }
        if constexpr (Store) {
            out->xa[i] = x1;
            out->xb[i] = x2;
            out->pa[i] = a;
            out->pb[i] = b;
            out->pc[i] = c;
        }
    }
    if constexpr (Store) out->loglik = ll;
    return ll;
}

inline cplx cadd(cplx a, cplx b) { return {a.real() + b.real(), a.imag() + b.imag()}; }
inline cplx csub(cplx a, cplx b) { return {a.real() - b.real(), a.imag() - b.imag()}; }

// --- window_unknown -------------------------------------------------------------------

template <typename P>
std::vector<std::uint8_t> window_unknown_impl(const FrameSpec& spec, std::span<const P> grid_pos, bool ook_possible,
                                              P reach) {
    std::vector<std::uint8_t> out(grid_pos.size(), 0);
    const std::int64_t unit = floordiv(spec.cw_window(), config::CW_UNITS);
    const std::int64_t lo_u = (ook_possible && spec.waveform() == frame::Waveform::CE)
                                  ? static_cast<std::int64_t>(ce::OOK_FIRST_KEYED_UNIT)
                                  : static_cast<std::int64_t>(config::CW_LEAD_UNITS);
    const std::int64_t hi_u = config::CW_LEAD_UNITS + config::CW_ROOM_UNITS;
    for (const std::int64_t pw : spec.win_start_pos()) {
        const P lo = static_cast<P>(lo_u * unit) - reach;
        const P hi = static_cast<P>(hi_u * unit - 1) + reach;
        for (std::size_t i = 0; i < grid_pos.size(); ++i) {
            const P d = grid_pos[i] - static_cast<P>(pw);
            if (d >= lo && d <= hi) out[i] = 1;
        }
    }
    return out;
}

// --- arange of floats -------------------------------------------------------------------

// np.arange(start, stop, step) for float arguments.
std::vector<double> arange_f(double start, double stop, double step) {
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

// np.argmin with numpy's NaN rule (the first NaN wins).
std::size_t np_argmin(std::span<const double> x) {
    std::size_t best = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (std::isnan(x[i])) return i;
        if (x[i] < x[best]) best = i;
    }
    return best;
}

}  // namespace

// --- derotation ---------------------------------------------------------------

namespace {

std::vector<cplx> derotate_with(const ChanCapture& chan, const std::vector<double>& f) {
    const std::size_t n = chan.ch.size();
    if (f.size() != n) broadcast_error(f.size(), n);
    const double mix = chan.mix_audio_hz();
    const auto fs = static_cast<double>(chan.fs);
    std::vector<double> cum(n);
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double g = (f[i] - mix) / fs;
        acc = i ? acc + g : g;
        cum[i] = acc;
    }
    double c0 = 0.0;
    if (n) {
        std::vector<double> idx(n);
        for (std::size_t i = 0; i < n; ++i) idx[i] = static_cast<double>(i);
        c0 = math::interp(chan.t0_index, idx, cum);
    }
    std::vector<cplx> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = cum[i] - c0;
        const double turns = x - std::floor(x);
        const cplx e = exp_i(-TWO_PI * turns);
        const cplx z(static_cast<double>(chan.ch[i].real()), static_cast<double>(chan.ch[i].imag()));
        out[i] = np::mul(z, e);
    }
    return out;
}

std::vector<double> chan_times(const ChanCapture& chan) {
    std::vector<double> t(chan.ch.size());
    for (std::size_t i = 0; i < t.size(); ++i) t[i] = chan.t_s(static_cast<double>(i));
    return t;
}

}  // namespace

std::vector<cplx> derotate(const ChanCapture& chan, const PathFn& fn) {
    return derotate_with(chan, fn(chan_times(chan)));
}

std::vector<cplx> derotate(const ChanCapture& chan, const PathEval& fn) {
    return derotate_with(chan, fn(chan_times(chan)));
}

// --- the symbol matched filter --------------------------------------------------

double pulse_at(double x, double t_sym) {
    if (t_sym == config::T_SYM) return ce::pulse(x);
    if (t_sym == config::L_T_SYM)
        return std::fabs(x) <= static_cast<double>(SPAN) ? ce::rrc(x) / lwave_rx::PULSE_NORM : 0.0;
    throw std::invalid_argument("no symbol pulse with period " + frame::py_repr(t_sym) + " s");
}

std::vector<double> pulse_at(std::span<const double> x, double t_sym) {
    if (t_sym != config::T_SYM && t_sym != config::L_T_SYM) pulse_at(0.0, t_sym);   // the refusal
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = pulse_at(x[i], t_sym);
    return out;
}

double pulse_g0(double t_sym) { return t_sym == config::T_SYM ? ce::g0() : lwave::g0(); }

namespace {

MfTable build_table(double t_sym) {
    const double sps = t_sym * config::CH_FS;
    const std::int64_t half = py_ceil(static_cast<double>(SPAN) * sps) + 1;
    const std::int64_t lo = -half * MF_RES, hi = half * MF_RES + 1;
    std::vector<double> tab(static_cast<std::size_t>(hi - lo));
    for (std::int64_t k = lo; k < hi; ++k) {
        const double d = static_cast<double>(k) / static_cast<double>(MF_RES);
        tab[static_cast<std::size_t>(k - lo)] = pulse_at(d / sps, t_sym);
    }
    return MfTable(t_sym, sps, static_cast<int>(half), std::move(tab), pulse_g0(t_sym));
}

}  // namespace

const MfTable& mf_table(double t_sym) {
    if (t_sym == config::T_SYM) {
        static const MfTable ce_table = build_table(config::T_SYM);
        return ce_table;
    }
    if (t_sym == config::L_T_SYM) {
        static const MfTable l_table = build_table(config::L_T_SYM);
        return l_table;
    }
    // the reference's failure: int(math.ceil(SPAN * sps)) first, then the pulse
    py_ceil(static_cast<double>(SPAN) * (t_sym * config::CH_FS));
    throw std::invalid_argument("no symbol pulse with period " + frame::py_repr(t_sym) + " s");
}

std::vector<cplx> symbol_mf(std::span<const cplx> z, const Timing& timing, std::span<const double> pos,
                            double t0_index, std::int64_t n_pos, const MfTable* table) {
    return symbol_mf_impl<cplx>(z, timing, pos, t0_index, n_pos, table);
}

std::vector<cplx> symbol_mf(std::span<const double> z, const Timing& timing, std::span<const double> pos,
                            double t0_index, std::int64_t n_pos, const MfTable* table) {
    return symbol_mf_impl<double>(z, timing, pos, t0_index, n_pos, table);
}

std::vector<double> psi_at(std::optional<std::span<const double>> keep, const Timing& timing,
                           std::span<const double> pos, double t0_index, std::int64_t n_pos, const MfTable* table) {
    if (!keep) return std::vector<double>(pos.size(), 1.0);
    const MfTable& tb = table ? *table : mf_table();
    const std::vector<cplx> m = symbol_mf(*keep, timing, pos, t0_index, n_pos, &tb);
    std::vector<double> out(m.size());
    for (std::size_t i = 0; i < m.size(); ++i) out[i] = np_clip(m[i].real() / tb.g0(), 0.0, 1.0);
    return out;
}

std::vector<double> psi_at(const ChanCapture& chan, const Timing& timing, std::span<const double> pos,
                           std::int64_t n_pos, const MfTable* table) {
    if (!chan.keep) return psi_at(std::nullopt, timing, pos, chan.t0_index, n_pos, table);
    const std::vector<double> k(chan.keep->begin(), chan.keep->end());
    return psi_at(std::span<const double>(k), timing, pos, chan.t0_index, n_pos, table);
}

std::vector<double> lowpass_taps(double hz, double fs) {
    if (hz == 0.0) throw ZeroDivision("float division by zero");
    const std::int64_t n = py_round(4.0 / hz * fs) | 1;
    if (n <= 0) return {};
    const std::int64_t mid = floordiv(n, 2);
    const double scale = 2 * hz / fs;
    // np.hanning(n + 2)[1:-1]
    const auto M = static_cast<double>(n + 2);
    std::vector<double> h(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        const double x = PI * (scale * static_cast<double>(i - mid));
        const double y = x != 0.0 ? x : std::numeric_limits<double>::epsilon();
        const double sinc = std::sin(y) / y;
        const double m = (1.0 - M) + static_cast<double>(i + 1) * 2.0;
        const double hann = 0.5 + 0.5 * std::cos(PI * m / (M - 1));
        h[static_cast<std::size_t>(i)] = sinc * hann;
    }
    const double s = np_sum(h);
    for (double& v : h) v /= s;
    return h;
}

std::vector<cplx> lowpass(std::span<const cplx> x, double hz, double fs) {
    const std::vector<double> h = lowpass_taps(hz, fs);
    return math::oaconvolve(x, std::span<const double>(h), math::ConvMode::Same);
}

std::vector<double> lowpass(std::span<const double> x, double hz, double fs) {
    const std::vector<double> h = lowpass_taps(hz, fs);
    return math::oaconvolve(x, std::span<const double>(h), math::ConvMode::Same);
}

std::vector<double> mf_taps(double fs, double t_sym) {
    const std::int64_t half = mf_table(t_sym).half();
    const double w = t_sym * fs;
    std::vector<double> out;
    for (std::int64_t k = -half + 1; k < half; ++k) out.push_back(pulse_at(static_cast<double>(k) / w, t_sym) / w);
    return out;
}

// --- known symbols and classes ---------------------------------------------------

Known known_symbols(const FrameSpec& spec, const frame::HdrBits* hdr_bits) {
    ce_only(spec, "track.known_symbols", "lframe.layout's known_idx/known_val");
    const auto lay = frame::layout(spec);
    std::vector<std::int64_t> idx = lay->known_idx;
    std::vector<double> val(lay->known_val_ce.begin(), lay->known_val_ce.end());
    if (hdr_bits && spec.n_hdr()) {
        if (!hdr_bits->one_d)
            throw std::invalid_argument("all the input arrays must have same number of dimensions");
        idx.insert(idx.end(), lay->hdr_bits.begin(), lay->hdr_bits.end());
        for (const double b : hdr_bits->values) val.push_back(1.0 - 2.0 * b);
    }
    Known k;
    k.pos.reserve(idx.size());
    for (const std::int64_t i : idx) k.pos.push_back(lay->pos[static_cast<std::size_t>(i)]);
    k.val = std::move(val);
    return k;
}

Classes make_classes(const FrameSpec& spec, const frame::HdrBits* hdr_bits, const types::EmPrior* prior) {
    ce_only(spec, "track.make_classes", "ltrack.make_classes_l");
    const auto lay = frame::layout(spec);
    const auto n = static_cast<std::size_t>(spec.n_sym());
    Classes c;
    c.mu.assign(n, 0.0);
    c.nu.assign(n, 1.0);
    c.known.assign(n, 0);
    for (std::size_t k = 0; k < lay->known_idx.size(); ++k) {
        const auto i = static_cast<std::size_t>(lay->known_idx[k]);
        c.mu[i] = static_cast<double>(lay->known_val_ce[k]);
        c.nu[i] = 0.0;
        c.known[i] = 1;
    }
    if (hdr_bits && spec.n_hdr()) {
        std::vector<double> v;
        if (hdr_bits->one_d)
            for (const double b : hdr_bits->values) v.push_back(1.0 - 2.0 * b);
        else
            throw std::invalid_argument("shape mismatch: value array could not be broadcast to indexing result of shape (" +
                                        std::to_string(lay->hdr_bits.size()) + ",)");
        assign_at(c.mu, lay->hdr_bits, v);
        for (const std::int64_t i : lay->hdr_bits) {
            c.nu[static_cast<std::size_t>(i)] = 0.0;
            c.known[static_cast<std::size_t>(i)] = 1;
        }
    }
    if (prior) {
        assign_at(c.mu, lay->data, prior->x_hat);
        std::vector<double> v(prior->v.size());
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = np_clip(prior->v[i], 0.0, 1.0);
        assign_at(c.nu, lay->data, v);
    }
    if (spec.hdr_rho() > 0) {
        if (hdr_bits) {
            c.spread = frame::spread_symbols(spec, hdr_bits);
        } else {
            c.spread_rho = spec.hdr_rho();
            for (const std::int64_t i : lay->data) c.nu[static_cast<std::size_t>(i)] += c.spread_rho;
        }
    }
    c.hdr_known = hdr_bits != nullptr;
    return c;
}

std::vector<cplx> remove_spread(std::span<const cplx> z, double t0_index, const FrameSpec& spec, const Timing& timing,
                                const std::vector<double>* spread) {
    ce_only(spec, "track.remove_spread", "nothing: L's spread copy is linear, in ltrack.make_classes_l's mu");
    if (!spread) return std::vector<cplx>(z.begin(), z.end());
    const std::size_t n = z.size();
    std::vector<double> rel(n);
    for (std::size_t i = 0; i < n; ++i) rel[i] = static_cast<double>(i) - t0_index;
    const std::vector<double> tau = tau_of_rx(timing, rel, spec.n_pos());
    const auto lo = static_cast<double>(-SPAN - 2), hi = static_cast<double>(spec.n_pos() + SPAN + 2);
    std::vector<double> tn;
    std::vector<std::size_t> at;
    for (std::size_t i = 0; i < n; ++i)
        if (tau[i] > lo && tau[i] < hi) {
            tn.push_back(tau[i]);
            at.push_back(i);
        }
    std::vector<double> phi(n, 0.0);
    if (!at.empty()) {
        const std::vector<double> ph = ce::phase_at(tn, *spread, spec);
        for (std::size_t k = 0; k < at.size(); ++k) phi[at[k]] = ph[k];
    }
    std::vector<cplx> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        // np.exp(-1j * phi): (-0.0, -1.0) times (phi, 0), numpy's fused multiply
        const cplx arg = np::mul(cplx(-0.0, -1.0), cplx(phi[i], 0.0));
        const cplx e = arg.real() == 0.0 ? exp_i(arg.imag()) : std::exp(arg);
        out[i] = np::mul(z[i], e);
    }
    return out;
}

Templates templates(const FrameSpec& spec, std::span<const std::int64_t> grid_pos, std::span<const double> mu,
                    std::span<const double> nu, const ce::Keying* keying, bool ook,
                    std::optional<double> keyed_from_pos) {
    ce_only(spec, "track.templates", "ltrack's L templates");
    constexpr std::int64_t fs = config::CH_FS;
    const std::int64_t L = ce::GRID_FS / fs;   // 64
    if (grid_pos.empty()) index_error(0, 0);
    const std::int64_t n0 =
        py_floor(static_cast<double>((grid_pos.front() - SPAN - 2) * 485) / static_cast<double>(L));
    const std::int64_t n1 =
        py_ceil(static_cast<double>((grid_pos.back() + SPAN + 2) * 485) / static_cast<double>(L));
    const std::int64_t n = n1 - n0;
    const std::size_t count = n > 0 ? static_cast<std::size_t>(n) : 0;
    std::vector<double> tau(count);
    for (std::size_t i = 0; i < count; ++i)
        tau[i] = static_cast<double>((n0 + static_cast<std::int64_t>(i)) * L) / 485.0;
    std::vector<double> phi = ce::phase_grid(fs, n0, n, mu, spec);
    std::vector<double> v = ce::var_grid(fs, n0, n, nu, spec);
    phi.resize(count, 0.0);
    v.resize(count, 0.0);
    const double lo = keyed_from_pos ? *keyed_from_pos : spec.keyed_start_pos();
    const double hi = spec.keyed_end_pos();
    const ce::CwPhase cw = ce::cw_phase_turns(tau, spec, keying, ook);
    std::vector<cplx> wave(count);
    std::vector<double> selfw(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double keyed = (tau[i] >= lo && tau[i] < hi) ? 1.0 : 0.0;
        const double x = phi[i] / TWO_PI + cw.theta[i];
        const double turns = x - std::floor(x);
        const double amp = keyed * cw.a[i];
        const cplx e = exp_i(TWO_PI * turns);
        wave[i] = np::mul(np::mul(cplx(amp, 0.0), e), cplx(std::exp(-v[i] / 2), 0.0));
        selfw[i] = amp * amp * (1.0 - std::exp(-v[i]));
    }
    Templates out;
    out.c = ce_rx::matched_filter(std::span<const cplx>(wave), fs, n0, grid_pos);
    std::vector<double> sf = ce_rx::matched_filter(std::span<const double>(selfw), fs, n0, grid_pos);
    const double g0 = ce::g0();
    for (double& s : sf) s = np_maximum(s / g0, 0.0);
    out.sf = std::move(sf);
    return out;
}

// --- Kalman filter and RTS smoother ------------------------------------------------

double kf_forward(std::span<const cplx> mu, std::span<const double> R, double q, double Pu) {
    return kf_impl<false>(mu, R, q, Pu, nullptr);
}

KfPass kf_forward_pass(std::span<const cplx> mu, std::span<const double> R, double q, double Pu) {
    KfPass p;
    kf_impl<true>(mu, R, q, Pu, &p);
    return p;
}

KalmanResult kalman_rts(std::span<const cplx> mu, std::span<const double> R, double q, double Pu, bool causal) {
    KfPass f = kf_forward_pass(mu, R, q, Pu);
    const std::size_t n = mu.size();
    KalmanResult out;
    out.loglik = f.loglik;
    if (causal || n == 0) {
        out.u = std::move(f.xa);
        out.P = std::move(f.pa);
        return out;
    }
    std::vector<cplx> us(n);
    std::vector<double> Ps(n);
    cplx s1 = f.xa[n - 1], s2 = f.xb[n - 1];
    double sa = f.pa[n - 1], sb = f.pb[n - 1], sc = f.pc[n - 1];
    us[n - 1] = s1;
    Ps[n - 1] = sa;
    for (std::size_t ii = n - 1; ii-- > 0;) {
        const std::size_t i = ii;
        const double pa = f.fa[i + 1], pb = f.fb[i + 1], pc = f.fc[i + 1];
        double det = pa * pc - pb * pb;
        if (det <= 0) det = 1e-300;
        const double ia = pc / det, ib = -pb / det, ic = pa / det;
        const double fa = f.pa[i], fb = f.pb[i], fc = f.pc[i];
        const double m11 = fa + fb, m12 = fb, m21 = fb + fc, m22 = fc;
        const double g11 = m11 * ia + m12 * ib;
        const double g12 = m11 * ib + m12 * ic;
        const double g21 = m21 * ia + m22 * ib;
        const double g22 = m21 * ib + m22 * ic;
        const cplx d1 = csub(s1, f.fx1[i + 1]);
        const cplx d2 = csub(s2, f.fx2[i + 1]);
        const cplx n1 = cadd(cadd(f.xa[i], py_rmul(g11, d1)), py_rmul(g12, d2));
        const cplx n2 = cadd(cadd(f.xb[i], py_rmul(g21, d1)), py_rmul(g22, d2));
        const double ea = sa - pa, eb = sb - pb, ec = sc - pc;
        const double t11 = g11 * ea + g12 * eb;
        const double t12 = g11 * eb + g12 * ec;
        const double t21 = g21 * ea + g22 * eb;
        const double t22 = g21 * eb + g22 * ec;
        const double na = fa + t11 * g11 + t12 * g12;
        const double nb = fb + t11 * g21 + t12 * g22;
        const double nc = fc + t21 * g21 + t22 * g22;
        s1 = n1;
        s2 = n2;
        sa = na;
        sb = nb;
        sc = nc;
        us[i] = s1;
        Ps[i] = py_max(sa, 0.0);
    }
    out.u = std::move(us);
    out.P = std::move(Ps);
    return out;
}

std::optional<std::pair<double, double>> select_q(std::span<const cplx> mu, std::span<const double> R, double Pu,
                                                  std::span<const double> grid) {
    std::optional<std::pair<double, double>> best;
    for (const double q : grid) {
        const double ll = kf_forward(mu, R, q, Pu);
        if (!best || ll > best->second) best = std::make_pair(q, ll);
    }
    return best;
}

// --- the tracker's pieces ------------------------------------------------------------

std::vector<std::uint8_t> window_unknown(const FrameSpec& spec, std::span<const std::int64_t> grid_pos,
                                         bool ook_possible, std::int64_t reach) {
    return window_unknown_impl<std::int64_t>(spec, grid_pos, ook_possible, reach);
}

std::vector<std::uint8_t> window_unknown(const FrameSpec& spec, std::span<const double> grid_pos, bool ook_possible,
                                         double reach) {
    return window_unknown_impl<double>(spec, grid_pos, ook_possible, reach);
}

std::vector<double> smooth(std::span<const double> x, std::int64_t n) {
    if (n <= 1) return std::vector<double>(x.begin(), x.end());
    if (x.empty()) throw std::invalid_argument("can't extend empty axis 0 using modes other than 'constant' or 'empty'");
    const double k = 1.0 / static_cast<double>(n);
    const std::int64_t left = floordiv(n, 2);
    const auto nx = static_cast<std::int64_t>(x.size());
    std::vector<double> out(x.size());
    for (std::int64_t i = 0; i < nx; ++i) {
        double s = 0.0;
        for (std::int64_t j = 0; j < n; ++j) {
            const std::int64_t src = std::min(std::max(i + j - left, std::int64_t(0)), nx - 1);
            s += x[static_cast<std::size_t>(src)] * k;
        }
        out[static_cast<std::size_t>(i)] = s;
    }
    return out;
}

Measurements measurements(const TrackResult& tr, std::span<const double> kappa, std::span<const double> sf,
                          std::span<const double> pu_loc, const std::vector<cplx>* c_arg,
                          const std::vector<std::uint8_t>* meas_arg) {
    const std::vector<cplx>& c = c_arg ? *c_arg : tr.c;
    const std::vector<std::uint8_t>& meas = meas_arg ? *meas_arg : tr.meas;
    const std::size_t n = c.size();
    same_length(n, tr.psi.size());
    same_length(meas.size(), n);
    Measurements out;
    out.mu.assign(n, cplx(0.0, 0.0));
    out.R.assign(n, INF);
    for (std::size_t i = 0; i < n; ++i) {
        const cplx cp = np::mul(c[i], cplx(tr.psi[i], 0.0));
        const double acp = np::abs(cp);
        if (!(meas[i] && acp >= rt::C_MIN)) continue;
        if (i >= tr.m.size() || i >= tr.N.size() || i >= kappa.size() || i >= pu_loc.size() || i >= sf.size())
            index_error(static_cast<std::int64_t>(i), std::min({tr.m.size(), tr.N.size(), kappa.size(), pu_loc.size(), sf.size()}));
        out.mu[i] = np::div(tr.m[i], cp);
        out.R[i] = (tr.N[i] + kappa[i] * pu_loc[i] * sf[i]) / (acp * acp) + rt::R_FLOOR * pu_loc[i];
    }
    return out;
}

double pu_initial(const TrackResult& tr) {
    const std::size_t n = tr.c.size();
    same_length(n, tr.psi.size());
    same_length(tr.meas.size(), n);
    std::vector<double> e;
    for (std::size_t i = 0; i < n; ++i) {
        const cplx cp = np::mul(tr.c[i], cplx(tr.psi[i], 0.0));
        const double acp = np::abs(cp);
        if (!(tr.meas[i] && acp >= rt::C_MIN)) continue;
        if (i >= tr.m.size() || i >= tr.N.size()) index_error(static_cast<std::int64_t>(i), std::min(tr.m.size(), tr.N.size()));
        e.push_back((abs2(tr.m[i]) - tr.N[i]) / (acp * acp));
    }
    if (e.empty()) return 1e-30;
    const double mean = np_mean(e);
    const double med = math::median(std::span<const double>(tr.N));
    return py_max(py_max(mean, 1e-3 * med), 1e-30);
}

Recenter recenter(const TrackResult& tr) {
    const std::vector<cplx>& u = tr.u;
    const double t_sym = tr.spec.t_sym();
    const std::vector<double> t = tr.t_pos_s();
    const std::int64_t n_w = std::max<std::int64_t>(1, py_round(rt::RECENTER_WIN_S / t_sym));
    std::vector<std::int64_t> idx;
    for (std::size_t i = 0; i < tr.pos.size(); ++i)
        if (tr.pos[i] >= -SPAN) idx.push_back(static_cast<std::int64_t>(i));
    if (idx.empty()) index_error(0, 0);
    const auto nu = static_cast<std::int64_t>(u.size());
    std::vector<double> tp, df, wt;
    for (std::int64_t a = idx.front(); a < idx.back(); a += n_w) {
        const std::int64_t b = std::min(a + n_w, nu - 1);
        if (b - a < floordiv(n_w, 4)) continue;
        std::vector<cplx> prod;
        std::vector<double> quart;
        for (std::int64_t k = a; k < b; ++k) {
            const cplx u0 = u[static_cast<std::size_t>(k)], u1 = u[static_cast<std::size_t>(k + 1)];
            const double a0 = np::abs(u0);
            prod.push_back(np::mul(np::mul(u1, std::conj(u0)), cplx(a0 * a0, 0.0)));
            quart.push_back(std::pow(a0, 4.0));
        }
        const cplx s = np_sum(prod);
        const double e = np_sum(quart);
        if (e <= 0) continue;
        if (static_cast<std::size_t>(b) >= t.size()) index_error(b, t.size());
        tp.push_back(0.5 * (t[static_cast<std::size_t>(a)] + t[static_cast<std::size_t>(b)]));
        df.push_back(std::atan2(s.imag(), s.real()) / (TWO_PI * t_sym));
        wt.push_back(std::hypot(s.real(), s.imag()));
    }
    Recenter out;
    if (tp.empty()) return out;
    const std::vector<double> td = arange_f(t.front(), t.back() + 1.0, 2.0);
    std::vector<double> fd = tr.freq(td);
    const std::vector<double> add = math::interp(std::span<const double>(td), tp, df);
    for (std::size_t i = 0; i < fd.size(); ++i) fd[i] += add[i];
    const double wmax = py_max(np_max(wt), 1e-300);
    std::vector<double> wn(wt.size());
    for (std::size_t i = 0; i < wt.size(); ++i) wn[i] = wt[i] / wmax;
    std::vector<double> wd = math::interp(std::span<const double>(td), tp, wn);
    for (double& v : wd) v += 1e-3;
    std::vector<double> adf(df.size());
    for (std::size_t i = 0; i < df.size(); ++i) adf[i] = std::fabs(df[i]);
    out.df_max = np_max(adf);
    out.fn.emplace(PathFn(td, fd, std::span<const double>(wd)));
    return out;
}

bool lead_supported(const ChanCapture& chan, std::span<const cplx> z_d, const Timing& tm, const FrameSpec& spec,
                    double lead, const NoisePsdFn& noise_psd) {
    const std::int64_t n = py_floor(py_min(lead, rt::LEAD_KEEP_S) / config::T_SYM);
    if (n < 4) return true;
    std::vector<double> pl, pf;
    for (std::int64_t p = -SPAN - n; p < -SPAN - 1; ++p) pl.push_back(static_cast<double>(p));
    for (std::int64_t p = 0; p < std::min<std::int64_t>(300, spec.n_pos()); ++p) pf.push_back(static_cast<double>(p));
    const std::vector<cplx> ml = symbol_mf(z_d, tm, pl, chan.t0_index, spec.n_pos());
    const std::vector<cplx> mf = symbol_mf(z_d, tm, pf, chan.t0_index, spec.n_pos());
    std::vector<double> t;
    for (const double p : pl) t.push_back(p * config::T_SYM);
    for (const double p : pf) t.push_back(p * config::T_SYM);
    std::vector<double> N = noise_psd_at(chan, t, noise_psd);
    for (double& v : N) v /= config::T_SYM;
    std::vector<double> dl(pl.size()), dfv(pf.size());
    for (std::size_t i = 0; i < pl.size(); ++i) dl[i] = abs2(ml[i]) - N[i];
    for (std::size_t i = 0; i < pf.size(); ++i) dfv[i] = abs2(mf[i]) - N[pl.size() + i];
    const double el = np_mean(dl);
    const double ef = np_mean(dfv) / 0.55;
    return el > ef / 3;
}

// --- the tracker -----------------------------------------------------------------------

namespace {

// track()'s closures over one run: measure, pu_loc, run, near.
class TrackRun {
public:
    TrackRun(TrackResult& tr, std::shared_ptr<const ChanCapture> chan, const NoisePsdFn& noise_psd,
             std::vector<std::uint8_t> meas, std::vector<double> kappa, std::vector<double> sf)
        : tr_(tr), chan_(std::move(chan)), noise_psd_(noise_psd), meas_(std::move(meas)), kappa_(std::move(kappa)),
          sf_(std::move(sf)), grid_d_(as_doubles(tr.pos)) {}

    void measure() {
        const std::int64_t n_pos = tr_.spec.n_pos();
        tr_.m = symbol_mf(*tr_.z_d, tr_.timing, grid_d_, chan_->t0_index, n_pos);
        const std::array<double, 3> key{tr_.timing.tau0, tr_.timing.ppm, tr_.timing.gamma};
        if (!psi_key_ || *psi_key_ != key) {
            psi_ = psi_at(*chan_, tr_.timing, grid_d_, n_pos);
            psi_key_ = key;
        }
        tr_.psi = psi_;
        tr_.N = noise_psd_at(*chan_, tr_.t_pos_s(), noise_psd_);
        for (double& v : tr_.N) v /= config::T_SYM;
        tr_.meas.assign(meas_.size(), 0);
        for (std::size_t i = 0; i < meas_.size(); ++i) tr_.meas[i] = (meas_[i] && tr_.psi[i] >= rt::PSI_MIN) ? 1 : 0;
    }

    std::pair<std::vector<double>, double> pu_loc() const {
        if (!have_u_) {
            const double pu_g = pu_initial(tr_);
            return {std::vector<double>(tr_.pos.size(), pu_g), pu_g};
        }
        std::vector<double> x(tr_.u.size());
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = abs2(tr_.u[i]) + tr_.P[i];
        std::vector<double> loc = smooth(x, py_trunc(rt::PU_SMOOTH_S / config::T_SYM));
        for (double& v : loc) v = np_maximum(v, 1e-30);
        std::vector<double> sel;
        for (std::size_t i = 0; i < tr_.pos.size(); ++i)
            if (tr_.pos[i] >= -SPAN) sel.push_back(loc[i]);
        return {std::move(loc), py_max(np_mean(sel), 1e-30)};
    }

    double run(std::span<const double> q_grid) {
        auto [loc, pu_g] = pu_loc();
        const Measurements ms = measurements(tr_, kappa_, sf_, loc);
        const double q = q_grid.size() == 1 ? q_grid[0] : select_q(ms.mu, ms.R, pu_g, q_grid).value().first;
        KalmanResult k = kalman_rts(ms.mu, ms.R, q, pu_g);
        tr_.u = std::move(k.u);
        tr_.P = std::move(k.P);
        tr_.q = q;
        tr_.loglik = k.loglik;
        tr_.pu = std::move(loc);
        have_u_ = true;
        return q;
    }

private:
    TrackResult& tr_;
    std::shared_ptr<const ChanCapture> chan_;
    const NoisePsdFn& noise_psd_;
    std::vector<std::uint8_t> meas_;
    std::vector<double> kappa_, sf_;
    std::vector<double> grid_d_;
    std::optional<std::array<double, 3>> psi_key_;
    std::vector<double> psi_;
    bool have_u_ = false;
};

// near(q): Q_GRID's points within two of q's (by |log q - log q_i|).
std::vector<double> near_q(double q) {
    if (!(q > 0) && !std::isnan(q)) throw std::invalid_argument("math domain error");
    std::array<double, Q_GRID.size()> d{};
    const double lq = std::log(q);
    for (std::size_t i = 0; i < Q_GRID.size(); ++i) d[i] = std::fabs(std::log(Q_GRID[i]) - lq);
    const auto i = static_cast<std::int64_t>(np_argmin(d));
    const auto n = static_cast<std::int64_t>(Q_GRID.size());
    const std::int64_t lo = std::max<std::int64_t>(0, i - 2), hi = std::min<std::int64_t>(i + 3, n);
    return std::vector<double>(Q_GRID.begin() + lo, Q_GRID.begin() + hi);
}

}  // namespace

TrackResult track(std::shared_ptr<const ChanCapture> chan, const FrameSpec& spec, const types::Detection& det,
                  const TrackArgs& args) {
    ce_only(spec, "track.track", "ltrack.track_l");
    if (!chan) throw std::invalid_argument("track: no channel");
    Classes classes =
        args.classes ? *args.classes : make_classes(spec, nullptr, args.prior ? &*args.prior : nullptr);
    const ce::Keying* keying = args.keying ? &*args.keying : nullptr;
    const bool ook = args.ook;
    PathFn fn = args.freq ? *args.freq : freq_path_spline(det.path);
    if (!args.timing && !det.timing) throw std::invalid_argument("track: no timing (det.timing is None)");
    const Timing tm = args.timing ? *args.timing : *det.timing;
    double lead = args.lead_in_s ? *args.lead_in_s : det.lead_in_s;
    lead = py_min(py_max(lead, 0.0), 10.0);
    std::shared_ptr<const std::vector<cplx>> z_d =
        args.z_d ? args.z_d : std::make_shared<const std::vector<cplx>>(derotate(*chan, fn));
    if (classes.spread)
        z_d = std::make_shared<const std::vector<cplx>>(remove_spread(*z_d, chan->t0_index, spec, tm, &*classes.spread));
    if (lead > 0 && !lead_supported(*chan, *z_d, tm, spec, lead, args.noise_psd)) lead = 0.0;
    const std::int64_t n_lead = py_floor(py_min(lead, rt::LEAD_KEEP_S) / config::T_SYM);
    const std::int64_t p_lo = -SPAN - n_lead;
    const std::int64_t n_pos = spec.n_pos();
    std::vector<std::int64_t> grid;
    for (std::int64_t p = p_lo; p < n_pos; ++p) grid.push_back(p);
    const auto lay = frame::layout(spec);
    const std::vector<std::int64_t> pos_known = masked_pos(*lay, classes.known);
    std::vector<std::uint8_t> is_known(grid.size(), 0);
    for (const std::int64_t p : pos_known) {
        const std::int64_t g = p - p_lo;
        if (g < 0 || g >= static_cast<std::int64_t>(grid.size())) index_error(g, grid.size());
        is_known[static_cast<std::size_t>(g)] = 1;
    }

    const double keyed_from = spec.keyed_start_pos() - lead / config::T_SYM;
    const bool withhold = args.withhold_known;
    std::vector<double> mu_t = classes.mu, nu_t = classes.nu;
    if (withhold) {
        same_length(classes.known.size(), mu_t.size());
        same_length(classes.known.size(), nu_t.size());
        for (std::size_t s = 0; s < classes.known.size(); ++s)
            if (classes.known[s]) {
                mu_t[s] = 0.0;
                nu_t[s] = 1.0;
            }
    }
    Templates tp = templates(spec, grid, mu_t, nu_t, keying, ook, keyed_from);
    std::vector<double> kappa(grid.size());
    for (std::size_t i = 0; i < grid.size(); ++i)
        kappa[i] = (is_known[i] && !withhold) ? rt::KAPPA_SELF_KNOWN : rt::KAPPA_SELF;

    std::vector<std::uint8_t> meas(grid.size(), 1);
    if (!keying && spec.n_win()) {
        const std::vector<std::uint8_t> wu = window_unknown(spec, std::span<const std::int64_t>(grid));
        for (std::size_t i = 0; i < grid.size(); ++i) meas[i] = meas[i] && !wu[i];
    }
    if (withhold)
        for (std::size_t i = 0; i < grid.size(); ++i) meas[i] = meas[i] && !is_known[i];
    if (n_lead == 0)
        for (std::size_t i = 0; i < grid.size(); ++i) meas[i] = meas[i] && grid[i] >= -SPAN;

    TrackResult tr(spec, chan, tm, fn);
    tr.pos = grid;
    tr.z_d = z_d;
    tr.c = tp.c;
    tr.meas = meas;
    tr.classes = classes;
    if (keying) tr.keying = *keying;
    tr.ook = ook;
    tr.lead_in_s = lead;

    TrackRun run(tr, chan, args.noise_psd, meas, kappa, tp.sf);
    run.measure();
    run.run(Q_GRID);
    if (args.refine && !withhold) {
        std::vector<double> mu_w = classes.mu, nu_w = classes.nu;
        same_length(classes.known.size(), mu_w.size());
        same_length(classes.known.size(), nu_w.size());
        for (std::size_t s = 0; s < classes.known.size(); ++s)
            if (classes.known[s]) {
                mu_w[s] = 0.0;
                nu_w[s] = 1.0;
            }
        const Templates tw = templates(spec, grid, mu_w, nu_w, keying, ook, keyed_from);
        const std::vector<double> kap_w(grid.size(), rt::KAPPA_SELF);
        for (int it = 0; it < 3; ++it) {
            auto [loc, pu_g] = run.pu_loc();
            std::vector<std::uint8_t> meas_w(tr.meas.size());
            for (std::size_t i = 0; i < meas_w.size(); ++i) meas_w[i] = tr.meas[i] && !is_known[i];
            const Measurements mw = measurements(tr, kap_w, tw.sf, loc, &tw.c, &meas_w);
            const double qw = select_q(mw.mu, mw.R, pu_g).value().first;
            const double one[1] = {qw};
            run.run(one);
            const std::optional<Timing> new_tm = refine_timing(tr);
            if (!new_tm) break;
            const double moved = std::fabs(new_tm->tau0 - tr.timing.tau0) +
                                 std::fabs((new_tm->ppm - tr.timing.ppm) * 1e-6 * static_cast<double>(n_pos) *
                                           (spec.t_sym() * config::CH_FS));
            tr.timing = *new_tm;
            if (classes.spread)
                tr.z_d = std::make_shared<const std::vector<cplx>>(
                    remove_spread(derotate(*chan, tr.freq), chan->t0_index, spec, tr.timing, &*classes.spread));
            run.measure();
            if (moved < 0.02) break;
        }
    }
    double q = run.run(Q_GRID);
    double df_max = 0.0;
    for (int it = 0; it < args.outer + 1; ++it) {
        if (it == args.outer && df_max <= rt::RECENTER_MIN_HZ) break;
        Recenter rc = recenter(tr);
        df_max = rc.df_max;
        if (df_max < 0.002) break;
        if (rc.fn) tr.freq = std::move(*rc.fn);
        const std::vector<cplx> zr = derotate(*chan, tr.freq);
        tr.z_d = std::make_shared<const std::vector<cplx>>(
            remove_spread(zr, chan->t0_index, spec, tr.timing, classes.spread ? &*classes.spread : nullptr));
        run.measure();
        const std::vector<double> nq = near_q(q);
        q = run.run(nq);
    }
    tr.df_max = df_max;
    return tr;
}

// --- refine_timing ----------------------------------------------------------------------

std::optional<Timing> refine_timing(const TrackResult& tr, int iters, const std::vector<cplx>* u_ref,
                                    const std::vector<double>* P_ref) {
    const FrameSpec& spec = tr.spec;
    ce_only(spec, "track.refine_timing", "ltrack.refine_timing_l");
    const std::int64_t n_pos = spec.n_pos();
    const double sps = spec.t_sym() * config::CH_FS;
    std::vector<std::int64_t> kpos;
    if (const auto* known = known_of(tr))
        kpos = masked_pos(*frame::layout(spec), *known);
    else
        kpos = known_symbols(spec).pos;
    const std::int64_t p_lo = tr.p_lo();
    const auto npos = static_cast<std::int64_t>(tr.pos.size());
    std::vector<std::size_t> gk;
    for (const std::int64_t p : kpos) {
        const std::int64_t g = p - p_lo;
        if (g < 0 || g >= npos) continue;
        if (static_cast<std::size_t>(g) >= tr.meas.size()) index_error(g, tr.meas.size());
        if (tr.meas[static_cast<std::size_t>(g)]) gk.push_back(static_cast<std::size_t>(g));
    }
    if (gk.size() < 8) return std::nullopt;
    const std::vector<cplx>& U = u_ref ? *u_ref : tr.u;
    const std::vector<double>& PP = P_ref ? *P_ref : tr.P;
    const std::size_t k = gk.size();
    std::vector<double> pk(k), var(k), w(k), N(k);
    std::vector<cplx> target(k), Uk(k);
    for (std::size_t j = 0; j < k; ++j) {
        const std::size_t g = gk[j];
        pk[j] = static_cast<double>(tr.pos[g]);
        target[j] = np::mul(np::mul(U.at(g), tr.c.at(g)), cplx(tr.psi.at(g), 0.0));
        var[j] = tr.N.at(g) + rt::KAPPA_SELF_KNOWN * tr.pu.at(g) * 0.1 + abs2(tr.c[g]) * PP.at(g);
        w[j] = tr.c[g].imag();
        Uk[j] = U[g];
        N[j] = tr.N[g];
    }
    if (!tr.z_d) throw std::invalid_argument("refine_timing: the track has no z_d");
    if (!tr.chan) throw std::invalid_argument("refine_timing: the track has no channel");
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
    auto ell = [&](const std::array<double, 3>& th) {
        const std::vector<cplx> m = symbol_mf(z_d, timing_of(th), pk, t0_index, n_pos);
        std::vector<double> r(k);
        for (std::size_t j = 0; j < k; ++j) r[j] = abs2(csub(m[j], target[j])) / var[j];
        return np_sum(r);
    };
    auto zk = [&](const std::array<double, 3>& th) {
        const std::vector<cplx> m = symbol_mf(z_d, timing_of(th), pk, t0_index, n_pos);
        std::vector<double> a(k), b(k);
        for (std::size_t j = 0; j < k; ++j) {
            a[j] = w[j] * np::mul(m[j], std::conj(Uk[j])).imag();
            b[j] = w[j] * w[j] * abs2(Uk[j]) * N[j] / 2;
        }
        const double num = np_sum(a);
        const double den = std::sqrt(py_max(np_sum(b), 1e-300));
        return num / den;
    };
    struct Newton {
        std::array<double, 3> th;
        std::optional<math::Matrix> H;
    };
    auto newton = [&](std::array<double, 3> th, const std::vector<std::size_t>& dims) {
        const double h = 0.3;
        volatile double two = 2.0;
        const double h2 = std::pow(h, static_cast<double>(two));   // Python's h ** 2
        std::optional<math::Matrix> H;
        for (int it = 0; it < iters; ++it) {
            const double f0 = ell(th);
            const std::size_t kd = dims.size();
            std::vector<double> g(kd, 0.0), fp(kd, 0.0), fm(kd, 0.0);
            math::Matrix Hm(kd, kd);
            for (std::size_t a = 0; a < kd; ++a) {
                std::array<double, 3> tp = th, tm = th;
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
                    std::array<double, 3> tp = th;
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
            math::EighResult eg;
            try {
                eg = math::eigh(Hm);
            } catch (const math::LinAlgError&) {
                break;
            }
            bool nonpos = false;
            for (const double x : eg.w) nonpos = nonpos || x <= 0;
            if (nonpos) break;
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
    if (!r.H || !finite(r.th)) return std::nullopt;
    std::vector<std::size_t> dims{0, 1};
    if (static_cast<double>(n_pos) * spec.t_sym() > 10000 * config::T_SYM) {
        Newton r3 = newton(r.th, {0, 1, 2});
        if (r3.H && finite(r3.th) && zk(r3.th) > zk(r.th) + 1.0) {
            r = r3;
            dims = {0, 1, 2};
        }
    }
    math::Matrix cov_s;
    try {
        cov_s = math::inv(*r.H);
    } catch (const math::LinAlgError&) {
        return std::nullopt;
    }
    for (std::size_t a = 0; a < dims.size(); ++a)
        if (cov_s(a, a) <= 0) return std::nullopt;
    const double jdiag[3] = {1.0, 1e6 / (static_cast<double>(n_pos) * sps), 1.0};
    const std::size_t kd = dims.size();
    math::Matrix J(kd, kd);
    for (std::size_t a = 0; a < kd; ++a)
        for (std::size_t b = 0; b < kd; ++b) J(a, b) = dims[a] == dims[b] ? jdiag[dims[a]] : 0.0;
    // J @ cov_s @ J.T
    math::Matrix A(kd, kd), C(kd, kd);
    for (std::size_t i = 0; i < kd; ++i)
        for (std::size_t j = 0; j < kd; ++j) {
            double s = 0.0;
            for (std::size_t q = 0; q < kd; ++q) s += J(i, q) * cov_s(q, j);
            A(i, j) = s;
        }
    for (std::size_t i = 0; i < kd; ++i)
        for (std::size_t j = 0; j < kd; ++j) {
            double s = 0.0;
            for (std::size_t q = 0; q < kd; ++q) s += A(i, q) * J(j, q);
            C(i, j) = s;
        }
    math::Matrix cov(3, 3);
    for (std::size_t a = 0; a < kd; ++a)
        for (std::size_t b = 0; b < kd; ++b) cov(dims[a], dims[b]) = C(a, b);
    Timing out = timing_of(r.th);
    out.z = zk(r.th);
    out.cov = cov;
    return out;
}

// --- the verification gate and the report ---------------------------------------------

std::shared_ptr<const frame::Layout> layout_of(const FrameSpec& spec) { return frame::layout(spec); }

double z_ref_stat(const TrackResult& tr, std::span<const cplx> c_known, std::optional<std::span<const cplx>> d_known) {
    const auto* known = known_of(tr);
    if (!known) throw std::invalid_argument("z_ref_stat: the track has no classes");
    const std::vector<std::int64_t> kp = masked_pos(*layout_of(tr.spec), *known);
    const std::int64_t p_lo = tr.p_lo();
    const auto npos = static_cast<std::int64_t>(tr.pos.size());
    std::vector<std::size_t> gk;
    for (const std::int64_t p : kp) {
        const std::int64_t g = p - p_lo;
        if (g < 0 || g >= npos) continue;
        if (static_cast<std::size_t>(g) >= tr.psi.size()) index_error(g, tr.psi.size());
        if (tr.psi[static_cast<std::size_t>(g)] >= rt::PSI_MIN) gk.push_back(static_cast<std::size_t>(g));
    }
    if (gk.empty()) return 0.0;
    std::vector<double> wv, tv, Vv;
    for (const std::size_t g : gk) {
        const cplx m = tr.m.at(g), u = tr.u.at(g);
        const double N = tr.N.at(g), P = tr.P.at(g);
        if (g >= c_known.size()) index_error(static_cast<std::int64_t>(g), c_known.size());
        const double u2 = abs2(u);
        const double c2 = abs2(c_known[g]);
        double t, V, w;
        if (!d_known) {
            const double ci = c_known[g].imag();
            t = ci * np::mul(m, std::conj(u)).imag();
            V = ci * ci * (N * u2 + c2 * u2 * P + N * P) / 2;
            if (!(V > 0)) continue;
            w = ci * ci * u2 / V;
        } else {
            if (g >= d_known->size()) index_error(static_cast<std::int64_t>(g), d_known->size());
            const cplx d = (*d_known)[g];
            const double d2 = abs2(d);
            t = np::mul(m, std::conj(np::mul(u, d))).real();
            V = d2 * (N * u2 + c2 * u2 * P + N * P) / 2;
            if (!(V > 0)) continue;
            w = d2 * u2 / V;
        }
        wv.push_back(w);
        tv.push_back(t);
        Vv.push_back(V);
    }
    if (wv.empty()) return 0.0;
    std::vector<double> a(wv.size()), b(wv.size());
    for (std::size_t i = 0; i < wv.size(); ++i) {
        a[i] = wv[i] * wv[i] * Vv[i];
        b[i] = wv[i] * tv[i];
    }
    const double den = np_sum(a);
    return np_sum(b) / std::sqrt(py_max(den, 1e-300));
}

void erase(TrackResult& tr, std::span<const std::pair<double, double>> spans) {
    if (spans.empty()) return;
    const double t_sym = tr.spec.t_sym();
    const std::vector<double> t = tr.t_pos_s();
    same_length(tr.psi.size(), t.size());
    for (std::size_t i = 0; i < t.size(); ++i) {
        bool gone = false;
        for (const auto& [a, b] : spans) gone = gone || (t[i] > a - t_sym && t[i] < b + t_sym);
        if (gone) tr.psi[i] = 0.0;
    }
}

types::TrackReport report(const TrackResult& tr, double kappa, bool suspect, double noise_floor) {
    const FrameSpec& spec = tr.spec;
    const std::vector<double> t = tr.t_pos_s();
    std::vector<double> tf;
    std::vector<cplx> u;
    same_length(tr.u.size(), tr.pos.size());
    for (std::size_t i = 0; i < tr.pos.size(); ++i)
        if (tr.pos[i] >= 0) {
            tf.push_back(t[i]);
            u.push_back(tr.u[i]);
        }
    const std::vector<double> fA = tr.freq(tf);
    const std::vector<double> line =
        tf.size() > 2 ? math::polyfit(tf, fA, 1) : std::vector<double>{0.0, tr.freq(0.0)};
    const std::size_t nu = u.size();
    const std::size_t nd = nu ? nu - 1 : 0;
    std::vector<double> w(nd), pr(nd), pi(nd);
    for (std::size_t k = 0; k < nd; ++k) {
        w[k] = np::abs(u[k + 1]) * np::abs(u[k]);
        const cplx p = np::mul(u[k + 1], std::conj(u[k]));
        pr[k] = p.real();
        pi[k] = p.imag();
    }
    const double t_sym = spec.t_sym();
    const std::int64_t n3 = std::max<std::int64_t>(1, py_round(3.0 / t_sym));
    const std::vector<double> sr = smooth(pr, n3), si = smooth(pi, n3);
    std::vector<double> tf1(tf.begin(), tf.begin() + static_cast<std::ptrdiff_t>(std::min(nd, tf.size())));
    const std::vector<double> f1 = tr.freq(tf1);
    const std::vector<double> pl = math::polyval(line, std::span<const double>(tf1));
    std::vector<double> res(nd);
    for (std::size_t k = 0; k < nd; ++k) {
        const cplx j = np::mul(cplx(0.0, 1.0), cplx(si[k], 0.0));   // 1j * smooth(imag)
        const cplx num(sr[k] + j.real(), 0.0 + j.imag());
        const double finst = f1[k] + std::atan2(num.imag(), num.real()) / (TWO_PI * t_sym);
        res[k] = finst - pl[k];
    }
    double wander = 0.0;
    if (nd) {
        const double thr = 0.25 * math::median(std::span<const double>(w));
        std::vector<double> sq;
        for (std::size_t k = 0; k < nd; ++k)
            if (w[k] > thr) sq.push_back(res[k] * res[k]);
        if (!sq.empty()) wander = std::sqrt(np_mean(sq));
    }
    const double kend = spec.keyed_end_pos();
    std::vector<double> pk;
    for (std::size_t i = 0; i < tr.pos.size(); ++i)
        if (tr.pos[i] >= 0 && static_cast<double>(tr.pos[i]) < kend) pk.push_back(abs2(tr.u[i]));
    const double pu = pk.empty() ? 0.0 : np_mean(pk);
    const double ratio = py_max(pu, 1e-30) / py_max(noise_floor * 2500.0, 1e-300);
    if (ratio <= 0) throw std::invalid_argument("math domain error");
    const double snr = 10 * std::log10(ratio);
    same_length(tr.pu.size(), tr.u.size());
    bool slip = false;
    for (std::size_t i = 0; i + 1 < tr.u.size(); ++i) {
        const bool big0 = abs2(tr.u[i]) > tr.pu[i] / 4, big1 = abs2(tr.u[i + 1]) > tr.pu[i + 1] / 4;
        const cplx p = np::mul(tr.u[i + 1], std::conj(tr.u[i]));
        const double d = std::atan2(p.imag(), p.real());
        if (std::fabs(d) > PI / 2 && big1 && big0) slip = true;
    }
    types::TrackReport r;
    r.offset_hz = tr.freq(0.0);
    r.drift_hz_per_min = line[0] * 60.0;
    r.wander_hz_rms = wander;
    r.doppler_hz = tr.doppler_hz();
    r.ppm = tr.timing.ppm;
    r.snr2500_db = snr;
    r.z_ref = tr.z_ref;
    r.kappa = kappa;
    r.slip_free = !slip;
    r.suspect = suspect;
    return r;
}

types::TrackReport report(const TrackResult& tr, double kappa, bool suspect, const NoisePsdFn& noise_psd) {
    if (!tr.chan) throw std::invalid_argument("report: the track has no channel");
    return report(tr, kappa, suspect, noise_floor(*tr.chan, noise_psd));
}

}  // namespace sstvae::qrss::track
