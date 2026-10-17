#include "qrss/lwave.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "dsp/fft.hpp"
#include "qrss/ce.hpp"
#include "qrss/filters.hpp"
#include "qrss/npsum.hpp"
#include "qrss/tables.hpp"

namespace sstvae::qrss::lwave {

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;   // numpy's pi
constexpr std::int64_t L_T_NUM = config::L_T_NUM;
constexpr std::int64_t SPAN = config::SPAN;
constexpr std::int64_t CW = config::L_CW_WINDOW;

// Python's // and % (floor), which C++'s / and % (truncating) are not.
std::int64_t floor_div(std::int64_t a, std::int64_t b) {
    std::int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

std::int64_t floor_mod(std::int64_t a, std::int64_t b) { return a - floor_div(a, b) * b; }

// -(-a // b): Python's ceiling division.
std::int64_t ceil_div(std::int64_t a, std::int64_t b) { return -floor_div(-a, b); }

// (a * b) mod m for 0 <= a, b < m, without a 128-bit type (MSVC has none):
// exact for any m < 2^63 (r + a stays under 2^64).
std::uint64_t mulmod(std::uint64_t a, std::uint64_t b, std::uint64_t m) {
    if (a == 0 || b == 0) return 0;
    if (a <= std::numeric_limits<std::uint64_t>::max() / b) return (a * b) % m;
    std::uint64_t r = 0;
    a %= m;
    while (b) {
        if (b & 1) {
            r += a;
            if (r >= m) r -= m;
        }
        a += a;
        if (a >= m) a -= m;
        b >>= 1;
    }
    return r;
}

void require_l(const frame::FrameSpec& spec, std::string_view what) {
    frame::require_waveform(spec, frame::Waveform::L, what);
}

// ce.py's helpers on L's own symbol, shared with CE so the two cannot drift.
using ce::hann_int;
using ce::ramp_envelope;
using ce::turns_to_phasor;

// numpy's float power with a scalar exponent: its loop's fast paths, then pow.
double np_power(double x, double k) {
    if (k == -1.0) return 1.0 / x;
    if (k == 0.0) return 1.0;
    if (k == 0.5) return std::sqrt(x);
    if (k == 1.0) return x;
    if (k == 2.0) return x * x;
    return std::pow(x, k);
}

std::string str(std::int64_t v) { return std::to_string(v); }

}  // namespace

// --- constants ----------------------------------------------------------------------

double a_pilot() {
    static const double v = [] {
        const double pf = config::L_PILOT_FRAC.value();   // float(Fraction(3, 20)), correctly rounded
        return std::sqrt(pf / (1.0 - pf));
    }();
    return v;
}

double a_clip() {
    static const double v =
        // A_PILOT ** 2 is libm's pow(x, 2) in the reference, which equals
        // x*x for this x (pinned by the ctest); 10 ** 0.0 is exactly 1.
        std::sqrt(1.0 + a_pilot() * a_pilot()) * std::pow(10.0, config::L_CLIP_HEADROOM_DB / 20);
    return v;
}

double a_id() {
    static const double v = std::sqrt(L_AVG_POWER);
    return v;
}

double g0() {
    static const double v = npsum::sum(pulse_table()) / static_cast<double>(L_T_NUM);
    return v;
}

std::span<const double> pulse_table() { return {tables::L_PULSE_TABLE.data(), tables::L_PULSE_TABLE.size()}; }

double np_abs(cplx z) {
    constexpr double inf = std::numeric_limits<double>::infinity();
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    double re = std::fabs(z.real()), im = std::fabs(z.imag());
    const bool re_inf = re == inf, im_inf = im == inf;
    if (re_inf) im = inf;
    if (im_inf) re = inf;
    const bool re_ok = !std::isnan(re), im_ok = !std::isnan(im);
    if (!re_ok) im = nan;
    if (!im_ok) re = nan;
    const double larger = (re > im || std::isnan(re)) ? re : im;
    const double smaller = (im < re || std::isnan(im)) ? im : re;
    const bool div = !(larger == 0.0 || smaller == inf);
    const double ratio = div ? smaller / larger : 0.0;
    return std::sqrt(std::fma(ratio, ratio, 1.0)) * larger;
}

// --- the data on a sample grid ----------------------------------------------------------

std::int64_t grid_step(std::int64_t fs) {
    if (fs <= 0 || GRID_FS % fs)
        throw std::invalid_argument("fs must divide " + str(GRID_FS) + ", not " + str(fs));
    return GRID_FS / fs;
}

namespace {

template <typename T>
std::vector<T> grid_sum_impl(std::int64_t fs, std::int64_t n0, std::int64_t n, std::span<const T> vals,
                             std::span<const double> table) {
    const std::int64_t G = grid_step(fs);
    if (n <= 0) return {};
    const auto n_pos = static_cast<std::int64_t>(vals.size());
    const std::int64_t p_lo = floor_div(G * n0, L_T_NUM) - SPAN - 1;
    const std::int64_t p_hi = ceil_div(G * (n0 + n - 1), L_T_NUM) + SPAN + 2;
    const std::int64_t a = std::max<std::int64_t>(p_lo, 0), b = std::min(p_hi, n_pos);
    if (a >= b) return std::vector<T>(static_cast<std::size_t>(n), T{});
    std::vector<T> x(static_cast<std::size_t>(p_hi - p_lo), T{});
    for (std::int64_t p = a; p < b; ++p) x[static_cast<std::size_t>(p - p_lo)] = vals[static_cast<std::size_t>(p)];
    const std::int64_t z0 = floor_mod(L_T_NUM * p_lo - HALF, G);
    const std::int64_t first = floor_div(G * n0 - L_T_NUM * p_lo + HALF + z0, G);
    std::vector<double> h(static_cast<std::size_t>(z0), 0.0);
    h.insert(h.end(), table.begin(), table.end());
    std::vector<T> y;
    if constexpr (std::is_same_v<T, cplx>) {
        y = filters::upfirdn(std::span<const double>(h), std::span<const cplx>(x), L_T_NUM, G);
    } else {
        y = filters::upfirdn<double>(std::span<const double>(h), std::span<const double>(x), L_T_NUM, G);
    }
    std::vector<T> out(static_cast<std::size_t>(n), T{});
    const auto ny = static_cast<std::int64_t>(y.size());
    for (std::int64_t i = 0; i < n && first + i < ny; ++i)
        if (first + i >= 0) out[static_cast<std::size_t>(i)] = y[static_cast<std::size_t>(first + i)];
    return out;
}

}  // namespace

std::vector<cplx> grid_sum(std::int64_t fs, std::int64_t n0, std::int64_t n, std::span<const cplx> vals,
                           std::span<const double> table) {
    return grid_sum_impl<cplx>(fs, n0, n, vals, table);
}

std::vector<double> grid_sum(std::int64_t fs, std::int64_t n0, std::int64_t n, std::span<const double> vals,
                             std::span<const double> table) {
    return grid_sum_impl<double>(fs, n0, n, vals, table);
}

void check_sym_shape(const frame::FrameSpec& spec, std::span<const std::int64_t> shape) {
    if (shape.size() != 1 || shape[0] != spec.n_sym())
        throw std::invalid_argument("sym has shape " + frame::shape_repr(shape) + ", expected (" +
                                    str(spec.n_sym()) + ",)");
}

std::vector<cplx> at_positions(std::span<const cplx> sym, const frame::FrameSpec& spec) {
    const std::int64_t shape[1] = {static_cast<std::int64_t>(sym.size())};
    check_sym_shape(spec, shape);
    require_l(spec, "lwave.at_positions");
    const auto lay = frame::layout(spec);
    std::vector<cplx> out(static_cast<std::size_t>(spec.n_pos()), cplx(0.0, 0.0));
    for (std::size_t s = 0; s < sym.size(); ++s) out[static_cast<std::size_t>(lay->pos[s])] = sym[s];
    return out;
}

std::vector<cplx> data_grid(std::int64_t fs, std::int64_t n0, std::int64_t n, std::span<const cplx> sym,
                            const frame::FrameSpec& spec) {
    const auto vals = at_positions(sym, spec);
    return grid_sum(fs, n0, n, vals, pulse_table());
}

std::vector<double> mix_turns(std::int64_t fs, std::int64_t n0, std::int64_t n, const Rational& f_hz) {
    // fs <= 0 is refused (the reference divides by a zero or negative period).
    if (fs <= 0) throw std::invalid_argument("mix_turns needs a positive fs, not " + str(fs));
    const std::int64_t den = f_hz.den();
    if (den > std::numeric_limits<std::int64_t>::max() / fs)
        throw std::overflow_error("mix_turns: the period den * fs leaves int64");
    const std::int64_t period = den * fs;
    const auto num_mod = static_cast<std::uint64_t>(floor_mod(f_hz.num(), period));
    const auto P = static_cast<std::uint64_t>(period);
    const double pd = static_cast<double>(period);
    std::vector<double> out(static_cast<std::size_t>(std::max<std::int64_t>(n, 0)));
    for (std::size_t i = 0; i < out.size(); ++i) {
        const auto nn = static_cast<std::uint64_t>(floor_mod(n0 + static_cast<std::int64_t>(i), period));
        out[i] = static_cast<double>(mulmod(nn, num_mod, P)) / pd;
    }
    return out;
}

std::vector<double> tau_grid(std::int64_t fs, std::int64_t n0, std::int64_t n) {
    std::vector<double> out(static_cast<std::size_t>(std::max<std::int64_t>(n, 0)));
    const std::int64_t G = grid_step(fs);
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = static_cast<double>((n0 + static_cast<std::int64_t>(i)) * G) / static_cast<double>(L_T_NUM);
    return out;
}

// --- pilot: keying, callsign windows and lead-in -------------------------------------

std::string lead_in_refusal(std::string_view shown) {
    return "lead-in must be 0 to " + str(config::LEAD_IN_MAX_S) + " s, not " + std::string(shown);
}

LeadInRefusal::LeadInRefusal(double v) : std::invalid_argument(lead_in_refusal(frame::py_repr(v))), lead(v) {}

std::pair<double, double> keyed_span_pos(const frame::FrameSpec& spec, double lead_in_s) {
    const double lead = lead_in_s;
    if (!(0.0 <= lead && lead <= static_cast<double>(config::LEAD_IN_MAX_S))) throw LeadInRefusal(lead_in_s);
    return {spec.keyed_start_pos() - lead / config::L_T_SYM, spec.keyed_end_pos()};
}

Keying Keying::units(std::span<const std::uint8_t> mask) {
    Keying k;
    k.shape = {static_cast<std::int64_t>(mask.size())};
    k.values.assign(mask.begin(), mask.end());
    return k;
}

std::optional<std::vector<std::uint8_t>> window_keying(const Keying* keying, const frame::FrameSpec& spec) {
    if (keying == nullptr) return std::nullopt;
    std::vector<std::int64_t> shape = keying->shape;
    const bool one = shape.size() == 1;
    if (one) shape = {spec.n_win(), keying->shape[0]};
    bool ok = shape.size() == 2 && shape[0] == spec.n_win() && shape[1] == CW;
    // The reference tests the values of the broadcast array: with no windows
    // a (192,) mask broadcasts to (0, 192) and none of its values is looked at.
    if (ok && shape[0] > 0) {
        for (double v : keying->values)
            if (v != 0 && v != 1) {
                ok = false;
                break;
            }
    }
    if (!ok)
        throw std::invalid_argument("keying must be 0/1 of shape (" + str(CW) + ",) or (" + str(spec.n_win()) + ", " +
                                    str(CW) + "), not " + frame::shape_repr(shape));
    const auto rows = static_cast<std::size_t>(spec.n_win());
    std::vector<std::uint8_t> out(rows * static_cast<std::size_t>(CW));
    for (std::size_t w = 0; w < rows; ++w)
        for (std::size_t u = 0; u < static_cast<std::size_t>(CW); ++u)
            out[w * CW + u] = static_cast<std::uint8_t>(keying->values[(one ? 0 : w * CW) + u]);
    return out;
}

double raised(double y) {
    // np.clip(y, 0, 1) (NaN passes through), then 0.5 (1 - cos(pi y)).
    if (y < 0.0) y = 0.0;
    if (y > 1.0) y = 1.0;
    return 0.5 * (1.0 - std::cos(PI * y));
}

std::vector<double> cw_turns(std::span<const double> tau, const frame::FrameSpec& spec, const Keying* keying) {
    std::vector<double> theta(tau.size(), 0.0);
    const auto k_all = window_keying(keying, spec);
    if (!k_all) return theta;
    const auto starts = spec.win_start_pos();
    for (std::int64_t w = 0; w < spec.n_win(); ++w) {
        const std::uint8_t* k = k_all->data() + static_cast<std::size_t>(w * CW);
        std::int64_t total = 0;
        std::vector<std::int64_t> cum(static_cast<std::size_t>(CW) + 1, 0);   // cum[i] = sum k[:i]
        for (std::int64_t u = 0; u < CW; ++u) {
            total += k[u];
            cum[static_cast<std::size_t>(u) + 1] = total;
        }
        if (total == 0) continue;
        const double p0 = static_cast<double>(starts[static_cast<std::size_t>(w)]) - 0.5;
        const double p1 = p0 + static_cast<double>(CW);
        const double tot = static_cast<double>(total);
        for (std::size_t i = 0; i < tau.size(); ++i) {
            const double t = tau[i];
            if (t >= p1) theta[i] -= tot;
            if (!(t >= p0 && t < p1)) continue;
            const double x = t - p0;
            const auto f = static_cast<std::int64_t>(std::floor(x));
            double th = -static_cast<double>(cum[static_cast<std::size_t>(std::clamp<std::int64_t>(f - 1, 0, CW))]);
            for (std::int64_t j = -1; j <= 1; ++j) {
                const std::int64_t u = f + j;
                const bool ok = u >= 0 && u < CW;
                const double ku = ok ? static_cast<double>(k[std::clamp<std::int64_t>(u, 0, CW - 1)]) : 0.0;
                const double y = x - static_cast<double>(u);
                th -= ku * (hann_int(y) - hann_int(y - 1));
            }
            theta[i] += th;
        }
    }
    return theta;
}

std::vector<double> id_weight(std::span<const double> tau, const frame::FrameSpec& spec, double lead_in_s) {
    std::vector<double> W(tau.size(), 0.0);
    const double r = static_cast<double>(ID_RAMP_UNITS);
    if (lead_in_s > 0) {
        const double start = spec.keyed_start_pos();   // the preamble's first pulse
        for (std::size_t i = 0; i < tau.size(); ++i)
            W[i] = tau[i] < start ? 1.0 : 1.0 - raised((tau[i] - start) / -start);
    }
    for (const std::int64_t p : spec.win_start_pos()) {
        const double off = static_cast<double>(p) - 0.5;
        for (std::size_t i = 0; i < tau.size(); ++i) {
            const double x = tau[i] - off;
            if (!(x >= 0 && x < static_cast<double>(CW))) continue;
            const double up = raised(x / r);
            const double down = 1.0 - raised((x - (static_cast<double>(CW) - r)) / r);
            W[i] = std::isnan(up) || up < down ? up : down;   // np.minimum
        }
    }
    return W;
}

std::vector<cplx> pilot(std::span<const double> tau, const frame::FrameSpec& spec, const Keying* keying,
                        double lead_in_s, double ramp_s) {
    const auto [lo, hi] = keyed_span_pos(spec, lead_in_s);
    const double ramp = ramp_s / config::L_T_SYM;
    std::vector<double> env(tau.size());
    for (std::size_t i = 0; i < tau.size(); ++i) env[i] = ramp_envelope(tau[i], lo, hi, ramp);
    const auto W = id_weight(tau, spec, lead_in_s);
    const auto th = cw_turns(tau, spec, keying);
    const double ap = a_pilot(), ai = a_id();
    std::vector<cplx> out(tau.size());
    for (std::size_t i = 0; i < tau.size(); ++i) {
        const double base = ap * (1.0 - W[i]);
        const cplx id = np_mul(ai * W[i], turns_to_phasor(th[i]));
        const cplx s(base + id.real(), 0.0 + id.imag());   // float + complex: (base + 0j) + id
        out[i] = np_mul(env[i], s);
    }
    return out;
}

// --- composite and clipper ------------------------------------------------------------

std::vector<cplx> composite(std::span<const cplx> sym, const frame::FrameSpec& spec, std::int64_t fs, std::int64_t n0,
                            std::int64_t n, const Keying* keying, double lead_in_s, double ramp_s) {
    const auto d = data_grid(fs, n0, n, sym, spec);
    const auto turns = mix_turns(fs, n0, n, Rational(config::L_PILOT_HZ.num, config::L_PILOT_HZ.den));
    const auto p = pilot(tau_grid(fs, n0, n), spec, keying, lead_in_s, ramp_s);
    std::vector<cplx> out(d.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
        const cplx m = np_mul(d[i], turns_to_phasor(-turns[i]));
        out[i] = cplx(m.real() + p[i].real(), m.imag() + p[i].imag());
    }
    return out;
}

std::vector<double> band_mask(std::span<const double> freqs, double guard_hz, double taper_hz) {
    const Rational edge = Rational(config::L_PILOT_HZ.num, config::L_PILOT_HZ.den) +
                          Rational(config::L_EDGE_HZ.num, config::L_EDGE_HZ.den);
    const double lo = -edge.to_double() - guard_hz;
    const double hi = guard_hz;
    const double t = taper_hz;
    std::vector<double> out(freqs.size());
    for (std::size_t i = 0; i < freqs.size(); ++i) {
        const double f = freqs[i];
        const double rise = raised((f - (lo - t / 2)) / t);
        const double fall = 1.0 - raised((f - (hi - t / 2)) / t);
        out[i] = std::isnan(rise) || rise < fall ? rise : fall;   // np.minimum
    }
    return out;
}

std::int64_t next_fast_len(std::int64_t target) {
    // pocketfft / ducc0 good_size_cmplx: the smallest 2^a 3^b 5^c 7^d 11^e >= n.
    if (target <= 12) return target;
    const std::int64_t n = target;
    std::int64_t best = 2 * n;
    for (std::int64_t f11 = 1; f11 < best; f11 *= 11)
        for (std::int64_t f117 = f11; f117 < best; f117 *= 7)
            for (std::int64_t f1175 = f117; f1175 < best; f1175 *= 5) {
                std::int64_t x = f1175;
                while (x < n) x *= 2;
                for (;;) {
                    if (x < n) {
                        x *= 3;
                    } else if (x > n) {
                        if (x < best) best = x;
                        if (x & 1) break;
                        x >>= 1;
                    } else {
                        return n;
                    }
                }
            }
    return best;
}

std::vector<double> fftfreq(std::int64_t n, double d) {
    const double val = 1.0 / (static_cast<double>(n) * d);
    std::vector<double> out(static_cast<std::size_t>(std::max<std::int64_t>(n, 0)));
    const std::int64_t N = floor_div(n - 1, 2) + 1;
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int64_t k = i < N ? i : -floor_div(n, 2) + (i - N);
        out[static_cast<std::size_t>(i)] = static_cast<double>(k) * val;
    }
    return out;
}

std::vector<cplx> clip(std::span<const cplx> zin, std::int64_t fs, double threshold, std::span<const double> overshoot,
                       double guard_hz) {
    std::vector<cplx> z(zin.begin(), zin.end());
    const std::size_t len = z.size();
    const std::int64_t nfft = next_fast_len(static_cast<std::int64_t>(len) + 4 * fs);
    const auto mask = band_mask(fftfreq(nfft, 1.0 / static_cast<double>(fs)), guard_hz);
    std::vector<double> mag(len);
    std::vector<cplx> corr(static_cast<std::size_t>(nfft));
    for (const double k : overshoot) {
        bool any = false;
        for (std::size_t i = 0; i < len; ++i) {
            mag[i] = np_abs(z[i]);
            any = any || mag[i] > threshold;
        }
        if (!any) continue;
        std::fill(corr.begin(), corr.end(), cplx(0.0, 0.0));
        for (std::size_t i = 0; i < len; ++i) {
            const double scale = mag[i] > threshold ? np_power(threshold / mag[i], k) : 1.0;
            corr[i] = np_mul(scale - 1.0, z[i]);
        }
        auto spec = dsp::fft(corr, true);
        for (std::size_t i = 0; i < spec.size(); ++i) spec[i] = np_mul(spec[i], cplx(mask[i], 0.0));
        const auto back = dsp::fft(spec, false);
        for (std::size_t i = 0; i < len; ++i) z[i] = cplx(z[i].real() + back[i].real(), z[i].imag() + back[i].imag());
    }
    return z;
}

Baseband frame_baseband(std::span<const cplx> sym, const frame::FrameSpec& spec, const Keying* keying,
                        double lead_in_s, std::int64_t fs, double ramp_s, double margin_s) {
    const auto [lo, hi] = keyed_span_pos(spec, lead_in_s);
    const double t_lo = lo * config::L_T_SYM - ramp_s, t_hi = hi * config::L_T_SYM + ramp_s;
    const double fsd = static_cast<double>(fs);
    const auto n0 = static_cast<std::int64_t>(std::floor((t_lo - margin_s) * fsd));
    const auto n1 = static_cast<std::int64_t>(std::ceil((t_hi + margin_s) * fsd));
    const auto z0 = composite(sym, spec, fs, n0, n1 - n0, keying, lead_in_s, ramp_s);
    Baseband bb;
    bb.z = clip(z0, fs);
    bb.n0 = n0;
    bb.fs = fs;
    // z /= A_ID: numpy's complex division by (A_ID + 0j), a multiply by 1/A_ID.
    const double rat = 0.0 / a_id();
    const double scl = 1.0 / (a_id() + 0.0 * rat);
    for (std::size_t i = 0; i < bb.z.size(); ++i) {
        const double t = static_cast<double>(n0 + static_cast<std::int64_t>(i)) / fsd;
        cplx v = bb.z[i];
        if (t < t_lo || t >= t_hi) v = cplx(0.0, 0.0);
        bb.z[i] = cplx((v.real() + v.imag() * rat) * scl, (v.imag() - v.real() * rat) * scl);
    }
    return bb;
}

// --- to 8 kHz ---------------------------------------------------------------------

std::vector<double> interp_taps(std::int64_t up) {
    std::vector<double> h(tables::L_INTERP_TAPS.begin(), tables::L_INTERP_TAPS.end());
    const double u = static_cast<double>(up);
    for (double& v : h) v = v * u;
    return h;
}

void resample_into(std::span<const cplx> z, std::int64_t n0, std::int64_t fs, std::int64_t fs_out, std::int64_t i0,
                   std::span<cplx> out) {
    const std::int64_t up = fs > 0 ? floor_div(fs_out, fs) : 0;
    if (fs <= 0 || up * fs != fs_out || up <= 0)
        throw std::invalid_argument("fs_out must be a multiple of " + str(fs));
    const auto len = static_cast<std::int64_t>(z.size());
    const auto m = static_cast<std::int64_t>(out.size());
    if (up == 1) {
        for (std::int64_t i = 0; i < m; ++i) {
            const std::int64_t k = i0 + i - n0;
            out[static_cast<std::size_t>(i)] = (k >= 0 && k < len) ? z[static_cast<std::size_t>(k)] : cplx(0.0, 0.0);
        }
        return;
    }
    // u[k] = sum_n z[n] h[k - up n] (scipy's upfirdn), summed in ascending n:
    // output sample s takes the frame samples n with 0 <= s + D - up n <= 2D.
    // An accumulator that starts at +0 never becomes -0, so the frame's own
    // zeros and the filter's zero padding, which scipy adds, change nothing.
    const auto h = interp_taps(up);
    const std::int64_t D = INTERP_HALF;
    const auto last = static_cast<std::int64_t>(h.size()) - 1;
    // s + D = up q + r (0 <= r < up): the taps are h[r + up j], j = q - n,
    // for 0 <= j <= (last - r) / up. q and r step with s, so the divisions
    // happen once.
    std::vector<std::int64_t> j_max(static_cast<std::size_t>(up));
    for (std::int64_t r = 0; r < up; ++r) j_max[static_cast<std::size_t>(r)] = floor_div(last - r, up);
    const std::int64_t n_last = n0 + len - 1;
    std::int64_t q = floor_div(i0 + D, up);
    std::int64_t r = i0 + D - up * q;
    const double* hp = h.data();
    for (std::int64_t i = 0; i < m; ++i) {
        const std::int64_t lo = std::max(q - j_max[static_cast<std::size_t>(r)], n0);
        const std::int64_t hi = std::min(q, n_last);
        double re = 0.0, im = 0.0;
        for (std::int64_t n = lo; n <= hi; ++n) {
            const double t = hp[r + up * (q - n)];
            const cplx x = z[static_cast<std::size_t>(n - n0)];
            re += x.real() * t;
            im += x.imag() * t;
        }
        out[static_cast<std::size_t>(i)] = cplx(re, im);
        if (++r == up) {
            r = 0;
            ++q;
        }
    }
}

std::vector<cplx> resample(std::span<const cplx> z, std::int64_t n0, std::int64_t fs, std::int64_t fs_out,
                           std::int64_t i0, std::int64_t m) {
    if (m < 0) throw std::invalid_argument("negative dimensions are not allowed");
    std::vector<cplx> out(static_cast<std::size_t>(m));
    resample_into(z, n0, fs, fs_out, i0, out);
    return out;
}

std::vector<cplx> resample(const Baseband& bb, std::int64_t fs_out, std::int64_t i0, std::int64_t m) {
    return resample(bb.z, bb.n0, bb.fs, fs_out, i0, m);
}

std::vector<cplx> to_data_centre(std::span<const cplx> z, std::int64_t n0, std::int64_t fs) {
    const auto turns = mix_turns(fs, n0, static_cast<std::int64_t>(z.size()),
                                 Rational(config::L_PILOT_HZ.num, config::L_PILOT_HZ.den));
    std::vector<cplx> out(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) out[i] = np_mul(z[i], turns_to_phasor(turns[i]));
    return out;
}

}  // namespace sstvae::qrss::lwave
