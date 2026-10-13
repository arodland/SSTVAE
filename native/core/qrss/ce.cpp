#include "qrss/ce.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>

#include "qrss/filters.hpp"
#include "qrss/npsum.hpp"
#include "qrss/tables.hpp"

namespace sstvae::qrss::ce {

namespace {

using std::int64_t;

constexpr double PI = std::numbers::pi;   // np.pi
constexpr double TWO_PI = 2 * PI;          // 2 * np.pi, exact
constexpr int64_t T_NUM = config::T_NUM;
constexpr int64_t SPAN = config::SPAN;
constexpr int64_t CW_UNITS = config::CW_UNITS;
constexpr int64_t TABLE_LEN = 2 * HALF + 1;
static_assert(tables::CE_PULSE_TABLE.size() == static_cast<std::size_t>(TABLE_LEN));

// Python's // and % (floor), for b != 0.
int64_t floor_div(int64_t a, int64_t b) {
    const int64_t q = a / b;
    return ((a % b != 0) && ((a < 0) != (b < 0))) ? q - 1 : q;
}
int64_t floor_mod(int64_t a, int64_t b) { return a - floor_div(a, b) * b; }

// The reference's scalars, each computed as Python computes it.
struct Consts {
    double a = ALPHA_F;
    double one_minus_a = 1 - ALPHA_F;            // 1 - a
    double one_plus_a = 1 + ALPHA_F;             // 1 + a
    double four_a = 4.0 * ALPHA_F;               // 4.0 * a (= 4 * a)
    double beta2 = BETA_F * BETA_F;              // _BETA ** 2 (pow(x, 2) is x*x, correctly rounded)
    double four_pi2 = 4 * (PI * PI);             // 4 * np.pi ** 2
    double rrc_zero = 0;                         // 1 - a + 4 * a / np.pi
    double rrc_sing = 0;                         // the value at 4 alpha |tau| = 1
    double norm = 0;                             // PULSE_NORM
    double pi_norm = 0;                          // np.pi * PULSE_NORM
    double c1 = 0, c2 = 0;                       // np.pi * (1 - a), np.pi * (1 + a)
    std::array<double, N_OFFSETS> cos1{}, sin1{}, cos2{}, sin2{};   // math.cos(c1 * j), ... j = -8..8
    double g0 = 0;
    std::vector<double> table_sq;                // PULSE_TABLE ** 2

    Consts() {
        rrc_zero = one_minus_a + four_a / PI;
        const double r = PI / four_a;            // np.pi / (4 * a)
        rrc_sing = (a / std::sqrt(2.0)) * ((1 + 2 / PI) * std::sin(r) + (1 - 2 / PI) * std::cos(r));
    }
};

double rrc_with(const Consts& k, double tau);

const Consts& consts() {
    static const Consts k = [] {
        Consts c;
        // PULSE_NORM = float(np.sqrt(np.sum(_rrc(_M / T_NUM) ** 2) / T_NUM))
        std::vector<double> sq(static_cast<std::size_t>(TABLE_LEN));
        for (int64_t m = -HALF; m <= HALF; ++m) {
            const double h = rrc_with(c, static_cast<double>(m) / static_cast<double>(T_NUM));
            sq[static_cast<std::size_t>(m + HALF)] = h * h;
        }
        c.norm = std::sqrt(npsum::sum(std::span<const double>(sq)) / static_cast<double>(T_NUM));
        c.pi_norm = PI * c.norm;
        c.c1 = PI * c.one_minus_a;
        c.c2 = PI * c.one_plus_a;
        for (int64_t j = -SPAN; j <= SPAN; ++j) {
            const auto i = static_cast<std::size_t>(j + SPAN);
            const double jd = static_cast<double>(j);
            c.cos1[i] = std::cos(c.c1 * jd);
            c.sin1[i] = std::sin(c.c1 * jd);
            c.cos2[i] = std::cos(c.c2 * jd);
            c.sin2[i] = std::sin(c.c2 * jd);
        }
        const auto& t = tables::CE_PULSE_TABLE;
        c.g0 = npsum::sum(std::span<const double>(t.data(), t.size())) / static_cast<double>(T_NUM);
        c.table_sq.resize(t.size());
        for (std::size_t i = 0; i < t.size(); ++i) c.table_sq[i] = t[i] * t[i];
        return c;
    }();
    return k;
}

double rrc_with(const Consts& k, double tau) {
    // zero = |tau| < 1e-12; sing = |1 - (4 a tau)^2| < 1e-9; else the closed form.
    if (std::fabs(tau) < 1e-12) return k.rrc_zero;
    const double q = k.four_a * tau;
    const double one_m = 1.0 - q * q;
    if (std::fabs(one_m) < 1e-9) return k.rrc_sing;
    const double den = (PI * tau) * one_m;
    const double pt = PI * tau;
    return (std::sin(pt * k.one_minus_a) + (k.four_a * tau) * std::cos(pt * k.one_plus_a)) / den;
}

std::string str(int64_t v) { return std::to_string(v); }

std::string fmt_g(double v) { return frame::py_format(v, "%g"); }

// p(tau)/c over the 17 offsets j of one d: ce._pulse_offsets, element-wise.
void offsets_into(const Consts& k, double d, bool square, double* out) {
    const double s1 = std::sin(k.c1 * d), k1 = std::cos(k.c1 * d);
    const double s2 = std::sin(k.c2 * d), k2 = std::cos(k.c2 * d);
    for (int64_t j = -SPAN; j <= SPAN; ++j) {
        const auto i = static_cast<std::size_t>(j + SPAN);
        const double tau = d - static_cast<double>(j);
        const double sn = s1 * k.cos1[i] - k1 * k.sin1[i];
        const double cs = k2 * k.cos2[i] + s2 * k.sin2[i];
        const double q = k.four_a * tau;
        const double den = (k.pi_norm * tau) * (1.0 - q * q);
        double p;
        if ((j == 0 || j == 2 || j == -2) && std::fabs(den) < 1e-9)
            p = rrc_with(k, tau) / k.norm;
        else
            p = (sn + q * cs) / den;
        if ((j == SPAN || j == -SPAN) && !(std::fabs(tau) <= static_cast<double>(SPAN))) p = 0.0;
        out[i] = square ? p * p : p;
    }
}

// The position values of `sym` (n_sym of them), as at_positions.
std::vector<double> positions_of(std::span<const double> sym, const FrameSpec& spec) {
    if (static_cast<int64_t>(sym.size()) != spec.n_sym()) {
        const std::int64_t shape[1] = {static_cast<int64_t>(sym.size())};
        check_sym(spec, shape);
    }
    std::vector<double> out(static_cast<std::size_t>(spec.n_pos()), 0.0);
    const auto lay = frame::layout(spec);
    for (std::size_t s = 0; s < sym.size(); ++s) out[static_cast<std::size_t>(lay->pos[s])] = sym[s];
    return out;
}

// phi (and v) at each tau, from position values vals (and vvar).
void phase_at_into(std::span<const double> tau, const std::vector<double>& vals, const std::vector<double>* vvar,
                   double* phi, double* v) {
    const Consts& k = consts();
    const auto n_pos = static_cast<int64_t>(vals.size());
    double p[N_OFFSETS];
    for (std::size_t i = 0; i < tau.size(); ++i) {
        const double t = tau[i];
        const double centre = std::floor(t + 0.5);
        // near: -SPAN - 1 <= centre <= n_pos + SPAN (NaN and inf are not near)
        if (!(centre >= static_cast<double>(-SPAN - 1) && centre <= static_cast<double>(n_pos + SPAN))) {
            phi[i] = 0.0;
            if (v) v[i] = 0.0;
            continue;
        }
        const auto ci = static_cast<int64_t>(centre);
        const double d = t - centre;
        offsets_into(k, d, false, p);
        double acc = 0.0;
        for (int64_t j = -SPAN; j <= SPAN; ++j) {
            const int64_t at = ci + j;
            const double x = (at >= 0 && at < n_pos) ? vals[static_cast<std::size_t>(at)] : 0.0;
            acc += x * p[j + SPAN];
        }
        phi[i] = BETA_F * acc;
        if (v) {
            offsets_into(k, d, true, p);
            double acc2 = 0.0;
            for (int64_t j = -SPAN; j <= SPAN; ++j) {
                const int64_t at = ci + j;
                const double x = (at >= 0 && at < n_pos) ? (*vvar)[static_cast<std::size_t>(at)] : 0.0;
                acc2 += x * p[j + SPAN];
            }
            v[i] = k.beta2 * acc2;
        }
    }
}

std::vector<double> scaled(std::vector<double> v, double s) {
    for (double& x : v) x = s * x;
    return v;
}

}  // namespace

// --- pulse -------------------------------------------------------------------------------

double rrc(double tau) { return rrc_with(consts(), tau); }

double pulse_norm() { return consts().norm; }

double pulse(double tau) {
    const Consts& k = consts();
    return std::fabs(tau) <= static_cast<double>(SPAN) ? rrc_with(k, tau) / k.norm : 0.0;
}

std::span<const double> pulse_table() {
    return std::span<const double>(tables::CE_PULSE_TABLE.data(), tables::CE_PULSE_TABLE.size());
}

double g0() { return consts().g0; }

std::int64_t grid_step(std::int64_t fs) {
    if (fs <= 0 || GRID_FS % fs != 0)
        throw std::invalid_argument("fs must divide " + str(GRID_FS) + ", not " + str(fs));
    return GRID_FS / fs;
}

// --- symbols on the position axis -------------------------------------------------------

void check_sym(const FrameSpec& spec, std::span<const std::int64_t> shape, std::string_view name) {
    if (shape.size() == 1 && shape[0] == spec.n_sym()) return;
    const std::int64_t want[1] = {spec.n_sym()};
    throw std::invalid_argument(std::string(name) + " has shape " + frame::shape_repr(shape) + ", expected " +
                                frame::shape_repr(want));
}

std::vector<double> at_positions(std::span<const double> sym, const FrameSpec& spec) {
    return positions_of(sym, spec);
}

// --- phase -------------------------------------------------------------------------------

std::array<double, N_OFFSETS> pulse_offsets(double d, bool square) {
    std::array<double, N_OFFSETS> out{};
    offsets_into(consts(), d, square, out.data());
    return out;
}

std::vector<double> phase_at(std::span<const double> tau, std::span<const double> sym, const FrameSpec& spec) {
    const std::vector<double> vals = positions_of(sym, spec);
    std::vector<double> phi(tau.size());
    phase_at_into(tau, vals, nullptr, phi.data(), nullptr);
    return phi;
}

std::pair<std::vector<double>, std::vector<double>> phase_at(std::span<const double> tau,
                                                             std::span<const double> sym,
                                                             std::span<const double> var,
                                                             const FrameSpec& spec) {
    const std::vector<double> vals = positions_of(sym, spec);
    if (static_cast<int64_t>(var.size()) != spec.n_sym()) {
        const std::int64_t shape[1] = {static_cast<int64_t>(var.size())};
        check_sym(spec, shape, "var");
    }
    const std::vector<double> vvar = positions_of(var, spec);
    std::vector<double> phi(tau.size()), v(tau.size());
    phase_at_into(tau, vals, &vvar, phi.data(), v.data());
    return {std::move(phi), std::move(v)};
}

std::vector<double> grid_sum(std::int64_t fs, std::int64_t n0, std::int64_t n, std::span<const double> vals,
                             std::span<const double> table) {
    const int64_t L = grid_step(fs);
    if (n <= 0) return {};
    // Samples this far from t0 are nowhere near any frame at any rate (a
    // frame spans < 2^40 grid steps), and the reference's Python ints give
    // zeros there; the bound keeps every product below int64's range.
    constexpr int64_t FAR = std::numeric_limits<int64_t>::max() / (4 * GRID_FS);
    if (n0 > FAR || n0 < -FAR || n > FAR) return std::vector<double>(static_cast<std::size_t>(n), 0.0);
    const auto n_pos = static_cast<int64_t>(vals.size());
    // The positions that can reach samples [n0, n0 + n): the reference's
    // local window [p_lo, p_hi), clipped to the frame.
    const int64_t p_lo = floor_div(L * n0, T_NUM) - SPAN - 1;
    const int64_t p_hi = -floor_div(-(L * (n0 + n - 1)), T_NUM) + SPAN + 2;
    const int64_t a = std::max<int64_t>(p_lo, 0), b = std::min(p_hi, n_pos);
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    if (a >= b) return out;
    const auto len_t = static_cast<int64_t>(table.size());
    for (int64_t i = 0; i < n; ++i) {
        const int64_t g = L * (n0 + i) + HALF;   // table index of position 0
        // positions with 0 <= g - 485 p < len_t, inside [a, b)
        const int64_t first = std::max(a, floor_div(g - len_t, T_NUM) + 1);
        const int64_t last = std::min(b - 1, floor_div(g, T_NUM));
        double acc = 0.0;
        for (int64_t p = first; p <= last; ++p)
            acc += vals[static_cast<std::size_t>(p)] * table[static_cast<std::size_t>(g - T_NUM * p)];
        out[static_cast<std::size_t>(i)] = acc;
    }
    return out;
}

std::vector<double> phase_grid(std::int64_t fs, std::int64_t n0, std::int64_t n, std::span<const double> sym,
                               const FrameSpec& spec) {
    const std::vector<double> vals = positions_of(sym, spec);
    return scaled(grid_sum(fs, n0, n, vals, pulse_table()), BETA_F);
}

std::vector<double> var_grid(std::int64_t fs, std::int64_t n0, std::int64_t n, std::span<const double> var,
                             const FrameSpec& spec) {
    if (static_cast<int64_t>(var.size()) != spec.n_sym()) {
        const std::int64_t shape[1] = {static_cast<int64_t>(var.size())};
        check_sym(spec, shape, "var");
    }
    const std::vector<double> vals = positions_of(var, spec);
    const Consts& k = consts();
    return scaled(grid_sum(fs, n0, n, vals, k.table_sq), k.beta2);
}

// --- callsign windows ----------------------------------------------------------------------

double hann_int(double y) {
    if (y < -0.5) return 0.0;
    if (y > 0.5) return y;
    const double s = y + 0.5;
    return (s * s) / 2 - (std::cos(TWO_PI * y) + 1) / consts().four_pi2;
}

std::string keying_refusal(const FrameSpec& spec, std::span<const std::int64_t> shape) {
    std::vector<int64_t> sh(shape.begin(), shape.end());
    if (sh.size() == 1) sh = {spec.n_win(), sh[0]};   // shown as broadcast, as the reference shows it
    const std::int64_t s1[1] = {CW_UNITS};
    const std::int64_t s2[2] = {spec.n_win(), CW_UNITS};
    return "keying must be 0/1 of shape " + frame::shape_repr(s1) + " or " + frame::shape_repr(s2) + ", not " +
           frame::shape_repr(sh);
}

Keying window_keying(std::span<const double> values, std::span<const std::int64_t> shape, const FrameSpec& spec) {
    const int64_t n_win = spec.n_win();
    std::vector<int64_t> sh(shape.begin(), shape.end());
    const bool one = sh.size() == 1;
    if (one) sh = {n_win, sh[0]};   // np.broadcast_to(k, (n_win, len(k)))
    bool ok = sh.size() == 2 && sh[0] == n_win && sh[1] == CW_UNITS &&
              values.size() == static_cast<std::size_t>(one ? CW_UNITS : n_win * CW_UNITS);
    // np.any((k != 0) & (k != 1)) over the (broadcast) array: nothing to
    // look at when there are no windows.
    if (ok && n_win > 0)
        for (const double v : values)
            if (v != 0 && v != 1) {
                ok = false;
                break;
            }
    if (!ok) throw std::invalid_argument(keying_refusal(spec, shape));
    Keying k(static_cast<std::size_t>(n_win * CW_UNITS));
    for (std::size_t i = 0; i < k.size(); ++i)
        k[i] = static_cast<std::uint8_t>(values[one ? i % static_cast<std::size_t>(CW_UNITS) : i]);
    return k;
}

Keying window_keying(std::span<const std::uint8_t> units, const FrameSpec& spec) {
    std::vector<double> v(units.begin(), units.end());
    const std::int64_t shape[1] = {static_cast<int64_t>(units.size())};
    return window_keying(v, shape, spec);
}

double window_units(double tau, const FrameSpec& spec, std::int64_t w) {
    const auto starts = spec.win_start_pos();
    const auto n = static_cast<int64_t>(starts.size());
    if (w < -n || w >= n) throw std::out_of_range("tuple index out of range");
    const int64_t p_w = starts[static_cast<std::size_t>(w < 0 ? w + n : w)];
    return (tau - (static_cast<double>(p_w) - 0.5)) / config::CW_UNIT;
}

CwPhase cw_phase_turns(std::span<const double> tau, const FrameSpec& spec, const Keying* keying, bool ook) {
    const auto starts = spec.win_start_pos();
    const std::size_t n_win = starts.size();
    if (keying && keying->size() != n_win * static_cast<std::size_t>(CW_UNITS))
        throw std::invalid_argument("cw_phase_turns: keying is not (n_win, 192)");
    struct Win {
        double p0, p1;
        const std::uint8_t* k;
        int64_t total;
        std::array<int64_t, CW_UNITS + 1> cum;
    };
    static const std::array<std::uint8_t, CW_UNITS> zeros{};
    std::vector<Win> wins(n_win);
    for (std::size_t w = 0; w < n_win; ++w) {
        Win& v = wins[w];
        v.p0 = static_cast<double>(starts[w]) - 0.5;
        v.p1 = v.p0 + config::CW_WINDOW;
        v.k = keying ? keying->data() + w * static_cast<std::size_t>(CW_UNITS) : zeros.data();
        v.cum[0] = 0;
        for (int64_t u = 0; u < CW_UNITS; ++u) v.cum[static_cast<std::size_t>(u + 1)] = v.cum[static_cast<std::size_t>(u)] + v.k[u];
        v.total = v.cum[static_cast<std::size_t>(CW_UNITS)];
    }
    CwPhase out{std::vector<double>(tau.size(), 0.0), std::vector<double>(tau.size(), 1.0)};
    for (std::size_t i = 0; i < tau.size(); ++i) {
        const double t = tau[i];
        double th = 0.0, a = 1.0;
        for (const Win& v : wins) {
            const bool inside = t >= v.p0 && t < v.p1;
            if (ook) {
                if (inside) {
                    const double x = (t - v.p0) / config::CW_UNIT;
                    const auto u = static_cast<int64_t>(std::floor(x));
                    const bool keyed = u >= OOK_FIRST_KEYED_UNIT && u <= OOK_LAST_KEYED_UNIT;
                    a = keyed ? static_cast<double>(v.k[std::clamp<int64_t>(u, 0, CW_UNITS - 1)]) : 1.0;
                }
                continue;
            }
            if (v.total == 0) continue;
            if (t >= v.p1) {
                th -= static_cast<double>(v.total);
            } else if (inside) {
                const double x = (t - v.p0) / config::CW_UNIT;
                const auto f = static_cast<int64_t>(std::floor(x));
                // Units at or before f - 2 are complete, after f + 1 not begun.
                // -cum[clip(f - 1, 0, 192)].astype(float64): a float negation
                double w_th = -static_cast<double>(v.cum[static_cast<std::size_t>(std::clamp<int64_t>(f - 1, 0, CW_UNITS))]);
                for (int64_t j = -1; j <= 1; ++j) {
                    const int64_t u = f + j;
                    const bool in = u >= 0 && u < CW_UNITS;
                    const double ku = in ? static_cast<double>(v.k[u]) : 0.0;
                    const double y = x - static_cast<double>(u);
                    w_th -= ku * (hann_int(y) - hann_int(y - 1));
                }
                th += w_th;
            }
        }
        out.theta[i] = th;
        out.a[i] = a;
    }
    return out;
}

// --- keyed span -----------------------------------------------------------------------------

std::string lead_in_refusal(std::string_view shown) {
    return "lead-in must be 0 to " + str(config::LEAD_IN_MAX_S) + " s, not " + std::string(shown);
}

std::pair<double, double> keyed_span_pos(const FrameSpec& spec, double lead_in_s) {
    if (!(0.0 <= lead_in_s && lead_in_s <= config::LEAD_IN_MAX_S))
        throw LeadInRefusal(lead_in_refusal(frame::py_repr(lead_in_s)));
    return {spec.keyed_start_pos() - lead_in_s / config::T_SYM, spec.keyed_end_pos()};
}

double ramp_envelope(double tau, double lo, double hi, double ramp_sym) {
    double env = (tau >= lo && tau < hi) ? 1.0 : 0.0;
    if (ramp_sym > 0) {
        const double start = lo - ramp_sym;
        if (tau >= start && tau < lo) env = 0.5 * (1 - std::cos(PI * (tau - start) / ramp_sym));
        if (tau >= hi && tau < hi + ramp_sym) env = 0.5 * (1 + std::cos(PI * (tau - hi) / ramp_sym));
    }
    return env;
}

std::vector<double> ramp_envelope(std::span<const double> tau, double lo, double hi, double ramp_sym) {
    std::vector<double> out(tau.size());
    for (std::size_t i = 0; i < tau.size(); ++i) out[i] = ramp_envelope(tau[i], lo, hi, ramp_sym);
    return out;
}

std::complex<double> turns_to_phasor(double turns) {
    const double ang = TWO_PI * (turns - std::floor(turns));
    return {std::cos(ang), std::sin(ang)};
}

// --- baseband and audio ---------------------------------------------------------------------

BasebandPlan baseband_plan(const FrameSpec& spec, double fs, double start_s, const BasebandOptions& opt) {
    BasebandPlan plan;
    // n0 = round(start_s * fs), as Python's round() of a float.
    const double x = start_s * fs;
    if (std::isnan(x)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(x)) throw std::overflow_error("cannot convert float infinity to integer");
    const double r = std::nearbyint(x);
    // [-2^63, 2^63): what np.arange(n0, ..., dtype=np.int64) takes.
    plan.n0_overflows = !(r >= -0x1p63 && r < 0x1p63);
    plan.n0 = plan.n0_overflows ? 0 : static_cast<int64_t>(r);
    // time_scale == 1.0 and int(fs) == fs and GRID_FS % int(fs) == 0 and |x - n0| < 1e-9
    // (x - n0 is float - int in Python: n0 converts exactly, being r).
    if (opt.time_scale == 1.0 && std::trunc(fs) == fs && std::fabs(fs) < 0x1p62) {
        const auto fs_i = static_cast<int64_t>(fs);
        if (fs_i == 0) throw ZeroDivision("integer modulo by zero");
        plan.exact = floor_mod(GRID_FS, fs_i) == 0 && std::fabs(x - r) < 1e-9;
    }
    std::tie(plan.lo, plan.hi) = keyed_span_pos(spec, opt.lead_in_s);
    return plan;
}

std::vector<std::complex<double>> baseband(std::span<const double> sym, const FrameSpec& spec, double fs,
                                           double start_s, std::int64_t n, const BasebandOptions& opt) {
    if (static_cast<int64_t>(sym.size()) != spec.n_sym()) {
        const std::int64_t shape[1] = {static_cast<int64_t>(sym.size())};
        check_sym(spec, shape);
    }
    const BasebandPlan plan = baseband_plan(spec, fs, start_s, opt);
    const int64_t n0 = plan.n0;
    const bool exact = plan.exact;
    const int64_t fs_i = exact ? static_cast<int64_t>(fs) : 0;
    const double lo = plan.lo, hi = plan.hi;
    const double ramp = opt.ramp_s / config::T_SYM;
    const std::size_t n_win = static_cast<std::size_t>(spec.n_win());
    if (opt.keying && opt.keying->size() != n_win * static_cast<std::size_t>(CW_UNITS))
        throw std::invalid_argument("baseband: keying is not (n_win, 192)");
    if (n < 0) throw std::invalid_argument("negative dimensions are not allowed");
    if (opt.chunk <= 0) throw std::invalid_argument("baseband: chunk must be positive");
    std::vector<std::complex<double>> out(static_cast<std::size_t>(n));
    std::vector<double> tau;
    for (int64_t i0 = 0; i0 < n; i0 += opt.chunk) {
        const int64_t m = std::min(opt.chunk, n - i0);
        tau.resize(static_cast<std::size_t>(m));
        if (exact) {
            // np.arange(n0 + i0, n0 + i0 + m, dtype=np.int64) * _grid_step(fs): the
            // start must be an int64 (numpy's OverflowError otherwise), and
            // numpy's int64 product wraps, as the unsigned one here does.
            if (plan.n0_overflows || n0 > std::numeric_limits<int64_t>::max() - i0)
                throw std::overflow_error("Python int too large to convert to C long");
            const auto L = static_cast<std::uint64_t>(grid_step(fs_i));
            const auto first = static_cast<std::uint64_t>(n0 + i0);
            for (int64_t i = 0; i < m; ++i)
                tau[static_cast<std::size_t>(i)] =
                    static_cast<double>(static_cast<int64_t>((first + static_cast<std::uint64_t>(i)) * L)) /
                    static_cast<double>(T_NUM);
        } else {
            for (int64_t i = 0; i < m; ++i)
                tau[static_cast<std::size_t>(i)] =
                    (start_s + static_cast<double>(i0 + i) / fs) * opt.time_scale / config::T_SYM;
        }
        if (tau.back() < lo - ramp || tau.front() >= hi + ramp) continue;   // unkeyed: stays zero
        const std::vector<double> phi = exact ? phase_grid(fs_i, n0 + i0, m, sym, spec) : phase_at(tau, sym, spec);
        const CwPhase cw = cw_phase_turns(tau, spec, opt.keying, opt.ook);
        for (int64_t i = 0; i < m; ++i) {
            const auto k = static_cast<std::size_t>(i);
            const double amp = ramp_envelope(tau[k], lo, hi, ramp) * cw.a[k];
            const std::complex<double> e = turns_to_phasor(phi[k] / TWO_PI + cw.theta[k]);
            // (amp + 0j) * e, numpy's complex product
            out[static_cast<std::size_t>(i0 + i)] =
                std::complex<double>(amp * e.real() - 0.0 * e.imag(), amp * e.imag() + 0.0 * e.real());
        }
    }
    return out;
}

std::int64_t mulmod(std::int64_t a, std::int64_t b, std::int64_t m) {
    using U = std::uint64_t;
    const U um = static_cast<U>(m);
    U ua = static_cast<U>(a), ub = static_cast<U>(b);
    if (ub == 0 || ua <= std::numeric_limits<U>::max() / ub) return static_cast<std::int64_t>((ua * ub) % um);
    // Double-and-add; every partial stays below m < 2^63, so nothing wraps.
    ua %= um;
    U r = 0;
    while (ub) {
        if (ub & 1U) r = (r >= um - ua) ? r - (um - ua) : r + ua;
        ub >>= 1;
        if (ub) ua = (ua >= um - ua) ? ua - (um - ua) : ua + ua;
    }
    return static_cast<std::int64_t>(r);
}

std::vector<double> to_audio(std::span<const std::complex<double>> z_in, std::int64_t fs, std::int64_t num,
                             std::int64_t den, double amplitude, std::int64_t n0) {
    std::vector<std::complex<double>> resampled;
    std::span<const std::complex<double>> z = z_in;
    if (fs != config::FS) {
        const int64_t g = std::gcd(static_cast<int64_t>(config::FS), fs);
        resampled = filters::resample_poly<std::complex<double>>(z_in, config::FS / g, fs / g);
        z = resampled;
    }
    if (den <= 0) throw std::invalid_argument("carrier_hz_den must be positive");
    // period = den * 8000 is a Python int there, and numpy refuses it past int64.
    if (den > std::numeric_limits<int64_t>::max() / config::FS)
        throw std::overflow_error("Python int too large to convert to C long");
    const int64_t period = den * config::FS;
    const int64_t nm = floor_mod(num, period);
    const double dp = static_cast<double>(period);
    std::vector<double> out(z.size());
    int64_t nn = floor_mod(n0, period);
    for (std::size_t i = 0; i < z.size(); ++i) {
        const double turns = static_cast<double>(mulmod(nn, nm, period)) / dp;
        const double y = TWO_PI * turns;
        const double c = std::cos(y), s = std::sin(y);
        // Re(z * e^{jy}) as numpy's SIMD complex multiply forms it (fused).
        out[i] = amplitude * std::fma(z[i].real(), c, -(z[i].imag() * s));
        if (++nn == period) nn = 0;
    }
    return out;
}

std::string carrier_refusal(std::string_view shown) {
    return "carrier " + std::string(shown) + " Hz is outside the " + fmt_g(config::CARRIER_BAND_HZ[0]) + "-" +
           fmt_g(config::CARRIER_BAND_HZ[1]) + " Hz audio band that receivers search";
}

double check_carrier(double carrier_hz, std::string_view shown) {
    const double lo = config::CARRIER_BAND_HZ[0], hi = config::CARRIER_BAND_HZ[1];
    if (!(std::isfinite(carrier_hz) && lo <= carrier_hz && carrier_hz <= hi))
        throw std::invalid_argument(carrier_refusal(shown));
    return carrier_hz;
}

double check_carrier(double carrier_hz) { return check_carrier(carrier_hz, frame::py_repr(carrier_hz)); }

Rational carrier_fraction(double carrier_hz) { return Rational::from_double_limited(carrier_hz, CARRIER_MAX_DEN); }

Rational carrier_fraction(std::string_view carrier_hz) {
    return Rational::parse(carrier_hz).limit_denominator(CARRIER_MAX_DEN);
}

Rational carrier_fraction(std::int64_t carrier_hz) { return Rational(carrier_hz, 1); }

}  // namespace sstvae::qrss::ce
