#include "qrss/trackbase.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include "math/errors.hpp"
#include "math/npstats.hpp"

namespace sstvae::qrss::track {

namespace {

constexpr double PI = std::numbers::pi;   // math.pi

// (tau / n_pos) ** 2 on a numpy array: numpy's power with exponent 2 is square.
inline double sq(double x) { return x * x; }

// Python's float // float (floatobject.c _float_div_mod), then int().
std::int64_t py_floordiv_int(double vx, double wx) {
    if (wx == 0.0) throw ZeroDivision("float floor division by zero");
    double mod = std::fmod(vx, wx);
    double div = (vx - mod) / wx;
    if (mod != 0.0) {
        if ((wx < 0) != (mod < 0)) {
            mod += wx;
            div -= 1.0;
        }
    }
    double floordiv;
    if (div != 0.0) {
        floordiv = std::floor(div);
        if (div - floordiv > 0.5) floordiv += 1.0;
    } else {
        floordiv = std::copysign(0.0, vx / wx);
    }
    if (std::isnan(floordiv)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(floordiv)) throw std::overflow_error("cannot convert float infinity to integer");
    if (!(floordiv >= -9223372036854775808.0 && floordiv < 9223372036854775808.0))
        throw std::overflow_error("int() of a float outside int64");
    return static_cast<std::int64_t>(floordiv);
}

// np.clip(t, lo, hi) for floats: _NPY_MIN(_NPY_MAX(t, lo), hi), NaN kept.
inline double np_clip(double t, double lo, double hi) {
    double x = std::isnan(t) ? t : (t > lo ? t : lo);
    return std::isnan(x) ? x : (x < hi ? x : hi);
}

// numpy's argsort order for float64 with a stable tie order: NaNs last.
std::vector<std::size_t> argsort(std::span<const double> t) {
    std::vector<std::size_t> idx(t.size());
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
        const double x = t[a], y = t[b];
        if (std::isnan(x)) return false;
        if (std::isnan(y)) return true;
        return x < y;
    });
    return idx;
}

}  // namespace

// --- process noise ----------------------------------------------------------

double q_of_spread(double spread_hz, double t_sym) {
    return std::pow(2 * PI * 0.5 * spread_hz * t_sym, 3.0);
}

double spread_of_q(double q, double t_sym) { return 2.0 * std::pow(q, 1.0 / 3.0) / (2 * PI * t_sym); }

std::array<double, rxconfig::track::Q_SPREADS_HZ.size()> q_grid(double t_sym) {
    std::array<double, rxconfig::track::Q_SPREADS_HZ.size()> out{};
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = q_of_spread(rxconfig::track::Q_SPREADS_HZ[i], t_sym);
    return out;
}

// --- timing -----------------------------------------------------------------

double rx_index(const types::Timing& timing, double tau, std::int64_t n_pos, int fs, double t_sym) {
    const double sps = t_sym * fs;
    return timing.tau0 + tau * sps * (1 + timing.ppm * 1e-6) +
           timing.gamma * sq(tau / static_cast<double>(n_pos));
}

std::vector<double> rx_index(const types::Timing& timing, std::span<const double> tau, std::int64_t n_pos, int fs,
                             double t_sym) {
    std::vector<double> out(tau.size());
    for (std::size_t i = 0; i < tau.size(); ++i) out[i] = rx_index(timing, tau[i], n_pos, fs, t_sym);
    return out;
}

double tau_of_rx(const types::Timing& timing, double rel, std::int64_t n_pos, double t_sym) {
    const double s = t_sym * config::CH_FS * (1 + timing.ppm * 1e-6);
    const double np = static_cast<double>(n_pos);
    double tau = (rel - timing.tau0) / s;
    for (int k = 0; k < 3; ++k) tau = (rel - timing.tau0 - timing.gamma * sq(tau / np)) / s;
    return tau;
}

std::vector<double> tau_of_rx(const types::Timing& timing, std::span<const double> rel, std::int64_t n_pos,
                              double t_sym) {
    std::vector<double> out(rel.size());
    for (std::size_t i = 0; i < rel.size(); ++i) out[i] = tau_of_rx(timing, rel[i], n_pos, t_sym);
    return out;
}

// --- the narrow channel -----------------------------------------------------

NoisePsdSlot& NoisePsdSlot::operator=(const NoisePsdSlot& o) {
    if (this == &o) return *this;
    auto v = o.get();
    std::lock_guard<std::mutex> lock(mutex_);
    value_ = std::move(v);
    return *this;
}

std::shared_ptr<const NoisePsd> NoisePsdSlot::get() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return value_;
}

void NoisePsdSlot::set(std::shared_ptr<const NoisePsd> v) {
    std::lock_guard<std::mutex> lock(mutex_);
    value_ = std::move(v);
}

std::shared_ptr<const NoisePsd> NoisePsdSlot::get_or_compute(const std::function<NoisePsd()>& compute) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!value_) value_ = std::make_shared<const NoisePsd>(compute());
    return value_;
}

double ChanCapture::mix_audio_hz() const { return config::FE_CENTER_HZ + f_mix.to_double(); }

double ChanCapture::t_s(double n) const { return (n - t0_index) / fs; }

std::vector<double> ChanCapture::t_s(std::span<const double> n) const {
    std::vector<double> out(n.size());
    for (std::size_t i = 0; i < n.size(); ++i) out[i] = t_s(n[i]);
    return out;
}

// --- the frequency path -----------------------------------------------------

PathFn::PathFn(std::span<const double> t_in, std::span<const double> f_in, std::optional<std::span<const double>> w_in,
               double knots_s) {
    if (f_in.size() != t_in.size() || (w_in && w_in->size() != t_in.size()))
        throw std::invalid_argument("PathFn: t, f and w must have the same length");
    const std::size_t n = t_in.size();
    if (n == 0) throw std::out_of_range("index 0 is out of bounds for axis 0 with size 0");
    const auto order = argsort(t_in);
    std::vector<double> t(n), f(n), w(n), sw(n);
    for (std::size_t i = 0; i < n; ++i) {
        t[i] = t_in[order[i]];
        f[i] = f_in[order[i]];
        const double wi = w_in ? (*w_in)[order[i]] : 1.0;
        // np.maximum(w, 0.0) + 1e-9 (maximum propagates NaN)
        w[i] = (std::isnan(wi) ? wi : (wi > 0.0 ? wi : 0.0)) + 1e-9;
        sw[i] = std::sqrt(w[i]);
    }
    lo_ = t.front();
    hi_ = t.back();
    if (n >= 2)
        line_ = math::polyfit(t, f, 1, std::span<const double>(sw));
    else
        line_ = {0.0, f[0]};
    const std::int64_t n_int = py_floordiv_int(hi_ - lo_, knots_s);
    if (n >= 8 && n_int >= 1) {
        std::vector<double> knots(4, lo_);
        const double cut = hi_ - 0.25 * knots_s;
        for (std::int64_t i = 1; i <= n_int; ++i) {
            const double v = lo_ + knots_s * static_cast<double>(i);
            if (v < cut) knots.push_back(v);
        }
        knots.insert(knots.end(), 4, hi_);
        try {
            spl_ = math::make_lsq_spline(t, f, knots, 3, std::span<const double>(sw));
        } catch (const math::LinAlgError&) {
            spl_.reset();
        } catch (const std::invalid_argument&) {
            spl_.reset();
        }
        if (spl_ && !std::all_of(spl_->c.begin(), spl_->c.end(), [](double v) { return std::isfinite(v); }))
            spl_.reset();   // knots without data (Schoenberg-Whitney): a line
    }
    prepare_ends();
}

PathFn PathFn::from_parts(double lo, double hi, std::vector<double> line, std::optional<math::BSpline> spline) {
    PathFn p;
    p.lo_ = lo;
    p.hi_ = hi;
    p.line_ = std::move(line);
    p.spl_ = std::move(spline);
    p.prepare_ends();
    return p;
}

void PathFn::prepare_ends() {
    if (!spl_) return;
    const math::BSpline der = spl_->derivative();
    s_lo_ = (*spl_)(lo_);
    d_lo_ = der(lo_);
    s_hi_ = (*spl_)(hi_);
    d_hi_ = der(hi_);
}

double PathFn::operator()(double t) const {
    if (!spl_) return math::polyval(line_, t);
    if (t < lo_) return s_lo_ + d_lo_ * (t - lo_);
    if (t > hi_) return s_hi_ + d_hi_ * (t - hi_);
    return (*spl_)(np_clip(t, lo_, hi_));
}

std::vector<double> PathFn::operator()(std::span<const double> t) const {
    if (!spl_) return math::polyval(line_, t);
    std::vector<double> tc(t.size());
    for (std::size_t i = 0; i < t.size(); ++i) tc[i] = np_clip(t[i], lo_, hi_);
    std::vector<double> out = (*spl_)(tc);
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (t[i] < lo_)
            out[i] = s_lo_ + d_lo_ * (t[i] - lo_);
        else if (t[i] > hi_)
            out[i] = s_hi_ + d_hi_ * (t[i] - hi_);
    }
    return out;
}

PathFn freq_path_spline(const types::FreqPath& path, double knots_s) {
    return PathFn(path.t_s, path.f_hz, std::span<const double>(path.weight), knots_s);
}

// --- matched-filter tables --------------------------------------------------

namespace {

void check_taps(int half, std::size_t len) {
    const std::int64_t want = 2 * static_cast<std::int64_t>(half) * rxconfig::track::MF_RES + 1;
    if (static_cast<std::int64_t>(len) != want)
        throw std::invalid_argument("tab must hold " + std::to_string(want) + " taps, not " + std::to_string(len));
}

}  // namespace

MfTable::MfTable(double t_sym, double sps, int half, std::vector<double> tab, double g0)
    : t_sym_(t_sym), sps_(sps), half_(half), g0_(g0) {
    check_taps(half, tab.size());
    tab_ = std::make_shared<const std::vector<double>>(std::move(tab));
}

MfTable::MfTable(double t_sym, double sps, int half, std::vector<std::complex<double>> tab, double g0)
    : t_sym_(t_sym), sps_(sps), half_(half), g0_(g0) {
    check_taps(half, tab.size());
    ctab_ = std::make_shared<const std::vector<std::complex<double>>>(std::move(tab));
}

std::size_t MfTable::size() const { return ctab_ ? ctab_->size() : tab_->size(); }

std::span<const double> MfTable::tab() const {
    if (!tab_) throw std::logic_error("MfTable: a complex table has no real taps");
    return *tab_;
}

std::span<const std::complex<double>> MfTable::ctab() const {
    if (!ctab_) throw std::logic_error("MfTable: a real table has no complex taps");
    return *ctab_;
}

// --- the tracker's result ---------------------------------------------------

std::int64_t TrackResult::p_lo() const {
    if (pos.empty()) throw std::out_of_range("index 0 is out of bounds for axis 0 with size 0");
    return pos.front();
}

std::vector<std::int64_t> TrackResult::gi(std::span<const std::int64_t> p) const {
    const std::int64_t lo = p_lo();
    std::vector<std::int64_t> out(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) out[i] = p[i] - lo;
    return out;
}

double TrackResult::doppler_hz() const { return spread_of_q(q, spec.t_sym()); }

std::vector<double> TrackResult::t_pos_s() const {
    std::vector<double> p(pos.size());
    for (std::size_t i = 0; i < pos.size(); ++i) p[i] = static_cast<double>(pos[i]);
    return t_pos_s(p);
}

std::vector<double> TrackResult::t_pos_s(std::span<const double> p) const {
    std::vector<double> out = rx_index(timing, p, spec.n_pos(), config::CH_FS, spec.t_sym());
    for (double& v : out) v /= config::CH_FS;
    return out;
}

}  // namespace sstvae::qrss::track
