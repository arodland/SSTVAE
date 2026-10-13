// Port of sstvae/qrss/ldemod.py; see ldemod.hpp (and demod.hpp) for what is
// matched and how.

#include "qrss/ldemod.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "math/npstats.hpp"
#include "qrss/config.hpp"
#include "qrss/lwave.hpp"
#include "qrss/pytext.hpp"

namespace sstvae::qrss::ldemod {

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();

inline double sq(double x) { return x * x; }

// Python's max(a, b) of two floats: the first unless the second is greater.
inline double py_max(double a, double b) { return b > a ? b : a; }

const FrameSpec& spec_of(const TrackView& tr) {
    if (!tr.spec) throw std::invalid_argument("ldemod: a track view without a frame");
    return *tr.spec;
}

// _l_only: "<what> is L's; 'short' is waveform 0: use demod's".
void l_only(const FrameSpec& spec, std::string_view what) {
    if (spec.waveform() == frame::Waveform::L) return;
    throw frame::WrongWaveform(std::string(what) + " is L's; " + pytext::repr(spec.name()) + " is waveform " +
                               std::to_string(static_cast<int>(spec.waveform())) + ": use demod's");
}

template <typename T>
std::vector<T> take(std::span<const T> a, std::span<const std::int64_t> idx) {
    std::vector<T> out(idx.size());
    for (std::size_t i = 0; i < idx.size(); ++i) out[i] = a[demod::np::index(idx[i], a.size())];
    return out;
}

std::vector<double> repeat2(const std::vector<double>& x) {
    std::vector<double> out(2 * x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out[2 * i] = out[2 * i + 1] = x[i];
    return out;
}

// numpy's mean of x[i] where keep[i].
double masked_mean(const std::vector<double>& x, const std::vector<std::uint8_t>& keep) {
    std::vector<double> sel;
    sel.reserve(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        if (keep[i]) sel.push_back(x[i]);
    return math::mean(sel);
}

// The usable-symbol mask both estimate loops share: s2n finite and under s2_max.
std::vector<std::uint8_t> usable(const std::vector<double>& s2n, double s2_max, std::size_t& cnt) {
    std::vector<std::uint8_t> ok(s2n.size());
    cnt = 0;
    for (std::size_t i = 0; i < s2n.size(); ++i) {
        ok[i] = std::isfinite(s2n[i]) && (s2n[i] < s2_max);
        cnt += ok[i];
    }
    return ok;
}

}  // namespace

double d_pass_x(const FrameSpec& spec) { return D_PASS_L * (1.0 + spec.hdr_rho()); }

// --- symbol estimates -----------------------------------------------------------------

SymbolEstimatesL symbol_estimates_l(const TrackView& tr, std::span<const std::int64_t> stream_idx, double gain) {
    const std::vector<std::int64_t> g = demod::np::grid_of(tr, stream_idx);
    const std::vector<cplx> m = take(tr.m, g);
    const std::vector<cplx> u = take(tr.u, g);
    const std::vector<double> P = take(tr.P, g);
    const std::vector<double> N = take(tr.N, g);
    const std::vector<double> psi = take(tr.psi, g);
    const std::size_t n = g.size();
    std::vector<double> u2(n);
    for (std::size_t i = 0; i < n; ++i) u2[i] = sq(lwave::np_abs(u[i]));
    // uu = |u[pos >= 0]|^2; med = median(uu) if len(uu) else 0.0
    const std::vector<double> uu = demod::np::u2_at_nonneg(tr);
    const double med = uu.empty() ? 0.0 : math::median(uu);
    const double floor = demod::U_MIN_REL * med;
    const double gg = gain * gain;
    SymbolEstimatesL out;
    out.r.assign(n, cplx(0.0, 0.0));
    out.s2n.assign(n, INF);
    for (std::size_t i = 0; i < n; ++i) {
        const bool ok = !((psi[i] < 0.5) || (u2[i] < floor) || (u2[i] <= 0));
        if (!ok) continue;
        // m conj(u) / (u2 psi g): numpy's fused complex multiply, then its
        // complex division by the real array (cast to (d, 0)).
        out.r[i] = demod::np::cdiv(lwave::np_mul(m[i], std::conj(u[i])), cplx((u2[i] * psi[i]) * gain, 0.0));
        out.s2n[i] = (N[i] + gg * P[i]) / ((gg * u2[i]) * (psi[i] * psi[i]));
    }
    return out;
}

demod::DataEstimates data_estimates_l(const TrackView& tr) {
    const FrameSpec& spec = spec_of(tr);
    const auto lay = frame::layout(spec);
    const SymbolEstimatesL se = symbol_estimates_l(tr, lay->data, G_DATA_L);
    demod::DataEstimates out;
    out.y = frame::unpair(se.r);
    out.s2n = repeat2(se.s2n);
    const double rho = spec.hdr_rho();
    if (rho <= 0) return out;
    const double s = std::sqrt(1.0 + rho);
    for (double& v : out.y) v = s * v;
    for (double& v : out.s2n) v = (1.0 + rho) * v;
    if (!tr.hdr_bits) {
        out.extra = rho;
        return out;
    }
    const frame::HdrBits bits = frame::HdrBits::of(*tr.hdr_bits);
    const std::vector<double> h = frame::spread_signs(spec, &bits);
    if (h.size() != out.y.size())
        throw std::invalid_argument("operands could not be broadcast together with shapes (" +
                                    std::to_string(out.y.size()) + ",) (" + std::to_string(h.size()) + ",) ");
    const double sr = std::sqrt(rho);
    for (std::size_t i = 0; i < out.y.size(); ++i) {
        const double y = out.y[i] - sr * h[i];
        out.y[i] = std::isfinite(out.s2n[i]) ? y : 0.0;
    }
    out.extra = 0.0;
    return out;
}

HeaderEstimates header_estimates_l(const TrackView& tr) {
    const auto lay = frame::layout(spec_of(tr));
    const SymbolEstimatesL se = symbol_estimates_l(tr, lay->hdr_sym, G_HDR_L);
    return {frame::unpair(se.r), repeat2(se.s2n)};
}

// --- calibration ------------------------------------------------------------------------

std::vector<std::pair<double, double>> kappa_estimates_l(const TrackView& tr) {
    const FrameSpec& spec = spec_of(tr);
    l_only(spec, "ldemod.kappa_estimates_l");
    const auto lay = frame::layout(spec);
    std::vector<std::pair<double, double>> ests;
    double s2_max = demod::KAPPA_S2_MAX;
    demod::DataEstimates de;
    const double dx = d_pass_x(spec);
    if (spec.n_data()) {
        de = data_estimates_l(tr);
        std::vector<double> fin;
        fin.reserve(de.s2n.size());
        for (const double v : de.s2n)
            if (std::isfinite(v)) fin.push_back(v);
        if (!fin.empty()) s2_max = py_max(demod::KAPPA_S2_MAX, demod::KAPPA_S2_REL * math::median(fin));
    }
    const std::vector<std::int64_t>& ref = lay->ref;
    if (ref.size() >= 16) {
        const SymbolEstimatesL se = symbol_estimates_l(tr, ref, G_REF_L);
        // a = known_val[searchsorted(known_idx, ref)]: ref is a subset of known_idx
        const std::vector<std::int64_t>& kidx = lay->known_idx;
        std::vector<cplx> a(ref.size());
        for (std::size_t i = 0; i < ref.size(); ++i) {
            const auto it = std::lower_bound(kidx.begin(), kidx.end(), ref[i]);
            const auto j = static_cast<std::size_t>(it - kidx.begin());
            a[i] = lay->known_val_l[demod::np::index(static_cast<std::int64_t>(j), lay->known_val_l.size())];
        }
        std::size_t cnt = 0;
        const std::vector<std::uint8_t> ok = usable(se.s2n, s2_max, cnt);
        if (cnt >= 16) {
            const std::vector<std::int64_t> gall = demod::np::grid_of(tr, ref);
            std::vector<std::int64_t> gi;
            gi.reserve(cnt);
            for (std::size_t i = 0; i < ref.size(); ++i)
                if (ok[i]) gi.push_back(gall[i]);
            // P_ = P[gi] / (|u[gi]|^2 psi[gi]^2), counted only where the tracker measured.
            std::vector<double> pm(gi.size());
            for (std::size_t i = 0; i < gi.size(); ++i) {
                const double P = tr.P[demod::np::index(gi[i], tr.P.size())];
                const double u2 = sq(lwave::np_abs(tr.u[demod::np::index(gi[i], tr.u.size())]));
                const double psi = tr.psi[demod::np::index(gi[i], tr.psi.size())];
                const double P_ = P / (u2 * (psi * psi));
                const bool meas = !tr.meas.empty() && tr.meas[demod::np::index(gi[i], tr.meas.size())] != 0;
                pm[i] = meas ? P_ : 0.0;
            }
            const double p_m = math::mean(pm);
            const double den = masked_mean(se.s2n, ok) - p_m;   // mean(N_ + P_u)
            if (den > 0) {
                std::vector<double> e2;
                e2.reserve(cnt);
                for (std::size_t i = 0; i < ref.size(); ++i)
                    if (ok[i]) e2.push_back(sq(lwave::np_abs(se.r[i] - a[i])));
                const double k = ((math::mean(e2) - D_REF_L) + p_m) / den;
                const double res = py_max(den - p_m, 0.0) + D_REF_L;   // E|e|^2 at kappa 1
                const double var = 2.0 * demod::np::py_pow(res, 2.0) / static_cast<double>(2 * cnt) /
                                   demod::np::py_pow(den, 2.0);
                ests.emplace_back(k, var);
            }
        }
    }
    if (spec.n_data() >= 256) {
        std::size_t cnt = 0;
        const std::vector<std::uint8_t> ok = usable(de.s2n, s2_max, cnt);
        if (cnt >= 256) {
            const double den = masked_mean(de.s2n, ok);
            if (den > 0) {
                std::vector<double> y2;
                y2.reserve(cnt);
                for (std::size_t i = 0; i < de.y.size(); ++i)
                    if (ok[i]) y2.push_back(sq(de.y[i]));
                const double k = (((math::mean(y2) - 1.0) - de.extra) - dx) / den;
                const double var = 2.0 * demod::np::py_pow(((1.0 + de.extra) + dx) + den, 2.0) /
                                   static_cast<double>(cnt) / demod::np::py_pow(den, 2.0);
                ests.emplace_back(k, var);
            }
        }
    }
    return ests;
}

demod::Calibration calibrate_l(const TrackView& tr) {
    l_only(spec_of(tr), "ldemod.calibrate_l");
    return demod::combine(kappa_estimates_l(tr));
}

// --- extract ----------------------------------------------------------------------------

demod::Extraction extract_l(const TrackView& tr, const FrameSpec& spec, std::uint64_t q,
                            std::string_view estimator, bool calibrate_kappa, std::string_view shown) {
    l_only(spec, "ldemod.extract_l");
    const demod::Calibration cal = calibrate_kappa ? calibrate_l(tr) : demod::Calibration{1.0, false};
    const double scale = !cal.suspect ? cal.kappa : 1.0;
    const double dx = d_pass_x(spec);
    const demod::DataEstimates de = data_estimates_l(tr);
    demod::Extraction ex;
    ex.s2.resize(de.s2n.size());
    for (std::size_t i = 0; i < de.s2n.size(); ++i) ex.s2[i] = (de.s2n[i] * scale + dx) + de.extra;
    demod::finish_data(ex, de.y, ex.s2, q, estimator, shown);
    std::vector<double> llr(spec.has_header() ? static_cast<std::size_t>(config::N_HDR_BITS) : 0, 0.0);
    if (spec.n_hdr()) {
        HeaderEstimates he = header_estimates_l(tr);
        std::vector<double> blk(he.y.size());
        for (std::size_t i = 0; i < he.y.size(); ++i) {
            const double s2h = he.s2n[i] * scale + D_HDR_L;
            blk[i] = std::isfinite(s2h) ? (2.0 * he.y[i]) / s2h : 0.0;
        }
        if (blk.size() != llr.size())
            throw std::invalid_argument("operands could not be broadcast together with shapes (" +
                                        std::to_string(llr.size()) + ",) (" + std::to_string(blk.size()) + ",) ");
        for (std::size_t i = 0; i < llr.size(); ++i) llr[i] = llr[i] + blk[i];
        ex.y_hdr = std::move(he.y);
        ex.llr_block = std::move(blk);
    }
    if (spec.hdr_rho() > 0 && de.extra > 0) {
        std::vector<double> s2s(de.s2n.size());
        for (std::size_t i = 0; i < de.s2n.size(); ++i) s2s[i] = de.s2n[i] * scale + dx;
        std::vector<double> spr = demod::spread_llr(de.y, s2s, spec);
        if (spr.size() != llr.size())
            throw std::invalid_argument("operands could not be broadcast together with shapes (" +
                                        std::to_string(llr.size()) + ",) (" + std::to_string(spr.size()) + ",) ");
        for (std::size_t i = 0; i < llr.size(); ++i) llr[i] = llr[i] + spr[i];
        ex.llr_spread = std::move(spr);
    }
    ex.llr.resize(llr.size());
    for (std::size_t i = 0; i < llr.size(); ++i) ex.llr[i] = static_cast<float>(llr[i]);
    ex.kappa = cal.kappa;
    ex.suspect = cal.suspect;
    ex.y = de.y;
    ex.estimator = std::string(estimator);
    return ex;
}

}  // namespace sstvae::qrss::ldemod
