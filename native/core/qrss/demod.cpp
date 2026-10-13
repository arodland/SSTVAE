// Port of sstvae/qrss/demod.py; see demod.hpp for what is matched and how.

#include "qrss/demod.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "math/npstats.hpp"
#include "qrss/config.hpp"
#include "qrss/lwave.hpp"
#include "qrss/npsum.hpp"
#include "qrss/precoder.hpp"
#include "qrss/pytext.hpp"
#include "qrss/sequences.hpp"

namespace sstvae::qrss::demod {

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();

inline double sq(double x) { return x * x; }

// float(BETA), as Python computes it from the Fraction.
constexpr double BETA_F = static_cast<double>(config::BETA.num) / static_cast<double>(config::BETA.den);

// The callers' refusal for a frame of the other waveform.
[[noreturn]] void wrong_waveform(const FrameSpec& spec, std::string_view what, std::string_view use) {
    throw frame::WrongWaveform(std::string(what) + " is CE's; " + pytext::repr(spec.name()) + " is waveform " +
                               std::to_string(static_cast<int>(spec.waveform())) + ": use " + std::string(use));
}

const FrameSpec& spec_of(const TrackView& tr) {
    if (!tr.spec) throw std::invalid_argument("demod: a track view without a frame");
    return *tr.spec;
}

// The values of `a` at numpy indices `idx`.
template <typename T>
std::vector<T> take(std::span<const T> a, std::span<const std::int64_t> idx) {
    std::vector<T> out(idx.size());
    for (std::size_t i = 0; i < idx.size(); ++i) out[i] = a[np::index(idx[i], a.size())];
    return out;
}

// Python's max(a, b) and min(a, b) of two floats: the first unless the
// second compares strictly greater (smaller), NaN passing as numpy's does not.
inline double py_max(double a, double b) { return b > a ? b : a; }
inline double py_min(double a, double b) { return b < a ? b : a; }

// Python's sum() of the estimates' 1/v and k/v. Each variance is a numpy
// float64 (a division by ok.sum(), an np.int64), so the terms are numpy
// scalars, not exact floats, and CPython's sum() leaves its fast paths
// (Neumaier's compensation among them) for plain `+`: 0 + t0 + t1 ...,
// left to right. With at most two terms it would agree with the
// compensated sum anyway.
double py_fsum(std::span<const double> xs) {
    double f = 0.0;
    for (const double x : xs) f = f + x;
    return f;
}

// numpy's "operands could not be broadcast together" for two 1-D arrays.
[[noreturn]] void broadcast_error(std::size_t a, std::size_t b) {
    throw std::invalid_argument("operands could not be broadcast together with shapes (" + std::to_string(a) +
                                ",) (" + std::to_string(b) + ",) ");
}

// Pairwise mean of x[i] where keep[i].
double masked_mean(std::span<const double> x, const std::vector<std::uint8_t>& keep) {
    std::vector<double> sel;
    sel.reserve(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        if (keep[i]) sel.push_back(x[i]);
    return math::mean(sel);
}

}  // namespace

// --- numpy's arithmetic -----------------------------------------------------------

namespace np {

double py_pow(double x, double y) {
    // A volatile exponent keeps the compiler from folding pow(x, 2.0) into
    // x * x: Python's float ** calls the C library's pow.
    volatile double yy = y;
    return std::pow(x, static_cast<double>(yy));
}

cplx cdiv(cplx a, cplx b) {
    const double ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
    const double abs_br = std::fabs(br), abs_bi = std::fabs(bi);
    if (abs_br >= abs_bi) {
        if (abs_br == 0 && abs_bi == 0) return {ar / abs_br, ai / abs_br};
        const double rat = bi / br;
        const double scl = 1.0 / (br + bi * rat);
        return {(ar + ai * rat) * scl, (ai - ar * rat) * scl};
    }
    const double rat = br / bi;
    const double scl = 1.0 / (bi + br * rat);
    return {(ar * rat + ai) * scl, (ai * rat - ar) * scl};
}

std::size_t index(std::int64_t i, std::size_t size) {
    const auto n = static_cast<std::int64_t>(size);
    const std::int64_t j = i < 0 ? i + n : i;
    if (j < 0 || j >= n)
        throw std::out_of_range("index " + std::to_string(i) + " is out of bounds for axis 0 with size " +
                                std::to_string(size));
    return static_cast<std::size_t>(j);
}

std::vector<std::int64_t> grid_of(const TrackView& tr, std::span<const std::int64_t> stream_idx) {
    // p = layout.pos[stream_idx]; gi = p - int(tr.pos[0]). Each array is
    // then indexed by gi, checked against its own length (`take`).
    const auto lay = frame::layout(spec_of(tr));
    const std::span<const std::int64_t> lpos(lay->pos);
    std::vector<std::int64_t> res;
    res.reserve(stream_idx.size());
    for (const std::int64_t s : stream_idx) res.push_back(lpos[index(s, lpos.size())]);
    const std::int64_t p_lo = tr.pos[index(0, tr.pos.size())];
    for (auto& p : res) p -= p_lo;
    return res;
}

std::vector<double> u2_at_nonneg(const TrackView& tr) {
    if (tr.u.size() != tr.pos.size())
        throw std::out_of_range("boolean index did not match indexed array along axis 0; size of axis is " +
                                std::to_string(tr.u.size()) + " but size of corresponding boolean axis is " +
                                std::to_string(tr.pos.size()));
    std::vector<double> out;
    out.reserve(tr.u.size());
    for (std::size_t i = 0; i < tr.u.size(); ++i)
        if (tr.pos[i] >= 0) out.push_back(sq(lwave::np_abs(tr.u[i])));
    return out;
}

}  // namespace np

namespace {

inline std::vector<std::int64_t> gi_of(const TrackView& tr, std::span<const std::int64_t> stream_idx) {
    return np::grid_of(tr, stream_idx);
}

}  // namespace

TrackView view(const track::TrackResult& tr) {
    TrackView v;
    v.spec = &tr.spec;
    v.pos = tr.pos;
    v.m = tr.m;
    v.u = tr.u;
    v.c = tr.c;
    v.P = tr.P;
    v.N = tr.N;
    v.psi = tr.psi;
    v.meas = tr.meas;
    if (const auto* c = tr.ce_classes()) {
        v.spread_rho = c->spread_rho;
    } else if (const auto* lc = tr.l_classes()) {
        v.spread_rho = lc->spread_rho;
        if (lc->hdr_bits) v.hdr_bits = std::span<const std::uint8_t>(*lc->hdr_bits);
    }
    return v;
}

// --- symbol estimates ------------------------------------------------------------

SymbolEstimates symbol_estimates(const TrackView& tr, std::span<const std::int64_t> stream_idx) {
    const std::vector<std::int64_t> g = gi_of(tr, stream_idx);
    const std::vector<cplx> m = take(tr.m, g);
    const std::vector<cplx> u = take(tr.u, g);
    const std::vector<double> P = take(tr.P, g);
    const std::vector<double> N = take(tr.N, g);
    const std::vector<double> psi = take(tr.psi, g);
    const std::size_t n = g.size();
    std::vector<double> u2(n);
    for (std::size_t i = 0; i < n; ++i) u2[i] = sq(lwave::np_abs(u[i]));
    // med = float(np.median(np.abs(tr.u[tr.pos >= 0]) ** 2)) if len(tr.u) else 0.0
    const double med = tr.u.empty() ? 0.0 : math::median(np::u2_at_nonneg(tr));
    const double floor = U_MIN_REL * med;
    const double k = config::K_LIN;
    const double c_ap = np::py_pow(config::A0, 2.0) + np::py_pow(k, 2.0);   // A0 ** 2 + K_LIN ** 2
    const double k2 = np::py_pow(k, 2.0);                                  // K_LIN ** 2
    SymbolEstimates out;
    out.y.assign(n, 0.0);
    out.s2n.assign(n, INF);
    out.s2.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const bool bad = (psi[i] < 0.5) || (u2[i] < floor) || (u2[i] <= 0);
        if (!bad) {
            // Im(m conj(u)), numpy's fused complex multiply
            const double im = lwave::np_mul(m[i], std::conj(u[i])).imag();
            out.y[i] = im / ((k * u2[i]) * psi[i]);
            out.s2n[i] = (N[i] / 2 + (c_ap * P[i]) / 2) / ((k2 * u2[i]) * (psi[i] * psi[i]));
        }
        out.s2[i] = out.s2n[i] + D_PASS;
    }
    return out;
}

DataEstimates data_estimates(const TrackView& tr) {
    const auto lay = frame::layout(spec_of(tr));
    SymbolEstimates se = symbol_estimates(tr, lay->data);
    DataEstimates out;
    const double rho = tr.spread_rho;
    if (rho <= 0) {   // a NaN rho goes on, as Python's does
        out.y = std::move(se.y);
        out.s2n = std::move(se.s2n);
        out.extra = 0.0;
        return out;
    }
    const double g = std::exp(-np::py_pow(BETA_F, 2.0) * rho / 2);
    const double g2 = np::py_pow(g, 2.0);
    out.y.resize(se.y.size());
    out.s2n.resize(se.s2n.size());
    for (std::size_t i = 0; i < se.y.size(); ++i) {
        out.y[i] = se.y[i] / g;
        out.s2n[i] = se.s2n[i] / g2;
    }
    out.extra = rho;
    return out;
}

std::vector<double> spread_llr(std::span<const double> y, std::span<const double> s2, const FrameSpec& spec) {
    if (y.size() != s2.size()) broadcast_error(y.size(), s2.size());
    const auto n_data = static_cast<std::size_t>(spec.n_data());
    if (y.size() != n_data) broadcast_error(y.size(), n_data);
    const double a = 2.0 * std::sqrt(spec.hdr_rho());
    const std::vector<std::int8_t> wh = sequences::spread_whitener(n_data);
    std::vector<double> out(static_cast<std::size_t>(config::N_HDR_BITS), 0.0);
    for (std::size_t i = 0; i < n_data; ++i) {
        const bool ok = std::isfinite(s2[i]);
        const double l = ok ? (a * y[i]) / (s2[i] + 1.0) : 0.0;
        out[i % out.size()] += l * static_cast<double>(wh[i]);
    }
    return out;
}

// --- calibration ------------------------------------------------------------------

Calibration combine(std::span<const std::pair<double, double>> ests) {
    if (ests.empty()) return {1.0, false};
    std::vector<double> inv, kv;
    inv.reserve(ests.size());
    kv.reserve(ests.size());
    for (const auto& [k, v] : ests) {
        inv.push_back(1.0 / v);
        kv.push_back(k / v);
    }
    const double wsum = py_fsum(inv);
    const double k_hat = py_fsum(kv) / wsum;
    const double sd = 1.0 / std::sqrt(wsum);
    const double w0 = 1.0 / np::py_pow(KAPPA_PRIOR_SD, 2.0);
    double kappa = (k_hat * wsum + 1.0 * w0) / (wsum + w0);
    const double lo = KAPPA_RANGE[0], hi = KAPPA_RANGE[1];
    kappa = py_min(py_max(kappa, lo), hi);
    const bool suspect = (k_hat < lo - 3 * sd) || (k_hat > hi + 3 * sd);
    return {kappa, suspect};
}

Calibration calibrate(const TrackView& tr) {
    const FrameSpec& spec = spec_of(tr);
    if (spec.waveform() != frame::Waveform::CE) wrong_waveform(spec, "demod.calibrate", "ldemod.calibrate_l");
    const auto lay = frame::layout(spec);
    std::vector<std::pair<double, double>> ests;
    const std::vector<std::int64_t>& ref = lay->ref;
    double s2_max = KAPPA_S2_MAX;
    DataEstimates de;
    if (spec.n_data()) {
        de = data_estimates(tr);
        std::vector<double> fin;
        fin.reserve(de.s2n.size());
        for (const double v : de.s2n)
            if (std::isfinite(v)) fin.push_back(v);
        if (!fin.empty()) s2_max = py_max(KAPPA_S2_MAX, KAPPA_S2_REL * math::median(fin));
    }
    if (ref.size() >= 16) {
        const SymbolEstimates se = symbol_estimates(tr, ref);
        const std::vector<std::int64_t> g = gi_of(tr, ref);
        const std::vector<cplx> c = take(tr.c, g);
        std::vector<std::uint8_t> ok(ref.size());
        std::size_t cnt = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
            ok[i] = std::isfinite(se.s2n[i]) && (se.s2n[i] < s2_max);
            cnt += ok[i];
        }
        if (cnt >= 16) {
            const double den = masked_mean(se.s2n, ok);
            if (den > 0) {
                std::vector<double> r2;
                r2.reserve(cnt);
                for (std::size_t i = 0; i < ref.size(); ++i)
                    if (ok[i]) r2.push_back(sq(se.y[i] - c[i].imag() / config::K_LIN));
                const double k = (math::mean(r2) - D_REF) / den;
                const double var =
                    2.0 * np::py_pow(den + D_REF, 2.0) / static_cast<double>(cnt) / np::py_pow(den, 2.0);
                ests.emplace_back(k, var);
            }
        }
    }
    if (spec.n_data() >= 256) {
        std::vector<std::uint8_t> ok(de.s2n.size());
        std::size_t cnt = 0;
        for (std::size_t i = 0; i < de.s2n.size(); ++i) {
            ok[i] = std::isfinite(de.s2n[i]) && (de.s2n[i] < s2_max);
            cnt += ok[i];
        }
        if (cnt >= 256) {
            const double den = masked_mean(de.s2n, ok);
            if (den > 0) {
                std::vector<double> y2;
                y2.reserve(cnt);
                for (std::size_t i = 0; i < de.y.size(); ++i)
                    if (ok[i]) y2.push_back(sq(de.y[i]));
                const double k = (math::mean(y2) - 1.0 - de.extra - D_PASS) / den;
                const double var = 2.0 * np::py_pow(1.0 + de.extra + D_PASS + den, 2.0) / static_cast<double>(cnt) /
                                   np::py_pow(den, 2.0);
                ests.emplace_back(k, var);
            }
        }
    }
    return combine(ests);
}

// --- block estimators --------------------------------------------------------------

void check_estimator(std::string_view estimator, std::string_view shown) {
    if (estimator == "joint" || estimator == "plain") return;
    throw std::invalid_argument("estimator must be 'joint' or 'plain', not " +
                                (shown.empty() ? pytext::repr(estimator) : std::string(shown)));
}

BlockEstimate block_estimate(std::span<const double> y, std::span<const double> s2, std::uint64_t q,
                             std::string_view estimator, std::string_view shown) {
    const std::size_t n = y.size();
    const std::vector<std::int8_t> csign = sequences::scrambler(q, n);
    const std::vector<std::int64_t> bi = precoder::block_index(n);
    const std::vector<std::size_t> sizes = precoder::block_sizes(n);
    // The reference's else branch (an unknown estimator) touches nothing that
    // can fail first; either known branch then broadcasts s2 against y.
    if (estimator != "plain" && estimator != "joint") check_estimator(estimator, shown);
    // np.where(erased, 0.0, y): numpy's three-operand broadcast text
    if (s2.size() != n)
        throw std::invalid_argument("operands could not be broadcast together with shapes (" +
                                    std::to_string(s2.size()) + ",) () (" + std::to_string(n) + ",) ");
    std::vector<std::uint8_t> erased(n);
    for (std::size_t i = 0; i < n; ++i) erased[i] = !std::isfinite(s2[i]);
    const std::size_t nb = sizes.size();
    BlockEstimate out;
    std::vector<double> v(nb, 0.0);
    if (estimator == "plain") {
        std::vector<double> yy(n);
        for (std::size_t i = 0; i < n; ++i) {
            yy[i] = erased[i] ? 0.0 : y[i];
            v[static_cast<std::size_t>(bi[i])] += erased[i] ? 1.0 : s2[i];
        }
        for (std::size_t b = 0; b < nb; ++b) v[b] = v[b] / static_cast<double>(sizes[b]);
        const std::vector<double> w = precoder::wht(yy);
        out.z.resize(n);
        for (std::size_t i = 0; i < n; ++i) out.z[i] = static_cast<double>(csign[i]) * w[i];
    } else if (estimator == "joint") {
        std::vector<double> d(n), yy(n);
        std::vector<double> gam(nb, 0.0), sum_ds2(nb, 0.0), sum_cross(nb, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            d[i] = erased[i] ? 0.0 : 1.0 / (1.0 + s2[i]);
            yy[i] = erased[i] ? 0.0 : y[i];
            gam[static_cast<std::size_t>(bi[i])] += d[i];
        }
        for (std::size_t b = 0; b < nb; ++b) gam[b] = gam[b] / static_cast<double>(sizes[b]);
        for (std::size_t i = 0; i < n; ++i) {
            const auto b = static_cast<std::size_t>(bi[i]);
            sum_ds2[b] += erased[i] ? 0.0 : d[i] * d[i] * s2[i];
        }
        for (std::size_t i = 0; i < n; ++i) {
            const auto b = static_cast<std::size_t>(bi[i]);
            sum_cross[b] += sq(d[i] - gam[b]);
        }
        for (std::size_t b = 0; b < nb; ++b) {
            const double num = (sum_ds2[b] + sum_cross[b]) / static_cast<double>(sizes[b]);
            v[b] = gam[b] > 0 ? num / sq(gam[b]) : INF;
        }
        std::vector<double> dy(n);
        for (std::size_t i = 0; i < n; ++i) dy[i] = d[i] * yy[i];
        const std::vector<double> w = precoder::wht(dy);
        out.z.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double gb = gam[static_cast<std::size_t>(bi[i])];
            const double a = (static_cast<double>(csign[i]) * w[i]) / (gb > 0 ? gb : 1.0);
            out.z[i] = gb > 0 ? a : 0.0;
        }
    } else {
        check_estimator(estimator, shown);
    }
    out.v.resize(n);
    for (std::size_t i = 0; i < n; ++i) out.v[i] = v[static_cast<std::size_t>(bi[i])];
    return out;
}

// --- extract ------------------------------------------------------------------------

void finish_data(Extraction& ex, std::span<const double> y, std::span<const double> s2, std::uint64_t q,
                 std::string_view estimator, std::string_view shown) {
    const BlockEstimate be = block_estimate(y, s2, q, estimator, shown);
    const std::size_t n = be.z.size();
    ex.z.resize(n);
    ex.w.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double v = be.v[i];
        const double w = (std::isfinite(v) && v > 0) ? 1.0 / v : 0.0;
        ex.w[i] = static_cast<float>(w);
        ex.z[i] = static_cast<float>(w > 0 ? be.z[i] : 0.0);
    }
}

Extraction extract(const TrackView& tr, const FrameSpec& spec, std::uint64_t q, std::string_view estimator,
                   bool calibrate_kappa, std::string_view shown) {
    if (spec.waveform() != frame::Waveform::CE) wrong_waveform(spec, "demod.extract", "ldemod.extract_l");
    const auto lay = frame::layout(spec);
    const Calibration cal = calibrate_kappa ? calibrate(tr) : Calibration{1.0, false};
    const double scale = !cal.suspect ? cal.kappa : 1.0;
    const DataEstimates de = data_estimates(tr);
    Extraction ex;
    ex.s2.resize(de.s2n.size());
    for (std::size_t i = 0; i < de.s2n.size(); ++i) ex.s2[i] = de.s2n[i] * scale + D_PASS + de.extra;
    finish_data(ex, de.y, ex.s2, q, estimator, shown);
    std::vector<double> llr(spec.has_header() ? static_cast<std::size_t>(config::N_HDR_BITS) : 0, 0.0);
    if (spec.n_hdr()) {
        SymbolEstimates se = symbol_estimates(tr, lay->hdr_bits);
        std::vector<double> blk(se.y.size());
        for (std::size_t i = 0; i < se.y.size(); ++i) {
            const double s2h = se.s2n[i] * scale + D_PASS;
            blk[i] = std::isfinite(s2h) ? 2.0 * se.y[i] / s2h : 0.0;
        }
        if (blk.size() != llr.size()) broadcast_error(llr.size(), blk.size());
        for (std::size_t i = 0; i < llr.size(); ++i) llr[i] = llr[i] + blk[i];
        ex.y_hdr = std::move(se.y);
        ex.llr_block = std::move(blk);
    }
    if (spec.hdr_rho() > 0 && de.extra > 0) {
        std::vector<double> s2s(de.s2n.size());
        for (std::size_t i = 0; i < de.s2n.size(); ++i) s2s[i] = de.s2n[i] * scale + D_PASS;
        std::vector<double> spr = spread_llr(de.y, s2s, spec);
        if (spr.size() != llr.size()) broadcast_error(llr.size(), spr.size());
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

}  // namespace sstvae::qrss::demod
