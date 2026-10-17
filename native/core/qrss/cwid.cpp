// cwid.py: see cwid.hpp.

#include "qrss/cwid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

#include "math/npcomplex.hpp"
#include "qrss/filters.hpp"
#include "qrss/morse.hpp"
#include "qrss/npsum.hpp"
#include "qrss/rational.hpp"

namespace sstvae::qrss::cwid {
namespace {

using cdouble = std::complex<double>;
namespace rxc = rxconfig::cwid;
constexpr double INF = std::numeric_limits<double>::infinity();

// int(x) of a float, with int()'s refusals.
std::int64_t py_int(double x) {
    if (std::isnan(x)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(x)) throw std::overflow_error("cannot convert float infinity to integer");
    return static_cast<std::int64_t>(std::trunc(x));
}

// ndarray.astype(np.int64) of a float on x86-64: NaN, infinities and
// values past int64 are INT64_MIN.
std::int64_t to_int64(double v) {
    if (!(v >= -9223372036854775808.0 && v < 9223372036854775808.0)) return std::numeric_limits<std::int64_t>::min();
    return static_cast<std::int64_t>(v);
}

double np_maximum(double a, double b) {
    if (std::isnan(a)) return a;
    if (std::isnan(b)) return b;
    return a >= b ? a : b;
}

// np.argmax: the first maximum, or the first NaN.
std::size_t np_argmax(const double* x, std::size_t n) {
    std::size_t k = 0;
    double mp = x[0];
    if (std::isnan(mp)) return 0;
    for (std::size_t i = 1; i < n; ++i) {
        if (!(x[i] <= mp)) {
            mp = x[i];
            k = i;
            if (std::isnan(mp)) break;
        }
    }
    return k;
}

// np.convolve(r, np.ones(15), "same") for a row of 192: numpy's correlate
// calls its BLAS dot on each window, which for under 16 terms is a running
// sum from 0 (the products by 1 exact), and DOUBLE_dot adds that onto 0.
std::array<double, UNITS> smooth(const std::array<double, UNITS>& r) {
    constexpr int H = rxc::AMP_SMOOTH_UNITS / 2;
    std::array<double, UNITS> out{};
    for (int u = 0; u < UNITS; ++u) {
        const int lo = std::max(0, u - H), hi = std::min(UNITS - 1, u + H);
        double d = 0.0;
        for (int i = lo; i <= hi; ++i) d += r[static_cast<std::size_t>(i)];
        out[static_cast<std::size_t>(u)] = 0.0 + d;
    }
    return out;
}

// np.dot(a, b) of two contiguous float64 vectors as numpy's OpenBLAS runs
// it on an AVX-512 machine (dot_compute + ddot_microk_skylakex-2.c, built
// with GCC's default FP contraction, so every multiply-add is one fma):
// the 16k leading terms in the kernel -- four 8-lane accumulators over
// blocks of 32, folded to 4 lanes, then the same four 4-lane accumulators
// over blocks of 16, summed in order and reduced pairwise -- the rest a
// running fma; DOUBLE_dot adds the result onto 0. Verified bit for bit
// against np.dot at n = 7, 16, 32, 40, 100 and 176 (2000 draws each);
// another machine's OpenBLAS kernel orders it differently, which is why
// match() is held to a tolerance and not to the bit.
double blas_dot(const double* x, const double* y, std::size_t n) {
    double dot = 0.0;
    const std::size_t n1 = n & ~std::size_t{15};
    if (n1) {
        double a5[4][8] = {};
        std::size_t i = 0;
        const std::size_t n32 = n1 & ~std::size_t{31};
        for (; i < n32; i += 32)
            for (std::size_t a = 0; a < 4; ++a)
                for (std::size_t l = 0; l < 8; ++l)
                    a5[a][l] = std::fma(x[i + 8 * a + l], y[i + 8 * a + l], a5[a][l]);
        double acc[4][4];
        for (std::size_t a = 0; a < 4; ++a)
            for (std::size_t l = 0; l < 4; ++l) acc[a][l] = a5[a][l] + a5[a][l + 4];
        for (; i < n1; i += 16)
            for (std::size_t a = 0; a < 4; ++a)
                for (std::size_t l = 0; l < 4; ++l) acc[a][l] = std::fma(x[i + 4 * a + l], y[i + 4 * a + l], acc[a][l]);
        double s[4];
        for (std::size_t l = 0; l < 4; ++l) s[l] = ((acc[0][l] + acc[1][l]) + acc[2][l]) + acc[3][l];
        const double h0 = s[0] + s[2], h1 = s[1] + s[3];
        dot = h0 + h1;
    }
    for (std::size_t i = n1; i < n; ++i) dot = std::fma(y[i], x[i], dot);
    return 0.0 + dot;
}

// The inverse Morse table.
const std::map<std::string, char, std::less<>>& inverse() {
    static const std::map<std::string, char, std::less<>> inv = [] {
        std::map<std::string, char, std::less<>> m;
        for (const config::MorseCode& c : config::MORSE) m.emplace(std::string(c.code), c.ch);
        return m;
    }();
    return inv;
}

std::vector<std::int64_t> pattern_lens() {
    std::vector<std::int64_t> lens;
    for (const Pattern& p : patterns()) lens.push_back(static_cast<std::int64_t>(p.units.size()));
    return lens;
}

}  // namespace

// --- the unit sums --------------------------------------------------------------

UnitSource unit_source(const track::TrackResult& tr, const track::NoisePsdFn& fn) {
    if (!tr.z_d || !tr.chan) throw std::invalid_argument("unit_source: the track has no z_d or channel");
    UnitSource s;
    s.z = std::span<const cdouble>(*tr.z_d);
    s.t0_index = tr.chan->t0_index;
    s.fs = static_cast<double>(tr.chan->fs);
    s.timing = tr.timing;
    std::shared_ptr<const track::ChanCapture> chan = tr.chan;
    s.noise_at = [chan, fn](std::span<const double> n) {
        const std::vector<double> t = chan->t_s(n);
        return track::noise_psd_at(*chan, std::span<const double>(t), fn);
    };
    return s;
}

UnitSums unit_sums(const UnitSource& src, const frame::FrameSpec& spec) {
    const std::int64_t n_win = spec.n_win();
    const std::int64_t unit = spec.cw_window() / UNITS;   // symbols per Morse unit (CE 2, L 1)
    UnitSums out;
    out.n_win = static_cast<std::size_t>(n_win);
    out.x0.assign(out.n_win * UNITS, cdouble(0.0, 0.0));
    out.x1.assign(out.n_win * UNITS, cdouble(0.0, 0.0));
    out.s2.assign(out.n_win * UNITS, 1.0);
    const std::span<const cdouble> z = src.z;
    const auto nz = static_cast<std::int64_t>(z.size());
    const std::vector<std::int64_t> ws = spec.win_start_pos();
    std::vector<double> ta(UNITS), tb(UNITS), mid(UNITS);
    std::vector<std::int64_t> first(UNITS), cnt(UNITS);
    std::vector<cdouble> v, p;
    for (std::size_t w = 0; w < out.n_win; ++w) {
        const double pw = static_cast<double>(ws[w]) - 0.5;
        for (int u = 0; u < UNITS; ++u) {
            ta[static_cast<std::size_t>(u)] = pw + static_cast<double>(unit * u);
            tb[static_cast<std::size_t>(u)] = ta[static_cast<std::size_t>(u)] + static_cast<double>(unit);
        }
        std::vector<double> na = track::rx_index(src.timing, ta, spec.n_pos(), config::CH_FS, spec.t_sym());
        std::vector<double> nb = track::rx_index(src.timing, tb, spec.n_pos(), config::CH_FS, spec.t_sym());
        for (int u = 0; u < UNITS; ++u) {
            na[static_cast<std::size_t>(u)] = src.t0_index + na[static_cast<std::size_t>(u)];
            nb[static_cast<std::size_t>(u)] = src.t0_index + nb[static_cast<std::size_t>(u)];
        }
        // np.max(np.ceil(nb) - first): NaN-propagating
        double wmax = -INF;
        for (int u = 0; u < UNITS; ++u) {
            first[static_cast<std::size_t>(u)] = to_int64(std::ceil(na[static_cast<std::size_t>(u)]));
            const double d = std::ceil(nb[static_cast<std::size_t>(u)]) -
                             static_cast<double>(first[static_cast<std::size_t>(u)]);
            if (std::isnan(wmax)) continue;
            if (std::isnan(d) || d > wmax) wmax = d;
        }
        const std::int64_t width = py_int(wmax) + 1;
        if (nz == 0 && width > 0) throw std::out_of_range("index -1 is out of bounds for axis 0 with size 0");
        const std::size_t W = width > 0 ? static_cast<std::size_t>(width) : 0;
        v.assign(W, cdouble(0.0, 0.0));
        p.assign(W, cdouble(0.0, 0.0));
        for (int u = 0; u < UNITS; ++u) {
            const std::size_t us = static_cast<std::size_t>(u);
            const double a = na[us], b = nb[us];
            const double span = b - a;
            std::int64_t c = 0;
            for (std::size_t j = 0; j < W; ++j) {
                const std::int64_t nn = first[us] + static_cast<std::int64_t>(j);
                const bool inside = (static_cast<double>(nn) < b) && nn >= 0 && nn < nz;
                const double x = (static_cast<double>(nn) - a) / span;
                const cdouble val = inside ? z[static_cast<std::size_t>(std::clamp<std::int64_t>(nn, 0, nz - 1))]
                                           : cdouble(0.0, 0.0);
                // np.exp(2j * np.pi * x): (0 + 2 pi j) * (x + 0j) by numpy's
                // FMA product, then glibc's cexp
                const cdouble arg = math::np::mul(cdouble(0.0, 6.283185307179586), cdouble(x, 0.0));
                v[j] = val;
                p[j] = math::np::mul(val, std::exp(arg));
                c += inside ? 1 : 0;
            }
            out.x0[w * UNITS + us] = npsum::sum(std::span<const cdouble>(v));
            out.x1[w * UNITS + us] = npsum::sum(std::span<const cdouble>(p));
            cnt[us] = c;
            mid[us] = 0.5 * (a + b);
        }
        const std::vector<double> psd = src.noise_at(mid);
        if (psd.size() != 1 && psd.size() != static_cast<std::size_t>(UNITS))
            throw std::invalid_argument("noise_psd_at returned " + std::to_string(psd.size()) + " values for " +
                                        std::to_string(UNITS) + " units");
        for (int u = 0; u < UNITS; ++u) {
            const std::size_t us = static_cast<std::size_t>(u);
            const double ps = psd.size() == 1 ? psd[0] : psd[us];
            out.s2[w * UNITS + us] = np_maximum(ps * src.fs * static_cast<double>(cnt[us]), 1e-300);
        }
    }
    return out;
}

// --- soft values and LLRs -------------------------------------------------------

void fsk_only(const frame::FrameSpec& spec, bool ook) {
    if (ook && spec.waveform() == frame::Waveform::L)
        throw std::invalid_argument(spec.name() + ": waveform L keys its callsign by FSK only");
}

namespace {

// |X|**2: numpy's SIMD complex absolute value, squared.
double power(cdouble x) {
    const double a = math::np::abs(x);
    return a * a;
}

// E = max(mean(p0[PLAIN] - s2[PLAIN]), 0) of one window.
double plain_energy(const double* p0, const double* s2) {
    std::array<double, rxc::PLAIN_UNITS.size()> d{};
    for (std::size_t i = 0; i < d.size(); ++i) {
        const auto u = static_cast<std::size_t>(rxc::PLAIN_UNITS[i]);
        d[i] = p0[u] - s2[u];
    }
    const double m = npsum::sum(std::span<const double>(d)) / static_cast<double>(d.size());
    return np_maximum(m, 0.0);
}

}  // namespace

Units32 window_soft(const UnitSource& src, const frame::FrameSpec& spec, bool ook) {
    Units32 out;
    if (spec.n_win() == 0) return out;
    fsk_only(spec, ook);
    const UnitSums us = unit_sums(src, spec);
    out.rows = us.n_win;
    out.data.resize(out.rows * CALL_N);
    const double r2 = std::sqrt(2.0);
    std::array<double, UNITS> p0{}, p1{}, s2{};
    for (std::size_t w = 0; w < us.n_win; ++w) {
        for (int u = 0; u < UNITS; ++u) {
            const std::size_t k = w * UNITS + static_cast<std::size_t>(u);
            p0[static_cast<std::size_t>(u)] = power(us.x0[k]);
            p1[static_cast<std::size_t>(u)] = power(us.x1[k]);
            s2[static_cast<std::size_t>(u)] = us.s2[k];
        }
        const double E = ook ? plain_energy(p0.data(), s2.data()) : 0.0;
        for (int i = 0; i < CALL_N; ++i) {
            const auto u = static_cast<std::size_t>(CALL_LO + i);
            const double soft = !ook ? (p1[u] - p0[u]) / (r2 * s2[u]) : ((p0[u] - s2[u]) - E / 2) / s2[u];
            out.data[w * CALL_N + static_cast<std::size_t>(i)] = static_cast<float>(soft);
        }
    }
    return out;
}

double ln_i0(double x) { return std::log(filters::i0e(x)) + x; }

Units32 window_llr(const UnitSource& src, const frame::FrameSpec& spec, bool ook) {
    Units32 out;
    if (spec.n_win() == 0) return out;
    fsk_only(spec, ook);
    const UnitSums us = unit_sums(src, spec);
    out.rows = us.n_win;
    out.data.resize(out.rows * CALL_N);
    const bool is_l = spec.waveform() == frame::Waveform::L;
    std::array<double, UNITS> use{}, ones{};
    for (int u = 0; u < UNITS; ++u) {
        use[static_cast<std::size_t>(u)] = (u >= CALL_LO && u < CALL_LO + CALL_N) ? 1.0 : 0.0;
        ones[static_cast<std::size_t>(u)] = 1.0;
    }
    std::array<double, UNITS> cnt = smooth(is_l ? use : ones);
    if (is_l)
        for (double& c : cnt) c = np_maximum(c, 1.0);
    std::array<double, UNITS> p0{}, p1{}, s2{}, e{};
    for (std::size_t w = 0; w < us.n_win; ++w) {
        for (int u = 0; u < UNITS; ++u) {
            const std::size_t k = w * UNITS + static_cast<std::size_t>(u);
            p0[static_cast<std::size_t>(u)] = power(us.x0[k]);
            p1[static_cast<std::size_t>(u)] = power(us.x1[k]);
            s2[static_cast<std::size_t>(u)] = us.s2[k];
        }
        if (!ook) {
            for (std::size_t u = 0; u < UNITS; ++u) {
                e[u] = (p0[u] + p1[u]) - 2 * s2[u];
                if (is_l) e[u] = e[u] * use[u];
            }
            const std::array<double, UNITS> num = smooth(e);
            for (int i = 0; i < CALL_N; ++i) {
                const auto u = static_cast<std::size_t>(CALL_LO + i);
                const double a = std::sqrt(np_maximum(num[u] / cnt[u], 0.0));
                const double llr = ln_i0(2 * a * std::sqrt(p1[u]) / s2[u]) - ln_i0(2 * a * std::sqrt(p0[u]) / s2[u]);
                out.data[w * CALL_N + static_cast<std::size_t>(i)] = static_cast<float>(llr);
            }
        } else {
            const double a = std::sqrt(plain_energy(p0.data(), s2.data()));
            for (int i = 0; i < CALL_N; ++i) {
                const auto u = static_cast<std::size_t>(CALL_LO + i);
                const double llr = ln_i0(2 * a * std::sqrt(p0[u]) / s2[u]) - a * a / s2[u];
                out.data[w * CALL_N + static_cast<std::size_t>(i)] = static_cast<float>(llr);
            }
        }
    }
    return out;
}

// --- the reader -----------------------------------------------------------------

Combined combine(std::span<const double> values, std::size_t rows, std::size_t cols, bool one_d) {
    Combined c;
    if (one_d) {
        c.s.assign(values.begin(), values.end());
        c.nrow = 1;
        return c;
    }
    if (values.size() != rows * cols) throw std::invalid_argument("combine: values must hold rows x cols");
    c.s.assign(cols, 0.0);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j) c.s[j] = c.s[j] + values[i * cols + j];
    c.nrow = rows;
    return c;
}

std::optional<std::string> decode_units(std::span<const std::uint8_t> hard) {
    std::size_t last = hard.size();
    for (std::size_t i = hard.size(); i-- > 0;)
        if (hard[i]) {
            last = i;
            break;
        }
    if (last == hard.size()) return std::nullopt;
    std::vector<std::pair<bool, std::size_t>> runs;
    std::size_t i = 0;
    const std::size_t n = last + 1;
    while (i < n) {
        std::size_t j = i;
        const bool val = hard[i] != 0;
        while (j < n && (hard[j] != 0) == val) ++j;
        runs.emplace_back(val, j - i);
        i = j;
    }
    std::size_t r0 = runs[0].first ? 0 : 1;   // tolerate a late first element
    std::vector<std::string> chars;
    std::string cur;
    for (std::size_t r = r0; r < runs.size(); ++r) {
        const auto [val, len] = runs[r];
        if (val) {
            if (len <= 2) cur += '.';
            else if (len <= 5) cur += '-';
            else return std::nullopt;
        } else {
            if (len <= 2) continue;
            if (len <= 6) {
                chars.push_back(cur);
                cur.clear();
            } else {
                return std::nullopt;
            }
        }
    }
    chars.push_back(cur);
    std::string text;
    const auto& inv = inverse();
    for (const std::string& c : chars) {
        const auto it = inv.find(c);
        if (it == inv.end()) return std::nullopt;
        text += it->second;
    }
    try {
        return morse::check_callsign(text);
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    }
}

const std::vector<Pattern>& patterns() {
    static const std::vector<Pattern> pats = [] {
        std::vector<config::MorseCode> codes(config::MORSE.begin(), config::MORSE.end());
        std::sort(codes.begin(), codes.end(),
                  [](const config::MorseCode& a, const config::MorseCode& b) { return a.ch < b.ch; });
        std::vector<Pattern> out;
        for (const config::MorseCode& c : codes) {
            Pattern p{c.ch, {}};
            for (std::size_t j = 0; j < c.code.size(); ++j) {
                if (j) p.units.push_back(0);
                const int len = c.code[j] == '.' ? 1 : 3;
                for (int k = 0; k < len; ++k) p.units.push_back(1);
            }
            out.push_back(std::move(p));
        }
        return out;
    }();
    return pats;
}

Dp dp(const std::vector<std::vector<double>>& gains, std::span<const std::int64_t> lens, std::int64_t n,
      std::int64_t max_chars, std::optional<std::pair<std::int64_t, std::int64_t>> forbid) {
    const std::size_t n_chars = gains.size();
    if (n < 0) throw std::invalid_argument("dp: n must be at least 0");
    const auto n1 = static_cast<std::size_t>(n + 1);
    for (const std::vector<double>& g : gains)
        if (g.size() != n1) throw std::invalid_argument("dp: every gain row must hold n + 1 values");
    if (lens.size() != n_chars) throw std::invalid_argument("dp: one length per character");
    if (n_chars == 0) throw std::invalid_argument("need at least one array to stack");
    // G with the forbidden (character, start) excluded
    auto G = [&](std::size_t c, std::size_t e) {
        if (forbid && static_cast<std::size_t>(forbid->first) == c && static_cast<std::size_t>(forbid->second) == e)
            return -INF;
        return gains[c][e];
    };
    if (forbid) {
        auto& [fc, fe] = *forbid;
        const auto nc = static_cast<std::int64_t>(n_chars);
        const auto ne = static_cast<std::int64_t>(n1);
        if (fc < -nc || fc >= nc)
            throw std::out_of_range("index " + std::to_string(fc) + " is out of bounds for axis 0 with size " +
                                    std::to_string(nc));
        if (fe < -ne || fe >= ne)
            throw std::out_of_range("index " + std::to_string(fe) + " is out of bounds for axis 1 with size " +
                                    std::to_string(ne));
        if (fc < 0) fc += nc;
        if (fe < 0) fe += ne;
    }
    // characters of one length end together: keep the best of each length
    std::vector<std::int64_t> ulen(lens.begin(), lens.end());
    std::sort(ulen.begin(), ulen.end());
    ulen.erase(std::unique(ulen.begin(), ulen.end()), ulen.end());
    const std::size_t nu = ulen.size();
    std::vector<double> Gm(nu * n1, -INF);
    std::vector<std::int64_t> Ga(nu * n1, 0);
    for (std::size_t i = 0; i < nu; ++i) {
        std::vector<std::size_t> cs;
        for (std::size_t c = 0; c < n_chars; ++c)
            if (lens[c] == ulen[i]) cs.push_back(c);
        std::vector<double> col(cs.size());
        for (std::size_t e = 0; e < n1; ++e) {
            for (std::size_t q = 0; q < cs.size(); ++q) col[q] = G(cs[q], e);
            const std::size_t j = np_argmax(col.data(), col.size());
            Ga[i * n1 + e] = static_cast<std::int64_t>(cs[j]);
            Gm[i * n1 + e] = col[j];
        }
    }
    if (max_chars + 1 < 0) throw std::invalid_argument("negative dimensions are not allowed");
    if (max_chars + 1 <= 1)
        throw std::out_of_range("index 1 is out of bounds for axis 0 with size " + std::to_string(max_chars + 1));
    Dp d;
    d.kmax = static_cast<std::size_t>(max_chars + 1);
    d.n1 = n1;
    d.best.assign(d.kmax * n1, -INF);
    d.back.assign(d.kmax * n1 * 2, -1);
    auto best = [&](std::size_t k, std::size_t e) -> double& { return d.best[k * n1 + e]; };
    auto back = [&](std::size_t k, std::size_t e, int q) -> std::int64_t& { return d.back[(k * n1 + e) * 2 + static_cast<std::size_t>(q)]; };
    for (std::size_t i = 0; i < nu; ++i)
        if (ulen[i] <= n && ulen[i] >= 0) {
            const auto e = static_cast<std::size_t>(ulen[i]);
            best(1, e) = Gm[i * n1];
            back(1, e, 0) = Ga[i * n1];
            back(1, e, 1) = 0;
        }
    std::vector<std::size_t> fin;
    for (std::size_t k = 1; k + 1 < d.kmax; ++k) {
        fin.clear();
        for (std::size_t e = 0; e < n1; ++e)
            if (std::isfinite(best(k, e))) fin.push_back(e);
        for (const std::size_t e : fin) {
            const auto st = static_cast<std::int64_t>(e) + 3;
            if (st >= n) continue;
            for (std::size_t i = 0; i < nu; ++i) {
                const std::int64_t end = st + ulen[i];
                if (!(end <= n)) continue;
                const double v = best(k, e) + Gm[i * n1 + static_cast<std::size_t>(st)];
                double& b = best(k + 1, static_cast<std::size_t>(end));
                if (v > b) {
                    b = v;
                    back(k + 1, static_cast<std::size_t>(end), 0) = Ga[i * n1 + static_cast<std::size_t>(st)];
                    back(k + 1, static_cast<std::size_t>(end), 1) = static_cast<std::int64_t>(e);
                }
            }
        }
    }
    return d;
}

Trace trace(const Dp& d, std::span<const std::int64_t> lens) {
    Trace tr;
    if (d.best.empty()) throw std::invalid_argument("attempt to get argmax of an empty sequence");
    const std::size_t idx = np_argmax(d.best.data(), d.best.size());
    std::size_t k = idx / d.n1, e = idx % d.n1;
    if (!std::isfinite(d.best[idx])) {
        tr.score = -INF;
        return tr;
    }
    tr.score = d.best[idx];
    const std::vector<Pattern>& pats = patterns();
    std::string text;
    std::vector<std::pair<std::int64_t, std::int64_t>> items;
    while (k > 0) {
        const std::int64_t c = d.back[(k * d.n1 + e) * 2];
        const std::int64_t pe = d.back[(k * d.n1 + e) * 2 + 1];
        const auto nl = static_cast<std::int64_t>(lens.size());
        const std::int64_t ci = c < 0 ? c + nl : c;
        if (ci < 0 || ci >= nl) throw std::out_of_range("trace: no character recorded");
        items.emplace_back(c, static_cast<std::int64_t>(e) - lens[static_cast<std::size_t>(ci)]);
        k -= 1;
        e = static_cast<std::size_t>(pe);
    }
    std::reverse(items.begin(), items.end());
    for (const auto& [c, st] : items) {
        const auto np = static_cast<std::int64_t>(pats.size());
        const std::int64_t ci = c < 0 ? c + np : c;
        if (ci < 0 || ci >= np)
            throw std::out_of_range("list index out of range");
        text += pats[static_cast<std::size_t>(ci)].ch;
    }
    tr.text = text;
    tr.items = std::move(items);
    return tr;
}

ReadMl read_ml(const Combined& llr, std::int64_t max_chars, bool margin) {
    const std::vector<double>& s = llr.s;
    const auto n = static_cast<std::int64_t>(s.size());
    const double total = npsum::sum(std::span<const double>(s));
    const double norm = std::sqrt(static_cast<double>(static_cast<std::int64_t>(n) * static_cast<std::int64_t>(llr.nrow)));
    const std::vector<Pattern>& pats = patterns();
    const std::vector<std::int64_t> lens = pattern_lens();
    std::vector<std::vector<double>> gains;
    gains.reserve(pats.size());
    std::vector<double> tmp;
    for (const Pattern& pat : pats) {
        const auto L = static_cast<std::int64_t>(pat.units.size());
        std::vector<std::int64_t> on;
        for (std::int64_t j = 0; j < L; ++j)
            if (pat.units[static_cast<std::size_t>(j)]) on.push_back(j);
        std::vector<double> g(static_cast<std::size_t>(n + 1), -INF);
        if (L <= n) {
            tmp.resize(on.size());
            for (std::int64_t st = 0; st <= n - L; ++st) {
                for (std::size_t q = 0; q < on.size(); ++q) tmp[q] = s[static_cast<std::size_t>(st + on[q])];
                g[static_cast<std::size_t>(st)] = 2 * npsum::sum(std::span<const double>(tmp));
            }
        }
        gains.push_back(std::move(g));
    }
    const Trace t = trace(dp(gains, lens, n, max_chars), lens);
    ReadMl out;
    if (!t.text) return out;
    const double S = t.score - total;   // sum over on units minus sum over off units
    if (norm == 0.0) throw ZeroDivision("float division by zero");
    out.text = t.text;
    out.z = S / norm;
    if (!margin) return out;
    double m = t.score / 2;            // against every unit key-up
    for (const auto& it : t.items) {
        const Trace t2 = trace(dp(gains, lens, n, max_chars, it), lens);
        if (!t2.text) continue;
        const double c = (t.score - t2.score) / 2;
        if (c < m) m = c;
    }
    out.margin = m;
    return out;
}

std::optional<std::string> read(const Combined& llr) {
    const ReadMl r = read_ml(llr, 8, true);
    if (!r.text || r.margin < rxc::READ_LLR_MIN) return std::nullopt;
    return r.text;
}

double match(const Combined& soft, std::string_view callsign) {
    const std::vector<std::uint8_t> ku = morse::keying_units(callsign);
    std::array<double, CALL_N> k{};
    for (int i = 0; i < CALL_N; ++i) k[static_cast<std::size_t>(i)] = static_cast<double>(ku[static_cast<std::size_t>(CALL_LO + i)]) * 2 - 1;
    if (soft.s.size() != static_cast<std::size_t>(CALL_N))
        throw std::invalid_argument("shapes (" + std::to_string(CALL_N) + ",) and (" + std::to_string(soft.s.size()) +
                                    ",) not aligned: " + std::to_string(CALL_N) + " (dim 0) != " +
                                    std::to_string(soft.s.size()) + " (dim 0)");
    const double dot = blas_dot(k.data(), soft.s.data(), k.size());
    return dot / std::sqrt(static_cast<double>(static_cast<std::int64_t>(CALL_N) * static_cast<std::int64_t>(soft.nrow)));
}

std::string keying_type(const UnitSource& src, const frame::FrameSpec& spec, std::optional<std::string_view> callsign) {
    if (spec.n_win() == 0 || spec.waveform() == frame::Waveform::L) return "fsk";
    if (callsign) {
        auto as_combined = [](const Units32& u) {
            std::vector<double> v(u.data.begin(), u.data.end());
            return combine(v, u.rows, CALL_N, false);
        };
        const double zf = match(as_combined(window_soft(src, spec, false)), *callsign);
        const double zo = match(as_combined(window_soft(src, spec, true)), *callsign);
        return zf >= zo ? "fsk" : "ook";
    }
    const UnitSums us = unit_sums(src, spec);
    std::vector<double> e;
    e.reserve(us.n_win * CALL_N);
    for (std::size_t w = 0; w < us.n_win; ++w)
        for (int i = 0; i < CALL_N; ++i) {
            const std::size_t k = w * UNITS + static_cast<std::size_t>(CALL_LO + i);
            e.push_back((power(us.x1[k]) - us.s2[k]) / us.s2[k]);
        }
    const double z = npsum::sum(std::span<const double>(e)) / std::sqrt(static_cast<double>(e.size()));
    return z > 3.0 ? "fsk" : "ook";
}

}  // namespace sstvae::qrss::cwid
