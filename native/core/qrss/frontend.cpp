#include "qrss/frontend.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <list>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dsp/dsp.hpp"
#include "dsp/fft.hpp"   // pocketfft (np.fft.fft is numpy's pocketfft)
#include "math/convolve.hpp"
#include "math/npcomplex.hpp"
#include "math/npstats.hpp"
#include "qrss/filters.hpp"
#include "qrss/npsum.hpp"
#include "qrss/rules.hpp"

namespace sstvae::qrss::frontend {

namespace {

namespace np = sstvae::math::np;

constexpr double PI = 3.141592653589793;   // math.pi / np.pi

// --- Python integer and float helpers ------------------------------------------

std::int64_t floordiv(std::int64_t a, std::int64_t b) {
    if (b == 0) throw ZeroDivision("integer division or modulo by zero");
    std::int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

std::int64_t floormod(std::int64_t a, std::int64_t b) {
    if (b == 0) throw ZeroDivision("integer modulo by zero");
    std::int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}

// int(x) of a float that is already integral (round, floor, ceil): Python's
// refusals for NaN and inf.
std::int64_t int_of(double x) {
    if (std::isnan(x)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(x)) throw std::overflow_error("cannot convert float infinity to integer");
    if (!(x >= -9223372036854775808.0 && x < 9223372036854775808.0))
        throw std::overflow_error("Python int too large to convert to C int64");
    return static_cast<std::int64_t>(x);
}

// int(round(x)): half to even.
std::int64_t round_int(double x) { return int_of(std::nearbyint(x)); }

// np.remainder(a, b) for floats (npy_divmod's modulus).
double np_remainder(double a, double b) {
    double mod = std::fmod(a, b);
    if (b == 0.0) return mod;
    if (mod != 0.0) {
        if ((b < 0) != (mod < 0)) mod += b;
    } else {
        mod = std::copysign(0.0, b);
    }
    return mod;
}

// np.fft.fftfreq(n, d): k / (n*d) for k = 0..(n-1)//2, then -(n//2)..-1.
std::vector<double> fftfreq(std::int64_t n, double d) {
    std::vector<double> f(static_cast<std::size_t>(std::max<std::int64_t>(n, 0)));
    const double nd = static_cast<double>(n) * d;
    if (nd == 0.0) throw ZeroDivision("float division by zero");   // numpy's 1.0 / (n * d)
    if (n < 0) return f;
    const double val = 1.0 / nd;
    const std::int64_t N = (n - 1) / 2 + 1;
    for (std::int64_t i = 0; i < N; ++i) f[static_cast<std::size_t>(i)] = static_cast<double>(i) * val;
    for (std::int64_t i = N; i < n; ++i)
        f[static_cast<std::size_t>(i)] = static_cast<double>(i - n) * val;
    return f;
}

// A small thread-safe LRU cache of immutable values (functools.lru_cache).
// The value is computed outside the lock; two threads asking for the same
// new key both compute it (the same value) and the first insert wins.
template <typename Key, typename Value>
class Lru {
public:
    explicit Lru(std::size_t capacity) : capacity_(capacity) {}

    template <typename F>
    std::shared_ptr<const Value> get(const Key& key, F&& compute) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto it = items_.begin(); it != items_.end(); ++it) {
                if (it->first == key) {
                    items_.splice(items_.begin(), items_, it);
                    return items_.front().second;
                }
            }
        }
        auto value = std::make_shared<const Value>(compute());
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = items_.begin(); it != items_.end(); ++it)
            if (it->first == key) return it->second;
        items_.emplace_front(key, value);
        if (items_.size() > capacity_) items_.pop_back();
        return value;
    }

private:
    std::size_t capacity_;
    std::mutex mutex_;
    std::list<std::pair<Key, std::shared_ptr<const Value>>> items_;
};

// Doubles compared by bit pattern (a cache key; equal values give equal
// results, so keying on bits only ever costs a recomputation).
std::uint64_t bits(double x) { return std::bit_cast<std::uint64_t>(x); }

using RateKey = std::tuple<std::int64_t, std::int64_t, std::int64_t>;
using BandKey = std::tuple<std::int64_t, std::int64_t, std::int64_t, std::uint64_t, std::uint64_t>;
using PowerKey = std::tuple<std::int64_t, std::int64_t, std::int64_t, std::uint64_t, std::uint64_t, std::uint64_t>;

// The filter resample_poly(., 1, decim, window=("kaiser", 10)) designs.
std::vector<double> decimator_taps(std::int64_t decim) {
    if (decim == 0) throw ZeroDivision("float division by zero");   // the reference's 1.0 / decim
    return filters::firwin(2 * 10 * decim + 1, 1.0 / static_cast<double>(decim), filters::Window::kaiser(10.0));
}

// np.abs(freqz(h, worN=f, fs=fs)[1]) ** 2
std::vector<double> power_response(std::span<const double> h, std::span<const double> f, double fs) {
    const std::vector<cd> H = math::freqz(h, f, fs);
    std::vector<double> out(H.size());
    for (std::size_t i = 0; i < H.size(); ++i) {
        const double a = np::abs(H[i]);
        out[i] = a * a;
    }
    return out;
}

std::vector<std::uint8_t> band_mask(const std::vector<double>& f, Band band) {
    std::vector<std::uint8_t> m(f.size());
    for (std::size_t i = 0; i < f.size(); ++i) {
        const double a = std::fabs(f[i]);
        m[i] = (a >= band.first) && (a <= band.second);
    }
    return m;
}

FoldPower compute_fold_power(std::int64_t fs_in, std::int64_t fs_out, std::int64_t nfft, Band band) {
    const std::int64_t decim = floordiv(fs_in, fs_out);
    const std::vector<double> f = fftfreq(nfft, 1.0 / static_cast<double>(fs_out));
    FoldPower out;
    const auto m = band_mask(f, band);
    for (std::size_t i = 0; i < m.size(); ++i)
        if (m[i]) out.idx.push_back(static_cast<std::int64_t>(i));
    const std::size_t ni = out.idx.size();
    if (decim == 1) {
        out.n_fold = 1;
        for (const std::int64_t i : out.idx) out.fk.push_back(f[static_cast<std::size_t>(i)]);
        out.P.assign(ni, 1.0);
        return out;
    }
    const std::vector<double> h = decimator_taps(decim);
    const std::int64_t k_lo = -floordiv(decim, 2), k_hi = decim - floordiv(decim, 2);
    out.n_fold = static_cast<std::size_t>(std::max<std::int64_t>(k_hi - k_lo, 0));
    for (std::int64_t k = k_lo; k < k_hi; ++k) {
        std::vector<double> row(ni);
        const auto shift = static_cast<double>(k * fs_out);
        for (std::size_t j = 0; j < ni; ++j) row[j] = f[static_cast<std::size_t>(out.idx[j])] + shift;
        const std::vector<double> p = power_response(h, row, static_cast<double>(fs_in));
        out.fk.insert(out.fk.end(), row.begin(), row.end());
        out.P.insert(out.P.end(), p.begin(), p.end());
    }
    return out;
}

std::vector<double> compute_noise_response(std::int64_t fs_in, std::int64_t fs_out, std::int64_t nfft) {
    const std::int64_t decim = floordiv(fs_in, fs_out);
    if (decim == 1) return std::vector<double>(static_cast<std::size_t>(std::max<std::int64_t>(nfft, 0)), 1.0);
    const std::vector<double> h = decimator_taps(decim);
    const std::vector<double> f = fftfreq(nfft, 1.0 / static_cast<double>(fs_out));
    std::vector<double> r(f.size(), 0.0);
    std::vector<double> row(f.size());
    for (std::int64_t k = -floordiv(decim, 2); k < decim - floordiv(decim, 2); ++k) {
        const auto shift = static_cast<double>(k * fs_out);
        for (std::size_t j = 0; j < f.size(); ++j) row[j] = f[j] + shift;
        const std::vector<double> p = power_response(h, row, static_cast<double>(fs_in));
        for (std::size_t j = 0; j < r.size(); ++j) r[j] = r[j] + p[j];
    }
    return r;
}

FePower compute_fe_power(std::int64_t fs_in, std::int64_t fs_out, std::int64_t nfft, double audio_hz, Band band) {
    const auto fp = fold_power(fs_in, fs_out, nfft, band);
    const std::size_t ni = fp->idx.size(), nk = fp->n_fold;
    // (audio_hz + fk - FE_CENTER_HZ) % FS, raveled row by row
    std::vector<double> worn(nk * ni);
    for (std::size_t i = 0; i < worn.size(); ++i)
        worn[i] = np_remainder((audio_hz + fp->fk[i]) - static_cast<double>(config::FE_CENTER_HZ),
                               static_cast<double>(config::FS));
    const std::vector<double> G2 = power_response(fe_taps(), worn, static_cast<double>(config::FS));
    FePower out;
    out.idx = fp->idx;
    out.num.assign(ni, 0.0);
    out.den.assign(ni, 0.0);
    // np.sum(., axis=0) over the folds: row after row
    for (std::size_t k = 0; k < nk; ++k) {
        for (std::size_t j = 0; j < ni; ++j) {
            const double P = fp->P[k * ni + j];
            const double t = P * G2[k * ni + j];
            out.num[j] = k == 0 ? t : out.num[j] + t;
            out.den[j] = k == 0 ? P : out.den[j] + P;
        }
    }
    return out;
}

Lru<RateKey, std::vector<double>>& noise_response_cache() {
    static Lru<RateKey, std::vector<double>> c(8);
    return c;
}
Lru<BandKey, FoldPower>& fold_power_cache() {
    static Lru<BandKey, FoldPower> c(8);
    return c;
}
Lru<PowerKey, FePower>& fe_power_cache() {
    static Lru<PowerKey, FePower> c(64);
    return c;
}

// numpy's float32 |z|^2 as frontend writes it: re**2 + im**2 in float32.
inline float power32(cf z) {
    const float re = z.real(), im = z.imag();
    return re * re + im * im;
}

// x[i] for i in 0..n-1 through np.interp, chunk by chunk (1 << 20).
std::vector<float> interp_chunks(std::int64_t n, std::span<const double> xp, std::span<const double> fp) {
    std::vector<float> out(static_cast<std::size_t>(n));
    constexpr std::int64_t CHUNK = 1 << 20;
    std::vector<double> x;
    for (std::int64_t a = 0; a < n; a += CHUNK) {
        const std::int64_t b = std::min(n, a + CHUNK);
        x.resize(static_cast<std::size_t>(b - a));
        for (std::int64_t i = a; i < b; ++i) x[static_cast<std::size_t>(i - a)] = static_cast<double>(i);
        const std::vector<double> y = math::interp(std::span<const double>(x), xp, fp);
        for (std::size_t i = 0; i < y.size(); ++i) out[static_cast<std::size_t>(a) + i] = static_cast<float>(y[i]);
    }
    return out;
}

// For each sample, the distance to the nearest dead one (n + 1 when none).
std::vector<std::int64_t> distance_to(const std::vector<std::uint8_t>& dead) {
    const auto n = static_cast<std::int64_t>(dead.size());
    std::vector<std::int64_t> d(dead.size(), n + 1);
    std::int64_t last = -1;
    for (std::int64_t i = 0; i < n; ++i) {
        if (dead[static_cast<std::size_t>(i)]) last = i;
        if (last >= 0) d[static_cast<std::size_t>(i)] = i - last;
    }
    last = -1;
    for (std::int64_t i = n - 1; i >= 0; --i) {
        if (dead[static_cast<std::size_t>(i)]) last = i;
        if (last >= 0) d[static_cast<std::size_t>(i)] = std::min(d[static_cast<std::size_t>(i)], last - i);
    }
    return d;
}

// Python's min() over the window medians: the first unless a later one is
// smaller.
double py_min(const std::vector<double>& v) {
    double r = v.front();
    for (std::size_t i = 1; i < v.size(); ++i)
        if (v[i] < r) r = v[i];
    return r;
}

double median_of(std::vector<double>&& v) { return math::median(std::span<const double>(v)); }

constexpr double LN2 = 0.6931471805599453;   // math.log(2)

}  // namespace

const std::vector<double>& fe_taps() {
    static const std::vector<double> taps = filters::firwin(255, 1250.0, filters::Window::kaiser(8.0),
                                                            filters::PassZero::Yes, true,
                                                            static_cast<double>(config::FS));
    return taps;
}

// --- audio <-> FE ----------------------------------------------------------------

std::vector<cf> audio_to_fe(std::span<const double> x8k) {
    static_assert(config::FE_CENTER_HZ == 1500, "to_baseband's FCENTER is the FE centre");
    const double s = std::sqrt(2.0);
    std::vector<cd> z = dsp::to_baseband(x8k);
    for (cd& v : z) v = np::mul(v, s);
    const std::vector<cd> y = math::oaconvolve(std::span<const cd>(z), std::span<const double>(fe_taps()),
                                               math::ConvMode::Same);
    std::vector<cf> out;
    out.reserve((y.size() + FE_DECIM - 1) / FE_DECIM);
    for (std::size_t i = 0; i < y.size(); i += FE_DECIM)
        out.emplace_back(static_cast<float>(y[i].real()), static_cast<float>(y[i].imag()));
    return out;
}

std::vector<double> fe_to_audio(std::span<const cd> fe) {
    std::vector<cd> up(fe.size() * FE_DECIM, cd(0.0, 0.0));
    for (std::size_t i = 0; i < fe.size(); ++i) up[i * FE_DECIM] = np::mul(fe[i], static_cast<double>(FE_DECIM));
    const std::vector<cd> y = math::oaconvolve(std::span<const cd>(up), std::span<const double>(fe_taps()),
                                               math::ConvMode::Same);
    // mix up: the conjugate of to_baseband's exact 16-phasor table
    const std::vector<double> ones(y.size(), 1.0);
    const std::vector<cd> het = dsp::to_baseband(ones);
    const double s = std::sqrt(2.0);
    std::vector<double> out(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) out[i] = s * np::mul(std::conj(het[i]), y[i]).real();
    return out;
}

// --- blanker and inverse AGC -------------------------------------------------------

RunningMedian running_median(std::span<const float> p, double fs, double win_s, double hop_s) {
    const auto n = static_cast<std::int64_t>(p.size());
    const std::int64_t hop = std::max<std::int64_t>(1, round_int(hop_s * fs));
    const std::int64_t w = std::max<std::int64_t>(1, round_int(win_s * fs));
    RunningMedian out;
    for (std::int64_t c = floordiv(hop, 2); c < std::max<std::int64_t>(n, 1); c += hop) out.centres.push_back(c);
    if (n <= w) {
        const float m = n ? math::median(p) : 0.0f;
        out.medians.assign(out.centres.size(), m);
        return out;
    }
    out.medians.resize(out.centres.size());
    for (std::size_t i = 0; i < out.centres.size(); ++i) {
        const std::int64_t start = std::clamp<std::int64_t>(out.centres[i] - floordiv(w, 2), 0, n - w);
        out.medians[i] = math::median(p.subspan(static_cast<std::size_t>(start), static_cast<std::size_t>(w)));
    }
    return out;
}

std::vector<float> interp_to_samples(std::int64_t n, std::span<const double> centres, std::span<const double> vals) {
    // np.empty(n) first; np.interp (and its refusals) only for n > 0
    if (n < 0) throw std::invalid_argument("negative dimensions are not allowed");
    return interp_chunks(n, centres, vals);
}

Blanked blank(std::span<const cf> fe, double k_sigma, double guard_ms, double win_s, double hop_s, double fs,
              double taper_ms) {
    const std::size_t n = fe.size();
    Blanked out;
    out.keep.assign(n, 1.0f);
    if (n == 0) return out;
    std::vector<float> p(n);
    for (std::size_t i = 0; i < n; ++i) p[i] = power32(fe[i]);
    const RunningMedian rm = running_median(p, fs, win_s, hop_s);
    const std::vector<double> cen(rm.centres.begin(), rm.centres.end());
    const std::vector<double> med(rm.medians.begin(), rm.medians.end());
    const std::vector<float> nu = interp_to_samples(static_cast<std::int64_t>(n), cen, med);
    const auto thr = static_cast<float>(k_sigma * k_sigma);
    std::vector<std::uint8_t> bad(n);
    bool any = false;
    for (std::size_t i = 0; i < n; ++i) {
        bad[i] = p[i] > thr * nu[i];
        any = any || bad[i];
    }
    if (!any) {
        out.fe.assign(fe.begin(), fe.end());
        return out;
    }
    const std::int64_t g = round_int(guard_ms * 1e-3 * fs);
    const std::vector<std::uint8_t> dead = math::maximum_filter1d(bad, 2 * g + 1);
    const std::int64_t L = round_int(taper_ms * 1e-3 * fs);
    if (L >= 1) {
        // the reference's loop from k = L down to 1 (nearer samples overwrite
        // farther ones) leaves each sample within L of a dead one at its
        // distance's value: a max filter of width 2k+1 in mode 'reflect' is
        // the window clipped to the stream
        const std::vector<std::int64_t> d = distance_to(dead);
        std::vector<float> val(static_cast<std::size_t>(L) + 1);
        for (std::int64_t k = 1; k <= L; ++k)
            val[static_cast<std::size_t>(k)] = static_cast<float>(
                0.5 * (1 - std::cos(PI * static_cast<double>(k) / static_cast<double>(L + 1))));
        for (std::size_t i = 0; i < n; ++i)
            if (d[i] >= 1 && d[i] <= L) out.keep[i] = val[static_cast<std::size_t>(d[i])];
    }
    for (std::size_t i = 0; i < n; ++i)
        if (dead[i]) out.keep[i] = 0.0f;
    out.fe.resize(n);
    for (std::size_t i = 0; i < n; ++i) out.fe[i] = np::mul(fe[i], out.keep[i]);
    return out;
}

Normalised normalise(std::span<const cf> fe, std::optional<std::span<const float>> keep, double win_ms, double fs,
                     double hop_ms, double min_kept) {
    const auto n = static_cast<std::int64_t>(fe.size());
    Normalised out;
    if (n == 0) {
        return out;
    }
    if (keep && static_cast<std::int64_t>(keep->size()) != n)   // numpy's refusal of p[keep < 1]
        throw std::out_of_range("boolean index did not match indexed array along axis 0; size of axis is " +
                                std::to_string(n) + " but size of corresponding boolean axis is " +
                                std::to_string(keep->size()));
    const std::int64_t w = std::min(n, std::max<std::int64_t>(1, round_int(win_ms * 1e-3 * fs)));
    const std::int64_t hop = std::max<std::int64_t>(1, round_int(hop_ms * 1e-3 * fs));
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> p(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < p.size(); ++i) {
        p[i] = power32(fe[i]);
        if (keep && (*keep)[i] < 1.0f) p[i] = nan;
    }
    std::vector<double> ok_centres, ok_med;
    std::vector<float> buf;
    buf.reserve(static_cast<std::size_t>(w));
    const std::int64_t need = std::max<std::int64_t>(1, int_of(std::trunc(min_kept * static_cast<double>(w))));
    const float inf = std::numeric_limits<float>::infinity();
    for (std::int64_t c = floordiv(hop, 2); c < n; c += hop) {
        const std::int64_t start = std::clamp<std::int64_t>(c - floordiv(w, 2), 0, n - w);
        buf.clear();
        for (std::int64_t j = start; j < start + w; ++j) {
            const float v = p[static_cast<std::size_t>(j)];
            if (!std::isnan(v)) buf.push_back(v);
        }
        const auto cnt = static_cast<std::int64_t>(buf.size());
        float med;
        if (cnt == 0) {
            med = 0.5f * (inf + inf);
        } else {
            // the middle of the kept count, as the reference's sort with the
            // unkept ones at +inf picks it
            const std::int64_t c_ = cnt;
            const auto i_lo = static_cast<std::size_t>((c_ - 1) / 2), i_hi = static_cast<std::size_t>(c_ / 2);
            std::nth_element(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(i_hi), buf.end());
            const float hi = buf[i_hi];
            const float lo = i_lo == i_hi ? hi
                                          : *std::max_element(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(i_hi));
            med = 0.5f * (lo + hi);
        }
        if (cnt >= need) {
            ok_centres.push_back(static_cast<double>(c));
            // np.maximum(med, float32(1e-30)): NaN propagates
            const float floor_ = 1e-30f;
            ok_med.push_back(static_cast<double>(std::isnan(med) ? med : (med > floor_ ? med : floor_)));
        }
    }
    if (ok_centres.empty()) {
        out.level.assign(static_cast<std::size_t>(n), 1.0f);
    } else {
        out.level = interp_chunks(n, ok_centres, ok_med);
    }
    out.fe.resize(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < out.fe.size(); ++i) out.fe[i] = np::div(fe[i], std::sqrt(out.level[i]));
    return out;
}

// --- channeliser -----------------------------------------------------------------------

Rational quantise_mix(double f_mix_hz) {
    return Rational(round_int(f_mix_hz * F_MIX_DEN), F_MIX_DEN);
}

namespace {

// mix_phasor's one period: exp(-j 2 pi ((i k) mod D)/D), i < P.
struct MixTable {
    std::vector<cd> tab;
    std::int64_t shift;   // n0 mod P
};

MixTable mix_table(std::int64_t n0, const Rational& f_mix, std::int64_t fs_in) {
    // num = int(f_mix * 64), truncated toward zero
    const Rational scaled = f_mix * Rational(F_MIX_DEN, 1);
    const std::int64_t num = scaled.num() / scaled.den();
    if (!(Rational(num, F_MIX_DEN) == f_mix))
        throw std::invalid_argument("f_mix " + f_mix.str() + " is not on the 1/" + std::to_string(F_MIX_DEN) +
                                    " Hz grid");
    const std::int64_t D = F_MIX_DEN * fs_in;
    const std::int64_t k = floormod(num, D);
    const std::int64_t P = floordiv(D, std::gcd(k, D));
    MixTable t;
    if (P <= 0) {   // a negative fs_in: numpy's arange(P) is empty
        t.shift = 0;
        return t;
    }
    t.tab.resize(static_cast<std::size_t>(P));
    const double Dd = static_cast<double>(D);
    for (std::int64_t i = 0; i < P; ++i) {
        // (i * k) % D: both below 64 * fs_in, so the product fits int64 for
        // any fs_in below 2^25
        const double turns = static_cast<double>(floormod(i * k, D)) / Dd;
        const double ang = (2 * PI) * turns;
        t.tab[static_cast<std::size_t>(i)] = cd(std::cos(ang), -std::sin(ang));
    }
    t.shift = floormod(n0, P);
    return t;
}

}  // namespace

std::vector<cd> mix_phasor(std::int64_t n0, std::int64_t n, const Rational& f_mix, std::int64_t fs_in) {
    const MixTable t = mix_table(n0, f_mix, fs_in);
    const auto P = static_cast<std::int64_t>(t.tab.size());
    // np.tile(tab, -(-n // P))[:n]: empty for -P < n <= 0 and refused below
    // that (a negative repeat count); an empty period (negative fs_in) gives
    // nothing to tile, which a caller asking for samples cannot use
    if (n <= 0 || P == 0) {
        if (n > 0 || (n < 0 && (P == 0 || n <= -P)))
            throw std::invalid_argument("negative dimensions are not allowed");
        return {};
    }
    std::vector<cd> out(static_cast<std::size_t>(n));
    std::int64_t j = t.shift;
    for (std::int64_t i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = t.tab[static_cast<std::size_t>(j)];
        if (++j == P) j = 0;
    }
    return out;
}

Channelised channelise(std::span<const cd> fe, std::int64_t fs_in, double f_mix_hz, std::int64_t fs_out,
                       std::int64_t n0) {
    if (floormod(fs_in, fs_out) != 0)
        throw std::invalid_argument("fs_out " + std::to_string(fs_out) + " must divide fs_in " + std::to_string(fs_in));
    Channelised out;
    out.f_mix = quantise_mix(f_mix_hz);
    const MixTable t = mix_table(n0, out.f_mix, fs_in);
    const auto P = static_cast<std::int64_t>(t.tab.size());
    if (P == 0 && !fe.empty()) throw std::invalid_argument("negative dimensions are not allowed");
    std::vector<cd> y(fe.size());
    std::int64_t j = t.shift;
    for (std::size_t i = 0; i < fe.size(); ++i) {
        y[i] = np::mul(fe[i], t.tab[static_cast<std::size_t>(j)]);
        if (++j == P) j = 0;
    }
    const std::int64_t decim = floordiv(fs_in, fs_out);
    if (decim > 1) y = filters::resample_poly<cd>(y, 1, decim, filters::Window::kaiser(10.0));
    out.ch.resize(y.size());
    for (std::size_t i = 0; i < y.size(); ++i)
        out.ch[i] = cf(static_cast<float>(y[i].real()), static_cast<float>(y[i].imag()));
    return out;
}

Channelised channelise(std::span<const cf> fe, std::int64_t fs_in, double f_mix_hz, std::int64_t fs_out,
                       std::int64_t n0) {
    const std::vector<cd> wide(fe.begin(), fe.end());
    return channelise(std::span<const cd>(wide), fs_in, f_mix_hz, fs_out, n0);
}

std::vector<float> channelise_keep(std::span<const double> keep, std::int64_t fs_in, std::int64_t fs_out) {
    const std::int64_t decim = floordiv(fs_in, fs_out);
    std::vector<double> k(keep.begin(), keep.end());
    if (decim > 1) k = filters::resample_poly<double>(k, 1, decim, filters::Window::kaiser(10.0));
    return std::vector<float>(k.begin(), k.end());
}

std::vector<float> channelise_keep(std::span<const float> keep, std::int64_t fs_in, std::int64_t fs_out) {
    const std::vector<double> k(keep.begin(), keep.end());
    return channelise_keep(std::span<const double>(k), fs_in, fs_out);
}

// --- the noise floor ---------------------------------------------------------------------

std::shared_ptr<const std::vector<double>> noise_response(std::int64_t fs_in, std::int64_t fs_out, std::int64_t nfft) {
    return noise_response_cache().get(RateKey{fs_in, fs_out, nfft},
                                      [&] { return compute_noise_response(fs_in, fs_out, nfft); });
}

std::shared_ptr<const FoldPower> fold_power(std::int64_t fs_in, std::int64_t fs_out, std::int64_t nfft, Band band_hz) {
    return fold_power_cache().get(BandKey{fs_in, fs_out, nfft, bits(band_hz.first), bits(band_hz.second)},
                                  [&] { return compute_fold_power(fs_in, fs_out, nfft, band_hz); });
}

std::shared_ptr<const FePower> fe_power(std::int64_t fs_in, std::int64_t fs_out, std::int64_t nfft, double audio_hz,
                                        Band band_hz) {
    return fe_power_cache().get(
        PowerKey{fs_in, fs_out, nfft, bits(audio_hz), bits(band_hz.first), bits(band_hz.second)},
        [&] { return compute_fe_power(fs_in, fs_out, nfft, audio_hz, band_hz); });
}

std::vector<std::uint8_t> fe_flat(std::int64_t fs_in, std::int64_t fs_out, std::int64_t nfft, double audio_hz,
                                  Band band_hz) {
    const auto fp = fe_power(fs_in, fs_out, nfft, audio_hz, band_hz);
    std::vector<std::uint8_t> flat(static_cast<std::size_t>(std::max<std::int64_t>(nfft, 0)), 0);
    const double thr = std::pow(10.0, -FE_FLAT_DB / 10.0);
    for (std::size_t j = 0; j < fp->idx.size(); ++j)
        flat[static_cast<std::size_t>(fp->idx[j])] = fp->num[j] >= thr * fp->den[j];
    return flat;
}

std::vector<double> fe_response(std::int64_t fs_in, std::int64_t fs_out, std::int64_t nfft, double audio_hz,
                                Band band_hz) {
    const auto fp = fe_power(fs_in, fs_out, nfft, audio_hz, band_hz);
    std::vector<double> r(static_cast<std::size_t>(std::max<std::int64_t>(nfft, 0)), 1.0);
    for (std::size_t j = 0; j < fp->idx.size(); ++j) r[static_cast<std::size_t>(fp->idx[j])] = fp->num[j] / fp->den[j];
    return r;
}

std::vector<float> noise_psd(std::span<const cf> ch, std::int64_t fs, std::span<const double> exclude_hz, double win_s,
                             const NoisePsdOptions& opt) {
    std::int64_t w = std::max<std::int64_t>(16, round_int(win_s * static_cast<double>(fs)));
    const auto n = static_cast<std::int64_t>(ch.size());
    if (n < w) w = n;
    const std::int64_t nwin = std::max<std::int64_t>(1, -floordiv(-n, w));
    const std::vector<double> f = fftfreq(w, 1.0 / static_cast<double>(fs));
    const auto W = static_cast<std::size_t>(w);
    std::vector<std::uint8_t> band = band_mask(f, opt.band_hz);
    std::vector<std::uint8_t> clear(W, 1);
    for (const double fx : exclude_hz)
        for (const std::int64_t alias : {-fs, std::int64_t{0}, fs})
            for (std::size_t j = 0; j < W; ++j)
                clear[j] = clear[j] && (std::fabs(f[j] - fx - static_cast<double>(alias)) > opt.guard_hz);
    // the fallback's skirt bins and the low-pass's response there
    std::optional<std::pair<std::vector<std::size_t>, std::vector<double>>> skirt;
    const std::int64_t fs_in = opt.fs_in;
    if (opt.audio_hz && !opt.old_edge_noise.value_or(rules::old(rules::EDGE_NOISE)) && floormod(fs_in, fs) == 0) {
        const std::vector<std::uint8_t> ff = fe_flat(fs_in, fs, w, *opt.audio_hz, opt.band_hz);
        std::vector<std::uint8_t> flat(W);
        bool flat_any = false, same = true;
        for (std::size_t j = 0; j < W; ++j) {
            flat[j] = band[j] && ff[j];
            flat_any = flat_any || flat[j];
            same = same && (flat[j] == band[j]);
        }
        if (flat_any && !same) {
            std::int64_t clear_flat = 0;
            for (std::size_t j = 0; j < W; ++j) clear_flat += (flat[j] && clear[j]);
            if (static_cast<double>(clear_flat) < FE_MIN_HZ * static_cast<double>(w) / static_cast<double>(fs)) {
                const std::vector<double> r = fe_response(fs_in, fs, w, *opt.audio_hz, opt.band_hz);
                const double floor_ = std::pow(10.0, -FE_SKIRT_DB / 10.0);
                std::vector<std::size_t> sk;
                std::vector<double> rs;
                for (std::size_t j = 0; j < W; ++j) {
                    if (band[j] && !flat[j] && clear[j] && r[j] >= floor_) {
                        sk.push_back(j);
                        rs.push_back(r[j]);
                    }
                }
                if (!sk.empty()) skirt.emplace(std::move(sk), std::move(rs));
            }
            band = flat;
        }
    }
    std::vector<std::uint8_t> sel(W);
    bool sel_any = false;
    for (std::size_t j = 0; j < W; ++j) {
        sel[j] = band[j] && clear[j];
        sel_any = sel_any || sel[j];
    }
    std::vector<std::vector<std::uint8_t>> sides;
    if (sel_any && !skirt) {
        sides.push_back(sel);
    } else {
        std::vector<std::uint8_t> lo(W), hi(W);
        for (std::size_t j = 0; j < W; ++j) {
            lo[j] = band[j] && f[j] < 0;
            hi[j] = band[j] && f[j] > 0;
        }
        sides.push_back(std::move(lo));
        sides.push_back(std::move(hi));
    }
    std::vector<std::vector<std::size_t>> side_idx;
    for (const auto& s : sides) {
        std::vector<std::size_t> idx;
        for (std::size_t j = 0; j < W; ++j)
            if (s[j]) idx.push_back(j);
        side_idx.push_back(std::move(idx));
    }
    bool any_side = false;
    for (const auto& s : side_idx) any_side = any_side || !s.empty();
    if (!any_side) throw std::invalid_argument("noise_psd: no bin in the band");
    std::shared_ptr<const std::vector<double>> resp;
    if (floormod(fs_in, fs) == 0) {
        resp = noise_response(fs_in, fs, w);
    } else {
        resp = std::make_shared<const std::vector<double>>(W, 1.0);
    }
    const std::vector<double> win = filters::hann(w, false);
    std::vector<double> win2(W);
    for (std::size_t j = 0; j < W; ++j) win2[j] = win[j] * win[j];
    const double scale = 1.0 / (static_cast<double>(fs) * npsum::sum(std::span<const double>(win2)));
    std::vector<std::size_t> band_idx;
    if (skirt)
        for (std::size_t j = 0; j < W; ++j)
            if (band[j]) band_idx.push_back(j);

    std::vector<float> out(static_cast<std::size_t>(nwin));
    std::vector<cd> buf(W), X(W);
    std::vector<double> p(W);
    const pocketfft::shape_t shape{W};
    const pocketfft::stride_t stride{static_cast<std::ptrdiff_t>(sizeof(cd))};
    for (std::int64_t i = 0; i < nwin; ++i) {
        const std::int64_t a = std::min(i * w, n - w);
        for (std::size_t j = 0; j < W; ++j) {
            const cf v = ch[static_cast<std::size_t>(a) + j];
            buf[j] = np::mul(cd(v.real(), v.imag()), win[j]);
        }
        pocketfft::c2c(shape, stride, stride, pocketfft::shape_t{0}, true, buf.data(), X.data(), 1.0, 1);
        for (std::size_t j = 0; j < W; ++j) {
            const double re = X[j].real(), im = X[j].imag();
            p[j] = (re * re + im * im) * scale / (*resp)[j];
        }
        double v;
        if (!skirt) {
            std::vector<double> meds;
            for (const auto& idx : side_idx) {
                if (idx.empty()) continue;
                std::vector<double> vals;
                vals.reserve(idx.size());
                for (const std::size_t j : idx) vals.push_back(p[j]);
                meds.push_back(median_of(std::move(vals)));
            }
            v = py_min(meds) / LN2;
        } else {
            std::vector<double> vals;
            vals.reserve(band_idx.size());
            for (const std::size_t j : band_idx) vals.push_back(p[j]);
            const double hi = median_of(std::move(vals));
            std::vector<double> lo_vals;
            lo_vals.reserve(skirt->first.size());
            for (std::size_t s = 0; s < skirt->first.size(); ++s) lo_vals.push_back(p[skirt->first[s]] / skirt->second[s]);
            const double lo = median_of(std::move(lo_vals));
            v = std::sqrt(hi * lo) / LN2;
        }
        out[static_cast<std::size_t>(i)] = static_cast<float>(v);
    }
    return out;
}

// --- live.StreamFE -----------------------------------------------------------------------

StreamFE::StreamFE(std::int64_t block) : block_(std::max<std::int64_t>(ALIGN, floordiv(block, ALIGN) * ALIGN)) {}

std::optional<std::int64_t> StreamFE::next_index() const {
    if (!a0_) return std::nullopt;
    return *a0_ + static_cast<std::int64_t>(pend_.size());
}

std::vector<StreamFE::Chunk> StreamFE::push(std::int64_t a, std::span<const double> x) {
    if (!a0_ || a != *next_index()) {
        // (re)start on the grid: drop the samples before the next multiple of ALIGN
        const std::int64_t skip = floormod(-a, ALIGN);
        a0_ = a + skip;
        const std::size_t s = std::min<std::size_t>(static_cast<std::size_t>(skip), x.size());
        pend_.assign(x.begin() + static_cast<std::ptrdiff_t>(s), x.end());
    } else {
        pend_.insert(pend_.end(), x.begin(), x.end());
    }
    std::vector<Chunk> out;
    while (static_cast<std::int64_t>(pend_.size()) >= 2 * HALO + ALIGN) {
        const std::int64_t n =
            std::min(block_, floordiv(static_cast<std::int64_t>(pend_.size()) - 2 * HALO, ALIGN) * ALIGN);
        const std::vector<cf> fe =
            audio_to_fe(std::span<const double>(pend_.data(), static_cast<std::size_t>(n + 2 * HALO)));
        const std::int64_t k0 = floordiv(HALO, FE_DECIM);
        const std::int64_t m = floordiv(n, FE_DECIM);
        Chunk c;
        c.index = floordiv(*a0_ + HALO, FE_DECIM);
        const auto lo = static_cast<std::size_t>(std::min<std::int64_t>(k0, static_cast<std::int64_t>(fe.size())));
        const auto hi = static_cast<std::size_t>(std::min<std::int64_t>(k0 + m, static_cast<std::int64_t>(fe.size())));
        c.fe.assign(fe.begin() + static_cast<std::ptrdiff_t>(lo), fe.begin() + static_cast<std::ptrdiff_t>(hi));
        out.push_back(std::move(c));
        pend_.erase(pend_.begin(), pend_.begin() + static_cast<std::ptrdiff_t>(n));
        *a0_ += n;
    }
    return out;
}

}  // namespace sstvae::qrss::frontend
