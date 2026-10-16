#include "qrss/chan.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "math/npcomplex.hpp"
#include "math/npstats.hpp"
#include "qrss/qrss_config.hpp"
#include "qrss/frontend.hpp"
#include "qrss/rational.hpp"

namespace sstvae::qrss::track {

namespace {

using cf = std::complex<float>;

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

// int(x) of an integral float (math.floor, math.ceil, round): Python's
// refusals for NaN and inf.
std::int64_t int_of(double x) {
    if (std::isnan(x)) throw std::invalid_argument("cannot convert float NaN to integer");
    if (std::isinf(x)) throw std::overflow_error("cannot convert float infinity to integer");
    if (!(x >= -9223372036854775808.0 && x < 9223372036854775808.0))
        throw std::overflow_error("Python int too large to convert to C int64");
    return static_cast<std::int64_t>(x);
}

// seg[a - i_a : b - i_a] = src[a:b], numpy's slice assignment: the source
// slice clamps to src's length and must then match (or be one element,
// which broadcasts).
template <typename T>
void assign_slice(std::vector<T>& seg, std::int64_t off, std::span<const T> src, std::int64_t a, std::int64_t b) {
    const std::int64_t want = b - a;
    const std::int64_t have = std::max<std::int64_t>(0, std::min<std::int64_t>(b, static_cast<std::int64_t>(src.size())) - a);
    if (have == want) {
        std::copy(src.begin() + a, src.begin() + b, seg.begin() + off);
    } else if (have == 1) {
        std::fill(seg.begin() + off, seg.begin() + off + want, src[static_cast<std::size_t>(a)]);
    } else {
        throw std::invalid_argument("could not broadcast input array from shape (" + std::to_string(have) +
                                    ",) into shape (" + std::to_string(want) + ",)");
    }
}

}  // namespace

ChanCapture make_chan(std::span<const cf> fe, std::optional<std::span<const float>> keep, double t0_index_fe,
                      std::int64_t fs_fe, double f_audio_hz, double t_lo_s, double t_hi_s,
                      std::span<const double> exclude_hz) {
    const std::int64_t decim = floordiv(fs_fe, config::CH_FS);
    const double fs = static_cast<double>(fs_fe);
    std::int64_t i_a = int_of(std::floor(t0_index_fe + t_lo_s * fs));
    i_a -= floormod(i_a, decim);
    std::int64_t i_b = int_of(std::ceil(t0_index_fe + t_hi_s * fs));
    i_b += floormod(-(i_b - i_a), decim);
    const std::int64_t n = i_b - i_a;
    if (n < 0) throw std::invalid_argument("negative dimensions are not allowed");
    std::vector<cf> seg(static_cast<std::size_t>(n));
    const std::int64_t a = std::max<std::int64_t>(i_a, 0);
    const std::int64_t b = std::min<std::int64_t>(i_b, static_cast<std::int64_t>(fe.size()));
    if (b > a) assign_slice<cf>(seg, a - i_a, fe, a, b);
    auto [ch, f_mix] = frontend::channelise(std::span<const cf>(seg), fs_fe, frontend::fe_hz(f_audio_hz),
                                            config::CH_FS, i_a);
    ChanCapture c;
    if (keep) {
        std::vector<float> ks(static_cast<std::size_t>(n));
        if (b > a) assign_slice<float>(ks, a - i_a, *keep, a, b);
        c.keep = frontend::channelise_keep(std::span<const float>(ks), fs_fe, config::CH_FS);
    }
    c.ch = std::move(ch);
    c.fs = config::CH_FS;
    c.t0_index = (t0_index_fe - static_cast<double>(i_a)) / static_cast<double>(decim);
    c.f_mix = f_mix;
    c.exclude_hz.assign(exclude_hz.begin(), exclude_hz.end());
    return c;
}

std::vector<float> default_noise_psd(std::span<const cf> ch, std::int64_t fs, std::span<const double> exclude_hz,
                                     double win_s, std::optional<double> audio_hz) {
    frontend::NoisePsdOptions opts;
    opts.audio_hz = audio_hz;
    return frontend::noise_psd(ch, fs, exclude_hz, win_s, opts);
}

namespace {

NoisePsd compute_table(const ChanCapture& c, const NoisePsdFn& fn) {
    const double mix = c.mix_audio_hz();
    std::vector<double> ex;
    ex.reserve(c.exclude_hz.size());
    for (const double h : c.exclude_hz) ex.push_back(h - mix);
    const std::int64_t len = static_cast<std::int64_t>(c.ch.size());
    const std::int64_t w = std::max<std::int64_t>(16, int_of(std::nearbyint(NOISE_WIN_S * static_cast<double>(c.fs))));
    const std::vector<float> psd = fn ? fn(c.ch, c.fs, ex, NOISE_WIN_S, mix)
                                      : default_noise_psd(c.ch, c.fs, ex, NOISE_WIN_S, mix);
    const std::size_t nwin = psd.size();
    const double half = static_cast<double>(w) / 2.0;
    std::vector<double> cen(nwin);
    for (std::size_t i = 0; i < nwin; ++i) {
        const std::int64_t start = std::min<std::int64_t>(static_cast<std::int64_t>(i) * w,
                                                          std::max<std::int64_t>(len - w, 0));
        cen[i] = static_cast<double>(start) + half;
    }
    // windows that are mostly zero padding read low: take them from the rest
    std::vector<bool> good(nwin);
    bool any = false;
    for (std::size_t i = 0; i < nwin; ++i) {
        double frac = 0.0;
        if (len > 0) {
            const double lo_f = std::max(cen[i] - half, 0.0);   // Python max(x, 0): x unless 0 > x
            std::int64_t lo = static_cast<std::int64_t>(lo_f);
            std::int64_t hi = static_cast<std::int64_t>(cen[i] + half);
            lo = std::clamp<std::int64_t>(lo, 0, len);
            hi = std::clamp<std::int64_t>(hi, 0, len);
            if (hi <= lo) {
                frac = std::nan("");
            } else {
                std::int64_t cnt = 0;
                for (std::int64_t k = lo; k < hi; ++k) {
                    // np.abs(ch) > 0: false for a NaN part (|NaN| is NaN),
                    // which a plain "nonzero" test would count
                    cnt += math::np::abs(c.ch[static_cast<std::size_t>(k)]) > 0.0f ? 1 : 0;
                }
                frac = static_cast<double>(cnt) / static_cast<double>(hi - lo);
            }
        }
        good[i] = (frac > 0.9) && (psd[i] > 0.0f);
        any = any || good[i];
    }
    NoisePsd out;
    out.t_s = c.t_s(std::span<const double>(cen));
    out.psd.resize(nwin);
    if (any) {
        std::vector<float> kept;
        for (std::size_t i = 0; i < nwin; ++i)
            if (good[i]) kept.push_back(psd[i]);
        const float med = math::median(std::span<const float>(kept));
        for (std::size_t i = 0; i < nwin; ++i) out.psd[i] = static_cast<double>(good[i] ? psd[i] : med);
    } else {
        const double m = static_cast<double>(math::median(std::span<const float>(psd)));
        const double v = (1e-30 > m) ? 1e-30 : m;   // max(m, 1e-30): NaN stays NaN
        std::fill(out.psd.begin(), out.psd.end(), v);
    }
    return out;
}

}  // namespace

std::shared_ptr<const NoisePsd> noise_psd_table(const ChanCapture& c, const NoisePsdFn& fn) {
    return c.psd.get_or_compute([&] { return compute_table(c, fn); });
}

double noise_psd_at(const ChanCapture& c, double t_s, const NoisePsdFn& fn) {
    const auto table = noise_psd_table(c, fn);
    return math::interp(t_s, table->t_s, table->psd);
}

std::vector<double> noise_psd_at(const ChanCapture& c, std::span<const double> t_s, const NoisePsdFn& fn) {
    const auto table = noise_psd_table(c, fn);
    return math::interp(t_s, table->t_s, table->psd);
}

double noise_floor(const ChanCapture& c, const NoisePsdFn& fn) {
    (void)noise_psd_at(c, 0.0, fn);
    return math::median(std::span<const double>(c.psd.get()->psd));
}

}  // namespace sstvae::qrss::track
