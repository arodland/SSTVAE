#include "dsp/dsp.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "dsp/fft.hpp"

namespace sstvae::dsp {
namespace {

using config::FCENTER;
using config::FS;

constexpr double PI = std::numbers::pi;
constexpr double TWO_PI = 2.0 * PI;

std::int64_t gcd_i(std::int64_t a, std::int64_t b) {
    while (b) {
        const std::int64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

// The heterodyne is exactly periodic: FCENTER/FS reduces to 3/16, so
// there are only 16 distinct phasors and `n` never reaches exp() at all.
//
// This matters more here than anywhere else in the modem because `n`
// runs over a whole recording rather than one symbol. Unreduced, a mode
// C transmission reaches |theta| = 895,000 rad -- one ulp there is
// 1.16e-10 -- and accumulates 1.47e-10 rad of error, ~5000x the error in
// the OFDM matrices. sstvae/modem/dsp.py now does the same thing; see
// docs/todo.md.
//
// Derived from the config rather than hardcoded as 3/16, so a change to
// FCENTER or FS stays correct instead of silently building a table for
// the wrong frequency.
struct Heterodyne {
    std::int64_t period;
    std::int64_t step;
    std::vector<cdouble> table;

    Heterodyne() {
        const std::int64_t g = gcd_i(FCENTER, FS);
        period = FS / g;
        step = FCENTER / g;
        table.resize(static_cast<std::size_t>(period));
        for (std::int64_t k = 0; k < period; ++k)
            // k/period is exact for a power-of-two period, so these are
            // as accurate as exp() can be.
            table[static_cast<std::size_t>(k)] = std::polar(
                1.0, -TWO_PI * (static_cast<double>(k) / static_cast<double>(period)));
    }
};

const Heterodyne& heterodyne() {
    static const Heterodyne h;
    return h;
}

// numpy.sinc: sin(pi x)/(pi x), 1 at 0.
double sinc(double x) {
    if (x == 0.0) return 1.0;
    const double y = PI * x;
    return std::sin(y) / y;
}

// scipy.signal.windows.hamming(M, sym=True), which is
// general_cosine(M, [0.54, 0.46]) -> 0.54 + 0.46*cos(fac),
// fac = linspace(-pi, pi, M).
std::vector<double> hamming(int m) {
    std::vector<double> w(static_cast<std::size_t>(m));
    if (m == 1) {
        w[0] = 1.0;
        return w;
    }
    const double step = TWO_PI / static_cast<double>(m - 1);
    for (int n = 0; n < m; ++n) {
        // numpy's linspace pins the last element to `stop` exactly
        // rather than letting start + n*step land near it.
        const double fac = (n == m - 1) ? PI : -PI + static_cast<double>(n) * step;
        w[static_cast<std::size_t>(n)] = 0.54 + 0.46 * std::cos(fac);
    }
    return w;
}

// The shared body of scipy's firwin for a single passband, with
// scale=True and a caller-supplied window (Hamming for the modem's own
// filters, Kaiser for resampling).
std::vector<double> firwin_band(int numtaps, double left, double right,
                                const std::vector<double>& win) {
    const double alpha = 0.5 * (numtaps - 1);
    std::vector<double> h(static_cast<std::size_t>(numtaps));
    std::vector<double> m(static_cast<std::size_t>(numtaps));
    for (int i = 0; i < numtaps; ++i) {
        m[static_cast<std::size_t>(i)] = static_cast<double>(i) - alpha;
        const double mi = m[static_cast<std::size_t>(i)];
        h[static_cast<std::size_t>(i)] = right * sinc(right * mi) - left * sinc(left * mi);
    }

    for (std::size_t i = 0; i < h.size(); ++i) h[i] *= win[i];

    // scale=True: normalize the response at the passband centre (or at
    // DC for a lowpass).
    double scale_frequency;
    if (left == 0.0)
        scale_frequency = 0.0;
    else if (right == 1.0)
        scale_frequency = 1.0;
    else
        scale_frequency = 0.5 * (left + right);

    double s = 0.0;
    for (std::size_t i = 0; i < h.size(); ++i)
        s += h[i] * std::cos(PI * m[i] * scale_frequency);
    for (double& v : h) v /= s;
    return h;
}

template <typename T>
std::vector<T> convolve_same_impl(std::span<const T> a, std::span<const double> v) {
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(a.size());
    const std::ptrdiff_t m = static_cast<std::ptrdiff_t>(v.size());
    if (n < m)
        throw std::invalid_argument(
            "convolve_same: the reference only ever convolves a long signal "
            "with a short kernel; the other case has different semantics");
    // full[k] = sum_j a[j] * v[k-j]; 'same' is full[(m-1)/2 ...] of
    // length n. Verified against numpy for odd and even kernels.
    const std::ptrdiff_t offset = (m - 1) / 2;
    std::vector<T> out(static_cast<std::size_t>(n));
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const std::ptrdiff_t k = i + offset;
        T acc{};
        const std::ptrdiff_t jlo = std::max<std::ptrdiff_t>(0, k - m + 1);
        const std::ptrdiff_t jhi = std::min<std::ptrdiff_t>(n - 1, k);
        for (std::ptrdiff_t j = jlo; j <= jhi; ++j)
            acc += a[static_cast<std::size_t>(j)] *
                   v[static_cast<std::size_t>(k - j)];
        out[static_cast<std::size_t>(i)] = acc;
    }
    return out;
}

// numpy.convolve(a, v, mode="same") for complex a / real v, via FFT
// rather than the direct sum convolve_same_impl uses. Only sync_lowpass
// wants this: its 129-tap kernel against a's full length (the whole
// ring-buffer snapshot, not a bounded search window) made the direct sum
// the single largest cost in a live decode-loop profile -- a poll-cycle
// tax that scaled with total buffer duration rather than with anything
// acquisition actually needs. Same offset arithmetic as
// convolve_same_impl, just applied to the FFT-computed full convolution
// instead of a materialized one.
std::vector<cdouble> fftconvolve_same(std::span<const cdouble> a,
                                       std::span<const double> v) {
    const std::size_t full = a.size() + v.size() - 1;
    const std::size_t n = next_fast_len(full, /*real=*/false);
    std::vector<cdouble> pa(n, cdouble{}), pv(n, cdouble{});
    std::copy(a.begin(), a.end(), pa.begin());
    for (std::size_t i = 0; i < v.size(); ++i) pv[i] = cdouble{v[i], 0.0};
    std::vector<cdouble> fa = fft(pa, true);
    const std::vector<cdouble> fv = fft(pv, true);
    for (std::size_t i = 0; i < n; ++i) fa[i] *= fv[i];
    const std::vector<cdouble> conv = fft(fa, false);

    const std::size_t offset = (v.size() - 1) / 2;
    return std::vector<cdouble>(
        conv.begin() + static_cast<std::ptrdiff_t>(offset),
        conv.begin() + static_cast<std::ptrdiff_t>(offset + a.size()));
}

}  // namespace

std::vector<cdouble> to_baseband(std::span<const double> x) {
    const Heterodyne& h = heterodyne();
    std::vector<cdouble> out(x.size());
    std::int64_t k = 0;
    for (std::size_t n = 0; n < x.size(); ++n) {
        out[n] = x[n] * h.table[static_cast<std::size_t>(k)];
        // Equivalent to (step*n) % period, kept incremental so it stays
        // exact for arbitrarily long recordings without a 64-bit product.
        k += h.step;
        if (k >= h.period) k -= h.period;
    }
    return out;
}

double wrap_cycles(double cycles) { return cycles - std::floor(cycles); }

std::vector<cdouble> freq_correct(std::span<const cdouble> z, double f_hz) {
    std::vector<cdouble> out(z.size());
    for (std::size_t n = 0; n < z.size(); ++n) {
        const double cycles =
            wrap_cycles(f_hz * static_cast<double>(n) / static_cast<double>(FS));
        out[n] = z[n] * std::polar(1.0, -TWO_PI * cycles);
    }
    return out;
}

std::vector<double> firwin_lowpass(int numtaps, double cutoff_hz) {
    const double nyq = 0.5 * static_cast<double>(FS);
    return firwin_band(numtaps, 0.0, cutoff_hz / nyq, hamming(numtaps));
}

std::vector<double> firwin_bandpass(int numtaps, double lo_hz, double hi_hz) {
    const double nyq = 0.5 * static_cast<double>(FS);
    return firwin_band(numtaps, lo_hz / nyq, hi_hz / nyq, hamming(numtaps));
}

double bessel_i0(double x) {
    // I0(x) = sum_k (x^2/4)^k / (k!)^2.
    //
    // Every term is positive, so there is no cancellation and the error
    // is just accumulated rounding -- accurate to the last couple of
    // ulp over the range this is used for. `resample_poly` fixes
    // beta = 5, so the argument never exceeds 5; the series is still
    // fine well past that, but it is not the right algorithm for large
    // x, where the asymptotic form is needed. Hence the guard.
    const double ax = std::abs(x);
    if (ax > 30.0) {
        throw std::invalid_argument(
            "bessel_i0: the power series is not appropriate above ~30; "
            "an asymptotic branch would be needed");
    }
    const double y = 0.25 * ax * ax;
    double term = 1.0;
    double sum = 1.0;
    for (int k = 1; k < 64; ++k) {
        term *= y / (static_cast<double>(k) * static_cast<double>(k));
        sum += term;
        if (term < 1e-18 * sum) break;
    }
    return sum;
}

std::vector<double> kaiser(int m, double beta) {
    std::vector<double> w(static_cast<std::size_t>(m));
    if (m == 1) {
        w[0] = 1.0;
        return w;
    }
    const double alpha = 0.5 * (m - 1);
    const double denom = bessel_i0(beta);
    for (int n = 0; n < m; ++n) {
        const double t = (static_cast<double>(n) - alpha) / alpha;
        w[static_cast<std::size_t>(n)] =
            bessel_i0(beta * std::sqrt(std::max(0.0, 1.0 - t * t))) / denom;
    }
    return w;
}

// scipy's _output_len: the length upfirdn returns.
std::size_t upfirdn_out_len(std::size_t len_h, std::size_t len_x, int up, int down) {
    const std::size_t in_len = (len_x - 1) * static_cast<std::size_t>(up) + len_h;
    return (in_len - 1) / static_cast<std::size_t>(down) + 1;
}

std::vector<double> resample_poly(std::span<const double> x, int up, int down,
                                  std::optional<std::vector<double>>* filter_cache) {
    if (up <= 0 || down <= 0) throw std::invalid_argument("resample_poly: up/down must be positive");
    const int g = static_cast<int>(std::gcd(up, down));
    up /= g;
    down /= g;
    if (up == 1 && down == 1) return std::vector<double>(x.begin(), x.end());
    if (x.empty()) return {};

    const std::size_t n_in = x.size();
    std::size_t n_out = n_in * static_cast<std::size_t>(up);
    n_out = n_out / static_cast<std::size_t>(down) +
            ((n_out % static_cast<std::size_t>(down)) ? 1 : 0);

    // Filter design, exactly scipy's: cutoff 1/max(up,down) relative to
    // Nyquist, 10*max(up,down) taps either side, Kaiser(5.0).
    //
    // `slot` is the caller's filter_cache if one was given, or a local
    // optional that only lives for this one call otherwise -- either
    // way the design-and-populate logic below runs unchanged, and reads
    // back through the same `h` reference.
    const int max_rate = std::max(up, down);
    const int half_len = 10 * max_rate;
    std::optional<std::vector<double>> local_filter;
    std::optional<std::vector<double>>& slot = filter_cache ? *filter_cache : local_filter;
    if (!slot.has_value()) {
        const int numtaps = 2 * half_len + 1;
        std::vector<double> h = firwin_band(numtaps, 0.0, 1.0 / max_rate, kaiser(numtaps, 5.0));
        for (double& v : h) v *= up;
        slot = std::move(h);
    }
    const std::vector<double>& h = *slot;

    // Zero-pad the filter so the output samples land at the centre.
    const std::size_t n_pre_pad =
        static_cast<std::size_t>(down - half_len % down);
    std::size_t n_post_pad = 0;
    const std::size_t n_pre_remove =
        (static_cast<std::size_t>(half_len) + n_pre_pad) / static_cast<std::size_t>(down);
    while (upfirdn_out_len(h.size() + n_pre_pad + n_post_pad, n_in, up, down) <
           n_out + n_pre_remove) {
        ++n_post_pad;
    }
    std::vector<double> hp(n_pre_pad, 0.0);
    hp.insert(hp.end(), h.begin(), h.end());
    hp.insert(hp.end(), n_post_pad, 0.0);

    // upfirdn with mode='constant', cval=0: insert up-1 zeros between
    // input samples, convolve, keep every `down`th. Written as a direct
    // polyphase sum -- for each output we touch only the taps that align
    // with a real input sample, so the inserted zeros are never visited.
    const std::size_t n_full = upfirdn_out_len(hp.size(), n_in, up, down);
    std::vector<double> y(n_full, 0.0);
    const std::ptrdiff_t up_s = up;
    for (std::size_t i = 0; i < n_full; ++i) {
        const std::ptrdiff_t t = static_cast<std::ptrdiff_t>(i) * down;
        // Taps k with (t - k) a multiple of `up` and inside the signal.
        std::ptrdiff_t k = t % up_s;
        double acc = 0.0;
        for (; k <= t && k < static_cast<std::ptrdiff_t>(hp.size()); k += up_s) {
            const std::ptrdiff_t j = (t - k) / up_s;
            if (j < static_cast<std::ptrdiff_t>(n_in)) {
                acc += hp[static_cast<std::size_t>(k)] * x[static_cast<std::size_t>(j)];
            }
        }
        y[i] = acc;
    }

    return std::vector<double>(y.begin() + static_cast<std::ptrdiff_t>(n_pre_remove),
                               y.begin() + static_cast<std::ptrdiff_t>(n_pre_remove + n_out));
}

std::vector<double> convolve_same(std::span<const double> a,
                                  std::span<const double> v) {
    return convolve_same_impl<double>(a, v);
}

std::vector<cdouble> convolve_same(std::span<const cdouble> a,
                                   std::span<const double> v) {
    return convolve_same_impl<cdouble>(a, v);
}

std::vector<cdouble> sync_lowpass(std::span<const cdouble> z) {
    static const std::vector<double> taps = firwin_lowpass(129, 850.0);
    return fftconvolve_same(z, std::span<const double>(taps));
}

std::vector<cdouble> hilbert(std::span<const double> x) {
    const std::size_t n = x.size();
    if (n == 0) return {};
    std::vector<cdouble> xf(n);
    for (std::size_t i = 0; i < n; ++i) xf[i] = cdouble(x[i], 0.0);
    std::vector<cdouble> spectrum = fft(xf, /*forward=*/true);

    // scipy.signal.hilbert's frequency-domain mask: keep DC (and the
    // Nyquist bin when n is even), double the positive frequencies, zero
    // the negatives.
    if (n % 2 == 0) {
        for (std::size_t i = 1; i < n / 2; ++i) spectrum[i] *= 2.0;
        for (std::size_t i = n / 2 + 1; i < n; ++i) spectrum[i] = 0.0;
    } else {
        for (std::size_t i = 1; i < (n + 1) / 2; ++i) spectrum[i] *= 2.0;
        for (std::size_t i = (n + 1) / 2; i < n; ++i) spectrum[i] = 0.0;
    }
    return fft(spectrum, /*forward=*/false);
}

std::size_t next_fast_len(std::size_t n, bool real) {
    return real ? pocketfft::detail::util::good_size_real(n)
                : pocketfft::detail::util::good_size_cmplx(n);
}

std::vector<cdouble> fftconvolve_valid(std::span<const cdouble> a,
                                       std::span<const cdouble> v) {
    if (a.size() < v.size())
        throw std::invalid_argument("fftconvolve_valid: len(a) must be >= len(v)");
    const std::size_t full = a.size() + v.size() - 1;
    const std::size_t n = next_fast_len(full, /*real=*/false);

    std::vector<cdouble> pa(n, cdouble{}), pv(n, cdouble{});
    std::copy(a.begin(), a.end(), pa.begin());
    std::copy(v.begin(), v.end(), pv.begin());
    std::vector<cdouble> fa = fft(pa, true);
    const std::vector<cdouble> fv = fft(pv, true);
    for (std::size_t i = 0; i < n; ++i) fa[i] *= fv[i];
    const std::vector<cdouble> conv = fft(fa, false);

    const std::size_t valid = a.size() - v.size() + 1;
    return std::vector<cdouble>(conv.begin() + static_cast<std::ptrdiff_t>(v.size() - 1),
                                conv.begin() + static_cast<std::ptrdiff_t>(v.size() - 1 + valid));
}

std::vector<double> fftconvolve_valid(std::span<const double> a,
                                      std::span<const double> v) {
    if (a.size() < v.size())
        throw std::invalid_argument("fftconvolve_valid: len(a) must be >= len(v)");
    const std::size_t full = a.size() + v.size() - 1;
    const std::size_t n = next_fast_len(full, /*real=*/true);

    // r2c/c2r, as scipy does for real inputs. Going through a complex
    // transform instead would be algebraically identical and would round
    // differently.
    const std::size_t nc = n / 2 + 1;
    std::vector<double> pa(n, 0.0), pv(n, 0.0);
    std::copy(a.begin(), a.end(), pa.begin());
    std::copy(v.begin(), v.end(), pv.begin());

    std::vector<cdouble> fa(nc), fv(nc);
    const pocketfft::shape_t shape{n};
    const pocketfft::stride_t stride_r{static_cast<std::ptrdiff_t>(sizeof(double))};
    const pocketfft::stride_t stride_c{static_cast<std::ptrdiff_t>(sizeof(cdouble))};
    pocketfft::r2c(shape, stride_r, stride_c, pocketfft::shape_t{0}, true,
                   pa.data(), fa.data(), 1.0);
    pocketfft::r2c(shape, stride_r, stride_c, pocketfft::shape_t{0}, true,
                   pv.data(), fv.data(), 1.0);
    for (std::size_t i = 0; i < nc; ++i) fa[i] *= fv[i];

    std::vector<double> conv(n);
    pocketfft::c2r(shape, stride_c, stride_r, pocketfft::shape_t{0}, false,
                   fa.data(), conv.data(), 1.0 / static_cast<double>(n));

    const std::size_t valid = a.size() - v.size() + 1;
    return std::vector<double>(conv.begin() + static_cast<std::ptrdiff_t>(v.size() - 1),
                               conv.begin() + static_cast<std::ptrdiff_t>(v.size() - 1 + valid));
}

std::vector<double> tx_condition(std::span<const double> x,
                                 double clip_headroom_db, int iterations) {
    double power = 0.0;
    for (double v : x) power += v * v;
    power /= static_cast<double>(x.size());
    std::vector<double> out(x.begin(), x.end());
    if (power == 0.0) return out;

    // Mean envelope power is 2x mean real power.
    const double thresh =
        std::sqrt(2.0 * power) * std::pow(10.0, clip_headroom_db / 20.0);
    static const std::vector<double> taps =
        firwin_bandpass(201, config::TX_BANDPASS_LO, config::TX_BANDPASS_HI);

    for (int it = 0; it < iterations; ++it) {
        const std::vector<cdouble> z = hilbert(out);
        for (std::size_t i = 0; i < out.size(); ++i) {
            const double mag = std::abs(z[i]);
            const double scale = std::min(1.0, thresh / std::max(mag, 1e-12));
            out[i] = (z[i] * scale).real();
        }
        out = convolve_same(std::span<const double>(out),
                            std::span<const double>(taps));
    }

    double rms = 0.0;
    for (double v : out) rms += v * v;
    rms = std::sqrt(rms / static_cast<double>(out.size()));
    for (double& v : out) v /= rms;
    return out;
}

double papr_db(std::span<const double> x) {
    const std::vector<cdouble> z = hilbert(x);
    double peak = 0.0;
    double mean = 0.0;
    for (const cdouble& v : z) {
        const double e2 = std::norm(v);
        peak = std::max(peak, e2);
        mean += e2;
    }
    mean /= static_cast<double>(z.size());
    return 10.0 * std::log10(peak / mean);
}

std::vector<std::int16_t> to_int16(std::span<const double> x, double peak) {
    double amax = 0.0;
    for (double v : x) amax = std::max(amax, std::abs(v));
    std::vector<std::int16_t> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        // std::nearbyint under the default rounding mode is
        // half-to-even, which is what np.round does. std::round is
        // half-away-from-zero and would differ.
        const double scaled = x[i] / amax * peak * 32767.0;
        out[i] = static_cast<std::int16_t>(std::nearbyint(scaled));
    }
    return out;
}

}  // namespace sstvae::dsp
