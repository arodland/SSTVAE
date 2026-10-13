// acquire.py's shared statistics: see acquire_stats.hpp.

#include "qrss/acquire_stats.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>

#include "math/ndfilter.hpp"
#include "math/npfft.hpp"
#include "math/npstats.hpp"
#include "math/special.hpp"
#include "qrss/filters.hpp"
#include "qrss/lwave.hpp"
#include "qrss/npsum.hpp"
#include "qrss/rational.hpp"

namespace sstvae::qrss::acquire {
namespace {

using cdouble = std::complex<double>;

// np.float64 scalar ** 2: numpy's scalar power calls libm pow (not always
// x*x); the volatile exponent stops GCC folding pow(x, 2.0) to x*x.
double scalar_square(double x) {
    volatile double two = 2.0;
    return std::pow(x, two);
}

// Python's round(x) to an int: half to even.
std::int64_t py_round(double x) { return static_cast<std::int64_t>(std::nearbyint(x)); }

// np.round(v).astype(np.int64): what x86-64's conversion gives for NaN,
// infinities and values past int64 is INT64_MIN.
std::int64_t round_to_int64(double v) {
    const double r = std::nearbyint(v);
    if (!(r >= -9223372036854775808.0 && r < 9223372036854775808.0)) return std::numeric_limits<std::int64_t>::min();
    return static_cast<std::int64_t>(r);
}

template <typename S>
Grid32 spectrogram_impl(std::span<const S> x, double fs, std::span<const double> starts,
                        std::optional<std::span<const std::int64_t>> bins) {
    const std::int64_t L = frame_len(fs);
    const std::vector<double> w = frame_window(fs);
    const std::size_t nb = bins ? bins->size() : static_cast<std::size_t>(std::max<std::int64_t>(L, 0));
    Grid32 out(starts.size(), nb);
    if (starts.empty()) return out;
    // x[np.clip(idx, 0, 0)] of an empty x fails before the FFT does.
    if (x.empty()) throw std::out_of_range("index 0 is out of bounds for axis 0 with size 0");
    // X[:, bins]: numpy's fancy index, checked when the first chunk runs.
    std::vector<std::size_t> cols(nb);
    for (std::size_t j = 0; j < nb; ++j) {
        std::int64_t b = bins ? (*bins)[j] : static_cast<std::int64_t>(j);
        if (b < -L || b >= L)
            throw std::out_of_range("index " + std::to_string(b) + " is out of bounds for axis 1 with size " +
                                    std::to_string(L));
        if (b < 0) b += L;
        cols[j] = static_cast<std::size_t>(b);
    }
    const auto n = static_cast<std::int64_t>(x.size());
    std::vector<cdouble> fr(static_cast<std::size_t>(L));
    for (std::size_t i = 0; i < starts.size(); ++i) {
        const std::int64_t st = round_to_int64(starts[i]);
        for (std::int64_t k = 0; k < L; ++k) {
            // idx = st + k, in int64 as numpy adds (no overflow for the
            // starts a capture can have; INT64_MIN + k stays negative).
            const std::int64_t idx = st + k;
            const double wk = w[static_cast<std::size_t>(k)];
            if (idx >= 0 && idx < n) {
                const S v = x[static_cast<std::size_t>(idx)];
                fr[static_cast<std::size_t>(k)] =
                    cdouble(static_cast<double>(v.real()) * wk, static_cast<double>(v.imag()) * wk);
            } else {
                fr[static_cast<std::size_t>(k)] = cdouble(0.0 * wk, 0.0 * wk);
            }
        }
        sstvae::math::fft_inplace(fr);
        for (std::size_t j = 0; j < nb; ++j) {
            const cdouble X = fr[cols[j]];
            const double re2 = X.real() * X.real();
            const double im2 = X.imag() * X.imag();
            out(i, j) = static_cast<float>(re2 + im2);
        }
    }
    return out;
}

template <typename T>
std::vector<double> spectral_shape_impl(const Grid<T>& Pn, int local_bins,
                                        std::optional<std::span<const std::uint8_t>> rows) {
    std::vector<std::size_t> keep;
    if (rows) {
        if (rows->size() != Pn.rows)
            throw std::out_of_range("boolean index did not match indexed array along axis 0; size of axis is " +
                                    std::to_string(Pn.rows) + " but size of corresponding boolean axis is " +
                                    std::to_string(rows->size()));
        for (std::size_t i = 0; i < Pn.rows; ++i)
            if ((*rows)[i]) keep.push_back(i);
    } else {
        keep.resize(Pn.rows);
        std::iota(keep.begin(), keep.end(), std::size_t{0});
    }
    const std::size_t nbins = Pn.cols;
    if (keep.empty()) return std::vector<double>(nbins, 1.0);
    // np.median(Pn, axis=0).astype(np.float64) / LN2
    std::vector<double> lvl(nbins);
    std::vector<T> col(keep.size());
    for (std::size_t j = 0; j < nbins; ++j) {
        for (std::size_t r = 0; r < keep.size(); ++r) col[r] = Pn(keep[r], j);
        lvl[j] = static_cast<double>(sstvae::math::median(std::span<const T>(col))) / rxa::LN2;
    }
    const std::int64_t len = static_cast<std::int64_t>(nbins);
    const std::int64_t w = len > 2 ? std::min<std::int64_t>(local_bins | 1, 2 * (len / 2) - 1) : 1;
    if (w < 1 || w % 2 == 0 || w > std::numeric_limits<int>::max())
        throw std::invalid_argument("spectral_shape: median window " + std::to_string(w) + " is not odd and positive");
    std::vector<double> sh = sstvae::math::median_filter(std::span<const double>(lvl), static_cast<int>(w));
    for (double& v : sh)
        if (!(v > 0)) v = 1.0;
    return sh;
}

template <typename T>
Grid32 cfar_normalise_impl(const Grid<T>& P, std::optional<int> local_bins) {
    // med = np.median(P, axis=1, keepdims=True) / LN2, in P's precision
    // (a float32 median over a Python float stays float32).
    std::vector<T> med(P.rows);
    std::vector<std::uint8_t> live(P.rows);
    for (std::size_t i = 0; i < P.rows; ++i) {
        med[i] = sstvae::math::median(P.row(i)) / static_cast<T>(rxa::LN2);
        live[i] = med[i] > 0;
    }
    Grid<T> out(P.rows, P.cols);
    for (std::size_t i = 0; i < P.rows; ++i)
        for (std::size_t j = 0; j < P.cols; ++j) out(i, j) = live[i] ? P(i, j) / med[i] : T(0);
    Grid32 res(P.rows, P.cols);
    if (local_bins && *local_bins != 0) {
        const std::vector<double> sh = spectral_shape_impl(out, *local_bins, std::span<const std::uint8_t>(live));
        for (std::size_t i = 0; i < P.rows; ++i)
            for (std::size_t j = 0; j < P.cols; ++j)
                res(i, j) = static_cast<float>(static_cast<double>(out(i, j)) / sh[j]);
    } else {
        for (std::size_t k = 0; k < out.data.size(); ++k) res.data[k] = static_cast<float>(out.data[k]);
    }
    return res;
}

template <typename T>
double parabola_impl(T ym, T y0, T yp) {
    const T den = ym - T(2) * y0 + yp;
    if (!std::isfinite(den) || den >= 0) return 0.0;
    const T v = T(0.5) * (ym - yp) / den;
    // np.clip(v, -0.5, 0.5): maximum then minimum, NaN through.
    if (std::isnan(v)) return static_cast<double>(v);
    return static_cast<double>(std::min(std::max(v, T(-0.5)), T(0.5)));
}

// _band_cov's frame spectra, per fs: r0 = |fft(w*w)| / e and
// r1 = |fft(w1)| / e (lru_cache in the reference).
struct BandCovTables {
    std::vector<double> r0, r1;
};

std::shared_ptr<const BandCovTables> band_cov_tables(double fs) {
    static std::mutex mu;
    static std::map<double, std::shared_ptr<const BandCovTables>> cache;
    static std::vector<double> order;   // insertion order, for a bound of 8
    {
        std::lock_guard<std::mutex> lock(mu);
        const auto it = cache.find(fs);
        if (it != cache.end()) return it->second;
    }
    const std::vector<double> w = frame_window(fs);
    const std::size_t L = w.size();
    const auto H = py_round(rxa::STFT_HOP_S * fs);
    std::vector<double> ww(L);
    for (std::size_t i = 0; i < L; ++i) ww[i] = w[i] * w[i];
    const double e = npsum::sum(std::span<const double>(ww));
    std::vector<double> w1(L, 0.0);
    // w1[H:] = w[H:] * w[:L - H] (numpy slices clamp; a negative H counts
    // from the end).
    const auto Ls = static_cast<std::int64_t>(L);
    auto clamp_start = [&](std::int64_t s) { return s < 0 ? std::max<std::int64_t>(s + Ls, 0) : std::min(s, Ls); };
    const std::int64_t a0 = clamp_start(H);
    const std::int64_t b1 = clamp_start(Ls - H);   // w[:L - H] ends here
    const std::int64_t m = std::min(Ls - a0, b1);
    if (m != Ls - a0) throw std::invalid_argument("_band_cov: hop does not fit the frame");
    for (std::int64_t i = 0; i < m; ++i)
        w1[static_cast<std::size_t>(a0 + i)] = w[static_cast<std::size_t>(a0 + i)] * w[static_cast<std::size_t>(i)];
    auto spectrum = [&](const std::vector<double>& v) {
        const std::vector<cdouble> X = sstvae::math::fft(std::span<const double>(v));
        std::vector<double> r(X.size());
        for (std::size_t i = 0; i < X.size(); ++i) r[i] = lwave::np_abs(X[i]) / e;
        return r;
    };
    auto tables = std::make_shared<BandCovTables>();
    tables->r0 = spectrum(ww);
    tables->r1 = spectrum(w1);
    std::lock_guard<std::mutex> lock(mu);
    const auto it = cache.find(fs);
    if (it != cache.end()) return it->second;
    if (order.size() >= 8) {
        cache.erase(order.front());
        order.erase(order.begin());
    }
    cache.emplace(fs, tables);
    order.push_back(fs);
    return tables;
}

}  // namespace

std::int64_t frame_len(double fs) { return py_round(rxa::STFT_FRAME_S * fs); }

std::vector<double> frame_window(double fs) { return filters::hann(frame_len(fs), false); }

Grid32 spectrogram(std::span<const std::complex<float>> x, double fs, std::span<const double> starts,
                   std::optional<std::span<const std::int64_t>> bins) {
    return spectrogram_impl(x, fs, starts, bins);
}

Grid32 spectrogram(std::span<const std::complex<double>> x, double fs, std::span<const double> starts,
                   std::optional<std::span<const std::int64_t>> bins) {
    return spectrogram_impl(x, fs, starts, bins);
}

BandBins band_bins(double fs, std::array<double, 2> band_hz, double centre_hz) {
    const std::int64_t L = frame_len(fs);
    std::vector<double> f = sstvae::math::fftfreq(L, 1.0 / fs);
    for (double& v : f) v = v + centre_hz;
    std::vector<std::int64_t> sel;
    for (std::size_t i = 0; i < f.size(); ++i)
        if (f[i] >= band_hz[0] && f[i] <= band_hz[1]) sel.push_back(static_cast<std::int64_t>(i));
    // argsort of distinct frequencies: any sort gives numpy's order.
    std::stable_sort(sel.begin(), sel.end(), [&](std::int64_t a, std::int64_t b) {
        return f[static_cast<std::size_t>(a)] < f[static_cast<std::size_t>(b)];
    });
    BandBins out;
    out.sel = sel;
    out.f_hz.reserve(sel.size());
    for (const std::int64_t i : sel) out.f_hz.push_back(f[static_cast<std::size_t>(i)]);
    return out;
}

Grid32 cfar_normalise(const Grid<float>& P, std::optional<int> local_bins) { return cfar_normalise_impl(P, local_bins); }
Grid32 cfar_normalise(const Grid<double>& P, std::optional<int> local_bins) { return cfar_normalise_impl(P, local_bins); }

std::vector<double> spectral_shape(const Grid<float>& Pn, int local_bins,
                                   std::optional<std::span<const std::uint8_t>> rows) {
    return spectral_shape_impl(Pn, local_bins, rows);
}
std::vector<double> spectral_shape(const Grid<double>& Pn, int local_bins,
                                   std::optional<std::span<const std::uint8_t>> rows) {
    return spectral_shape_impl(Pn, local_bins, rows);
}

double parabola(double ym, double y0, double yp) { return parabola_impl(ym, y0, yp); }
double parabola(float ym, float y0, float yp) { return parabola_impl(ym, y0, yp); }

std::pair<double, double> a1_gamma(std::int64_t M, double fs) {
    const std::vector<double> w = frame_window(fs);
    const std::int64_t H = py_round(rxa::STFT_HOP_S * fs);
    const auto L = static_cast<std::int64_t>(w.size());
    double rho = 0.0;
    if (H < L) {
        // w[:-H] * w[H:] (numpy slices: H <= 0 gives empty and full
        // operands, which do not broadcast unless both are empty).
        const std::int64_t n1 = H > 0 ? L - H : (H == 0 ? 0 : std::min<std::int64_t>(-H, L));
        const std::int64_t s2 = H >= 0 ? H : std::max<std::int64_t>(L + H, 0);
        const std::int64_t n2 = L - s2;
        if (n1 != n2)
            throw std::invalid_argument("operands could not be broadcast together with shapes (" + std::to_string(n1) +
                                        ",) (" + std::to_string(n2) + ",) ");
        std::vector<double> prod(static_cast<std::size_t>(n1)), sq(static_cast<std::size_t>(L));
        for (std::int64_t i = 0; i < n1; ++i)
            prod[static_cast<std::size_t>(i)] = w[static_cast<std::size_t>(i)] * w[static_cast<std::size_t>(s2 + i)];
        for (std::int64_t i = 0; i < L; ++i) sq[static_cast<std::size_t>(i)] = w[static_cast<std::size_t>(i)] * w[static_cast<std::size_t>(i)];
        rho = scalar_square(npsum::sum(std::span<const double>(prod)) / npsum::sum(std::span<const double>(sq)));
    }
    const double var = static_cast<double>(M) + static_cast<double>(2 * (M - 1)) * rho;
    return {static_cast<double>(M * M) / var, var / static_cast<double>(M)};
}

double a1_threshold(std::int64_t M, double pfa, double fs) {
    const auto [k, th] = a1_gamma(M, fs);
    return sstvae::math::gammainccinv(k, pfa) * th;
}

double a2_threshold(double pfa, std::int64_t n_chunks, std::int64_t n_scales) {
    if (n_scales == 0) throw ZeroDivision("float division by zero");
    return sstvae::math::gammainccinv(static_cast<double>(n_chunks), pfa / static_cast<double>(n_scales));
}

std::vector<Scale> preamble_scales(std::span<const std::complex<double>> S, std::size_t n_df, std::size_t n_tau,
                                   std::size_t nc, std::span<const double> var, std::span<const int> scales) {
    if (S.size() != n_df * n_tau * nc)
        throw std::invalid_argument("preamble_scales: S holds " + std::to_string(S.size()) + " values, not " +
                                    std::to_string(n_df * n_tau * nc));
    if (var.size() != n_tau * nc)
        throw std::invalid_argument("preamble_scales: var holds " + std::to_string(var.size()) + " values, not " +
                                    std::to_string(n_tau * nc));
    for (const int g : scales) {
        if (g == 0) throw ZeroDivision("integer division or modulo by zero");
        if (g < 0) throw std::invalid_argument("preamble_scales: a scale must be positive, not " + std::to_string(g));
    }
    // Sn = S / np.sqrt(var)[None]: numpy's complex division by a real
    // promoted to complex (Smith, through a reciprocal).
    std::vector<double> sq(var.size());
    for (std::size_t i = 0; i < var.size(); ++i) sq[i] = std::sqrt(var[i]);
    std::vector<double> sn(2 * S.size());   // interleaved (re, im)
    for (std::size_t f = 0; f < n_df; ++f)
        for (std::size_t t = 0; t < n_tau; ++t)
            for (std::size_t c = 0; c < nc; ++c) {
                const std::size_t i = (f * n_tau + t) * nc + c;
                const double br = sq[t * nc + c], bi = 0.0;
                const double ar = S[i].real(), ai = S[i].imag();
                double re, im;
                if (std::fabs(br) >= std::fabs(bi)) {
                    if (br == 0 && bi == 0) {
                        re = ar / std::fabs(br);
                        im = ai / std::fabs(br);
                    } else {
                        const double rat = bi / br;
                        const double scl = 1.0 / (br + bi * rat);
                        re = (ar + ai * rat) * scl;
                        im = (ai - ar * rat) * scl;
                    }
                } else {   // unreachable for a real divisor; numpy's other branch
                    const double rat = br / bi;
                    const double scl = 1.0 / (bi + br * rat);
                    re = (ar * rat + ai) * scl;
                    im = (ai * rat - ar) * scl;
                }
                sn[2 * i] = re;
                sn[2 * i + 1] = im;
            }
    std::vector<Scale> out;
    out.reserve(scales.size());
    std::vector<double> mag2;
    for (const int gi : scales) {
        const auto g = static_cast<std::size_t>(gi);
        const std::size_t k = nc / g;
        Scale sc;
        sc.k = static_cast<std::int64_t>(k);
        sc.lam.assign(n_df * n_tau, 0.0);
        mag2.assign(k, 0.0);
        for (std::size_t cell = 0; cell < n_df * n_tau; ++cell) {
            const double* base = sn.data() + 2 * cell * nc;
            for (std::size_t q = 0; q < k; ++q) {
                // Sg = sum over the run of g: pairwise onto 0 + 0j.
                double rr = 0.0, ri = 0.0;
                npsum::pairwise_sum_complex(rr, ri, base + 2 * q * g, 2 * g);
                const double sr = 0.0 + rr, si = 0.0 + ri;
                const double a = lwave::np_abs(cdouble(sr, si));
                mag2[q] = a * a;
            }
            sc.lam[cell] = npsum::sum(std::span<const double>(mag2)) / static_cast<double>(gi);
        }
        out.push_back(std::move(sc));
    }
    return out;
}

PreambleP preamble_p(std::span<const std::complex<double>> S, std::size_t n_df, std::size_t n_tau, std::size_t nc,
                     std::span<const double> var, std::span<const int> scales) {
    PreambleP r;
    r.st = preamble_scales(S, n_df, n_tau, nc, var, scales);
    const std::size_t cells = n_df * n_tau;
    const std::size_t ns = r.st.size();
    if (ns == 0) throw std::invalid_argument("need at least one array to stack");
    std::vector<std::vector<double>> ps(ns);
    for (std::size_t s = 0; s < ns; ++s)
        ps[s] = sstvae::math::gammaincc(static_cast<double>(r.st[s].k), std::span<const double>(r.st[s].lam));
    r.p.resize(cells);
    r.best.resize(cells);
    for (std::size_t c = 0; c < cells; ++c) {
        // np.argmin over the scales: the first NaN, else the first minimum.
        std::size_t b = 0;
        double v = ps[0][c];
        if (!std::isnan(v))
            for (std::size_t s = 1; s < ns; ++s) {
                const double u = ps[s][c];
                if (std::isnan(u)) {
                    b = s;
                    v = u;
                    break;
                }
                if (u < v) {
                    b = s;
                    v = u;
                }
            }
        r.best[c] = static_cast<std::int64_t>(b);
        const double scaled = v * static_cast<double>(ns);
        r.p[c] = std::isnan(scaled) ? scaled : std::min(scaled, 1.0);
    }
    return r;
}

std::pair<double, double> tbd_gumbel(std::int64_t n_frames) {
    if (n_frames < 0) throw std::invalid_argument("math domain error");
    constexpr std::size_t N = rxa::TBD_GUMBEL_N.size();
    std::array<double, N> r{};
    for (std::size_t i = 0; i < N; ++i) r[i] = std::sqrt(static_cast<double>(rxa::TBD_GUMBEL_N[i]));
    const double x = std::sqrt(static_cast<double>(n_frames));
    std::size_t i;
    if (x <= r[0]) {
        i = 0;
    } else if (x >= r[N - 1]) {
        i = N - 2;
    } else {
        // np.searchsorted(r, x) (side left): the first r[j] >= x.
        i = static_cast<std::size_t>(std::lower_bound(r.begin(), r.end(), x) - r.begin()) - 1;
    }
    const double w = (x - r[i]) / (r[i + 1] - r[i]);
    const auto& mu = rxa::TBD_GUMBEL_MU;
    const auto& be = rxa::TBD_GUMBEL_BETA;
    return {mu[i] + w * (mu[i + 1] - mu[i]), be[i] + w * (be[i + 1] - be[i])};
}

double gumbel_p(double z, std::int64_t n_frames) {
    const auto [mu, beta] = tbd_gumbel(n_frames);
    return -std::expm1(-std::exp(-(z - mu) / beta));
}

std::pair<double, double> band_cov(std::int64_t n_bins, double fs) {
    const std::shared_ptr<const BandCovTables> t = band_cov_tables(fs);
    const auto L = static_cast<std::int64_t>(t->r0.size());
    if (n_bins <= 0) return {0.0, 0.0};
    // k = arange(-(n-1), n); mult = n - |k|; sum(mult * r[k % L] ** 2)
    const std::size_t cnt = static_cast<std::size_t>(2 * n_bins - 1);
    std::vector<double> a(cnt), b(cnt);
    for (std::size_t j = 0; j < cnt; ++j) {
        const std::int64_t k = -(n_bins - 1) + static_cast<std::int64_t>(j);
        const auto mult = static_cast<double>(n_bins - (k < 0 ? -k : k));
        const std::int64_t idx = ((k % L) + L) % L;
        const double r0 = t->r0[static_cast<std::size_t>(idx)], r1 = t->r1[static_cast<std::size_t>(idx)];
        a[j] = mult * (r0 * r0);
        b[j] = mult * (r1 * r1);
    }
    return {npsum::sum(std::span<const double>(a)), npsum::sum(std::span<const double>(b))};
}

}  // namespace sstvae::qrss::acquire
