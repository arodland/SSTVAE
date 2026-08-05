#include "sync/sync.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <string>
#include <utility>

#include "dsp/dsp.hpp"
#include "dsp/fft.hpp"
#include "ofdm/ofdm.hpp"

namespace sstvae::sync {
namespace {

using config::FRAME_SAMPLES;
using config::FS;
using config::M;
using config::PREAMBLE_CORR_WINDOW;
using config::PREAMBLE_CP;
using config::PREAMBLE_REPEATS;
using config::PREAMBLE_SAMPLES;

constexpr double PI = std::numbers::pi;

// Sliding lag-M autocorrelation over a PREAMBLE_CORR_WINDOW window.
// The window, not the preamble's length, is what sets this metric's
// noise floor -- it must be widened along with the preamble or the
// extra repeats buy nothing. See config::PREAMBLE_REPEATS.
struct AutocorrMetric {
    std::vector<double> metric;
    std::vector<cdouble> a;
};

AutocorrMetric autocorr_metric(std::span<const cdouble> z) {
    const std::size_t n = z.size();
    std::vector<cdouble> prod(n - M);
    std::vector<double> power(n);
    for (std::size_t i = 0; i < n; ++i) power[i] = std::norm(z[i]);
    for (std::size_t i = 0; i < n - M; ++i) prod[i] = z[i + M] * std::conj(z[i]);

    constexpr std::size_t W = PREAMBLE_CORR_WINDOW;
    const std::vector<cdouble> kernel_c(W, cdouble{1.0, 0.0});
    const std::vector<double> kernel_r(W, 1.0);
    const std::vector<cdouble> a = dsp::fftconvolve_valid(prod, kernel_c);
    const std::vector<double> e1 = dsp::fftconvolve_valid(
        std::span<const double>(power.data(), n - M), kernel_r);
    const std::vector<double> e2 = dsp::fftconvolve_valid(
        std::span<const double>(power.data() + M, n - M), kernel_r);

    // Floor the energies at a fraction of the typical window energy so
    // near-silent regions (filter ringing) cannot produce inflated
    // metrics.
    double mean_power = 0.0;
    for (double p : power) mean_power += p;
    mean_power /= static_cast<double>(n);
    const double floor = 1e-3 * static_cast<double>(W) * mean_power;

    AutocorrMetric out;
    out.a = a;
    out.metric.resize(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double energy =
            std::sqrt(std::max(e1[i], floor) * std::max(e2[i], floor)) + 1e-12;
        out.metric[i] = std::abs(a[i]) / energy;
    }
    return out;
}

// np.argmax: first maximum wins.
std::size_t argmax(std::span<const double> v) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < v.size(); ++i)
        if (v[i] > v[best]) best = i;
    return best;
}

// np.median, including the even-length average of the two middle values.
double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const std::size_t n = v.size();
    if (n % 2 == 1) {
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(n / 2),
                         v.end());
        return v[n / 2];
    }
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(n / 2),
                     v.end());
    const double hi = v[n / 2];
    const double lo = *std::max_element(v.begin(),
                                        v.begin() + static_cast<std::ptrdiff_t>(n / 2));
    return 0.5 * (lo + hi);
}

}  // namespace

Acquisition acquire(std::span<const cdouble> z_in, double threshold, int max_bins,
                    std::optional<SearchWindow> search) {
    if (z_in.size() < static_cast<std::size_t>(PREAMBLE_SAMPLES + 2 * M))
        throw SyncError("signal too short");

    const std::vector<cdouble> z = dsp::sync_lowpass(z_in);
    AutocorrMetric am = autocorr_metric(z);
    std::vector<double>& metric = am.metric;

    if (search) {
        const std::int64_t s0 = std::max<std::int64_t>(0, search->start);
        const std::int64_t s1 =
            std::min<std::int64_t>(static_cast<std::int64_t>(metric.size()), search->end);
        if (s1 - s0 < 1)
            throw SyncError("empty search window");
        // Masked with -1 rather than truncated, so the returned index is
        // still an index into the whole signal.
        std::vector<double> masked(metric.size(), -1.0);
        for (std::int64_t i = s0; i < s1; ++i)
            masked[static_cast<std::size_t>(i)] = metric[static_cast<std::size_t>(i)];
        metric = std::move(masked);
    }

    const std::size_t n_star = argmax(metric);
    if (metric[n_star] < threshold)
        throw SyncError("no preamble found (peak metric " +
                        std::to_string(metric[n_star]) + ")");

    const double f_frac =
        std::arg(am.a[n_star]) / (2 * PI * static_cast<double>(M) / FS);

    // Integer-bin CFO search + fine timing via template correlation.
    const std::vector<cdouble> templ = ofdm::preamble_template();
    double t_norm = 0.0;
    for (const cdouble& t : templ) t_norm += std::norm(t);
    t_norm = std::sqrt(t_norm);

    const std::int64_t lo =
        std::max<std::int64_t>(0, static_cast<std::int64_t>(n_star) - PREAMBLE_CP - 200);
    const std::int64_t hi =
        std::min<std::int64_t>(static_cast<std::int64_t>(z.size()) - PREAMBLE_SAMPLES,
                               static_cast<std::int64_t>(n_star) + 200);
    if (hi <= lo) throw SyncError("preamble at signal edge");

    const std::span<const cdouble> seg(
        z.data() + lo, static_cast<std::size_t>(hi + PREAMBLE_SAMPLES - lo));

    // conj(template[::-1]) -- the matched filter.
    std::vector<cdouble> kernel(templ.size());
    for (std::size_t i = 0; i < templ.size(); ++i)
        kernel[i] = std::conj(templ[templ.size() - 1 - i]);

    bool have_best = false;
    double best_score = 0.0;
    std::int64_t best_p0 = 0;
    double best_f = 0.0;
    for (int m_bin = -max_bins; m_bin <= max_bins; ++m_bin) {
        const double f_cand = f_frac + static_cast<double>(m_bin) * FS / M;
        const std::vector<cdouble> seg_c = dsp::freq_correct(seg, f_cand);
        const std::vector<cdouble> corr = dsp::fftconvolve_valid(seg_c, kernel);

        std::vector<double> mag(corr.size());
        for (std::size_t i = 0; i < corr.size(); ++i) mag[i] = std::abs(corr[i]);
        const std::size_t peak = argmax(mag);

        double seg_energy = 0.0;
        for (std::size_t i = peak;
             i < std::min(peak + PREAMBLE_SAMPLES, seg_c.size()); ++i)
            seg_energy += std::norm(seg_c[i]);
        seg_energy = std::sqrt(seg_energy);

        const double score = mag[peak] / (t_norm * seg_energy + 1e-12);
        if (!have_best || score > best_score) {
            have_best = true;
            best_score = score;
            best_p0 = lo + static_cast<std::int64_t>(peak);
            best_f = f_cand;
        }
    }

    std::int64_t p0 = best_p0;
    double f_hat = best_f;

    // Refine CFO from the phase between successive preamble periods at
    // the now-known timing: the same lag-M estimate, but noise-averaged
    // at the exact alignment, over every repeat rather than one pair.
    constexpr std::int64_t N_PRE = PREAMBLE_REPEATS * M;
    const std::int64_t u0 = p0 + PREAMBLE_CP;
    if (u0 >= 0 && u0 + N_PRE <= static_cast<std::int64_t>(z.size())) {
        const std::span<const cdouble> win(z.data() + u0, N_PRE);
        const std::vector<cdouble> zc = dsp::freq_correct(win, f_hat);
        cdouble d{0.0, 0.0};
        for (std::int64_t i = 0; i < N_PRE - M; ++i)
            d += zc[static_cast<std::size_t>(M + i)] *
                 std::conj(zc[static_cast<std::size_t>(i)]);
        if (std::abs(d) > 0.0)
            f_hat += std::arg(d) / (2 * PI * static_cast<double>(M) / FS);
    }

    return Acquisition{p0, f_hat, metric[n_star]};
}

BlindAcquisition acquire_blind(std::span<const cdouble> z, double max_offset_hz,
                               double bin_step_hz, int min_periods,
                               double threshold,
                               std::optional<SearchWindow> search) {
    const std::vector<cdouble> templ = ofdm::pilot_template();
    std::vector<cdouble> kernel(templ.size());
    for (std::size_t i = 0; i < templ.size(); ++i)
        kernel[i] = std::conj(templ[templ.size() - 1 - i]);

    const std::size_t seg_off =
        search ? static_cast<std::size_t>(search->start) : 0;
    const std::size_t seg_end =
        search ? static_cast<std::size_t>(search->end) : z.size();
    if (seg_end <= seg_off)
        throw SyncError("window too short for blind acquisition");
    const std::span<const cdouble> seg(z.data() + seg_off, seg_end - seg_off);

    if (seg.size() < static_cast<std::size_t>(FRAME_SAMPLES) *
                         static_cast<std::size_t>(min_periods))
        throw SyncError("window too short for blind acquisition");

    const int n_bins = static_cast<int>(std::ceil(max_offset_hz / bin_step_hz));
    const std::size_t n_fft =
        dsp::next_fast_len(seg.size() + kernel.size() - 1, /*real=*/false);
    const double bin_hz = static_cast<double>(FS) / static_cast<double>(n_fft);
    const std::size_t lo = kernel.size() - 1;
    const std::size_t valid_len = seg.size() - kernel.size() + 1;

    // Searching many CFO candidates against one segment is a Doppler
    // search, so rather than re-modulating the (long) segment per
    // candidate, take one FFT and circularly shift its spectrum:
    // modulation by f is exactly a shift of the DFT by f/bin_hz bins.
    std::vector<cdouble> pad_seg(n_fft, cdouble{});
    std::copy(seg.begin(), seg.end(), pad_seg.begin());
    std::vector<cdouble> pad_kernel(n_fft, cdouble{});
    std::copy(kernel.begin(), kernel.end(), pad_kernel.begin());
    const std::vector<cdouble> Sf = dsp::fft(pad_seg, true);
    const std::vector<cdouble> Tf = dsp::fft(pad_kernel, true);

    bool have_best = false;
    double best_score = 0.0;
    std::size_t best_phase = 0;
    double best_f = 0.0;
    std::vector<cdouble> shifted(n_fft);

    for (int k = -n_bins; k <= n_bins; ++k) {
        const long shift_bins = static_cast<long>(
            std::nearbyint(static_cast<double>(k) * bin_step_hz / bin_hz));
        const double f_cand = static_cast<double>(shift_bins) * bin_hz;

        // np.roll(Sf, -shift_bins): out[i] = Sf[(i + shift_bins) mod n]
        for (std::size_t i = 0; i < n_fft; ++i) {
            long src = static_cast<long>(i) + shift_bins;
            src %= static_cast<long>(n_fft);
            if (src < 0) src += static_cast<long>(n_fft);
            shifted[i] = Sf[static_cast<std::size_t>(src)] * Tf[i];
        }
        const std::vector<cdouble> mf = dsp::fft(shifted, false);

        const std::size_t n_periods = valid_len / FRAME_SAMPLES;
        if (n_periods < static_cast<std::size_t>(min_periods)) continue;

        std::vector<double> folded(FRAME_SAMPLES, 0.0);
        for (std::size_t p = 0; p < n_periods; ++p)
            for (int i = 0; i < FRAME_SAMPLES; ++i)
                folded[static_cast<std::size_t>(i)] +=
                    std::norm(mf[lo + p * FRAME_SAMPLES + static_cast<std::size_t>(i)]);

        const std::size_t phase = argmax(folded);
        const double score = folded[phase] / (median(folded) + 1e-12);
        if (!have_best || score > best_score) {
            have_best = true;
            best_score = score;
            best_phase = phase;
            best_f = f_cand;
        }
    }

    if (!have_best)
        throw SyncError("signal too short for blind acquisition at any CFO bin");
    if (best_score < threshold)
        throw SyncError("no periodic pilot found (peak prominence " +
                        std::to_string(best_score) + ")");

    return BlindAcquisition{static_cast<std::int64_t>(seg_off + best_phase), best_f,
                            best_score};
}

}  // namespace sstvae::sync
