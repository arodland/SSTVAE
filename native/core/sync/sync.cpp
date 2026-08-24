#include "sync/sync.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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

// The earliest local maximum within FIRST_PATH_SEARCH samples *ahead* of
// `peak` that still holds FIRST_PATH_FRAC of its power; `peak` itself if
// there is none. Mirrors sync.first_path -- see config.FIRST_PATH_SEARCH
// on the Python side for why the argmax is the wrong pick on a two-path
// channel, and why the caller must keep *scoring* at the argmax.
//
// `cyclic` distinguishes the blind path's fold, which wraps modulo
// FRAME_SAMPLES, from the preamble path's plain correlation, which does
// not and simply declines to look past its own edges.
std::size_t first_path(std::span<const double> power, std::size_t peak,
                       bool cyclic) {
    const std::size_t n = power.size();
    if (n < 3) return peak;
    const double thr = config::FIRST_PATH_FRAC * power[peak];
    for (int d = config::FIRST_PATH_SEARCH; d >= 1; --d) {
        std::int64_t i = static_cast<std::int64_t>(peak) - d;
        if (cyclic) {
            i %= static_cast<std::int64_t>(n);
            if (i < 0) i += static_cast<std::int64_t>(n);
        } else if (i < 1 || i + 1 >= static_cast<std::int64_t>(n)) {
            continue;
        }
        const auto u = static_cast<std::size_t>(i);
        const std::size_t lo = (u + n - 1) % n;
        const std::size_t hi = (u + 1) % n;
        if (power[u] >= thr && power[u] >= power[lo] && power[u] >= power[hi])
            return u;
    }
    return peak;
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
        std::vector<double> cpow(corr.size());
        for (std::size_t i = 0; i < corr.size(); ++i) {
            mag[i] = std::abs(corr[i]);
            cpow[i] = mag[i] * mag[i];
        }
        const std::size_t peak = argmax(cpow);

        double seg_energy = 0.0;
        for (std::size_t i = peak;
             i < std::min(peak + PREAMBLE_SAMPLES, seg_c.size()); ++i)
            seg_energy += std::norm(seg_c[i]);
        seg_energy = std::sqrt(seg_energy);

        // Scored at the argmax (TEMPLATE_SCORE_THRESHOLD is calibrated
        // against that), timed at the first path.
        const double score = mag[peak] / (t_norm * seg_energy + 1e-12);
        if (!have_best || score > best_score) {
            have_best = true;
            best_score = score;
            best_p0 = lo + static_cast<std::int64_t>(first_path(cpow, peak, false));
            best_f = f_cand;
        }
    }

    if (best_score < config::TEMPLATE_SCORE_THRESHOLD) {
        // The winning candidate is the *best available* one, not
        // necessarily a *good* one -- the lag-M metric above only rules
        // out pure noise, and real transmission data elsewhere in the
        // buffer can pass it too. This is the second gate: no candidate
        // here explains enough of the template's energy to trust as an
        // actual preamble.
        throw SyncError("no preamble found (best candidate score " +
                        std::to_string(best_score) + " at " + std::to_string(best_f) +
                        " Hz)");
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

double refine_cfo(std::span<const double> freqs, std::span<const double> scores,
                  std::size_t i) {
    if (i == 0 || i + 1 >= freqs.size()) return freqs[i];
    const double x0 = freqs[i - 1], x1 = freqs[i], x2 = freqs[i + 1];
    const double y0 = scores[i - 1], y1 = scores[i], y2 = scores[i + 1];
    const double d1 = x1 - x0, d2 = x1 - x2;
    const double denom = d1 * (y1 - y2) - d2 * (y1 - y0);
    if (denom == 0.0) return x1;
    const double vertex = x1 - 0.5 * (d1 * d1 * (y1 - y2) - d2 * d2 * (y1 - y0)) / denom;
    // A parabola through three noisy points can put its vertex anywhere;
    // outside the bracketing bins it is an extrapolation, not a peak.
    const double lo_x = std::min(x0, x2), hi_x = std::max(x0, x2);
    return std::min(std::max(vertex, lo_x), hi_x);
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

    // Every bin's score is kept, not just the running best, because the
    // winner's neighbours are what refine_cfo needs; one double per bin
    // next to an FFT each costs nothing.
    const std::size_t n_cand = static_cast<std::size_t>(2 * n_bins + 1);
    std::vector<double> cand_freqs(n_cand, 0.0);
    std::vector<double> cand_scores(n_cand, -std::numeric_limits<double>::infinity());
    std::vector<std::size_t> cand_phases(n_cand, 0);
    bool have_best = false;
    std::vector<cdouble> shifted(n_fft);

    for (int k = -n_bins; k <= n_bins; ++k) {
        const std::size_t ci = static_cast<std::size_t>(k + n_bins);
        const long shift_bins = static_cast<long>(
            std::nearbyint(static_cast<double>(k) * bin_step_hz / bin_hz));
        const double f_cand = static_cast<double>(shift_bins) * bin_hz;
        cand_freqs[ci] = f_cand;

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

        // Score at the argmax (BLIND_SCORE_THRESHOLD is calibrated
        // against that), report the first path's timing.
        const std::size_t pk = argmax(folded);
        cand_scores[ci] = folded[pk] / (median(folded) + 1e-12);
        cand_phases[ci] = first_path(folded, pk, true);
        have_best = true;
    }

    if (!have_best)
        throw SyncError("signal too short for blind acquisition at any CFO bin");
    const std::size_t best = argmax(cand_scores);
    const double best_score = cand_scores[best];
    if (best_score < threshold)
        throw SyncError("no periodic pilot found (peak prominence " +
                        std::to_string(best_score) + ")");

    return BlindAcquisition{static_cast<std::int64_t>(seg_off + cand_phases[best]),
                            refine_cfo(cand_freqs, cand_scores, best), best_score};
}

BlindAccumulator::BlindAccumulator(double max_offset_hz, double bin_step_hz, int min_periods,
                                   double threshold, std::optional<int> block_samples,
                                   std::vector<std::optional<double>> window_s)
    : min_periods_(min_periods), threshold_(threshold) {
    if (window_s.empty()) throw SyncError("BlindAccumulator: window_s must not be empty");
    const std::vector<cdouble> templ = ofdm::pilot_template();
    std::vector<cdouble> kernel(templ.size());
    for (std::size_t i = 0; i < templ.size(); ++i)
        kernel[i] = std::conj(templ[templ.size() - 1 - i]);
    m_ = static_cast<int>(kernel.size());

    // Sized from BLIND_BLOCK_RES_HZ, **not** from bin_step_hz: the block
    // is chosen so overlap-save is efficient, and the shift quantization
    // FS/block that it buys is the finest the search grid could ever be.
    // Tying it to the grid -- which is what this did while the two were
    // the same number -- collapses the block the moment the grid is
    // coarsened and hands back most of the saving.
    const int min_block =
        static_cast<int>(std::ceil(static_cast<double>(FS) / config::BLIND_BLOCK_RES_HZ));
    block_ = static_cast<int>(dsp::next_fast_len(
        static_cast<std::size_t>(std::max(min_block, 4 * m_)), /*real=*/false));
    step_ = block_ - (m_ - 1);
    if (step_ <= 0) throw SyncError("block_samples too small for the pilot kernel");
    if (block_samples) {
        block_ = *block_samples;
        step_ = block_ - (m_ - 1);
    }
    const double bin_hz = static_cast<double>(FS) / static_cast<double>(block_);

    n_bins_ = static_cast<int>(std::ceil(max_offset_hz / bin_step_hz));
    for (int k = -n_bins_; k <= n_bins_; ++k) {
        const int shift = static_cast<int>(std::nearbyint(static_cast<double>(k) * bin_step_hz / bin_hz));
        shift_bins_.push_back(shift);
        freqs_.push_back(static_cast<double>(shift) * bin_hz);
    }

    std::vector<cdouble> pad_kernel(static_cast<std::size_t>(block_), cdouble{});
    std::copy(kernel.begin(), kernel.end(), pad_kernel.begin());
    kernel_f_ = dsp::fft(pad_kernel, true);

    // One decay factor per timescale, applied once per processed block,
    // uniformly across every phase and CFO bin within that timescale --
    // see the BlindAccumulator docstring in sync.py for why this is
    // exponential decay rather than exact eviction, and why several
    // timescales run in parallel off the same per-block matched-filter
    // result rather than a single one.
    n_scales_ = static_cast<int>(window_s.size());
    decay_per_block_.resize(static_cast<std::size_t>(n_scales_));
    for (int t = 0; t < n_scales_; ++t)
        decay_per_block_[static_cast<std::size_t>(t)] =
            window_s[static_cast<std::size_t>(t)]
                ? std::exp(-static_cast<double>(step_) /
                          (*window_s[static_cast<std::size_t>(t)] * static_cast<double>(FS)))
                : 1.0;

    folded_.assign(static_cast<std::size_t>(n_scales_) * shift_bins_.size() *
                       static_cast<std::size_t>(FRAME_SAMPLES),
                   0.0);
}

void BlindAccumulator::push(std::span<const cdouble> z, std::int64_t start_sample) {
    if (buf_start_ < 0) {
        buf_start_ = start_sample;
    } else if (start_sample != buf_start_ + static_cast<std::int64_t>(buf_.size())) {
        throw SyncError(
            "BlindAccumulator::push: expected a contiguous continuation at sample " +
            std::to_string(buf_start_ + static_cast<std::int64_t>(buf_.size())) + ", got " +
            std::to_string(start_sample));
    }
    buf_.insert(buf_.end(), z.begin(), z.end());

    const int B = block_, m = m_, step = step_;
    std::vector<cdouble> shifted(static_cast<std::size_t>(B));
    std::size_t pos = 0;
    while (pos + static_cast<std::size_t>(B) <= buf_.size()) {
        const std::vector<cdouble> block(
            buf_.begin() + static_cast<std::ptrdiff_t>(pos),
            buf_.begin() + static_cast<std::ptrdiff_t>(pos) + B);
        const std::vector<cdouble> block_f = dsp::fft(block, true);

        // Same convention as acquire_blind's p2[j]: mf[i] (0-indexed
        // within the valid slice, i.e. mf_full[m-1+i]) is the matched
        // filter's response for a pilot window starting at block-local
        // index i, so no (m - 1) belongs in abs0.
        const std::int64_t abs0 = buf_start_ + static_cast<std::int64_t>(pos);
        const int phase0 = static_cast<int>(((abs0 % FRAME_SAMPLES) + FRAME_SAMPLES) % FRAME_SAMPLES);

        // Decay every timescale's whole array first -- cheap
        // (n_scales_ x n_bins_ x FRAME_SAMPLES scalars) next to the
        // per-bin FFT below, which is why adding timescales barely
        // moves push()'s cost.
        const std::size_t scale_stride =
            shift_bins_.size() * static_cast<std::size_t>(FRAME_SAMPLES);
        for (int t = 0; t < n_scales_; ++t) {
            const double d = decay_per_block_[static_cast<std::size_t>(t)];
            if (d == 1.0) continue;
            double* base = &folded_[static_cast<std::size_t>(t) * scale_stride];
            for (std::size_t i = 0; i < scale_stride; ++i) base[i] *= d;
        }

        for (std::size_t bi = 0; bi < shift_bins_.size(); ++bi) {
            const int shift = shift_bins_[bi];
            // np.roll(block_f, -shift): out[i] = block_f[(i + shift) mod B]
            for (int i = 0; i < B; ++i) {
                long src = static_cast<long>(i) + shift;
                src %= B;
                if (src < 0) src += B;
                shifted[static_cast<std::size_t>(i)] =
                    block_f[static_cast<std::size_t>(src)] * kernel_f_[static_cast<std::size_t>(i)];
            }
            const std::vector<cdouble> mf = dsp::fft(shifted, false);
            // Every timescale folds in the same per-block matched-filter
            // power -- only the decay each one already applied to its
            // own history (above) differs.
            for (int t = 0; t < n_scales_; ++t) {
                double* row = &folded_[static_cast<std::size_t>(t) * scale_stride +
                                       bi * static_cast<std::size_t>(FRAME_SAMPLES)];
                int phase = phase0;
                for (int i = 0; i < step; ++i) {
                    row[phase] += std::norm(mf[static_cast<std::size_t>(m - 1 + i)]);
                    if (++phase == FRAME_SAMPLES) phase = 0;
                }
            }
        }
        n_valid_ += step;
        pos += static_cast<std::size_t>(step);
    }

    buf_.erase(buf_.begin(), buf_.begin() + static_cast<std::ptrdiff_t>(pos));
    buf_start_ += static_cast<std::int64_t>(pos);
}

BlindAcquisition BlindAccumulator::result(std::int64_t origin) const {
    if (n_valid_ < static_cast<std::int64_t>(FRAME_SAMPLES) * min_periods_)
        throw SyncError("window too short for blind acquisition");

    // Best (timescale, bin) pair -- reports whichever timescale's peak
    // score is highest, not a fixed one, since which timescale that is
    // depends on which mode (if any) is actually transmitting.
    const std::size_t scale_stride =
        shift_bins_.size() * static_cast<std::size_t>(FRAME_SAMPLES);
    // Every (timescale, bin) score is kept rather than a running best,
    // because refine_cfo needs the winner's two neighbours within its
    // own timescale.
    std::vector<double> scores(static_cast<std::size_t>(n_scales_) * shift_bins_.size(), 0.0);
    bool have_best = false;
    double best_score = 0.0;
    std::size_t best_bin = 0;
    int best_scale = 0;
    for (int t = 0; t < n_scales_; ++t) {
        const double* scale_base = &folded_[static_cast<std::size_t>(t) * scale_stride];
        for (std::size_t bi = 0; bi < shift_bins_.size(); ++bi) {
            const std::span<const double> row(
                scale_base + bi * static_cast<std::size_t>(FRAME_SAMPLES),
                static_cast<std::size_t>(FRAME_SAMPLES));
            const std::size_t peak = argmax(row);
            const double score =
                row[peak] / (median(std::vector<double>(row.begin(), row.end())) + 1e-12);
            scores[static_cast<std::size_t>(t) * shift_bins_.size() + bi] = score;
            if (!have_best || score > best_score) {
                have_best = true;
                best_score = score;
                best_bin = bi;
                best_scale = t;
            }
        }
    }
    if (best_score < threshold_)
        throw SyncError("no periodic pilot found (peak prominence " +
                        std::to_string(best_score) + ")");

    const double* scale_base = &folded_[static_cast<std::size_t>(best_scale) * scale_stride];
    const std::span<const double> row(scale_base + best_bin * static_cast<std::size_t>(FRAME_SAMPLES),
                                      static_cast<std::size_t>(FRAME_SAMPLES));
    // Score at the argmax (the threshold above is calibrated against it),
    // timing at the first path -- see config.FIRST_PATH_SEARCH.
    const std::size_t phase = first_path(row, argmax(row), true);

    // Sub-bin, against the winning timescale's own per-bin scores -- the
    // grid is coarse by design and the raw bin centre is several Hz out,
    // which the demodulator cannot absorb. See refine_cfo.
    const std::span<const double> scale_scores(
        scores.data() + static_cast<std::size_t>(best_scale) * shift_bins_.size(),
        shift_bins_.size());
    // Rebase from the fold's absolute coordinate to the caller's --
    // see the header comment on `origin`.
    const std::int64_t f = FRAME_SAMPLES;
    const std::int64_t rebased =
        ((static_cast<std::int64_t>(phase) - origin) % f + f) % f;
    return BlindAcquisition{rebased, refine_cfo(freqs_, scale_scores, best_bin),
                            best_score};
}

}  // namespace sstvae::sync
