#include "modem/diversity.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "framing/framing.hpp"

namespace sstvae::modem::diversity {

namespace {

void validate_branches(const std::vector<DemodResult>& results) {
    if (results.empty()) throw std::invalid_argument("needs at least one branch");
    const std::string_view name = results[0].mode.name;
    for (std::size_t i = 1; i < results.size(); ++i) {
        if (results[i].mode.name != name)
            throw std::invalid_argument("branch mode mismatch");
    }
}

// snr_lin: each branch's linear SNR (with the "nothing usable" fallback
// to all-ones already applied). w: (n_branches * n_latents) row-major
// inverse-variance (MRC) combining weight, snr_lin[i] * weight[i][k]**2.
struct MrcWeights {
    std::vector<double> snr_lin;
    std::vector<double> w;
    std::size_t n_latents;
};

MrcWeights mrc_weights(const std::vector<DemodResult>& results) {
    const std::size_t nb = results.size();
    const std::size_t nl = results[0].latents.size();
    std::vector<double> snr_lin(nb);
    bool any_positive = false;
    for (std::size_t i = 0; i < nb; ++i) {
        const double s = results[i].snr_db;
        snr_lin[i] = std::isfinite(s) ? std::pow(10.0, s / 10.0) : 0.0;
        any_positive = any_positive || snr_lin[i] > 0.0;
    }
    if (!any_positive) {
        // No branch has a usable SNR estimate -- fall back to unweighted
        // averaging rather than producing an all-zero combine.
        std::fill(snr_lin.begin(), snr_lin.end(), 1.0);
    }

    std::vector<double> w(nb * nl);
    for (std::size_t i = 0; i < nb; ++i) {
        const double s = snr_lin[i];
        const std::vector<double>& wt = results[i].weights;
        for (std::size_t k = 0; k < nl; ++k) {
            w[i * nl + k] = s * wt[k] * wt[k];
        }
    }
    return {std::move(snr_lin), std::move(w), nl};
}

}  // namespace

DemodResult combine_demod_results(const std::vector<DemodResult>& results) {
    validate_branches(results);
    if (results.size() == 1) return results[0];

    const MrcWeights mw = mrc_weights(results);
    const std::size_t nb = results.size();
    const std::size_t nl = mw.n_latents;
    const double ref = *std::max_element(mw.snr_lin.begin(), mw.snr_lin.end());

    std::vector<double> combined_latents(nl, 0.0);
    std::vector<double> combined_weights(nl, 0.0);
    for (std::size_t k = 0; k < nl; ++k) {
        double num = 0.0, denom = 0.0;
        for (std::size_t i = 0; i < nb; ++i) {
            const double wk = mw.w[i * nl + k];
            num += wk * results[i].latents[k];
            denom += wk;
        }
        if (denom > 0.0) {
            combined_latents[k] = num / denom;
            combined_weights[k] = std::min(1.0, std::sqrt(denom / ref));
        }
    }

    // The strongest branch by SNR (ties keep the first) supplies every
    // scalar/metadata field the combine itself doesn't produce -- mode,
    // freq_offset, beacon, callsign, preamble_start.
    std::size_t best_i = 0;
    double best_snr = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < nb; ++i) {
        const double s = results[i].snr_db;
        if (std::isfinite(s) && s > best_snr) {
            best_snr = s;
            best_i = i;
        }
    }

    int frames_received = results[0].frames_received;
    for (const DemodResult& r : results)
        frames_received = std::max(frames_received, r.frames_received);

    double total_snr_lin = 0.0;
    for (double s : mw.snr_lin) total_snr_lin += s;

    DemodResult out = results[best_i];
    out.latents = std::move(combined_latents);
    out.weights = std::move(combined_weights);
    out.frames_received = frames_received;
    out.snr_db = 10.0 * std::log10(total_snr_lin);
    return out;
}

std::vector<std::vector<double>> branch_contribution(
    const std::vector<DemodResult>& results) {
    validate_branches(results);
    const std::size_t nb = results.size();
    const std::size_t nl = results[0].latents.size();
    std::vector<std::vector<double>> frac(nb, std::vector<double>(nl, 0.0));

    if (nb == 1) {
        for (std::size_t k = 0; k < nl; ++k)
            frac[0][k] = results[0].weights[k] > 0.0 ? 1.0 : 0.0;
        return frac;
    }

    const MrcWeights mw = mrc_weights(results);
    for (std::size_t k = 0; k < nl; ++k) {
        double denom = 0.0;
        for (std::size_t i = 0; i < nb; ++i) denom += mw.w[i * nl + k];
        if (denom > 0.0) {
            for (std::size_t i = 0; i < nb; ++i) frac[i][k] = mw.w[i * nl + k] / denom;
        }
    }
    return frac;
}

images::Picture contribution_image(const std::vector<DemodResult>& results, int scale) {
    if (results.size() != 2)
        throw std::invalid_argument("contribution_image needs exactly two branches");
    validate_branches(results);

    const std::vector<std::vector<double>> frac = branch_contribution(results);
    const int n_frames = results[0].mode.n_frames;
    const int channels = config::LATENT_CHANNELS;
    constexpr int per_channel = config::LATENT_H * config::LATENT_W;

    // Per (channel, frame) cell, averaged over whichever latents the
    // interleaver placed there -- a frame's latents are scattered across
    // channels, not one-per-frame, so most cells are an average of a few.
    std::vector<double> sum_r(static_cast<std::size_t>(channels) * n_frames, 0.0);
    std::vector<double> sum_b(sum_r.size(), 0.0);
    std::vector<int> counts(sum_r.size(), 0);

    for (int f = 0; f < n_frames; ++f) {
        const framing::FrameSlots fs = framing::slot_range_for_frame(f);
        for (const std::int64_t idx : fs.indices) {
            const int ch = static_cast<int>(idx / per_channel);
            const std::size_t cell = static_cast<std::size_t>(ch) * n_frames + f;
            sum_r[cell] += frac[0][static_cast<std::size_t>(idx)];
            sum_b[cell] += frac[1][static_cast<std::size_t>(idx)];
            ++counts[cell];
        }
    }

    if (scale < 1) scale = 1;
    const int out_w = n_frames * scale;
    const int out_h = channels * scale;
    images::Picture img(out_w, out_h);

    // Nearest-neighbor replication by construction, not a smoothing
    // resize -- each cell is a categorical (channel, frame) value, and
    // stb's srgb resizer (images::resize) would blur cell boundaries
    // that are meaningful, not noise.
    for (int ch = 0; ch < channels; ++ch) {
        for (int f = 0; f < n_frames; ++f) {
            const std::size_t cell = static_cast<std::size_t>(ch) * n_frames + f;
            std::uint8_t r = 0, b = 0;
            if (counts[cell] > 0) {
                r = static_cast<std::uint8_t>(
                    std::clamp(sum_r[cell] / counts[cell] * 255.0, 0.0, 255.0));
                b = static_cast<std::uint8_t>(
                    std::clamp(sum_b[cell] / counts[cell] * 255.0, 0.0, 255.0));
            }
            for (int dy = 0; dy < scale; ++dy) {
                const int y = ch * scale + dy;
                for (int dx = 0; dx < scale; ++dx) {
                    const int x = f * scale + dx;
                    const std::size_t px =
                        (static_cast<std::size_t>(y) * static_cast<std::size_t>(out_w) +
                         static_cast<std::size_t>(x)) * 3;
                    img.rgb[px + 0] = r;
                    img.rgb[px + 2] = b;
                }
            }
        }
    }
    return img;
}

}  // namespace sstvae::modem::diversity
